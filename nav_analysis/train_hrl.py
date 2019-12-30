#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import os.path as osp
import random
import signal
import sys
import threading
from collections import deque
from time import sleep, time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict
from torch.utils import tensorboard

from habitat import logger
from nav_analysis.rl.ppo import PPO, Policy, RolloutStorage
from nav_analysis.rl.ppo.driver_policy import DriverPolicy
from nav_analysis.rl.ppo.hrl_policy import HRLPolicy
from nav_analysis.rl.ppo.utils import (
    batch_obs,
    ppo_args,
    update_linear_schedule,
)
from nav_analysis.train_ppo import construct_envs

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


WORLD_RANK = -1
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", 0)
STATE_FILE = osp.join(
    os.environ["HOME"], ".interrupted_states", "{}.pt".format(SLURM_JOB_ID)
)

INTERRUPTED = threading.Event()
INTERRUPTED.clear()

REQUEUE = threading.Event()
REQUEUE.clear()


def requeue_handler(signum, frame):
    # define the handler function
    # note that this is not executed here, but rather
    # when the associated signal is sent
    print("signaled for requeue")
    INTERRUPTED.set()
    REQUEUE.set()


def clean_exit_handler(signum, frame):
    print("exiting cleanly")
    INTERRUPTED.set()


signal.signal(signal.SIGUSR1, requeue_handler)

signal.signal(signal.SIGUSR2, clean_exit_handler)
signal.signal(signal.SIGTERM, clean_exit_handler)
signal.signal(signal.SIGINT, clean_exit_handler)


def before_exit():
    dist.barrier()

    if WORLD_RANK != 0:
        return

    # Make sure rank 0 is the last to exit.  GPU clean-up gets weird otherwise
    sleep(1)
    if REQUEUE.is_set():
        print("requeuing job " + os.environ["SLURM_JOB_ID"])
        os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])


def init_distrib():
    master_port = int(os.environ.get("MASTER_PORT", 1234))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID")))
    world_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS")))

    torch.cuda.set_device(torch.device("cuda", local_rank))

    tcp_store = dist.TCPStore(master_addr, master_port, world_size, world_rank == 0)
    dist.init_process_group(
        dist.Backend.NCCL, store=tcp_store, rank=world_rank, world_size=world_size
    )

    return local_rank, tcp_store


def main():
    global WORLD_RANK
    args = ppo_args()

    local_rank, tcp_store = init_distrib()

    args.general.local_rank = local_rank
    args.general.sim_gpu_id = args.general.local_rank
    device = torch.device("cuda", args.general.local_rank)
    torch.cuda.set_device(device)

    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    WORLD_RANK = world_rank
    random.seed(args.general.seed + world_rank)
    torch.manual_seed(args.general.seed + world_rank)
    torch.cuda.manual_seed_all(args.general.seed + world_rank)
    np.random.seed(args.general.seed + world_rank)

    is_done_store = dist.PrefixStore("rollout_tracker", tcp_store)
    if WORLD_RANK == 0:
        is_done_store.set(f"num_done", "0")

    checkpoint_folder = args.logging.checkpoint_folder
    if WORLD_RANK == 0 and not os.path.isdir(args.logging.checkpoint_folder):
        os.makedirs(args.logging.checkpoint_folder)

    tensorboard_dir = args.logging.tensorboard_dir

    output_log_file = args.logging.log_file
    if WORLD_RANK == 0:
        logger.add_filehandler(output_log_file)

    with construct_envs(args) as envs:

        num_recurrent_layers = args.model.num_recurrent_layers
        pointnav_agent = Policy(
            observation_space=envs.observation_spaces[0],
            action_space=envs.action_spaces[0],
            hidden_size=args.model.hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            blind=args.model.blind,
            use_aux_losses=False,
            rnn_type=args.model.rnn_type,
            resnet_baseplanes=args.model.resnet_baseplanes,
            backbone=args.model.backbone,
            task=args.task.nav_task,
            norm_visual_inputs=args.model.norm_visual_inputs,
            two_headed=args.model.two_headed,
        )

        pointnav_agent.load_state_dict(
            {
                k[len("actor_critic.") :]: v
                for k, v in torch.load(
                    args.transfer.pretrained_path, map_location="cpu"
                )["state_dict"].items()
                if "ddp" not in k and "actor_critic" in k
            }
        )
        pointnav_agent.eval()
        for param in pointnav_agent.parameters():
            param.requires_grad_(False)

        pointnav_agent.to(device)
        actor_critic = HRLPolicy(pointnav_agent)

        actor_critic.to(device)
        agent = PPO(
            actor_critic,
            args.ppo.clip_param,
            args.ppo.ppo_epoch,
            args.ppo.num_mini_batch,
            args.ppo.value_loss_coef,
            args.ppo.entropy_coef,
            lr=args.optimizer.lr,
            eps=args.optimizer.eps,
            max_grad_norm=args.optimizer.max_grad_norm,
            weight_decay=args.optimizer.weight_decay,
        )

        env_time = 0
        pth_time = 0
        sync_time = 0
        count_steps = torch.tensor(0, device=device, dtype=torch.int64)
        count_checkpoints = 0
        update_start_from = 0
        prev_time = 0

        if osp.exists(STATE_FILE):
            ckpt = torch.load(STATE_FILE, map_location=device)
            agent.load_state_dict(
                {k: v for k, v in ckpt["state_dict"].items() if "ddp" not in k}
            )
            agent.optimizer.load_state_dict(ckpt["optim_state"])

            env_time = ckpt["extra"]["env_time"]
            pth_time = ckpt["extra"]["pth_time"]
            sync_time = ckpt["extra"]["sync_time"]
            count_steps = ckpt["extra"]["count_steps"]
            count_checkpoints = ckpt["extra"]["count_checkpoints"]
            update_start_from = ckpt["extra"]["update"]
            prev_time = ckpt["extra"]["prev_time"]
            output_log_file = ckpt["extra"]["output_log_file"]
            checkpoint_folder = ckpt["extra"]["checkpoint_folder"]
            tensorboard_dir = ckpt["extra"]["tensorboard_dir"]

            if WORLD_RANK == 0:
                logger.add_filehandler(output_log_file)
                logger.info("Starting requeued job")

            del ckpt

        agent.init_distributed()
        actor_critic = agent.actor_critic

        if WORLD_RANK == 0:
            logger.info("-" * 50)
            logger.info("args:\n" + args.pretty())
            logger.info("-" * 50)
            logger.info(
                "agent number of parameters: {}".format(
                    sum(
                        param.numel()
                        for param in actor_critic.parameters()
                        if param.requires_grad
                    )
                )
            )

        observations = envs.reset()
        batch = batch_obs(observations)
        for k, v in batch.items():
            batch[k] = v.to(device)

        features = pointnav_agent.net.cnn[0](
            pointnav_agent.net.running_mean_and_var(
                F.avg_pool2d(batch["rgb"].float().permute(0, 3, 1, 2), 2)
            )
        )
        observations = None
        batch = None

        logger.info(envs.observation_spaces[0])

        rollouts = RolloutStorage(
            args.ppo.num_steps,
            envs.num_envs,
            SpaceDict(
                {
                    "features": spaces.Box(
                        low=np.iinfo(np.uint32).min,
                        high=np.iinfo(np.uint32).max,
                        shape=(128, 4, 4),
                        dtype=np.float32,
                    ),
                    "prev_pointgoal": spaces.Box(
                        low=np.iinfo(np.uint32).min,
                        high=np.iinfo(np.uint32).max,
                        shape=(3,),
                        dtype=np.float32,
                    ),
                    "pg_model_steps": spaces.Box(
                        low=np.iinfo(np.uint32).min,
                        high=np.iinfo(np.uint32).max,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                    **envs.observation_spaces[0].spaces,
                }
            ),
            envs.action_spaces[0],
            pointnav_agent.net._hidden_size,
            num_recurrent_layers=pointnav_agent.net.num_recurrent_layers + 4,
        )
        rollouts.to(device)

        rollouts.observations["features"][0].copy_(features)

        episode_rewards = torch.zeros(envs.num_envs, 1).to(device)
        episode_counts = torch.zeros(envs.num_envs, 1).to(device)
        current_episode_reward = torch.zeros(envs.num_envs, 1).to(device)

        window_episode_reward = deque(maxlen=args.logging.reward_window_size)
        window_episode_counts = deque(maxlen=args.logging.reward_window_size)

        if args.task.nav_task == "flee":
            episode_flee_dist = torch.zeros(envs.num_envs, 1, device=device)

            window_episode_flee_dist = deque(maxlen=args.logging.reward_window_size)
        elif args.task.nav_task == "explore":
            episode_explore_amount = torch.zeros(envs.num_envs, 1, device=device)

            window_episode_explore_amount = deque(
                maxlen=args.logging.reward_window_size
            )

        t_start = time()

        tb_enabled = WORLD_RANK == 0
        if tb_enabled:
            writer_kwargs = dict(log_dir=tensorboard_dir, purge_step=count_steps.item())

        with (
            tensorboard.SummaryWriter(**writer_kwargs)
            if tb_enabled
            else contextlib.suppress()
        ) as writer:
            for update in range(update_start_from, args.ppo.num_updates):

                if args.ppo.linear_lr_decay:
                    update_linear_schedule(
                        agent.optimizer, update, args.ppo.num_updates, args.optimizer.lr
                    )

                if args.ppo.linear_clip_decay:
                    agent.clip_param = args.ppo.clip_param * (
                        1 - update / args.ppo.num_updates
                    )

                actor_critic.eval()
                for step in range(args.ppo.num_steps):
                    t_sample_action = time()
                    # sample actions
                    with torch.no_grad():
                        step_observation = {
                            k: v[step] for k, v in rollouts.observations.items()
                        }

                        (
                            values,
                            actions,
                            actions_log_probs,
                            entropy,
                            recurrent_hidden_states,
                            goal_preds,
                            pg_masks,
                            pg_model_steps,
                        ) = actor_critic.act(
                            step_observation,
                            rollouts.recurrent_hidden_states[step],
                            rollouts.prev_actions[step],
                            rollouts.masks[step],
                        )

                    pth_time += time() - t_sample_action

                    t_step_env = time()

                    outputs = envs.step([a[0].item() for a in actions])
                    observations, rewards, dones, infos = [
                        list(x) for x in zip(*outputs)
                    ]

                    env_time += time() - t_step_env

                    t_update_stats = time()
                    batch = batch_obs(observations)
                    for k, v in batch.items():
                        batch[k] = v.to(device)

                    rewards = torch.tensor(
                        rewards, dtype=torch.float, device=device
                    ).unsqueeze(-1)

                    masks = torch.tensor(
                        [[0.0] if done else [1.0] for done in dones],
                        dtype=torch.float,
                        device=device,
                    )

                    current_episode_reward += rewards
                    episode_rewards += (1.0 - masks) * current_episode_reward
                    episode_counts += 1.0 - masks
                    current_episode_reward *= masks

                    if args.task.nav_task == "flee":
                        episode_flee_dist += torch.tensor(
                            [
                                [info["flee_distance"]] if done else [0.0]
                                for info, done in zip(infos, dones)
                            ],
                            dtype=torch.float,
                            device=device,
                        )
                    elif args.task.nav_task == "explore":
                        episode_explore_amount += torch.tensor(
                            [
                                [info["visited"]] if done else [0.0]
                                for info, done in zip(infos, dones)
                            ],
                            dtype=torch.float,
                            device=device,
                        )

                    features = pointnav_agent.net.cnn[0](
                        pointnav_agent.net.running_mean_and_var(
                            F.avg_pool2d(batch["rgb"].float().permute(0, 3, 1, 2), 2)
                        )
                    )
                    batch["features"] = features
                    batch["prev_pointgoal"] = goal_preds
                    batch["pg_model_steps"] = pg_model_steps

                    rollouts.insert(
                        batch,
                        recurrent_hidden_states,
                        actions,
                        actions_log_probs,
                        values,
                        rewards,
                        masks,
                        entropy,
                    )
                    observations = None
                    batch = None

                    pth_time += time() - t_update_stats

                    if (step + 1) >= (args.ppo.num_steps / 4) and int(
                        is_done_store.get("num_done")
                    ) >= (world_size * args.ddppo.sync_frac):
                        break

                is_done_store.add("num_done", 1)

                t_sync = time()
                dist.barrier()
                t_sync = torch.tensor(
                    time() - t_sync, device=device, dtype=torch.float32
                )
                dist.all_reduce(t_sync)
                sync_time += t_sync.item() / world_size

                step_delta = torch.full_like(count_steps, rollouts.step * envs.num_envs)
                dist.all_reduce(step_delta)
                count_steps += step_delta

                t_update_model = time()
                with torch.no_grad():
                    last_observation = {
                        k: v[rollouts.step] for k, v in rollouts.observations.items()
                    }
                    next_value = actor_critic.get_value(
                        last_observation,
                        rollouts.recurrent_hidden_states[rollouts.step],
                        rollouts.prev_actions[rollouts.step],
                        rollouts.masks[rollouts.step],
                    ).detach()

                rollouts.compute_returns(
                    next_value, args.ppo.use_gae, args.ppo.gamma, args.ppo.tau
                )

                actor_critic.train()

                value_loss, action_loss, dist_entropy = agent.update(rollouts)

                losses = torch.tensor(
                    [value_loss, action_loss, dist_entropy],
                    device=device,
                    dtype=torch.float32,
                )
                dist.all_reduce(losses)
                losses /= world_size

                rollouts.after_update()
                pth_time += time() - t_update_model

                torch.cuda.synchronize()
                if WORLD_RANK == 0:
                    is_done_store.set("num_done", "0")

                if args.task.nav_task == "flee":
                    stats = torch.cat(
                        [episode_rewards, episode_counts, episode_flee_dist], 1
                    )
                elif args.task.nav_task == "explore":
                    stats = torch.cat(
                        [episode_rewards, episode_counts, episode_explore_amount], 1
                    )
                else:
                    stats = torch.cat([episode_rewards, episode_counts], 1)

                dist.all_reduce(stats)

                if args.task.nav_task == "flee":
                    window_episode_reward.append(stats[:, 0])
                    window_episode_counts.append(stats[:, 1])
                    window_episode_flee_dist.append(stats[:, 2])
                elif args.task.nav_task == "explore":
                    window_episode_reward.append(stats[:, 0])
                    window_episode_counts.append(stats[:, 1])
                    window_episode_explore_amount.append(stats[:, 2])
                else:
                    window_episode_reward.append(stats[:, 0])
                    window_episode_counts.append(stats[:, 1])

                if tb_enabled:
                    if args.task.nav_task == "flee":
                        stats = zip(
                            ["count", "reward", "flee_dist"],
                            [
                                window_episode_counts,
                                window_episode_reward,
                                window_episode_flee_dist,
                            ],
                        )
                    elif args.task.nav_task == "explore":
                        stats = zip(
                            ["count", "reward", "visited"],
                            [
                                window_episode_counts,
                                window_episode_reward,
                                window_episode_explore_amount,
                            ],
                        )
                    else:
                        stats = zip(
                            ["count", "reward"],
                            [window_episode_counts, window_episode_reward],
                        )

                    deltas = {
                        k: (
                            (v[-1] - v[0]).sum().item()
                            if len(v) > 1
                            else v[0].sum().item()
                        )
                        for k, v in stats
                    }
                    deltas["count"] = max(deltas["count"], 1.0)

                    writer.add_scalar(
                        "reward", deltas["reward"] / deltas["count"], count_steps
                    )

                    writer.add_scalars(
                        "losses",
                        {
                            k: l.item() * s
                            for l, k, s in zip(
                                losses, ["value", "policy", "entropy"], [1, 1, 0.1]
                            )
                        },
                        count_steps,
                    )

                    if args.task.nav_task in {"explore", "flee"}:
                        writer.add_scalars(
                            "metrics",
                            {
                                k: deltas[k] / deltas["count"]
                                for k in [
                                    "flee_dist"
                                    if args.task.nav_task == "flee"
                                    else "visited"
                                ]
                            },
                            count_steps,
                        )

                def _save_state():
                    checkpoint = {
                        "state_dict": {
                            k: v
                            for k, v in agent.state_dict().items()
                            if "ddp" not in k
                        },
                        "optim_state": agent.optimizer.state_dict(),
                        "args": args,
                    }
                    checkpoint["extra"] = dict(
                        prev_time=prev_time + (time() - t_start),
                        count_steps=count_steps,
                        update=update,
                        count_checkpoints=count_checkpoints,
                        pth_time=pth_time,
                        env_time=env_time,
                        sync_time=sync_time,
                        output_log_file=output_log_file,
                        checkpoint_folder=checkpoint_folder,
                        tensorboard_dir=tensorboard_dir,
                    )
                    torch.save(checkpoint, STATE_FILE)

                if INTERRUPTED.is_set():
                    if world_rank == 0:
                        logger.info("Interrupted, REQUEUE: {}".format(REQUEUE.is_set()))
                    if world_rank == 0 and REQUEUE.is_set():
                        logger.info("Saving state for requeue")
                        _save_state()

                    return

                if world_rank == 0:
                    if update > 0 and update % args.logging.save_state_interval == 0:
                        _save_state()

                    # log stats
                    if update > 0 and update % args.logging.log_interval == 0:
                        logger.info(
                            "update: {}\tfps: {:.3f}".format(
                                update, count_steps / ((time() - t_start) + prev_time)
                            )
                        )

                        logger.info(
                            "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\tsync-time: {:.3f}s\tsync-frac: {:.3f}"
                            "\tframes: {}".format(
                                update,
                                env_time,
                                pth_time,
                                sync_time,
                                sync_time
                                / max((env_time + pth_time + sync_time), 1e-8),
                                count_steps.item(),
                            )
                        )

                        window_rewards = (
                            window_episode_reward[-1] - window_episode_reward[0]
                        ).sum()
                        window_counts = (
                            window_episode_counts[-1] - window_episode_counts[0]
                        ).sum()
                        if args.task.nav_task == "flee":
                            window_flee = (
                                window_episode_flee_dist[-1]
                                - window_episode_flee_dist[0]
                            ).sum()
                        elif args.task.nav_task == "explore":
                            window_explore = (
                                window_episode_explore_amount[-1]
                                - window_episode_explore_amount[0]
                            ).sum()

                        if window_counts > 0:

                            if args.task.nav_task == "flee":
                                logger.info(
                                    "Average window size {} reward: {:.3f}\tflee-dist: {:.3f}".format(
                                        len(window_episode_reward),
                                        (window_rewards / window_counts).item(),
                                        (window_flee / window_counts).item(),
                                    )
                                )
                            elif args.task.nav_task == "explore":
                                logger.info(
                                    "Average window size {} reward: {:.3f}\texplore-amount: {:.3f}".format(
                                        len(window_episode_reward),
                                        (window_rewards / window_counts).item(),
                                        (window_explore / window_counts).item(),
                                    )
                                )
                            else:
                                logger.info(
                                    "Average window size {} reward: {:.3f}".format(
                                        len(window_episode_reward),
                                        (window_rewards / window_counts).item(),
                                    )
                                )
                        else:
                            logger.info("No episodes finish in current window")

                    # checkpoint model
                    if update % args.logging.checkpoint_interval == 0:
                        checkpoint = {
                            "state_dict": {
                                k: v
                                for k, v in agent.state_dict().items()
                                if "ddp" not in k
                            },
                            "args": args,
                            "num_frames": count_steps,
                            "wall_time": (time() - t_start) + prev_time,
                        }
                        torch.save(
                            checkpoint,
                            os.path.join(
                                checkpoint_folder,
                                "ckpt.{}.pth".format(count_checkpoints),
                            ),
                        )
                        count_checkpoints += 1


if __name__ == "__main__":
    main()

    before_exit()