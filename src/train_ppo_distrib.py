#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from time import time, sleep
from collections import deque
import random

import torch
from habitat import logger
from src.rl.ppo import PPO, Policy, RolloutStorage
from src.rl.ppo.utils import update_linear_schedule, ppo_args, batch_obs
import torch.distributed as dist
from src.train_ppo import construct_envs
import threading
import os.path as osp
import signal
import contextlib
import tensorboardX


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
    if WORLD_RANK != 0:
        return

    # Make sure rank 0 is the last to exit.  GPU clean-up gets weird otherwise
    sleep(1)
    if REQUEUE.is_set():
        print("requeuing job " + os.environ["SLURM_JOB_ID"])
        os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
    elif not REQUEUE.is_set() and osp.exists(STATE_FILE):
        os.remove(STATE_FILE)


def setup_distrib_env():
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "1234"

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")

    if "RANK" not in os.environ:
        os.environ["RANK"] = os.environ.get("SLURM_PROCID", "0")

    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", "1")


def main():
    global WORLD_RANK
    parser = ppo_args()
    args = parser.parse_args()

    setup_distrib_env()

    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(device)

    dist.init_process_group(dist.Backend.NCCL)

    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    WORLD_RANK = world_rank
    random.seed(args.seed + world_rank)

    args.sim_gpu_id = args.local_rank

    checkpoint_folder = args.checkpoint_folder
    if WORLD_RANK == 0 and not os.path.isdir(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)

    output_log_file = args.log_file
    if WORLD_RANK == 0:
        logger.add_filehandler(output_log_file)

    with construct_envs(args) as envs:

        num_recurrent_layers = args.num_recurrent_layers
        actor_critic = Policy(
            observation_space=envs.observation_spaces[0],
            action_space=envs.action_spaces[0],
            hidden_size=args.hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            blind=args.blind,
            use_aux_losses=args.use_aux_losses,
            rnn_type=args.rnn_type,
            resnet_baseplanes=args.resnet_baseplanes,
        )
        actor_critic.to(device)

        agent = PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
        )

        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        update_start_from = 0
        prev_time = 0

        if osp.exists(STATE_FILE):
            ckpt = torch.load(STATE_FILE, map_location=device)
            agent.load_state_dict(
                {k: v for k, v in ckpt["state_dict"].items() if "ddp" not in k}
            )

            env_time = ckpt["extra"]["env_time"]
            pth_time = ckpt["extra"]["pth_time"]
            count_steps = ckpt["extra"]["count_steps"]
            count_checkpoints = ckpt["extra"]["count_checkpoints"]
            update_start_from = ckpt["extra"]["update"]
            prev_time = ckpt["extra"]["prev_time"]
            output_log_file = ckpt["extra"]["output_log_file"]
            checkpoint_folder = ckpt["extra"]["checkpoint_folder"]

            if WORLD_RANK == 0:
                logger.add_filehandler(output_log_file)
                logger.info("Starting requeued job")

        agent.init_distributed()
        actor_critic = agent.actor_critic

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in actor_critic.parameters())
            )
        )

        if WORLD_RANK == 0:
            logger.info("-" * 50)
            logger.info("args:")
            for p in sorted(list(vars(args))):
                logger.info("{}: {}".format(p, getattr(args, p)))
            logger.info("-" * 50)

        observations = envs.reset()

        batch = batch_obs(observations)
        logger.info(envs.observation_spaces[0])

        rollouts = RolloutStorage(
            args.num_steps,
            envs.num_envs,
            envs.observation_spaces[0],
            envs.action_spaces[0],
            args.hidden_size,
            num_recurrent_layers=actor_critic.net.num_recurrent_layers,
        )
        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])
        rollouts.to(device)

        episode_spls = torch.zeros(envs.num_envs, 1).to(device)
        episode_successes = torch.zeros(envs.num_envs, 1).to(device)
        episode_rewards = torch.zeros(envs.num_envs, 1).to(device)
        episode_counts = torch.zeros(envs.num_envs, 1).to(device)

        current_episode_reward = torch.zeros(envs.num_envs, 1).to(device)

        window_episode_reward = deque(maxlen=args.reward_window_size)
        window_episode_counts = deque(maxlen=args.reward_window_size)
        window_episode_spl = deque(maxlen=args.reward_window_size)
        window_episode_successes = deque(maxlen=args.reward_window_size)

        t_start = time()

        tb_enabled = WORLD_RANK == 0 and args.env_name is not None
        if tb_enabled:
            writer_kwargs = dict(
                log_dir=osp.join("runs", args.env_name), purge_step=count_steps
            )

        with (
            tensorboardX.SummaryWriter(**writer_kwargs)
            if tb_enabled
            else contextlib.suppress()
        ) as writer:
            for update in range(update_start_from, args.num_updates):
                if args.use_linear_lr_decay:
                    update_linear_schedule(
                        agent.optimizer, update, args.num_updates, args.lr
                    )

                agent.clip_param = args.clip_param * (
                    1 - update / args.num_updates
                )

                actor_critic.eval()
                for step in range(args.num_steps):
                    t_sample_action = time()
                    # sample actions
                    with torch.no_grad():
                        step_observation = {
                            k: v[step]
                            for k, v in rollouts.observations.items()
                        }

                        (
                            values,
                            actions,
                            actions_log_probs,
                            recurrent_hidden_states,
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
                    rewards = torch.tensor(
                        rewards, dtype=torch.float, device=device
                    )
                    rewards = rewards.unsqueeze(1)

                    masks = torch.tensor(
                        [[0.0] if done else [1.0] for done in dones],
                        dtype=torch.float,
                        device=device,
                    )

                    current_episode_reward += rewards
                    episode_rewards += (1.0 - masks) * current_episode_reward
                    episode_counts += 1.0 - masks
                    current_episode_reward *= masks

                    if args.nav_task == "pointnav":
                        key_spl = "spl"
                    else:
                        key_spl = "loop_spl"

                    episode_spls += torch.tensor(
                        [
                            [info[key_spl]] if done else [0.0]
                            for info, done in zip(infos, dones)
                        ],
                        dtype=torch.float,
                        device=device,
                    )
                    episode_successes += torch.tensor(
                        [
                            [1.0] if done and info[key_spl] > 0 else [0.0]
                            for info, done in zip(infos, dones)
                        ],
                        dtype=torch.float,
                        device=device,
                    )

                    rollouts.insert(
                        batch,
                        recurrent_hidden_states,
                        actions,
                        actions_log_probs,
                        values,
                        rewards,
                        masks,
                    )

                    count_steps += envs.num_envs * world_size
                    pth_time += time() - t_update_stats

                stats = torch.cat(
                    [
                        episode_rewards,
                        episode_spls,
                        episode_successes,
                        episode_counts,
                    ],
                    1,
                )
                dist.all_reduce(stats)
                window_episode_reward.append(stats[:, 0])
                window_episode_spl.append(stats[:, 1])
                window_episode_successes.append(stats[:, 2])
                window_episode_counts.append(stats[:, 3])

                t_update_model = time()
                with torch.no_grad():
                    last_observation = {
                        k: v[-1] for k, v in rollouts.observations.items()
                    }
                    next_value = actor_critic.get_value(
                        last_observation,
                        rollouts.recurrent_hidden_states[-1],
                        rollouts.prev_actions[-1],
                        rollouts.masks[-1],
                    ).detach()

                rollouts.compute_returns(
                    next_value, args.use_gae, args.gamma, args.tau
                )

                actor_critic.train()
                value_loss, action_loss, dist_entropy = agent.update(rollouts)

                rollouts.after_update()
                pth_time += time() - t_update_model

                if tb_enabled:
                    stats = zip(
                        ["count", "reward", key_spl, "success"],
                        [
                            window_episode_counts,
                            window_episode_reward,
                            window_episode_spl,
                            window_episode_successes,
                        ],
                    )
                    deltas = {
                        k: (
                            (v[-1] - v[0]).sum().item()
                            if len(v) > 1
                            else v[0].sum().item()
                        )
                        for k, v in stats
                    }

                    writer.add_scalar(
                        "reward",
                        deltas["reward"] / deltas["count"],
                        count_steps,
                    )

                    writer.add_scalars(
                        "metrics",
                        {
                            k: deltas[k] / deltas["count"]
                            for k in [key_spl, "success"]
                        },
                        count_steps,
                    )

                def _save_state():
                    checkpoint = {
                        "state_dict": {
                            k: v
                            for k, v in agent.state_dict().items()
                            if "ddp" not in k
                        }
                    }
                    checkpoint["extra"] = dict(
                        prev_time=prev_time + (time() - t_start),
                        count_steps=count_steps,
                        update=update,
                        count_checkpoints=count_checkpoints,
                        pth_time=pth_time,
                        env_time=env_time,
                        output_log_file=output_log_file,
                        checkpoint_folder=checkpoint_folder
                    )
                    torch.save(checkpoint, STATE_FILE)

                if INTERRUPTED.is_set():
                    if world_rank == 0:
                        logger.info("Interrupted, REQUEUE: {}".format(
                            REQUEUE.is_set()))
                    if world_rank == 0 and REQUEUE.is_set():
                        logger.info("Saving state for requeue")
                        _save_state()

                    return

                if world_rank == 0:
                    # log stats
                    if update > 0 and update % args.log_interval == 0:
                        _save_state()

                        logger.info(
                            "update: {}\tfps: {:.3f}".format(
                                update,
                                count_steps / ((time() - t_start) + prev_time),
                            )
                        )

                        logger.info(
                            "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s"
                            "frames: {}".format(
                                update, env_time, pth_time, count_steps
                            )
                        )

                        window_rewards = (
                            window_episode_reward[-1]
                            - window_episode_reward[0]
                        ).sum()
                        window_spl = (
                            window_episode_spl[-1] - window_episode_spl[0]
                        ).sum()
                        window_successes = (
                            window_episode_successes[-1]
                            - window_episode_successes[0]
                        ).sum()
                        window_counts = (
                            window_episode_counts[-1]
                            - window_episode_counts[0]
                        ).sum()

                        if window_counts > 0:
                            logger.info(
                                "Average window size {} reward: {:3f}\t{}: {:3f}\t success: {:3f}".format(
                                    len(window_episode_reward),
                                    (window_rewards / window_counts).item(),
                                    key_spl,
                                    (window_spl / window_counts).item(),
                                    (window_successes / window_counts).item(),
                                )
                            )
                        else:
                            logger.info("No episodes finish in current window")

                    # checkpoint model
                    if update % args.checkpoint_interval == 0:
                        checkpoint = {
                            "state_dict": {
                                k: v
                                for k, v in agent.state_dict().items()
                                if "ddp" not in k
                            }
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
