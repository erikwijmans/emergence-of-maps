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
import numpy as np
import json


import threading
from collections import deque
from time import sleep, time

import torch
import torch.distributed as dist
from torch.utils import tensorboard

from habitat import logger
from nav_analysis.rl.ppo import PPO, Policy, RolloutStorage
from nav_analysis.rl.ppo.utils import batch_obs, ppo_args, update_linear_schedule
from nav_analysis.train_ppo import construct_envs


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def before_exit():
    dist.barrier()


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
    parser.add_argument("--backprop", type=int, default=1)
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
    torch.manual_seed(args.seed + world_rank)
    torch.cuda.manual_seed_all(args.seed + world_rank)
    np.random.seed(args.seed + world_rank)

    args.sim_gpu_id = args.local_rank

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
            backbone=args.backbone,
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
            weight_decay=args.weight_decay,
        )

        env_time = 0
        pth_time = 0
        sync_time = 0
        count_steps = 0

        agent.init_distributed()
        actor_critic = agent.actor_critic

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

        t_start = time()
        for update in range(0, args.num_updates):
            if args.use_linear_lr_decay:
                update_linear_schedule(
                    agent.optimizer, update, args.num_updates, args.lr
                )

            if args.use_linear_clip_decay:
                agent.clip_param = args.clip_param * (1 - update / args.num_updates)

            actor_critic.eval()
            for step in range(args.num_steps):
                count_steps += envs.num_envs * world_size
                t_sample_action = time()
                # sample actions

                if args.backprop:
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
                        ) = actor_critic.act(
                            step_observation,
                            rollouts.recurrent_hidden_states[step],
                            rollouts.prev_actions[step],
                            rollouts.masks[step],
                        )
                else:
                    actions = torch.randint_like(rollouts.prev_actions[step], 0, 4)

                pth_time += time() - t_sample_action

                t_step_env = time()

                outputs = envs.step([a[0].item() for a in actions])
                observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

                env_time += time() - t_step_env

                if not args.backprop:
                    continue

                t_update_stats = time()
                batch = batch_obs(observations)
                rewards = torch.tensor(rewards, dtype=torch.float, device=device)
                rewards = rewards.unsqueeze(1)

                masks = torch.tensor(
                    [[0.0] if done else [1.0] for done in dones],
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
                    entropy,
                )

                pth_time += time() - t_update_stats

            t_sync = time()
            dist.barrier()
            t_sync = torch.tensor(time() - t_sync, device=device, dtype=torch.float32)
            dist.all_reduce(t_sync, op=dist.ReduceOp.MAX)
            sync_time += t_sync.item()

            t_update_model = time()

            if args.backprop:
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

                rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

                actor_critic.train()
                value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

            pth_time += time() - t_update_model

            if WORLD_RANK == 0:
                if update < args.num_updates - 1:
                    logger.info(
                        "FPS={:.3f}\tsync-frac={:.3f}".format(
                            count_steps / (time() - t_start),
                            (sync_time) / (time() - t_start),
                        )
                    )
                else:
                    res = dict(
                        fps=count_steps / (time() - t_start),
                        sync_frac=sync_time / (time() - t_start),
                    )
                    print(json.dumps(res))


if __name__ == "__main__":
    main()

    before_exit()
