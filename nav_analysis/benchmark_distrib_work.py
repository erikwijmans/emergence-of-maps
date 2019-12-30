#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import json
import logging
import os
import os.path as osp
import random
import signal
import socket
import sys
import threading
from collections import deque
from time import sleep, time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils import tensorboard

from habitat import logger
from nav_analysis.rl.ppo import PPO, Policy, RolloutStorage
from nav_analysis.rl.ppo.utils import (
    batch_obs,
    ppo_args,
    update_linear_schedule,
)
from nav_analysis.train_ppo import construct_envs

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args, is_done_store):
    global WORLD_RANK

    device = torch.device("cuda", args.general.local_rank)
    torch.cuda.set_device(device)

    world_size = dist.get_world_size()
    world_rank = dist.get_rank()

    WORLD_RANK = world_rank

    random.seed(args.general.seed + world_rank)
    torch.manual_seed(args.general.seed + world_rank)
    torch.cuda.manual_seed_all(args.general.seed + world_rank)
    np.random.seed(args.general.seed + world_rank)

    if args.ddppo.sync_frac < 1.0:
        if WORLD_RANK == 0:
            is_done_store.set(f"num_done", "0")

    args.general.sim_gpu_id = args.general.local_rank

    with construct_envs(args, one_scene=True) as envs:

        num_recurrent_layers = args.model.num_recurrent_layers
        actor_critic = Policy(
            observation_space=envs.observation_spaces[0],
            action_space=envs.action_spaces[0],
            hidden_size=args.model.hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            blind=args.model.blind,
            use_aux_losses=args.model.use_aux_losses,
            rnn_type=args.model.rnn_type,
            resnet_baseplanes=args.model.resnet_baseplanes,
            backbone=args.model.backbone,
        )
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
        rollout_time = 0
        opt_time = 0
        count_steps = torch.tensor(0, dtype=torch.long, device=device)

        agent.init_distributed()
        actor_critic = agent.actor_critic

        observations = envs.reset()

        batch = batch_obs(observations)
        logger.info(envs.observation_spaces[0])

        rollouts = RolloutStorage(
            args.ppo.num_steps,
            envs.num_envs,
            envs.observation_spaces[0],
            envs.action_spaces[0],
            args.model.hidden_size,
            num_recurrent_layers=actor_critic.net.num_recurrent_layers,
        )
        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])
        rollouts.to(device)

        t_start = time()
        for update in range(0, args.ppo.num_updates):
            if update == args.ppo.num_updates // 2:
                t_start = time()
                env_time = 0
                pth_time = 0
                sync_time = 0
                rollout_time = 0
                opt_time = 0
                count_steps = torch.tensor(0, dtype=torch.long, device=device)

            if args.ppo.use_linear_lr_decay:
                update_linear_schedule(
                    agent.optimizer, update, args.ppo.num_updates, args.optimizer.lr
                )

            if args.ppo.use_linear_clip_decay:
                agent.clip_param = args.ppo.clip_param * (
                    1 - update / args.ppo.num_updates
                )

            actor_critic.eval()
            t_rollout_start = time()
            for step in range(args.ppo.num_steps):
                t_sample_action = time()
                # sample actions

                if args.general.backprop:
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

                if args.general.backprop:
                    t_update_stats = time()
                    batch = batch_obs(observations)
                    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
                    rewards = rewards.unsqueeze(1)

                    masks = torch.ones(
                        (len(dones), 1), dtype=torch.float32, device=device
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

                if (
                    args.ddppo.sync_frac < 1.0
                    and (step + 1) >= (args.ppo.num_steps / 4)
                    and int(is_done_store.get("num_done"))
                    > (world_size * args.ddppo.sync_frac)
                ):
                    break

            if args.ddppo.sync_frac < 1.0:
                is_done_store.add("num_done", 1)

            t_sync = time()
            dist.barrier()
            rollout_time += time() - t_rollout_start
            t_sync = torch.tensor(time() - t_sync, device=device)
            dist.all_reduce(t_sync)
            sync_time += t_sync.item() / world_size

            step_delta = torch.full_like(count_steps, (step + 1) * envs.num_envs)
            dist.all_reduce(step_delta)
            count_steps += step_delta

            t_update_model = time()

            if args.general.backprop:
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

            rollouts.after_update()

            pth_time += time() - t_update_model
            opt_time += time() - t_update_model

            if args.ddppo.sync_frac < 1.0:
                torch.cuda.synchronize()
                if WORLD_RANK == 0:
                    is_done_store.set("num_done", "0")

            if WORLD_RANK == 0:
                logger.info(
                    "FPS={:.3f}\tsync-frac={:.3f}".format(
                        count_steps.item() / (time() - t_start),
                        (sync_time) / (time() - t_start),
                    )
                )
                logger.info(
                    "rollout-time={:.3f}\topt-time={:.3f}".format(
                        rollout_time, opt_time
                    )
                )

            res = dict(
                seed=args.general.seed,
                sync_frac=args.ddppo.sync_frac,
                ngpu=args.general.ngpu,
                fps=count_steps.item() / (time() - t_start),
                sync_amount=sync_time / (time() - t_start),
            )

    return res
