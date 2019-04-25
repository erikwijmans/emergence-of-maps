#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import random

import imageio
import torch
import torch.nn.functional as F
import tqdm
from src.config.default import cfg as cfg_baseline
from src.rl.ppo import PPO, Policy
from src.rl.ppo.utils import batch_obs
from train_ppo import make_env_fn

import habitat
from habitat.config.default import get_config
import numpy as np

from src.train_ppo import make_env_fn
from src.rl.ppo import PPO, Policy
from src.rl.ppo.utils import batch_obs
import tqdm
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

import numpy as np
from habitat.tasks.nav.nav_task import NavigationEpisode, NavigationGoal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--sim-gpu-id", type=int, required=True)
    parser.add_argument("--pth-gpu-id", type=int, required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-recurrent-layers", type=int, default=1)
    parser.add_argument(
        "--sensors",
        type=str,
        default="RGB_SENSOR,DEPTH_SENSOR",
        help="comma separated string containing different"
        "sensors to use, currently 'RGB_SENSOR' and"
        "'DEPTH_SENSOR' are supported",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default="tasks/pointnav.yaml",
        help="path to config yaml containing information about task",
    )
    parser.add_argument("--blind", type=int, default=0)
    parser.add_argument(
        "--pointgoal-sensor-type",
        type=str,
        default="DENSE",
        choices=["DENSE", "SPARSE"],
    )
    parser.add_argument(
        "--rnn-type", type=str, default="GRU", choices=["GRU", "LSTM"]
    )
    parser.add_argument("--count-test-episodes", type=int, default=1000)
    parser.add_argument("--pointgoal-sensor-dimensions", type=int, default=2)
    parser.add_argument("--pointgoal-sensor-format", type=str, default="POLAR")
    parser.add_argument("--resnet-baseplanes", type=int, default=32)

    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.pth_gpu_id))

    env_configs = []
    baseline_configs = []

    for _ in range(args.num_processes):
        config_env = get_config(config_file=args.task_config)
        config_env.defrost()
        config_env.DATASET.SPLIT = "val"
        config_env.DATASET.TYPE = "PointNav-v1"

        agent_sensors = args.sensors.strip().split(",")
        for sensor in agent_sensors:
            assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]

        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors
        config_env.SIMULATOR.HABITAT_SIM_V0.COMPRESS_TEXTURES = False

        config_env.TASK.POINTGOAL_SENSOR.SENSOR_TYPE = (
            args.pointgoal_sensor_type
        )
        config_env.TASK.POINTGOAL_SENSOR.SENSOR_DIMENSIONS = (
            args.pointgoal_sensor_dimensions
        )

        config_env.freeze()
        env_configs.append(config_env)

        config_baseline = cfg_baseline()
        baseline_configs.append(config_baseline)

    random.seed(env_configs[0].SEED)
    torch.random.manual_seed(env_configs[0].SEED)
    torch.backends.cudnn.deterministic = True
    dummy_dataset = PointNavDatasetV1(env_configs[0].DATASET)
    args.count_test_episodes = len(dummy_dataset.episodes)

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(
                    env_configs,
                    baseline_configs,
                    [10000 for _ in range(args.num_processes)],
                    range(args.num_processes),
                )
            )
        ),
    )

    ckpt = torch.load(args.model_path, map_location=device)
    print(envs.observation_spaces[0])

    actor_critic = Policy(
        observation_space=envs.observation_spaces[0],
        action_space=envs.action_spaces[0],
        hidden_size=args.hidden_size,
        num_recurrent_layers=args.num_recurrent_layers,
        blind=args.blind,
        use_aux_losses=False,
        rnn_type=args.rnn_type,
        resnet_baseplanes=args.resnet_baseplanes,
    )
    actor_critic.to(device)
    if args.blind:
        assert actor_critic.net.cnn is None

    ppo = PPO(
        actor_critic=actor_critic,
        clip_param=0.1,
        ppo_epoch=4,
        num_mini_batch=32,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        lr=2.5e-4,
        eps=1e-5,
        max_grad_norm=0.5,
    )

    ppo.load_state_dict(
        {k: v for k, v in ckpt["state_dict"].items() if "ddp" not in k}
    )

    actor_critic = ppo.actor_critic
    actor_critic.eval()

    observations = envs.reset()
    batch = batch_obs(observations)
    for sensor in batch:
        batch[sensor] = batch[sensor].to(device)

    episode_rewards = torch.zeros(envs.num_envs, 1, device=device)
    episode_spls = torch.zeros(envs.num_envs, 1, device=device)
    episode_success = torch.zeros(envs.num_envs, 1, device=device)
    episode_counts = torch.zeros(envs.num_envs, 1, device=device)
    current_episode_reward = torch.zeros(envs.num_envs, 1, device=device)

    test_recurrent_hidden_states = torch.zeros(
        actor_critic.net.num_recurrent_layers,
        args.num_processes,
        args.hidden_size,
        device=device,
    )
    not_done_masks = torch.zeros(args.num_processes, 1, device=device)
    prev_actions = torch.zeros(
        args.num_processes, 1, device=device, dtype=torch.int64
    )

    with tqdm.tqdm(total=args.count_test_episodes * envs.num_envs) as pbar:
        while (
            episode_counts < args.count_test_episodes
        ).float().sum().item() > 0:
            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states = actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)

            outputs = envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations)
            for sensor in batch:
                batch[sensor] = batch[sensor].to(device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=device
            ).unsqueeze(1)
            current_episode_reward += rewards

            for i in range(not_done_masks.shape[0]):
                if (
                    not_done_masks[i].item() == 0
                    and episode_counts[i] < args.count_test_episodes
                ):
                    pbar.update()

                    episode_counts[i] += 1
                    episode_rewards[i] += current_episode_reward[i]

                    episode_spls[i] += infos[i]["spl"]
                    if infos[i]["spl"] > 0:
                        episode_success[i] += 1

                    pbar.set_postfix(
                        dict(
                            spl=(
                                episode_spls.sum() / episode_counts.sum()
                            ).item()
                        ),
                        refresh=True,
                    )

            current_episode_reward *= not_done_masks

    episode_reward_mean = (episode_rewards / episode_counts).mean().item()
    episode_spl_mean = (episode_spls / episode_counts).mean().item()
    episode_success_mean = (episode_success / episode_counts).mean().item()

    print("Average episode reward: {:.6f}".format(episode_reward_mean))
    print("Average episode success: {:.6f}".format(episode_success_mean))
    print("Average episode spl: {:.6f}".format(episode_spl_mean))


if __name__ == "__main__":
    main()
