#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from time import time
from collections import deque
import random
import numpy as np

import torch
import habitat
from habitat import logger
from habitat.sims.habitat_simulator import SimulatorActions, SIM_NAME_TO_ACTION
from habitat.config.default import get_config as cfg_env
from src.config.default import cfg as cfg_baseline
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets import make_dataset
from src.rl.ppo import PPO, Policy, RolloutStorage
from src.rl.ppo.utils import update_linear_schedule, ppo_args, batch_obs


class NavRLEnv(habitat.RLEnv):
    def __init__(self, config_env, config_baseline, dataset):
        self._config_env = config_env.TASK
        self._config_baseline = config_baseline
        self._previous_target_distance = None
        self._current_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        super().__init__(config_env, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        return observations

    def step(self, action):
        self._previous_action = action
        observations, reward, done, info = super().step(action)
        return observations, reward, done, info

    def get_reward_range(self):
        return (
            self._config_baseline.BASELINE.RL.SLACK_REWARD - 1.0,
            self._config_baseline.BASELINE.RL.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._config_baseline.BASELINE.RL.SLACK_REWARD

        current_target_distance = self._distance_target()

        # check for infinity geodesic distance
        self._current_target_distance = current_target_distance
        if not (-100 <= self._current_target_distance <= 100):
            logger.info("Infinite geodesic distance observed in get_reward")
            return 0.0

        reward += self._previous_target_distance - current_target_distance
        self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += (
                self._config_baseline.BASELINE.RL.SUCCESS_REWARD
                * self.get_info(observations)["spl"]
            )

        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance

    def _episode_success(self):
        if (
            self._previous_action
            == SIM_NAME_TO_ACTION[SimulatorActions.STOP.value]
            and self._distance_target() < self._config_env.SUCCESS_DISTANCE
        ):
            return True
        return False

    def get_done(self, observations):
        done = False

        # check for infinity geodesic distance
        if not (-100 <= self._current_target_distance <= 100):
            logger.info("Infinite geodesic distance observed in get_done")
            return True

        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        # info = {}

        # if self.get_done(observations):
        info = self.habitat_env.get_metrics()

        # check for infinity geodesic distance
        if not (-100 <= self._current_target_distance <= 100):
            info["spl"] = 0.0
            logger.info("Infinite geodesic distance observed in get_info")
            return info

        return info


def make_env_fn(config_env, config_baseline, shuffle_interval, rank):
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    try:
        dataset.shuffle_episodes(shuffle_interal=shuffle_interval)
    except:
        pass
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config_env.freeze()
    env = NavRLEnv(
        config_env=config_env, config_baseline=config_baseline, dataset=dataset
    )
    env.seed(rank)
    return env


def construct_envs(args):
    env_configs = []
    baseline_configs = []

    basic_config = cfg_env(config_file=args.task_config)

    scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)

    if len(scenes) > 0:
        random.shuffle(scenes)

        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )
        scene_split_size = int(np.floor(len(scenes) / args.num_processes))

    for i in range(args.num_processes):
        config_env = cfg_env(config_file=args.task_config)
        config_env.defrost()

        if len(scenes) > 0:
            config_env.DATASET.POINTNAVV1.CONTENT_SCENES = scenes[
                i * scene_split_size : (i + 1) * scene_split_size
            ]

        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.sim_gpu_id
        config_env.TASK.POINTGOAL_SENSOR.SENSOR_TYPE = (
            args.pointgoal_sensor_type
        )
        config_env.TASK.POINTGOAL_SENSOR.SENSOR_DIMENSIONS = (
            args.pointgoal_sensor_dimensions
        )
        config_env.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = (
            args.pointgoal_sensor_format
        )

        agent_sensors = [
            s for s in args.sensors.strip().split(",") if len(s) > 0
        ]
        for sensor in agent_sensors:
            assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]
        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors

        if args.blind:
            config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = 2
            config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = 2

            config_env.SIMULATOR.RGB_SENSOR.HEIGHT = 2
            config_env.SIMULATOR.RGB_SENSOR.WIDTH = 2

        config_env.freeze()
        env_configs.append(config_env)

        config_baseline = cfg_baseline()
        baseline_configs.append(config_baseline)

        logger.info("config_env: {}".format(config_env))

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(
                    env_configs,
                    baseline_configs,
                    [args.shuffle_interval for _ in range(args.num_processes)],
                    range(args.num_processes),
                )
            )
        ),
    )

    return envs


def main():
    parser = ppo_args()
    args = parser.parse_args()

    random.seed(args.seed)

    device = torch.device("cuda:{}".format(args.pth_gpu_id))

    logger.add_filehandler(args.log_file)

    if not os.path.isdir(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)

    for p in sorted(list(vars(args))):
        logger.info("{}: {}".format(p, getattr(args, p)))

    envs = construct_envs(args)

    actor_critic = Policy(
        observation_space=envs.observation_spaces[0],
        action_space=envs.action_spaces[0],
        net=args.net,
        hidden_size=args.hidden_size,
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

    start_update = 0

    if args.load_ckpt is not None:
        ckpt = torch.load(args.load_ckpt, map_location=device)
        agent.load_state_dict(ckpt["state_dict"])
        actor_critic = agent.actor_critic
        start_update = args.iters_load_ckpt

    logger.info(
        "agent number of parameters: {}".format(
            sum(param.numel() for param in agent.parameters())
        )
    )

    observations = envs.reset()

    batch = batch_obs(observations)

    rollouts = RolloutStorage(
        args.num_steps,
        envs.num_envs,
        envs.observation_spaces[0],
        envs.action_spaces[0],
        args.hidden_size,
    )
    for sensor in rollouts.observations:
        rollouts.observations[sensor][0].copy_(batch[sensor])
    rollouts.to(device)

    episode_rewards = torch.zeros(envs.num_envs, 1)
    episode_spl = torch.zeros(envs.num_envs, 1)
    episode_success = torch.zeros(envs.num_envs, 1)
    episode_counts = torch.zeros(envs.num_envs, 1)
    current_episode_reward = torch.zeros(envs.num_envs, 1)

    window_episode_reward = deque()
    window_episode_spl = deque()
    window_episode_success = deque()
    window_episode_counts = deque()

    t_start = time()
    env_time = 0
    pth_time = 0
    count_steps = 0
    count_checkpoints = 0

    logger.info(
        "start_update: {}, num_updates: {}".format(
            start_update, args.num_updates
        )
    )

    for update in range(start_update, args.num_updates):
        if args.use_linear_lr_decay:
            update_linear_schedule(
                agent.optimizer, update, args.num_updates, args.lr
            )

        agent.clip_param = args.clip_param * (1 - update / args.num_updates)

        for step in range(args.num_steps):
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
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    step_observation,
                    rollouts.recurrent_hidden_states[step],
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
            rewards = torch.tensor(rewards, dtype=torch.float)
            rewards = rewards.unsqueeze(1)

            masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones], dtype=torch.float
            )

            # update SPL and success
            for ep_i in range(envs.num_envs):
                if dones[ep_i]:
                    episode_spl[ep_i] += infos[ep_i]["spl"]
                    if infos[ep_i]["spl"] > 0:
                        episode_success[ep_i] += 1.0

            # update rewards and episode counts
            current_episode_reward += rewards
            episode_rewards += (1 - masks) * current_episode_reward
            episode_counts += 1 - masks
            current_episode_reward *= masks

            rollouts.insert(
                batch,
                recurrent_hidden_states,
                actions,
                actions_log_probs,
                values,
                rewards,
                masks,
            )

            count_steps += envs.num_envs
            pth_time += time() - t_update_stats

        if len(window_episode_reward) == args.reward_window_size:
            window_episode_reward.popleft()
            window_episode_spl.popleft()
            window_episode_success.popleft()
            window_episode_counts.popleft()

        window_episode_reward.append(episode_rewards.clone())
        window_episode_spl.append(episode_spl.clone())
        window_episode_success.append(episode_success.clone())
        window_episode_counts.append(episode_counts.clone())

        t_update_model = time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts.observations.items()
            }
            next_value = actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(
            next_value, args.use_gae, args.gamma, args.tau
        )

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()
        pth_time += time() - t_update_model

        # log stats
        if update > 0 and update % args.log_interval == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    update, count_steps / (time() - t_start)
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(update, env_time, pth_time, count_steps)
            )

            window_rewards = (
                window_episode_reward[-1] - window_episode_reward[0]
            ).sum()

            window_spl = (window_episode_spl[-1] - window_episode_spl[0]).sum()

            window_success = (
                window_episode_success[-1] - window_episode_success[0]
            ).sum()

            window_counts = (
                window_episode_counts[-1] - window_episode_counts[0]
            ).sum()

            window_counts = (
                window_episode_counts[-1] - window_episode_counts[0]
            ).sum()

            if window_counts > 0:
                logger.info(
                    "Average window size {}, reward: {:.3f}, spl: {:.3f}, "
                    "success: {:.3f}".format(
                        len(window_episode_reward),
                        (window_rewards / window_counts).item(),
                        (window_spl / window_counts).item(),
                        (window_success / window_counts).item(),
                    )
                )
            else:
                logger.info("No episodes finish in current window")

        # checkpoint model
        if update % args.checkpoint_interval == 0 and update > 0:
            checkpoint = {"state_dict": agent.state_dict()}

            window_rewards = (
                window_episode_reward[-1] - window_episode_reward[0]
            ).sum()
            window_counts = (
                window_episode_counts[-1] - window_episode_counts[0]
            ).sum()
            avg_reward = (window_rewards / window_counts).item()
            if np.isnan(avg_reward):
                avg_reward = 0

            torch.save(
                checkpoint,
                os.path.join(
                    args.checkpoint_folder,
                    "ckpt.{}.reward.{:.3f}.pth".format(
                        avg_reward, count_checkpoints
                    ),
                ),
            )

            count_checkpoints += 1


if __name__ == "__main__":
    main()
