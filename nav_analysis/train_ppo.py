#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import os.path as osp
import random
from collections import deque
from time import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import habitat
import habitat_sim
from habitat_sim.utils.common import quat_from_two_vectors, angle_between_quats

import nav_analysis
from habitat import logger
from habitat.config.default import get_config as cfg_env
from habitat.datasets import make_dataset
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.sims.habitat_simulator import SimulatorActions
from nav_analysis.config.default import cfg as cfg_baseline
from nav_analysis.rl import splitnet_nav_envs
from nav_analysis.rl.ppo import PPO, Policy, RolloutStorage
from nav_analysis.rl.ppo.utils import (
    batch_obs,
    ppo_args,
    update_linear_schedule,
)
from gym import spaces

CFG_DIR = osp.join(osp.dirname(nav_analysis.__file__), "configs")


def filter_obs(obs, give_obs):
    if give_obs:
        return obs

    always_keep_sensors = {"episode_stage", "initial_hidden_state", "pointgoal"}
    for k, v in obs.items():
        if k not in always_keep_sensors:
            obs[k] = np.zeros_like(v)

    return obs


def heading_error(env, spath):
    if len(spath.points) < 2:
        return 0.0

    gt_heading = quat_from_two_vectors(
        habitat_sim.geo.FRONT,
        spath.points[1] - spath.points[0] + 1e-5 * habitat_sim.geo.LEFT,
    )

    return angle_between_quats(env.sim.get_agent_state().rotation, gt_heading) / np.pi


class ObsStackEnv(habitat.RLEnv):
    def __init__(self, env: habitat.RLEnv, num_stacks: int = 1):
        self._env = env
        self.num_stacks = num_stacks
        self._stack = {}
        self._flattened_stack = {}

        self.observation_space = self._env.observation_space
        for k, v in self.observation_space.spaces.items():
            new_shape = list(v.shape) + [num_stacks]
            v.shape = tuple(new_shape)
            self.observation_space.spaces[k] = v

        self.observation_space.spaces["prev_action"] = spaces.Box(
            low=0, high=100, shape=(1, num_stacks), dtype=np.int64
        )

        self.action_space = self._env.action_space

    def reset(self):
        for k, v in self._stack.items():
            self._stack[k] = deque(maxlen=self.num_stacks)

        for k, v in self._flattened_stack.items():
            self._flattened_stack[k] = np.zeros_like(v)

        observations = self._env.reset()
        observations["prev_action"] = np.array([self.action_space.n])

        self.update_stack(observations)

        return self.flatten_stack()

    def update_stack(self, observations):
        for k, v in observations.items():
            if k not in self._stack:
                self._stack[k] = deque(maxlen=self.num_stacks)

            self._stack[k].append(v)

    def flatten_stack(self):
        for k, v in self._stack.items():
            if k not in self._flattened_stack:
                self._flattened_stack[k] = np.stack(
                    [np.zeros_like(v[0]) for _ in range(self.num_stacks)], axis=-1
                )

            for i, ele in enumerate(v):
                self._flattened_stack[k][..., i] = ele

        return self._flattened_stack

    def step(self, action):
        observations, reward, done, info = self._env.step(action)
        observations["prev_action"] = np.array([action + 1])
        self.update_stack(observations)

        return self.flatten_stack(), reward, done, info

    def close(self):
        self._env.close()


class NavRLEnv(habitat.RLEnv):
    def __init__(self, config_env, config_baseline, dataset):
        self._config_env = config_env.TASK
        self._config_baseline = config_baseline
        self._previous_target_distance = None
        self._current_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        self._give_obs = self._config_env.LOOPNAV_GIVE_RETURN_OBS
        self._heading_reward = not self._config_env.LOOPNAV_GIVE_RETURN_OBS

        if config_env.TASK.VERBOSE is True:
            logger.add_filehandler(os.environ.get("LOG_FILE"))

        super().__init__(config_env, dataset)

    def reset(self):
        self._previous_action = None

        observations = super().reset()

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        if self._heading_reward:
            self._previous_heading_error = heading_error(self._env, self._spath())

        if self._config_env.VERBOSE is True:
            agent_state = self._env.sim.get_agent_state()
            logger.info("CURRENT_EPISODE: {}".format(self._env.current_episode))
            logger.info("START_POSITION A: {}".format(agent_state.position))
            logger.info("START_ROTATION A: {}".format(agent_state.rotation))
            logger.info(
                "GOAL_POSITION A: {}".format(
                    self._env.current_episode.goals[0].position
                )
            )

        # Do not filter on reset!
        #  observations = filter_obs(observations, self._give_obs)

        return observations

    def step(self, action):
        self._previous_action = action
        observations, reward, done, info = super().step(action)

        observations = filter_obs(observations, self._give_obs)
        return observations, reward, done, info

    def get_reward_range(self):
        return (
            self._config_baseline.BASELINE.RL.SLACK_REWARD - 1.0,
            self._config_baseline.BASELINE.RL.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._config_baseline.BASELINE.RL.SLACK_REWARD

        spath = self._spath()
        current_target_distance = spath.geodesic_distance

        # check for infinity geodesic distance
        self._current_target_distance = current_target_distance
        if not (-100 <= self._current_target_distance <= 100):
            logger.info("Infinite geodesic distance observed in get_reward")
            return 0.0

        if self._config_env.LOOPNAV_GIVE_RETURN_OBS:
            reward += 1.0 * (self._previous_target_distance - current_target_distance)
        else:
            reward += 10.0 * (
                (self._previous_target_distance - current_target_distance)
                / self._env.current_episode.info["geodesic_distance"]
            )

        self._previous_target_distance = current_target_distance

        if self._heading_reward:
            new_heading_error = heading_error(self._env, spath)
            reward += 0.25 * (new_heading_error - self._previous_heading_error)
            self._previous_heading_error = new_heading_error

        if self._episode_success():
            reward += self._config_baseline.BASELINE.RL.SUCCESS_REWARD

        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(current_position, target_position)
        return distance

    def _spath(self):
        current_position = self._env.sim.get_agent_state().position
        target_position = self._env.current_episode.goals[0].position
        return self._env.sim.shortest_path(current_position, target_position)

    def _episode_success(self):
        if (
            self._previous_action == SimulatorActions.STOP.value
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


class LoopNavRLEnv(NavRLEnv):
    def __init__(self, config_env, config_baseline, dataset, task):
        self._episode_stage = None
        self._stages_successful = []
        self._sparse_goal_sensor = (
            config_env.TASK.POINTGOAL_SENSOR.SENSOR_TYPE == "SPARSE"
        )
        self.task = task
        super().__init__(config_env, config_baseline, dataset)

        self._give_obs = True

    def reset(self):
        self._episode_stage = 0
        self._stages_successful = [False, False]
        self._give_obs = True
        output = super().reset()
        return output

    def step(self, action):

        curr_episode_over = False
        teleport = False

        if action == SimulatorActions.STOP.value:
            if self._episode_stage == 0:
                if self._previous_target_distance < self._config_env.SUCCESS_DISTANCE:
                    # zeroth stage is successful
                    self._stages_successful[0] = True

                    if self._config_env.VERBOSE is True:
                        agent_state = self._env.sim.get_agent_state()
                        logger.info(
                            "STAGE-1 ENDING_POSITION: {}".format(agent_state.position)
                        )
                        logger.info(
                            "STAGE-1 ENDING_ROTATION: {}".format(agent_state.rotation)
                        )
                else:
                    curr_episode_over = True

                self._orig_goal = list(self._env.current_episode.goals[0].position)
                self._orig_start = list(self._env.current_episode.start_position)
                # swap start position and goal for stage-1

                if self.task == "loopnav":
                    if not self._sparse_goal_sensor:
                        (
                            self._env.current_episode.start_position,
                            self._env.current_episode.goals[0].position,
                        ) = (
                            self._env.current_episode.goals[0].position,
                            self._env.current_episode.start_position,
                        )
                    else:
                        # If we are doing the static point goal sensor thing, only put the goal at the start!
                        self._env.current_episode.goals[
                            0
                        ].position = self._env.current_episode.start_position
                else:
                    teleport = self._stages_successful[0]

                self._previous_target_distance = self._distance_target()
            else:
                curr_episode_over = True

                if self._previous_target_distance < self._config_env.SUCCESS_DISTANCE:
                    self._stages_successful[1] = True

                    if self._config_env.VERBOSE is True:
                        agent_state = self._env.sim.get_agent_state()
                        logger.info(
                            "STAGE-2 ENDING_POSITION: {}".format(agent_state.position)
                        )
                        logger.info(
                            "STAGE-2 ENDING_ROTATION: {}".format(agent_state.rotation)
                        )

        observations, reward, done, info = super().step(action)

        if teleport:
            self._env.sim.set_agent_state(
                self._env.current_episode.start_position,
                self._env.sim.get_agent_state().rotation,
            )

            sim_obs = self._env.sim._sim.get_sensor_observations()
            observations = self._env.sim._sensor_suite.get_observations(sim_obs)
            observations.update(
                self._env.task.sensor_suite.get_observations(
                    observations=observations, episode=self._env.current_episode
                )
            )
            self._previous_target_distance = self._distance_target()

        # Do this all after getting observations to world-model probes their first input
        if (
            action == SimulatorActions.STOP.value and self._episode_stage == 0
        ) and not self._config_env.LOOPNAV_GIVE_RETURN_OBS:
            self._give_obs = False

        # update episode stage
        if action == SimulatorActions.STOP.value and self._episode_stage == 0:
            self._episode_stage = 1

        if curr_episode_over:
            done = True
            self._env.episode_over = True
            self._env.current_episode.goals[0].position = self._orig_goal
            self._env.current_episode.start_position = self._orig_start

        return observations, reward, done, info

    def get_reward(self, observations):
        reward = self._config_baseline.BASELINE.RL.SLACK_REWARD

        spath = self._spath()
        current_target_distance = spath.geodesic_distance

        # check for infinity geodesic distance
        self._current_target_distance = current_target_distance
        if not (-100 <= self._current_target_distance <= 1000):
            logger.info("Infinite geodesic distance observed in get_reward")
            return 0.0

        reward += 10.0 * (
            (self._previous_target_distance - current_target_distance)
            / self._env.current_episode.info["geodesic_distance"]
        )
        self._previous_target_distance = current_target_distance

        if self._heading_reward:
            new_heading_error = heading_error(self._env, spath)
            reward += 0.25 * (new_heading_error - self._previous_heading_error)
            self._previous_heading_error = new_heading_error

        if self._episode_success():
            reward = self._config_baseline.BASELINE.RL.SUCCESS_REWARD
        elif (
            self._previous_action == SimulatorActions.STOP.value
            and self._stages_successful[0]
            and self._episode_stage == 0
        ):
            reward = self._config_baseline.BASELINE.RL.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        if (
            self._episode_stage == 1
            and self._previous_action == SimulatorActions.STOP.value
            and self._distance_target() < self._config_env.SUCCESS_DISTANCE
            and self._stages_successful[0]
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
        info = self.habitat_env.get_metrics()

        # check for infinity geodesic distance
        if not (-100 <= self._current_target_distance <= 100):
            info["loop_spl"] = 0.0
            logger.info("Infinite geodesic distance observed in get_info")
            print("Infinite geodesic distance observed in get_info", flush=True)

        return info


def make_env_fn(args, config_env, config_baseline, shuffle_interval, rank):
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    try:
        dataset.shuffle_episodes(shuffle_interal=shuffle_interval)
    except AttributeError:
        pass
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config_env.freeze()

    task = args.task.nav_task
    train_jointly = args.task.training_stage == -1
    if task in ["loopnav", "teleportnav"] and train_jointly:
        env = LoopNavRLEnv(
            config_env=config_env,
            config_baseline=config_baseline,
            dataset=dataset,
            task=task,
        )
    elif task == "flee":
        env = splitnet_nav_envs.RunAwayRLEnv(config_env, dataset)
    elif task == "explore":
        env = splitnet_nav_envs.ExplorationRLEnv(config_env, dataset)
    else:
        env = NavRLEnv(
            config_env=config_env, config_baseline=config_baseline, dataset=dataset,
        )

    env.seed(rank)

    if args.model.max_memory_length:
        env = ObsStackEnv(env, args.model.max_memory_length)

    return env


def construct_envs(
    args, split="train", one_scene=False, dset_measures=False, scenes=None
):
    env_configs = []
    baseline_configs = []

    if scenes is None:
        basic_config = cfg_env(config_file=args.task.task_config, config_dir=CFG_DIR)

        basic_config.defrost()
        basic_config.DATASET.SPLIT = split
        basic_config.freeze()
        print(basic_config.DATASET)
        scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)

    random.shuffle(scenes)
    scene_splits = [[] for _ in range(args.ppo.num_processes)]

    if len(scenes) < args.ppo.num_processes:
        scenes_per_proc = math.ceil(len(scenes) / args.ppo.num_processes)

        idx = 0
        for proc_id in range(args.ppo.num_processes):
            for _ in range(scenes_per_proc):
                scene_splits[proc_id].append(scenes[idx])

                idx = (idx + 1) % len(scenes)
                if idx == 0:
                    random.shuffle(scenes)
    else:
        for idx, s in enumerate(scenes):
            scene_splits[idx % args.ppo.num_processes].append(s)

    #  scene_splits = [scenes[0:1] for _ in range(args.ppo.num_processes)]
    for i in range(args.ppo.num_processes):
        config_env = cfg_env(config_file=args.task.task_config, config_dir=CFG_DIR)
        config_env.defrost()

        config_env.DATASET.SPLIT = split
        if split == "val":
            config_env.DATASET.TYPE = "PointNav-v1"
        else:
            config_env.DATASET.TYPE = "PointNavOTF-v1"

        if args.task.training_stage == 2:
            config_env.DATASET.TYPE = "Stage2"
            config_env.DATASET.POINTNAVV1.EPISODE_PATH = osp.realpath(
                osp.join(osp.dirname(args.stage_2_args.stage_1_model), "episodes")
            )
            config_env.DATASET.POINTNAVV1.TASK_TYPE = args.stage_2_args.stage_2_task
            config_env.TASK.SENSORS = list(
                set(config_env.TASK.SENSORS + ["INITIAL_HIDDEN_STATE"])
            )

            config_env.TASK.INITIAL_HIDDEN_STATE.SHAPE = (
                args.model.num_recurrent_layers * 2,
                args.model.hidden_size,
            )
            config_env.TASK.INITIAL_HIDDEN_STATE.STATE_TYPE = (
                args.stage_2_args.state_type
            )

        config_env.DATASET.POINTNAVV1.CONTENT_SCENES = scene_splits[i]

        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.general.sim_gpu_id
        config_env.TASK.POINTGOAL_SENSOR.SENSOR_TYPE = args.task.pointgoal_sensor_type
        config_env.TASK.POINTGOAL_SENSOR.SENSOR_DIMENSIONS = (
            args.task.pointgoal_sensor_dimensions
        )
        config_env.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = args.task.pointgoal_sensor_format

        agent_sensors = args.task.agent_sensors
        for sensor in agent_sensors:
            assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]

        if args.model.blind:
            agent_sensors = []

        config_env.SIMULATOR.AGENT_0.SENSORS = list(agent_sensors)
        config_env.TASK.LOOPNAV_GIVE_RETURN_OBS = args.task.loopnav_give_return_inputs

        if args.task.nav_task in ["loopnav", "teleportnav"]:
            config_env.SIMULATOR.AGENT_0.TURNAROUND = args.task.training_stage == -1
            config_env.TASK.MEASUREMENTS = list(
                set(config_env.TASK.MEASUREMENTS + ["LOOPSPL", "LOOP_D_DELTA"])
            )
            config_env.TASK.LOOPSPL.BREAKDOWN_METRIC = True

            if args.task.nav_task == "teleportnav":
                config_env.TASK.LOOPSPL.TELEPORT = True
                config_env.TASK.LOOP_D_DELTA.TELEPORT = True
            else:
                config_env.TASK.LOOPSPL.TELEPORT = False
                config_env.TASK.LOOP_D_DELTA.TELEPORT = False
        else:
            config_env.TASK.MEASUREMENTS = list(
                set(config_env.TASK.MEASUREMENTS + ["SPL"])
            )

        if dset_measures:
            config_env.TASK.MEASUREMENTS = list(
                set(
                    config_env.TASK.MEASUREMENTS
                    + [
                        "EGO_POSE",
                        "GOAL_POSE",
                        "COLLISIONS",
                        #  "TOP_DOWN_OCCUPANCY_GRID",
                        #  "GEO_DISTANCES",
                    ]
                )
            )

        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.task.max_episode_timesteps

        config_env.freeze()
        env_configs.append(config_env)

        config_baseline = cfg_baseline(config_dir=CFG_DIR)
        baseline_configs.append(config_baseline)

        #  logger.info("config_env: {}".format(config_env))

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(
                    [args for _ in range(args.ppo.num_processes)],
                    env_configs,
                    baseline_configs,
                    [args.task.shuffle_interval for _ in range(args.ppo.num_processes)],
                    range(args.ppo.num_processes),
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
        hidden_size=args.hidden_size,
        num_recurrent_layers=1,
        blind=args.blind,
        use_aux_losses=0,
        rnn_type="LSTM",
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
        "start_update: {}, num_updates: {}".format(start_update, args.num_updates)
    )

    for update in range(start_update, args.num_updates):
        if args.use_linear_lr_decay:
            update_linear_schedule(agent.optimizer, update, args.num_updates, args.lr)

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
            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

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
            last_observation = {k: v[-1] for k, v in rollouts.observations.items()}
            next_value = actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()
        pth_time += time() - t_update_model

        # log stats
        if update > 0 and (update + 1) % args.log_interval == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    update + 1, count_steps / (time() - t_start)
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(update + 1, env_time, pth_time, count_steps)
            )

            window_rewards = (
                window_episode_reward[-1] - window_episode_reward[0]
            ).sum()

            window_spl = (window_episode_spl[-1] - window_episode_spl[0]).sum()

            window_success = (
                window_episode_success[-1] - window_episode_success[0]
            ).sum()

            window_counts = (window_episode_counts[-1] - window_episode_counts[0]).sum()

            window_counts = (window_episode_counts[-1] - window_episode_counts[0]).sum()

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
        if (update + 1) % args.checkpoint_interval == 0 and update > 0:
            checkpoint = {"state_dict": agent.state_dict()}

            window_rewards = (
                window_episode_reward[-1] - window_episode_reward[0]
            ).sum()
            window_counts = (window_episode_counts[-1] - window_episode_counts[0]).sum()
            avg_reward = (window_rewards / window_counts).item()
            if np.isnan(avg_reward):
                avg_reward = 0

            torch.save(
                checkpoint,
                os.path.join(
                    args.checkpoint_folder,
                    "ckpt.{}.reward.{:.3f}.pth".format(avg_reward, count_checkpoints),
                ),
            )

            count_checkpoints += 1


if __name__ == "__main__":
    main()
