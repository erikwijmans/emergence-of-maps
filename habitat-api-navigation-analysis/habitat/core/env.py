#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import time
import weakref
from typing import Any, Dict, List, Optional, Tuple, Type

import gym
import numpy as np
from gym import Space, spaces
from gym.spaces.dict_space import Dict as SpaceDict

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks import make_task


class Env:
    """Fundamental environment class for habitat. All the information needed
    for working on embodied tasks with simulator is abstracted inside
    Env. Acts as a base for other derived environment classes. Env consists
    of three major components: dataset (episodes), simulator and task and
    connects all the three components together.

    Args:
        config: config for the environment. Should contain id for simulator and
            task_name which are passed into make_sim and make_task.
        dataset: reference to dataset for task instance level information.
            Can be defined as None in which case _episodes should be populated
            from outside.

    Attributes:
        observation_space: SpaceDict object corresponding to sensor in sim
            and task.
        action_space: gym.space object corresponding to valid actions.
    """

    observation_space: SpaceDict
    action_space: SpaceDict
    _config: Config
    _dataset: Optional[Dataset]
    _episodes: List[Type[Episode]]
    _current_episode_index: Optional[int]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool

    def __init__(self, config: Config, dataset: Optional[Dataset] = None) -> None:
        assert config.is_frozen(), (
            "Freeze the config before creating the " "environment, use config.freeze()"
        )
        self._config = config
        self._dataset = dataset
        self._current_episode_index = None
        if self._dataset is None and config.DATASET.TYPE:
            self._dataset = make_dataset(
                id_dataset=config.DATASET.TYPE, config=config.DATASET
            )
        self._episodes = self._dataset.episodes if self._dataset else []

        # load the first scene if dataset is present
        if self._dataset:
            assert len(self._dataset.episodes) > 0, (
                "dataset should have " "non-empty episodes list"
            )
            self._dataset.env = weakref.proxy(self)
            self._config.defrost()
            self._config.SIMULATOR.SCENE = self._dataset.episodes[0].scene_id
            self._config.freeze()

        self._sim = make_sim(
            id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
        )
        self._task = make_task(
            self._config.TASK.TYPE,
            task_config=self._config.TASK,
            sim=self._sim,
            dataset=self._dataset,
        )
        self.observation_space = SpaceDict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
            }
        )
        self.action_space = self._sim.action_space
        self._max_episode_seconds = self._config.ENVIRONMENT.MAX_EPISODE_SECONDS
        self._max_episode_steps = self._config.ENVIRONMENT.MAX_EPISODE_STEPS
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False

    @property
    def current_episode(self) -> Type[Episode]:
        assert (
            self._current_episode_index is not None
            and self._current_episode_index < len(self._episodes)
        )
        return self._episodes[self._current_episode_index]

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._episodes

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
        assert len(episodes) > 0, "Environment doesn't accept empty episodes list."
        self._episodes = episodes

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @episode_over.setter
    def episode_over(self, episode_over: bool) -> None:
        self._episode_over = episode_over

    @property
    def task(self) -> EmbodiedTask:
        return self._task

    @property
    def _elapsed_seconds(self) -> float:
        assert (
            self._episode_start_time
        ), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_metrics(self) -> Metrics:
        return self._task.measurements.get_metrics()

    def _past_limit(self) -> bool:
        if (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ):
            return True
        elif (
            self._max_episode_seconds != 0
            and self._max_episode_seconds <= self._elapsed_seconds
        ):
            return True
        return False

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False

    def reset(self) -> Observations:
        """Resets the environments and returns the initial observations.

        Returns:
            Initial observations from the environment
        """
        self._reset_stats()

        assert len(self.episodes) > 0, "Episodes list is empty"

        # Switch to next episode in a loop
        if self._current_episode_index is None:
            self._current_episode_index = 0
        else:
            self._current_episode_index = (self._current_episode_index + 1) % len(
                self._episodes
            )

        if "POSITIONS_FILE" in os.environ:
            with open(os.environ["POSITIONS_FILE"], "rb") as f:
                swapped_start_goal = pickle.load(f)

            if self._current_episode_index < len(swapped_start_goal):
                current_episode_id = self.current_episode.episode_id
                s1_starting_pos = swapped_start_goal[current_episode_id][
                    "start_position"
                ]
                s1_ending_pos = swapped_start_goal[current_episode_id][
                    "s1_ending_position"
                ]
                s1_ending_rot = swapped_start_goal[current_episode_id][
                    "s1_ending_rotation"
                ]
                sts_success = swapped_start_goal[current_episode_id]["success"]

                if sts_success is True:
                    assert s1_ending_pos is not None
                    assert s1_ending_rot is not None
                    self._episodes[
                        self._current_episode_index
                    ].start_position = s1_ending_pos
                    self._episodes[
                        self._current_episode_index
                    ].start_rotation = s1_ending_rot
                    self._episodes[self._current_episode_index].goals[
                        0
                    ].position = s1_starting_pos
                else:

                    self._episodes[
                        self._current_episode_index
                    ].start_position, self._episodes[self._current_episode_index].goals[
                        0
                    ].position = (
                        self._episodes[self._current_episode_index].goals[0].position,
                        self._episodes[self._current_episode_index].start_position,
                    )

                    init_angle = np.random.uniform(0, 2 * np.pi)
                    self._episodes[self._current_episode_index].start_rotation = [
                        0,
                        np.sin(init_angle / 2),
                        0,
                        np.cos(init_angle / 2),
                    ]

        self.reconfigure(self._config)

        observations = self._sim.reset()
        observations.update(
            self.task.sensor_suite.get_observations(
                observations=observations, episode=self.current_episode
            )
        )

        self._task.measurements.reset_measures(episode=self.current_episode)

        if self._dataset is not None:
            self._dataset.step_taken()

        return observations

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._sim.is_episode_active
        if self._past_limit():
            self._episode_over = True

        if self._dataset is not None:
            self._dataset.step_taken()

    def step(self, action: int) -> Observations:
        """Perform an action in the environment and return observations

        Args:
            action: action (belonging to action_space) to be performed inside
                the environment.

        Returns:
            observations after taking action in environment.
        """

        assert self._episode_start_time is not None, (
            "Cannot call step " "before calling reset"
        )
        assert self._episode_over is False, (
            "Episode over, call reset " "before calling step"
        )

        observations = self._sim.step(action)
        observations.update(
            self._task.sensor_suite.get_observations(
                observations=observations, episode=self.current_episode
            )
        )

        self._task.measurements.update_measures(
            episode=self.current_episode, action=action
        )

        self._update_step_stats()

        return observations

    def seed(self, seed: int) -> None:
        self._sim.seed(seed)

    def reconfigure(self, config: Config) -> None:
        self._config = config

        self._config.defrost()
        self._config.SIMULATOR = self._task.overwrite_sim_config(
            self._config.SIMULATOR, self.current_episode
        )
        self._config.freeze()

        self._sim.reconfigure(self._config.SIMULATOR)

    def render(self, mode="rgb") -> np.ndarray:
        return self._sim.render(mode)

    def close(self) -> None:
        self._sim.close()


class RLEnv(gym.Env):
    """Reinforcement Learning (RL) environment class which subclasses gym.Env.
    This is a wrapper over habitat.Env for RL users. To create custom RL
    environments users should subclass RLEnv and define the following methods:

        get_reward_range
        get_reward
        get_done
        get_info

    As this is a subclass of gym.Env, it implements
        reset
        step

    Args:
        config: config to construct habitat.Env.
        dataset: dataset to construct habtiat.Env.
    """

    _env: Env

    def __init__(self, config: Config, dataset: Optional[Dataset] = None) -> None:
        self._env = Env(config, dataset)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.reward_range = self.get_reward_range()

    @property
    def habitat_env(self) -> Env:
        return self._env

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._env.episodes

    @property
    def current_episode(self) -> Type[Episode]:
        return self._env.current_episode

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
        self._env.episodes = episodes

    @property
    def current_episode(self) -> Type[Episode]:
        return self._env.current_episode

    def reset(self) -> Observations:
        return self._env.reset()

    def get_reward_range(self):
        """Get min, max range of reward

        Returns:
             [min, max] range of reward
        """
        raise NotImplementedError

    def get_reward(self, observations: Observations) -> Any:
        """Returns reward after action has been performed. This method
        is called inside the step method.

        Args:
            observations: observations from simulator and task

        Returns:
            reward after performing the last action.
        """
        raise NotImplementedError

    def get_done(self, observations: Observations) -> bool:
        """Returns boolean indicating whether episode is done after performing
        the last action. This method is called inside the step method.

        Args:
            observations: observations from simulator and task

        Returns:
            done boolean after performing the last action.
        """
        raise NotImplementedError

    def get_info(self, observations) -> Dict[Any, Any]:
        """
        Args:
            observations: observations from simulator and task

        Returns:
            info after performing the last action
        """
        raise NotImplementedError

    def step(self, action: int) -> Tuple[Observations, Any, bool, dict]:
        """Perform an action in the environment and return
        (observations, reward, done, info)

        Args:
            action: action (belonging to action_space) to be performed inside
                the environment.

        Returns:
            (observations, reward, done, info)
        """

        observations = self._env.step(action)
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def seed(self, seed: int) -> None:
        self._env.seed(seed)

    def render(self, mode: str = "rgb") -> np.ndarray:
        return self._env.render(mode)

    def close(self) -> None:
        self._env.close()
