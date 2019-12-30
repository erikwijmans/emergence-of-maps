#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, Optional

from habitat.config.default import DEFAULT_CONFIG_DIR, get_config
from habitat.core.agent import Agent
from habitat.core.env import Env


class Benchmark:
    """Benchmark for evaluating agents in environments.


    Args:
        config_file: file to be used for creating the environment.
        config_dir: directory where config_file is located.
    """

    def __init__(
        self, config_file: Optional[str] = None, config_dir: str = DEFAULT_CONFIG_DIR
    ) -> None:
        config_env = get_config(config_file=config_file, config_dir=config_dir)
        self._env = Env(config=config_env)

    def evaluate(
        self, agent: Agent, num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Args:
            agent: agent to be evaluated in environment.
            num_episodes: count of number of episodes for which the evaluation
                should be run.

        Returns:
            dict containing metrics tracked by environment.
        """

        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(num_episodes, len(self._env.episodes))
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0
        while count_episodes < num_episodes:
            agent.reset()
            observations = self._env.reset()

            while not self._env.episode_over:
                action = agent.act(observations)
                observations = self._env.step(action)

            metrics = self._env.get_metrics()
            for m, v in metrics.items():
                agg_metrics[m] += v
            count_episodes += 1

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics
