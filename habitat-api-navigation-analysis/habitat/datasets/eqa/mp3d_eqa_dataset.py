#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
from typing import List

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.tasks.eqa.eqa_task import EQAEpisode, QuestionData
from habitat.tasks.nav.nav_task import ObjectGoal, ShortestPathPoint

EQA_MP3D_V1_VAL_EPISODE_COUNT = 1950


def get_default_mp3d_v1_config(split: str = "val"):
    config = Config()
    config.name = "MP3DEQA-v1"
    config.DATA_PATH = "data/datasets/eqa/mp3d/v1/{split}.json.gz"
    config.DATA_PATH = "data/scene_datasets/mp3d"
    config.SPLIT = split
    return config


class Matterport3dDatasetV1(Dataset):
    """Class inherited from Dataset that loads Matterport3D
    Embodied Question Answering dataset.

    This class can then be used as follows::
        eqa_config.dataset = get_default_mp3d_v1_config()
        eqa = habitat.make_task(eqa_config.task_name, config=eqa_config)
    """

    episodes: List[EQAEpisode]

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(config.MP3DEQAV1.DATA_PATH.format(split=config.SPLIT))

    def __init__(self, config: Config = None) -> None:
        self.episodes = []

        if config is None:
            return

        with gzip.open(
            config.MP3DEQAV1.DATA_PATH.format(split=config.SPLIT), "rt"
        ) as f:
            self.from_json(f.read())

    def from_json(self, json_str: str) -> None:
        deserialized = json.loads(json_str)
        self.__dict__.update(deserialized)
        for ep_index, episode in enumerate(deserialized["episodes"]):
            episode = EQAEpisode(**episode)
            episode.question = QuestionData(**episode.question)
            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = ObjectGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)
            self.episodes[ep_index] = episode
