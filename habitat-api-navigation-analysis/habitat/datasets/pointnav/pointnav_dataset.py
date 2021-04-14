#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
import os.path as osp
import random
import weakref
from typing import List

import habitat_sim
import lmdb
import msgpack
import msgpack_numpy

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.datasets.pointnav.generator import generate_pointnav_episode
from habitat.tasks.nav.nav_task import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)

ALL_SCENES_MASK = "*"
CONTENT_SCENES_PATH_FIELD = "content_scenes_path"

msgpack_numpy.patch()


def _swizzle_msgpack(fname):
    with gzip.open(fname, "rt") as f:
        return json.load(f)

    base_name = osp.splitext(osp.splitext(fname)[0])[0]
    msg_pack_name = base_name + ".msg"

    if not osp.exists(msg_pack_name):
        with gzip.open(fname, "rt") as infile, open(msg_pack_name, "wb") as outfile:
            msgpack.dump(json.load(infile), outfile, use_bin_type=True)

    with open(msg_pack_name, "rb") as f:
        return msgpack.load(f, raw=False)


class PointNavDatasetV1(Dataset):
    """
        Class inherited from Dataset that loads Point Navigation dataset.
    """

    episodes: List[NavigationEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(config.POINTNAVV1.DATA_PATH.format(split=config.SPLIT))

    @staticmethod
    def get_scenes_to_load(config: Config) -> List[str]:
        """Return list of scene ids for which dataset has separate files with
        episodes.
        """
        assert PointNavDatasetV1.check_config_paths_exist(config)
        dataset_dir = os.path.dirname(
            config.POINTNAVV1.DATA_PATH.format(split=config.SPLIT)
        )

        cfg = config.clone()
        cfg.defrost()
        cfg.POINTNAVV1.CONTENT_SCENES = []
        dataset = PointNavDatasetV1(cfg)
        return PointNavDatasetV1._get_scenes_from_folder(
            content_scenes_path=dataset.content_scenes_path, dataset_dir=dataset_dir
        )

    @classmethod
    def _get_scenes_from_folder(cls, content_scenes_path, dataset_dir):
        scenes = []
        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    def __init__(self, config: Config = None) -> None:
        self.episodes = []

        if config is None:
            return

        datasetfile_path = config.POINTNAVV1.DATA_PATH.format(split=config.SPLIT)
        self.from_json(_swizzle_msgpack(datasetfile_path))

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        scenes = config.POINTNAVV1.CONTENT_SCENES
        if ALL_SCENES_MASK in scenes:
            scenes = PointNavDatasetV1._get_scenes_from_folder(
                content_scenes_path=self.content_scenes_path, dataset_dir=dataset_dir,
            )

        last_ptr = 0
        for scene in scenes:
            scene_filename = self.content_scenes_path.format(
                data_path=dataset_dir, scene=scene
            )
            self.from_json(_swizzle_msgpack(scene_filename))

            random.shuffle(self.episodes[last_ptr:])
            last_ptr = len(self.episodes)

        for ep in self.episodes:
            ep.origin_position = ep.start_position
            ep.origin_rotation = ep.start_rotation

    def from_json(self, deserialized) -> None:
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode in deserialized["episodes"]:
            episode = NavigationEpisode(**episode)
            if not os.path.exists(episode.scene_id):
                episode.scene_id = os.path.join(
                    "data", "scene_datasets", episode.scene_id
                )
            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)
            self.episodes.append(episode)

    def step_taken(self):
        pass


class PointNavDatasetOTFV1(PointNavDatasetV1):
    """
        Class inherited from Dataset that loads Point Navigation dataset.
    """

    class OTFList(object):
        def __init__(self, parent):
            self.parent = parent
            self.cur_idx = None
            self.cur_ep = None

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            if idx != self.cur_idx:
                self.cur_idx = idx
                self.cur_ep = self.parent.gen_episode()

            return self.cur_ep

    episodes: OTFList

    def __init__(self, config: Config = None) -> None:
        assert config.SPLIT in ["train", "val", "train-2plus"]
        self._init_orientation = getattr(config.POINTNAVV1, "INIT_ORIENTATION")
        assert self._init_orientation in {"random", "spath"}
        self.episodes = []

        if config is None:
            return

        datasetfile_path = config.POINTNAVV1.DATA_PATH.format(split=config.SPLIT)
        self.from_json(_swizzle_msgpack(datasetfile_path))

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        scenes = config.POINTNAVV1.CONTENT_SCENES
        if ALL_SCENES_MASK in scenes:
            scenes = PointNavDatasetV1._get_scenes_from_folder(
                content_scenes_path=self.content_scenes_path, dataset_dir=dataset_dir
            )

        self.scene_paths = {}
        self.episodes_by_scene = {}
        for scene in scenes:
            scene_filename = self.content_scenes_path.format(
                data_path=dataset_dir, scene=scene
            )
            self.episodes = []
            self.from_json(_swizzle_msgpack(scene_filename))

            self.scene_paths[scene] = self.episodes[0].scene_id
            self.episodes_by_scene[scene] = self.episodes

        self.episodes = []
        self.episodes = PointNavDatasetOTFV1.OTFList(weakref.proxy(self))
        self.current_scene_idx = 0
        self.shuffle_interval = 10000
        self.scenes = scenes
        random.shuffle(self.scenes)

        self._set_next_shuffle()
        self._steps_taken = 0

        nav_mesh = osp.splitext(self.scene_paths[self.current_scene])[0] + ".navmesh"
        assert osp.exists(nav_mesh), f"{nav_mesh} not found"
        self.pf = habitat_sim.PathFinder()
        self.pf.load_nav_mesh(nav_mesh)
        assert self.pf.is_loaded, f"Could not load {nav_mesh}"

    def _set_next_shuffle(self):
        self.next_shuffle = random.randint(
            0.8 * self.shuffle_interval, 1.2 * self.shuffle_interval
        )

    @property
    def current_scene(self):
        return self.scenes[self.current_scene_idx % len(self.scenes)]

    def shuffle_episodes(self, shuffle_interal):
        self.shuffle_interval = shuffle_interal
        self._set_next_shuffle()

    def switch_scene(self):
        self.current_scene_idx += 1
        self._set_next_shuffle()
        self._steps_taken = 0
        nav_mesh = osp.splitext(self.scene_paths[self.current_scene])[0] + ".navmesh"
        assert osp.exists(nav_mesh), f"{nav_mesh} not found"
        self.pf = habitat_sim.PathFinder()
        self.pf.load_nav_mesh(nav_mesh)
        assert self.pf.is_loaded, f"Could not load {nav_mesh}"

    def gen_episode(self):
        if self._steps_taken >= self.next_shuffle:
            self.switch_scene()

        ep = random.choice(self.episodes_by_scene[self.current_scene])
        while ep is None:
            ep = next(
                generate_pointnav_episode(
                    self.pf,
                    self.scene_paths[self.current_scene],
                    1,
                    self._init_orientation,
                )
            )

            if ep is None:
                self.switch_scene()

        ep.scene_id = self.scene_paths[self.current_scene]
        ep.origin_position = ep.start_position
        ep.origin_rotation = ep.start_rotation

        return ep

    def step_taken(self):
        self._steps_taken += 1


class Stage2Dataset(PointNavDatasetOTFV1):
    def __init__(self, config: Config = None) -> None:
        super().__init__(config)
        self._config = config
        self._path = config.POINTNAVV1.EPISODE_PATH
        self._task_type = config.POINTNAVV1.TASK_TYPE

        assert self._task_type in {"loopnav", "teleportnav"}

        f = osp.join(self._path, f"{self.current_scene}.lmdb")
        assert osp.exists(f), f"{f} does not exist"

        self._load_db(f)

    def _load_db(self, f):
        self._lmdb_env = lmdb.open(f, map_size=1 << 34, readonly=True, lock=False)
        self._len = self._lmdb_env.stat()["entries"]

    def _read_entry(self, idx):
        with self._lmdb_env.begin(buffers=True) as txn:
            return msgpack.unpackb(txn.get(str(idx).encode()), raw=False)

    def gen_episode(self):
        if self._steps_taken >= self.next_shuffle:
            self.current_scene_idx += 1
            self._set_next_shuffle()
            self._steps_taken = 0

            f = osp.join(self._path, f"{self.current_scene}.lmdb")
            assert osp.exists(f), f"{f} does not exist"
            self._load_db(f)

        ep_id = random.randint(0, self._len - 1)
        ep = self._read_entry(ep_id)

        if self._task_type == "loopnav":
            goal = NavigationGoal(**ep["orig_goal"])
            goal.position = ep["orig_start_position"]

            nav_ep = NavigationEpisode(
                episode_id=str(ep_id),
                goals=[goal],
                scene_id=self.scene_paths[self.current_scene],
                start_position=ep["loop_start_position"],
                start_rotation=ep["loop_start_rotation"],
                shortest_paths=None,
                info=dict(geodesic_distance=ep["loop_geodesic_distance"]),
            )
        else:
            goal = NavigationGoal(**ep["orig_goal"])

            nav_ep = NavigationEpisode(
                episode_id=str(ep_id),
                goals=[goal],
                scene_id=self.scene_paths[self.current_scene],
                start_position=ep["orig_start_position"],
                start_rotation=ep["loop_start_rotation"],
                shortest_paths=None,
                info=dict(geodesic_distance=ep["orig_geodesic_distance"]),
            )

        nav_ep.trained_initial_hidden_state = ep["hidden_state"]
        nav_ep.random_initial_hidden_state = ep["random_hidden_state"]
        nav_ep.origin_position = ep["orig_start_position"]
        nav_ep.origin_rotation = ep["orig_start_rotation"]

        return nav_ep
