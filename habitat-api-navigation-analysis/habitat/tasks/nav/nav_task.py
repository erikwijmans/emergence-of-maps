#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, List, Optional, Type

import cv2
import fastdtw
import lazy_property
import numpy as np
import quaternion
from gym import spaces
from habitat_sim.utils.common import (
    quat_from_coeffs,
    quat_rotate_vector,
    quat_to_coeffs,
)

import habitat
from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import Measurements
from habitat.core.simulator import (
    SensorSuite,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_rotate_vector,
    quaternion_to_rotation,
)
from habitat.utils.visualizations import maps

COLLISION_PROXIMITY_TOLERANCE: float = 1e-3
NAVIGABLE_POINT_TOLERANCE: float = 0.01
NAVIGABLE_POINT_HEIGHT_CHECK_DISTANCE: float = 0.1
MAP_THICKNESS_SCALAR: int = 1250


def merge_sim_episode_config(sim_config: Config, episode: Type[Episode]) -> Any:
    sim_config.defrost()
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if episode.start_position is not None and episode.start_rotation is not None:
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config


class NavigationGoal:
    """Base class for a goal specification hierarchy.
    """

    position: List[float]
    radius: Optional[float]

    def __init__(
        self, position: List[float], radius: Optional[float] = None, **kwargs
    ) -> None:
        self.position = position
        self.radius = radius


class ObjectGoal(NavigationGoal):
    """Object goal that can be specified by object_id or position or object
    category.
    """

    object_id: str
    object_name: Optional[str]
    object_category: Optional[str]
    room_id: Optional[str]
    room_name: Optional[str]

    def __init__(
        self,
        object_id: str,
        room_id: Optional[str] = None,
        object_name: Optional[str] = None,
        object_category: Optional[str] = None,
        room_name: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.object_id = object_id
        self.object_name = object_name
        self.object_category = object_category
        self.room_id = room_id
        self.room_name = room_name


class RoomGoal(NavigationGoal):
    """Room goal that can be specified by room_id or position with radius.
    """

    room_id: str
    room_name: Optional[str]

    def __init__(self, room_id: str, room_name: Optional[str] = None, **kwargs) -> None:
        super().__init__(**kwargs)  # type: ignore
        self.room_id = room_id
        self.room_name = room_name


class _SE3:
    def __init__(self, rot, trans):
        self.rot = rot
        self.trans = trans

    def inv(self):
        rot_inv = self.rot.inverse()
        return _SE3(rot_inv, quat_rotate_vector(rot_inv, -self.trans))

    def inverse(self):
        return self.inv()

    def __mul__(self, other):
        if isinstance(other, _SE3):
            return _SE3(
                self.rot * other.rot,
                self.trans + quat_rotate_vector(self.rot, other.trans),
            )
        else:
            return quat_rotate_vector(self.rot, other) + self.trans


class NavigationEpisode(Episode):
    """Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """

    goals: List[NavigationGoal]
    start_room: Optional[str]
    shortest_paths: Optional[List[ShortestPathPoint]]

    def __init__(
        self,
        goals: List[NavigationGoal],
        start_room: Optional[str] = None,
        shortest_paths: Optional[List[ShortestPathPoint]] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.goals = goals
        self.shortest_paths = shortest_paths
        self.start_room = start_room

    @lazy_property.LazyProperty
    def transform_world_start(self):
        return _SE3(
            quat_from_coeffs(self.origin_rotation), np.array(self.origin_position)
        )

    @lazy_property.LazyProperty
    def transform_start_world(self):
        return self.transform_world_start.inv()

    def __getstate__(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k
            not in {
                "_transform_start_world",
                "_transform_world_start",
                "trained_initial_hidden_state",
                "random_initial_hidden_state",
            }
        }


class EpisodicGPSAndCompassSensor(habitat.Sensor):
    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "gps_and_compass"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(6,),
            dtype=np.float32,
        )

    def get_observation(self, observations, episode):
        state = self._sim.get_agent_state()
        transform_world_curr = _SE3(state.rotation, state.position)

        transform_start_curr = episode.transform_start_world * transform_world_curr

        look_dir = np.array([0, 0, -1], dtype=np.float32)
        heading_vector = quat_rotate_vector(transform_start_curr.rot, look_dir)

        pos = transform_start_curr.trans

        return np.concatenate([heading_vector, pos]).astype(np.float32)


class EpisodicCompassSensor(habitat.Sensor):
    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "compass"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def get_observation(self, observations, episode):
        state = self._sim.get_agent_state()
        transform_world_curr = _SE3(state.rotation, state.position)

        transform_start_curr = episode.transform_start_world * transform_world_curr

        look_dir = np.array([0, 0, -1], dtype=np.float32)
        heading_vector = quat_rotate_vector(transform_start_curr.rot, look_dir)

        return heading_vector


class BumpSensor(habitat.Sensor):
    def __init__(self, sim, config):
        self._sim = sim
        super().__init__(config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "bump"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=1, shape=(1,), dtype=np.int)

    def get_observation(self, observations, episode):
        return np.array([int(self._sim.previous_step_collided)])


class EpisodeStage(habitat.Sensor):
    def __init__(self, sim, config):
        self._sim = sim
        super().__init__(config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "episode_stage"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=1, shape=(1,), dtype=np.int)

    def get_observation(self, observations, episode):
        return np.array([int(getattr(self._sim, "_count_stop", 0))])


class PointGoalSensor(habitat.Sensor):
    """
    Sensor for PointGoal observations which are used in the PointNav task.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim

        self._goal_format = getattr(config, "GOAL_FORMAT", "CARTESIAN")
        self._sensor_type = getattr(config, "SENSOR_TYPE", "DENSE")
        self._ndims = getattr(config, "SENSOR_DIMENSIONS", 2)
        assert self._goal_format in ["CARTESIAN", "POLAR"]

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "pointgoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        if self._goal_format == "CARTESIAN":
            sensor_shape = (3,)
        else:
            sensor_shape = (self._ndims + 1,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode):
        if self._sensor_type == "DENSE":
            agent_state = self._sim.get_agent_state()
            ref_position = agent_state.position
            rotation_world_agent = agent_state.rotation
        else:
            ref_position = episode.origin_position
            rotation_world_agent = quat_from_coeffs(episode.origin_rotation)

        direction_vector = (
            np.array(episode.goals[0].position, dtype=np.float32) - ref_position
        )
        direction_vector_agent = quaternion_rotate_vector(
            rotation_world_agent.inverse(), direction_vector
        )

        if self._goal_format == "POLAR":
            if self._ndims == 2:
                v = np.array([-direction_vector_agent[2], direction_vector_agent[0]])
            else:
                v = direction_vector_agent.copy()

            rho = np.linalg.norm(v)
            if rho > 1e-5:
                direction_vector_agent = np.array([rho] + v.tolist(), dtype=np.float32)
            else:
                direction_vector_agent = np.array([0, 1, 0], dtype=np.float32)

        return direction_vector_agent


class StaticPointGoalSensor(habitat.Sensor):
    """
    Sensor for PointGoal observations which are used in the StaticPointNav
    task. For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.
    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._goal_format = getattr(config, "GOAL_FORMAT", "CARTESIAN")
        assert self._goal_format in ["CARTESIAN", "POLAR"]

        super().__init__(sim, config)
        self._initial_vector = None
        self.current_episode_id = None

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "static_pointgoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.STATIC_GOAL_VECTOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        if self._goal_format == "CARTESIAN":
            sensor_shape = (3,)
        else:
            sensor_shape = (2,)
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode):
        episode_id = (episode.episode_id, episode.scene_id)
        if self.current_episode_id != episode_id:
            # Only compute the direction vector when a new episode is started.
            self.current_episode_id = episode_id
            agent_state = self._sim.get_agent_state()
            ref_position = agent_state.position
            rotation_world_agent = agent_state.rotation

            direction_vector = (
                np.array(episode.goals[0].position, dtype=np.float32) - ref_position
            )
            direction_vector_agent = quaternion_rotate_vector(
                rotation_world_agent.inverse(), direction_vector
            )

            if self._goal_format == "POLAR":
                rho, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                direction_vector_agent = np.array([rho, -phi], dtype=np.float32)

            self._initial_vector = direction_vector_agent
        return self._initial_vector


class HeadingSensor(habitat.Sensor):
    """
    Sensor for observing the agent's heading in the global coordinate frame.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "heading"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)

    def get_observation(self, observations, episode):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation

        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(
            rotation_world_agent.inverse(), direction_vector
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array(phi)


class ProximitySensor(habitat.Sensor):
    """
    Sensor for observing the distance to the closest obstacle

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(self, sim, config):
        self._sim = sim
        self._max_detection_radius = getattr(config, "MAX_DETECTION_RADIUS", 2.0)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "proximity"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0, high=self._max_detection_radius, shape=(1,), dtype=np.float
        )

    def get_observation(self, observations, episode):
        current_position = self._sim.get_agent_state().position

        return self._sim.distance_to_closest_obstacle(
            current_position, self._max_detection_radius
        )


class EgoMotionSensor(habitat.Sensor):
    def __init__(self, sim, config):
        self._sim = sim
        self._t_world_prev = _SE3(np.quaternion(1, 0, 0, 0), np.array([0, 0, 0]))
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "ego_motion"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-100.0, high=100.0, shape=(2, 3), dtype=np.float)

    def get_observation(self, observations, episode):
        state = self._sim.get_agent_state()
        transform_world_curr = _SE3(state.rotation, state.position)

        t_curr_prev = transform_world_curr.inv() * self._t_world_prev
        self._t_world_prev = transform_world_curr

        trans = t_curr_prev.trans
        rot = quaternion.as_rotation_matrix(t_curr_prev.rot)
        trans = trans[[0, 2]]
        rot = rot[[0, 0, 2, 2], [0, 2, 0, 2]].reshape(2, 2)

        return np.concatenate((rot, trans[:, np.newaxis]), axis=-1)


class DeltaGPSSensor(habitat.Sensor):
    def __init__(self, sim, config):
        self._sim = sim
        self._t_world_prev = _SE3(np.quaternion(1, 0, 0, 0), np.array([0, 0, 0]))
        self._prev_gps = np.array([0, 0, 0], dtype=np.float32)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "delta_gps"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-100.0, high=100.0, shape=(3,), dtype=np.float)

    def get_observation(self, observations, episode):
        state = self._sim.get_agent_state()
        if False:
            gps = episode.transform_start_world * state.position

            delta_gps = gps - self._prev_gps

            self._prev_gps = gps

            return delta_gps
        else:
            transform_world_curr = _SE3(state.rotation, state.position)

            t_curr_prev = transform_world_curr.inv() * self._t_world_prev
            self._t_world_prev = transform_world_curr

            return t_curr_prev.trans


class InitialHiddenState(habitat.Sensor):
    def __init__(self, sim, config):
        self._sim = sim
        self._shape = tuple(config.SHAPE)
        self._state_type = config.STATE_TYPE
        assert self._state_type in {"trained", "random"}
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "initial_hidden_state"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-100.0, high=100.0, shape=self._shape, dtype=np.float)

    def get_observation(self, observations, episode):
        if self._state_type == "trained":
            return episode.trained_initial_hidden_state
        else:
            return episode.random_initial_hidden_state


class DistanceToGoal(habitat.Sensor):
    def __init__(self, sim, config):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "dist_to_goal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float)

    def get_observation(self, observations, episode):
        dg = np.linalg.norm(
            self._sim.get_agent_state(0).position
            - np.array(episode.goals[0].position, dtype=np.float32)
        )
        dg = min(dg, 0.5)

        return np.array([dg], dtype=np.float32)


class SPL(habitat.Measure):
    """SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    """

    def __init__(self, sim: Simulator, config: Config):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "spl"

    def reset_metric(self, episode):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        self._agent_episode_distance = 0.0
        self._metric = None

    @staticmethod
    def euclidean_distance(position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def update_metric(self, episode, action):
        ep_success = 0
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        if distance_to_target == float("inf"):
            print("Hit inf")
            print("Prev pose:", self._previous_position)
            print("Current pose:", current_position)
            raise RuntimeError("Inf go dist")

        if (
            action == self._sim.index_stop_action
            and distance_to_target < self._config.SUCCESS_DISTANCE
        ):
            ep_success = 1

        self._agent_episode_distance += self.euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(self._start_end_episode_distance, self._agent_episode_distance)
        )


class LoopSPL(habitat.Measure):
    def __init__(self, sim: Simulator, config: Config):
        self._previous_position = None
        self._start_end_episode_distances = [None, None]
        self._agent_episode_distances = [None, None]
        self._episode_successes = [None, None]
        self._sim = sim
        self._config = config
        self._episode_stage = None
        self._breakdown_metric = config.BREAKDOWN_METRIC
        self._teleport = config.TELEPORT

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "loop_spl"

    def reset_metric(self, episode):
        self._episode_stage = 0
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distances = [
            episode.info["geodesic_distance"],
            episode.info["geodesic_distance"],
        ]
        self._episode_successes = [False, False]
        self._agent_episode_distances = [0.0, 0.0]
        self._agent_episode_goals = [
            episode.goals[0].position,
            episode.goals[0].position if self._teleport else episode.start_position,
        ]
        self._metric = None

    def update_metric(self, episode, action):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, self._agent_episode_goals[self._episode_stage]
        )

        if distance_to_target == float("inf"):
            print("Hit inf")
            print("Prev pose:", self._previous_position)
            print("Current pose:", current_position)
            raise RuntimeError("Inf go dist")

        self._agent_episode_distances[self._episode_stage] += SPL.euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        if (
            self._episode_stage == 0
            and action == self._sim.index_stop_action
            and distance_to_target < self._config.SUCCESS_DISTANCE
        ):
            self._episode_successes[0] = True
            if self._teleport:
                self._previous_position = episode.start_position
        elif (
            self._episode_stage == 1
            and action == self._sim.index_stop_action
            and distance_to_target < self._config.SUCCESS_DISTANCE
        ):
            self._episode_successes[1] = True

        if self._episode_stage == 0 and action == self._sim.index_stop_action:
            self._episode_stage = 1

        stage_spls = [
            float(self._episode_successes[i])
            * (
                self._start_end_episode_distances[i]
                / max(
                    self._start_end_episode_distances[i],
                    self._agent_episode_distances[i],
                )
            )
            for i in range(2)
        ]

        assert np.all(np.array(stage_spls) <= 1.0)

        if self._breakdown_metric:
            self._metric = {
                "total_spl": np.sqrt(np.prod(stage_spls)),
                "stage_1_spl": stage_spls[0],
                "stage_2_spl": stage_spls[1],
            }
        else:
            self._metric = np.sqrt(np.prod(stage_spls))


class LoopDDelta(habitat.Measure):
    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._config = config
        self._teleport = config.TELEPORT
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "loop_d_delta"

    def reset_metric(self, episode):
        self._episode_stage = 0
        self._stage_d_deltas = [0.0, 0.0]
        self._start_end_episode_distances = [
            episode.info["geodesic_distance"],
            episode.info["geodesic_distance"],
        ]
        self._stage_targets = [
            np.array(episode.goals[0].position),
            np.array(episode.goals[0].position)
            if self._teleport
            else np.array(episode.start_position),
        ]
        self._metric = None

    def update_metric(self, episode, action):
        for i in range(2):
            if self._episode_stage == i:
                self._stage_d_deltas[i] = (
                    self._start_end_episode_distances[i]
                    - self._sim.geodesic_distance(
                        self._stage_targets[i], self._sim.get_agent_state().position
                    )
                ) / self._start_end_episode_distances[i]

        if self._episode_stage == 0 and action == self._sim.index_stop_action:
            self._episode_stage = 1

        self._metric = {
            "stage_1": self._stage_d_deltas[0],
            "stage_2": self._stage_d_deltas[1],
        }


class LoopCompareDTW(habitat.Measure):
    def __init__(self, sim: Simulator, config: Config):
        self._episode_stage = None
        self._stage_paths = None
        self._episode_successes = None
        self._sim = sim
        self._teleport = config.TELEPORT
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "loop_compare"

    def reset_metric(self, episode):
        self._episode_stage = 0
        self._stage_paths = [[np.array(episode.start_position)], []]
        self._episode_successes = [False, False]
        self._metric = 0.0

    def _path_length(self, path):
        _len = 0.0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            if np.linalg.norm(p2 - p1) > 1e-3:
                _len += self._sim.geodesic_distance(p2, p1)

        return max(_len, 1e-2)

    def update_metric(self, episode, action):
        current_position = self._sim.get_agent_state().position
        if self._episode_stage == 0 and action == self._sim.index_stop_action:
            self._stage_paths[0].append(current_position)
            self._episode_stage = 1

            if self._teleport:
                self._stage_paths[1].append(np.array(episode.start_position))
            else:
                self._stage_paths[1].append(current_position)
        else:
            self._stage_paths[self._episode_stage].append(current_position)

        if self._episode_stage == 1 and action == self._sim.index_stop_action:
            if len(self._stage_paths[0]) > 0 and len(self._stage_paths[1]) > 0:
                s0_path = np.stack(self._stage_paths[0]).astype(np.float32)
                s1_path = np.stack(self._stage_paths[1]).astype(np.float32)

                dist, _ = fastdtw.dtw(
                    s0_path, s1_path, dist=self._sim.geodesic_distance
                )

                self._metric = (
                    np.exp(-dist / self._path_length(s0_path))
                    + np.exp(-dist / self._path_length(s1_path))
                ) / 2.0


class AgentPose(habitat.Measure):
    def __init__(self, sim, config):
        self._sim = sim
        self._config = config
        self._metric = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "agent_pose"

    def reset_metric(self, episode):
        self.update_metric(episode, None)

    def update_metric(self, episode, action):
        state = self._sim.get_agent_state()
        self._metric = (state.position, quat_to_coeffs(state.rotation))


class EgocentricPose(habitat.Measure):
    def __init__(self, sim, config):
        self._sim = sim
        self._config = config
        self._metric = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "ego_pose"

    def reset_metric(self, episode):
        self.update_metric(episode, None)

    def update_metric(self, episode, action):
        state = self._sim.get_agent_state()
        transform_world_curr = _SE3(state.rotation, state.position)

        transform_start_curr = episode.transform_start_world * transform_world_curr
        pos = transform_start_curr.trans
        self._metric = (
            transform_start_curr.trans,
            quat_to_coeffs(transform_start_curr.rot),
        )


class GeoDistances(habitat.Measure):
    def __init__(self, sim, config):
        self._sim = sim
        self._config = config
        self._metric = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "geo_distances"

    def reset_metric(self, episode: NavigationEpisode):
        self._start_pos = episode.start_position
        self._goal_pos = episode.goals[0].position
        self.update_metric(episode, None)

    def update_metric(self, episode, action):
        self._metric = {}

        state = self._sim.get_agent_state()
        self._metric["dist_to_start"] = self._sim.geodesic_distance(
            state.position, self._start_pos
        )
        self._metric["dist_to_goal"] = self._sim.geodesic_distance(
            state.position, self._goal_pos
        )


class GoalcentricPose(habitat.Measure):
    def __init__(self, sim, config):
        self._sim = sim
        self._config = config
        self._metric = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "goal_pose"

    def reset_metric(self, episode):
        self._goal_pose = np.array(episode.goals[0].position)
        self._start_rot = self._sim.get_agent_state().rotation.inverse()

        self.update_metric(episode, None)

    def update_metric(self, episode, action):
        self._metric = self._sim.get_agent_state().position - self._goal_pose
        self._metric = quaternion_rotate_vector(self._start_rot, self._metric)
        self._metric = np.array([-self._metric[2], self._metric[0]])


class Collisions(habitat.Measure):
    def __init__(self, sim, config):
        self._sim = sim
        self._config = config
        self._metric = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "collisions"

    def reset_metric(self, episode):
        self._metric = None

    def update_metric(self, episode, action):
        if self._metric is None:
            self._metric = dict(count=0, is_collision=False)

        if self._sim.previous_step_collided:
            self._metric["count"] += 1
            self._metric["is_collision"] = True
        else:
            self._metric["is_collision"] = False


class TopDownOccupancyGrid(habitat.Measure):
    """Top Down Map measure
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        self._top_down_map = None
        self._cell_scale = 0.25
        resolution = int(96 * 1.25)
        self._map_resolution = (resolution, resolution)
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "top_down_occupancy_grid"

    def reset_metric(self, episode):
        top_down_map = np.zeros(self._map_resolution, dtype=np.uint8)
        h2 = top_down_map.shape[0] // 2
        w2 = top_down_map.shape[1] // 2
        jitters = [
            np.array([0, 0, 0.5]),
            np.array([0.5, 0, 0]),
            np.array([0.5, 0, 0.5]),
            np.array([0, 0, 0]),
        ]
        for x in range(top_down_map.shape[0]):
            for y in range(top_down_map.shape[1]):
                pt_start = np.array([x - h2, 0, y - w2], dtype=np.float32)
                valid_point = any(
                    self._sim.is_navigable(
                        episode.transform_world_start
                        * ((pt_start + jit) * self._cell_scale)
                    )
                    for jit in jitters
                )
                top_down_map[x, y] = 1 if valid_point else 0

        h2 = top_down_map.shape[0] // 2 - 1
        w2 = top_down_map.shape[1] // 2 - 1
        crop = int(96 / 2)
        top_down_map = top_down_map[h2 - crop : h2 + crop, w2 - crop : w2 + crop]

        self._metric = top_down_map

    def update_metric(self, episode, action):
        pass


class TopDownMap(habitat.Measure):
    """Top Down Map measure
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._draw_source_and_target = config.DRAW_SOURCE_AND_TARGET
        self._draw_border = config.DRAW_BORDER
        self._grid_delta = 3
        self._step_count = 0
        self._max_steps = config.MAX_EPISODE_STEPS
        self._map_resolution = (config.MAP_RESOLUTION, config.MAP_RESOLUTION)
        self._num_samples = 20000
        self._ind_x_min = None
        self._ind_x_max = None
        self._ind_y_min = None
        self._ind_y_max = None
        self._previous_xy_location = None
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        self._top_down_map = None
        self._cell_scale = (
            self._coordinate_max - self._coordinate_min
        ) / self._map_resolution[0]
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "top_down_map"

    def _check_valid_nav_point(self, point: List[float]):
        return (
            NAVIGABLE_POINT_TOLERANCE
            < self._sim.geodesic_distance(
                point,
                [point[0], point[1] + NAVIGABLE_POINT_HEIGHT_CHECK_DISTANCE, point[2]],
            )
            < (NAVIGABLE_POINT_HEIGHT_CHECK_DISTANCE + NAVIGABLE_POINT_TOLERANCE)
        )

    def get_original_map(self, episode):
        top_down_map = maps.get_topdown_map(
            self._sim, self._map_resolution, self._num_samples, self._draw_border
        )

        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]

        self._ind_x_min = range_x[0]
        self._ind_x_max = range_x[-1]
        self._ind_y_min = range_y[0]
        self._ind_y_max = range_y[-1]

        if self._draw_source_and_target:
            # mark source point
            s_x, s_y = maps.to_grid(
                episode.start_position[0],
                episode.start_position[2],
                self._coordinate_min,
                self._coordinate_max,
                self._map_resolution,
            )
            point_padding = 2 * int(
                np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR)
            )
            top_down_map[
                s_x - point_padding : s_x + point_padding + 1,
                s_y - point_padding : s_y + point_padding + 1,
            ] = 4

            # mark target point
            t_x, t_y = maps.to_grid(
                episode.goals[0].position[0],
                episode.goals[0].position[2],
                self._coordinate_min,
                self._coordinate_max,
                self._map_resolution,
            )
            top_down_map[
                t_x - point_padding : t_x + point_padding + 1,
                t_y - point_padding : t_y + point_padding + 1,
            ] = 6

        return top_down_map

    def reset_metric(self, episode):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map(episode)
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self._previous_xy_location = (a_y, a_x)

    def update_metric(self, episode, action):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        house_map = house_map[
            self._ind_x_min - self._grid_delta : self._ind_x_max + self._grid_delta,
            self._ind_y_min - self._grid_delta : self._ind_y_max + self._grid_delta,
        ]

        self._metric = dict(
            map=house_map,
            map_agent_pos=[
                map_agent_x - self._ind_x_min + self._grid_delta,
                map_agent_y - self._ind_y_min + self._grid_delta,
            ],
            agent_angle=self.get_polar_angle(),
        )

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # Quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array(phi)

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        color = (
            min(self._step_count * 245 / self._max_steps, 245) + 10
            if self._top_down_map[a_x, a_y] != 4
            else self._top_down_map[a_x, a_y]
        )
        color = int(color)

        thickness = int(np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR))
        cv2.line(
            self._top_down_map,
            self._previous_xy_location,
            (a_y, a_x),
            color,
            thickness=thickness,
        )

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y


class NavigationTask(habitat.EmbodiedTask):
    def __init__(
        self, task_config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:

        task_measurements = []
        for measurement_name in task_config.MEASUREMENTS:
            measurement_cfg = getattr(task_config, measurement_name)
            is_valid_measurement = hasattr(
                habitat.tasks.nav.nav_task, measurement_cfg.TYPE  # type: ignore
            )
            assert is_valid_measurement, "invalid measurement type {}".format(
                measurement_cfg.TYPE
            )
            task_measurements.append(
                getattr(
                    habitat.tasks.nav.nav_task, measurement_cfg.TYPE  # type: ignore
                )(sim, measurement_cfg)
            )
        self.measurements = Measurements(task_measurements)

        task_sensors = []
        for sensor_name in task_config.SENSORS:
            sensor_cfg = getattr(task_config, sensor_name)
            is_valid_sensor = hasattr(
                habitat.tasks.nav.nav_task, sensor_cfg.TYPE  # type: ignore
            )
            assert is_valid_sensor, "invalid sensor type {}".format(sensor_cfg.TYPE)
            task_sensors.append(
                getattr(habitat.tasks.nav.nav_task, sensor_cfg.TYPE)(  # type: ignore
                    sim, sensor_cfg
                )
            )

        self.sensor_suite = SensorSuite(task_sensors)
        super().__init__(config=task_config, sim=sim, dataset=dataset)

    def overwrite_sim_config(self, sim_config: Any, episode: Type[Episode]) -> Any:
        return merge_sim_episode_config(sim_config, episode)
