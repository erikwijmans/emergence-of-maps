#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, List, Optional

import habitat_sim
import numpy as np
from gym import Space, spaces

import habitat
from habitat import Config, SensorSuite
from habitat.core.logging import logger
from habitat.core.simulator import (
    AgentState,
    DepthSensor,
    RGBSensor,
    SemanticSensor,
    ShortestPathPoint,
)

RGBSENSOR_DIMENSION = 3


def overwrite_config(config_from: Config, config_to: Any) -> None:
    for attr, value in config_from.items():
        if hasattr(config_to, attr.lower()):
            setattr(config_to, attr.lower(), value)


def check_sim_obs(obs, sensor):
    assert obs is not None, (
        "Observation corresponding to {} not present in "
        "simulator's observations".format(sensor.uuid)
    )


class SimulatorActions(Enum):
    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    STOP = 3


class HabitatSimRGBSensor(RGBSensor):
    sim_sensor_type: habitat_sim.SensorType

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        # remove alpha channel
        obs = obs[:, :, :RGBSENSOR_DIMENSION]
        check_sim_obs(obs, self)

        return obs


class HabitatSimDepthSensor(DepthSensor):
    sim_sensor_type: habitat_sim.SensorType
    min_depth_value: float
    max_depth_value: float

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH

        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)

        if isinstance(obs, np.ndarray):
            obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)
            obs = obs[..., np.newaxis]
        else:
            obs = obs.clamp(self.config.MIN_DEPTH, self.config.MAX_DEPTH)
            obs = obs.unsqueeze(-1)

        if self.config.NORMALIZE_DEPTH:
            # normalize depth observation to [0, 1]
            obs = (obs - self.config.MIN_DEPTH) / self.config.MAX_DEPTH

        return obs


class HabitatSimSemanticSensor(SemanticSensor):
    sim_sensor_type: habitat_sim.SensorType

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.uint32).min,
            high=np.iinfo(np.uint32).max,
            shape=(self.config.HEIGHT, self.config.WIDTH),
            dtype=np.uint32,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)
        return obs


class HabitatSim(habitat.Simulator):
    """Simulator wrapper over habitat-sim

    habitat-sim repo: https://github.com/facebookresearch/habitat-sim

    Args:
        config: configuration for initializing the simulator.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        agent_config = self._get_agent_config()

        sim_sensors = []
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.config, sensor_name)
            is_valid_sensor = hasattr(
                habitat.sims.habitat_simulator, sensor_cfg.TYPE  # type: ignore
            )
            assert is_valid_sensor, "invalid sensor type {}".format(sensor_cfg.TYPE)
            sim_sensors.append(
                getattr(
                    habitat.sims.habitat_simulator, sensor_cfg.TYPE  # type: ignore
                )(sensor_cfg)
            )

        self._sensor_suite = SensorSuite(sim_sensors)
        self._agent_turnaround = agent_config.TURNAROUND
        self._count_stop = None
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene.id
        self._sim = habitat_sim.Simulator(self.sim_config)
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )

        self._is_episode_active = False

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        sim_config.scene.id = self.config.SCENE
        sim_config.gpu_device_id = self.config.HABITAT_SIM_V0.GPU_DEVICE_ID
        sim_config.allow_sliding = True
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(config_from=self._get_agent_config(), config_to=agent_config)

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            sim_sensor_cfg = habitat_sim.SensorSpec()
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(sensor.observation_space.shape[:2])
            sim_sensor_cfg.parameters["hfov"] = str(sensor.config.HFOV)
            sim_sensor_cfg.position = sensor.config.POSITION
            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type  # type: ignore
            sim_sensor_cfg.gpu2gpu_transfer = False
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = {
            SimulatorActions.LEFT.value: habitat_sim.ActionSpec(
                "turn_left", habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE)
            ),
            SimulatorActions.RIGHT.value: habitat_sim.ActionSpec(
                "turn_right", habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE)
            ),
            SimulatorActions.FORWARD.value: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(amount=self.config.FORWARD_STEP_SIZE),
            ),
            SimulatorActions.STOP.value: habitat_sim.ActionSpec("stop"),
        }

        if self.config.NOISY_ACTIONS:
            agent_config.action_space[
                SimulatorActions.LEFT.value
            ] = habitat_sim.ActionSpec(
                "pyrobot_noisy_turn_left",
                habitat_sim.PyRobotNoisyActuationSpec(amount=self.config.TURN_ANGLE),
            )
            agent_config.action_space[
                SimulatorActions.RIGHT.value
            ] = habitat_sim.ActionSpec(
                "pyrobot_noisy_turn_right",
                habitat_sim.PyRobotNoisyActuationSpec(amount=self.config.TURN_ANGLE),
            )
            agent_config.action_space[
                SimulatorActions.FORWARD.value
            ] = habitat_sim.ActionSpec(
                "pyrobot_noisy_move_forward",
                habitat_sim.PyRobotNoisyActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            )

        return habitat_sim.Configuration(sim_config, [agent_config])

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    @property
    def action_space(self) -> Space:
        return self._action_space

    @property
    def is_episode_active(self) -> bool:
        return self._is_episode_active

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, _ in enumerate(self.config.AGENTS):
            agent_cfg = self._get_agent_config(agent_id)
            if agent_cfg.IS_SET_START_STATE:
                self.set_agent_state(
                    agent_cfg.START_POSITION, agent_cfg.START_ROTATION, agent_id
                )
                is_updated = True
        return is_updated

    def reset(self):
        self._count_stop = 0
        sim_obs = self._sim.reset()
        if self._update_agents_state():
            sim_obs = self._sim.get_sensor_observations()

        self._is_episode_active = True
        self._prev_sim_obs = sim_obs
        return self._sensor_suite.get_observations(sim_obs)

    def step(self, action):
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )
        if action == SimulatorActions.STOP.value and (
            not self._agent_turnaround or self._count_stop == 1
        ):
            self._is_episode_active = False
            sim_obs = self._sim.get_sensor_observations()
        elif action == SimulatorActions.STOP.value and self._agent_turnaround:
            sim_obs = self._sim.get_sensor_observations()
            self._count_stop += 1
        else:
            sim_obs = self._sim.step(action)

        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        return observations

    def render(self, mode: str = "rgb") -> Any:
        """
        Args:
            mode: sensor whose observation is used for returning the frame,
                eg: "rgb", "depth", "semantic"

        Returns:
            rendered frame according to the mode
        """
        sim_obs = self._sim.get_sensor_observations()
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)

        output = observations.get(mode)
        assert output is not None, "mode {} sensor is not active".format(mode)

        return output

    def seed(self, seed):
        self._sim.seed(seed)

    def reconfigure(self, config: Config) -> None:
        # TODO(maksymets): Switch to Habitat-Sim more efficient caching
        is_same_scene = config.SCENE == self._current_scene
        self.config = config
        self.sim_config = self.create_sim_config(self._sensor_suite)
        if not is_same_scene:
            self._current_scene = config.SCENE
            self._sim.close()
            del self._sim
            self._sim = habitat_sim.Simulator(self.sim_config)

        self._update_agents_state()
        #  self._sim.pathfinder.max_slide_dist = self.config.MAX_SLIDE_DIST

    def geodesic_distance(self, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(position_a, dtype=np.float32)
        path.requested_end = np.array(position_b, dtype=np.float32)
        self._sim.pathfinder.find_path(path)
        return path.geodesic_distance

    def shortest_path(self, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(position_a, dtype=np.float32)
        path.requested_end = np.array(position_b, dtype=np.float32)
        self._sim.pathfinder.find_path(path)

        return path

    def action_space_shortest_path(
        self, source: AgentState, targets: List[AgentState], agent_id: int = 0
    ) -> List[ShortestPathPoint]:
        """
        Returns:
            List of agent states and actions along the shortest path from
            source to the nearest target (both included). If one of the
            target(s) is identical to the source, a list containing only
            one node with the identical agent state is returned. Returns
            an empty list in case none of the targets are reachable from
            the source. For the last item in the returned list the action
            will be None.
        """
        raise NotImplementedError(
            "This function is no longer implemented.  Please use the greedy follower instead"
        )

    @property
    def up_vector(self):
        return np.array([0.0, 1.0, 0.0])

    @property
    def forward_vector(self):
        return -np.array([0.0, 0.0, 1.0])

    def get_straight_shortest_path_points(self, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = position_a
        path.requested_end = position_b
        self._sim.pathfinder.find_path(path)
        return path.points

    def sample_navigable_point(self):
        return self._sim.pathfinder.get_random_navigable_point().tolist()

    def is_navigable(self, point: List[float]):
        return self._sim.pathfinder.is_navigable(point)

    def semantic_annotations(self):
        """
        Returns:
            SemanticScene which is a three level hierarchy of semantic
            annotations for the current scene. Specifically this method
            returns a SemanticScene which contains a list of SemanticLevel's
            where each SemanticLevel contains a list of SemanticRegion's where
            each SemanticRegion contains a list of SemanticObject's.

            SemanticScene has attributes: aabb(axis-aligned bounding box) which
            has attributes aabb.center and aabb.sizes which are 3d vectors,
            categories, levels, objects, regions.

            SemanticLevel has attributes: id, aabb, objects and regions.

            SemanticRegion has attributes: id, level, aabb, category (to get
            name of category use category.name()) and objects.

            SemanticObject has attributes: id, region, aabb, obb (oriented
            bounding box) and category.

            SemanticScene contains List[SemanticLevels]
            SemanticLevel contains List[SemanticRegion]
            SemanticRegion contains List[SemanticObject]

            Example to loop through in a hierarchical fashion:
            for level in semantic_scene.levels:
                for region in level.regions:
                    for obj in region.objects:
        """
        return self._sim.semantic_scene

    def close(self):
        self._sim.close()

    @property
    def index_stop_action(self):
        return SimulatorActions.STOP.value

    @property
    def index_forward_action(self):
        return SimulatorActions.FORWARD.value

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config

    def get_agent_state(self, agent_id: int = 0):
        assert agent_id == 0, "No support of multi agent in {} yet.".format(
            self.__class__.__name__
        )
        return self._sim.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float] = None,
        rotation: List[float] = None,
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> None:
        """Sets agent state similar to initialize_agent, but without agents
        creation.

        Args:
            position: numpy ndarray containing 3 entries for (x, y, z).
            rotation: numpy ndarray with 4 entries for (x, y, z, w) elements
            of unit quaternion (versor) representing agent 3D orientation,
            (https://en.wikipedia.org/wiki/Versor)
            agent_id: int identification of agent from multiagent setup.
            reset_sensors: bool for if sensor changes (e.g. tilt) should be
                reset).
        """
        agent = self._sim.get_agent(agent_id)
        state = self.get_agent_state(agent_id)
        state.position = position
        state.rotation = rotation

        # NB: The agent state also contains the sensor states in _absolute_ coordinates.
        # In order to set the agent's body to a specific location and have the sensors follow,
        # we must not provide any state for the sensors.
        # This will cause them to follow the agent's body
        state.sensor_states = dict()

        agent.set_state(state, reset_sensors)

        self._check_agent_position(position, agent_id)

    # TODO (maksymets): Remove check after simulator became stable
    def _check_agent_position(self, position, agent_id=0):
        if not np.allclose(position, self.get_agent_state(agent_id).position):
            logger.info("Agent state diverges from configured start position.")

    def distance_to_closest_obstacle(self, position, max_search_radius=2.0):
        return self._sim.pathfinder.distance_to_closest_obstacle(
            position, max_search_radius
        )

    @property
    def previous_step_collided(self):
        r"""Whether or not the previous step resulted in a collision
         Returns:
            bool: True if the previous step resulted in a collision, false otherwise
         Warning:
            This feild is only updated when :meth:`step`, :meth:`reset`, or :meth:`get_observations_at` are
            called.  It does not update when the agent is moved to a new loction.  Furthermore, it
            will _always_ be false after :meth:`reset` or :meth:`get_observations_at` as neither of those
            result in an action (step) being taken.
        """
        return self._prev_sim_obs.get("collided", False)
