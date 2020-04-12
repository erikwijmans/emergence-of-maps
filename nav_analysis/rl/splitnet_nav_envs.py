from typing import Any

import numpy as np

import habitat


class ExplorationRLEnv(habitat.RLEnv):
    def __init__(self, config, datasets):
        self._grid_size = 1
        self._slack_reward = 0.0
        self._new_grid_cell_reward = 0.25
        self._collision_reward = 0
        self._return_visited_grid = False
        self._camera_height = config.SIMULATOR.RGB_SENSOR.POSITION[1]
        self._visited = set()
        self._regions = None
        super().__init__(config, datasets)

    def _check_grid_cell(self):
        location = self._current_pose[0].copy()
        quantized_location = (location / self._grid_size).astype(np.int32)
        quantized_loc_tuple = tuple(quantized_location.tolist())
        if quantized_loc_tuple not in self._visited:
            self._visited.add(quantized_loc_tuple)
            return True

        return False

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if "pointgoal" in obs:
            obs["pointgoal"] = np.zeros_like(obs["pointgoal"])

        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        self._visited = set()
        self._current_pose = (
            self.habitat_env.sim.get_agent_state().position,
            self.habitat_env.sim.get_agent_state().rotation,
        )
        self._check_grid_cell()
        return obs

    def get_reward_range(self):
        return (
            self._slack_reward + self._collision_reward,
            self._slack_reward + self._new_grid_cell_reward,
        )

    def get_reward(self, observations) -> Any:
        reward = self._slack_reward

        self._current_pose = (
            self.habitat_env.sim.get_agent_state().position,
            self.habitat_env.sim.get_agent_state().rotation,
        )

        in_new_grid_cell = self._check_grid_cell()
        if in_new_grid_cell:
            reward += self._new_grid_cell_reward
        return reward

    def get_done(self, observations) -> bool:
        return self.habitat_env.episode_over

    def _to_grid(self, padding=5):
        if len(self._visited) == 0:
            return np.zeros((3, 3), dtype=np.uint8)
        visited_cells = np.array(list(self._visited))
        visited_cells = visited_cells[:, [0, 2]]  # Ignore y axis
        mins = np.min(visited_cells, axis=0)
        maxes = np.max(visited_cells, axis=0)
        grid = np.zeros((maxes - mins + 2 * padding + 1), dtype=np.uint8)
        # Mark the visited ones
        visited_cells = visited_cells + padding - mins
        np.ravel(grid)[
            np.ravel_multi_index(visited_cells.T, grid.shape, mode="clip")
        ] = 1
        grid = np.rot90(grid, 2)
        return grid

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()

        info["visited"] = len(self._visited)

        return info


class RunAwayRLEnv(habitat.RLEnv):
    def __init__(self, config, datasets):
        self._start_position = None
        self._prev_distance = None
        self._slack_reward = 0.0
        self._collision_reward = 0
        self._max_dist = None
        super().__init__(config, datasets)

    def reset(self):
        self._max_dist = None
        while self._max_dist is None:
            self._start_position = None
            obs = super().reset()
            self._start_position = self.habitat_env.sim.get_agent_state().position
            self._prev_distance = 0

            try:
                self.max_dist()
            except ValueError:
                pass

        return obs

    def _get_distance(self):
        current_position = self.habitat_env.sim.get_agent_state().position
        distance = self.habitat_env.sim.geodesic_distance(
            current_position, self._start_position
        )
        return distance

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if "pointgoal" in obs:
            obs["pointgoal"] = np.array([5.0, 1.0, 0.0], dtype=np.float32)

        return obs, reward, done, info

    def get_reward_range(self):
        return (self._slack_reward, self._slack_reward + 10.0)

    def get_reward(self, observations) -> Any:
        reward = self._slack_reward

        new_dist = self._get_distance()
        reward += 5.0 * (new_dist - self._prev_distance) / self.max_dist()
        self._prev_distance = new_dist

        return reward

    def get_done(self, observations) -> bool:
        return self.habitat_env.episode_over

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()

        info["flee_distance"] = self._get_distance() / self.max_dist()

        return info

    def max_dist(self):
        if self._max_dist is None:
            hsim = self.habitat_env.sim

            step = 1.0
            xs = np.arange(-10.0, 10.0 + step, step=step)
            zs = xs.copy()
            dists = [
                hsim.geodesic_distance(
                    self._start_position, np.array([x, self._start_position[1], z])
                )
                for x in xs
                for z in zs
            ]

            self._max_dist = np.max(np.array([v for v in dists if v < 100]))

        return self._max_dist
