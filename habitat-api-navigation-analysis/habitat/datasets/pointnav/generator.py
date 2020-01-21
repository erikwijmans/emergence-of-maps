import os.path as osp
import random
from typing import Optional

import habitat_sim
import habitat_sim.utils
import numpy as np

from habitat.tasks.nav.nav_task import NavigationEpisode, NavigationGoal

# from utils.visualize.gen_video import make_path_video, make_greedy_path_video

GEODESIC_TO_EUCLID_RATIO_THRESHOLD = 1.1
NUMBER_RETRIES_PER_TARGET = 100
NEAR_DIST_LIMIT = 2.5
FAR_DIST_LIMIT = 30
ISLAND_RADIUS_LIMIT = 1.5

DIFFICULTIES_PROP = {
    "easy": {"start": NEAR_DIST_LIMIT, "end": 6},
    "medium": {"start": 6, "end": 8},
    "hard": {"start": 8, "end": FAR_DIST_LIMIT},
}
# Perfect spit on 10 k
# DIFFICULTIES_PROP = {
#     "easy": {"start": NEAR_DIST_LIMIT, "end": 5.4},
#     "medium": {"start": 5.4, "end": 7.8},
#     "hard": {"start": 7.8, "end": FAR_DIST_LIMIT},
# }


def _build_path(pf, s, t):
    path = habitat_sim.ShortestPath()
    path.requested_start = s
    path.requested_end = t

    pf.find_path(path)

    return path


def geodesic_distance(pf, s, t):
    return _build_path(pf, s, t).geodesic_distance


def get_straight_shortest_path_points(pf, s, t):
    return _build_path(pf, s, t).points


def get_difficulty(geo_distance: float) -> str:
    for difficulty, limits in DIFFICULTIES_PROP.items():
        if limits["start"] <= geo_distance < limits["end"]:
            return difficulty


RETRIES = 0
RETRIES_DIFF_LEVELS = 0
RETRIES_NO_PATH = 0
RETRIES_DIS_RANGE = 0
RETRIES_SAMPL = 0


def _ratio_sample_rate(ratio):
    """

    :param ratio: geodesic distance ratio to Euclid distance
    :return: value between 0.008 and 0.144 for ration 1 and 1.1
    """
    return 20 * (ratio - 0.98) ** 2


def is_compatible_episode(s, t, pf, near_dist, far_dist):
    global RETRIES, RETRIES_DIFF_LEVELS, RETRIES_NO_PATH, RETRIES_SAMPL, RETRIES_DIS_RANGE
    RETRIES += 1

    euclid_dist = np.power(np.power(s - t, 2).sum(0), 0.5)
    if np.abs(s[1] - t[1]) > 0.5:
        RETRIES_DIFF_LEVELS += 1
        # print("RETRIES_DIFF_LEVELS", np.abs(s[1] - t[1]))
        return False, 0
    d_separation = geodesic_distance(pf, s, t)
    if d_separation == np.inf:
        RETRIES_NO_PATH += 1
        return False, 0
    if not near_dist <= d_separation <= far_dist:
        RETRIES_DIS_RANGE += 1
        return False, 0
    distances_ratio = d_separation / euclid_dist
    if distances_ratio < GEODESIC_TO_EUCLID_RATIO_THRESHOLD and np.random.rand() > _ratio_sample_rate(
        distances_ratio
    ):
        # print("RETRIES_SAMPL")
        RETRIES_SAMPL += 1
        return False, 0
    if pf.island_radius(s) < ISLAND_RADIUS_LIMIT:
        # print("ISLAND_RADIUS_LIMIT")
        return False, 0
    return True, d_separation


def _create_episode(
    episode_id,
    scene_id,
    start_position,
    start_rotation,
    target_position,
    shortest_paths=None,
    info=None,
) -> Optional[NavigationEpisode]:
    goals = [NavigationGoal(position=target_position)]
    return NavigationEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        shortest_paths=shortest_paths,
        info=info,
    )


NUMBER_RETRIES_PER_EPISODE = 50


def generate_pointnav_episode(pf, scene, num_episodes=-1, init_orientation="random"):

    episode = None
    episode_count = 0
    retry_count = 0
    # while True: #not episode:
    while episode_count < num_episodes or num_episodes < 0:
        if retry_count == NUMBER_RETRIES_PER_EPISODE:
            yield None
        retry_count += 1

        target_position = pf.get_random_navigable_point()
        if not pf.is_navigable(target_position):
            continue

        # print("TARGET radius: ", env._sim._sim.pathfinder.island_radius(
        #    target_position), target_position[1])
        if pf.island_radius(target_position) < ISLAND_RADIUS_LIMIT:
            # print("ISLAND_RADIUS_LIMIT")
            continue

        for retry in range(NUMBER_RETRIES_PER_TARGET):
            source_position = pf.get_random_navigable_point()
            if not pf.is_navigable(source_position):
                continue
            # print("source radius: ", env._sim._sim.pathfinder.island_radius(
            #    source_position), source_position[1])
            is_compatible, dist = is_compatible_episode(
                source_position,
                target_position,
                pf,
                near_dist=DIFFICULTIES_PROP["easy"]["start"],
                far_dist=DIFFICULTIES_PROP["hard"]["end"],
            )
            if is_compatible:
                if init_orientation == "random":
                    angle = np.random.uniform(0, 2 * np.pi)
                    source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
                elif init_orientation == "spath":
                    spath_points = get_straight_shortest_path_points(
                        pf, source_position, target_position
                    )
                    source_rotation = habitat_sim.utils.quat_to_coeffs(
                        habitat_sim.utils.quat_from_two_vectors(
                            habitat_sim.geo.FRONT, spath_points[1] - spath_points[0]
                        )
                    ).tolist()

                shortest_paths = None
                episode = _create_episode(
                    episode_id=episode_count,
                    scene_id=scene,
                    start_position=source_position.tolist(),
                    start_rotation=source_rotation,
                    target_position=target_position.tolist(),
                    shortest_paths=shortest_paths,
                    info={
                        "geodesic_distance": dist,
                        "difficulty": get_difficulty(dist),
                    },
                )

                episode_count += 1
                yield episode
                retry_count = 0
                break
