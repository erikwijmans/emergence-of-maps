import glob
import gzip
import json
import multiprocessing
import os
import os.path as osp

import tqdm

import habitat
import habitat_sim
from habitat.datasets.pointnav.generator import generate_pointnav_episode

num_episodes_per_scene = int(1e3)


def _generate_fn(scene):
    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene
    cfg.SIMULATOR.AGENT_0.SENSORS = []
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

    dset = habitat.datasets.make_dataset("PointNav-v1")
    dset.episodes = list(
        generate_pointnav_episode(
            sim._sim.pathfinder,
            scene,
            num_episodes=num_episodes_per_scene,
        )
    )
    for ep in dset.episodes:
        ep.scene_id = ep.scene_id[len("data/scene_datasets/") :]

    scene_key = osp.splitext(osp.basename(scene))[0]
    out_file = (
        f"./data/datasets/pointnav/mdp/v1/{scene_key}/content/{scene_key}.json.gz"
    )
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())

    out_file = f"./data/datasets/pointnav/mdp/v1/{scene_key}/{scene_key}.json.gz"
    with gzip.open(out_file, "wt") as f:
        json.dump(dict(episodes=[]), f)


scenes = [
    "data/scene_datasets/mp3d/RPmz2sHmrrY/RPmz2sHmrrY.glb",
    "data/scene_datasets/mp3d/jtcxE69GiFV/jtcxE69GiFV.glb",
    "data/scene_datasets/gibson/Scioto.glb",
]

with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
    for _ in pool.imap_unordered(_generate_fn, scenes):
        pbar.update()
