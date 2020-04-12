import argparse
import asyncio
import atexit
import os
import os.path as osp
import random

import fairtask
import fairtask_slurm
import habitat_sim
import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import torch
import tqdm
import glob

import nav_analysis
from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from nav_analysis.rl.ppo.policy import Policy
from nav_analysis.rl.ppo.utils import batch_obs
from nav_analysis.train_ppo import construct_envs

msgpack_numpy.patch()

CFG_DIR = osp.join(osp.dirname(nav_analysis.__file__), "configs")


fairtask.queue.TASK_MAX_RETRIES = 30

slurm_q = fairtask_slurm.SLURMQueueConfig(
    name="dump-rollouts",
    num_workers_per_node=4,
    cpus_per_worker=10,
    mem_gb_per_worker=100,
    num_jobs=64,
    partition="learnfair",
    maxtime_mins=int(24 * 60),
    gres="gpu:1",
    log_directory="/checkpoint/{user}/fairtask",
    output="slurm-%j.out",
    error="slurm-%j.err",
)

qs = fairtask.TaskQueues(
    {"local": fairtask.LocalQueueConfig(num_workers=1), "slurm": slurm_q},
    no_workers=False,
)


@atexit.register
def close_queues():
    qs.close()


class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = DotDict(value)
            self[key] = value

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-paths-glob", type=str, required=True)
    parser.add_argument("--sample-per-scene", type=float, default=2e4)

    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-processes", type=int, default=16)

    return parser


@qs.task("slurm")
def collect_traj_for_scene(args, trained_args, random_weights_state, scene):
    device = torch.device("cuda", random.randint(0, torch.cuda.device_count() - 1))
    trained_ckpt = torch.load(args.model_path, map_location="cpu")

    trajectories = []
    with construct_envs(
        trained_args, "train", dset_measures=False, scenes=[scene]
    ) as envs:
        policy_args = dict(
            observation_space=envs.observation_spaces[0],
            action_space=envs.action_spaces[0],
            hidden_size=trained_args.model.hidden_size,
            num_recurrent_layers=trained_args.model.num_recurrent_layers,
            blind=trained_args.model.blind,
            use_aux_losses=False,
            rnn_type=trained_args.model.rnn_type,
            resnet_baseplanes=trained_args.model.resnet_baseplanes,
            backbone=trained_args.model.backbone,
            two_headed=trained_args.model.two_headed,
            task=trained_args.task.nav_task
            if trained_args.task.training_stage == -1
            else "loopnav",
        )

        actor_critic = Policy(**policy_args)
        actor_critic.load_state_dict(
            {
                k[len("actor_critic.") :]: v
                for k, v in trained_ckpt["state_dict"].items()
                if "ddp" not in k and "actor_critic" in k
            }
        )

        actor_critic = actor_critic.to(device)
        actor_critic.eval()

        random_agent = Policy(**policy_args)
        random_agent.load_state_dict(
            torch.load(random_weights_state, map_location="cpu")
        )
        random_agent = random_agent.to(device)
        random_agent.eval()

        if trained_args.model.blind:
            assert actor_critic.net.cnn is None

        observations = envs.reset()
        batch = batch_obs(observations)
        for sensor in batch:
            batch[sensor] = batch[sensor].to(device)

        test_recurrent_hidden_states = torch.zeros(
            actor_critic.net.num_recurrent_layers,
            args.num_processes,
            trained_args.model.hidden_size,
            device=device,
        )

        random_recurrent_hidden_states = torch.zeros(
            actor_critic.net.num_recurrent_layers,
            args.num_processes,
            trained_args.model.hidden_size,
            device=device,
        )

        not_done_masks = torch.zeros(args.num_processes, 1, device=device)

        prev_actions = torch.zeros(
            args.num_processes, 1, device=device, dtype=torch.int64
        )

        dones = [True for _ in range(args.num_processes)]
        infos = None

        current_episodes = envs.current_episodes()
        pbar = tqdm.tqdm(total=args.sample_per_scene, ascii=True)
        while len(trajectories) < args.sample_per_scene:
            with torch.no_grad():
                _, actions, _, _, test_recurrent_hidden_states = actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                _, _, _, _, random_recurrent_hidden_states = random_agent.act(
                    batch,
                    random_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

            outputs = envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations)
            for sensor in batch:
                batch[sensor] = batch[sensor].to(device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=device,
            )
            prev_actions.copy_(actions)

            if infos is None or not any(dones):
                continue

            for i in range(args.num_processes):
                if dones[i]:
                    if infos[i]["spl"] > 0:
                        pbar.update()
                        trajectories.append(
                            dict(
                                hidden_state=test_recurrent_hidden_states[:, i]
                                .clone()
                                .cpu()
                                .numpy()
                                .astype(np.float32),
                                random_hidden_state=random_recurrent_hidden_states[:, i]
                                .clone()
                                .cpu()
                                .numpy()
                                .astype(np.float32),
                                orig_start_position=current_episodes[i].start_position,
                                orig_start_rotation=current_episodes[i].start_rotation,
                                orig_goal=vars(current_episodes[i].goals[0]),
                                orig_geodesic_distance=current_episodes[i].info[
                                    "geodesic_distance"
                                ],
                                loop_start_position=infos[i]["agent_pose"][0].tolist(),
                                loop_start_rotation=infos[i]["agent_pose"][1].tolist(),
                                mesh_path=current_episodes[i].scene_id,
                            )
                        )

                        assert (
                            scene in current_episodes[i].scene_id
                        ), f"{scene} not in {current_episodes[i].scene_id}"

                    current_episodes[i] = envs.current_episodes()[i]

    output_path = osp.join(osp.dirname(args.model_path), "episodes", f"{scene}.lmdb")
    os.makedirs(osp.dirname(output_path), exist_ok=True)

    pf = None

    lmdb_env = lmdb.open(output_path, map_size=1 << 32)
    with lmdb_env.begin(write=True) as txn:
        db = lmdb_env.open_db()
        txn.drop(db)
        idx = 0

        for traj in trajectories:
            if pf is None:
                pf = habitat_sim.PathFinder()
                pf.load_nav_mesh(osp.splitext(traj["mesh_path"])[0] + ".navmesh")

                assert pf.is_loaded

            path = habitat_sim.ShortestPath()
            path.requested_start = traj["loop_start_position"]
            path.requested_end = traj["orig_start_position"]

            pf.find_path(path)

            traj["loop_geodesic_distance"] = path.geodesic_distance
            if not (-100.0 < path.geodesic_distance < 100.0):
                raise ValueError(
                    "Inf geo dist with {}, {} in scene {}".format(
                        path.requested_start, path.requested_end, scene
                    )
                )
            else:
                txn.put(str(idx).encode(), msgpack.packb(traj, use_bin_type=True))
                idx += 1


async def main():
    args = build_parser().parse_args()
    args.model_paths = glob.glob(args.model_paths_glob)

    trained_ckpt = torch.load(args.model_paths[0], map_location="cpu")
    trained_args = trained_ckpt["args"]

    trained_args.ppo.num_processes = args.num_processes
    trained_args.general.sim_gpu_id = args.gpu_id

    trained_args.general.video = False

    basic_config = cfg_env(
        config_file=trained_args.task.task_config, config_dir=CFG_DIR
    )

    basic_config.defrost()
    basic_config.DATASET.SPLIT = "train"
    basic_config.freeze()
    scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)

    random_weights_states_files = dict()

    with construct_envs(
        trained_args, "train", dset_measures=False, scenes=[scenes[0]]
    ) as envs:
        policy_args = dict(
            observation_space=envs.observation_spaces[0],
            action_space=envs.action_spaces[0],
            hidden_size=trained_args.model.hidden_size,
            num_recurrent_layers=trained_args.model.num_recurrent_layers,
            blind=trained_args.model.blind,
            use_aux_losses=False,
            rnn_type=trained_args.model.rnn_type,
            resnet_baseplanes=trained_args.model.resnet_baseplanes,
            backbone=trained_args.model.backbone,
            two_headed=trained_args.model.two_headed,
            task=trained_args.task.nav_task
            if trained_args.task.training_stage == -1
            else "loopnav",
        )
        for model_path in args.model_paths:
            random_weights_state = osp.join(
                osp.dirname(model_path), "episodes", f"random_weights_state.ckpt"
            )
            random_weights_states_files[model_path] = random_weights_state

            os.makedirs(osp.dirname(random_weights_state), exist_ok=True)
            if not osp.exists(random_weights_state):
                actor_critic = Policy(**policy_args)
                torch.save(actor_critic.state_dict(), random_weights_state)

    tasks = []
    for model_path in args.model_paths:
        job_args = DotDict(vars(args))
        job_args.model_path = model_path
        random_weights_state = random_weights_states_files[model_path]
        for s in scenes:
            output_path = osp.join(
                osp.dirname(job_args.model_path), "episodes", f"{s}.lmdb"
            )

            if not osp.exists(output_path):
                tasks.append(
                    collect_traj_for_scene(
                        job_args, trained_args, random_weights_state, s
                    )
                )

    with tqdm.tqdm(total=len(tasks)) as pbar:
        for task in asyncio.as_completed(tasks):
            _ = await task
            pbar.update()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
