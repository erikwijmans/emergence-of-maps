import argparse
import os.path as osp
import random

import cv2
import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import torch
import tqdm
from pydash import py_

from nav_analysis.rl.ppo.policy import Policy
from nav_analysis.rl.ppo.two_agent_policy import TwoAgentPolicy
from nav_analysis.rl.ppo.utils import batch_obs
from nav_analysis.train_ppo import construct_envs

msgpack_numpy.patch()


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--num-val", type=float, default=2e3)
    parser.add_argument("--num-train", type=float, default=5e4)

    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-processes", type=int, default=14)

    parser.add_argument("--output-path", type=str, required=True)

    return parser


def main():
    args = build_parser().parse_args()

    device = torch.device("cuda", args.gpu_id)

    trained_ckpt = torch.load(args.model_path, map_location=device)
    trained_args = trained_ckpt["args"]
    trained_args.task.task_config = "tasks/loopnav/mp3d.loopnav.yaml"

    trained_args.ppo.num_processes = args.num_processes
    trained_args.general.sim_gpu_id = args.gpu_id

    trained_args.general.video = False
    trained_args.task.shuffle_interval = int(1e4)
    random_actor_critic = None

    for split in ["train", "val"][::-1]:
        num_samples = getattr(args, f"num_{split}")
        trained_args.task.nav_task = "loopnav"
        trained_args.task.training_stage = -1
        with construct_envs(trained_args, split, dset_measures=True) as envs, tqdm.tqdm(
            total=num_samples
        ) as pbar:
            trained_args.task.nav_task = "loopnav"
            policy_kwargs = dict(
                observation_space=envs.observation_spaces[0],
                action_space=envs.action_spaces[0],
                hidden_size=trained_args.model.hidden_size,
                num_recurrent_layers=trained_args.model.num_recurrent_layers,
                blind=trained_args.model.blind,
                use_aux_losses=False,
                rnn_type=trained_args.model.rnn_type,
                resnet_baseplanes=trained_args.model.resnet_baseplanes,
                backbone=trained_args.model.backbone,
                task=trained_args.task.nav_task
                if trained_args.task.training_stage == -1
                else "loopnav",
                norm_visual_inputs=trained_args.model.norm_visual_inputs,
                two_headed=trained_args.model.two_headed,
            )
            policy_kwargs["stage_2_state_type"] = trained_args.stage_2_args.state_type
            actor_critic = TwoAgentPolicy(**policy_kwargs)
            stage_1_state = torch.load(
                trained_args.stage_2_args.stage_1_model, map_location="cpu"
            )["state_dict"]
            stage_2_state = trained_ckpt["state_dict"]

            actor_critic.agent1.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in stage_1_state.items()
                    if "ddp" not in k and "actor_critic" in k
                }
            )

            actor_critic.agent2.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in stage_2_state.items()
                    if "ddp" not in k and "actor_critic" in k
                }
            )

            actor_critic = actor_critic.to(device)
            actor_critic.eval()

            if random_actor_critic is None:
                random_actor_critic = TwoAgentPolicy(**policy_kwargs)
                random_actor_critic.eval()
                random_actor_critic.to(device)

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
            random_recurrent_hidden_states = test_recurrent_hidden_states.clone()
            not_done_masks = torch.zeros(args.num_processes, 1, device=device)
            prev_actions = torch.zeros(
                args.num_processes, 1, device=device, dtype=torch.int64
            )

            dones = [True] * args.num_processes
            prev_infos = None
            infos = None
            next_idx = 0
            avg_spl = 0.0
            num_done = 0.0
            trajectories = [[] for _ in range(args.num_processes)]
            lmdb_env = lmdb.open(f"{args.output_path}_{split}.lmdb", map_size=1 << 40)
            with lmdb_env.begin(write=True) as txn:
                db = lmdb_env.open_db()
                txn.drop(db)

            current_episodes = envs.current_episodes()
            with lmdb_env.begin(write=True) as txn:
                while next_idx < num_samples:
                    with torch.no_grad():
                        (
                            _,
                            actions,
                            _,
                            _,
                            test_recurrent_hidden_states,
                        ) = actor_critic.act(
                            batch,
                            test_recurrent_hidden_states,
                            prev_actions,
                            not_done_masks,
                            deterministic=False,
                        )

                        (
                            _,
                            _,
                            _,
                            _,
                            random_recurrent_hidden_states,
                        ) = random_actor_critic.act(
                            batch,
                            random_recurrent_hidden_states,
                            prev_actions,
                            not_done_masks,
                            deterministic=False,
                        )

                    for i in range(args.num_processes):
                        if next_idx >= num_samples:
                            break

                        if infos is None:
                            break

                        if not dones[i]:
                            trajectories[i].append(
                                dict(
                                    hidden_state=test_recurrent_hidden_states[:, i].cpu().numpy(),
                                    random_hidden_state=random_recurrent_hidden_states[
                                        :, i
                                    ].cpu().numpy(),
                                    collision=int(
                                        infos[i]["collisions"]["is_collision"]
                                    ),
                                    positions=infos[i]["ego_pose"][0].copy(),
                                    rotations=infos[i]["ego_pose"][1].copy(),
                                    d_goal=infos[i]["geo_distances"]["dist_to_goal"],
                                    d_start=infos[i]["geo_distances"]["dist_to_start"],
                                    actions=prev_actions[i].item(),
                                )
                            )
                        else:
                            avg_spl = (
                                num_done * avg_spl + infos[i]["loop_spl"]["total_spl"]
                            ) / (num_done + 1)
                            num_done += 1
                            pbar.set_postfix(avg_spl=avg_spl)
                            assert (
                                len(trajectories[i])
                                <= trained_args.task.max_episode_timesteps
                            )

                            if (
                                np.random.uniform(0.0, 1.0)
                                < max(len(trajectories[i]) / 50, 1.0)
                                and infos[i]["loop_spl"]["total_spl"] > 0
                            ):
                                v = {
                                    k: [dic[k] for dic in trajectories[i]]
                                    for k in trajectories[i][0]
                                }
                                ep_info = current_episodes[i].__getstate__()
                                ep_info["goal"] = vars(current_episodes[i].goals[0])
                                del ep_info["goals"]
                                v["episode"] = ep_info
                                v["spl"] = infos[i]["loop_spl"]["total_spl"]
                                v["top_down_occupancy_grid"] = np.ascontiguousarray(
                                    prev_infos[i]["top_down_occupancy_grid"].astype(
                                        np.uint8
                                    )
                                )

                                txn.put(
                                    str(next_idx).encode(),
                                    msgpack.packb(v, use_bin_type=True),
                                )

                                pbar.update()
                                next_idx += 1

                            trajectories[i] = []

                            current_episodes[i] = envs.current_episodes()[i]

                    outputs = envs.step([a[0].item() for a in actions])

                    prev_infos = infos
                    observations, rewards, dones, infos = [
                        list(x) for x in zip(*outputs)
                    ]
                    batch = batch_obs(observations)
                    for sensor in batch:
                        batch[sensor] = batch[sensor].to(device)

                    not_done_masks = torch.tensor(
                        [[0.0] if done else [1.0] for done in dones],
                        dtype=torch.float,
                        device=device,
                    )
                    prev_actions.copy_(actions)


if __name__ == "__main__":
    main()
