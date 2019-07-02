import argparse

import h5py as h5
import numpy as np
import torch
import tqdm
from pydash import py_

from nav_analysis.rl.ppo.policy import Policy
from nav_analysis.rl.ppo.utils import batch_obs
from nav_analysis.train_ppo import construct_envs


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--num-val-samples", type=float, default=1e4)
    parser.add_argument("--num-train-samples", type=float, default=1e5)

    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-processes", type=int, default=8)

    parser.add_argument("--output-path", type=str, required=True)

    return parser


def main():
    args = build_parser().parse_args()

    device = torch.device("cuda", args.gpu_id)

    trained_ckpt = torch.load(args.model_path, map_location=device)
    trained_args = trained_ckpt["args"]
    trained_args.task_config = "tasks/gibson.pointnav.yaml"

    trained_args.num_processes = args.num_processes
    trained_args.sim_gpu_ids = args.gpu_id
    trained_args.pth_gpu_id = args.gpu_id

    trained_args.nav_task = "pointnav"

    trained_args.nav_env_verbose = False
    trained_args.video = False
    trained_args.shuffle_interval = int(1e3)

    for split in ["train", "val"]:
        num_samples = getattr(args, f"num_{split}_samples")
        with construct_envs(trained_args, split) as envs, tqdm.tqdm(
            total=num_samples
        ) as pbar:
            actor_critic = Policy(
                observation_space=envs.observation_spaces[0],
                action_space=envs.action_spaces[0],
                hidden_size=trained_args.hidden_size,
                num_recurrent_layers=trained_args.num_recurrent_layers,
                blind=trained_args.blind,
                use_aux_losses=trained_args.use_aux_losses,
                rnn_type=trained_args.rnn_type,
                resnet_baseplanes=trained_args.resnet_baseplanes,
                backbone=trained_args.backbone,
            )
            actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in trained_ckpt["state_dict"].items()
                    if "ddp" not in k and "actor_critic" in k
                }
            )

            actor_critic = actor_critic.to(device)
            actor_critic.eval()

            if trained_args.blind:
                assert actor_critic.net.cnn is None

            observations = envs.reset()
            batch = batch_obs(observations)
            for sensor in batch:
                batch[sensor] = batch[sensor].to(device)

            test_recurrent_hidden_states = torch.zeros(
                actor_critic.net.num_recurrent_layers,
                args.num_processes,
                trained_args.hidden_size,
                device=device,
            )
            not_done_masks = torch.zeros(args.num_processes, 1, device=device)
            prev_actions = torch.zeros(
                args.num_processes, 1, device=device, dtype=torch.int64
            )

            num_samples = int(num_samples)
            collision_labels = np.zeros((num_samples,), dtype=np.int64)
            positions = np.zeros((num_samples, 2), dtype=np.float32)
            goal_centric_positions = np.zeros(
                (num_samples, 2), dtype=np.float32
            )
            goal_vectors = np.zeros((num_samples, 3), dtype=np.float32)
            hidden_states = np.zeros(
                (
                    num_samples,
                    actor_critic.net.num_recurrent_layers
                    * trained_args.hidden_size,
                ),
                dtype=np.float32,
            )
            num_collisions = 0.0
            dones = [True] * args.num_processes
            next_idx = 0
            while next_idx < num_samples:
                with torch.no_grad():
                    _, actions, _, _, test_recurrent_hidden_states = actor_critic.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=False,
                    )

                for i in range(args.num_processes):
                    if dones[i]:
                        continue

                    if np.random.uniform(0, 1.0) > 0.1:
                        continue

                    if next_idx >= num_samples:
                        break

                    if np.linalg.norm(infos[i]["ego_pose"]) < 1e-2:
                        continue

                    collision_labels[next_idx] = int(
                        infos[i]["collisions"]["is_collision"]
                    )
                    positions[next_idx] = infos[i]["ego_pose"]
                    goal_centric_positions[next_idx] = infos[i]["goal_pose"]
                    goal_vectors[next_idx] = (
                        batch["pointgoal"][i].cpu().numpy()
                    )
                    hidden_states[next_idx] = (
                        test_recurrent_hidden_states[:, i]
                        .cpu()
                        .view(-1)
                        .numpy()
                    )
                    num_collisions += collision_labels[next_idx]
                    next_idx += 1

                    pbar.update()
                    pbar.set_postfix(
                        pc="{:.3f}".format(num_collisions / next_idx)
                    )

                outputs = envs.step([a[0].item() for a in actions])

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

        with h5.File(f"{args.output_path}_{split}.h5", "w") as f:
            f.create_dataset("collision_labels", data=collision_labels)
            f.create_dataset("positions", data=positions)
            f.create_dataset("hidden_states", data=hidden_states)
            f.create_dataset("goal_vectors", data=goal_vectors)
            f.create_dataset(
                "goal_centric_positions", data=goal_centric_positions
            )


if __name__ == "__main__":
    main()
