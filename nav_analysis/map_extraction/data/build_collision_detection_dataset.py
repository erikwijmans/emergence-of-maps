import argparse

import h5py as h5
import numpy as np
import collections
import os.path as osp
import torch
import tqdm
from pydash import py_

from nav_analysis.rl.ppo.policy import Policy
from nav_analysis.rl.ppo.utils import batch_obs
from nav_analysis.train_ppo import construct_envs


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--num-val-samples", type=float, default=2e4)
    parser.add_argument("--num-train-samples", type=float, default=1e6)

    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-processes", type=int, default=8)

    parser.add_argument("--output-path", type=str, required=True)

    return parser


def build_h5(args):
    device = torch.device("cuda", args.gpu_id)

    trained_ckpt = torch.load(args.model_path, map_location=device)
    trained_args = trained_ckpt["args"]
    trained_args.task.task_config = "tasks/loopnav/gibson-public.loopnav.yaml"

    trained_args.ppo.num_processes = args.num_processes
    trained_args.general.sim_gpu_id = args.gpu_id

    trained_args.general.video = False
    trained_args.task.shuffle_interval = int(1e3)

    for split in ["train", "val"][1:]:
        num_samples = getattr(args, f"num_{split}_samples")
        trained_args.task.nav_task = "pointnav"
        with construct_envs(trained_args, split, dset_measures=True) as envs, tqdm.tqdm(
            total=num_samples
        ) as pbar, torch.no_grad():
            trained_args.task.nav_task = "loopnav"
            actor_critic = Policy(
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
                task=trained_args.task.nav_task,
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
            not_done_masks = torch.zeros(args.num_processes, 1, device=device)
            prev_actions = torch.zeros(
                args.num_processes, 1, device=device, dtype=torch.int64
            )

            num_samples = int(num_samples)
            collision_labels = np.zeros((num_samples,), dtype=np.bool)
            dset_prev_actions = np.zeros((num_samples), np.int64)
            hidden_states = np.zeros(
                (
                    num_samples,
                    actor_critic.net.num_recurrent_layers
                    * trained_args.model.hidden_size,
                ),
                dtype=np.float32,
            )
            num_collisions = 0.0
            dones = [True] * args.num_processes
            infos = None
            episode_lens = [0.0] * args.num_processes
            next_idx = 0
            avg_spl = 0.0
            num_done = 0.0
            current_episodes = envs.current_episodes()
            while next_idx < num_samples:
                _, actions, _, _, test_recurrent_hidden_states = actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                for i in range(args.num_processes):
                    if infos is None:
                        break

                    if dones[i]:
                        avg_spl = (num_done * avg_spl + infos[i]["spl"]) / (
                            num_done + 1
                        )
                        num_done += 1
                        continue

                    if np.random.uniform(0, 1.0) > 0.1:
                        continue

                    if next_idx >= num_samples:
                        break

                    if np.linalg.norm(infos[i]["ego_pose"][0]) < 1e-2:
                        continue

                    collision_labels[next_idx] = int(
                        infos[i]["collisions"]["is_collision"]
                    )
                    hidden_states[next_idx] = (
                        test_recurrent_hidden_states[:, i].cpu().view(-1).numpy()
                    )
                    dset_prev_actions[next_idx] = prev_actions[i].item()
                    num_collisions += collision_labels[next_idx]
                    next_idx += 1

                    pbar.update()
                    pbar.set_postfix(
                        pc="{:.3f}".format(num_collisions / next_idx), avg_spl=avg_spl
                    )

                current_episodes = envs.current_episodes()
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

        with h5.File(f"{args.output_path}_{split}.h5", "w") as f:
            f.create_dataset("collision_labels", data=collision_labels)
            f.create_dataset("hidden_states", data=hidden_states)
            f.create_dataset("prev_actions", data=dset_prev_actions)


def build_ffcv(args):
    with h5.File(f"{args.output_path}_train.h5", "r") as f:
        x_train = f["hidden_states"][()]
        y_train = f["collision_labels"][()].astype(np.bool)

    with h5.File(f"{args.output_path}_val.h5", "r") as f:
        x_val = f["hidden_states"][()]
        y_val = f["collision_labels"][()].astype(np.bool)

    mean, std = np.mean(x_train, 0, keepdims=True), np.std(x_train, 0, keepdims=True)

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_train, y_train, x_val, y_val = map(
        lambda t: torch.from_numpy(t), (x_train, y_train, x_val, y_val)
    )

    train_dset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dset = torch.utils.data.TensorDataset(x_val, y_val)


def main():
    args = build_parser().parse_args()
    build_h5(args)

    #  build_ffcv(args)


if __name__ == "__main__":
    main()
