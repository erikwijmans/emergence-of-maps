#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import getpass
import logging
import os
import os.path as osp
import random
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from pydash import py_
from torch.utils.tensorboard import SummaryWriter

import habitat
import nav_analysis
from habitat import logger
from habitat.config.default import get_config
from habitat.datasets import make_dataset
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.utils.visualizations import maps
from nav_analysis.config.default import cfg as cfg_baseline
from nav_analysis.evaluate_ppo import construct_val_envs
from nav_analysis.rl import splitnet_nav_envs
from nav_analysis.rl.ppo import PPO, Policy
from nav_analysis.rl.ppo.driver_policy import DriverPolicy
from nav_analysis.rl.ppo.utils import batch_obs
from nav_analysis.train_ppo import LoopNavRLEnv, NavRLEnv, make_env_fn

CFG_DIR = osp.join(osp.dirname(nav_analysis.__file__), "configs")


if getpass.getuser() == "erikwijmans":
    logger.handlers[-1].setLevel(level=logging.WARNING)


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def poll_checkpoint_folder(checkpoint_folder, previous_ckpt_ind):
    assert os.path.isdir(checkpoint_folder), "invalid checkpoint folder path"
    models = os.listdir(checkpoint_folder)
    models.sort(key=lambda x: int(x.strip().split(".")[1]))

    #  models = list(reversed(models))

    ind = previous_ckpt_ind + 1
    if ind < len(models):
        return os.path.join(checkpoint_folder, models[ind])

    exit()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-model-dir", type=str, required=True)
    parser.add_argument("--sim-gpu-ids", type=str, required=True)
    parser.add_argument("--pth-gpu-id", type=int, required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--count-test-episodes", type=int, required=True)
    parser.add_argument(
        "--nav-task",
        type=str,
        required=True,
        choices=["pointnav", "loopnav", "flee", "explore"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard-dir", type=str, required=True)
    parser.add_argument("--nav-env-verbose", type=int, required=True)
    parser.add_argument("--max-memory-length", type=int, default=None)
    parser.add_argument("--eval-task-config", type=str, required=True)
    parser.add_argument("--video", type=int, default=0, choices=[0, 1])
    parser.add_argument("--out-dir-video", type=str)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    logger.add_filehandler(args.log_file)

    prev_ckpt_ind = -1

    with SummaryWriter(log_dir=args.tensorboard_dir) as tb_writer:

        while True:
            current_ckpt = None
            while current_ckpt is None:
                current_ckpt = poll_checkpoint_folder(
                    args.checkpoint_model_dir, prev_ckpt_ind
                )
                time.sleep(2)  # sleep for 2 seconds before polling again

            logger.warning("current_ckpt: {}".format(current_ckpt))

            prev_ckpt_ind += 1

            device = torch.device("cuda", args.pth_gpu_id)

            trained_ckpt = torch.load(current_ckpt, map_location=device)
            trained_args = trained_ckpt["args"]
            trained_args.task.task_config = args.eval_task_config
            trained_args.general.sim_gpu_id = int(args.sim_gpu_ids)
            trained_args.task.nav_env_verbose = bool(args.nav_env_verbose)

            trained_args.ppo.num_processes = args.num_processes

            trained_args.task.nav_task = args.nav_task

            if trained_args.task.nav_task == "pointnav":
                key_spl = "spl"
            elif trained_args.task.nav_task == "loopnav":
                key_spl = "loop_spl"

            envs = construct_val_envs(trained_args)

            pointnav_agent = Policy(
                observation_space=envs.observation_spaces[0],
                action_space=envs.action_spaces[0],
                hidden_size=trained_args.model.hidden_size,
                num_recurrent_layers=trained_args.model.num_recurrent_layers,
                blind=trained_args.model.blind,
                use_aux_losses=False,
                rnn_type=trained_args.model.rnn_type,
                resnet_baseplanes=trained_args.model.resnet_baseplanes,
                backbone=trained_args.model.backbone,
                task=trained_args.task.nav_task,
                norm_visual_inputs=trained_args.model.norm_visual_inputs,
                two_headed=trained_args.model.two_headed,
            )
            pointnav_agent.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in torch.load(
                        trained_args.transfer.pretrained_path, map_location="cpu"
                    )["state_dict"].items()
                    if "ddp" not in k and "actor_critic" in k
                }
            )
            pointnav_agent.eval()
            for param in pointnav_agent.parameters():
                param.requires_grad_(False)

            pointnav_agent.to(device)

            actor_critic = DriverPolicy(pointnav_agent)
            actor_critic.to(device)
            actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in trained_ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )

            current_episode_reward = torch.zeros(envs.num_envs, 1, device=device)

            test_recurrent_hidden_states = torch.zeros(
                pointnav_agent.net.num_recurrent_layers + 4,
                trained_args.ppo.num_processes,
                pointnav_agent.net._hidden_size,
                device=device,
            )
            not_done_masks = torch.zeros(args.num_processes, 1, device=device)
            prev_actions = torch.zeros(
                args.num_processes, 1, device=device, dtype=torch.int64
            )

            observations = envs.reset()
            batch = batch_obs(observations)
            for sensor in batch:
                batch[sensor] = batch[sensor].to(device)

            features = pointnav_agent.net.cnn[0](
                pointnav_agent.net.running_mean_and_var(
                    F.avg_pool2d(batch["rgb"].float().permute(0, 3, 1, 2), 2)
                )
            )
            batch["features"] = features

            with tqdm.tqdm(total=args.count_test_episodes, ncols=0) as pbar:
                total_episode_counts = 0
                stats_episodes = {}

                while total_episode_counts < args.count_test_episodes:
                    current_episodes = envs.current_episodes()

                    with torch.no_grad():
                        _, actions, _, _, test_recurrent_hidden_states = actor_critic.act(
                            batch,
                            test_recurrent_hidden_states,
                            prev_actions,
                            not_done_masks,
                            deterministic=False,
                        )

                        prev_actions.copy_(actions)

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

                    rewards = torch.tensor(
                        rewards, dtype=torch.float, device=device
                    ).unsqueeze(1)
                    current_episode_reward += rewards

                    features = pointnav_agent.net.cnn[0](
                        pointnav_agent.net.running_mean_and_var(
                            F.avg_pool2d(batch["rgb"].float().permute(0, 3, 1, 2), 2)
                        )
                    )
                    batch["features"] = features

                    next_episodes = envs.current_episodes()
                    envs_to_pause = []
                    n_envs = envs.num_envs
                    for i in range(n_envs):
                        if next_episodes[i].episode_id in stats_episodes:
                            envs_to_pause.append(i)

                        if not_done_masks[i].item() == 0:
                            # new episode ended, record stats
                            pbar.update()
                            total_episode_counts += 1

                            if trained_args.task.nav_task == "flee":
                                stats_episodes[current_episodes[i].episode_id] = {
                                    "flee_dist": infos[i]["flee_distance"]
                                }
                            elif trained_args.task.nav_task == "explore":
                                stats_episodes[current_episodes[i].episode_id] = {
                                    "visited": infos[i]["visited"]
                                }

                    current_episode_reward *= not_done_masks

                    def _avg(k):
                        return "{:.3f}".format(
                            py_()
                            .values()
                            .map(k)
                            .thru(lambda lst: np.array(lst, dtype=np.float32).mean())(
                                stats_episodes
                            )
                            if len(stats_episodes) > 0
                            else 0.0
                        )

                    if trained_args.task.nav_task == "flee":
                        pbar.set_postfix(flee_dist=_avg("flee_dist"))
                    elif trained_args.task.nav_task == "explore":
                        pbar.set_postfix(visited=_avg("visited"))

                    if len(envs_to_pause) > 0:
                        state_index = list(range(envs.num_envs))
                        for idx in reversed(envs_to_pause):
                            state_index.pop(idx)
                            envs.pause_at(idx)

                        # indexing along the batch dimensions
                        test_recurrent_hidden_states = test_recurrent_hidden_states[
                            :, state_index
                        ]
                        prev_actions = prev_actions[state_index]
                        not_done_masks = not_done_masks[state_index]
                        current_episode_reward = current_episode_reward[state_index]

                        for k, v in batch.items():
                            batch[k] = v[state_index]

            logger.info("Checkpoint {} results:".format(current_ckpt))

            if trained_args.task.nav_task == "flee":
                avg_flee_dist = py_().values().map("flee_dist").mean()(stats_episodes)
                tb_writer.add_scalars(
                    "val", {"flee_dist": avg_flee_dist}, trained_ckpt["num_frames"]
                )
            elif trained_args.task.nav_task == "explore":
                avg_flee_dist = py_().values().map("visited").mean()(stats_episodes)
                tb_writer.add_scalars(
                    "val", {"visited": avg_flee_dist}, trained_ckpt["num_frames"]
                )

            envs.close()


if __name__ == "__main__":
    main()
