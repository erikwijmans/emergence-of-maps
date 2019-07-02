#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import os.path as osp
import random
import time
import getpass

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
from nav_analysis.rl.ppo import PPO, Policy
from nav_analysis.rl.ppo.utils import batch_obs
from nav_analysis.train_ppo import LoopNavRLEnv, NavRLEnv, make_env_fn
from nav_analysis.rl.rnn_memory_buffer import RNNMemoryBuffer

CFG_DIR = osp.join(osp.dirname(nav_analysis.__file__), "configs")


if getpass.getuser() == "erikwijmans":
    logger.handlers[-1].setLevel(level=logging.WARNING)


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def val_env_fn(config_env, config_baseline, rank):
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config_env.freeze()

    if config_env.SIMULATOR.AGENT_0.TURNAROUND:
        env = LoopNavRLEnv(
            config_env=config_env,
            config_baseline=config_baseline,
            dataset=dataset,
        )
    else:
        env = NavRLEnv(
            config_env=config_env,
            config_baseline=config_baseline,
            dataset=dataset,
        )

    env.seed(rank)

    return env


def images_to_video(images, output_dir, video_name):
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(os.path.join(output_dir, video_name), fps=10)
    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()
    logger.info(
        "Generated video: {}".format(os.path.join(output_dir, video_name))
    )


def poll_checkpoint_folder(checkpoint_folder, previous_ckpt_ind):
    assert os.path.isdir(checkpoint_folder), "invalid checkpoint folder path"
    models = os.listdir(checkpoint_folder)
    models.sort(key=lambda x: int(x.strip().split(".")[1]))

    #  models = list(reversed(models))

    ind = previous_ckpt_ind + 1
    if ind < len(models):
        return os.path.join(checkpoint_folder, models[ind])
    return None


def construct_val_envs(args):
    env_configs = []
    baseline_configs = []

    basic_config = get_config(config_file=args.task_config, config_dir=CFG_DIR)
    basic_config.defrost()
    basic_config.DATASET.SPLIT = "val"
    basic_config.freeze()

    scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)

    assert len(scenes) >= args.num_processes, (
        "reduce the number of processes as there "
        "aren't enough number of scenes"
    )
    scene_splits = [[] for _ in range(args.num_processes)]
    next_split_id = 0
    for s in scenes:
        scene_splits[next_split_id].append(s)
        next_split_id = (next_split_id + 1) % len(scene_splits)

    assert sum(map(len, scene_splits)) == len(scenes)
    sim_gpus = [int(x) for x in args.sim_gpu_ids.strip().split(",")]

    for i in range(args.num_processes):
        config_env = get_config(
            config_file=args.task_config, config_dir=CFG_DIR
        )
        config_env.defrost()

        config_env.DATASET.SPLIT = "val"
        config_env.DATASET.POINTNAVV1.CONTENT_SCENES = scene_splits[i]

        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = sim_gpus[
            i % len(sim_gpus)
        ]
        config_env.TASK.POINTGOAL_SENSOR.SENSOR_TYPE = (
            args.pointgoal_sensor_type
        )
        config_env.TASK.POINTGOAL_SENSOR.SENSOR_DIMENSIONS = (
            args.pointgoal_sensor_dimensions
        )
        config_env.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = (
            args.pointgoal_sensor_format
        )
        config_env.DATASET.TYPE = "PointNav-v1"

        agent_sensors = [
            s for s in args.sensors.strip().split(",") if len(s) > 0
        ]

        if args.video == 1 and "RGB_SENSOR" not in agent_sensors:
            agent_sensors.append("RGB_SENSOR")

        for sensor in agent_sensors:
            assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]
        if args.blind:
            agent_sensors = []

        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors

        if args.blind and args.video == 0:
            config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = 2
            config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = 2

            config_env.SIMULATOR.RGB_SENSOR.HEIGHT = 2
            config_env.SIMULATOR.RGB_SENSOR.WIDTH = 2

        config_env.SIMULATOR.AGENT_0.TURNAROUND = args.nav_task == "loopnav"

        if args.nav_task == "loopnav":
            config_env.TASK.MEASUREMENTS = ["LOOPSPL", "LOOP_D_DELTA"]
            config_env.TASK.LOOPSPL.BREAKDOWN_METRIC = True
            config_env.TASK.LOOPNAV_GIVE_RETURN_OBS = False
            config_env.TASK.SENSORS += ["EPO_GPS_AND_COMPASS", "EPISODE_STAGE"]
        else:
            config_env.TASK.MEASUREMENTS = ["SPL"]

        if args.video == 1:
            config_env.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config_env.TASK.MEASUREMENTS.append("COLLISIONS")
            config_env.SIMULATOR.RGB_SENSOR.WIDTH = 1024
            config_env.SIMULATOR.RGB_SENSOR.HEIGHT = 1024

            if "DEPTH_SENSOR" in agent_sensors:
                config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = 1024
                config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = 1024

        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_timesteps

        config_env.TASK.VERBOSE = bool(args.nav_env_verbose)

        config_env.freeze()
        env_configs.append(config_env)

        config_baseline = cfg_baseline()
        baseline_configs.append(config_baseline)

        logger.info("config_env: {}".format(config_env))

    envs = habitat.VectorEnv(
        make_env_fn=val_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(env_configs, baseline_configs, range(args.num_processes))
            )
        ),
    )

    if args.video == 1 and args.sensors == "DEPTH_SENSOR":
        del envs.observation_spaces[0].spaces["rgb"]
        envs.observation_spaces[0].spaces["depth"].shape = (256, 256, 1)

    return envs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-model-dir", type=str, required=True)
    parser.add_argument("--sim-gpu-ids", type=str, required=True)
    parser.add_argument("--pth-gpu-id", type=int, required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--count-test-episodes", type=int, required=True)
    parser.add_argument("--video", type=int, default=0, choices=[0, 1])
    parser.add_argument("--out-dir-video", type=str)
    parser.add_argument(
        "--nav-task", type=str, required=True, choices=["pointnav", "loopnav"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard-dir", type=str, required=True)
    parser.add_argument("--nav-env-verbose", type=int, required=True)
    parser.add_argument("--max-memory-length", type=int, default=None)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    logger.add_filehandler(args.log_file)

    if args.video == 1:
        assert args.out_dir_video is not None, "Video dir not specified"

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

            if args.video == 1:
                rgb_frames = [[]] * args.num_processes
                if not os.path.exists(args.out_dir_video):
                    os.makedirs(args.out_dir_video)
            else:
                rgb_frames = None

            prev_ckpt_ind += 1

            device = torch.device("cuda", args.pth_gpu_id)

            trained_ckpt = torch.load(current_ckpt, map_location=device)
            trained_args = trained_ckpt["args"]
            trained_args.task_config = "tasks/gibson.pointnav.yaml"

            trained_args.num_processes = args.num_processes
            trained_args.sim_gpu_ids = args.sim_gpu_ids
            trained_args.pth_gpu_id = args.pth_gpu_id

            trained_args.nav_task = args.nav_task

            if trained_args.nav_task == "pointnav":
                key_spl = "spl"
            else:
                key_spl = "loop_spl"

            trained_args.nav_env_verbose = args.nav_env_verbose
            trained_args.video = args.video

            envs = construct_val_envs(trained_args)

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
                task=trained_args.nav_task,
                norm_visual_inputs=getattr(
                    trained_args, "norm_visual_inputs", False
                ),
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
            rnn_memory_buffer = RNNMemoryBuffer(
                actor_critic,
                num_processes=args.num_processes,
                memory_length=args.max_memory_length,
            )

            if trained_args.blind:
                assert actor_critic.net.cnn is None

            observations = envs.reset()
            batch = batch_obs(observations)
            for sensor in batch:
                batch[sensor] = batch[sensor].to(device)

            current_episode_reward = torch.zeros(
                envs.num_envs, 1, device=device
            )

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

            rnn_memory_buffer.gt_hidden = test_recurrent_hidden_states.clone()

            with tqdm.tqdm(total=args.count_test_episodes, ncols=0) as pbar:
                total_episode_counts = 0
                stats_episodes = {}

                while total_episode_counts < args.count_test_episodes:
                    current_episodes = envs.current_episodes()

                    with torch.no_grad():
                        test_recurrent_hidden_states = (
                            rnn_memory_buffer.get_hidden_states()
                        )

                        _, actions, _, _, test_recurrent_hidden_states = actor_critic.act(
                            batch,
                            test_recurrent_hidden_states,
                            prev_actions,
                            not_done_masks,
                            deterministic=False,
                        )

                        rnn_memory_buffer.add(
                            batch,
                            prev_actions,
                            not_done_masks,
                            test_recurrent_hidden_states,
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

                            if key_spl == "loop_spl":
                                res = {}
                                for k in infos[i][key_spl]:
                                    res[k] = infos[i][key_spl][k]

                                res["success"] = int(
                                    infos[i][key_spl]["total_spl"] > 0
                                )
                                res["stage_1_d_delta"] = infos[i][
                                    "loop_d_delta"
                                ]["stage_1"]
                                res["stage_2_d_delta"] = infos[i][
                                    "loop_d_delta"
                                ]["stage_2"]

                                logger.info(
                                    "EP {}, S1 SPL: {:.3f}, "
                                    "S2 SPL: {:.3f}, "
                                    "T SPL: {:.3f}".format(
                                        current_episodes[i].episode_id,
                                        infos[i][key_spl]["stage_1_spl"],
                                        infos[i][key_spl]["stage_2_spl"],
                                        infos[i][key_spl]["total_spl"],
                                    )
                                )

                                logger.info(
                                    "Num parallel envs: {}".format(
                                        envs.num_envs
                                    )
                                )

                                stats_episodes[
                                    current_episodes[i].episode_id
                                ] = res
                            else:
                                stats_episodes[
                                    current_episodes[i].episode_id
                                ] = {
                                    key_spl: infos[i][key_spl],
                                    "success": (infos[i][key_spl] > 0),
                                }

                                logger.info(
                                    "EP {}, SPL {}, Success {}".format(
                                        current_episodes[i].episode_id,
                                        infos[i][key_spl],
                                        infos[i][key_spl] > 0,
                                    )
                                )

                            if args.video == 1:
                                if isinstance(infos[i][key_spl], dict):
                                    video_name = "{}_{}_spl1_{:.2f}_spl2_{:.2f}".format(
                                        current_episodes[i].episode_id,
                                        "apt",
                                        infos[i][key_spl]["stage_1_spl"],
                                        infos[i][key_spl]["stage_2_spl"],
                                    )
                                else:
                                    video_name = "{}_{}_{}_{:.2f}".format(
                                        current_episodes[i].episode_id,
                                        "apt",
                                        key_spl,
                                        infos[i][key_spl],
                                    )

                                images_to_video(
                                    rgb_frames[i],
                                    args.out_dir_video,
                                    video_name,
                                )
                                rgb_frames[i] = []

                        elif args.video == 1:
                            # episode continuing, record frames
                            size = observations[i]["rgb"].shape[0]
                            frame = np.empty(
                                (size, 2 * size, 3), dtype=np.uint8
                            )
                            frame[:, :size] = observations[i]["rgb"][:, :, :3]

                            if infos[i]["collisions"]["is_collision"]:
                                frame[:, 1024:] = [0, 0, 0]

                                mask = np.ones(
                                    (frame.shape[0], frame.shape[1])
                                )
                                mask[30:-30, 30 : 1024 - 30] = 0
                                mask = mask == 1
                                alpha = 0.5
                                frame[mask] = (
                                    alpha * np.array([255, 0, 0])
                                    + (1.0 - alpha) * frame
                                )[mask]

                            top_down_map = infos[i]["top_down_map"]["map"]
                            scale = 1024.0 / max(top_down_map.shape)
                            scale_x = scale_y = scale
                            top_down_map = maps.lut_top_down_map[top_down_map]
                            top_down_map = maps.resize_img(
                                top_down_map,
                                round(scale * top_down_map.shape[0]),
                                round(scale * top_down_map.shape[1]),
                            )

                            map_agent_pos = infos[i]["top_down_map"][
                                "map_agent_pos"
                            ]
                            map_agent_pos[0] = int(map_agent_pos[0] * scale_x)
                            map_agent_pos[1] = int(map_agent_pos[1] * scale_y)
                            top_down_map = maps.draw_agent(
                                top_down_map,
                                map_agent_pos,
                                -infos[i]["top_down_map"]["agent_angle"]
                                + np.pi / 2,
                                agent_radius_px=7 * 4,
                            )
                            if top_down_map.shape[0] > top_down_map.shape[1]:
                                top_down_map = np.rot90(top_down_map, 1)

                            # white background
                            frame[:, 1024:] = [255, 255, 255]
                            frame[
                                : top_down_map.shape[0],
                                1024 : 1024 + top_down_map.shape[1],
                            ] = top_down_map
                            rgb_frames[i].append(frame)

                    current_episode_reward *= not_done_masks

                    def _avg(k):
                        return "{:.3f}".format(
                            py_()
                            .values()
                            .map(k)
                            .thru(
                                lambda lst: np.array(
                                    lst, dtype=np.float32
                                ).mean()
                            )(stats_episodes)
                            if len(stats_episodes) > 0
                            else 0.0
                        )

                    if key_spl != "loop_spl":
                        pbar.set_postfix(
                            spl=_avg("spl"), success=_avg("success")
                        )
                    else:

                        pbar.set_postfix(
                            total_spl=_avg("total_spl"),
                            success=_avg("success"),
                            stage_1_spl=_avg("stage_1_spl"),
                            stage_2_spl=_avg("stage_2_spl"),
                            stage_1_d_delta=_avg("stage_1_d_delta"),
                            stage_2_d_delta=_avg("stage_2_d_delta"),
                        )

                    if len(envs_to_pause) > 0:
                        state_index = list(range(envs.num_envs))
                        for idx in reversed(envs_to_pause):
                            state_index.pop(idx)
                            envs.pause_at(idx)
                            rnn_memory_buffer.pause_at(idx)

                        # indexing along the batch dimensions
                        test_recurrent_hidden_states = test_recurrent_hidden_states[
                            :, state_index
                        ]
                        prev_actions = prev_actions[state_index]
                        not_done_masks = not_done_masks[state_index]
                        current_episode_reward = current_episode_reward[
                            state_index
                        ]

                        for k, v in batch.items():
                            batch[k] = v[state_index]

                        if args.video == 1:
                            rgb_frames = rgb_frames[state_index]

            logger.info("Checkpoint {} results:".format(current_ckpt))

            total_success = (
                py_().values().map("success").map(int).sum()(stats_episodes)
            )

            logger.info(
                "Average episode success: {:.6f}".format(
                    total_success / len(stats_episodes)
                )
            )

            if key_spl == "loop_spl":
                total_stage_1_spl = 0.0
                total_stage_2_spl = 0.0

                for k, v in stats_episodes.items():
                    total_stage_1_spl += v["stage_1_spl"]
                    total_stage_2_spl += v["stage_2_spl"]

                avg_stage_1_spl = total_stage_1_spl / len(stats_episodes)
                avg_stage_2_spl = total_stage_2_spl / len(stats_episodes)

                logger.info(
                    "Average episode stage-1 SPL: {:.6f}".format(
                        avg_stage_1_spl
                    )
                )
                logger.info(
                    "Average episode stage-2 SPL: {:.6f}".format(
                        avg_stage_2_spl
                    )
                )

                tb_writer.add_scalars(
                    "val",
                    {
                        "stage-1 SPL": avg_stage_1_spl,
                        "stage-2 SPL": avg_stage_2_spl,
                        "Success": total_success / len(stats_episodes),
                    },
                    prev_ckpt_ind,
                )
            else:
                avg_spl = py_().values().map("spl").mean()(stats_episodes)

                logger.info("Average episode SPL: {:.6f}".format(avg_spl))

                tb_writer.add_scalars(
                    "val",
                    {
                        "SPL": avg_spl,
                        "Success": total_success / len(stats_episodes),
                    },
                    prev_ckpt_ind,
                )

            envs.close()

            if args.max_memory_length is not None:
                import json
                import os.path as osp
                import pprint
                import gzip

                res = dict(mem_len=[], spl=[], success=[])
                if osp.exists("spl_vs_mem_len.json.gz"):
                    with gzip.open("spl_vs_mem_len.json.gz", "rt") as f:
                        res = json.load(f)

                for v in stats_episodes.values():
                    res["mem_len"].append(args.max_memory_length)
                    res["spl"].append(v["spl"])
                    res["success"].append(int(v["success"]))

                with gzip.open("spl_vs_mem_len.json.gz", "wt") as f:
                    json.dump(res, f)

                print("=" * 10)
                print(
                    json.dumps(
                        dict(
                            spl=avg_spl,
                            success=total_success / len(stats_episodes),
                            mem_len=args.max_memory_length,
                        )
                    )
                )
                print("=" * 10)
                return


if __name__ == "__main__":
    main()
