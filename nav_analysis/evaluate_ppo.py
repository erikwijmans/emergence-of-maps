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
import gzip
import json

import imageio
import numpy as np
import torch
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
from nav_analysis.rl import splitnet_nav_envs
from nav_analysis.rl.ppo import Policy
from nav_analysis.rl.ppo.two_agent_policy import TwoAgentPolicy
from nav_analysis.rl.ppo.memory_limited_policy import MemoryLimitedPolicy
from nav_analysis.rl.ppo.utils import batch_obs
from nav_analysis.rl.rnn_memory_buffer import RNNMemoryBuffer
from nav_analysis.utils import radar
from nav_analysis.train_ppo import LoopNavRLEnv, NavRLEnv, ObsStackEnv


CFG_DIR = osp.join(osp.dirname(nav_analysis.__file__), "configs")


class StreamingMean:
    def __init__(self, update_fn=None):
        self._count = 0.0
        self._mean = 0.0
        self.all_vals = []
        self.update_fn = update_fn

    def add(self, v):
        if v is None:
            return
        if self.update_fn is None:
            _sum = self._mean * self._count + v
            self._count += 1.0
            self._mean = _sum / self._count
        else:
            self._count, self._mean = self.update_fn(self._count, self._mean, v)
        self.all_vals.append(v)

    @property
    def mean(self):
        return self._mean


if True:
    logger.handlers[-1].setLevel(level=logging.WARNING)


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def val_env_fn(args, task, config_env, config_baseline, rank):
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config_env.freeze()

    if task in {"loopnav", "teleportnav"}:
        env = LoopNavRLEnv(
            config_env=config_env,
            config_baseline=config_baseline,
            dataset=dataset,
            task=task,
        )
    elif task == "flee":
        env = splitnet_nav_envs.RunAwayRLEnv(config_env, dataset)
    elif task == "explore":
        env = splitnet_nav_envs.ExplorationRLEnv(config_env, dataset)
    else:
        env = NavRLEnv(
            config_env=config_env, config_baseline=config_baseline, dataset=dataset
        )

    env.seed(rank)

    if args.model.max_memory_length:
        env = ObsStackEnv(env, args.model.max_memory_length)

    return env


def images_to_video(images, output_dir, video_name):
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(os.path.join(output_dir, video_name), fps=10)
    for im in tqdm.tqdm(images, leave=False):
        writer.append_data(im)
    writer.close()
    logger.info("Generated video: {}".format(os.path.join(output_dir, video_name)))


def poll_checkpoint_folder(checkpoint_folder, previous_ckpt_ind, exit_immediately):
    if not os.path.isdir(checkpoint_folder):
        return "done" if previous_ckpt_ind != -1 else checkpoint_folder

    models = os.listdir(checkpoint_folder)
    models = list(filter(lambda x: x.endswith(".pth"), models))
    models.sort(key=lambda x: int(x.strip().split(".")[1]))

    ind = previous_ckpt_ind + 1
    if ind < len(models):
        return os.path.join(checkpoint_folder, models[ind])

    return "done" if exit_immediately else None


def construct_val_envs(args, split):
    env_configs = []
    baseline_configs = []

    basic_config = get_config(config_file=args.task.task_config, config_dir=CFG_DIR)
    basic_config.defrost()
    basic_config.DATASET.SPLIT = split
    basic_config.freeze()

    scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)

    assert len(scenes) >= args.ppo.num_processes, (
        "reduce the number of processes as there " "aren't enough number of scenes"
    )
    scene_splits = [[] for _ in range(args.ppo.num_processes)]
    next_split_id = 0
    for s in scenes:
        scene_splits[next_split_id].append(s)
        next_split_id = (next_split_id + 1) % len(scene_splits)

    assert sum(map(len, scene_splits)) == len(scenes)
    sim_gpus = [args.general.sim_gpu_id]

    for i in range(args.ppo.num_processes):
        config_env = get_config(config_file=args.task.task_config, config_dir=CFG_DIR)
        config_env.defrost()

        config_env.DATASET.SPLIT = split
        config_env.DATASET.POINTNAVV1.CONTENT_SCENES = scene_splits[i]

        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = sim_gpus[i % len(sim_gpus)]
        config_env.TASK.POINTGOAL_SENSOR.SENSOR_TYPE = args.task.pointgoal_sensor_type
        config_env.TASK.POINTGOAL_SENSOR.SENSOR_DIMENSIONS = (
            args.task.pointgoal_sensor_dimensions
        )
        config_env.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = args.task.pointgoal_sensor_format
        config_env.DATASET.TYPE = "PointNav-v1"

        agent_sensors = list(args.task.agent_sensors)

        if args.model.blind:
            agent_sensors = []

        if args.general.video and "RGB_SENSOR" not in agent_sensors:
            agent_sensors.append("RGB_SENSOR")

        for sensor in agent_sensors:
            assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]

        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors

        if args.model.blind and args.general.video:
            config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = 2
            config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = 2

            config_env.SIMULATOR.RGB_SENSOR.HEIGHT = 2
            config_env.SIMULATOR.RGB_SENSOR.WIDTH = 2

        if args.task.nav_task in ["loopnav", "teleportnav"]:
            config_env.SIMULATOR.AGENT_0.TURNAROUND = True
            config_env.TASK.MEASUREMENTS = ["LOOPSPL", "LOOP_D_DELTA", "LOOP_COMPARE"]
            config_env.TASK.LOOPSPL.BREAKDOWN_METRIC = True
            config_env.TASK.LOOPNAV_GIVE_RETURN_OBS = (
                args.task.loopnav_give_return_inputs
            )

            if args.task.nav_task == "teleportnav":
                config_env.TASK.LOOPSPL.TELEPORT = True
                config_env.TASK.LOOP_D_DELTA.TELEPORT = True
                config_env.TASK.LOOP_COMPARE.TELEPORT = True
            else:
                config_env.TASK.LOOPSPL.TELEPORT = False
                config_env.TASK.LOOP_D_DELTA.TELEPORT = False
                config_env.TASK.LOOP_COMPARE.TELEPORT = False
        else:
            config_env.TASK.MEASUREMENTS = ["SPL", "LOOP_D_DELTA"]

        if args.general.video:
            config_env.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config_env.TASK.MEASUREMENTS.append("COLLISIONS")
            config_env.TASK.SENSORS.append("POINTGOAL_WITH_GPS_COMPASS")
            config_env.SIMULATOR.RGB_SENSOR.WIDTH = 1024
            config_env.SIMULATOR.RGB_SENSOR.HEIGHT = 1024

            if "DEPTH_SENSOR" in agent_sensors:
                config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = 1024
                config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = 1024

        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.task.max_episode_timesteps

        config_env.TASK.VERBOSE = args.task.nav_env_verbose
        config_env.TASK.MEASUREMENTS.append("GEO_DISTANCES")

        config_env.freeze()
        env_configs.append(config_env)

        config_baseline = cfg_baseline()
        baseline_configs.append(config_baseline)

        logger.info("config_env: {}".format(config_env))

    envs = habitat.VectorEnv(
        make_env_fn=val_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(
                    [args for _ in range(args.ppo.num_processes)],
                    [args.task.nav_task for _ in range(args.ppo.num_processes)],
                    env_configs,
                    baseline_configs,
                    range(args.ppo.num_processes),
                )
            )
        ),
    )

    if args.general.video and args.model.blind:
        del envs.observation_spaces[0].spaces["rgb"]
    elif args.general.video and args.task.agent_sensors == ["DEPTH_SENSOR"]:
        del envs.observation_spaces[0].spaces["rgb"]
        envs.observation_spaces[0].spaces["depth"].shape = (256, 256, 1)

    return envs


def eval_checkpoint(args, current_ckpt):
    if args.video == 1:
        rgb_frames = [[] for _ in range(args.num_processes)]
        extra_infos = [[] for _ in range(args.num_processes)]
        if not os.path.exists(args.out_dir_video):
            os.makedirs(args.out_dir_video)
    else:
        rgb_frames = None

    device = torch.device("cuda", args.pth_gpu_id)
    torch.cuda.set_device(device)

    trained_ckpt = torch.load(current_ckpt, map_location=device)
    trained_args = trained_ckpt["args"]
    trained_args.task.task_config = args.eval_task_config
    trained_args.general.sim_gpu_id = int(args.sim_gpu_ids)
    trained_args.general.video = bool(args.video)
    trained_args.task.nav_env_verbose = bool(args.nav_env_verbose)

    trained_args.ppo.num_processes = args.num_processes

    trained_args.task.nav_task = args.nav_task
    trained_args.task.max_episode_timesteps = 2000

    if trained_args.task.nav_task == "pointnav":
        key_spl = "spl"
    elif trained_args.task.nav_task in {"loopnav", "teleportnav"}:
        key_spl = "loop_spl"

    envs = construct_val_envs(trained_args, args.split)

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
    if trained_args.task.training_stage == 2:
        policy_kwargs["stage_2_state_type"] = trained_args.stage_2_args.state_type
        actor_critic = TwoAgentPolicy(**policy_kwargs)
        stage_1_state = torch.load(
            trained_args.stage_2_args.stage_1_model.replace(
                "/private/home/erikwijmans/projects/navigation-analysis-habitat/",
                os.getcwd() + "/",
            ),
            map_location="cpu",
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

        if trained_args.stage_2_args.state_type == "random":
            random_weights_state = torch.load(
                osp.join(
                    osp.dirname(trained_args.stage_2_args.stage_1_model),
                    "episodes",
                    "random_weights_state.ckpt",
                ),
                map_location="cpu",
            )

            actor_critic.random_policy.load_state_dict(random_weights_state)

    elif trained_args.model.max_memory_length:
        policy_kwargs["max_memory_length"] = trained_args.model.max_memory_length
        actor_critic = MemoryLimitedPolicy(**policy_kwargs)

        actor_critic.load_state_dict(
            {
                k[len("actor_critic.") :]: v
                for k, v in trained_ckpt["state_dict"].items()
                if "ddp" not in k and "actor_critic" in k
            }
        )
    else:

        actor_critic = (
            TwoAgentPolicy(**policy_kwargs)
            if trained_args.model.double_agent
            else Policy(**policy_kwargs)
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
    batch = batch_obs(observations, device)

    current_episode_reward = torch.zeros(envs.num_envs, 1, device=device)

    test_recurrent_hidden_states = torch.zeros(
        actor_critic.net.num_recurrent_layers,
        trained_args.ppo.num_processes,
        trained_args.model.hidden_size,
        device=device,
    )
    not_done_masks = torch.zeros(args.num_processes, 1, device=device)
    prev_actions = torch.zeros(args.num_processes, 1, device=device, dtype=torch.int64)

    rnn_memory_buffer.gt_hidden = test_recurrent_hidden_states.clone()

    infos = None
    with tqdm.tqdm(total=args.count_test_episodes, ncols=0) as pbar:
        total_episode_counts = 0
        stats_episodes = {}
        stats_means = {}

        while total_episode_counts < args.count_test_episodes and envs.num_envs > 0:
            current_episodes = envs.current_episodes()

            with torch.no_grad():
                test_recurrent_hidden_states = rnn_memory_buffer.get_hidden_states()

                (_, actions, _, _, test_recurrent_hidden_states,) = actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                rnn_memory_buffer.add(
                    batch, prev_actions, not_done_masks, test_recurrent_hidden_states,
                )

                prev_actions.copy_(actions)

                actions = actions.cpu()

            outputs = envs.step([a[0].item() for a in actions])

            last_infos = infos
            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=device,
            )

            rewards = torch.tensor(rewards, dtype=torch.float, device=device).unsqueeze(
                1
            )
            current_episode_reward += rewards

            next_episodes = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                next_key = "{}:{}".format(
                    next_episodes[i].episode_id, next_episodes[i].scene_id
                )
                if next_key in stats_episodes:
                    envs_to_pause.append(i)

                if not_done_masks[i].item() == 0:
                    # new episode ended, record stats
                    pbar.update()
                    total_episode_counts += 1

                    if trained_args.task.nav_task in {"loopnav", "teleportnav"}:
                        res = {}
                        for k in infos[i][key_spl]:
                            res[k] = infos[i][key_spl][k]

                        res["success"] = int(infos[i][key_spl]["total_spl"] > 0)
                        res["stage_1_d_delta"] = infos[i]["loop_d_delta"]["stage_1"]
                        res["stage_2_d_delta"] = infos[i]["loop_d_delta"]["stage_2"]
                        if (
                            res["success"]
                            and "loop_compare" in infos[i]
                            and infos[i]["loop_compare"] is not None
                        ):
                            res.update(
                                {
                                    "loop_compare." + k: v
                                    for k, v in infos[i]["loop_compare"].items()
                                }
                            )

                    elif trained_args.task.nav_task == "pointnav":
                        res = {
                            key_spl: infos[i][key_spl],
                            "success": int(infos[i][key_spl] > 0),
                            "d_delta": infos[i]["loop_d_delta"]["stage_1"],
                        }

                        logger.info(
                            "EP {}, SPL {}, Success {}".format(
                                current_episodes[i].episode_id,
                                infos[i][key_spl],
                                infos[i][key_spl] > 0,
                            )
                        )
                    elif trained_args.task.nav_task == "flee":
                        res = {"flee_dist": infos[i]["flee_distance"]}
                    elif trained_args.task.nav_task == "explore":
                        res = {"visited": infos[i]["visited"]}

                    res["initial_l2_to_goal"] = infos[i]["geo_distances"][
                        "initial_l2_to_goal"
                    ]
                    res["initial_geo_to_goal"] = infos[i]["geo_distances"][
                        "initial_geo_to_goal"
                    ]

                    current_key = "{}:{}".format(
                        current_episodes[i].episode_id, current_episodes[i].scene_id,
                    )
                    stats_episodes[current_key] = res

                    for k, v in stats_episodes[current_key].items():
                        if k not in stats_means:
                            stats_means[k] = StreamingMean()

                        stats_means[k].add(v)

                    if args.video == 1:
                        if isinstance(infos[i][key_spl], dict):
                            video_name = "{}_{}_dist={:.2f}_spl1={:.2f}_spl2={:.2f}_dd1={:.2f}_dd2={:.2f}_nact={}".format(
                                current_episodes[i].episode_id,
                                osp.splitext(
                                    osp.basename(current_episodes[i].scene_id)
                                )[0],
                                current_episodes[i].info["geodesic_distance"],
                                infos[i][key_spl]["stage_1_spl"],
                                infos[i][key_spl]["stage_2_spl"] or 0.0,
                                infos[i]["loop_d_delta"]["stage_1"],
                                infos[i]["loop_d_delta"]["stage_2"] or 1.0,
                                len(rgb_frames[i]) + 1,
                            )
                        else:
                            video_name = "{}_{}_{}_{:.2f}".format(
                                current_episodes[i].episode_id,
                                osp.splitext(
                                    osp.basename(current_episodes[i].scene_id)
                                )[0],
                                key_spl,
                                infos[i][key_spl],
                            )

                        images_to_video(rgb_frames[i], args.out_dir_video, video_name)
                        top_down_map = last_infos[i]["top_down_map"]["map"]
                        video_name = video_name.replace(" ", "_").replace("\n", "_")
                        map_name = video_name + ".npy"
                        np.save(
                            osp.join(args.out_dir_video, map_name),
                            top_down_map,
                            allow_pickle=False,
                        )
                        extra_name = video_name + "_extra.npz"
                        np.savez_compressed(
                            osp.join(args.out_dir_video, extra_name),
                            **{
                                k: np.stack([v[k] for v in extra_infos[i]], 0)
                                for k in extra_infos[i][0].keys()
                            }
                        )

                        rgb_frames[i] = []
                        extra_infos[i] = []

                elif args.video == 1:
                    # episode continuing, record frames
                    extra_infos[i].append(
                        dict(
                            hidden_states=test_recurrent_hidden_states[:, i]
                            .cpu()
                            .view(-1)
                            .numpy(),
                            raw_rgb=observations[i]["rgb"].copy(),
                            collision=np.array(
                                [infos[i]["collisions"]["is_collision"]]
                            ),
                            action=actions[i].cpu().numpy(),
                        )
                    )
                    size = observations[i]["rgb"].shape[0]
                    frame = np.empty((size, 2 * size, 3), dtype=np.uint8)
                    rgb = observations[i]["rgb"][..., :3]
                    pg = observations[i]["pointgoal_with_gps_compass"]

                    radar.draw_goal_radar(
                        pg,
                        rgb,
                        radar.Rect(0, 0, int(size / 4), int(size / 4)),
                        start_angle=0,
                        fov=90,
                    )
                    frame[:, :size] = rgb

                    if infos[i]["collisions"]["is_collision"]:
                        frame[:, 1024:] = [0, 0, 0]

                        mask = np.ones((frame.shape[0], frame.shape[1]))
                        mask[30:-30, 30 : 1024 - 30] = 0
                        mask = mask == 1
                        alpha = 0.5
                        frame[mask] = (
                            alpha * np.array([255, 0, 0]) + (1.0 - alpha) * frame
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

                    map_agent_pos = infos[i]["top_down_map"]["map_agent_pos"]
                    map_agent_pos[0] = int(map_agent_pos[0] * scale_x)
                    map_agent_pos[1] = int(map_agent_pos[1] * scale_y)
                    top_down_map = maps.draw_agent(
                        top_down_map,
                        map_agent_pos,
                        -infos[i]["top_down_map"]["agent_angle"] + np.pi / 2,
                        agent_radius_px=7 * 4,
                    )
                    if top_down_map.shape[0] > top_down_map.shape[1]:
                        top_down_map = np.rot90(top_down_map, 1)

                    # white background
                    frame[:, 1024:] = [255, 255, 255]
                    frame[
                        : top_down_map.shape[0], 1024 : 1024 + top_down_map.shape[1],
                    ] = top_down_map
                    rgb_frames[i].append(frame)

            current_episode_reward *= not_done_masks

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
                current_episode_reward = current_episode_reward[state_index]

                for k, v in batch.items():
                    batch[k] = v[state_index]

                if args.video == 1:
                    rgb_frames = [rgb_frames[idx] for idx in state_index]
                    extra_infos = [extra_infos[idx] for idx in state_index]

            def _avg(k):
                return stats_means[k].mean if k in stats_means else 0.0

            if trained_args.task.nav_task == "pointnav":
                pbar.set_postfix(
                    spl=_avg("spl"), success=_avg("success"), d_delta=_avg("d_delta"),
                )

            elif trained_args.task.nav_task in {"loopnav", "teleportnav"}:
                pbar.set_postfix(
                    total_spl=_avg("total_spl"),
                    success=_avg("success"),
                    stage_1_spl=_avg("stage_1_spl"),
                    stage_2_spl=_avg("stage_2_spl"),
                    stage_1_success=_avg("stage_1_success"),
                    stage_2_success=_avg("stage_2_success"),
                    stage_1_d_delta=_avg("stage_1_d_delta"),
                    stage_2_d_delta=_avg("stage_2_d_delta"),
                    loop_compare=_avg("loop_compare_chamfer_probe_agent"),
                )
            elif trained_args.task.nav_task == "flee":
                pbar.set_postfix(flee_dist=_avg("flee_dist"))
            elif trained_args.task.nav_task == "explore":
                pbar.set_postfix(visited=_avg("visited"))

    envs.close()

    return (
        stats_means,
        stats_episodes,
        trained_args,
        float(trained_ckpt["num_frames"]),
        args,
    )


class ModelEvaluator:
    @staticmethod
    def build_parser():
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
            "--nav-task",
            type=str,
            required=True,
            choices=["pointnav", "loopnav", "flee", "explore", "teleportnav"],
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--tensorboard-dir", type=str, required=True)
        parser.add_argument("--nav-env-verbose", type=int, required=True)
        parser.add_argument("--max-memory-length", type=int, default=None)
        parser.add_argument("--eval-task-config", type=str, required=True)
        parser.add_argument("--exit-immediately", action="store_true")
        parser.add_argument("--split", type=str, default="val")

        return parser

    def __init__(self, prev_ckpt_ind=-1, last_step=0):
        self.prev_ckpt_ind = prev_ckpt_ind
        self.last_step = last_step

    def checkpoint(self, args):
        import submitit

        return submitit.helpers.DelayedSubmission(
            ModelEvaluator(self.prev_ckpt_ind, self.last_step), args
        )

    def __call__(self, args):
        random.seed(args.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        logger.add_filehandler(args.log_file)

        if args.video == 1:
            assert args.out_dir_video is not None, "Video dir not specified"

        best_spl = -1
        best_stats = None
        with SummaryWriter(
            log_dir=args.tensorboard_dir, purge_step=self.last_step
        ) as tb_writer:
            while True:
                current_ckpt = None
                while current_ckpt is None:
                    current_ckpt = poll_checkpoint_folder(
                        args.checkpoint_model_dir,
                        self.prev_ckpt_ind,
                        args.exit_immediately,
                    )

                    if current_ckpt == "done":
                        spls = np.array(py_().values().map("spl")(best_stats)) * 100.0
                        success = (
                            np.array(py_().values().map("success")(best_stats)) * 100.0
                        )

                        if len(spls):
                            print(
                                np.mean(spls), 1.96 / np.sqrt(len(spls)) * np.std(spls)
                            )

                        if len(success):
                            print(
                                np.mean(success),
                                1.96 / np.sqrt(len(success)) * np.std(success),
                            )

                        return

                    time.sleep(2)  # sleep for 2 seconds before polling again

                logger.warning("current_ckpt: {}".format(current_ckpt))

                (
                    stats_means,
                    stats_episodes,
                    trained_args,
                    num_frames,
                    _,
                ) = eval_checkpoint(args, current_ckpt)

                if trained_args.task.nav_task in {"loopnav", "teleportnav"}:
                    total_success = (
                        py_().values().map("success").map(int).sum()(stats_episodes)
                    )

                    logger.warn(
                        "Average episode success: {:.6f}".format(
                            total_success / len(stats_episodes)
                        )
                    )
                    avg_stage_1_spl = stats_means["stage_1_spl"].mean
                    avg_stage_2_spl = stats_means["stage_2_spl"].mean

                    logger.warn(
                        "Average episode stage-1 SPL: {:.6f}".format(avg_stage_1_spl)
                    )
                    logger.warn(
                        "Average episode stage-2 SPL: {:.6f}".format(avg_stage_2_spl)
                    )

                    val_metrics = {
                        "stage-1 SPL": avg_stage_1_spl,
                        "stage-2 SPL": avg_stage_2_spl,
                        "stage-1 Success": stats_means["stage_1_success"].mean,
                        "stage-2 Success": stats_means["stage_2_success"].mean,
                        "Success": total_success / len(stats_episodes),
                    }

                    val_metrics.update(
                        {
                            k: v.mean
                            for k, v in stats_means.items()
                            if "loop_compare" in k
                        }
                    )

                    tb_writer.add_scalars(
                        "val", val_metrics, num_frames,
                    )
                elif trained_args.task.nav_task == "pointnav":
                    avg_spl = py_().values().map("spl").mean()(stats_episodes)
                    total_success = (
                        py_().values().map("success").map(int).sum()(stats_episodes)
                    )

                    if avg_spl > best_spl:
                        best_spl = avg_spl
                        best_stats = stats_episodes

                        spls = np.array(py_().values().map("spl")(best_stats)) * 100.0
                        success = (
                            np.array(py_().values().map("success")(best_stats)) * 100.0
                        )

                        print(np.mean(spls), 1.96 / np.sqrt(len(spls)) * np.std(spls))
                        print(
                            np.mean(success),
                            1.96 / np.sqrt(len(success)) * np.std(success),
                        )

                    logger.info("Average episode SPL: {:.6f}".format(avg_spl))

                    tb_writer.add_scalars(
                        "val",
                        {
                            "SPL": avg_spl,
                            "Success": total_success / len(stats_episodes),
                        },
                        num_frames,
                    )
                elif trained_args.task.nav_task == "flee":
                    avg_flee_dist = (
                        py_().values().map("flee_dist").mean()(stats_episodes)
                    )
                    tb_writer.add_scalars(
                        "val", {"flee_dist": avg_flee_dist}, num_frames
                    )
                elif trained_args.task.nav_task == "explore":
                    avg_flee_dist = py_().values().map("visited").mean()(stats_episodes)
                    tb_writer.add_scalars("val", {"visited": avg_flee_dist}, num_frames)

                self.prev_ckpt_ind += 1
                self.last_step = num_frames


if __name__ == "__main__":
    args = ModelEvaluator.build_parser().parse_args()
    ModelEvaluator()(args)
