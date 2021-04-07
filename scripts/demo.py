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
from nav_analysis.train_ppo import LoopNavRLEnv, NavRLEnv, ObsStackEnv
from nav_analysis.evaluate_ppo import (
    DotDict,
    StreamingMean,
    val_env_fn,
    images_to_video,
)
from nav_analysis.map_extraction.training.train_visited_predictor import (
    Model as HiddenPredictorModel,
)
from nav_analysis.map_extraction.viz.make_visited_videos import (
    colorize_map,
    scale_up_color,
    _scale_up_binary,
)
from nav_analysis.map_extraction.viz.top_down_occ_figure import (
    to_grid,
    create_occupancy_grid_mask,
    make_groundtruth,
    draw_path,
)


CFG_DIR = osp.join(osp.dirname(nav_analysis.__file__), "configs")

logger.handlers[-1].setLevel(level=logging.WARNING)

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def construct_val_envs(args):
    env_configs = []
    baseline_configs = []

    basic_config = get_config(config_file=args.task.task_config, config_dir=CFG_DIR)
    basic_config.defrost()
    basic_config.DATASET.SPLIT = "demo"
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

        config_env.DATASET.SPLIT = "demo"
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

        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors

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
            config_env.TASK.MEASUREMENTS.append("EGO_POSE")

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

    return envs


def eval_checkpoint(args, current_ckpt):
    rgb_frames = [[] for _ in range(args.num_processes)]
    if not os.path.exists(args.out_dir_video):
        os.makedirs(args.out_dir_video)

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

    key_spl = "loop_spl"

    envs = construct_val_envs(trained_args)

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
            "data/checkpoints/demo_agent.pth", map_location="cpu",
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

    actor_critic = actor_critic.to(device)
    actor_critic.eval()

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

    total_episode_counts = 0
    stats_episodes = {}
    stats_means = {}

    prev_hidden_state = None
    agent_final_hidden_state = None
    agent_positions = []
    probe_positions = []
    episode = envs.current_episodes()[0]
    while total_episode_counts < args.count_test_episodes and envs.num_envs > 0:
        current_episodes = envs.current_episodes()

        with torch.no_grad():
            prev_hidden_state = test_recurrent_hidden_states.clone()
            (_, actions, _, _, test_recurrent_hidden_states,) = actor_critic.act(
                batch,
                test_recurrent_hidden_states,
                prev_actions,
                not_done_masks,
                deterministic=True,
            )

            if actions.item() == 3 and agent_final_hidden_state is None:
                agent_final_hidden_state = prev_hidden_state.clone()

            prev_actions.copy_(actions)

        outputs = envs.step([a[0].item() for a in actions])

        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        batch = batch_obs(observations, device)

        not_done_masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=device,
        )

        rewards = torch.tensor(rewards, dtype=torch.float, device=device).unsqueeze(1)
        current_episode_reward += rewards

        n_envs = envs.num_envs
        for i in range(n_envs):
            if agent_final_hidden_state is None:
                agent_positions.append(infos[i]["ego_pose"][0])
            else:
                probe_positions.append(infos[i]["ego_pose"][0])

            if not_done_masks[i].item() == 0:
                # new episode ended, record stats
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

                if isinstance(infos[i][key_spl], dict):
                    video_name = "{}_{}_dist={:.2f}_spl1={:.2f}_spl2={:.2f}_dd1={:.2f}_dd2={:.2f}_nact={}".format(
                        current_episodes[i].episode_id,
                        osp.splitext(osp.basename(current_episodes[i].scene_id))[0],
                        current_episodes[i].info["geodesic_distance"],
                        infos[i][key_spl]["stage_1_spl"],
                        infos[i][key_spl]["stage_2_spl"],
                        infos[i]["loop_d_delta"]["stage_1"],
                        infos[i]["loop_d_delta"]["stage_2"],
                        len(rgb_frames[i]) + 1,
                    )
                else:
                    video_name = "{}_{}_{}_{:.2f}".format(
                        current_episodes[i].episode_id,
                        "apt",
                        key_spl,
                        infos[i][key_spl],
                    )

                images_to_video(rgb_frames[i], args.out_dir_video, video_name)
                rgb_frames[i] = []

            else:
                size = 1024
                frame = np.empty((size, size, 3), dtype=np.uint8)

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
                frame[:, :] = [255, 255, 255]
                frame[: top_down_map.shape[0], : top_down_map.shape[1],] = top_down_map
                rgb_frames[i].append(frame)

        current_episode_reward *= not_done_masks

    envs.close()

    mask = np.zeros((96, 96), dtype=np.int64)
    for pos in agent_positions:
        x, y = to_grid(pos, mask.shape[-1])

        if x <= 0 or x >= mask.shape[0]:
            continue
        if y <= 0 or y >= mask.shape[1]:
            continue

        mask[x, y] = 1

    mask = create_occupancy_grid_mask(mask, napply=5)

    map_predictor = HiddenPredictorModel(512 * 6, [96, 96])
    map_predictor.load_state_dict(
        torch.load("data/checkpoints/best_occ_predictor-trained.pt", map_location="cpu")
    )
    map_predictor.to(device)
    map_predictor.eval()

    _, _, occupancy_logits, _, _ = map_predictor(agent_final_hidden_state.view(1, -1))

    scaling_factor = 6
    color_map = [[236, 240, 241], [149, 165, 166], [255, 255, 255]]

    pred = torch.argmax(occupancy_logits, -1).cpu().numpy().squeeze(0)
    gt_map = make_groundtruth(episode.__getstate__(), pred.shape[0] * scaling_factor)

    pred[mask == 0] = 2
    pred = scale_up_color(colorize_map(pred, color_map), scaling_factor)

    gt_map[_scale_up_binary(mask, scaling_factor) == 0] = 2
    gt_map = colorize_map(gt_map, color_map)

    gt_map = draw_path(agent_positions, gt_map)
    pred = draw_path(agent_positions, pred)

    gt_map = draw_path(probe_positions, gt_map, colors=[[155, 89, 182], [155, 89, 182]])
    pred = draw_path(probe_positions, pred, colors=[[155, 89, 182], [155, 89, 182]])

    imageio.imwrite("results/predicted_map.png", pred)
    imageio.imwrite("results/gt_map.png", gt_map)


if __name__ == "__main__":
    args = DotDict()
    args.pth_gpu_id = 0
    args.out_dir_video = "results"
    args.sim_gpu_ids = 0
    args.video = 1
    args.count_test_episodes = 1
    args.nav_env_verbose = 0
    args.num_processes = 1
    args.eval_task_config = "tasks/loopnav/demo.loopnav.yaml"
    args.nav_task = "loopnav"

    eval_checkpoint(args, "data/checkpoints/demo_probe.pth")
