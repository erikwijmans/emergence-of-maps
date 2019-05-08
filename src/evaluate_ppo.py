#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
import numpy as np
import time

import torch
from src.config.default import cfg as cfg_baseline

import habitat
from habitat.config.default import get_config

import imageio

from src.train_ppo import NavRLEnv, LoopNavRLEnv
from src.rl.ppo import PPO, Policy
from src.rl.ppo.utils import batch_obs
import tqdm
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat import logger
from habitat.datasets import make_dataset
from habitat.utils.visualizations import maps


def val_env_fn(config_env, config_baseline, rank):
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config_env.freeze()

    if config_env.SIMULATOR.AGENT_0.TURNAROUND:
        env = LoopNavRLEnv(
            config_env=config_env,
            config_baseline=config_baseline,
            dataset=dataset
        )
    else:
        env = NavRLEnv(
            config_env=config_env,
            config_baseline=config_baseline,
            dataset=dataset
        )

    env.seed(rank)

    return env


def images_to_video(images, output_dir, video_name):
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(os.path.join(output_dir, video_name), fps=10)
    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()
    print("Generated video: {}".format(os.path.join(output_dir, video_name)))


def poll_checkpoint_folder(checkpoint_folder, previous_ckpt_ind):
    assert os.path.isdir(checkpoint_folder), "invalid checkpoint folder path"
    models = os.listdir(checkpoint_folder)
    models.sort(key=lambda x: int(x.strip().split(".")[1]))
    ind = previous_ckpt_ind + 1
    if ind < len(models):
        return os.path.join(checkpoint_folder, models[ind])
    return None


def construct_val_envs(args):
    env_configs = []
    baseline_configs = []

    basic_config = get_config(config_file=args.task_config)
    basic_config.defrost()
    basic_config.DATASET.SPLIT = "val"
    basic_config.freeze()

    scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)

    assert len(scenes) >= args.num_processes, (
        "reduce the number of processes as there "
        "aren't enough number of scenes"
    )
    scene_split_size = int(np.floor(len(scenes) / args.num_processes))
    sim_gpus = [int(x) for x in args.sim_gpu_ids.strip().split(",")]

    for i in range(args.num_processes):
        config_env = get_config(config_file=args.task_config)
        config_env.defrost()

        config_env.DATASET.SPLIT = "val"
        config_env.DATASET.POINTNAVV1.CONTENT_SCENES = \
            scenes[i * scene_split_size: (i + 1) * scene_split_size]

        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = \
            sim_gpus[i % len(sim_gpus)]
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
        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors

        if args.blind and args.video == 0:
            config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = 2
            config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = 2

            config_env.SIMULATOR.RGB_SENSOR.HEIGHT = 2
            config_env.SIMULATOR.RGB_SENSOR.WIDTH = 2

        config_env.SIMULATOR.AGENT_0.TURNAROUND = (args.nav_task == "loopnav")

        if args.nav_task == "loopnav":
            config_env.TASK.MEASUREMENTS = ["LOOPSPL"]
            config_env.TASK.LOOPSPL.BREAKDOWN_METRIC = True
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

        config_env.TASK.VERBOSE = True

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
                    env_configs,
                    baseline_configs,
                    range(args.num_processes),
                )
            )
        )
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
    parser.add_argument("--nav-task", type=str, required=True,
                        choices=["pointnav", "loopnav"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    assert args.num_processes == 1, "Currently only single environment is " \
                                    "supported for evaulation"

    logger.add_filehandler(args.log_file)

    if args.video == 1:
        assert args.out_dir_video is not None, "Video dir not specified"

    prev_ckpt_ind = -1

    while True:
        current_ckpt = None

        while current_ckpt is None:
            current_ckpt = poll_checkpoint_folder(
                args.checkpoint_model_dir, prev_ckpt_ind
            )

            time.sleep(2)  # sleep for 2 seconds before polling again

        logger.info("current_ckpt: {}".format(current_ckpt))

        if args.video == 1:
            rgb_frames = [[]] * args.num_processes

            if not os.path.exists(args.out_dir_video):
                os.makedirs(args.out_dir_video)
        else:
            rgb_frames = None

        prev_ckpt_ind += 1

        trained_ckpt = torch.load(current_ckpt)
        trained_args = trained_ckpt["args"]

        device = torch.device("cuda:{}".format(args.pth_gpu_id))

        trained_args.num_processes = args.num_processes
        trained_args.sim_gpu_ids = args.sim_gpu_ids
        trained_args.pth_gpu_id = args.pth_gpu_id

        trained_args.nav_task = args.nav_task

        if trained_args.nav_task == "pointnav":
            key_spl = "spl"
        else:
            key_spl = "loop_spl"

        trained_args.video = args.video

        envs = construct_val_envs(trained_args)

        random.seed(args.seed)
        torch.random.manual_seed(args.seed)

        torch.backends.cudnn.deterministic = True

        actor_critic = Policy(
            observation_space=envs.observation_spaces[0],
            action_space=envs.action_spaces[0],
            hidden_size=trained_args.hidden_size,
            num_recurrent_layers=trained_args.num_recurrent_layers,
            blind=trained_args.blind,
            use_aux_losses=trained_args.use_aux_losses,
            rnn_type=trained_args.rnn_type,
            resnet_baseplanes=trained_args.resnet_baseplanes
        )

        agent = PPO(
            actor_critic=actor_critic,
            clip_param=trained_args.clip_param,
            ppo_epoch=trained_args.ppo_epoch,
            num_mini_batch=trained_args.num_mini_batch,
            value_loss_coef=trained_args.value_loss_coef,
            entropy_coef=trained_args.entropy_coef,
            lr=trained_args.lr,
            eps=trained_args.eps,
            max_grad_norm=trained_args.max_grad_norm,
        )

        agent.load_state_dict(
            {k: v for k, v in trained_ckpt["state_dict"].items()
             if "ddp" not in k}
        )

        actor_critic = agent.actor_critic
        actor_critic = actor_critic.to(device)
        actor_critic.eval()

        if trained_args.blind:
            assert actor_critic.net.cnn is None

        observations = envs.reset()
        batch = batch_obs(observations)
        for sensor in batch:
            batch[sensor] = batch[sensor].to(device)

        episode_rewards = torch.zeros(envs.num_envs, 1, device=device)
        episode_spls = torch.zeros(envs.num_envs, 1, device=device)
        episode_success = torch.zeros(envs.num_envs, 1, device=device)
        episode_counts = torch.zeros(envs.num_envs, 1, device=device)
        current_episode_reward = torch.zeros(envs.num_envs, 1, device=device)

        if key_spl == "loop_spl":
            episode_spls_stage_1 = torch.zeros(envs.num_envs, 1, device=device)
            episode_spls_stage_2 = torch.zeros(envs.num_envs, 1, device=device)
        else:
            episode_spls_stage_1 = None
            episode_spls_stage_2 = None

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

        with tqdm.tqdm(total=args.count_test_episodes * envs.num_envs) as pbar:
            while (
                episode_counts < args.count_test_episodes
            ).float().sum().item() > 0:
                with torch.no_grad():
                    _, actions, _, test_recurrent_hidden_states = \
                        actor_critic.act(
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

                for i in range(not_done_masks.shape[0]):
                    if (
                        not_done_masks[i].item() == 0
                        and episode_counts[i] < args.count_test_episodes
                    ):
                        pbar.update()
                        episode_id = int(episode_counts.sum())

                        episode_counts[i] += 1
                        episode_rewards[i] += current_episode_reward[i]

                        if key_spl == "loop_spl":
                            episode_spls[i] += infos[i][key_spl]["total_spl"]

                            episode_spls_stage_1[i] += \
                                infos[i][key_spl]["stage_1_spl"]
                            episode_spls_stage_2[i] += \
                                infos[i][key_spl]["stage_2_spl"]

                            if infos[i][key_spl]["total_spl"] > 0:
                                episode_success[i] += 1

                            logger.info("EP {}, S1 SPL: {:.3f}, "
                                        "S2 SPL: {:.3f}, "
                                        "T SPL: {:.3f}".format(
                                            episode_counts[i].item(),
                                            infos[i][key_spl]["stage_1_spl"],
                                            infos[i][key_spl]["stage_2_spl"],
                                            infos[i][key_spl]["total_spl"]))
                        else:
                            episode_spls[i] += infos[i][key_spl]
                            if infos[i][key_spl] > 0:
                                episode_success[i] += 1

                            logger.info("EP {}, T SPL: {:.3f}".format(
                                episode_counts, infos[i][key_spl]))

                        if args.video == 1:

                            if isinstance(infos[i][key_spl], dict):
                                video_name = \
                                    "{}_{}_spl1_{:.2f}_spl2_{:.2f}".format(
                                        episode_id, "apt",
                                        infos[i][key_spl]["stage_1_spl"],
                                        infos[i][key_spl]["stage_2_spl"]
                                    )
                            else:
                                video_name = "{}_{}_{}_{:.2f}".format(
                                    episode_id, "apt", key_spl,
                                    infos[i][key_spl])

                            images_to_video(
                                rgb_frames[i],
                                args.out_dir_video,
                                video_name
                            )
                            rgb_frames[i] = []

                        pbar.set_postfix(
                            dict(
                                spl=(
                                    episode_spls.sum() / episode_counts.sum()
                                ).item()
                            ),
                            refresh=True,
                        )
                    elif args.video == 1:
                        size = observations[i]["rgb"].shape[0]
                        frame = np.empty((size, 2 * size, 3), dtype=np.uint8)
                        frame[:, :size] = observations[i]["rgb"][:, :, :3]

                        if infos[i]["collisions"]["is_collision"]:
                            frame[:, 1024:] = [0, 0, 0]

                            mask = np.ones((frame.shape[0], frame.shape[1]))
                            mask[30:-30, 30:1024 - 30] = 0
                            mask = (mask == 1)
                            alpha = 0.5
                            frame[mask] = (alpha * np.array([255, 0, 0]) +
                                           (1.0 - alpha) * frame)[mask]

                        top_down_map = infos[i]["top_down_map"]["map"]
                        scale = 1024.0 / max(top_down_map.shape)
                        scale_x = scale_y = scale
                        top_down_map = maps.lut_top_down_map[top_down_map]
                        top_down_map = maps.resize_img(
                            top_down_map, round(scale * top_down_map.shape[0]),
                            round(scale * top_down_map.shape[1])
                        )

                        map_agent_pos = \
                            infos[i]["top_down_map"]["map_agent_pos"]
                        map_agent_pos[0] = int(map_agent_pos[0] * scale_x)
                        map_agent_pos[1] = int(map_agent_pos[1] * scale_y)
                        top_down_map = maps.draw_agent(
                            top_down_map,
                            map_agent_pos,
                            -infos[i]["top_down_map"]["agent_angle"] +
                            np.pi / 2,
                            agent_radius_px=7 * 4,
                        )
                        if top_down_map.shape[0] > top_down_map.shape[1]:
                            top_down_map = np.rot90(top_down_map, 1)

                        # white background
                        frame[:, 1024:] = [255, 255, 255]
                        frame[
                            :top_down_map.shape[0],
                            1024:1024 + top_down_map.shape[1]] = \
                            top_down_map
                        rgb_frames[i].append(frame)

                current_episode_reward *= not_done_masks

        episode_reward_mean = (episode_rewards / episode_counts).mean().item()
        episode_spl_mean = (episode_spls / episode_counts).mean().item()
        episode_success_mean = (episode_success / episode_counts).mean().item()

        logger.info("Checkpoint {} results:".format(current_ckpt))
        logger.info(
            "Average episode reward: {:.6f}".format(episode_reward_mean)
        )
        logger.info(
            "Average episode success: {:.6f}".format(episode_success_mean)
        )

        if key_spl == "loop_spl":
            logger.info(
                "Average episode stage-1 SPL: {:.6f}".format(
                    (episode_spls_stage_1 / episode_counts).mean().item()
                )
            )
            logger.info(
                "Average episode stage-2 SPL: {:.6f}".format(
                    (episode_spls_stage_2 / episode_counts).mean().item()
                )
            )

        logger.info(
            "Average episode {}: {:.6f}".format(key_spl, episode_spl_mean)
        )

        envs.close()


if __name__ == "__main__":
    main()
