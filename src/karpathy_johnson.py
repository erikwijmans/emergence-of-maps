#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import random

import imageio
import torch
import tqdm
from src.config.default import cfg as cfg_baseline
from src.rl.ppo import PPO, Policy
from src.rl.ppo.utils import batch_obs
from src.train_ppo import make_env_fn

import habitat
from habitat.config.default import get_config
from src.config.default import cfg as cfg_baseline
import numpy as np

from src.train_ppo import make_env_fn
from src.rl.ppo import PPO, Policy
from src.rl.ppo.utils import batch_obs
import tqdm
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

import numpy as np
from habitat.tasks.nav.nav_task import NavigationEpisode, NavigationGoal
from habitat.tasks.visualizations import vis_utils


vis_utils.lut_top_down_map[10:] = np.array(
    [vis_utils.b_to_r(i / 245.0) for i in range(0, 246)], dtype=np.uint8
)


def images_to_video(images, output_dir, video_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(os.path.join(output_dir, video_name), fps=10)
    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()
    print(
        "Generated video: {}".format(str(os.path.join(output_dir, video_name)))
    )


def basename(path):
    return os.path.splitext(os.path.basename(path))[0]


def calc_metrics(episode_metrics):
    metrics = {
        "avg spl": ("spl", "spl"),
        "avg success": ("spl", "success"),
        "avg d_T": ("spl", "distance_to_target"),
        "avg steps": ("pose", "step_count"),
        "avg collisions": ("pose", "collision_count"),
        "avg geo distance": ("episode_info", "info", "geodesic_distance"),
    }

    aggregated_metrics = {}
    for episode_id, metrics_values in episode_metrics.items():
        for metric, path in metrics.items():
            value = metrics_values
            for item in path:
                value = value[item]
            # print(metric, ": ", value)
            aggregated_metrics[metric] = (
                aggregated_metrics.get(metric, 0) + value
            )
            # print(metric, aggregated_metrics[metric])

    for metric in metrics.keys():
        aggregated_metrics[metric] /= len(episode_metrics.keys())
    # print("len(episode_metrics):=", len(episode_metrics.keys()))
    return aggregated_metrics


def get_run_name(args) -> str:
    if args.output_file:
        return args.output_file
    else:
        return "N{}".format(args.neuron)


def save_eval(eval_data, args):
    class DatasetJSONEncoder(json.JSONEncoder):
        def default(self, object):
            print(object)
            if type(object) == np.ndarray:
                return None
            return object.__dict__

    json_str = DatasetJSONEncoder().encode(eval_data)
    json_str = json_str.encode("utf-8")
    file_path = "{}/{}.json".format(args.output_dir, get_run_name(args))
    with open(file_path, "wb") as f:
        f.write(json_str)
    print("Evaluation file saved to: {}".format(file_path))


def quaternion_to_euler(quat_rotation):
    x, y, z, w = tuple(quat_rotation)
    import math

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    X = math.atan2(t0, t1)
    Y = math.asin(t2)
    Z = math.atan2(t3, t4)
    return [X, Y, Z]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--sim-gpu-id", type=int, required=True)
    parser.add_argument("--pth-gpu-id", type=int, required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-recurrent-layers", type=int, default=1)
    parser.add_argument("--count-test-episodes", type=int, default=1500)
    parser.add_argument("--shuffle-interval", type=int, required=True)
    parser.add_argument("--record-video", type=bool, default=False)
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument(
        "--sensors",
        type=str,
        default="RGB_SENSOR,DEPTH_SENSOR",
        help="comma separated string containing different"
        "sensors to use, currently 'RGB_SENSOR' and"
        "'DEPTH_SENSOR' are supported",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default="tasks/pointnav.yaml",
        help="path to config yaml containing information about task",
    )
    parser.add_argument("--blind", type=int, default=0)
    parser.add_argument("--use-aux-losses", type=int, default=1)
    parser.add_argument(
        "--pointgoal-sensor-type",
        type=str,
        default="SPARSE",
        choices=["DENSE", "SPARSE"],
    )
    parser.add_argument(
        "--rnn-type", type=str, default="LSTM", choices=["LSTM"]
    )
    parser.add_argument("--neuron", type=int, default=0)
    args = parser.parse_args()
    if args.blind:
        args.net = "blind"
    else:
        args.net = "vision"

    args.record_video = True

    device = torch.device("cuda:{}".format(args.pth_gpu_id))

    env_configs = []
    baseline_configs = []

    for _ in range(args.num_processes):
        config_env = get_config(config_file=args.task_config)
        config_env.defrost()
        config_env.DATASET.SPLIT = "val"
        config_env.DATASET.TYPE = "PointNav-v1"

        agent_sensors = args.sensors.strip().split(",")
        for sensor in agent_sensors:
            assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]

        config_env.SIMULATOR.HABITAT_SIM_V0.COMPRESS_TEXTURES = False

        config_env.TASK.POINTGOAL_SENSOR.SENSOR_TYPE = (
            args.pointgoal_sensor_type
        )
        if args.record_video and "RGB_SENSOR" not in agent_sensors:
            agent_sensors.append("RGB_SENSOR")
        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors
        print(
            "config_env.SIMULATOR.AGENT_0.SENSORS",
            config_env.SIMULATOR.AGENT_0.SENSORS,
        )
        config_env.TASK.MEASUREMENTS = ["SPL", "EPISODE_INFO", "POSE"]
        config_env.SIMULATOR.RGB_SENSOR.WIDTH = 1024
        config_env.SIMULATOR.RGB_SENSOR.HEIGHT = 1024
        config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = 1024
        config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = 1024

        config_env.freeze()
        env_configs.append(config_env)

        config_baseline = cfg_baseline()
        baseline_configs.append(config_baseline)

    random.seed(env_configs[0].SEED)
    torch.random.manual_seed(env_configs[0].SEED)
    torch.backends.cudnn.deterministic = True
    dummy_dataset = PointNavDatasetV1(env_configs[0].DATASET)

    if len(dummy_dataset.episodes) > args.count_test_episodes:
        dummy_dataset.episodes = dummy_dataset.episodes[
            : args.count_test_episodes
        ]
    assert len(baseline_configs) > 0, "empty list of datasets"

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(
                    env_configs,
                    baseline_configs,
                    [args.shuffle_interval for _ in range(args.num_processes)],
                    range(args.num_processes),
                )
            )
        ),
    )
    print("if args.record_video", args.record_video)
    if args.record_video and args.sensors == "DEPTH_SENSOR":
        # Remove RGB as input for critic
        del envs.observation_spaces[0].spaces["rgb"]

    ckpt = torch.load(args.model_path, map_location=device)
    print(envs.observation_spaces[0])

    actor_critic = Policy(
        observation_space=envs.observation_spaces[0],
        action_space=envs.action_spaces[0],
        hidden_size=args.hidden_size,
        num_recurrent_layers=args.num_recurrent_layers,
        blind=args.blind,
        use_aux_losses=args.use_aux_losses,
        rnn_type=args.rnn_type,
    )
    actor_critic.to(device)
    if args.blind:
        assert actor_critic.net.cnn is None

    ppo = PPO(
        actor_critic=actor_critic,
        clip_param=0.1,
        ppo_epoch=4,
        num_mini_batch=32,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        lr=2.5e-4,
        eps=1e-5,
        max_grad_norm=0.5,
    )

    ppo.load_state_dict(
        {k: v for k, v in ckpt["state_dict"].items() if "ddp" not in k}
    )

    actor_critic = ppo.actor_critic
    actor_critic.eval()

    observations = envs.reset()
    batch = batch_obs(observations)
    for sensor in batch:
        batch[sensor] = batch[sensor].to(device)

    episode_rewards = torch.zeros(envs.num_envs, 1, device=device)
    episode_spls = torch.zeros(envs.num_envs, 1, device=device)
    episode_success = torch.zeros(envs.num_envs, 1, device=device)
    episode_counts = torch.zeros(envs.num_envs, 1, device=device)
    current_episode_reward = torch.zeros(envs.num_envs, 1, device=device)

    test_recurrent_hidden_states = torch.zeros(
        actor_critic.net.num_recurrent_layers,
        args.num_processes,
        args.hidden_size,
        device=device,
    )
    not_done_masks = torch.zeros(args.num_processes, 1, device=device)
    prev_actions = torch.zeros(
        args.num_processes, 1, device=device, dtype=torch.int64
    )

    rgb_frames = [[]] * args.num_processes
    output_dir = "data/eval/karpathy_johnson_v2/{}".format(get_run_name(args))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    episode_metrics = {}
    pbar = tqdm.tqdm(total=len(dummy_dataset.episodes))
    print(
        "UNIQUE EPISODES ID: ",
        {ep.episode_id for ep in dummy_dataset.episodes},
    )
    map_changes = []
    while episode_counts.sum() < len(dummy_dataset.episodes):
        with torch.no_grad():
            _, actions, _, test_recurrent_hidden_states = actor_critic.act(
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

        for i in range(not_done_masks.shape[0]):
            if not_done_masks[i].item() == 0:
                pbar.update(1)
                episode_id = int(episode_counts.sum())
                episode_counts.sum()
                scene_path = infos[i]["episode_info"]["scene_id"]
                scene_name = basename(scene_path)
                print(episode_id, infos[i]["spl"])
                episode_metrics[episode_id] = infos[i]
                if args.record_video:
                    images_to_video(
                        rgb_frames[i],
                        output_dir,
                        "{id}_{scene_name}_spl={spl:.2f}_d={dist:.2f}".format(
                            id=episode_id,
                            scene_name=scene_name,
                            spl=infos[i]["spl"]["spl"],
                            dist=infos[i]["episode_info"]["info"][
                                "geodesic_distance"
                            ],
                        ),
                    )
                    rgb_frames[i] = []
                episode_spls[i] += infos[i]["spl"]["spl"]
                if infos[i]["spl"]["spl"] > 0:
                    episode_success[i] += 1
                del episode_metrics[episode_id]["pose"]["map"]
                del episode_metrics[episode_id]["pose"]["agent_angle"]
                map_changes = []
            else:
                if args.record_video:
                    size = observations[i]["rgb"].shape[0]
                    frame = np.empty((size, 2 * size, 3), dtype=np.uint8)
                    frame[:, :size] = observations[i]["rgb"][:, :, :3]

                    if "pose" in infos[i] and infos[i]["pose"]["is_collision"]:
                        frame[:, 1024:] = [0, 0, 0]

                        mask = np.ones((frame.shape[0], frame.shape[1]))
                        mask[20:-20, 20:1004] = 0
                        mask = mask == 1
                        alpha = 0.5
                        frame[mask] = (
                            alpha * np.array([255, 0, 0])
                            + (1.0 - alpha) * frame
                        )[mask]
                    agent_position = infos[i]["pose"]["position"]
                    agent_rotation = -quaternion_to_euler(
                        infos[i]["pose"]["rotation"]
                    )[1]

                    map = infos[i]["pose"]["map"]
                    ax, ay = infos[i]["pose"]["prev_map_agent_pos"]
                    map_changes.append(
                        (
                            (ax, ay),
                            int(
                                (
                                    test_recurrent_hidden_states[
                                        1, i, args.neuron
                                    ]
                                    .tanh()
                                    .item()
                                    + 1.0
                                )
                                / 2.0
                                * 245
                            )
                            + 10,
                        )
                    )
                    for (ax, ay), v in map_changes:
                        map[ax, ay] = v
                        map[ax - 3 : ax + 3, ay - 3 : ay + 3] = v

                    scale = 1024.0 / max(map.shape)
                    scale_x = scale_y = scale
                    map = vis_utils.lut_top_down_map[map]
                    map = vis_utils.resize_img(
                        map,
                        round(scale * map.shape[0]),
                        round(scale * map.shape[1]),
                    )
                    map_agent_pos = infos[i]["pose"]["map_agent_pos"]
                    map_agent_pos[0] = int(map_agent_pos[0] * scale_x)
                    map_agent_pos[1] = int(map_agent_pos[1] * scale_y)
                    #  map = vis_utils.draw_agent(
                    #  map,
                    #  map_agent_pos,
                    #  infos[i]["pose"]["agent_angle"] - np.pi / 2,
                    #  agent_radius_px=7,
                    #  )
                    if map.shape[0] > map.shape[1]:
                        map = np.rot90(map, 1)
                    # white background
                    frame[:, 1024:] = [255, 255, 255]
                    frame[: map.shape[0], 1024 : 1024 + map.shape[1]] = map
                    rgb_frames[i].append(frame)

        rewards = torch.tensor(
            rewards, dtype=torch.float, device=device
        ).unsqueeze(1)
        current_episode_reward += rewards
        episode_rewards += (1 - not_done_masks) * current_episode_reward
        episode_counts += 1 - not_done_masks
        current_episode_reward *= not_done_masks

    print("Episode agg metrics:\n")
    agg_metrics = calc_metrics(episode_metrics)
    for k, v in calc_metrics(episode_metrics).items():
        print("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
