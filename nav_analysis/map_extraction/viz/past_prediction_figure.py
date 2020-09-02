import argparse
import os
import os.path as osp

import cv2
import imageio
import imageio_ffmpeg
import lmdb
import msgpack
import msgpack_numpy
import numba
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from habitat_sim.utils.common import (
    d3_40_colors_rgb,
    quat_from_coeffs,
    quat_rotate_vector,
)
from PIL import Image
from pydash import py_

from habitat.tasks.nav.nav_task import _SE3
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.visualizations import maps
from nav_analysis.map_extraction.training.train_visited_predictor import (
    Model,
    create_occupancy_grid_mask,
)
from nav_analysis.map_extraction.viz.top_down_occ_figure import (
    make_groundtruth,
    to_grid,
)
from nav_analysis.map_extraction.viz.make_visited_videos import (
    colorize_map,
    scale_up_color,
    _scale_up_binary,
)

msgpack_numpy.patch()

num_bins = np.array([96, 96])
bin_size = 0.25
bin_range = np.arange(num_bins[0] + 1)
bin_range = (bin_range - bin_range.max() / 2) * bin_size
bins = [bin_range.copy(), bin_range.copy()]


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--trajectories", type=str, required=True)
    parser.add_argument("--num-images", type=float, default=1e2)

    parser.add_argument("--output-path", type=str, required=True)

    return parser


def main():
    scaling_factor = 6
    args = build_parser().parse_args()
    args.num_images = int(args.num_images)
    os.makedirs(args.output_path, exist_ok=True)

    device = torch.device("cuda")
    input_size = 512 * 6
    model = Model(input_size, num_bins)
    model.to(device)
    model.eval()

    trained_ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(trained_ckpt)

    env = lmdb.open(args.trajectories)
    length = env.stat()["entries"]
    print(length)
    prng = np.random.RandomState(0)
    ordering = prng.permutation(length)
    with env.begin(buffers=True) as txn:
        for i in tqdm.trange(74, 74 + 1):
            v = txn.get(str(ordering[i]).encode())
            value = msgpack.unpackb(v, raw=False)

            actions = value["actions"]
            stop_idxs = [i for i in range(len(actions)) if actions[i] == 3]
            assert len(stop_idxs) == 1
            stop_idx = stop_idxs[0]
            positions = value["positions"]
            agent_route = positions[0:stop_idx]

            xs = value["hidden_state"][stop_idx : stop_idx + 1]
            with torch.no_grad():
                xs = torch.from_numpy(np.stack(xs, 0)).view(-1, input_size).to(device)

                model_outputs = [model(v) for v in torch.split(xs, 64, 0)]

                past_visited_logits = torch.cat([m[0] for m in model_outputs], 0)
                future_visited_logits = torch.cat([m[1] for m in model_outputs], 0)
                occupancy_logits = torch.cat([m[2] for m in model_outputs], 0)

                d_start_preds = torch.cat([m[3] for m in model_outputs], 0)
                d_goal_preds = torch.cat([m[4] for m in model_outputs], 0)

            past_visited_probs = (
                F.softmax(past_visited_logits[0], -1).cpu()[..., 1:].numpy()
            )
            past_heat_map = (
                (1 - past_visited_probs) * np.array([[[255, 255, 255]]])
                + past_visited_probs * np.array([[[0, 0, 255]]])
            ).astype(np.uint8)

            def draw_path(route, _map):
                _map = _map.copy()

                def _convert_pt(pt):
                    return tuple(reversed(to_grid(pt, num_bins=_map.shape[0])))

                prev_pt = route[0]
                for i in range(1, len(route)):
                    beta = 0.0 + 1.0 * (i / (len(route) - 1))
                    beta = 1.0
                    color = tuple(
                        (
                            np.array([52, 152, 219]) * beta
                            + (1 - beta) * np.array([22, 160, 133])
                        ).tolist()
                    )

                    cv2.line(
                        _map,
                        _convert_pt(prev_pt),
                        _convert_pt(route[i]),
                        color,
                        5,
                        lineType=cv2.LINE_8,
                    )

                    prev_pt = route[i]

                x, y = to_grid([0, 0, 0], num_bins=_map.shape[0])
                _map[x, y] = 0

                goal_pos = np.array(value["episode"]["goal"]["position"])
                transform_world_start = _SE3(
                    quat_from_coeffs(value["episode"]["start_rotation"]),
                    np.array(value["episode"]["start_position"]),
                )
                goal_pos = transform_world_start.inv() * np.array(goal_pos)
                x, y = to_grid(goal_pos, num_bins=_map.shape[0])
                if (0 <= x < _map.shape[0]) and (0 <= y < _map.shape[1]):
                    _map[x, y] = 0

                return _map

            past_heat_map = scale_up_color(past_heat_map, scaling_factor)
            past_visited_probs = scale_up_color(past_visited_probs, scaling_factor)
            mask = np.all(past_heat_map > 220, -1) == 0

            gt = make_groundtruth(value["episode"], past_heat_map.shape[0])
            color_map = [[236, 240, 241], [149, 165, 166], [255, 255, 255]]
            gt = colorize_map(gt, color_map)

            route = draw_path(agent_route, gt)
            route = gt.copy()

            probs_in_mask = past_visited_probs[mask]
            probs_in_mask = (probs_in_mask - probs_in_mask.min()) / (
                probs_in_mask.max() - probs_in_mask.min()
            )
            probs_in_mask = probs_in_mask / probs_in_mask[probs_in_mask < 1].max()
            probs_in_mask[probs_in_mask > 1] = 1
            past_heat_map = (
                (1 - probs_in_mask) * np.array([[[255, 255, 255]]])
                + probs_in_mask * np.array([[[0, 0, 255]]])
            ).astype(np.uint8)

            img = route.copy()
            img[mask] = 0.35 * img[mask] + 0.65 * past_heat_map

            img = np.clip(img, 0, 255).astype(np.uint8)

            image_name = f"{i}.png"

            imageio.imwrite(osp.join(args.output_path, image_name), img)


if __name__ == "__main__":
    main()
