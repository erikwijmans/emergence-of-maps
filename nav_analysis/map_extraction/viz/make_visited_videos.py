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

msgpack_numpy.patch()

num_bins = np.array([96, 96])
bin_size = 0.25
bin_range = np.arange(num_bins[0] + 1)
bin_range = (bin_range - bin_range.max() / 2) * bin_size
bins = [bin_range.copy(), bin_range.copy()]


@numba.jit(nopython=True, parallel=True)
def _scale_up_binary(img, scaling: int = 4):
    h, w = img.shape
    new_img = np.zeros((h * scaling, w * scaling), dtype=img.dtype)
    for j in range(h * scaling):
        for i in range(w * scaling):
            new_img[j, i] = img[j // scaling, i // scaling]

    return new_img


@numba.jit(nopython=True, parallel=True)
def scale_up_color(img, scaling: int = 4):
    h, w, c = img.shape
    new_img = np.zeros((h * scaling, w * scaling, c), dtype=img.dtype)
    for j in range(h * scaling):
        for i in range(w * scaling):
            new_img[j, i] = img[j // scaling, i // scaling]

    return new_img


def colorize_map(_map, coloring=d3_40_colors_rgb):
    img = Image.new("P", tuple(reversed(_map.shape)))
    img.putdata(_map.flatten().astype(np.uint8))
    img.putpalette(np.array(coloring, dtype=np.uint8).flatten())
    img = img.convert("RGBA")
    img = np.array(img)

    return img[..., 0:3]


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--trajectories", type=str, required=True)
    parser.add_argument("--num-videos", type=float, default=5e1)

    parser.add_argument("--output-path", type=str, required=True)

    return parser


def main():
    args = build_parser().parse_args()
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
    with env.begin(buffers=True) as txn, tqdm.tqdm(total=int(args.num_videos)) as pbar:
        i = 0
        vids = 0
        while vids < int(args.num_videos):
            v = txn.get(str(ordering[i]).encode())
            value = msgpack.unpackb(v, raw=False)
            i += 1
            positions = value["positions"]
            goal_pos = np.array(value["episode"]["goal"]["position"])
            if np.abs(np.stack(positions, 0)).mean() < 0.0:
                continue

            xs = value["hidden_state"]
            with torch.no_grad():
                xs = torch.from_numpy(np.stack(xs, 0)).view(-1, input_size).to(device)

                model_outputs = [model(v) for v in torch.split(xs, 64, 0)]

                past_visited_logits = torch.cat([m[0] for m in model_outputs], 0)
                future_visited_logits = torch.cat([m[1] for m in model_outputs], 0)
                occupancy_logits = torch.cat([m[2] for m in model_outputs], 0)

                d_start_preds = torch.cat([m[3] for m in model_outputs], 0)
                d_goal_preds = torch.cat([m[4] for m in model_outputs], 0)

            scene_id = osp.splitext(osp.basename(value["episode"]["scene_id"]))[0]
            epid = value["episode"]["episode_id"]
            vid_name = f"{i}-epid={epid}-scene={scene_id}-nacts={len(xs)}-spl={value['spl']:.3f}.mp4"
            #  vid_name = f"{i}-nacts={len(xs)}.mp4"
            tqdm.tqdm.write(vid_name)
            writer = imageio.get_writer(
                osp.join(args.output_path, vid_name), fps=10, quality=5
            )

            top_down_occupancy_grid = value["top_down_occupancy_grid"]
            _map = np.zeros(num_bins, dtype=np.int64)
            prev_xy = None
            collisions = value["collision"]
            transform_world_start = _SE3(
                quat_from_coeffs(value["episode"]["start_rotation"]),
                np.array(value["episode"]["start_position"]),
            )
            goal_pos = transform_world_start.inv() * np.array(goal_pos)
            gx = int(np.searchsorted(bins[0], [goal_pos[0]])[0])
            gy = int(np.searchsorted(bins[1], [goal_pos[2]])[0])
            for j in tqdm.trange(len(xs), leave=False):
                x, _, y = positions[j]
                x = int(np.searchsorted(bins[0], [x])[0])
                y = int(np.searchsorted(bins[1], [y])[0])

                if x <= 0 or x >= _map.shape[0]:
                    continue
                if y <= 0 or y >= _map.shape[1]:
                    continue

                if prev_xy is not None:
                    px, py = prev_xy
                    _map[px, py] = 1

                prev_xy = (x, y)

                _map[x, y] = 2

                future_visited_probs = (
                    F.softmax(future_visited_logits[j], -1).cpu()[..., 1:].numpy()
                )
                past_visited_probs = (
                    F.softmax(past_visited_logits[j], -1).cpu()[..., 1:].numpy()
                )

                visited_thresh = 0.35
                future_pred_map = np.squeeze(
                    np.where(future_visited_probs > visited_thresh, 1, 0), -1
                )
                future_pred_map = np.array(
                    [[255, 255, 255], [255, 0, 0]], dtype=np.uint8
                )[future_pred_map]

                past_pred_map = np.squeeze(
                    np.where(past_visited_probs > visited_thresh, 1, 0), -1
                )
                past_pred_map = np.array(
                    [[255, 255, 255], [0, 0, 255]], dtype=np.uint8
                )[past_pred_map]

                pred_img = (0.5 * past_pred_map + 0.5 * future_pred_map).astype(
                    np.uint8
                )
                pred_img[x, y] = d3_40_colors_rgb[2]

                future_heat_map = (
                    (1 - future_visited_probs) * np.array([[[255, 255, 255]]])
                    + future_visited_probs * np.array([[[255, 0, 0]]])
                ).astype(np.uint8)

                past_heat_map = (
                    (1 - past_visited_probs) * np.array([[[255, 255, 255]]])
                    + past_visited_probs * np.array([[[0, 0, 255]]])
                ).astype(np.uint8)

                heat_map = (1.0 * past_heat_map + 0.0 * future_heat_map).astype(
                    np.uint8
                )

                heat_map[x, y] = d3_40_colors_rgb[2]

                occupancy_mask = create_occupancy_grid_mask(_map, napply=8)
                train_occupancy_mask = create_occupancy_grid_mask(_map, napply=2)
                occupancy_mask = np.ones_like(occupancy_mask)
                train_occupancy_mask = np.ones_like(train_occupancy_mask)

                _top_down_occ = top_down_occupancy_grid.copy()
                _top_down_occ = (
                    torch.from_numpy(_top_down_occ)
                    .float()
                    .view(1, 1, *_top_down_occ.shape)
                )
                _top_down_occ = F.avg_pool2d(_top_down_occ, 2).squeeze() * 4
                _top_down_occ = _top_down_occ >= 2
                _tmp = _top_down_occ.numpy().astype(np.uint8)
                _tmp[x, y] = 2

                gt_occupancy = colorize_map(_tmp)
                predicted_occupancy = (
                    torch.argmax(occupancy_logits[j], -1).cpu().numpy()
                )
                predicted_occupancy[x, y] = 2
                predicted_occupancy = colorize_map(predicted_occupancy)
                predicted_occupancy[train_occupancy_mask == 0] = (
                    0.75 * predicted_occupancy[train_occupancy_mask == 0]
                )
                predicted_occupancy[occupancy_mask == 0] = 175

                occupancy_probs = (
                    F.softmax(occupancy_logits[j], -1).cpu()[..., 0].numpy()
                )
                occupancy_probs[occupancy_mask == 0] = 1.0
                occupancy_heat = cv2.applyColorMap(
                    (occupancy_probs * 255.0).astype(np.uint8), cv2.COLORMAP_OCEAN
                )[..., ::-1]
                occupancy_heat[x, y] = d3_40_colors_rgb[2]

                def _pad(vx):
                    return np.pad(
                        vx,
                        [(3, 3), (3, 3), (0, 0)],
                        mode="constant",
                        constant_values=255,
                    )

                goal_color_key = 4

                def _add_goal(img):
                    if 0 <= gx < img.shape[0] and 0 <= gy < img.shape[1]:
                        img[gx, gy] = d3_40_colors_rgb[goal_color_key]

                ref_rotation = quat_from_coeffs(value["rotations"][j])

                heading_vector = quat_rotate_vector(
                    ref_rotation.inverse(), np.array([0, 0, -1])
                )

                phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]

                def _draw_agent(img):
                    return maps.draw_agent(
                        img, (4 * x + 2, 4 * y + 2), phi + np.pi / 2, agent_radius_px=7
                    )

                gt_img = colorize_map(_map)

                _add_goal(gt_img)
                gt_img = scale_up_color(gt_img)
                gt_img = _draw_agent(gt_img)
                gt_img = _pad(gt_img)

                _add_goal(pred_img)
                pred_img = scale_up_color(pred_img)
                pred_img = _draw_agent(pred_img)
                pred_img = _pad(pred_img)

                _add_goal(heat_map)
                heat_map = scale_up_color(heat_map)
                heat_map = _draw_agent(heat_map)
                heat_map = _pad(heat_map)

                _add_goal(gt_occupancy)
                gt_occupancy = scale_up_color(gt_occupancy)
                gt_occupancy = _draw_agent(gt_occupancy)
                gt_occupancy = _pad(gt_occupancy)

                _add_goal(predicted_occupancy)
                predicted_occupancy = scale_up_color(predicted_occupancy)
                predicted_occupancy = _draw_agent(predicted_occupancy)
                predicted_occupancy = _pad(predicted_occupancy)

                _add_goal(occupancy_heat)
                occupancy_heat = scale_up_color(occupancy_heat)
                occupancy_heat = _draw_agent(occupancy_heat)
                occupancy_heat = _pad(occupancy_heat)

                img = np.concatenate((heat_map, pred_img, gt_img), 1)
                img2 = np.concatenate(
                    (occupancy_heat, predicted_occupancy, gt_occupancy), 1
                )
                img = np.concatenate((img, img2), 0)

                if collisions[j]:
                    h, w, _ = img.shape
                    mask = np.ones((h, w))
                    mask[
                        int(0.025 * h) : int(0.975 * h), int(0.025 * w) : int(0.975 * w)
                    ] = 0
                    mask = mask == 1
                    img[mask] = (
                        0.75 * img[mask] + 0.25 * np.array([255, 0, 0])
                    ).astype(np.uint8)

                h, w, _ = img.shape

                def _draw_text(img, txt):
                    return cv2.putText(
                        img,
                        txt,
                        (10, img.shape[0] // 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (25, 25, 25),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

                visited_label = np.full((h // 2, w // 8, 3), 255, dtype=np.uint8)
                visited_label = _draw_text(visited_label, "visited")

                occ_label = np.full((h // 2, w // 8, 3), 255, dtype=np.uint8)
                occ_label = _draw_text(occ_label, "occ.")

                lbl = np.concatenate([visited_label, occ_label], axis=0)
                img = np.concatenate([lbl, img], axis=1)

                d_start_bar = np.full((h // 2, w // 8, 3), 255, dtype=np.uint8)
                d_start_bar = _draw_text(d_start_bar, "d_start")
                d_start_bar[h // 4 :] = _draw_text(
                    d_start_bar[h // 4 :], "{:2.3f}".format(d_start_preds[j].item())
                )

                d_goal_bar = np.full((h // 2, w // 8, 3), 255, dtype=np.uint8)
                d_goal_bar = _draw_text(d_goal_bar, "d_goal")
                d_goal_bar[h // 4 :] = _draw_text(
                    d_goal_bar[h // 4 :], "{:2.3f}".format(d_goal_preds[j].item())
                )

                d_bar = np.concatenate([d_start_bar, d_goal_bar], 0)
                img = np.concatenate([img, d_bar], 1)

                writer.append_data(img)

            writer.close()
            vids += 1
            pbar.update()


if __name__ == "__main__":
    main()
