import argparse
import imageio
import imageio_ffmpeg

import numba
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from pydash import py_
import msgpack
import msgpack_numpy
import lmdb
import os.path as osp
import cv2
from PIL import Image

from habitat_sim.utils import d3_40_colors_rgb

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


def colorize_map(_map):
    _map = _scale_up_binary(_map)
    img = Image.new("P", tuple(reversed(_map.shape)))
    img.putdata(_map.flatten().astype(np.uint8))
    img.putpalette(d3_40_colors_rgb.flatten())
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
            if np.abs(np.stack(positions, 0)).mean() < 0.0:
                continue

            xs = value["hidden_state"]
            with torch.no_grad():
                xs = torch.from_numpy(np.stack(xs, 0)).view(-1, input_size).to(device)
                visited_logits = torch.cat(
                    [model(v)[0] for v in torch.split(xs, 64, 0)], 0
                )
                occupancy_logits = torch.cat(
                    [model(v)[1] for v in torch.split(xs, 64, 0)], 0
                )

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
            for j in tqdm.trange(len(xs), leave=False):
                x, y = positions[j]
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

                gt_img = colorize_map(_map)
                pred_map = torch.argmax(visited_logits[j], -1).cpu().numpy()
                pred_map[x, y] = 2
                pred_img = colorize_map(pred_map)
                probs = F.softmax(visited_logits[j], -1).cpu()[..., 0].numpy()

                heat_map = cv2.applyColorMap(
                    (probs * 255.0).astype(np.uint8), cv2.COLORMAP_OCEAN
                )[..., ::-1]
                heat_map[x, y] = d3_40_colors_rgb[2]
                heat_map = scale_up_color(heat_map)

                occupancy_mask = create_occupancy_grid_mask(_map, napply=8)
                train_occupancy_mask = create_occupancy_grid_mask(_map, napply=2)

                _tmp = top_down_occupancy_grid.copy()
                _tmp[x, y] = 2
                gt_occupancy = colorize_map(_tmp)
                predicted_occupancy = (
                    torch.argmax(occupancy_logits[j], -1).cpu().numpy()
                )
                predicted_occupancy[x, y] = 2
                predicted_occupancy = colorize_map(predicted_occupancy)
                train_occupancy_mask = _scale_up_binary(train_occupancy_mask)
                predicted_occupancy[train_occupancy_mask == 0] = (
                    0.75 * predicted_occupancy[train_occupancy_mask == 0]
                )
                predicted_occupancy[_scale_up_binary(occupancy_mask) == 0] = 175

                occupancy_probs = (
                    F.softmax(occupancy_logits[j], -1).cpu()[..., 0].numpy()
                )
                occupancy_probs[occupancy_mask == 0] = 1.0
                occupancy_heat = cv2.applyColorMap(
                    (occupancy_probs * 255.0).astype(np.uint8), cv2.COLORMAP_OCEAN
                )[..., ::-1]
                occupancy_heat[x, y] = d3_40_colors_rgb[2]
                occupancy_heat = scale_up_color(occupancy_heat)

                def _pad(vx):
                    return np.pad(
                        vx,
                        [(3, 3), (3, 3), (0, 0)],
                        mode="constant",
                        constant_values=255,
                    )

                gt_img = _pad(gt_img)
                pred_img = _pad(pred_img)
                heat_map = _pad(heat_map)

                gt_occupancy = _pad(gt_occupancy)
                predicted_occupancy = _pad(predicted_occupancy)
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
                        int(0.05 * h) : int(0.95 * h), int(0.05 * w) : int(0.95 * w)
                    ] = 0
                    mask = mask == 1
                    img[mask] = (
                        0.75 * img[mask] + 0.25 * np.array([255, 0, 0])
                    ).astype(np.uint8)

                writer.append_data(img)

            writer.close()
            vids += 1
            pbar.update()


if __name__ == "__main__":
    main()
