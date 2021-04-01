import argparse
import contextlib
import glob
import os.path as osp
import numpy as np
import gym
import math

import h5py as h5
import imageio
import numba
import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import lmdb
import msgpack
import msgpack_numpy
import tqdm
import skimage
import skimage.draw
from scipy import ndimage
from torch.utils import tensorboard
from collections import OrderedDict, defaultdict

import habitat_sim
from habitat_sim.utils.common import (
    d3_40_colors_rgb,
    quat_from_coeffs,
    quat_rotate_vector,
    quat_from_angle_axis,
    quat_from_two_vectors,
    angle_between_quats,
)

from nav_analysis.train_ppo import filter_obs
from nav_analysis.evaluate_ppo import StreamingMean
from nav_analysis.rl.ppo import Policy
from nav_analysis.rl.ppo.utils import batch_obs
from habitat.tasks.nav.nav_task import _SE3
from nav_analysis.map_extraction.training.train_visited_predictor import (
    Model,
    focal_loss,
    create_occupancy_grid_mask,
    num_bins,
)
from nav_analysis.map_extraction.viz.top_down_occ_figure import (
    to_grid,
    colorize_map,
    scale_up_color,
)

num_bins = np.array([96, 96])
bin_size = 0.25
bin_range = np.arange(num_bins[0] + 1)
bin_range = (bin_range - bin_range.max() / 2) * bin_size
bins = [bin_range.copy(), bin_range.copy()]

probe_model_ckpts = [
    "data/checkpoints/probes/mp3d-gibson-all-teleportnav-stage-2-trained-state-final-run_0_{}-blind/ckpt.{}.pth".format(
        *v
    )
    for v in [(0, 40), (1, 40), (2, 40), (0, 40), (4, 40)]
]
#  probe_model_ckpts = [
#  "data/checkpoints/probes/mp3d-gibson-all-teleportnav-stage-2-trained-state-final-run_0_{}-no-inputs/ckpt.{}.pth".format(
#  *v
#  )
#  for v in [(0, 145), (1, 121), (2, 147), (0, 148), (4, 137)]
#  ]


def mean_of_angles(angles):
    angles = np.array(angles)
    return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))


def _angle_distance(a1, a2):
    v1 = np.array([np.cos(a1), np.sin(a1)])
    v2 = np.array([np.cos(a2), np.sin(a2)])

    return np.arccos(np.dot(v1, v2))

    #  return np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], np.dot(v1, v2))


def mean_of_angles_streaming_mean(count, mean, a):
    mean = np.arctan2(np.sin(mean) * count + np.sin(a), np.cos(mean) * count, np.cos(a))

    return count + 1, mean


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--trajectories", type=str, required=True)
    parser.add_argument("--epsilon", default=20.0, type=float)

    return parser


@numba.njit(parallel=True)
def reflect(img, m, b):
    reflected = img.copy()
    for y in numba.prange(img.shape[0]):
        for x in range(img.shape[1]):
            u = int(((1 - m * m) * x + 2 * m * y - 2 * m * b) / (m * m + 1))
            v = int(((m * m - 1) * y + 2 * m * x + 2 * b) / (m * m + 1))

            if u >= 0 and u < img.shape[1] and v >= 0 and v < img.shape[0]:
                reflected[y, x] = img[v, u]
            else:
                reflected[y, x] = 0

    return reflected


def make_target_map_reflection(grid, mask, goal_pos):
    grid = grid.copy()
    mask = mask.copy()

    origin = to_grid([0, 0, 0])

    m = (origin[0] - goal_pos[0]) / (origin[1] - goal_pos[1] + 1e-5)
    b = -m * origin[1] + origin[0]

    mask = reflect(mask, m, b)
    grid = reflect(grid, m, b)

    mask = create_occupancy_grid_mask(mask, napply=5)

    return grid, mask


def make_target_map(grid, mask, goal_pos):
    grid = grid.copy()
    mask = mask.copy()

    origin = to_grid([0, 0, 0])
    rr, cc, _ = skimage.draw.line_aa(*origin, *goal_pos)
    for i in range(len(rr)):
        if not 0 <= rr[i] < mask.shape[0] or not 0 <= cc[i] < mask.shape[1]:
            rr = rr[0:i]
            cc = cc[0:i]
            break

    mask[rr, cc] = 1

    mask = create_occupancy_grid_mask(mask, napply=5)

    grid[rr, cc] = 1

    rr_shift = np.clip(rr + 1, 0, grid.shape[0] - 1)
    cc_shift = np.clip(cc + 1, 0, grid.shape[1] - 1)
    grid[rr_shift, cc_shift] = 1

    rr_shift = np.clip(rr - 1, 0, grid.shape[0] - 1)
    cc_shift = np.clip(cc - 1, 0, grid.shape[1] - 1)
    grid[rr_shift, cc_shift] = 1

    return grid, mask


def pgd_attack(
    model,
    orig_hidden_state: torch.Tensor,
    target_map,
    mask,
    epsilon=5.0,
    lr=1e-2,
    norm_type="l2",
):
    assert norm_type in {"l2", "inf"}
    orig_hidden_state = orig_hidden_state.detach().clone()

    if norm_type == "l2":
        delta = torch.randn_like(orig_hidden_state)
        norm = torch.norm(delta, dim=1, keepdim=True)
        r = (torch.rand_like(norm) + 1e-3) ** (1 / 2.0)
        delta = epsilon / 10.0 * r / norm * delta
    else:
        delta = (2 * torch.rand_like(orig_hidden_state) - 1) * epsilon

    hidden_state = orig_hidden_state.clone().detach() + delta
    hidden_state.requires_grad = True

    max_iters = int(2.5 * epsilon / lr)

    with tqdm.tqdm(total=max_iters, leave=False) as pbar:
        for _ in range(max_iters):
            pre_step = hidden_state.detach().clone()
            hidden_state.requires_grad = True
            hidden_state.grad = None
            pred = model(hidden_state)
            loss = focal_loss(pred, target_map, mask)
            loss = loss.mean()
            loss.backward()

            grad = hidden_state.grad

            if norm_type == "l2":
                grad_norm = torch.norm(grad, dim=1, keepdim=True)
                tmp = hidden_state.clone()
                hidden_state.data.add_(grad / grad_norm, alpha=-lr)
            else:
                hidden_state.data.add_(torch.sign(grad), alpha=-lr)

            with torch.no_grad():
                if norm_type == "l2":
                    norm = torch.norm(
                        hidden_state - orig_hidden_state, dim=1, keepdim=True
                    )
                    shift = (
                        torch.where(
                            norm > epsilon, torch.full_like(norm, epsilon), norm
                        )
                        / norm
                    )

                    hidden_state.copy_(
                        orig_hidden_state + shift * (hidden_state - orig_hidden_state)
                    )
                else:
                    hidden_state.copy_(
                        orig_hidden_state
                        + torch.clamp(
                            hidden_state - orig_hidden_state, -epsilon, epsilon
                        )
                    )

                pbar.set_postfix(loss=loss.item(), norm=norm.mean().item())
                pbar.update()

    with torch.no_grad():
        pred = model(hidden_state)

    return hidden_state, pred


def save_examples(gt_grid, target_grid, pgd_grid, gt_mask, target_mask, idx):
    def _to_cpu(t):
        return t.cpu().squeeze(0).numpy()

    gt_grid = _to_cpu(gt_grid)
    target_grid = _to_cpu(target_grid)
    pgd_grid = _to_cpu(pgd_grid)

    gt_mask = _to_cpu(gt_mask)
    target_mask = _to_cpu(target_mask)

    gt_grid[gt_mask == 0] = 2
    target_grid[target_mask == 0] = 2
    pgd_grid[target_mask == 0] = 2

    color_map = [[236, 240, 241], [149, 165, 166], [255, 255, 255]]

    gt_map = scale_up_color(colorize_map(gt_grid, color_map))
    target_map = scale_up_color(colorize_map(target_grid, color_map))
    pgd_map = scale_up_color(colorize_map(pgd_grid, color_map))

    imageio.imwrite("pgd_viz/gt{}.png".format(idx), gt_map)
    imageio.imwrite("pgd_viz/target{}.png".format(idx), target_map)
    imageio.imwrite("pgd_viz/pgd_pred{}.png".format(idx), pgd_map)


def get_heading_of_first_forward(
    probe: Policy,
    hidden_state,
    goal,
    transform_world_start: _SE3,
    initial_position,
    initial_rotation,
):
    transform_start_world = transform_world_start.inv()
    device = hidden_state.device
    prev_action = torch.full((1, 1), 3, device=device, dtype=torch.long)
    delta_heading = 0

    def _build_obs(rotation, position):
        pg = transform_world_start.inv() * goal
        v = np.array([-pg[2], pg[0]])

        rho = np.linalg.norm(v)
        if rho > 1e-5:
            pg = np.array([rho] + v.tolist(), dtype=np.float32)
        else:
            pg = np.array([0, 1, 0], dtype=np.float32)

        dg = np.linalg.norm(position - goal)
        dg = np.array([min(dg, 0.5)], dtype=np.float32)
        stage = np.array([1], dtype=np.long)

        transform_world_curr = _SE3(rotation, position)
        transform_start_curr = transform_start_world * transform_world_curr
        look_dir = np.array([0, 0, -1], dtype=np.float32)
        heading_vector = quat_rotate_vector(transform_start_curr.rot, look_dir)
        pos = transform_start_curr.trans

        gps_and_compass = np.concatenate([heading_vector, pos]).astype(np.float32)

        obs = dict(
            pointgoal=pg,
            gps_and_compass=gps_and_compass,
            dist_to_goal=dg,
            episode_stage=stage,
        )

        obs = filter_obs(obs, True)

        return obs

    position = initial_position
    rotation = initial_rotation

    not_done_masks = torch.ones(1, 1, device=device)
    while True:
        obs = batch_obs([_build_obs(rotation, position)], device)
        with torch.no_grad():
            (_, actions, _, _, hidden_state) = probe.act(
                obs, hidden_state, prev_action, not_done_masks, deterministic=False
            )
        prev_action.copy_(actions)
        actions = actions.item()

        if actions == 0:
            break
        elif actions == 1:
            rotation = rotation * quat_from_angle_axis(
                np.deg2rad(10), habitat_sim.geo.UP
            )
            delta_heading += 10
        elif actions == 2:
            rotation = rotation * quat_from_angle_axis(
                np.deg2rad(-10), habitat_sim.geo.UP
            )
            delta_heading -= 10
        elif actions == 3:
            break
        else:
            raise RuntimeError(f"Unknown action: {actions}")

    return rotation, delta_heading


def main():
    args = build_parser().parse_args()

    device = torch.device("cuda")
    input_size = 512 * 6
    model = Model(input_size, num_bins)
    model.to(device)
    model.eval()

    trained_ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(trained_ckpt)

    observation_space = gym.spaces.Dict(
        {
            "pointgoal": gym.spaces.Box(1, 1, (3,)),
            "gps_and_compass": gym.spaces.Box(1, 1, (6,)),
            "episode_stage": gym.spaces.Box(1, 1, (1,)),
            "dist_to_goal": gym.spaces.Box(1, 1, (1,)),
        }
    )

    probes = []
    for ckpt in probe_model_ckpts:
        trained_ckpt = torch.load(ckpt, map_location=device)
        trained_args = trained_ckpt["args"]
        policy_kwargs = dict(
            observation_space=observation_space,
            action_space=gym.spaces.Discrete(4),
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
        probe = Policy(**policy_kwargs)
        probe.to(device)
        probe.load_state_dict(
            {
                k[len("actor_critic.") :]: v
                for k, v in trained_ckpt["state_dict"].items()
                if "ddp" not in k and "actor_critic" in k
            }
        )
        probe.eval()

        probes.append(probe)

    occ_model = model.occ_model
    for param in occ_model.parameters():
        param.requires_grad = False

    def _to_device(t):
        return torch.from_numpy(t.copy()).to(device=device).unsqueeze(0)

    env = lmdb.open(args.trajectories, readonly=True, lock=False)
    length = env.stat()["entries"]
    prng = np.random.RandomState(0)
    ordering = prng.permutation(length)

    avg_heading_errors = OrderedDict(
        [
            ("pgd", StreamingMean()),
            ("control", StreamingMean()),
            ("orig", StreamingMean()),
            ("zero", StreamingMean()),
        ]
    )

    #  length = 50
    with env.begin(buffers=True) as txn, tqdm.tqdm(total=length) as pbar:
        for idx in range(length):
            v = txn.get(str(ordering[idx]).encode())
            value = msgpack.unpackb(v, raw=False)
            actions = value["actions"]
            xs = value["hidden_state"]
            positions = value["positions"]
            rotations = value["rotations"]

            stop_idxs = [i for i in range(len(actions)) if actions[i] == 3]
            assert len(stop_idxs) == 1
            stop_idx = stop_idxs[0]

            goal_pos = np.array(value["episode"]["goal"]["position"])
            transform_world_start = _SE3(
                quat_from_coeffs(value["episode"]["start_rotation"]),
                np.array(value["episode"]["start_position"]),
            )

            goal_delta = goal_pos - transform_world_start.trans
            goal_delta[1] = 0
            if np.linalg.norm(goal_delta) < 1e-3:
                continue

            mask = np.zeros((96, 96), dtype=np.int64)
            for j in range(stop_idxs[0]):
                x, y = to_grid(positions[j], mask.shape[-1])

                if x <= 0 or x >= mask.shape[0]:
                    continue
                if y <= 0 or y >= mask.shape[1]:
                    continue

                mask[x, y] = 1

            _top_down_occ = value["top_down_occupancy_grid"]
            _top_down_occ = (
                torch.from_numpy(_top_down_occ.copy())
                .float()
                .view(1, 1, *_top_down_occ.shape)
            )
            _top_down_occ = F.avg_pool2d(_top_down_occ, 2).squeeze() * 4
            _top_down_occ = _top_down_occ >= 2

            grid = _top_down_occ.to(dtype=torch.uint8).numpy()
            hidden_state = xs[stop_idx]
            hidden_state = _to_device(hidden_state).view(1, -1)

            pred_grid = occ_model(hidden_state)
            pred_grid = (
                (F.softmax(pred_grid, -1)[..., 1] > 0.5).long().squeeze(0).cpu().numpy()
            )

            gt_grid = grid.copy()
            gt_mask = create_occupancy_grid_mask(mask, napply=5)

            pred_grid[gt_mask == 0] = 0

            attack_type = "line"

            grid_goal = to_grid(transform_world_start.inv() * goal_pos)
            if attack_type == "reflection":
                target_grid, target_mask = make_target_map_reflection(
                    pred_grid, mask, grid_goal
                )
                target_mask = (target_mask == 1) | (gt_mask == 1)
            else:
                target_grid, target_mask = make_target_map(pred_grid, mask, grid_goal)

            gt_grid = _to_device(gt_grid).long()
            gt_mask = _to_device(gt_mask).bool()
            target_grid = _to_device(target_grid).long()
            target_mask = _to_device(target_mask).bool()
            pred_grid = _to_device(pred_grid)

            iou = (
                torch.masked_select(((pred_grid == 1) & (gt_grid == 1)), gt_mask)
                .float()
                .sum()
                / torch.masked_select(((gt_grid == 1) | (pred_grid == 1)), gt_mask)
                .float()
                .sum()
            )

            if iou < 0.33:
                continue

            pgd_states, pgd_logits = pgd_attack(
                occ_model,
                hidden_state.repeat(len(probes) * 5, 1),
                target_grid.repeat(len(probes) * 5, 1, 1),
                target_mask.repeat(len(probes) * 5, 1, 1),
                epsilon=args.epsilon,
            )

            probs = F.softmax(pgd_logits[0:1], -1)
            pgd_grid = (probs[..., 1] > 0.5).long()

            if False:
                save_examples(
                    pred_grid,
                    target_grid,
                    pgd_grid,
                    gt_mask,
                    (target_mask == 1) | (gt_mask == 1),
                    idx,
                )
            to_add = defaultdict(list)

            idx = 0
            for probe in probes:
                goal_delta = goal_pos - transform_world_start.trans
                goal_delta[1] = 0

                goal_heading = quat_from_two_vectors(np.array([0, 0, -1]), goal_delta)
                if attack_type == "reflection":
                    base_angles = []
                    for _ in range(15):
                        rotation, delta_heading = get_heading_of_first_forward(
                            probe,
                            hidden_state.view(6, 1, 512),
                            goal_pos,
                            transform_world_start,
                            transform_world_start.trans,
                            transform_world_start.rot
                            * quat_from_coeffs(rotations[stop_idx]),
                        )
                        look_dir = np.array([0, 0, -1])
                        look_dir = quat_rotate_vector(rotation, look_dir)
                        look_dir = quat_rotate_vector(goal_heading.inverse(), look_dir)
                        angle = np.arctan2(-look_dir[2], look_dir[0])

                        base_angles.append(angle)

                    base_angle = mean_of_angles(base_angles)
                    base_angle = -base_angle

                    if abs(base_angle) < 0.1:
                        to_add = None
                        break

                    state_angles = defaultdict(list)

                for _ in range(5):
                    pgd_hidden = pgd_states[idx : idx + 1]
                    idx += 1
                    control_delta = torch.rand_like(hidden_state) * 2 - 1
                    if True:
                        control_delta = (
                            control_delta
                            * torch.norm(hidden_state - pgd_hidden)
                            / torch.norm(control_delta)
                        )
                    else:
                        n = torch.norm(hidden_state - pgd_hidden, p=math.inf)
                        control_delta = torch.clamp(control_delta, -n, n)
                    control_state = hidden_state + control_delta

                    for name, state in [
                        ("pgd", pgd_hidden),
                        ("control", control_state),
                        ("orig", hidden_state),
                        ("zero", torch.zeros_like(hidden_state)),
                    ]:
                        for _ in range(3):
                            rotation, delta_heading = get_heading_of_first_forward(
                                probe,
                                state.view(6, 1, 512),
                                goal_pos,
                                transform_world_start,
                                transform_world_start.trans,
                                transform_world_start.rot
                                * quat_from_coeffs(rotations[stop_idx]),
                            )

                            if attack_type == "reflection":
                                look_dir = np.array([0, 0, -1])
                                look_dir = quat_rotate_vector(rotation, look_dir)
                                look_dir = quat_rotate_vector(
                                    goal_heading.inverse(), look_dir
                                )
                                angle = np.arctan2(-look_dir[2], look_dir[0])

                                state_angles[name].append(angle)
                            else:
                                angle = angle_between_quats(goal_heading, rotation)
                                avg_heading_errors[name].add(angle)

                if attack_type == "reflection":
                    for name, angles in state_angles.items():
                        angle = mean_of_angles(angles)

                        delta = _angle_distance(base_angle, angle)

                        to_add[name].append(delta)

            if attack_type == "reflection":
                if to_add is not None:
                    for k, vs in to_add.items():
                        for v in vs:
                            avg_heading_errors[k].add(v)

            pbar.set_postfix(**{k: v.mean for k, v in avg_heading_errors.items()})
            pbar.update()

    with open("causality_results.msg", "wb") as f:
        msgpack.pack(
            {k: v.all_vals for k, v in avg_heading_errors.items()}, f, use_bin_type=True
        )


if __name__ == "__main__":
    main()
