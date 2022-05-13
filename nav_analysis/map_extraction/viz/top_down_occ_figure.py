import tqdm
import msgpack_numpy
import lmdb
import torch.nn.functional as F
import os
import torch
import numpy as np
import statsmodels.api as sm
import imageio
import cv2
from scipy.stats import ks_2samp, wilcoxon
import habitat_sim

import matplotlib

matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["backend"] = "Agg"

import seaborn as sns
import matplotlib.pyplot as plt

from nav_analysis.map_extraction.viz.make_visited_videos import (
    colorize_map,
    scale_up_color,
    _scale_up_binary,
)

from nav_analysis.map_extraction.training.train_visited_predictor import (
    Model,
    VisitedDataset,
    mapping_acc,
    create_occupancy_grid_mask,
)


from habitat.tasks.nav.nav_task import SE3, quat_from_coeffs


def make_groundtruth(ep_data, resolution):
    transform_world_start = SE3(
        quat_from_coeffs(ep_data["origin_rotation"]),
        np.array(ep_data["origin_position"]),
    )
    cell_scale = 0.25 * 96 / resolution

    top_down_map = np.zeros(
        (int(resolution * 1.25 + 1), int(resolution * 1.25 + 1)), dtype=np.uint8
    )
    h2 = top_down_map.shape[0] // 2
    w2 = top_down_map.shape[1] // 2
    jitters = [
        np.array([0, 0, 0.5]),
        np.array([0.5, 0, 0]),
        np.array([0.5, 0, 0.5]),
        np.array([0, 0, 0]),
    ]

    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(ep_data["scene_id"].replace("glb", "navmesh"))
    assert pf.is_loaded, ep_data["scene_id"]
    for x in range(top_down_map.shape[0]):
        for y in range(top_down_map.shape[1]):
            pt_start = np.array([x - h2, 0, y - w2], dtype=np.float32)
            valid_point = any(
                pf.is_navigable(transform_world_start * ((pt_start + jit) * cell_scale))
                for jit in jitters
            )
            top_down_map[x, y] = 1 if valid_point else 0

    h2 = top_down_map.shape[0] // 2 - 1
    w2 = top_down_map.shape[1] // 2 - 1
    crop = int(resolution / 2)
    top_down_map = top_down_map[h2 - crop : h2 + crop, w2 - crop : w2 + crop]

    return np.ascontiguousarray(top_down_map.copy())


def to_grid(pt, num_bins=96):
    bin_size = 0.25 * 96 / num_bins
    num_bins = np.array([num_bins, num_bins])
    bin_range = np.arange(num_bins[0] + 1)
    bin_range = (bin_range - bin_range.max() / 2) * bin_size
    bins = [bin_range.copy(), bin_range.copy()]

    x, _, y = pt

    x = int(np.searchsorted(bins[0], [x])[0])
    y = int(np.searchsorted(bins[1], [y])[0])

    return x, y


sns.set(style="white", font_scale=1.7)


def custom_kde_plot(data, label, color):
    kde = sm.nonparametric.KDEUnivariate(data)
    kde.fit(kernel="gau", cut=3, clip=(0.0, 1.0), bw="scott")
    x, y = kde.support, kde.density
    y = y / np.sum(y)

    ax = plt.gca()

    ax.plot(x, y, color=color, label=label)

    shade_kws = dict(facecolor=color, alpha=0.25, clip_on=True, zorder=1)
    ax.fill_between(x, 0, y, **shade_kws)

    ax.set_ylim(0, auto=None)

    ax.axvline(np.mean(data), color=color, linestyle="--")
    #  plt.text(
    #  np.mean(data) - 0.025,
    #  1.0,
    #  "{:.1f}%".format(np.round(np.mean(data) * 100, 1)),
    #  transform=ax.get_xaxis_transform(),
    #  )


@torch.no_grad()
def accuracy_and_examples(state_type):
    device = torch.device("cuda")
    input_size = 512 * 6
    model = Model(input_size, [96, 96])
    model.to(device)
    model.eval()

    trained_ckpt = torch.load(
        "data/checkpoints/best_occ_predictor-{}.pt".format(state_type),
        #  "data/best_vision_occ_predictor-{}.pt".format(state_type),
        map_location=device,
    )
    model.load_state_dict(trained_ckpt)

    env = lmdb.open(
        "data/map_extraction/positions_maps/loopnav-final-mp3d-blind-with-random-states_val.lmdb",
        #  "data/map_extraction/positions_maps/depth-model_val.lmdb",
        map_size=1 << 40,
    )
    length = env.stat()["entries"]

    top_down_preds_and_accs = []
    all_probs = []
    all_grids = []
    all_masks = []

    with env.begin(buffers=True) as txn:
        for idx in tqdm.trange(length):
            v = txn.get(str(idx).encode())
            value = msgpack_numpy.unpackb(v, raw=False)

            xs = value["hidden_state"]
            rxs = value["random_hidden_state"]
            actions = value["actions"]
            positions = value["positions"]

            stop_idxs = [i for i in range(len(actions)) if actions[i] == 3]
            assert len(stop_idxs) in (0, 1)
            if len(stop_idxs) == 0:
                stop_idxs = [len(actions)]

            mask = np.zeros((96, 96), dtype=np.int64)
            agent_route = []
            for j in range(stop_idxs[0]):
                agent_route.append(positions[j])
                x, y = to_grid(positions[j], mask.shape[-1])

                if x <= 0 or x >= mask.shape[0]:
                    continue
                if y <= 0 or y >= mask.shape[1]:
                    continue

                mask[x, y] = 1

            mask = create_occupancy_grid_mask(mask, napply=5)
            mask = torch.from_numpy(mask).to(device=device) == 1

            _top_down_occ = value["top_down_occupancy_grid"]
            _top_down_occ = (
                torch.from_numpy(_top_down_occ.copy())
                .float()
                .view(1, 1, *_top_down_occ.shape)
                .to(device=device)
            )
            _top_down_occ = F.avg_pool2d(_top_down_occ, 2).squeeze() * 4
            _top_down_occ = _top_down_occ >= 2
            grid = _top_down_occ.to(device=device, dtype=torch.uint8)

            if state_type == "random":
                x = torch.from_numpy(rxs[stop_idxs[0] - 1].copy())
            else:
                x = torch.from_numpy(xs[stop_idxs[0] - 1].copy())

            x = x.to(device=device, dtype=torch.float32).view(1, -1)

            past_logits, future_logits, occupancy_logits, _, _ = model(x)

            occupancy_logits[..., 1].masked_fill_((mask == 0).unsqueeze(0), -1e10)

            probs = F.softmax(occupancy_logits, -1).squeeze(0)
            all_probs.append(probs)
            all_grids.append(grid)
            all_masks.append(mask)

            _map = (probs[..., 1] > 0.5).long()

            iou = (
                torch.masked_select(((_map == 1) & (grid == 1)), mask).float().sum()
                / torch.masked_select(((_map == 1) | (grid == 1)), mask).float().sum()
            )
            acc = float(iou)
            bal_acc = (
                float(
                    torch.masked_select((_map == grid) & (grid == 1), mask)
                    .float()
                    .sum()
                    / torch.masked_select(grid == 1, mask).float().sum()
                    + torch.masked_select((_map == grid) & (grid == 0), mask)
                    .float()
                    .sum()
                    / torch.masked_select(grid == 0, mask).float().sum()
                )
                / 2.0
            )

            errors = torch.full_like(_map, 2)
            errors.masked_fill_((_map != grid) & (grid == 0), 0)
            errors.masked_fill_((_map != grid) & (grid == 1), 1)

            top_down_preds_and_accs.append(
                [
                    _map.cpu().numpy(),
                    float(acc),
                    agent_route,
                    value["episode"],
                    mask.cpu().numpy(),
                    errors.cpu().numpy(),
                    bal_acc,
                ]
            )

    return (
        [v[1] for v in top_down_preds_and_accs],
        [v[0] for v in top_down_preds_and_accs],
        [v[2] for v in top_down_preds_and_accs],
        [v[3] for v in top_down_preds_and_accs],
        [v[4] for v in top_down_preds_and_accs],
        [v[5] for v in top_down_preds_and_accs],
        [v[6] for v in top_down_preds_and_accs],
    )


def make_examples(
    trained_accs, trained_bal_accs, trained_preds, agent_routes, episodes, masks, errors
):
    scaling_factor = 6

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

        return _map

    prng = np.random.RandomState(0)

    acc_ranges = [[0.0, 0.1], [0.115, 0.13], [0.31, 0.33], [0.55, 0.65]]

    num = 0
    for acc_rg in acc_ranges:
        lb, ub = acc_rg
        inds = [i for i in range(len(trained_accs)) if lb <= trained_accs[i] <= ub]
        ind = prng.choice(inds)

        print(trained_accs[ind])

        color_map = [[236, 240, 241], [149, 165, 166], [255, 255, 255]]

        pred = trained_preds[ind].copy()
        gt = make_groundtruth(episodes[ind], pred.shape[0] * scaling_factor)
        gt_full = gt.copy()
        mask = masks[ind].copy()

        pred[mask == 0] = 2

        gt[_scale_up_binary(mask, scaling_factor) == 0] = 2

        pred = scale_up_color(colorize_map(pred, color_map), scaling_factor)
        gt = colorize_map(gt, color_map)
        gt_full = colorize_map(gt_full, color_map)

        err = errors[ind]
        err[mask == 0] = 3
        color_map = [
            [231, 76, 60],
            np.array([192, 57, 43]) * 0.85,
            [39, 174, 96],
            [255, 255, 255],
        ]
        err = scale_up_color(colorize_map(err, color_map), scaling_factor)

        gt = draw_path(agent_routes[ind], gt)
        gt_full = draw_path(agent_routes[ind], gt_full)
        pred = draw_path(agent_routes[ind], pred)

        mask = _scale_up_binary(mask, scaling_factor)

        range_x = np.nonzero(np.any(mask, 1))[0]
        range_y = np.nonzero(np.any(mask, 0))[0]

        x_min, x_max = range_x[0], range_x[-1]
        y_min, y_max = range_y[0], range_y[-1]

        pred = pred[x_min:x_max, y_min:y_max]
        gt = gt[x_min:x_max, y_min:y_max]
        err = err[x_min:x_max, y_min:y_max]

        imageio.imwrite("occ_figure/pred{}.png".format(num), pred)
        imageio.imwrite("occ_figure/gt{}.png".format(num), gt)
        imageio.imwrite("occ_figure/gt_full{}.png".format(num), gt_full)
        imageio.imwrite("occ_figure/err{}.png".format(num), err)
        num += 1

    trained_accs = np.array(trained_accs)
    trained_bal_accs = np.array(trained_bal_accs)

    deltas = np.sort(trained_bal_accs - trained_accs)
    cut_off = np.sort(deltas)[int(0.9 * len(deltas))]
    print("Acc delta cuttoff", cut_off)
    taken_inds = set()
    for num in range(5):
        while True:
            ind = int(prng.choice(len(deltas)))

            if ind in taken_inds:
                continue

            if deltas[ind] >= cut_off and trained_bal_accs[ind] > 0.67:
                break

        taken_inds.add(ind)

        print(deltas[ind], trained_bal_accs[ind])

        color_map = [[236, 240, 241], [149, 165, 166], [255, 255, 255]]

        pred = trained_preds[ind].copy()
        mask = masks[ind].copy()
        gt = make_groundtruth(episodes[ind], pred.shape[0] * scaling_factor)

        pred[mask == 0] = 2
        gt[_scale_up_binary(mask, scaling_factor) == 0] = 2

        pred = scale_up_color(colorize_map(pred, color_map), scaling_factor)
        gt = colorize_map(gt, color_map)

        err = errors[ind]
        err[mask == 0] = 3
        color_map = [
            [231, 76, 60],
            np.array([192, 57, 43]) * 0.85,
            [39, 174, 96],
            [255, 255, 255],
        ]
        err = scale_up_color(colorize_map(err, color_map), scaling_factor)

        pred = draw_path(agent_routes[ind], pred)
        gt = draw_path(agent_routes[ind], gt)

        mask = _scale_up_binary(mask, scaling_factor)

        range_x = np.nonzero(np.any(mask, 1))[0]
        range_y = np.nonzero(np.any(mask, 0))[0]

        x_min, x_max = range_x[0], range_x[-1]
        y_min, y_max = range_y[0], range_y[-1]

        pred = pred[x_min:x_max, y_min:y_max]
        gt = gt[x_min:x_max, y_min:y_max]

        imageio.imwrite("occ_figure/big-diff-{}.png".format(num), pred)
        imageio.imwrite("occ_figure/big-diff-gt-{}.png".format(num), gt)


def main():
    (
        trained_accs,
        trained_preds,
        agent_routes,
        episodes,
        masks,
        errors,
        trained_bal_accs,
    ) = accuracy_and_examples("trained")

    print("Trained IOU", np.mean(trained_accs))
    print("Trained Bal Accs", np.mean(trained_bal_accs))

    random_accs, *_, random_bal_acs = accuracy_and_examples("random")
    print("Random IOU", np.mean(random_accs))
    print("Random Bal Accs", np.mean(random_bal_acs))

    print(ks_2samp(trained_accs, random_accs))
    res = wilcoxon(trained_accs, random_accs)
    print(res)
    print(res.pvalue)

    os.makedirs("occ_figure", exist_ok=True)

    make_examples(
        trained_accs,
        trained_bal_accs,
        trained_preds,
        agent_routes,
        episodes,
        masks,
        errors,
    )

    fig = plt.figure(figsize=(10, 5))

    #  custom_kde_plot(random_accs, label="\\texttt{RandomEmbedding}", color="#c0392b")
    #  custom_kde_plot(trained_accs, label="\\texttt{TrainedEmbedding}", color="#f39c12")
    custom_kde_plot(random_accs, label="UntrainedAgentMemory", color="#c0392b")
    custom_kde_plot(trained_accs, label="TrainedAgentMemory", color="#f39c12")

    ax = plt.gca()
    ax.legend(loc="best")
    plt.xlabel("Map Prediction Accuracy (IoU)")

    ax.get_yaxis().set_visible(False)

    sns.despine(fig=fig)

    plt.savefig("occ_figure/top_down_acc_plot.pdf", format="pdf", bbox_inches="tight")

    fig = plt.figure(figsize=(10, 5))
    custom_kde_plot(random_bal_acs, label="UntrainedAgentMemory", color="#c0392b")
    custom_kde_plot(trained_bal_accs, label="TrainedAgentMemory", color="#f39c12")

    ax = plt.gca()
    ax.legend(loc="best")
    plt.xlabel("Map Prediction Accuracy (Class Balanced Accuracy)")

    ax.get_yaxis().set_visible(False)

    sns.despine(fig=fig)

    plt.savefig(
        "occ_figure/top_down_acc_plot_bal_acc.pdf", format="pdf", bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
