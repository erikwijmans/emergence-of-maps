import os.path as osp
import torch
import glob
import lmdb
import msgpack_numpy
import numpy as np
import tqdm
import argparse
from nav_analysis.map_extraction.training.train_position_predictor import (
    VisitationPredictor,
)
from nav_analysis.map_extraction.training.train_visited_predictor import Model
import torch.nn.functional as F

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def to_grid(pt, num_bins=96):
    bin_size = 0.25 * 96 / num_bins
    num_bins = np.array([num_bins, num_bins])
    bin_range = np.arange(num_bins[0] + 1)
    bin_range = (bin_range - bin_range.max() / 2) * bin_size
    bins = [bin_range.copy(), bin_range.copy()]

    x, _, y = pt

    x = int(np.searchsorted(bins[0], [x])[0])
    y = int(np.searchsorted(bins[1], [y])[0])

    if (0 <= x < num_bins[0]) and (0 <= y < num_bins[1]):
        return x, y
    else:
        return None


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--annotated-data",
        default="data/map_extraction/positions_maps/loopnav-final-mp3d-blind_val.lmdb",
    )

    parser.add_argument(
        "--chance-run", default="False", type=str, choices={"True", "False"}
    )

    return parser


@torch.no_grad()
def main():
    args = build_parser().parse_args()
    args.chance_run = args.chance_run == "True"

    device = torch.device("cuda", 0)

    top_down_model = Model(512 * 6, [96, 96])
    top_down_model.to(device)
    top_down_model.load_state_dict(
        torch.load(
            "data/checkpoints/best_occ_predictor-{}.pt".format("trained"),
            map_location=device,
        )
    )
    top_down_model.eval()

    if args.chance_run:
        log_dir = "data/visitation_predictor-chance"
    else:
        log_dir = "data/visitation_predictor-v2"

    log_dirs = glob.glob(osp.join(log_dir, "time_offset=*"))
    models = dict()

    for log_dir in tqdm.tqdm(log_dirs):
        ckpts = list(
            filter(
                lambda ckpt: "hpc" not in ckpt,
                glob.glob(osp.join(log_dir, "**/*.ckpt"), recursive=True),
            )
        )
        if len(ckpts) == 0:
            continue

        time_offset = int(log_dir.split("=")[-1])

        model = VisitationPredictor.load_from_checkpoint(
            list(sorted(ckpts, key=osp.getmtime, reverse=True,))[0]
        ).to(device=device)
        model.eval()

        models[time_offset] = model

    detailed_results = []
    with lmdb.open(
        args.annotated_data, map_size=1 << 40, readonly=True
    ) as lmdb_env, lmdb_env.begin(buffers=True, write=False) as txn:
        for idx in tqdm.tqdm(range(lmdb_env.stat()["entries"])):
            ep = msgpack_numpy.unpackb(txn.get(str(idx).encode()), raw=False)
            if "excursions" not in ep:
                continue

            actions = ep["actions"]

            stop_pos = -1
            for i in range(len(actions)):
                if actions[i] == 3:
                    stop_pos = i
                    break

            assert stop_pos > 0
            hidden_state = ep["hidden_state"][0:stop_pos]
            positions = ep["positions"][0:stop_pos]
            excursions = ep["excursion_label"][0:stop_pos]

            hidden_state = (
                torch.from_numpy(np.array(hidden_state)).to(device).view(stop_pos, -1)
            )

            time_offset_preds = dict()
            for offset, model in models.items():
                time_offset_preds[offset] = model(hidden_state).cpu().numpy()

            past_map = torch.cat(
                [top_down_model(h)[0] for h in torch.chunk(hidden_state, 10)], 0
            )
            past_map = F.softmax(past_map, -1).cpu().numpy()

            episode_detail = []
            for i in range(stop_pos):
                step = dict(
                    excursion=int(excursions[i]),
                    excursion_length=int((excursions[i] == excursions).sum()),
                    excursion_step=int(
                        i - np.nonzero(excursions[i] == excursions)[0].min()
                    ),
                    position=positions[i],
                    predicted=[None for _ in range(0, 513)],
                    top_down_prob=[None for _ in range(0, 513)],
                )

                for offset, pred in time_offset_preds.items():
                    j = -offset + i
                    if 0 <= j < stop_pos:
                        step["predicted"][offset + 256] = pred[j] + positions[j]

                        grid_pos = to_grid(positions[i])
                        if grid_pos is not None:
                            step["top_down_prob"][offset + 256] = float(
                                past_map[j, grid_pos[0], grid_pos[1], 1]
                            )

                episode_detail.append(step)

            detailed_results.append(episode_detail)

    print(len(detailed_results))

    with open("calibrated_position_predictor_detial.msg", "wb") as f:
        msgpack_numpy.pack(detailed_results, f, use_bin_type=True)


if __name__ == "__main__":
    main()
