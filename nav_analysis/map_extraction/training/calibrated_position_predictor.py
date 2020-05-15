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

    if args.chance_run:
        log_dir = "data/visitation_predictor-chance"
    else:
        log_dir = "data/visitation_predictor"

    device = torch.device("cuda", 0)
    log_dirs = glob.glob(osp.join(log_dir, "time_offset=*"))
    models = dict()

    for log_dir in log_dirs:
        if len(glob.glob(osp.join(log_dir, "*.ckpt"))) == 0:
            continue

        time_offset = int(log_dir.split("=")[-1])

        model = VisitationPredictor.load_from_checkpoint(
            list(
                sorted(
                    glob.glob(osp.join(log_dir, "*.ckpt")),
                    key=osp.getmtime,
                    reverse=True,
                )
            )[0]
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

            if len(ep["excursions"]) == 0:
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

            for i in range(stop_pos):
                step = dict(
                    excursion=excursions[i],
                    excursion_length=(excursions[i] == excursions).sum(),
                    excursion_step=i - np.nonzero(excursions[i] == excursions)[0].min(),
                    position=positions[i],
                    predicted=[None for _ in range(0, 513)],
                )

                for offset, pred in time_offset_preds.items():
                    j = offset + i
                    if j < 0 or j >= stop_pos:
                        continue

                    step["predicted"][offset + 256] = pred[j]

                detailed_results.append(step)

    print(len(detailed_results))

    with open("calibrated_position_predictor_detial.msg", "wb") as f:
        msgpack_numpy.pack(detailed_results, f, use_bin_type=True)


if __name__ == "__main__":
    main()
