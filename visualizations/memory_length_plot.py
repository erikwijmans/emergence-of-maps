import os.path as osp
import torch
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import pydash
import glob
import numpy as np

import tensorflow as tf
from tensorboard.backend.event_processing import (
    plugin_event_multiplexer as event_multiplexer,
)

SIZE_GUIDANCE = {"scalars": 5000}


sns.set(style="whitegrid", font_scale=1.7)


def extract_scalars(multiplexer, run, tag, vmul=1.0):
    """Extract tabular data from the scalars at a given run and tag.
  The result is a list of 3-tuples (wall_time, step, value).
  """
    try:
        tensor_events = multiplexer.Tensors(run, tag)
    except KeyError as e:
        raise e
        return []
    values = [
        (event.step, tf.make_ndarray(event.tensor_proto).item())
        for event in tensor_events
    ]

    beta = 0.0
    ema = values[0][1]
    ema_values = []
    for s, v in values:
        ema = beta * ema + (1.0 - beta) * v
        ema_values.append([s / 1e6, ema * vmul])

    return ema_values


def create_multiplexer(logdir, run_names):
    multiplexer = event_multiplexer.EventMultiplexer(
        run_path_map={rn: osp.join(logdir, rn) for rn in run_names},
        tensor_size_guidance=SIZE_GUIDANCE,
        max_reload_threads=40,
    )
    multiplexer.Reload()
    return multiplexer


def main():
    all_logs = defaultdict(list)
    for run_dir in glob.glob("runs/memory-length/*"):
        mem_len = int(run_dir.split("mem_")[1].split("-")[0])

        all_logs[mem_len].append(osp.basename(run_dir))

    def _make_key(v, m):
        if m == "spl":
            m = "SPL"
        elif m == "success":
            m = "Success"

        return osp.join(v, "val_" + m)

    all_run_names = []
    for v in pydash.chain().values().flatten()(all_logs):
        all_run_names.append(_make_key(v, "spl"))
        all_run_names.append(_make_key(v, "success"))

    multiplexer = create_multiplexer("runs/memory-length", all_run_names)

    data = dict(x=[], y=[], Metric=[])

    for mem_len in all_logs.keys():
        for rn in all_logs[mem_len]:
            rn_spls = [
                s[1]
                for s in extract_scalars(
                    multiplexer, _make_key(rn, "spl"), "val", 100.0
                )
                if s[0] < 1000
            ]

            best_idx = np.argmax(rn_spls)

            spl = rn_spls[best_idx]

            succ = extract_scalars(multiplexer, _make_key(rn, "success"), "val", 100.0)[
                best_idx
            ][1]

            data["x"].append(mem_len)
            data["x"].append(mem_len)

            data["y"].append(spl)
            data["y"].append(succ)

            data["Metric"].append("SPL")
            data["Metric"].append("Success")

    data = pd.DataFrame.from_dict(data)

    f = plt.figure()
    sns.lineplot(x="x", y="y", hue="Metric", data=data)

    sns.despine(fig=f)

    plt.title("")
    plt.xlabel("Memory Length")
    plt.ylabel("Performance (higher is better)")
    plt.ylim(0, 100)

    plt.savefig("memory_length_vs_perf.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
