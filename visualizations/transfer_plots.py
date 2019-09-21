import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import argparse
import os.path as osp
import re
import functools
import pprint
from pydash import py_
import numpy as np
from collections import defaultdict

import tensorflow as tf
from tensorboard.backend.event_processing import (
    plugin_event_multiplexer as event_multiplexer,
)


run_name_parser = re.compile(
    r"""
        gibson-public
        -(?P<task>.*?)
        -(?P<type>.*?)
        -rgb
        -r(?P<r>\d)
        $
    """,
    re.VERBOSE,
)

sns.set(style="whitegrid", font_scale=1.2)


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--logdir", type=str, default="runs")

    return parser


SIZE_GUIDANCE = {"scalars": 500}


def extract_scalars(multiplexer, run, tag):
    """Extract tabular data from the scalars at a given run and tag.
  The result is a list of 3-tuples (wall_time, step, value).
  """
    try:
        tensor_events = multiplexer.Tensors(run, tag)
    except KeyError as e:
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
        ema_values.append([s / 1e6, ema])

    #  return [(np.log(s), v) for s, v in ema_values]
    if len(ema_values) == 39:
        ema_values = ema_values[1:]
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
    pretty_type_names = {
        "scratch": r"Scratch",
        "imagenet-xfer": r"ImageNet",
        "pointnav-xfer": r"Visual Encoder",
        "pointnav-ftune": r"Fine-tune",
        "driver": r"Neural Controller",
        "controller": r"$\nabla$ Neural Controller",
        "controller-512": r"$\nabla$ Neural Controller",
    }

    args = build_parser().parse_args()

    type_order = [
        "scratch",
        "imagenet-xfer",
        "pointnav-xfer",
        "pointnav-ftune",
        #  "driver",
        #  "controller",
        "controller-512",
    ]

    all_logs = defaultdict(lambda: defaultdict(list))
    for run_dir in glob.glob(args.logdir + "/*"):
        run_name = osp.basename(run_dir)
        match = run_name_parser.match(run_name)
        if match is None:
            continue

        type_ = match.group("type")
        task = match.group("task")

        if type_ not in type_order:
            continue

        all_logs[task][type_].append(run_name)

    def _make_key(v, dset, task):
        return osp.join(
            v,
            "metrics" if dset == "train" else "val",
            "flee_dist" if task == "flee" else "visited",
        )

    all_run_names = []
    for task, results in all_logs.items():
        for v in py_().values().flatten()(results):
            all_run_names.append(_make_key(v, "train", task))
            all_run_names.append(_make_key(v, "val", task))

    multiplexer = create_multiplexer(args.logdir, all_run_names)

    data = {"metric": [], "iter": [], "": [], "dset": [], "task": []}
    for task, results in all_logs.items():
        for type_ in type_order:
            if type_ not in results:
                continue
            runs = results[type_]
            print(task, type_)

            val_curves = [
                extract_scalars(multiplexer, _make_key(rn, "val", task), "val")
                for rn in runs
                if len(extract_scalars(multiplexer, _make_key(rn, "val", task), "val"))
                > 0
            ]
            min_len = min(map(len, val_curves))
            print(min_len)
            val_curves = np.array([v[0:min_len] for v in val_curves])

            for i in range(val_curves.shape[1]):
                for j in range(val_curves.shape[0]):
                    data["metric"].append(val_curves[j, i, 1])
                    data["iter"].append(val_curves[:, i, 0].mean())
                    data[""].append(pretty_type_names[type_])
                    data["dset"].append("val")
                    data["task"].append(
                        r"{}".format("Flee" if task == "flee" else r"Exploration")
                    )

            continue

            train_curves = np.array(
                [
                    extract_scalars(
                        multiplexer, _make_key(rn, "train", task), "metrics"
                    )
                    for rn in runs
                    if len(
                        extract_scalars(
                            multiplexer, _make_key(rn, "train", task), "metrics"
                        )
                    )
                    > 0
                ]
            )
            for i in range(train_curves.shape[1]):
                for j in range(train_curves.shape[0]):
                    data["metric"].append(train_curves[j, i, 1])
                    data["iter"].append(train_curves[:, i, 0].mean())
                    data["Method"].append(pretty_type_names[type_])
                    data["dset"].append("train")

    palette = {
        k: v
        for k, v in zip(
            [pretty_type_names[name] for name in type_order], sns.color_palette()
        )
    }

    data = pd.DataFrame.from_dict(data)
    #  f = plt.figure()
    #  sns.lineplot(
    #  x="iter",
    #  y="metric",
    #  hue="Method",
    #  #  style="dset",
    #  data=data,
    #  legend="full",
    #  palette=palette,
    #  )
    g = (
        sns.relplot(
            x="iter",
            y="metric",
            hue="",
            #  style="dset",
            data=data,
            legend="full",
            palette=palette,
            col="task",
            kind="line",
            height=4,
            aspect=1.4,
            facet_kws=dict(sharey=False),
        )
        .despine()
        .set_titles("{col_name}")
        .set_axis_labels("Steps (experience) in millions", "")
        .savefig("data/transfer.pdf".format(task), format="pdf", bbox_inches="tight")
    )


if __name__ == "__main__":
    main()
