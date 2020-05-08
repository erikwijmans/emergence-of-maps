import matplotlib

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
import itertools

import tensorflow as tf
from tensorboard.backend.event_processing import (
    plugin_event_multiplexer as event_multiplexer,
)

matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["backend"] = "pgf"


run_name_parser = re.compile(
    r"""
    mp3d-gibson-all
    -(?P<task>.*?)
    -stage-2
    -(?P<state>.*?)-state
    -final
    -run_(?P<r>\d)_(?P<s>\d)
    -(?P<input_type>.*?)
    $
    """,
    re.VERBOSE,
)

sns.set(style="whitegrid", font_scale=1.3)


def mean_ci(vals, sigfigs=1):
    vals = np.array(vals)
    sem = 1.96 * np.std(vals) / np.sqrt(len(vals))
    return np.round(np.mean(vals), sigfigs), np.round(sem, sigfigs + 1)


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--logdir", type=str, default="runs")

    return parser


def make_row(chamfers, success, spls):
    row = ""
    for i, metric in enumerate([chamfers, success, spls]):
        if i != 0:
            m, std = mean_ci(metric)
            row += " & "
            row += r"{:.1f}\scriptsize{{$\pm${:.2f}}}".format(m, std)
        else:
            m, std = mean_ci(metric, 2)
            row += r"{:.2f}\scriptsize{{$\pm${:.3f}}}".format(m, std)

    return row


SIZE_GUIDANCE = {"scalars": 500}


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
    args = build_parser().parse_args()

    all_logs = defaultdict(list)
    for run_dir in glob.glob(args.logdir + "/*"):
        run_name = osp.basename(run_dir)
        match = run_name_parser.match(run_name)
        if match is None:
            continue

        state = match.group("state")
        task = match.group("task")
        input_type = match.group("input_type")

        all_logs[(task, state, input_type)].append(run_name)

    def _make_key(v, m):
        if m == "spl":
            m = "stage-2 SPL"
        elif m == "success":
            m = "stage-2 Success"

        return osp.join(v, "val_" + m)

    all_run_names = []
    for v in py_().values().flatten()(all_logs):
        all_run_names.append(_make_key(v, "spl"))
        all_run_names.append(_make_key(v, "success"))
        all_run_names.append(_make_key(v, "loop_compare_chamfer"))

    multiplexer = create_multiplexer(args.logdir, all_run_names)

    task_types = ["teleportnav", "loopnav"]
    state_types = ["zero", "random", "trained"]
    input_types = ["blind", "no-inputs"]

    for input_type, state, task in itertools.product(
        input_types, state_types, task_types
    ):
        spls = []
        successes = []
        chamfers = []
        for rn in all_logs[(task, state, input_type)]:
            rn_spls = [
                s[1]
                for s in extract_scalars(
                    multiplexer, _make_key(rn, "spl"), "val", 100.0
                )
            ]

            best_idx = np.argmax(rn_spls)

            spls.append(rn_spls[best_idx])

            successes.append(
                extract_scalars(multiplexer, _make_key(rn, "success"), "val", 100.0)[
                    best_idx
                ][1]
            )

            chamfers.append(
                extract_scalars(
                    multiplexer, _make_key(rn, "loop_compare_chamfer"), "val", 1.0
                )[best_idx][1]
            )

        print(input_type, state, task)
        print(make_row(chamfers, successes, spls))


if __name__ == "__main__":
    main()
