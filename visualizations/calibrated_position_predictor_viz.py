import msgpack_numpy
import pandas as pd
import pydash
import numpy as np
import matplotlib
import numpy as np

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.7)


def bin_int(v, bin_size=5):
    v = int(int(v / bin_size) * bin_size)
    return v


def parse_excursion(i, step, positions, excursion_label):
    errors = []
    for offset in reversed(range(-64, 0)):
        j = -offset + i
        if j >= len(excursion_label):
            break

        if excursion_label[i] != excursion_label[j] and excursion_label[j] != 0:
            break

        if np.linalg.norm(positions[i] - positions[j]) < 0.5:
            continue

        if step["predicted"][offset + 256] is not None:
            if step["is_entry"]:
                t = "Excursion"
            elif step["is_exit"]:
                t = "Exit"
            else:
                t = "Excursion"

            errors.append(
                dict(
                    t=t,
                    x=bin_int(-offset),
                    y=np.linalg.norm(step["predicted"][offset + 256] - step["position"])
                    / np.linalg.norm(positions[i] - positions[j]),
                )
            )
    return errors


def parse_excursion_new(i, step, positions, excursion_label):
    errors = []
    for offset in reversed(range(-64, 0)):
        j = -offset + i
        if j >= len(excursion_label):
            break

        if excursion_label[i] != excursion_label[j]:
            break

        if np.linalg.norm(positions[i] - positions[j]) < 0.5:
            continue

        if step["predicted"][offset + 256] is not None:
            if step["is_entry"]:
                t = "Excursion"
            elif step["is_exit"]:
                t = "Exit"
            else:
                t = "Excursion"

            errors.append(
                dict(
                    t=t,
                    x=bin_int(-offset),
                    y=np.linalg.norm(step["predicted"][offset + 256] - step["position"])
                    / np.linalg.norm(positions[i] - positions[j]),
                )
            )

    return errors


def parse_nonexcursion(i, step, positions, excursion_label):
    errors = []
    for offset in reversed(range(-64, 0)):
        j = -offset + i
        if j >= len(excursion_label):
            break

        if excursion_label[j] != 0:
            break

        if np.linalg.norm(positions[i] - positions[j]) < 0.5:
            continue

        if step["predicted"][offset + 256] is not None:
            errors.append(
                dict(
                    t="Non-Excursion",
                    x=bin_int(-offset),
                    y=np.linalg.norm(step["predicted"][offset + 256] - step["position"])
                    / np.linalg.norm(positions[i] - positions[j]),
                )
            )

    return errors


def parse_episode(episode):
    excursion_label = np.array(pydash.map_(episode, "excursion"))
    positions = np.array(pydash.map_(episode, "position"))
    errors = []

    for i in range(len(episode)):
        step = episode[i]
        if excursion_label[i] == 0:
            errors += parse_nonexcursion(i, step, positions, excursion_label)
        else:
            errors += parse_excursion(i, step, positions, excursion_label)

    return errors


def erode_excursions(episode):
    erosion_amount = 0.2
    for i in range(len(episode)):
        episode[i]["is_entry"] = False
        episode[i]["is_exit"] = False

    excursion_label = np.array(pydash.map_(episode, "excursion"))

    for excursion_id in np.unique(excursion_label):
        if excursion_id == 0:
            continue

        start_step = np.nonzero(excursion_label == excursion_id)[0].min()
        end_step = np.nonzero(excursion_label == excursion_id)[0].max() + 1
        excur_len = end_step - start_step

        for i in range(start_step, end_step):
            episode[i]["is_entry"] = (i - start_step) < (erosion_amount / 2 * excur_len)
            episode[i]["is_exit"] = (end_step - i) < (erosion_amount / 2 * excur_len)

    return episode


with open("calibrated_position_predictor_detial.msg", "rb") as f:
    calibrated_position_predictor_detial = msgpack_numpy.unpack(f, raw=False)

#  calibrated_position_predictor_detial = calibrated_position_predictor_detial[0:500]


data = (
    pydash.chain()
    .map(erode_excursions)
    .map(parse_episode)
    .flatten()(calibrated_position_predictor_detial)
)

data = pd.DataFrame(data)


fig = plt.figure()
sns.lineplot("x", "y", hue="t", data=data)

plt.ylabel("Relative L2 Error")
plt.xlabel("Steps in the Past")
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles[1:],
    labels[1:],
    bbox_to_anchor=(1.05, 0.5),
    loc="center left",
    borderaxespad=0.0,
    frameon=False,
    fancybox=False,
    shadow=False,
)
sns.despine(fig=fig)

plt.savefig("calibrated_position_predictor.pdf", format="pdf", bbox_inches="tight")
