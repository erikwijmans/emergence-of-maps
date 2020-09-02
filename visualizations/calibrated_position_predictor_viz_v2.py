import msgpack_numpy
import pandas as pd
import pydash
import numpy as np
import matplotlib
import numpy as np
import tqdm

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.7)

MAX_OFFSET = 64
MAX_EXCUR_LENGTH = 128

PROB_BASELINES = np.zeros((256,))
PROB_BASELINE_COUNTS = np.zeros((256,))

L2_BASELINES = np.zeros((256,))
L2_BASELINE_COUNTS = np.zeros((256,))


def running_avg_update(avg, count, val):
    avg = (avg * count + val) / (count + 1)
    count += 1

    return avg, count


def cut_excurs_in_half(episode):
    excursion_label = np.array(pydash.map_(episode, "excursion"))
    for excursion_id in np.unique(excursion_label):
        if excursion_id != 0:
            start_step = np.nonzero(excursion_label == excursion_id)[0].min()
            end_step = np.nonzero(excursion_label == excursion_id)[0].max() + 1

            new_end_step = start_step + (end_step - start_step) // 2

            for idx in range(new_end_step, end_step):
                episode[idx]["excursion"] = 0

    return episode


def calc_exur_length(episode):

    excursion_label = np.array(pydash.map_(episode, "excursion"))
    lengths = []
    for excursion_id in np.unique(excursion_label):
        if excursion_id != 0:
            start_step = np.nonzero(excursion_label == excursion_id)[0].min()
            end_step = np.nonzero(excursion_label == excursion_id)[0].max() + 1
            if (end_step - start_step) <= MAX_EXCUR_LENGTH:
                lengths.append(end_step - start_step)

    return lengths


def calc_l2_error(episode, start_step, end_step, base_offset):

    errors = []
    for i in range(start_step, end_step):
        step = episode[i]
        pred_idx = 256 + i - end_step + base_offset
        if pred_idx >= 0 and step["predicted"][pred_idx] is not None:
            errors.append(
                np.linalg.norm(step["predicted"][pred_idx] - step["position"])
                / L2_BASELINES[pred_idx]
            )
        else:
            return None

    if len(errors) == 0:
        return None

    return np.mean(np.array(errors))


def calc_prob_error(episode, start_step, end_step, base_offset):

    prev_pos = np.array([500, 500, 500])
    errors = []
    for i in range(start_step, end_step):
        step = episode[i]
        if np.linalg.norm(step["position"] - prev_pos) < 1e-3:
            continue

        prev_pos = step["position"]

        pred_idx = 256 + i - end_step + base_offset
        if pred_idx >= 0 and step["top_down_prob"][pred_idx] is not None:
            errors.append(
                (1.0 - step["top_down_prob"][pred_idx]) / PROB_BASELINES[pred_idx]
            )
        else:
            return None

    if len(errors) == 0:
        return None

    return np.mean(np.array(errors))


def parse_excursion(episode, excursion_id, excursion_label):
    start_step = np.nonzero(excursion_label == excursion_id)[0].min()
    end_step = np.nonzero(excursion_label == excursion_id)[0].max() + 1

    if (end_step - start_step) > MAX_EXCUR_LENGTH:
        return []

    baseline_l2 = calc_l2_error(episode, start_step, end_step, 0)
    baseline_prob = calc_prob_error(episode, start_step, end_step, 0)
    if baseline_l2 is None or baseline_prob is None:
        return []

    errors = []
    i = end_step
    for offset in reversed(range(-MAX_OFFSET, 0)):
        j = -offset + i
        if j >= len(excursion_label):
            break

        if excursion_label[i] != excursion_label[j] and excursion_label[j] != 0:
            break

        if excursion_label[j] != 0:
            continue

        l2_error = calc_l2_error(episode, start_step, end_step, offset)
        prob_error = calc_prob_error(episode, start_step, end_step, offset)

        if l2_error is None or prob_error is None:
            continue

        errors.append(
            dict(
                t="Excursion",
                x=-offset,
                l2_baseline=baseline_l2,
                l2_error=calc_l2_error(episode, start_step, end_step, offset),
                prob_error=calc_prob_error(episode, start_step, end_step, offset),
                prob_baseline=baseline_prob,
            )
        )

    return errors


def parse_non_excursion(episode, excursion_label, excursion_lengths):
    errors = []
    non_excursion_ranges = []
    start_step = None
    for i in range(len(excursion_label)):
        if start_step is None and excursion_label[i] == 0:
            start_step = i

        if start_step is not None and excursion_label[i] != 0:
            non_excursion_ranges.append((start_step, i))
            start_step = None
            break

    if start_step is not None:
        non_excursion_ranges.append((start_step, len(excursion_label)))

    def select_excur_length():
        for _ in range(100):
            simulated_excur_length = np.random.choice(excursion_lengths)
            valids = [
                i
                for i in range(len(non_excursion_ranges))
                if (non_excursion_ranges[i][1] - non_excursion_ranges[i][0])
                > simulated_excur_length
            ]

            if len(valids) > 0:
                return np.random.choice(valids), simulated_excur_length

        return None, None

    num_types = len(np.unique(excursion_label))
    for _ in range(num_types if num_types == 1 else num_types * 3):
        rng_idx, exur_length = select_excur_length()
        if rng_idx is None:
            continue

        real_start_step, real_end_step = non_excursion_ranges[rng_idx]
        for start_step_offset in range(
            max(real_end_step - real_start_step - exur_length - MAX_OFFSET, 0) + 1
        ):
            start_step = real_start_step + start_step_offset
            end_step = start_step + exur_length

            baseline_l2 = calc_l2_error(episode, start_step, end_step, 0)
            baseline_prob = calc_prob_error(episode, start_step, end_step, 0)
            if baseline_l2 is None or baseline_prob is None:
                continue

            i = end_step
            for offset in reversed(range(-MAX_OFFSET, 0)):
                j = -offset + i
                if j >= len(excursion_label):
                    break

                # Exit on transition
                if excursion_label[j - 1] != 0 and excursion_label[j] == 0:
                    break

                l2_error = calc_l2_error(episode, start_step, end_step, offset)
                prob_error = calc_prob_error(episode, start_step, end_step, offset)

                if l2_error is None or prob_error is None:
                    continue

                #  l2_error = l2_error / baseline_l2
                #  prob_error = prob_error / baseline_prob

                if False:
                    errors.append(
                        dict(
                            t="Non-Excursion (All)",
                            x=-offset,
                            l2_baseline=baseline_l2,
                            l2_error=l2_error,
                            prob_error=prob_error,
                            prob_baseline=baseline_prob,
                        )
                    )

                if num_types > 1:
                    errors.append(
                        dict(
                            t="Non-Excursion (SoftBound)",
                            x=-offset,
                            l2_baseline=baseline_l2,
                            l2_error=l2_error,
                            prob_error=prob_error,
                            prob_baseline=baseline_prob,
                        )
                    )

                if excursion_label[j] == 0 and num_types > 1:
                    errors.append(
                        dict(
                            t="Non-Excursion (HardBound)",
                            x=-offset,
                            l2_baseline=baseline_l2,
                            l2_error=l2_error,
                            prob_error=prob_error,
                            prob_baseline=baseline_prob,
                        )
                    )

                if False and num_types == 1:
                    errors.append(
                        dict(
                            t="Non-Excursion (EpOnly)",
                            x=-offset,
                            l2_baseline=baseline_l2,
                            l2_error=l2_error,
                            prob_error=prob_error,
                            prob_baseline=baseline_prob,
                        )
                    )

    return errors


use_full = True
if use_full:
    with open("calibrated_position_predictor_detial.msg", "rb") as f:
        calibrated_position_predictor_detial = msgpack_numpy.unpack(f, raw=False)

    with open("calibrated_position_predictor_detial_short.msg", "wb") as f:
        msgpack_numpy.pack(
            calibrated_position_predictor_detial[0:20], f, use_bin_type=True
        )
else:
    with open("calibrated_position_predictor_detial_short.msg", "rb") as f:
        calibrated_position_predictor_detial = msgpack_numpy.unpack(f, raw=False)


excursion_lengths = np.array(
    pydash.chain().map(calc_exur_length).flatten()(calibrated_position_predictor_detial)
)

for episode in tqdm.tqdm(calibrated_position_predictor_detial):
    excursion_label = np.array(pydash.map_(episode, "excursion"))
    if len(np.unique(excursion_label)) == 1:
        continue

    prev_label = 0
    for i in range(len(episode)):
        step = episode[i]
        prev_label = excursion_label[i]
        if i != 0 and excursion_label[i] == 0 and excursion_label[i] == 1:
            break

        for offset in reversed(range(-256, 0)):
            j = -offset + i
            if j >= len(excursion_label):
                break

            if (
                excursion_label[i] == 0
                and prev_label != excursion_label[j]
                and excursion_label[j] == 0
            ):
                break

            if (
                excursion_label[i] != 0
                and prev_label != excursion_label[j]
                and excursion_label[j] != 0
            ):
                break

            prev_label = excursion_label[j]

            pred_idx = 256 + offset

            if step["top_down_prob"][pred_idx] is not None:
                (
                    PROB_BASELINES[pred_idx],
                    PROB_BASELINE_COUNTS[pred_idx],
                ) = running_avg_update(
                    PROB_BASELINES[pred_idx],
                    PROB_BASELINE_COUNTS[pred_idx],
                    1.0 - step["top_down_prob"][pred_idx],
                )

            if step["predicted"][pred_idx] is not None:
                (
                    L2_BASELINES[pred_idx],
                    L2_BASELINE_COUNTS[pred_idx],
                ) = running_avg_update(
                    L2_BASELINES[pred_idx],
                    L2_BASELINE_COUNTS[pred_idx],
                    np.linalg.norm(step["predicted"][pred_idx] - step["position"]),
                )


def parse_episode(episode):
    excursion_label = np.array(pydash.map_(episode, "excursion"))
    errors = []

    for excursion_id in np.unique(excursion_label):
        if excursion_id == 1:
            errors += parse_excursion(episode, excursion_id, excursion_label)

    errors += parse_non_excursion(episode, excursion_label, excursion_lengths)

    return errors


data = pydash.chain().map(parse_episode).flatten()(calibrated_position_predictor_detial)


data = pd.DataFrame(data)


fig = plt.figure()
sns.lineplot("x", "prob_baseline", hue="t", data=data)
plt.ylabel("Baseline Error")
plt.xlabel("Steps from Excursion")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


plt.savefig(
    "calibrated_position_predictor_prob_baseline.pdf", format="pdf", bbox_inches="tight"
)

fig = plt.figure()
sns.lineplot("x", "prob_error", hue="t", data=data)
plt.ylabel("Relative Error")
plt.xlabel("Steps from Excursion")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


plt.savefig(
    "calibrated_position_predictor_prob_error.pdf", format="pdf", bbox_inches="tight"
)

fig = plt.figure()
sns.lineplot("x", "l2_baseline", hue="t", data=data)
plt.ylabel("Baseline L2 Error")
plt.xlabel("Steps from Excursion")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


plt.savefig(
    "calibrated_position_predictor_l2_baseline.pdf", format="pdf", bbox_inches="tight"
)

fig = plt.figure()
sns.lineplot("x", "l2_error", hue="t", data=data)
plt.ylabel("Relative Error")
plt.xlabel("Steps from Excursion")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


plt.savefig(
    "calibrated_position_predictor_l2_error.pdf", format="pdf", bbox_inches="tight"
)
