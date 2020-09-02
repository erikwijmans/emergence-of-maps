import msgpack_numpy
import pydash
import numpy as np
import tqdm
import imageio
import numba


@numba.jit(nopython=True, parallel=True)
def scale_up_color(img, scaling: int = 4):
    h, w, c = img.shape
    new_img = np.zeros((h * scaling, w * scaling, c), dtype=img.dtype)
    for j in range(h * scaling):
        for i in range(w * scaling):
            new_img[j, i] = img[j // scaling, i // scaling]

    return new_img


MAX_EXCUR_LENGTH = 64

num_bins = 192 // 2
bin_size = 0.125 * 2


def to_grid(*args):
    return tuple(int(v / bin_size + 0.5 + num_bins / 2) for v in args)


use_full = False
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


#  calibrated_position_predictor_detial = calibrated_position_predictor_detial[4:6]


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
            errors.append(1.0 - step["top_down_prob"][pred_idx])
        else:
            return None

    if len(errors) == 0:
        return None

    return np.mean(np.array(errors))


pbar = tqdm.tqdm(total=len(calibrated_position_predictor_detial))


def parse_episode(episode):
    positions = np.array(pydash.map_(episode, "position"))
    excursion_label = np.array(pydash.map_(episode, "excursion"))
    if np.unique(excursion_label).shape[0] == 1:
        pbar.update()
        return []

    error_gradients = []
    for i in tqdm.trange(32, len(positions), leave=False):
        for exur_length in range(16, MAX_EXCUR_LENGTH + 1):
            end_step = i + exur_length
            if end_step >= len(positions):
                break

            baseline_prob_error = calc_prob_error(episode, i, end_step, 0)
            if baseline_prob_error is None:
                continue

            for offset in reversed(range(-8, 0)):
                prev_error = calc_prob_error(episode, i, end_step, offset)
                if prev_error is None:
                    continue

                max_grad = 0.0
                best_offset = 1
                for forgetting_offset in reversed(range(-1, 0)):
                    prob_error = calc_prob_error(
                        episode, i, end_step, offset + forgetting_offset
                    )
                    if prob_error is None:
                        continue

                    if prob_error - prev_error > max_grad:
                        max_grad = prob_error - prev_error
                        best_offset = forgetting_offset

                if best_offset == 1:
                    continue

                error_gradients.append(
                    dict(
                        grad=max_grad / baseline_prob_error,
                        start_step=i,
                        end_step=end_step,
                        offset=offset,
                        forgetting_offset=best_offset,
                    )
                )

    pbar.update()
    return error_gradients


def make_prob_image(episode, error_event):
    positions = np.array(pydash.map_(episode, "position"))
    start_step = error_event["start_step"]
    end_step = error_event["end_step"]
    offset = error_event["offset"]

    map1 = np.ones((num_bins, num_bins, 3))
    for i in range(len(positions)):
        x, y = to_grid(positions[i][0], positions[i][2])
        if 0 <= x < num_bins and 0 <= y < num_bins:
            map1[x, y] = np.array([41, 128, 185]) / 255.0 * 0.5 + 0.5

    for i in range(0, start_step):
        x, y = to_grid(positions[i][0], positions[i][2])
        if 0 <= x < num_bins and 0 <= y < num_bins:
            map1[x, y] = np.array([41, 128, 185]) / 255.0

    map2 = map1.copy()
    for i in range(start_step, end_step):
        step = episode[i]
        x, y = to_grid(positions[i][0], positions[i][2])

        pred_idx1 = 256 + i - end_step + offset
        pred_idx2 = 256 + i - end_step + (offset + error_event["forgetting_offset"])
        if (
            pred_idx2 >= 0
            and step["top_down_prob"][pred_idx1] is not None
            and step["top_down_prob"][pred_idx2] is not None
        ):
            p1 = step["top_down_prob"][pred_idx1]
            p2 = step["top_down_prob"][pred_idx2]

            map1[x, y] = p1 * np.array([44, 62, 80]) / 255.0 + (1 - p1) * np.ones((3,))
            map2[x, y] = p2 * np.array([44, 62, 80]) / 255.0 + (1 - p2) * np.ones((3,))

    i = end_step - offset
    x, y = to_grid(positions[i][0], positions[i][2])
    map1[x, y] = 0

    i = end_step - offset - error_event["forgetting_offset"]
    x, y = to_grid(positions[i][0], positions[i][2])
    map2[x, y] = 0

    full_deal = np.concatenate([map1, np.ones((num_bins, 10, 3)), map2], 1)

    full_deal = (full_deal * 255.0).astype(np.uint8)
    full_deal = scale_up_color(full_deal)

    return full_deal


error_grads_all = pydash.map_(calibrated_position_predictor_detial, parse_episode)
pbar.close()
grads = np.array(pydash.chain().flatten().map("grad")(error_grads_all))

print(grads.mean())
print(grads.std())

cutoff = np.quantile(grads, [0.99])


image_idx = 0
for i in tqdm.trange(len(calibrated_position_predictor_detial)):
    episode = calibrated_position_predictor_detial[i]
    error_grads = error_grads_all[i]

    for err in error_grads:
        if err["grad"] > cutoff:
            print(err["grad"])
            image = make_prob_image(episode, err)

            imageio.imsave(
                "fast_forgetting_images/{:0>4d}.png".format(image_idx), image
            )

            image_idx += 1
