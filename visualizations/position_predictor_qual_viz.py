import msgpack_numpy
import pydash
import numpy as np
import matplotlib
import cv2
import os.path as osp
import os
import pydash

GRID_SIZE = 1024
CELL_SIZE = 0.125 / 2


def to_grid(pt):
    x, _, y = pt

    x = int(x / CELL_SIZE + GRID_SIZE / 2)
    y = int(y / CELL_SIZE + GRID_SIZE / 2)

    return x, y


def draw_path(episode):
    positions = np.array(pydash.map_(episode, "position"))
    if len(episode) < 50:
        return

    errors = []
    for i in range(len(episode) - 1):
        step = episode[i]

        all_past_preds = []
        all_future_positions = []
        for offset in range(-80, -40):
            j = i - offset

            if step["predicted"][offset + 256] is None:
                continue

            if np.linalg.norm(positions[i] - positions[j]) < 0.25:
                continue

            all_past_preds.append(step["predicted"][offset + 256])
            all_future_positions.append(positions[j])

        if len(all_past_preds) == 0:
            break

        all_past_preds = np.array(all_past_preds)
        all_future_positions = np.array(all_future_positions)

        error = min(
            np.mean(
                np.linalg.norm(all_past_preds - step["position"][None, :], axis=-1)
                / np.linalg.norm(positions[i][None, :] - all_future_positions, axis=-1)
            ),
            1.0,
        )

        errors.append(error)

    errors = np.array(errors)
    print(errors)

    path_map = np.ones((GRID_SIZE, GRID_SIZE, 3))
    for i in range(len(errors) - 1):

        pt1 = to_grid(positions[i])
        pt2 = to_grid(positions[i + 1])

        base_color = np.array([26, 188, 156][::-1]) / 255.0

        err = (errors[i] + errors[i + 1]) / 2
        color = base_color * (1 - err) + err

        cv2.line(path_map, pt1, pt2, color, 10, cv2.LINE_AA)

        if i == 0:
            pt0 = pt1

    for i in range(len(errors) - 1):

        pt1 = to_grid(positions[i])
        pt2 = to_grid(positions[i + 1])

        base_color = np.array([52, 152, 219][::-1]) / 255.0

        cv2.line(path_map, pt1, pt2, base_color, 2, cv2.LINE_AA)

    x, y = pt0
    path_map[x - 5 : x + 5, y - 5 : y + 5] = np.array([39, 174, 96][::-1]) / 255.0

    cv2.imshow("path", path_map)
    cv2.waitKey(0)


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


for episode in calibrated_position_predictor_detial:
    path = draw_path(episode)
