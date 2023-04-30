from flask import Flask, render_template, request, jsonify
import lmdb
import msgpack_numpy
import numpy as np
from habitat.tasks.nav.nav_task import _SE3
from habitat_sim.utils.common import quat_from_coeffs
import tqdm

dbase = "/private/home/erikwijmans/projects/navigation-analysis-habitat/data/map_extraction/positions_maps/loopnav-final-mp3d-blind_val.lmdb"
dbase = "/private/home/erikwijmans/projects/navigation-analysis-habitat/data/map_extraction/positions_maps/loopnav-final-mp3d-blind_val.lmdb"
dbase = "/srv/share3/ewijmans3/emergence-of-maps-data/map_extraction/positions_maps/loopnav-final-mp3d-blind_val.lmdb"
#  dbase = "/private/home/erikwijmans/projects/navigation-analysis-habitat/data/map_extraction/positions_maps/devel.lmdb"

with lmdb.open(dbase, map_size=1 << 40,readonly=True) as lmdb_env:
    num_elements = lmdb_env.stat()["entries"]

app = Flask(__name__)

num_bins = 192
bin_size = 0.125


def santize_idx(idx):
    idx = int(idx % num_elements)
    return str(idx).encode()


def to_grid(v):
    return int(v / bin_size + 0.5 + num_bins / 2)


def add_excursions_label(ep):
    excursions = ep["excursions"]

    positions = ep["positions"]
    actions = ep["actions"]
    grid = ep["top_down_occupancy_grid"]

    excursion_label = np.zeros((len(positions)), dtype=np.int16)
    if len(excursions) > 0:
        ptr = 0
        prev_xy = None
        for i, pos in enumerate(positions):
            x, _, y = pos
            x = to_grid(x)
            y = to_grid(y)

            if actions[i] == 3:
                break

            if x <= 0 or x >= grid.shape[0]:
                continue

            if y <= 0 or y >= grid.shape[1]:
                continue

            if not (prev_xy == [x, y]):
                ptr += 1
                prev_xy = [x, y]

            for eidx in range(0, len(excursions) - 1, 2):
                if (ptr >= excursions[eidx]) and (ptr <= excursions[eidx + 1]):
                    excursion_label[i] = eidx // 2 + 1

    ep["excursion_label"] = excursion_label


def build_display_info(idx):
    with lmdb.open(dbase, map_size=1 << 40,) as lmdb_env, lmdb_env.begin(
        buffers=True
    ) as txn:
        ep = msgpack_numpy.unpackb(txn.get(idx), raw=False)

    grid = ep["top_down_occupancy_grid"]
    positions = ep["positions"]
    actions = ep["actions"]
    goal_pos = np.array(ep["episode"]["goal"]["position"])
    transform_world_start = _SE3(
        quat_from_coeffs(ep["episode"]["start_rotation"]),
        np.array(ep["episode"]["start_position"]),
    )
    goal_pos = transform_world_start.inv() * goal_pos
    gx = to_grid(goal_pos[0])
    gy = to_grid(goal_pos[2])

    all_path_1_pts = []
    all_path_2_pts = []

    path1 = []
    path2 = []
    active_path = path1
    active_all_pts = all_path_1_pts
    for i, pos in enumerate(positions):
        x, _, y = pos
        x = to_grid(x)
        y = to_grid(y)

        if actions[i] == 3:
            active_path = path2
            active_all_pts = all_path_2_pts

        if len(active_all_pts) > 0:
            if not np.allclose(active_all_pts[-1], pos, atol=1e-3):
                active_all_pts.append(pos)
        else:
            active_all_pts.append(pos)

        if x <= 0 or x >= grid.shape[0]:
            continue

        if y <= 0 or y >= grid.shape[1]:
            continue

        if len(active_path) > 0:
            prev_xy = active_path[-1]
            if prev_xy == [x, y]:
                continue

        active_path.append([x, y])

    all_path_1_pts = np.array(all_path_1_pts)
    all_path_2_pts = np.array(all_path_2_pts)
    path_difference = [
        float(
            np.linalg.norm(
                all_path_1_pts[:, np.newaxis, :] - all_path_2_pts[np.newaxis, ...],
                axis=-1,
            )
            .min(-1)
            .mean()
        ),
        float(
            np.linalg.norm(
                all_path_2_pts[:, np.newaxis, :] - all_path_1_pts[np.newaxis, ...],
                axis=-1,
            )
            .min(-1)
            .mean()
        ),
    ]

    return dict(
        grid=grid.tolist(),
        path1=path1,
        path2=path2,
        goal=[int(gx), int(gy)],
        excursions=ep.get("excursions", []),
        path_difference=path_difference,
    )


@app.route("/")
def main():
    first_unlabeled = 0
    with lmdb.open(dbase, map_size=1 << 40,readonly=True) as lmdb_env, lmdb_env.begin(
        buffers=True
    ) as txn:
        for first_unlabeled in range(num_elements):
            ep = msgpack_numpy.unpackb(txn.get(santize_idx(first_unlabeled)), raw=False)

            if "excursions" not in ep:
                break

    return render_template("index.html", idx=first_unlabeled)


@app.route("/api/get-task-data", methods=["POST"])
def get_task_data():
    data = request.get_json()
    idx = santize_idx(int(data["idx"]))
    return jsonify(build_display_info(idx))


@app.route("/api/save-result", methods=["POST"])
def save_result():
    data = request.get_json()
    idx = santize_idx(int(data["idx"]))

    with lmdb.open(dbase, map_size=1 << 40,) as lmdb_env, lmdb_env.begin(
        write=True
    ) as txn:
        ep = msgpack_numpy.unpackb(txn.get(idx), raw=False)
        ep["excursions"] = data["excursions"]

        add_excursions_label(ep)

        txn.replace(idx, msgpack_numpy.packb(ep, use_bin_type=True))

    return jsonify(True)


if __name__ == "__main__":
    if False:
        for idx in tqdm.trange(num_elements):
            with lmdb.open(dbase, map_size=1 << 40,) as lmdb_env, lmdb_env.begin(
                write=True
            ) as txn:
                idx = santize_idx(idx)
                ep = msgpack_numpy.unpackb(txn.get(idx), raw=False)

                if "excursions" in ep:
                    add_excursions_label(ep)

                    txn.replace(idx, msgpack_numpy.packb(ep, use_bin_type=True))
