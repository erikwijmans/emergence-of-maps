import argparse
import os.path as osp

import lmdb
import msgpack
import msgpack_numpy
import h5py as h5
import numpy as np
import tqdm

num_bins = np.array([96, 96])
bin_size = 0.25
bin_range = np.arange(num_bins[0] + 1)
bin_range = (bin_range - bin_range.max() / 2) * bin_size
bins = [bin_range.copy(), bin_range.copy()]


msgpack_numpy.patch()


def _build_dset(_env, T=5000):
    _xs = []
    _maps = []
    occupancy_grids = []
    samples_per = 30
    _len = _env.stat()["entries"]
    print(_len)
    episode_lens = []
    with _env.begin(buffers=True) as txn, tqdm.tqdm(total=_len) as pbar:
        num_skipped = 0
        total = 0
        for i in range(_len):
            v = txn.get(str(i).encode())
            value = msgpack.unpackb(v, raw=False)

            xs = value["hidden_state"][0:T]
            positions = value["positions"][0:T]

            episode_lens.append(len(xs))

            _map = np.zeros(num_bins, dtype=np.uint8)

            total += len(positions)
            _new_xs = []
            _new_maps = []
            for i, pos in enumerate(positions):
                x, y = pos

                x = np.searchsorted(bins[0], [x])[0]
                if x <= 0 or x >= _map.shape[0]:
                    num_skipped += 1
                    continue

                y = np.searchsorted(bins[1], [y])[0]
                if y <= 0 or y >= _map.shape[1]:
                    num_skipped += 1
                    continue

                _map[x, y] = 1

                _new_xs.append(xs[i])
                _new_maps.append(_map.copy())

            if _map.sum() > 4 and len(_new_xs) > samples_per:
                take_ids = np.linspace(
                    len(_new_xs) // samples_per, len(_new_xs) - 1, num=samples_per
                ).astype(np.int64)
                _xs.append(np.stack(_new_xs, 0)[take_ids])
                _maps.append(np.stack(_new_maps, 0)[take_ids])
                occupancy_grids.append(value["top_down_occupancy_grid"])

            pbar.set_postfix(num_skipped=num_skipped / total, dset_len=len(_xs))
            pbar.update()

    assert len(_xs) == len(_maps)

    print(np.min(episode_lens))
    print(np.mean(episode_lens))
    print(np.max(episode_lens))

    return np.stack(_xs, 0), np.stack(_maps, 0), np.stack(occupancy_grids, 0)


for split in ["train", "val"]:
    fname = f"data/map_extraction/positions_maps/loopnav-static-pg-v4_{split}"
    with lmdb.open(fname + ".lmdb") as _env:
        xs, maps, occupancy_grids = _build_dset(_env)

    with h5.File(fname + "_dset.h5", "w") as f:
        f.attrs.create("len", len(xs))
        f.attrs.create("samples_per", xs.shape[1])
        f.attrs.create("maps_shape", maps.shape[2:])

        f.create_dataset("xs", data=xs)
        f.create_dataset("maps", data=maps)
        f.create_dataset("occupancy_grids", data=occupancy_grids)
