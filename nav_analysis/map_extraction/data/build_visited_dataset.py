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
    returns = dict(
        xs=[], maps=[], occupancy_grids=[], d_goal=[], d_start=[]
    )
    samples_per = 30
    _len = _env.stat()["entries"]
    episode_lens = []

    with _env.begin(buffers=True) as txn, tqdm.tqdm(total=_len) as pbar:
        num_skipped = 0
        total = 0
        for i in range(_len):
            v = txn.get(str(i).encode())
            value = msgpack.unpackb(v, raw=False)

            positions = value["positions"][0:T]

            xs = np.stack(value["hidden_state"][0:T], 0)
            d_goal = np.stack(value["d_goal"][0:T], 0)
            d_start = np.stack(value["d_start"][0:T], 0)

            episode_lens.append(len(xs))

            _map = np.zeros(num_bins, dtype=np.uint8)

            total += len(positions)
            _new_maps = []
            valid_ids = []
            for i, pos in enumerate(positions):
                x, _, y = pos

                x = np.searchsorted(bins[0], [x])[0]
                if x <= 0 or x >= _map.shape[0]:
                    num_skipped += 1
                    continue

                y = np.searchsorted(bins[1], [y])[0]
                if y <= 0 or y >= _map.shape[1]:
                    num_skipped += 1
                    continue

                _map[x, y] = 1

                valid_ids.append(i)
                _new_maps.append(_map.copy())

            xs = xs[valid_ids]
            d_goal = d_goal[valid_ids]
            d_start = d_start[valid_ids]

            if _map.sum() > 4 and len(valid_ids) > samples_per:
                take_ids = np.linspace(
                    len(valid_ids) // samples_per, len(valid_ids) - 1, num=samples_per
                ).astype(np.int64)
                returns["occupancy_grids"].append(value["top_down_occupancy_grid"])

                returns["xs"].append(xs[take_ids])
                returns["maps"].append(np.stack(_new_maps, 0)[take_ids])
                returns["d_goal"].append(d_goal[take_ids])
                returns["d_start"].append(d_start[take_ids])

            pbar.set_postfix(
                num_skipped=num_skipped / total, dset_len=len(returns["xs"])
            )
            pbar.update()

    print(np.min(episode_lens))
    print(np.mean(episode_lens))
    print(np.max(episode_lens))

    for k, v in returns.items():
        print(k)
        returns[k] = np.stack(v, 0)
        print(k, returns[k].shape)

    return returns


for split in ["train", "val"][::-1]:
    fname = f"data/map_extraction/positions_maps/loopnav-with-grad_{split}"
    with lmdb.open(fname + ".lmdb") as _env:
        returns = _build_dset(_env)

    with h5.File(fname + "_dset.h5", "w") as f:
        xs = returns["xs"]
        maps = returns["maps"]
        f.attrs.create("len", len(xs))
        f.attrs.create("samples_per", xs.shape[1])
        f.attrs.create("maps_shape", maps.shape[2:])

        for k, v in returns.items():
            print(k, v.shape)
            f.create_dataset(k, data=v)
