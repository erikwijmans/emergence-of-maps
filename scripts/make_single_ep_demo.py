import json
import gzip
import os
import os.path as osp


def _trim(fname):
    with gzip.open(fname, "rt") as f:
        dset = json.load(f)

    eps = dset["episodes"]
    assert len(eps) > 0
    dset["episodes"] = eps[7:8]

    with gzip.open(fname, "wt") as f:
        json.dump(dset, f)

_trim("data/datasets/pointnav/habitat-test-scenes/v1/demo/content/skokloster-castle.json.gz")
