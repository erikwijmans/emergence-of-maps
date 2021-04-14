import json
import gzip
import os
import os.path as osp


def _unbundle(fname):
    with gzip.open(fname, "rt") as f:
        dset = json.load(f)

    eps = dset["episodes"]
    assert len(eps) > 0
    scenes = list(set(ep["scene_id"] for ep in eps))

    os.makedirs(osp.join(osp.dirname(fname), "content"), exist_ok=True)
    for scene in scenes:
        scene_eps = []
        for ep in eps:
            if ep["scene_id"] == scene:
                scene_eps.append(ep)

        with gzip.open(
            osp.join(
                osp.dirname(fname),
                "content",
                osp.splitext(osp.basename(scene))[0] + ".json.gz",
            ),
            "wt",
        ) as f:
            json.dump(dict(episodes=scene_eps), f)

    with gzip.open(fname, "wt") as f:
        json.dump(dict(episodes=[]), f)


_unbundle("data/datasets/pointnav/habitat-test-scenes/v1/test/test.json.gz")
_unbundle("data/datasets/pointnav/habitat-test-scenes/v1/train/train.json.gz")
_unbundle("data/datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz")
