import random
import math
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
import lmdb
import msgpack_numpy
import tqdm
import numpy as np
import argparse

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument(
        "--train-dataset",
        default="data/map_extraction/positions_maps/loopnav-static-pg-v5_train.lmdb",
    )
    parser.add_argument(
        "--val-dataset",
        default="data/map_extraction/positions_maps/loopnav-static-pg-v5_val.lmdb",
    )
    parser.add_argument("--time-offset", type=int, default=-1)

    return parser


def build_visitation_dataset_cache(lmdb_filename, time_range_size):
    cache_name = osp.join(osp.splitext(lmdb_filename)[0], "time_offset_cache.lmdb")

    if osp.exists(cache_name):
        return cache_name

    with lmdb.open(
        lmdb_filename, map_size=1 << 40, readonly=True, lock=False
    ) as lmdb_env, lmdb_env.begin(buffers=True) as txn, txn.cursor() as curr:

        all_hidden_states = []
        all_positions = []
        episode_ranges = []
        for _, v in tqdm.tqdm(curr, total=lmdb_env.stat()["entries"]):
            ele = msgpack_numpy.unpackb(v, raw=False)
            hidden_state = ele["hidden_state"]
            positions = ele["positions"]
            episode_ranges.append(
                (np.arange(len(positions)) + len(all_hidden_states)).tolist()
            )

            all_hidden_states += hidden_state
            all_positions += positions

    os.makedirs(cache_name, exist_ok=True)

    with lmdb.open(cache_name, map_size=1 << 40) as lmdb_env, lmdb_env.begin(
        write=True
    ) as txn:
        for i, hidden_state in tqdm.tqdm(
            enumerate(all_hidden_states), total=len(all_hidden_states)
        ):
            txn.put(
                f"hidden_state/{i}".encode(),
                msgpack_numpy.packb(hidden_state.astype(np.float32), use_bin_type=True),
            )

        all_positions = np.array(all_positions, dtype=np.float32)

        for time_offset in tqdm.trange(-time_range_size, time_range_size + 1):
            if time_offset == 0:
                continue

            time_range_dset = []
            for ep in episode_ranges:
                nvalid = len(ep) - abs(time_offset)
                if nvalid <= 0:
                    continue

                states = np.arange(ep[0], ep[-1]).astype(np.int32)
                positions = all_positions[ep[0] : ep[-1]]

                if time_offset < 0:
                    states = states[-time_offset:]
                    positions = positions[0:time_offset] - positions[-time_offset:]
                else:
                    states = states[0:-time_offset]
                    positions = positions[time_offset:] - positions[0:-time_offset]

                for i in range(len(states)):
                    time_range_dset.append(
                        dict(position=positions[i], hidden_state_idx=states[i],)
                    )

            txn.put(
                f"time_offset/{time_offset}".encode(),
                msgpack_numpy.packb(time_range_dset, use_bin_type=True),
            )

    return cache_name


class VisititationPredictionDataset(torch.utils.data.IterableDataset):
    def __init__(self, lmdb_filename, time_offset):
        super().__init__()
        self.lmdb_filename = lmdb_filename
        self.preload_size = 10000
        self._preload = []
        self._current_idx = 0

        with lmdb.open(
            lmdb_filename, map_size=1 << 40, lock=False, readonly=True
        ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
            self.time_range_dset = msgpack_numpy.unpackb(
                txn.get(f"time_offset/{time_offset}".encode()), raw=False
            )

    def __next__(self):
        if len(self._preload) == 0:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                start = 0
                end = len(self.time_range_dset)
            else:
                per_worker = int(
                    math.ceil(
                        len(self.time_range_dset) / float(worker_info.num_workers)
                    )
                )

                start = per_worker * worker_info.id
                end = min(start + per_worker, len(self.time_range_dset))

            if (self._current_idx + start) == end:
                raise StopIteration

            with lmdb.open(
                self.lmdb_filename, map_size=1 << 40, readonly=True, lock=False
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for _ in range(self.preload_size):
                    if (self._current_idx + start) == end:
                        break
                    ele = self.time_range_dset[self._current_idx + start].copy()
                    ele["hidden_state"] = msgpack_numpy.unpackb(
                        txn.get(
                            "hidden_state/{}".format(ele["hidden_state_idx"]).encode()
                        ),
                        raw=False,
                    ).flatten()

                    self._preload.append(ele)

                    self._current_idx += 1

            random.shuffle(self._preload)

        return self._preload.pop()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.time_range_dset)


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_filename, time_offset):
        super().__init__()
        self.lmdb_filename = lmdb_filename
        self._lmdb_env = None

        with lmdb.open(
            lmdb_filename, map_size=1 << 40, lock=False, readonly=True
        ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
            self.time_range_dset = msgpack_numpy.unpackb(
                txn.get(f"time_offset/{time_offset}".encode()), raw=False
            )

    def __getitem__(self, idx):
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self.lmdb_filename, map_size=1 << 40, lock=False, readonly=True
            )

        ele = self.time_range_dset[idx].copy()
        with self._lmdb_env.begin(buffers=True) as txn:
            ele["hidden_state"] = msgpack_numpy.unpackb(
                txn.get("hidden_state/{}".format(ele["hidden_state_idx"]).encode()),
                raw=False,
            ).flatten()

        return ele

    def __len__(self):
        return len(self.time_range_dset)


class VisitationPredictor(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.model = self.__build_model()

    def __build_model(self):
        return nn.Sequential(
            nn.Linear(512 * 6, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        preds = self.forward(batch["hidden_state"])

        loss = F.smooth_l1_loss(preds, batch["position"])
        l2_error = torch.norm(preds.detach() - batch["position"], p=2, dim=-1).mean()

        return dict(
            loss=loss, l2_error=l2_error, log=dict(loss=loss, l2_error=l2_error)
        )

    def validation_step(self, batch, batch_idx):
        preds = self.forward(batch["hidden_state"])

        loss = F.smooth_l1_loss(preds, batch["position"])
        l2_error = torch.norm(preds.detach() - batch["position"], p=2, dim=-1).mean()

        return dict(val_loss=loss, val_l2_error=l2_error)

    def validation_end(self, outputs):
        res = dict()
        for k in outputs[0].keys():
            res[k] = torch.stack([x[k] for x in outputs]).mean()

        res["log"] = res.copy()

        return res

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def __build_dataloader(self, split):
        if split == "train":
            dset = VisititationPredictionDataset(
                getattr(self.hparams, f"{split}_dataset"), self.hparams.time_offset
            )
        if split == "val":
            dset = ListDataset(
                getattr(self.hparams, f"{split}_dataset"), self.hparams.time_offset
            )

        dloader = torch.utils.data.DataLoader(
            dset,
            num_workers=4,
            batch_size=self.hparams.batch_size,
            drop_last=split == "train",
        )

        return dloader

    @pl.data_loader
    def train_dataloader(self):
        return self.__build_dataloader("train")

    @pl.data_loader
    def val_dataloader(self):
        return self.__build_dataloader("val")


def main():
    args = build_parser().parse_args()

    args.val_dataset = build_visitation_dataset_cache(args.val_dataset, 256)
    args.train_dataset = build_visitation_dataset_cache(args.train_dataset, 256)

    model = VisitationPredictor(args)

    logger = TensorBoardLogger(
        "data/visitation_predictor/time_offset={}".format(args.time_offset),
        name="",
        version=0,
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath="data/visitation_predictor/time_offset={}".format(args.time_offset),
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )

    trainer = pl.Trainer(
        max_nb_epochs=300,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        val_check_interval=1000,
        gpus=[0],
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
