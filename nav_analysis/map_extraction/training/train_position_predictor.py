import random
import math
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pytorch_lightning as pl
import glob
from pytorch_lightning.logging import TensorBoardLogger
import lmdb
import msgpack_numpy
import tqdm
import numpy as np
import argparse
import shutil

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "0")


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

    parser.add_argument("--mode", default="train", type=str, choices={"train", "eval"})
    parser.add_argument(
        "--chance-run", default="False", type=str, choices={"True", "False"}
    )

    return parser


def did_stop(pos_a, rot_a, pos_b, rot_b):
    return np.allclose(pos_a, pos_b) and np.allclose(rot_a, rot_b)


def copy_to_scratch_jail(lmdb_filename):
    jail = osp.join("/scratch/slurm_tmpdir", SLURM_JOB_ID)

    if osp.exists(jail):
        jail_name = osp.join(jail, osp.abspath(lmdb_filename)[1:])

        if not osp.exists(jail_name):
            shutil.copytree(lmdb_filename, jail_name)

        lmdb_filename = jail_name

    return lmdb_filename


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
        for episode_key, v in tqdm.tqdm(curr, total=lmdb_env.stat()["entries"]):
            ele = msgpack_numpy.unpackb(v, raw=False)
            hidden_state = ele["hidden_state"]
            positions = ele["positions"]
            actions = ele["actions"]

            stop_pos = -1
            for i in range(len(positions)):
                if actions[i] == 3:
                    stop_pos = i
                    break

            assert stop_pos > 0
            hidden_state = hidden_state[0:stop_pos]
            positions = positions[0:stop_pos]

            episode_ranges.append(
                (
                    (np.arange(len(positions)) + len(all_hidden_states)).tolist(),
                    [(episode_key, i) for i in range(len(positions))],
                    [-(stop_pos - i) for i in range(stop_pos)]
                    + [(i - stop_pos) for i in range(stop_pos, len(positions))],
                )
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
                msgpack_numpy.packb(hidden_state.astype(np.float16), use_bin_type=True),
            )

        all_positions = np.array(all_positions, dtype=np.float32)

        for time_offset in tqdm.trange(-time_range_size, time_range_size + 1):
            if time_offset == 0:
                continue

            time_range_dset = []
            for ep, ep_keys, steps_from_stop in episode_ranges:
                nvalid = len(ep) - abs(time_offset)
                if nvalid <= 0:
                    continue

                states = np.arange(ep[0], ep[-1]).astype(np.int32)
                positions = all_positions[ep[0] : ep[-1]]

                if time_offset < 0:
                    states = states[-time_offset:]
                    positions = positions[0:time_offset] - positions[-time_offset:]

                    steps_from_stop_curr = steps_from_stop[-time_offset:]
                    steps_from_stop_tgt = steps_from_stop[0:time_offset]
                else:
                    states = states[0:-time_offset]
                    positions = positions[time_offset:] - positions[0:-time_offset]

                    steps_from_stop_curr = steps_from_stop[0:-time_offset]
                    steps_from_stop_tgt = steps_from_stop[time_offset:]

                for i in range(len(states)):
                    time_range_dset.append(
                        dict(
                            position=positions[i],
                            hidden_state_idx=states[i],
                            steps_from_stop_curr=steps_from_stop_curr[i],
                            steps_from_stop_tgt=steps_from_stop_tgt[i],
                        )
                    )

            txn.put(
                f"time_offset/{time_offset}".encode(),
                msgpack_numpy.packb(time_range_dset, use_bin_type=True),
            )

    return cache_name


def _block_shuffle(lst, block_size=1, shuffle=True):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    if shuffle:
        random.shuffle(blocks)

    return blocks


class VisititationPredictionDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, lmdb_filename, time_offset, chance_run=False, shuffle=True, truncate=-1
    ):
        super().__init__()
        self.lmdb_filename = lmdb_filename
        self.preload_size = int(256 * 40)
        self._preload = []
        self._current_idx = 0
        self._chance_run = chance_run
        self._truncate = truncate

        with lmdb.open(
            lmdb_filename, map_size=1 << 40, lock=False, readonly=True
        ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
            self.time_range_dset = msgpack_numpy.unpackb(
                txn.get(f"time_offset/{time_offset}".encode()), raw=False
            )

        self._num_hidden_states = (
            max(ele["hidden_state_idx"] for ele in self.time_range_dset) + 1
        )

        self._all_load_blocks = _block_shuffle(
            list(range(len(self.time_range_dset))),
            block_size=self.preload_size,
            shuffle=shuffle,
        )
        self._shuffle = shuffle

    def __next__(self):
        if len(self._preload) == 0:
            if len(self._load_blocks) == 0:
                raise StopIteration

            with lmdb.open(
                self.lmdb_filename, map_size=1 << 40, readonly=True, lock=False
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                block = self._load_blocks.pop()
                for idx in block:
                    ele = self.time_range_dset[idx].copy()
                    if self._chance_run:
                        hidden_idx = random.randint(0, self._num_hidden_states - 1)
                    else:
                        hidden_idx = ele["hidden_state_idx"]

                    ele["hidden_state"] = (
                        msgpack_numpy.unpackb(
                            txn.get(f"hidden_state/{hidden_idx}".encode()), raw=False,
                        )
                        .astype(np.float32)
                        .flatten()
                    )

                    self._preload.append(ele)

            if self._shuffle:
                random.shuffle(self._preload)

            self._preload = list(reversed(self._preload))

        return self._preload.pop()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self._load_blocks = list(self._all_load_blocks)
        else:
            per_worker = int(
                math.ceil(len(self._all_load_blocks) / float(worker_info.num_workers))
            )

            start = per_worker * worker_info.id
            end = min(start + per_worker, len(self._all_load_blocks))

            self._load_blocks = list(self._all_load_blocks[start:end])

        if self._shuffle:
            random.shuffle(self._load_blocks)

        if self._truncate != -1:
            rank = 0 if worker_info is None else worker_info.id
            world_size = 1 if worker_info is None else worker_info.num_workers

            truncate = self._truncate / world_size
            if rank == world_size - 1:
                truncate = int(math.ceil(truncate))
            else:
                truncate = int(math.floor(truncate))

            self._load_blocks = self._load_blocks[0:truncate]

        return self


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
            ele["hidden_state"] = (
                msgpack_numpy.unpackb(
                    txn.get("hidden_state/{}".format(ele["hidden_state_idx"]).encode()),
                    raw=False,
                )
                .astype(np.float32)
                .flatten()
            )

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

        loss = F.smooth_l1_loss(preds, batch["position"], reduction="none").mean(dim=-1)
        l2_error = torch.norm(preds.detach() - batch["position"], p=2, dim=-1)

        gt_norms = torch.norm(batch["position"], p=2, dim=-1)
        norm_l2_error = torch.full_like(l2_error, -1e5)
        norm_mask = gt_norms > 1e-1
        norm_l2_error[norm_mask] = l2_error[norm_mask] / gt_norms[norm_mask]

        return dict(
            val_loss=loss, val_l2_error=l2_error, val_norm_l2_error=norm_l2_error
        )

    def validation_end(self, outputs):
        res = dict()
        for k in outputs[0].keys():
            res[k] = torch.cat([x[k] for x in outputs]).mean()

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
                getattr(self.hparams, f"{split}_dataset"),
                self.hparams.time_offset,
                self.hparams.chance_run,
                truncate=25,
                shuffle=True,
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
    args.chance_run = args.chance_run == "True"

    args.val_dataset = build_visitation_dataset_cache(args.val_dataset, 256)
    args.train_dataset = build_visitation_dataset_cache(args.train_dataset, 256)

    args.val_dataset = copy_to_scratch_jail(args.val_dataset)
    args.train_dataset = copy_to_scratch_jail(args.train_dataset)

    if args.chance_run:
        log_dir = "data/visitation_predictor-chance/time_offset={}".format(
            args.time_offset
        )
    else:
        log_dir = "data/visitation_predictor/time_offset={}".format(args.time_offset)
    if args.mode == "train":
        model = VisitationPredictor(args)
        logger = TensorBoardLogger(log_dir, name="", version=0,)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=log_dir,
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
            prefix="",
        )
        early_stopping_cb = pl.callbacks.EarlyStopping(
            mode="min", patience=3, verbose=True
        )

        trainer = pl.Trainer(
            max_epochs=300,
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stopping_cb,
            val_check_interval=500,
            gpus=[0],
        )

        trainer.fit(model)
    else:
        device = torch.device("cuda", 0)
        log_dirs = glob.glob(osp.join(osp.dirname(log_dir), "time_offset=*"))

        eval_results = dict(
            time_offset=[],
            val_loss=[],
            val_l2_error=[],
            val_norm_l2_error=[],
            val_loss_sem=[],
            val_l2_error_sem=[],
            val_norm_l2_error_sem=[],
        )

        for log_dir in tqdm.tqdm(log_dirs):
            if len(glob.glob(osp.join(log_dir, "*.ckpt"))) == 0:
                continue

            time_offset = int(log_dir.split("=")[-1])

            model = VisitationPredictor.load_from_checkpoint(
                list(
                    sorted(
                        glob.glob(osp.join(log_dir, "*.ckpt")),
                        key=osp.getmtime,
                        reverse=True,
                    )
                )[0]
            ).to(device=device)
            model.eval()
            model.hparams.val_dataset = args.val_dataset
            model.hparams.train_dataset = args.train_dataset

            with torch.no_grad():
                val_outputs = []
                for dl in model.val_dataloader():
                    for batch in dl:
                        batch["hidden_state"] = batch["hidden_state"].to(device=device)
                        batch["position"] = batch["position"].to(device=device)
                        val_outputs.append(model.validation_step(batch, None))

                        val_outputs[-1]["steps_from_stop_tgt"] = batch[
                            "steps_from_stop_tgt"
                        ]
                        val_outputs[-1]["steps_from_stop_curr"] = batch[
                            "steps_from_stop_curr"
                        ]

                loss = torch.cat([o["val_loss"] for o in val_outputs], dim=0)
                l2_error = torch.cat([o["val_l2_error"] for o in val_outputs], dim=0)
                norm_l2_error = torch.cat(
                    [o["val_norm_l2_error"] for o in val_outputs], dim=0
                )
                norm_l2_error = norm_l2_error[norm_l2_error > 0.0]

                eval_results["val_loss"].append(float(loss.mean()))
                eval_results["val_l2_error"].append(float(l2_error.mean()))
                eval_results["val_norm_l2_error"].append(float(norm_l2_error.mean()))

                eval_results["val_loss_sem"].append(
                    2.0 * float(loss.std() / np.sqrt(loss.numel()))
                )
                eval_results["val_l2_error_sem"].append(
                    2.0 * float(l2_error.std() / np.sqrt(l2_error.numel()))
                )
                eval_results["val_norm_l2_error_sem"].append(
                    2.0 * float(norm_l2_error.std() / np.sqrt(norm_l2_error.numel()))
                )

                eval_results["time_offset"].append(time_offset)

        if args.chance_run:
            output_name = "position_predictor_eval_results_chance.msg"
        else:
            output_name = "position_predictor_eval_results.msg"

        with open(output_name, "wb") as f:
            msgpack_numpy.pack(eval_results, f, use_bin_type=True)


if __name__ == "__main__":
    main()
