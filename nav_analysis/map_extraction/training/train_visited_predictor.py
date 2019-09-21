import contextlib
import argparse
import glob
import os.path as osp

import numba
import h5py as h5
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as distrib
import torch.nn.functional as F
import torch.utils.data
import tqdm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from torch.utils import tensorboard
import imageio

from scipy import ndimage


from nav_analysis.utils.ddp_utils import init_distrib_slurm

from apex import amp

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


num_bins = None


def create_occupancy_grid_mask(visited_map, rank=2, napply=1):
    mask = visited_map != 0.0
    struct = ndimage.generate_binary_structure(len(mask.shape), rank)
    for _ in range(napply):
        mask = ndimage.binary_dilation(mask, structure=struct).astype(np.uint8)

    return mask


class VisitedDataset(torch.utils.data.Dataset):
    bins = None

    def __init__(self, split, T=100):
        super().__init__()
        global num_bins
        self.split = split
        self.T = T
        self.fname = (
            f"data/map_extraction/positions_maps/loopnav-static-pg-v4_{split}_dset.h5"
        )

        self._f = h5.File(self.fname, "r")
        self._len = self._f.attrs["len"]
        self._samples_per = self._f.attrs["samples_per"]
        num_bins = self._f.attrs["maps_shape"][()]

        self._f = None

    def __getitem__(self, idx):
        if self._f is None:
            self._f = h5.File(self.fname, "r")
        _map = self._f["maps"][idx // self._samples_per][
            idx % self._samples_per
        ].astype(np.int64)
        xs = np.reshape(
            self._f["xs"][idx // self._samples_per][idx % self._samples_per], -1
        )
        grid = self._f["occupancy_grids"][idx // self._samples_per].astype(np.int64)

        mask = create_occupancy_grid_mask(_map, napply=2)

        return xs, _map, grid, mask

    def __len__(self):
        return self._len * self._samples_per


def _fuse_coords(x, coords=None):
    device = x.device
    _, _, h, w = x.size()
    if coords is None:
        xs, ys = torch.meshgrid(
            1 - 2 * torch.arange(0, h, device=device, dtype=x.dtype) / h,
            2 * torch.arange(0, w, device=device, dtype=x.dtype) / w,
        )

        xs = xs.view(1, 1, h, w)
        ys = ys.view(1, 1, h, w)

        coords = torch.cat([xs, ys], 1)

    x = torch.cat([x, coords.expand(x.size(0), 2, h, w)], 1)

    return x, coords


class CoordDeconv(nn.Module):
    def __init__(self, inp, out, stride):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            inp + 2,
            out,
            stride=stride,
            kernel_size=3,
            dilation=1,
            padding=0,
            bias=False,
        )

        self._coords = None

    def forward(self, x):
        x, self._coords = _fuse_coords(x, self._coords)
        x = self.conv(x)
        return x


class CoordConv(nn.Module):
    def __init__(self, inp, out, kernel_size=3, bias=True):
        super().__init__()

        self.conv = nn.Conv2d(
            inp + 2, out, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias
        )
        self._coords = None

    def forward(self, x):
        x, self._coords = _fuse_coords(x, self._coords)
        x = self.conv(x)
        return x


class Model(nn.Module):
    def __init__(self, input_size, num_bins, linear=False):
        super().__init__()

        def _make_layer(inp, out, p=0.0):
            return nn.Sequential(
                nn.Dropout(p=p),
                nn.Linear(inp, out, bias=False),
                nn.BatchNorm1d(out),
                nn.ReLU(True),
            )

        def _make_coord_deconv(inp, out, p=0.0):
            return nn.Sequential(
                nn.Dropout2d(p=p),
                CoordDeconv(inp, out, 2),
                nn.BatchNorm2d(out),
                nn.ReLU(True),
            )

        def _make_coord_conv(inp, out, p=0.0):
            return nn.Sequential(
                nn.Dropout2d(p=p),
                CoordConv(inp, out, 2, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(True),
            )

        self.num_bins = num_bins
        hidden_size = 256
        div = 16
        self.hidden_spatial_size = np.array(
            [64, num_bins[0] // div, num_bins[1] // div]
        )
        self.pose_preder = nn.Sequential(
            _make_layer(input_size, hidden_size),
            _make_layer(hidden_size, np.prod(self.hidden_spatial_size)),
        )

        self.deconvs = nn.Sequential(
            _make_coord_conv(64, 64),
            _make_coord_deconv(64, 64),
            _make_coord_conv(64, 64),
            _make_coord_deconv(64, 128),
            _make_coord_conv(128, 128),
            _make_coord_deconv(128, 128),
            _make_coord_conv(128, 128),
            _make_coord_deconv(128, 128),
            #  _make_coord_conv(128, 256),
            #  _make_coord_deconv(256, 256),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, 4, 1),
        )

    def forward(self, x):
        pose_preds = self.pose_preder(x)
        pose_preds = pose_preds.view(-1, *self.hidden_spatial_size)

        pose_preds = self.deconvs(pose_preds)
        res = pose_preds[:, :, 0 : self.num_bins[0], 0 : self.num_bins[1]].permute(
            0, 2, 3, 1
        )
        return res[..., 0:2].contiguous(), res[..., 2:].contiguous()


class FocalLoss:
    def __init__(self, alpha=0.75, gamma=2.0):
        self.weights = torch.tensor([alpha, 1.0 - alpha])
        self.gamma = torch.tensor(gamma)

    def __call__(self, logits, y, mask=None):
        logits = logits.float()

        logits = logits.view(-1, 2)
        y = y.view(-1)
        if mask is not None:
            mask = mask.view(-1)
            inds = mask.nonzero().squeeze(-1)

            logits = torch.index_select(logits, 0, inds)
            y = torch.index_select(y, 0, inds)

        device = logits.device
        self.weights = self.weights.to(device)
        self.gamma = self.gamma.to(device)

        if self.gamma > 0.0:
            loss = F.cross_entropy(logits, y, weight=self.weights, reduction="none")

            with torch.no_grad():
                probs = F.softmax(logits.detach(), -1)
                focal_weights = (
                    torch.gather(1 - probs, dim=-1, index=y.view(-1, 1))
                    .pow(self.gamma)
                    .view(-1)
                )
            return focal_weights * loss
        else:
            return F.cross_entropy(logits, y, weight=weights)


focal_loss = FocalLoss()


@torch.no_grad()
def mapping_acc(logits, y, mask=None):
    preds = torch.argmax(logits, -1)
    corrects = preds == y

    gt_false = y == 0
    gt_true = y == 1

    if mask is not None:
        preds = torch.masked_select(preds, mask)
        corrects = torch.masked_select(corrects, mask)

        gt_false = torch.masked_select(gt_false, mask)
        gt_true = torch.masked_select(gt_true, mask)

    return [
        (corrects & gt_false).float().sum(),
        gt_false.float().sum(),
        (corrects & gt_true).float().sum(),
        gt_true.float().sum(),
    ]


class EpochMetrics:
    def __init__(self, device):
        self.device = device
        self.total_visited_acc = []
        self.total_unbal_visited_acc = 0.0

        self.total_occupancy_acc = []
        self.total_unbal_occupancy_acc = 0.0

        self.unbalanced_acc = 0.0
        self.occupancy_count = torch.tensor(0.0, device=device)
        self.visited_count = torch.tensor(0.0, device=device)

        self.total_visited_loss = 0.0
        self.total_occupancy_loss = 0.0

    @torch.no_grad()
    def update(
        self, batch, visited_logits, occupancy_logits, visited_loss, occupancy_loss
    ):
        x, y, grid, mask = batch

        self.visited_count += y.numel()
        self.occupancy_count += mask.float().sum()

        self.total_visited_loss += visited_loss.sum()
        self.total_occupancy_loss += occupancy_loss.sum()

        self.total_unbal_occupancy_acc += (
            torch.masked_select((torch.argmax(occupancy_logits, -1) == grid), mask)
            .float()
            .sum()
        )
        self.total_unbal_visited_acc += (
            (torch.argmax(visited_logits, -1) == y).float().sum()
        )

        self.total_occupancy_acc.append(mapping_acc(occupancy_logits, grid, mask))
        self.total_visited_acc.append(mapping_acc(visited_logits, y))

    def finalize(self):
        self.total_occupancy_acc = torch.tensor(
            self.total_occupancy_acc, device=self.device
        ).sum(0)
        self.total_visited_acc = torch.tensor(
            self.total_visited_acc, device=self.device
        ).sum(0)

        distrib.all_reduce(self.total_visited_acc)
        distrib.all_reduce(self.total_unbal_visited_acc)
        distrib.all_reduce(self.total_visited_loss)
        distrib.all_reduce(self.visited_count)

        distrib.all_reduce(self.total_occupancy_acc)
        distrib.all_reduce(self.total_unbal_occupancy_acc)
        distrib.all_reduce(self.total_occupancy_loss)
        distrib.all_reduce(self.occupancy_count)

        self.total_visited_acc = (
            self.total_visited_acc[0] / self.total_visited_acc[1]
            + self.total_visited_acc[2] / self.total_visited_acc[3]
        ) / 2.0
        self.total_unbal_visited_acc /= self.visited_count
        self.total_visited_loss /= self.visited_count

        self.total_occupancy_acc = (
            self.total_occupancy_acc[0] / self.total_occupancy_acc[1]
            + self.total_occupancy_acc[2] / self.total_occupancy_acc[3]
        ) / 2.0
        self.total_unbal_occupancy_acc /= self.occupancy_count
        self.total_occupancy_loss /= self.occupancy_count


def train_epoch(model, optim, lr_sched, loader, writer, step):
    device = next(model.parameters()).device
    model.train()

    metrics = EpochMetrics(device)

    world_rank = distrib.get_rank()
    with tqdm.tqdm(
        total=len(loader)
    ) if world_rank == 0 else contextlib.suppress() as pbar:
        for batch in loader:
            batch = tuple(v.to(device) for v in batch)
            x, y, grid, mask = batch

            visited_logits, occupancy_logits = model(x)

            visited_loss = focal_loss(visited_logits, y)
            occupancy_loss = focal_loss(occupancy_logits, grid, mask)

            optim.zero_grad()
            with amp.scale_loss(
                50.0 * visited_loss.mean() + occupancy_loss.mean(), optim
            ) as scaled_loss:
                scaled_loss.backward()
            optim.step()
            lr_sched.step()

            metrics.update(
                batch, visited_logits, occupancy_logits, visited_loss, occupancy_loss
            )

            step += 1

            if world_rank == 0:
                writer.add_scalars(
                    "visited_loss", {"train": visited_loss.mean().item()}, step
                )
                writer.add_scalars(
                    "occupancy_loss", {"train": occupancy_loss.mean().item()}, step
                )

                pbar.update()
                pbar.refresh()

    metrics.finalize()

    if distrib.get_rank() == 0:
        writer.add_scalars(
            "visited_bal_acc", {"train": metrics.total_visited_acc * 1e2}, step
        )
        writer.add_scalars(
            "occupancy_bal_acc", {"train": metrics.total_occupancy_acc * 1e2}, step
        )

    return step


best_eval_acc = 0.0


def eval_epoch(model, loader, writer, step):
    global best_eval_acc
    device = next(model.parameters()).device
    model.eval()

    metrics = EpochMetrics(device)

    for batch in loader:
        batch = tuple(v.to(device) for v in batch)
        x, y, grid, mask = batch

        with torch.no_grad():
            visited_logits, occupancy_logits = model(x)

            visited_loss = focal_loss(visited_logits, y)
            occupancy_loss = focal_loss(occupancy_logits, grid, mask)

            metrics.update(
                batch, visited_logits, occupancy_logits, visited_loss, occupancy_loss
            )

    metrics.finalize()

    if distrib.get_rank() == 0:
        writer.add_scalars("visited_loss", {"val": metrics.total_visited_loss}, step)
        writer.add_scalars(
            "occupancy_loss", {"val": metrics.total_occupancy_loss}, step
        )
        writer.add_scalars(
            "visited_bal_acc", {"val": metrics.total_visited_acc * 1e2}, step
        )
        writer.add_scalars(
            "occupancy_bal_acc", {"val": metrics.total_occupancy_acc * 1e2}, step
        )
    else:
        return

    total_acc = metrics.total_occupancy_acc.item() + metrics.total_visited_acc.item()
    total_acc = metrics.total_occupancy_acc.item()
    if total_acc < best_eval_acc:
        return

    best_eval_acc = total_acc

    torch.save(model.module.state_dict(), "data/best_visited_predictor.pt")


def softmax_classifier():
    local_rank, _ = init_distrib_slurm()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    batch_size = 128

    train_dset = VisitedDataset("train")
    val_dset = VisitedDataset("val")

    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size,
        sampler=torch.utils.data.DistributedSampler(train_dset),
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size,
        sampler=torch.utils.data.DistributedSampler(val_dset),
        pin_memory=True,
        num_workers=4,
    )

    world_size = distrib.get_world_size()
    base_lr = 1e-3
    num_epochs = 300

    train_len = len(train_loader)

    def warmup_lr(step):
        step = step / train_len
        if step < 5:
            return batch_size * (1 + step / 5 * (world_size - 1)) / 64
        else:
            return world_size * batch_size / 64

    input_size = 512 * 6
    model = Model(input_size, num_bins)
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0)

    model, optim = amp.initialize(model, optim, opt_level="O2")

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optim, warmup_lr)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )

    step = 0
    with tensorboard.SummaryWriter(
        log_dir="map_extraction_runs/visited_predictor", purge_step=step, flush_secs=30
    ) if distrib.get_rank() == 0 else contextlib.suppress() as writer, tqdm.tqdm(
        total=300
    ) if distrib.get_rank() == 0 else contextlib.suppress() as pbar:
        eval_epoch(model, val_loader, writer, step)
        for i in range(num_epochs):
            train_loader.sampler.set_epoch(i)
            val_loader.sampler.set_epoch(i)

            step = train_epoch(model, optim, lr_sched, train_loader, writer, step)
            eval_epoch(model, val_loader, writer, step)

            if distrib.get_rank() == 0:
                pbar.update()
                pbar.refresh()


if __name__ == "__main__":
    softmax_classifier()
