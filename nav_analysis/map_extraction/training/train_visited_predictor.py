import argparse
import contextlib
import glob
import os.path as osp

import h5py as h5
import imageio
import numba
import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import tqdm
from apex import amp
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from torch.utils import tensorboard

from nav_analysis.utils.ddp_utils import init_distrib_slurm

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
        _future_map = self._f["maps"][idx // self._samples_per][-1].astype(np.int64)

        xs = np.reshape(
            self._f["xs"][idx // self._samples_per][idx % self._samples_per], -1
        )
        grid = self._f["occupancy_grids"][idx // self._samples_per].astype(np.int64)

        mask = create_occupancy_grid_mask(_map, napply=2)
        mask = np.ones_like(mask).astype(np.bool)

        #  d_goal = self._f["d_goal"][idx // self._samples_per][idx % self._samples_per]
        #  d_start = self._f["d_start"][idx // self._samples_per][idx % self._samples_per]

        d_goal = np.array([0.0], dtype=np.float32)[0]
        d_start = d_goal

        return xs, _map, _future_map - _map, grid, mask, d_start, d_goal

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
            inp + 2, out, stride=stride, kernel_size=3, padding=0, bias=False
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


def _make_layer(inp, out, p=0.0):
    return nn.Sequential(
        nn.Dropout(p=p),
        nn.Linear(inp, out, bias=False),
        nn.BatchNorm1d(out),
        nn.ReLU(True),
    )


def _make_coord_deconv(inp, out, p=0.0):
    return nn.Sequential(
        nn.Dropout2d(p=p), CoordDeconv(inp, out, 2), nn.BatchNorm2d(out), nn.ReLU(True)
    )


def _make_coord_conv(inp, out, p=0.0):
    return nn.Sequential(
        nn.Dropout2d(p=p),
        CoordConv(inp, out, 3, bias=False),
        nn.BatchNorm2d(out),
        nn.ReLU(True),
    )


class BaseModel(nn.Module):
    def __init__(self, input_size, num_bins, linear=False):
        super().__init__()

        self.num_bins = num_bins
        hidden_size = 256
        div = 16
        self.hidden_spatial_size = np.array(
            [32, num_bins[0] // div, num_bins[1] // div]
        )
        self.hidden_reshape = nn.Sequential(
            _make_layer(input_size, hidden_size),
            _make_layer(hidden_size, np.prod(self.hidden_spatial_size)),
        )

        self.backbone = nn.Sequential(
            _make_coord_conv(32, 32),
            _make_coord_deconv(32, 32),
            _make_coord_conv(32, 32),
            _make_coord_deconv(32, 64),
            _make_coord_conv(64, 64),
            _make_coord_deconv(64, 64),
            _make_coord_conv(64, 64),
            _make_coord_deconv(64, 64),
            _make_coord_conv(64, 64),
        )

    def forward(self, x):
        x = self.hidden_reshape(x)
        x = x.view(-1, *self.hidden_spatial_size)
        x = self.backbone(x)
        return x


class VisitedModel(nn.Module):
    def __init__(self, input_size, num_bins, linear=False):
        super().__init__()
        self.visited_model = BaseModel(input_size, num_bins)
        self.future_head = nn.Sequential(
            _make_coord_conv(64, 64), nn.Dropout(p=0.0), nn.Conv2d(64, 2, 1)
        )
        self.past_head = nn.Sequential(
            _make_coord_conv(64, 64), nn.Dropout(p=0.0), nn.Conv2d(64, 2, 1)
        )
        self.num_bins = num_bins

    def _cleanup_pred(self, pred):
        pred = pred[:, :, 0 : self.num_bins[0], 0 : self.num_bins[1]].permute(
            0, 2, 3, 1
        )
        return pred.contiguous()

    def forward(self, x):
        visited_feats = self.visited_model(x)
        past = self._cleanup_pred(self.past_head(visited_feats))
        fut = self._cleanup_pred(self.future_head(visited_feats))

        return past, fut


class OccupancyModel(nn.Module):
    def __init__(self, input_size, num_bins, linear=False):
        super().__init__()
        self.occupancy_model = BaseModel(input_size, num_bins)
        self.occupancy_head = nn.Sequential(nn.Dropout(p=0.0), nn.Conv2d(64, 2, 1))
        self.num_bins = num_bins

    def _cleanup_pred(self, pred):
        pred = pred[:, :, 0 : self.num_bins[0], 0 : self.num_bins[1]].permute(
            0, 2, 3, 1
        )
        return pred.contiguous()

    def forward(self, x):
        occ = self._cleanup_pred(self.occupancy_head(self.occupancy_model(x)))

        return occ


class GeoModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.geo_shared = nn.Sequential(
            nn.Linear(input_size, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(True)
        )
        self.d_start = nn.Linear(512, 1)
        self.d_goal = nn.Linear(512, 1)

    def forward(self, x):
        geo_feats = self.geo_shared(x)

        return (self.d_start(geo_feats).squeeze(-1), self.d_goal(geo_feats).squeeze(-1))


class Model(nn.Module):
    def __init__(self, input_size, num_bins, linear=False):
        super().__init__()
        self.visited_model = VisitedModel(input_size, num_bins)

        self.occ_model = OccupancyModel(input_size, num_bins)

        self.geo_model = GeoModel(input_size)

    def forward(self, x):
        past, fut = self.visited_model(x)
        occ = self.occ_model(x)
        d_start, d_goal = self.geo_model(x)

        return past, fut, occ, d_start, d_goal


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


class VisitedEpochMetrics:
    def __init__(self, device):
        self.device = device
        self.total_visited_acc = []
        self.total_unbal_visited_acc = 0.0

        self.unbalanced_acc = 0.0
        self.visited_count = torch.tensor(0.0, device=device)

        self.total_visited_loss = 0.0

    @torch.no_grad()
    def update(self, y, visited_logits, visited_loss):
        self.visited_count += y.numel()

        self.total_visited_loss += visited_loss.sum()

        self.total_unbal_visited_acc += (
            (torch.argmax(visited_logits, -1) == y).float().sum()
        )

        self.total_visited_acc.append(mapping_acc(visited_logits, y))

    def finalize(self):
        self.total_visited_acc = torch.tensor(
            self.total_visited_acc, device=self.device
        ).sum(0)

        distrib.all_reduce(self.total_visited_acc)
        distrib.all_reduce(self.total_unbal_visited_acc)
        distrib.all_reduce(self.total_visited_loss)
        distrib.all_reduce(self.visited_count)

        self.total_visited_acc = (
            self.total_visited_acc[0] / self.total_visited_acc[1]
            + self.total_visited_acc[2] / self.total_visited_acc[3]
        ) / 2.0
        self.total_unbal_visited_acc /= self.visited_count
        self.total_visited_loss /= self.visited_count


class EpochMetrics:
    def __init__(self, device):
        self.device = device

        self.past_visited = VisitedEpochMetrics(device)
        self.future_visited = VisitedEpochMetrics(device)

        self.total_occupancy_acc = []
        self.total_unbal_occupancy_acc = 0.0

        self.unbalanced_acc = 0.0
        self.occupancy_count = torch.tensor(0.0, device=device)

        self.total_occupancy_loss = 0.0

    @torch.no_grad()
    def update(
        self,
        batch,
        past_visited_logits,
        future_visited_logits,
        occupancy_logits,
        past_visited_loss,
        future_visited_loss,
        occupancy_loss,
    ):
        x, y_past, y_future, grid, mask, _, _ = batch
        self.past_visited.update(y_past, past_visited_logits, past_visited_loss)
        self.future_visited.update(y_future, future_visited_logits, future_visited_loss)

        self.occupancy_count += mask.float().sum()

        self.total_occupancy_loss += occupancy_loss.sum()

        self.total_unbal_occupancy_acc += (
            torch.masked_select((torch.argmax(occupancy_logits, -1) == grid), mask)
            .float()
            .sum()
        )

        self.total_occupancy_acc.append(mapping_acc(occupancy_logits, grid, mask))

    def finalize(self):
        self.future_visited.finalize()
        self.past_visited.finalize()
        self.total_occupancy_acc = torch.tensor(
            self.total_occupancy_acc, device=self.device
        ).sum(0)

        distrib.all_reduce(self.total_occupancy_acc)
        distrib.all_reduce(self.total_unbal_occupancy_acc)
        distrib.all_reduce(self.total_occupancy_loss)
        distrib.all_reduce(self.occupancy_count)

        self.total_occupancy_acc = (
            self.total_occupancy_acc[0] / self.total_occupancy_acc[1]
            + self.total_occupancy_acc[2] / self.total_occupancy_acc[3]
        ) / 2.0
        self.total_unbal_occupancy_acc /= self.occupancy_count
        self.total_occupancy_loss /= self.occupancy_count


def train_epoch(model, optims, lr_scheds, loader, writer, step):
    device = next(model.parameters()).device
    model.train()

    metrics = EpochMetrics(device)

    world_rank = distrib.get_rank()
    with tqdm.tqdm(
        total=len(loader)
    ) if world_rank == 0 else contextlib.suppress() as pbar:
        for batch in loader:
            batch = tuple(v.to(device) for v in batch)
            x, y_past, y_future, grid, mask, gt_d_start, gt_d_goal = batch

            past_logits, future_logits, occupancy_logits, d_start, d_goal = model(x)

            future_visted_loss = focal_loss(future_logits, y_future)
            past_visited_loss = focal_loss(past_logits, y_past)
            occupancy_loss = focal_loss(occupancy_logits, grid, mask)
            start_loss = F.smooth_l1_loss(d_start, gt_d_start)
            goal_loss = F.smooth_l1_loss(d_goal, gt_d_goal)

            for optim in optims:
                optim.zero_grad()

            with amp.scale_loss(
                past_visited_loss.mean()
                + future_visted_loss.mean()
                + occupancy_loss.mean()
                + start_loss
                + goal_loss,
                optims,
            ) as scaled_loss:
                scaled_loss.backward()

            for i in range(len(optims)):
                optims[i].step()
                lr_scheds[i].step()

            metrics.update(
                batch,
                past_logits,
                future_logits,
                occupancy_logits,
                past_visited_loss,
                future_visted_loss,
                occupancy_loss,
            )

            step += 1

            if world_rank == 0:
                writer.add_scalars(
                    "visited_loss",
                    {"train_future": future_visted_loss.mean().item()},
                    step,
                )
                writer.add_scalars(
                    "visited_loss",
                    {"train_past": past_visited_loss.mean().item()},
                    step,
                )
                writer.add_scalars(
                    "occupancy_loss", {"train": occupancy_loss.mean().item()}, step
                )
                writer.add_scalars(
                    "geo_loss",
                    {"train_start": start_loss.item(), "train_goal": goal_loss.item()},
                    step,
                )

                pbar.update()
                pbar.refresh()

    metrics.finalize()

    if distrib.get_rank() == 0:
        writer.add_scalars(
            "visited_bal_acc",
            {"train_future": metrics.future_visited.total_visited_acc * 1e2},
            step,
        )
        writer.add_scalars(
            "visited_bal_acc",
            {"train_past": metrics.past_visited.total_visited_acc * 1e2},
            step,
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

        with torch.no_grad():
            batch = tuple(v.to(device) for v in batch)
            x, y_past, y_future, grid, mask, _, _ = batch

            past_logits, future_logits, occupancy_logits, _, _ = model(x)

            future_visted_loss = focal_loss(future_logits, y_future)
            past_visited_loss = focal_loss(past_logits, y_past)
            occupancy_loss = focal_loss(occupancy_logits, grid, mask)

            metrics.update(
                batch,
                past_logits,
                future_logits,
                occupancy_logits,
                past_visited_loss,
                future_visted_loss,
                occupancy_loss,
            )

    metrics.finalize()

    if distrib.get_rank() == 0:
        writer.add_scalars(
            "visited_loss",
            {"val_future": metrics.future_visited.total_visited_loss},
            step,
        )
        writer.add_scalars(
            "visited_loss", {"val_past": metrics.past_visited.total_visited_loss}, step
        )
        writer.add_scalars(
            "occupancy_loss", {"val": metrics.total_occupancy_loss}, step
        )
        writer.add_scalars(
            "visited_bal_acc",
            {"val_future": metrics.future_visited.total_visited_acc * 1e2},
            step,
        )
        writer.add_scalars(
            "visited_bal_acc",
            {"val_past": metrics.past_visited.total_visited_acc * 1e2},
            step,
        )
        writer.add_scalars(
            "occupancy_bal_acc", {"val": metrics.total_occupancy_acc * 1e2}, step
        )
    else:
        return

    total_acc = metrics.total_occupancy_acc.item()
    if total_acc < best_eval_acc:
        return

    best_eval_acc = total_acc

    torch.save(
        model.module.state_dict(), "data/best_future_visited_predictor_with_grad.pt"
    )


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

    models = [model.occ_model, model.visited_model, model.geo_model]
    optims = [
        torch.optim.Adam(m.parameters(), lr=base_lr, weight_decay=0.0) for m in models
    ]

    models, optims = amp.initialize(models, optims, opt_level="O2", enabled=False)

    model.occ_model = models[0]
    model.visited_model = models[1]
    model.geo_model = models[2]

    lr_scheds = [
        torch.optim.lr_scheduler.LambdaLR(optim, warmup_lr) for optim in optims
    ]

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], output_device=device, find_unused_parameters=True
    )

    step = 0
    with tensorboard.SummaryWriter(
        log_dir="map_extraction_runs/future_visited_predictor",
        purge_step=step,
        flush_secs=30,
    ) if distrib.get_rank() == 0 else contextlib.suppress() as writer, tqdm.tqdm(
        total=300
    ) if distrib.get_rank() == 0 else contextlib.suppress() as pbar:
        #  eval_epoch(model, val_loader, writer, step)
        for i in range(num_epochs):
            train_loader.sampler.set_epoch(i)
            val_loader.sampler.set_epoch(i)

            step = train_epoch(model, optims, lr_scheds, train_loader, writer, step)
            eval_epoch(model, val_loader, writer, step)

            if distrib.get_rank() == 0:
                pbar.update()
                pbar.refresh()


if __name__ == "__main__":
    softmax_classifier()
