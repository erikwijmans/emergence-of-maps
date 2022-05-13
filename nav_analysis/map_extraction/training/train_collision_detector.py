import argparse
import glob
import os.path as osp
import math
import functools
import gzip
import submitit

import h5py as h5
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import tqdm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from torch.utils import tensorboard
import torch.nn.utils.prune as prune
import copy

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda", 0)

num_bins = 30
bin_size = 1.0


class Model(nn.Module):
    def __init__(self, input_size, linear=False):
        super().__init__()

        if not linear:
            self.collisions_cls = nn.Sequential(
                nn.Linear(input_size, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(256, 2),
            )
        else:
            self.collisions_cls = nn.Linear(input_size, 1, bias=True)

        #  def _make_layer(inp, out, p=0.0):
        #  return nn.Sequential(
        #  nn.Dropout(p=p),
        #  nn.Linear(inp, out, bias=False),
        #  nn.BatchNorm1d(out),
        #  nn.ReLU(True),
        #  )

        #  hidden_size = 256
        #  self.pose_preder = nn.Sequential(
        #  _make_layer(input_size, hidden_size, p=0.0),
        #  #  _make_layer(hidden_size, hidden_size, p=0.0),
        #  #  _make_layer(hidden_size, hidden_size, p=0.0),
        #  nn.Linear(hidden_size, num_bins * 2),
        #  )

        #  hidden_size = 512
        #  self.goal_preder = nn.Sequential(
        #  #  _make_layer(input_size, hidden_size),
        #  #  _make_layer(hidden_size, hidden_size, p=0.5),
        #  #  nn.Dropout(p=0.5),
        #  nn.Linear(input_size, 3)
        #  )

    def forward(self, x):
        logits = self.collisions_cls(x)
        #  pose_preds = self.pose_preder(x)
        #  goal_pred = self.goal_preder(x)
        #  goal_pred = torch.cat(
        #  [
        #  goal_pred[:, 0:1],
        #  goal_pred[:, 1:] / torch.norm(goal_pred[:, 1:], dim=-1, keepdim=True),
        #  ],
        #  -1,
        #  )

        return logits.squeeze()  # , pose_preds, goal_pred


def focal_loss(logits, y, gamma=0.0):
    return F.cross_entropy(logits, y)
    alpha = 0.5

    weights = torch.tensor([1.0 - alpha, alpha], device=device)
    weights.fill_(1.0)

    if gamma > 0.0:
        loss = F.cross_entropy(logits, y, weight=weights, reduction="none")

        probs = F.softmax(logits, -1)
        loss = (1 - probs[:, y].detach()).pow(gamma) * loss

        return loss.mean()
    else:
        return F.cross_entropy(logits, y, weight=weights)


def pose_bins_loss(preds, gt):
    return F.cross_entropy(preds[:, 0:num_bins], gt[:, 0]) + F.cross_entropy(
        preds[:, num_bins:], gt[:, 1]
    )


def pose_bins_err(preds, gt):
    return (
        (
            (torch.argmax(preds[:, 0:num_bins], -1) - gt[:, 0]).abs() * bin_size[0]
            + (torch.argmax(preds[:, num_bins:], -1) - gt[:, 1]).abs() * bin_size[1]
        )
        .float()
        .mean()
        .item()
    )


def pose_loss(preds, gt, rel=True):
    if rel:
        return (
            F.smooth_l1_loss(preds, gt, reduction="none")
            / torch.norm(gt, dim=-1, keepdim=True)
        ).mean()
    else:
        return F.smooth_l1_loss(preds, gt)


def pose_rel_l2_error(preds, gt, rel=True):
    if rel:
        return (torch.norm(preds - gt, dim=-1) / torch.norm(gt, dim=-1)).mean().item()
    else:
        return torch.norm(preds - gt, dim=-1).mean().item()


class CollisionDataset(object):
    bins = None

    def __init__(self, split):

        fname = "data/map_extraction/collisions/collision-dset-2022-20-01_{split}.h5"
        slice_to_load = slice(None)
        with h5.File(fname.format(split=split), "r") as f:
            self.xs = f["hidden_states"][slice_to_load]
            #  self.xs = self.xs.reshape(self.xs.shape[0], -1, 512)
            self.ys = f["collision_labels"][slice_to_load].astype(np.float32)

        if split == "train":
            mean, std = (
                np.mean(self.xs, 0, keepdims=True),
                np.std(self.xs, 0, keepdims=True),
            )
        else:
            with h5.File(fname.format(split="train"), "r") as f:
                tmp = f["hidden_states"][slice_to_load]
            mean, std = np.mean(tmp, 0, keepdims=True), np.std(tmp, 0, keepdims=True)

        self.xs = (self.xs - mean) / std

        self.tensors = [self.xs, self.ys]


def train_epoch(model, optim, loader, writer, step, l1_pen):
    model.train()
    total_pose_err = 0.0
    for batch in tqdm.tqdm(loader, total=len(loader), leave=False):
        optim.zero_grad()
        batch = tuple(v.to(device, non_blocking=True) for v in batch)
        x, y = batch

        #  logits, pose_preds, goal_preds = model(x)
        #  loss = F.binary_cross_entropy_with_logits(logits, y)
        #  l_pose = pose_bins_loss(pose_preds, poses)

        l1_pen_loss = l1_pen * model.collisions_cls.weight.abs().mean()

        #  (
        #  loss + l_pose + pose_loss(goal_preds, goal_gt, rel=False) + l1_pen_loss
        #  ).backward()

        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        (loss + l1_pen_loss).backward()

        optim.step()

        acc = (y == (torch.sigmoid(logits) > 0.5).float()).float().mean().item()
        writer.add_scalars("acc", {"train": acc}, step)
        writer.add_scalars("loss", {"train": loss.item()}, step)
        #  writer.add_scalars(
        #  "pose_error", {"train": pose_bins_err(pose_preds, poses)}, step
        #  )
        #  total_pose_err += pose_bins_err(pose_preds, poses)

        step += 1

    #  print(total_pose_err / len(loader))

    return step


@torch.no_grad()
def compute_num_sig_neurons(model, loader):
    assert len(loader) == 1

    for batch in loader:
        batch = tuple(v.to(device, non_blocking=True) for v in batch)
        x, y = batch

    orig_weight = model.collisions_cls.weight.data.detach()
    _, l1_prune_ordering = torch.sort(orig_weight.abs().squeeze(0))
    num_neurons = orig_weight.size(1)
    pruned_weights = orig_weight.unsqueeze(0).repeat(num_neurons, 1, 1)
    for i in range(1, orig_weight.size(1)):
        pruned_weights[i, :].index_fill_(1, l1_prune_ordering[0:i], 0)

    preds = (
        (
            torch.sigmoid(
                torch.einsum("noi, bi -> nbo", pruned_weights, x)
                + model.collisions_cls.bias
            )
            > 0.5
        )
        .float()
        .squeeze(-1)
    )
    y = y.view(1, -1).expand_as(preds)
    bal_acc = 0.5 * (y == preds)[y == 1.0].float().view(num_neurons, -1).mean(
        -1
    ) + 0.5 * (y == preds)[y == 0.0].float().view(num_neurons, -1).mean(-1)
    best_acc = bal_acc.max()

    num_pruned = torch.max(
        torch.nonzero(bal_acc > (0.99 * best_acc), as_tuple=False).squeeze()
    ).item()

    return (
        num_neurons - num_pruned,
        [int(v) for v in pruned_weights[num_pruned].squeeze().nonzero().cpu()],
    )


@torch.no_grad()
def eval_epoch(model, loader, writer, step):
    model.eval()
    total_pose_err = 0.0
    total_goal_err = 0.0

    assert len(loader) == 1

    for batch in loader:
        batch = tuple(v.to(device, non_blocking=True) for v in batch)
        x, y = batch

    def _eval(model):
        total_acc = 0.0
        total_loss = 0.0

        with torch.no_grad():
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (y == preds).float().mean()

            bal_acc = (
                0.5 * (y == preds)[y == 1.0].float().mean()
                + 0.5 * (y == preds)[y == 0.0].float().mean()
            )

        return acc.item(), bal_acc.item(), loss.item()

    total_acc, bal_acc, total_loss = _eval(model)

    #  tqdm.tqdm.write("Num sig neurons: {:d}".format(n_sig_neurons))

    writer.add_scalars("acc", {"val": total_acc}, step)
    writer.add_scalars("loss", {"val": total_loss}, step)
    #  writer.add_scalars("pose_error", {"val": total_pose_err / len(loader)}, step)

    #  tqdm.tqdm.write("Acc={:.3f}".format(bal_acc * 1e2))
    #  tqdm.tqdm.write("PoseErr={:.3f}".format(total_pose_err / len(loader)))
    #  tqdm.tqdm.write("GoalErr={:.3f}".format(total_goal_err / len(loader)))

    return total_acc, bal_acc


def _build_dsets():
    def _t(n):
        return torch.from_numpy(n).to(device=device)

    return (
        torch.utils.data.TensorDataset(*map(_t, CollisionDataset("train").tensors)),
        torch.utils.data.TensorDataset(*map(_t, CollisionDataset("val").tensors)),
        512 * 6,
    )


def softmax_classifier(l1_pen: float, dset_res):
    batch_size = 256
    train_dset, val_dset, input_size = dset_res
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dset, len(val_dset), pin_memory=False, num_workers=0,
    )

    model = Model(input_size, linear=True)
    model = model.to(device)
    optim = torch.optim._multi_tensor.Adam(model.parameters(), lr=5e-4,)

    best_acc = 0.0
    best_bal_acc = 0.0
    ret = None
    best_model = None
    step = 0
    with tensorboard.SummaryWriter(
        log_dir="map_extraction_runs/testing", purge_step=step, flush_secs=30
    ) as writer:
        for _ in tqdm.trange(20, leave=False):
            step = train_epoch(model, optim, train_loader, writer, step, l1_pen)
            acc, bal_acc = eval_epoch(model, val_loader, writer, step)

            if bal_acc > best_bal_acc:
                best_acc = acc
                best_bal_acc = bal_acc
                with torch.no_grad():
                    best_model = copy.deepcopy(model)

    num_sig, sig_nueron_inds = compute_num_sig_neurons(best_model, val_loader)

    return best_acc, best_bal_acc, num_sig, sig_nueron_inds


def _sklearn_classify(clf, pca=False):
    dset = CollisionDataset("train")
    train_x = dset.xs
    train_y = dset.ys

    dset = CollisionDataset("val")
    val_x = dset.xs
    val_y = dset.ys

    if pca:
        pca = PCA(n_components=0.6)
        pca.fit(train_x)
        train_x = pca.transform(train_x)
        val_x = pca.transform(val_x)

    clf.fit(train_x, train_y)
    return clf, (val_x, val_y)


def random_forest():
    clf = RandomForestClassifier(n_estimators=1000, max_depth=3, n_jobs=-1)
    _sklearn_classify(clf)


def svm():
    clf = LinearSVC(max_iter=int(1e5))
    _sklearn_classify(clf, True)


def sklearn_logistic(l1_pen):
    clf = LogisticRegression(
        C=float(1.0 / l1_pen) if l1_pen > 1.0 else 1.0,
        penalty="l1",
        solver="liblinear",
        n_jobs=-1,
    )
    clf, (val_x, val_y) = _sklearn_classify(clf)
    print(clf.score(val_x, val_y))

    coefs = clf.coef_.ravel()
    print(np.mean(coefs == 0.0))


def run_job(l1_pens, rank):
    dset_res = _build_dsets()
    data = []
    for l1_pen in tqdm.tqdm(l1_pens):
        for _ in tqdm.trange(5, leave=False):
            acc, bal_acc, n_sig, sig_nueron_inds = softmax_classifier(l1_pen, dset_res)

            data.append(
                dict(
                    l1_pen=float(l1_pen),
                    acc=float(acc),
                    n_sig=int(n_sig),
                    bal_acc=float(bal_acc),
                    sig_nueron_inds=sig_nueron_inds,
                )
            )

    with gzip.open(f"collision_detector_res/{rank}.json.gz", "wt") as f:
        json.dump(data, f)


if __name__ == "__main__":
    executor = submitit.AutoExecutor(
        folder="/checkpoint/erikwijmans/submitit/%j", cluster="slurm"
    )
    executor.update_parameters(
        mem_gb=50,
        gpus_per_node=1,
        tasks_per_node=1,
        cpus_per_task=5,
        nodes=1,
        timeout_min=60 * 24,
        slurm_partition="devlab",
        name="nav-analysis-eval",
        slurm_signal_delay_s=60,
    )
    l1_pens = [0.0] + list(torch.logspace(math.log2(4), math.log2(1024), 200, base=2.0))
    chunks = 32
    num_per_chunk = int(math.ceil(len(l1_pens) / chunks))
    jobs = []
    with executor.batch():
        for r in range(chunks):
            j = executor.submit(
                run_job, l1_pens[r * num_per_chunk : (r + 1) * num_per_chunk], r
            )
            jobs.append(j)

    for job in tqdm.tqdm(jobs):
        print(job.results())

    data = []
    for rank in range(chunks):
        with gzip.open(f"collision_detector_res/{rank}.json.gz", "rt") as f:
            data += json.load(f)

    with gzip.open("collision_detection_and_pruning.json.gz", "wt") as f:
        json.dump(data, f)
