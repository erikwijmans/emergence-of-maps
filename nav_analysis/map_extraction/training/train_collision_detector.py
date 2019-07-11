import argparse
import glob
import os.path as osp

import h5py as h5
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import tqdm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from torch.utils import tensorboard

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda", 0)

num_bins = 20
bin_size = None


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
            self.collisions_cls = nn.Linear(input_size, 1)

        def _make_layer(inp, out, p=0.0):
            return nn.Sequential(
                nn.Dropout(p=p),
                nn.Linear(inp, out, bias=False),
                nn.BatchNorm1d(out),
                nn.ReLU(True),
            )

        hidden_size = 256
        self.pose_preder = nn.Sequential(
            _make_layer(input_size, hidden_size),
            #  _make_layer(hidden_size, hidden_size, p=0.0),
            #  _make_layer(hidden_size, hidden_size, p=0.0),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, num_bins * 2),
        )

        hidden_size = 512
        self.goal_preder = nn.Sequential(
            #  _make_layer(input_size, hidden_size),
            #  _make_layer(hidden_size, hidden_size, p=0.5),
            #  nn.Dropout(p=0.5),
            nn.Linear(input_size, 3)
        )

    def forward(self, x):
        logits = self.collisions_cls(x)
        pose_preds = self.pose_preder(x)
        goal_pred = self.goal_preder(x)
        goal_pred = torch.cat(
            [
                goal_pred[:, 0:1],
                goal_pred[:, 1:]
                / torch.norm(goal_pred[:, 1:], dim=-1, keepdim=True),
            ],
            -1,
        )

        return logits.squeeze(), pose_preds, goal_pred


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
            (torch.argmax(preds[:, 0:num_bins], -1) - gt[:, 0]).abs()
            * bin_size[0]
            + (torch.argmax(preds[:, num_bins:], -1) - gt[:, 1]).abs()
            * bin_size[1]
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
        return (
            (torch.norm(preds - gt, dim=-1) / torch.norm(gt, dim=-1))
            .mean()
            .item()
        )
    else:
        return torch.norm(preds - gt, dim=-1).mean().item()


class CollisionDataset(object):
    bins = None

    def __init__(self, split):
        global bin_size

        fname = f"data/map_extraction/collisions/collision_and_pose_dset_{split}.h5"
        with h5.File(fname, "r") as f:
            self.xs = f["hidden_states"][()]
            self.ys = f["collision_labels"][()].astype(np.float32)
            self.poses = f["positions"][()]
            self.goals = f["goal_vectors"][()]

        new_xs = np.zeros_like(self.xs)
        inds = np.array([1690, 923, 1938, 725, 2027])
        new_xs[:, inds] = self.xs[:, inds]
        #  self.xs = new_xs

        if split == "train":
            x_range = [self.poses.min(0)[0], self.poses.max(0)[0]]
            y_range = [self.poses.min(0)[1], self.poses.max(0)[1]]
            type(self).bins = (
                np.linspace(x_range[0], x_range[1], num=num_bins),
                np.linspace(y_range[0], y_range[1], num=num_bins),
            )
            bin_size = (
                self.bins[0][1] - self.bins[0][0],
                self.bins[1][1] - self.bins[1][0],
            )

        self.poses = np.stack(
            [
                np.searchsorted(self.bins[0], self.poses[:, 0]),
                np.searchsorted(self.bins[1], self.poses[:, 1]),
            ],
            -1,
        ).astype(np.int64)

        self.tensors = [self.xs, self.ys, self.poses, self.goals]
        self.tensors = [torch.from_numpy(v) for v in self.tensors]


def train_epoch(model, optim, loader, writer, step):
    model.train()
    total_pose_err = 0.0
    for batch in tqdm.tqdm(loader, total=len(loader), leave=False):
        batch = tuple(v.to(device) for v in batch)
        x, y, poses, goal_gt = batch

        logits, pose_preds, goal_preds = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        l_pose = pose_bins_loss(pose_preds, poses)

        l1_pen_loss = 0.0 * model.collisions_cls.weight.abs().mean()

        optim.zero_grad()
        (
            loss
            + l_pose
            + pose_loss(goal_preds, goal_gt, rel=False)
            + l1_pen_loss
        ).backward()
        optim.step()

        acc = (
            (y == (torch.sigmoid(logits) > 0.5).float()).float().mean().item()
        )
        writer.add_scalars("acc", {"train": acc}, step)
        writer.add_scalars("loss", {"train": loss.item()}, step)
        writer.add_scalars(
            "pose_error", {"train": pose_bins_err(pose_preds, poses)}, step
        )
        total_pose_err += pose_bins_err(pose_preds, poses)

        step += 1

    print(total_pose_err / len(loader))

    return step


def eval_epoch(model, loader, writer, step):
    model.eval()
    total_loss = 0.0
    total_pose_err = 0.0
    total_goal_err = 0.0
    total_acc = 0.0
    for batch in loader:
        batch = tuple(v.to(device) for v in batch)
        x, y, poses, goal_gt = batch

        with torch.no_grad():
            logits, pose_preds, goal_pred = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            acc = (y == (torch.sigmoid(logits) > 0.5).float()).float().mean()

        total_acc += acc.item()
        total_loss += loss.item()
        total_pose_err += pose_bins_err(pose_preds, poses)
        total_goal_err += pose_rel_l2_error(goal_pred, goal_gt, rel=False)

    writer.add_scalars("acc", {"val": total_acc / len(loader)}, step)
    writer.add_scalars("loss", {"val": total_loss / len(loader)}, step)
    writer.add_scalars(
        "pose_error", {"val": total_pose_err / len(loader)}, step
    )

    tqdm.tqdm.write("Acc={:.3f}".format(total_acc / len(loader) * 1e2))
    tqdm.tqdm.write("PoseErr={:.3f}".format(total_pose_err / len(loader)))
    tqdm.tqdm.write("GoalErr={:.3f}".format(total_goal_err / len(loader)))


def _build_dsets():
    return (
        torch.utils.data.TensorDataset(*CollisionDataset("train").tensors),
        torch.utils.data.TensorDataset(*CollisionDataset("val").tensors),
        2048,
    )


def softmax_classifier():
    batch_size = 128
    train_dset, val_dset, input_size = _build_dsets()
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dset, 256, pin_memory=False, num_workers=0
    )

    model = Model(input_size, linear=True)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.0)

    step = 0
    with tensorboard.SummaryWriter(
        log_dir="map_extraction_runs/testing", purge_step=step, flush_secs=30
    ) as writer:
        for _ in tqdm.trange(300):
            eval_epoch(model, val_loader, writer, step)
            step = train_epoch(model, optim, train_loader, writer, step)


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
    print(clf.score(train_x, train_y))


def random_forest():
    clf = RandomForestClassifier(n_estimators=1000, max_depth=3, n_jobs=-1)
    _sklearn_classify(clf)


def svm():
    clf = LinearSVC(max_iter=int(1e5))
    _sklearn_classify(clf, True)


if __name__ == "__main__":
    softmax_classifier()
