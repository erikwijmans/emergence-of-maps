import attr
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distrib
import torchvision
import torchvision.transforms as T


from nav_analysis.utils.ddp_utils import init_distrib_slurm
from nav_analysis.rl.ppo.policy import ResNetEncoder
from nav_analysis.rl.running_mean_and_var import RunningMeanAndVar
import nav_analysis.rl.resnet

from apex import amp

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


@attr.s(auto_attribs=True)
class WarmUpSGDR:
    train_len: int
    warmup_amount: float
    T0: int = 1
    T_mult: int = 2
    warm_up_epochs: int = 3

    def __attrs_post_init__(self):
        self.n_restarts = 0
        self.last_restart = 0
        self._init_warmup_amount = self.warmup_amount

    def __call__(self, i):
        epoch = i / self.train_len
        if epoch < self.warm_up_epochs:
            return 1 + (self.warmup_amount - 1) * epoch / self.warm_up_epochs

        epoch -= self.warm_up_epochs

        if epoch >= self.T0 * (self.T_mult ** self.n_restarts):
            self.n_restarts += 1
            self.last_restart = epoch

            if epoch >= 30:
                self.warmup_amount = 0.1 * self._init_warmup_amount
            elif epoch >= 60:
                self.warmup_amount = 0.01 * self._init_warmup_amount

        return 1e-5 + (
            (self.warmup_amount - 1e-5)
            * (
                1
                + np.cos(
                    (epoch - self.last_restart)
                    / (self.T0 * (self.T_mult ** self.n_restarts))
                    * np.pi
                )
            )
            / 2.0
        )


class Model(nn.Module):
    def __init__(self, backbone="se_resneXt50"):
        super().__init__()
        resnet_baseplanes = 32
        encoder = ResNetEncoder(
            3,
            resnet_baseplanes,
            resnet_baseplanes // 2,
            128,
            make_backbone=getattr(nav_analysis.rl.resnet, backbone),
        )
        self.cnn = nn.Sequential(encoder)

        self.running_mean_and_var = RunningMeanAndVar(3)

        self.classifier = nn.Linear(np.prod(encoder.output_size), 1000)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = self.running_mean_and_var(x)
        x = self.cnn(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


def _run_batch(model, batch, stats, optim=None, lr_sched=None):
    x, y = batch

    logits = model(x)
    logits = logits.float()
    loss = F.cross_entropy(logits, y)

    if optim is not None:
        optim.zero_grad()
        with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()

        optim.step()

        lr_sched.step()

    with torch.no_grad():
        stats[0] += loss * y.size(0)
        stats[1] += (torch.argmax(logits, -1) == y).float().sum()
        stats[2] += y.size(0)


def train_epoch(model, optim, lr_sched, loader, epoch):
    device = next(model.parameters()).device
    model.train()

    train_stats = torch.zeros(3, device=device)
    for batch in loader:
        batch = tuple(v.to(device) for v in batch)
        _run_batch(model, batch, train_stats, optim, lr_sched)

    distrib.all_reduce(train_stats)

    if distrib.get_rank() == 0:
        print("\n{0} Train -- Epoch {1:3d} {0}\n".format("=" * 5, epoch))
        print(
            "Loss: {:.3f}   Acc: {:.3f}".format(
                (train_stats[0] / train_stats[2]).item(),
                (train_stats[1] / train_stats[2]).item(),
            )
        )


best_eval_acc = 0.0


def eval_epoch(model, val_loader, epoch):
    global best_eval_acc
    device = next(model.parameters()).device

    def _run_eval(model, loader):
        stats = torch.zeros(3, device=device)
        with torch.no_grad():
            for batch in loader:
                batch = tuple(v.to(device) for v in batch)
                _run_batch(model, batch, stats)

        distrib.all_reduce(stats)

        return stats

    val_stats = _run_eval(model, val_loader)
    if distrib.get_rank() == 0:
        print("\n{0} Val   -- Epoch {1:3d} {0}\n".format("=" * 5, epoch))
        print(
            "Loss: {:.3f}   Acc: {:.3f}".format(
                (val_stats[0] / val_stats[2]).item(),
                (val_stats[1] / val_stats[2]).item(),
            )
        )

    if distrib.get_rank() == 0:
        eval_acc = (val_stats[0] / val_stats[2]).item()
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            state = model.module.state_dict(prefix="actor_critic.net.")
            state = {k: v.float() for k, v in state.items() if "classifier" not in k}
            torch.save(state, "./data/checkpoints/se_resneXt50_imagenet.pth")


def main():
    BATCH_SIZE = 128
    LR = 0.01
    EPOCHS = 103

    local_rank, _ = init_distrib_slurm()

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    train_set = torchvision.datasets.ImageFolder(
        "/datasets01_101/imagenet_full_size/061417/train",
        transform=T.Compose(
            [T.RandomResizedCrop(256), T.RandomHorizontalFlip(), T.ToTensor()]
        ),
    )
    val_transforms = T.Compose([T.CenterCrop(256), T.ToTensor()])
    val_set = torchvision.datasets.ImageFolder(
        "/datasets01_101/imagenet_full_size/061417/val", transform=val_transforms
    )

    loaders = {
        k: torch.utils.data.DataLoader(
            dataset,
            BATCH_SIZE,
            sampler=torch.utils.data.DistributedSampler(dataset),
            drop_last=k == "train",
            pin_memory=True,
            num_workers=8,
        )
        for k, dataset in zip(["train", "val"], [train_set, val_set])
    }

    model = Model()
    model.to(device)

    optim = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=0.9, nesterov=True, weight_decay=1e-4
    )

    model, optim = amp.initialize(model, optim, opt_level="O2", enabled=False)

    lr_lambda = WarmUpSGDR(len(loaders["train"]), 40.0)

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    model = torch.nn.parallel.DistributedDataParallel(model, [device], device)

    for epoch in range(EPOCHS):
        for v in loaders.values():
            v.sampler.set_epoch(epoch)

        train_epoch(model, optim, lr_sched, loaders["train"], epoch)
        eval_epoch(model, loaders["val"], epoch)


if __name__ == "__main__":
    main()
