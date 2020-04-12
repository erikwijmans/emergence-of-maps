import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta.pow(2) * count * batch_count / tot_count

    mean = mean + delta * batch_count / tot_count
    var = M2 / tot_count
    count = tot_count

    return mean, var, count


class RunningMeanAndVar(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, *shape))
        self.register_buffer("_var", torch.zeros(1, *shape))
        self.register_buffer("_count", torch.full((), 0.0))
        self._shape = shape

        self._x_buffer = []

    def update(self, x):
        if distrib.is_initialized():
            self._x_buffer.append(x.clone())
        else:
            self._update(x)

    def _update(self, x):
        batch_mean = x.sum(0, keepdim=True)
        batch_count = torch.full_like(self._count, x.size(0))

        if distrib.is_initialized():
            shape = batch_mean.size()
            n = np.prod(shape)
            tmp = torch.cat([batch_mean.view(-1), batch_count.view(-1)])
            distrib.all_reduce(tmp)

            batch_mean = tmp[0:n].view(shape)
            batch_count = tmp[n]

        batch_mean = batch_mean / batch_count

        batch_var = (x - batch_mean).pow(2).sum(0, keepdim=True)
        if distrib.is_initialized():
            distrib.all_reduce(batch_var)

        batch_var = batch_var / batch_count

        self._mean, self._var, self._count = update_mean_var_count_from_moments(
            self._mean, self._var, self._count, batch_mean, batch_var, batch_count
        )

    @property
    def stdev(self):
        return torch.sqrt(
            torch.max(
                self._var,
                torch.tensor(1e-4, device=self._var.device, dtype=self._var.dtype),
            )
        )

    def sync(self):
        if distrib.is_initialized():
            self._update(torch.cat(self._x_buffer, 0))

            self._x_buffer = []


class ImageAutoRunningMeanAndVar(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("_var", torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("_count", torch.full((), 0.0))

    def update(self, x):
        if self.training:
            batch_mean = F.adaptive_avg_pool2d(x, 1).mean(0, keepdim=True)
            batch_count = torch.full_like(
                self._count, x.size(0) * x.size(2) * x.size(3)
            )

            if distrib.is_initialized():
                shape = batch_mean.size()
                n = np.prod(shape)
                tmp = torch.cat([batch_mean.view(-1), batch_count.view(-1)])
                distrib.all_reduce(tmp)

                batch_mean = tmp[0:n].view(shape) / distrib.get_world_size()
                batch_count = tmp[n]

            batch_var = F.adaptive_avg_pool2d((x - batch_mean).pow(2), 1).mean(
                0, keepdim=True
            )
            if distrib.is_initialized():
                distrib.all_reduce(batch_var)
                batch_var = batch_var / distrib.get_world_size()

            self._mean, self._var, self._count = update_mean_var_count_from_moments(
                self._mean, self._var, self._count, batch_mean, batch_var, batch_count
            )

    def forward(self, x):
        if x.size(1) == 3 and self._mean.mean() > 1.0 and x.mean() < 1.0:
            x = x * 255.0

        self.update(x)

        return (x - self._mean) / self.stdev

    @property
    def stdev(self):
        return torch.sqrt(
            torch.max(
                self._var,
                torch.tensor(1e-4, device=self._var.device, dtype=self._var.dtype),
            )
        )
