import torch
import torch.distributed as distrib
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def update_mean_var_count_from_moments(
    mean, var, count, all_batch_mean, all_batch_var, all_batch_count
):
    T = all_batch_mean.size(0)

    for t in range(T):
        batch_mean = all_batch_mean[t]
        batch_var = all_batch_var[t]
        batch_count = all_batch_count[t]

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
    def __init__(self, shape, use_distrib=True, always_training=False):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, *shape))
        self.register_buffer("_var", torch.full((1, *shape), 0.0))
        self.register_buffer("_count", torch.full((), 0.0))
        self._shape = shape

        self._distributed = distrib.is_initialized() if use_distrib else False
        self._always_training = always_training

    def update(self, x):
        if self.training or self._always_training:
            if x.ndim == 4:
                batch_mean = F.adaptive_avg_pool2d(x, 1).mean(0, keepdim=True)
                batch_count = torch.full_like(
                    self._count, x.size(0) * x.size(2) * x.size(3)
                )
                batch_var = F.adaptive_avg_pool2d((x - batch_mean).pow(2), 1).mean(
                    0, keepdim=True
                )
            elif x.ndim == 3:
                batch_mean = F.adaptive_avg_pool1d(x, 1).mean(0, keepdim=True)
                batch_count = torch.full_like(self._count, x.size(0) * x.size(2))
                batch_var = F.adaptive_avg_pool1d((x - batch_mean).pow(2), 1).mean(
                    0, keepdim=True
                )
            else:
                batch_mean = x.mean(0, keepdim=True)
                batch_count = torch.full_like(self._count, x.size(0))
                batch_var = (x - batch_mean).pow(2).mean(0, keepdim=True)

            self._update_from_moments(batch_mean, batch_var, batch_count)

    def forward(self, x):
        self.update(x)

        # Note this is a hack to deal with an issue of me not dividing by 255
        # for RGB observations on a couple runs
        if self._mean.mean() > 1.0 and x.mean() < 1.0:
            x = x * 255.0
        elif self._mean.mean() < 1.0 and x.mean() > 1.0:
            x = x / 255.0

        return (x - self._mean) / self.stdev

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        if self._distributed:
            world_size = distrib.get_world_size()

            def _gather(t):
                all_t = [torch.empty_like(t) for _ in range(world_size)]
                distrib.all_gather(all_t, t)
                return torch.stack(all_t, 0)

            batch_mean = _gather(batch_mean)
            batch_count = _gather(batch_count)
            batch_var = _gather(batch_var)
        else:
            batch_mean = batch_mean.unsqueeze(0)
            batch_var = batch_var.unsqueeze(0)
            batch_count = batch_count.unsqueeze(0)

        self._mean, self._var, self._count = update_mean_var_count_from_moments(
            self._mean, self._var, self._count, batch_mean, batch_var, batch_count
        )

    @property
    def stdev(self):
        return torch.sqrt(torch.max(self._var, torch.full_like(self._var, 1e-2)))

    def sync(self):
        batch_mean = self._mean
        batch_var = self._var
        batch_count = self._count

        self._mean = torch.zeros_like(self._mean)
        self._var = torch.zeros_like(self._var)
        self._count = torch.zeros_like(self._count)

        self._update_from_moments(batch_mean, batch_var, batch_count)
