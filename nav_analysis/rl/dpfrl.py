from typing import List

import numpy as np
import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F


class PFJitGRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.ih = nn.Linear(input_size, 4 * hidden_size)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tensor
        x_results = self.ih(x)
        h_results = self.hh(hidden)

        x_results = x_results.squeeze()
        h_results = h_results.squeeze()

        i_r, i_z, i_n, i_s = x_results.chunk(4, 1)
        h_r, h_z, h_n, h_s = h_results.chunk(4, 1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        s = i_s + h_s
        n = torch.tanh(i_n + r * h_n) + torch.randn_like(s) * s

        return n - torch.mul(n, z) + torch.mul(z, hidden)


class DPFRL(jit.ScriptModule):
    __constants__ = ["K", "hidden_size"]

    def __init__(self, input_size=512, hidden_size=512, K=16, alpha=0.5):
        super().__init__()

        self.f_trans = PFJitGRUCell(input_size, hidden_size)

        self.f_obs = nn.Linear(input_size + hidden_size, 1)
        self.v = nn.Linear(hidden_size, hidden_size)

        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer(
            "inv_sqrt_hidden_size", torch.tensor(1.0 / np.sqrt(hidden_size))
        )
        self.K = K
        self.hidden_size = hidden_size

    @jit.script_method
    def forward(self, obs, particles):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        T = obs.size(0)
        N = obs.size(1)

        h, w = particles[..., : self.hidden_size], particles[..., -1]

        w = torch.where(
            w.sum(0, keepdim=True) < 1e-5, torch.full_like(w, 1 / self.K), w
        )

        results = []
        for t in range(T):
            o = obs[t].view(1, N, -1).repeat(self.K, 1, 1).view(self.K * N, -1)

            h = h.view(self.K * N, -1)
            h = self.f_trans(o, h)

            p_obs = self.f_obs(torch.cat([h, o], -1))
            p_obs = F.softmax(p_obs.view(self.K, N), 0)
            w = p_obs * w
            w = w / torch.sum(w, 0, keepdim=True)

            w = w / (self.alpha * w + (1 - self.alpha) / self.K)
            resamples = torch.multinomial(w.t(), self.K, replacement=True).t()

            w = torch.gather(w, dim=0, index=resamples)
            w = w / torch.sum(w, 0, keepdim=True)

            h = h.view(self.K, N, -1)
            h = torch.gather(h, dim=0, index=resamples.view(self.K, N, 1).expand_as(h))

            mean_particle = (w.view(self.K, N, 1) * h).sum(0)

            mgf = (
                w.view(self.K, N, 1)
                * torch.exp(
                    self.v(h.view(self.K * N, -1)).view(self.K, N, -1)
                    * self.inv_sqrt_hidden_size
                )
            ).sum(0)

            results.append(torch.cat([mean_particle, mgf], -1))

        particles = torch.cat([h, w.unsqueeze(-1)], -1)

        results = torch.stack(results, 0)
        return results, particles
