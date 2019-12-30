from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class TLU(nn.Module):
    def __init__(self, nchannel: int):
        super().__init__()
        self.tau = nn.Parameter(torch.zeros(nchannel, 1))

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(orig_shape[0], orig_shape[1], -1)

        return torch.max(x, self.tau).view(orig_shape)


class FRNLayer(nn.Module):
    _affine: Final[bool]
    learnable_eps: Final[bool]
    nchannel: Final[int]
    _tlu: Final[bool]

    def __init__(
        self,
        nchannel: int,
        affine: bool = True,
        eps: float = 1e-6,
        learnable_eps: bool = True,
        tlu: bool = True,
    ):
        super().__init__()

        self._learnable_eps = learnable_eps
        if learnable_eps:
            self.eps_l = nn.Parameter(torch.zeros(nchannel, 1))

        self._affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(nchannel, 1))
            self.beta = nn.Parameter(torch.zeros(nchannel, 1))

        self.register_buffer("eps", torch.tensor(eps))

        self._tlu = tlu
        if self._tlu:
            self.tlu_act = TLU(nchannel)

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(orig_shape[0], orig_shape[1], -1)
        nu2 = x.pow(2).mean(dim=2, keepdim=True)

        eps = self.eps
        if self._learnable_eps:
            eps = eps + self.eps_l.abs()

        x = x * torch.rsqrt(nu2 + eps)

        if self._affine:
            x = torch.addcmul(self.beta, self.gamma, x)

        if self._tlu:
            x = self.tlu_act(x)

        return x.view(orig_shape)
