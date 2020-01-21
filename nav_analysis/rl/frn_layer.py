import torch
import torch.nn as nn
from torch.jit import Final

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class TLUFunctional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tau, inplace):
        if inplace:
            ctx.mark_dirty(x)
            output = x
        else:
            output = x.clone()

        torch.max(x, tau, out=output)

        ctx.save_for_backward(x, tau)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        x_grad = grad_out.clone()
        tau_grad = grad_out.clone()

        x, tau = ctx.saved_tensors

        x_grad[x <= tau] = 0
        tau_grad[x > tau] = 0

        return x_grad, tau_grad.sum(0), None


tlu_functional = TLUFunctional.apply


class TLU(nn.Module):
    _inplace: Final[bool]

    def __init__(self, nchannel: int, inplace: bool = True):
        super().__init__()
        self.tau = nn.Parameter(torch.zeros(nchannel, 1))
        self._inplace = inplace

    def forward(self, x):
        orig_size = x.size()
        x = x.view(orig_size[0], orig_size[1], -1)

        res = tlu_functional(x, self.tau, self._inplace)

        return res.view(orig_size)


class FRNLayer(nn.Module):
    _affine: Final[bool]
    _learnable_eps: Final[bool]
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
        orig_size = x.size()
        x = x.view(orig_size[0], orig_size[1], -1)
        nu2 = x.pow(2).mean(dim=2, keepdim=True)

        eps = self.eps
        if self._learnable_eps:
            eps = eps + self.eps_l.abs()

        x = x * torch.rsqrt(nu2 + eps)

        if self._affine:
            x = torch.addcmul(self.beta, self.gamma, x)

        if self._tlu:
            x = self.tlu_act(x)

        return x.view(orig_size)
