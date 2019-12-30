import time

import torch
import torch.distributed as distrib
import torch.optim

from .gossiper import Gossiper


def reduce_list(lst):
    all_eles = torch.cat([ele.view(-1) for ele in lst])
    distrib.all_reduce(all_eles)
    all_eles /= distrib.get_world_size()

    ptr = 0
    reduced_lst = []
    for ele in lst:
        reduced_lst.append(all_eles[ptr : ptr + ele.numel()].view_as(ele))
        ptr += ele.numel()

    return reduced_lst


class SlowMoGossipOptim(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        base_optim: torch.optim.Adam,
        lr=1e-4,
        slow_lr=1,
        beta=0.4,
        tau=40,
        gossip_freq=1,
    ):
        defaults = dict(lr=lr, slow_lr=slow_lr, beta=beta)
        super().__init__(params, defaults)

        self.base_optim = base_optim

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]

                state["slow_mo"] = torch.zeros_like(p.data, requires_grad=False)
                state["x"] = p.data.clone()

        self.world_size = distrib.get_world_size()
        self.step_num = 0
        self.tau = tau
        self.gossip_freq = gossip_freq

        self.gossiper = None

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["base_optim"] = self.base_optim.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        self.base_optim.load_state_dict(state_dict["base_optim"])
        del state_dict["base_optim"]
        super().load_state_dict(state_dict)

    def gossip_tensor_list(self):
        tensors = [p.data for group in self.param_groups for p in group["params"]]
        all_params = list(self.base_optim.state.keys())
        for state_key in self.base_optim.state[all_params[0]]:
            if state_key != "step":
                tensors += [self.base_optim.state[p][state_key] for p in all_params]

        return tensors

    def gossip(self):
        if self.gossiper is None:
            self.gossiper = Gossiper(self.gossip_tensor_list())

        self.gossiper.begin_gossip(self.gossip_tensor_list())

    def finish_gossip(self):
        if self.gossiper is None:
            return

        self.gossiper.finish_gossip(self.gossip_tensor_list())

    def step(self, closure=None):
        self.step_num += 1

        ret = self.base_optim.step(closure)

        if (self.step_num % self.gossip_freq) == 0 and (self.step_num % self.tau) != 0:
            self.finish_gossip()
            self.gossip()

        if (self.step_num % self.tau) == 0:
            self.finish_gossip()
            for group in self.param_groups:
                inv_lr = 1.0 / group["lr"]
                reduced_params = reduce_list([p.data for p in group["params"]])

                for i, p in enumerate(group["params"]):
                    state = self.state[p]

                    ascent_dir = reduced_params[i]
                    ascent_dir.add_(-1.0, state["x"])
                    state["slow_mo"].mul_(group["beta"])
                    state["slow_mo"].add_(-inv_lr, ascent_dir)
                    state["x"].add_(-group["lr"] * group["slow_lr"], state["slow_mo"])

                    p.data.copy_(state["x"])

            all_params = list(self.base_optim.state.keys())
            for state_key in self.base_optim.state[all_params[0]]:
                if state_key != "step":
                    reduced = reduce_list(
                        [self.base_optim.state[p][state_key] for p in all_params]
                    )

                    for i, p in enumerate(all_params):
                        self.base_optim.state[p][state_key] = reduced[i]

        return ret
