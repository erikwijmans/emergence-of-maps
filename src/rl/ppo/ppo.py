#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import numpy as np
from distutils.version import StrictVersion


EPS_PPO = 1e-3


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        normalized_advantage=False,
    ):

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.device = next(self.actor_critic.parameters()).device
        self.normalized_advantage = normalized_advantage
        self.reducer = None

    def init_distributed(self):
        class Gaurd(object):
            def __init__(self, module, device):
                # In pytorch 1.0,
                # DDP's hooks for backwards work regardless of if we actually use it for the forward pass,
                # so we can just init them and they will do our bidding!
                # This allows us to leverage all the work done so backdrop and all_reduce can happen
                # simultaneously
                self.ddp = torch.nn.parallel.DistributedDataParallel(
                    module, device_ids=[device], output_device=device
                )

        self.ddp = Gaurd(self.actor_critic, self.device)
        self.all_advantages = None
        self.world_size = dist.get_world_size()

        self.get_advantages = self._get_advantages_distributed

        torch_version = StrictVersion(torch.__version__)
        assert torch_version >= StrictVersion("1.0")

        if torch_version >= StrictVersion("1.1"):
            self.reducer = self.ddp.ddp.reducer

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

        if not self.normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def _get_advantages_distributed(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

        if not self.normalized_advantage:
            return advantages

        if self.all_advantages is None:
            self.all_advantages = [
                advantages.clone() for _ in range(self.world_size)
            ]

        # Gather all the advantages across all the rollouts so we can compute mean and std
        dist.all_gather(self.all_advantages, advantages)
        gathered_adv = torch.cat(self.all_advantages)

        return (advantages - gathered_adv.mean()) / (
            gathered_adv.std() + EPS_PPO
        )

    def update(self, rollouts):
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                    aux_preds,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                aux_loss = 0.0
                if aux_preds is not None:
                    (egomotion_preds, gps_preds, delta_pos_preds) = aux_preds
                    n = recurrent_hidden_states_batch.size(1)
                    t = int(masks_batch.size(0) / n)

                    egomotion_loss = F.cross_entropy(
                        egomotion_preds,
                        actions_batch.view(t, n)[:-1].view(-1),
                        reduction="none",
                    )
                    egomotion_loss = (
                        0.01
                        * (
                            masks_batch.view(t, n)[:-1].view(-1)
                            * egomotion_loss
                        ).sum()
                        / max((masks_batch.view(t, n)[:-1]).sum().item(), 32.0)
                    )

                    gps_gt = obs_batch["gps"]
                    gps_loss = F.smooth_l1_loss(gps_preds, gps_gt)

                    pos_gt = obs_batch["pos"].view(t, n, -1)
                    delta_pos_gt = pos_gt[1:] - pos_gt[:-1]
                    delta_pos_gt = torch.cat(
                        [
                            delta_pos_gt[..., 0:2].norm(dim=-1, keepdim=True),
                            delta_pos_gt[..., 2:],
                        ],
                        -1,
                    )
                    delta_pos_preds = delta_pos_preds.view(t, n, -1)
                    delta_pos_loss = F.smooth_l1_loss(
                        delta_pos_preds[1:], delta_pos_gt, reduction="none"
                    ).mean(-1)

                    delta_pos_loss = (
                        masks_batch.view(t, n)[1:] * delta_pos_loss
                    ).sum() / max(
                        masks_batch.view(t, n)[1:].sum().item(), 32.0
                    )

                    aux_loss = egomotion_loss + gps_loss + delta_pos_loss

                self.optimizer.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                    + aux_loss
                )

                if self.reducer is not None:
                    find_unused_params = False
                    if find_unused_params:
                        self.reducer.prepare_for_backward([total_loss])
                    else:
                        self.reducer.prepare_for_backward([])

                total_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
