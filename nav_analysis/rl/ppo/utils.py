#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os.path as osp
from collections import defaultdict

import numpy as np
import omegaconf
import torch
import torch.nn as nn

import nav_analysis


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, task, two_headed):
        super().__init__()

        if task == "pointnav":
            self.linear = nn.Linear(num_inputs, num_outputs)
        else:
            self.forward_actor = nn.Sequential(
                nn.Linear(num_inputs, num_inputs // 2),
                nn.ReLU(True),
                nn.Linear(num_inputs // 2, num_outputs),
            )
            if two_headed:
                self.backward_actor = nn.Sequential(
                    nn.Linear(num_inputs, num_inputs // 2),
                    nn.ReLU(True),
                    nn.Linear(num_inputs // 2, num_inputs // 2),
                    nn.ReLU(True),
                    nn.Linear(num_inputs // 2, num_outputs),
                )
            self._two_headed = two_headed

        self._task = task

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x, obs):
        if self._task == "loopnav":
            if self._two_headed:
                logits = torch.where(
                    obs["episode_stage"].view(-1, 1) == 1,
                    self.backward_actor(x),
                    self.forward_actor(x),
                )
            else:
                logits = self.forward_actor(x)
        else:
            logits = self.linear(x)

        return CustomFixedCategorical(logits=logits)


def _flatten_helper(t, n, tensor):
    return tensor.view(t * n, *tensor.size()[2:])


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class RolloutStorage:
    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.entropy = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        self.masks = torch.ones(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)
        self.entropy = self.entropy.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        entropy,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.entropy[self.step].copy_(entropy.unsqueeze(-1))
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(self.observations[sensor][-1])

        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.prev_actions[0].copy_(self.prev_actions[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        max_ent_coeff = 0.0
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (
                    self.rewards[step]
                    + max_ent_coeff * self.entropy[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                    + max_ent_coeff * self.entropy[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][:-1, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind]
                )

                actions_batch.append(self.actions[:, ind])
                prev_actions_batch.append(self.prev_actions[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind]
                )

                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (num_layers, N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            )

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = _flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = _flatten_helper(T, N, actions_batch)
            prev_actions_batch = _flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                prev_actions_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )


def batch_obs(observations):
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(torch.from_numpy(np.array(obs[sensor])))

    for sensor in batch:
        batch[sensor] = torch.stack(batch[sensor]).float()
    return batch


def ppo_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--extra-confs", type=str, nargs="*")
    parser.add_argument("--opts", type=str, nargs="*")

    args = parser.parse_args()

    opts = omegaconf.OmegaConf.load(
        osp.join(
            osp.dirname(nav_analysis.__file__),
            "configs/experiments/base_conf.yaml",
        )
    )
    for conf_name in args.extra_confs if args.extra_confs is not None else []:
        opts.merge_with(
            omegaconf.OmegaConf.load(
                osp.join(osp.dirname(nav_analysis.__file__), conf_name)
            )
        )

    if args.opts is not None:
        opts.merge_with_dotlist(args.opts)

    return opts
