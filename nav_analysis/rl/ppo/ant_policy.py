#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import nav_analysis.rl.efficient_net
import nav_analysis.rl.resnet
from habitat.sims.habitat_simulator import SimulatorActions
from nav_analysis.rl.dpfrl import DPFRL
from nav_analysis.rl.layer_norm_lstm import LayerNormLSTM
from nav_analysis.rl.ppo.utils import CategoricalNet, Flatten
from nav_analysis.rl.running_mean_and_var import ImageAutoRunningMeanAndVar


class AntPolicy(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=512,
        num_recurrent_layers=1,
        blind=0,
        use_aux_losses=True,
        rnn_type="GRU",
        resnet_baseplanes=32,
        backbone="resnet50",
        task="pointnav",
        norm_visual_inputs=False,
        two_headed=False,
    ):
        super().__init__()
        assert not two_headed
        self.dim_actions = action_space.n

        self.net = Net(
            observation_space=observation_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            blind=blind,
            rnn_type=rnn_type,
            backbone=backbone,
            resnet_baseplanes=resnet_baseplanes,
            norm_visual_inputs=norm_visual_inputs,
            task=task,
        )

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions, task=task
        )

        assert not use_aux_losses

    def forward(self, *x):
        return None

    def act(
        self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False
    ):
        value, actor_features, rnn_hidden_states, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(actor_features, observations)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return (
            value,
            action,
            action_log_probs,
            distribution.entropy(),
            rnn_hidden_states,
        )

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        value, _, _, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return value

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        value, actor_features, rnn_hidden_states, cnn_feats = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(actor_features, observations)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        aux_preds = None

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            aux_preds,
        )


class Net(nn.Module):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space,
        hidden_size,
        num_recurrent_layers,
        blind,
        rnn_type,
        backbone,
        resnet_baseplanes,
        norm_visual_inputs,
        task,
    ):
        super().__init__()
        self._task = task

        rnn_input_size = 0
        self.prev_action_embedding = nn.Embedding(5, 32)
        rnn_input_size += 32
        self.bump_embedding = nn.Embedding(2, 32)
        rnn_input_size += 32

        def _build_embed(inp_size):
            return nn.Linear(inp_size, 32)

        self._n_input_goal = observation_space.spaces["pointgoal"].shape[0]
        self.tgt_embed = _build_embed(self._n_input_goal)
        rnn_input_size += 32

        self.compass_embed = _build_embed(observation_space.spaces["compass"].shape[0])
        rnn_input_size += 32

        self.delta_gps_embed = _build_embed(
            observation_space.spaces["delta_gps"].shape[0]
        )
        rnn_input_size += 32

        self._hidden_size = hidden_size

        self._rnn_type = rnn_type
        self._num_recurrent_layers = num_recurrent_layers
        self.rnn = getattr(nn, rnn_type)(
            rnn_input_size, hidden_size, num_layers=num_recurrent_layers
        )

        self.critic_linear = nn.Linear(self.output_size, 1)

        self.layer_init()
        self.train()

    @property
    def output_size(self):
        return self._hidden_size * (2 if self._rnn_type == "DPFRL" else 1)

    @property
    def num_recurrent_layers(self):
        if self._rnn_type == "DPFRL":
            return self.rnn.K

        return self._num_recurrent_layers * (2 if "LSTM" in self._rnn_type else 1)

    def layer_init(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name and len(param.size()) >= 2:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def _pack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = torch.cat([hidden_states[0], hidden_states[1]], dim=0)

        return hidden_states

    def _unpack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = (
                hidden_states[0 : self._num_recurrent_layers],
                hidden_states[self._num_recurrent_layers :],
            )

        return hidden_states

    def _mask_hidden(self, hidden_states, masks):
        if isinstance(hidden_states, tuple):
            hidden_states = tuple(v * masks for v in hidden_states)
        else:
            hidden_states = masks * hidden_states

        return hidden_states

    def forward_rnn(self, x, hidden_states, masks, prev_actions):
        if x.size(0) == hidden_states.size(1):
            hidden_states = self._unpack_hidden(hidden_states)
            x, hidden_states = self.rnn(
                x.unsqueeze(0), self._mask_hidden(hidden_states, masks.unsqueeze(0))
            )
            x = x.squeeze(0)
        else:
            # x is a (T, N, -1) tensor flattened to (T * N, -1)
            n = hidden_states.size(1)
            t = int(x.size(0) / n)

            # unflatten
            x = x.view(t, n, x.size(1))
            masks = masks.view(t, n)
            not_stop_mask = (
                (prev_actions != SimulatorActions.STOP.value).float().view(t, n)
            )
            orig_masks = masks.clone()
            masks = masks * not_stop_mask

            # steps in sequence which have zero for any agent. Assume t=0 has
            # a zero in it.
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]  # handle scalar
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [t]

            masks = orig_masks

            hidden_states = self._unpack_hidden(hidden_states)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # process steps that don't have any zeros in masks together
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                if isinstance(hidden_states, tuple):
                    hidden_states = tuple(
                        torch.where(
                            not_stop_mask[start_idx].view(1, n, 1).bool(), v, v.detach()
                        )
                        for v in hidden_states
                    )
                else:
                    hidden_states = torch.where(
                        not_stop_mask[start_idx].view(1, n, 1).bool(),
                        hidden_states,
                        hidden_states.detach(),
                    )

                rnn_scores, hidden_states = self.rnn(
                    x[start_idx:end_idx],
                    self._mask_hidden(hidden_states, masks[start_idx].view(1, -1, 1)),
                )

                outputs.append(rnn_scores)

            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            x = x.view(t * n, -1)  # flatten

        hidden_states = self._pack_hidden(hidden_states)
        return x, hidden_states

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        rnn_input = []

        rnn_input.append(self.tgt_embed(observations["pointgoal"]))

        rnn_input.append(
            self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().squeeze(-1)
            )
        )

        rnn_input.append(self.bump_embedding(observations["bump"].long().squeeze(-1)))

        rnn_input.append(self.delta_gps_embed(observations["delta_gps"]))
        rnn_input.append(self.compass_embed(observations["compass"]))

        x = torch.cat(rnn_input, dim=1)

        x, rnn_hidden_states = self.forward_rnn(
            x, rnn_hidden_states, masks, prev_actions
        )

        value = self.critic_linear(x)

        return value, x, rnn_hidden_states, None
