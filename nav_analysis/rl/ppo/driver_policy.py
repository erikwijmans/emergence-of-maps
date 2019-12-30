#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import nav_analysis.rl.resnet
from nav_analysis.rl.layer_norm_lstm import LayerNormLSTM
from nav_analysis.rl.ppo.policy import Net as PPONet, Policy
from nav_analysis.rl.ppo.utils import CategoricalNet, Flatten
from nav_analysis.rl.running_mean_and_var import RunningMeanAndVar


class DriverPolicy(nn.Module):
    def __init__(
        self,
        pointnav_agent: Policy,
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="LSTM",
    ):
        super().__init__()
        self.pointnav_agent = pointnav_agent

        self.net = Net(
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
        )

    def forward(self, *x):
        return None

    def _pack_hidden(self, net_hidden_states, pointnav_hidden_states):
        #  net_hidden_states = net_hidden_states.view(2, -1, 512)
        return torch.cat([net_hidden_states, pointnav_hidden_states], 0)

    def _unpack_hidden(self, hidden_states):
        net_hidden_states = hidden_states[0:4]
        #  net_hidden_states = net_hidden_states.view(4, -1, 256)
        pointnav_hidden_states = hidden_states[4:]

        return (net_hidden_states, pointnav_hidden_states)

    def act(
        self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False
    ):
        net_hidden_states, pointnav_hidden_states = self._unpack_hidden(
            rnn_hidden_states
        )

        value, goal_preds, net_hidden_states = self.net(
            observations, net_hidden_states, prev_actions, masks
        )
        observations["pointgoal"] = goal_preds

        _, action, action_log_probs, entropy, pointnav_hidden_states = self.pointnav_agent.act(
            observations, pointnav_hidden_states, prev_actions, masks, deterministic
        )

        return (
            value,
            action,
            action_log_probs,
            entropy,
            self._pack_hidden(net_hidden_states, pointnav_hidden_states),
        )

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        net_hidden_states, pointnav_hidden_states = self._unpack_hidden(
            rnn_hidden_states
        )

        value, goal_preds, net_hidden_states = self.net(
            observations, net_hidden_states, prev_actions, masks
        )
        return value

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        net_hidden_states, pointnav_hidden_states = self._unpack_hidden(
            rnn_hidden_states
        )

        value, goal_preds, net_hidden_states = self.net(
            observations, net_hidden_states, prev_actions, masks
        )
        observations["pointgoal"] = goal_preds

        _, action_log_probs, distribution_entropy, rnn_hidden_states, _ = self.pointnav_agent.evaluate_actions(
            observations, pointnav_hidden_states, prev_actions, masks, action
        )
        return (value, action_log_probs, distribution_entropy, rnn_hidden_states, None)


class Net(PPONet):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, hidden_size, num_recurrent_layers, rnn_type):
        nn.Module.__init__(self)

        self.pointnav_agent_action_embedding = nn.Embedding(5, 32)
        self._n_prev_action = 32

        self._hidden_size = hidden_size
        self.hidden_size = hidden_size

        self.feature_compress = nn.Sequential(
            Flatten(), nn.Linear(2048, hidden_size), nn.ReLU(True)
        )
        self.goal_predictor = nn.Linear(self.hidden_size, 3)
        self.critic_linear = nn.Linear(self.hidden_size, 1)

        self._rnn_type = rnn_type
        self._num_recurrent_layers = num_recurrent_layers
        if rnn_type == "LN-LSTM":
            self.rnn = LayerNormLSTM(
                hidden_size + self._n_prev_action,
                hidden_size,
                num_layers=num_recurrent_layers,
            )
        else:
            self.rnn = getattr(nn, rnn_type)(
                hidden_size + self._n_prev_action,
                hidden_size,
                num_layers=num_recurrent_layers,
            )

        self.layer_init()
        self.train()

    def layer_init(self):

        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        prev_actions = self.pointnav_agent_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(-1)
        )

        features = observations["features"]
        features = self.feature_compress(features)
        x = torch.cat([features, prev_actions], dim=1)
        x, rnn_hidden_states = self.forward_rnn(x, rnn_hidden_states, masks)

        goal_preds = self.goal_predictor(x)
        r = F.elu(goal_preds[:, 0:1]) + 0.75
        xy = torch.tanh(goal_preds[:, 1:])
        goal_preds = torch.cat([r, xy], dim=-1)

        return self.critic_linear(x), goal_preds, rnn_hidden_states
