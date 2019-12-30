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


class HRLPolicy(nn.Module):
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

        value, goal_preds, net_hidden_states, pg_masks, pg_model_steps = self.net(
            observations, net_hidden_states, prev_actions, masks
        )
        observations["pointgoal"] = goal_preds

        (
            _,
            action,
            action_log_probs,
            entropy,
            pointnav_hidden_states,
        ) = self.pointnav_agent.act(
            observations, pointnav_hidden_states, prev_actions, pg_masks, deterministic
        )

        return (
            value,
            action,
            action_log_probs,
            entropy,
            self._pack_hidden(net_hidden_states, pointnav_hidden_states),
            goal_preds,
            pg_masks,
            pg_model_steps,
        )

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        net_hidden_states, pointnav_hidden_states = self._unpack_hidden(
            rnn_hidden_states
        )

        return self.net(observations, net_hidden_states, prev_actions, masks)[0]

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        net_hidden_states, pointnav_hidden_states = self._unpack_hidden(
            rnn_hidden_states
        )

        value, goal_preds, net_hidden_states, pg_masks, _ = self.net(
            observations, net_hidden_states, prev_actions, masks
        )
        observations["pointgoal"] = goal_preds

        (
            _,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            _,
        ) = self.pointnav_agent.evaluate_actions(
            observations, pointnav_hidden_states, prev_actions, pg_masks, action
        )
        return (value, action_log_probs, distribution_entropy, rnn_hidden_states, None)


@torch.jit.script
def update_ego_motion(
    prev_pg, ego, goal_preds, masks, pg_model_steps, pg_masks, updated_goal_preds
):
    r = torch.norm(goal_preds, p=2, dim=-1, keepdim=True)
    xy = goal_preds / r
    goal_preds = torch.cat([r, xy], dim=-1)

    T = ego.size(0)
    for t in range(T):
        r = prev_pg[:, 0:1]
        xy = r * prev_pg[:, 1:]
        xy = torch.stack([xy[:, 1], -xy[:, 0]], -1)
        xy = torch.baddbmm(
            ego[t, :, :, 2:], ego[t, :, :, 0:2], xy.unsqueeze(-1)
        ).squeeze(-1)
        xy = torch.stack([-xy[:, 1], xy[:, 0]], -1)

        r = torch.norm(xy, p=2, dim=-1, keepdim=True)
        xy = xy / r

        new_pg = torch.cat([r, xy], -1)
        pg_masks[t] = (
            (new_pg[:, 0] < 0.25).float()
            * masks[t]
            * (pg_model_steps[t] == 0.0).float()
        )

        updated_goal_preds[t] = torch.where(
            pg_masks[t].byte().unsqueeze(-1), new_pg, goal_preds[t]
        )
        prev_pg = updated_goal_preds[t].clone()

    return pg_masks, updated_goal_preds


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
        self.goal_predictor = nn.Linear(self.hidden_size, 2)
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

        nn.init.constant_(self.goal_predictor.bias, 1.0 / np.sqrt(2))

    def _update_pg(self, observations, goal_preds, masks, N):
        prev_pg = observations["prev_pointgoal"]
        T = prev_pg.size(0) // N
        prev_pg = prev_pg.view(T, N, -1)
        ego = observations["ego_motion"]
        ego = ego.view(T, N, *ego.size()[1:])
        pg_model_steps = torch.fmod(observations["pg_model_steps"].view(T, N) + 1, 10)

        masks = masks.view(T, N)
        goal_preds = goal_preds.view(T, N, -1)
        pg_masks = torch.zeros(T, N, device=masks.device)
        updated_goal_preds = prev_pg.clone()
        prev_pg = prev_pg[0]
        pg_masks, updated_goal_preds = update_ego_motion(
            prev_pg,
            ego,
            goal_preds,
            masks,
            pg_model_steps,
            pg_masks,
            updated_goal_preds,
        )

        updated_goal_preds = updated_goal_preds.view(T * N, -1)
        pg_masks = pg_masks.view(T * N, 1)
        pg_model_steps = pg_model_steps.view(T * N, 1)

        return updated_goal_preds, pg_masks, pg_model_steps

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        prev_actions = self.pointnav_agent_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(-1)
        )

        features = observations["features"]
        features = self.feature_compress(features)
        x = torch.cat([features, prev_actions], dim=1)
        x, rnn_hidden_states = self.forward_rnn(x, rnn_hidden_states, masks)

        goal_preds = self.goal_predictor(x)

        goal_preds, pg_masks, pg_model_steps = self._update_pg(
            observations, goal_preds, masks, rnn_hidden_states.size(1)
        )

        return (
            self.critic_linear(x),
            goal_preds,
            rnn_hidden_states,
            pg_masks,
            pg_model_steps,
        )
