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
from nav_analysis.rl.ppo.utils import CustomFixedCategorical, Flatten
from nav_analysis.rl.running_mean_and_var import ImageAutoRunningMeanAndVar


class TwoAgentCategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.forward_actor = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_inputs, num_inputs // 2),
                    nn.ReLU(True),
                    nn.Linear(num_inputs // 2, num_outputs),
                ),
                nn.Sequential(
                    nn.Linear(num_inputs, num_inputs // 2),
                    nn.ReLU(True),
                    nn.Linear(num_inputs // 2, num_outputs),
                ),
            ]
        )

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x, obs):
        stage = obs["episode_stage"]
        logits = torch.where(
            stage == 0, self.forward_actor[0](x), self.forward_actor[1](x)
        )

        return CustomFixedCategorical(logits=logits)

    def sync_params(self):
        self.actor[1].load_state_dict(self.actor[0].state_dict())


class TwoAgentPolicy(nn.Module):
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
        share_grad=False,
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
            share_grad=share_grad,
        )

        self.action_distribution = TwoAgentCategoricalNet(
            self.net.output_size, self.dim_actions
        )

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

    def sync_params(self):
        self.net.sync_params()
        self.action_distribution.sync_params()


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
        share_grad,
    ):
        super().__init__()
        self._task = task
        self._share_grad = share_grad

        self.prev_action_embedding = nn.ModuleList(
            [nn.Embedding(5, 32), nn.Embedding(5, 32)]
        )
        self._n_prev_action = 32

        self._n_input_goal = observation_space.spaces["pointgoal"].shape[0]
        self._old_goal_format = False
        if self._old_goal_format:
            self._n_input_goal -= 1

        self.tgt_embed = nn.ModuleList(
            [nn.Linear(self._n_input_goal, 32), nn.Linear(self._n_input_goal, 32)]
        )
        self._n_input_goal = 32

        self.gps_compass_embed = nn.ModuleList(
            [
                nn.Linear(observation_space.spaces["gps_and_compass"].shape[0], 32),
                nn.Linear(observation_space.spaces["gps_and_compass"].shape[0], 32),
            ]
        )
        self._n_input_goal += 32

        self.dist_to_goal_embed = nn.ModuleList(
            [
                nn.Linear(observation_space.spaces["dist_to_goal"].shape[0], 32),
                nn.Linear(observation_space.spaces["dist_to_goal"].shape[0], 32),
            ]
        )
        self._n_input_goal += 32

        assert "episode_stage" in observation_space.spaces

        self._hidden_size = hidden_size

        rnn_input_size = self._n_input_goal + self._n_prev_action

        self._rnn_type = rnn_type
        self._num_recurrent_layers = num_recurrent_layers
        self.rnn = nn.ModuleList(
            [
                getattr(nn, rnn_type)(
                    rnn_input_size, hidden_size, num_layers=num_recurrent_layers
                ),
                getattr(nn, rnn_type)(
                    rnn_input_size, hidden_size, num_layers=num_recurrent_layers
                ),
            ]
        )

        self.critic = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(True),
                    nn.Linear(hidden_size // 2, 1),
                ),
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(True),
                    nn.Linear(hidden_size // 2, 1),
                ),
            ]
        )

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
        if False and self.cnn is not None:
            for layer in self.cnn.modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(
                        layer.weight, nn.init.calculate_gain("relu")
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)

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

    def forward_rnn(self, x, hidden_states, masks, prev_actions, stage):
        if x.size(0) == hidden_states.size(1):
            hidden_states = self._unpack_hidden(hidden_states)
            init_hidden = self._mask_hidden(hidden_states, masks.unsqueeze(0))
            x_0, hidden_states_0 = self.rnn[0](x.unsqueeze(0), init_hidden)
            x_1, hidden_states_1 = self.rnn[1](x.unsqueeze(0), init_hidden)

            hidden_states = torch.where(
                stage.view(1, -1, 1) == 0,
                self._pack_hidden(hidden_states_0),
                self._pack_hidden(hidden_states_1),
            )

            x = torch.where(stage == 0, x_0.squeeze(0), x_1.squeeze(0))
        else:
            # x is a (T, N, -1) tensor flattened to (T * N, -1)
            n = hidden_states.size(1)
            t = int(x.size(0) / n)

            # unflatten
            stage = stage.view(t, n)
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

                if not self._share_grad:
                    if isinstance(hidden_states, tuple):
                        hidden_states = tuple(
                            torch.where(
                                not_stop_mask[start_idx].view(1, n, 1).bool(),
                                v,
                                v.detach(),
                            )
                            for v in hidden_states
                        )
                    else:
                        hidden_states = torch.where(
                            not_stop_mask[start_idx].view(1, n, 1).bool(),
                            hidden_states,
                            hidden_states.detach(),
                        )

                init_hidden = self._mask_hidden(
                    hidden_states, masks[start_idx].view(1, -1, 1)
                )

                rnn_scores_0, hidden_states_0 = self.rnn[0](
                    x[start_idx:end_idx], init_hidden
                )

                rnn_scores_1, hidden_states_1 = self.rnn[1](
                    x[start_idx:end_idx], init_hidden
                )

                rnn_scores = torch.where(
                    stage[start_idx:end_idx].view(-1, n, 1) == 0,
                    rnn_scores_0,
                    rnn_scores_1,
                )
                hidden_states = torch.where(
                    stage[end_idx - 1].view(1, -1, 1) == 0,
                    self._pack_hidden(hidden_states_0),
                    self._pack_hidden(hidden_states_1),
                )
                hidden_states = self._unpack_hidden(hidden_states)

                outputs.append(rnn_scores)

            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            x = x.view(t * n, -1)  # flatten

            hidden_states = self._pack_hidden(hidden_states)

        return x, hidden_states

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        goal_observations = observations["pointgoal"]
        if self._old_goal_format:
            rho_obs = goal_observations[:, 0].clone()
            phi_obs = -torch.atan2(goal_observations[:, 2], goal_observations[:, 1])
            goal_observations = torch.stack([rho_obs, phi_obs], -1)

        stage = observations["episode_stage"]
        goal_observations = torch.where(
            stage == 0,
            self.tgt_embed[0](goal_observations),
            self.tgt_embed[1](goal_observations),
        )
        prev_actions_emb = torch.where(
            stage == 0,
            self.prev_action_embedding[0](
                ((prev_actions.float() + 1) * masks).long().squeeze(-1)
            ),
            self.prev_action_embedding[1](
                ((prev_actions.float() + 1) * masks).long().squeeze(-1)
            ),
        )

        x = [
            goal_observations,
            prev_actions_emb,
            torch.where(
                stage == 0,
                self.gps_compass_embed[0](observations["gps_and_compass"]),
                self.gps_compass_embed[1](observations["gps_and_compass"]),
            ),
            torch.where(
                stage == 0,
                self.dist_to_goal_embed[0](observations["dist_to_goal"]),
                self.dist_to_goal_embed[1](observations["dist_to_goal"]),
            ),
        ]

        x = torch.cat(x, dim=1)

        x, rnn_hidden_states = self.forward_rnn(
            x, rnn_hidden_states, masks, prev_actions, stage
        )

        value = torch.where(stage == 0, self.critic[0](x), self.critic[1](x))

        return value, x, rnn_hidden_states, None

    def sync_params(self):
        all_lists = [
            self.rnn,
            self.critic,
            self.prev_action_embedding,
            self.tgt_embed,
            self.gps_compass_embed,
        ]
        for mlist in all_lists:
            mlist[1].load_state_dict(mlist[0].state_dict())
