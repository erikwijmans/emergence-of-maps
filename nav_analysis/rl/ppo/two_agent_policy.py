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
from habitat.sims.habitat_simulator import SimulatorActions
from nav_analysis.rl.ppo.policy import Net, Policy


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
        stage_2_state_type="trained",
    ):
        super().__init__()
        assert not two_headed
        self.dim_actions = action_space.n

        policy_kwargs = dict(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            blind=blind,
            use_aux_losses=use_aux_losses,
            rnn_type=rnn_type,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            task=task,
            norm_visual_inputs=norm_visual_inputs,
        )

        self.agent1 = Policy(**policy_kwargs)

        self.agent2 = Policy(**policy_kwargs)

        if stage_2_state_type == "random":
            self.random_policy = Policy(**policy_kwargs)

        self._stage_2_state_type = stage_2_state_type

        class NetStub:
            def __init__(self, policy):
                self.num_recurrent_layers = policy.net.num_recurrent_layers * (
                    2 if stage_2_state_type == "random" else 1
                )
                self.output_size = policy.net.output_size
                self.hidden_size = hidden_size

        self.net = NetStub(self.agent1)

    def forward(self, *x):
        return None

    def build_hidden(self, rnn_hidden_states, prev_actions):
        if self._stage_2_state_type == "random":
            random_hidden_states = rnn_hidden_states[
                self.agent1.net.num_recurrent_layers :
            ].clone()
            rnn_hidden_states = rnn_hidden_states[
                0 : self.agent1.net.num_recurrent_layers
            ].clone()

        state2 = rnn_hidden_states.clone()

        if self._stage_2_state_type == "trained":
            pass
        elif self._stage_2_state_type == "random":
            state2 = torch.where(prev_actions == 3, random_hidden_states, state2)
        elif self._stage_2_state_type == "zero":
            state2 = torch.where(prev_actions == 3, torch.zeros_like(state2), state2)
        else:
            raise RuntimeError(
                f"Unknown stage_2_state_type: {self._stage_2_state_type}"
            )

        return rnn_hidden_states.contiguous(), state2.contiguous()

    def act(
        self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False
    ):
        assert rnn_hidden_states.size(1) > 0
        state1, state2 = self.build_hidden(rnn_hidden_states, prev_actions)

        a1 = self.agent1.act(observations, state1, prev_actions, masks, deterministic)
        a2 = self.agent2.act(observations, state2, prev_actions, masks, deterministic)

        stage = observations["episode_stage"]

        res = []
        for i in range(len(a1)):
            if i < len(a1) - 1:
                res.append(torch.where(stage == 0, a1[i], a2[i]))
            else:
                res.append(torch.where((stage == 0).view(1, -1, 1), a1[i], a2[i]))

        if self._stage_2_state_type == "random":
            res[-1] = torch.cat(
                [
                    res[-1],
                    self.random_policy.act(
                        observations,
                        rnn_hidden_states[self.agent1.net.num_recurrent_layers :],
                        prev_actions,
                        masks,
                        deterministic,
                    )[-1],
                ],
                dim=0,
            )

        return tuple(res)

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        value, _, _, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return value

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        raise NotImplementedError()
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
