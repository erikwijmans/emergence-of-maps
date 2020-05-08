#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nav_analysis.rl.efficient_net
import nav_analysis.rl.resnet
from habitat.sims.habitat_simulator import SimulatorActions
from nav_analysis.rl.dpfrl import DPFRL
from nav_analysis.rl.frn_layer import FRNLayer
from nav_analysis.rl.layer_norm_lstm import LayerNormLSTM
from nav_analysis.rl.ppo.utils import CategoricalNet, Flatten
from nav_analysis.rl.running_mean_and_var import ImageAutoRunningMeanAndVar
from nav_analysis.rl.ppo.policy import Policy, ResNetEncoder, Net


def gather_last_from_packed_seq(
    x: torch.nn.utils.rnn.PackedSequence, lengths: torch.Tensor,
) -> torch.Tensor:
    r"""Takes a packed sequence returns the last element in each sequence contained.

    When working with rnns and padded seqs, it is nice to do the following

    .. code::python

        x = <padded_rnn_input>
        lengths = <length of each seq in x>

        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)

        x, _ = rnn(x)

        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x)

        x = torch.gather(x, dim=0, index=(lengths - 1).view(1, x.size(1), 1).expand_as(x[0:1])).squeeze(0)

    to produce an encoding of a sequence.  However, padding the packed sequence is expensive and memory inefficient,
    this method can be used instead!

    .. code::python

        x = <padded_rnn_input>
        lengths = <length of each seq in x>

        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)

        x, _ = rnn(x)

        x = gather_last_from_packed_seq(x, lengths)


    Not that for this method, batch_first=False and batch_first=True are identical!


    :param x: The packed sequence
    :param lengths: The length of each sequence

    :return: A tensor of size (lengths.size(0), x.data.size(1))
    """
    B = len(lengths)

    sorted_lengths = torch.index_select(lengths, dim=0, index=x.sorted_indices)

    offsets = torch.cumsum(
        torch.cat(
            (
                torch.zeros(
                    (1,), device=x.batch_sizes.device, dtype=x.batch_sizes.dtype
                ),
                x.batch_sizes,
            )
        ).to(device=x.data.device),
        dim=0,
    )

    gather_inds = torch.index_select(
        offsets, dim=0, index=sorted_lengths - 1
    ) + torch.arange(B, device=x.data.device, dtype=sorted_lengths.dtype)

    gather_inds = (
        torch.index_select(gather_inds, dim=0, index=x.unsorted_indices)
        .view(B, 1)
        .expand(B, x.data.size(1))
    )

    return torch.gather(x.data, dim=0, index=gather_inds)


class MemoryLimitedPolicy(Policy):
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
        max_memory_length=2,
    ):
        nn.Module.__init__(self)
        assert not two_headed
        self.dim_actions = action_space.n

        self.net = MemoryLimitedNet(
            observation_space=observation_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            blind=blind,
            rnn_type=rnn_type,
            backbone=backbone,
            resnet_baseplanes=resnet_baseplanes,
            norm_visual_inputs=norm_visual_inputs,
            task=task,
            max_memory_length=max_memory_length,
        )

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions, task=task
        )

        assert not blind or not use_aux_losses
        if use_aux_losses:
            self.representation_egomotion_head = nn.Linear(
                hidden_size, self.dim_actions
            )
            self.gps_head = nn.Linear(
                hidden_size, observation_space.spaces["gps"].shape[0]
            )
            self.delta_pos_head = nn.Linear(hidden_size, 3)

        self.use_aux_losses = use_aux_losses


class MemoryLimitedNet(Net):
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
        max_memory_length,
    ):
        super().__init__(
            observation_space,
            hidden_size,
            num_recurrent_layers,
            blind,
            rnn_type,
            backbone,
            resnet_baseplanes,
            norm_visual_inputs,
            task,
        )

        self._max_memory_length = max_memory_length

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        # Observations come in as a (B, ..., C, self._max_memory_length) tensor,
        # Put things as (self._max_memory_length * B, ..., C) to then make
        # but we want (self._max_memory_length, B, C) later
        for k, v in observations.items():
            dim_order = list(range(v.ndim))
            dim_order = dim_order[-1:] + dim_order[0:-1]

            v = v.permute(*dim_order)
            size = list(v.size())
            size = [size[0] * size[1]] + size[2:]
            v = v.reshape(*size)

            observations[k] = v

        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            rgb_observations = (self.grayscale_kernel * rgb_observations).sum(
                1, keepdim=True
            )

            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)

            cnn_input.append(depth_observations)

        assert not self._old_goal_format
        goal_observations = self.tgt_embed(observations["pointgoal"])

        prev_actions_emb = self.prev_action_embedding(
            observations["prev_action"].long().squeeze(-1)
        )

        x = []
        cnn_feats = None
        if len(cnn_input) > 0:
            cnn_input = torch.cat(cnn_input, dim=1)
            cnn_input = F.interpolate(
                cnn_input, size=(self._sq_size, self._sq_size), mode="area"
            )
            cnn_input = F.avg_pool2d(cnn_input, 2)
            if self._norm_inputs:
                cnn_input = self.running_mean_and_var(cnn_input)

            cnn_feats = self.cnn(cnn_input)
            x += [cnn_feats]

        x += [goal_observations, prev_actions_emb]
        if self.gps_compass_embed is not None:
            x += [self.gps_compass_embed(observations["gps_and_compass"])]

        if self.dist_to_goal_embed is not None:
            x += [self.dist_to_goal_embed(observations["dist_to_goal"])]

        x = torch.cat(x, dim=1)  # concatenate goal vector

        x = x.view(self._max_memory_length, x.size(0) // self._max_memory_length, -1)

        lengths = (
            (observations["prev_action"].view(x.size(0), x.size(1)) != 0).long().sum(0)
        )
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=False, enforce_sorted=False
        )

        x, _ = self.rnn(x)

        x = gather_last_from_packed_seq(x, lengths)

        if self._task == "pointnav":
            value = self.critic_linear(x)
        else:
            value = self.critic(x)

        return value, x, rnn_hidden_states, cnn_feats
