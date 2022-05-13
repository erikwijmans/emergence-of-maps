#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

import nav_analysis.rl.efficient_net
import nav_analysis.rl.resnet
from habitat.sims.habitat_simulator import SimulatorActions
from nav_analysis.rl.dpfrl import DPFRL
from nav_analysis.rl.frn_layer import FRNLayer
from nav_analysis.rl.layer_norm_lstm import LayerNormLSTM
from nav_analysis.rl.ppo.utils import CategoricalNet, Flatten
from nav_analysis.rl.running_mean_and_var import ImageAutoRunningMeanAndVar


def _invert_permutation(permutation: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(permutation)
    output.scatter_(
        0,
        permutation,
        torch.arange(0, permutation.numel(), device=permutation.device),
    )
    return output


def _build_pack_info_from_dones(
    dones: torch.Tensor,
    T: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Create the indexing info needed to make the PackedSequence
    based on the dones.

    PackedSequences are PyTorch's way of supporting a single RNN forward
    call where each input in the batch can have an arbitrary sequence length

    They work as follows: Given the sequences [c], [x, y, z], [a, b],
    we generate data [x, a, c, y, b, z] and batch_sizes [3, 2, 1].  The
    data is a flattened out version of the input sequences (the ordering in
    data is determined by sequence length).  batch_sizes tells you that
    for each index, how many sequences have a length of (index + 1) or greater.

    This method will generate the new index ordering such that you can
    construct the data for a PackedSequence from a (T*N, ...) tensor
    via x.index_select(0, select_inds)
    """
    dones = dones.view(T, -1)
    N = dones.size(1)

    rollout_boundaries = dones.clone().detach()
    # Force a rollout boundary for t=0.  We will use the
    # original dones for masking later, so this is fine
    # and simplifies logic considerably
    rollout_boundaries[0] = True
    rollout_boundaries = rollout_boundaries.nonzero(as_tuple=False)

    # The rollout_boundaries[:, 0]*N will make the episode_starts index into
    # the T*N flattened tensors
    episode_starts = rollout_boundaries[:, 0] * N + rollout_boundaries[:, 1]

    # We need to create a transposed start indexing so we can compute episode lengths
    # As if we make the starts index into a N*T tensor, then starts[1] - starts[0]
    # will compute the length of the 0th episode
    episode_starts_transposed = rollout_boundaries[:, 1] * T + rollout_boundaries[:, 0]
    # Need to sort so the above logic is correct
    episode_starts_transposed, sorted_indices = torch.sort(
        episode_starts_transposed, descending=False
    )

    # Calculate length of episode rollouts
    rollout_lengths = episode_starts_transposed[1:] - episode_starts_transposed[:-1]
    last_len = N * T - episode_starts_transposed[-1]
    rollout_lengths = torch.cat([rollout_lengths, last_len.unsqueeze(0)])
    # Undo the sort above
    rollout_lengths = rollout_lengths.index_select(
        0, _invert_permutation(sorted_indices)
    )

    # Resort in descending order of episode length
    lengths, sorted_indices = torch.sort(rollout_lengths, descending=True)

    # We will want these on the CPU for torch.unique_consecutive,
    # so move now.
    cpu_lengths = lengths.to(device="cpu", non_blocking=True)

    episode_starts = episode_starts.index_select(0, sorted_indices)
    select_inds = torch.empty((T * N), device=dones.device, dtype=torch.int64)

    max_length = int(cpu_lengths[0].item())
    # batch_sizes is *always* on the CPU
    batch_sizes = torch.empty((max_length,), device="cpu", dtype=torch.long)

    offset = 0
    prev_len = 0
    num_valid_for_length = lengths.size(0)

    unique_lengths = torch.unique_consecutive(cpu_lengths)
    # Iterate over all unique lengths in reverse as they sorted
    # in decreasing order
    for next_len in reversed(unique_lengths):
        valids = lengths[0:num_valid_for_length] > prev_len
        num_valid_for_length = int(valids.float().sum())

        batch_sizes[prev_len:next_len] = num_valid_for_length

        # Creates this array
        # [step * N + start for step in range(prev_len, next_len)
        #                   for start in episode_starts[0:num_valid_for_length]
        # * N because each step is seperated by N elements
        new_inds = (
            torch.arange(prev_len, next_len, device=episode_starts.device).view(
                next_len - prev_len, 1
            )
            * N
            + episode_starts[0:num_valid_for_length].view(1, num_valid_for_length)
        ).view(-1)

        select_inds[offset : offset + new_inds.numel()] = new_inds

        offset += new_inds.numel()

        prev_len = next_len

    # Make sure we have an index for all elements
    assert offset == T * N

    # This is used in conjunction with episode_starts to get
    # the RNN hidden states
    rnn_state_batch_inds = episode_starts % N
    # This indicates that a given episode is the last one
    # in that rollout.  In other words, there are N places
    # where this is True, and for each n, True indicates
    # that this episode is the last contiguous block of experience,
    # This is needed for getting the correct hidden states after
    # the RNN forward pass
    last_episode_in_batch_mask = ((episode_starts + (lengths - 1) * N) // N) == (T - 1)

    return (
        select_inds,
        batch_sizes,
        episode_starts,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
    )


def build_rnn_inputs(
    x: torch.Tensor, not_dones: torch.Tensor, rnn_states: torch.Tensor
) -> Tuple[PackedSequence, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Create a PackedSequence input for an RNN such that each
    set of steps that are part of the same episode are all part of
    a batch in the PackedSequence.

    Use the returned select_inds and build_rnn_out_from_seq to invert this.

    :param x: A (T * N, -1) tensor of the data to build the PackedSequence out of
    :param not_dones: A (T * N) tensor where not_dones[i] == False indicates an episode is done
    :param rnn_states: A (-1, N, -1) tensor of the rnn_hidden_states

    :return: tuple(x_seq, rnn_states, select_inds, rnn_state_batch_inds, last_episode_in_batch_mask)
        WHERE
        x_seq is the PackedSequence version of x to pass to the RNN

        rnn_states are the corresponding rnn state

        select_inds can be passed to build_rnn_out_from_seq to retrieve the
            RNN output

        rnn_state_batch_inds indicates which of the rollouts in the batch a hidden
            state came from/is for

        last_episode_in_batch_mask indicates if an episode is the last in that batch.
            There will be exactly N places where this is True

    """

    N = rnn_states.size(1)
    T = x.size(0) // N
    dones = torch.logical_not(not_dones)

    (
        select_inds,
        batch_sizes,
        episode_starts,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
    ) = _build_pack_info_from_dones(dones.detach().to(device="cpu"), T)

    select_inds = select_inds.to(device=x.device)
    episode_starts = episode_starts.to(device=x.device)
    rnn_state_batch_inds = rnn_state_batch_inds.to(device=x.device)
    last_episode_in_batch_mask = last_episode_in_batch_mask.to(device=x.device)

    x_seq = PackedSequence(x.index_select(0, select_inds), batch_sizes, None, None)

    # Just select the rnn_states by batch index, the masking bellow will set things
    # to zero in the correct locations
    rnn_states = rnn_states.index_select(1, rnn_state_batch_inds)
    # Now zero things out in the correct locations
    rnn_states = torch.where(
        not_dones.view(1, -1, 1).index_select(1, episode_starts),
        rnn_states,
        rnn_states.new_zeros(()),
    )

    return (
        x_seq,
        rnn_states,
        select_inds,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
    )


def build_rnn_out_from_seq(
    x_seq: PackedSequence,
    hidden_states,
    select_inds,
    rnn_state_batch_inds,
    last_episode_in_batch_mask,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Construct the output of the rnn from a packed sequence returned by
        forward propping an RNN on the packed sequence returned by :ref:`build_rnn_inputs`.

    :param x_seq: The packed sequence output from the rnn
    :param hidden_statess: The hidden states output from the rnn
    :param select_inds: Returned from :ref:`build_rnn_inputs`
    :param rnn_state_batch_inds: Returned from :ref:`build_rnn_inputs`
    :param last_episode_in_batch_mask: Returned from :ref:`build_rnn_inputs`
    :param N: The number of simulator instances in the batch of experience.
    """
    x = x_seq.data.index_select(0, _invert_permutation(select_inds))

    last_hidden_states = torch.masked_select(
        hidden_states,
        last_episode_in_batch_mask.view(1, hidden_states.size(1), 1),
    ).view(hidden_states.size(0), N, hidden_states.size(2))
    output_hidden_states = torch.empty_like(last_hidden_states)
    scatter_inds = (
        torch.masked_select(rnn_state_batch_inds, last_episode_in_batch_mask)
        .view(1, N, 1)
        .expand_as(output_hidden_states)
    )
    output_hidden_states.scatter_(1, scatter_inds, last_hidden_states)

    return x, output_hidden_states


class Policy(nn.Module):
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

    def _unit_circle(self, x):
        x = x.tanh()
        x = x / x.norm(dim=-1, keepdim=True)
        return x

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
        if self.use_aux_losses:
            n = rnn_hidden_states.size(1)
            t = int(masks.size(0) / n)
            cnn_feats = cnn_feats.view(t, n, -1)
            egomotion_preds = self.representation_egomotion_head(
                (cnn_feats[1:] - cnn_feats[:-1]).view((t - 1) * n, -1)
            )

            gps_preds = self.gps_head(actor_features)
            rho_preds = gps_preds[:, 0:1].clone()
            dir_preds = gps_preds[:, 1:].clone()

            dir_preds = self._unit_circle(dir_preds)
            gps_preds = torch.cat([rho_preds, dir_preds], -1)

            delta_pos_preds = self.delta_pos_head(actor_features)
            #  delta_xy_preds = delta_pos_preds[:, 0:1].clone()
            #  delta_dir_preds = delta_pos_preds[:, 1:].clone()

            #  delta_dir_preds = self._unit_circle(delta_dir_preds)
            #  delta_pos_preds = torch.cat([delta_xy_preds, delta_dir_preds], -1)

            aux_preds = (egomotion_preds, gps_preds, delta_pos_preds)

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            aux_preds,
        )


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        input_channels=3,
        baseplanes=32,
        ngroups=32,
        spatial_size=128,
        flat_output_size=2048,
        backbone=None,
    ):
        super().__init__()

        self.backbone = backbone

        final_spatial = int(spatial_size * self.backbone.final_spatial_compress)
        bn_size = int(round(flat_output_size / (final_spatial ** 2)))
        self.output_size = (bn_size, final_spatial, final_spatial)

        self.bn = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                bn_size,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, bn_size),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.backbone(x)

        return self.bn(x)


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

        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            if False:
                self.register_buffer(
                    "grayscale_kernel",
                    torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32).view(
                        1, 3, 1, 1
                    ),
                )
                self._n_input_rgb = 1

            self._sq_size = min(observation_space.spaces["rgb"].shape[0:2])
            spatial_size = self._sq_size // 2
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            self._sq_size = min(observation_space.spaces["depth"].shape[0:2])
            spatial_size = self._sq_size // 2
        else:
            self._n_input_depth = 0

        self._norm_inputs = norm_visual_inputs
        if self._norm_inputs and not blind:
            self.running_mean_and_var = ImageAutoRunningMeanAndVar(
                n_channels=self._n_input_rgb + self._n_input_depth
            )

        self.prev_action_embedding = nn.Embedding(5, 32)
        self._n_prev_action = 32

        self._n_input_goal = observation_space.spaces["pointgoal"].shape[0]
        self._old_goal_format = False
        if self._old_goal_format:
            self._n_input_goal -= 1
        self.tgt_embed = nn.Linear(self._n_input_goal, 32)
        self._n_input_goal = 32

        if "gps_and_compass" in observation_space.spaces:
            self.gps_compass_embed = nn.Linear(
                observation_space.spaces["gps_and_compass"].shape[0], 32
            )
            self._n_input_goal += 32
        else:
            self.gps_compass_embed = None

        if "dist_to_goal" in observation_space.spaces:
            self.dist_to_goal_embed = nn.Linear(
                observation_space.spaces["dist_to_goal"].shape[0], 32
            )
            self._n_input_goal += 32
        else:
            self.dist_to_goal_embed = None

        self._hidden_size = hidden_size

        rnn_input_size = self._n_input_goal + self._n_prev_action
        if not blind:
            assert self._n_input_depth + self._n_input_rgb > 0

            if "resnet" in backbone or "resneXt" in backbone:
                backbone = getattr(nav_analysis.rl.resnet, backbone)(
                    self._n_input_depth + self._n_input_rgb,
                    resnet_baseplanes,
                    resnet_baseplanes // 2,
                )
            else:
                backbone = nav_analysis.rl.efficient_net.EfficientNet.from_name(
                    self._n_input_depth + self._n_input_rgb, backbone
                )

            encoder = ResNetEncoder(
                self._n_input_depth + self._n_input_rgb,
                resnet_baseplanes,
                resnet_baseplanes // 2,
                spatial_size,
                backbone=backbone,
            )
            self.cnn = nn.Sequential(
                encoder,
                Flatten(),
                nn.Linear(np.prod(encoder.output_size), hidden_size, bias=True),
                nn.ReLU(True),
            )
            rnn_input_size += self._hidden_size
        else:
            self._n_input_rgb = 0
            self._n_input_depth = 0
            self.cnn = None

        self._rnn_type = rnn_type
        self._num_recurrent_layers = num_recurrent_layers
        if rnn_type is None:
            print("NO RNN", flush=True)

            class RNNDummy(nn.Module):
                def __init__(self, rnn_input_size, hidden_size):
                    super().__init__()
                    self.layer_in = nn.Sequential(
                        nn.Linear(rnn_input_size, hidden_size), nn.ReLU(True)
                    )

                    self.rnn = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(True),
                        nn.Linear(hidden_size, hidden_size, bias=False),
                        nn.LayerNorm(hidden_size),
                        nn.ReLU(True),
                    )

                def forward(self, x):
                    x = self.layer_in(x)
                    return self.rnn(x) + x

            self.rnn = RNNDummy(rnn_input_size, hidden_size)

            self._rnn_type = "None"
            self._num_recurrent_layers = 1
        elif rnn_type == "LN-LSTM":
            self.rnn = LayerNormLSTM(
                rnn_input_size, hidden_size, num_layers=num_recurrent_layers
            )
        elif rnn_type == "DPFRL":
            self.rnn = DPFRL(rnn_input_size, hidden_size)
        else:
            self.rnn = getattr(nn, rnn_type)(
                rnn_input_size, hidden_size, num_layers=num_recurrent_layers
            )

        self._task = task
        if self._task in ["loopnav", "teleportnav"]:
            self.critic = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(True),
                nn.Linear(hidden_size // 2, 1),
            )
        else:
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
        if self.cnn is not None:
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

    def _mask_hidden(self, hidden_states, masks, initial_hidden_state=None):

        if initial_hidden_state is None:
            if isinstance(hidden_states, tuple):
                hidden_states = tuple(v * masks for v in hidden_states)
            else:
                hidden_states = masks * hidden_states
        else:
            initial_hidden_state = initial_hidden_state.permute(1, 0, 2)
            initial_hidden_state = self._unpack_hidden(initial_hidden_state)

            if isinstance(hidden_states, tuple):
                for curr, init in zip(hidden_states, initial_hidden_state):
                    assert curr.size() == init.size()

                hidden_states = tuple(
                    torch.where(masks == 1, curr, init)
                    for curr, init in zip(hidden_states, initial_hidden_state)
                )
            else:
                assert hidden_states.size() == initial_hidden_state.size()
                hidden_states = torch.where(
                    masks == 1, hidden_states, initial_hidden_state
                )

        return hidden_states

    def forward_rnn(self, x, hidden_states, masks, prev_actions, observations):
        if x.size(0) == hidden_states.size(1):
            hidden_states = self._unpack_hidden(hidden_states)
            x, hidden_states = self.rnn(
                x.unsqueeze(0),
                self._mask_hidden(
                    hidden_states,
                    masks.unsqueeze(0),
                    observations.get("initial_hidden_state", None),
                ),
            )
            x = x.squeeze(0)
            hidden_states = self._pack_hidden(hidden_states)
        elif "initial_hidden_state" not in observations:
            N = hidden_states.size(1)

            (
                x_seq,
                hidden_states,
                select_inds,
                rnn_state_batch_inds,
                last_episode_in_batch_mask,
            ) = build_rnn_inputs(x, masks.bool(), hidden_states)

            x_seq, hidden_states = self.rnn(x_seq, self._unpack_hidden(hidden_states))
            hidden_states = self._pack_hidden(hidden_states)

            x, hidden_states = build_rnn_out_from_seq(
                x_seq,
                hidden_states,
                select_inds,
                rnn_state_batch_inds,
                last_episode_in_batch_mask,
                N,
            )
        else:
            # x is a (T, N, -1) tensor flattened to (T * N, -1)
            n = hidden_states.size(1)
            t = int(x.size(0) / n)

            initial_hidden_states = observations.get("initial_hidden_state", None)
            if initial_hidden_states is not None:
                initial_hidden_states = initial_hidden_states.view(
                    t, n, *initial_hidden_states.size()[1:]
                )

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
                    self._mask_hidden(
                        hidden_states,
                        masks[start_idx].view(1, -1, 1),
                        initial_hidden_states[start_idx]
                        if initial_hidden_states is not None
                        else None,
                    ),
                )

                outputs.append(rnn_scores)

            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            x = x.view(t * n, -1)  # flatten

            hidden_states = self._pack_hidden(hidden_states)
        return x, hidden_states

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            #  rgb_observations = (self.grayscale_kernel * rgb_observations).sum(
            #  1, keepdim=True
            #  )

            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)

            cnn_input.append(depth_observations)

        goal_observations = observations["pointgoal"]
        if self._old_goal_format:
            rho_obs = goal_observations[:, 0].clone()
            phi_obs = -torch.atan2(goal_observations[:, 2], goal_observations[:, 1])
            goal_observations = torch.stack([rho_obs, phi_obs], -1)

        goal_observations = self.tgt_embed(goal_observations)

        start_tok = torch.full_like(
            prev_actions, 4 if "initial_hidden_state" in observations else 0
        )
        prev_actions_emb = self.prev_action_embedding(
            torch.where(masks == 1.0, prev_actions + 1, start_tok).squeeze(-1)
        )

        x = []
        cnn_feats = None
        if len(cnn_input) > 0:
            cnn_input = torch.cat(cnn_input, dim=1)
            if cnn_input.size(2) != self._sq_size or cnn_input.size(3) != self._sq_size:
                cnn_input = F.interpolate(
                    cnn_input, size=(self._sq_size, self._sq_size), mode="area"
                )
                assert False

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

        if self._rnn_type == "None":
            x = self.rnn(x)
        else:
            x, rnn_hidden_states = self.forward_rnn(
                x, rnn_hidden_states, masks, prev_actions, observations
            )

        if self._task in ["loopnav", "teleportnav"]:
            value = self.critic(x)
        else:
            value = self.critic_linear(x)

        return value, x, rnn_hidden_states, cnn_feats
