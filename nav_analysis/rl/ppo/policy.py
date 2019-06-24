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
from nav_analysis.rl.ppo.utils import CategoricalNet, Flatten

#  from nav_analysis.rl.layer_norm_lstm import LayerNormLSTM


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
    ):
        super().__init__()
        self.dim_actions = action_space.n

        self.net = Net(
            observation_space=observation_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            blind=blind,
            rnn_type=rnn_type,
            backbone=backbone,
            resnet_baseplanes=resnet_baseplanes,
        )

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
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
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        value, actor_features, rnn_hidden_states, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(actor_features)

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
        value, _, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
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
        distribution = self.action_distribution(actor_features)

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
        make_backbone=None,
    ):
        super().__init__()

        self.backbone = make_backbone(input_channels, baseplanes, ngroups)

        final_spatial = int(
            spatial_size * self.backbone.final_spatial_compress
        )
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
    ):
        super().__init__()

        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            self._n_input_rgb = 3
            self.register_buffer(
                "grayscale_kernel",
                torch.tensor(
                    [0.2126, 0.7152, 0.0722], dtype=torch.float32
                ).view(1, 3, 1, 1),
            )
            spatial_size = observation_space.spaces["rgb"].shape[0] // 2
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            spatial_size = observation_space.spaces["depth"].shape[0] // 2
        else:
            self._n_input_depth = 0

        self._norm_inputs = False
        if backbone.split("_")[-1] == "norm":
            self.register_buffer("_mean", torch.zeros(1, 4, 1, 1))
            self.register_buffer("_var", torch.full((1, 4, 1, 1), 0.0))
            self.register_buffer("_count", torch.full((), 0.0))
            self._norm_inputs = True

            backbone = "_".join(backbone.split("_")[:-1])

        self.prev_action_embedding = nn.Embedding(5, 32)
        self._n_prev_action = 32

        self._n_input_goal = observation_space.spaces["pointgoal"].shape[0]
        self._old_goal_format = False
        if self._old_goal_format:
            self._n_input_goal -= 1
        self.tgt_embed = nn.Linear(self._n_input_goal, 32)
        self._n_input_goal = 32

        self._hidden_size = hidden_size

        rnn_input_size = self._n_input_goal + self._n_prev_action
        if not blind:
            assert self._n_input_depth + self._n_input_rgb > 0
            encoder = ResNetEncoder(
                self._n_input_depth + self._n_input_rgb,
                resnet_baseplanes,
                resnet_baseplanes // 2,
                spatial_size,
                make_backbone=getattr(nav_analysis.rl.resnet, backbone),
            )
            self.cnn = nn.Sequential(
                encoder,
                Flatten(),
                nn.Linear(np.prod(encoder.output_size), hidden_size),
                nn.ReLU(True),
            )
            rnn_input_size += self._hidden_size
        else:
            self._n_input_rgb = 0
            self._n_input_depth = 0
            self.cnn = None

        self._rnn_type = rnn_type
        self._num_recurrent_layers = num_recurrent_layers
        if rnn_type == "LN-LSTM":
            self.rnn = LayerNormLSTM(
                rnn_input_size, hidden_size, num_layers=num_recurrent_layers
            )
        else:
            self.rnn = getattr(nn, rnn_type)(
                rnn_input_size, hidden_size, num_layers=num_recurrent_layers
            )
        self.critic_linear = nn.Linear(hidden_size, 1)

        self.layer_init()
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self._num_recurrent_layers * (
            2 if "LSTM" in self._rnn_type else 1
        )

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
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

        nn.init.orthogonal_(self.critic_linear.weight, gain=1)
        nn.init.constant_(self.critic_linear.bias, val=0)

    def _pack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = torch.cat(
                [hidden_states[0], hidden_states[1]], dim=0
            )

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

    def forward_rnn(self, x, hidden_states, masks):
        if x.size(0) == hidden_states.size(1):
            hidden_states = self._unpack_hidden(hidden_states)
            x, hidden_states = self.rnn(
                x.unsqueeze(0),
                self._mask_hidden(hidden_states, masks.unsqueeze(0)),
            )
            x = x.squeeze(0)
        else:
            # x is a (T, N, -1) tensor flattened to (T * N, -1)
            n = hidden_states.size(1)
            t = int(x.size(0) / n)

            # unflatten
            x = x.view(t, n, x.size(1))
            masks = masks.view(t, n)

            # steps in sequence which have zero for any agent. Assume t=0 has
            # a zero in it.
            has_zeros = (
                (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
            )

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]  # handle scalar
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [t]

            hidden_states = self._unpack_hidden(hidden_states)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # process steps that don't have any zeros in masks together
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hidden_states = self.rnn(
                    x[start_idx:end_idx],
                    self._mask_hidden(
                        hidden_states, masks[start_idx].view(1, -1, 1)
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
            #  rgb_observations = rgb_observations / 255.0  # normalize RGB
            #  rgb_observations = (rgb_observations * self.grayscale_kernel).sum(
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
            phi_obs = -torch.atan2(
                goal_observations[:, 2], goal_observations[:, 1]
            )
            goal_observations = torch.stack([rho_obs, phi_obs], -1)

        goal_observations = self.tgt_embed(goal_observations)
        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(-1)
        )

        x = []
        cnn_feats = None
        if len(cnn_input) > 0:
            cnn_input = torch.cat(cnn_input, dim=1)
            cnn_input = F.avg_pool2d(cnn_input, 2)
            if self._norm_inputs:
                if self.training:
                    import torch.distributed as distrib

                    new_mean = F.adaptive_avg_pool2d(cnn_input, 1).mean(
                        0, keepdim=True
                    )
                    new_count = torch.full_like(self._count, 1)

                    if distrib.is_initialized():
                        distrib.all_reduce(new_mean)
                        distrib.all_reduce(new_count)

                    new_mean /= new_count

                    new_var = F.adaptive_avg_pool2d(
                        (cnn_input - new_mean).pow(2), 1
                    ).mean(0, keepdim=True)

                    if distrib.is_initialized():
                        distrib.all_reduce(new_var)

                    # No - 1 on all the variance as the number of pixels
                    # seen over training is simply absurd, so it doesn't matter
                    new_var /= new_count

                    m_a = self._var * (self._count)
                    m_b = new_var * (new_count)
                    M2 = (
                        m_a
                        + m_b
                        + (new_mean - self._mean).pow(2)
                        * self._count
                        * new_count
                        / (self._count + new_count)
                    )

                    self._var = M2 / (self._count + new_count)
                    self._mean = (
                        self._count * self._mean + new_count * new_mean
                    ) / (self._count + new_count)

                    self._count += new_count

                stdev = torch.sqrt(
                    torch.max(self._var, torch.full_like(self._var, 1e-2))
                )
                cnn_input = (cnn_input - self._mean) / stdev

            cnn_feats = self.cnn(cnn_input)
            x += [cnn_feats]

        x += [goal_observations, prev_actions]

        x = torch.cat(x, dim=1)  # concatenate goal vector
        x, rnn_hidden_states = self.forward_rnn(x, rnn_hidden_states, masks)

        return self.critic_linear(x), x, rnn_hidden_states, cnn_feats
