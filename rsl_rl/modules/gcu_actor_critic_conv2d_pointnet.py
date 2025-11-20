# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

from rsl_rl.utils import resolve_nn_activation

from rsl_rl.modules.actor_critic_conv2d_pointnet import ActorCriticConv2dPointNet
from rsl_rl.modules.gcu_actor_critic import GCUActorCritic

class GCUActorCriticConv2dPointNet(GCUActorCritic, ActorCriticConv2dPointNet):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        image_input_shape,
        conv_layers_params,
        conv_linear_output_size,
        pointnet_layers_params,
        pointnet_in_dim,
        pointnet_num_points,
        actor_hidden_dims,
        critic_hidden_dims,
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        # Initialize ActorCriticConv2dPointNet first to set up the PointNet architecture
        ActorCriticConv2dPointNet.__init__(
            self,
            num_actor_obs,
            num_critic_obs,
            num_actions,
            image_input_shape,
            conv_layers_params,
            conv_linear_output_size,
            pointnet_layers_params,
            pointnet_in_dim,
            pointnet_num_points,
            actor_hidden_dims,
            critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            **kwargs,
        )
        
        # Initialize GCUActorCritic to set up GCU-specific attributes
        # We skip the parent's __init__ by calling nn.Module.__init__ directly in GCUActorCritic
        # But we need to set up the GCU-specific attributes
        self.placement_dist = None
        self.orientation_dist = None
        self.device = None
