# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

from rsl_rl.utils import resolve_nn_activation

from rsl_rl.modules.actor_critic_conv2d import ActorCriticConv2d

class GCUActorCritic(ActorCriticConv2d):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        image_input_shape,
        conv_layers_params,
        conv_linear_output_size,
        actor_hidden_dims,
        critic_hidden_dims,
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        super().__init__(
            num_actor_obs,
            num_critic_obs,
            num_actions,
            image_input_shape,
            conv_layers_params,
            conv_linear_output_size,
            actor_hidden_dims,
            critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            **kwargs,
        )
        
        # Separate distributions for placement and orientation
        self.placement_dist = None
        self.orientation_dist = None
        
        # Store device for KL divergence calculation
        # Initialize device after super().__init__ to ensure parameters are available
        self.device = None

    def update_distribution(self, observations):
        """Update both placement and orientation distributions."""
        # Initialize device if not set
        if self.device is None:
            self.device = next(self.parameters()).device
            
        # Get raw actor output: [batch, 4] where [:, :2] are placement means, [:, 2:] are orientation logits
        action_raw = self.actor(observations)
        
        # Extract placement parameters (first 2 dimensions)
        mean_xy = action_raw[:, :2]  # [batch, 2] for x, y placement
        
        # Extract orientation logits (last 2 dimensions)
        logits_o = action_raw[:, 2:]  # [batch, 2] for orientation logits
        
        # Compute placement standard deviation
        std_xy = self.std[:2].expand_as(mean_xy)

        # Create distributions
        self.placement_dist = Normal(mean_xy, std_xy)
        self.orientation_dist = Categorical(logits=logits_o)
        
        # Store the raw action output for compatibility with action_mean/action_std
        self.distribution = Normal(action_raw, torch.ones_like(action_raw))

    def act(self, observations, **kwargs):
        """Sample actions: raw 4D output [x, y, logit0, logit1] for compatibility with _convert_to_pos_quat."""
        self.update_distribution(observations)
        
        # Sample placement from Gaussian
        xy = self.placement_dist.sample()  # [batch, 2]
        
        # Sample orientation from categorical
        orientation = self.orientation_dist.sample()  # [batch]
        
        # Convert orientation to one-hot encoding for logits
        orientation_onehot = torch.zeros_like(self.orientation_dist.logits)
        orientation_onehot.scatter_(1, orientation.unsqueeze(1), 1.0)
        
        # Combine into raw 4D action tensor: [batch, 4] where [:, :2] are placement, [:, 2:] are orientation logits
        actions = torch.cat([xy, orientation_onehot], dim=-1)
        
        return actions

    def get_actions_log_prob(self, actions):
        """Compute log probability of actions."""
        # Split actions: [batch, 4] -> xy [batch, 2] and orientation logits [batch, 2]
        xy = actions[:, :2]
        orientation_logits = actions[:, 2:]
        
        # Convert logits to orientation index
        orientation = torch.argmax(orientation_logits, dim=-1)
        
        # Compute log probabilities
        log_prob_xy = self.placement_dist.log_prob(xy).sum(dim=-1)
        log_prob_o = self.orientation_dist.log_prob(orientation)
        
        # Total log probability
        log_prob_total = log_prob_xy + log_prob_o
        
        return log_prob_total

    @property
    def action_mean(self):
        """Return the mean of the raw action output for storage compatibility."""
        if self.distribution is None:
            return torch.zeros(1, 4)  # Default shape
        return self.distribution.mean

    @property
    def action_std(self):
        if self.placement_dist is None:
            return torch.ones(1,4)
        # build a 4‑vector: [σ_x, σ_y, 0, 0]  (or whatever makes sense for the discrete part)
        std_xy = self.placement_dist.stddev
        zeros = torch.zeros_like(self.orientation_dist.logits)
        return torch.cat([std_xy, zeros], dim=-1)


    @property
    def entropy(self):
        """Compute total entropy of both distributions."""
        if self.placement_dist is None or self.orientation_dist is None:
            return torch.tensor(0.0)
        
        placement_entropy = self.placement_dist.entropy().sum(dim=-1)
        orientation_entropy = self.orientation_dist.entropy()
        
        return placement_entropy + orientation_entropy

    def act_inference(self, observations):
        """Get deterministic actions for inference."""
        action_raw = self.actor(observations)
        
        # Get placement mean
        xy = action_raw[:, :2]
        
        # Get orientation with highest probability
        logits_o = action_raw[:, 2:]
        orientation = torch.argmax(logits_o, dim=-1)
        
        # Convert to one-hot encoding
        orientation_onehot = torch.zeros_like(logits_o)
        orientation_onehot.scatter_(1, orientation.unsqueeze(1), 1.0)
        
        # Combine into raw 4D action tensor
        actions = torch.cat([xy, orientation_onehot], dim=-1)
        
        return actions

    def compute_kl_divergence(self, old_mu, old_sigma):
        """Compute KL divergence for mixed distribution (Gaussian placement + categorical orientation)."""
        if self.placement_dist is None or self.orientation_dist is None:
            return torch.tensor(0.0, device=self.device)
        
        # Extract placement parameters from old distribution
        old_mu_xy = old_mu[:, :2]  # Placement means
        old_sigma_xy = old_sigma[:, :2]  # Placement stds
        
        # Extract orientation logits from old distribution
        old_logits_o = old_mu[:, 2:]  # Orientation logits
        
        # Compute KL divergence for placement (Gaussian)
        current_mu_xy = self.placement_dist.mean
        current_sigma_xy = self.placement_dist.stddev
        
        kl_placement = torch.sum(
            torch.log(current_sigma_xy / old_sigma_xy + 1.0e-5)
            + (torch.square(old_sigma_xy) + torch.square(old_mu_xy - current_mu_xy))
            / (2.0 * torch.square(current_sigma_xy))
            - 0.5,
            axis=-1,
        )
        
        # Compute KL divergence for orientation (categorical)
        current_logits_o = self.orientation_dist.logits
        old_probs_o = torch.softmax(old_logits_o, dim=-1)
        current_probs_o = torch.softmax(current_logits_o, dim=-1)
        
        kl_orientation = torch.sum(
            old_probs_o * (torch.log(old_probs_o + 1e-8) - torch.log(current_probs_o + 1e-8)),
            dim=-1
        )
        
        # Total KL divergence
        total_kl = kl_placement + kl_orientation
        
        return total_kl
