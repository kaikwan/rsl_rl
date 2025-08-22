# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.actor_critic_conv2d import ActorCriticConv2d


class MixtureGaussian6D:
    """Mixture of Gaussians for 6D rotation representation."""

    def __init__(self, mixture_logits, means, stds):
        """
        Args:
            mixture_logits: [B, K]
            means: [B, K, 6]
            stds: [B, K, 6] - direct standard deviations (not log)
            base_stds: [B, K, 6] - base standard deviations from initialization (optional)
        """
        self.mixture_logits = mixture_logits
        self.mixture_probs = torch.softmax(mixture_logits, dim=-1)  # [B, K]
        self.means = means
        self.stds = stds
            
        self.K = means.shape[1]

    def sample(self):
        """Sample one component and then a Gaussian within it."""
        B = self.mixture_logits.shape[0]
        cat = Categorical(self.mixture_probs)
        k = cat.sample()  # [B]
        batch_idx = torch.arange(B, device=k.device)
        means = self.means[batch_idx, k]   # [B, 6]
        stds = self.stds[batch_idx, k]     # [B, 6]
        dist = Normal(means, stds)
        return dist.sample(), k

    def log_prob(self, x):
        """Marginal log prob of x under mixture."""
        # log N(x | μ_k, σ_k)
        log_probs = Normal(self.means, self.stds).log_prob(x[:, None, :]).sum(-1)  # [B, K]
        log_mix = torch.log_softmax(self.mixture_logits, dim=-1)                   # [B, K]
        return torch.logsumexp(log_mix + log_probs, dim=-1)                        # [B]

    def entropy(self):
        """Approximate entropy = H(Categorical) + avg H(Normal)."""
        cat = Categorical(self.mixture_probs)
        H_cat = cat.entropy()                      # [B]
        H_gauss = Normal(self.means, self.stds).entropy().sum(-1).mean(-1)  # [B]
        return H_cat + H_gauss


class GCUActorCritic(ActorCriticConv2d):
    """Actor-Critic with Gaussian placement + mixture Gaussian 6D rotation."""

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
        num_mixture_components=1,
        **kwargs,
    ):
        self.K = num_mixture_components
        
        # Call parent with the actual num_actions (8: 2 placement + 6 rotation)
        super().__init__(
            num_actor_obs,
            num_critic_obs,
            num_actions=num_actions,  # Use actual num_actions (8), not total_std_dims
            image_input_shape=image_input_shape,
            conv_layers_params=conv_layers_params,
            conv_linear_output_size=conv_linear_output_size,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            **kwargs,
        )

        # Output dim = 2 (placement means) + K (mixture logits) + K*6 (rotation means)
        self.output_dim = 2 + self.K + self.K * 6

        # Replace actor head to output the mixture parameters
        self.actor.mlp[-1] = nn.Linear(actor_hidden_dims[-1], self.output_dim)

        # Initialize standard deviation parameters like actor_critic.py
        # Placement standard deviations (2D)
        self.placement_std = nn.Parameter(init_noise_std * torch.ones(2))
        # Rotation standard deviations (K*6D)
        self.rotation_std = nn.Parameter(init_noise_std * torch.ones(self.K * 6))

        self.placement_dist = None
        self.rotation_dist = None
        self.distribution = None
        
        # Initialize device attribute for KL divergence computation
        self.device = next(self.parameters()).device
        
        print(f"Initialized placement_std parameter with shape: {self.placement_std.shape}")
        print(f"Initialized rotation_std parameter with shape: {self.rotation_std.shape}")
        print(f"Placement stds: {self.placement_std}")
        print(f"Rotation stds: {self.rotation_std.view(self.K, 6)}")

    def update_distribution(self, observations):
        params = self.actor(observations)  # [B, output_dim]
        B = params.shape[0]
        
        # Placement mean + fixed nn.Parameter std
        mean_xy = params[:, :2]
        std_xy = self.placement_std.expand_as(mean_xy)
        self.placement_dist = Normal(mean_xy, std_xy)
        
        # Rotation mixture
        logits = params[:, 2:2+self.K]  # [B, K]
        rot_means = params[:, 2+self.K:].view(B, self.K, 6)
        stds = self.rotation_std.view(self.K, 6).expand(B, self.K, 6)
        self.rotation_dist = MixtureGaussian6D(logits, rot_means, stds)
        
        # dummy distribution for API
        self.distribution = self.placement_dist

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        xy = self.placement_dist.sample()
        rot6d, _ = self.rotation_dist.sample()
        return torch.cat([xy, rot6d], dim=-1)  # [B, 8] env actions

    def get_actions_log_prob(self, actions):
        xy = actions[:, :2]
        rot6d = actions[:, 2:8]
        logp_xy = self.placement_dist.log_prob(xy).sum(-1)
        logp_rot = self.rotation_dist.log_prob(rot6d)
        return logp_xy + logp_rot

    @property
    def action_mean(self):
        """Return the full 8-dimensional action mean for storage compatibility."""
        if self.placement_dist is None or self.rotation_dist is None:
            return torch.zeros(1, 8, device=self.placement_std.device)
        
        # Get placement mean (2D)
        xy_mean = self.placement_dist.mean
        
        # Get rotation mean from best mixture component
        logits = self.rotation_dist.mixture_logits
        best_k = torch.argmax(logits, dim=-1)
        batch_idx = torch.arange(best_k.shape[0], device=best_k.device)
        rot_mean = self.rotation_dist.means[batch_idx, best_k]
        
        # Concatenate to get 8D action mean
        # Ensure the shape matches the expected action shape [batch_size, 8]
        return torch.cat([xy_mean, rot_mean], dim=-1)

    @property
    def action_std(self):
        """Return the full 8-dimensional action std for storage compatibility."""
        if self.placement_dist is None or self.rotation_dist is None:
            return torch.ones(1, 8, device=self.placement_std.device)
        
        # Get placement std (2D)
        xy_std = self.placement_dist.stddev
        
        # Get rotation std from best mixture component
        logits = self.rotation_dist.mixture_logits
        best_k = torch.argmax(logits, dim=-1)
        batch_idx = torch.arange(best_k.shape[0], device=best_k.device)
        rot_std = self.rotation_dist.stds[batch_idx, best_k]
        
        # Concatenate to get 8D action std
        # Ensure the shape matches the expected action shape [batch_size, 8]
        return torch.cat([xy_std, rot_std], dim=-1)

    @property
    def entropy(self):
        return self.placement_dist.entropy().sum(-1) + self.rotation_dist.entropy()

    def act_inference(self, observations):
        params = self.actor(observations)
        xy = params[:, :2]
        logits = params[:, 2:2+self.K]
        rot_means = params[:, 2+self.K:].view(-1, self.K, 6)

        best_k = torch.argmax(logits, dim=-1)
        batch_idx = torch.arange(best_k.shape[0], device=best_k.device)
        rot6d = rot_means[batch_idx, best_k]
        return torch.cat([xy, rot6d], dim=-1)  # deterministic [B, 8]

    def compute_kl_divergence(self, old_mu, old_sigma):
        """Compute KL divergence for mixed distribution (Gaussian placement + mixture rotation)."""
        if self.placement_dist is None or self.rotation_dist is None:
            return torch.tensor(0.0, device=self.device)
        
        # Extract placement parameters from old distribution
        old_mu_xy = old_mu[:, :2]  # Placement means
        old_sigma_xy = old_sigma[:, :2]  # Placement stds
        
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
        
        # For rotation, use a simplified KL approximation
        # Since we have a mixture, we'll use the KL between the best components
        if self.rotation_dist is not None:
            # Get the best component for current policy
            best_k_current = torch.argmax(self.rotation_dist.mixture_probs, dim=-1)
            batch_idx = torch.arange(best_k_current.shape[0], device=best_k_current.device)
            
            # Extract the best component parameters
            current_rot_mean = self.rotation_dist.means[batch_idx, best_k_current]
            current_rot_std = self.rotation_dist.stds[batch_idx, best_k_current]
            
            # For old parameters, we'll use a reasonable approximation
            # Since old_mu and old_sigma are 8-dimensional, we need to extract rotation
            if old_mu.shape[1] >= 8:
                old_rot_mean = old_mu[:, 2:8]
                old_rot_std = old_sigma[:, 2:8].clamp(min=1e-5)  # Ensure positive
                
                kl_rotation = torch.sum(
                    torch.log(current_rot_std / old_rot_std + 1.0e-5)
                    + (torch.square(old_rot_std) + torch.square(old_rot_mean - current_rot_mean))
                    / (2.0 * torch.square(current_rot_std))
                    - 0.5,
                    axis=-1,
                )
            else:
                kl_rotation = torch.zeros_like(kl_placement)
        else:
            kl_rotation = torch.zeros_like(kl_placement)
        
        # Total KL divergence
        total_kl = kl_placement + kl_rotation
        
        return total_kl
