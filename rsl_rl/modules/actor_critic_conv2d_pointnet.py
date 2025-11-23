# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories
from rsl_rl.modules.actor_critic_conv2d import ResidualBlock

class PointNetEncoder(nn.Module):
    def __init__(self, in_dim=8, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1),
            nn.ReLU(),
            # nn.Conv1d(64, 128, 1),
            # nn.ReLU(),
            nn.Conv1d(64, out_dim, 1),
            nn.ReLU(),
        )

    def forward(self, x):  # x: [B, P, D]
        x = self.mlp(x)                       # [B, P, out]
        x = x.max(dim=2).values 
        return x

class ConvolutionalNetworkWithPointNet(nn.Module):
    def __init__(
        self,
        proprio_input_dim,
        output_dim,
        image_input_shape,
        conv_layers_params,
        pointnet_layers_params,
        pointnet_in_dim,
        pointnet_num_points,
        hidden_dims,
        activation_fn,
        conv_linear_output_size,
    ):
        super().__init__()

        self.image_input_shape = image_input_shape  # (C, H, W)
        self.image_obs_size = torch.prod(torch.tensor(self.image_input_shape)).item()
        self.proprio_obs_size = proprio_input_dim
        self.input_dim = self.proprio_obs_size + self.image_obs_size
        self.activation_fn = activation_fn
        self.pointnet_in_dim = pointnet_in_dim
        self.pointnet_num_points = pointnet_num_points
        self.conv_linear_output_size = conv_linear_output_size

        # Build the PointNet encoder and get its output size
        self.pointnet_encoder = self.build_pointnet_net(pointnet_layers_params, in_dim=pointnet_in_dim)
        with torch.no_grad():
            dummy_pointnet = torch.zeros(1, pointnet_in_dim, pointnet_num_points)
            pointnet_output = self.pointnet_encoder(dummy_pointnet)
            self.encoded_pointnet_size = pointnet_output.shape[1]

        # Check if we need a projection layer for RNN output (when proprio_input_dim != PointNet input size)
        expected_pointnet_input_size = pointnet_in_dim * pointnet_num_points
        if proprio_input_dim != expected_pointnet_input_size:
            # RNN output case: add projection from RNN output to PointNet encoder output size
            self.proprio_projection = nn.Linear(proprio_input_dim, self.encoded_pointnet_size)
            self.use_projection = True
        else:
            # Raw observations case: no projection needed
            self.proprio_projection = nn.Identity()  # Use Identity instead of None for JIT compatibility
            self.use_projection = False

        # Build conv network and get its output size
        self.conv_net = self.build_conv_net(conv_layers_params)
        with torch.no_grad():
            dummy_image = torch.zeros(1, *self.image_input_shape)
            conv_output = self.conv_net(dummy_image)
            self.image_feature_size = conv_output.view(1, -1).shape[1]

        # Build the connection layers between conv net and mlp
        self.conv_linear = nn.Linear(self.image_feature_size, conv_linear_output_size)
        self.layernorm = nn.LayerNorm(conv_linear_output_size)

        # Build the mlp
        self.mlp = nn.Sequential(
            nn.Linear(self.encoded_pointnet_size + conv_linear_output_size, hidden_dims[0]),
            self.activation_fn,
            *[
                layer
                for dim in zip(hidden_dims[:-1], hidden_dims[1:])
                for layer in (nn.Linear(dim[0], dim[1]), self.activation_fn)
            ],
            nn.Linear(hidden_dims[-1], output_dim),
        )
        
        # Store output dimension for JIT compatibility (avoid computing in forward)
        self.output_dim = output_dim

        # Initialize the weights
        self._initialize_weights()

    def build_conv_net(self, conv_layers_params):
        layers = []
        in_channels = self.image_input_shape[0]
        for idx, params in enumerate(conv_layers_params[:-1]):
            layers.extend([
                nn.Conv2d(
                    in_channels,
                    params["out_channels"],
                    kernel_size=params.get("kernel_size", 3),
                    stride=params.get("stride", 1),
                    padding=params.get("padding", 0),
                ),
                nn.BatchNorm2d(params["out_channels"]),
                nn.ReLU(inplace=True),
                ResidualBlock(params["out_channels"]) if idx > 0 else nn.Identity(),
            ])
            in_channels = params["out_channels"]
        last_params = conv_layers_params[-1]
        layers.append(
            nn.Conv2d(
                in_channels,
                last_params["out_channels"],
                kernel_size=last_params.get("kernel_size", 3),
                stride=last_params.get("stride", 1),
                padding=last_params.get("padding", 0),
            )
        )
        layers.append(nn.BatchNorm2d(last_params["out_channels"]))
        return nn.Sequential(*layers)

    def build_pointnet_net(self, pointnet_layers_params, in_dim=8):
        layers = []
        in_channels = in_dim
        for idx, params in enumerate(pointnet_layers_params[:-1]):
            layers.extend([
                nn.Conv1d(
                    in_channels,
                    params["out_channels"],
                    kernel_size=params.get("kernel_size", 1),
                ),
                nn.BatchNorm1d(params["out_channels"]) if params.get("batch_norm", True) else nn.Identity(),
                nn.ReLU(inplace=True),
            ])
            in_channels = params["out_channels"]
        last_params = pointnet_layers_params[-1]
        layers.append(
            nn.Conv1d(
                in_channels,
                last_params["out_channels"],
                kernel_size=last_params.get("kernel_size", 1),
            )
        )
        if last_params.get("batch_norm", True):
            layers.append(nn.BatchNorm1d(last_params["out_channels"]))
        if last_params.get("activation", True):
            layers.append(nn.ReLU(inplace=True))
        
        class PointNetModule(nn.Module):
            def __init__(self, mlp, input_dim):
                super().__init__()
                self.mlp = mlp
                self.input_dim = input_dim
            
            def forward(self, x):  # x: [B, D, P] format (always reshaped before calling)
                # Input is always in [B, D, P] format when called from forward()
                # Removed dynamic shape check for JIT compatibility
                x = self.mlp(x)  # [B, out_channels, P]
                x = x.max(dim=2).values  # [B, out_channels]
                return x
        
        return PointNetModule(nn.Sequential(*layers), in_dim)

    def _initialize_weights(self):
        for m in self.conv_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.pointnet_encoder.mlp.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.kaiming_normal_(self.conv_linear.weight, mode="fan_out", nonlinearity="tanh")
        nn.init.constant_(self.conv_linear.bias, 0)
        nn.init.constant_(self.layernorm.weight, 1.0)
        nn.init.constant_(self.layernorm.bias, 0.0)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias) if layer.bias is not None else None
        
        # Initialize projection layer if it exists
        if self.use_projection:
            nn.init.orthogonal_(self.proprio_projection.weight, gain=0.01)
            nn.init.zeros_(self.proprio_projection.bias) if self.proprio_projection.bias is not None else None

    def forward(self, observations):
        proprio_obs = observations[:, : -self.image_obs_size]
        image_obs = observations[:, -self.image_obs_size :]
        
        # Process proprioceptive observations based on initialization flag (JIT-compatible)
        # This avoids dynamic shape checks that JIT doesn't support
        if self.use_projection:
            # RNN output case: project to match PointNet encoder output size
            pointnet_features = self.proprio_projection(proprio_obs)
        else:
            # Raw observations case: process through PointNet
            pointnet_features = self.pointnet_encoder(proprio_obs.view(-1, self.pointnet_in_dim, self.pointnet_num_points))

        batch_size = image_obs.size(0)
        
        # Handle empty batch case (use stored output_dim for JIT compatibility)
        if batch_size == 0:
            # Return empty tensor with correct output shape
            return torch.empty(0, self.output_dim, device=observations.device, dtype=observations.dtype)
        
        image = image_obs.view(batch_size, *self.image_input_shape)

        conv_features = self.conv_net(image)
        
        # Handle empty conv_features case (use stored output_dim for JIT compatibility)
        if conv_features.numel() == 0:
            return torch.empty(0, self.output_dim, device=observations.device, dtype=observations.dtype)
        
        flattened_conv_features = conv_features.view(batch_size, -1)
        normalized_conv_output = self.layernorm(self.conv_linear(flattened_conv_features))
        combined_input = torch.cat([pointnet_features, normalized_conv_output], dim=1)
        output = self.mlp(combined_input)
        return output



class ActorCriticConv2dPointNet(nn.Module):
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
        nn.Module.__init__(self)

        self.image_input_shape = image_input_shape  # (C, H, W)
        self.activation_fn = resolve_nn_activation(activation)

        self.actor = ConvolutionalNetworkWithPointNet(
            proprio_input_dim=num_actor_obs,
            output_dim=num_actions,
            image_input_shape=image_input_shape,
            conv_layers_params=conv_layers_params,
            pointnet_layers_params=pointnet_layers_params,
            pointnet_in_dim=pointnet_in_dim,
            pointnet_num_points=pointnet_num_points,
            hidden_dims=actor_hidden_dims,
            activation_fn=self.activation_fn,
            conv_linear_output_size=conv_linear_output_size,
        )

        self.critic = ConvolutionalNetworkWithPointNet(
            proprio_input_dim=num_critic_obs,
            output_dim=1,
            image_input_shape=image_input_shape,
            conv_layers_params=conv_layers_params,
            pointnet_layers_params=pointnet_layers_params,
            pointnet_in_dim=pointnet_in_dim,
            pointnet_num_points=pointnet_num_points,
            hidden_dims=critic_hidden_dims,
            activation_fn=self.activation_fn,
            conv_linear_output_size=conv_linear_output_size,
        )

        print(f"Modified Actor Network: {self.actor}")
        print(f"Modified Critic Network: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


class ActorCriticConv2dPointNetRecurrent(ActorCriticConv2dPointNet):
    is_recurrent = True

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
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=rnn_hidden_dim,
            num_critic_obs=rnn_hidden_dim,
            num_actions=num_actions,
            image_input_shape=image_input_shape,
            conv_layers_params=conv_layers_params,
            conv_linear_output_size=conv_linear_output_size,
            pointnet_layers_params=pointnet_layers_params,
            pointnet_in_dim=pointnet_in_dim,
            pointnet_num_points=pointnet_num_points,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            **kwargs,
        )

        activation = resolve_nn_activation(activation)

        # Compute image observation size from image_input_shape
        num_image_obs = torch.prod(torch.tensor(image_input_shape)).item()
        
        # Store sizes for splitting observations
        self.num_proprio_obs = num_actor_obs  # Proprioceptive observation size (4096)
        self.num_image_obs = num_image_obs    # Image observation size (1924)
        
        # Memory should process the full concatenated observations (proprio + image)
        # The RNN maintains temporal dependencies across both proprioceptive and image features
        # Full observation size = num_actor_obs + num_image_obs (4096 + 1924 = 6020)
        self.memory_a = Memory(num_actor_obs + num_image_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.memory_c = Memory(num_critic_obs + num_image_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

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

    def act(self, observations, masks=None, hidden_states=None):
        # Process full observations (proprio + image) through RNN
        # RNN maintains temporal dependencies across both proprioceptive and image features
        rnn_output = self.memory_a(observations, masks, hidden_states)
        
        # Extract image observations - need to handle batch mode (with masks) vs inference mode
        if masks is not None:
            # Batch mode: observations is [seq_len, batch, obs_features]
            # RNN output after unpad_trajectories has shape [seq_len, num_valid, hidden_dim]
            # Extract image part from full observations: [seq_len, batch, image_features]
            image_obs_padded = observations[:, :, self.num_proprio_obs:]
            # Apply same unpad logic as RNN output
            image_obs = unpad_trajectories(image_obs_padded, masks)
            # Both rnn_output and image_obs have shape [seq_len, num_valid, features]
            # They should have the same seq_len and num_valid since they use the same masks
            seq_len = rnn_output.shape[0]
            num_valid = rnn_output.shape[1]
            
            # Ensure shapes match
            assert image_obs.shape[0] == seq_len, \
                f"Image obs seq_len {image_obs.shape[0]} doesn't match RNN seq_len {seq_len}"
            assert image_obs.shape[1] == num_valid, \
                f"Image obs num_valid {image_obs.shape[1]} doesn't match RNN num_valid {num_valid}"
            
            # Flatten to [seq_len * num_valid, features] (which equals [total_batch, features])
            rnn_output = rnn_output.reshape(-1, rnn_output.shape[-1])  # [seq_len * num_valid, rnn_hidden_dim]
            image_obs = image_obs.reshape(-1, image_obs.shape[-1])  # [seq_len * num_valid, image_features]
        else:
            # Inference mode: observations is [batch, obs_features]
            rnn_output = rnn_output.squeeze(0)  # [batch, rnn_hidden_dim]
            image_obs = observations[:, self.num_proprio_obs:]  # [batch, image_features]
        
        # Concatenate RNN output with image observations to pass to actor
        # Actor expects [proprio_features, image_obs] where proprio_features is rnn_hidden_dim
        actor_input = torch.cat([rnn_output, image_obs], dim=1)
        
        self.update_distribution(actor_input)
        
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
        # Handle 3D actions: [num_aug/seq_len, batch, action_dim] -> [total_batch, action_dim]
        original_shape = actions.shape
        if actions.dim() == 3:
            actions = actions.reshape(-1, actions.shape[-1])  # Flatten first two dimensions
        
        # Split actions: [batch, action_dim] -> xy [batch, 2] and orientation logits [batch, action_dim-2]
        xy = actions[:, :2]
        orientation_logits = actions[:, 2:]
        
        # Convert logits to orientation index
        orientation = torch.argmax(orientation_logits, dim=-1)
        
        # Ensure distribution batch size matches actions batch size
        dist_batch_size = self.placement_dist.loc.shape[0]
        action_batch_size = xy.shape[0]
        
        if dist_batch_size != action_batch_size:
            # If batch sizes don't match, slice the distribution to match actions
            # This can happen when actions are augmented or come from different batches
            if dist_batch_size >= action_batch_size:
                # Distribution has enough samples, use only the first action_batch_size
                placement_dist_loc = self.placement_dist.loc[:action_batch_size]
                placement_dist_scale = self.placement_dist.scale[:action_batch_size]
                placement_dist = Normal(placement_dist_loc, placement_dist_scale)
                orientation_dist_logits = self.orientation_dist.logits[:action_batch_size]
                orientation_dist = Categorical(logits=orientation_dist_logits)
            else:
                # Actions have more samples than distribution - this shouldn't happen normally
                # Use the full distribution and pad/truncate actions (shouldn't reach here)
                placement_dist = self.placement_dist
                orientation_dist = self.orientation_dist
                xy = xy[:dist_batch_size]
                orientation = orientation[:dist_batch_size]
        else:
            placement_dist = self.placement_dist
            orientation_dist = self.orientation_dist
        
        # Compute log probabilities
        log_prob_xy = placement_dist.log_prob(xy).sum(dim=-1)
        log_prob_o = orientation_dist.log_prob(orientation)
        
        # Total log probability
        log_prob_total = log_prob_xy + log_prob_o
        
        # Don't reshape back to 3D - keep it flattened to match other flattened tensors in PPO
        # The PPO update code expects flattened tensors, so we keep it as 1D/2D
        # If reshape is needed elsewhere, it should be done at the call site
        
        return log_prob_total

    @property
    def action_mean(self):
        """Return the mean action as [x, y, onehot_orientation_0, onehot_orientation_1]."""
        # Mean placement (continuous)
        xy_mean = self.placement_dist.mean  # [batch, 2]

        # Orientation: argmax logits → one-hot
        orientation = torch.argmax(self.orientation_dist.logits, dim=-1)  # [batch]
        orientation_onehot = torch.zeros_like(self.orientation_dist.logits)  # [batch, 2]
        orientation_onehot.scatter_(1, orientation.unsqueeze(1), 1.0)

        # Combine into full mean action
        return torch.cat([xy_mean, orientation_onehot], dim=-1)  # [batch, 4]

    @property
    def action_std(self):
        """Return std for placement only; keep categorical entropy separately."""
        return self.placement_dist.stddev

    @property
    def entropy(self):
        """Compute total entropy of both distributions."""
        if self.placement_dist is None or self.orientation_dist is None:
            return torch.tensor(0.0)
        
        placement_entropy = self.placement_dist.entropy().sum(dim=-1)
        orientation_entropy = self.orientation_dist.entropy()
        
        return placement_entropy + orientation_entropy

    @property
    def placement_entropy(self):
        """Compute entropy of placement distribution only."""
        if self.placement_dist is None:
            if self.device is None:
                self.device = next(self.parameters()).device
            return torch.tensor(0.0, device=self.device)
        return self.placement_dist.entropy().sum(dim=-1)

    @property
    def orientation_entropy(self):
        """Compute entropy of orientation distribution only."""
        if self.orientation_dist is None:
            if self.device is None:
                self.device = next(self.parameters()).device
            return torch.tensor(0.0, device=self.device)
        return self.orientation_dist.entropy()

    def act_inference(self, observations):
        """Get deterministic actions for inference."""
        # Process full observations (proprio + image) through RNN
        rnn_output = self.memory_a(observations)
        rnn_output = rnn_output.squeeze(0)  # [batch, rnn_hidden_dim]
        
        # Split observations to get image part
        image_obs = observations[:, self.num_proprio_obs:]
        
        # Concatenate RNN output with image observations to pass to actor
        actor_input = torch.cat([rnn_output, image_obs], dim=1)
        
        action_raw = self.actor(actor_input)
        
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

    def compute_kl_divergence(self, mu, sigma, old_mu, old_sigma):
        """Compute KL divergence between old and current policy (Gaussian + categorical)."""
        if self.placement_dist is None or self.orientation_dist is None:
            return torch.tensor(0.0, device=self.device)

        # Handle 3D tensors if needed (from recurrent storage format)
        original_mu_shape = mu.shape
        original_sigma_shape = sigma.shape
        original_old_mu_shape = old_mu.shape
        original_old_sigma_shape = old_sigma.shape
        
        # Flatten mu and old_mu if they are 3D
        if mu.dim() == 3:
            mu = mu.reshape(-1, mu.shape[-1])
        if old_mu.dim() == 3:
            old_mu = old_mu.reshape(-1, old_mu.shape[-1])
        
        # Flatten sigma and old_sigma if they are 3D
        if sigma.dim() == 3:
            sigma = sigma.reshape(-1, sigma.shape[-1])
        if old_sigma.dim() == 3:
            old_sigma = old_sigma.reshape(-1, old_sigma.shape[-1])
        
        # Ensure batch sizes match (take minimum if they don't)
        mu_batch_size = mu.shape[0]
        old_mu_batch_size = old_mu.shape[0]
        if mu_batch_size != old_mu_batch_size:
            min_batch = min(mu_batch_size, old_mu_batch_size)
            mu = mu[:min_batch]
            old_mu = old_mu[:min_batch]
        
        sigma_batch_size = sigma.shape[0]
        old_sigma_batch_size = old_sigma.shape[0]
        if sigma_batch_size != old_sigma_batch_size:
            min_batch = min(sigma_batch_size, old_sigma_batch_size)
            sigma = sigma[:min_batch]
            old_sigma = old_sigma[:min_batch]

        # ---- Gaussian part ----
        mu_xy = mu[:, :2]
        sigma_xy = sigma[:, :2]
        old_mu_xy = old_mu[:, :2]
        old_sigma_xy = old_sigma[:, :2]

        kl_placement = torch.sum(
            torch.log((sigma_xy + 1e-8) / (old_sigma_xy + 1e-8))
            + (old_sigma_xy.pow(2) + (old_mu_xy - mu_xy).pow(2))
            / (2.0 * sigma_xy.pow(2))
            - 0.5,
            dim=-1,
        )

        # ---- Categorical part ----
        current_logits_o = mu[:, 2:]
        old_logits_o = old_mu[:, 2:]

        old_probs_o = torch.softmax(old_logits_o, dim=-1).detach()
        current_probs_o = torch.softmax(current_logits_o, dim=-1)


        kl_orientation = torch.sum(
            current_probs_o * (torch.log(current_probs_o + 1e-8) - torch.log(old_probs_o + 1e-8)),
            dim=-1,
        )

        # ---- Combine ----
        total_kl = kl_placement + kl_orientation
        
        # Reshape back to original shape if needed
        # Only reshape if the total_kl size matches the expected reshape size
        if len(original_old_mu_shape) == 3:
            expected_size = original_old_mu_shape[0] * original_old_mu_shape[1]
            if total_kl.numel() == expected_size:
                total_kl = total_kl.reshape(original_old_mu_shape[0], original_old_mu_shape[1])
        elif len(original_mu_shape) == 3:
            expected_size = original_mu_shape[0] * original_mu_shape[1]
            if total_kl.numel() == expected_size:
                total_kl = total_kl.reshape(original_mu_shape[0], original_mu_shape[1])
        
        return total_kl
    
    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """Evaluate critic with recurrent memory."""
        # Process full observations (proprio + image) through RNN
        rnn_output = self.memory_c(critic_observations, masks, hidden_states)
        
        # Extract image observations - need to handle batch mode (with masks) vs inference mode
        if masks is not None:
            # Batch mode: observations is [seq_len, batch, obs_features]
            # RNN output after unpad_trajectories has shape [seq_len, num_valid, hidden_dim]
            # Extract image part from full observations: [seq_len, batch, image_features]
            image_obs_padded = critic_observations[:, :, self.num_proprio_obs:]
            # Apply same unpad logic as RNN output
            image_obs = unpad_trajectories(image_obs_padded, masks)
            # Both rnn_output and image_obs have shape [seq_len, num_valid, features]
            # They should have the same seq_len and num_valid since they use the same masks
            seq_len = rnn_output.shape[0]
            num_valid = rnn_output.shape[1]
            
            # Ensure shapes match
            assert image_obs.shape[0] == seq_len, \
                f"Image obs seq_len {image_obs.shape[0]} doesn't match RNN seq_len {seq_len}"
            assert image_obs.shape[1] == num_valid, \
                f"Image obs num_valid {image_obs.shape[1]} doesn't match RNN num_valid {num_valid}"
            
            # Flatten to [seq_len * num_valid, features] (which equals [total_batch, features])
            rnn_output = rnn_output.reshape(-1, rnn_output.shape[-1])  # [seq_len * num_valid, rnn_hidden_dim]
            image_obs = image_obs.reshape(-1, image_obs.shape[-1])  # [seq_len * num_valid, image_features]
        else:
            # Inference mode: observations is [batch, obs_features]
            rnn_output = rnn_output.squeeze(0)  # [batch, rnn_hidden_dim]
            image_obs = critic_observations[:, self.num_proprio_obs:]  # [batch, image_features]
        
        # Concatenate RNN output with image observations to pass to critic
        critic_input = torch.cat([rnn_output, image_obs], dim=1)
        
        value = self.critic(critic_input)
        return value

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)