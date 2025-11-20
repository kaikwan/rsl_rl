# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization, GCUActorCriticConv2dPointNet
from rsl_rl.runners import OnPolicyRunnerConv2d
from rsl_rl.utils import store_code_state
from rsl_rl.utils import normalize_and_flatten_image_obs

class OnPolicyRunnerConv2dPointNet(OnPolicyRunnerConv2d):
    """Custom on-policy runner for training and evaluation with convolutional actor-critic with PointNet."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve training type depending on the algorithm
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        # resolve type of privileged observations
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"  # actor-critic reinforcement learnig, e.g., PPO
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"  # policy distillation
            else:
                self.privileged_obs_type = None

        # resolve dimensions of privileged observations
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # resolve dimension of rnd gated state
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env
            
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        # Convert from [N, H, W, C] to [C, H, W]
        input_image_shape = extras["observations"]["sensor"].permute(0, 3, 1, 2).shape[1:]
        num_image_obs = torch.prod(torch.tensor(input_image_shape)).item()

        # init the actor-critic networks
        actor_critic: GCUActorCriticConv2dPointNet = GCUActorCriticConv2dPointNet(
            num_obs, num_critic_obs, self.env.num_actions, input_image_shape, **self.policy_cfg
        ).to(self.device)

        # init the ppo algorithmw
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
            self.image_obs_normalizer = EmpiricalNormalization(shape=[num_image_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            
        # init storage and model
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs + num_image_obs],
            [num_critic_obs + num_image_obs],
            [self.env.num_actions],
        )

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        self._reward_hist_window = self.cfg.get("reward_histogram_window", 128)
        if self.log_dir is not None and not self.disable_logs:
            self._episode_reward_traces = [[] for _ in range(self.env.num_envs)]
            self._historical_reward_traces = deque(maxlen=self._reward_hist_window)
        else:
            self._episode_reward_traces = None
            self._historical_reward_traces = None
        self._hist_episode_counter = 0
