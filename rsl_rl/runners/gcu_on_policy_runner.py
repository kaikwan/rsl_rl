# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import statistics
import time
import torch
from collections import deque
import multiprocessing as mp
from functools import partial
import numpy as np

import rsl_rl
from rsl_rl.algorithms import PPO, Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from rsl_rl.utils import store_code_state
from rsl_rl.runners import OnPolicyRunnerConv2d

import rsl_rl.utils.bpp_utils as bpp_utils
from tote_consolidation.tasks.manager_based.pack.utils.tote_helpers import (
    calculate_rotated_bounding_box,
)
from packing3d import (Attitude, Transform, Position)
import matplotlib.pyplot as plt


class GCUOnPolicyRunner(OnPolicyRunnerConv2d):
    """Runner for on-policy algorithms in GCU environments."""
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        env.num_actions = 4 # Override the number of actions to 4 fpr policy
        super().__init__(env, train_cfg, log_dir, device)
        self.tote_manager = env.unwrapped.tote_manager
        self.num_obj_per_env = self.tote_manager.num_objects

        args = {
            "decreasing_vol": False,  # Whether to use decreasing volume for packing
            "use_stability": False,  # Whether to use stability checks for packing
            "use_subset_sum": False,  # Whether to use subset sum for packing
            "use_multiprocessing": False,  # Disable multiprocessing to avoid CUDA issues
            "max_workers": 20,  # Use single worker when multiprocessing is enabled
        }

        self.bpp = bpp_utils.BPP(
            self.tote_manager, env.num_envs, torch.arange(self.num_obj_per_env, device=env.unwrapped.device), **args
        )
        env.unwrapped.bpp = self.bpp


