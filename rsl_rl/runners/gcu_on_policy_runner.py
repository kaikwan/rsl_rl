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



class GCUOnPolicyRunner(OnPolicyRunnerConv2d):
    """Runner for on-policy algorithms in GCU environments."""
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        env.num_actions = 5 # Override the number of actions to 5 fpr policy
        super().__init__(env, train_cfg, log_dir, device)



