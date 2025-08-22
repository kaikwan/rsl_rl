# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunnerConv2d



class GCUOnPolicyRunner(OnPolicyRunnerConv2d):
    """Runner for on-policy algorithms in GCU environments."""
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        # For K=4 mixture components: 2 (placement means) + 2 (placement log-stds) + 4 (logits) + 24 (rotation means) + 24 (rotation log-stds) = 56
        env.num_actions = 8  # Override the number of actions for mixture-of-Gaussians policy
        super().__init__(env, train_cfg, log_dir, device)



