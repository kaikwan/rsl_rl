# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        # Initialize wandb
        wandb.init(project=project, entity=entity, name=run_name)

        # Add log directory to wandb
        wandb.config.update({"log_dir": log_dir})

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        wandb.config.update({"runner_cfg": runner_cfg})
        wandb.config.update({"policy_cfg": policy_cfg})
        wandb.config.update({"alg_cfg": alg_cfg})
        try:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            wandb.config.update({"env_cfg": asdict(env_cfg)})

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({self._map_path(tag): scalar_value}, step=global_step)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats="CHW"):
        super().add_image(tag, img_tensor, global_step=global_step, walltime=walltime, dataformats=dataformats)
        # Convert tensor to numpy for WandB
        import torch
        import numpy as np
        
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.detach().cpu().numpy()
        else:
            img_np = np.array(img_tensor)
        
        # Handle different data formats
        if dataformats == "CHW":
            # Convert from CHW to HWC for WandB
            if img_np.ndim == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            # If single channel, repeat to RGB
            if img_np.ndim == 2:
                img_np = np.expand_dims(img_np, -1)
            if img_np.shape[-1] == 1:
                img_np = np.repeat(img_np, 3, axis=-1)
        elif dataformats == "HWC":
            # Already in HWC format
            if img_np.ndim == 2:
                img_np = np.expand_dims(img_np, -1)
            if img_np.shape[-1] == 1:
                img_np = np.repeat(img_np, 3, axis=-1)
        
        # Normalize to [0, 255] if needed
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        wandb.log({self._map_path(tag): wandb.Image(img_np)}, step=global_step)

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        # Convert matplotlib figure to image for WandB (to avoid Plotly conversion issues)
        import io
        import numpy as np
        
        # Render figure to buffer
        buf = io.BytesIO()
        figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to numpy array
        try:
            from PIL import Image
            img = Image.open(buf)
            img_np = np.array(img)
            buf.close()
        except ImportError:
            # Skip WandB logging if PIL not available
            buf.close()
            img_np = None
        
        if img_np is not None:
            # Log to WandB as image
            wandb.log({self._map_path(tag): wandb.Image(img_np)}, step=global_step)
        
        # Then log to TensorBoard (which may close the figure)
        super().add_figure(tag, figure, global_step=global_step, close=close, walltime=walltime)

    def stop(self):
        wandb.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        wandb.save(path, base_path=os.path.dirname(path))

    """
    Private methods.
    """

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path
