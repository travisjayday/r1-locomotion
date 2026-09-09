# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-bucketed evaluator for body-frame velocity steering."""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.config import SteeringEvaluatorConfig
from protomotions.utils import rotations


class SteeringEvaluator(BaseEvaluator):
    """Report tracking quality separately for behaviorally important commands.

    Buckets are intentionally overlapping: for example, a fast backward turn is
    represented in ``backward``, ``high_speed``, and ``high_yaw_rate``. This
    keeps a strong forward average from concealing a failed deployment mode.
    """

    config: SteeringEvaluatorConfig

    def initialize_eval(self) -> Dict:
        metrics = super().initialize_eval()
        self._bucket_step = 0
        self._bucket_stats = {
            name: torch.zeros(8, device=self.device, dtype=torch.float64)
            for name in (
                "all",
                "stand",
                "turn_in_place",
                "forward",
                "backward",
                "lateral",
                "high_speed",
                "high_yaw_rate",
            )
        }
        return metrics

    def _on_episode_step(self, env_ids: Tensor, extras: Dict, actions: Tensor) -> None:
        step = self._bucket_step
        self._bucket_step += 1
        if step < self.config.bucket_warmup_steps:
            return

        context = self.env.context
        target_velocity = context.steering.tar_local_vel[env_ids]
        target_yaw_rate = context.steering.tar_yaw_rate[env_ids]
        current = context.current
        heading_inv = rotations.calc_heading_quat_inv(
            current.anchor_rot[env_ids], w_last=True
        )
        actual_velocity = rotations.quat_rotate(
            heading_inv, current.anchor_vel[env_ids], w_last=True
        )[:, :2]
        actual_yaw_rate = current.anchor_ang_vel[env_ids, 2]

        target_speed = torch.linalg.vector_norm(target_velocity, dim=-1)
        linear_error = torch.linalg.vector_norm(
            actual_velocity - target_velocity, dim=-1
        )
        yaw_error = torch.abs(actual_yaw_rate - target_yaw_rate)
        tracking_success = (
            (linear_error <= self.config.linear_velocity_tolerance)
            & (yaw_error <= self.config.yaw_rate_tolerance)
        )
        is_moving = target_speed > 0.1
        has_yaw = torch.abs(target_yaw_rate) > 0.1
        masks = {
            "all": torch.ones_like(is_moving),
            "stand": (~is_moving) & (~has_yaw),
            "turn_in_place": (~is_moving) & has_yaw,
            "forward": is_moving & (target_velocity[:, 0] > 0.1),
            "backward": is_moving & (target_velocity[:, 0] < -0.1),
            "lateral": is_moving
            & (torch.abs(target_velocity[:, 1]) >= torch.abs(target_velocity[:, 0])),
            "high_speed": target_speed >= self.config.high_speed_threshold,
            "high_yaw_rate": (
                torch.abs(target_yaw_rate) >= self.config.high_yaw_rate_threshold
            ),
        }

        # [count, linear error, yaw error, successful steps, target vx,
        #  actual vx, target speed, actual speed]
        actual_speed = torch.linalg.vector_norm(actual_velocity, dim=-1)
        for name, mask in masks.items():
            if not torch.any(mask):
                continue
            stats = self._bucket_stats[name]
            stats[0] += mask.sum()
            stats[1] += linear_error[mask].double().sum()
            stats[2] += yaw_error[mask].double().sum()
            stats[3] += tracking_success[mask].double().sum()
            stats[4] += target_velocity[mask, 0].double().sum()
            stats[5] += actual_velocity[mask, 0].double().sum()
            stats[6] += target_speed[mask].double().sum()
            stats[7] += actual_speed[mask].double().sum()

    def process_eval_results(self) -> Tuple[Dict, Optional[float], int]:
        to_log, score, num_eval_items = super().process_eval_results()
        metric_names = (
            "sample_count",
            "linear_velocity_mae",
            "yaw_rate_mae",
            "tracking_fraction",
            "target_vx_mean",
            "actual_vx_mean",
            "target_speed_mean",
            "actual_speed_mean",
        )
        for bucket, stats in self._bucket_stats.items():
            count = stats[0].item()
            prefix = f"eval/bucket/{bucket}"
            to_log[f"{prefix}/{metric_names[0]}"] = count
            if count == 0:
                continue
            for index, metric_name in enumerate(metric_names[1:], start=1):
                to_log[f"{prefix}/{metric_name}"] = stats[index].item() / count
        return to_log, score, num_eval_items

    def cleanup_after_evaluation(self) -> None:
        self._bucket_stats = {}
        super().cleanup_after_evaluation()
