# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Unitree torque-speed/friction actuator, adapted from WBC-AGILE."""

from __future__ import annotations

from dataclasses import MISSING

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions


class UnitreeActuator(DelayedPDActuator):
    cfg: "UnitreeActuatorCfg"

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._effort_y1 = self._parse_joint_parameter(cfg.Y1, 1e9)
        self._effort_y2 = self._parse_joint_parameter(cfg.Y2, cfg.Y1)
        self._velocity_x1 = self._parse_joint_parameter(cfg.X1, 1e9)
        self._velocity_x2 = self._parse_joint_parameter(cfg.X2, 1e9)
        self._friction_static = self._parse_joint_parameter(cfg.Fs, 0.0)
        self._friction_dynamic = self._parse_joint_parameter(cfg.Fd, 0.0)
        self._activation_vel = self._parse_joint_parameter(cfg.Va, 0.01)
        # Per-environment multiplicative voltage/torque uncertainty. The
        # simulator's DR pipeline updates this tensor at curriculum changes.
        self.motor_strength_scale = torch.ones_like(self.applied_effort)

    def compute(self, control_action: ArticulationActions, joint_pos, joint_vel):
        self._joint_vel[:] = joint_vel
        control_action = super().compute(control_action, joint_pos, joint_vel)
        self.applied_effort -= self._friction_static * torch.tanh(
            joint_vel / self._activation_vel
        ) + self._friction_dynamic * joint_vel
        self.applied_effort *= self.motor_strength_scale
        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort
        return control_action

    def _clip_effort(self, effort):
        same_direction = self._joint_vel * effort > 0
        limit = torch.where(same_direction, self._effort_y1, self._effort_y2)
        slope = -limit / (self._velocity_x2 - self._velocity_x1)
        speed_limited = (
            slope * (self._joint_vel.abs() - self._velocity_x1) + limit
        ).clip(min=0.0)
        limit = torch.where(self._joint_vel.abs() < self._velocity_x1, limit, speed_limited)
        return torch.clip(effort, -limit, limit)


@configclass
class UnitreeActuatorCfg(DelayedPDActuatorCfg):
    class_type: type = UnitreeActuator
    X1: float = 1e9
    X2: float = 1e9
    Y1: float = MISSING
    Y2: float | None = None
    Fs: float = 0.0
    Fd: float = 0.0
    Va: float = 0.01

    def __post_init__(self):
        if self.Y2 is None:
            self.Y2 = self.Y1
        self.effort_limit_sim = self.Y2
        self.velocity_limit_sim = self.X2
