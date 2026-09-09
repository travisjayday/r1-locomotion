# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Translate ProtoMotions actuator descriptions into MuJoCo joint torques."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class ActuatorModelTranslation:
    """Vectorized PD actuator model in MuJoCo DOF order.

    ProtoMotions robot configurations describe actuator behavior independently
    of a simulator. IsaacLab consumes those descriptions through actuator
    classes. MuJoCo position actuators cannot represent torque-speed envelopes,
    so this class applies the same equations before writing motor torques.
    """

    _IDEAL = 0
    _UNITREE = 1

    def __init__(self, control_info: Mapping[str, Any], dof_names: Sequence[str]):
        infos = []
        missing = []
        for name in dof_names:
            info = control_info.get(name)
            if info is None:
                missing.append(name)
            else:
                infos.append(info)
        if missing:
            raise KeyError(f"Missing control information for MuJoCo DOFs: {missing}")

        self.dof_names = tuple(dof_names)
        self.kp = self._array(infos, "stiffness", required=True)
        self.kd = self._array(infos, "damping", required=True)
        self.effort_limit = self._array(infos, "effort_limit", default=np.inf)

        model_names = [getattr(info, "actuator_model", None) for info in infos]
        unsupported = sorted(
            {name for name in model_names if name not in (None, "unitree")}
        )
        if unsupported:
            raise ValueError(
                "Unsupported MuJoCo actuator model(s): " + ", ".join(unsupported)
            )

        self.model_kind = np.array(
            [
                self._UNITREE if name == "unitree" else self._IDEAL
                for name in model_names
            ],
            dtype=np.int8,
        )
        self.has_custom_models = bool(np.any(self.model_kind != self._IDEAL))

        self.x1 = self._array(infos, "torque_speed_x1", default=np.inf)
        self.x2 = self._array(infos, "torque_speed_x2", default=np.inf)
        self.y1 = self._array(
            infos, "torque_same_direction", fallback=self.effort_limit
        )
        self.y2 = self._array(
            infos, "torque_opposite_direction", fallback=self.effort_limit
        )
        self.friction_static = self._array(
            infos, "friction_static", default=0.0
        )
        self.friction_dynamic = self._array(
            infos, "friction_dynamic", default=0.0
        )
        self.friction_activation_velocity = self._array(
            infos, "friction_activation_velocity", default=0.01
        )

        unitree = self.model_kind == self._UNITREE
        invalid_envelope = unitree & (
            ~np.isfinite(self.x1)
            | ~np.isfinite(self.x2)
            | (self.x2 <= self.x1)
            | ~np.isfinite(self.y1)
            | ~np.isfinite(self.y2)
        )
        if np.any(invalid_envelope):
            names = [self.dof_names[index] for index in np.flatnonzero(invalid_envelope)]
            raise ValueError(f"Invalid Unitree torque-speed parameters for: {names}")
        if np.any(unitree & (self.friction_activation_velocity <= 0.0)):
            raise ValueError("Unitree friction activation velocity must be positive")

    @staticmethod
    def _array(
        infos: Sequence[Any],
        field: str,
        *,
        default: float | None = None,
        fallback: np.ndarray | None = None,
        required: bool = False,
    ) -> np.ndarray:
        values = []
        for index, info in enumerate(infos):
            value = getattr(info, field, None)
            if value is None and fallback is not None:
                value = fallback[index]
            if value is None:
                if required:
                    raise ValueError(f"Actuator parameter '{field}' is required")
                value = default
            values.append(float(value))
        return np.asarray(values, dtype=np.float64)

    def compute(
        self,
        targets: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
    ) -> np.ndarray:
        """Compute actuator torque using the IsaacLab Unitree equations."""
        targets = np.asarray(targets, dtype=np.float64)
        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        joint_vel = np.asarray(joint_vel, dtype=np.float64)
        expected_shape = self.kp.shape
        for name, value in (
            ("targets", targets),
            ("joint_pos", joint_pos),
            ("joint_vel", joint_vel),
        ):
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {value.shape}, expected {expected_shape}"
                )

        raw_effort = self.kp * (targets - joint_pos) - self.kd * joint_vel
        limit = self.effort_limit.copy()

        unitree = self.model_kind == self._UNITREE
        if np.any(unitree):
            indices = np.flatnonzero(unitree)
            unitree_velocity = joint_vel[indices]
            same_direction = unitree_velocity * raw_effort[indices] > 0.0
            envelope_limit = np.where(
                same_direction, self.y1[indices], self.y2[indices]
            )
            slope = -envelope_limit / (self.x2[indices] - self.x1[indices])
            speed_limited = np.maximum(
                slope * (np.abs(unitree_velocity) - self.x1[indices])
                + envelope_limit,
                0.0,
            )
            envelope_limit = np.where(
                np.abs(unitree_velocity) < self.x1[indices],
                envelope_limit,
                speed_limited,
            )
            limit[indices] = envelope_limit

        effort = np.clip(raw_effort, -limit, limit)
        if np.any(unitree):
            friction = self.friction_static * np.tanh(
                joint_vel / self.friction_activation_velocity
            ) + self.friction_dynamic * joint_vel
            effort[unitree] -= friction[unitree]

        # IsaacLab also configures the physics-side effort limit. Preserve that
        # final safety clamp after the explicit motor-friction term.
        return np.clip(effort, -self.effort_limit, self.effort_limit)


__all__ = ["ActuatorModelTranslation"]
