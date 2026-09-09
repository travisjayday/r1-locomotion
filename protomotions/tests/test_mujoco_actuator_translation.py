# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

from protomotions.simulator.mujoco.actuator_translation import (
    ActuatorModelTranslation,
)
from protomotions.simulator.mujoco.realtime import RealTimePacer


def _info(**overrides):
    values = {
        "stiffness": 10.0,
        "damping": 1.0,
        "effort_limit": 5.0,
        "actuator_model": None,
        "torque_speed_x1": None,
        "torque_speed_x2": None,
        "torque_same_direction": None,
        "torque_opposite_direction": None,
        "friction_static": None,
        "friction_dynamic": None,
        "friction_activation_velocity": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ideal_translation_computes_and_clips_pd_torque():
    model = ActuatorModelTranslation({"joint": _info()}, ["joint"])

    torque = model.compute(
        targets=np.array([1.0]),
        joint_pos=np.array([0.0]),
        joint_vel=np.array([2.0]),
    )

    assert model.has_custom_models is False
    np.testing.assert_allclose(torque, [5.0])


def test_unitree_translation_matches_torque_speed_and_friction_equations():
    info = _info(
        stiffness=84.3355,
        damping=5.369,
        effort_limit=66.7,
        actuator_model="unitree",
        torque_speed_x1=8.4,
        torque_speed_x2=15.3,
        torque_same_direction=53.7,
        torque_opposite_direction=66.7,
        friction_static=0.6,
        friction_dynamic=0.06,
        friction_activation_velocity=0.01,
    )
    model = ActuatorModelTranslation({"joint": info}, ["joint"])

    torque = model.compute(
        targets=np.array([2.0]),
        joint_pos=np.array([0.0]),
        joint_vel=np.array([10.0]),
    )

    envelope = 53.7 - 53.7 / (15.3 - 8.4) * (10.0 - 8.4)
    friction = 0.6 * np.tanh(10.0 / 0.01) + 0.06 * 10.0
    assert model.has_custom_models is True
    np.testing.assert_allclose(torque, [envelope - friction])


def test_translation_rejects_unknown_actuator_models():
    with pytest.raises(ValueError, match="Unsupported MuJoCo actuator model"):
        ActuatorModelTranslation(
            {"joint": _info(actuator_model="unimplemented")}, ["joint"]
        )


class _FakeClock:
    def __init__(self):
        self.now = 10.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


def test_realtime_pacer_sleeps_when_ahead_and_rebases_when_far_behind():
    clock = _FakeClock()
    pacer = RealTimePacer(clock=clock, sleeper=clock.sleep, max_lag_s=0.1)
    pacer.reset(3.0)

    pacer.wait(3.02)
    assert clock.sleeps == pytest.approx([0.02])

    clock.now += 0.5
    pacer.wait(3.04)
    assert pacer._wall_origin == pytest.approx(clock.now)
    assert pacer._sim_origin == pytest.approx(3.04)


def test_realtime_pacer_validates_configuration():
    with pytest.raises(ValueError, match="real_time_factor"):
        RealTimePacer(real_time_factor=0.0)
    with pytest.raises(ValueError, match="max_lag_s"):
        RealTimePacer(max_lag_s=-0.1)
