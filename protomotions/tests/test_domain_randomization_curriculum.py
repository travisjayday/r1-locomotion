# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from protomotions.simulator.base_simulator.config import (
    ActuatorDomainRandomizationConfig,
    ActionNoiseDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    DomainRandomizationConfig,
    DomainRandomizationCurriculumStageConfig,
    FrictionDomainRandomizationConfig,
    PushDomainRandomizationConfig,
    RigidBodyDomainRandomizationConfig,
    RobotNoiseConfig,
)
from protomotions.simulator.base_simulator.simulator import Simulator


def _config():
    return DomainRandomizationConfig(
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(-0.02, 0.02), dof_indices=[0, 1]
        ),
        friction=FrictionDomainRandomizationConfig(
            static_friction_range=(0.6, 1.4),
            dynamic_friction_range=(0.5, 1.3),
            restitution_range=(0.0, 0.08),
            body_indices=[0],
        ),
        center_of_mass=CenterOfMassDomainRandomizationConfig(
            com_range={"x": (-0.02, 0.02), "y": (-0.03, 0.03), "z": (-0.01, 0.01)},
            body_indices=[0],
        ),
        observation_noise=RobotNoiseConfig(
            dof_pos_noise=0.01,
            root_rot_noise=[0.02, 0.01, 0.04],
        ),
        push=PushDomainRandomizationConfig(
            max_linear_velocity=(0.3, 0.2, 0.1),
            max_angular_velocity=(0.2, 0.2, 0.4),
        ),
        actuator=ActuatorDomainRandomizationConfig(
            stiffness_scale_range=(0.8, 1.2),
            damping_scale_range=(0.7, 1.3),
            motor_strength_scale_range=(0.9, 1.1),
            delay_steps_range=(0, 2),
            dof_indices=[0, 1],
        ),
        rigid_body=RigidBodyDomainRandomizationConfig(
            mass_scale_range=(0.9, 1.1),
            inertia_scale_range=(0.8, 1.2),
            body_indices=[0],
        ),
        curriculum=[
            DomainRandomizationCurriculumStageConfig(0, 0.5),
            DomainRandomizationCurriculumStageConfig(10, 1.0),
        ],
    )


def test_domain_randomization_curriculum_scales_about_physical_nominals():
    config = _config()
    mild = config.resolved_for_epoch(0)
    final = config.resolved_for_epoch(10)

    assert mild.action_noise.action_noise_range == pytest.approx((-0.01, 0.01))
    assert mild.friction.static_friction_range == pytest.approx((0.8, 1.2))
    assert mild.friction.restitution_range == pytest.approx((0.0, 0.04))
    assert mild.center_of_mass.com_range["y"] == pytest.approx((-0.015, 0.015))
    assert mild.observation_noise.dof_pos_noise == pytest.approx(0.005)
    assert mild.observation_noise.root_rot_noise == pytest.approx([0.01, 0.005, 0.02])
    assert mild.push.max_linear_velocity == pytest.approx((0.15, 0.1, 0.05))
    assert mild.actuator.stiffness_scale_range == pytest.approx((0.9, 1.1))
    assert mild.actuator.delay_steps_range == (0, 1)
    assert mild.rigid_body.mass_scale_range == pytest.approx((0.95, 1.05))
    assert final.action_noise.action_noise_range == pytest.approx((-0.02, 0.02))
    assert final.actuator.delay_steps_range == (0, 2)
    assert final.curriculum is None
    assert config.curriculum is not None


def test_domain_randomization_curriculum_validation():
    with pytest.raises(ValueError, match="start at epoch 0"):
        DomainRandomizationConfig(
            curriculum=[DomainRandomizationCurriculumStageConfig(1, 0.5)]
        )
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        DomainRandomizationCurriculumStageConfig(0, 1.1)
    with pytest.raises(ValueError, match="positive"):
        ActuatorDomainRandomizationConfig(
            stiffness_scale_range=(0.0, 1.0), dof_indices=[0]
        )


def test_actuator_and_rigid_body_processing_samples_expected_shapes():
    simulator = SimpleNamespace(
        num_envs=8,
        device=torch.device("cpu"),
        _dof_names=["hip", "knee", "head"],
        _body_names=["pelvis", "foot"],
    )

    actuator = Simulator._process_actuator_domain_randomization(
        simulator,
        ActuatorDomainRandomizationConfig(
            stiffness_scale_range=(0.8, 1.2),
            damping_scale_range=(0.7, 1.3),
            motor_strength_scale_range=(0.9, 1.1),
            delay_steps_range=(0, 2),
            dof_names=["hip|knee"],
        ),
    )
    rigid = Simulator._process_rigid_body_domain_randomization(
        simulator,
        RigidBodyDomainRandomizationConfig(
            mass_scale_range=(0.9, 1.1),
            inertia_scale_range=(0.8, 1.2),
            body_names=[".*"],
        ),
    )

    assert actuator["dof_indices"] == [0, 1]
    assert actuator["stiffness_scale"].shape == (8, 2)
    assert actuator["delay_steps"].shape == (8,)
    assert torch.all((actuator["delay_steps"] >= 0) & (actuator["delay_steps"] <= 2))
    assert rigid["body_indices"] == [0, 1]
    assert rigid["mass_scale"].shape == (8, 2)


def test_epoch_transition_resamples_and_calls_backend_once():
    simulator = SimpleNamespace()
    simulator._domain_randomization_config = _config()
    simulator._domain_randomization_stage_index = 0
    simulator._domain_randomization_stage_metric = torch.tensor(0.0)
    simulator._domain_randomization_intensity_metric = torch.tensor(0.5)
    simulator._active_domain_randomization_config = (
        simulator._domain_randomization_config.resolved_for_epoch(0)
    )
    calls = []
    simulator._process_domain_randomization = lambda config: {"config": config}
    simulator._configure_active_push_randomization = lambda: calls.append("push")
    simulator._apply_robot_domain_randomization = lambda: calls.append("robot")

    Simulator.on_epoch_end(simulator, 9)
    assert calls == []

    Simulator.on_epoch_end(simulator, 10)
    assert calls == ["push", "robot"]
    assert simulator._domain_randomization_stage_index == 1
    assert simulator._domain_randomization_stage_metric.item() == 1.0
    assert simulator._domain_randomization_intensity_metric.item() == 1.0
    assert simulator._domain_randomization["config"].actuator.delay_steps_range == (0, 2)
