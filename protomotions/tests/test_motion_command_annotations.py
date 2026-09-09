# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch

from protomotions.utils.motion_command_annotations import (
    ReferenceCommandTrajectory,
    SimulationRootAnnotationConfig,
    annotate_motion_clip,
    annotate_packed_motion_library,
)


def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    rotation = torch.zeros(len(yaw), 4)
    rotation[:, 2] = torch.sin(0.5 * yaw)
    rotation[:, 3] = torch.cos(0.5 * yaw)
    return rotation


def _config(**updates) -> SimulationRootAnnotationConfig:
    values = {
        "position_body_index": 1,
        "facing_body_index": 0,
        "position_body_name": "spine",
        "facing_body_name": "pelvis",
        "smoothing_window_s": 0.75,
        "polynomial_order": 3,
    }
    values.update(updates)
    return SimulationRootAnnotationConfig(**values)


def test_simulation_root_suppresses_step_scale_position_oscillation():
    dt = 0.02
    time = torch.arange(251) * dt
    body_pos = torch.zeros(len(time), 2, 3)
    body_pos[:, 1, 0] = 0.6 * time + 0.035 * torch.sin(4.0 * torch.pi * time)
    body_rot = torch.zeros(len(time), 2, 4)
    body_rot[..., 3] = 1.0

    annotation = annotate_motion_clip(body_pos, body_rot, dt, _config())

    raw_velocity = torch.gradient(body_pos[:, 1, 0], spacing=(dt,))[0]
    command_velocity = annotation.motion_commands[:, 0]
    interior = slice(30, -30)
    assert command_velocity[interior].mean().item() == pytest.approx(0.6, abs=0.02)
    assert command_velocity[interior].std() < 0.2 * raw_velocity[interior].std()
    assert torch.max(torch.abs(annotation.motion_commands[:, 1:])) < 1.0e-5


def test_linear_extrapolation_preserves_nonzero_endpoint_velocity():
    frames = 151
    dt = 0.02
    time = torch.arange(frames) * dt
    body_pos = torch.zeros((frames, 2, 3))
    body_pos[:, 1, 0] = 1.25 * time
    body_rot = torch.zeros((frames, 2, 4))
    body_rot[..., 3] = 1.0

    annotation = annotate_motion_clip(body_pos, body_rot, dt, _config())

    assert torch.allclose(
        annotation.simulation_root_vel[[0, -1], 0],
        torch.tensor([1.25, 1.25]),
        atol=1.0e-4,
    )


def test_projected_facing_stays_continuous_across_yaw_wrap():
    dt = 0.02
    time = torch.arange(201) * dt
    unwrapped_yaw = 3.0 + 0.3 * time
    body_pos = torch.zeros(len(time), 2, 3)
    body_rot = torch.zeros(len(time), 2, 4)
    body_rot[..., 3] = 1.0
    body_rot[:, 0] = _yaw_quaternion(unwrapped_yaw)

    annotation = annotate_motion_clip(body_pos, body_rot, dt, _config())

    yaw_rate = annotation.motion_commands[25:-25, 2]
    assert yaw_rate.median().item() == pytest.approx(0.3, abs=0.02)
    assert torch.max(torch.abs(yaw_rate - 0.3)) < 0.05


def test_source_commands_match_achieved_timing_and_scale():
    dt = 0.02
    time = np.arange(201) * dt
    delay = 0.2
    source_vx = 0.5 * (1.0 + np.tanh((time - 1.5) / 0.2))
    source_wz = 0.7 * np.exp(-np.square((time - 2.2) / 0.45))
    source_yaw = np.cumsum(source_wz) * dt
    reference = ReferenceCommandTrajectory(
        time_s=time,
        commands=np.stack((source_vx, np.zeros_like(time), source_wz), axis=-1),
        yaw=source_yaw,
        source_path="synthetic/commands.csv",
    )

    delayed_time = time - delay
    achieved_vx = 0.75 * np.interp(delayed_time, time, source_vx)
    achieved_yaw = 0.9 * np.interp(delayed_time, time, source_yaw)
    body_pos = torch.zeros((len(time), 2, 3))
    body_pos[:, 1, 0] = torch.from_numpy(
        np.cumsum(np.cos(achieved_yaw) * achieved_vx) * dt
    )
    body_pos[:, 1, 1] = torch.from_numpy(
        np.cumsum(np.sin(achieved_yaw) * achieved_vx) * dt
    )
    body_rot = torch.zeros((len(time), 2, 4))
    body_rot[..., 3] = 1.0
    body_rot[:, 0] = _yaw_quaternion(torch.from_numpy(achieved_yaw).float())

    annotation = annotate_motion_clip(
        body_pos,
        body_rot,
        dt,
        _config(smoothing_window_s=0.0),
        reference_commands=reference,
    )

    metadata = annotation.motion_command_metadata
    assert metadata["command_source"] == "calibrated_source_trajectory"
    assert metadata["source_time_offset_s"] == pytest.approx(-delay, abs=0.04)
    assert metadata["source_velocity_gain"] == pytest.approx(0.75, abs=0.03)
    assert metadata["source_yaw_gain"] == pytest.approx(0.9, abs=0.04)
    assert torch.allclose(
        annotation.motion_commands[:, 2],
        annotation.source_motion_commands[:, 2] * metadata["source_yaw_gain"],
    )


def test_packed_annotations_do_not_filter_across_clip_boundaries():
    dt = 0.02
    count = 101
    time = torch.arange(count) * dt
    body_pos = torch.zeros(2 * count, 2, 3)
    body_pos[:count, 1, 0] = 0.5 * time
    body_pos[count:, 1, 0] = 100.0 - 0.4 * time
    body_rot = torch.zeros(2 * count, 2, 4)
    body_rot[..., 3] = 1.0
    packed = {
        "gts": body_pos,
        "grs": body_rot,
        "length_starts": torch.tensor([0, count]),
        "motion_num_frames": torch.tensor([count, count]),
        "motion_dt": torch.tensor([dt, dt]),
    }

    annotation = annotate_packed_motion_library(packed, _config())

    assert annotation.motion_commands[count - 1, 0].item() == pytest.approx(0.5, abs=0.02)
    assert annotation.motion_commands[count, 0].item() == pytest.approx(-0.4, abs=0.02)
    assert torch.max(torch.abs(annotation.motion_commands[:, 0])) < 0.55


def test_manual_root_columns_override_procedural_simulation_root():
    dt = 0.1
    frames = 20
    body_pos = torch.randn(frames, 2, 3)
    body_rot = torch.zeros(frames, 2, 4)
    body_rot[..., 3] = 1.0
    time = np.arange(frames) * dt
    manual = {
        "simulation_root_x": 0.7 * time,
        "simulation_root_y": -0.2 * time,
        "simulation_root_yaw": 0.4 * time,
    }

    annotation = annotate_motion_clip(
        body_pos,
        body_rot,
        dt,
        _config(),
        manual_root=manual,
    )

    assert annotation.motion_command_metadata["manually_edited"] is True
    assert annotation.simulation_root_pos[:, 0].numpy() == pytest.approx(
        manual["simulation_root_x"]
    )
    assert annotation.simulation_root_ang_vel[:, 2].mean().item() == pytest.approx(
        0.4, abs=1.0e-5
    )
