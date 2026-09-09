# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate editable locomotion-command labels from packed motion libraries.

The synthetic ``simulation root`` intentionally differs from the robot root.
Its planar position follows a smoothed upper-spine link, while its heading
follows the smoothed projected forward direction of a hip/pelvis link.  This
removes step-scale pelvis sway before planar velocity and yaw-rate labels are
computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from scipy.signal import savgol_filter
from torch import Tensor

from protomotions.utils import rotations


ANNOTATION_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SimulationRootAnnotationConfig:
    """Configuration for Savitzky--Golay simulation-root extraction."""

    position_body_index: int
    facing_body_index: int
    position_body_name: str
    facing_body_name: str
    semantic_forward_axis_xy: tuple[float, float] = (1.0, 0.0)
    smoothing_window_s: float = 0.75
    polynomial_order: int = 3
    boundary_mode: str = "linear_extrapolate"
    source_match_max_offset_s: float = 0.4
    source_velocity_gain_bounds: tuple[float, float] = (0.0, 2.0)
    source_yaw_gain_bounds: tuple[float, float] = (0.0, 2.0)

    def validate(self, num_bodies: int) -> None:
        for name, index in (
            ("position_body_index", self.position_body_index),
            ("facing_body_index", self.facing_body_index),
        ):
            if not 0 <= index < num_bodies:
                raise ValueError(f"{name}={index} is outside [0, {num_bodies - 1}]")
        if self.smoothing_window_s < 0.0:
            raise ValueError("smoothing_window_s must be non-negative")
        if self.polynomial_order < 0:
            raise ValueError("polynomial_order must be non-negative")
        if self.boundary_mode not in {
            "linear_extrapolate",
            "interp",
            "mirror",
            "nearest",
            "constant",
            "wrap",
        }:
            raise ValueError(f"unsupported Savitzky--Golay boundary mode: {self.boundary_mode}")
        axis = np.asarray(self.semantic_forward_axis_xy, dtype=np.float64)
        if axis.shape != (2,) or np.linalg.norm(axis) < 1.0e-8:
            raise ValueError("semantic_forward_axis_xy must be a nonzero XY vector")
        if self.source_match_max_offset_s < 0.0:
            raise ValueError("source_match_max_offset_s must be non-negative")
        for name, bounds in (
            ("source_velocity_gain_bounds", self.source_velocity_gain_bounds),
            ("source_yaw_gain_bounds", self.source_yaw_gain_bounds),
        ):
            if len(bounds) != 2 or bounds[0] < 0.0 or bounds[0] > bounds[1]:
                raise ValueError(f"{name} must be ordered non-negative bounds")


@dataclass(frozen=True)
class ReferenceCommandTrajectory:
    """Original command trajectory used to generate one source motion."""

    time_s: np.ndarray
    commands: np.ndarray
    yaw: np.ndarray
    source_path: Optional[str] = None

    def validate(self) -> None:
        if self.time_s.ndim != 1 or len(self.time_s) < 1:
            raise ValueError("reference command time_s must be a nonempty vector")
        if self.commands.shape != (len(self.time_s), 3):
            raise ValueError("reference commands must have shape [frames, 3]")
        if self.yaw.shape != self.time_s.shape:
            raise ValueError("reference command yaw must match time_s")
        if not (
            np.isfinite(self.time_s).all()
            and np.isfinite(self.commands).all()
            and np.isfinite(self.yaw).all()
        ):
            raise ValueError("reference command trajectory contains non-finite values")
        if len(self.time_s) > 1 and np.any(np.diff(self.time_s) <= 0.0):
            raise ValueError("reference command timestamps must be strictly increasing")


@dataclass(frozen=True)
class SimulationRootAnnotations:
    """Per-frame synthetic root transform and locomotion command labels."""

    simulation_root_pos: Tensor
    simulation_root_rot: Tensor
    simulation_root_vel: Tensor
    simulation_root_ang_vel: Tensor
    motion_commands: Tensor
    achieved_motion_commands: Tensor
    source_motion_commands: Optional[Tensor]
    motion_command_metadata: dict

    def as_dict(self) -> dict:
        result = {
            "simulation_root_pos": self.simulation_root_pos,
            "simulation_root_rot": self.simulation_root_rot,
            "simulation_root_vel": self.simulation_root_vel,
            "simulation_root_ang_vel": self.simulation_root_ang_vel,
            "motion_commands": self.motion_commands,
            "achieved_motion_commands": self.achieved_motion_commands,
            "motion_command_metadata": self.motion_command_metadata,
        }
        if self.source_motion_commands is not None:
            result["source_motion_commands"] = self.source_motion_commands
        return result


def _adaptive_savgol_window(
    num_frames: int,
    dt: float,
    window_s: float,
    polynomial_order: int,
) -> Optional[int]:
    """Return the largest useful odd window no longer than the clip."""
    if num_frames <= polynomial_order or window_s <= 0.0:
        return None
    requested = max(1, int(round(window_s / max(dt, 1.0e-8))))
    if requested % 2 == 0:
        requested += 1
    largest_odd = num_frames if num_frames % 2 == 1 else num_frames - 1
    window = min(requested, largest_odd)
    minimum = polynomial_order + 1
    if minimum % 2 == 0:
        minimum += 1
    if window < minimum:
        return None
    return window


def _linear_edge_padding(values: np.ndarray, pad: int, fit_width: int) -> np.ndarray:
    """Extend each edge using a least-squares linear trend over nearby frames."""
    if pad == 0:
        return values
    fit_width = min(fit_width, len(values))
    sample_time = np.arange(fit_width, dtype=np.float64)
    centered_time = sample_time - sample_time.mean()
    denominator = np.dot(centered_time, centered_time)

    def extrapolate(edge_values: np.ndarray, query_time: np.ndarray) -> np.ndarray:
        mean = edge_values.mean(axis=0)
        slope = np.tensordot(centered_time, edge_values - mean, axes=(0, 0)) / denominator
        time_shape = (...,) + (None,) * (values.ndim - 1)
        return mean + (query_time - sample_time.mean())[time_shape] * slope

    left = extrapolate(values[:fit_width], np.arange(-pad, 0, dtype=np.float64))
    right = extrapolate(
        values[-fit_width:],
        np.arange(fit_width, fit_width + pad, dtype=np.float64),
    )
    return np.concatenate((left, values, right), axis=0)


def _smooth(
    values: np.ndarray,
    window: Optional[int],
    config: SimulationRootAnnotationConfig,
) -> np.ndarray:
    if window is None:
        return values.copy()
    if config.boundary_mode == "linear_extrapolate":
        pad = window // 2
        padded = _linear_edge_padding(values, pad, window)
        return savgol_filter(
            padded,
            window_length=window,
            polyorder=config.polynomial_order,
            axis=0,
            mode="interp",
        )[pad:-pad]
    return savgol_filter(
        values,
        window_length=window,
        polyorder=config.polynomial_order,
        axis=0,
        mode=config.boundary_mode,
    )


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values)
    return np.gradient(
        values,
        dt,
        axis=0,
        edge_order=2 if len(values) >= 3 else 1,
    )


def _yaw_to_xyzw(yaw: np.ndarray) -> np.ndarray:
    result = np.zeros((len(yaw), 4), dtype=yaw.dtype)
    result[:, 2] = np.sin(0.5 * yaw)
    result[:, 3] = np.cos(0.5 * yaw)
    return result


def _manual_root_arrays(
    manual_root: Mapping[str, Sequence[float]],
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    required = ("simulation_root_x", "simulation_root_y", "simulation_root_yaw")
    missing = [key for key in required if key not in manual_root]
    if missing:
        raise ValueError(f"manual simulation-root data is missing columns: {missing}")
    columns = [np.asarray(manual_root[key], dtype=np.float64) for key in required]
    if any(column.shape != (num_frames,) for column in columns):
        raise ValueError(f"manual simulation-root columns must each contain {num_frames} frames")
    position_xy = np.stack(columns[:2], axis=-1)
    yaw = np.unwrap(columns[2])
    return position_xy, yaw


def _interpolate_reference(
    reference: ReferenceCommandTrajectory,
    target_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    commands = np.stack(
        [
            np.interp(
                target_time,
                reference.time_s,
                reference.commands[:, component],
            )
            for component in range(3)
        ],
        axis=-1,
    )
    yaw = np.interp(target_time, reference.time_s, np.unwrap(reference.yaw))
    return commands, yaw


def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> Optional[float]:
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    if denominator < 1.0e-8:
        return None
    return float(np.dot(first_centered, second_centered) / denominator)


def _source_match_score(source: np.ndarray, achieved: np.ndarray) -> Optional[float]:
    """Score temporal agreement while ignoring constant command channels."""
    scores = []
    source_channels = [source[:, 0], source[:, 1], source[:, 2]]
    achieved_channels = [achieved[:, 0], achieved[:, 1], achieved[:, 2]]
    source_channels.append(np.linalg.norm(source[:, :2], axis=-1))
    achieved_channels.append(np.linalg.norm(achieved[:, :2], axis=-1))
    for source_channel, achieved_channel in zip(source_channels, achieved_channels):
        variation_threshold = max(
            0.05,
            0.1 * float(np.max(np.abs(source_channel))),
        )
        if np.ptp(source_channel) < variation_threshold:
            continue
        score = _normalized_correlation(source_channel, achieved_channel)
        if score is not None:
            scores.append(score)
    return float(np.mean(scores)) if scores else None


def _match_source_time_offset(
    reference: ReferenceCommandTrajectory,
    target_time: np.ndarray,
    achieved_commands: np.ndarray,
    dt: float,
    max_offset_s: float,
) -> tuple[float, Optional[float], Optional[float]]:
    """Find a small clip-level source-time shift for changing command profiles."""
    if max_offset_s <= 0.0:
        return 0.0, None, None
    offsets = np.arange(-max_offset_s, max_offset_s + 0.5 * dt, dt)
    zero_commands, _ = _interpolate_reference(reference, target_time)
    zero_score = _source_match_score(zero_commands, achieved_commands)
    if zero_score is None:
        return 0.0, None, None
    best_offset = 0.0
    best_score = zero_score
    best_regularized_score = zero_score
    for offset in offsets:
        source_commands, _ = _interpolate_reference(reference, target_time + offset)
        score = _source_match_score(source_commands, achieved_commands)
        if score is None:
            continue
        offset_fraction = abs(float(offset)) / max(max_offset_s, 1.0e-8)
        regularized_score = score - 0.03 * offset_fraction**2
        if regularized_score > best_regularized_score + 1.0e-9 or (
            abs(regularized_score - best_regularized_score) <= 1.0e-9
            and abs(offset) < abs(best_offset)
        ):
            best_score = score
            best_regularized_score = regularized_score
            best_offset = float(offset)
    return best_offset, zero_score, best_score


def _calibrate_reference_commands(
    reference: ReferenceCommandTrajectory,
    target_time: np.ndarray,
    achieved_commands: np.ndarray,
    achieved_yaw: np.ndarray,
    dt: float,
    config: SimulationRootAnnotationConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Match timing and achieved scale while retaining the smooth source profile."""
    reference.validate()
    time_offset, zero_match_score, matched_score = _match_source_time_offset(
        reference,
        target_time,
        achieved_commands,
        dt,
        config.source_match_max_offset_s,
    )
    source_commands, source_yaw = _interpolate_reference(
        reference,
        target_time + time_offset,
    )

    source_velocity = source_commands[:, :2]
    velocity_denominator = float(np.sum(np.square(source_velocity)))
    if velocity_denominator > 1.0e-8:
        velocity_gain = float(
            np.sum(source_velocity * achieved_commands[:, :2])
            / velocity_denominator
        )
        velocity_gain = float(
            np.clip(velocity_gain, *config.source_velocity_gain_bounds)
        )
    else:
        velocity_gain = 0.0

    source_yaw_centered = source_yaw - source_yaw.mean()
    achieved_yaw_centered = achieved_yaw - achieved_yaw.mean()
    yaw_denominator = float(np.dot(source_yaw_centered, source_yaw_centered))
    if yaw_denominator > 1.0e-8:
        yaw_gain = float(
            np.dot(source_yaw_centered, achieved_yaw_centered) / yaw_denominator
        )
        yaw_gain = float(np.clip(yaw_gain, *config.source_yaw_gain_bounds))
    else:
        yaw_gain = 0.0

    calibrated = source_commands.copy()
    calibrated[:, :2] *= velocity_gain
    calibrated[:, 2] *= yaw_gain
    metadata = {
        "source_command_path": reference.source_path,
        "source_time_offset_s": time_offset,
        "source_match_score_at_zero": zero_match_score,
        "source_match_score": matched_score,
        "source_velocity_gain": velocity_gain,
        "source_yaw_gain": yaw_gain,
    }
    return calibrated, source_commands, metadata


def annotate_motion_clip(
    body_pos: Tensor,
    body_rot: Tensor,
    dt: float,
    config: SimulationRootAnnotationConfig,
    *,
    manual_root: Optional[Mapping[str, Sequence[float]]] = None,
    reference_commands: Optional[ReferenceCommandTrajectory] = None,
) -> SimulationRootAnnotations:
    """Annotate one clip without allowing filtering across clip boundaries."""
    if body_pos.ndim != 3 or body_pos.shape[-1] != 3:
        raise ValueError("body_pos must have shape [frames, bodies, 3]")
    if body_rot.shape != (*body_pos.shape[:-1], 4):
        raise ValueError("body_rot must have shape [frames, bodies, 4]")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    config.validate(body_pos.shape[1])

    num_frames = body_pos.shape[0]
    window = _adaptive_savgol_window(
        num_frames,
        dt,
        config.smoothing_window_s,
        config.polynomial_order,
    )

    if manual_root is None:
        position_xy_raw = (
            body_pos[:, config.position_body_index, :2].detach().cpu().double().numpy()
        )
        position_xy = _smooth(position_xy_raw, window, config)

        axis_xy = np.asarray(config.semantic_forward_axis_xy, dtype=np.float64)
        axis_xy /= np.linalg.norm(axis_xy)
        local_forward = body_pos.new_zeros((num_frames, 3))
        local_forward[:, :2] = torch.as_tensor(
            axis_xy, device=body_pos.device, dtype=body_pos.dtype
        )
        world_forward = rotations.quat_rotate(
            body_rot[:, config.facing_body_index],
            local_forward,
            w_last=True,
        )[:, :2]
        world_forward = world_forward.detach().cpu().double().numpy()
        raw_norm = np.linalg.norm(world_forward, axis=-1, keepdims=True)
        world_forward /= np.maximum(raw_norm, 1.0e-8)
        smooth_forward = _smooth(world_forward, window, config)
        smooth_norm = np.linalg.norm(smooth_forward, axis=-1, keepdims=True)
        degenerate = smooth_norm[:, 0] < 1.0e-8
        smooth_forward /= np.maximum(smooth_norm, 1.0e-8)
        smooth_forward[degenerate] = world_forward[degenerate]
        yaw = np.unwrap(np.arctan2(smooth_forward[:, 1], smooth_forward[:, 0]))
        manually_edited = False
    else:
        position_xy, yaw = _manual_root_arrays(manual_root, num_frames)
        manually_edited = True

    world_velocity_xy = _finite_difference(position_xy, dt)
    yaw_rate = _finite_difference(yaw, dt)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    local_velocity_xy = np.stack(
        (
            cos_yaw * world_velocity_xy[:, 0] + sin_yaw * world_velocity_xy[:, 1],
            -sin_yaw * world_velocity_xy[:, 0] + cos_yaw * world_velocity_xy[:, 1],
        ),
        axis=-1,
    )

    root_pos = np.zeros((num_frames, 3), dtype=np.float64)
    root_pos[:, :2] = position_xy
    root_vel = np.zeros_like(root_pos)
    root_vel[:, :2] = world_velocity_xy
    root_ang_vel = np.zeros_like(root_pos)
    root_ang_vel[:, 2] = yaw_rate
    achieved_commands = np.concatenate(
        (local_velocity_xy, yaw_rate[:, None]),
        axis=-1,
    )
    source_commands = None
    source_metadata = {
        "command_source": "achieved_simulation_root",
        "source_command_path": None,
        "source_time_offset_s": 0.0,
        "source_match_score_at_zero": None,
        "source_match_score": None,
        "source_velocity_gain": 1.0,
        "source_yaw_gain": 1.0,
    }
    commands = achieved_commands
    if reference_commands is not None:
        commands, source_commands, calibration_metadata = (
            _calibrate_reference_commands(
                reference_commands,
                np.arange(num_frames, dtype=np.float64) * dt,
                achieved_commands,
                yaw,
                dt,
                config,
            )
        )
        source_metadata.update(calibration_metadata)
        source_metadata["command_source"] = "calibrated_source_trajectory"

    tensor_kwargs = {"device": body_pos.device, "dtype": body_pos.dtype}
    metadata = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "method": (
            "savgol_simulation_root_with_calibrated_source_commands"
            if reference_commands is not None
            else "savgol_simulation_root"
        ),
        "position_body_name": config.position_body_name,
        "position_body_index": config.position_body_index,
        "facing_body_name": config.facing_body_name,
        "facing_body_index": config.facing_body_index,
        "semantic_forward_axis_xy": tuple(config.semantic_forward_axis_xy),
        "smoothing_window_s": config.smoothing_window_s,
        "polynomial_order": config.polynomial_order,
        "boundary_mode": config.boundary_mode,
        "effective_window_frames": window,
        "manually_edited": manually_edited,
        **source_metadata,
    }
    return SimulationRootAnnotations(
        simulation_root_pos=torch.as_tensor(root_pos, **tensor_kwargs),
        simulation_root_rot=torch.as_tensor(_yaw_to_xyzw(yaw), **tensor_kwargs),
        simulation_root_vel=torch.as_tensor(root_vel, **tensor_kwargs),
        simulation_root_ang_vel=torch.as_tensor(root_ang_vel, **tensor_kwargs),
        motion_commands=torch.as_tensor(commands, **tensor_kwargs),
        achieved_motion_commands=torch.as_tensor(achieved_commands, **tensor_kwargs),
        source_motion_commands=(
            torch.as_tensor(source_commands, **tensor_kwargs)
            if source_commands is not None
            else None
        ),
        motion_command_metadata=metadata,
    )


def annotate_packed_motion_library(
    packed_motion: Mapping[str, object],
    config: SimulationRootAnnotationConfig,
    *,
    manual_roots: Optional[Mapping[int, Mapping[str, Sequence[float]]]] = None,
    reference_commands: Optional[Mapping[int, ReferenceCommandTrajectory]] = None,
) -> SimulationRootAnnotations:
    """Annotate every clip in a packed MotionLib dictionary independently."""
    required = ("gts", "grs", "length_starts", "motion_num_frames", "motion_dt")
    missing = [key for key in required if key not in packed_motion]
    if missing:
        raise ValueError(f"packed motion library is missing fields: {missing}")
    body_pos = packed_motion["gts"]
    body_rot = packed_motion["grs"]
    if not isinstance(body_pos, Tensor) or not isinstance(body_rot, Tensor):
        raise TypeError("packed gts and grs fields must be tensors")

    outputs: dict[str, list[Tensor]] = {
        "simulation_root_pos": [],
        "simulation_root_rot": [],
        "simulation_root_vel": [],
        "simulation_root_ang_vel": [],
        "motion_commands": [],
        "achieved_motion_commands": [],
    }
    uses_reference_commands = reference_commands is not None
    if uses_reference_commands:
        outputs["source_motion_commands"] = []
    clip_metadata = []
    manual_roots = manual_roots or {}
    reference_commands = reference_commands or {}
    starts = packed_motion["length_starts"]
    counts = packed_motion["motion_num_frames"]
    dts = packed_motion["motion_dt"]
    for motion_id, (start, count, motion_dt) in enumerate(zip(starts, counts, dts)):
        start_i = int(start)
        count_i = int(count)
        if uses_reference_commands and motion_id not in reference_commands:
            raise ValueError(f"reference commands are missing motion {motion_id}")
        annotation = annotate_motion_clip(
            body_pos[start_i : start_i + count_i],
            body_rot[start_i : start_i + count_i],
            float(motion_dt),
            config,
            manual_root=manual_roots.get(motion_id),
            reference_commands=reference_commands.get(motion_id),
        )
        for key in outputs:
            value = getattr(annotation, key)
            if value is None:
                raise RuntimeError(f"annotation field {key} is unexpectedly missing")
            outputs[key].append(value)
        clip_metadata.append(annotation.motion_command_metadata)

    metadata = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "method": (
            "savgol_simulation_root_with_calibrated_source_commands"
            if uses_reference_commands
            else "savgol_simulation_root"
        ),
        "position_body_name": config.position_body_name,
        "position_body_index": config.position_body_index,
        "facing_body_name": config.facing_body_name,
        "facing_body_index": config.facing_body_index,
        "semantic_forward_axis_xy": tuple(config.semantic_forward_axis_xy),
        "smoothing_window_s": config.smoothing_window_s,
        "polynomial_order": config.polynomial_order,
        "boundary_mode": config.boundary_mode,
        "source_match_max_offset_s": config.source_match_max_offset_s,
        "source_velocity_gain_bounds": config.source_velocity_gain_bounds,
        "source_yaw_gain_bounds": config.source_yaw_gain_bounds,
        "clip_metadata": clip_metadata,
    }
    concatenated = {key: torch.cat(value, dim=0) for key, value in outputs.items()}
    return SimulationRootAnnotations(
        simulation_root_pos=concatenated["simulation_root_pos"],
        simulation_root_rot=concatenated["simulation_root_rot"],
        simulation_root_vel=concatenated["simulation_root_vel"],
        simulation_root_ang_vel=concatenated["simulation_root_ang_vel"],
        motion_commands=concatenated["motion_commands"],
        achieved_motion_commands=concatenated["achieved_motion_commands"],
        source_motion_commands=concatenated.get("source_motion_commands"),
        motion_command_metadata=metadata,
    )


__all__ = [
    "ANNOTATION_SCHEMA_VERSION",
    "ReferenceCommandTrajectory",
    "SimulationRootAnnotationConfig",
    "SimulationRootAnnotations",
    "annotate_motion_clip",
    "annotate_packed_motion_library",
]
