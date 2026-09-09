# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command labels and command-matched sampling for motion libraries."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import Tensor

from protomotions.utils import rotations


def get_achieved_reference_commands(
    motion_lib,
    anchor_body_index: int,
    smoothing_window_s: float = 0.2,
) -> Tensor:
    """Return cached body-frame ``[vx, vy, yaw_rate]`` for every motion frame.

    Velocities are averaged independently inside each clip.  This removes
    footstep-scale velocity oscillation without blending across clip boundaries.
    """
    cache_key = (int(anchor_body_index), float(smoothing_window_s))
    cache = getattr(motion_lib, "_achieved_reference_command_cache", None)
    if cache is None:
        cache = {}
        motion_lib._achieved_reference_command_cache = cache
    if cache_key in cache:
        return cache[cache_key]

    # Annotated libraries carry a clip-local, smoothed synthetic root whose
    # velocity and yaw rate deliberately reject pelvis/footstep oscillations.
    # Prefer those persistent labels over rebuilding commands from the raw
    # anchor velocity. The smoothing argument only applies to legacy libraries.
    annotated_commands = getattr(motion_lib, "motion_commands", None)
    if annotated_commands is not None:
        if annotated_commands.shape != (motion_lib.gts.shape[0], 3):
            raise ValueError(
                "motion_commands must have shape [total_motion_frames, 3], "
                f"got {tuple(annotated_commands.shape)}"
            )
        commands = annotated_commands.to(
            device=motion_lib.gts.device,
            dtype=motion_lib.gts.dtype,
        )
        cache[cache_key] = commands
        return commands

    anchor_rot = motion_lib.grs[:, anchor_body_index]
    heading_inv = rotations.calc_heading_quat_inv(anchor_rot, w_last=True)
    local_vel = rotations.quat_rotate(
        heading_inv,
        motion_lib.gvs[:, anchor_body_index],
        w_last=True,
    )[:, :2]
    commands = torch.cat(
        (
            local_vel,
            motion_lib.gavs[:, anchor_body_index, 2:3],
        ),
        dim=-1,
    )

    if smoothing_window_s > 0.0:
        smoothed = commands.clone()
        for start, num_frames, motion_dt in zip(
            motion_lib.length_starts.tolist(),
            motion_lib.motion_num_frames.tolist(),
            motion_lib.motion_dt.tolist(),
        ):
            if num_frames <= 1:
                continue
            half_window = max(
                0,
                int(round(0.5 * smoothing_window_s / max(motion_dt, 1.0e-8))),
            )
            if half_window == 0:
                continue
            clip = commands[start : start + num_frames]
            prefix = torch.cat(
                (torch.zeros_like(clip[:1]), torch.cumsum(clip, dim=0)),
                dim=0,
            )
            frame_ids = torch.arange(num_frames, device=commands.device)
            lower = (frame_ids - half_window).clamp_min(0)
            upper = (frame_ids + half_window + 1).clamp_max(num_frames)
            smoothed[start : start + num_frames] = (
                prefix[upper] - prefix[lower]
            ) / (upper - lower).unsqueeze(-1)
        commands = smoothed

    cache[cache_key] = commands
    return commands


def reference_commands_at_times(
    motion_lib,
    motion_ids: Tensor,
    motion_times: Tensor,
    anchor_body_index: int,
    smoothing_window_s: float = 0.2,
) -> Tensor:
    """Look up achieved commands at the nearest stored reference frames."""
    frame_ids = torch.round(
        motion_times / motion_lib.motion_dt[motion_ids].clamp_min(1.0e-8)
    ).long()
    frame_ids = torch.minimum(
        frame_ids.clamp_min(0),
        motion_lib.motion_num_frames[motion_ids] - 1,
    )
    global_frame_ids = motion_lib.length_starts[motion_ids] + frame_ids
    commands = get_achieved_reference_commands(
        motion_lib,
        anchor_body_index=anchor_body_index,
        smoothing_window_s=smoothing_window_s,
    )
    return commands[global_frame_ids]


def sample_command_matched_motions(
    motion_manager,
    target_commands: Tensor,
    anchor_body_index: int,
    candidate_count: int = 64,
    command_scales: Sequence[float] = (0.5, 0.5, 0.5),
    smoothing_window_s: float = 0.2,
    init_start_probability: float | Tensor = 0.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Choose the closest achieved command among random reference candidates.

    Candidate motion IDs retain the motion manager's configured weighting and
    exclusions.  Sampling exact stored frames makes matching inexpensive and
    ensures the returned time points agree with the cached command labels.
    """
    if target_commands.ndim != 2 or target_commands.shape[-1] != 3:
        raise ValueError("target_commands must have shape [num_samples, 3]")
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    scales = torch.as_tensor(
        command_scales,
        device=target_commands.device,
        dtype=target_commands.dtype,
    )
    if scales.shape != (3,) or torch.any(scales <= 0.0):
        raise ValueError("command_scales must contain three positive values")

    num_samples = target_commands.shape[0]
    if num_samples == 0:
        empty_ids = torch.empty(
            0, device=target_commands.device, dtype=torch.long
        )
        return empty_ids, target_commands.new_empty(0), target_commands.clone()

    flat_motion_ids = motion_manager.sample_n_motion_ids(
        num_samples * candidate_count
    )
    candidate_motion_ids = flat_motion_ids.view(num_samples, candidate_count)
    num_frames = motion_manager.motion_lib.motion_num_frames[candidate_motion_ids]
    candidate_frame_ids = torch.floor(
        torch.rand(
            num_samples,
            candidate_count,
            device=target_commands.device,
        )
        * num_frames
    ).long()

    start_probability = torch.as_tensor(
        init_start_probability,
        device=target_commands.device,
        dtype=target_commands.dtype,
    )
    if start_probability.ndim == 0:
        start_probability = start_probability.expand(num_samples)
    if start_probability.shape != (num_samples,):
        raise ValueError(
            "init_start_probability must be scalar or have shape [num_samples]"
        )
    if torch.any((start_probability < 0.0) | (start_probability > 1.0)):
        raise ValueError("init_start_probability must be in [0, 1]")
    choose_start = torch.rand(
        num_samples,
        candidate_count,
        device=target_commands.device,
    ) < start_probability.unsqueeze(-1)
    candidate_frame_ids = torch.where(
        choose_start,
        torch.zeros_like(candidate_frame_ids),
        candidate_frame_ids,
    )

    global_frame_ids = (
        motion_manager.motion_lib.length_starts[candidate_motion_ids]
        + candidate_frame_ids
    )
    command_cache = get_achieved_reference_commands(
        motion_manager.motion_lib,
        anchor_body_index=anchor_body_index,
        smoothing_window_s=smoothing_window_s,
    )
    candidate_commands = command_cache[global_frame_ids]
    command_error = torch.sum(
        torch.square(
            (candidate_commands - target_commands.unsqueeze(1)) / scales
        ),
        dim=-1,
    )
    best_candidate = torch.argmin(command_error, dim=1)
    row_ids = torch.arange(num_samples, device=target_commands.device)
    matched_motion_ids = candidate_motion_ids[row_ids, best_candidate]
    matched_frame_ids = candidate_frame_ids[row_ids, best_candidate]
    matched_commands = candidate_commands[row_ids, best_candidate]
    matched_times = (
        matched_frame_ids
        * motion_manager.motion_lib.motion_dt[matched_motion_ids]
    )
    return matched_motion_ids, matched_times, matched_commands


__all__ = [
    "get_achieved_reference_commands",
    "reference_commands_at_times",
    "sample_command_matched_motions",
]
