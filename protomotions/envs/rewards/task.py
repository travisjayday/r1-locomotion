# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-specific reward compute kernels.

Pure tensor functions (kernels) for computing task-specific rewards.
Use MdpComponent in experiment configs to bind kernels to context paths:

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.task import compute_heading_velocity_rew

    reward_components = {
        "heading_velocity": MdpComponent(
            compute_func=compute_heading_velocity_rew,
            dynamic_vars={
                "root_pos": EnvContext.current.root_pos,
                "prev_root_pos": EnvContext.steering.prev_root_pos,
                "root_rot": EnvContext.current.root_rot,
                "tar_dir": EnvContext.steering.tar_dir,
                "tar_speed": EnvContext.steering.tar_speed,
                "tar_face_dir": EnvContext.steering.tar_face_dir,
                "dt": EnvContext.dt,
            },
        ),
    }

Provides reward functions for specific tasks:
- Steering/locomotion rewards
- Path following rewards
- Target reaching rewards
"""

import torch
from torch import Tensor

from protomotions.utils import rotations
from protomotions.utils.rotations import calc_heading_quat, quat_rotate


# =============================================================================
# Steering Reward Kernels
# =============================================================================

def compute_heading_velocity_rew(
    root_pos: Tensor,
    prev_root_pos: Tensor,
    root_rot: Tensor,
    tar_dir: Tensor,
    tar_speed: Tensor,
    tar_face_dir: Tensor,
    dt: float,
) -> Tensor:
    """Reward for moving in target direction at target speed while facing that direction.

    Computes weighted combination of:
    - Direction reward: exponential penalty on velocity error and tangent velocity
    - Facing reward: alignment between robot heading and target direction

    Args:
        root_pos: Current root position [num_envs, 3].
        prev_root_pos: Previous root position [num_envs, 3].
        root_rot: Root orientation quaternions [num_envs, 4] (w-last).
        tar_dir: Target movement direction [num_envs, 2].
        tar_speed: Target speed [num_envs].
        tar_face_dir: Target facing direction [num_envs, 2] (can differ from tar_dir).
        dt: Simulation timestep.

    Returns:
        Reward [num_envs] in range [0, 1].
    """

    vel_err_scale = 0.25
    tangent_err_w = 0.1

    dir_reward_w = 0.7
    facing_reward_w = 0.3

    # Compute velocity in target direction
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)

    # Compute tangent (perpendicular) velocity
    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * tar_dir
    tangent_vel = root_vel[..., :2] - tar_dir_vel
    tangent_vel_err = torch.sum(torch.square(tangent_vel), dim=-1)

    # Direction reward: penalize velocity error and tangent movement
    tar_vel_err = tar_speed - tar_dir_speed
    dir_reward = torch.exp(
        -vel_err_scale * (tar_vel_err * tar_vel_err + tangent_err_w * tangent_vel_err)
    )

    # A stop command should reward low planar speed regardless of the arbitrary
    # retained target direction. Without this branch, an exactly stationary
    # robot receives zero direction reward because tar_dir_speed <= 0.
    is_stop_command = tar_speed <= 1.0e-4
    stop_reward = torch.exp(
        -vel_err_scale * torch.sum(torch.square(root_vel[..., :2]), dim=-1)
    )
    dir_reward = torch.where(is_stop_command, stop_reward, dir_reward)

    # For non-stop commands, reject motion opposite the requested direction.
    speed_mask = (tar_dir_speed <= 0) & ~is_stop_command
    dir_reward[speed_mask] = 0

    # Facing reward: robot should face the target facing direction
    heading_rot = calc_heading_quat(root_rot, w_last=True)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir, w_last=True)

    facing_err = torch.sum(tar_face_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)
    # Facing reward is orientation-only and pays out even at zero velocity,
    # so during an active walk/run command it must not soften the speed_mask
    # penalty above -- otherwise standing still and only turning to face
    # tar_face_dir becomes a safe local optimum that never has to risk
    # learning to actually move. Stop commands are untouched: turning in
    # place to face a direction while stationary is the intended behavior
    # there, handled entirely by the stop_reward branch above.
    facing_reward[speed_mask] = 0

    reward = dir_reward_w * dir_reward + facing_reward_w * facing_reward

    return reward


def compute_track_lin_vel_xy_yaw_frame_exp(
    root_vel: Tensor,
    root_rot: Tensor,
    tar_local_vel: Tensor,
    std: float = 0.5,
) -> Tensor:
    """Track commanded planar velocity in the robot's yaw-only frame."""
    heading_inv = rotations.calc_heading_quat_inv(root_rot, w_last=True)
    local_velocity = rotations.quat_rotate(
        heading_inv, root_vel, w_last=True
    )[..., :2]
    velocity_error = torch.sum(
        torch.square(local_velocity - tar_local_vel), dim=-1
    )
    return torch.exp(-velocity_error / (std * std))


def compute_track_ang_vel_z_world_exp(
    root_ang_vel: Tensor,
    tar_yaw_rate: Tensor,
    std: float = 0.5,
) -> Tensor:
    """Track commanded world-frame yaw rate."""
    yaw_rate_error = torch.square(root_ang_vel[..., 2] - tar_yaw_rate)
    return torch.exp(-yaw_rate_error / (std * std))


def compute_lin_vel_z_l2(root_vel: Tensor) -> Tensor:
    """Penalize vertical root velocity."""
    return torch.square(root_vel[..., 2])


def compute_ang_vel_xy_l2(root_ang_vel: Tensor) -> Tensor:
    """Penalize root roll and pitch angular velocity."""
    return torch.sum(torch.square(root_ang_vel[..., :2]), dim=-1)


def compute_body_orientation_l2(root_rot: Tensor) -> Tensor:
    """Penalize pelvis tilt using the horizontal projected-gravity components."""
    gravity = torch.zeros(
        *root_rot.shape[:-1], 3, device=root_rot.device, dtype=root_rot.dtype
    )
    gravity[..., 2] = -1.0
    projected_gravity = rotations.quat_rotate_inverse(
        root_rot, gravity, w_last=True
    )
    return torch.sum(torch.square(projected_gravity[..., :2]), dim=-1)


# =============================================================================
# Path Following Reward Kernels
# =============================================================================

def compute_path_following_rew(
    head_pos: Tensor,
    tar_pos: Tensor,
    height_conditioned: bool,
    pos_err_scale: float = 2.0,
    height_err_scale: float = 10.0,
) -> Tensor:
    """Reward for following a path (staying close to target position).

    Computes exponential reward based on:
    - Horizontal distance to target position
    - Optionally: vertical distance to target position

    Args:
        head_pos: Current head position [num_envs, 3] (ground-relative).
        tar_pos: Target position from path [num_envs, 3] (ground-relative).
        height_conditioned: Whether to include height in reward.
        pos_err_scale: Coefficient for position error.
        height_err_scale: Coefficient for height error.

    Returns:
        Reward [num_envs] in range [0, 1].
    """
    pos_diff = tar_pos[..., 0:2] - head_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    height_diff = tar_pos[..., 2] - head_pos[..., 2]
    height_err = height_diff * height_diff

    pos_reward = torch.exp(-pos_err_scale * pos_err)
    height_reward = torch.exp(-height_err_scale * height_err)

    if height_conditioned:
        # Multiplicative reward ensures both terms are properly met.
        reward = pos_reward * height_reward
    else:
        reward = pos_reward

    return reward


# =============================================================================
# Target Reaching Reward Kernels
# =============================================================================

def compute_target_rew(
    root_pos: Tensor,
    tar_pos: Tensor,
    tar_proximity_threshold: float,
    pos_err_scale: float = 0.42,
) -> Tensor:
    """Distance-based target reaching reward."""
    pos_diff = tar_pos[..., :2] - root_pos[..., :2]
    dist = torch.linalg.norm(pos_diff, dim=-1)
    reward = torch.exp(-pos_err_scale * dist)
    return torch.where(dist < tar_proximity_threshold, torch.ones_like(reward), reward)


__all__ = [
    "compute_heading_velocity_rew",
    "compute_track_lin_vel_xy_yaw_frame_exp",
    "compute_track_ang_vel_z_world_exp",
    "compute_lin_vel_z_l2",
    "compute_ang_vel_xy_l2",
    "compute_body_orientation_l2",
    "compute_path_following_rew",
    "compute_target_rew",
]
