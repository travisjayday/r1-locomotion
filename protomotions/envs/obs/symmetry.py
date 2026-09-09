# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observation packing raw physical state for the PPO left-right symmetry loss.

This is a side-channel, not an actor input: it packs exactly the physical
quantities ``PPOAgentConfig.symmetry`` needs to reconstruct a correctly
mirrored ``proprio``/``steering`` observation pair during the actor update
(see ``protomotions.utils.mirroring`` and ``protomotions.agents.ppo.agent``).
"""

from torch import Tensor

from protomotions.utils.mirroring import pack_symmetry_state


def compute_symmetry_state_obs(
    dof_pos: Tensor,
    dof_vel: Tensor,
    anchor_rot: Tensor,
    root_local_ang_vel: Tensor,
    tar_local_vel: Tensor,
    tar_yaw_rate: Tensor,
) -> Tensor:
    """Pack raw physical state for the PPO symmetry loss. See ``pack_symmetry_state``."""
    return pack_symmetry_state(
        dof_pos, dof_vel, anchor_rot, root_local_ang_vel, tar_local_vel, tar_yaw_rate
    )


__all__ = ["compute_symmetry_state_obs"]
