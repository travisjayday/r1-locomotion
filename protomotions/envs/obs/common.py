# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Small reusable observation tensor transforms."""

from torch import Tensor

from protomotions.utils.rotations import quat_to_tan_norm


def flatten_batch(x: Tensor) -> Tensor:
    """Convert an observation to float and flatten every non-batch axis."""
    return x.float().reshape(x.shape[0], -1)


def quaternion_to_tan_norm_and_flatten(
    quaternion: Tensor, w_last: bool = True
) -> Tensor:
    """Encode quaternions as sign-invariant 6D rotations and flatten horizons.

    A quaternion and its negation describe the same rotation, but exposing the
    four raw components to a policy makes that equivalent pair look maximally
    different. The tangent/normal representation rotates the canonical X and Z
    axes instead, so ``q`` and ``-q`` produce identical observations.

    Args:
        quaternion: Quaternion tensor shaped ``[batch, ..., 4]``.
        w_last: Whether the quaternion scalar component is last.

    Returns:
        Tensor shaped ``[batch, product(..., 6)]``.
    """
    rotation_6d = quat_to_tan_norm(quaternion, w_last=w_last)
    return rotation_6d.float().reshape(quaternion.shape[0], -1)


def select_bodies_and_flatten(
    x: Tensor, body_indices: list[int], body_axis: int = -2
) -> Tensor:
    """Select a body axis and flatten non-batch axes."""
    if body_axis == -1:
        selected = x[..., body_indices]
    elif body_axis == -2:
        selected = x[..., body_indices, :]
    else:
        raise ValueError("select_bodies_and_flatten supports body_axis -1 or -2")
    return selected.float().reshape(x.shape[0], -1)
