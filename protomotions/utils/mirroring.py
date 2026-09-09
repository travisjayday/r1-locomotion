# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Left-right (sagittal-plane) mirroring for symmetric humanoids.

Reflects world-frame kinematic quantities across the robot's sagittal (XZ)
plane -- i.e. negates the lateral Y axis. This is the standard ProtoMotions
world convention (X-forward, Y-left, Z-up; see e.g. the ``lateral_offset``
usage in ``pyroki/visualize_g1_r1_retarget.py``).

Two pieces compose to mirror a full character pose:

1. The per-tensor math below (``mirror_position``, ``mirror_quaternion``,
   ``mirror_linear_velocity``, ``mirror_angular_velocity``) -- reflection
   formulas, robot-agnostic.
2. A per-robot **mirror table**: which body/DOF index swaps with which under
   left<->right reflection, and whether that DOF's *own* value additionally
   flips sign (roll/yaw axes flip; pitch/hinge axes don't -- see
   ``classify_dof_mirror_sign``). Build one with ``build_mirror_table``, from
   any ``RobotConfig.kinematic_info.body_names`` / ``dof_names`` that follow
   the ``left_*``/``right_*`` naming convention (r1, g1, ...).

Formulas are verified in ``protomotions/tests/test_mirroring.py`` both in
isolation (random rotations/angular velocities against a from-scratch
reflection) and end-to-end against MuJoCo forward kinematics on the R1 model.

Usage in a diagnostic or augmentation script::

    from protomotions.utils.mirroring import build_mirror_table, mirror_motion_state

    table = build_mirror_table(robot_config.kinematic_info.body_names,
                                robot_config.kinematic_info.dof_names)
    mirrored = mirror_motion_state(
        root_pos=root_pos, root_rot=root_rot,
        root_vel=root_vel, root_ang_vel=root_ang_vel,
        dof_pos=dof_pos, dof_vel=dof_vel,
        body_pos=body_pos, body_rot=body_rot,
        body_vel=body_vel, body_ang_vel=body_ang_vel,
        table=table, w_last=True,
    )
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor


# =============================================================================
# Per-tensor reflection math (robot-agnostic)
# =============================================================================


@torch.jit.script
def mirror_position(pos: Tensor) -> Tensor:
    """Reflect a world-frame position (or linear velocity) across the XZ plane."""
    out = pos.clone()
    out[..., 1] = -out[..., 1]
    return out


# Linear velocity transforms exactly like position under a reflection.
mirror_linear_velocity = mirror_position


@torch.jit.script
def mirror_local_planar_velocity(vel2: Tensor) -> Tensor:
    """Reflect a 2D body-frame planar velocity ``[..., 2]`` (vx, vy) across the sagittal plane.

    Used for steering-task commands (``tar_local_vel``): vx (forward) is
    unaffected, vy (lateral) flips sign -- the same rule as ``mirror_position``,
    restricted to a 2D body-frame vector that never carries a Z component.
    """
    out = vel2.clone()
    out[..., 1] = -out[..., 1]
    return out


@torch.jit.script
def mirror_angular_velocity(ang_vel: Tensor) -> Tensor:
    """Reflect a world-frame angular velocity (a pseudovector) across the XZ plane.

    Pseudovectors pick up the opposite pattern from ordinary vectors under an
    improper (reflection) transform: the components *in* the mirror plane
    (X, Z) flip sign and the component normal to it (Y) is unchanged --
    verified in test_mirroring.py against a finite-difference measurement of
    the actual angular velocity of a mirrored quaternion trajectory.
    """
    out = ang_vel.clone()
    out[..., 0] = -out[..., 0]
    out[..., 2] = -out[..., 2]
    return out


@torch.jit.script
def mirror_quaternion(quat: Tensor, w_last: bool) -> Tensor:
    """Reflect a world-frame orientation across the XZ plane.

    For wxyz quaternions this is exactly ``(w, -x, y, -z)`` -- verified
    numerically (max error ~1e-7 over random rotations) against reflecting
    the rotation matrix directly (``N @ R @ N`` for the Y-flip Householder
    matrix N, which is the unique proper-rotation formula satisfying
    ``mirrored_R(mirror(v)) == mirror(R(v))`` for all v).
    """
    shape = quat.shape
    flat = quat.reshape(-1, 4)
    if w_last:
        # xyzw: negate x and z, keep y and w.
        flat = flat * torch.tensor([-1.0, 1.0, -1.0, 1.0], device=quat.device, dtype=quat.dtype)
    else:
        # wxyz: negate x and z, keep w and y.
        flat = flat * torch.tensor([1.0, -1.0, 1.0, -1.0], device=quat.device, dtype=quat.dtype)
    return flat.reshape(shape)


# =============================================================================
# Per-robot mirror table
# =============================================================================


@dataclass(frozen=True)
class MirrorTable:
    """Left<->right index/sign maps for one robot's bodies and DOFs.

    ``*_index[i]`` is the index whose value moves into slot ``i`` after
    mirroring (itself, for a midline body/DOF). ``dof_sign[i]`` additionally
    flips the DOF's own scalar value where the mirrored joint axis reverses
    (roll, yaw); it is +1 for axes unaffected by the reflection (pitch,
    hinge-type joints such as knee/elbow).
    """

    body_index: List[int]
    dof_index: List[int]
    dof_sign: List[float]


def _mirror_name(name: str) -> Optional[str]:
    """Return the left<->right-swapped name, or None if it has no side prefix."""
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    return None


def _build_name_permutation(names: List[str]) -> List[int]:
    by_name = {name: i for i, name in enumerate(names)}
    permutation = []
    for i, name in enumerate(names):
        mirrored_name = _mirror_name(name)
        if mirrored_name is None:
            permutation.append(i)  # midline: maps to itself
            continue
        if mirrored_name not in by_name:
            raise ValueError(
                f"{name!r} has no mirror partner {mirrored_name!r} in {names}"
            )
        permutation.append(by_name[mirrored_name])
    return permutation


# Suffixes for joint axes that reverse sign under a left-right reflection
# (roll: rotation about the forward axis; yaw: rotation about the vertical
# axis -- both lie in the XZ mirror plane). Pitch/hinge axes (rotation about
# the lateral Y axis: pitch joints, and pure hinges like knee/elbow/ankle
# rod joints) are unaffected -- see the module docstring and test_mirroring.py
# for the derivation and per-joint URDF-limit cross-check.
_SIGN_FLIP_SUFFIXES = ("_roll", "_roll_joint", "_yaw", "_yaw_joint")
_NO_FLIP_SUFFIXES = (
    "_pitch", "_pitch_joint",
    "knee", "elbow",
)


def classify_dof_mirror_sign(dof_name: str) -> float:
    """Infer whether a DOF's value flips sign under left-right mirroring.

    Roll and yaw axes lie in the sagittal mirror plane and flip; pitch and
    hinge (knee/elbow) axes are perpendicular to it and don't. Robots whose
    joints don't follow this naming should build a ``MirrorTable`` by hand
    instead of via ``build_mirror_table``.
    """
    stem = dof_name[:-len("_joint")] if dof_name.endswith("_joint") else dof_name
    if any(stem.endswith(suffix) for suffix in ("_roll", "_yaw")):
        return -1.0
    if any(stem.endswith(suffix) for suffix in ("_pitch",)) or "knee" in stem or "elbow" in stem:
        return 1.0
    raise ValueError(
        f"Cannot infer mirror sign for DOF {dof_name!r} from its name; "
        "pass an explicit dof_sign_overrides entry to build_mirror_table."
    )


def build_mirror_table(
    body_names: List[str],
    dof_names: List[str],
    dof_sign_overrides: Optional[dict] = None,
) -> MirrorTable:
    """Build a MirrorTable from ``left_*``/``right_*``-named bodies and DOFs.

    Args:
        body_names: ``RobotConfig.kinematic_info.body_names``.
        dof_names: ``RobotConfig.kinematic_info.dof_names``.
        dof_sign_overrides: Optional ``{dof_name: sign}`` to override
            ``classify_dof_mirror_sign`` for joints whose axis convention
            can't be inferred from the name (e.g. a genuinely asymmetric
            axial joint).
    """
    overrides = dof_sign_overrides or {}
    body_index = _build_name_permutation(body_names)
    dof_index = _build_name_permutation(dof_names)
    dof_sign = [
        overrides[name] if name in overrides else classify_dof_mirror_sign(name)
        for name in dof_names
    ]
    return MirrorTable(body_index=body_index, dof_index=dof_index, dof_sign=dof_sign)


# =============================================================================
# Full-state mirroring
# =============================================================================


def mirror_dof(dof: Tensor, table: MirrorTable) -> Tensor:
    """Mirror a [..., num_dofs] DOF-space tensor (position or velocity)."""
    sign = torch.tensor(table.dof_sign, device=dof.device, dtype=dof.dtype)
    return dof[..., table.dof_index] * sign


def mirror_bodies(
    body_pos: Tensor,
    body_rot: Tensor,
    body_vel: Optional[Tensor],
    body_ang_vel: Optional[Tensor],
    table: MirrorTable,
    w_last: bool,
) -> tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """Mirror [..., num_bodies, {3,4}] rigid-body tensors."""
    idx = table.body_index
    pos = mirror_position(body_pos[..., idx, :])
    rot = mirror_quaternion(body_rot[..., idx, :], w_last=w_last)
    vel = mirror_linear_velocity(body_vel[..., idx, :]) if body_vel is not None else None
    ang_vel = (
        mirror_angular_velocity(body_ang_vel[..., idx, :])
        if body_ang_vel is not None
        else None
    )
    return pos, rot, vel, ang_vel


def mirror_motion_state(
    root_pos: Tensor,
    root_rot: Tensor,
    dof_pos: Tensor,
    table: MirrorTable,
    w_last: bool,
    root_vel: Optional[Tensor] = None,
    root_ang_vel: Optional[Tensor] = None,
    dof_vel: Optional[Tensor] = None,
    body_pos: Optional[Tensor] = None,
    body_rot: Optional[Tensor] = None,
    body_vel: Optional[Tensor] = None,
    body_ang_vel: Optional[Tensor] = None,
) -> dict:
    """Mirror a full character state (root + DOFs, optionally all bodies too).

    Every argument keeps its input shape; only root_pos/root_rot/dof_pos are
    required (e.g. for mirroring a live sim qpos). Pass the rigid-body and
    velocity tensors too when mirroring a recorded trajectory (a MotionLib
    ``gts``/``grs``/``gvs``/``gavs``/``dps``/``dvs``-style dict).
    """
    result = {
        "root_pos": mirror_position(root_pos),
        "root_rot": mirror_quaternion(root_rot, w_last=w_last),
        "dof_pos": mirror_dof(dof_pos, table),
    }
    if root_vel is not None:
        result["root_vel"] = mirror_linear_velocity(root_vel)
    if root_ang_vel is not None:
        result["root_ang_vel"] = mirror_angular_velocity(root_ang_vel)
    if dof_vel is not None:
        result["dof_vel"] = mirror_dof(dof_vel, table)
    if body_pos is not None:
        pos, rot, vel, ang_vel = mirror_bodies(
            body_pos, body_rot, body_vel, body_ang_vel, table, w_last=w_last
        )
        result["body_pos"] = pos
        result["body_rot"] = rot
        if vel is not None:
            result["body_vel"] = vel
        if ang_vel is not None:
            result["body_ang_vel"] = ang_vel
    return result


# =============================================================================
# Symmetry-loss state packing (PPOAgentConfig.symmetry)
# =============================================================================


def pack_symmetry_state(
    dof_pos: Tensor,
    dof_vel: Tensor,
    anchor_rot: Tensor,
    root_local_ang_vel: Tensor,
    tar_local_vel: Tensor,
    tar_yaw_rate: Tensor,
) -> Tensor:
    """Pack the raw physical quantities a symmetry loss needs to remirror obs.

    These are exactly the inputs to ``compute_humanoid_reduced_coords_observations``
    (minus the unused root_height/root_vel variants) and
    ``compute_yaw_rate_steering_obs`` -- packing them lets the PPO actor loss
    reconstruct a correctly-mirrored ``proprio``/``steering`` pair by calling
    those same functions again on mirrored inputs, rather than hand-deriving a
    mirror rule for the already-flattened observation vectors.

    Concatenation order: ``[dof_pos(D), dof_vel(D), anchor_rot(4),
    root_local_ang_vel(3), tar_local_vel(2), tar_yaw_rate(1)]``. Keep in sync
    with ``unpack_symmetry_state``.
    """
    num_envs = dof_pos.shape[0]
    return torch.cat(
        (
            dof_pos.reshape(num_envs, -1),
            dof_vel.reshape(num_envs, -1),
            anchor_rot.reshape(num_envs, -1),
            root_local_ang_vel.reshape(num_envs, -1),
            tar_local_vel.reshape(num_envs, -1),
            tar_yaw_rate.reshape(num_envs, -1),
        ),
        dim=-1,
    )


def unpack_symmetry_state(
    packed: Tensor, num_dofs: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Inverse of ``pack_symmetry_state``.

    Returns ``(dof_pos, dof_vel, anchor_rot, root_local_ang_vel,
    tar_local_vel, tar_yaw_rate)``.
    """
    i = 0
    dof_pos = packed[..., i : i + num_dofs]
    i += num_dofs
    dof_vel = packed[..., i : i + num_dofs]
    i += num_dofs
    anchor_rot = packed[..., i : i + 4]
    i += 4
    root_local_ang_vel = packed[..., i : i + 3]
    i += 3
    tar_local_vel = packed[..., i : i + 2]
    i += 2
    tar_yaw_rate = packed[..., i : i + 1].squeeze(-1)
    return dof_pos, dof_vel, anchor_rot, root_local_ang_vel, tar_local_vel, tar_yaw_rate
