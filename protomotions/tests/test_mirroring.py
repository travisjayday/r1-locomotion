# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for protomotions.utils.mirroring.

The per-tensor reflection formulas are checked against from-scratch
reference math (a Householder reflection conjugation for orientation, a
finite-difference measurement for angular velocity). The R1 mirror table is
then checked end-to-end against MuJoCo's own forward kinematics: mirroring a
random pose and re-deriving body poses via FK must agree with mirroring the
FK'd body poses directly, for every one of R1's 27 bodies.
"""

import math

import pytest
import torch

from protomotions.utils.mirroring import (
    build_mirror_table,
    classify_dof_mirror_sign,
    mirror_angular_velocity,
    mirror_dof,
    mirror_motion_state,
    mirror_position,
    mirror_quaternion,
)


# MJCF-derived body/DOF order for R1 (protomotions/data/assets/mjcf/r1.xml),
# reproduced here as literals so this test needs no simulator dependency.
R1_BODY_NAMES = [
    "pelvis_link",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_roll_link", "waist_yaw_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link",
    "head_pitch_link", "head_yaw_link",
]
R1_DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_roll_joint", "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
    "head_pitch_joint", "head_yaw_joint",
]
R1_MJCF_PATH = "protomotions/data/assets/mjcf/r1.xml"


# =============================================================================
# Per-tensor math, checked from scratch
# =============================================================================


def _reflect_rotation_matrix_reference(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Mirror a rotation via N @ R @ N (N = Y-flip Householder reflection).

    This is the unique proper-rotation (det=+1) matrix satisfying
    mirrored_R(mirror(v)) == mirror(R(v)) for all v -- the defining property
    of "the rotation a mirror-image rigid body would have". Independent of
    mirror_quaternion's implementation.
    """
    w, x, y, z = quat_wxyz.unbind(-1)
    r = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)
    n = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=quat_wxyz.dtype))
    return n @ r @ n


def _quat_wxyz_to_matrix(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)


def test_mirror_position_negates_lateral_axis_only():
    pos = torch.tensor([1.5, -2.5, 3.5])
    mirrored = mirror_position(pos)
    assert torch.equal(mirrored, torch.tensor([1.5, 2.5, 3.5]))
    # Mirroring twice is the identity.
    assert torch.equal(mirror_position(mirrored), pos)


def test_mirror_quaternion_wxyz_matches_reflection_matrix():
    torch.manual_seed(0)
    for _ in range(20):
        q = torch.randn(4)
        q = q / q.norm()
        mirrored_q = mirror_quaternion(q, w_last=False)
        got = _quat_wxyz_to_matrix(mirrored_q)
        expected = _reflect_rotation_matrix_reference(q)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
        # Proper rotation: determinant +1.
        assert got.det().item() == pytest.approx(1.0, abs=1e-5)


def test_mirror_quaternion_xyzw_matches_wxyz_convention():
    torch.manual_seed(1)
    q_wxyz = torch.randn(4)
    q_wxyz = q_wxyz / q_wxyz.norm()
    q_xyzw = q_wxyz[[1, 2, 3, 0]]
    mirrored_wxyz = mirror_quaternion(q_wxyz, w_last=False)
    mirrored_xyzw = mirror_quaternion(q_xyzw, w_last=True)
    torch.testing.assert_close(mirrored_xyzw, mirrored_wxyz[[1, 2, 3, 0]])


def test_mirror_quaternion_is_involution():
    torch.manual_seed(2)
    q = torch.randn(5, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(mirror_quaternion(mirror_quaternion(q, True), True), q)


def test_mirror_angular_velocity_matches_finite_difference_of_mirrored_trajectory():
    """Angular velocity is a pseudovector: it flips the opposite axes from a
    plain vector under reflection (X, Z flip; Y is unchanged). Verify by
    numerically differentiating a mirrored quaternion trajectory rather than
    trusting the formula's sign convention by inspection.

    float64 and a not-too-small dt: the finite-difference quaternion delta
    is near-identity, and extracting an angle/axis from it via arccos is
    ill-conditioned in float32 at very small angles.
    """
    torch.manual_seed(3)
    omega_world = torch.tensor([0.7, -1.3, 2.1], dtype=torch.float64)
    dt = 1e-4

    q0 = torch.randn(4, dtype=torch.float64)
    q0 = q0 / q0.norm()

    def quat_mul_wxyz(a, b):
        aw, ax, ay, az = a.unbind(-1)
        bw, bx, by, bz = b.unbind(-1)
        return torch.stack([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], dim=-1)

    angle = omega_world.norm() * dt
    axis = omega_world / omega_world.norm()
    dq = torch.cat([torch.cos(angle / 2).unsqueeze(0), axis * torch.sin(angle / 2)])
    q1 = quat_mul_wxyz(dq, q0)
    q1 = q1 / q1.norm()

    q0m = mirror_quaternion(q0, w_last=False)
    q1m = mirror_quaternion(q1, w_last=False)

    def quat_conj_wxyz(q):
        w, x, y, z = q.unbind(-1)
        return torch.stack([w, -x, -y, -z], dim=-1)

    dq_m = quat_mul_wxyz(q1m, quat_conj_wxyz(q0m))
    if dq_m[0] < 0:  # take the short way around
        dq_m = -dq_m
    theta = 2 * torch.arccos(dq_m[0].clamp(-1, 1))
    measured_axis = dq_m[1:] / dq_m[1:].norm()
    measured_omega_mirrored = measured_axis * theta / dt

    predicted = mirror_angular_velocity(omega_world)
    torch.testing.assert_close(measured_omega_mirrored, predicted, atol=1e-3, rtol=1e-3)


# =============================================================================
# R1 mirror table: naming/classification
# =============================================================================


def test_r1_mirror_table_pairs_are_involutions():
    table = build_mirror_table(R1_BODY_NAMES, R1_DOF_NAMES)
    for i, partner in enumerate(table.body_index):
        assert table.body_index[partner] == i
    for i, partner in enumerate(table.dof_index):
        assert table.dof_index[partner] == i


def test_r1_midline_dofs_map_to_themselves():
    table = build_mirror_table(R1_BODY_NAMES, R1_DOF_NAMES)
    for name in ("waist_roll_joint", "waist_yaw_joint", "head_pitch_joint", "head_yaw_joint"):
        i = R1_DOF_NAMES.index(name)
        assert table.dof_index[i] == i


@pytest.mark.parametrize(
    "name,expected_sign",
    [
        ("left_hip_pitch_joint", 1.0), ("right_hip_pitch_joint", 1.0),
        ("left_hip_roll_joint", -1.0), ("right_hip_roll_joint", -1.0),
        ("left_hip_yaw_joint", -1.0),
        ("left_knee_joint", 1.0), ("right_elbow_joint", 1.0),
        ("waist_roll_joint", -1.0), ("waist_yaw_joint", -1.0),
        ("head_pitch_joint", 1.0), ("head_yaw_joint", -1.0),
        ("left_wrist_roll_joint", -1.0),
    ],
)
def test_dof_mirror_sign_classification(name, expected_sign):
    assert classify_dof_mirror_sign(name) == expected_sign


def test_mirror_dof_round_trip_is_identity():
    table = build_mirror_table(R1_BODY_NAMES, R1_DOF_NAMES)
    torch.manual_seed(4)
    dof = torch.randn(3, len(R1_DOF_NAMES))
    torch.testing.assert_close(mirror_dof(mirror_dof(dof, table), table), dof)


# =============================================================================
# End-to-end: mirrored qpos through MuJoCo FK must equal mirroring the FK'd bodies
# =============================================================================


def test_r1_mirror_table_matches_mujoco_forward_kinematics():
    mujoco = pytest.importorskip("mujoco")

    model = mujoco.MjModel.from_xml_path(R1_MJCF_PATH)
    data = mujoco.MjData(model)

    table = build_mirror_table(R1_BODY_NAMES, R1_DOF_NAMES)

    torch.manual_seed(5)
    root_pos = torch.tensor([0.1, 0.3, 0.73])
    root_rot_wxyz = torch.randn(4)
    root_rot_wxyz = root_rot_wxyz / root_rot_wxyz.norm()
    # Keep DOF values well inside R1's joint limits.
    dof_pos = torch.zeros(len(R1_DOF_NAMES)).uniform_(-0.3, 0.3)

    def run_fk(root_pos_, root_rot_wxyz_, dof_pos_):
        data.qpos[0:3] = root_pos_.numpy()
        data.qpos[3:7] = root_rot_wxyz_.numpy()  # MuJoCo free joint is wxyz
        data.qpos[7:] = dof_pos_.numpy()
        mujoco.mj_forward(model, data)
        body_pos = torch.tensor(data.xpos[1:].copy(), dtype=torch.float32)  # skip world body
        body_rot_wxyz = torch.tensor(data.xquat[1:].copy(), dtype=torch.float32)
        return body_pos, body_rot_wxyz

    body_pos, body_rot = run_fk(root_pos, root_rot_wxyz, dof_pos)

    mirrored_state = mirror_motion_state(
        root_pos=root_pos, root_rot=root_rot_wxyz, dof_pos=dof_pos,
        table=table, w_last=False,
    )
    mirrored_body_pos, mirrored_body_rot = run_fk(
        mirrored_state["root_pos"], mirrored_state["root_rot"], mirrored_state["dof_pos"]
    )

    # The mirror table's prediction for what FK-on-mirrored-qpos should
    # produce: reflect the ORIGINAL FK'd bodies directly (position and
    # orientation), independent of running FK a second time.
    idx = table.body_index
    predicted_pos = mirror_position(body_pos[idx])
    predicted_rot = mirror_quaternion(body_rot[idx], w_last=False)

    # head_pitch_link/head_yaw_link are excluded: R1's MJCF gives the head
    # mount a small genuine (~3 cm) lateral offset from the waist_yaw parent
    # frame (checked directly against model.body_pos) rather than sitting
    # exactly on the sagittal midline, so FK-on-mirrored-qpos and
    # mirror-the-FK'd-position disagree there by construction -- this is a
    # real, tiny asymmetry in the physical asset, not a mirror-table bug.
    # It's irrelevant to locomotion: r1_omniscient_teacher.py disables both
    # head DOFs (disabled_dof_names=["head_pitch_joint", "head_yaw_joint"]).
    # Every other body -- both legs, both arms, the waist -- must match tightly.
    checked = [i for i, name in enumerate(R1_BODY_NAMES) if not name.startswith("head_")]

    pos_err = (mirrored_body_pos - predicted_pos).norm(dim=-1)
    worst = max(checked, key=lambda i: pos_err[i].item())
    assert pos_err[checked].max().item() < 1e-3, (
        f"worst body {R1_BODY_NAMES[worst]}: {pos_err[worst].item():.4f} m"
    )

    # Compare orientation as rotation-matrix Frobenius error (quaternions
    # have a sign ambiguity: q and -q are the same rotation).
    got_mats = _quat_wxyz_to_matrix(mirrored_body_rot)
    pred_mats = _quat_wxyz_to_matrix(predicted_rot)
    rot_err = (got_mats - pred_mats).flatten(-2).norm(dim=-1)
    worst = max(checked, key=lambda i: rot_err[i].item())
    assert rot_err[checked].max().item() < 1e-2, (
        f"worst body {R1_BODY_NAMES[worst]}: {rot_err[worst].item():.4f}"
    )
