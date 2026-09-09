# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch-retarget Kimodo G1 CSV or PyRoki G1 NPZ trajectories to Unitree R1.

This is robot-to-robot retargeting. Kimodo CSV input is headerless and contains
root xyz in meters, root quaternion wxyz, then 29 G1 joint angles in radians.
NPZ input is expected to contain ``base_frame_pos``, ``base_frame_wxyz``, and
``joint_angles``. G1 forward kinematics supplies the targets; R1 root poses and
joint angles are optimized over the full trajectory.

The morphology scale factors are measured from the two neutral URDF poses at
runtime.  In particular, limbs are scaled segment-by-segment instead of using
one height scale for the whole robot.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import NamedTuple, TypedDict
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_G1_URDF = (
    SCRIPT_DIR / "../protomotions/data/assets/urdf/for_retargeting/g1.urdf"
).resolve()
DEFAULT_R1_URDF = (
    SCRIPT_DIR / "../protomotions/data/assets/urdf/for_retargeting/r1.urdf"
).resolve()

# The link frames at these locations represent the same anatomical landmarks.
SEMANTIC_LINKS = (
    ("pelvis", "pelvis_contour_link", "pelvis_link"),
    ("left_hip", "left_hip_yaw_link", "left_hip_yaw_link"),
    ("left_knee", "left_knee_link", "left_knee_link"),
    ("left_ankle", "left_ankle_roll_link", "left_ankle_roll_link"),
    ("right_hip", "right_hip_yaw_link", "right_hip_yaw_link"),
    ("right_knee", "right_knee_link", "right_knee_link"),
    ("right_ankle", "right_ankle_roll_link", "right_ankle_roll_link"),
    ("torso", "torso_link", "waist_yaw_link"),
    ("left_shoulder", "left_shoulder_pitch_link", "left_shoulder_pitch_link"),
    ("left_elbow", "left_elbow_link", "left_elbow_link"),
    ("left_wrist", "left_wrist_roll_link", "left_wrist_roll_link"),
    ("right_shoulder", "right_shoulder_pitch_link", "right_shoulder_pitch_link"),
    ("right_elbow", "right_elbow_link", "right_elbow_link"),
    ("right_wrist", "right_wrist_roll_link", "right_wrist_roll_link"),
)
SEMANTIC_NAMES = tuple(item[0] for item in SEMANTIC_LINKS)
SEMANTIC_INDEX = {name: i for i, name in enumerate(SEMANTIC_NAMES)}

# Four sole-surface samples, expressed in each ankle-roll frame.
# Order: rear-left, rear-right, front-left, front-right.  The R1 contact
# spheres are centered at z=-0.045 with radius 0.01, so their contact surface
# is z=-0.055.  Using their centers would place the collider 1 cm underground.
G1_SOLE_POINTS = np.array(
    [[-0.05, 0.025, -0.03], [-0.05, -0.025, -0.03],
     [0.12, 0.03, -0.03], [0.12, -0.03, -0.03]], dtype=np.float32
)
R1_SOLE_POINTS = np.array(
    [[-0.04, 0.025, -0.055], [-0.04, -0.025, -0.055],
     [0.115, 0.025, -0.055], [0.115, -0.025, -0.055]], dtype=np.float32
)

# Collision meshes are deliberately not loaded by the batch solver, so use a
# compact geometric proxy for the R1 wrist/hand mesh.  Its STL spans local
# x=[-0.002, 0.143] m and is about 0.076 m wide.  These centerline samples are
# checked against capsule approximations of both thighs.  A 9.5 cm centerline
# clearance covers the roughly 4 cm hand and 5 cm thigh half-widths plus a
# small numerical/mesh-shape margin.
R1_HAND_CENTERLINE_POINTS = np.array(
    [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0],
     [0.10, 0.0, 0.0], [0.14, 0.0, 0.0]], dtype=np.float32
)
R1_HAND_THIGH_CLEARANCE = 0.095


class Weights(TypedDict):
    landmark: float
    root_position: float
    root_orientation: float
    root_roll_orientation: float
    root_yaw_orientation: float
    hip_roll_tracking: float
    torso_orientation: float
    arm_orientation: float
    hand_thigh_clearance: float
    arm_smoothness: float
    arm_acceleration: float
    swing_foot_orientation: float
    seed: float
    joint_smoothness: float
    joint_acceleration: float
    root_smoothness: float
    joint_velocity_limit: float
    foot_slip: float
    foot_anchor: float
    foot_penetration: float
    foot_height: float
    foot_tilt: float
    swing_foot_height: float


DEFAULT_WEIGHTS = Weights(
    landmark=32.0,
    root_position=3.0,
    root_orientation=0.5,
    root_roll_orientation=16.0,
    # Root yaw and waist/hip yaw form an otherwise weakly observable gauge:
    # equal-and-opposite rotations leave world-space torso and feet unchanged.
    # Track source heading firmly while retaining root-pitch freedom for the
    # G1/R1 torso morphology mismatch.
    root_yaw_orientation=16.0,
    hip_roll_tracking=12.0,
    torso_orientation=16.0,
    # Elbow and wrist-roll link orientations disambiguate axial shoulder and
    # wrist rotation when an arm is nearly straight. Position landmarks alone
    # are singular in that configuration and otherwise encourage overreach.
    arm_orientation=4.0,
    # Keep the relatively long R1 hand meshes out of both thighs.  This is a
    # one-sided morphology constraint, not a pose-tracking term.
    hand_thigh_clearance=400.0,
    arm_smoothness=20.0,
    arm_acceleration=20.0,
    swing_foot_orientation=12.0,
    seed=0.35,
    joint_smoothness=2.0,
    joint_acceleration=1.0,
    root_smoothness=1.0,
    joint_velocity_limit=20.0,
    foot_slip=160.0,
    # Deliberately much weaker than foot_slip. The anchor is a spring to an
    # absolute stance median, so its error does not shrink as the foot moves:
    # at 200 it yanks the foot at touchdown and forbids the pivot that spin
    # motions need, which is jerk baked into the reference. foot_slip is a
    # frame-to-frame difference and cannot do that, so it carries stance
    # coherence instead. Measured on walk_spin_backward_clockwise, 200 -> 80
    # halves foot jerk (p99 1074 -> 531 m/s^3), cuts peak foot acceleration
    # 47 -> 26 m/s^2 and removes every >40 deg direction reversal, for 1.2 mm
    # per frame of extra stance drift. Below 80 buys no further smoothness
    # and keeps costing slip. A mimic policy can learn out slip -- friction
    # and the contact_slip reward both oppose it -- but it cannot learn out
    # jerk, because the jerk is the tracking target.
    foot_anchor=80.0,
    foot_penetration=100.0,
    foot_height=10.0,
    foot_tilt=30.0,
    swing_foot_height=70.0,
)


class Morphology(NamedTuple):
    hip: float
    thigh: float
    shin: float
    torso: float
    shoulder: float
    upper_arm: float
    forearm: float
    locomotion: float


def _mean_distance(points: np.ndarray, pairs: list[tuple[str, str]]) -> float:
    lengths = [
        np.linalg.norm(points[SEMANTIC_INDEX[b]] - points[SEMANTIC_INDEX[a]])
        for a, b in pairs
    ]
    return float(np.mean(lengths))


def derive_morphology(g1_neutral: np.ndarray, r1_neutral: np.ndarray) -> Morphology:
    """Measure segment ratios from neutral-pose semantic link positions."""
    bilateral = {
        "hip": [("pelvis", "left_hip"), ("pelvis", "right_hip")],
        "thigh": [("left_hip", "left_knee"), ("right_hip", "right_knee")],
        "shin": [("left_knee", "left_ankle"), ("right_knee", "right_ankle")],
        "shoulder": [
            ("pelvis", "left_shoulder"), ("pelvis", "right_shoulder")
        ],
        "upper_arm": [
            ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow")
        ],
        "forearm": [
            ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist")
        ],
    }
    ratios = {
        name: _mean_distance(r1_neutral, pairs) / _mean_distance(g1_neutral, pairs)
        for name, pairs in bilateral.items()
    }
    torso = (
        np.linalg.norm(
            r1_neutral[SEMANTIC_INDEX["torso"]]
            - r1_neutral[SEMANTIC_INDEX["pelvis"]]
        )
        / np.linalg.norm(
            g1_neutral[SEMANTIC_INDEX["torso"]]
            - g1_neutral[SEMANTIC_INDEX["pelvis"]]
        )
    )
    g1_leg = _mean_distance(g1_neutral, bilateral["thigh"]) + _mean_distance(
        g1_neutral, bilateral["shin"]
    )
    r1_leg = _mean_distance(r1_neutral, bilateral["thigh"]) + _mean_distance(
        r1_neutral, bilateral["shin"]
    )
    return Morphology(
        hip=ratios["hip"], thigh=ratios["thigh"], shin=ratios["shin"],
        torso=float(torso), shoulder=ratios["shoulder"],
        upper_arm=ratios["upper_arm"], forearm=ratios["forearm"],
        locomotion=float(r1_leg / g1_leg),
    )


def load_joint_metadata(urdf_path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Read actuated joint ordering, limits, and velocities in URDF order."""
    root = ET.parse(urdf_path).getroot()
    names, lower, upper, velocity = [], [], [], []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") not in ("revolute", "continuous", "prismatic"):
            continue
        limit = joint.find("limit")
        names.append(joint.attrib["name"])
        lower.append(float(limit.attrib.get("lower", "-inf")))
        upper.append(float(limit.attrib.get("upper", "inf")))
        velocity.append(float(limit.attrib.get("velocity", "20")))
    return names, np.asarray(lower), np.asarray(upper), np.asarray(velocity)


def forward_kinematics_world(
    robot: pk.Robot, root_wxyz_xyz: jnp.ndarray, joints: jnp.ndarray
) -> jaxlie.SE3:
    """Vectorized world-space FK for a trajectory."""
    def one(root_pose: jnp.ndarray, cfg: jnp.ndarray) -> jnp.ndarray:
        world_root = jaxlie.SE3(root_pose)
        root_links = jaxlie.SE3(robot.forward_kinematics(cfg=cfg))
        return (world_root @ root_links).wxyz_xyz

    return jaxlie.SE3(jax.vmap(one)(root_wxyz_xyz, joints))


def transform_points(T_world_link: jaxlie.SE3, points: jnp.ndarray) -> jnp.ndarray:
    """Apply each trajectory link transform to the same local point set."""
    rotations = T_world_link.rotation().as_matrix()
    return (
        jnp.einsum("tlij,pj->tlpi", rotations, points)
        + T_world_link.translation()[:, :, None, :]
    )


def scale_landmarks(source: np.ndarray, morphology: Morphology) -> np.ndarray:
    """Rebuild an R1-proportioned skeleton from G1 segment directions."""
    out = np.empty_like(source)
    pelvis = SEMANTIC_INDEX["pelvis"]
    src_root = source[:, pelvis]
    out[:, pelvis] = src_root[0] + (source[:, pelvis] - src_root[0]) * morphology.locomotion
    # Scale initial absolute root height as well; x/y origins remain unchanged.
    out[:, pelvis, 2] += src_root[0, 2] * (morphology.locomotion - 1.0)

    for side in ("left", "right"):
        hip, knee, ankle = [SEMANTIC_INDEX[f"{side}_{x}"] for x in ("hip", "knee", "ankle")]
        out[:, hip] = out[:, pelvis] + morphology.hip * (source[:, hip] - source[:, pelvis])
        out[:, knee] = out[:, hip] + morphology.thigh * (source[:, knee] - source[:, hip])
        out[:, ankle] = out[:, knee] + morphology.shin * (source[:, ankle] - source[:, knee])

        shoulder, elbow, wrist = [
            SEMANTIC_INDEX[f"{side}_{x}"] for x in ("shoulder", "elbow", "wrist")
        ]
        out[:, shoulder] = out[:, pelvis] + morphology.shoulder * (
            source[:, shoulder] - source[:, pelvis]
        )
        out[:, elbow] = out[:, shoulder] + morphology.upper_arm * (
            source[:, elbow] - source[:, shoulder]
        )
        out[:, wrist] = out[:, elbow] + morphology.forearm * (
            source[:, wrist] - source[:, elbow]
        )

    torso = SEMANTIC_INDEX["torso"]
    out[:, torso] = out[:, pelvis] + morphology.torso * (
        source[:, torso] - source[:, pelvis]
    )
    return out


def infer_contacts(
    sole_points: np.ndarray, fps: float, height_margin: float, speed_threshold: float
) -> np.ndarray:
    """Infer left/right stance phases from G1 sole height and horizontal speed."""
    center = sole_points.mean(axis=2)  # [T, 2, 3]
    speed = np.zeros(center.shape[:2], dtype=np.float32)
    speed[1:] = np.linalg.norm(np.diff(center[..., :2], axis=0), axis=-1) * fps
    speed[0] = speed[1] if len(speed) > 1 else 0.0
    min_height = np.percentile(sole_points[..., 2], 5.0, axis=(0, 2))
    low = np.min(sole_points[..., 2], axis=2) <= min_height[None] + height_margin
    contacts = low & (speed <= speed_threshold)
    # Remove isolated detections and softly cross-fade transitions.
    kernel = np.ones(5, dtype=np.float32) / 5.0
    return np.stack(
        [np.convolve(contacts[:, side].astype(np.float32), kernel, mode="same")
         for side in range(2)], axis=1
    )


def align_contact_onsets(
    point_contacts: np.ndarray, source_sole: np.ndarray, margin: float
) -> np.ndarray:
    """Delay each labeled touchdown until the source foot is actually down.

    The embedded Kimodo labels are not derived from the G1 sole geometry, and
    they lead it: measured over this dataset, 40% of touchdown labels fire
    with the lowest sole sample still more than 2 cm above the clip's own
    ground plane, 16% more than 4 cm, worst case 8 cm. The solver is then
    given contradictory instructions -- the landmark cost tracks a source
    ankle that is still descending, while the height/anchor costs plant the
    R1 foot on z=0 -- and it resolves the contradiction by throwing the leg
    down over the preceding frames and catching it, which reads as a snap
    with a reversal of the swing foot's travel direction.

    Softening the labels only spreads that conflict over more frames. Moving
    the onset to the frame the source foot genuinely arrives removes it.

    Args:
        point_contacts: [T, 2, P] labels in [0, 1].
        source_sole: [T, 2, P, 3] world-space source sole samples.
        margin: Clearance in metres, above each foot's own 5th-percentile
            ground plane, that still counts as ground contact.

    Returns:
        Same shape, with the leading frames of every stance interval cleared
        up to the first frame satisfying the margin. Intervals that never
        satisfy it are moved to their own lowest frame, so no stance is lost.
    """
    if len(point_contacts) < 2:
        return point_contacts.astype(np.float32)
    lowest = source_sole[..., 2].min(axis=2)
    clearance = lowest - np.percentile(lowest, 5.0, axis=0)[None]
    aligned = point_contacts.astype(np.float32).copy()
    active = point_contacts.max(axis=2) > 0.5
    for side in range(active.shape[1]):
        column = active[:, side]
        starts = np.flatnonzero(column & ~np.r_[False, column[:-1]])
        ends = np.flatnonzero(column & ~np.r_[column[1:], False]) + 1
        for start, end in zip(starts, ends):
            grounded = np.flatnonzero(clearance[start:end, side] <= margin)
            first = start + (
                int(grounded[0]) if len(grounded)
                else int(np.argmin(clearance[start:end, side]))
            )
            aligned[start:first, side, :] = 0.0
    return aligned


def soften_contact_labels(
    point_contacts: np.ndarray, ramp_frames: int
) -> np.ndarray:
    """Cross-fade binary contact labels at stance boundaries.

    Every contact-scaled cost in foot_costs uses sqrt(contact), which the
    comment there describes as following "the soft contact label linearly
    instead of quadratically" -- that only holds for labels that actually
    ramp. infer_contacts already produces soft labels via a box filter, but
    embedded (e.g. Kimodo) labels arrive strictly binary and bypass it, so
    every cost switched on/off as a step: at the last swing frame the swing
    clearance cost was at full strength holding the foot up, and one frame
    later it was exactly zero while the stance anchor pulled at full weight,
    dropping the foot several centimetres in a single frame.

    Uses the same box filter as infer_contacts so both paths agree.

    Args:
        point_contacts: [T, 2, P] labels in [0, 1].
        ramp_frames: Box-filter width; <= 1 leaves the labels untouched.

    Returns:
        Same shape, cross-faded across every transition.
    """
    if ramp_frames <= 1 or len(point_contacts) < 2:
        return point_contacts.astype(np.float32)
    kernel = np.ones(int(ramp_frames), dtype=np.float32) / float(ramp_frames)
    softened = np.empty_like(point_contacts, dtype=np.float32)
    for side in range(point_contacts.shape[1]):
        for point in range(point_contacts.shape[2]):
            softened[:, side, point] = np.convolve(
                point_contacts[:, side, point].astype(np.float32), kernel, mode="same"
            )
    return softened


def taper_swing_clearance(
    swing_clearance_targets: np.ndarray,
    contacts: np.ndarray,
    taper_frames: int,
) -> np.ndarray:
    """Fade the requested swing clearance to zero as touchdown approaches.

    The target is the source sole's height above its own stance baseline,
    amplified by --swing-height-scale. The embedded contact label can call
    touchdown while the source foot is still several centimetres up and
    descending, so without a taper the solver is asked to hold the foot high
    right up to the frame the anchor plants it -- an impossible gap it closes
    with a single-frame drop. Scaling the request down over the approach
    leaves the foot already near the ground when contact begins.

    Args:
        swing_clearance_targets: [T, 2] requested clearance in metres.
        contacts: [T, 2] contact labels used only to locate touchdown frames.
        taper_frames: Frames over which the request ramps back to full value
            ahead of touchdown; <= 0 disables the taper.

    Returns:
        [T, 2] tapered targets.
    """
    if taper_frames <= 0 or len(swing_clearance_targets) < 2:
        return swing_clearance_targets.astype(np.float32)
    binary = contacts > 0.5
    frames, sides = binary.shape
    distance = np.full((frames, sides), np.inf, dtype=np.float32)
    for side in range(sides):
        next_onset = np.inf
        for t in range(frames - 1, -1, -1):
            if binary[t, side] and (t == 0 or not binary[t - 1, side]):
                next_onset = float(t)
            if np.isfinite(next_onset):
                distance[t, side] = next_onset - t
    ramp = np.clip(distance / float(taper_frames), 0.0, 1.0)
    # Smoothstep, not the raw linear ramp. A linear taper is continuous in
    # position but not in velocity: at the frame the taper switches on, the
    # clearance request goes from constant to falling at full rate, which the
    # solver reproduces as a one-frame reversal of the foot's vertical
    # velocity mid-swing. s^2(3-2s) has zero slope at both ends, so the
    # request eases in at taper onset and eases out at touchdown.
    scale = ramp * ramp * (3.0 - 2.0 * ramp)
    return (swing_clearance_targets * scale).astype(np.float32)


def suppress_anchor_rampin(
    soft_contacts: np.ndarray, hard_contacts: np.ndarray
) -> np.ndarray:
    """Zero the pre-touchdown half of a softened contact label.

    soften_contact_labels widens every stance interval symmetrically, which is
    what the height and tilt costs want. The stance anchor is different: its
    target is a fixed world x/y median of the stance, so giving it any weight
    while the foot is still airborne drags the swing foot horizontally toward
    a spot it has not reached yet. The result is a visible one-frame reversal
    of the foot's travel direction a few frames before touchdown -- the same
    energy as the original touchdown snap, just moved earlier into swing.

    Trailing-edge softening has no such problem: at toe-off the foot is
    already at the anchor, so fading the anchor out is what lets it leave
    smoothly. So keep the ramp-out and drop the ramp-in.

    Args:
        soft_contacts: [T, 2, P] cross-faded labels.
        hard_contacts: [T, 2, P] original binary labels.

    Returns:
        [T, 2, P] labels equal to soft_contacts except on swing frames that
        are closer to the next touchdown than to the previous lift-off, which
        are zeroed.
    """
    binary = hard_contacts > 0.5
    frames = len(binary)
    if frames == 0:
        return soft_contacts.astype(np.float32)
    index = np.arange(frames, dtype=np.float32)
    result = soft_contacts.astype(np.float32).copy()
    flat_binary = binary.reshape(frames, -1)
    flat_result = result.reshape(frames, -1)
    for channel in range(flat_binary.shape[1]):
        active = flat_binary[:, channel]
        if not active.any():
            flat_result[:, channel] = 0.0
            continue
        stance = index[active]
        # Distance to the nearest stance frame ahead of / behind each frame.
        ahead = np.searchsorted(stance, index, side="left")
        behind = ahead - 1
        to_next = np.where(ahead < len(stance), stance[np.minimum(ahead, len(stance) - 1)] - index, np.inf)
        to_prev = np.where(behind >= 0, index - stance[np.maximum(behind, 0)], np.inf)
        ramp_in = ~active & (to_next <= to_prev)
        flat_result[ramp_in, channel] = 0.0
    return result


def build_stance_anchors(
    nominal_points: np.ndarray, point_contacts: np.ndarray
) -> np.ndarray:
    """Fix each heel/toe sole sample only during its own contact interval.

    The target stays the stance median for the whole interval. Easing it from
    the landing position instead was measured to be much worse: it asks the
    solver to walk the foot from where it landed to the median, which is
    sustained slip by construction (mean 0.018 -> 0.033 m/s, peak 0.20 ->
    0.50 m/s on walk_spin_backward_clockwise). The onset discontinuity is
    handled on the weight side instead, in suppress_anchor_rampin.
    """
    anchors = nominal_points.copy()
    for side in range(2):
        for point in range(nominal_points.shape[2]):
            active = point_contacts[:, side, point] > 0.1
            starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
            ends = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
            for start, end in zip(starts, ends):
                core = point_contacts[start:end, side, point] > 0.5
                samples = nominal_points[start:end, side, point][core]
                if len(samples) == 0:
                    samples = nominal_points[start:end, side, point]
                anchor_xy = np.median(samples[:, :2], axis=0)
                anchors[start:end, side, point, :2] = anchor_xy
    anchors[..., 2] = 0.0
    return anchors.astype(np.float32)


def pad_or_trim(array: np.ndarray, frames: int) -> tuple[np.ndarray, int]:
    actual = min(len(array), frames) if frames > 0 else len(array)
    if frames <= 0 or len(array) == frames:
        return array, actual
    if len(array) > frames:
        return array[:frames], actual
    return np.concatenate([array, np.repeat(array[-1:], frames - len(array), axis=0)]), actual


def contact_quality_metrics(
    robot: pk.Robot,
    roots: jaxlie.SE3,
    joints: np.ndarray,
    ankle_indices: np.ndarray,
    contacts: np.ndarray,
    ground_heights: np.ndarray,
    fps: float,
    sole_offsets: np.ndarray = R1_SOLE_POINTS,
    point_contacts: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure stance-foot behavior on the solved trajectory."""
    links = forward_kinematics_world(robot, roots.wxyz_xyz, jnp.asarray(joints))
    ankles = jaxlie.SE3(links.wxyz_xyz[:, ankle_indices])
    rotations = np.asarray(ankles.rotation().as_matrix())
    sole = np.asarray(
        jnp.einsum("tfij,pj->tfpi", ankles.rotation().as_matrix(), jnp.asarray(sole_offsets))
        + ankles.translation()[:, :, None, :]
    )
    speeds = np.zeros(sole.shape[:3], dtype=np.float32)
    if len(sole) > 1:
        speeds[1:] = np.linalg.norm(np.diff(sole, axis=0), axis=-1) * fps
    normals = rotations[..., :, 2]
    tilt_deg = np.degrees(np.arccos(np.clip(normals[..., 2], -1.0, 1.0)))
    height_error = np.abs(sole[..., 2] - ground_heights[:, :, None])
    penetration = np.maximum(
        ground_heights[:, :, None] - sole[..., 2], 0.0
    )
    stance = contacts > 0.5
    if point_contacts is None:
        point_stance = np.broadcast_to(stance[:, :, None], speeds.shape)
        tilt_stance = stance
    else:
        point_stance = np.asarray(point_contacts) > 0.5
        tilt_stance = np.all(point_stance, axis=2)
    def add_persistent_point_slip(metrics: dict[str, float]) -> dict[str, float]:
        if point_contacts is None or len(speeds) < 2:
            return metrics
        supported = np.asarray(point_contacts) > 0.5
        persistent = supported[1:] & supported[:-1]
        values = speeds[1:][persistent]
        metrics.update(
            {
                "persistent_contact_mean_speed_mps": (
                    float(np.mean(values)) if len(values) else 0.0
                ),
                "persistent_contact_max_speed_mps": (
                    float(np.max(values)) if len(values) else 0.0
                ),
            }
        )
        return metrics
    if not np.any(point_stance):
        return add_persistent_point_slip({
            "stance_samples": 0.0,
            "stance_mean_speed_mps": 0.0,
            "stance_max_speed_mps": 0.0,
            "stance_mean_tilt_deg": 0.0,
            "stance_max_tilt_deg": 0.0,
            "stance_mean_height_error_m": 0.0,
            "stance_max_height_error_m": 0.0,
            "max_ground_penetration_m": float(np.max(penetration)),
        })
    return add_persistent_point_slip({
        "stance_samples": float(np.count_nonzero(stance)),
        "stance_mean_speed_mps": float(np.mean(speeds[point_stance])),
        "stance_max_speed_mps": float(np.max(speeds[point_stance])),
        "stance_mean_tilt_deg": (
            float(np.mean(tilt_deg[tilt_stance])) if np.any(tilt_stance) else 0.0
        ),
        "stance_max_tilt_deg": (
            float(np.max(tilt_deg[tilt_stance])) if np.any(tilt_stance) else 0.0
        ),
        "stance_mean_height_error_m": float(np.mean(height_error[point_stance])),
        "stance_max_height_error_m": float(np.max(height_error[point_stance])),
        "max_ground_penetration_m": float(np.max(penetration)),
    })


def hand_thigh_quality_metrics(
    robot: pk.Robot,
    roots: jaxlie.SE3,
    joints: np.ndarray,
    semantic_link_indices: np.ndarray,
) -> dict[str, float]:
    """Measure the same hand/thigh proxy clearance used by the optimizer."""
    links = forward_kinematics_world(robot, roots.wxyz_xyz, jnp.asarray(joints))
    semantic = jaxlie.SE3(links.wxyz_xyz[:, semantic_link_indices])
    positions = np.asarray(semantic.translation())
    rotations = np.asarray(semantic.rotation().as_matrix())
    wrist_indices = np.asarray([10, 13])
    hand_points = (
        np.einsum(
            "thij,pj->thpi",
            rotations[:, wrist_indices],
            R1_HAND_CENTERLINE_POINTS,
        )
        + positions[:, wrist_indices, None, :]
    )
    hip = positions[:, [1, 4]]
    knee = positions[:, [2, 5]]
    segment = knee - hip
    point_delta = hand_points[:, :, :, None, :] - hip[:, None, None, :, :]
    segment_sq = np.sum(segment * segment, axis=-1)
    fraction = np.clip(
        np.sum(point_delta * segment[:, None, None, :, :], axis=-1)
        / np.maximum(segment_sq[:, None, None, :], 1.0e-8),
        0.0,
        1.0,
    )
    closest = (
        hip[:, None, None, :, :]
        + fraction[..., None] * segment[:, None, None, :, :]
    )
    distance = np.linalg.norm(
        hand_points[:, :, :, None, :] - closest, axis=-1
    )
    shortfall = np.maximum(R1_HAND_THIGH_CLEARANCE - distance, 0.0)
    return {
        "hand_thigh_min_clearance_m": float(np.min(distance)),
        "hand_thigh_max_shortfall_m": float(np.max(shortfall)),
        "hand_thigh_violation_fraction": float(np.mean(shortfall > 0.0)),
    }


def continuity_quality_metrics(
    roots: jaxlie.SE3, joints: np.ndarray, fps: float
) -> dict[str, float]:
    """Report frame discontinuities that look like IK branch switches."""
    if len(joints) < 2:
        return {
            "root_max_translation_step_m": 0.0,
            "root_max_rotation_step_deg": 0.0,
            "joint_max_step_rad": 0.0,
            "root_max_linear_speed_mps": 0.0,
            "root_max_angular_speed_dps": 0.0,
        }
    root_array = np.asarray(roots.wxyz_xyz)
    translation_step = np.linalg.norm(np.diff(root_array[:, 4:], axis=0), axis=-1)
    previous_rotation = jaxlie.SO3(jnp.asarray(root_array[:-1, :4]))
    next_rotation = jaxlie.SO3(jnp.asarray(root_array[1:, :4]))
    rotation_step = np.linalg.norm(
        np.asarray((previous_rotation.inverse() @ next_rotation).log()), axis=-1
    )
    joint_step = np.abs(np.diff(joints, axis=0))
    return {
        "root_max_translation_step_m": float(np.max(translation_step)),
        "root_max_rotation_step_deg": float(np.degrees(np.max(rotation_step))),
        "joint_max_step_rad": float(np.max(joint_step)),
        "root_max_linear_speed_mps": float(np.max(translation_step) * fps),
        "root_max_angular_speed_dps": float(np.degrees(np.max(rotation_step)) * fps),
    }


@jaxls.Cost.factory
def joint_velocity_limit_cost(
    var_values: jaxls.VarValues,
    q_curr: jaxls.Var[jnp.ndarray],
    q_prev: jaxls.Var[jnp.ndarray],
    limits: jnp.ndarray,
    dt: float,
    weight: float,
) -> jax.Array:
    excess = jnp.maximum(jnp.abs((var_values[q_curr] - var_values[q_prev]) / dt) - limits, 0.0)
    return excess.flatten() * weight


@jaxls.Cost.factory
def joint_acceleration_cost(
    var_values: jaxls.VarValues,
    q_next: jaxls.Var[jnp.ndarray],
    q_curr: jaxls.Var[jnp.ndarray],
    q_prev: jaxls.Var[jnp.ndarray],
    weight: float,
) -> jax.Array:
    return (var_values[q_next] - 2.0 * var_values[q_curr] + var_values[q_prev]).flatten() * weight


@jaxls.Cost.factory
def foot_contact_cost(
    var_values: jaxls.VarValues,
    root_curr: jaxls.SE3Var,
    root_prev: jaxls.SE3Var,
    q_curr: jaxls.Var[jnp.ndarray],
    q_prev: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    ankle_indices: jnp.ndarray,
    sole_offsets: jnp.ndarray,
    contacts: jnp.ndarray,
    point_contacts: jnp.ndarray,
    prev_point_contacts: jnp.ndarray,
    anchor_contacts: jnp.ndarray,
    ground_heights: jnp.ndarray,
    stance_anchors: jnp.ndarray,
    swing_clearance_targets: jnp.ndarray,
    slip_weight: float,
    anchor_weight: float,
    penetration_weight: float,
    height_weight: float,
    tilt_weight: float,
    swing_height_weight: float,
) -> jax.Array:
    def sole(root_var: jaxls.SE3Var, q_var: jaxls.Var[jnp.ndarray]):
        links = var_values[root_var] @ jaxlie.SE3(robot.forward_kinematics(var_values[q_var]))
        ankle = jaxlie.SE3(links.wxyz_xyz[ankle_indices])
        points = jnp.einsum("fij,pj->fpi", ankle.rotation().as_matrix(), sole_offsets)
        return points + ankle.translation()[:, None, :], ankle.rotation().as_matrix()

    curr_points, curr_rot = sole(root_curr, q_curr)
    prev_points, _ = sole(root_prev, q_prev)
    # Least-squares squares residuals. sqrt(contact) makes the objective's
    # strength follow the soft contact label linearly instead of quadratically.
    contact_scale = jnp.sqrt(jnp.clip(point_contacts, 0.0, 1.0))
    # A touchdown frame must not be compared with the preceding airborne
    # frame. Slip only exists when the same sole sample is supported on both
    # sides of the finite difference.
    slip_scale = jnp.sqrt(
        jnp.clip(point_contacts * prev_point_contacts, 0.0, 1.0)
    )
    slip = (
        (curr_points - prev_points) * slip_scale[:, :, None]
    ).flatten() * slip_weight
    # Deliberately not contact_scale: the anchor must not act before the foot
    # has actually landed. See suppress_anchor_rampin.
    anchor_scale = jnp.sqrt(jnp.clip(anchor_contacts, 0.0, 1.0))
    anchor = (
        (curr_points[..., :2] - stance_anchors[..., :2])
        * anchor_scale[:, :, None]
    ).flatten() * anchor_weight
    # Ground is unilateral even in swing: a foot may be above the plane but
    # never below it. This prevents the late-contact underground-then-snap-up
    # failure without pinning swing height to the ground.
    penetration = jnp.maximum(
        ground_heights[:, None] - curr_points[..., 2], 0.0
    ).flatten() * penetration_weight
    height = (
        (curr_points[..., 2] - ground_heights[:, None])
        * contact_scale
    ).flatten() * height_weight
    # Penalizing both horizontal components of the sole normal gives a useful
    # gradient even near level; (R[2,2]-1) alone becomes quartic near zero tilt.
    full_contact_scale = jnp.sqrt(jnp.min(jnp.clip(point_contacts, 0.0, 1.0), axis=1))
    tilt = (curr_rot[:, :2, 2] * full_contact_scale[:, None]).flatten() * tilt_weight
    # Preserve source step clearance explicitly. This is deliberately a
    # one-sided cost: being higher than the requested swing height is allowed,
    # while a compressed/dragging swing trajectory is penalized. Fade it out
    # as contact approaches so it cannot fight the stance-height objective.
    swing_scale = jnp.sqrt(jnp.clip(1.0 - contacts, 0.0, 1.0))
    center_height = curr_points[..., 2].mean(axis=1) - ground_heights
    swing_height = (
        jnp.maximum(swing_clearance_targets - center_height, 0.0)
        * swing_scale
        * swing_height_weight
    )
    return jnp.concatenate([slip, anchor, penetration, height, tilt, swing_height])


@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    target_positions: jnp.ndarray,
    target_rotations: jnp.ndarray,
    root_init: jaxlie.SE3,
    joint_init: jnp.ndarray,
    contacts: jnp.ndarray,
    point_contacts: jnp.ndarray,
    anchor_contacts: jnp.ndarray,
    ground_heights: jnp.ndarray,
    stance_anchors: jnp.ndarray,
    swing_clearance_targets: jnp.ndarray,
    r1_link_indices: jnp.ndarray,
    ankle_indices: jnp.ndarray,
    velocity_limits: jnp.ndarray,
    dt: float,
    weights: Weights,
) -> tuple[jaxlie.SE3, jnp.ndarray]:
    timesteps = target_positions.shape[0]
    q = robot.joint_var_cls(jnp.arange(timesteps))
    root = jaxls.SE3Var(jnp.arange(timesteps))

    landmark_weights = jnp.array(
        [3.0, 1.0, 2.0, 4.0, 1.0, 2.0, 4.0, 2.0,
         1.5, 2.0, 3.0, 1.5, 2.0, 3.0], dtype=jnp.float32
    )

    @jaxls.Cost.factory
    def landmark_cost(var_values, root_var, q_var, targets):
        links = var_values[root_var] @ jaxlie.SE3(robot.forward_kinematics(var_values[q_var]))
        positions = links.translation()[r1_link_indices]
        return ((positions - targets) * landmark_weights[:, None]).flatten() * weights["landmark"]

    @jaxls.Cost.factory
    def root_tracking_cost(var_values, root_var, target_pose):
        current = var_values[root_var]
        translation = (current.translation() - target_pose.translation()) * weights["root_position"]
        rotation_error = (target_pose.rotation().inverse() @ current.rotation()).log()
        # R1 has no waist-pitch joint, so pelvis pitch needs freedom to align
        # its torso. Keep roll independently stiff to prevent hip hiking and
        # side-to-side pelvis tilt from satisfying swing-foot height cheaply.
        rotation_weights = jnp.array(
            [
                weights["root_roll_orientation"],
                weights["root_orientation"],
                weights["root_yaw_orientation"],
            ]
        )
        rotation = rotation_error * rotation_weights
        return jnp.concatenate([translation, rotation])

    @jaxls.Cost.factory
    def link_orientation_cost(var_values, root_var, q_var, rotations, contact):
        links = var_values[root_var] @ jaxlie.SE3(robot.forward_kinematics(var_values[q_var]))
        current = links.rotation().as_matrix()[r1_link_indices]
        # Torso, ankles, elbows, and wrist-roll links. The wrist-roll link is
        # upstream of G1's unsupported wrist pitch/yaw joints, so its complete
        # orientation is shared by both robots and safe to track.
        torso_i, left_i, right_i = 7, 3, 6
        torso = (current[torso_i] - rotations[torso_i]).flatten() * weights["torso_orientation"]
        arm_indices = jnp.asarray([9, 10, 12, 13])
        arms = (
            current[arm_indices] - rotations[arm_indices]
        ).flatten() * weights["arm_orientation"]
        feet = []
        for semantic_i, side in ((left_i, 0), (right_i, 1)):
            w = weights["swing_foot_orientation"] + contact[side] * weights["foot_tilt"]
            feet.append((current[semantic_i] - rotations[semantic_i]).flatten() * w)
        return jnp.concatenate([torso, arms, *feet])

    @jaxls.Cost.factory
    def seed_cost(var_values, q_var, seed):
        return (var_values[q_var] - seed).flatten() * weights["seed"]

    @jaxls.Cost.factory
    def hand_thigh_clearance_cost(var_values, root_var, q_var):
        """Keep the R1 wrist/hand collision volumes outside both thighs."""
        links = var_values[root_var] @ jaxlie.SE3(
            robot.forward_kinematics(var_values[q_var])
        )
        semantic = jaxlie.SE3(links.wxyz_xyz[r1_link_indices])
        positions = semantic.translation()

        # Hand centerlines in world coordinates: [left/right, sample, xyz].
        wrist_indices = jnp.asarray([10, 13])
        wrist = jaxlie.SE3(semantic.wxyz_xyz[wrist_indices])
        hand_points = (
            jnp.einsum(
                "hij,pj->hpi",
                wrist.rotation().as_matrix(),
                jnp.asarray(R1_HAND_CENTERLINE_POINTS),
            )
            + wrist.translation()[:, None, :]
        )

        # The upper-leg collision mesh follows the hip-to-knee segment. Check
        # every hand sample against both thighs so cross-body arm motions are
        # also protected. Shape of distance is [hand, sample, thigh].
        hip = positions[jnp.asarray([1, 4])]
        knee = positions[jnp.asarray([2, 5])]
        segment = knee - hip
        point_delta = hand_points[:, :, None, :] - hip[None, None, :, :]
        segment_sq = jnp.sum(segment * segment, axis=-1)
        fraction = jnp.clip(
            jnp.sum(point_delta * segment[None, None, :, :], axis=-1)
            / jnp.maximum(segment_sq[None, None, :], 1.0e-8),
            0.0,
            1.0,
        )
        closest = hip[None, None, :, :] + fraction[..., None] * segment[None, None, :, :]

        distance = jnp.sqrt(
            jnp.sum((hand_points[:, :, None, :] - closest) ** 2, axis=-1)
            + 1.0e-8
        )
        penetration = jnp.maximum(R1_HAND_THIGH_CLEARANCE - distance, 0.0)
        return penetration.flatten() * weights["hand_thigh_clearance"]

    hip_roll_indices = jnp.asarray(
        [
            list(robot.joints.actuated_names).index("left_hip_roll_joint"),
            list(robot.joints.actuated_names).index("right_hip_roll_joint"),
        ]
    )
    arm_joint_indices = jnp.asarray(
        [
            i for i, name in enumerate(robot.joints.actuated_names)
            if any(part in name for part in ("shoulder", "elbow", "wrist"))
        ]
    )

    @jaxls.Cost.factory
    def arm_smoothness_cost(var_values, curr, prev):
        return (
            var_values[curr][arm_joint_indices]
            - var_values[prev][arm_joint_indices]
        ) * weights["arm_smoothness"]

    @jaxls.Cost.factory
    def arm_acceleration_cost(var_values, next_var, curr, prev):
        return (
            var_values[next_var][arm_joint_indices]
            - 2.0 * var_values[curr][arm_joint_indices]
            + var_values[prev][arm_joint_indices]
        ) * weights["arm_acceleration"]

    @jaxls.Cost.factory
    def hip_roll_tracking_cost(var_values, q_var, seed):
        """Prevent swing-clearance costs from being met by hip hiking."""
        return (
            var_values[q_var][hip_roll_indices] - seed[hip_roll_indices]
        ) * weights["hip_roll_tracking"]

    @jaxls.Cost.factory
    def root_smoothness_cost(var_values, curr, prev):
        return (var_values[prev].inverse() @ var_values[curr]).log().flatten() * weights["root_smoothness"]

    costs = [
        landmark_cost(root, q, target_positions),
        root_tracking_cost(root, root_init),
        link_orientation_cost(root, q, target_rotations, contacts),
        hand_thigh_clearance_cost(root, q),
        seed_cost(q, joint_init),
        hip_roll_tracking_cost(q, joint_init),
        pk.costs.limit_cost(jax.tree.map(lambda x: x[None], robot), q, 100.0),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(timesteps - 1)),
            weights["joint_smoothness"],
        ),
        root_smoothness_cost(
            jaxls.SE3Var(jnp.arange(1, timesteps)),
            jaxls.SE3Var(jnp.arange(timesteps - 1)),
        ),
        joint_velocity_limit_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(timesteps - 1)),
            velocity_limits[None], dt, weights["joint_velocity_limit"],
        ),
        arm_smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(timesteps - 1)),
        ),
    ]
    if timesteps > 2:
        costs.append(
            joint_acceleration_cost(
                robot.joint_var_cls(jnp.arange(2, timesteps)),
                robot.joint_var_cls(jnp.arange(1, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(timesteps - 2)),
                weights["joint_acceleration"],
            )
        )
        costs.append(
            arm_acceleration_cost(
                robot.joint_var_cls(jnp.arange(2, timesteps)),
                robot.joint_var_cls(jnp.arange(1, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(timesteps - 2)),
            )
        )
    for t in range(timesteps):
        prev_t = max(0, t - 1)
        costs.append(
            foot_contact_cost(
                jaxls.SE3Var(t), jaxls.SE3Var(prev_t),
                robot.joint_var_cls(t), robot.joint_var_cls(prev_t), robot,
                ankle_indices, jnp.asarray(R1_SOLE_POINTS), contacts[t],
                point_contacts[t], point_contacts[prev_t],
                anchor_contacts[t],
                ground_heights[t], stance_anchors[t],
                swing_clearance_targets[t],
                weights["foot_slip"], weights["foot_anchor"],
                weights["foot_penetration"], weights["foot_height"],
                weights["foot_tilt"], weights["swing_foot_height"],
            )
        )

    solution = (
        jaxls.LeastSquaresProblem(costs, [q, root])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make(
                [q.with_value(joint_init), root.with_value(root_init)]
            ),
            termination=jaxls.TerminationConfig(max_iterations=500),
            # jaxls verbose logging inserts jax.debug.callback host callbacks.
            # JAX will not persistently cache an executable containing them.
            verbose=False,
        )
    )
    return solution[root], solution[q]


def process_motion(
    path: Path,
    output_path: Path,
    g1_robot: pk.Robot,
    r1_robot: pk.Robot,
    g1_names: list[str],
    r1_names: list[str],
    r1_lower: np.ndarray,
    r1_upper: np.ndarray,
    r1_velocity: np.ndarray,
    g1_link_indices: np.ndarray,
    r1_link_indices: np.ndarray,
    morphology: Morphology,
    rotation_alignment: np.ndarray,
    weights: Weights,
    args: argparse.Namespace,
) -> None:
    data = None
    embedded_contacts = None
    if path.suffix.lower() == ".csv":
        csv = np.loadtxt(path, delimiter=",", dtype=np.float32)
        csv = np.atleast_2d(csv)
        expected_columns = 7 + len(g1_names)
        if csv.shape[1] != expected_columns:
            raise ValueError(
                f"{path}: {csv.shape[1]} CSV columns, expected {expected_columns} "
                "(xyz, quaternion wxyz, then G1 joints)"
            )
        pos, quat, g1_q = csv[:, :3], csv[:, 3:7], csv[:, 7:]
        source_fps = float(args.input_fps)
    elif path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        required = ("base_frame_pos", "base_frame_wxyz", "joint_angles")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path}: missing NPZ fields {missing}")
        source_fps = float(data["fps"]) if "fps" in data else args.input_fps
        pos = np.asarray(data["base_frame_pos"], dtype=np.float32)
        quat = np.asarray(data["base_frame_wxyz"], dtype=np.float32)
        g1_q = np.asarray(data["joint_angles"], dtype=np.float32)
        embedded_contacts = (
            data["foot_point_contacts"]
            if "foot_point_contacts" in data
            else data["foot_contacts"] if "foot_contacts" in data else None
        )
    else:
        raise ValueError(f"Unsupported input extension: {path.suffix}")

    stride = args.subsample_factor
    pos, quat, g1_q = pos[::stride], quat[::stride], g1_q[::stride]
    if g1_q.shape[1] != len(g1_names):
        raise ValueError(
            f"{path}: {g1_q.shape[1]} joint columns, expected {len(g1_names)} in G1 URDF order"
        )
    output_fps = source_fps / stride

    root_array = np.concatenate([quat, pos], axis=-1)
    world_links = forward_kinematics_world(
        g1_robot, jnp.asarray(root_array), jnp.asarray(g1_q)
    )
    semantic_T = jaxlie.SE3(world_links.wxyz_xyz[:, g1_link_indices])
    source_positions = np.asarray(semantic_T.translation())
    source_rotations = np.asarray(semantic_T.rotation().as_matrix())
    # Link frame conventions are not guaranteed to be identical between the
    # URDFs. Preserve each G1 link's motion relative to its neutral frame, then
    # express the target using the corresponding neutral R1 link frame.
    target_rotations = np.einsum(
        "tlij,ljk->tlik", source_rotations, rotation_alignment
    ).astype(np.float32)
    target_positions = scale_landmarks(source_positions, morphology)

    ankle_semantic = np.array(
        [SEMANTIC_INDEX["left_ankle"], SEMANTIC_INDEX["right_ankle"]]
    )
    source_ankles = jaxlie.SE3(semantic_T.wxyz_xyz[:, ankle_semantic])
    source_sole = np.asarray(
        jnp.einsum(
            "tfij,pj->tfpi", source_ankles.rotation().as_matrix(), jnp.asarray(G1_SOLE_POINTS)
        ) + source_ankles.translation()[:, :, None, :]
    )
    # Measure clearance relative to each source sole's own stance baseline.
    # This removes any constant G1 ground offset before morphology scaling.
    source_sole_centers = source_sole.mean(axis=2)
    source_ground = np.percentile(source_sole_centers[..., 2], 5.0, axis=0)
    swing_clearance_targets = np.maximum(
        source_sole_centers[..., 2] - source_ground[None], 0.0
    ).astype(np.float32)
    swing_clearance_targets *= morphology.locomotion * args.swing_height_scale
    if embedded_contacts is not None:
        raw_contacts = np.asarray(embedded_contacts)[::stride]
        if raw_contacts.ndim == 3 and raw_contacts.shape[1:] == (2, 4):
            point_contacts = raw_contacts
        elif raw_contacts.ndim == 2 and raw_contacts.shape[1] == 4:
            # Kimodo ordering: left heel, left toe, right heel, right toe.
            point_contacts = np.empty(
                (len(raw_contacts), 2, len(R1_SOLE_POINTS)), dtype=np.float32
            )
            point_contacts[:, 0, :2] = raw_contacts[:, 0, None]
            point_contacts[:, 0, 2:] = raw_contacts[:, 1, None]
            point_contacts[:, 1, :2] = raw_contacts[:, 2, None]
            point_contacts[:, 1, 2:] = raw_contacts[:, 3, None]
        elif raw_contacts.shape[1] == 2:
            point_contacts = np.repeat(raw_contacts[:, :, None], 4, axis=2)
        else:
            raise ValueError(
                f"{path}: contacts must be [T,2], [T,4], or [T,2,4], "
                f"got {raw_contacts.shape}"
            )
        point_contacts = point_contacts.astype(np.float32)
        # Embedded labels are not derived from the sole geometry and lead it.
        # Correct the timing before softening: a cross-fade around the wrong
        # frame is still centred on the wrong frame.
        if args.align_contact_onsets:
            point_contacts = align_contact_onsets(
                point_contacts, source_sole, args.contact_height_margin
            )
        # Embedded labels are binary and would otherwise bypass the soft-label
        # cross-fade that infer_contacts() applies on the other branch.
        hard_point_contacts = point_contacts.copy()
        point_contacts = soften_contact_labels(
            point_contacts, args.contact_ramp_frames
        )
    else:
        contacts = infer_contacts(
            source_sole, output_fps, args.contact_height_margin,
            args.contact_speed_threshold,
        )
        point_contacts = np.repeat(contacts[:, :, None], 4, axis=2)
        hard_point_contacts = (point_contacts > 0.5).astype(np.float32)
    contacts = np.max(point_contacts, axis=2).astype(np.float32)
    anchor_contacts = suppress_anchor_rampin(point_contacts, hard_point_contacts)
    # Locate touchdown from the thresholded label, then stop asking the solver
    # to hold clearance the foot is about to give up anyway.
    swing_clearance_targets = taper_swing_clearance(
        swing_clearance_targets, contacts, args.swing_clearance_taper_frames
    )

    # Kimodo is a flat-ground dataset. Do not derive a different ground height
    # for every frame from the source foot pose: doing so turns source tilt and
    # tracking noise into a moving R1 floor. Build morphology-scaled nominal
    # sole samples, then hold each labeled heel/toe sample fixed in x/y while
    # constraining contacting samples to the single world plane z=0.
    target_ankles = target_positions[:, ankle_semantic]
    target_ankle_rot = target_rotations[:, ankle_semantic]
    nominal_sole_points = (
        np.einsum("tfij,pj->tfpi", target_ankle_rot, R1_SOLE_POINTS)
        + target_ankles[:, :, None, :]
    )
    ground_heights = np.zeros((len(g1_q), 2), dtype=np.float32)
    stance_anchors = build_stance_anchors(nominal_sole_points, anchor_contacts)

    joint_init = np.zeros((len(g1_q), len(r1_names)), dtype=np.float32)
    for r1_i, name in enumerate(r1_names):
        if name in g1_names:
            joint_init[:, r1_i] = g1_q[:, g1_names.index(name)]
    joint_init = np.clip(joint_init, r1_lower[None], r1_upper[None])

    # Root orientation is shared; root position is initialized from the scaled pelvis target.
    root_init_array = np.concatenate([quat, target_positions[:, 0]], axis=-1)
    arrays = [
        target_positions, target_rotations, root_init_array, joint_init,
        contacts, point_contacts, anchor_contacts, ground_heights,
        stance_anchors, swing_clearance_targets,
    ]
    padded, actual = [], None
    for array in arrays:
        result, count = pad_or_trim(array, args.target_frames)
        padded.append(result)
        actual = count if actual is None else min(actual, count)
    (
        target_positions, target_rotations, root_init_array, joint_init,
        contacts, point_contacts, anchor_contacts, ground_heights,
        stance_anchors, swing_clearance_targets,
    ) = padded

    ankle_indices = np.array(
        [list(r1_robot.links.names).index("left_ankle_roll_link"),
         list(r1_robot.links.names).index("right_ankle_roll_link")], dtype=np.int32
    )
    roots, joints = solve_retargeting(
        r1_robot, jnp.asarray(target_positions), jnp.asarray(target_rotations),
        jaxlie.SE3(jnp.asarray(root_init_array)), jnp.asarray(joint_init),
        jnp.asarray(contacts), jnp.asarray(point_contacts),
        jnp.asarray(anchor_contacts),
        jnp.asarray(ground_heights),
        jnp.asarray(stance_anchors),
        jnp.asarray(swing_clearance_targets),
        jnp.asarray(r1_link_indices), jnp.asarray(ankle_indices),
        jnp.asarray(r1_velocity, dtype=jnp.float32), 1.0 / output_fps,
        weights,
    )
    roots_trimmed = jaxlie.SE3(roots.wxyz_xyz[:actual])
    joints_trimmed = np.asarray(joints[:actual])
    metrics = contact_quality_metrics(
        r1_robot, roots_trimmed, joints_trimmed, ankle_indices,
        np.asarray(contacts[:actual]), np.asarray(ground_heights[:actual]), output_fps,
        point_contacts=np.asarray(point_contacts[:actual]),
    )
    metrics.update(
        hand_thigh_quality_metrics(
            r1_robot, roots_trimmed, joints_trimmed, r1_link_indices
        )
    )
    metrics.update(
        continuity_quality_metrics(roots_trimmed, joints_trimmed, output_fps)
    )
    print(
        "  contact QA: "
        f"persistent mean/max slip="
        f"{metrics['persistent_contact_mean_speed_mps']:.4f}/"
        f"{metrics['persistent_contact_max_speed_mps']:.4f} m/s, "
        f"mean/max tilt={metrics['stance_mean_tilt_deg']:.2f}/"
        f"{metrics['stance_max_tilt_deg']:.2f} deg, "
        f"mean/max |ground error|={metrics['stance_mean_height_error_m']:.4f}/"
        f"{metrics['stance_max_height_error_m']:.4f} m, "
        f"max penetration={metrics['max_ground_penetration_m']:.4f} m\n"
        "  arm QA: "
        f"min hand-thigh clearance={metrics['hand_thigh_min_clearance_m']:.4f} m, "
        f"max proxy overlap={metrics['hand_thigh_max_shortfall_m']:.4f} m, "
        f"violation fraction={metrics['hand_thigh_violation_fraction']:.2%}\n"
        "  continuity QA: "
        f"max root step={metrics['root_max_translation_step_m']:.4f} m/"
        f"{metrics['root_max_rotation_step_deg']:.2f} deg, "
        f"max joint step={metrics['joint_max_step_rad']:.4f} rad"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        base_frame_pos=np.asarray(roots_trimmed.wxyz_xyz[:, 4:]),
        base_frame_wxyz=np.asarray(roots_trimmed.wxyz_xyz[:, :4]),
        joint_angles=joints_trimmed,
        joint_names=np.asarray(r1_names),
        foot_contacts=np.asarray(contacts[:actual]),
        foot_point_contacts=np.asarray(point_contacts[:actual]),
        swing_foot_clearance_target=np.asarray(swing_clearance_targets[:actual]),
        fps=np.float32(output_fps),
        source_fps=np.float32(source_fps),
        subsample_factor=np.int32(stride),
        **{f"qa_{key}": np.float32(value) for key, value in metrics.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--input-dir", help="Folder of Kimodo G1 CSV and/or PyRoki G1 NPZ files"
    )
    inputs.add_argument(
        "--input-file", type=Path, help="Retarget only one G1 CSV or NPZ"
    )
    parser.add_argument("--output-dir", required=True, help="Folder for R1 NPZ files")
    parser.add_argument("--g1-urdf", type=Path, default=DEFAULT_G1_URDF)
    parser.add_argument("--r1-urdf", type=Path, default=DEFAULT_R1_URDF)
    parser.add_argument("--input-fps", type=float, default=30.0, help="Fallback when NPZ has no fps")
    parser.add_argument("--subsample-factor", type=int, default=1)
    parser.add_argument(
        "--target-frames", type=int, default=0,
        help="Fixed compiled trajectory length (pads/trims); default 0 preserves full motions",
    )
    parser.add_argument("--contact-height-margin", type=float, default=0.025)
    parser.add_argument("--contact-speed-threshold", type=float, default=0.15)
    parser.add_argument(
        "--swing-height-scale", type=float, default=3.0,
        help=(
            "Multiplier on morphology-scaled G1 swing-foot clearance; values "
            "above 1 encourage higher steps (for example 1.2)"
        ),
    )
    parser.add_argument(
        "--align-contact-onsets", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Delay each embedded touchdown label to the first frame whose "
            "source sole is within --contact-height-margin of the ground. "
            "Embedded labels lead the geometry by up to 8 cm, which the "
            "solver resolves as a snap."
        ),
    )
    parser.add_argument(
        "--contact-ramp-frames", type=int, default=3,
        help=(
            "Box-filter width used to cross-fade embedded (binary) contact "
            "labels at stance boundaries, matching what infer_contacts() "
            "already does for inferred labels. 1 disables the cross-fade and "
            "restores the previous hard on/off switching. Default 3 measured "
            "on walk_spin_backward_clockwise: it removes as much touchdown "
            "snap as a wider 5-frame fade (1.71 vs 1.80 cm worst touchdown "
            "step, down from 8.11 cm) while loosening the stance anchor less "
            "(max slip 0.237 vs 0.254 m/s, max ground error 2.2 vs 2.5 cm)."
        ),
    )
    parser.add_argument(
        "--swing-clearance-taper-frames", type=int, default=5,
        help=(
            "Frames over which the requested swing clearance ramps back to "
            "its full value ahead of touchdown, so the solver is not asked to "
            "hold the foot high on the frame the stance anchor plants it. "
            "0 disables the taper."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--weight", action="append", default=[], metavar="NAME=VALUE",
        help="Override a retarget cost weight; may be repeated",
    )
    args = parser.parse_args()
    if args.subsample_factor < 1:
        parser.error("--subsample-factor must be >= 1")
    if args.target_frames != 0 and args.target_frames < 3:
        parser.error("--target-frames must be 0 or >= 3")
    if args.swing_height_scale < 0:
        parser.error("--swing-height-scale must be non-negative")
    if args.contact_ramp_frames < 1:
        parser.error("--contact-ramp-frames must be at least 1")
    if args.swing_clearance_taper_frames < 0:
        parser.error("--swing-clearance-taper-frames must be non-negative")

    # Kinematic retargeting does not need visual meshes. Avoid loading them so
    # headless batch jobs do not depend on URDF mesh-path conventions.
    g1_load_kwargs = {"load_meshes": False}
    r1_load_kwargs = {"load_meshes": False}
    g1_urdf = yourdfpy.URDF.load(str(args.g1_urdf), **g1_load_kwargs)
    r1_urdf = yourdfpy.URDF.load(str(args.r1_urdf), **r1_load_kwargs)
    g1_robot, r1_robot = pk.Robot.from_urdf(g1_urdf), pk.Robot.from_urdf(r1_urdf)
    g1_names, _, _, _ = load_joint_metadata(args.g1_urdf)
    r1_names, r1_lower, r1_upper, r1_velocity = load_joint_metadata(args.r1_urdf)
    if g1_names != list(g1_robot.joints.actuated_names):
        raise ValueError("G1 PyRoki joint order differs from actuated URDF order")
    if r1_names != list(r1_robot.joints.actuated_names):
        raise ValueError("R1 PyRoki joint order differs from actuated URDF order")

    g1_links, r1_links = list(g1_robot.links.names), list(r1_robot.links.names)
    g1_link_indices = np.asarray([g1_links.index(x[1]) for x in SEMANTIC_LINKS])
    r1_link_indices = np.asarray([r1_links.index(x[2]) for x in SEMANTIC_LINKS])
    g1_zero = jaxlie.SE3(g1_robot.forward_kinematics(jnp.zeros(len(g1_names))))
    r1_zero = jaxlie.SE3(r1_robot.forward_kinematics(jnp.zeros(len(r1_names))))
    morphology = derive_morphology(
        np.asarray(g1_zero.translation())[g1_link_indices],
        np.asarray(r1_zero.translation())[r1_link_indices],
    )
    g1_neutral_rotations = np.asarray(g1_zero.rotation().as_matrix())[g1_link_indices]
    r1_neutral_rotations = np.asarray(r1_zero.rotation().as_matrix())[r1_link_indices]
    rotation_alignment = np.einsum(
        "lij,ljk->lik",
        np.swapaxes(g1_neutral_rotations, -1, -2),
        r1_neutral_rotations,
    ).astype(np.float32)
    print("URDF-derived R1/G1 morphology ratios:")
    for name, value in morphology._asdict().items():
        print(f"  {name:12s}: {value:.4f}")

    if args.input_file is not None:
        paths = [args.input_file]
    else:
        paths = [
            Path(p) for extension in ("*.csv", "*.npz")
            for p in glob.glob(os.path.join(args.input_dir, extension))
        ]
        paths.sort()
    if not paths:
        raise FileNotFoundError(f"No CSV or NPZ files found in {args.input_dir}")
    weights = dict(DEFAULT_WEIGHTS)
    for override in args.weight:
        try:
            name, raw_value = override.split("=", 1)
            value = float(raw_value)
        except ValueError:
            parser.error(f"Invalid --weight {override!r}; expected NAME=VALUE")
        if name not in weights:
            parser.error(
                f"Unknown weight {name!r}; choices: {', '.join(weights)}"
            )
        if value < 0:
            parser.error(f"Weight {name!r} must be non-negative")
        weights[name] = value
    print("Retarget weights:")
    for name, value in weights.items():
        print(f"  {name:24s}: {value:g}")
    for i, path in enumerate(paths, 1):
        output_path = Path(args.output_dir) / f"{path.stem}_r1.npz"
        if (
            args.skip_existing
            and output_path.exists()
            and output_path.stat().st_mtime_ns >= path.stat().st_mtime_ns
        ):
            print(f"[{i}/{len(paths)}] skipping {output_path.name}")
            continue
        print(f"[{i}/{len(paths)}] retargeting {path.name}")
        process_motion(
            path, output_path, g1_robot, r1_robot, g1_names, r1_names,
            r1_lower, r1_upper, r1_velocity, g1_link_indices,
            r1_link_indices, morphology, rotation_alignment, weights, args,
        )
        print(f"  saved {output_path}")


if __name__ == "__main__":
    main()
