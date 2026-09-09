# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remove stance-foot sliding from G1 CSV or G1/R1 NPZ trajectories with PyRoki.

For G1 CSV input, contacts are detected from the original sole motion. For NPZ
input, existing source-derived contact labels are preserved. Each contiguous
stance interval receives a fixed four-point touchdown anchor. A trajectory-level
IK solve adjusts the floating root and robot joints while tracking the original
pose, respecting limits, and matching those anchors. Corrected output uses the
standard robot NPZ format and includes the contact labels used by the solve.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import TypedDict

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy

from batch_retarget_g1_npz_to_r1 import (
    DEFAULT_G1_URDF,
    DEFAULT_R1_URDF,
    G1_SOLE_POINTS,
    R1_SOLE_POINTS,
    SEMANTIC_LINKS,
    contact_quality_metrics,
    forward_kinematics_world,
    load_joint_metadata,
    pad_or_trim,
)


class Weights(TypedDict):
    pose_landmarks: float
    root_position: float
    root_orientation: float
    joint_seed: float
    joint_smoothness: float
    joint_acceleration: float
    root_acceleration: float
    foot_anchor: float
    foot_orientation: float


WEIGHTS = Weights(
    pose_landmarks=5.0,
    root_position=2.0,
    root_orientation=2.0,
    joint_seed=0.6,
    joint_smoothness=2.0,
    joint_acceleration=4.0,
    root_acceleration=3.0,
    foot_anchor=120.0,
    foot_orientation=20.0,
)


def _remove_short_runs(mask: np.ndarray, minimum_frames: int) -> np.ndarray:
    result = mask.copy()
    start = 0
    while start < len(result):
        value = result[start]
        end = start + 1
        while end < len(result) and result[end] == value:
            end += 1
        if value and end - start < minimum_frames:
            result[start:end] = False
        start = end
    return result


def _fill_short_gaps(mask: np.ndarray, maximum_gap: int) -> np.ndarray:
    result = mask.copy()
    start = 0
    while start < len(result):
        value = result[start]
        end = start + 1
        while end < len(result) and result[end] == value:
            end += 1
        if not value and start > 0 and end < len(result) and end - start <= maximum_gap:
            result[start:end] = True
        start = end
    return result


def detect_contacts(
    sole: np.ndarray,
    fps: float,
    height_margin: float,
    horizontal_speed_threshold: float,
    vertical_speed_threshold: float,
    minimum_stance_frames: int,
) -> np.ndarray:
    """Detect stance robustly enough to tolerate moderate source foot sliding."""
    center = sole.mean(axis=2)
    velocity = np.zeros_like(center)
    if len(center) > 1:
        velocity[1:] = np.diff(center, axis=0) * fps
        velocity[0] = velocity[1]
    lowest = sole[..., 2].min(axis=2)
    ground = np.percentile(sole[..., 2], 5.0, axis=(0, 2))
    low = lowest <= ground[None] + height_margin
    slow_xy = np.linalg.norm(velocity[..., :2], axis=-1) <= horizontal_speed_threshold
    slow_z = np.abs(velocity[..., 2]) <= vertical_speed_threshold
    binary = low & slow_xy & slow_z
    for side in range(2):
        binary[:, side] = _fill_short_gaps(binary[:, side], maximum_gap=2)
        binary[:, side] = _remove_short_runs(binary[:, side], minimum_stance_frames)

    # Ramp *inside* each detected stance.  Frames below 0.5 pull the foot
    # toward its anchor before/after firm contact, but are not reported as
    # contact by QA or downstream conversion.  Extending the ramp outside the
    # stance makes an ordinary swing-to-touchdown displacement look like slip.
    contacts = binary.astype(np.float32)
    for side in range(2):
        start = 0
        while start < len(binary):
            if not binary[start, side]:
                start += 1
                continue
            end = start + 1
            while end < len(binary) and binary[end, side]:
                end += 1
            length = end - start
            ramp = np.ones(length, dtype=np.float32)
            if length == 1:
                ramp[0] = 0.25
            elif length == 2:
                ramp[:] = 0.25
            elif length == 3:
                ramp[:] = (0.25, 1.0, 0.25)
            else:
                ramp[:2] = (0.25, 0.5)
                ramp[-2:] = (0.5, 0.25)
            contacts[start:end, side] = ramp
            start = end
    return contacts


def build_anchor_targets(
    sole: np.ndarray,
    rotations: np.ndarray,
    contacts: np.ndarray,
    sole_offsets: np.ndarray = G1_SOLE_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Freeze sole position and yaw separately for every stance interval."""
    target_points = sole.copy()
    target_rotations = rotations.copy()
    # Include the low-weight ramp frames in the same constant anchor segment.
    active = contacts > 0.0
    for side in range(2):
        start = 0
        while start < len(active):
            if not active[start, side]:
                start += 1
                continue
            end = start + 1
            while end < len(active) and active[end, side]:
                end += 1
            # Robust average touchdown heading; pitch/roll are intentionally removed.
            forward = rotations[start:end, side, :2, 0].mean(axis=0)
            yaw = float(np.arctan2(forward[1], forward[0]))
            cy, sy = np.cos(yaw), np.sin(yaw)
            level_rotation = np.array(
                [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )
            center_xy = np.median(sole[start:end, side].mean(axis=1)[:, :2], axis=0)
            ground_z = float(np.percentile(sole[start:end, side, :, 2], 10.0))
            anchored = sole_offsets @ level_rotation.T
            anchored[:, :2] += center_xy
            anchored[:, 2] += ground_z - anchored[:, 2].min()
            target_points[start:end, side] = anchored[None]
            target_rotations[start:end, side] = level_rotation[None]
            start = end
    return target_points, target_rotations


@jaxls.Cost.factory
def acceleration_cost(var_values, q_next, q_curr, q_prev, weight: float):
    return (
        var_values[q_next] - 2.0 * var_values[q_curr] + var_values[q_prev]
    ).flatten() * weight


@jaxls.Cost.factory
def root_acceleration_cost(var_values, root_next, root_curr, root_prev, weight: float):
    previous_delta = (var_values[root_prev].inverse() @ var_values[root_curr]).log()
    next_delta = (var_values[root_curr].inverse() @ var_values[root_next]).log()
    return (next_delta - previous_delta).flatten() * weight


@jaxls.Cost.factory
def anchor_cost(
    var_values,
    root_var,
    q_var,
    robot: pk.Robot,
    ankle_indices: jnp.ndarray,
    targets: jnp.ndarray,
    target_rotations: jnp.ndarray,
    contacts: jnp.ndarray,
    sole_offsets: jnp.ndarray,
    point_weight: float,
    orientation_weight: float,
):
    links = var_values[root_var] @ jaxlie.SE3(robot.forward_kinematics(var_values[q_var]))
    ankles = jaxlie.SE3(links.wxyz_xyz[ankle_indices])
    rotation = ankles.rotation().as_matrix()
    points = (
        jnp.einsum("fij,pj->fpi", rotation, sole_offsets)
        + ankles.translation()[:, None, :]
    )
    point_residual = (
        (points - targets) * contacts[:, None, None]
    ).flatten() * point_weight
    rotation_residual = (
        (rotation - target_rotations) * contacts[:, None, None]
    ).flatten() * orientation_weight
    return jnp.concatenate([point_residual, rotation_residual])


@jdc.jit
def solve(
    robot: pk.Robot,
    root_init: jaxlie.SE3,
    q_init: jnp.ndarray,
    landmark_targets: jnp.ndarray,
    landmark_indices: jnp.ndarray,
    ankle_indices: jnp.ndarray,
    anchor_targets: jnp.ndarray,
    anchor_rotations: jnp.ndarray,
    contacts: jnp.ndarray,
    sole_offsets: jnp.ndarray,
    weights: Weights,
) -> tuple[jaxlie.SE3, jnp.ndarray]:
    timesteps = q_init.shape[0]
    q = robot.joint_var_cls(jnp.arange(timesteps))
    root = jaxls.SE3Var(jnp.arange(timesteps))

    @jaxls.Cost.factory
    def landmarks(var_values, root_var, q_var, targets):
        links = var_values[root_var] @ jaxlie.SE3(robot.forward_kinematics(var_values[q_var]))
        # Track the original body everywhere, but downweight ankles because
        # their fixed stance anchors intentionally override the sliding source.
        residual = links.translation()[landmark_indices] - targets
        per_link = jnp.array(
            [2.0, 1.0, 1.5, 0.25, 1.0, 1.5, 0.25,
             1.5, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0]
        )
        return (residual * per_link[:, None]).flatten() * weights["pose_landmarks"]

    @jaxls.Cost.factory
    def root_tracking(var_values, root_var, target):
        current = var_values[root_var]
        pos = (current.translation() - target.translation()) * weights["root_position"]
        rot = (target.rotation().inverse() @ current.rotation()).log() * weights["root_orientation"]
        return jnp.concatenate([pos, rot])

    @jaxls.Cost.factory
    def seed(var_values, q_var, target):
        return (var_values[q_var] - target).flatten() * weights["joint_seed"]

    costs = [
        landmarks(root, q, landmark_targets),
        root_tracking(root, root_init),
        seed(q, q_init),
        pk.costs.limit_cost(jax.tree.map(lambda x: x[None], robot), q, 100.0),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(timesteps - 1)),
            weights["joint_smoothness"],
        ),
    ]
    if timesteps > 2:
        costs.extend([
            acceleration_cost(
                robot.joint_var_cls(jnp.arange(2, timesteps)),
                robot.joint_var_cls(jnp.arange(1, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(timesteps - 2)),
                weights["joint_acceleration"],
            ),
            root_acceleration_cost(
                jaxls.SE3Var(jnp.arange(2, timesteps)),
                jaxls.SE3Var(jnp.arange(1, timesteps - 1)),
                jaxls.SE3Var(jnp.arange(timesteps - 2)),
                weights["root_acceleration"],
            ),
        ])
    for t in range(timesteps):
        costs.append(
            anchor_cost(
                jaxls.SE3Var(t), robot.joint_var_cls(t), robot, ankle_indices,
                anchor_targets[t], anchor_rotations[t], contacts[t],
                sole_offsets,
                weights["foot_anchor"], weights["foot_orientation"],
            )
        )
    solution = (
        jaxls.LeastSquaresProblem(costs, [q, root]).analyze().solve(
            initial_vals=jaxls.VarValues.make(
                [q.with_value(q_init), root.with_value(root_init)]
            ),
            termination=jaxls.TerminationConfig(max_iterations=500),
            # jaxls verbose logging uses jax.debug.callback. Executables with
            # host callbacks are deliberately excluded from JAX's persistent
            # compilation cache, causing every batch invocation to recompile.
            verbose=False,
        )
    )
    return solution[root], solution[q]


def process(path: Path, output: Path, robot: pk.Robot, args: argparse.Namespace) -> None:
    urdf_path = args.g1_urdf if args.robot == "g1" else args.r1_urdf
    sole_offsets = G1_SOLE_POINTS if args.robot == "g1" else R1_SOLE_POINTS
    semantic_column = 1 if args.robot == "g1" else 2
    joint_names, _, _, _ = load_joint_metadata(urdf_path)
    embedded_contacts = None
    fps = args.input_fps
    if path.suffix.lower() == ".csv":
        csv = np.atleast_2d(np.loadtxt(path, delimiter=",", dtype=np.float32))
        if csv.shape[1] != 7 + len(joint_names):
            raise ValueError(f"{path}: expected {7 + len(joint_names)} columns, got {csv.shape[1]}")
        pos, quat, q = csv[:, :3], csv[:, 3:7], csv[:, 7:]
    else:
        data = np.load(path, allow_pickle=True)
        pos = np.asarray(data["base_frame_pos"], dtype=np.float32)
        quat = np.asarray(data["base_frame_wxyz"], dtype=np.float32)
        q = np.asarray(data["joint_angles"], dtype=np.float32)
        fps = float(data["fps"]) if "fps" in data else fps
        if "joint_names" in data and list(map(str, data["joint_names"])) != joint_names:
            raise ValueError(f"{path}: joint_names do not match {args.robot.upper()} URDF order")
        if "foot_contacts" in data:
            embedded_contacts = np.asarray(data["foot_contacts"], dtype=np.float32)
    root_array = np.concatenate([quat, pos], axis=-1)
    world = forward_kinematics_world(robot, jnp.asarray(root_array), jnp.asarray(q))
    link_names = list(robot.links.names)
    landmark_indices = np.array([link_names.index(item[semantic_column]) for item in SEMANTIC_LINKS])
    ankle_indices = np.array(
        [link_names.index("left_ankle_roll_link"), link_names.index("right_ankle_roll_link")]
    )
    landmarks = np.asarray(world.translation())[:, landmark_indices]
    ankles = jaxlie.SE3(world.wxyz_xyz[:, ankle_indices])
    rotations = np.asarray(ankles.rotation().as_matrix())
    sole = np.asarray(
        jnp.einsum("tfij,pj->tfpi", ankles.rotation().as_matrix(), jnp.asarray(sole_offsets))
        + ankles.translation()[:, :, None, :]
    )
    if embedded_contacts is not None and embedded_contacts.shape[1] == 4:
        # Only heel+toe overlap is safe to treat as rigid full-foot stance in
        # this optional preprocessor. Preserve the original point labels for
        # the retargeter so heel strike and toe-off remain articulated.
        contacts = np.stack(
            [embedded_contacts[:, :2].min(axis=1),
             embedded_contacts[:, 2:].min(axis=1)], axis=1
        )
        output_contacts = np.stack(
            [embedded_contacts[:, :2].max(axis=1),
             embedded_contacts[:, 2:].max(axis=1)], axis=1
        )
        point_contacts = np.empty((len(contacts), 2, 4), dtype=np.float32)
        point_contacts[:, 0, :2] = embedded_contacts[:, 0, None]
        point_contacts[:, 0, 2:] = embedded_contacts[:, 1, None]
        point_contacts[:, 1, :2] = embedded_contacts[:, 2, None]
        point_contacts[:, 1, 2:] = embedded_contacts[:, 3, None]
    else:
        contacts = embedded_contacts if embedded_contacts is not None else detect_contacts(
            sole, fps, args.height_margin, args.speed_threshold,
            args.vertical_speed_threshold, args.minimum_stance_frames,
        )
        if contacts.shape[1] != 2:
            raise ValueError(f"{path}: expected 2 or 4 contact columns, got {contacts.shape}")
        output_contacts = contacts
        point_contacts = np.repeat(contacts[:, :, None], 4, axis=2)
    anchor_targets, anchor_rotations = build_anchor_targets(
        sole, rotations, contacts, sole_offsets
    )

    arrays = [
        root_array, q, landmarks, anchor_targets, anchor_rotations, contacts,
        output_contacts, point_contacts,
    ]
    padded, actual = [], len(q)
    for array in arrays:
        value, count = pad_or_trim(array, args.target_frames)
        padded.append(value)
        actual = min(actual, count)
    (
        root_array, q, landmarks, anchor_targets, anchor_rotations, contacts,
        output_contacts, point_contacts,
    ) = padded
    roots, joints = solve(
        robot, jaxlie.SE3(jnp.asarray(root_array)), jnp.asarray(q),
        jnp.asarray(landmarks), jnp.asarray(landmark_indices),
        jnp.asarray(ankle_indices), jnp.asarray(anchor_targets),
        jnp.asarray(anchor_rotations), jnp.asarray(contacts),
        jnp.asarray(sole_offsets), WEIGHTS,
    )
    roots = jaxlie.SE3(roots.wxyz_xyz[:actual])
    joints = np.asarray(joints[:actual])
    # Anchor height varies by stance segment, so QA ground is taken directly
    # from each active target sole plane.
    ground = np.min(anchor_targets[:actual, ..., 2], axis=2)
    metrics = contact_quality_metrics(
        robot, roots, joints, ankle_indices, contacts[:actual], ground,
        fps, sole_offsets,
    )
    print(
        f"  corrected QA: mean/max slip={metrics['stance_mean_speed_mps']:.4f}/"
        f"{metrics['stance_max_speed_mps']:.4f} m/s, mean/max tilt="
        f"{metrics['stance_mean_tilt_deg']:.2f}/{metrics['stance_max_tilt_deg']:.2f} deg, "
        f"mean/max ground error={metrics['stance_mean_height_error_m']:.4f}/"
        f"{metrics['stance_max_height_error_m']:.4f} m"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        base_frame_pos=np.asarray(roots.wxyz_xyz[:, 4:]),
        base_frame_wxyz=np.asarray(roots.wxyz_xyz[:, :4]),
        joint_angles=joints,
        joint_names=np.asarray(joint_names),
        foot_contacts=np.asarray(output_contacts[:actual]),
        foot_point_contacts=np.asarray(point_contacts[:actual]),
        fps=np.float32(fps),
        **{f"qa_{key}": np.float32(value) for key, value in metrics.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot", choices=("g1", "r1"), default="g1")
    parser.add_argument("--g1-urdf", type=Path, default=DEFAULT_G1_URDF)
    parser.add_argument("--r1-urdf", type=Path, default=DEFAULT_R1_URDF)
    parser.add_argument("--input-fps", type=float, default=30.0)
    parser.add_argument("--target-frames", type=int, default=0)
    parser.add_argument("--height-margin", type=float, default=0.035)
    parser.add_argument("--speed-threshold", type=float, default=0.25)
    parser.add_argument("--vertical-speed-threshold", type=float, default=0.25)
    parser.add_argument("--minimum-stance-frames", type=int, default=3)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.target_frames not in (0,) and args.target_frames < 3:
        parser.error("--target-frames must be 0 or >= 3")
    urdf_path = args.g1_urdf if args.robot == "g1" else args.r1_urdf
    urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False)
    robot = pk.Robot.from_urdf(urdf)
    # Canonical G1 NPZ inputs are used when converting recursively generated
    # Kimodo datasets; process() already supports the same pose schema as CSV.
    extensions = ("*.csv", "*.npz") if args.robot == "g1" else ("*.npz",)
    paths = [Path(p) for ext in extensions for p in sorted(glob.glob(os.path.join(args.input_dir, ext)))]
    if not paths:
        raise FileNotFoundError(f"No {args.robot.upper()} inputs found in {args.input_dir}")
    for index, path in enumerate(paths, 1):
        output = args.output_dir / f"{path.stem}.npz"
        if args.skip_existing and output.exists():
            print(f"[{index}/{len(paths)}] skipping {output.name}")
            continue
        print(f"[{index}/{len(paths)}] correcting {path.name}")
        process(path, output, robot, args)
        print(f"  saved {output}")


if __name__ == "__main__":
    main()
