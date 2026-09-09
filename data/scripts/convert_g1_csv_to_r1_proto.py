# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Kimodo G1 CSV/NPZ -> R1 ProtoMotions dataset conversion.

The PyRoki stage is launched with a separate interpreter because ProtoMotions
and PyRoki commonly use incompatible JAX/CUDA dependency sets. This command
produces both individual ``.motion`` files and a packaged MotionLib ``.pt``.

Native Kimodo dataset clips are supported recursively. Each ``motion.npz`` is
paired with its sibling ``g1_qpos.csv`` because Kimodo's NPZ stores skeleton
rotations/positions rather than the 29 actuated G1 joint coordinates required
for robot-to-robot retargeting. Kimodo's heel/toe contacts are preserved and
mapped to individual R1 sole samples during retargeting.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from protomotions.components.pose_lib import (
    compute_cartesian_velocity,
    extract_kinematic_info,
    extract_qpos_from_transforms,
    extract_transforms_from_qpos,
    fk_from_transforms_with_velocities,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RETARGET_SCRIPT = REPO_ROOT / "pyroki/batch_retarget_g1_npz_to_r1.py"
FOOT_FIX_SCRIPT = REPO_ROOT / "pyroki/preprocess_g1_csv_fix_foot_sliding.py"
DEFAULT_R1_MJCF = REPO_ROOT / "protomotions/data/assets/mjcf/r1.xml"
DEFAULT_JAX_CACHE_DIR = Path.home() / ".cache" / "jax-pyroki-r1"


def pyroki_subprocess_env(jax_cache_dir: Path) -> dict[str, str]:
    """Return an environment enabling JAX's persistent compilation cache."""
    environment = os.environ.copy()
    # Each distinct clip-length shape triggers its own jaxls compilation; XLA's
    # CUDA-graph "command buffer" optimization keeps every one of those graphs
    # alive for the life of the process instead of releasing them, so a batch
    # spanning enough distinct frame counts (a few hundred clips reliably does)
    # exhausts GPU memory with RESOURCE_EXHAUSTED / "N alive graphs" well before
    # the dataset finishes. Disabling command buffers trades a little per-call
    # dispatch overhead for bounded memory across an arbitrarily long batch.
    existing_xla_flags = environment.get("XLA_FLAGS", "")
    environment["XLA_FLAGS"] = (
        existing_xla_flags + " --xla_gpu_enable_command_buffer="
    ).strip()
    environment.update(
        {
            "JAX_COMPILATION_CACHE_DIR": str(jax_cache_dir),
            "JAX_ENABLE_COMPILATION_CACHE": "true",
            # Retarget compilations are expensive enough to retain all entries,
            # including any short warm-up variants produced by small clips.
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
            # Fail visibly instead of silently recompiling if cache I/O or
            # executable deserialization fails.
            "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true",
            # Source locations and transient metadata must not perturb keys
            # for otherwise identical solver executables across invocations.
            "JAX_COMPILATION_CACHE_INCLUDE_METADATA_IN_KEY": "false",
        }
    )
    return environment


def _kimodo_clip_name(input_dir: Path, motion_npz: Path) -> str:
    """Create a stable, collision-resistant name from a nested clip path."""
    relative_parent = motion_npz.parent.relative_to(input_dir)
    parts = list(relative_parent.parts)
    if not parts:
        parts = [motion_npz.parent.name]
    if parts[-1].endswith(".motion"):
        parts[-1] = parts[-1][:-len(".motion")]
    return "__".join(parts)


def _kimodo_fps(motion_npz: Path, fallback: float) -> float:
    metadata_path = motion_npz.parent / "metadata.json"
    if not metadata_path.exists():
        return fallback
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    return float(metadata.get("fps", fallback))


def _collapse_kimodo_contacts(contacts: np.ndarray, frames: int) -> np.ndarray:
    """Collapse [L heel, L toe, R heel, R toe] into left/right labels."""
    contacts = np.asarray(contacts)
    if contacts.shape == (frames, 2):
        return contacts.astype(np.float32)
    if contacts.shape != (frames, 4):
        raise ValueError(
            f"Kimodo foot_contacts must have shape [T, 4] or [T, 2], got "
            f"{contacts.shape}"
        )
    return np.stack(
        (contacts[:, :2].any(axis=1), contacts[:, 2:4].any(axis=1)), axis=-1
    ).astype(np.float32)


def prepare_kimodo_npz_inputs(
    input_dir: Path,
    staging_dir: Path,
    fallback_fps: float,
    force_remake: bool,
) -> Path | None:
    """Convert a nested Kimodo dataset into canonical G1 NPZ retarget inputs."""
    native_npzs = []
    for path in sorted(input_dir.rglob("motion.npz")):
        with np.load(path, allow_pickle=True) as data:
            if "base_frame_pos" not in data.files:
                native_npzs.append(path)
    if not native_npzs:
        return None

    print(
        f"Preparing {len(native_npzs)} native Kimodo clips from {input_dir} "
        f"using sibling g1_qpos.csv files"
    )
    staging_dir.mkdir(parents=True, exist_ok=True)
    used_names: dict[str, Path] = {}
    for index, motion_npz in enumerate(native_npzs, 1):
        clip_name = _kimodo_clip_name(input_dir, motion_npz)
        if clip_name in used_names:
            raise ValueError(
                f"Kimodo clip-name collision for {motion_npz} and "
                f"{used_names[clip_name]}: {clip_name}"
            )
        used_names[clip_name] = motion_npz
        qpos_path = motion_npz.parent / "g1_qpos.csv"
        if not qpos_path.exists():
            raise FileNotFoundError(
                f"{motion_npz}: sibling g1_qpos.csv is required because native "
                "Kimodo motion.npz has no actuated G1 joint angles"
            )
        output_path = staging_dir / f"{clip_name}.npz"
        newest_input_mtime = max(
            motion_npz.stat().st_mtime_ns,
            qpos_path.stat().st_mtime_ns,
            (motion_npz.parent / "metadata.json").stat().st_mtime_ns
            if (motion_npz.parent / "metadata.json").exists()
            else 0,
        )
        schema_is_current = False
        if output_path.exists():
            try:
                with np.load(output_path, allow_pickle=True) as staged:
                    schema_is_current = (
                        "foot_contacts" in staged
                        and staged["foot_contacts"].ndim == 2
                        and staged["foot_contacts"].shape[1] == 4
                    )
            except (OSError, ValueError):
                schema_is_current = False
        if (
            output_path.exists()
            and not force_remake
            and schema_is_current
            and output_path.stat().st_mtime_ns >= newest_input_mtime
        ):
            continue

        qpos = np.atleast_2d(
            np.loadtxt(qpos_path, delimiter=",", dtype=np.float32)
        )
        if qpos.shape[1] != 36:
            raise ValueError(
                f"{qpos_path}: expected 36 columns (xyz, wxyz, 29 G1 joints), "
                f"got {qpos.shape[1]}"
            )
        with np.load(motion_npz, allow_pickle=True) as kimodo:
            if "foot_contacts" not in kimodo:
                raise ValueError(f"{motion_npz}: missing foot_contacts")
            point_contacts = np.asarray(kimodo["foot_contacts"], dtype=np.float32)
            contacts = _collapse_kimodo_contacts(point_contacts, len(qpos))
        fps = _kimodo_fps(motion_npz, fallback_fps)
        np.savez_compressed(
            output_path,
            base_frame_pos=qpos[:, :3],
            base_frame_wxyz=qpos[:, 3:7],
            joint_angles=qpos[:, 7:],
            # Keep the original [L heel, L toe, R heel, R toe] labels. The
            # retargeter maps them to individual sole samples, allowing a foot
            # to pivot naturally during heel strike and toe-off.
            foot_contacts=point_contacts,
            kimodo_foot_contacts=contacts,
            fps=np.float32(fps),
            source_motion_npz=np.asarray(str(motion_npz)),
            source_qpos_csv=np.asarray(str(qpos_path)),
        )
        if index <= 5 or index == len(native_npzs) or index % 50 == 0:
            print(f"  [{index}/{len(native_npzs)}] prepared {output_path.name}")
    return staging_dir


def ankle_sole_depth_from_mjcf(mjcf_path: Path) -> float:
    """Return positive ankle-origin-to-lowest-collider depth for the R1 feet."""
    root = ET.parse(mjcf_path).getroot()
    depths = []
    for side in ("left", "right"):
        body = root.find(f".//body[@name='{side}_ankle_roll_link']")
        if body is None:
            raise ValueError(f"{mjcf_path}: missing {side}_ankle_roll_link body")
        lowest = None
        for geom in body.findall("geom"):
            # contype=0 is visual-only geometry and must not define ground height.
            if geom.get("contype") == "0":
                continue
            pos = [float(x) for x in geom.get("pos", "0 0 0").split()]
            geom_type = geom.get("type", "sphere")
            size = [float(x) for x in geom.get("size", "").split()]
            if geom_type == "sphere" and size:
                bottom = pos[2] - size[0]
            elif geom_type == "box" and len(size) >= 3:
                bottom = pos[2] - size[2]  # MJCF box sizes are half-extents.
            else:
                continue
            lowest = bottom if lowest is None else min(lowest, bottom)
        if lowest is None:
            raise ValueError(f"{mjcf_path}: no supported {side} foot collision geoms")
        depths.append(-lowest)
    if not np.isclose(depths[0], depths[1], atol=1e-5):
        raise ValueError(f"Asymmetric R1 sole depths in {mjcf_path}: {depths}")
    return float(np.mean(depths))


def run_g1_foot_preprocessing(
    args: argparse.Namespace, input_dir: Path, corrected_dir: Path
) -> None:
    command = [
        str(args.pyroki_python), str(FOOT_FIX_SCRIPT),
        "--input-dir", str(input_dir),
        "--output-dir", str(corrected_dir),
        "--input-fps", str(args.input_fps),
        "--target-frames", str(args.target_frames),
        "--height-margin", str(args.contact_height_margin),
        "--speed-threshold", str(args.g1_contact_speed_threshold),
    ]
    if not args.force_remake:
        command.append("--skip-existing")
    print("Running source G1 stance-foot correction:")
    print("  " + " ".join(command))
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        env=pyroki_subprocess_env(args.jax_cache_dir),
    )


def run_retargeting(
    args: argparse.Namespace, retarget_input_dir: Path, retargeted_dir: Path
) -> None:
    command = [
        str(args.pyroki_python),
        str(RETARGET_SCRIPT),
        "--input-dir", str(retarget_input_dir),
        "--output-dir", str(retargeted_dir),
        "--input-fps", str(args.input_fps),
        "--subsample-factor", str(args.retarget_subsample_factor),
        "--target-frames", str(args.target_frames),
        "--contact-height-margin", str(args.contact_height_margin),
        "--contact-speed-threshold", str(args.contact_speed_threshold),
    ]
    if not args.force_remake:
        command.append("--skip-existing")
    print("Running G1 -> R1 PyRoki retargeting:")
    print("  " + " ".join(command))
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        env=pyroki_subprocess_env(args.jax_cache_dir),
    )


def run_r1_foot_postprocessing(
    args: argparse.Namespace, retargeted_dir: Path, corrected_dir: Path
) -> None:
    command = [
        str(args.pyroki_python), str(FOOT_FIX_SCRIPT),
        "--robot", "r1",
        "--input-dir", str(retargeted_dir),
        "--output-dir", str(corrected_dir),
        "--input-fps", str(args.input_fps),
        "--target-frames", str(args.target_frames),
    ]
    if not args.force_remake:
        command.append("--skip-existing")
    print("Running final R1 stance-foot correction:")
    print("  " + " ".join(command))
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        env=pyroki_subprocess_env(args.jax_cache_dir),
    )


def resample_arrays(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joints: np.ndarray,
    contacts: np.ndarray,
    input_fps: float,
    output_fps: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if output_fps is None:
        return root_pos, root_quat, joints, contacts, input_fps
    ratio = input_fps / output_fps
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor):
        raise ValueError(
            f"Input FPS {input_fps:g} must be an integer multiple of output FPS {output_fps:g}"
        )
    return (
        root_pos[::factor], root_quat[::factor], joints[::factor],
        contacts[::factor], float(output_fps),
    )


def repair_dynamic_contacts(
    contacts: np.ndarray,
    ankle_heights: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Recover brief running contacts from low-rate vertical foot minima.

    Soft crossfades can peak at exactly 0.5 and disappear when converted to a
    boolean MotionLib. Dynamic steps can also be missed entirely when neither
    sampled frame has low horizontal velocity. Preserve all supplied labels,
    then add conservative two-frame contact cores at low, alternating foot
    minima that are not already close to a labeled event.
    """
    repaired = contacts.copy()
    frames = len(repaired)
    min_event_gap = max(3, int(round(0.13 * fps)))
    existing_centers: list[tuple[int, int]] = []
    for side in range(2):
        active = repaired[:, side] >= 0.5
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
        existing_centers.extend(
            ((int(start + end - 1) // 2, side) for start, end in zip(starts, ends))
        )

    baseline = np.percentile(ankle_heights, 5.0, axis=0)
    candidates: list[tuple[int, int]] = []
    for side in range(2):
        z = ankle_heights[:, side]
        minima = np.flatnonzero((z[1:-1] <= z[:-2]) & (z[1:-1] < z[2:])) + 1
        for frame in minima:
            # A contacting foot should be close to its learned stance height
            # and materially lower than the contralateral swing foot.
            if z[frame] > baseline[side] + 0.015:
                continue
            if z[frame] > ankle_heights[frame, 1 - side] - 0.010:
                continue
            lo, hi = max(0, frame - min_event_gap), min(frames, frame + min_event_gap + 1)
            if np.any(repaired[lo:hi, side] > 0.0):
                continue
            if any(abs(frame - center) < min_event_gap for center, _ in existing_centers):
                continue
            candidates.append((int(frame), side))

    added: list[tuple[int, int]] = []
    accepted = sorted(existing_centers)
    for frame, side in sorted(candidates):
        if any(abs(frame - center) < min_event_gap for center, _ in accepted):
            continue
        # Use the lower adjacent sample to make a two-frame (>=67 ms at
        # 30 FPS) boolean stance core, then retain a small soft crossfade.
        neighbor = frame - 1 if ankle_heights[frame - 1, side] <= ankle_heights[min(frame + 1, frames - 1), side] else min(frame + 1, frames - 1)
        core_start, core_end = sorted((frame, neighbor))
        repaired[core_start : core_end + 1, side] = 1.0
        if core_start > 0:
            repaired[core_start - 1, side] = max(repaired[core_start - 1, side], 0.25)
        if core_end + 1 < frames:
            repaired[core_end + 1, side] = max(repaired[core_end + 1, side], 0.25)
        accepted.append((frame, side))
        added.append((frame, side))
    return repaired, added


def convert_one(
    npz_path: Path,
    motion_path: Path,
    kinematic_info,
    left_foot_idx: int,
    right_foot_idx: int,
    output_fps: float | None,
    sole_depth: float,
    sole_clearance: float,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    required = ("base_frame_pos", "base_frame_wxyz", "joint_angles", "foot_contacts")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"{npz_path}: missing fields {missing}")

    root_pos = np.asarray(data["base_frame_pos"], dtype=np.float32)
    root_quat = np.asarray(data["base_frame_wxyz"], dtype=np.float32)
    joints = np.asarray(data["joint_angles"], dtype=np.float32)
    contacts = np.asarray(data["foot_contacts"], dtype=np.float32)
    fps = float(data["fps"]) if "fps" in data else 30.0
    root_pos, root_quat, joints, contacts, fps = resample_arrays(
        root_pos, root_quat, joints, contacts, fps, output_fps
    )

    if "joint_names" in data:
        input_joint_names = [str(name) for name in data["joint_names"]]
        if input_joint_names != list(kinematic_info.dof_names):
            raise ValueError(
                f"{npz_path}: R1 joint order does not match r1.xml\n"
                f"NPZ:  {input_joint_names}\nMJCF: {kinematic_info.dof_names}"
            )
    if joints.shape[1] != kinematic_info.num_dofs:
        raise ValueError(
            f"{npz_path}: {joints.shape[1]} joint columns, expected {kinematic_info.num_dofs}"
        )
    if contacts.shape != (len(joints), 2):
        raise ValueError(f"{npz_path}: expected foot_contacts [T, 2], got {contacts.shape}")
    quat_norm = np.linalg.norm(root_quat, axis=1)
    if np.any(~np.isfinite(quat_norm)) or np.max(np.abs(quat_norm - 1.0)) > 1e-3:
        raise ValueError(f"{npz_path}: invalid or non-unit root quaternion")

    device, dtype = torch.device("cpu"), torch.float32
    root_pos_t = torch.from_numpy(root_pos).to(device, dtype)
    root_quat_t = torch.from_numpy(root_quat).to(device, dtype)
    joints_t = torch.from_numpy(joints).to(device, dtype)
    qpos = torch.cat([root_pos_t, root_quat_t, joints_t], dim=-1)
    root_from_qpos, joint_rot_mats = extract_transforms_from_qpos(kinematic_info, qpos)
    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=root_from_qpos,
        joint_rot_mats=joint_rot_mats,
        fps=fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )

    canonical_qpos = extract_qpos_from_transforms(
        kinematic_info, root_pos_t, joint_rot_mats
    )
    motion.dof_pos = canonical_qpos[:, 7:]
    motion.dof_vel = compute_cartesian_velocity(
        batched_robot_pos=joints_t.unsqueeze(1), fps=fps
    ).squeeze(1)

    ankle_heights_np = (
        motion.rigid_body_pos[:, [left_foot_idx, right_foot_idx], 2].cpu().numpy()
    )
    contacts, added_contacts = repair_dynamic_contacts(
        contacts, ankle_heights_np, fps
    )
    if added_contacts:
        labels = ", ".join(
            f"{'L' if side == 0 else 'R'}@{frame}" for frame, side in added_contacts
        )
        print(f"  recovered dynamic contact events: {labels}")

    # Apply one constant vertical translation, never a per-frame correction:
    # per-frame height fixing can manufacture vertical foot slip. R1 contact
    # surface is ``sole_depth`` below the ankle-roll body origins. This depth
    # is parsed from the active MJCF collision geometry, including radii.
    ankle_heights = motion.rigid_body_pos[:, [left_foot_idx, right_foot_idx], 2]
    stance = torch.from_numpy(contacts >= 0.5)
    if stance.any():
        estimated_ground = (ankle_heights - sole_depth)[stance].median()
    else:
        estimated_ground = (ankle_heights - sole_depth).min()
    height_shift = torch.zeros(3, dtype=motion.rigid_body_pos.dtype)
    # Keep a small collision-safe gap. A zero/median alignment leaves roughly
    # half of noisy stance samples penetrating the PhysX ground, which can
    # generate large impulses during visualization or simulator resets.
    height_shift[2] = -estimated_ground + sole_clearance
    motion.translate(height_shift)

    rigid_contacts = torch.zeros(
        len(joints), kinematic_info.num_bodies, dtype=torch.bool, device=device
    )
    rigid_contacts[:, left_foot_idx] = torch.from_numpy(contacts[:, 0] >= 0.5)
    rigid_contacts[:, right_foot_idx] = torch.from_numpy(contacts[:, 1] >= 0.5)
    motion.rigid_body_contacts = rigid_contacts
    motion.local_rigid_body_rot = None  # Avoid MotionLib interpolating cached locals.

    tensors = (
        motion.dof_pos, motion.dof_vel, motion.rigid_body_pos,
        motion.rigid_body_rot, motion.rigid_body_vel,
    )
    if not all(torch.isfinite(value).all() for value in tensors if value is not None):
        raise ValueError(f"{npz_path}: non-finite values after ProtoMotions FK")

    motion_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(motion.to_dict(), motion_path)
    print(
        f"  saved {motion_path} ({len(joints)} frames, {fps:g} FPS, "
        f"{int(stance.sum())} stance labels)"
    )


def package_library(proto_dir: Path, library_file: Path) -> None:
    library_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO_ROOT / "protomotions/components/motion_lib.py"),
        "--motion-path", str(proto_dir),
        "--output-file", str(library_file),
        "--device", "cpu",
    ]
    print("Packaging R1 MotionLib:")
    print("  " + " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Flat Kimodo G1 CSV/canonical-NPZ directory, or a recursively nested "
            "Kimodo dataset containing *.motion/motion.npz and g1_qpos.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="R1 dataset output root")
    parser.add_argument(
        "--pyroki-python", type=Path, required=True,
        help="Python interpreter from the PyRoki environment",
    )
    parser.add_argument(
        "--jax-cache-dir",
        type=Path,
        default=DEFAULT_JAX_CACHE_DIR,
        help=(
            "Persistent JAX compilation-cache directory used by PyRoki "
            f"(default: {DEFAULT_JAX_CACHE_DIR})"
        ),
    )
    parser.add_argument("--input-fps", type=float, default=30.0)
    parser.add_argument("--output-fps", type=float, default=None)
    parser.add_argument("--retarget-subsample-factor", type=int, default=1)
    parser.add_argument("--target-frames", type=int, default=0)
    parser.add_argument("--contact-height-margin", type=float, default=0.025)
    parser.add_argument("--contact-speed-threshold", type=float, default=0.15)
    parser.add_argument("--g1-contact-speed-threshold", type=float, default=0.25)
    parser.add_argument(
        "--r1-sole-clearance", type=float, default=0.003,
        help="Constant R1 sole clearance above z=0 in meters (default: 0.003)",
    )
    parser.add_argument(
        "--fix-g1-feet", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Correct source G1 full-foot stance before retargeting. Native "
            "Kimodo G1 motion is preserved by default"
        ),
    )
    parser.add_argument(
        "--fix-r1-feet", action=argparse.BooleanOptionalAction, default=False,
        help="Apply fixed stance anchors to retargeted R1 motion (default: disabled)",
    )
    parser.add_argument("--r1-mjcf", type=Path, default=DEFAULT_R1_MJCF)
    parser.add_argument("--library-file", type=Path, default=None)
    parser.add_argument("--skip-retarget", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--force-remake", action="store_true")
    args = parser.parse_args()

    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.jax_cache_dir = args.jax_cache_dir.expanduser().resolve()
    args.jax_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Persistent PyRoki JAX compilation cache: {args.jax_cache_dir}")
    canonical_g1_dir = args.output_dir / "g1-kimodo-input"
    retargeted_dir = args.output_dir / "r1-retargeted"
    corrected_g1_dir = args.output_dir / "g1-foot-fixed"
    corrected_r1_dir = args.output_dir / "r1-foot-fixed"
    proto_dir = args.output_dir / "proto-r1"
    library_file = (
        args.library_file.resolve()
        if args.library_file is not None
        else args.output_dir / "kimodo_r1_motions.pt"
    )
    if not args.skip_retarget:
        prepared_dir = prepare_kimodo_npz_inputs(
            args.input_dir,
            canonical_g1_dir,
            args.input_fps,
            args.force_remake,
        )
        retarget_input_dir = prepared_dir or args.input_dir
        if args.fix_g1_feet:
            run_g1_foot_preprocessing(args, retarget_input_dir, corrected_g1_dir)
            retarget_input_dir = corrected_g1_dir
        run_retargeting(args, retarget_input_dir, retargeted_dir)

    final_r1_dir = retargeted_dir
    if args.fix_r1_feet:
        run_r1_foot_postprocessing(args, retargeted_dir, corrected_r1_dir)
        final_r1_dir = corrected_r1_dir

    npz_files = sorted(final_r1_dir.rglob("*_r1.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No R1 retargeted NPZ files found in {final_r1_dir}")
    kinematic_info = extract_kinematic_info(str(args.r1_mjcf))
    left_foot_idx = kinematic_info.body_names.index("left_ankle_roll_link")
    right_foot_idx = kinematic_info.body_names.index("right_ankle_roll_link")
    sole_depth = ankle_sole_depth_from_mjcf(args.r1_mjcf)
    print(f"R1 collision sole depth below ankle origin: {sole_depth:.4f} m")

    failures = []
    for index, npz_path in enumerate(npz_files, 1):
        relative_npz = npz_path.relative_to(final_r1_dir)
        motion_path = (proto_dir / relative_npz).with_suffix(".motion")
        # Reuse only an output that is at least as new as its actual NPZ input.
        # Retarget/postprocess stages may regenerate NPZs while an older motion
        # with the same name remains; blindly skipping it packages stale poses.
        motion_is_current = (
            motion_path.exists()
            and motion_path.stat().st_mtime_ns >= npz_path.stat().st_mtime_ns
        )
        if motion_is_current and not args.force_remake:
            print(f"[{index}/{len(npz_files)}] skipping {motion_path.name}")
            continue
        print(f"[{index}/{len(npz_files)}] converting {npz_path.name}")
        try:
            convert_one(
                npz_path, motion_path, kinematic_info,
                left_foot_idx, right_foot_idx, args.output_fps, sole_depth,
                args.r1_sole_clearance,
            )
        except Exception as exc:
            failures.append((npz_path, exc))
            print(f"  ERROR: {exc}", file=sys.stderr)

    if failures:
        details = "\n".join(f"  {path}: {exc}" for path, exc in failures)
        raise RuntimeError(f"Failed to convert {len(failures)} motion(s):\n{details}")
    if not any(proto_dir.rglob("*.motion")):
        raise RuntimeError(f"No .motion files produced in {proto_dir}")
    if not args.skip_package:
        package_library(proto_dir, library_file)
        print(f"Finished: motions={proto_dir}, library={library_file}")
    else:
        print(f"Finished: motions={proto_dir} (library packaging skipped)")


if __name__ == "__main__":
    with torch.no_grad():
        main()
