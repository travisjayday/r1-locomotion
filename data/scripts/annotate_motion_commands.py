#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add smoothed simulation-root and ``[vx, vy, yaw_rate]`` labels to MotionLib data."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protomotions.robot_configs.factory import robot_config
from protomotions.utils.motion_command_annotations import (
    ReferenceCommandTrajectory,
    SimulationRootAnnotationConfig,
    SimulationRootAnnotations,
    annotate_packed_motion_library,
)


CSV_FIELDS = (
    "frame",
    "time_s",
    "simulation_root_x",
    "simulation_root_y",
    "simulation_root_yaw",
    "world_vx",
    "world_vy",
    "local_vx",
    "local_vy",
    "yaw_rate",
    "achieved_local_vx",
    "achieved_local_vy",
    "achieved_yaw_rate",
    "source_vx",
    "source_vy",
    "source_yaw_rate",
)


def annotation_csv_name(motion_id: int, motion_files: Sequence[str]) -> str:
    stem = Path(motion_files[motion_id]).stem if motion_id < len(motion_files) else "motion"
    return f"{motion_id:04d}__{stem}.csv"


def load_manual_roots(
    csv_dir: Path,
    motion_files: Sequence[str],
    motion_num_frames: Sequence,
) -> dict[int, dict[str, list[float]]]:
    """Load any edited CSVs present; missing clips retain procedural labels."""
    roots: dict[int, dict[str, list[float]]] = {}
    for motion_id, count in enumerate(motion_num_frames):
        path = csv_dir / annotation_csv_name(motion_id, motion_files)
        if not path.is_file():
            continue
        columns = {
            "simulation_root_x": [],
            "simulation_root_y": [],
            "simulation_root_yaw": [],
        }
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [key for key in columns if key not in (reader.fieldnames or ())]
            if missing:
                raise ValueError(f"{path} is missing editable columns: {missing}")
            for row in reader:
                for key in columns:
                    columns[key].append(float(row[key]))
        expected = int(count)
        if len(columns["simulation_root_x"]) != expected:
            raise ValueError(
                f"{path} has {len(columns['simulation_root_x'])} rows; "
                f"expected {expected}"
            )
        roots[motion_id] = columns
    return roots


def source_command_csv_path(source_root: Path, motion_file: str) -> Path:
    """Map ``category__clip_r1.motion`` back to the Kimodo command CSV."""
    stem = Path(motion_file).stem
    if stem.endswith("_r1"):
        stem = stem[:-3]
    if "__" not in stem:
        raise ValueError(f"cannot map motion filename without category separator: {motion_file}")
    category, clip = stem.split("__", 1)
    return source_root / category / f"{clip}.motion" / "commands.csv"


def load_reference_commands(
    source_root: Path,
    motion_files: Sequence[str],
) -> dict[int, ReferenceCommandTrajectory]:
    """Load the original Kimodo command trajectory corresponding to every clip."""
    trajectories = {}
    missing = []
    for motion_id, motion_file in enumerate(motion_files):
        path = source_command_csv_path(source_root, motion_file)
        if not path.is_file():
            missing.append(path)
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = ("time", "vx", "vy", "wz", "yaw")
        if not rows or any(key not in rows[0] for key in required):
            raise ValueError(f"{path} must contain columns {required}")
        trajectory = ReferenceCommandTrajectory(
            time_s=np.asarray([float(row["time"]) for row in rows]),
            commands=np.asarray(
                [
                    [float(row["vx"]), float(row["vy"]), float(row["wz"])]
                    for row in rows
                ]
            ),
            yaw=np.asarray([float(row["yaw"]) for row in rows]),
            source_path=str(path.resolve()),
        )
        trajectory.validate()
        trajectories[motion_id] = trajectory
    if missing:
        examples = "\n".join(f"  {path}" for path in missing[:10])
        raise FileNotFoundError(
            f"missing source command CSVs for {len(missing)} motions:\n{examples}"
        )
    return trajectories


def _yaw_from_xyzw(rotation: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(rotation, -1, 0)
    return np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def export_csv_annotations(
    csv_dir: Path,
    packed_motion: Mapping[str, object],
    annotations: SimulationRootAnnotations,
) -> None:
    """Export an editable CSV per clip; root x/y/yaw are authoritative on re-import."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    motion_files = tuple(packed_motion.get("motion_files", ()))
    starts = packed_motion["length_starts"]
    counts = packed_motion["motion_num_frames"]
    dts = packed_motion["motion_dt"]
    pos = annotations.simulation_root_pos.detach().cpu().numpy()
    rot = annotations.simulation_root_rot.detach().cpu().numpy()
    vel = annotations.simulation_root_vel.detach().cpu().numpy()
    commands = annotations.motion_commands.detach().cpu().numpy()
    achieved = annotations.achieved_motion_commands.detach().cpu().numpy()
    source = (
        annotations.source_motion_commands.detach().cpu().numpy()
        if annotations.source_motion_commands is not None
        else None
    )
    for motion_id, (start, count, dt) in enumerate(zip(starts, counts, dts)):
        start_i, count_i, dt_f = int(start), int(count), float(dt)
        sl = slice(start_i, start_i + count_i)
        yaw = _yaw_from_xyzw(rot[sl])
        path = csv_dir / annotation_csv_name(motion_id, motion_files)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for frame_id in range(count_i):
                global_id = start_i + frame_id
                writer.writerow(
                    {
                        "frame": frame_id,
                        "time_s": frame_id * dt_f,
                        "simulation_root_x": pos[global_id, 0],
                        "simulation_root_y": pos[global_id, 1],
                        "simulation_root_yaw": yaw[frame_id],
                        "world_vx": vel[global_id, 0],
                        "world_vy": vel[global_id, 1],
                        "local_vx": commands[global_id, 0],
                        "local_vy": commands[global_id, 1],
                        "yaw_rate": commands[global_id, 2],
                        "achieved_local_vx": achieved[global_id, 0],
                        "achieved_local_vy": achieved[global_id, 1],
                        "achieved_yaw_rate": achieved[global_id, 2],
                        "source_vx": source[global_id, 0] if source is not None else "",
                        "source_vy": source[global_id, 1] if source is not None else "",
                        "source_yaw_rate": (
                            source[global_id, 2] if source is not None else ""
                        ),
                    }
                )


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.in_place:
        return args.input
    if args.output is not None:
        return args.output
    return args.input.with_name(f"{args.input.stem}_commands{args.input.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Packed MotionLib .pt file")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", type=Path, help="Annotated output .pt")
    output_group.add_argument("--in-place", action="store_true", help="Replace the input file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output",
    )
    parser.add_argument("--robot", default="r1", help="Robot configuration name")
    parser.add_argument("--position-body", default="waist_yaw_link")
    parser.add_argument("--facing-body", default="pelvis_link")
    parser.add_argument("--window-sec", type=float, default=0.75)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument(
        "--source-command-root",
        type=Path,
        help="Kimodo g1_locomotion root containing <category>/<clip>.motion/commands.csv",
    )
    parser.add_argument(
        "--source-match-max-offset-sec",
        type=float,
        default=0.4,
        help="Maximum source-time shift used to align changing command profiles",
    )
    parser.add_argument(
        "--boundary-mode",
        choices=("linear_extrapolate", "interp", "mirror", "nearest", "constant", "wrap"),
        default="linear_extrapolate",
        help=(
            "Savitzky--Golay endpoint behavior; linear_extrapolate preserves a fitted "
            "nonzero endpoint trend"
        ),
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        help="Also export one editable CSV per motion",
    )
    parser.add_argument(
        "--manual-csv-dir",
        type=Path,
        help="Override generated roots with edited CSVs found in this directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = resolve_output_path(args)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if output.exists() and output != args.input and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
    if output == args.input and not args.overwrite:
        raise FileExistsError("in-place annotation requires --overwrite")

    packed_motion = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(packed_motion, dict):
        raise TypeError(f"{args.input} does not contain a packed MotionLib dictionary")
    cfg = robot_config(args.robot)
    body_names = cfg.kinematic_info.body_names
    for body_name in (args.position_body, args.facing_body):
        if body_name not in body_names:
            raise ValueError(f"body {body_name!r} is not present in {args.robot}: {body_names}")
    annotation_config = SimulationRootAnnotationConfig(
        position_body_index=body_names.index(args.position_body),
        facing_body_index=body_names.index(args.facing_body),
        position_body_name=args.position_body,
        facing_body_name=args.facing_body,
        semantic_forward_axis_xy=tuple(cfg.semantic_forward_axis_xy),
        smoothing_window_s=args.window_sec,
        polynomial_order=args.polyorder,
        boundary_mode=args.boundary_mode,
        source_match_max_offset_s=args.source_match_max_offset_sec,
    )

    motion_files = tuple(packed_motion.get("motion_files", ()))
    reference_commands = None
    if args.source_command_root is not None:
        reference_commands = load_reference_commands(
            args.source_command_root,
            motion_files,
        )
        print(f"Loaded source command trajectories for {len(reference_commands)} motions")
    manual_roots = None
    if args.manual_csv_dir is not None:
        manual_roots = load_manual_roots(
            args.manual_csv_dir,
            motion_files,
            packed_motion["motion_num_frames"],
        )
        print(f"Loaded manual simulation roots for {len(manual_roots)} motions")
    annotations = annotate_packed_motion_library(
        packed_motion,
        annotation_config,
        manual_roots=manual_roots,
        reference_commands=reference_commands,
    )
    packed_motion.update(annotations.as_dict())
    packed_motion["motion_command_metadata"].update(
        {
            "source_motion_file": str(args.input.resolve()),
            "manual_csv_dir": (
                str(args.manual_csv_dir.resolve()) if args.manual_csv_dir is not None else None
            ),
            "source_command_root": (
                str(args.source_command_root.resolve())
                if args.source_command_root is not None
                else None
            ),
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    torch.save(packed_motion, temporary)
    os.replace(temporary, output)
    if args.csv_dir is not None:
        export_csv_annotations(args.csv_dir, packed_motion, annotations)

    commands = annotations.motion_commands
    quantiles = torch.quantile(commands.abs(), torch.tensor([0.5, 0.95, 0.99]), dim=0)
    print(f"Annotated {len(packed_motion['motion_num_frames'])} motions / {len(commands)} frames")
    print(f"Saved {output}")
    print("Absolute command quantiles (50%, 95%, 99%):")
    for index, label in enumerate(("vx", "vy", "yaw_rate")):
        values = ", ".join(f"{value:.3f}" for value in quantiles[:, index].tolist())
        print(f"  {label}: {values}")
    if args.csv_dir is not None:
        print(f"Editable per-motion CSVs: {args.csv_dir}")


if __name__ == "__main__":
    main()
