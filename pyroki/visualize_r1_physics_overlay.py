# SPDX-License-Identifier: Apache-2.0
"""Viser R1-mesh overlay of a kinematic reference and an RL physics rollout.

The viewer accepts either an explicit ``--reference``/``--physics`` pair or an
entire physics motion library via ``--dataset``. Dataset mode parses the
category/clip hierarchy out of the physics library's embedded
``motion_files`` metadata (``category__clip`` filenames, same convention the
retargeting pipeline uses) and resolves each entry's kinematic reference
relative to the dataset root, exposing category and motion selectors in the
Viser GUI.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import viser
from viser.extras import ViserUrdf
import yourdfpy


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# Common .motion files pickle StateConversion from the protomotions package.
# Direct script execution otherwise places only pyroki/ on sys.path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_R1_URDF = (
    SCRIPT_DIR / "../protomotions/data/assets/urdf/for_retargeting/r1.urdf"
).resolve()
DEFAULT_G1_URDF = (
    SCRIPT_DIR / "../protomotions/data/assets/urdf/for_retargeting/g1.urdf"
).resolve()
DEFAULT_G1_MESH_DIR = (
    SCRIPT_DIR / "../protomotions/data/assets/mesh/G1"
).resolve()
DEFAULT_R1_MESH_DIR = (
    SCRIPT_DIR / "../protomotions/data/assets/mesh/R1"
).resolve()
LEFT_FOOT_BODY = 6
RIGHT_FOOT_BODY = 12
FACING_ARROW_COLOR = (0, 180, 255)
VELOCITY_ARROW_COLOR = (0, 230, 80)
YAW_ARROW_COLOR = (255, 70, 210)


@dataclass(frozen=True)
class LibraryEntry:
    category: str
    clip: str
    motion_id: int
    reference_path: Path
    reference_motion_id: int


def natural_sort_key(value: str) -> tuple[object, ...]:
    """Sort embedded numbers numerically while keeping names deterministic."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def clip_label(entry: LibraryEntry) -> str:
    return f"({entry.motion_id}) {entry.clip}"


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def arrow_line_segments(
    origin: np.ndarray,
    vector: np.ndarray,
    *,
    head_length: float = 0.08,
    head_width: float = 0.05,
) -> np.ndarray:
    """Represent a planar arrow as a shaft and two arrowhead segments."""
    origin = np.asarray(origin, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-8:
        return np.repeat(origin[None, None, :], 6, axis=0).reshape(3, 2, 3)
    direction = vector / norm
    perpendicular = np.array([-direction[1], direction[0], 0.0])
    tip = origin + vector
    effective_head_length = min(head_length, 0.45 * norm)
    effective_head_width = min(head_width, 0.3 * norm)
    head_base = tip - effective_head_length * direction
    return np.asarray(
        [
            [origin, tip],
            [tip, head_base + effective_head_width * perpendicular],
            [tip, head_base - effective_head_width * perpendicular],
        ]
    )


def _facing_xy_from_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Rotate local +X by a quaternion and return its normalized XY projection."""
    x, y, z, w = rotation
    facing = np.asarray(
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z)]
    )
    norm = np.linalg.norm(facing)
    return facing / max(float(norm), 1.0e-8)


def _slice_r1_motion(
    data: dict, motion_id: int, path: Path
) -> dict[str, np.ndarray | float]:
    """Slice one motion out of an already-loaded MotionLib .pt or .motion dict."""
    if "length_starts" in data:
        count = len(data["length_starts"])
        if motion_id < 0 or motion_id >= count:
            raise ValueError(f"motion ID {motion_id} is outside [0, {count - 1}]")
        start = int(data["length_starts"][motion_id].item())
        frames = int(data["motion_num_frames"][motion_id].item())
        sl = slice(start, start + frames)
        pos, rot, joints = data["gts"][sl], data["grs"][sl], data["dps"][sl]
        contacts = data.get("contacts")
        contacts = contacts[sl] if contacts is not None else None
        simulation_root_pos = data.get("simulation_root_pos")
        simulation_root_rot = data.get("simulation_root_rot")
        simulation_root_vel = data.get("simulation_root_vel")
        motion_commands = data.get("motion_commands")
        achieved_motion_commands = data.get("achieved_motion_commands")
        simulation_root_pos = (
            simulation_root_pos[sl] if simulation_root_pos is not None else None
        )
        simulation_root_rot = (
            simulation_root_rot[sl] if simulation_root_rot is not None else None
        )
        simulation_root_vel = (
            simulation_root_vel[sl] if simulation_root_vel is not None else None
        )
        motion_commands = motion_commands[sl] if motion_commands is not None else None
        achieved_motion_commands = (
            achieved_motion_commands[sl]
            if achieved_motion_commands is not None
            else None
        )
        fps = 1.0 / float(data["motion_dt"][motion_id].item())
    elif "rigid_body_pos" in data:
        pos, rot, joints = (
            data["rigid_body_pos"],
            data["rigid_body_rot"],
            data["dof_pos"],
        )
        contacts = data.get("rigid_body_contacts")
        simulation_root_pos = data.get("simulation_root_pos")
        simulation_root_rot = data.get("simulation_root_rot")
        simulation_root_vel = data.get("simulation_root_vel")
        motion_commands = data.get("motion_commands")
        achieved_motion_commands = data.get("achieved_motion_commands")
        fps = float(data.get("fps", 30.0))
    else:
        raise ValueError(f"{path} is not a supported ProtoMotions motion file")

    if pos.shape[1] != 27 or joints.shape[1] != 26:
        raise ValueError(
            f"{path} is not R1: got {pos.shape[1]} bodies and {joints.shape[1]} DOFs"
        )
    return {
        "root_pos": pos[:, 0].numpy(),
        "root_wxyz": _xyzw_to_wxyz(rot[:, 0].numpy()),
        "joint_angles": joints.numpy(),
        "body_pos": pos.numpy(),
        "contacts": contacts.numpy() if contacts is not None else None,
        "simulation_root_pos": (
            simulation_root_pos.numpy() if simulation_root_pos is not None else None
        ),
        "simulation_root_xyzw": (
            simulation_root_rot.numpy() if simulation_root_rot is not None else None
        ),
        "simulation_root_vel": (
            simulation_root_vel.numpy() if simulation_root_vel is not None else None
        ),
        "motion_commands": motion_commands.numpy() if motion_commands is not None else None,
        "achieved_motion_commands": (
            achieved_motion_commands.numpy()
            if achieved_motion_commands is not None
            else None
        ),
        "fps": fps,
    }


def find_g1_source(reference_path: Path) -> Path | None:
    """Locate the G1 clip a given R1 reference was retargeted from.

    The pipeline writes ``<root>/proto-r1/<stem>_r1.motion`` from
    ``<root>/g1-kimodo-input/<stem>.npz``, so the source is recoverable from
    the reference path alone. Mirrored clips have no G1 counterpart -- the
    mirror is generated after retargeting -- so they return None.
    """
    stem = reference_path.stem
    if stem.endswith("_mirrored"):
        return None
    if not stem.endswith("_r1"):
        return None
    stem = stem[: -len("_r1")]
    for parent in (reference_path.parent.parent, reference_path.parent):
        for sub in ("g1-kimodo-input", "g1-foot-fixed"):
            candidate = parent / sub / f"{stem}.npz"
            if candidate.is_file():
                return candidate
    return None


def load_g1_motion(path: Path) -> dict:
    """Load a G1 source clip, reordered into the URDF's actuated-joint order."""
    with np.load(path, allow_pickle=True) as data:
        joint_angles = np.asarray(data["joint_angles"], dtype=np.float32)
        result = {
            "root_pos": np.asarray(data["base_frame_pos"], dtype=np.float32),
            "root_wxyz": np.asarray(data["base_frame_wxyz"], dtype=np.float32),
            "joint_angles": joint_angles,
            "joint_names": (
                [str(x) for x in data["joint_names"]]
                if "joint_names" in data
                else None
            ),
            "fps": float(data["fps"]) if "fps" in data else 30.0,
        }
    return result


def load_r1_motion(path: Path, motion_id: int = 0) -> dict[str, np.ndarray | float]:
    """Load either a common .motion or a packaged MotionLib .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return _slice_r1_motion(data, motion_id, path)


def _drop_zero_pad_frame(motion: dict[str, np.ndarray | float]) -> dict[str, np.ndarray | float]:
    """Drop a trailing all-zero frame some MotionLib saves leave uninitialized.

    Predicted motion libraries written by the mimic evaluator can be one
    frame longer than the data actually captured, leaving body_pos/joint
    angles at exactly zero for the last frame (a root-position teleport to
    the origin during playback otherwise).
    """
    body_pos = motion["body_pos"]
    if len(body_pos) > 1 and not np.any(body_pos[-1]):
        for key in (
            "root_pos",
            "root_wxyz",
            "joint_angles",
            "body_pos",
            "simulation_root_pos",
            "simulation_root_xyzw",
            "simulation_root_vel",
            "motion_commands",
            "achieved_motion_commands",
        ):
            if motion[key] is None:
                continue
            motion[key] = motion[key][:-1]
        if motion["contacts"] is not None:
            motion["contacts"] = motion["contacts"][:-1]
    return motion


def _align_physics_to_reference(
    physics: dict[str, np.ndarray | float], reference: dict[str, np.ndarray | float]
) -> dict[str, np.ndarray | float]:
    """Cancel the physics rollout's per-env world-grid offset for display.

    Predicted motion libraries saved from large multi-env training runs bake
    in each environment's world-space spawn offset (envs are tiled across a
    grid to avoid collisions during rollout), which has nothing to do with
    the reference motion's own coordinate frame. Shift the rollout's XY so
    its first frame lines up with the reference's first frame instead of
    trusting the saved absolute world coordinates.
    """
    xy_offset = physics["root_pos"][0, :2] - reference["root_pos"][0, :2]
    physics["root_pos"] = physics["root_pos"].copy()
    physics["body_pos"] = physics["body_pos"].copy()
    physics["root_pos"][:, :2] -= xy_offset
    physics["body_pos"][:, :, :2] -= xy_offset
    if physics["simulation_root_pos"] is not None:
        physics["simulation_root_pos"] = physics["simulation_root_pos"].copy()
        physics["simulation_root_pos"][:, :2] -= xy_offset
    return physics


def prepare_pair(
    reference: dict[str, np.ndarray | float], physics: dict[str, np.ndarray | float]
) -> tuple[dict[str, np.ndarray | float], dict[str, np.ndarray | float]]:
    reference = _drop_zero_pad_frame(reference)
    physics = _drop_zero_pad_frame(physics)
    physics = _align_physics_to_reference(physics, reference)
    return reference, physics


def discover_motion_directory(dataset_root: Path) -> list[LibraryEntry]:
    """Enumerate a directory of .motion files for reference-only browsing.

    discover_physics_library takes its clip list from a rollout library's
    motion_files, which only exists once something has been trained. A freshly
    retargeted dataset has no rollout yet, so scan the files directly instead;
    the ids here are just a stable display order, not motion-library indices.
    """
    paths = sorted(dataset_root.rglob("*.motion"))
    if not paths:
        raise FileNotFoundError(f"No .motion files found under {dataset_root}")
    entries: list[LibraryEntry] = []
    for motion_id, path in enumerate(
        sorted(paths, key=lambda item: natural_sort_key(item.stem))
    ):
        stem = path.stem
        category, clip = stem.split("__", 1) if "__" in stem else ("uncategorized", stem)
        entries.append(LibraryEntry(category, clip, motion_id, path.resolve(), 0))
    return sorted(
        entries, key=lambda e: (natural_sort_key(e.category), natural_sort_key(e.clip))
    )


def discover_physics_library(
    physics_raw: dict, physics_path: Path, dataset_root: Path
) -> list[LibraryEntry]:
    """Parse category/clip names from a physics MotionLib's motion_files metadata."""
    if "length_starts" not in physics_raw:
        raise ValueError(f"{physics_path} is not a MotionLib-format .pt file")
    motion_files = physics_raw.get("motion_files")
    if not motion_files:
        raise ValueError(
            f"{physics_path} has no embedded motion_files metadata; re-save it "
            "with a MotionLib build that records source motion files"
        )
    num_frames = physics_raw["motion_num_frames"]

    entries: list[LibraryEntry] = []
    missing: list[str] = []
    for motion_id, rel_path in enumerate(motion_files):
        if int(num_frames[motion_id].item()) < 1:
            continue  # masked: never rolled out during evaluation
        reference_path = (dataset_root / rel_path).resolve()
        if not reference_path.is_file():
            missing.append(rel_path)
            continue
        base_stem = Path(rel_path).stem
        if "__" in base_stem:
            category, clip = base_stem.split("__", 1)
        else:
            category, clip = "uncategorized", base_stem
        entries.append(LibraryEntry(category, clip, motion_id, reference_path, 0))

    if missing:
        print(
            f"Warning: skipped {len(missing)} motions with missing reference "
            f"files under {dataset_root}"
        )
        for rel_path in missing[:5]:
            print(f"  missing {rel_path}")
    if not entries:
        raise FileNotFoundError(
            f"No browsable motions with resolvable reference files found in "
            f"{physics_path} under {dataset_root}"
        )
    return sorted(
        entries, key=lambda e: (natural_sort_key(e.category), natural_sort_key(e.clip))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, help="Kinematic reference .motion (single-pair mode)"
    )
    parser.add_argument(
        "--physics", type=Path,
        help=(
            "Physics rollout MotionLib .pt. Required unless --reference-only "
            "is combined with --reference (single-clip mode needs no physics "
            "file at all); still used to enumerate categories/clips in "
            "--dataset mode even when --reference-only skips loading its "
            "trajectory data."
        ),
    )
    parser.add_argument(
        "--reference-only", action="store_true",
        help=(
            "Show only the kinematic reference, played at its own native FPS "
            "with no physics rollout loaded, aligned, or rendered. Use this "
            "to pinpoint an issue in the reference motion itself without the "
            "shared reference/physics timeline (sized to the shorter of the "
            "two durations at the longer of the two FPS) changing which "
            "native reference frame a given scrubber position maps to."
        ),
    )
    parser.add_argument("--reference-motion-id", type=int, default=0)
    parser.add_argument("--physics-motion-id", type=int, default=0)
    parser.add_argument(
        "--dataset",
        type=Path,
        help=(
            "With --physics: root used to resolve that library's embedded "
            "motion_files paths (e.g. the repo root, since motion_files are "
            "stored relative to it). With --reference-only and no --physics: a "
            "directory of .motion files, scanned recursively, for browsing a "
            "retargeted dataset before anything has been trained on it. Either "
            "way this enables the Category/Motion selectors in the Viser GUI "
            "instead of a fixed --reference/--physics-motion-id pair"
        ),
    )
    parser.add_argument("--category", help="Initial category in dataset mode")
    parser.add_argument("--clip", help="Initial clip in dataset mode")
    parser.add_argument("--g1-urdf", type=Path, default=DEFAULT_G1_URDF)
    parser.add_argument("--g1-mesh-dir", type=Path, default=DEFAULT_G1_MESH_DIR)
    parser.add_argument(
        "--no-g1-source", action="store_true",
        help=(
            "Do not show the G1 clip each R1 reference was retargeted from. "
            "The source is found automatically next to the reference; mirrored "
            "clips have no G1 counterpart and never show one."
        ),
    )
    parser.add_argument("--r1-urdf", type=Path, default=DEFAULT_R1_URDF)
    parser.add_argument("--r1-mesh-dir", type=Path, default=DEFAULT_R1_MESH_DIR)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--separation", type=float, default=0.0,
        help="Lateral reference offset; zero gives a direct overlay.",
    )
    parser.add_argument(
        "--velocity-arrow-scale", type=float, default=0.5,
        help="Displayed metres per m/s of annotated planar velocity.",
    )
    parser.add_argument(
        "--yaw-arrow-scale", type=float, default=0.25,
        help="Displayed metres per rad/s of annotated yaw rate.",
    )
    args = parser.parse_args()

    # --physics is required only when a rollout is actually rendered.
    # --reference-only browses either a single --reference or a --dataset
    # directory scanned directly, so neither needs a library.
    if not args.reference_only and args.physics is None:
        parser.error("--physics is required unless --reference-only is given")

    physics_raw = (
        torch.load(args.physics, map_location="cpu", weights_only=False)
        if args.physics is not None
        else None
    )
    if args.dataset is not None:
        if args.reference is not None:
            parser.error("--dataset cannot be combined with --reference")
        entries = (
            discover_motion_directory(args.dataset.resolve())
            if physics_raw is None
            else discover_physics_library(
                physics_raw, args.physics, args.dataset.resolve()
            )
        )
    else:
        if args.reference is None:
            parser.error("provide either --dataset or --reference")
        entries = [
            LibraryEntry(
                "single",
                args.reference.stem,
                args.physics_motion_id,
                args.reference.resolve(),
                args.reference_motion_id,
            )
        ]

    categories = sorted({entry.category for entry in entries}, key=natural_sort_key)
    initial_category = args.category or categories[0]
    if initial_category not in categories:
        parser.error(
            f"unknown --category {initial_category!r}; choose from {categories}"
        )
    entries_by_key = {(entry.category, entry.clip): entry for entry in entries}
    clips_by_category = {
        category: sorted(
            [entry.clip for entry in entries if entry.category == category],
            key=natural_sort_key,
        )
        for category in categories
    }
    labels_by_category = {
        category: [
            clip_label(entries_by_key[(category, clip)]) for clip in clips
        ]
        for category, clips in clips_by_category.items()
    }
    clip_by_label = {
        clip_label(entry): entry.clip for entry in entries
    }
    initial_clips = clips_by_category[initial_category]
    initial_clip = args.clip or initial_clips[0]
    if initial_clip not in initial_clips:
        parser.error(
            f"unknown --clip {initial_clip!r} in category {initial_category!r}"
        )
    active_entry = entries_by_key[(initial_category, initial_clip)]
    reference = load_r1_motion(active_entry.reference_path, active_entry.reference_motion_id)
    if args.reference_only:
        physics = None
        reference = _drop_zero_pad_frame(reference)
        duration = len(reference["joint_angles"]) / float(reference["fps"])
    else:
        physics = _slice_r1_motion(physics_raw, active_entry.motion_id, args.physics)
        reference, physics = prepare_pair(reference, physics)
        duration = min(
            len(reference["joint_angles"]) / float(reference["fps"]),
            len(physics["joint_angles"]) / float(physics["fps"]),
        )

    def load_urdf():
        return yourdfpy.URDF.load(
            str(args.r1_urdf),
            filename_handler=lambda fname: str(
                args.r1_mesh_dir / Path(fname).name
            ),
        )

    g1_urdf = None
    g1_joint_order = None
    if not args.no_g1_source:
        try:
            g1_urdf = yourdfpy.URDF.load(
                str(args.g1_urdf),
                filename_handler=lambda fname: str(
                    args.g1_mesh_dir / Path(fname).name
                ),
            )
            g1_joint_order = list(g1_urdf.actuated_joint_names)
        except Exception as exc:  # pragma: no cover - asset availability
            print(f"G1 source display disabled ({exc})")
            g1_urdf = None

    def load_g1_for(entry) -> dict | None:
        if g1_urdf is None:
            return None
        source = find_g1_source(entry.reference_path)
        if source is None:
            return None
        motion = load_g1_motion(source)
        names = motion["joint_names"]
        angles = motion["joint_angles"]
        if names is not None:
            index = {n: i for i, n in enumerate(names)}
            motion["joint_angles"] = np.stack(
                [
                    angles[:, index[n]] if n in index
                    else np.zeros(len(angles), dtype=np.float32)
                    for n in g1_joint_order
                ],
                axis=1,
            )
        motion["path"] = source
        return motion

    g1_motion = load_g1_for(active_entry)

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid(
        "/ground", width=20.0, height=20.0, cell_size=0.1,
        section_size=0.5, plane="xy", plane_opacity=0.2,
    )
    g1_root = None
    g1_vis = None
    if g1_urdf is not None:
        g1_root = server.scene.add_frame("/g1_source", show_axes=False)
        g1_vis = ViserUrdf(
            server, g1_urdf, root_node_name="/g1_source",
            mesh_color_override=(0.30, 0.85, 0.35, 0.40),
        )
    ref_root = server.scene.add_frame("/reference", show_axes=False)
    ref_vis = ViserUrdf(
        server, load_urdf(), root_node_name="/reference",
        mesh_color_override=(0.20, 0.55, 1.0, 0.45),
    )
    if args.reference_only:
        phy_root = None
        phy_vis = None
    else:
        phy_root = server.scene.add_frame("/physics", show_axes=False)
        phy_vis = ViserUrdf(
            server, load_urdf(), root_node_name="/physics",
            mesh_color_override=(1.0, 0.35, 0.10, 0.75),
        )

    playing = server.gui.add_checkbox("Playing", True)
    speed = server.gui.add_slider("Playback speed", 0.1, 2.0, 0.1, 1.0)
    overlay = server.gui.add_checkbox("Overlay", args.separation == 0.0)
    show_ref = server.gui.add_checkbox("Show kinematic reference", True)
    show_g1 = server.gui.add_checkbox(
        "Show G1 retarget source", g1_motion is not None
    )
    show_ref_contacts = server.gui.add_checkbox(
        "Show reference foot contacts", reference["contacts"] is not None
    )
    if args.reference_only:
        show_phy = None
        show_contacts = None
        show_commands = None
    else:
        show_phy = server.gui.add_checkbox("Show physics rollout", True)
        show_contacts = server.gui.add_checkbox("Show physics foot contacts", True)
        show_commands = server.gui.add_checkbox(
            "Show simulation root commands",
            physics["motion_commands"] is not None,
        )
    category_selector = server.gui.add_dropdown(
        "Category", categories, initial_value=initial_category,
        visible=args.dataset is not None,
    )
    clip_selector = server.gui.add_dropdown(
        "Motion", labels_by_category[initial_category], initial_value=clip_label(active_entry),
        visible=args.dataset is not None,
    )
    if args.reference_only:
        timeline_fps = float(reference["fps"])
    else:
        timeline_fps = max(float(reference["fps"]), float(physics["fps"]))
    timeline_frames = max(1, round(duration * timeline_fps))
    frame = server.gui.add_slider("Frame", 0, timeline_frames - 1, 1, 0)
    status = server.gui.add_markdown(
        f"**{active_entry.category}/{active_entry.clip}**  \n"
        f"{timeline_frames} frames at {timeline_fps:g} FPS"
        + ("  \nreference-only: native FPS, no physics loaded" if args.reference_only else "")
    )
    command_status = (
        None
        if args.reference_only
        else server.gui.add_markdown(
            "Simulation-root annotations loaded"
            if physics["motion_commands"] is not None
            else "No simulation-root annotations in this physics file"
        )
    )

    print(f"Viser: http://localhost:{args.port}")
    if args.reference_only:
        print(
            "Blue=kinematic reference (reference-only: no physics rollout loaded), "
            "cyan=reference foot contact, green=G1 retarget source"
        )
        print(f"Reference {float(reference['fps']):.2f} Hz; duration {duration:.2f} s")
    else:
        print(
            "Blue=kinematic reference, orange=RL physics rollout, "
            "magenta=physics foot contact, cyan=reference foot contact, "
            "green=G1 retarget source"
        )
        print("Cyan=simulation-root facing, green=planar command, pink=signed yaw command")
        print(
            f"Reference {float(reference['fps']):.2f} Hz; physics "
            f"{float(physics['fps']):.2f} Hz; duration {duration:.2f} s"
        )
    print(f"Loaded {len(entries)} motions in {len(categories)} categories")
    last_advance = time.perf_counter()
    last_reload_check = last_advance
    if args.reference_only:
        physics_mtime = None
    else:
        print(f"Watching {args.physics} for updates")
        physics_mtime = args.physics.stat().st_mtime_ns
    selected_category = initial_category
    selected_clip = initial_clip
    while True:
        now = time.perf_counter()
        if category_selector.value != selected_category:
            selected_category = category_selector.value
            clip_selector.options = labels_by_category[selected_category]
            selected_clip = clip_by_label[clip_selector.value]

        if clip_by_label[clip_selector.value] != selected_clip:
            selected_clip = clip_by_label[clip_selector.value]

        selected_entry = entries_by_key[(selected_category, selected_clip)]
        if selected_entry != active_entry:
            active_entry = selected_entry
            reference = load_r1_motion(
                active_entry.reference_path, active_entry.reference_motion_id
            )
            g1_motion = load_g1_for(active_entry)
            show_g1.value = g1_motion is not None
            if args.reference_only:
                reference = _drop_zero_pad_frame(reference)
                duration = len(reference["joint_angles"]) / float(reference["fps"])
                timeline_fps = float(reference["fps"])
            else:
                physics = _slice_r1_motion(physics_raw, active_entry.motion_id, args.physics)
                reference, physics = prepare_pair(reference, physics)
                duration = min(
                    len(reference["joint_angles"]) / float(reference["fps"]),
                    len(physics["joint_angles"]) / float(physics["fps"]),
                )
                timeline_fps = max(float(reference["fps"]), float(physics["fps"]))
            timeline_frames = max(1, round(duration * timeline_fps))
            frame.max = timeline_frames - 1
            frame.value = 0
            status.content = (
                f"**{active_entry.category}/{active_entry.clip}**  \n"
                f"{timeline_frames} frames at {timeline_fps:g} FPS"
                + ("  \nreference-only: native FPS, no physics loaded" if args.reference_only else "")
            )
            if not args.reference_only:
                show_commands.value = physics["motion_commands"] is not None
            last_advance = now
            print(f"Loaded {active_entry.category}/{active_entry.clip} ({timeline_frames} frames)")

        if not args.reference_only and now - last_reload_check >= 0.5:
            last_reload_check = now
            new_mtime = args.physics.stat().st_mtime_ns
            if new_mtime != physics_mtime:
                # The trainer writes a new checkpoint atomically enough for
                # normal local use. If we catch it mid-write, retry next poll.
                try:
                    new_raw = torch.load(args.physics, map_location="cpu", weights_only=False)
                    new_physics = _slice_r1_motion(new_raw, active_entry.motion_id, args.physics)
                    _, new_physics = prepare_pair(reference, new_physics)
                except (OSError, ValueError, EOFError, RuntimeError):
                    pass
                else:
                    physics_raw = new_raw
                    physics = new_physics
                    physics_mtime = new_mtime
                    duration = min(
                        len(reference["joint_angles"]) / float(reference["fps"]),
                        len(physics["joint_angles"]) / float(physics["fps"]),
                    )
                    timeline_fps = max(float(reference["fps"]), float(physics["fps"]))
                    timeline_frames = max(1, round(duration * timeline_fps))
                    frame.max = timeline_frames - 1
                    frame.value = min(frame.value, timeline_frames - 1)
                    show_commands.value = physics["motion_commands"] is not None
                    print(f"Reloaded physics rollout ({timeline_frames} frames)")

        period = 1.0 / (timeline_fps * speed.value)
        if playing.value and now - last_advance >= period:
            frame.value = (frame.value + 1) % timeline_frames
            last_advance = now

        seconds = frame.value / timeline_fps
        ri = min(
            round(seconds * float(reference["fps"])),
            len(reference["joint_angles"]) - 1,
        )
        lateral = 0.0 if overlay.value else args.separation
        ref_vis.show_visual = show_ref.value
        if not args.reference_only:
            pi = min(
                round(seconds * float(physics["fps"])),
                len(physics["joint_angles"]) - 1,
            )
            phy_vis.show_visual = show_phy.value
        with server.atomic():
            ref_root.wxyz = reference["root_wxyz"][ri]
            ref_root.position = reference["root_pos"][ri] + np.array([0.0, lateral, 0.0])
            ref_vis.update_cfg(reference["joint_angles"][ri])

            # The G1 source, on its own native clock. It is the same take the
            # R1 was retargeted from, so any difference between the green and
            # blue skeletons is retargeting error rather than a different
            # motion. Absent for mirrored clips, which are generated after
            # retargeting and have no G1 counterpart.
            if g1_vis is not None:
                visible = show_g1.value and g1_motion is not None
                g1_vis.show_visual = visible
                if visible:
                    gi = min(
                        int(frame.value / timeline_fps * g1_motion["fps"]),
                        len(g1_motion["joint_angles"]) - 1,
                    )
                    g1_root.wxyz = g1_motion["root_wxyz"][gi]
                    g1_root.position = g1_motion["root_pos"][gi] + np.array(
                        [0.0, -lateral, 0.0]
                    )
                    g1_vis.update_cfg(g1_motion["joint_angles"][gi])

            # The reference's own contact labels, drawn at its feet. These come
            # from the retargeter's foot_contacts (ultimately the Kimodo
            # labels), not from the R1 sole geometry, so they can disagree with
            # where this skeleton's foot actually is -- which is exactly what
            # this overlay is for. Cyan = labelled in contact, dark = airborne.
            ref_contacts = reference["contacts"]
            if show_ref_contacts.value and ref_contacts is not None:
                foot_ids = [LEFT_FOOT_BODY, RIGHT_FOOT_BODY]
                active = ref_contacts[ri, foot_ids] > 0.5
                points = reference["body_pos"][ri, foot_ids] + np.array(
                    [0.0, lateral, 0.0]
                )
                colors = np.asarray(
                    [[0, 220, 255] if value else [40, 60, 70] for value in active],
                    dtype=np.uint8,
                )
                server.scene.add_point_cloud(
                    "/reference_contacts", points=points, colors=colors,
                    point_size=0.045, point_shape="circle",
                )

            if not args.reference_only:
                phy_root.wxyz = physics["root_wxyz"][pi]
                phy_root.position = physics["root_pos"][pi]
                phy_vis.update_cfg(physics["joint_angles"][pi])

                contacts = physics["contacts"]
                if show_contacts.value and contacts is not None:
                    foot_ids = [LEFT_FOOT_BODY, RIGHT_FOOT_BODY]
                    active = contacts[pi, foot_ids] > 0.5
                    points = physics["body_pos"][pi, foot_ids]
                    colors = np.asarray(
                        [[220, 0, 255] if value else [80, 80, 80] for value in active],
                        dtype=np.uint8,
                    )
                    server.scene.add_point_cloud(
                        "/physics_contacts", points=points, colors=colors,
                        point_size=0.035, point_shape="circle",
                    )

                commands = physics["motion_commands"]
                command_visible = show_commands.value and commands is not None
                if command_visible:
                    root_pos = physics["simulation_root_pos"][pi].copy()
                    root_pos[2] = 0.025
                    root_rotation = physics["simulation_root_xyzw"][pi]
                    facing_xy = _facing_xy_from_xyzw(root_rotation)
                    facing_vector = np.array([facing_xy[0], facing_xy[1], 0.0]) * 0.35
                    local_velocity_xy = commands[pi, :2]
                    velocity_xy = np.array(
                        [
                            facing_xy[0] * local_velocity_xy[0]
                            - facing_xy[1] * local_velocity_xy[1],
                            facing_xy[1] * local_velocity_xy[0]
                            + facing_xy[0] * local_velocity_xy[1],
                        ]
                    )
                    velocity_vector = np.array(
                        [velocity_xy[0], velocity_xy[1], 0.0]
                    ) * args.velocity_arrow_scale
                    heading_tip = root_pos + facing_vector
                    positive_yaw_tangent = np.array(
                        [-facing_xy[1], facing_xy[0], 0.0]
                    )
                    yaw_rate = float(commands[pi, 2])
                    yaw_vector = (
                        positive_yaw_tangent * yaw_rate * args.yaw_arrow_scale
                    )
                    command_status.content = (
                        f"**Command**: vx={commands[pi, 0]:+.3f} m/s, "
                        f"vy={commands[pi, 1]:+.3f} m/s, "
                        f"yaw={yaw_rate:+.3f} rad/s"
                    )
                    achieved_commands = physics["achieved_motion_commands"]
                    if achieved_commands is not None:
                        achieved = achieved_commands[pi]
                        command_status.content += (
                            f"  \n**Smoothed achieved**: vx={achieved[0]:+.3f} m/s, "
                            f"vy={achieved[1]:+.3f} m/s, "
                            f"yaw={achieved[2]:+.3f} rad/s"
                        )
                else:
                    hidden = np.array([0.0, 0.0, -100.0])
                    root_pos = hidden
                    heading_tip = hidden
                    facing_vector = velocity_vector = yaw_vector = np.zeros(3)
                    if commands is None:
                        command_status.content = "No simulation-root annotations in this physics file"

                for name, origin, vector, color in (
                    ("facing", root_pos, facing_vector, FACING_ARROW_COLOR),
                    ("velocity", root_pos, velocity_vector, VELOCITY_ARROW_COLOR),
                    ("yaw_rate", heading_tip, yaw_vector, YAW_ARROW_COLOR),
                ):
                    server.scene.add_line_segments(
                        f"/command_annotations/{name}",
                        points=arrow_line_segments(origin, vector),
                        colors=color,
                        thickness=0.012,
                        visible=command_visible,
                    )
        time.sleep(0.002)


if __name__ == "__main__":
    main()
