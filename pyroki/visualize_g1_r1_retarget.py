# SPDX-License-Identifier: Apache-2.0
"""Fast Viser comparison of source G1 and retargeted R1 NPZ motions.

The viewer accepts either one explicit ``--g1``/``--r1`` pair or an entire
conversion output via ``--dataset``. Dataset mode reconstructs the original
Kimodo category/clip hierarchy from ``category__clip`` filenames and exposes
category and motion selectors in the Viser GUI.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_G1_URDF = (
    SCRIPT_DIR / "../protomotions/data/assets/urdf/for_retargeting/g1.urdf"
).resolve()
DEFAULT_R1_URDF = (
    SCRIPT_DIR / "../protomotions/data/assets/urdf/for_retargeting/r1.urdf"
).resolve()
DEFAULT_G1_MESH_DIR = (
    SCRIPT_DIR / "../protomotions/data/assets/mesh/G1"
).resolve()
DEFAULT_R1_MESH_DIR = (
    SCRIPT_DIR / "../protomotions/data/assets/mesh/R1"
).resolve()

G1_SOLE_POINTS = np.array(
    [[-0.05, 0.025, -0.03], [-0.05, -0.025, -0.03],
     [0.12, 0.03, -0.03], [0.12, -0.03, -0.03]], dtype=np.float32
)
R1_SOLE_POINTS = np.array(
    [[-0.04, 0.025, -0.055], [-0.04, -0.025, -0.055],
     [0.115, 0.025, -0.055], [0.115, -0.025, -0.055]], dtype=np.float32
)
G1_CONTACT_ACTIVE_COLOR = np.array([0, 255, 255], dtype=np.uint8)
G1_CONTACT_INACTIVE_COLOR = np.array([35, 65, 90], dtype=np.uint8)
R1_CONTACT_ACTIVE_COLOR = np.array([255, 245, 0], dtype=np.uint8)
R1_CONTACT_INACTIVE_COLOR = np.array([100, 55, 30], dtype=np.uint8)


@dataclass(frozen=True)
class MotionPair:
    category: str
    clip: str
    g1_path: Path
    r1_path: Path


def natural_sort_key(value: str) -> tuple[object, ...]:
    """Sort embedded numbers numerically while keeping names deterministic."""
    import re

    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def discover_dataset(dataset: Path) -> list[MotionPair]:
    """Pair G1 and R1 NPZs from a converter output directory."""
    # Native Kimodo is now the default retarget input; retain the old corrected
    # directory as a fallback for legacy conversion outputs.
    g1_candidates = (dataset / "g1-kimodo-input", dataset / "g1-foot-fixed")
    g1_dir = next((path for path in g1_candidates if path.is_dir()), None)
    r1_dir = dataset / "r1-retargeted"
    if g1_dir is None:
        raise FileNotFoundError(
            f"Missing source G1 directory; checked {', '.join(map(str, g1_candidates))}"
        )
    if not r1_dir.is_dir():
        raise FileNotFoundError(f"Missing retargeted R1 directory: {r1_dir}")

    pairs: list[MotionPair] = []
    missing: list[Path] = []
    for r1_path in sorted(r1_dir.rglob("*_r1.npz")):
        relative = r1_path.relative_to(r1_dir)
        base_stem = r1_path.stem.removesuffix("_r1")
        relative_g1 = relative.with_name(f"{base_stem}.npz")
        candidates = (g1_dir / relative_g1, g1_dir / f"{base_stem}.npz")
        g1_path = next((path for path in candidates if path.is_file()), None)
        if g1_path is None:
            missing.append(r1_path)
            continue

        if "__" in base_stem:
            category, clip = base_stem.split("__", 1)
        elif relative.parent != Path("."):
            category, clip = relative.parent.as_posix(), base_stem
        else:
            category, clip = "uncategorized", base_stem
        pairs.append(MotionPair(category, clip, g1_path, r1_path))

    if missing:
        print(f"Warning: skipped {len(missing)} R1 motions without matching G1 NPZs")
        for path in missing[:5]:
            print(f"  missing G1 pair for {path}")
    if not pairs:
        raise FileNotFoundError(
            f"No paired G1/R1 NPZ motions found under {dataset}"
        )
    return sorted(pairs, key=lambda pair: (
        natural_sort_key(pair.category), natural_sort_key(pair.clip)
    ))


def load_motion(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=True) as data:
        required = ("base_frame_pos", "base_frame_wxyz", "joint_angles")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        result: dict[str, np.ndarray | float] = {
            key: np.asarray(data[key]) for key in required
        }
        result["foot_contacts"] = (
            np.asarray(data["foot_contacts"])
            if "foot_contacts" in data
            else np.zeros((len(result["joint_angles"]), 2), dtype=np.float32)
        )
        result["foot_point_contacts"] = (
            np.asarray(data["foot_point_contacts"])
            if "foot_point_contacts" in data
            else contacts_to_sole_points(np.asarray(result["foot_contacts"]))
        )
        result["fps"] = float(data["fps"]) if "fps" in data else 30.0
    return result


def contacts_to_sole_points(contacts: np.ndarray) -> np.ndarray:
    """Normalize side or Kimodo heel/toe contacts to [T, foot, 4 points]."""
    contacts = np.asarray(contacts, dtype=np.float32)
    if contacts.ndim == 3 and contacts.shape[1:] == (2, 4):
        return contacts
    if contacts.ndim != 2:
        raise ValueError(f"Unsupported contact shape: {contacts.shape}")
    if contacts.shape[1] == 2:
        return np.repeat(contacts[:, :, None], 4, axis=2)
    if contacts.shape[1] == 4:
        result = np.empty((len(contacts), 2, 4), dtype=np.float32)
        result[:, 0, :2] = contacts[:, 0, None]
        result[:, 0, 2:] = contacts[:, 1, None]
        result[:, 1, :2] = contacts[:, 2, None]
        result[:, 1, 2:] = contacts[:, 3, None]
        return result
    raise ValueError(f"Unsupported contact shape: {contacts.shape}")


def quaternion_matrix(wxyz: np.ndarray) -> np.ndarray:
    """Convert one normalized wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(wxyz, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def sole_points_world(
    urdf: yourdfpy.URDF,
    root_position: np.ndarray,
    root_wxyz: np.ndarray,
    sole_points: np.ndarray,
    lateral_offset: float,
) -> np.ndarray:
    """Return eight sole samples in Viser world coordinates."""
    root_rotation = quaternion_matrix(root_wxyz)
    root_translation = np.asarray(root_position, dtype=np.float32) + np.array(
        [0.0, lateral_offset, 0.0], dtype=np.float32
    )
    result = []
    for side in ("left", "right"):
        root_T_ankle = np.asarray(
            urdf.get_transform(f"{side}_ankle_roll_link"), dtype=np.float32
        )
        points_root = (
            sole_points @ root_T_ankle[:3, :3].T + root_T_ankle[:3, 3]
        )
        result.append(points_root @ root_rotation.T + root_translation)
    return np.concatenate(result, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1", type=Path, help="Source G1 NPZ (single-pair mode)")
    parser.add_argument("--r1", type=Path, help="Retargeted R1 NPZ (single-pair mode)")
    parser.add_argument(
        "--dataset",
        type=Path,
        help=(
            "Converter output root containing g1-kimodo-input/ (or legacy "
            "g1-foot-fixed/) and r1-retargeted/; enables motion selection"
        ),
    )
    parser.add_argument("--category", help="Initial category in dataset mode")
    parser.add_argument("--clip", help="Initial clip in dataset mode")
    parser.add_argument("--g1-urdf", type=Path, default=DEFAULT_G1_URDF)
    parser.add_argument("--r1-urdf", type=Path, default=DEFAULT_R1_URDF)
    parser.add_argument("--g1-mesh-dir", type=Path, default=DEFAULT_G1_MESH_DIR)
    parser.add_argument("--r1-mesh-dir", type=Path, default=DEFAULT_R1_MESH_DIR)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--separation", type=float, default=0.7,
        help="G1 lateral offset in meters; set 0 for an overlay",
    )
    args = parser.parse_args()

    if args.dataset is not None:
        if args.g1 is not None or args.r1 is not None:
            parser.error("--dataset cannot be combined with --g1 or --r1")
        pairs = discover_dataset(args.dataset.resolve())
    else:
        if args.g1 is None or args.r1 is None:
            parser.error("provide either --dataset or both --g1 and --r1")
        pairs = [MotionPair("single", args.r1.stem, args.g1.resolve(), args.r1.resolve())]

    categories = sorted({pair.category for pair in pairs}, key=natural_sort_key)
    initial_category = args.category or categories[0]
    if initial_category not in categories:
        parser.error(
            f"unknown --category {initial_category!r}; choose from {categories}"
        )
    pairs_by_key = {(pair.category, pair.clip): pair for pair in pairs}
    clips_by_category = {
        category: sorted(
            [pair.clip for pair in pairs if pair.category == category],
            key=natural_sort_key,
        )
        for category in categories
    }
    initial_clips = clips_by_category[initial_category]
    initial_clip = args.clip or initial_clips[0]
    if initial_clip not in initial_clips:
        parser.error(
            f"unknown --clip {initial_clip!r} in category {initial_category!r}"
        )
    active_pair = pairs_by_key[(initial_category, initial_clip)]
    g1, r1 = load_motion(active_pair.g1_path), load_motion(active_pair.r1_path)
    frames = min(len(g1["joint_angles"]), len(r1["joint_angles"]))
    if frames < 1:
        raise ValueError("Motions contain no frames")

    # The two retarget URDFs use different relative-path conventions. Resolve
    # by basename so this diagnostic viewer is independent of the current cwd.
    g1_urdf = yourdfpy.URDF.load(
        str(args.g1_urdf),
        filename_handler=lambda fname: str(args.g1_mesh_dir / Path(fname).name),
    )
    r1_urdf = yourdfpy.URDF.load(
        str(args.r1_urdf),
        filename_handler=lambda fname: str(args.r1_mesh_dir / Path(fname).name),
    )
    server = viser.ViserServer(port=args.port)
    server.scene.add_grid(
        "/ground", width=10.0, height=10.0, cell_size=0.1,
        section_size=0.5, plane="xy", plane_opacity=0.15,
    )
    g1_root = server.scene.add_frame("/g1", show_axes=False)
    r1_root = server.scene.add_frame("/r1", show_axes=False)
    contacts_root = server.scene.add_frame("/contacts", show_axes=False)
    g1_vis = ViserUrdf(
        server, g1_urdf, root_node_name="/g1",
        mesh_color_override=(0.25, 0.55, 1.0, 0.55),
    )
    r1_vis = ViserUrdf(
        server, r1_urdf, root_node_name="/r1",
        mesh_color_override=(1.0, 0.45, 0.15, 0.85),
    )

    playing = server.gui.add_checkbox("Playing", True)
    speed = server.gui.add_slider("Playback speed", 0.1, 2.0, 0.1, 1.0)
    overlay = server.gui.add_checkbox("Overlay robots", args.separation == 0.0)
    show_g1 = server.gui.add_checkbox("Show G1", True)
    show_r1 = server.gui.add_checkbox("Show R1", True)
    show_contacts = server.gui.add_checkbox("Show foot contacts", True)
    server.gui.add_markdown(
        "Contact markers: **cyan=G1 active**, **yellow=R1 active**; "
        "dark markers are inactive sole samples."
    )
    category_selector = server.gui.add_dropdown(
        "Category", categories, initial_value=initial_category,
        visible=args.dataset is not None,
    )
    clip_selector = server.gui.add_dropdown(
        "Motion", initial_clips, initial_value=initial_clip,
        visible=args.dataset is not None,
    )
    frame = server.gui.add_slider("Frame", 0, frames - 1, 1, 0)
    status = server.gui.add_markdown(
        f"**{active_pair.category}/{active_pair.clip}**  \n"
        f"{frames} frames at {float(r1['fps']):g} FPS"
    )

    print(f"Viser: http://localhost:{args.port}")
    print(
        "Blue=source G1, orange=retargeted R1; "
        "cyan/yellow=active G1/R1 sole contacts; grid plane is z=0"
    )
    print(f"Loaded {len(pairs)} paired motions in {len(categories)} categories")
    print(f"Watching {active_pair.r1_path} for updates")
    last_advance = time.perf_counter()
    last_reload_check = last_advance
    r1_mtime = active_pair.r1_path.stat().st_mtime_ns
    selected_category = initial_category
    selected_clip = initial_clip
    while True:
        now = time.perf_counter()
        if category_selector.value != selected_category:
            selected_category = category_selector.value
            new_clips = clips_by_category[selected_category]
            clip_selector.options = new_clips
            selected_clip = clip_selector.value

        if clip_selector.value != selected_clip:
            selected_clip = clip_selector.value

        selected_pair = pairs_by_key[(selected_category, selected_clip)]
        if selected_pair != active_pair:
            active_pair = selected_pair
            g1, r1 = load_motion(active_pair.g1_path), load_motion(active_pair.r1_path)
            frames = min(len(g1["joint_angles"]), len(r1["joint_angles"]))
            if frames < 1:
                raise ValueError(f"{active_pair.clip}: motions contain no frames")
            frame.max = frames - 1
            frame.value = 0
            r1_mtime = active_pair.r1_path.stat().st_mtime_ns
            status.content = (
                f"**{active_pair.category}/{active_pair.clip}**  \n"
                f"{frames} frames at {float(r1['fps']):g} FPS"
            )
            last_advance = now
            print(f"Loaded {active_pair.category}/{active_pair.clip} ({frames} frames)")

        if now - last_reload_check >= 0.5:
            last_reload_check = now
            new_mtime = active_pair.r1_path.stat().st_mtime_ns
            if new_mtime != r1_mtime:
                # The retargeter writes a new NPZ atomically enough for normal
                # local use. If we catch it mid-write, retry on the next poll.
                try:
                    new_r1 = load_motion(active_pair.r1_path)
                except (OSError, ValueError, EOFError):
                    pass
                else:
                    r1 = new_r1
                    r1_mtime = new_mtime
                    frames = min(len(g1["joint_angles"]), len(r1["joint_angles"]))
                    frame.max = frames - 1
                    frame.value = min(frame.value, frames - 1)
                    print(f"Reloaded R1 result ({frames} frames)")
        period = 1.0 / (float(r1["fps"]) * speed.value)
        if playing.value and now - last_advance >= period:
            frame.value = (frame.value + 1) % frames
            last_advance = now
        t = frame.value

        g1_vis.show_visual = show_g1.value
        r1_vis.show_visual = show_r1.value
        contacts_root.visible = show_contacts.value
        lateral = 0.0 if overlay.value else args.separation
        with server.atomic():
            g1_root.wxyz = np.asarray(g1["base_frame_wxyz"])[t]
            g1_root.position = np.asarray(g1["base_frame_pos"])[t] + np.array(
                [0.0, lateral, 0.0]
            )
            r1_root.wxyz = np.asarray(r1["base_frame_wxyz"])[t]
            r1_root.position = np.asarray(r1["base_frame_pos"])[t]
            g1_cfg = np.asarray(g1["joint_angles"])[t]
            r1_cfg = np.asarray(r1["joint_angles"])[t]
            g1_vis.update_cfg(g1_cfg)
            r1_vis.update_cfg(r1_cfg)

            if show_contacts.value:
                # Keep FK explicit rather than placing labels at ankle origins:
                # heel/toe-only phases should visibly pivot about the supported
                # samples while the other sole samples remain dark.
                g1_urdf.update_cfg(g1_cfg)
                r1_urdf.update_cfg(r1_cfg)
                g1_points = sole_points_world(
                    g1_urdf,
                    np.asarray(g1["base_frame_pos"])[t],
                    np.asarray(g1["base_frame_wxyz"])[t],
                    G1_SOLE_POINTS,
                    lateral,
                )
                r1_points = sole_points_world(
                    r1_urdf,
                    np.asarray(r1["base_frame_pos"])[t],
                    np.asarray(r1["base_frame_wxyz"])[t],
                    R1_SOLE_POINTS,
                    0.0,
                )
                g1_active = np.asarray(g1["foot_point_contacts"])[t].reshape(-1) > 0.5
                r1_active = np.asarray(r1["foot_point_contacts"])[t].reshape(-1) > 0.5
                g1_colors = np.where(
                    g1_active[:, None],
                    G1_CONTACT_ACTIVE_COLOR,
                    G1_CONTACT_INACTIVE_COLOR,
                ).astype(np.uint8)
                r1_colors = np.where(
                    r1_active[:, None],
                    R1_CONTACT_ACTIVE_COLOR,
                    R1_CONTACT_INACTIVE_COLOR,
                ).astype(np.uint8)
                server.scene.add_point_cloud(
                    "/contacts/g1", g1_points, g1_colors, point_size=0.018
                )
                server.scene.add_point_cloud(
                    "/contacts/r1", r1_points, r1_colors, point_size=0.022
                )
        time.sleep(0.002)


if __name__ == "__main__":
    main()
