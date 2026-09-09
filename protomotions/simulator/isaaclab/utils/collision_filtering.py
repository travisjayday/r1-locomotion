# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab collision-filtering helpers not representable by its MJCF importer."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def _environment_root(path: str) -> str | None:
    """Return ``/World/envs/env_N`` for a prim below an environment clone."""
    parts = path.split("/")
    if (
        len(parts) >= 5
        and parts[1:3] == ["World", "envs"]
        and parts[3].startswith("env_")
    ):
        return "/".join(parts[:4])
    return None


def collect_body_environment_filter_paths(
    prim_paths: Iterable[str], body_names: Iterable[str]
) -> dict[str, tuple[str, ...]]:
    """Map selected robot bodies to external collider roots in their environment.

    The terrain is global. Scene objects and projectiles are local to an
    environment clone. Robot paths are deliberately not targets: their
    self-collision policy remains controlled by MJCF ``contact/exclude`` pairs.
    """
    paths = tuple(str(path) for path in prim_paths)
    selected_names = set(body_names)
    bodies_by_environment: dict[str, list[str]] = defaultdict(list)
    externals_by_environment: dict[str, list[str]] = defaultdict(list)

    for path in paths:
        environment = _environment_root(path)
        if environment is None:
            continue
        relative = path[len(environment) + 1 :]
        if relative.startswith("Robot/") and path.rsplit("/", 1)[-1] in selected_names:
            bodies_by_environment[environment].append(path)
        elif "/" not in relative and (
            relative.startswith("Object_") or relative.startswith("Projectile_")
        ):
            externals_by_environment[environment].append(path)

    terrain_targets = tuple(
        path for path in paths if path in ("/World/ground", "/World/ground/terrain/mesh")
    )
    result: dict[str, tuple[str, ...]] = {}
    for environment, body_paths in bodies_by_environment.items():
        targets = tuple(
            dict.fromkeys((*terrain_targets, *externals_by_environment[environment]))
        )
        for body_path in body_paths:
            result[body_path] = targets
    return result


def apply_body_environment_collision_filters(stage, body_names: Iterable[str]) -> int:
    """Disable selected body collisions with terrain and environment assets.

    Returns the number of selected body prims updated. Imports USD lazily so
    non-Isaac backends can import ProtoMotions without a Kit/USD installation.
    """
    names = tuple(body_names)
    if not names:
        return 0

    from pxr import Sdf, UsdPhysics

    prim_paths = [prim.GetPath().pathString for prim in stage.Traverse()]
    filters = collect_body_environment_filter_paths(prim_paths, names)
    for body_path, target_paths in filters.items():
        body_prim = stage.GetPrimAtPath(body_path)
        api = UsdPhysics.FilteredPairsAPI.Apply(body_prim)
        relationship = api.CreateFilteredPairsRel()
        existing_targets = set(relationship.GetTargets())
        for target_path in target_paths:
            target = Sdf.Path(target_path)
            if target not in existing_targets:
                relationship.AddTarget(target)
    return len(filters)
