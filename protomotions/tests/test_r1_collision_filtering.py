# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the R1's selective collision filtering."""

from pathlib import Path
import xml.etree.ElementTree as ET

from protomotions.simulator.isaaclab.utils.collision_filtering import (
    collect_body_environment_filter_paths,
)


R1_MJCF = Path(__file__).parents[1] / "data" / "assets" / "mjcf" / "r1.xml"

COLLISION_BODY_NAMES = {
    "left_hip_pitch_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "waist_roll_link",
    "waist_yaw_link",
    "left_shoulder_pitch_link",
    "left_shoulder_yaw_link",
    "left_wrist_roll_link",
    "right_shoulder_pitch_link",
    "right_shoulder_yaw_link",
    "right_wrist_roll_link",
    "head_pitch_link",
    "head_yaw_link",
}


def _wrist_hip_pairs():
    return {
        frozenset((f"{hip}_hip_pitch_link", f"{wrist}_wrist_roll_link"))
        for hip in ("left", "right")
        for wrist in ("left", "right")
    }


def _allowed_self_collision_pairs():
    return _wrist_hip_pairs() | {
        frozenset(("left_hip_yaw_link", "right_hip_yaw_link"))
    }


def _collision_masks():
    root = ET.parse(R1_MJCF).getroot()
    default_geom = root.find("./default/geom")
    default_contype = int(default_geom.get("contype", "1"))
    default_conaffinity = int(default_geom.get("conaffinity", "1"))

    masks = {}
    for geom in root.findall(".//worldbody//geom"):
        contype = int(geom.get("contype", default_contype))
        conaffinity = int(geom.get("conaffinity", default_conaffinity))
        if contype or conaffinity:
            name = geom.get("name")
            if name is not None:
                masks[name] = (contype, conaffinity)
    return masks


def _can_collide(first, second):
    first_type, first_affinity = first
    second_type, second_affinity = second
    return bool(
        (first_type & second_affinity) or (second_type & first_affinity)
    )


def test_r1_self_collision_is_limited_to_selected_pairs():
    masks = _collision_masks()
    robot_pairs = {
        frozenset((first, second))
        for first, first_mask in masks.items()
        for second, second_mask in masks.items()
        if first < second and _can_collide(first_mask, second_mask)
    }

    expected_geom_pairs = {
        frozenset((f"{hip}_hip_pitch_collision", f"{wrist}_wrist_roll_collision"))
        for hip in ("left", "right")
        for wrist in ("left", "right")
    } | {
        frozenset(("left_hip_yaw_collision", "right_hip_yaw_collision"))
    }
    assert robot_pairs == expected_geom_pairs

    # Isaac Lab 3's MJCF-to-USD converter does not preserve general MuJoCo
    # contype/conaffinity masks. It does preserve explicit contact excludes as
    # UsdPhysics.FilteredPairsAPI relationships, so validate that representation
    # independently of the masks used by native MuJoCo.
    root = ET.parse(R1_MJCF).getroot()
    excluded_pairs = {
        frozenset((exclude.get("body1"), exclude.get("body2")))
        for exclude in root.findall("./contact/exclude")
    }
    all_collision_pairs = {
        frozenset((first, second))
        for first in COLLISION_BODY_NAMES
        for second in COLLISION_BODY_NAMES
        if first < second
    }
    assert excluded_pairs == all_collision_pairs - _allowed_self_collision_pairs()


def test_r1_hip_yaw_colliders_only_use_their_private_mask():
    external_mask = (1, 1)
    masks = _collision_masks()
    hip_yaw_geoms = {
        "left_hip_yaw_collision",
        "right_hip_yaw_collision",
    }
    assert all(
        not _can_collide(masks[name], external_mask) for name in hip_yaw_geoms
    )
    assert all(
        _can_collide(mask, external_mask)
        for name, mask in masks.items()
        if name not in hip_yaw_geoms
    )


def test_r1_isaaclab_filters_hip_yaw_bodies_from_environment_only():
    hip_yaw_bodies = [
        "left_hip_yaw_link",
        "right_hip_yaw_link",
    ]

    paths = [
        "/World/ground",
        "/World/ground/terrain/mesh",
        "/World/envs/env_0/Robot/pelvis/left_hip_yaw_link",
        "/World/envs/env_0/Robot/pelvis/right_hip_yaw_link",
        "/World/envs/env_0/Robot/pelvis/left_hip_pitch_link",
        "/World/envs/env_0/Object_0",
        "/World/envs/env_0/Projectile_0",
        "/World/envs/env_1/Robot/pelvis/left_hip_yaw_link",
        "/World/envs/env_1/Robot/pelvis/right_hip_yaw_link",
        "/World/envs/env_1/Object_0",
    ]
    filters = collect_body_environment_filter_paths(
        paths, hip_yaw_bodies
    )

    assert set(filters) == {
        "/World/envs/env_0/Robot/pelvis/left_hip_yaw_link",
        "/World/envs/env_0/Robot/pelvis/right_hip_yaw_link",
        "/World/envs/env_1/Robot/pelvis/left_hip_yaw_link",
        "/World/envs/env_1/Robot/pelvis/right_hip_yaw_link",
    }
    assert filters["/World/envs/env_0/Robot/pelvis/left_hip_yaw_link"] == (
        "/World/ground",
        "/World/ground/terrain/mesh",
        "/World/envs/env_0/Object_0",
        "/World/envs/env_0/Projectile_0",
    )
    assert filters["/World/envs/env_1/Robot/pelvis/right_hip_yaw_link"] == (
        "/World/ground",
        "/World/ground/terrain/mesh",
        "/World/envs/env_1/Object_0",
    )
