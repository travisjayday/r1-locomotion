# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unitree R1 26-DoF configuration using WBC-AGILE actuator parameters."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.base import (
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
)


DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.30,
    ".*_knee_joint": 0.65,
    ".*_ankle_pitch_joint": -0.35,
    ".*_elbow_joint": 0.55,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
}


def _unitree_control(stiffness: float, damping: float, *, n6014: bool,
                     parallel: bool = False) -> ControlInfo:
    if n6014:
        x1, x2, y1, y2, armature, fs, fd = 11.8, 27.2, 31.7, 34.4, 0.025392063, 0.1, 0.01
    else:
        x1, x2, y1, y2, armature, fs, fd = 8.4, 15.3, 53.7, 66.7, 0.003347153, 0.6, 0.06
    return ControlInfo(
        stiffness=stiffness,
        damping=damping,
        effort_limit=y2,
        velocity_limit=x2,
        armature=armature * (2.0 if parallel else 1.0),
        actuator_model="unitree",
        torque_speed_x1=x1,
        torque_speed_x2=x2,
        torque_same_direction=y1,
        torque_opposite_direction=y2,
        friction_static=fs,
        friction_dynamic=fd,
        friction_activation_velocity=0.01,
    )


def _head_control() -> ControlInfo:
    # WBC-AGILE's R1 is 24-DoF and intentionally excludes the two head joints.
    return ControlInfo(stiffness=20.0, damping=2.0, effort_limit=17.0,
                       velocity_limit=37.7, armature=0.01)


@dataclass
class R1RobotConfig(RobotConfig):
    semantic_forward_axis_xy: Tuple[float, float] = (1.0, 0.0)
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["left_ankle_roll_link"],
            "all_right_foot_bodies": ["right_ankle_roll_link"],
            "all_left_hand_bodies": ["left_wrist_roll_link"],
            "all_right_hand_bodies": ["right_wrist_roll_link"],
            "head_body_name": ["head_yaw_link"],
            "torso_body_name": ["pelvis_link"],
        }
    )
    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "waist_yaw_link", "head_yaw_link",
            "left_ankle_roll_link", "right_ankle_roll_link",
            "left_wrist_roll_link", "right_wrist_roll_link",
        ]
    )
    default_root_height: float = 0.73
    default_dof_pos: Dict[str, float] = field(default_factory=lambda: DEFAULT_JOINT_POS)
    anchor_body_name: str = "pelvis_link"
    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/r1.xml",
            # Self-contact is limited to wrist-roll <-> hip-pitch and the two
            # hip-yaw links. The latter pair is private: it does not contact
            # terrain or other environment assets.
            self_collisions=True,
            environment_collision_exclusion_bodies=[
                "left_hip_yaw_link",
                "right_hip_yaw_link",
            ],
            replace_cylinder_with_capsule=True,
            thickness=0.01,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )
    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            action_effort_fraction=0.25,
            override_control_info={
                ".*_(hip|knee)_.*": _unitree_control(84.3355, 5.369, n6014=False),
                ".*_ankle_.*": _unitree_control(31.0834, 1.9788, n6014=True, parallel=True),
                "waist_.*": _unitree_control(84.3355, 5.369, n6014=False, parallel=True),
                ".*_shoulder_(pitch|roll)_.*": _unitree_control(84.3355, 5.369, n6014=False),
                ".*_(shoulder_yaw|elbow|wrist_roll)_.*": _unitree_control(15.5417, 0.9894, n6014=True),
                "head_.*": _head_control(),
            },
        )
    )
