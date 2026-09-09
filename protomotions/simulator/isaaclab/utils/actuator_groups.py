# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import re
from typing import Dict, Mapping, Tuple

from protomotions.components.pose_lib import ControlInfo, KinematicInfo
from protomotions.robot_configs.base import ControlType


_ACTUATOR_PARAM_FIELDS = (
    ("stiffness", "stiffness"),
    ("damping", "damping"),
    ("armature", "armature"),
    ("effort_limit_sim", "effort_limit"),
    ("velocity_limit_sim", "velocity_limit"),
    ("friction", "friction"),
)


@dataclass(frozen=True)
class SingleActuatorSpec:
    name: str
    joint_names_expr: Tuple[str, ...]
    params: dict
    actuator_model: str | None = None


@dataclass(frozen=True)
class IsaacLabJointNameMap:
    semantic_to_backend: Mapping[str, str]
    backend_to_semantic: Mapping[str, str]


def build_isaaclab_joint_name_map(
    kinematic_info: KinematicInfo,
) -> IsaacLabJointNameMap:
    """Map MJCF joint names to IsaacLab 3 PhysX multi-axis DOF names."""
    semantic_to_backend = {}
    dof_offset = 0
    for body_index in range(kinematic_info.num_bodies):
        num_body_dofs = len(kinematic_info.hinge_axes_map.get(body_index, ()))
        semantic_names = kinematic_info.dof_names[
            dof_offset : dof_offset + num_body_dofs
        ]
        dof_offset += num_body_dofs

        if num_body_dofs <= 1:
            backend_names = semantic_names
        else:
            backend_names = [
                f"{semantic_names[0]}:{axis_index}"
                for axis_index in range(num_body_dofs)
            ]
        semantic_to_backend.update(zip(semantic_names, backend_names))

    if dof_offset != len(kinematic_info.dof_names):
        raise ValueError(
            "Kinematic body DOF counts do not cover all semantic joint names"
        )

    backend_to_semantic = {
        backend: semantic for semantic, backend in semantic_to_backend.items()
    }
    if len(backend_to_semantic) != len(semantic_to_backend):
        raise ValueError("IsaacLab joint-name mapping is not one-to-one")

    return IsaacLabJointNameMap(
        semantic_to_backend=semantic_to_backend,
        backend_to_semantic=backend_to_semantic,
    )


def single_actuator_params_by_joint(
    control_info_by_dof: Dict[str, ControlInfo],
    *,
    zero_stiffness_and_damping: bool = False,
) -> SingleActuatorSpec:
    joint_names_expr = tuple(re.escape(dof_name) for dof_name in control_info_by_dof)
    params = {}
    for config_key, control_key in _ACTUATOR_PARAM_FIELDS:
        if zero_stiffness_and_damping and config_key in {"stiffness", "damping"}:
            params[config_key] = {
                re.escape(dof_name): 0.0 for dof_name in control_info_by_dof
            }
            continue
        values = {
            re.escape(dof_name): getattr(control_info, control_key)
            for dof_name, control_info in control_info_by_dof.items()
            if getattr(control_info, control_key) is not None
        }
        if values:
            params[config_key] = values
    return SingleActuatorSpec(
        name="actuator_group_0",
        joint_names_expr=joint_names_expr,
        params=params,
    )


def resolve_actuator_specs_for_control_type(
    control_info_by_dof: Dict[str, ControlInfo],
    control_type: ControlType,
) -> Tuple[SingleActuatorSpec, ...]:
    if not control_info_by_dof:
        return ()

    # Explicit motor models can have actuator-class-specific parameters, so
    # keep them separate from the normal ideal-PD group.  Group joints whose
    # complete configurations are identical: IsaacLab calls ``compute`` once
    # per actuator group on every physics substep, making one group per joint
    # disproportionately expensive for single-environment inference.
    if any(info.actuator_model for info in control_info_by_dof.values()):
        groups = []
        for dof_name, info in control_info_by_dof.items():
            params = {}
            for config_key, control_key in _ACTUATOR_PARAM_FIELDS:
                value = getattr(info, control_key)
                if value is not None:
                    params[config_key] = value
            if info.actuator_model == "unitree":
                params.update(
                    X1=info.torque_speed_x1,
                    X2=info.torque_speed_x2,
                    Y1=info.torque_same_direction,
                    Y2=info.torque_opposite_direction,
                    Fs=info.friction_static,
                    Fd=info.friction_dynamic,
                    Va=info.friction_activation_velocity,
                    min_delay=0,
                    max_delay=0,
                )
            matching_group = next(
                (
                    group
                    for group in groups
                    if group["actuator_model"] == info.actuator_model
                    and group["params"] == params
                ),
                None,
            )
            if matching_group is None:
                matching_group = {
                    "joint_names_expr": [],
                    "params": params,
                    "actuator_model": info.actuator_model,
                }
                groups.append(matching_group)
            matching_group["joint_names_expr"].append(re.escape(dof_name))

        return tuple(
            SingleActuatorSpec(
                name=f"actuator_group_{index}",
                joint_names_expr=tuple(group["joint_names_expr"]),
                params=group["params"],
                actuator_model=group["actuator_model"],
            )
            for index, group in enumerate(groups)
        )

    zero_stiffness_and_damping = control_type != ControlType.BUILT_IN_PD
    collapsed_spec = single_actuator_params_by_joint(
        control_info_by_dof,
        zero_stiffness_and_damping=zero_stiffness_and_damping,
    )
    if all(
        len(values) == len(control_info_by_dof)
        for values in collapsed_spec.params.values()
    ):
        return (collapsed_spec,)

    specs = []
    for index, (dof_name, control_info) in enumerate(control_info_by_dof.items()):
        params = {}
        for config_key, control_key in _ACTUATOR_PARAM_FIELDS:
            value = getattr(control_info, control_key)
            if zero_stiffness_and_damping and config_key in {
                "stiffness",
                "damping",
            }:
                value = 0.0
            if value is not None:
                params[config_key] = value
        specs.append(
            SingleActuatorSpec(
                name=f"actuator_group_{index}",
                joint_names_expr=(re.escape(dof_name),),
                params=params,
            )
        )
    return tuple(specs)
