# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pure reward kernels."""

import math

import torch

from protomotions.envs.rewards import base, regularization, task, tracking


def _identity_quat(*shape: int) -> torch.Tensor:
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def test_base_reward_primitives_handle_shapes_indices_and_exp_modes():
    x = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[2.0, 2.0], [4.0, 4.0]],
        ]
    )
    ref = torch.zeros_like(x)
    indices = torch.tensor([1])

    assert torch.allclose(
        base.mean_squared_error(x, ref),
        torch.tensor([0.5, 10.0]),
    )
    assert torch.allclose(
        base.mean_squared_error(x, ref, indices=indices),
        torch.tensor([1.0, 16.0]),
    )
    assert torch.allclose(
        base.mean_squared_error(torch.tensor([[1.0, 3.0]]), torch.zeros(1, 2)),
        torch.tensor([5.0]),
    )
    assert torch.allclose(
        base.mean_squared_error(torch.tensor([2.0]), torch.zeros(1)),
        torch.tensor([4.0]),
    )

    assert torch.allclose(
        base.mean_squared_error_exp(x, ref, coefficient=-1.0),
        torch.exp(torch.tensor([-0.5, -10.0])),
    )
    assert torch.allclose(
        base.mean_squared_error_exp(
            x,
            ref,
            coefficient=-1.0,
            indices=indices,
            mean_before_exp=False,
        ),
        torch.exp(torch.tensor([[-1.0], [-16.0]])).mean(dim=-1),
    )
    assert torch.allclose(
        base.mean_squared_error_exp(
            torch.tensor([[1.0, 3.0]]),
            torch.zeros(1, 2),
            coefficient=-2.0,
        ),
        torch.exp(torch.tensor([-10.0])),
    )
    assert torch.allclose(
        base.mean_squared_error_exp(
            torch.tensor([2.0]),
            torch.zeros(1),
            coefficient=-0.5,
        ),
        torch.exp(torch.tensor([-2.0])),
    )

    assert torch.allclose(
        base.norm(torch.tensor([[[3.0, 4.0], [5.0, 12.0]]])),
        torch.tensor([[5.0, 13.0]]),
    )
    assert torch.allclose(
        base.norm(torch.tensor([[[3.0, 4.0], [5.0, 12.0]]]), indices=indices),
        torch.tensor([[13.0]]),
    )
    assert torch.allclose(
        base.delta_norm(torch.tensor([[3.0, 4.0]]), torch.zeros(1, 2)),
        torch.tensor([5.0]),
    )
    assert torch.allclose(
        base.delta_norm(x, ref, indices=indices),
        torch.tensor([[2.0**0.5], [32.0**0.5]]),
    )
    assert torch.allclose(
        base.delta_logmeanexp(
            torch.tensor([[1.0, 3.0]]),
            torch.zeros(1, 2),
            beta=2.0,
        ),
        (torch.logsumexp(torch.tensor([[2.0, 6.0]]), dim=-1) - math.log(2)) / 2.0,
    )
    assert torch.allclose(
        base.delta_logmeanexp(x, ref, indices=indices, beta=2.0),
        torch.tensor([[1.0], [4.0]]),
    )
    assert torch.allclose(
        base.absolute_difference_sum(x, ref),
        torch.tensor([2.0, 12.0]),
    )
    assert torch.allclose(
        base.absolute_difference_sum(x, ref, indices=indices),
        torch.tensor([2.0, 8.0]),
    )
    assert torch.allclose(
        base.absolute_difference_sum(torch.tensor([[1.0, -2.0]]), torch.zeros(1, 2)),
        torch.tensor([3.0]),
    )
    assert torch.allclose(
        base.absolute_difference_sum(torch.tensor([-3.0]), torch.zeros(1)),
        torch.tensor([3.0]),
    )


def test_base_rotation_and_power_primitives():
    quat = _identity_quat(2, 2)
    ref_quat = quat.clone()
    ref_quat[1, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])

    assert torch.allclose(
        base.rotation_error_exp(quat, quat, coefficient=-1.0),
        torch.ones(2),
    )
    assert torch.allclose(
        base.rotation_error_exp(
            quat,
            quat,
            coefficient=-1.0,
            indices=torch.tensor([1]),
            mean_before_exp=False,
        ),
        torch.ones(2),
    )
    assert base.rotation_error_exp(quat, ref_quat, coefficient=-1.0)[1] < 1.0
    assert torch.allclose(base.rotation_error(quat, quat), torch.zeros(2))
    assert torch.allclose(
        base.rotation_error(quat, quat, indices=torch.tensor([1])),
        torch.zeros(2),
    )

    dof_forces = torch.tensor([[2.0, -3.0], [4.0, 5.0]])
    dof_vel = torch.tensor([[10.0, -2.0], [0.5, -1.0]])
    assert torch.allclose(
        base.power_consumption_sum(dof_forces, dof_vel),
        torch.tensor([26.0, 7.0]),
    )
    assert torch.allclose(
        base.power_consumption_sum(
            dof_forces,
            dof_vel,
            indices=torch.tensor([1]),
        ),
        torch.tensor([6.0, 5.0]),
    )
    assert torch.allclose(
        base.power_consumption_sum(dof_forces, dof_vel, use_torque_squared=True),
        torch.tensor([13.0, 41.0]),
    )
    assert torch.allclose(
        base.power_consumption_exp(dof_forces, dof_vel, coefficient=-0.1),
        torch.exp(torch.tensor([-2.6, -0.7])),
    )
    assert torch.allclose(
        base.power_consumption_exp(
            dof_forces,
            dof_vel,
            coefficient=-0.1,
            use_torque_squared=True,
            indices=torch.tensor([1]),
        ),
        torch.exp(torch.tensor([-0.9, -2.5])),
    )
    assert torch.allclose(
        base.velocity_squared_sum(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])),
        torch.tensor([30.0]),
    )
    assert torch.allclose(
        base.velocity_squared_sum(torch.tensor([[1.0, 2.0, 3.0]])),
        torch.tensor([14.0]),
    )
    assert torch.allclose(
        base.velocity_squared_sum(
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            indices=torch.tensor([1]),
        ),
        torch.tensor([25.0]),
    )


def test_regularization_rewards_and_helpers():
    current_action = torch.tensor([[1.0, 3.0], [4.0, 4.0]])
    previous_action = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    dof_pos = torch.tensor([[-2.0, 0.0, 2.0], [0.0, 2.0, 4.0]])
    lower = torch.tensor([-1.0, -1.0, -1.0])
    upper = torch.tensor([1.0, 1.0, 3.0])

    assert torch.allclose(
        regularization.compute_action_smoothness(current_action, previous_action),
        torch.tensor([2.0, 5.0]),
    )
    assert torch.allclose(
        regularization.compute_action_smoothness_logmeanexp(
            current_action,
            previous_action,
            beta=2.0,
        ),
        base.delta_logmeanexp(current_action, previous_action, beta=2.0),
    )
    assert torch.allclose(
        regularization.compute_pow_rew(
            torch.tensor([[2.0, -3.0]]),
            torch.tensor([[10.0, -2.0]]),
        ),
        torch.tensor([26.0]),
    )
    assert torch.allclose(
        regularization.compute_pow_rew(
            torch.tensor([[2.0, -3.0]]),
            torch.tensor([[10.0, -2.0]]),
            use_torque_squared=True,
        ),
        torch.tensor([13.0]),
    )
    assert torch.allclose(
        regularization.compute_soft_pos_limit_rew(dof_pos, lower, upper),
        torch.tensor([1.0, 2.0]),
    )
    assert torch.allclose(
        regularization.joint_limit_violation(
            dof_pos,
            lower,
            upper,
            indices=torch.tensor([0, 2]),
        ),
        torch.tensor([1.0, 1.0]),
    )

    sim_contacts = torch.tensor([[True, False, True], [False, True, False]])
    ref_contacts = torch.tensor([[True, True, False], [True, True, False]])
    assert torch.allclose(
        regularization.compute_contact_match_rew(
            sim_contacts,
            ref_contacts,
            contact_body_ids=torch.tensor([1, 2]),
        ),
        torch.tensor([2.0, 0.0]),
    )
    assert torch.allclose(
        regularization.contact_mismatch_sum(
            sim_contacts,
            ref_contacts,
            indices=torch.tensor([0, 1]),
        ),
        torch.tensor([1.0, 1.0]),
    )
    assert torch.allclose(
        regularization.compute_contact_force_change_rew(
            torch.tensor([[10.0, 50.0], [100.0, 0.0]]),
            torch.tensor([[0.0, 0.0], [20.0, 40.0]]),
            threshold=30.0,
        ),
        torch.tensor([20.0, 60.0]),
    )
    assert torch.allclose(
        regularization.impact_force_penalty(
            torch.tensor([[10.0, 50.0], [100.0, 0.0]]),
            torch.tensor([[0.0, 0.0], [20.0, 40.0]]),
            indices=torch.tensor([0]),
            threshold=30.0,
        ),
        torch.tensor([0.0, 50.0]),
    )


def test_contact_slip_rew_gates_on_actual_contact_not_reference():
    # bodies: [left_foot, right_foot, hand]. force_threshold=1.0.
    contact_force_magnitudes = torch.tensor(
        [
            [5.0, 0.0, 5.0],  # left planted, right airborne, hand irrelevant
            [0.0, 0.0, 0.0],  # nothing planted
            [5.0, 5.0, 0.0],  # both feet planted
        ]
    )
    # xyz velocity per body; z (vertical) must be ignored by the kernel.
    current_rigid_body_vel = torch.tensor(
        [
            [[0.5, 0.0, 9.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.3, 0.4, 0.0], [0.0, 0.0, 5.0], [3.0, 0.0, 0.0]],
        ]
    )
    body_indices = [0, 1]
    result = regularization.compute_contact_slip_rew(
        contact_force_magnitudes,
        current_rigid_body_vel,
        body_indices=body_indices,
        force_threshold=1.0,
    )
    # row 0: only left foot planted, horizontal speed 0.5, vertical (9.0) ignored.
    assert torch.allclose(result[0], torch.tensor(0.5))
    # row 1: neither foot planted -> zero penalty regardless of how fast they move.
    assert torch.allclose(result[1], torch.tensor(0.0))
    # row 2: both planted; left horizontal speed = hypot(0.3,0.4) = 0.5, right
    # has no xy velocity (only a large ignored z component) -> 0.5 + 0.0.
    assert torch.allclose(result[2], torch.tensor(0.5))


def test_task_rewards_cover_direction_path_target_and_object_terms():
    root_rot = _identity_quat(3)
    heading_reward = task.compute_heading_velocity_rew(
        root_pos=torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        prev_root_pos=torch.zeros(3, 3),
        root_rot=root_rot,
        tar_dir=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        tar_speed=torch.tensor([1.0, 1.0, 1.0]),
        tar_face_dir=torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        dt=1.0,
    )
    assert torch.allclose(heading_reward[0], torch.tensor(1.0))
    # index 1 moves backward under a non-stop speed command: dir_reward is
    # zeroed by speed_mask, and facing_reward must be zeroed too -- otherwise
    # standing still (or moving the wrong way) while merely facing tar_face_dir
    # would earn a free 0.3 without ever satisfying the speed command.
    assert torch.allclose(heading_reward[1], torch.tensor(0.0))
    assert heading_reward[2] < 0.7

    stop_reward = task.compute_heading_velocity_rew(
        root_pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        prev_root_pos=torch.zeros(2, 3),
        root_rot=_identity_quat(2),
        tar_dir=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        tar_speed=torch.zeros(2),
        tar_face_dir=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        dt=1.0,
    )
    assert torch.allclose(stop_reward[0], torch.tensor(1.0))
    assert stop_reward[1] < stop_reward[0]

    velocity_reward = task.compute_track_lin_vel_xy_yaw_frame_exp(
        root_vel=torch.tensor([[1.0, -0.5, 0.0], [0.0, 0.0, 0.0]]),
        root_rot=_identity_quat(2),
        tar_local_vel=torch.tensor([[1.0, -0.5], [1.0, 0.0]]),
        std=0.5,
    )
    assert torch.allclose(velocity_reward[0], torch.tensor(1.0))
    assert velocity_reward[1] < 0.02

    yaw_reward = task.compute_track_ang_vel_z_world_exp(
        root_ang_vel=torch.tensor([[0.2, -0.1, 0.8], [0.0, 0.0, -0.8]]),
        tar_yaw_rate=torch.tensor([0.8, 0.8]),
        std=0.5,
    )
    assert torch.allclose(yaw_reward[0], torch.tensor(1.0))
    assert yaw_reward[1] < yaw_reward[0]
    assert torch.allclose(
        task.compute_lin_vel_z_l2(torch.tensor([[0.0, 0.0, -0.5]])),
        torch.tensor([0.25]),
    )
    assert torch.allclose(
        task.compute_ang_vel_xy_l2(torch.tensor([[0.2, -0.3, 1.0]])),
        torch.tensor([0.13]),
    )
    assert torch.allclose(
        task.compute_body_orientation_l2(_identity_quat(1)), torch.zeros(1)
    )

    assert torch.allclose(
        task.compute_path_following_rew(
            head_pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
            tar_pos=torch.tensor([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]),
            height_conditioned=True,
            pos_err_scale=2.0,
            height_err_scale=1.0,
        ),
        torch.tensor([math.exp(-1.0), math.exp(-2.0)]),
    )
    assert torch.allclose(
        task.compute_path_following_rew(
            head_pos=torch.tensor([[0.0, 0.0, 0.0]]),
            tar_pos=torch.tensor([[0.0, 0.0, 10.0]]),
            height_conditioned=False,
        ),
        torch.ones(1),
    )

    target_reward = task.compute_target_rew(
        root_pos=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        tar_pos=torch.tensor([[0.1, 0.0, 0.0], [12.0, 0.0, 0.0]]),
        tar_proximity_threshold=0.5,
        pos_err_scale=0.5,
    )
    assert torch.allclose(target_reward, torch.tensor([1.0, math.exp(-1.0)]))


def test_tracking_rewards_cover_standard_and_beyond_mimic_variants():
    current_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    ref_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [2.0, 0.0, 0.0]],
        ]
    )
    current_rot = _identity_quat(2, 2)
    ref_rot = _identity_quat(2, 2)
    ref_rot[1, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    current_vel = torch.ones(2, 2, 3)
    current_anchor_pos = current_pos[:, 0, :]
    current_anchor_rot = current_rot[:, 0, :]

    assert torch.allclose(tracking.compute_gt_rew(current_pos, current_pos), torch.ones(2))
    assert torch.allclose(tracking.compute_gr_rew(current_rot, current_rot), torch.ones(2))
    assert torch.allclose(tracking.compute_gv_rew(current_vel, current_vel), torch.ones(2))
    assert torch.allclose(tracking.compute_gav_rew(current_vel, current_vel), torch.ones(2))
    assert torch.allclose(
        tracking.compute_rh_rew(current_pos[:, 0, 2], current_pos),
        torch.ones(2),
    )
    assert torch.allclose(
        tracking.compute_global_position_error_exp(current_pos, current_pos, sigma=0.5),
        torch.ones(2),
    )
    assert tracking.compute_global_position_error_exp(
        current_pos,
        ref_pos,
        sigma=1.0,
        indices=torch.tensor([1]),
    )[1] < 1.0
    assert torch.allclose(
        tracking.compute_global_anchor_pos_rew(
            current_anchor_pos,
            ref_pos,
            anchor_idx=0,
            sigma=1.0,
        ),
        torch.tensor([1.0, math.exp(-1.0)]),
    )
    assert torch.allclose(
        tracking.compute_global_orientation_error_exp(current_rot, current_rot, sigma=1.0),
        torch.ones(2),
    )
    assert tracking.compute_global_anchor_ori_rew(
        current_anchor_rot,
        ref_rot,
        anchor_idx=0,
        sigma=1.0,
    )[1] < 1.0

    assert torch.allclose(
        tracking.compute_relative_body_pos_rew(
            current_pos,
            current_pos,
            current_anchor_rot,
            current_rot,
            current_anchor_pos,
            anchor_idx=0,
            sigma=1.0,
        ),
        torch.ones(2),
    )
    assert tracking.compute_relative_body_pos_rew(
        current_pos,
        ref_pos,
        current_anchor_rot,
        ref_rot,
        current_anchor_pos,
        anchor_idx=0,
        sigma=1.0,
        body_indices=torch.tensor([1]),
    )[1] < 1.0
    assert torch.allclose(
        tracking.compute_relative_body_ori_rew(
            current_rot,
            current_rot,
            current_anchor_rot,
            anchor_idx=0,
            sigma=1.0,
        ),
        torch.ones(2),
    )
    assert tracking.compute_relative_body_ori_rew(
        current_rot,
        ref_rot,
        current_anchor_rot,
        anchor_idx=0,
        sigma=1.0,
        body_indices=torch.tensor([0]),
    )[1] < 1.0
    assert torch.allclose(
        tracking.compute_global_body_lin_vel_rew(current_vel, current_vel),
        torch.ones(2),
    )
    assert torch.allclose(
        tracking.compute_global_body_ang_vel_rew(current_vel, current_vel),
        torch.ones(2),
    )
    assert torch.allclose(
        tracking.compute_gt_rel_rew(
            current_pos,
            current_pos,
            current_anchor_rot,
            current_rot,
            anchor_idx=0,
            body_indices=[0, 1],
        ),
        torch.ones(2),
    )
    assert tracking.compute_gt_rel_rew(
        current_pos,
        ref_pos,
        current_anchor_rot,
        ref_rot,
        anchor_idx=0,
    )[1] < 1.0
    assert torch.allclose(
        tracking.compute_anchor_xy_rew(
            current_anchor_pos,
            current_pos,
            anchor_idx=0,
        ),
        torch.ones(2),
    )


def test_ref_stance_foot_lift_rew_prices_lift_only_where_reference_is_planted():
    # bodies: [left_foot, right_foot, hand]. Only the feet are scored.
    # Rows exercise: planted+lifted, planted+down, airborne reference, both feet.
    current_rigid_body_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.075], [0.0, 0.0, 0.055], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.055], [0.0, 0.0, 0.055], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.300], [0.0, 0.0, 0.055], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.085], [0.0, 0.0, 0.095], [0.0, 0.0, 1.0]],
        ]
    )
    ref_rigid_body_pos = torch.full((4, 3, 3), 0.0)
    ref_rigid_body_pos[..., 2] = 0.055
    ref_contacts = torch.tensor(
        [
            [True, True, True],    # both planted; hand must be ignored
            [True, True, False],
            [False, True, False],  # left foot airborne in the reference
            [True, True, False],
        ]
    )
    result = regularization.compute_ref_stance_foot_lift_rew(
        current_rigid_body_pos,
        ref_rigid_body_pos,
        ref_contacts,
        body_indices=[0, 1],
        deadband=0.005,
    )
    # row 0: left is 20 mm above reference, 5 mm forgiven -> 0.015; right flat.
    assert torch.allclose(result[0], torch.tensor(0.015))
    # row 1: both feet exactly on the reference -> no penalty.
    assert torch.allclose(result[1], torch.tensor(0.0))
    # row 2: left is 245 mm up but the reference has it airborne -> not priced.
    assert torch.allclose(result[2], torch.tensor(0.0))
    # row 3: both planted and lifted 30/40 mm -> (0.030-0.005)+(0.040-0.005).
    assert torch.allclose(result[3], torch.tensor(0.060))


def test_ref_stance_foot_lift_rew_never_rewards_pressing_into_the_ground():
    # A foot below the reference height must score zero, not a negative
    # penalty that would pay the policy for pushing through the floor.
    current_rigid_body_pos = torch.zeros(1, 2, 3)
    current_rigid_body_pos[0, 0, 2] = 0.010   # 45 mm *below* the reference
    current_rigid_body_pos[0, 1, 2] = 0.055
    ref_rigid_body_pos = torch.zeros(1, 2, 3)
    ref_rigid_body_pos[..., 2] = 0.055
    ref_contacts = torch.ones(1, 2, dtype=torch.bool)
    result = regularization.compute_ref_stance_foot_lift_rew(
        current_rigid_body_pos,
        ref_rigid_body_pos,
        ref_contacts,
        body_indices=[0, 1],
    )
    assert torch.allclose(result, torch.zeros(1))


def test_ref_stance_foot_lift_rew_is_continuous_unlike_binary_contact_match():
    # The point of this kernel over compute_contact_match_rew: the penalty
    # grows with how far the foot has strayed, so the gradient says how far
    # to come back down. Sweep one foot upward and require strict monotonic
    # increase past the deadband.
    heights = torch.linspace(0.055, 0.155, 11)
    ref_rigid_body_pos = torch.zeros(11, 2, 3)
    ref_rigid_body_pos[..., 2] = 0.055
    current_rigid_body_pos = torch.zeros(11, 2, 3)
    current_rigid_body_pos[:, 0, 2] = heights
    current_rigid_body_pos[:, 1, 2] = 0.055
    ref_contacts = torch.ones(11, 2, dtype=torch.bool)
    result = regularization.compute_ref_stance_foot_lift_rew(
        current_rigid_body_pos,
        ref_rigid_body_pos,
        ref_contacts,
        body_indices=[0, 1],
        deadband=0.005,
    )
    past_deadband = result[1:]
    assert torch.all(past_deadband[1:] > past_deadband[:-1])
    # A 2 cm lift -- the median measured on r1_locomotion_10 -- must register.
    assert result[2] > 0.0


def test_ref_touchdown_impact_rew_fires_only_on_the_touchdown_step():
    # bodies: [left_foot, right_foot, hand]. force_threshold=1.0.
    # Row 0: left touching down (in contact, was airborne) -> scored.
    # Row 1: left in contact but air_time 0 (already stood there) -> not scored.
    # Row 2: left airborne despite air_time>0 (no contact yet) -> not scored.
    contact_force_magnitudes = torch.tensor(
        [[50.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    contact_air_time = torch.tensor(
        [[0.30, 0.0, 0.0], [0.00, 0.0, 0.0], [0.30, 0.0, 0.0]]
    )
    prev_rigid_body_vel = torch.zeros(3, 3, 3)
    prev_rigid_body_vel[:, 0, 2] = -0.85          # landing at 0.85 m/s
    ref_rigid_body_vel = torch.zeros(3, 3, 3)
    ref_rigid_body_vel[:, 0, 2] = -0.20           # reference lands at 0.20
    result = regularization.compute_ref_touchdown_impact_rew(
        contact_force_magnitudes, contact_air_time, prev_rigid_body_vel,
        ref_rigid_body_vel, body_indices=[0, 1], force_threshold=1.0, deadband=0.05,
    )
    # row 0: |0.85| - |0.20| - 0.05 = 0.60
    assert torch.allclose(result[0], torch.tensor(0.60))
    assert torch.allclose(result[1], torch.tensor(0.0))
    assert torch.allclose(result[2], torch.tensor(0.0))


def test_ref_touchdown_impact_rew_allows_a_reference_that_lands_hard():
    # backwards clips land at ~0.44 m/s in the reference; matching that must be
    # free, which an absolute threshold could not express.
    contact_force_magnitudes = torch.tensor([[50.0, 0.0], [50.0, 0.0]])
    contact_air_time = torch.tensor([[0.3, 0.0], [0.3, 0.0]])
    prev_rigid_body_vel = torch.zeros(2, 2, 3)
    ref_rigid_body_vel = torch.zeros(2, 2, 3)
    prev_rigid_body_vel[0, 0, 2] = -0.44; ref_rigid_body_vel[0, 0, 2] = -0.44
    # a soft-landing reference (walk, 0.03) held to its own standard
    prev_rigid_body_vel[1, 0, 2] = -0.44; ref_rigid_body_vel[1, 0, 2] = -0.03
    result = regularization.compute_ref_touchdown_impact_rew(
        contact_force_magnitudes, contact_air_time, prev_rigid_body_vel,
        ref_rigid_body_vel, body_indices=[0, 1],
    )
    assert torch.allclose(result[0], torch.tensor(0.0))
    assert torch.allclose(result[1], torch.tensor(0.44 - 0.03 - 0.05))


def test_ref_touchdown_impact_rew_never_rewards_landing_softer_than_reference():
    contact_force_magnitudes = torch.tensor([[50.0, 0.0]])
    contact_air_time = torch.tensor([[0.3, 0.0]])
    prev_rigid_body_vel = torch.zeros(1, 2, 3); prev_rigid_body_vel[0, 0, 2] = -0.05
    ref_rigid_body_vel = torch.zeros(1, 2, 3); ref_rigid_body_vel[0, 0, 2] = -0.50
    result = regularization.compute_ref_touchdown_impact_rew(
        contact_force_magnitudes, contact_air_time, prev_rigid_body_vel,
        ref_rigid_body_vel, body_indices=[0, 1],
    )
    assert torch.allclose(result, torch.zeros(1))


def test_ref_touchdown_impact_rew_is_continuous_in_excess_speed():
    # The point over a binary impact flag: the penalty grows with how much too
    # hard the landing is, so the gradient says how much to soften.
    n = 9
    speeds = torch.linspace(0.20, 1.00, n)
    contact_force_magnitudes = torch.full((n, 2), 0.0); contact_force_magnitudes[:, 0] = 50.0
    contact_air_time = torch.zeros(n, 2); contact_air_time[:, 0] = 0.3
    prev_rigid_body_vel = torch.zeros(n, 2, 3); prev_rigid_body_vel[:, 0, 2] = -speeds
    ref_rigid_body_vel = torch.zeros(n, 2, 3); ref_rigid_body_vel[:, 0, 2] = -0.20
    result = regularization.compute_ref_touchdown_impact_rew(
        contact_force_magnitudes, contact_air_time, prev_rigid_body_vel,
        ref_rigid_body_vel, body_indices=[0, 1], deadband=0.05,
    )
    past = result[2:]
    assert torch.all(past[1:] > past[:-1])
    # the measured 0.586 vs 0.188 gap must register
    assert result[-1] > 0.7


def test_action_oscillation_is_zero_for_a_constant_ramp():
    """The distinction from action_smoothness: a fast ramp must cost nothing."""
    a2 = torch.zeros(1, 4)
    a1 = torch.full((1, 4), 0.08)
    a0 = torch.full((1, 4), 0.16)          # +0.08 every step, no reversal
    result = regularization.compute_action_oscillation(a0, a1, a2)
    assert torch.allclose(result, torch.zeros(1), atol=1e-6)
    # ...while action_smoothness charges the same ramp its full first difference
    assert regularization.compute_action_smoothness(a0, a1).item() > 0.1


def test_action_oscillation_is_maximal_for_alternation():
    # +0.08, -0.08, +0.08 -- identical first differences to the ramp above,
    # which is exactly the case action_smoothness cannot distinguish.
    a2 = torch.zeros(1, 4)
    a1 = torch.full((1, 4), 0.08)
    a0 = torch.zeros(1, 4)
    osc = regularization.compute_action_oscillation(a0, a1, a2)
    # second difference is -2*0.08 per element -> norm = 0.16*sqrt(4)
    assert torch.allclose(osc, torch.tensor([0.32]), atol=1e-6)
    ramp_first_diff = regularization.compute_action_smoothness(
        torch.full((1, 4), 0.16), a1
    )
    alt_first_diff = regularization.compute_action_smoothness(a0, a1)
    assert torch.allclose(ramp_first_diff, alt_first_diff)


def test_action_oscillation_grows_with_reversal_amplitude():
    n = 6
    amp = torch.linspace(0.01, 0.20, n)
    a2 = torch.zeros(n, 3)
    a1 = amp[:, None].repeat(1, 3)
    a0 = torch.zeros(n, 3)
    result = regularization.compute_action_oscillation(a0, a1, a2)
    assert torch.all(result[1:] > result[:-1])
    assert torch.allclose(result[0], torch.tensor(2 * 0.01 * 3 ** 0.5), atol=1e-5)


def test_action_oscillation_ignores_a_steady_offset():
    """A held target, however far from zero, is not oscillation."""
    held = torch.full((2, 5), 0.42)
    result = regularization.compute_action_oscillation(held, held, held)
    assert torch.allclose(result, torch.zeros(2), atol=1e-6)


def test_ref_swing_foot_low_rew_prices_only_a_foot_the_reference_has_airborne():
    # bodies: [left_foot, right_foot, hand].
    current_rigid_body_pos = torch.zeros(3, 3, 3)
    ref_rigid_body_pos = torch.zeros(3, 3, 3)
    # row 0: reference swings the left foot to 0.20, physics only reaches 0.12
    ref_rigid_body_pos[0, 0, 2] = 0.20; current_rigid_body_pos[0, 0, 2] = 0.12
    # row 1: same shortfall, but the reference says that foot is PLANTED
    ref_rigid_body_pos[1, 0, 2] = 0.20; current_rigid_body_pos[1, 0, 2] = 0.12
    # row 2: airborne and lifted HIGHER than the reference -> never penalised
    ref_rigid_body_pos[2, 0, 2] = 0.20; current_rigid_body_pos[2, 0, 2] = 0.31
    ref_contacts = torch.zeros(3, 3, dtype=torch.bool)
    ref_contacts[1, 0] = True                      # row 1 planted
    result = regularization.compute_ref_swing_foot_low_rew(
        current_rigid_body_pos, ref_rigid_body_pos, ref_contacts,
        body_indices=[0, 1], deadband=0.01,
    )
    assert torch.allclose(result[0], torch.tensor(0.07))   # 0.20-0.12-0.01
    assert torch.allclose(result[1], torch.tensor(0.0))    # stance: not this term's job
    assert torch.allclose(result[2], torch.tensor(0.0))    # higher than reference is free


def test_ref_swing_foot_low_rew_is_the_mirror_of_the_stance_term():
    """The pair must not both fire on the same foot in the same frame."""
    current = torch.zeros(2, 2, 3); ref = torch.zeros(2, 2, 3)
    ref[:, 0, 2] = 0.20
    current[0, 0, 2] = 0.10        # too low
    current[1, 0, 2] = 0.30        # too high
    swing = torch.zeros(2, 2, dtype=torch.bool)          # reference: airborne
    stance = torch.ones(2, 2, dtype=torch.bool)          # reference: planted
    low_in_swing = regularization.compute_ref_swing_foot_low_rew(
        current, ref, swing, body_indices=[0, 1])
    lift_in_swing = regularization.compute_ref_stance_foot_lift_rew(
        current, ref, swing, body_indices=[0, 1])
    lift_in_stance = regularization.compute_ref_stance_foot_lift_rew(
        current, ref, stance, body_indices=[0, 1])
    low_in_stance = regularization.compute_ref_swing_foot_low_rew(
        current, ref, stance, body_indices=[0, 1])
    assert low_in_swing[0] > 0 and lift_in_swing[0] == 0     # low during swing
    assert lift_in_stance[1] > 0 and low_in_stance[1] == 0   # high during stance
    assert lift_in_swing.sum() == 0 and low_in_stance.sum() == 0


def test_ref_swing_foot_low_rew_grows_with_the_shortfall():
    n = 7
    heights = torch.linspace(0.20, 0.05, n)
    current = torch.zeros(n, 2, 3); current[:, 0, 2] = heights
    ref = torch.zeros(n, 2, 3); ref[:, 0, 2] = 0.20
    ref_contacts = torch.zeros(n, 2, dtype=torch.bool)
    r = regularization.compute_ref_swing_foot_low_rew(
        current, ref, ref_contacts, body_indices=[0, 1], deadband=0.01)
    assert torch.all(r[2:] > r[1:-1])
    # the 43 mm mean deficit measured on r1_locomotion_16 must register
    assert r[1] > 0.0
