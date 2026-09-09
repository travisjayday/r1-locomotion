# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regularization reward compute kernels.

Pure tensor functions (kernels) for computing regularization rewards.
Use MdpComponent in experiment configs to bind kernels to context paths:

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.regularization import compute_action_smoothness
    
    reward_components = {
        "action_smoothness": MdpComponent(
            compute_func=compute_action_smoothness,
            dynamic_vars={
                "current_processed_action": EnvContext.current_processed_action,
                "previous_processed_action": EnvContext.previous_processed_action,
            },
        ),
    }

Includes:
- Action smoothness (L2 and Log-Mean-Exp variants)
- Power consumption
- Joint limit violations
- Contact matching
- Contact force change penalties
"""

import torch
from torch import Tensor
from typing import Optional

from protomotions.envs.rewards.base import power_consumption_sum, delta_norm, delta_logmeanexp
from protomotions.utils import rotations


# =============================================================================
# Regularization Reward Kernels
# =============================================================================

def compute_action_smoothness(
    current_processed_action: Tensor,
    previous_processed_action: Tensor,
) -> Tensor:
    """Action smoothness reward (L2 norm of processed action changes).
    
    Requires num_state_history_steps >= 1 in env config.
    
    Args:
        current_processed_action: Current processed action [num_envs, action_dim].
        previous_processed_action: Previous processed action [num_envs, action_dim].
    
    Returns:
        Smoothness penalty tensor [num_envs].
    """
    return delta_norm(current_processed_action, previous_processed_action)


def compute_action_oscillation(
    current_processed_action: Tensor,
    previous_processed_action: Tensor,
    prev2_processed_action: Tensor,
) -> Tensor:
    """Penalize reversals in the commanded PD target (second difference).

    compute_action_smoothness above is the L2 norm of the *first* difference,
    which measures how fast the target moves but is blind to whether it
    reverses: a target stepping +0.08, -0.08, +0.08 rad scores exactly the same
    as one stepping +0.08, +0.08, +0.08. Measured across r1_locomotion_13 and
    _14, its raw value correlates with eval jerk at -0.03 -- in run 13 the
    action deltas shrank (1.48 -> 1.34) while jerk grew (1062 -> 1326). It
    cannot see the failure.

    The failure it cannot see: reconstructing the commanded target from the
    rollout (target = q + (tau + kd*qd)/kp, with tau from MuJoCo inverse
    dynamics) shows it reversing 30 times a second against joints that reverse
    19.5 times -- the mechanism is low-pass filtering a near-Nyquist command.
    Commanded steps reach 0.084 rad on the hips while the joint moves 0.015.
    This is speed-independent, so it is invisible in sprint and dominates the
    slow clips, where reference foot speed is 0.25 m/s: foot acceleration
    reverses 22-26 times per second against the reference's 7.

    The second difference separates the two cases exactly -- zero for a ramp,
    maximal for alternation -- so it prices oscillation without taxing fast but
    consistent motion, which matters because sprint already tracks at only 88.8%
    of reference speed and must not be slowed further.

    Args:
        current_processed_action: Processed action this step [num_envs, action_dim].
        previous_processed_action: Processed action one step back.
        prev2_processed_action: Processed action two steps back.

    Returns:
        L2 norm of the second difference [num_envs].
    """
    second_difference = (
        current_processed_action
        - 2.0 * previous_processed_action
        + prev2_processed_action
    )
    return torch.norm(second_difference, dim=-1)


def compute_action_smoothness_logmeanexp(
    current_processed_action: Tensor,
    previous_processed_action: Tensor,
    beta: float = 3.0,
) -> Tensor:
    """Action smoothness using Log-Mean-Exp (soft L_infinity).
    
    Requires num_state_history_steps >= 1 in env config.
    
    Args:
        current_processed_action: Current processed action [num_envs, action_dim].
        previous_processed_action: Previous processed action [num_envs, action_dim].
        beta: Temperature parameter. Lower = more like mean, higher = more like max.
    
    Returns:
        Smoothness penalty tensor [num_envs].
    """
    return delta_logmeanexp(
        current_processed_action,
        previous_processed_action,
        beta=beta,
    )


def compute_pow_rew(
    dof_forces: Tensor,
    dof_vel: Tensor,
    use_torque_squared: bool = False,
) -> Tensor:
    """Power consumption reward.
    
    Args:
        dof_forces: Joint forces/torques [num_envs, num_dofs].
        dof_vel: Joint velocities [num_envs, num_dofs].
        use_torque_squared: Whether to use torque squared instead of absolute.
    
    Returns:
        Power consumption tensor [num_envs].
    """
    return power_consumption_sum(dof_forces, dof_vel, use_torque_squared)


def compute_soft_pos_limit_rew(
    dof_pos: Tensor,
    dof_limits_lower: Tensor,
    dof_limits_upper: Tensor,
    max_violation: float = None,
) -> Tensor:
    """Soft joint position limit penalty.

    Penalizes when joints approach or exceed limits.

    The penalty is linear in the violation and summed over every DOF, so it has
    no floor. That is harmless while the simulator is healthy and unbounded when
    it is not: joint positions that blow up produce an arbitrarily large negative
    reward, which then sets the reward-normalizer scale for many epochs
    afterwards. A raw task reward of -7.9e7 was traced to this term.

    Args:
        dof_pos: Joint positions [num_envs, num_dofs].
        dof_limits_lower: Lower joint limits [num_dofs].
        dof_limits_upper: Upper joint limits [num_dofs].
        max_violation: If set, clamp the summed violation (in radians) before it
            is weighted, which puts a floor on the penalty. Past the clamp the
            state is broken rather than merely bad, and a larger number carries
            no useful gradient. ``None`` (default) keeps the unbounded behaviour.

    Returns:
        Penalty tensor [num_envs].
    """
    out_of_limits = -(dof_pos - dof_limits_lower).clip(max=0.0)
    out_of_limits += (dof_pos - dof_limits_upper).clip(min=0.0)
    total = torch.sum(out_of_limits, dim=1)
    if max_violation is not None:
        total = total.clip(max=max_violation)
    return total


def compute_contact_match_rew(
    sim_contacts: Tensor,
    ref_contacts: Tensor,
    contact_body_ids: Tensor,
) -> Tensor:
    """Contact matching reward using foot contact bodies.
    
    Penalizes mismatch between simulated and reference foot contacts.
    Uses contact_body_ids (typically foot bodies).
    
    Args:
        sim_contacts: Simulated contact flags [num_envs, num_bodies].
        ref_contacts: Reference contact flags [num_envs, num_bodies].
        contact_body_ids: Indices of bodies to track contacts for [num_contact_bodies].
    
    Returns:
        Contact mismatch penalty tensor [num_envs].
    """
    sim_contacts_subset = sim_contacts[:, contact_body_ids]
    ref_contacts_subset = ref_contacts[:, contact_body_ids]
    return torch.abs(sim_contacts_subset.float() - ref_contacts_subset.float()).sum(dim=1)


def compute_undesired_contacts_rew(
    contact_force_magnitudes: Tensor,
    body_indices: Tensor,
    force_threshold: float = 1.0,
) -> Tensor:
    """Count PMG contacts on bodies other than allowed end effectors."""
    return (
        contact_force_magnitudes[:, body_indices] > force_threshold
    ).float().sum(dim=-1)


def compute_contact_force_change_rew(
    current_contact_force_magnitudes: Tensor,
    prev_contact_force_magnitudes: Tensor,
    threshold: float = 30.0,
) -> Tensor:
    """Contact force change penalty.
    
    Penalizes sudden contact force changes above a threshold (impact penalty).
    
    Args:
        current_contact_force_magnitudes: Current contact forces [num_envs, num_bodies].
        prev_contact_force_magnitudes: Previous contact forces [num_envs, num_bodies].
        threshold: Force change threshold below which changes are ignored (default: 30.0).
    
    Returns:
        Total force change above threshold [num_envs].
    """
    force_changes = torch.abs(current_contact_force_magnitudes - prev_contact_force_magnitudes)
    force_changes = torch.clamp(force_changes - threshold, min=0)
    return force_changes.sum(dim=-1)


def compute_contact_slip_rew(
    contact_force_magnitudes: Tensor,
    current_rigid_body_vel: Tensor,
    body_indices: list,
    force_threshold: float = 1.0,
) -> Tensor:
    """Penalize horizontal velocity of feet the simulator reports as planted.

    Gated on the actual simulated contact force, not the reference motion's
    contact labels -- a foot the physics engine says is bearing load should
    not be moving sideways regardless of what the (possibly imperfect)
    retargeted reference claims about contact at this frame. Vertical
    velocity is excluded deliberately: it's dominated by the settle/lift-off
    transients right at the force threshold crossing, not sliding.

    Args:
        contact_force_magnitudes: Current per-body contact force [num_envs, num_bodies].
        current_rigid_body_vel: Current world-frame body linear velocity [num_envs, num_bodies, 3].
        body_indices: Foot body indices to check.
        force_threshold: Force above which a body counts as planted.

    Returns:
        Summed in-contact horizontal speed over the selected bodies [num_envs].
    """
    in_contact = contact_force_magnitudes[:, body_indices] > force_threshold
    horizontal_speed = current_rigid_body_vel[:, body_indices, :2].norm(dim=-1)
    return (horizontal_speed * in_contact.float()).sum(dim=-1)


def compute_ref_stance_foot_lift_rew(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    ref_contacts: Tensor,
    body_indices: list,
    deadband: float = 0.005,
) -> Tensor:
    """Penalize lifting a foot the reference holds planted, in proportion to lift.

    Fills the gap that lets slow motions degenerate into foot tapping. In
    walking, backing up, spinning and turning in place the reference spends
    long stretches in double support, but nothing in the tracking reward
    prices a brief lift: measured on r1_locomotion_10 at epoch 6400, the
    policy raised a reference-planted foot a median 18.8 mm (p90 34 mm) for
    ~40 ms at a time, while root-relative foot tracking error in those same
    categories was 45-67 mm. The lift is comfortably inside the tracking
    tolerance, so the mimic terms are indifferent to it, and contact_slip
    only bites *while in contact* -- lifting is precisely the action that
    escapes it. The share of reference-stance frames lost this way was
    rising with training (backwards 9.1% -> 15.5% between epochs 2000 and
    6400) while fast motions, which have no double-support phase to fail,
    improved (sprint 3.3% -> 2.2%).

    Deliberately continuous, unlike compute_contact_match_rew, which prices
    the same failure as a binary state mismatch. r1_locomotion_9 ran that at
    -0.2 and it did not work: a foot already near the threshold minimizes a
    binary penalty most cheaply by tapping, so contact chatter roughly
    doubled while eval success fell 0.56 -> 0.35. Penalizing height above the
    reference gives a gradient proportional to how far the foot has strayed,
    telling the policy how far to come back down rather than only that it is
    wrong.

    Height is measured against the reference foot rather than a world ground
    plane, so no sole-offset or terrain-height assumption is needed, and a
    foot pressed slightly into the ground is never rewarded for it.

    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        ref_contacts: Reference contact flags [num_envs, num_bodies].
        body_indices: Foot body indices to check.
        deadband: Metres of clearance above the reference forgiven before the
            penalty starts, absorbing contact compliance and tracking noise.

    Returns:
        Summed excess height over reference-planted bodies, in metres [num_envs].
    """
    current_height = current_rigid_body_pos[:, body_indices, 2]
    ref_height = ref_rigid_body_pos[:, body_indices, 2]
    planted = ref_contacts[:, body_indices].to(current_height.dtype)
    excess = (current_height - ref_height - deadband).clamp_min(0.0)
    return (excess * planted).sum(dim=-1)


def compute_ref_swing_foot_low_rew(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    ref_contacts: Tensor,
    body_indices: list,
    deadband: float = 0.01,
) -> Tensor:
    """Penalize a swing foot carried lower than the reference carries it.

    The exact mirror of compute_ref_stance_foot_lift_rew: that term prices the
    foot being *above* the reference while the reference has it planted, this
    one prices it being *below* while the reference has it airborne. Together
    they say keep the foot where the reference puts it, in both phases.

    Only worth having once the reference itself is trustworthy. Retargeting
    previously inflated swing height to 1.41x the G1 source, and the policy
    under-lifted by ~28%, which cancelled almost exactly: measured physics apex
    was 1.02x (r1_locomotion_13) and 0.97x (r1_locomotion_15) of the true G1
    height. An earlier version of this term would have pushed the foot toward
    an inflated target and made the motion worse, which is why it was not added
    then. With --swing-height-scale 1.0 the reference is 1.03x G1 and the
    under-lift is exposed: r1_locomotion_16 at epoch 6600 reaches only 0.73x
    the G1 apex, 0.72x of its own reference, a 43 mm mean deficit across 848
    clips, worst on slow walks (0.61) and mildest on sprints (0.82).

    It is a reward gap, not a capability limit. Nothing penalises lift during
    swing -- every contact term is gated on contact -- leg joint velocity in
    swing averages 6.4% of the actuator limit, and a 43 mm apex error costs
    2.0% of relative_body_pos at its sigma of 0.3, which is the only term that
    can see it.

    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        ref_contacts: Reference contact flags [num_envs, num_bodies].
        body_indices: Foot body indices to score.
        deadband: Metres below the reference forgiven before the penalty starts.

    Returns:
        Summed shortfall below the reference over airborne feet, in metres
        [num_envs].
    """
    current_height = current_rigid_body_pos[:, body_indices, 2]
    ref_height = ref_rigid_body_pos[:, body_indices, 2]
    airborne = (~ref_contacts[:, body_indices].bool()).to(current_height.dtype)
    shortfall = (ref_height - current_height - deadband).clamp_min(0.0)
    return (shortfall * airborne).sum(dim=-1)


def compute_ref_touchdown_impact_rew(
    contact_force_magnitudes: Tensor,
    contact_air_time: Tensor,
    prev_rigid_body_vel: Tensor,
    ref_rigid_body_vel: Tensor,
    body_indices: list,
    force_threshold: float = 1.0,
    deadband: float = 0.05,
) -> Tensor:
    """Penalize landing harder than the reference lands, at the touchdown step.

    Measured on r1_locomotion_13 at epoch 6400, the policy touches down at
    0.586 m/s vertical against the reference's 0.188 m/s -- 3.1x harder -- and
    that impulse knocks the *other*, correctly planted foot loose: over 1298
    touchdowns where the contralateral foot stayed in contact throughout, its
    horizontal speed rose from 0.044 m/s before the landing to 0.169 m/s after
    (3.8x), exceeding 0.2 m/s on 26.9% of them. That is the visible foot
    micro-adjustment, and nothing in the reward priced it.

    Deliberately relative to the reference rather than an absolute threshold.
    Reference touchdown speed spans 14.6x across the library -- walk lands at
    0.030 m/s, backwards at 0.437 -- so one absolute limit would either permit
    a slammed walk or punish backwards for its legitimately firm heel plant.
    Sprint's reference lands softly too (0.094 m/s), because a good running
    touchdown is mostly horizontal.

    Chosen over raising contact_force_change, which prices the same impact via
    force rate, for two reasons. Force rate also penalizes push-off, and sprint
    already travels at only 88.8% of reference speed, so suppressing its
    propulsive impulse is the wrong direction. And a kinematic reference has no
    forces at all, so a force-based term cannot be made reference-relative --
    only a velocity one can.

    Reads velocity one step back, because by the step contact first registers
    the impact has already arrested the foot.

    Args:
        contact_force_magnitudes: Current per-body contact force [num_envs, num_bodies].
        contact_air_time: Seconds each body has been airborne, accumulated
            through the end of the previous step [num_envs, num_bodies].
        prev_rigid_body_vel: Body velocities at the end of the previous step
            [num_envs, num_bodies, 3] -- the pre-impact landing speed.
        ref_rigid_body_vel: Reference body velocities [num_envs, num_bodies, 3].
        body_indices: Foot body indices to score.
        force_threshold: Force above which a body counts as in contact.
        deadband: Metres per second of excess forgiven before the penalty starts.

    Returns:
        Summed excess vertical landing speed over feet touching down this step,
        in m/s [num_envs]. Zero on every step that is not a touchdown.
    """
    forces = contact_force_magnitudes[:, body_indices]
    air_time = contact_air_time[:, body_indices]
    # The touchdown step: in contact now, airborne through the previous step.
    first_contact = (forces > force_threshold) & (air_time > 0.0)
    landing = prev_rigid_body_vel[:, body_indices, 2].abs()
    reference = ref_rigid_body_vel[:, body_indices, 2].abs()
    excess = (landing - reference - deadband).clamp_min(0.0)
    return (excess * first_contact.float()).sum(dim=-1)


def compute_feet_air_time_rew(
    contact_force_magnitudes: Tensor,
    contact_air_time: Tensor,
    body_indices: list,
    force_threshold: float = 1.0,
    air_time_offset: float = 0.5,
) -> Tensor:
    """Reward touchdown in proportion to time spent airborne beforehand.

    Cures the "stays in place" AMP-plus-tracking collapse: standing still
    earns zero air time on every foot, so this term pushes the policy to
    actually pick its feet up, independent of any style or tracking signal.

    ``contact_air_time`` is the time (seconds) each body has spent out of
    contact, accumulated through the end of the previous step (see
    ``BaseEnv.contact_air_time``). A reward fires only on the step a
    tracked body's contact force first exceeds ``force_threshold`` -- the
    touchdown instant -- valued by how long that body was airborne minus
    ``air_time_offset`` (a target swing duration; touchdowns faster than
    this are not rewarded further, discouraging unnaturally long strides).

    Args:
        contact_force_magnitudes: Per-body contact force magnitude [num_envs, num_bodies].
        contact_air_time: Per-body seconds since last contact [num_envs, num_bodies].
        body_indices: Absolute indices of the bodies to reward (typically feet).
        force_threshold: Force magnitude above which a body counts as in contact.
        air_time_offset: Seconds subtracted from air time before rewarding.

    Returns:
        Touchdown air-time reward tensor [num_envs].
    """
    forces = contact_force_magnitudes[:, body_indices]
    air_time = contact_air_time[:, body_indices]
    in_contact = forces > force_threshold
    first_contact = in_contact & (air_time > 0.0)
    return torch.sum((air_time - air_time_offset) * first_contact.float(), dim=-1)


def compute_alive_bonus(progress_buf: Tensor) -> Tensor:
    """Constant per-step survival bonus.

    A flat +1 (scaled by the factory's ``weight``) for every environment that
    is still being stepped. Only closes the "learns to fall to dodge
    penalties" failure mode if paired with terminations that actually catch a
    fall/crawl -- otherwise it just pays the policy more for surviving via
    whatever exploit it found.

    Args:
        progress_buf: Episode progress counter [num_envs]. Used only to infer
            batch size and device; its value does not affect the reward.

    Returns:
        Ones tensor [num_envs].
    """
    return torch.ones_like(progress_buf, dtype=torch.float32)


def compute_foot_separation_rew(
    rigid_body_pos: Tensor,
    anchor_pos: Tensor,
    anchor_rot: Tensor,
    left_foot_body_id: int,
    right_foot_body_id: int,
    min_distance: float = 0.10,
    w_last: bool = True,
) -> Tensor:
    """Penalize the feet closing laterally to less than ``min_distance`` apart.

    Computed in the robot's own heading-local frame -- via the same
    ``calc_heading_quat_inv``/``quat_rotate`` pattern used to build body-frame
    steering commands -- so it reflects the robot's actual left-right sense
    regardless of which way it's currently facing (a nonzero commanded yaw
    rate continuously rotates the world-frame foot positions). Only the
    lateral axis is used: a normal front-back stride offset, or the feet
    passing near each other in the direction of travel during mid-stance,
    never trips this -- only the feet actually closing in on or crossing the
    sagittal midline (a scissoring, unstable gait) does.

    Args:
        rigid_body_pos: World-frame body positions [num_envs, num_bodies, 3].
        anchor_pos: World-frame anchor (root) position [num_envs, 3].
        anchor_rot: World-frame anchor orientation [num_envs, 4].
        left_foot_body_id: Body index of one foot.
        right_foot_body_id: Body index of the other foot.
        min_distance: Minimum acceptable lateral separation, in meters.
        w_last: Quaternion component order.

    Returns:
        Violation magnitude tensor [num_envs] (0 when >= min_distance apart).
    """
    left_foot_pos = rigid_body_pos[:, left_foot_body_id]
    right_foot_pos = rigid_body_pos[:, right_foot_body_id]
    heading_inv = rotations.calc_heading_quat_inv(anchor_rot, w_last=w_last)
    local_left = rotations.quat_rotate(
        heading_inv, left_foot_pos - anchor_pos, w_last=w_last
    )
    local_right = rotations.quat_rotate(
        heading_inv, right_foot_pos - anchor_pos, w_last=w_last
    )
    lateral_sep = torch.abs(local_left[..., 1] - local_right[..., 1])
    return torch.clamp(min_distance - lateral_sep, min=0.0)


# =============================================================================
# Helper Functions (used by kernels or for advanced use cases)
# =============================================================================

def joint_limit_violation(
    dof_pos: Tensor,
    dof_limits_lower: Tensor,
    dof_limits_upper: Tensor,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Sum of joint position limit violations.

    Penalizes positions outside [lower, upper] limits.

    Args:
        dof_pos: Joint positions [num_envs, num_dofs].
        dof_limits_lower: Lower limits [num_dofs].
        dof_limits_upper: Upper limits [num_dofs].
        indices: Optional DOF indices to subset.

    Returns:
        Total violation [num_envs].
    """
    if indices is not None:
        dof_pos = dof_pos[:, indices]
        dof_limits_lower = dof_limits_lower[indices]
        dof_limits_upper = dof_limits_upper[indices]

    below_lower = -(dof_pos - dof_limits_lower).clip(max=0.0)
    above_upper = (dof_pos - dof_limits_upper).clip(min=0.0)
    return torch.sum(below_lower + above_upper, dim=1)


def contact_mismatch_sum(
    sim_contacts: Tensor,
    ref_contacts: Tensor,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Sum of contact state mismatches.

    Computes sum(|sim_contacts - ref_contacts|).

    Args:
        sim_contacts: Simulated contacts [num_envs, num_bodies].
        ref_contacts: Reference contacts [num_envs, num_bodies].
        indices: Optional body indices to subset.

    Returns:
        Total mismatch [num_envs].
    """
    if indices is not None:
        sim_contacts = sim_contacts[:, indices]
        ref_contacts = ref_contacts[:, indices]

    return torch.abs(sim_contacts.float() - ref_contacts.float()).sum(dim=1)


def impact_force_penalty(
    current_forces: Tensor,
    previous_forces: Tensor,
    indices: Optional[Tensor] = None,
    threshold: float = 30.0,
) -> Tensor:
    """Sum of sudden contact force changes above a threshold (impact penalty).

    Penalizes abrupt force changes (both increases and decreases) that exceed
    the threshold. Small force changes below the threshold are ignored.

    Args:
        current_forces: Current contact forces [num_envs, num_bodies].
        previous_forces: Previous contact forces [num_envs, num_bodies].
        indices: Optional body indices to subset.
        threshold: Force change threshold below which changes are ignored (default: 30.0).

    Returns:
        Total force change above threshold [num_envs].
    """
    if indices is not None:
        current_forces = current_forces[:, indices]
        previous_forces = previous_forces[:, indices]

    force_changes = torch.abs(current_forces - previous_forces)
    force_changes = torch.clamp(force_changes - threshold, min=0)
    return force_changes.sum(dim=-1)


__all__ = [
    # Main reward kernels
    "compute_action_smoothness",
    "compute_action_smoothness_logmeanexp",
    "compute_pow_rew",
    "compute_soft_pos_limit_rew",
    "compute_contact_match_rew",
    "compute_contact_force_change_rew",
    "compute_feet_air_time_rew",
    "compute_alive_bonus",
    "compute_foot_separation_rew",
    # Helper functions
    "joint_limit_violation",
    "contact_mismatch_sum",
    "impact_force_penalty",
]
