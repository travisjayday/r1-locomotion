# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""R1 privileged physical-motion generator (OmniTrack Stage-I style).

This is a physics teacher, not a deployable robot policy.  The actor and critic
receive clean simulator-only global link states, forces, contacts, joint state,
and the complete multi-horizon reference state.  Physics and observation
domain randomization are disabled. During training only, the reference command
receives OmniTrack PMG's small uniform perturbations; reward targets and
evaluation remain clean. The real R1 effort limits and PD controller remain
active. MimicEvaluator records the simulated rollout as a ProtoMotions-compatible
``predicted_motion_lib_epoch_*.pt``.
"""

from __future__ import annotations

import argparse

from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.terrains.config import (
    CombineMode,
    TerrainConfig,
    TerrainSimConfig,
)
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--motion-id",
        type=int,
        default=None,
        help=(
            "Zero-based motion-library clip ID to train and export. All parallel "
            "environments track this clip at independently sampled phases."
        ),
    )
    parser.add_argument(
        "--reserve-future-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reserve max(future_steps) frames at the end of each clip as "
            "lookahead-only context. Enabled by default; pass "
            "--no-reserve-future-context to restore full-length rollouts."
        ),
    )
    parser.add_argument(
        "--reference-motion-perturbations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply OmniTrack PMG uniform noise to reference q/qdot and root "
            "pose commands during training (default: enabled)."
        ),
    )


def terrain_config(args: argparse.Namespace) -> TerrainConfig:
    return TerrainConfig(
        sim_config=TerrainSimConfig(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            combine_mode=CombineMode.AVERAGE,
        )
    )


def scene_lib_config(args: argparse.Namespace) -> SceneLibConfig:
    return SceneLibConfig(scene_file=getattr(args, "scenes_file", None))


def motion_lib_config(args: argparse.Namespace) -> MotionLibConfig:
    return MotionLibConfig(motion_file=args.motion_file)


def _tensor_observation(path):
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.common import flatten_batch

    return MdpComponent(
        # This function lives in an importable package module, unlike the
        # dynamically loaded experiment, so resolved_configs.pt is pickleable.
        compute_func=flatten_batch,
        dynamic_vars={"x": path},
        static_params={},
    )


def _quaternion_observation(path):
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.common import quaternion_to_tan_norm_and_flatten

    return MdpComponent(
        compute_func=quaternion_to_tan_norm_and_flatten,
        dynamic_vars={"quaternion": path},
        static_params={"w_last": True},
    )


def _body_tensor_observation(path, body_indices, body_axis=-2):
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.common import select_bodies_and_flatten

    return MdpComponent(
        compute_func=select_bodies_and_flatten,
        dynamic_vars={"x": path},
        static_params={"body_indices": body_indices, "body_axis": body_axis},
    )


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.action import make_bm_pd_action_config
    from protomotions.envs.component_factories import (
        action_oscillation_factory,
        action_smoothness_factory,
        anchor_heading_error_term_factory,
        anchor_height_floor_term_factory,
        contact_force_change_rew_factory,
        contact_body_ori_rew_factory,
        contact_slip_rew_factory,
        global_anchor_ori_rew_factory,
        global_anchor_pos_rew_factory,
        global_body_ang_vel_rew_factory,
        global_body_lin_vel_rew_factory,
        max_coords_obs_factory,
        previous_actions_factory,
        relative_body_ori_rew_factory,
        relative_body_pos_error_term_factory,
        ref_stance_foot_lift_rew_factory,
        ref_swing_foot_low_rew_factory,
        ref_touchdown_impact_rew_factory,
        relative_body_pos_rew_factory,
        undesired_contacts_rew_factory,
    )
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.control.mimic_control import (
        MimicControlConfig,
        ReferenceMotionPerturbationConfig,
    )
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.motion_manager.config import (
        AdaptiveBinSamplingConfig,
        MimicMotionManagerConfig,
    )
    from protomotions.envs.rewards import compute_soft_pos_limit_rew

    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
            # At 50 Hz these retain approximately the old 15 Hz lookahead
            # times: 60, 140, 260, and 540 ms versus 67, 133, 267, and 533 ms.
            future_steps=[3, 7, 13, 27],
            # Keep the final max(future_steps) control frames in the source
            # motion as lookahead-only context. This prevents the future root
            # targets from collapsing onto one terminal pose and cueing a stop.
            reserve_future_context=args.reserve_future_context,
            reference_motion_perturbation=(
                ReferenceMotionPerturbationConfig(
                    joint_pos=0.01,
                    joint_vel=0.5,
                    root_pos_xyz=(0.01, 0.01, 0.01),
                    root_ori_rpy=(0.05, 0.05, 0.05),
                )
                if args.reference_motion_perturbations
                else None
            ),
        )
    }

    body_names = robot_cfg.kinematic_info.body_names
    # OmniTrack PMG tracks 14 major links. The paper gives anatomical regions,
    # not exact G1 link names; these are their R1 morphology equivalents.
    major_body_names = [
        "pelvis_link",
        "left_hip_yaw_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_yaw_link", "right_knee_link", "right_ankle_roll_link",
        "waist_yaw_link",
        "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link",
        "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
    ]
    major_body_ids = [body_names.index(name) for name in major_body_names]
    foot_body_ids = [
        body_names.index("left_ankle_roll_link"),
        body_names.index("right_ankle_roll_link"),
    ]
    hand_body_ids = [
        body_names.index("left_wrist_roll_link"),
        body_names.index("right_wrist_roll_link"),
    ]
    end_effector_ids = set(foot_body_ids + hand_body_ids)
    undesired_contact_body_ids = [
        i for i in range(len(body_names)) if i not in end_effector_ids
    ]

    observation_components = {
        # Clean global state for every R1 link.  max-coords includes global
        # orientations/velocities and all non-anchor positions; root_global_pos
        # below restores the intentionally omitted anchor translation.
        "privileged_global_state": max_coords_obs_factory(
            use_noisy=False,
            local_obs=False,
            root_height_obs=True,
            observe_contacts=True,
        ),
        "root_global_pos": _tensor_observation(EnvContext.current.root_pos),
        "joint_pos": _tensor_observation(EnvContext.current.dof_pos),
        "joint_vel": _tensor_observation(EnvContext.current.dof_vel),
        "joint_forces": _tensor_observation(EnvContext.current.dof_forces),
        "contact_force_magnitudes": _tensor_observation(
            EnvContext.current_contact_force_magnitudes
        ),
        #"reference_contacts": _body_tensor_observation(
            #EnvContext.mimic.ref_state.rigid_body_contacts,
            #foot_body_ids,
            #body_axis=-1,
        #),
        # Compact OmniTrack PMG command: reference q/qdot and root pose. Four
        # horizons retain transition anticipation without duplicating complete
        # all-link trajectories. Encode root rotation in sign-invariant 6D form
        # so equivalent q/-q motion-library quaternions cannot create a policy-
        # input discontinuity.
        "reference_root_pos": _tensor_observation(
            EnvContext.mimic.command_root_pos
        ),
        "reference_root_rot": _quaternion_observation(
            EnvContext.mimic.command_root_rot
        ),
        "reference_joint_pos": _tensor_observation(
            EnvContext.mimic.command_dof_pos
        ),
        "reference_joint_vel": _tensor_observation(
            EnvContext.mimic.command_dof_vel
        ),
        # 2, not 1: action_oscillation penalises the second difference
        # ||a_t - 2*a_{t-1} + a_{t-2}||, so a_{t-2} has to be observable or the
        # policy is optimising an expectation over a quantity it cannot see --
        # with one previous action it cannot distinguish a reversal from a
        # continuation, which is precisely the distinction the term rewards.
        # Two previous actions is the exact information the reward depends on;
        # costs 26 inputs (806 -> 832).
        "previous_actions": previous_actions_factory(history_steps=2),
    }

    reward_components = {
        # OmniTrack PMG tracking terms (Table A.5).
        # sigma is a tolerance: the error at which the kernel falls to 1/e.
        # At 0.5 a small drift was nearly free -- the 6 cm lateral offset that
        # r1_locomotion_12 accumulated over the last stride of
        # transitions__transition_walk_idle_set01 cost 1.4% of this term, and
        # once the reference goes static that offset is frozen in permanently,
        # because correcting it needs a step the reference does not contain.
        # 0.3 nearly triples the gradient there (0.47 -> 1.28) while the term
        # still spans a useful range across the library (0.32 sprint to 0.86
        # idle_turn at 0.3, versus 0.47 to 0.94 at 0.5).
        #
        # This deliberately shifts emphasis from large errors to small ones:
        # sprint's root error averages 0.58 m and its gradient here drops
        # 1.20 -> 0.30. That is an accepted trade because global_root_lin_vel
        # (sigma 1.5 -> 0.4 in the same change) now carries the large-error
        # work. Do not sharpen past ~0.3: at 0.2 the sprint gradient collapses
        # to 0.006 and the term goes dead at the bottom, the same failure as
        # the old 1.5 velocity kernel at the top.
        "global_root_pos": global_anchor_pos_rew_factory(weight=1.0, sigma=0.3),
        # Perceptually important rotations get isolated terms so their errors
        # are not diluted by the all-body average below. R1 body indices follow
        # the MJCF preorder: pelvis=0, feet=6/12, upper torso (waist yaw)=14.
        "global_root_ori": global_anchor_ori_rew_factory(weight=1.5, sigma=0.4),
        "relative_body_pos": relative_body_pos_rew_factory(
            weight=1.0, sigma=0.3, body_indices=major_body_ids
        ),
        "relative_body_ori": relative_body_ori_rew_factory(
            weight=1.0, sigma=0.4, body_indices=major_body_ids
        ),
        "torso_relative_ori": relative_body_ori_rew_factory(
            weight=1.5, sigma=0.3, body_indices=[body_names.index('waist_yaw_link')]
        ),
        "contact_foot_ori": contact_body_ori_rew_factory(
            weight=1.5, sigma=0.25, body_indices=foot_body_ids
        ),
        "global_body_lin_vel": global_body_lin_vel_rew_factory(
            weight=1.0, sigma=1.0, body_indices=major_body_ids
        ),
        # Give translational tracking a direct, non-diluted signal.
        #
        # sigma is a tolerance: the error at which the kernel falls to 1/e. At
        # the previous 1.5 it declared a 1.5 m/s tolerance, larger than the
        # entire speed of most clips in the library, and the term was
        # consequently saturated for the whole of r1_locomotion_11 -- scaled
        # reward 1.380 at epoch 200 and 1.384 at epoch 9000, i.e. 92% of its
        # 1.5 maximum from before training began, flat for 9000 epochs and
        # contributing no gradient at any point. (The comment previously here
        # argued a broad kernel preserves gradient on fast clips; the measured
        # opposite is that it removes gradient on all of them.) 0.4 is set as
        # a specification -- the root-velocity accuracy actually wanted --
        # rather than fitted to a past run's error.
        "global_root_lin_vel": global_body_lin_vel_rew_factory(
            weight=1.5,
            sigma=0.4,
            body_indices=[robot_cfg.anchor_body_index],
        ),
        "global_body_ang_vel": global_body_ang_vel_rew_factory(
            weight=1.0, sigma=3.14, body_indices=major_body_ids
        ),
        "action_smoothness": action_smoothness_factory(weight=-0.1),
        # action_smoothness is the L2 norm of the FIRST difference of the PD
        # target: it prices how fast the target moves, not whether it reverses,
        # so +0.08/-0.08/+0.08 and +0.08/+0.08/+0.08 score identically. Its raw
        # value correlates with eval jerk at -0.03 across r1_locomotion_13 and
        # _14 -- in run 13 the deltas shrank 1.48 -> 1.34 while jerk grew
        # 1062 -> 1326. The second difference separates the two: zero for a
        # ramp, maximal for alternation. Reconstructing the commanded target
        # from the run 14 rollout shows it reversing 30 times/second against
        # joints reversing 19.5 -- the robot is low-pass filtering a
        # near-Nyquist command. Speed-independent, so invisible in sprint and
        # dominant in the slow clips (foot acceleration reverses 22-26 times/s
        # against the reference's 7). Weight set from the measured raw value;
        # see the smoke-test note in the launch checklist.
        "action_oscillation": action_oscillation_factory(weight=-0.05),
        "joint_position_limits": MdpComponent(
            compute_func=compute_soft_pos_limit_rew,
            dynamic_vars={"dof_pos": EnvContext.current.dof_pos},
            static_params={
                "weight": -10.0,
                "dof_limits_lower": robot_cfg.kinematic_info.dof_limits_lower,
                "dof_limits_upper": robot_cfg.kinematic_info.dof_limits_upper,
            },
        ),
        # The source now carries repaired sprint contacts.  Matching them also
        # penalizes undesired non-reference contacts and makes the exported
        # simulator contact masks coherent.
        "undesired_contacts": undesired_contacts_rew_factory(
            body_indices=undesired_contact_body_ids,
            weight=-0.2,
            force_threshold=1.0,
            zero_during_grace_period=True,
        ),
        "contact_force_change": contact_force_change_rew_factory(
            weight=-1.0e-5,
            min_value=-0.5,
            threshold=30.0,
            zero_during_grace_period=True,
        ),
        # Gated on the simulator's own contact force, not the reference
        # motion's contact labels -- the retargeted reference's foot
        # placement/contact quality isn't trusted enough to gate a penalty on
        # directly (see contact_foot_ori above, which does use it, but only
        # for orientation *tracking*, where being wrong just reduces reward
        # rather than actively encouraging a bad habit). Without this, a foot
        # that's actually planted per the physics sim can still slide freely
        # with no direct penalty; position-tracking reward alone doesn't
        # discourage sliding if it keeps average position roughly correct.
        # This penalty only fires while a foot is in contact, so it prices
        # contact without pricing its absence: on r1_locomotion_8 at epoch
        # 5600 the policy hovered the foot 5 mm (idle_turn) to 65 mm (sprint)
        # above the reference stance and lost contact during 19-56% of
        # reference-stance frames, because a few mm of clearance is nearly
        # free on position tracking and strictly cheaper on every
        # contact-gated penalty.
        #
        # contact_match_rew_factory is NOT the answer to that, despite looking
        # like it. r1_locomotion_9 ran it at -0.2 and made the symptom worse:
        # idle_turn foot-contact transitions went from 2.51x the reference
        # rate to 3.95x by epoch 6000 (4.56x by 13600) while hover barely
        # moved (5.2 -> 6.6 mm), and peak eval success fell 0.56 -> 0.35. It
        # penalizes instantaneous |sim_contact - ref_contact|, so a foot
        # already grazing the threshold minimizes it most cheaply by tapping
        # the ground rather than by committing to stance. Fixing the hover
        # needs a term that rewards *sustained* contact -- penalizing contact
        # transitions absent from the reference, or rewarding contact force
        # during reference stance -- not instantaneous state matching.
        "contact_slip": contact_slip_rew_factory(
            body_indices=foot_body_ids,
            weight=-0.2,
            force_threshold=1.0,
            zero_during_grace_period=True,
        ),
        # The other half of "keep the foot where the reference puts it":
        # contact_slip prices moving a foot that is down, this prices a foot
        # being up when the reference has it planted. Without it, lifting is
        # the one action that escapes every contact-gated penalty, and slow
        # motions degenerate into tapping -- on r1_locomotion_10 at epoch 6400
        # the policy raised a reference-planted foot a median 18.8 mm for
        # ~40 ms at a time, on 60-83% of the frames where the reference wanted
        # double support, and the share of stance frames lost that way was
        # still climbing (backwards 9.1% -> 15.5% between epochs 2000 and
        # 6400). Fast motions, which have no double-support phase to fail,
        # were unaffected (sprint 3.3% -> 2.2%).
        # The policy lands 3.1x harder than the reference (0.586 vs 0.188 m/s
        # measured on r1_locomotion_13 @6400), and that impulse knocks the
        # opposite, correctly planted foot loose -- its horizontal speed goes
        # 0.044 -> 0.169 m/s across a touchdown, exceeding 0.2 m/s on 26.9% of
        # them. Reference-relative because reference landing speed spans 14.6x
        # (walk 0.030, backwards 0.437), so no absolute threshold fits.
        # Preferred over raising contact_force_change: force rate would also
        # tax push-off, which sprint needs, and a kinematic reference has no
        # forces to be relative to.
        "ref_touchdown_impact": ref_touchdown_impact_rew_factory(
            body_indices=foot_body_ids,
            weight=-2.0,
            force_threshold=1.0,
            deadband=0.05,
        ),
        # Mirror of ref_stance_foot_lift below: that prices the foot being
        # above the reference while it should be planted, this prices it being
        # below while it should be airborne. Deliberately not added until the
        # retargeting swing-height inflation was fixed -- at 1.41x reference
        # apex the policy's ~28% under-lift cancelled it almost exactly
        # (physics reached 1.02x and 0.97x of true G1 height in runs 13 and
        # 15), so this term would have chased an inflated target. With
        # --swing-height-scale 1.0 the reference is 1.03x G1 and r1_locomotion_16
        # reaches only 0.73x: a 43 mm mean apex deficit over 848 clips, 0.61x
        # on slow walks. Verified to be a reward gap rather than a limit --
        # nothing penalises swing lift, swing joint velocity is 6.4% of the
        # actuator limit, and 43 mm costs 2.0% of relative_body_pos at sigma 0.3.
        # Weight set from a 64-env smoke test rather than by symmetry with
        # ref_stance_foot_lift: at -10.0 the scaled term was -0.265, 45x its
        # mirror (-0.006) and larger than action_smoothness (-0.15), because
        # the swing shortfall is a persistent ~27 mm rather than the rare
        # violation the stance term prices. -4.0 lands it near -0.11.
        "ref_swing_foot_low": ref_swing_foot_low_rew_factory(
            body_indices=foot_body_ids,
            weight=-4.0,
            deadband=0.01,
        ),
        "ref_stance_foot_lift": ref_stance_foot_lift_rew_factory(
            body_indices=foot_body_ids,
            weight=-10.0,
            deadband=0.005,
        ),
    }
    print(f"Motion ID: {args.motion_id}")

    return EnvConfig(
        # Do not inject the standard 5 cm spawn bump into the physical motion
        # exported by MimicEvaluator.
        ref_respawn_offset=0.0,
        ref_contact_smooth_window=3,
        max_episode_length=1000,
        # 2 (not 1) so processed_actions[:, 2] exists for the
        # second-difference action_oscillation penalty. The observation is
        # unaffected: previous_actions_factory(history_steps=1) selects only
        # the first historical entry regardless of buffer depth.
        num_state_history_steps=2,
        control_components=control_components,
        observation_components=observation_components,
        # OmniTrack uses a deliberately loose Stage-I early termination so raw
        # reference artifacts do not prevent coverage of difficult segments.
        termination_components={
            "large_tracking_error": relative_body_pos_error_term_factory(
                threshold=1.0,
                body_indices=major_body_ids,
            ),
            # Ground-truth complement to the tracking-error check above: a
            # mean- or relative-frame error can stay under its own threshold
            # well into an actual fall (a collapsed-but-still pose can have
            # small relative-pose error even though its absolute position is
            # nowhere reasonable). This fires immediately and unambiguously
            # once the pelvis is actually near the ground, independent of
            # what the reference happens to be doing at that instant.
            "fallen": anchor_height_floor_term_factory(min_height=0.3),
            # anchor_ori_error_term_factory ("bad_ref_ori") compares
            # projected-gravity Z and is yaw-invariant: spinning purely about
            # the vertical axis doesn't change how "upright" something is, so
            # it's blind to a robot that stays upright but turns the wrong
            # way -- exactly what walk_spin_left_counterclockwise does after
            # ~frame 70. anchor_heading_error_term_factory checks heading
            # directly instead. 0.5 rad (~29deg) sits well above normal
            # tracking noise (<7deg observed) and below where the reversal
            # commits (>90deg by frame ~110).
            "bad_heading": anchor_heading_error_term_factory(threshold=0.6),
        },
        reward_components=reward_components,
        # Use the full static-torque-equivalent joint displacement per unit
        # action. This explicit value overrides R1's conservative 0.25 default.
        action_config=make_bm_pd_action_config(
            robot_cfg,
            effort_fraction=1.0,
            disabled_dof_names=["head_pitch_joint", "head_yaw_joint"],
        ),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
            realign_motion_with_humanoid_on_each_step=False,
            fixed_motion_id=args.motion_id,
            # Within-motion curriculum: bias reset-time sampling toward the
            # specific 1s segment of a clip that's actually been failing,
            # rather than treating the whole clip as equally hard. See
            # AdaptiveBinSamplingConfig for the OmniTrack PMG Table A.7 reference.
            adaptive_bin_sampling=AdaptiveBinSamplingConfig(enabled=True),
        ),
    )


def agent_config(
    robot_config: RobotConfig,
    env_config: EnvConfig,
    args: argparse.Namespace,
) -> PPOAgentConfig:
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.agents.ppo.config import (
        AdaptiveLRConfig,
        AdvantageNormalizationConfig,
        PPOActorConfig,
        PPOModelConfig,
    )
    from protomotions.envs.component_factories import (
        gr_error_factory,
        gt_error_factory,
        max_joint_error_factory,
    )

    obs_keys = [
        "privileged_global_state",
        "root_global_pos",
        "joint_pos",
        "joint_vel",
        "joint_forces",
        "contact_force_magnitudes",
        #"reference_contacts",
        "reference_root_pos",
        "reference_root_rot",
        "reference_joint_pos",
        "reference_joint_vel",
        "previous_actions",
    ]

    actor = PPOActorConfig(
        num_out=robot_config.number_of_actions,
        actor_logstd=-2.3,
        learnable_std=True,
        in_keys=obs_keys,
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=obs_keys,
            normalize_obs=True,
            norm_clamp_value=10.0,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            # Widened 768->1024 and given one extra layer (5->6): the motion
            # library nearly doubled (454->908, original + mirrored) since
            # this was last sized, and a capacity-limited shared network
            # being asked to hold that much behaviorally distinct content
            # without interference is a second, independent contributor to
            # the forgetting diagnosed in r1_locomotion_5 (on top of the
            # motion-weight cascade fixed above). Fine for this network to
            # lean toward memorizing the reference set -- it's an omniscient,
            # privileged-observation teacher; generalization is the later
            # deployable student's problem, not this one's.
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
        ),
    )
    critic = MLPWithConcatConfig(
        in_keys=obs_keys,
        normalize_obs=True,
        norm_clamp_value=10.0,
        out_keys=["value"],
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    return PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=obs_keys,
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor,
            critic=critic,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        num_mini_epochs=2,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        # r1_locomotion_5 (learnable_std=True, this flat 0.005 default) ran
        # std_mean from 0.10 at epoch 0 to >1.5 by epoch 9600 with nothing to
        # check it -- a constant entropy bonus has no reason to weaken as the
        # policy matures, and logstd had no ceiling. Decaying it to 0 over the
        # first 8000 epochs (comfortably before the point where std got large
        # enough to visibly hurt eval/success_rate) plus PPOActorConfig's new
        # logstd_max=-0.5 default (std <= ~0.61) bound the runaway from two
        # directions instead of one.
        #
        # Decaying all the way to 0.0 overcorrected. r1_locomotion_8 ran that
        # schedule and its real-DOF std (excluding the two disabled head
        # joints, which are separately capped by logstd_max) peaked at 0.157
        # around epoch 2000 and then collapsed: 0.155 at 3000, 0.149 at 3800,
        # 0.121 by 6029. Progress stalled with it -- eval gt_error flattened
        # at ~0.27 after epoch 5200 and the task-reward slope fell from
        # +0.0021 to +0.0014 per 1k epochs. A small floor keeps exploration
        # pressure alive for the whole run while still shedding most of the
        # bonus early, which is what the decay was for.
        # Reverted to 0.002 after r1_locomotion_14. Raising the floor to 0.004
        # did hold std up as intended (0.171-0.178 through epoch 12000, where
        # run 13's slid to 0.147), but the run peaked at the same epoch as run
        # 13 (7000, success 0.9108 vs 0.9141) and then degraded to 0.7346 while
        # eval jerk doubled 1002 -> 2060. Sustained exploration past convergence
        # fed the oscillation exploit rather than finding better policies. The
        # note below is kept for the record of why 0.002 beats 0.0.
        #
        # 0.002 was still too low. r1_locomotion_13 ran that floor and its
        # real-DOF std (excluding the two clamped head joints) fell 0.1354 at
        # epoch 5000 -> 0.1095 at 8000 -> 0.1088, i.e. below the 0.1216 that
        # r1_locomotion_10 collapsed to with no floor at all; eval flattened
        # over the same window, with the gt_error slope going from -0.031 per
        # 1k epochs (3000-5000) to +0.005 (6500-8135). The floor slowed the
        # slide rather than stopping it. 0.004 is closer to the flat 0.005
        # that r1_locomotion_5 used, which is safe here in a way it was not
        # there: logstd_max=-0.5 now caps std at ~0.61, so the runaway that
        # motivated the decay in the first place is bounded independently.
        entropy_coef=0.005,
        entropy_coef_final=0.002,
        entropy_coef_decay_epochs=8000,
        adaptive_lr=AdaptiveLRConfig(
            enabled=True,
            desired_kl=0.013,
            min_lr=1.0e-7,
            max_lr=2.0e-4,
            # Tried kl_ema_alpha=0.15 here to smooth the once-per-eval KL
            # spike (see motion_weights_rules below). Reverted: _update_
            # learning_rate already reacts every epoch with a repeated 1.5x
            # step, so feeding it a lagged/smoothed reading instead of the
            # instantaneous one meant a spike stayed "visible" for several
            # consecutive epochs instead of one, and the controller kept
            # cutting LR each of those epochs -- compounding all the way to
            # min_lr (observed: crashed to the 1e-7 floor once per eval cycle
            # in r1_locomotion_6, versus ~1.4e-5 with the raw, unsmoothed
            # signal) and then needing an equally repeated overshoot to climb
            # back out. The un-smoothed spike this was meant to soften was
            # minor to begin with (self-corrected in ~3-4 epochs on its own,
            # see r1_locomotion_5's history) -- not worth this instability.
            # kl_ema_alpha left at its default (1.0 = raw KL, no smoothing).
        ),
        evaluator=MimicEvaluatorConfig(
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.50),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(threshold=1.0),
            },
            eval_metrics_every=200,
            eval_metrics_at_epochs=[3],
            save_predicted_motion_lib_every=1,
            # Policy runs at 50 Hz (200 Hz physics / decimation 4). Export at
            # the policy rate so the capture cadence divides each control step.
            trajectory_export_fps=50,
            save_video_every_epochs=200,
            save_video_at_epochs=[3],
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                # 0.0 (the previous value) is special-cased to an instant
                # snap to weight=1.0 the moment a motion fails one eval --
                # every failing motion jumps to full weight in a single step,
                # a discrete shock to the training distribution that showed
                # up as a real KL spike (see adaptive_lr above). This mirrors
                # the success side's own discount instead: raising a failing
                # motion's weight by only 1/0.999^200 ~= 1.22x per failing
                # eval, so it climbs toward 1.0 over many evals rather than
                # in one. Note this update rule is exponential in the
                # discount, not linear -- 0.9 here would multiply a failing
                # motion's weight by ~1.4e9 in a single eval (worse than the
                # snap it was meant to soften); 0.999 is the right order of
                # magnitude to mirror the success side's pace.
                motion_weights_update_failure_discount=0.999,
                # r1_locomotion_5 (908 motions): the default "1/num_motions"
                # floor (~=0.0011) let mastered motions decay to near-zero
                # sampling probability -- by the end of that run the median
                # weight across all 908 motions was 1.0 (checked directly in
                # the saved motion_manager state), meaning over half the
                # library was pinned at max weight while the rest saw almost
                # no rehearsal. Directly evidenced in the failed-motions logs:
                # 209 motions that were succeeding at the epoch-7600 peak had
                # become new failures by epoch 9600 -- real forgetting, not
                # noise, from a self-reinforcing loop where each new failure
                # crowded out rehearsal of everything else. A flat floor two
                # orders of magnitude higher than the old one keeps mastered
                # motions getting occasional practice even during a partial
                # relapse, at the cost of slightly less sampling sharpness on
                # the hardest content -- a deliberate trade given the model
                # forgetting things it already knew is the worse failure mode
                # here.
                min_motion_weight=0.05,
                # The floor above bounds the bottom; nothing bounded the top
                # until this. The failure branch multiplies a failing motion's
                # weight by 1/0.999^200 = 1.222 every eval with no ceiling, so
                # r1_locomotion_10 reached a max weight of 991429 against the
                # 0.05 floor by epoch 13400 -- effective sample size 39 of 908,
                # top-50 clips holding 95% of the sampling probability.
                # Replaying the failure logs shows the spiral: sampling
                # concentrates, neglected clips regress, the newly failing
                # clips get upweighted, concentration accelerates. Failures
                # bottomed at 349 (epoch 6000, ESS 160) and climbed to 556 by
                # 13000 while eval success fell 0.6156 -> 0.3877. Capping the
                # ratio holds ESS near 470-550 in the same replay, and a
                # persistently failing clip is still sampled 20x an easy one.
                max_motion_weight_ratio=20.0,
            ),
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True
        ),
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    args: argparse.Namespace,
):
    # OmniTrack PMG penalizes non-end-effector contacts. Sense all bodies except
    # the hands; feet remain sensed for reference contact matching.
    hand_names = {"left_wrist_roll_link", "right_wrist_roll_link"}
    robot_cfg.update_fields(
        contact_bodies=[
            name
            for name in robot_cfg.kinematic_info.body_names
            if name not in hand_names
        ],
        reset_noise=None,
    )
    simulator_cfg.sim.fps = 200
    simulator_cfg.sim.decimation = 4
    simulator_cfg.domain_randomization = None


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg: EnvConfig,
    agent_cfg: PPOAgentConfig,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
):
    simulator_cfg.domain_randomization = None
    robot_cfg.reset_noise = None
    env_cfg.termination_components = {}
    env_cfg.max_episode_length = 1_000_000
    env_cfg.motion_manager.init_start_prob = 1.0
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.control_components["mimic"].reference_motion_perturbation = None
