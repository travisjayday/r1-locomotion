# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""R1 deployable-observation locomotion trained with AMP steering.

The policy receives body-frame planar velocity and yaw-rate commands while an
unconditioned AMP discriminator learns style from the physically achieved R1
motion library exported by the omniscient teacher.
"""

from __future__ import annotations

import argparse

from examples.experiments.steering import mlp as base_steering
from protomotions.agents.amp.config import AMPAgentConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


def terrain_config(args: argparse.Namespace):
    return base_steering.terrain_config(args)


def scene_lib_config(args: argparse.Namespace):
    return base_steering.scene_lib_config(args)


def motion_lib_config(args: argparse.Namespace):
    return base_steering.motion_lib_config(args)


def configure_robot_and_simulator(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    args: argparse.Namespace,
):
    """Match the teacher's timing and add mild first-stage randomization."""
    from protomotions.simulator.base_simulator.config import (
        ActuatorDomainRandomizationConfig,
        ActionNoiseDomainRandomizationConfig,
        CenterOfMassDomainRandomizationConfig,
        DomainRandomizationConfig,
        DomainRandomizationCurriculumStageConfig,
        FrictionDomainRandomizationConfig,
        PushDomainRandomizationConfig,
        RigidBodyDomainRandomizationConfig,
        RobotNoiseConfig,
    )

    foot_names = {"left_ankle_roll_link", "right_ankle_roll_link"}
    robot_cfg.update_fields(
        # Every body needs an active contact sensor, including the hands --
        # a prior version excluded them here, which meant "fell and crawled on
        # hands" was invisible to both the undesired-contacts penalty and any
        # contact-based termination: the simulator never reported hand contact
        # force at all, not just an unpenalized one.
        contact_bodies=list(robot_cfg.kinematic_info.body_names),
        # Only feet may contact the ground without ending the episode (see
        # fall_termination_factory in env_config). Left at the default "all"
        # this check never fires, which is what let bracing on hands/knees
        # dodge the anchor-height-floor termination instead of triggering a
        # proper fall.
        non_termination_contact_bodies=list(foot_names),
        reset_noise=RobotNoiseConfig(
            dof_pos_noise=0.03,
            root_pos_noise=[0.02, 0.02, 0.01],
            root_rot_noise=[0.04, 0.04, 0.08],
            root_vel_noise=[0.05, 0.05, 0.03],
            root_ang_vel_noise=[0.05, 0.05, 0.08],
        ),
    )
    simulator_cfg.sim.fps = 200
    simulator_cfg.sim.decimation = 4
    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(-0.02, 0.02),
            dof_indices=[
                dof_id
                for dof_id, dof_name in enumerate(
                    robot_cfg.kinematic_info.dof_names
                )
                if dof_name not in {"head_pitch_joint", "head_yaw_joint"}
            ],
        ),
        friction=FrictionDomainRandomizationConfig(
            num_buckets=32,
            static_friction_range=(0.55, 1.45),
            dynamic_friction_range=(0.50, 1.35),
            restitution_range=(0.0, 0.08),
            body_names=[".*"],
        ),
        center_of_mass=CenterOfMassDomainRandomizationConfig(
            com_range={
                "x": (-0.02, 0.02),
                "y": (-0.03, 0.03),
                "z": (-0.02, 0.02),
            },
            body_names=robot_cfg.common_naming_to_robot_body_names[
                "torso_body_name"
            ],
        ),
        observation_noise=RobotNoiseConfig(
            dof_pos_noise=0.01,
            dof_vel_noise=0.40,
            anchor_ang_vel_noise=0.20,
            anchor_rot_noise=0.02,
        ),
        push=PushDomainRandomizationConfig(
            push_interval_range=(3.0, 6.0),
            max_linear_velocity=(0.30, 0.30, 0.10),
            max_angular_velocity=(0.30, 0.30, 0.50),
        ),
        actuator=ActuatorDomainRandomizationConfig(
            stiffness_scale_range=(0.85, 1.15),
            damping_scale_range=(0.80, 1.20),
            motor_strength_scale_range=(0.85, 1.15),
            # Physics substeps: 0--2 is 0--10 ms at the 200 Hz training rate.
            delay_steps_range=(0, 2),
            dof_indices=[
                dof_id
                for dof_id, dof_name in enumerate(
                    robot_cfg.kinematic_info.dof_names
                )
                if dof_name not in {"head_pitch_joint", "head_yaw_joint"}
            ],
        ),
        rigid_body=RigidBodyDomainRandomizationConfig(
            mass_scale_range=(0.90, 1.10),
            inertia_scale_range=(0.90, 1.10),
            body_names=[".*"],
        ),
        # A warm start retains useful locomotion while adapting immediately to
        # small model errors. Avoid a single large distribution jump: widen at
        # 750 epochs, then reach the deployment envelope at 1500.
        curriculum=[
            DomainRandomizationCurriculumStageConfig(
                start_epoch=0, intensity=0.50
            ),
            DomainRandomizationCurriculumStageConfig(
                start_epoch=750, intensity=0.75
            ),
            DomainRandomizationCurriculumStageConfig(
                start_epoch=1500, intensity=1.00
            ),
        ],
    )


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.action import make_bm_pd_action_config
    from protomotions.envs.component_factories import (
        action_smoothness_factory,
        alive_bonus_rew_factory,
        ang_vel_xy_l2_rew_factory,
        anchor_height_floor_term_factory,
        body_orientation_l2_rew_factory,
        dof_pos_tracking_rew_factory,
        fall_termination_factory,
        feet_air_time_rew_factory,
        foot_separation_rew_factory,
        historical_max_coords_obs_factory,
        lin_vel_z_l2_rew_factory,
        max_coords_obs_factory,
        pow_rew_factory,
        previous_actions_factory,
        reduced_coords_obs_factory,
        symmetry_state_obs_factory,
        track_ang_vel_z_rew_factory,
        track_lin_vel_xy_yaw_frame_rew_factory,
        undesired_contacts_rew_factory,
        yaw_rate_steering_obs_factory,
    )
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.control.steering_control import (
        RandomYawRateSteeringCommandSourceConfig,
        SteeringControlConfig,
        YawRateSteeringCurriculumStageConfig,
    )
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.motion_manager.config import MotionManagerConfig
    from protomotions.envs.rewards import compute_soft_pos_limit_rew

    body_names = robot_cfg.kinematic_info.body_names
    foot_names = {"left_ankle_roll_link", "right_ankle_roll_link"}
    foot_body_ids = [
        body_id
        for body_id, body_name in enumerate(body_names)
        if body_name in foot_names
    ]
    left_foot_body_id = body_names.index("left_ankle_roll_link")
    right_foot_body_id = body_names.index("right_ankle_roll_link")
    # Hands are deliberately NOT in the allowed set: a prior version let the
    # robot brace on hands (a "crawl") to keep its anchor above the height
    # floor without ever being penalized, since hand-ground contact was
    # excluded from both the undesired-contacts reward below AND the sensor
    # setup in configure_robot_and_simulator. Only feet are allowed contact
    # now, matching non_termination_contact_bodies there.
    allowed_contact_names = foot_names
    undesired_contact_body_ids = [
        body_id
        for body_id, body_name in enumerate(body_names)
        if body_name not in allowed_contact_names
    ]

    observation_components = {
        # The actor consumes only encoder/IMU state, its prior action, and the
        # externally supplied velocity command. Clean full-body state remains
        # available only to the asymmetric critic and AMP discriminator.
        "proprio": reduced_coords_obs_factory(
            use_noisy=True,
            root_height_obs=False,
            root_vel_obs=False,
        ),
        "previous_actions": previous_actions_factory(
            history_steps=4,
            processed=True,
        ),
        # Side-channel for the PPO symmetry loss (agent_config below) -- not
        # consumed by the actor/critic/discriminator. use_noisy=True matches
        # "proprio" above so the loss compares like-for-like.
        "symmetry_state": symmetry_state_obs_factory(use_noisy=True),
        "max_coords_obs": max_coords_obs_factory(
            use_noisy=False,
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
        ),
        "historical_max_coords_obs": historical_max_coords_obs_factory(
            use_noisy=False,
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
        ),
        "steering": yaw_rate_steering_obs_factory(),
    }

    reward_components = {
        "track_lin_vel_xy": track_lin_vel_xy_yaw_frame_rew_factory(
            weight=2.0, std=0.5
        ),
        "track_yaw_rate": track_ang_vel_z_rew_factory(weight=1.0, std=0.5),
        "lin_vel_z": lin_vel_z_l2_rew_factory(weight=-1.0),
        "ang_vel_xy": ang_vel_xy_l2_rew_factory(weight=-0.05),
        "body_orientation": body_orientation_l2_rew_factory(weight=-1.0),
        "energy": pow_rew_factory(weight=-1.0e-4, min_value=-0.5),
        "action_smoothness": action_smoothness_factory(weight=-0.02),
        "joint_position_limits": MdpComponent(
            compute_func=compute_soft_pos_limit_rew,
            dynamic_vars={"dof_pos": EnvContext.current.dof_pos},
            static_params={
                "weight": -10.0,
                "dof_limits_lower": robot_cfg.kinematic_info.dof_limits_lower,
                "dof_limits_upper": robot_cfg.kinematic_info.dof_limits_upper,
            },
        ),
        "undesired_contacts": undesired_contacts_rew_factory(
            body_indices=undesired_contact_body_ids,
            weight=-0.2,
            force_threshold=1.0,
            zero_during_grace_period=True,
        ),
        # Standing still earns zero air time on every foot, so this gives a
        # nonzero gradient toward stepping even before track_lin_vel_xy is
        # tracked well -- the standard fix for the AMP-plus-tracking
        # "stays in place" collapse.
        "feet_air_time": feet_air_time_rew_factory(
            body_indices=foot_body_ids,
            weight=1.0,
            force_threshold=1.0,
            air_time_offset=0.4,
        ),
        # Discourages a scissoring/crossed-leg gait: penalizes the feet
        # closing to less than 10cm apart laterally (heading-local, so it
        # holds regardless of which way the robot is facing). Only the
        # lateral axis is used, so the normal front-back stride offset
        # between the feet is never penalized.
        "foot_separation": foot_separation_rew_factory(
            left_foot_body_id=left_foot_body_id,
            right_foot_body_id=right_foot_body_id,
            weight=-10.0,
            min_distance=0.10,
        ),
        # A flat survival bonus. Only safe to add alongside the hard fall
        # termination below -- otherwise it would just pay the policy more
        # for the same fall-and-crawl exploit that termination closes.
        "alive": alive_bonus_rew_factory(weight=0.5),
        # Small explicit DeepMimic-style joint-angle imitation signal.
        # AMP's discriminator reward only asks the policy to look
        # statistically plausible -- it never requires closely matching any
        # specific reference trajectory. This is a no-op (reward saturates
        # at its maximum) unless track_reference_pose is set on the active
        # curriculum stage below, so it's safe to leave wired here across
        # the whole curriculum.
        "dof_pos_tracking": dof_pos_tracking_rew_factory(
            weight=0.15,
            coefficient=-10.0,
        ),
    }

    return EnvConfig(
        max_episode_length=300,
        reset_grace_period=5,
        num_state_history_steps=8,
        control_components={
            "steering": SteeringControlConfig(
                tar_speed_min=0.2,
                tar_speed_max=1.8,
                heading_change_steps_min=75,
                heading_change_steps_max=200,
                yaw_rate_commands=True,
                command_source=RandomYawRateSteeringCommandSourceConfig(
                    stand_probability=0.10,
                    turn_in_place_probability=0.15,
                    omnidirectional_probability=0.25,
                    forward_heading_range=0.60,
                    # Otherwise only reachable through the uniform
                    # omnidirectional slice (a thin sliver of the full
                    # circle), which leaves it chronically undertrained
                    # relative to forward walking. Cap explicit backward
                    # samples at 0.7 m/s, where the library has clean straight
                    # matches; faster nominal samples were usually matched to
                    # noisy turning frames instead.
                    backward_probability=0.25,
                    backward_heading_range=0.3,
                    backward_speed_max=0.7,
                    high_speed_probability=0.25,
                    high_speed_min=1.2,
                    # Mirrors high_speed_probability/high_speed_min. Without
                    # this, slow forward walking was only whatever fraction
                    # of the wide uniform(0.2, 1.8) baseline draw happened to
                    # land below ~1 m/s -- undertrained relative to jogging
                    # (which has its own dedicated share) and, after enough
                    # continued fine-tuning epochs, visibly degraded into an
                    # asymmetric limp (one leg stepping, the other trailing)
                    # even though the command distribution nominally still
                    # covered it.
                    low_speed_probability=0.25,
                    low_speed_max=0.8,
                    moving_yaw_rate_probability=0.50,
                    yaw_rate_min_abs=0.20,
                    yaw_rate_max_abs=0.90,
                    match_reference_resets=True,
                    reference_match_candidate_count=512,
                    reference_match_command_scales=(0.5, 0.5, 0.5),
                    reference_match_smoothing_window_s=0.2,
                    # A checkpoint supplied to a new experiment name is a
                    # warm start: go directly into targeted deployment
                    # fine-tuning instead of replaying the gait-discovery
                    # curriculum from epoch zero. True resumes use their
                    # frozen saved config and never execute this function.
                    curriculum=None
                    if getattr(args, "checkpoint", None) is not None
                    else [
                        # First discover a stable forward gait. A stationary
                        # policy cannot collect standing or zero-speed command
                        # rewards during this phase.
                        YawRateSteeringCurriculumStageConfig(
                            start_epoch=0,
                            tar_speed_min=0.3,
                            tar_speed_max=1.0,
                            stand_probability=0.0,
                            turn_in_place_probability=0.0,
                            omnidirectional_probability=0.0,
                            forward_heading_range=0.15,
                            moving_yaw_rate_probability=0.0,
                            yaw_rate_min_abs=0.2,
                            yaw_rate_max_abs=0.6,
                            init_start_prob=1.0,
                            advance_lin_vel_reward_threshold=0.60,
                            advance_yaw_rate_reward_threshold=0.80,
                            minimum_epochs=500,
                            # Bootstrap gait discovery with a small explicit
                            # imitation signal (see dof_pos_tracking reward
                            # below) -- AMP's discriminator alone only asks
                            # the policy to look plausible, not to closely
                            # track any specific trajectory. Turned off again
                            # once the policy already walks reasonably (stage
                            # 2 onward) so it doesn't fight the task/style
                            # reward once they're carrying the policy alone.
                            track_reference_pose=True,
                        ),
                        # Add gentle curved walking and rare stops/turns after
                        # the forward gait has had time to form.
                        YawRateSteeringCurriculumStageConfig(
                            start_epoch=500,
                            tar_speed_min=0.25,
                            tar_speed_max=1.2,
                            stand_probability=0.03,
                            turn_in_place_probability=0.03,
                            omnidirectional_probability=0.05,
                            forward_heading_range=0.30,
                            backward_probability=0.03,
                            backward_heading_range=0.2,
                            backward_speed_max=0.5,
                            moving_yaw_rate_probability=0.10,
                            yaw_rate_min_abs=0.2,
                            yaw_rate_max_abs=0.6,
                            init_start_prob=0.75,
                            advance_lin_vel_reward_threshold=0.55,
                            advance_yaw_rate_reward_threshold=0.65,
                            minimum_epochs=300,
                            track_reference_pose=True,
                        ),
                        # Broaden lateral direction, turning, and speed without
                        # jumping directly to the final joystick distribution.
                        YawRateSteeringCurriculumStageConfig(
                            start_epoch=1000,
                            tar_speed_min=0.2,
                            tar_speed_max=1.4,
                            stand_probability=0.06,
                            turn_in_place_probability=0.08,
                            omnidirectional_probability=0.15,
                            forward_heading_range=0.45,
                            backward_probability=0.06,
                            backward_heading_range=0.3,
                            backward_speed_max=0.8,
                            moving_yaw_rate_probability=0.22,
                            yaw_rate_min_abs=0.2,
                            yaw_rate_max_abs=0.9,
                            init_start_prob=0.5,
                            advance_lin_vel_reward_threshold=0.50,
                            advance_yaw_rate_reward_threshold=0.55,
                            high_speed_probability=0.15,
                            high_speed_min=1.1,
                            low_speed_probability=0.15,
                            low_speed_max=0.5,
                            minimum_epochs=400,
                            track_reference_pose=True,
                        ),
                        # Final deployment distribution.
                        YawRateSteeringCurriculumStageConfig(
                            start_epoch=1500,
                            tar_speed_min=0.2,
                            tar_speed_max=1.8,
                            stand_probability=0.10,
                            turn_in_place_probability=0.15,
                            omnidirectional_probability=0.25,
                            forward_heading_range=0.60,
                            backward_probability=0.25,
                            backward_heading_range=0.3,
                            backward_speed_max=0.7,
                            high_speed_probability=0.25,
                            high_speed_min=1.2,
                            low_speed_probability=0.25,
                            low_speed_max=0.8,
                            moving_yaw_rate_probability=0.50,
                            yaw_rate_min_abs=0.20,
                            yaw_rate_max_abs=0.90,
                            init_start_prob=0.5,
                            track_reference_pose=True,
                        ),
                    ],
                ),
            )
        },
        observation_components=observation_components,
        reward_components=reward_components,
        termination_components={
            "fallen": anchor_height_floor_term_factory(min_height=0.3),
            # Ends the episode when a disallowed body (hands, knees, torso --
            # anything but feet, per non_termination_contact_bodies in
            # configure_robot_and_simulator) is both touching the ground and
            # near it. anchor_height_floor_term alone let the robot brace on
            # its hands to keep its pelvis above the 0.3m floor indefinitely;
            # this catches that "fall and crawl" directly instead of relying
            # on the anchor height, which a braced posture can stay above.
            "unwanted_ground_contact": fall_termination_factory(
                termination_height=0.15
            ),
        },
        # Match the physically consistent teacher. A unit policy action maps
        # to effort_limit / stiffness around the R1 default standing pose.
        action_config=make_bm_pd_action_config(
            robot_cfg,
            effort_fraction=1.0,
            disabled_dof_names=["head_pitch_joint", "head_yaw_joint"],
        ),
        motion_manager=MotionManagerConfig(
            init_start_prob=1.0,
        ),
    )


def agent_config(
    robot_config: RobotConfig,
    env_config: EnvConfig,
    args: argparse.Namespace,
) -> AMPAgentConfig:
    """Reuse the established steering AMP actor, critic, and discriminator.

    r1_steering4 learned plausible stepping and survival but largely ignored
    commands. Its unconditioned discriminator sampled spins, backward motion,
    and sprints while the gait curriculum requested slow forward motion. This
    version conditions AMP on steering, matches expert/reset frames to each
    desired command, and makes task tracking dominate gait discovery.

    PPO actor/critic learning rates are adapted around a KL target. The AMP
    discriminator keeps its own fixed learning rate because its optimization
    is not governed by the policy KL.
    """
    from protomotions.agents.amp.config import AMPCommandMatchingConfig
    from protomotions.agents.ppo.config import AdaptiveLRConfig, SymmetryLossConfig
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.utils.reference_command_matching import (
        reference_commands_at_times,
    )

    agent_cfg = base_steering.agent_config(robot_config, env_config, args)
    # The base steering policy uses logstd=-2.9 (sigma ~= 0.055), which is
    # unnecessarily conservative for discovering a gait from scratch. Keep the
    # noise fixed, but raise it to sigma ~= 0.10 for R1 locomotion exploration.
    agent_cfg.model.actor.actor_logstd = -2.3
    actor_keys = ["proprio", "steering", "previous_actions"]
    critic_keys = [
        "max_coords_obs",
        "historical_max_coords_obs",
        "steering",
        "previous_actions",
    ]
    agent_cfg.model.actor.in_keys = actor_keys
    agent_cfg.model.actor.mu_model.in_keys = actor_keys
    agent_cfg.model.critic.in_keys = critic_keys
    agent_cfg.model.disc_critic.in_keys = [
        "max_coords_obs",
        "historical_max_coords_obs",
        "steering",
    ]
    agent_cfg.model.disc_critic.models[0].in_keys = [
        "max_coords_obs",
        "historical_max_coords_obs",
        "steering",
    ]
    agent_cfg.model.discriminator.in_keys = [
        "historical_max_coords_obs",
        "steering",
    ]
    agent_cfg.model.discriminator.models[0].in_keys = [
        "historical_max_coords_obs",
        "steering",
    ]
    agent_cfg.model.in_keys = [
        "proprio",
        "steering",
        "previous_actions",
        "max_coords_obs",
        "historical_max_coords_obs",
    ]
    agent_cfg.amp_parameters.discriminator_reward_threshold = 0.0
    # Gait discovery must prioritize following the command. Conditional AMP
    # remains a style regularizer instead of being the dominant objective.
    agent_cfg.task_reward_w = 0.5
    agent_cfg.amp_parameters.discriminator_reward_w = 0.5
    agent_cfg.amp_parameters.discriminator_grad_penalty = 10.0
    agent_cfg.amp_parameters.command_matching = AMPCommandMatchingConfig(
        observation_key="steering",
        candidate_count=512,
        command_scales=(0.5, 0.5, 0.5),
        smoothing_window_s=0.2,
        # Prevents the discriminator from separating real/fake purely on
        # whether the command looks self-consistent with the motion (expert
        # transitions always carry their true command; agent transitions
        # early in training often don't yet), which would trivialize its
        # task and leave the style reward carrying no useful gradient.
        command_dropout_prob=0.15,
    )
    agent_cfg.reference_obs_components["steering"] = MdpComponent(
        compute_func=reference_commands_at_times,
        dynamic_vars={},
        static_params={
            "anchor_body_index": robot_config.anchor_body_index,
            "smoothing_window_s": 0.2,
        },
        compile=False,
    )
    agent_cfg.model.discriminator_optimizer.lr = 1e-5
    agent_cfg.adaptive_lr = AdaptiveLRConfig(
        enabled=True,
        desired_kl=0.008,
        min_lr=1.0e-6,
        max_lr=1.0e-4,
    )
    # Penalizes the actor for responding differently to a left-right-mirrored
    # copy of its own observation, targeting the asymmetric limp gait
    # (big lead-leg stride, other leg just catching up to neutral) that a
    # pure average-speed tracking reward doesn't discourage. coef is a
    # starting point -- raise it if the limp persists, lower it if command
    # tracking degrades. previous_actions_history_steps must match
    # previous_actions_factory(history_steps=4, ...) above.
    # Raised from 1.0: r1_steering9 resurfaced the limp specifically at slow
    # speed, the regime that was also the most undersampled in the command
    # mixture (see low_speed_probability above) -- a plausible reason the
    # symmetry gradient wasn't enough to suppress it there even though it
    # kept jogging clean.
    agent_cfg.symmetry = SymmetryLossConfig(
        enabled=True,
        coef=2.0,
        previous_actions_history_steps=4,
    )
    # The legacy evaluator reports one aggregate velocity result. Add
    # overlapping command buckets so regressions in backward walking, jogging,
    # or high-yaw tracking remain visible even when forward walking is strong.
    from protomotions.agents.evaluators.config import SteeringEvaluatorConfig

    base_evaluator = agent_cfg.evaluator
    agent_cfg.evaluator = SteeringEvaluatorConfig(
        evaluation_components=base_evaluator.evaluation_components,
        max_eval_steps=base_evaluator.max_eval_steps,
        eval_metrics_every=base_evaluator.eval_metrics_every,
        eval_metrics_at_epochs=base_evaluator.eval_metrics_at_epochs,
        high_speed_threshold=1.2,
        high_yaw_rate_threshold=0.7,
        linear_velocity_tolerance=0.35,
        yaw_rate_tolerance=0.25,
    )
    return agent_cfg


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg: EnvConfig,
    agent_cfg: AMPAgentConfig,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args: argparse.Namespace,
):
    base_steering.apply_inference_overrides(
        robot_cfg,
        simulator_cfg,
        env_cfg,
        agent_cfg,
        terrain_cfg,
        motion_lib_cfg,
        scene_lib_cfg,
        args,
    )
    from protomotions.envs.component_factories import reduced_coords_obs_factory

    simulator_cfg.domain_randomization = None
    robot_cfg.reset_noise = None
    steering_cfg = env_cfg.control_components.get("steering")
    if steering_cfg is not None and hasattr(
        steering_cfg.command_source, "curriculum"
    ):
        # Inference and evaluation outside the trainer should expose the final
        # joystick command range rather than restarting at curriculum stage 0.
        steering_cfg.command_source.curriculum = None
    if env_cfg.motion_manager is not None:
        # This only chooses clip-start versus random-time reference resets.
        # Keep evaluation representative of the later training phases.
        env_cfg.motion_manager.init_start_prob = 0.5
    env_cfg.observation_components["proprio"] = reduced_coords_obs_factory(
        use_noisy=False,
        root_height_obs=False,
        root_vel_obs=False,
    )
