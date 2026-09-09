# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Steering control component for locomotion tasks.

Manages target direction and speed state for steering tasks.
The target direction and speed change periodically to encourage versatile locomotion.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext, SteeringContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.envs.control.steering_command_window import (
    SteeringCommandWindow,
)
from protomotions.utils import rotations
from protomotions.utils.reference_command_matching import (
    sample_command_matched_motions,
)
from protomotions.utils.hydra_replacement import get_class
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarkerConfig,
    MarkerState,
)

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class SteeringCommandSourceConfig:
    """Base configuration for steering command sources."""

    _target_: str = (
        "protomotions.envs.control.steering_control.RandomSteeringCommandSource"
    )


@dataclass
class RandomSteeringCommandSourceConfig(SteeringCommandSourceConfig):
    """Use the training-time random steering sampler."""


@dataclass
class YawRateSteeringCurriculumStageConfig:
    """Command distribution activated at ``start_epoch``."""

    start_epoch: int = 0
    tar_speed_min: float = 0.2
    tar_speed_max: float = 1.6
    stand_probability: float = 0.10
    turn_in_place_probability: float = 0.15
    omnidirectional_probability: float = 0.25
    forward_heading_range: float = 0.60
    # Backward walking is otherwise only reachable through the uniform
    # omnidirectional slice (a thin sliver near +-pi out of the full circle),
    # so it stays chronically undertrained relative to forward walking even
    # once omnidirectional commands are unlocked. This gives it its own
    # explicit probability, heading cone (around pi, mirroring
    # forward_heading_range's cone around 0), and speed cap -- reference
    # motion libraries typically have much sparser, slower backward-walking
    # clips than forward ones, so commanding fast backward motion invites
    # poor AMP command matches on top of the undertraining itself.
    backward_probability: float = 0.0
    backward_heading_range: float = 0.3
    backward_speed_max: Optional[float] = None
    # Force a configurable share of non-backward moving commands into the
    # upper end of the speed range. A plain uniform draw leaves relatively few
    # samples at jogging speed, especially after stand/turn/backward modes are
    # accounted for.
    high_speed_probability: float = 0.0
    high_speed_min: Optional[float] = None
    # Mirrors high_speed_probability/high_speed_min at the other end of the
    # range. Without this, "slow walking" is only the bottom slice of the
    # same wide uniform draw that high_speed_probability already pulls mass
    # away from -- it gets no dedicated share, so it stays chronically
    # undertrained relative to both jogging and the mixed-speed middle,
    # exactly as high_speed_probability was added to fix the symmetric
    # problem at the top of the range.
    low_speed_probability: float = 0.0
    low_speed_max: Optional[float] = None
    moving_yaw_rate_probability: float = 0.35
    yaw_rate_min_abs: float = 0.20
    yaw_rate_max_abs: float = 1.20
    init_start_prob: Optional[float] = None
    advance_lin_vel_reward_threshold: Optional[float] = None
    advance_yaw_rate_reward_threshold: Optional[float] = None
    # Minimum time spent in this stage after it is entered. This prevents a
    # delayed stage from making all already-due later stages collapse into a
    # handful of epochs while their tracking EMA is still unrepresentative.
    minimum_epochs: int = 0
    # AMP's discriminator reward only asks the policy to look statistically
    # plausible -- it never requires matching any specific reference
    # trajectory. When True, a reference clip is matched to whatever command
    # is currently active (reusing the same candidate search as
    # match_reference_resets) and advanced in sync with elapsed real time;
    # dof_pos_tracking_rew_factory then gives a small explicit DeepMimic-style
    # joint-angle MSE reward against it. Meant for early curriculum stages
    # only -- set False again once the policy already tracks reasonably, so
    # this scaffolding doesn't fight the discriminator/task reward once
    # they're carrying the policy on their own.
    track_reference_pose: bool = False


@dataclass
class RandomYawRateSteeringCommandSourceConfig(SteeringCommandSourceConfig):
    """Sample deployable body-frame velocity and yaw-rate commands.

    When ``curriculum`` is provided, its stages replace the fields below at
    their configured epochs. The top-level fields remain the final inference
    distribution and the fallback for non-curriculum experiments.
    """

    _target_: str = (
        "protomotions.envs.control.steering_control."
        "RandomYawRateSteeringCommandSource"
    )
    stand_probability: float = 0.10
    turn_in_place_probability: float = 0.15
    omnidirectional_probability: float = 0.25
    forward_heading_range: float = 0.60
    backward_probability: float = 0.0
    backward_heading_range: float = 0.3
    backward_speed_max: Optional[float] = None
    high_speed_probability: float = 0.0
    high_speed_min: Optional[float] = None
    low_speed_probability: float = 0.0
    low_speed_max: Optional[float] = None
    moving_yaw_rate_probability: float = 0.35
    yaw_rate_min_abs: float = 0.20
    yaw_rate_max_abs: float = 1.20
    track_reference_pose: bool = False
    curriculum: Optional[List[YawRateSteeringCurriculumStageConfig]] = None
    tracking_metric_ema_alpha: float = 0.01
    tracking_metric_std: float = 0.5
    match_reference_resets: bool = False
    reference_match_candidate_count: int = 64
    reference_match_command_scales: Tuple[float, float, float] = (
        0.5,
        0.5,
        0.5,
    )
    reference_match_smoothing_window_s: float = 0.2


@dataclass
class KeyboardSteeringCommandSourceConfig(SteeringCommandSourceConfig):
    """Interactive keyboard steering configuration.

    W/S/A/D select movement, Q/E turn, X stops turning (or aligns legacy
    facing), Space stops, and -/= decrease/increase speed. Commands affect the
    viewer's active environment.
    """

    _target_: str = (
        "protomotions.envs.control.steering_control.KeyboardSteeringCommandSource"
    )
    default_speed: float = 1.0
    default_yaw_rate: float = 0.6
    speed_increment: float = 0.25
    facing_turn_increment: float = 0.2617993877991494  # 15 degrees
    fail_if_headless: bool = True


@dataclass
class WindowSteeringCommandSourceConfig(KeyboardSteeringCommandSourceConfig):
    """Steering input from a dedicated simulator-independent window."""

    _target_: str = (
        "protomotions.envs.control.steering_control."
        "WindowSteeringCommandSource"
    )
    fail_if_headless: bool = False
    window_title: str = "ProtoMotions Steering"
    window_refresh_interval_ms: int = 50
    status_update_interval_s: float = 0.1
    startup_timeout_s: float = 5.0


@dataclass
class SteeringControlConfig(ControlComponentConfig):
    """Configuration for steering control component.

    Attributes:
        tar_speed_min: Minimum target speed.
        tar_speed_max: Maximum target speed.
        heading_change_steps_min: Minimum steps between heading changes.
        heading_change_steps_max: Maximum steps between heading changes.
        random_heading_probability: Probability of fully random heading vs incremental change.
        standard_heading_change: Maximum incremental heading change (radians).
        standard_speed_change: Maximum incremental speed change.
        stop_probability: Probability of setting speed to zero.
        enable_rand_facing: Enable independent random facing direction (for strafing, etc).
        yaw_rate_commands: Use body-frame planar velocity and signed yaw rate
            instead of legacy world direction and facing targets.
    """

    _target_: str = "protomotions.envs.control.steering_control.SteeringControl"

    tar_speed_min: float = 0.0
    tar_speed_max: float = 2.0
    heading_change_steps_min: int = 50
    heading_change_steps_max: int = 150
    random_heading_probability: float = 0.1
    standard_heading_change: float = 0.5  # radians
    standard_speed_change: float = 0.5
    stop_probability: float = 0.1
    enable_rand_facing: bool = True
    yaw_rate_commands: bool = False
    command_source: SteeringCommandSourceConfig = field(
        default_factory=RandomSteeringCommandSourceConfig
    )


class SteeringCommandSource:
    """Source of commands for :class:`SteeringControl`."""

    def __init__(
        self, config: SteeringCommandSourceConfig, control: "SteeringControl"
    ):
        self.config = config
        self.control = control

    def reset(self, env_ids: Tensor) -> None:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def on_epoch_end(self, current_epoch: int) -> None:
        pass

    def get_state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass

    def sample_reference_reset(self, env_ids: Tensor, motion_manager):
        return None


class RandomSteeringCommandSource(SteeringCommandSource):
    """Sample direction, speed, and facing commands during training."""

    def reset(self, env_ids: Tensor) -> None:
        control = self.control
        n = len(env_ids)
        device = control.env.device

        rand_probs = (
            torch.ones(n, device=device) * control.config.random_heading_probability
        )
        use_random = torch.bernoulli(rand_probs).bool()

        rand_dir_theta = 2 * np.pi * torch.rand(n, device=device) - np.pi
        rand_tar_speed = (
            control.config.tar_speed_max - control.config.tar_speed_min
        ) * torch.rand(n, device=device) + control.config.tar_speed_min

        dir_delta_theta = (
            2 * control.config.standard_heading_change * torch.rand(n, device=device)
            - control.config.standard_heading_change
        )
        inc_dir_theta = (
            dir_delta_theta + control._tar_dir_theta[env_ids] + np.pi
        ) % (2 * np.pi) - np.pi

        speed_delta = (
            2 * control.config.standard_speed_change * torch.rand(n, device=device)
            - control.config.standard_speed_change
        )
        inc_tar_speed = torch.clamp(
            speed_delta + control._tar_speed[env_ids],
            min=control.config.tar_speed_min,
            max=control.config.tar_speed_max,
        )

        dir_theta = torch.where(use_random, rand_dir_theta, inc_dir_theta)
        tar_speed = torch.where(use_random, rand_tar_speed, inc_tar_speed)
        tar_dir = torch.stack([torch.cos(dir_theta), torch.sin(dir_theta)], dim=-1)

        change_steps = torch.randint(
            low=control.config.heading_change_steps_min,
            high=control.config.heading_change_steps_max,
            size=(n,),
            device=device,
            dtype=torch.int64,
        )

        stop_probs = torch.ones(n, device=device) * control.config.stop_probability
        should_stop = torch.bernoulli(stop_probs)

        if control.config.enable_rand_facing:
            face_theta = 2 * np.pi * torch.rand(n, device=device) - np.pi
        else:
            face_theta = dir_theta
        tar_face_dir = torch.stack(
            [torch.cos(face_theta), torch.sin(face_theta)], dim=-1
        )

        control._tar_speed[env_ids] = tar_speed * (1.0 - should_stop)
        control._tar_dir_theta[env_ids] = dir_theta
        control._tar_dir[env_ids] = tar_dir
        control._tar_face_dir[env_ids] = tar_face_dir
        progress = control.env.progress_buf[env_ids]
        is_env_reset = (
            control.env.reset_buf[env_ids] | control.env.terminate_buf[env_ids]
        )
        progress = torch.where(is_env_reset, torch.zeros_like(progress), progress)
        control._heading_change_steps[env_ids] = progress + change_steps

    def step(self) -> None:
        control = self.control
        reset_task_mask = control.env.progress_buf >= control._heading_change_steps
        env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            control.reset(env_ids)


class RandomYawRateSteeringCommandSource(SteeringCommandSource):
    """Sample body-frame planar velocity and signed yaw-rate commands.

    The mixture emphasizes ordinary forward locomotion while retaining explicit
    standing, turning-in-place, omnidirectional, and curved-walking examples.
    """

    config: RandomYawRateSteeringCommandSourceConfig

    def __init__(
        self,
        config: RandomYawRateSteeringCommandSourceConfig,
        control: "SteeringControl",
    ):
        super().__init__(config, control)
        self.config = config
        self._current_epoch = 0
        self._active_stage_index = 0
        self._stage_enter_epoch = 0
        self._tracking_lin_vel_ema = torch.tensor(
            float("nan"), device=control.env.device
        )
        self._tracking_yaw_rate_ema = torch.tensor(
            float("nan"), device=control.env.device
        )
        self._validate_curriculum()
        self._apply_stage_reset_sampling()

    def _validate_distribution(self, distribution) -> None:
        speed_min, speed_max = self._speed_bounds(distribution)
        if speed_min < 0.0:
            raise ValueError("tar_speed_min must be non-negative")
        if speed_min > speed_max:
            raise ValueError("tar_speed_min must be <= tar_speed_max")
        if speed_max > self.control.config.tar_speed_max:
            raise ValueError(
                "curriculum tar_speed_max cannot exceed SteeringControlConfig."
                "tar_speed_max"
            )

        if not 0.0 <= distribution.stand_probability <= 1.0:
            raise ValueError("stand_probability must be in [0, 1]")
        if not 0.0 <= distribution.turn_in_place_probability <= 1.0:
            raise ValueError("turn_in_place_probability must be in [0, 1]")
        if (
            distribution.stand_probability
            + distribution.turn_in_place_probability
            > 1.0
        ):
            raise ValueError(
                "stand_probability + turn_in_place_probability must be <= 1"
            )
        if not 0.0 <= distribution.omnidirectional_probability <= 1.0:
            raise ValueError("omnidirectional_probability must be in [0, 1]")
        # getattr defaults throughout: configs pickled before these fields
        # existed (older checkpoints) unpickle without them.
        backward_probability = getattr(distribution, "backward_probability", 0.0)
        if not 0.0 <= backward_probability <= 1.0:
            raise ValueError("backward_probability must be in [0, 1]")
        if distribution.omnidirectional_probability + backward_probability > 1.0:
            raise ValueError(
                "omnidirectional_probability + backward_probability must be <= 1"
            )
        backward_heading_range = getattr(distribution, "backward_heading_range", 0.0)
        if not 0.0 <= backward_heading_range <= np.pi:
            raise ValueError("backward_heading_range must be in [0, pi]")
        backward_speed_max = getattr(distribution, "backward_speed_max", None)
        if backward_speed_max is not None and not (
            speed_min <= backward_speed_max <= speed_max
        ):
            raise ValueError(
                "backward_speed_max must be within [tar_speed_min, tar_speed_max] "
                "for this stage"
            )
        high_speed_probability = getattr(
            distribution, "high_speed_probability", 0.0
        )
        if not 0.0 <= high_speed_probability <= 1.0:
            raise ValueError("high_speed_probability must be in [0, 1]")
        high_speed_min = getattr(distribution, "high_speed_min", None)
        if high_speed_min is not None and not (
            speed_min <= high_speed_min <= speed_max
        ):
            raise ValueError(
                "high_speed_min must be within [tar_speed_min, tar_speed_max] "
                "for this stage"
            )
        low_speed_probability = getattr(distribution, "low_speed_probability", 0.0)
        if not 0.0 <= low_speed_probability <= 1.0:
            raise ValueError("low_speed_probability must be in [0, 1]")
        if low_speed_probability + high_speed_probability > 1.0:
            raise ValueError(
                "low_speed_probability + high_speed_probability must be <= 1"
            )
        low_speed_max = getattr(distribution, "low_speed_max", None)
        if low_speed_max is not None and not (
            speed_min <= low_speed_max <= speed_max
        ):
            raise ValueError(
                "low_speed_max must be within [tar_speed_min, tar_speed_max] "
                "for this stage"
            )
        if not 0.0 <= distribution.moving_yaw_rate_probability <= 1.0:
            raise ValueError("moving_yaw_rate_probability must be in [0, 1]")
        if not 0.0 <= distribution.forward_heading_range <= np.pi:
            raise ValueError("forward_heading_range must be in [0, pi]")
        if not (
            0.0
            <= distribution.yaw_rate_min_abs
            <= distribution.yaw_rate_max_abs
        ):
            raise ValueError(
                "yaw-rate bounds must satisfy 0 <= min_abs <= max_abs"
            )
        init_start_prob = getattr(distribution, "init_start_prob", None)
        if init_start_prob is not None and not (
            0.0 <= init_start_prob <= 1.0
        ):
            raise ValueError("init_start_prob must be in [0, 1]")
        if getattr(distribution, "minimum_epochs", 0) < 0:
            raise ValueError("minimum_epochs must be non-negative")
        for name in (
            "advance_lin_vel_reward_threshold",
            "advance_yaw_rate_reward_threshold",
        ):
            threshold = getattr(distribution, name, None)
            if threshold is not None and not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def _speed_bounds(self, distribution) -> Tuple[float, float]:
        return (
            getattr(
                distribution,
                "tar_speed_min",
                self.control.config.tar_speed_min,
            ),
            getattr(
                distribution,
                "tar_speed_max",
                self.control.config.tar_speed_max,
            ),
        )

    def _validate_curriculum(self) -> None:
        self._validate_distribution(self.config)
        if not 0.0 < self.config.tracking_metric_ema_alpha <= 1.0:
            raise ValueError("tracking_metric_ema_alpha must be in (0, 1]")
        if self.config.tracking_metric_std <= 0.0:
            raise ValueError("tracking_metric_std must be positive")
        if self.config.reference_match_candidate_count < 1:
            raise ValueError("reference_match_candidate_count must be positive")
        if self.config.reference_match_smoothing_window_s < 0.0:
            raise ValueError(
                "reference_match_smoothing_window_s must be non-negative"
            )
        stages = self.config.curriculum
        if not stages:
            return
        if stages[0].start_epoch != 0:
            raise ValueError("yaw-rate curriculum must start at epoch 0")
        start_epochs = [stage.start_epoch for stage in stages]
        if any(epoch < 0 for epoch in start_epochs):
            raise ValueError("curriculum start epochs must be non-negative")
        if start_epochs != sorted(set(start_epochs)):
            raise ValueError("curriculum start epochs must be strictly increasing")
        for stage in stages:
            self._validate_distribution(stage)

    @property
    def active_distribution(self):
        stages = self.config.curriculum
        if not stages:
            return self.config
        return stages[self._active_stage_index]

    def _stage_index_for_epoch(self, current_epoch: int) -> int:
        stages = self.config.curriculum
        if not stages:
            return 0
        stage_index = 0
        for index, stage in enumerate(stages):
            if current_epoch < stage.start_epoch:
                break
            stage_index = index
        return stage_index

    def record_tracking_rewards(
        self,
        lin_vel_reward: Tensor,
        yaw_rate_reward: Tensor,
    ) -> None:
        alpha = self.config.tracking_metric_ema_alpha
        lin_mean = lin_vel_reward.mean()
        yaw_mean = yaw_rate_reward.mean()
        if torch.isnan(self._tracking_lin_vel_ema):
            self._tracking_lin_vel_ema.copy_(lin_mean)
            self._tracking_yaw_rate_ema.copy_(yaw_mean)
        else:
            self._tracking_lin_vel_ema.lerp_(lin_mean, alpha)
            self._tracking_yaw_rate_ema.lerp_(yaw_mean, alpha)

    def _stage_performance_is_sufficient(self) -> bool:
        distribution = self.active_distribution
        thresholds = (
            (
                distribution.advance_lin_vel_reward_threshold,
                self._tracking_lin_vel_ema,
            ),
            (
                distribution.advance_yaw_rate_reward_threshold,
                self._tracking_yaw_rate_ema,
            ),
        )
        for threshold, metric in thresholds:
            if threshold is None:
                continue
            if torch.isnan(metric) or float(metric) < threshold:
                return False
        return True

    def _apply_stage_reset_sampling(self) -> None:
        distribution = self.active_distribution
        init_start_prob = getattr(distribution, "init_start_prob", None)
        motion_manager = getattr(self.control.env, "motion_manager", None)
        if init_start_prob is None or motion_manager is None:
            return
        motion_manager.config.init_start_prob = init_start_prob
        motion_manager.init_start_probs.fill_(init_start_prob)

    def sample_reference_reset(self, env_ids: Tensor, motion_manager):
        if not self.config.match_reference_resets:
            return None
        target_commands = torch.cat(
            (
                self.control._tar_local_vel[env_ids],
                self.control._tar_yaw_rate[env_ids, None],
            ),
            dim=-1,
        )
        init_start_probability = getattr(
            self.active_distribution,
            "init_start_prob",
            None,
        )
        if init_start_probability is None:
            init_start_probability = motion_manager.config.init_start_prob
        motion_ids, motion_times, matched_commands = sample_command_matched_motions(
            motion_manager,
            target_commands=target_commands,
            anchor_body_index=self.control.env.robot_config.anchor_body_index,
            candidate_count=self.config.reference_match_candidate_count,
            command_scales=self.config.reference_match_command_scales,
            smoothing_window_s=self.config.reference_match_smoothing_window_s,
            init_start_probability=init_start_probability,
        )
        self.control._reference_reset_match_mae = torch.mean(
            torch.abs(matched_commands - target_commands),
            dim=0,
        )
        return motion_ids, motion_times

    def reset(self, env_ids: Tensor) -> None:
        control = self.control
        distribution = self.active_distribution
        speed_min, speed_max = self._speed_bounds(distribution)
        n = len(env_ids)
        device = control.env.device

        mode = torch.rand(n, device=device)
        is_stand = mode < distribution.stand_probability
        is_turn_in_place = (
            mode >= distribution.stand_probability
        ) & (
            mode
            < distribution.stand_probability
            + distribution.turn_in_place_probability
        )
        is_moving = ~(is_stand | is_turn_in_place)

        # getattr defaults: configs pickled before these fields existed
        # (older checkpoints) unpickle without them.
        backward_probability = getattr(distribution, "backward_probability", 0.0)
        backward_heading_range = getattr(distribution, "backward_heading_range", 0.0)
        backward_speed_max = getattr(distribution, "backward_speed_max", None)
        if backward_speed_max is None:
            backward_speed_max = speed_max
        high_speed_probability = getattr(
            distribution, "high_speed_probability", 0.0
        )
        high_speed_min = getattr(distribution, "high_speed_min", None)
        if high_speed_min is None:
            high_speed_min = speed_max
        low_speed_probability = getattr(distribution, "low_speed_probability", 0.0)
        low_speed_max = getattr(distribution, "low_speed_max", None)
        if low_speed_max is None:
            low_speed_max = speed_min

        heading_mode = torch.rand(n, device=device)
        use_omnidirectional = heading_mode < distribution.omnidirectional_probability
        use_backward = (~use_omnidirectional) & (
            heading_mode
            < distribution.omnidirectional_probability + backward_probability
        )

        omni_theta = 2.0 * np.pi * torch.rand(n, device=device) - np.pi
        forward_theta = distribution.forward_heading_range * (
            2.0 * torch.rand(n, device=device) - 1.0
        )
        # Cone around pi (walking backward), mirroring forward_theta's cone
        # around 0 (walking forward). cos/sin below don't need this wrapped
        # back into [-pi, pi].
        backward_theta = np.pi + backward_heading_range * (
            2.0 * torch.rand(n, device=device) - 1.0
        )
        move_theta = torch.where(
            use_omnidirectional,
            omni_theta,
            torch.where(use_backward, backward_theta, forward_theta),
        )

        # Backward commands get their own (typically lower) speed ceiling:
        # reference motion libraries are usually much sparser and slower
        # walking backward than forward, so commanding fast backward motion
        # invites poor AMP command matches on top of the undertraining itself.
        env_speed_max = torch.where(
            use_backward,
            torch.full((n,), float(backward_speed_max), device=device),
            torch.full((n,), float(speed_max), device=device),
        )
        env_speed_max = torch.clamp(env_speed_max, min=speed_min)
        speed = (
            env_speed_max - speed_min
        ) * torch.rand(n, device=device) + speed_min
        # A single roll split into [0, high) / [high, high+low) / rest keeps
        # the two overrides mutually exclusive so their probabilities are
        # exact, instead of two independent Bernoulli draws that could both
        # fire (and silently let one clobber the other via where-ordering).
        speed_mode_roll = torch.rand(n, device=device)
        use_high_speed = (
            is_moving & ~use_backward & (speed_mode_roll < high_speed_probability)
        )
        use_low_speed = (
            is_moving
            & ~use_backward
            & (speed_mode_roll >= high_speed_probability)
            & (speed_mode_roll < high_speed_probability + low_speed_probability)
        )
        high_speed = (
            speed_max - high_speed_min
        ) * torch.rand(n, device=device) + high_speed_min
        low_speed = (
            low_speed_max - speed_min
        ) * torch.rand(n, device=device) + speed_min
        speed = torch.where(use_high_speed, high_speed, speed)
        speed = torch.where(use_low_speed, low_speed, speed)
        speed = torch.where(is_moving, speed, torch.zeros_like(speed))
        local_velocity = torch.stack(
            (speed * torch.cos(move_theta), speed * torch.sin(move_theta)),
            dim=-1,
        )

        yaw_magnitude = (
            distribution.yaw_rate_max_abs - distribution.yaw_rate_min_abs
        ) * torch.rand(n, device=device) + distribution.yaw_rate_min_abs
        yaw_sign = torch.where(
            torch.rand(n, device=device) < 0.5,
            -torch.ones(n, device=device),
            torch.ones(n, device=device),
        )
        use_moving_yaw = (
            torch.rand(n, device=device)
            < distribution.moving_yaw_rate_probability
        )
        use_yaw = is_turn_in_place | (is_moving & use_moving_yaw)
        yaw_rate = torch.where(
            use_yaw, yaw_sign * yaw_magnitude, torch.zeros_like(yaw_magnitude)
        )

        control.set_local_velocity_command(env_ids, local_velocity, yaw_rate)
        change_steps = torch.randint(
            low=control.config.heading_change_steps_min,
            high=control.config.heading_change_steps_max,
            size=(n,),
            device=device,
            dtype=torch.int64,
        )
        progress = control.env.progress_buf[env_ids]
        is_env_reset = (
            control.env.reset_buf[env_ids] | control.env.terminate_buf[env_ids]
        )
        progress = torch.where(is_env_reset, torch.zeros_like(progress), progress)
        control._heading_change_steps[env_ids] = progress + change_steps

        if getattr(distribution, "track_reference_pose", False):
            motion_manager = getattr(control.env, "motion_manager", None)
            if motion_manager is not None:
                target_commands = torch.cat(
                    (local_velocity, yaw_rate.unsqueeze(-1)), dim=-1
                )
                tracked_motion_ids, tracked_times, _ = (
                    sample_command_matched_motions(
                        motion_manager,
                        target_commands=target_commands,
                        anchor_body_index=control.env.robot_config.anchor_body_index,
                        candidate_count=self.config.reference_match_candidate_count,
                        command_scales=self.config.reference_match_command_scales,
                        smoothing_window_s=self.config.reference_match_smoothing_window_s,
                        # Looking for the closest ongoing matching motion to
                        # track, not specifically its first frame.
                        init_start_probability=0.0,
                    )
                )
                control.set_tracked_reference(
                    env_ids, tracked_motion_ids, tracked_times, progress
                )

    def step(self) -> None:
        control = self.control
        all_env_ids = torch.arange(
            control.env.num_envs, device=control.env.device, dtype=torch.long
        )
        control._update_world_direction_from_local(all_env_ids)
        reset_task_mask = control.env.progress_buf >= control._heading_change_steps
        env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            control.reset(env_ids)

    def on_epoch_end(self, current_epoch: int) -> None:
        self._current_epoch = int(current_epoch)
        stages = self.config.curriculum
        if not stages or self._active_stage_index >= len(stages) - 1:
            return
        next_stage_index = self._active_stage_index + 1
        minimum_epochs = getattr(self.active_distribution, "minimum_epochs", 0)
        if self._current_epoch - self._stage_enter_epoch < minimum_epochs:
            return
        if self._current_epoch < stages[next_stage_index].start_epoch:
            return
        if not self._stage_performance_is_sufficient():
            return
        self._active_stage_index = next_stage_index
        self._stage_enter_epoch = self._current_epoch
        self._apply_stage_reset_sampling()
        self._tracking_lin_vel_ema.fill_(float("nan"))
        self._tracking_yaw_rate_ema.fill_(float("nan"))
        env_ids = torch.arange(
            self.control.env.num_envs,
            device=self.control.env.device,
            dtype=torch.long,
        )
        self.control.reset(env_ids)

    def get_state_dict(self) -> dict:
        return {
            "current_epoch": self._current_epoch,
            "active_stage_index": self._active_stage_index,
            "stage_enter_epoch": self._stage_enter_epoch,
            "tracking_lin_vel_ema": self._tracking_lin_vel_ema.clone(),
            "tracking_yaw_rate_ema": self._tracking_yaw_rate_ema.clone(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._current_epoch = int(state_dict.get("current_epoch", 0))
        stages = self.config.curriculum
        max_stage_index = max(0, len(stages) - 1) if stages else 0
        self._active_stage_index = min(
            int(state_dict.get("active_stage_index", 0)),
            max_stage_index,
        )
        self._stage_enter_epoch = int(
            state_dict.get("stage_enter_epoch", self._current_epoch)
        )
        self._tracking_lin_vel_ema.copy_(
            torch.as_tensor(
                state_dict.get("tracking_lin_vel_ema", float("nan")),
                device=self.control.env.device,
            )
        )
        self._tracking_yaw_rate_ema.copy_(
            torch.as_tensor(
                state_dict.get("tracking_yaw_rate_ema", float("nan")),
                device=self.control.env.device,
            )
        )
        self._apply_stage_reset_sampling()


class KeyboardSteeringCommandSource(SteeringCommandSource):
    """Drive one active environment with registered simulator UI keys."""

    config: KeyboardSteeringCommandSourceConfig

    def __init__(
        self,
        config: KeyboardSteeringCommandSourceConfig,
        control: "SteeringControl",
    ):
        super().__init__(config, control)
        self.config = config
        if config.fail_if_headless and control.env.simulator.headless:
            raise RuntimeError(
                "Keyboard steering command source requires a non-headless simulator"
            )
        bindings = control.env.simulator.user_interface.scope("steering_control")
        bindings.register("W", "move_forward", "Move forward")
        bindings.register("S", "move_backward", "Move backward")
        bindings.register("A", "move_left", "Strafe left")
        bindings.register("D", "move_right", "Strafe right")
        bindings.register("Q", "face_left", "Turn left")
        bindings.register("E", "face_right", "Turn right")
        bindings.register("X", "align_facing", "Stop turning / align facing")
        bindings.register("SPACE", "stop", "Stop movement")
        bindings.register("-", "slower", "Decrease target speed")
        bindings.register("=", "faster", "Increase target speed")
        self._bindings = bindings

    def reset(self, env_ids: Tensor) -> None:
        control = self.control
        if control.config.yaw_rate_commands:
            control.set_local_velocity_command(
                env_ids,
                local_velocity=torch.zeros(
                    len(env_ids), 2, device=control.env.device
                ),
                yaw_rate=torch.zeros(len(env_ids), device=control.env.device),
            )
            control._heading_change_steps[env_ids] = torch.iinfo(torch.int64).max
            return
        forward = torch.tensor(
            [[1.0, 0.0]], device=control.env.device, dtype=torch.float32
        ).expand(len(env_ids), -1)
        control.set_heading_relative_command(
            env_ids,
            movement_direction=forward,
            speed=torch.zeros(len(env_ids), device=control.env.device),
            facing_direction=forward,
        )
        control._heading_change_steps[env_ids] = torch.iinfo(torch.int64).max

    def step(self) -> None:
        ui = self.control.env.simulator.user_interface
        env_id = int(ui.active_env_id)
        if env_id < 0 or env_id >= self.control.env.num_envs:
            raise IndexError(
                f"Active env id {env_id} is outside [0, {self.control.env.num_envs})"
            )
        actions = (
            (self._bindings.move_forward, "move_forward"),
            (self._bindings.move_backward, "move_backward"),
            (self._bindings.move_left, "move_left"),
            (self._bindings.move_right, "move_right"),
            (self._bindings.stop, "stop"),
            (self._bindings.slower, "slower"),
            (self._bindings.faster, "faster"),
            (self._bindings.align_facing, "align_facing"),
            (self._bindings.face_left, "face_left"),
            (self._bindings.face_right, "face_right"),
        )
        for handle, action in actions:
            if handle.consume():
                self._apply_action(action, env_id)

        if self.control.config.yaw_rate_commands:
            env_ids = torch.tensor([env_id], device=self.control.env.device)
            self.control._update_world_direction_from_local(env_ids)

    def _apply_action(self, action: str, env_id: int) -> None:
        control = self.control
        env_ids = torch.tensor([env_id], device=control.env.device)
        local_directions = {
            "move_forward": (1.0, 0.0),
            "move_backward": (-1.0, 0.0),
            "move_left": (0.0, 1.0),
            "move_right": (0.0, -1.0),
        }
        if action in local_directions:
            direction = local_directions[action]
            speed = max(float(control._tar_speed[env_id]), self.config.default_speed)
            if control.config.yaw_rate_commands:
                local_direction = torch.tensor(
                    [direction], device=control.env.device
                )
                control.set_local_velocity_command(
                    env_ids,
                    local_velocity=local_direction * speed,
                    yaw_rate=control._tar_yaw_rate[env_ids],
                )
            else:
                control.set_heading_relative_command(
                    env_ids,
                    movement_direction=torch.tensor(
                        [direction], device=control.env.device
                    ),
                    speed=torch.tensor([speed], device=control.env.device),
                    facing_direction=None,
                )
        elif action == "stop":
            self.control._tar_speed[env_id] = 0.0
            if self.control.config.yaw_rate_commands:
                self.control._tar_local_vel[env_id] = 0.0
                self.control._tar_yaw_rate[env_id] = 0.0
        elif action == "slower":
            self._adjust_speed(env_id, -self.config.speed_increment)
        elif action == "faster":
            self._adjust_speed(env_id, self.config.speed_increment)
        elif action == "align_facing":
            if self.control.config.yaw_rate_commands:
                self.control._tar_yaw_rate[env_id] = 0.0
            else:
                self.control._tar_face_dir[env_id] = self.control._tar_dir[env_id]
        elif action == "face_left":
            if self.control.config.yaw_rate_commands:
                self.control._tar_yaw_rate[env_id] = self.config.default_yaw_rate
            else:
                self._rotate_facing(env_id, self.config.facing_turn_increment)
        elif action == "face_right":
            if self.control.config.yaw_rate_commands:
                self.control._tar_yaw_rate[env_id] = -self.config.default_yaw_rate
            else:
                self._rotate_facing(env_id, -self.config.facing_turn_increment)
        else:
            raise ValueError(f"Unknown steering window action: {action}")

    def _adjust_speed(self, env_id: int, increment: float) -> None:
        control = self.control
        new_speed = torch.clamp(
            control._tar_speed[env_id] + increment,
            min=0.0,
            max=control.config.tar_speed_max,
        )
        if control.config.yaw_rate_commands:
            direction = control._tar_local_vel[env_id]
            norm = torch.linalg.norm(direction)
            if norm > 1.0e-6:
                control._tar_local_vel[env_id] = direction / norm * new_speed
            elif new_speed > 0.0:
                control._tar_local_vel[env_id, 0] = new_speed
            control._tar_speed[env_id] = new_speed
        else:
            control._tar_speed[env_id] = new_speed

    def _rotate_facing(self, env_id: int, angle: float) -> None:
        facing = self.control._tar_face_dir[env_id].clone()
        cos_angle = float(np.cos(angle))
        sin_angle = float(np.sin(angle))
        self.control._tar_face_dir[env_id, 0] = (
            cos_angle * facing[0] - sin_angle * facing[1]
        )
        self.control._tar_face_dir[env_id, 1] = (
            sin_angle * facing[0] + cos_angle * facing[1]
        )

    def close(self) -> None:
        self._bindings.unregister_all()


class WindowSteeringCommandSource(KeyboardSteeringCommandSource):
    """Drive one environment from a dedicated Tk command panel."""

    config: WindowSteeringCommandSourceConfig

    def __init__(
        self,
        config: WindowSteeringCommandSourceConfig,
        control: "SteeringControl",
    ):
        SteeringCommandSource.__init__(self, config, control)
        self.config = config
        if config.status_update_interval_s <= 0.0:
            raise ValueError("status_update_interval_s must be positive")
        self._window = SteeringCommandWindow(
            title=config.window_title,
            refresh_interval_ms=config.window_refresh_interval_ms,
            startup_timeout_s=config.startup_timeout_s,
        )
        self._next_status_update = 0.0

    def step(self) -> None:
        ui = self.control.env.simulator.user_interface
        env_id = int(ui.active_env_id)
        if env_id < 0 or env_id >= self.control.env.num_envs:
            raise IndexError(
                f"Active env id {env_id} is outside [0, {self.control.env.num_envs})"
            )
        for action in self._window.poll_actions():
            self._apply_action(action, env_id)

        if self.control.config.yaw_rate_commands:
            env_ids = torch.tensor([env_id], device=self.control.env.device)
            self.control._update_world_direction_from_local(env_ids)

        now = time.monotonic()
        if now >= self._next_status_update:
            self._window.update_status(self._status(env_id))
            self._next_status_update = now + self.config.status_update_interval_s

    def _status(self, env_id: int) -> Dict[str, object]:
        control = self.control
        local_velocity = control._tar_local_vel[env_id]
        move_direction = control._tar_dir[env_id]
        face_direction = control._tar_face_dir[env_id]
        return {
            "env_id": env_id,
            "mode": (
                "body velocity + yaw rate"
                if control.config.yaw_rate_commands
                else "world direction + facing"
            ),
            "local_vx": float(local_velocity[0]),
            "local_vy": float(local_velocity[1]),
            "speed": float(control._tar_speed[env_id]),
            "yaw_rate": float(control._tar_yaw_rate[env_id]),
            "move_heading_deg": float(
                torch.rad2deg(torch.atan2(move_direction[1], move_direction[0]))
            ),
            "face_heading_deg": float(
                torch.rad2deg(torch.atan2(face_direction[1], face_direction[0]))
            ),
        }

    def close(self) -> None:
        self._window.close()


class SteeringControl(ControlComponent):
    """Steering control component that manages target direction and speed.

    Provides target direction and speed that change periodically during training
    to encourage versatile locomotion. Exposes state via get_context() for
    observation and reward functions.

    Args:
        config: Steering control configuration.
        env: Parent environment instance.
    """

    def __init__(self, config: SteeringControlConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self.config: SteeringControlConfig = config

        # Task state buffers
        self._heading_change_steps = torch.zeros(
            self.env.num_envs, device=self.env.device, dtype=torch.int64
        )
        self._tar_dir_theta = torch.zeros(
            self.env.num_envs, device=self.env.device, dtype=torch.float
        )
        self._tar_dir = torch.zeros(
            self.env.num_envs, 2, device=self.env.device, dtype=torch.float
        )
        self._tar_dir[..., 0] = 1.0  # Default: forward direction

        # Target facing direction (2D) - can be different from tar_dir for strafing
        self._tar_face_dir = torch.zeros(
            self.env.num_envs, 2, device=self.env.device, dtype=torch.float
        )
        self._tar_face_dir[..., 0] = 1.0  # Default: forward direction

        self._tar_speed = torch.ones(
            self.env.num_envs, device=self.env.device, dtype=torch.float
        )
        self._tar_local_vel = torch.zeros(
            self.env.num_envs, 2, device=self.env.device, dtype=torch.float
        )
        self._tar_yaw_rate = torch.zeros(
            self.env.num_envs, device=self.env.device, dtype=torch.float
        )
        self._commands_sampled_before_reset = torch.zeros(
            self.env.num_envs, device=self.env.device, dtype=torch.bool
        )
        self._reference_reset_match_mae = torch.full(
            (3,), float("nan"), device=self.env.device
        )

        # DeepMimic-style joint-angle tracking target (see track_reference_pose):
        # the reference clip/time matched to whatever command is currently
        # active, advanced in sync with elapsed real time since that command
        # was set. Populated in the command source's reset() (fires on both
        # env reset and mid-episode command changes) and advanced in
        # populate_context(). Values are meaningless when track_reference_pose
        # is False for the active stage -- populate_context falls back to the
        # agent's own current pose in that case.
        self._tracked_motion_ids = torch.zeros(
            self.env.num_envs, device=self.env.device, dtype=torch.long
        )
        self._tracked_motion_start_time = torch.zeros(
            self.env.num_envs, device=self.env.device, dtype=torch.float
        )
        self._tracked_command_progress = torch.zeros(
            self.env.num_envs, device=self.env.device, dtype=torch.long
        )

        # Double buffer for root position (prev = t-1, curr = t)
        # Allows correct velocity computation in rewards
        self._prev_root_pos = torch.zeros(
            self.env.num_envs, 3, device=self.env.device, dtype=torch.float
        )
        self._curr_root_pos = torch.zeros(
            self.env.num_envs, 3, device=self.env.device, dtype=torch.float
        )
        source_config = getattr(
            config, "command_source", RandomSteeringCommandSourceConfig()
        )
        source_class = get_class(source_config._target_)
        self.command_source = source_class(source_config, self)

    def reset(self, env_ids: Tensor):
        """Reset steering task for given environments."""
        if len(env_ids) == 0:
            return

        robot_state = self.env.simulator.get_robot_state()
        anchor_body_index = self.env.robot_config.anchor_body_index
        anchor_pos = robot_state.rigid_body_pos[env_ids, anchor_body_index]
        self._prev_root_pos[env_ids] = anchor_pos
        self._curr_root_pos[env_ids] = anchor_pos
        commands_already_sampled = self._commands_sampled_before_reset[env_ids]
        if torch.any(~commands_already_sampled):
            self.command_source.reset(env_ids[~commands_already_sampled])
        if torch.any(commands_already_sampled) and self.config.yaw_rate_commands:
            self._update_world_direction_from_local(
                env_ids[commands_already_sampled]
            )
        self._commands_sampled_before_reset[env_ids] = False

    def before_env_reset(self, env_ids: Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.command_source.reset(env_ids)
        self._commands_sampled_before_reset[env_ids] = True

    def sample_reference_reset(self, env_ids: Tensor, motion_manager):
        return self.command_source.sample_reference_reset(
            env_ids,
            motion_manager,
        )

    def set_tracked_reference(
        self,
        env_ids: Tensor,
        motion_ids: Tensor,
        motion_times: Tensor,
        progress: Tensor,
    ) -> None:
        """Record the reference clip/time to track for track_reference_pose.

        Called from the command source whenever a new command is set (both
        full env resets and mid-episode command changes), so the tracked
        reference always matches whatever command is currently active.

        Args:
            env_ids: Environments whose command just changed.
            motion_ids: Matched reference motion IDs for those environments.
            motion_times: Matched reference start times for those environments.
            progress: episode progress_buf value at the moment the command
                was set (0 for a full reset), used to advance motion_times in
                sync with elapsed real time in populate_context().
        """
        self._tracked_motion_ids[env_ids] = motion_ids
        self._tracked_motion_start_time[env_ids] = motion_times
        self._tracked_command_progress[env_ids] = progress

    def step(self):
        """Check if any environments need their heading task updated."""
        # Rotate double buffer: prev <- curr, curr <- new position
        self._prev_root_pos[:] = self._curr_root_pos
        robot_state = self.env.simulator.get_robot_state()
        anchor_body_index = self.env.robot_config.anchor_body_index
        self._curr_root_pos[:] = robot_state.rigid_body_pos[:, anchor_body_index]

        record_tracking_rewards = getattr(
            self.command_source,
            "record_tracking_rewards",
            None,
        )
        if record_tracking_rewards is not None:
            anchor_rot = robot_state.rigid_body_rot[:, anchor_body_index]
            heading_inv = rotations.calc_heading_quat_inv(
                anchor_rot,
                w_last=True,
            )
            local_velocity = rotations.quat_rotate(
                heading_inv,
                robot_state.rigid_body_vel[:, anchor_body_index],
                w_last=True,
            )[:, :2]
            std = self.command_source.config.tracking_metric_std
            lin_vel_reward = torch.exp(
                -torch.sum(
                    torch.square(local_velocity - self._tar_local_vel),
                    dim=-1,
                )
                / (std * std)
            )
            yaw_rate_reward = torch.exp(
                -torch.square(
                    robot_state.rigid_body_ang_vel[:, anchor_body_index, 2]
                    - self._tar_yaw_rate
                )
                / (std * std)
            )
            record_tracking_rewards(lin_vel_reward, yaw_rate_reward)

        self.command_source.step()
        extras = getattr(self.env, "extras", None)
        if isinstance(extras, dict):
            extras["steering_target_speed"] = self._tar_speed
            extras["steering_target_yaw_rate"] = self._tar_yaw_rate
            extras["steering_target_yaw_rate_abs"] = torch.abs(
                self._tar_yaw_rate
            )
            stage_index = getattr(
                self.command_source, "_active_stage_index", 0
            )
            extras["steering_curriculum_stage"] = torch.tensor(
                float(stage_index), device=self.env.device
            )
            lin_vel_ema = getattr(
                self.command_source,
                "_tracking_lin_vel_ema",
                None,
            )
            yaw_rate_ema = getattr(
                self.command_source,
                "_tracking_yaw_rate_ema",
                None,
            )
            if lin_vel_ema is not None:
                extras["steering_tracking_lin_vel_ema"] = lin_vel_ema
                extras["steering_tracking_yaw_rate_ema"] = yaw_rate_ema
            if torch.all(torch.isfinite(self._reference_reset_match_mae)):
                labels = ("vx", "vy", "yaw_rate")
                for index, label in enumerate(labels):
                    extras[
                        f"steering_reference_reset_match_{label}_mae"
                    ] = self._reference_reset_match_mae[index]

    def close(self) -> None:
        """Release command-source resources such as keyboard bindings."""
        self.command_source.close()

    def on_epoch_end(self, current_epoch: int) -> None:
        self.command_source.on_epoch_end(current_epoch)

    def get_state_dict(self) -> dict:
        buffer_names = (
            "_heading_change_steps",
            "_tar_dir_theta",
            "_tar_dir",
            "_tar_face_dir",
            "_tar_speed",
            "_tar_local_vel",
            "_tar_yaw_rate",
            "_commands_sampled_before_reset",
            "_reference_reset_match_mae",
            "_tracked_motion_ids",
            "_tracked_motion_start_time",
            "_tracked_command_progress",
            "_prev_root_pos",
            "_curr_root_pos",
        )
        return {
            "command_source": self.command_source.get_state_dict(),
            "runtime_buffers": {
                name: getattr(self, name).clone() for name in buffer_names
            },
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.command_source.load_state_dict(
            state_dict.get("command_source", {})
        )
        # Older checkpoints contain only command_source. Runtime buffers are
        # primarily used by BaseEnv.save_state/restore_state to make an
        # evaluation rollout transparent to the following training epoch.
        for name, value in state_dict.get("runtime_buffers", {}).items():
            target = getattr(self, name, None)
            if isinstance(target, Tensor) and target.shape == value.shape:
                target.copy_(value.to(device=target.device, dtype=target.dtype))

    def set_world_command(
        self,
        env_ids: Tensor,
        movement_direction: Tensor,
        speed: Tensor,
        facing_direction: Tensor | None = None,
    ) -> None:
        """Set normalized world-frame movement/facing directions and speed.

        This is the direct external-program interface. A joystick adapter can
        either provide world-frame vectors here or use
        :meth:`set_heading_relative_command` for robot-relative axes.
        """
        env_ids = env_ids.to(device=self.env.device, dtype=torch.long)
        movement_direction = movement_direction.to(
            device=self.env.device, dtype=torch.float32
        )
        speed = speed.to(device=self.env.device, dtype=torch.float32).reshape(-1)
        if movement_direction.ndim == 1:
            movement_direction = movement_direction.unsqueeze(0)
        if movement_direction.shape != (len(env_ids), 2):
            raise ValueError(
                "movement_direction must have shape [len(env_ids), 2], got "
                f"{tuple(movement_direction.shape)}"
            )
        if speed.shape != (len(env_ids),):
            raise ValueError(
                f"speed must have shape [len(env_ids)], got {tuple(speed.shape)}"
            )

        movement_norm = torch.linalg.norm(movement_direction, dim=-1, keepdim=True)
        current_movement = self._tar_dir[env_ids]
        normalized_movement = torch.where(
            movement_norm > 1.0e-6,
            movement_direction / movement_norm.clamp_min(1.0e-6),
            current_movement,
        )
        self._tar_dir[env_ids] = normalized_movement
        self._tar_dir_theta[env_ids] = torch.atan2(
            normalized_movement[:, 1], normalized_movement[:, 0]
        )
        self._tar_speed[env_ids] = speed.clamp(
            min=0.0, max=self.config.tar_speed_max
        )

        if facing_direction is not None:
            facing_direction = facing_direction.to(
                device=self.env.device, dtype=torch.float32
            )
            if facing_direction.ndim == 1:
                facing_direction = facing_direction.unsqueeze(0)
            if facing_direction.shape != (len(env_ids), 2):
                raise ValueError(
                    "facing_direction must have shape [len(env_ids), 2], got "
                    f"{tuple(facing_direction.shape)}"
                )
            facing_norm = torch.linalg.norm(
                facing_direction, dim=-1, keepdim=True
            )
            self._tar_face_dir[env_ids] = torch.where(
                facing_norm > 1.0e-6,
                facing_direction / facing_norm.clamp_min(1.0e-6),
                self._tar_face_dir[env_ids],
            )

    def set_heading_relative_command(
        self,
        env_ids: Tensor,
        movement_direction: Tensor,
        speed: Tensor,
        facing_direction: Tensor | None = None,
    ) -> None:
        """Set movement/facing vectors in each robot's heading frame."""
        env_ids = env_ids.to(device=self.env.device, dtype=torch.long)
        robot_state = self.env.simulator.get_robot_state(env_ids)
        anchor_body_index = self.env.robot_config.anchor_body_index
        anchor_rot = robot_state.rigid_body_rot[:, anchor_body_index]
        heading_rot = rotations.calc_heading_quat(anchor_rot, w_last=True)

        movement_direction = movement_direction.to(
            device=self.env.device, dtype=torch.float32
        )
        if movement_direction.ndim == 1:
            movement_direction = movement_direction.unsqueeze(0)
        movement_3d = torch.cat(
            (
                movement_direction,
                torch.zeros(len(env_ids), 1, device=self.env.device),
            ),
            dim=-1,
        )
        world_movement = rotations.quat_rotate(
            heading_rot, movement_3d, w_last=True
        )[:, :2]

        world_facing = None
        if facing_direction is not None:
            facing_direction = facing_direction.to(
                device=self.env.device, dtype=torch.float32
            )
            if facing_direction.ndim == 1:
                facing_direction = facing_direction.unsqueeze(0)
            facing_3d = torch.cat(
                (
                    facing_direction,
                    torch.zeros(len(env_ids), 1, device=self.env.device),
                ),
                dim=-1,
            )
            world_facing = rotations.quat_rotate(
                heading_rot, facing_3d, w_last=True
            )[:, :2]

        self.set_world_command(
            env_ids,
            movement_direction=world_movement,
            speed=speed,
            facing_direction=world_facing,
        )

    def set_local_velocity_command(
        self,
        env_ids: Tensor,
        local_velocity: Tensor,
        yaw_rate: Tensor,
    ) -> None:
        """Set deployable body-yaw-frame planar velocity and yaw-rate commands."""
        env_ids = env_ids.to(device=self.env.device, dtype=torch.long)
        local_velocity = local_velocity.to(
            device=self.env.device, dtype=torch.float32
        )
        yaw_rate = yaw_rate.to(
            device=self.env.device, dtype=torch.float32
        ).reshape(-1)
        if local_velocity.ndim == 1:
            local_velocity = local_velocity.unsqueeze(0)
        if local_velocity.shape != (len(env_ids), 2):
            raise ValueError(
                "local_velocity must have shape [len(env_ids), 2], got "
                f"{tuple(local_velocity.shape)}"
            )
        if yaw_rate.shape != (len(env_ids),):
            raise ValueError(
                f"yaw_rate must have shape [len(env_ids)], got {tuple(yaw_rate.shape)}"
            )

        speed = torch.linalg.norm(local_velocity, dim=-1)
        scale = torch.where(
            speed > self.config.tar_speed_max,
            self.config.tar_speed_max / speed.clamp_min(1.0e-6),
            torch.ones_like(speed),
        )
        self._tar_local_vel[env_ids] = local_velocity * scale.unsqueeze(-1)
        self._tar_speed[env_ids] = speed.clamp(max=self.config.tar_speed_max)
        self._tar_yaw_rate[env_ids] = yaw_rate
        self._update_world_direction_from_local(env_ids)

    def _update_world_direction_from_local(self, env_ids: Tensor) -> None:
        """Update legacy world-direction state for markers and evaluators."""
        if len(env_ids) == 0:
            return
        robot_state = self.env.simulator.get_robot_state(env_ids)
        anchor_body_index = self.env.robot_config.anchor_body_index
        anchor_rot = robot_state.rigid_body_rot[:, anchor_body_index]
        heading_rot = rotations.calc_heading_quat(anchor_rot, w_last=True)
        local_velocity = self._tar_local_vel[env_ids]
        local_direction = local_velocity / torch.linalg.norm(
            local_velocity, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        local_direction = torch.where(
            self._tar_speed[env_ids].unsqueeze(-1) > 1.0e-6,
            local_direction,
            torch.tensor(
                [1.0, 0.0], device=self.env.device, dtype=torch.float32
            ).expand(len(env_ids), -1),
        )
        local_direction_3d = torch.cat(
            (
                local_direction,
                torch.zeros(len(env_ids), 1, device=self.env.device),
            ),
            dim=-1,
        )
        world_direction = rotations.quat_rotate(
            heading_rot, local_direction_3d, w_last=True
        )[:, :2]
        self._tar_dir[env_ids] = world_direction
        self._tar_dir_theta[env_ids] = torch.atan2(
            world_direction[:, 1], world_direction[:, 0]
        )
        yaw_preview = self._tar_yaw_rate[env_ids]
        heading_forward = torch.zeros(
            len(env_ids), 3, device=self.env.device, dtype=torch.float32
        )
        # Preview one second of the signed yaw-rate command in the blue marker.
        heading_forward[:, 0] = torch.cos(yaw_preview)
        heading_forward[:, 1] = torch.sin(yaw_preview)
        self._tar_face_dir[env_ids] = rotations.quat_rotate(
            heading_rot, heading_forward, w_last=True
        )[:, :2]

    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        """No terminations from steering control."""
        reset_buf = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        terminate_buf = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        return reset_buf, terminate_buf

    def populate_context(self, ctx: EnvContext) -> None:
        """Populate steering-specific view in the EnvContext."""
        env_ids = getattr(ctx, "env_ids", None)
        if env_ids is None:
            tar_dir = self._tar_dir
            tar_dir_theta = self._tar_dir_theta
            tar_speed = self._tar_speed
            tar_face_dir = self._tar_face_dir
            tar_local_vel = self._tar_local_vel
            tar_yaw_rate = self._tar_yaw_rate
            prev_root_pos = self._prev_root_pos
            tracked_motion_ids = self._tracked_motion_ids
            tracked_start_time = self._tracked_motion_start_time
            tracked_progress = self._tracked_command_progress
            progress_buf = self.env.progress_buf
        else:
            tar_dir = self._tar_dir[env_ids]
            tar_dir_theta = self._tar_dir_theta[env_ids]
            tar_speed = self._tar_speed[env_ids]
            tar_face_dir = self._tar_face_dir[env_ids]
            tar_local_vel = self._tar_local_vel[env_ids]
            tar_yaw_rate = self._tar_yaw_rate[env_ids]
            prev_root_pos = self._prev_root_pos[env_ids]
            tracked_motion_ids = self._tracked_motion_ids[env_ids]
            tracked_start_time = self._tracked_motion_start_time[env_ids]
            tracked_progress = self._tracked_command_progress[env_ids]
            progress_buf = self.env.progress_buf[env_ids]

        # Default to the current pose so the optional tracking reward remains
        # a no-op for command sources that do not support reference tracking
        # (notably keyboard inference). Lightweight callers may build a
        # steering-only context without ``current``; keep None in that case.
        distribution = getattr(self.command_source, "active_distribution", None)
        current = getattr(ctx, "current", None)
        ref_dof_pos = getattr(current, "dof_pos", None)
        if distribution is not None:
            # Falls back to the agent's own current pose (trivially zero
            # error) when tracking is off for the active curriculum stage,
            # so dof_pos_tracking_rew_factory can stay wired in
            # reward_components across the whole curriculum as a no-op
            # rather than needing to be added/removed per stage.
            if getattr(distribution, "track_reference_pose", False):
                elapsed = (
                    (progress_buf - tracked_progress).clamp(min=0).float()
                    * self.env.dt
                )
                ref_time = tracked_start_time + elapsed
                motion_lengths = self.env.motion_lib.get_motion_length(
                    tracked_motion_ids
                )
                ref_time = torch.clamp(ref_time, max=motion_lengths)
                ref_dof_pos = self.env.motion_lib.get_motion_state(
                    tracked_motion_ids, ref_time
                ).dof_pos

        ctx.steering = SteeringContext(
            tar_dir=tar_dir,
            tar_dir_theta=tar_dir_theta,
            tar_speed=tar_speed,
            tar_face_dir=tar_face_dir,
            prev_root_pos=prev_root_pos,
            tar_local_vel=tar_local_vel,
            tar_yaw_rate=tar_yaw_rate,
            ref_dof_pos=ref_dof_pos,
        )

    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, VisualizationMarkerConfig]:
        """Create steering direction markers.

        Creates two arrow markers:
        - Red arrow: movement direction (tar_dir)
        - Blue arrow: facing direction (tar_face_dir)
        """
        if headless:
            return {}

        # Movement direction marker (red, like ASE)
        movement_markers = [MarkerConfig(size="regular")]
        movement_markers_cfg = VisualizationMarkerConfig(
            type="arrow", color=(0.8, 0.0, 0.0), markers=movement_markers
        )

        # Facing direction marker (blue, like ASE)
        facing_markers = [MarkerConfig(size="regular")]
        facing_markers_cfg = VisualizationMarkerConfig(
            type="arrow", color=(0.0, 0.0, 0.8), markers=facing_markers
        )

        return {
            "movement_markers": movement_markers_cfg,
            "facing_markers": facing_markers_cfg,
        }

    def get_markers_state(self) -> Dict[str, MarkerState]:
        """Get marker states for visualization."""
        if self.env.simulator.headless:
            return {}

        robot_state = self.env.simulator.get_robot_state()
        anchor_body_index = self.env.robot_config.anchor_body_index
        root_pos = robot_state.rigid_body_pos[:, anchor_body_index]
        heading_axis = torch.zeros_like(root_pos)
        heading_axis[..., -1] = 1.0

        # Movement direction marker position and rotation
        movement_marker_pos = root_pos.clone()
        movement_marker_pos[..., 0:2] += self._tar_dir

        movement_theta = torch.atan2(self._tar_dir[..., 1], self._tar_dir[..., 0])
        movement_rot = rotations.quat_from_angle_axis(
            movement_theta, heading_axis, True
        )

        # Facing direction marker position and rotation
        facing_marker_pos = root_pos.clone()
        facing_marker_pos[..., 0:2] += self._tar_face_dir

        facing_theta = torch.atan2(
            self._tar_face_dir[..., 1], self._tar_face_dir[..., 0]
        )
        facing_rot = rotations.quat_from_angle_axis(
            facing_theta, heading_axis, True
        )

        return {
            "movement_markers": MarkerState(
                translation=movement_marker_pos.view(self.env.num_envs, -1, 3),
                orientation=movement_rot.view(self.env.num_envs, -1, 4),
            ),
            "facing_markers": MarkerState(
                translation=facing_marker_pos.view(self.env.num_envs, -1, 3),
                orientation=facing_rot.view(self.env.num_envs, -1, 4),
            ),
        }
