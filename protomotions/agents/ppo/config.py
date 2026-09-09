# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for PPO agent.

This module defines all configuration dataclasses for the Proximal Policy Optimization (PPO)
algorithm, including actor-critic architecture parameters, optimization settings, and
training hyperparameters.

Key Classes:
    - PPOAgentConfig: Main PPO agent configuration
    - PPOModelConfig: PPO model (actor-critic) configuration
    - PPOActorConfig: Policy network configuration
    - AdvantageNormalizationConfig: Advantage normalization settings
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from protomotions.agents.common.config import (
    ModuleContainerConfig,
)
from protomotions.agents.base_agent.config import (
    OptimizerConfig,
    BaseAgentConfig,
    BaseModelConfig,
)


@dataclass
class PPOActorConfig:
    """Configuration for PPO Actor network."""

    mu_key: str = field(metadata={"help": "The key of the output of the mu model."})
    in_keys: List[str] = field(
        default_factory=list, metadata={"help": "Input observation keys."}
    )
    out_keys: List[str] = field(
        default_factory=lambda: ["action", "mean_action", "neglogp"],
        metadata={"help": "Output keys: action, mean_action, neglogp."},
    )
    _target_: str = "protomotions.agents.ppo.model.PPOActor"
    mu_model: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Neural network model for action mean."},
    )
    num_out: int = field(
        default=None, metadata={"help": "Number of actions. Set from robot config."}
    )
    actor_logstd: float = field(
        default=-2.9, metadata={"help": "Initial log std for action distribution."}
    )
    learnable_std: bool = field(
        default=False,
        metadata={"help": "Make action log std learnable (requires_grad=True)."},
    )
    # Only meaningful when learnable_std=True. A fixed std never changes from
    # actor_logstd, so these bounds are inert for it. With nothing bounding a
    # learnable logstd, a constant entropy bonus (entropy_coef) with no
    # counterpressure can drift it upward indefinitely -- observed in practice
    # as std_mean climbing continuously from epoch 0 (0.10) to >1.5 by epoch
    # 9600 on an R1 mimic run, degrading tracking quality once std got large
    # enough to swamp the training signal. -0.5 (std ~= 0.61) is a generous
    # ceiling for mocap-tracking-style continuous control -- well above
    # typical fixed-std choices in this codebase (-2.3 to -2.9, std
    # ~= 0.055-0.10) but far below where the runaway becomes damaging.
    logstd_min: float = field(
        default=-5.0,
        metadata={"help": "Floor clamp on learnable logstd (numerical-stability hygiene; std ~= 0.0067)."},
    )
    logstd_max: float = field(
        default=-0.5,
        metadata={"help": "Ceiling clamp on learnable logstd, applied in-place after each actor optimizer step."},
    )


@dataclass
class PPOModelConfig(BaseModelConfig):
    """Configuration for PPO Model (Actor-Critic)."""

    _target_: str = "protomotions.agents.ppo.model.PPOModel"
    out_keys: List[str] = field(
        default_factory=lambda: ["action", "mean_action", "neglogp", "value"],
        metadata={"help": "Output keys including actions and value estimate."},
    )
    actor: PPOActorConfig = field(
        default_factory=PPOActorConfig,
        metadata={"help": "Actor (policy) network configuration."},
    )
    critic: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Critic (value) network configuration."},
    )
    actor_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5),
        metadata={"help": "Optimizer settings for actor network."},
    )
    critic_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=1e-4),
        metadata={"help": "Optimizer settings for critic network."},
    )


@dataclass
class AdvantageNormalizationConfig:
    """Configuration for advantage normalization."""

    enabled: bool = field(
        default=True, metadata={"help": "Whether to normalize advantages."}
    )
    shift_mean: bool = field(
        default=True, metadata={"help": "Subtract mean from advantages."}
    )
    # EMA parameters
    use_ema: bool = field(
        default=True, metadata={"help": "Use EMA for normalization statistics."}
    )
    ema_alpha: float = field(
        default=0.05, metadata={"help": "EMA weight for new data."}
    )
    min_std: float = field(
        default=0.02, metadata={"help": "Minimum std to prevent extreme normalization."}
    )
    clamp_range: float = field(
        default=4.0,
        metadata={"help": "Clamp normalized advantages to [-range, range]."},
    )


@dataclass
class AdaptiveLRConfig:
    """Configuration for adaptive learning rate based on KL divergence."""

    enabled: bool = field(
        default=False,
        metadata={"help": "Enable adaptive learning rate based on KL divergence."},
    )
    desired_kl: float = field(
        default=0.01,
        metadata={"help": "Target KL divergence for adaptive learning rate."},
    )
    min_lr: float = field(
        default=1e-5,
        metadata={"help": "Minimum learning rate for both actor and critic."},
    )
    max_lr: float = field(
        default=1e-2,
        metadata={"help": "Maximum learning rate for both actor and critic."},
    )
    # Default 1.0 = no smoothing (use the raw per-epoch KL directly), the
    # exact prior behavior. A single-epoch KL outlier -- e.g. the epoch right
    # after a full-dataset mimic evaluation, whose motion-weight update
    # deliberately reshapes the training distribution -- otherwise crashes
    # the LR by 1.5x on one noisy reading, then spends many epochs climbing
    # back (observed as a recurring sawtooth tied exactly to eval cadence on
    # an R1 mimic run). Values <1 fold the new reading into an EMA before
    # comparing to desired_kl, so one bad epoch no longer swings the LR alone.
    kl_ema_alpha: float = field(
        default=1.0,
        metadata={
            "help": (
                "EMA weight for the newest KL reading before adaptive-LR "
                "thresholding: kl_ema = alpha*kl + (1-alpha)*kl_ema. "
                "1.0 disables smoothing (prior behavior)."
            )
        },
    )


@dataclass
class L2C2Config:
    """L2C2 (Lipschitz-ratio) actor regularization (Kobayashi 2022).

    Penalizes the ratio  ||mu(noisy) - mu(clean)||^2 / ||noisy - clean||^2
    so the actor's Lipschitz constant stays bounded.
    """

    enabled: bool = field(
        default=False, metadata={"help": "Enable L2C2 regularization."}
    )
    lambda_l2c2: float = field(default=0.1, metadata={"help": "L2C2 loss coefficient."})
    obs_pairs: Dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Map from noisy actor obs key to clean counterpart key."},
    )


@dataclass
class SymmetryLossConfig:
    """Left-right (sagittal-plane) symmetry loss for bilaterally-symmetric legged robots.

    Penalizes the actor for producing a different action (after mirroring it
    back) when fed a left-right-mirrored copy of its own observation. This
    targets asymmetric "limp" gaits directly -- e.g. one leg taking a long
    lead stride while the other just catches up to neutral -- a failure mode
    that a plain average-speed tracking reward doesn't discourage, since a
    lopsided gait and a symmetric one can hit the same average speed.

    Requires an env observation from ``symmetry_state_obs_factory`` and a
    robot whose body/DOF names follow the ``left_*``/``right_*`` convention
    (see ``protomotions.utils.mirroring.build_mirror_table``).
    """

    enabled: bool = field(
        default=False, metadata={"help": "Enable the left-right symmetry loss."}
    )
    coef: float = field(
        default=1.0,
        metadata={"help": "Symmetry loss coefficient.", "min": 0.0},
    )
    mirror_state_key: str = field(
        default="symmetry_state",
        metadata={
            "help": "Obs key holding the packed state from symmetry_state_obs_factory."
        },
    )
    proprio_key: str = field(
        default="proprio",
        metadata={"help": "Actor obs key for reduced-coords proprioception."},
    )
    steering_key: str = field(
        default="steering",
        metadata={"help": "Actor obs key for the yaw-rate steering command."},
    )
    previous_actions_key: str = field(
        default="previous_actions",
        metadata={"help": "Actor obs key for the previous-actions history."},
    )
    previous_actions_history_steps: int = field(
        default=1,
        metadata={
            "help": "Must match history_steps passed to previous_actions_factory."
        },
    )
    root_height_obs: bool = field(
        default=False,
        metadata={
            "help": (
                "Must match root_height_obs passed to reduced_coords_obs_factory "
                "for proprio_key. Not yet supported -- enabling raises at "
                "agent construction time."
            )
        },
    )
    root_vel_obs: bool = field(
        default=False,
        metadata={
            "help": (
                "Must match root_vel_obs passed to reduced_coords_obs_factory "
                "for proprio_key. Not yet supported -- enabling raises at "
                "agent construction time."
            )
        },
    )


@dataclass
class PPOAgentConfig(BaseAgentConfig):
    """Main configuration class for PPO Agent."""

    _target_: str = "protomotions.agents.ppo.agent.PPO"

    # Model configuration
    model: PPOModelConfig = field(
        default_factory=PPOModelConfig, metadata={"help": "Model configuration."}
    )

    # PPO hyperparameters
    tau: float = field(
        default=0.95, metadata={"help": "GAE lambda for advantage estimation."}
    )
    e_clip: float = field(
        default=0.2, metadata={"help": "PPO clipping parameter epsilon."}
    )
    clip_critic_loss: bool = field(
        default=True, metadata={"help": "Clip critic loss similar to actor."}
    )

    # Actor update control
    actor_clip_frac_threshold: Optional[float] = field(
        default=0.6,
        metadata={"help": "Skip actor update if clip_frac > threshold (e.g., 0.5)."},
    )

    # Entropy regularization (used when actor has learnable_std). A flat,
    # non-annealed coefficient has no reason to weaken as training matures --
    # it keeps rewarding higher std even once the policy no longer needs the
    # exploration, which is what let std_mean climb unchecked for an entire
    # 9600-epoch R1 mimic run. entropy_coef_final/entropy_coef_decay_epochs
    # linearly decay it toward (near-)zero so exploration pressure fades as
    # training progresses; leaving entropy_coef_decay_epochs unset keeps the
    # old flat-coefficient behavior exactly (default matches prior default).
    entropy_coef: float = field(
        default=0.005,
        metadata={"help": "Initial entropy bonus coefficient for learnable std exploration."},
    )
    entropy_coef_final: float = field(
        default=0.0,
        metadata={"help": "Entropy bonus coefficient after the decay finishes. Unused unless entropy_coef_decay_epochs is set."},
    )
    entropy_coef_decay_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Epochs to linearly decay entropy_coef -> entropy_coef_final over. "
                "None disables annealing (entropy_coef stays constant, prior behavior)."
            )
        },
    )

    # L2C2 regularization
    l2c2: L2C2Config = field(
        default_factory=L2C2Config, metadata={"help": "L2C2 settings."}
    )

    # Left-right symmetry regularization
    symmetry: SymmetryLossConfig = field(
        default_factory=SymmetryLossConfig, metadata={"help": "Symmetry loss settings."}
    )

    # Adaptive learning rate
    adaptive_lr: AdaptiveLRConfig = field(
        default_factory=AdaptiveLRConfig,
        metadata={"help": "Adaptive learning rate settings."},
    )

    # Value normalization
    advantage_normalization: AdvantageNormalizationConfig = field(
        default_factory=AdvantageNormalizationConfig,
        metadata={"help": "Advantage normalization settings."},
    )
