# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for evaluators."""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

from protomotions.envs.mdp_component import MdpComponent


@dataclass
class EvaluatorConfig:
    """Configuration for base evaluator."""

    _target_: str = "protomotions.agents.evaluators.base_evaluator.BaseEvaluator"
    evaluation_components: Dict[str, MdpComponent] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of MdpComponent evaluation metrics for success/failure tracking."}
    )
    max_eval_steps: int = field(
        default=600,
        metadata={"help": "Maximum steps per evaluation episode.", "min": 1}
    )
    eval_metrics_every: Optional[int] = field(
        default=200,
        metadata={"help": "Evaluate metrics every N epochs. None = disabled.", "min": 1}
    )
    eval_metrics_at_epochs: List[int] = field(
        default_factory=list,
        metadata={"help": "Additional one-off epochs at which to run evaluation."},
    )


@dataclass
class SteeringEvaluatorConfig(EvaluatorConfig):
    """Evaluator with command-stratified steering tracking metrics."""

    _target_: str = (
        "protomotions.agents.evaluators.steering_evaluator.SteeringEvaluator"
    )
    bucket_warmup_steps: int = 10
    high_speed_threshold: float = 1.2
    high_yaw_rate_threshold: float = 0.7
    linear_velocity_tolerance: float = 0.35
    yaw_rate_tolerance: float = 0.25


@dataclass
class MotionWeightsRulesConfig:
    """Configuration for motion weights update rule."""

    motion_weights_update_success_discount: float = field(
        default=0.999,
        metadata={"help": "Discount factor for successful motion weights.", "min": 0.0, "max": 1.0}
    )
    motion_weights_update_failure_discount: float = field(
        default=0.999,
        metadata={"help": "Discount for failed motions. 0 = set weight straight to 1.", "min": 0.0, "max": 1.0}
    )
    min_motion_weight: Union[float, str] = field(
        default="1/num_motions",
        metadata={"help": "Minimum weight for any motion. '1/num_motions' or float value."}
    )
    max_motion_weight_ratio: float = field(
        default=20.0,
        metadata={
            "help": (
                "Cap on the hardest motion's sampling weight, as a multiple of "
                "min_motion_weight. The failure branch multiplies weights "
                "without any upper bound, so this is what keeps the curriculum "
                "a re-weighting rather than a takeover."
            ),
            "min": 1.0,
        },
    )


@dataclass
class MimicEvaluatorConfig(EvaluatorConfig):
    """Configuration for Mimic evaluator."""

    _target_: str = "protomotions.agents.evaluators.mimic_evaluator.MimicEvaluator"
    save_predicted_motion_lib_every: Optional[int] = field(
        default=3,
        metadata={"help": "Save pred_motion_lib every M evals. None = disabled.", "min": 1}
    )
    trajectory_export_fps: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional physics-state capture rate for exported predicted motions. "
                "The policy rate is unchanged; supported simulators sample actual "
                "states at intermediate physics substeps."
            ),
            "min": 1,
        },
    )
    save_video_every_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Save a headless actual-vs-reference tracking video every N training epochs.",
            "min": 1,
        },
    )
    save_video_at_epochs: List[int] = field(
        default_factory=list,
        metadata={"help": "Additional one-off epochs at which to save a tracking video."},
    )
    motion_weights_rules: MotionWeightsRulesConfig = field(
        default_factory=MotionWeightsRulesConfig,
        metadata={"help": "Rules for updating motion sampling weights."}
    )
    eval_action_ema_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "EMA smoothing factor for actions during evaluation only. "
                "Simulates deployment low-pass filtering. "
                "a_applied = alpha * a_policy + (1-alpha) * a_prev. "
                "None = disabled (raw actions). Typical values: 0.5-0.8."
                "Smaller alpha = more smoothing."
            ),
            "min": 0.0,
            "max": 1.0,
        }
    )
