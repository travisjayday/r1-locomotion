# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for motion manager components.

This module contains all configuration dataclasses for motion manager functionality,
co-located with the motion manager implementations in the same directory.
"""

from typing import Optional, List, Union
from dataclasses import dataclass, field


@dataclass
class MotionManagerConfig:
    """Configuration for motion management."""

    _target_: str = "protomotions.envs.motion_manager.motion_manager.MotionManager"

    init_start_prob: float = field(
        default=0.2,
        metadata={
            "help": "Probability to sample an initial pose instead of random time. Helps prevent local-minima in AMP.",
            "min": 0.0,
            "max": 1.0,
        }
    )

    fixed_motion_id: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional motion ID assigned to every environment for single-clip training and evaluation.",
            "min": 0,
        },
    )

    sample_time_truncate_s: Optional[float] = field(
        default=None,
        metadata={
            "help": "Optional extra seconds to remove from the end of random reset time sampling.",
            "min": 0.0,
        },
    )

    subset_method: Optional[Union[str, List[int]]] = field(
        default=None,
        metadata={
            "help": "Motion subset for evaluation: 'first', 'last', 'random', or list of motion IDs. None uses all motions.",
            "options": ["first", "last", "random"],
        }
    )

    exclude_motion_ids: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "Motion IDs to exclude from sampling. Useful for removing problematic motions.",
        }
    )

    exclude_motions_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to file with motion IDs to exclude (one per line). Can also be an expert training directory.",
        }
    )

    realign_motion_with_humanoid_on_each_step: bool = field(
        default=False,
        metadata={
            "help": "Realign motion with humanoid each step. Prevents tracking error accumulation for imperfect retargeting.",
        }
    )


@dataclass
class AdaptiveBinSamplingConfig:
    """Within-motion adaptive bin sampling for reset-time curriculum.

    Segments each motion into fixed-length time bins and tracks an
    exponentially-blended failure rate per bin. Reset-time start-time
    sampling is then drawn from a per-motion distribution proportional to
    those failure rates (clamped relative to the motion's uniform average),
    so training concentrates reset attempts on the specific segment of a
    clip that's actually hard rather than the whole clip equally -- e.g. a
    motion that only fails during one fast turn gets many more reset
    attempts starting near that turn, not just more attempts overall.

    Matches the adaptive sampling strategy described in the OmniTrack PMG
    paper (Table A.7: 1s bins, alpha=0.001, clamp to
    [0.75, 100] times the average bin probability).
    """

    enabled: bool = field(
        default=False,
        metadata={"help": "Enable within-motion adaptive bin sampling."},
    )
    bin_size_s: float = field(
        default=1.0,
        metadata={"help": "Bin duration in seconds.", "min": 0.01},
    )
    alpha: float = field(
        default=0.001,
        metadata={
            "help": "EMA blending rate for per-bin failure counts: new = alpha*current_failures + (1-alpha)*old.",
            "min": 0.0,
            "max": 1.0,
        },
    )
    prob_clamp_min_ratio: float = field(
        default=0.75,
        metadata={
            "help": "Lower clamp on a bin's sampling probability, as a multiple of that motion's uniform average (1/num_bins).",
            "min": 0.0,
        },
    )
    prob_clamp_max_ratio: float = field(
        default=100.0,
        metadata={
            "help": "Upper clamp on a bin's sampling probability, as a multiple of that motion's uniform average (1/num_bins).",
            "min": 0.0,
        },
    )


@dataclass
class MimicMotionManagerConfig(MotionManagerConfig):
    """Configuration for mimic motion management."""

    _target_: str = (
        "protomotions.envs.motion_manager.mimic_motion_manager.MimicMotionManager"
    )

    resample_on_reset: bool = field(
        default=True,
        metadata={"help": "Whether to resample motion on environment reset."}
    )

    adaptive_bin_sampling: Optional[AdaptiveBinSamplingConfig] = field(
        default=None,
        metadata={"help": "Optional within-motion adaptive bin sampling for reset-time curriculum."},
    )
