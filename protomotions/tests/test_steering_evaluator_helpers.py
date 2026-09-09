# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.config import SteeringEvaluatorConfig
from protomotions.agents.evaluators.steering_evaluator import SteeringEvaluator


def test_steering_evaluator_reports_backward_fast_and_high_yaw_buckets(monkeypatch):
    evaluator = object.__new__(SteeringEvaluator)
    evaluator.config = SteeringEvaluatorConfig(
        evaluation_components={"placeholder": object()},
        bucket_warmup_steps=0,
        high_speed_threshold=1.2,
        high_yaw_rate_threshold=0.7,
        linear_velocity_tolerance=0.35,
        yaw_rate_tolerance=0.25,
    )
    evaluator.fabric = SimpleNamespace(device=torch.device("cpu"))
    identity = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    )
    evaluator.agent = SimpleNamespace(
        env=SimpleNamespace(
            context=SimpleNamespace(
                steering=SimpleNamespace(
                    tar_local_vel=torch.tensor([[-0.6, 0.0], [1.5, 0.0]]),
                    tar_yaw_rate=torch.tensor([0.0, 0.8]),
                ),
                current=SimpleNamespace(
                    anchor_rot=identity,
                    anchor_vel=torch.tensor([[-0.5, 0.0, 0.0], [1.3, 0.0, 0.0]]),
                    anchor_ang_vel=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]]),
                ),
            )
        )
    )
    evaluator._bucket_step = 0
    evaluator._bucket_stats = {
        name: torch.zeros(8, dtype=torch.float64)
        for name in (
            "all",
            "stand",
            "turn_in_place",
            "forward",
            "backward",
            "lateral",
            "high_speed",
            "high_yaw_rate",
        )
    }
    monkeypatch.setattr(
        BaseEvaluator,
        "process_eval_results",
        lambda self: ({}, 0.5, 2),
    )

    evaluator._on_episode_step(torch.tensor([0, 1]), {}, torch.zeros(2, 1))
    metrics, score, count = evaluator.process_eval_results()

    assert score == 0.5
    assert count == 2
    assert metrics["eval/bucket/backward/sample_count"] == 1
    assert metrics["eval/bucket/backward/actual_vx_mean"] == pytest.approx(-0.5)
    assert metrics["eval/bucket/high_speed/sample_count"] == 1
    assert metrics["eval/bucket/high_yaw_rate/sample_count"] == 1
    assert metrics["eval/bucket/all/tracking_fraction"] == pytest.approx(1.0)
