# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for reusable observation tensor transforms."""

import math

import torch

from protomotions.envs.obs.common import quaternion_to_tan_norm_and_flatten


def test_quaternion_observation_is_invariant_to_quaternion_sign():
    quaternion = torch.tensor(
        [
            [
                [0.0, 0.0, -0.7, 0.7141428],
                [0.1, -0.2, 0.3, 0.9273618],
            ]
        ],
        dtype=torch.float32,
    )
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)

    observation = quaternion_to_tan_norm_and_flatten(quaternion)
    negated_observation = quaternion_to_tan_norm_and_flatten(-quaternion)

    assert observation.shape == (1, 12)
    assert torch.allclose(observation, negated_observation, atol=1.0e-6)


def test_quaternion_observation_preserves_batch_and_flattens_horizons():
    quaternion = torch.zeros(3, 4, 4)
    quaternion[..., 3] = 1.0

    observation = quaternion_to_tan_norm_and_flatten(quaternion)

    assert observation.shape == (3, 24)
    expected_single_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert torch.equal(observation[0].reshape(4, 6)[0], expected_single_rotation)


def test_quaternion_observation_is_continuous_across_wrapped_yaw():
    half_angle = math.radians(179.0) / 2.0
    before_wrap = torch.tensor(
        [[0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]]
    )
    after_wrap = torch.tensor(
        [[0.0, 0.0, -math.sin(half_angle), math.cos(half_angle)]]
    )

    before_observation = quaternion_to_tan_norm_and_flatten(before_wrap)
    after_observation = quaternion_to_tan_norm_and_flatten(after_wrap)

    # The rotations are only two degrees apart despite the wrapped yaw values.
    assert torch.linalg.vector_norm(after_observation - before_observation) < 0.05
