# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for motion sampling managers."""
from __future__ import annotations

import pytest
import torch

from protomotions.envs.motion_manager.config import (
    AdaptiveBinSamplingConfig,
    MimicMotionManagerConfig,
    MotionManagerConfig,
)
from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager
from protomotions.envs.motion_manager.motion_manager import MotionManager


class _MotionLib:
    def __init__(self):
        self.motion_weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float)
        self.motion_lengths = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float)
        self.motion_file = "motions.yaml"

    def num_motions(self):
        return len(self.motion_weights)


def _manager(
    config: MotionManagerConfig | None = None,
    *,
    num_envs: int = 3,
    env_dt: float = 0.25,
    fixed_motion_ids_per_env: torch.Tensor | None = None,
):
    return MotionManager(
        MotionManagerConfig(init_start_prob=0.0) if config is None else config,
        num_envs=num_envs,
        env_dt=env_dt,
        device=torch.device("cpu"),
        motion_lib=_MotionLib(),
        fixed_motion_ids_per_env=fixed_motion_ids_per_env,
    )


def test_motion_subset_methods_select_expected_available_ids():
    first = _manager(MotionManagerConfig(init_start_prob=0.0, subset_method="first"))
    last = _manager(MotionManagerConfig(init_start_prob=0.0, subset_method="last"))
    listed = _manager(MotionManagerConfig(init_start_prob=0.0, subset_method=[2, 0, 3]))

    assert torch.equal(first.available_motion_ids, torch.tensor([0, 1, 2]))
    assert torch.equal(last.available_motion_ids, torch.tensor([1, 2, 3]))
    assert torch.equal(listed.available_motion_ids, torch.tensor([2, 0, 3]))

    listed.sample_motions(torch.tensor([0, 1, 2]))
    assert torch.equal(listed.motion_ids, torch.tensor([2, 0, 3]))
    assert torch.all(listed.motion_times >= 0.0)
    assert torch.all(
        listed.motion_times
        <= _MotionLib().motion_lengths[listed.motion_ids] - listed.env_dt
    )


def test_motion_subset_random_is_fixed_for_session_and_disabled_when_more_envs_than_motions():
    torch.manual_seed(123)
    random_subset = _manager(
        MotionManagerConfig(init_start_prob=0.0, subset_method="random")
    )
    assert random_subset.available_motion_ids.shape == (3,)
    assert sorted(random_subset.available_motion_ids.tolist()) == sorted(
        set(random_subset.available_motion_ids.tolist())
    )
    assert all(0 <= motion_id < _MotionLib().num_motions() for motion_id in random_subset.available_motion_ids)

    too_many_envs = _manager(
        MotionManagerConfig(init_start_prob=0.0, subset_method="first"),
        num_envs=5,
    )
    assert too_many_envs.available_motion_ids is None


def test_motion_subset_rejects_invalid_methods_and_ids():
    with pytest.raises(ValueError, match="Unknown subset_method"):
        _manager(MotionManagerConfig(subset_method="middle"))

    with pytest.raises(ValueError, match="length .* must equal num_envs"):
        _manager(MotionManagerConfig(subset_method=[0, 1]))

    with pytest.raises(ValueError, match="out of range"):
        _manager(MotionManagerConfig(subset_method=[0, 1, 7]))

    with pytest.raises(ValueError, match="subset_method must be"):
        _manager(MotionManagerConfig(subset_method=object()))


def test_motion_exclusions_merge_config_ids_and_latest_failed_motion_file(tmp_path):
    failed_dir = tmp_path / "failed_motions"
    failed_dir.mkdir()
    (failed_dir / "failed_motions_epoch_1_rank_0.txt").write_text("1\n")
    (failed_dir / "failed_motions_epoch_3_rank_0.txt").write_text("2\n\n")
    (failed_dir / "ignored.txt").write_text("0\n")
    manager = _manager(
        MotionManagerConfig(
            init_start_prob=0.0,
            exclude_motion_ids=[0],
            exclude_motions_file=str(tmp_path),
        )
    )

    assert torch.equal(manager.excluded_motion_ids, torch.tensor([0, 2]))
    manager._apply_motion_exclusions()
    assert torch.equal(manager.motion_weights, torch.tensor([0.0, 0.2, 0.0, 0.4]))

    manager.update_sampling_weights(torch.ones(4))
    assert torch.equal(manager.motion_weights, torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_motion_exclusion_directory_uses_latest_numeric_rank_zero_file(tmp_path):
    failed_dir = tmp_path / "failed_motions"
    failed_dir.mkdir()
    (failed_dir / "failed_motions_epoch_2_rank_0.txt").write_text("1\n")
    (failed_dir / "failed_motions_epoch_10_rank_0.txt").write_text("3\n")
    (failed_dir / "failed_motions_epoch_99_rank_1.txt").write_text("0\n")

    manager = _manager(
        MotionManagerConfig(init_start_prob=0.0, exclude_motions_file=str(tmp_path))
    )

    assert torch.equal(manager.excluded_motion_ids, torch.tensor([3]))


def test_motion_exclusion_file_paths_and_validation(tmp_path, capsys):
    direct_file = tmp_path / "exclude.txt"
    direct_file.write_text("1\n3\n")
    from_file = _manager(
        MotionManagerConfig(init_start_prob=0.0, exclude_motions_file=str(direct_file))
    )
    assert torch.equal(from_file.excluded_motion_ids, torch.tensor([1, 3]))

    missing = _manager(
        MotionManagerConfig(
            init_start_prob=0.0,
            exclude_motions_file=str(tmp_path / "missing.txt"),
        )
    )
    assert missing.excluded_motion_ids is None
    assert "exclude_motions_file not found" in capsys.readouterr().out

    with pytest.raises(ValueError, match="exclude_motion_ids must be"):
        _manager(MotionManagerConfig(exclude_motion_ids="1"))

    with pytest.raises(ValueError, match="out of range"):
        _manager(MotionManagerConfig(exclude_motion_ids=[4]))


def test_empty_motion_exclusion_file_disables_exclusions(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("\n\n")

    manager = _manager(
        MotionManagerConfig(init_start_prob=0.0, exclude_motions_file=str(empty_file))
    )

    assert manager.excluded_motion_ids is None


@pytest.mark.parametrize("failed_motions_setup", ["missing", "empty", "no_rank_zero"])
def test_motion_exclusion_directory_without_rank_zero_failed_file_is_ignored(
    tmp_path, failed_motions_setup, capsys
):
    if failed_motions_setup != "missing":
        failed_dir = tmp_path / "failed_motions"
        failed_dir.mkdir()
        if failed_motions_setup == "no_rank_zero":
            (failed_dir / "failed_motions_epoch_9_rank_1.txt").write_text("2\n")

    manager = _manager(
        MotionManagerConfig(init_start_prob=0.0, exclude_motions_file=str(tmp_path))
    )

    assert manager.excluded_motion_ids is None
    assert "No rank-0 failed motions file found" in capsys.readouterr().out


def test_motion_exclusion_file_parsing_ignores_blank_lines_preserves_file_ids(tmp_path):
    exclude_file = tmp_path / "exclude.txt"
    exclude_file.write_text("\n 2 \n\n0\n2\n")

    manager = _manager(
        MotionManagerConfig(init_start_prob=0.0, exclude_motions_file=str(exclude_file))
    )

    assert torch.equal(manager.excluded_motion_ids, torch.tensor([0, 2]))


def test_motion_exclusion_file_rejects_non_integer_lines(tmp_path):
    exclude_file = tmp_path / "exclude.txt"
    exclude_file.write_text("1\nnot-an-int\n")

    with pytest.raises(ValueError, match="invalid literal"):
        _manager(
            MotionManagerConfig(
                init_start_prob=0.0, exclude_motions_file=str(exclude_file)
            )
        )


def test_sample_motions_honors_fixed_ids_random_sentinels_and_overrides():
    torch.manual_seed(7)
    manager = _manager(fixed_motion_ids_per_env=torch.tensor([2, -1, 1]))

    unique_motion_ids, first_env_indices = manager.get_unique_fixed_motions()
    assert torch.equal(unique_motion_ids, torch.tensor([1, 2]))
    assert torch.equal(first_env_indices, torch.tensor([2, 0]))

    manager.sample_motions(torch.tensor([0, 1, 2]))
    assert manager.motion_ids[0].item() == 2
    assert manager.motion_ids[2].item() == 1
    assert 0 <= manager.motion_ids[1].item() < _MotionLib().num_motions()

    manager.sample_motions(torch.tensor([0, 1, 2]), torch.tensor([3, -1, 0]))
    assert manager.motion_ids[0].item() == 3
    assert manager.motion_ids[2].item() == 0
    assert 0 <= manager.motion_ids[1].item() < _MotionLib().num_motions()

    with pytest.raises(AssertionError, match="same shape"):
        manager.sample_motions(torch.tensor([0, 1]), torch.tensor([0]))


def test_sample_motions_with_subset_resets_only_requested_envs():
    manager = _manager(
        MotionManagerConfig(init_start_prob=1.0, subset_method=[2, 0, 3]),
    )
    manager.motion_ids[:] = torch.tensor([1, 1, 1])
    manager.motion_times[:] = torch.tensor([0.5, 0.5, 0.5])

    manager.sample_motions(torch.tensor([0, 2]))

    assert torch.equal(manager.motion_ids, torch.tensor([2, 1, 3]))
    assert torch.equal(manager.motion_times, torch.tensor([0.0, 0.5, 0.0]))


def test_sample_motions_with_empty_env_ids_is_noop():
    manager = _manager()
    manager.motion_ids[:] = torch.tensor([0, 1, 2])
    manager.motion_times[:] = torch.tensor([0.1, 0.2, 0.3])

    manager.sample_motions(torch.tensor([], dtype=torch.long))

    assert torch.equal(manager.motion_ids, torch.tensor([0, 1, 2]))
    assert torch.allclose(manager.motion_times, torch.tensor([0.1, 0.2, 0.3]))


def test_fixed_motion_ids_empty_cases_and_shape_validation():
    no_fixed = _manager()
    empty_ids, empty_indices = no_fixed.get_unique_fixed_motions()
    assert empty_ids.numel() == 0
    assert empty_indices.numel() == 0

    all_random = _manager(fixed_motion_ids_per_env=torch.tensor([-1, -1, -1]))
    empty_ids, empty_indices = all_random.get_unique_fixed_motions()
    assert empty_ids.numel() == 0
    assert empty_indices.numel() == 0

    with pytest.raises(AssertionError, match="must be of shape"):
        _manager(fixed_motion_ids_per_env=torch.tensor([0, 1]))


def test_sampling_helpers_validate_truncation_and_state_dicts(capsys):
    manager = _manager()
    times = manager.sample_time(torch.tensor([0, 3]), truncate_time=0.5)
    assert torch.all(times >= 0.0)
    assert torch.all(times <= torch.tensor([0.5, 3.5]))

    with pytest.raises(AssertionError):
        manager.sample_time(torch.tensor([0]), truncate_time=-0.1)

    # Over-truncation (truncate_time > clip length) clamps to t=0 rather than
    # raising, so resets on short clips degrade gracefully instead of crashing.
    over_truncated = manager.sample_time(torch.tensor([0]), truncate_time=1.1)
    assert torch.all(over_truncated == 0.0)

    state = manager.get_state_dict()
    assert state["motion_file_name"] == "motions.yaml"
    assert torch.equal(state["motion_weights"], manager.motion_weights)

    manager.load_state_dict(
        {"motion_file_name": "motions.yaml", "motion_weights": torch.ones(4)}
    )
    assert torch.equal(manager.motion_weights, torch.ones(4))

    manager.load_state_dict(
        {"motion_file_name": "other.yaml", "motion_weights": torch.zeros(4)}
    )
    assert "motion file name mismatch" in capsys.readouterr().out
    assert torch.equal(manager.motion_weights, torch.ones(4))


def test_sample_n_motion_ids_applies_exclusions_before_multinomial():
    torch.manual_seed(0)
    manager = _manager(
        MotionManagerConfig(init_start_prob=0.0, exclude_motion_ids=[0, 1, 2]),
        num_envs=2,
    )

    sampled = manager.sample_n_motion_ids(10)

    assert torch.equal(sampled, torch.full((10,), 3))


def test_init_start_probability_forces_zero_start_time():
    manager = _manager(MotionManagerConfig(init_start_prob=1.0), num_envs=2)

    manager.sample_motions(torch.tensor([0, 1]), torch.tensor([1, 3]))

    assert torch.equal(manager.motion_ids, torch.tensor([1, 3]))
    assert torch.equal(manager.motion_times, torch.zeros(2))


def test_mimic_motion_manager_done_tracks_and_post_physics_step():
    manager = MimicMotionManager(
        MimicMotionManagerConfig(init_start_prob=0.0),
        num_envs=3,
        env_dt=0.25,
        device=torch.device("cpu"),
        motion_lib=_MotionLib(),
    )
    manager.motion_ids[:] = torch.tensor([0, 1, 2])
    manager.motion_times[:] = torch.tensor([0.8, 1.0, 2.9])

    assert torch.equal(manager.get_done_tracks(), torch.tensor([True, False, True]))
    assert torch.equal(manager.get_done_tracks(torch.tensor([1, 2])), torch.tensor([False, True]))

    manager.post_physics_step()

    assert torch.allclose(manager.motion_times, torch.tensor([1.05, 1.25, 3.15]))


def test_mimic_motion_manager_reserves_tail_for_future_context():
    manager = MimicMotionManager(
        MimicMotionManagerConfig(init_start_prob=0.0),
        num_envs=3,
        env_dt=0.25,
        device=torch.device("cpu"),
        motion_lib=_MotionLib(),
    )
    manager.reserve_future_context_steps(2)

    # The source lengths remain unchanged and available for lookahead queries.
    assert torch.equal(
        manager.motion_lib.motion_lengths,
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    assert torch.equal(
        manager.get_motion_end_times(),
        torch.tensor([0.5, 1.5, 2.5, 3.5]),
    )

    manager.motion_ids[:] = torch.tensor([0, 1, 2])
    manager.motion_times[:] = torch.tensor([0.24, 1.0, 2.24])
    assert torch.equal(manager.get_done_tracks(), torch.tensor([False, False, False]))
    manager.motion_times[:] += 0.01
    assert torch.equal(manager.get_done_tracks(), torch.tensor([True, False, True]))

    sampled = manager.sample_time(torch.tensor([1, 2, 3]), truncate_time=0.25)
    assert torch.all(sampled >= 0.0)
    assert torch.all(sampled <= torch.tensor([1.25, 2.25, 3.25]))


def test_mimic_motion_manager_resamples_only_done_tracks_when_configured():
    torch.manual_seed(11)
    manager = MimicMotionManager(
        MimicMotionManagerConfig(init_start_prob=0.0, resample_on_reset=False),
        num_envs=3,
        env_dt=0.25,
        device=torch.device("cpu"),
        motion_lib=_MotionLib(),
    )
    manager.motion_ids[:] = torch.tensor([0, 1, 2])
    manager.motion_times[:] = torch.tensor([0.1, 1.8, 1.0])

    manager.sample_motions(torch.tensor([0, 1, 2]))

    assert manager.motion_ids[0].item() == 0
    assert manager.motion_times[0].item() == pytest.approx(0.1)
    assert manager.motion_ids[2].item() == 2
    assert manager.motion_times[2].item() == pytest.approx(1.0)
    assert 0 <= manager.motion_ids[1].item() < _MotionLib().num_motions()
    assert manager.motion_times[1].item() != pytest.approx(1.8)


def test_mimic_motion_manager_returns_when_no_tracks_are_done():
    manager = MimicMotionManager(
        MimicMotionManagerConfig(init_start_prob=0.0, resample_on_reset=False),
        num_envs=2,
        env_dt=0.25,
        device=torch.device("cpu"),
        motion_lib=_MotionLib(),
    )
    manager.motion_ids[:] = torch.tensor([0, 1])
    manager.motion_times[:] = torch.tensor([0.1, 0.2])

    manager.sample_motions(torch.tensor([0, 1]))

    assert torch.equal(manager.motion_ids, torch.tensor([0, 1]))
    assert torch.allclose(manager.motion_times, torch.tensor([0.1, 0.2]))


def _adaptive_manager(cfg: AdaptiveBinSamplingConfig, **kwargs):
    kwargs.setdefault("num_envs", 3)
    kwargs.setdefault("env_dt", 0.25)
    return MimicMotionManager(
        MimicMotionManagerConfig(init_start_prob=0.0, adaptive_bin_sampling=cfg),
        device=torch.device("cpu"),
        motion_lib=_MotionLib(),
        **kwargs,
    )


def test_adaptive_bin_layout_matches_motion_lengths():
    # _MotionLib lengths [1.0, 2.0, 3.0, 4.0] with 1s bins -> [1, 2, 3, 4] bins.
    manager = _adaptive_manager(AdaptiveBinSamplingConfig(enabled=True, bin_size_s=1.0))

    assert torch.equal(manager._bins_per_motion, torch.tensor([1, 2, 3, 4]))
    assert torch.equal(manager._bin_starts, torch.tensor([0, 1, 3, 6]))
    assert manager._total_bins == 10
    assert torch.equal(manager._bin_failed_count, torch.zeros(10))


def test_adaptive_bin_layout_ceils_partial_tail_bins():
    manager = _adaptive_manager(AdaptiveBinSamplingConfig(enabled=True, bin_size_s=1.5))
    # lengths [1.0, 2.0, 3.0, 4.0] / 1.5s bins, ceil'd -> [1, 2, 2, 3]
    assert torch.equal(manager._bins_per_motion, torch.tensor([1, 2, 2, 3]))


def test_disabled_adaptive_sampling_skips_bin_setup_and_uses_uniform_time():
    manager = _adaptive_manager(AdaptiveBinSamplingConfig(enabled=False))
    assert not hasattr(manager, "_bin_failed_count")

    manager.record_bin_failures(torch.tensor([0]), torch.tensor([True]))  # no-op, must not raise

    torch.manual_seed(0)
    sampled = manager.sample_time(torch.tensor([3, 3, 3]))
    assert torch.all(sampled >= 0.0) and torch.all(sampled <= 4.0)


def test_record_bin_failures_blends_only_failed_envs_at_their_own_bin():
    manager = _adaptive_manager(
        AdaptiveBinSamplingConfig(enabled=True, bin_size_s=1.0, alpha=0.5)
    )
    # bin_starts = [0, 1, 3, 6]; motion 1 has 2 bins (flat 1,2), motion 3 has 4 (flat 6..9)
    manager.motion_ids[:] = torch.tensor([1, 2, 3])
    manager.motion_times[:] = torch.tensor([1.4, 0.5, 3.9])
    failed = torch.tensor([True, False, True])  # env 1 (motion 2) did NOT fail

    manager.record_bin_failures(torch.tensor([0, 1, 2]), failed)

    expected = torch.zeros(10)
    expected[2] = 0.5  # motion 1, bin 1 (floor(1.4) clamped to its last bin, index 1) -> flat 1+1
    expected[9] = 0.5  # motion 3, bin 3 (floor(3.9)) -> flat 6+3
    assert torch.allclose(manager._bin_failed_count, expected)

    # A second failure at the same bin blends further via the alpha EMA.
    manager.record_bin_failures(torch.tensor([0]), torch.tensor([True]))
    assert manager._bin_failed_count[2].item() == pytest.approx(0.5 * 1 + 0.5 * 0.5)


def test_record_bin_failures_noop_without_any_failures():
    manager = _adaptive_manager(AdaptiveBinSamplingConfig(enabled=True, bin_size_s=1.0))
    manager.motion_ids[:] = torch.tensor([1, 2, 3])
    manager.motion_times[:] = torch.tensor([1.4, 0.5, 3.9])

    manager.record_bin_failures(torch.tensor([0, 1, 2]), torch.tensor([False, False, False]))

    assert torch.equal(manager._bin_failed_count, torch.zeros(10))


def test_sample_adaptive_bins_clamps_dominant_bin_probability():
    manager = _adaptive_manager(
        AdaptiveBinSamplingConfig(
            enabled=True, bin_size_s=1.0, prob_clamp_min_ratio=0.75, prob_clamp_max_ratio=100.0
        ),
        num_envs=1,
    )
    # Synthesize a long (200-bin) motion 3 to make the ratio-relative clamp
    # bind at all: for a short clip, 100x the uniform average exceeds 1.0
    # and never actually restricts anything.
    n_bins = 200
    manager._bins_per_motion = torch.tensor([1, 2, 3, n_bins])
    manager._bin_starts = torch.tensor([0, 1, 3, 6])
    manager._total_bins = 6 + n_bins
    manager._bin_failed_count = torch.zeros(manager._total_bins)
    manager._bin_failed_count[6] = 1000.0  # all recorded failures on motion 3's bin 0

    torch.manual_seed(0)
    samples = torch.cat([manager._sample_adaptive_bins(torch.tensor([3])) for _ in range(4000)])
    freq = torch.bincount(samples, minlength=n_bins).float() / len(samples)

    # Without a clamp bin 0 would take ~100% of draws; with it, capped well
    # under that, and every other bin still gets a floor of nonzero mass.
    assert freq[0].item() < 0.5
    assert freq[0].item() > 0.2
    assert torch.all(freq[1:] > 0)


def test_sample_time_uses_adaptive_bins_and_respects_truncation():
    manager = _adaptive_manager(
        AdaptiveBinSamplingConfig(enabled=True, bin_size_s=1.0), num_envs=1
    )
    manager._bin_failed_count[manager._bin_starts[3]] = 1.0  # motion 3, bin 0

    torch.manual_seed(0)
    sampled = manager.sample_time(torch.tensor([3, 3, 3, 3]), truncate_time=0.1)
    assert torch.all(sampled >= 0.0)
    assert torch.all(sampled <= 4.0 - 0.1 + 1e-6)


def test_reserve_future_context_rebuilds_adaptive_bins_over_trackable_length():
    manager = _adaptive_manager(AdaptiveBinSamplingConfig(enabled=True, bin_size_s=1.0))
    manager._bin_failed_count[:] = 5.0  # would be stale after the tail shrinks bin counts

    manager.reserve_future_context_steps(4)  # 4 * env_dt(0.25) = 1.0s reserved tail
    # lengths [1,2,3,4] - 1.0 tail -> [0, 1, 2, 3] trackable seconds -> ceil -> bins [1,1,2,3]
    assert torch.equal(manager._bins_per_motion, torch.tensor([1, 1, 2, 3]))
    assert manager._total_bins == 7
    assert torch.equal(manager._bin_failed_count, torch.zeros(7))


def test_adaptive_bin_state_dict_round_trip():
    manager = _adaptive_manager(AdaptiveBinSamplingConfig(enabled=True, bin_size_s=1.0))
    manager._bin_failed_count[3] = 2.5

    state = manager.get_state_dict()
    assert torch.allclose(state["adaptive_bin_failed_count"], manager._bin_failed_count)

    restored = _adaptive_manager(AdaptiveBinSamplingConfig(enabled=True, bin_size_s=1.0))
    restored.load_state_dict(state)
    assert torch.allclose(restored._bin_failed_count, manager._bin_failed_count)
