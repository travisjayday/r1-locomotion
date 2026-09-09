# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
from protomotions.envs.motion_manager.motion_manager import MotionManager
from protomotions.components.motion_lib import MotionLib
import torch
from torch import Tensor
from typing import Optional


class MimicMotionManager(MotionManager):
    """Motion manager specialized for mimic environments.

    Extends the base MotionManager to handle mimic-specific motion sampling,
    including time progression and conditional resampling on reset.
    """

    config: MimicMotionManagerConfig

    def __init__(
        self,
        config: MimicMotionManagerConfig,
        num_envs: int,
        env_dt: float,
        device: torch.device,
        motion_lib: MotionLib,
        fixed_motion_ids_per_env: Optional[torch.Tensor] = None,
    ):
        """A motion manager that handles motion sampling and tracking for mimic environments.

        Args:
            config: Configuration object containing motion manager settings
            num_envs (int): Number of parallel environments
            env_dt (float): Environment timestep
            device (torch.device): Device to store tensors on
            motion_lib (MotionLib): Motion library containing reference motions
            fixed_motion_ids_per_env (Optional[torch.Tensor], optional): If provided, specifies fixed motion IDs to use for each environment. Defaults to None.
        """
        super().__init__(config, num_envs, env_dt, device, motion_lib, fixed_motion_ids_per_env)
        self._reserved_tail_s = 0.0
        if self.config.adaptive_bin_sampling is not None and self.config.adaptive_bin_sampling.enabled:
            self._rebuild_adaptive_bins()

    def reserve_future_context_steps(self, num_steps: int) -> None:
        """Reserve a motion tail for future-reference observations.

        The underlying motion remains unchanged and can still be queried by a
        control component. Only reset-time sampling and the episode/export end
        time are shortened.
        """
        if type(num_steps) is not int or num_steps < 0:
            raise ValueError("num_steps must be a non-negative integer")
        self._reserved_tail_s = num_steps * self.env_dt

        # A clip must contain at least one tracked step before its context tail.
        too_short = self.motion_lib.motion_lengths.to(self.device) < (
            self._reserved_tail_s + self.env_dt
        )
        self.motion_weights[too_short] = 0.0

        # Bins are laid out over the trackable (tail-excluded) length, which
        # just shrank; rebuild. Called during env setup before any episodes
        # have run, so resetting accumulated failure counts loses nothing.
        if self.config.adaptive_bin_sampling is not None and self.config.adaptive_bin_sampling.enabled:
            self._rebuild_adaptive_bins()

    def _rebuild_adaptive_bins(self) -> None:
        """(Re)build the flat per-motion bin layout and reset failure counts.

        Bins are packed into one flat tensor per motion_lib's own
        length_starts/motion_num_frames convention: bin b of motion m lives
        at flat index bin_starts[m] + b.
        """
        bin_size_s = self.config.adaptive_bin_sampling.bin_size_s
        lengths = self.get_motion_end_times()
        self._bins_per_motion = torch.ceil(lengths / bin_size_s).long().clamp(min=1)
        shifted = self._bins_per_motion.roll(1)
        shifted[0] = 0
        self._bin_starts = shifted.cumsum(0)
        self._total_bins = int((self._bin_starts[-1] + self._bins_per_motion[-1]).item())
        self._bin_failed_count = torch.zeros(self._total_bins, device=self.device)

    def record_bin_failures(self, env_ids: Tensor, failed: Tensor) -> None:
        """Blend adaptive-sampling failure stats for envs about to be reset.

        Must be called with each env's still-current (pre-reset) motion_id
        and motion_time -- i.e. before sample_motions overwrites them -- and
        a boolean mask of which of those envs terminated due to a genuine
        failure (a termination_component firing), not a timeout or a
        naturally-completed clip.
        """
        cfg = self.config.adaptive_bin_sampling
        if cfg is None or not cfg.enabled or not bool(failed.any()):
            return

        failed_env_ids = env_ids[failed]
        motion_ids = self.motion_ids[failed_env_ids]
        bin_idx = (self.motion_times[failed_env_ids] / cfg.bin_size_s).long()
        bin_idx = bin_idx.clamp(min=0, max=None).clamp(max=self._bins_per_motion[motion_ids] - 1)
        flat_idx = self._bin_starts[motion_ids] + bin_idx

        current_failed = torch.zeros_like(self._bin_failed_count)
        current_failed.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=current_failed.dtype))
        self._bin_failed_count = cfg.alpha * current_failed + (1 - cfg.alpha) * self._bin_failed_count

    def _sample_adaptive_bins(self, motion_ids: Tensor) -> Tensor:
        """Draw one bin index per motion_id from its clamped failure-weighted distribution."""
        cfg = self.config.adaptive_bin_sampling
        n_bins = self._bins_per_motion[motion_ids]  # [B]
        max_bins = int(self._bins_per_motion.max().item())

        bin_offset = torch.arange(max_bins, device=self.device).unsqueeze(0)  # [1, max_bins]
        valid = bin_offset < n_bins.unsqueeze(1)  # [B, max_bins]
        flat_idx = (self._bin_starts[motion_ids].unsqueeze(1) + bin_offset).clamp(max=self._total_bins - 1)
        raw = self._bin_failed_count[flat_idx] * valid  # [B, max_bins]

        uniform_p = 1.0 / n_bins.float().unsqueeze(1)
        total_failed = raw.sum(dim=1, keepdim=True)
        prob = torch.where(total_failed > 0, raw / total_failed.clamp(min=1e-8), uniform_p.expand_as(raw))

        prob = prob.clamp(min=cfg.prob_clamp_min_ratio * uniform_p, max=cfg.prob_clamp_max_ratio * uniform_p)
        prob = torch.where(valid, prob, torch.zeros_like(prob))
        prob = prob / prob.sum(dim=1, keepdim=True)

        return torch.multinomial(prob, num_samples=1).squeeze(-1)

    def get_motion_end_times(
        self, motion_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return usable episode endpoints, excluding reserved context tails."""
        lengths = self.motion_lib.motion_lengths
        if motion_ids is not None:
            lengths = lengths[motion_ids]
        return (lengths - self._reserved_tail_s).clamp(min=0.0)

    def sample_time(
        self, motion_ids: torch.Tensor, truncate_time: Optional[float] = None
    ) -> torch.Tensor:
        """Sample only from the tracked portion, never the reserved tail."""
        max_time = self.get_motion_end_times(motion_ids).clone()
        if truncate_time is not None:
            if truncate_time < 0.0:
                raise ValueError("truncate_time must be non-negative")
            max_time = (max_time - truncate_time).clamp(min=0.0)

        cfg = self.config.adaptive_bin_sampling
        if cfg is not None and cfg.enabled:
            bin_idx = self._sample_adaptive_bins(motion_ids)
            within_bin = torch.rand(motion_ids.shape, device=self.device)
            # truncate_time shaves at most one env_dt (plus an optional small
            # margin) off the trackable length the bins were built over, so
            # clamping the rare bin-in-the-shaved-sliver draw down to
            # max_time is a negligible approximation, not a real bias.
            return ((bin_idx.float() + within_bin) * cfg.bin_size_s).clamp(min=0.0, max=None).minimum(max_time)

        phase = torch.rand(motion_ids.shape, device=self.device)
        return phase * max_time

    def get_done_tracks(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Check which motion tracks have reached their end time.

        Args:
            env_ids: Optional tensor of environment indices to check. If None, checks all environments.

        Returns:
            Boolean tensor indicating which motion tracks are done (True) or still playing (False)
        """
        end_times = self.get_motion_end_times(self.motion_ids)
        done_clip = (self.motion_times + self.env_dt) >= end_times
        if env_ids is not None:
            done_clip = done_clip[env_ids]
        return done_clip

    def post_physics_step(self):
        """Advance motion playback time by one environment timestep.

        Called after each physics simulation step to update the current time
        in each motion track.
        """
        self.motion_times += self.env_dt

    def sample_motions(
        self, env_ids: torch.Tensor, new_motion_ids: Optional[torch.Tensor] = None
    ):
        """Sample new motions for environments.

        Extends base class to handle mimic-specific resample_on_reset logic:
        only resample motions that have finished playing.

        Args:
            env_ids (Tensor): Indices of the environments to reset.
            new_motion_ids (Tensor, optional):
                Force new motion IDs for the reset environments.
                If provided, this overrides fixed motion IDs.
        """
        # Mimic-specific: Only resample motions that have finished if resample_on_reset is False
        reset_env_ids = env_ids
        if not self.config.resample_on_reset:
            done_tracks = self.get_done_tracks(env_ids)
            reset_env_ids = env_ids[done_tracks]

        # Only proceed if there are environments to reset
        if len(reset_env_ids) == 0:
            return

        # Call parent sample_motions (handles fixed motion IDs)
        super().sample_motions(reset_env_ids, new_motion_ids)

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        cfg = self.config.adaptive_bin_sampling
        if cfg is not None and cfg.enabled:
            state_dict["adaptive_bin_failed_count"] = self._bin_failed_count.cpu().clone()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        cfg = self.config.adaptive_bin_sampling
        if cfg is not None and cfg.enabled and "adaptive_bin_failed_count" in state_dict:
            saved = state_dict["adaptive_bin_failed_count"]
            if saved.shape == self._bin_failed_count.shape:
                self._bin_failed_count[:] = saved.to(self.device)
            else:
                print(
                    "Warning: skip loading adaptive bin failure counts due to "
                    f"shape mismatch: {tuple(saved.shape)} != {tuple(self._bin_failed_count.shape)}"
                )
