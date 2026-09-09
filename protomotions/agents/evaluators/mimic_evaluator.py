# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.config import MimicEvaluatorConfig
from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.components.motion_lib import MotionLib
from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager

logger = logging.getLogger(__name__)


@dataclass
class MimicEpisodeContext:
    """Per-episode-batch state for mimic evaluation."""

    motion_ids: Tensor  # which motion each env is tracking
    frame_limits: Tensor  # how many frames before clip ends


class MimicEvaluator(BaseEvaluator):
    """Evaluator for Mimic agent's motion tracking performance."""

    def __init__(self, agent: Any, fabric: Any, config: MimicEvaluatorConfig):
        super().__init__(agent, fabric, config)
        self._physics_export_metrics: Optional[Dict[str, MotionMetrics]] = None
        self._physics_export_dt: Optional[float] = None
        self._physics_export_substep_interval: Optional[int] = None
        self._physics_export_actions: Optional[Tensor] = None
        self._physics_export_env_ids: Optional[Tensor] = None
        self._physics_export_motion_ids: Optional[Tensor] = None

    @property
    def motion_lib(self) -> MotionLib:
        """Motion library (from agent)."""
        return self.agent.motion_lib

    @property
    def motion_manager(self) -> MimicMotionManager:
        """Motion manager (from env)."""
        return self.env.motion_manager

    def _get_evaluation_motion_lengths(
        self, motion_ids: Optional[Tensor]
    ) -> Tensor:
        """Return tracked lengths while leaving reserved lookahead data intact."""
        get_end_times = getattr(self.motion_manager, "get_motion_end_times", None)
        if get_end_times is not None:
            return get_end_times(motion_ids)
        return self.motion_lib.get_motion_length(motion_ids)

    def _motion_frame_counts(
        self, motion_lengths: Tensor, dt: Optional[float] = None
    ) -> Tensor:
        """Convert seconds to frames without losing exact grid points to fp32."""
        sample_dt = self.env.dt if dt is None else dt
        return (motion_lengths / sample_dt + 1e-4).floor().long()

    def _register_plugins(self) -> None:
        """Register metric computation plugins."""
        self._register_smoothness_plugin(window_sec=0.4, high_jerk_threshold=6500.0)
        self._register_action_smoothness_plugin()

    def _create_metrics(
        self,
        num_motions: int,
        motion_num_frames: Tensor,
        max_eval_steps: int,
    ) -> Dict[str, MotionMetrics]:
        """Create MotionMetrics buffers for trajectory collection (robot state + actions)."""
        metrics = {}

        self._add_robot_state_metrics(
            metrics, num_motions, motion_num_frames, max_eval_steps
        )

        num_dofs = self.env.robot_config.kinematic_info.num_dofs
        metrics["actions"] = MotionMetrics(
            num_motions, motion_num_frames, max_eval_steps, num_dofs, device=self.device
        )

        return metrics

    def initialize_eval(self) -> Dict:
        """Initialize evaluation tracking and cache env state for restoration."""
        num_motions = self.motion_lib.num_motions()
        motion_lengths = self._get_evaluation_motion_lengths(None)
        motion_num_frames = self._motion_frame_counts(motion_lengths)
        motion_num_frames = motion_num_frames.clamp(max=self.config.max_eval_steps)
        self._init_eval_component_buffers(num_motions)

        # Keep these aliases for direct initialize/cleanup callers. The normal
        # evaluate() entry point additionally has BaseEvaluator's transaction,
        # which is the authoritative full-state restoration path.
        self._env_snapshot = self.env.save_state()
        self._cached_motion_ids = self.motion_manager.motion_ids.clone()
        self._cached_motion_times = self.motion_manager.motion_times.clone()

        metrics = self._create_metrics(
            num_motions, motion_num_frames, self.config.max_eval_steps
        )
        self._initialize_physics_export_metrics(num_motions, motion_lengths)
        return metrics

    def _initialize_physics_export_metrics(
        self, num_motions: int, motion_lengths: Tensor
    ) -> None:
        """Allocate a separate buffer for genuine high-rate physics samples."""
        self._physics_export_metrics = None
        self._physics_export_dt = None
        self._physics_export_substep_interval = None

        export_fps = self.config.trajectory_export_fps
        if export_fps is None:
            return

        physics_fps = int(self.env.simulator.config.sim.fps)
        if physics_fps % export_fps != 0:
            raise ValueError(
                f"trajectory_export_fps={export_fps} must divide physics fps "
                f"{physics_fps} exactly"
            )
        substep_interval = physics_fps // export_fps
        if self.env.simulator.decimation % substep_interval != 0:
            raise ValueError(
                f"Control decimation {self.env.simulator.decimation} must be divisible "
                f"by the export substep interval {substep_interval}"
            )

        export_dt = 1.0 / export_fps
        export_num_frames = self._motion_frame_counts(motion_lengths, export_dt)
        max_export_frames = int(
            self.config.max_eval_steps
            * self.env.simulator.decimation
            // substep_interval
        )
        export_num_frames = export_num_frames.clamp(max=max_export_frames)
        self._physics_export_metrics = self._create_metrics(
            num_motions, export_num_frames, max_export_frames
        )
        self._physics_export_dt = export_dt
        self._physics_export_substep_interval = substep_interval
        logger.info(
            "Physics trajectory export: policy=%.1f Hz, physics=%d Hz, "
            "capture=%d Hz (every %d physics substeps)",
            1.0 / self.env.dt,
            physics_fps,
            export_fps,
            substep_interval,
        )

    def _capture_physics_substep(self, substep_index: int, _physics_dt: float) -> None:
        """Capture an actual simulator state at the configured physics cadence."""
        interval = self._physics_export_substep_interval
        if interval is None or substep_index % interval != 0:
            return
        if self._physics_export_actions is None:
            return

        metrics = self._physics_export_metrics
        env_ids = self._physics_export_env_ids
        motion_ids = self._physics_export_motion_ids
        state = self.env.simulator.get_robot_state(env_ids)
        for key, metric in metrics.items():
            if key == "actions":
                metric.update(
                    motion_ids,
                    self._physics_export_actions[env_ids].detach(),
                )
                continue
            if getattr(state, key, None) is not None:
                values = state.flatten_bodies(key).detach().to(dtype=metric.dtype)
                metric.update(motion_ids, values)

    def _save_failed_motions(self, failed_motions: list, epoch: int) -> None:
        """
        Save list of failed motions to a text file.

        Args:
            failed_motions: List of motion IDs that failed tracking
            epoch: Current epoch number
        """
        filename = f"failed_motions_epoch_{epoch}_rank_{self.fabric.global_rank}.txt"
        self._save_list_to_file(failed_motions, filename, subdirectory="failed_motions")

    def _update_motion_sampling_weights(self) -> None:
        """Update motion sampling weights based on evaluation component failures."""
        if self._motion_failed is None:
            return

        failed_motions = torch.nonzero(self._motion_failed).flatten().tolist()
        success_motions = torch.nonzero(~self._motion_failed).flatten().tolist()

        self._save_failed_motions(failed_motions, self.agent.current_epoch)

        success_discount = math.pow(
            self.config.motion_weights_rules.motion_weights_update_success_discount,
            self.config.eval_metrics_every,
        )
        failure_discount = math.pow(
            self.config.motion_weights_rules.motion_weights_update_failure_discount,
            self.config.eval_metrics_every,
        )
        new_weights = self.env.motion_manager.motion_weights.clone()
        new_weights[success_motions] *= success_discount
        if failure_discount != 0:
            new_weights[failed_motions] /= failure_discount
        else:
            new_weights[failed_motions] = 1.0
        # The failure branch above is unbounded: at the 0.999 default with
        # eval_metrics_every=200 it multiplies a failing motion's weight by
        # 1/0.8186 = 1.222 every evaluation, forever. min_motion_weight bounds
        # the bottom but nothing bounded the top, so r1_locomotion_10 reached a
        # max weight of 991429 against the 0.05 floor by epoch 13400 -- a 2e7
        # dynamic range in which the top 50 clips held 95% of the sampling
        # probability and the effective sample size was 39 of 908. Training
        # rewards kept climbing on that shrinking subset while the full
        # unweighted eval fell from 0.6156 to 0.3877 success: the policy was
        # mastering ~39 motions and forgetting the rest. Cap the ratio so a
        # persistently failing clip is sampled a bounded multiple more often
        # than an easy one, never to the exclusion of the training set.
        floor = self._resolved_min_motion_weight(new_weights.shape[0])
        ceiling = floor * max(
            self.config.motion_weights_rules.max_motion_weight_ratio, 1.0
        )
        new_weights.clamp_(min=floor, max=ceiling)
        self.env.motion_manager.update_sampling_weights(new_weights)

    def _resolved_min_motion_weight(self, num_motions: int) -> float:
        """Resolve ``min_motion_weight``, which may be the string '1/num_motions'.

        Without this floor the success discount compounds without bound: at the
        default 0.999 with ``eval_metrics_every=200`` each update multiplies a
        succeeding motion's weight by 0.82, so after ~80 updates it sits at ~1e-7
        while any motion that failed once is reset to 1.0. Sampling then collapses
        onto whichever handful of clips failed most recently, and every
        distribution-sensitive metric starts tracking that churn rather than the
        policy. The floor keeps the curriculum a re-weighting rather than a
        replacement of the training set.
        """
        min_weight = self.config.motion_weights_rules.min_motion_weight
        if isinstance(min_weight, str):
            if min_weight == "1/num_motions":
                return 1.0 / max(num_motions, 1)
            return float(min_weight)
        return float(min_weight)

    def _park_inactive_envs(self, active_env_ids: Tensor) -> None:
        """Move envs not in ``active_env_ids`` far below the terrain.

        With scene-paired motions, ``_build_eval_batches`` returns a single
        batch whose ``env_ids`` (from ``get_unique_fixed_motions``) can be
        much smaller than ``num_envs`` -- the remaining envs would otherwise
        keep running physics with their (replicated) scenes, contributing
        substantially to the PhysX broadphase pair budget and triggering
        ``foundLostPairsCapacity`` overflow at large num_envs (silent contact
        drops -> tunneling -> phantom failures).

        Parking those envs at z << 0 removes their AABBs from the broadphase
        active region without changing the policy's view of the rollout.
        """
        if active_env_ids is None or active_env_ids.numel() >= self.num_envs:
            return
        all_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        all_mask[active_env_ids] = False
        inactive_env_ids = torch.nonzero(all_mask, as_tuple=False).flatten()
        if inactive_env_ids.numel() == 0:
            return
        self.env.simulator.park_envs(inactive_env_ids)

    def evaluate_episode(self, env_ids: torch.Tensor, max_steps: int) -> None:
        """Run a single episode batch, optionally with EMA action smoothing.

        When eval_action_ema_alpha is set, actions are low-pass filtered to
        simulate deployment conditions. Motions that fail under EMA get higher
        sampling weight, creating curriculum pressure toward smooth policies.
        """
        ema_alpha = self.config.eval_action_ema_alpha

        self._on_episode_start(env_ids)

        # Park envs that aren't part of this batch so they don't generate
        # PhysX broadphase pairs / contacts during the eval. BaseEvaluator's
        # transactional wrapper restores the pre-evaluation state afterward.
        self._park_inactive_envs(env_ids)

        obs, _ = self.env.reset(env_ids, **self._get_reset_kwargs())
        self.agent.pre_collect_step(0)
        obs = self.agent.add_agent_info_to_obs(obs)
        obs_td = self.agent.obs_dict_to_tensordict(obs)

        prev_actions = None
        capture_substeps = self._physics_export_metrics is not None
        if capture_substeps:
            self._physics_export_env_ids = env_ids
            self._physics_export_motion_ids = self._episode_ctx.motion_ids
            self.env.simulator.set_physics_substep_callback(
                self._capture_physics_substep
            )

        try:
            for step_idx in range(max_steps):
                actions = self._policy_action(obs_td)

                # Apply EMA smoothing (deployment simulation)
                if ema_alpha is not None:
                    if prev_actions is None:
                        prev_actions = actions.clone()
                    actions = ema_alpha * actions + (1.0 - ema_alpha) * prev_actions
                    prev_actions = actions.clone()

                self._physics_export_actions = actions if capture_substeps else None
                obs, rewards, dones, terminated, extras = self.env.step(actions)
                self.agent.pre_collect_step(step_idx + 1)
                obs = self.agent.add_agent_info_to_obs(obs)
                obs_td = self.agent.obs_dict_to_tensordict(obs)

                self._check_eval_components(env_ids, step_idx)
                self._on_episode_step(env_ids, extras, actions)
        finally:
            if capture_substeps:
                self.env.simulator.set_physics_substep_callback(None)
                self._physics_export_actions = None
                self._physics_export_env_ids = None
                self._physics_export_motion_ids = None

    def run_evaluation(self) -> None:
        """Run evaluation across multiple motions."""
        for env_ids, motion_ids in self._build_eval_batches():
            motion_lengths = self._get_evaluation_motion_lengths(motion_ids)
            max_len = min(
                self._motion_frame_counts(motion_lengths.max()).item(),
                self.config.max_eval_steps,
            )
            # Build episode context before evaluate_episode so hooks can read it
            self._episode_ctx = MimicEpisodeContext(
                motion_ids=motion_ids,
                frame_limits=self._motion_frame_counts(motion_lengths).clamp(
                    max=self.config.max_eval_steps
                ),
            )
            self.evaluate_episode(env_ids, max_len)

    def _build_eval_batches(self):
        """Build list of (env_ids, motion_ids) batches to evaluate.

        Returns:
            List of (env_ids, motion_ids) tuples
        """
        fixed_motion_ids, first_env_indices = (
            self.motion_manager.get_unique_fixed_motions()
        )

        if fixed_motion_ids.numel() > 0:
            print(f"Only evaluating fixed motions: {fixed_motion_ids}")
            return [(first_env_indices, fixed_motion_ids)]

        num_motions = self.motion_lib.num_motions()
        batches = []
        for start in range(0, num_motions, self.num_envs):
            end = min(start + self.num_envs, num_motions)
            motion_ids = torch.arange(start, end, device=self.device)
            env_ids = torch.arange(0, motion_ids.numel(), device=self.device)
            print(f"Evaluating motions {start} to {end}, out of total {num_motions}")
            batches.append((env_ids, motion_ids))
        return batches

    # --- Hook overrides ---

    def _on_episode_start(self, env_ids: Tensor) -> None:
        """Set motion_ids/times in the motion manager before reset."""
        self.motion_manager.motion_ids[env_ids] = self._episode_ctx.motion_ids
        self.motion_manager.motion_times[env_ids] = 0.0

    def _get_reset_kwargs(self) -> dict:
        """Customize env.reset() for mimic evaluation."""
        return {"sample_flat": True, "disable_motion_resample": True}

    def _check_eval_components(self, env_ids: Tensor, step_idx: int) -> None:
        """Filter by frame limits and check failures only for active clips."""
        still_active = self._episode_ctx.frame_limits > step_idx
        if still_active.any():
            active_env_ids = env_ids[still_active]
            active_motion_ids = self._episode_ctx.motion_ids[still_active]
            self._check_evaluation_failures(active_env_ids, active_motion_ids)

    def _on_episode_step(self, env_ids: Tensor, extras: Dict, actions: Tensor) -> None:
        """Collect smoothness metrics each step."""
        self._record_trajectory_step(
            self._metrics, extras, env_ids, self._episode_ctx.motion_ids, actions
        )

    def _record_trajectory_step(
        self,
        metrics: Dict,
        extras: Dict,
        active_env_ids: Tensor,
        active_motion_ids: Tensor,
        actions: Tensor,
    ) -> None:
        """Record robot state and actions into trajectory buffers for this step."""
        if "actions" in metrics and actions is not None:
            metrics["actions"].update(
                active_motion_ids, actions[active_env_ids].detach()
            )

        for k in metrics.keys():
            if k == "actions":
                continue
            if f"raw/{k}" in extras:
                metrics[k].update(
                    active_motion_ids, extras[f"raw/{k}"][active_env_ids].detach()
                )

    def process_eval_results(self) -> Tuple[Dict, Optional[float], int]:
        """Process results and update motion sampling weights."""
        to_log, success_rate, num_eval_items = super().process_eval_results()
        self._update_motion_sampling_weights()

        additional_metrics = self._compute_additional_metrics(self._metrics)
        to_log.update(additional_metrics)
        if (
            self._physics_export_metrics is not None
            and not self._physics_export_metrics["dof_pos"].frame_counts.any()
        ):
            raise RuntimeError(
                "trajectory_export_fps was requested, but the simulator did not "
                "produce any physics-substep samples"
            )
        export_metrics = self._physics_export_metrics or self._metrics
        export_dt = self._physics_export_dt or self.env.dt

        if self.fabric.global_rank == 0:
            if (
                self.config.save_predicted_motion_lib_every is not None
                and self.eval_count % self.config.save_predicted_motion_lib_every == 0
            ):
                self._save_predicted_motion_lib(
                    export_metrics,
                    epoch=self.agent.current_epoch,
                    sample_dt=export_dt,
                )
            video_every = self.config.save_video_every_epochs
            if (
                (
                    video_every is not None
                    and self.agent.current_epoch % video_every == 0
                )
                or self.agent.current_epoch
                in getattr(self.config, "save_video_at_epochs", [])
            ):
                self._save_tracking_video(
                    export_metrics,
                    self.agent.current_epoch,
                    sample_dt=export_dt,
                )

        return to_log, success_rate, num_eval_items

    def _save_tracking_video(
        self, metrics: Dict, epoch: int, sample_dt: Optional[float] = None
    ) -> None:
        """Render the fixed clip as a headless actual/reference skeleton MP4."""
        import imageio_ffmpeg
        import matplotlib

        # Isaac Sim owns multiple worker threads.  A GUI Matplotlib backend
        # creates Tcl/Tk objects whose garbage collection on another thread
        # aborts the entire Kit process ("Tcl_AsyncDelete: ... wrong thread").
        # Video rendering is headless, so force the non-GUI raster backend.
        matplotlib.use("Agg", force=True)
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter

        fixed_ids, _ = self.motion_manager.get_unique_fixed_motions()
        motion_id = int(fixed_ids[0].item()) if fixed_ids.numel() else 0
        actual_metric = metrics["rigid_body_pos"]
        frame_count = int(actual_metric.frame_counts[motion_id].item())
        if frame_count < 2:
            return

        num_bodies = self.env.robot_config.kinematic_info.num_bodies
        actual = (
            actual_metric.data[motion_id, :frame_count]
            .view(frame_count, num_bodies, 3)
            .detach()
            .cpu()
        )
        video_dt = self.env.dt if sample_dt is None else sample_dt
        reference_times = torch.arange(
            frame_count,
            device=self.env.device,
            dtype=torch.float32,
        ) * video_dt
        reference_motion_ids = torch.full(
            (frame_count,),
            motion_id,
            device=self.env.device,
            dtype=torch.long,
        )
        reference = (
            self.motion_lib.get_motion_state(reference_motion_ids, reference_times)
            .rigid_body_pos.detach()
            .cpu()
        )
        parents = self.env.robot_config.kinematic_info.parent_indices

        output_dir = self.root_dir / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"tracking_epoch_{epoch:06d}_motion_{motion_id}.mp4"
        fps = max(1, round(1.0 / video_dt))

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2400)
        with writer.saving(fig, str(output_path), dpi=120):
            for frame in range(frame_count):
                ax.clear()
                root = actual[frame, 0]
                ax.set_xlim(root[0] - 1.0, root[0] + 1.0)
                ax.set_ylim(root[1] - 1.0, root[1] + 1.0)
                ax.set_zlim(0.0, 1.8)
                ax.set_box_aspect((2.0, 2.0, 1.8))
                ax.set_title(f"R1 tracking | epoch {epoch} | frame {frame}")
                for pose, color, label in (
                    (reference[frame], "tab:orange", "reference"),
                    (actual[frame], "tab:blue", "physics"),
                ):
                    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], s=10, c=color, label=label)
                    for child, parent in enumerate(parents):
                        if parent >= 0:
                            segment = pose[[parent, child]]
                            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], c=color, linewidth=1.5)
                if frame == 0:
                    ax.legend(loc="upper right")
                writer.grab_frame()
        plt.close(fig)
        print(f"Saved tracking video to {output_path}")

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {"evaluation/tracking_video": wandb.Video(str(output_path), fps=fps, format="mp4")},
                    step=epoch,
                )
        except Exception as error:
            print(f"W&B video logging skipped: {error}")

    def cleanup_after_evaluation(self) -> None:
        """Restore direct-call state and release evaluator-owned buffers."""
        self.motion_manager.motion_ids.copy_(self._cached_motion_ids)
        self.motion_manager.motion_times.copy_(self._cached_motion_times)
        self.env.restore_state(self._env_snapshot)
        del self._env_snapshot
        del self._cached_motion_ids
        del self._cached_motion_times
        self._physics_export_metrics = None
        self._physics_export_dt = None
        self._physics_export_substep_interval = None
        super().cleanup_after_evaluation()

    def _plot_per_frame_metrics(
        self, metrics: Dict, actions_storage: list = None
    ) -> None:
        """
        Plot per-frame metrics vs time when evaluating a single motion.
        Uses base class plotting with custom colors for contact forces.

        Args:
            metrics: Dictionary of MotionMetrics objects
            actions_storage: List of action arrays for plotting (optional, currently unused)
        """
        # Define custom colors for specific metrics
        custom_colors = {}

        # Only plot metrics that were actually collected
        eval_metric_keys = list(self.config.evaluation_components.keys())
        available_keys = [k for k in eval_metric_keys if k in metrics]

        # Use base class generic plotting with custom colors
        super()._plot_per_frame_metrics(
            metrics,
            keys_to_plot=available_keys if available_keys else None,
            custom_colors=custom_colors,
            output_filename="metrics_per_frame_plot.png",
        )

    def _save_predicted_motion_lib(
        self,
        metrics: Dict[str, MotionMetrics],
        epoch: int,
        sample_dt: Optional[float] = None,
    ) -> None:
        """Pack collected predicted metrics and save as a MotionLib-compatible .pt file.

        This creates a "predicted" version of MotionLib where unknown fields are copied
        from the ground-truth self.motion_lib.

        Args:
            metrics: Dictionary of MotionMetrics objects containing predicted data
            epoch: Current epoch number for filename
        """
        required_keys = [
            "dof_pos",
            "dof_vel",
            "rigid_body_pos",
            "rigid_body_rot",
            "rigid_body_vel",
            "rigid_body_ang_vel",
            "rigid_body_contacts",
        ]

        # Ensure required data exists
        for k in required_keys:
            if k not in metrics:
                raise ValueError(
                    f"Missing metric '{k}' required to build predicted MotionLib"
                )

        device = self.device
        num_motions = self.motion_lib.num_motions()
        output_dt = self.env.dt if sample_dt is None else sample_dt

        motion_num_frames = metrics["dof_pos"].motion_lens.to(device=device).long()
        assert (
            motion_num_frames.shape[0] == num_motions
        ), "motion_num_frames size mismatch"

        # Mask motions that were never rolled out, so the saved lib doesn't
        # contain zero-filled phantom frames that trip the playback assertion.
        # frame_counts is incremented by MotionMetrics.update; zero here means
        # no env ever wrote a frame for this motion.  Motion IDs stay aligned
        # with the GT lib; un-rolled motions simply have length 0 in the saved file.
        rolled_out = metrics["dof_pos"].frame_counts.to(device=device) > 0
        num_skipped = int((~rolled_out).sum().item())
        if num_skipped > 0:
            print(
                f"Predicted MotionLib: masking {num_skipped} / {num_motions} "
                f"un-rolled-out motions (length set to 0)"
            )
        motion_num_frames = torch.where(
            rolled_out,
            motion_num_frames,
            torch.zeros_like(motion_num_frames),
        )

        lengths_shifted = motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        length_starts = lengths_shifted.cumsum(0)

        motion_dt = (
            torch.ones(num_motions, dtype=torch.float32, device=device) * output_dt
        )
        motion_lengths = motion_num_frames.to(dtype=torch.float32) * output_dt

        def pack_metric(metric_key: str) -> torch.Tensor:
            data = metrics[metric_key].data
            per_motion = []
            for m in range(num_motions):
                f = motion_num_frames[m].item()
                f = min(f, data.shape[1])
                per_motion.append(data[m, :f].detach().clone())
            return torch.cat(per_motion, dim=0)

        # Build packed tensors matching MotionLib field names
        dps = pack_metric("dof_pos")  # [total_frames, num_dofs]
        dvs = pack_metric("dof_vel")  # [total_frames, num_dofs]

        # Rigid body tensors are stored flattened in metrics; reshape to [*, num_bodies, C]
        num_bodies = self.env.robot_config.kinematic_info.num_bodies
        gts_flat = pack_metric("rigid_body_pos")  # [total_frames, num_bodies*3]
        grs_flat = pack_metric("rigid_body_rot")  # [total_frames, num_bodies*4]
        gvs_flat = pack_metric("rigid_body_vel")  # [total_frames, num_bodies*3]
        gavs_flat = pack_metric("rigid_body_ang_vel")  # [total_frames, num_bodies*3]

        # Validate and reshape
        assert (
            gts_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_pos dim mismatch: {gts_flat.shape[-1]} vs {num_bodies*3}"
        assert (
            grs_flat.shape[-1] == num_bodies * 4
        ), f"rigid_body_rot dim mismatch: {grs_flat.shape[-1]} vs {num_bodies*4}"
        assert (
            gvs_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_vel dim mismatch: {gvs_flat.shape[-1]} vs {num_bodies*3}"
        assert (
            gavs_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_ang_vel dim mismatch: {gavs_flat.shape[-1]} vs {num_bodies*3}"

        gts = gts_flat.view(-1, num_bodies, 3)
        grs = grs_flat.view(-1, num_bodies, 4)
        gvs = gvs_flat.view(-1, num_bodies, 3)
        gavs = gavs_flat.view(-1, num_bodies, 3)

        # Rigid body positions captured via "raw/rigid_body_pos" are in the
        # simulator's world frame, so they include the per-env respawn offset
        # the env applies on reset. The replay-time counterpart
        # ``get_spawn_to_ref_pose_offset_with_terrain_height_correction`` adds
        # back only ``scene_xy + fresh terrain height correction`` (and
        # deliberately NOT ``ref_respawn_offset``, which is a spawn-only
        # safety bump for physics). So to keep the saved lib replay-faithful,
        # we undo exactly what replay will re-add — no more, no less.
        #
        # Concretely: stored ``respawn_root_offset.z`` equals
        # ``terrain_height_at_spawn + ref_respawn_offset``; subtracting just
        # the ``terrain_height_at_spawn`` portion leaves the 5 cm spawn bump
        # baked into ``gts[0, root, z]`` so playback renders the "drop from
        # 5 cm, then settle" trajectory the policy actually experienced.
        # Velocities/rotations are invariant under a constant translation.
        per_motion_offset = torch.zeros(num_motions, 3, device=device, dtype=gts.dtype)
        unique_motion_ids, first_env_indices = (
            self.motion_manager.get_unique_fixed_motions()
        )
        if unique_motion_ids.numel() > 0:
            env_offsets = (
                self.env.respawn_root_offset[first_env_indices]
                .to(device=device, dtype=gts.dtype)
                .clone()
            )
            # Strip the spawn-only ref_respawn_offset from z; keep terrain
            # correction and scene xy.
            env_offsets[:, 2] -= float(self.env.config.ref_respawn_offset)
            per_motion_offset[unique_motion_ids] = env_offsets
        for m in range(num_motions):
            nframes = int(motion_num_frames[m].item())
            if nframes == 0:
                continue
            start = int(length_starts[m].item())
            gts[start : start + nframes] -= per_motion_offset[m].view(1, 1, 3)

        # Pack predicted contacts from metrics
        contacts_data = metrics[
            "rigid_body_contacts"
        ].data  # [num_motions, max_frames, num_bodies]
        contacts_list = []
        for m in range(num_motions):
            f = motion_num_frames[m].item()
            # Clamp to available frames
            f = min(f, contacts_data.shape[1])
            # Convert float contacts to bool for consistency with MotionLib format
            contacts_list.append(contacts_data[m, :f].bool().detach().clone())
        contacts = torch.cat(contacts_list, dim=0)

        # Copy ground-truth motion weights and files
        gt_lib = self.motion_lib
        motion_weights = getattr(
            gt_lib,
            "motion_weights",
            torch.ones(num_motions, dtype=torch.float32, device=device),
        )
        motion_files = getattr(
            gt_lib,
            "motion_files",
            tuple([f"predicted_motion_{i}" for i in range(num_motions)]),
        )

        save_data = {
            "gts": gts,
            "grs": grs,
            "gvs": gvs,
            "gavs": gavs,
            "dvs": dvs,
            "dps": dps,
            "length_starts": length_starts,
            "motion_lengths": motion_lengths,
            "motion_dt": motion_dt,
            "motion_num_frames": motion_num_frames,
            "motion_weights": motion_weights,
            "motion_files": motion_files,
            "contacts": contacts,  # Always save predicted contacts
        }

        # create dir if not exists
        output_dir = self.root_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"predicted_motion_lib_epoch_{epoch}.pt"
        torch.save(save_data, output_path)
        print(f"Predicted MotionLib saved to {output_path}")
