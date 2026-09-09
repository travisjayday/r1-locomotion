from types import SimpleNamespace

import torch

from protomotions.utils.reference_command_matching import (
    get_achieved_reference_commands,
    reference_commands_at_times,
    sample_command_matched_motions,
)


def _motion_lib():
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    rotations = identity.view(1, 1, 4).repeat(6, 1, 1)
    velocities = torch.zeros(6, 1, 3)
    angular_velocities = torch.zeros(6, 1, 3)
    velocities[:3, 0, 0] = 0.5
    angular_velocities[3:, 0, 2] = 1.0
    return SimpleNamespace(
        gts=torch.zeros(6, 1, 3),
        grs=rotations,
        gvs=velocities,
        gavs=angular_velocities,
        length_starts=torch.tensor([0, 3]),
        motion_num_frames=torch.tensor([3, 3]),
        motion_dt=torch.tensor([0.1, 0.1]),
    )


class _MotionManager:
    def __init__(self, motion_lib):
        self.motion_lib = motion_lib

    def sample_n_motion_ids(self, num_samples):
        return torch.arange(num_samples) % 2


def test_reference_command_labels_and_lookup_do_not_cross_clip_boundaries():
    motion_lib = _motion_lib()
    commands = get_achieved_reference_commands(
        motion_lib,
        anchor_body_index=0,
        smoothing_window_s=0.2,
    )

    torch.testing.assert_close(
        commands[:3],
        torch.tensor([[0.5, 0.0, 0.0]]).repeat(3, 1),
    )
    torch.testing.assert_close(
        commands[3:],
        torch.tensor([[0.0, 0.0, 1.0]]).repeat(3, 1),
    )
    looked_up = reference_commands_at_times(
        motion_lib,
        motion_ids=torch.tensor([0, 1]),
        motion_times=torch.tensor([0.2, 0.1]),
        anchor_body_index=0,
        smoothing_window_s=0.2,
    )
    torch.testing.assert_close(
        looked_up,
        torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )


def test_command_matched_sampling_selects_nearest_candidate():
    motion_lib = _motion_lib()
    manager = _MotionManager(motion_lib)
    targets = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 1.0]])

    motion_ids, motion_times, achieved = sample_command_matched_motions(
        manager,
        target_commands=targets,
        anchor_body_index=0,
        candidate_count=2,
        smoothing_window_s=0.0,
        init_start_probability=1.0,
    )

    assert torch.equal(motion_ids, torch.tensor([0, 1]))
    assert torch.equal(motion_times, torch.zeros(2))
    torch.testing.assert_close(achieved, targets)


def test_persistent_motion_command_annotations_override_raw_anchor_velocities():
    motion_lib = _motion_lib()
    motion_lib.motion_commands = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
            [-0.1, -0.2, -0.3],
            [-0.2, -0.3, -0.4],
            [-0.3, -0.4, -0.5],
        ]
    )

    commands = get_achieved_reference_commands(
        motion_lib,
        anchor_body_index=0,
        smoothing_window_s=99.0,
    )

    torch.testing.assert_close(commands, motion_lib.motion_commands)
