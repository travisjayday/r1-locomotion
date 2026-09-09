# SPDX-License-Identifier: Apache-2.0
"""Mirror an R1 .motion file left-right, using protomotions.utils.mirroring.

Produces a new .motion file whose kinematics are the sagittal-plane (XZ)
reflection of the input: a genuinely CCW clip mirrors into a genuinely CW
one (and vice versa), with dynamics preserved exactly (mirroring a
physically valid trajectory of a left-right-symmetric robot produces another
physically valid trajectory).

Two uses this was built for:

1. Isolating whether a checkpoint's one-sided failure (e.g. R1 locomotion
   training succeeding on clockwise walk_spin clips but failing on
   counterclockwise ones) comes from the policy or from the environment.
   Run the SAME checkpoint against the mirrored clip via the normal
   inference/eval path (unmodified obs/action/reward/termination code):
   if it now succeeds like the real clockwise clips do, that's evidence the
   dynamics are learnable and the checkpoint just hasn't mastered this
   handedness; if it still fails, the failure isn't specific to the
   original clip's authored direction.
2. Left-right data augmentation: mirror an entire motion directory into a
   sibling directory and train on both, doubling handedness coverage for
   free.

The R1 mirror table (which DOF/body swaps with which, and whether a DOF's
own value additionally flips sign) is derived from body/DOF names and
verified end-to-end against MuJoCo forward kinematics in
protomotions/tests/test_mirroring.py -- see that file for the derivation
and the one known caveat (a ~3cm genuine, task-irrelevant lateral offset in
R1's head mount; irrelevant here since head DOFs aren't part of any R1
locomotion action space).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protomotions.utils.mirroring import build_mirror_table, mirror_motion_state  # noqa: E402

# MJCF-derived body/DOF order for R1 (protomotions/data/assets/mjcf/r1.xml).
# Kept as literals so this script doesn't need dm_control/mujoco installed;
# cross-checked against MuJoCo directly in protomotions/tests/test_mirroring.py.
R1_BODY_NAMES = [
    "pelvis_link",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_roll_link", "waist_yaw_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link",
    "head_pitch_link", "head_yaw_link",
]
R1_DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_roll_joint", "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
    "head_pitch_joint", "head_yaw_joint",
]


def mirror_r1_motion_file(data: dict) -> dict:
    """Mirror one loaded R1 .motion (StateConversion-pickled) dict."""
    table = build_mirror_table(R1_BODY_NAMES, R1_DOF_NAMES)

    root_pos = data["rigid_body_pos"][:, 0]
    root_rot = data["rigid_body_rot"][:, 0]

    mirrored = mirror_motion_state(
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=data["dof_pos"],
        table=table,
        w_last=True,  # .motion files store xyzw quaternions
        dof_vel=data["dof_vel"],
        body_pos=data["rigid_body_pos"],
        body_rot=data["rigid_body_rot"],
        body_vel=data["rigid_body_vel"],
        body_ang_vel=data["rigid_body_ang_vel"],
    )

    out = dict(data)
    out["dof_pos"] = mirrored["dof_pos"]
    out["dof_vel"] = mirrored["dof_vel"]
    out["rigid_body_pos"] = mirrored["body_pos"]
    out["rigid_body_rot"] = mirrored["body_rot"]
    out["rigid_body_vel"] = mirrored["body_vel"]
    out["rigid_body_ang_vel"] = mirrored["body_ang_vel"]
    # Boolean per-body contact flags: swap left<->right bodies, no sign.
    out["rigid_body_contacts"] = data["rigid_body_contacts"][:, table.body_index]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path, help="Input R1 .motion file")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path (default: <stem>_mirrored.motion next to the input)",
    )
    args = parser.parse_args()

    output = args.output or args.motion.with_name(f"{args.motion.stem}_mirrored.motion")

    data = torch.load(args.motion, map_location="cpu", weights_only=False)
    required = ("dof_pos", "dof_vel", "rigid_body_pos", "rigid_body_rot",
                "rigid_body_vel", "rigid_body_ang_vel")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{args.motion}: missing {missing}; not a plain R1 .motion file")
    if data["rigid_body_pos"].shape[1] != len(R1_BODY_NAMES):
        raise ValueError(
            f"{args.motion}: {data['rigid_body_pos'].shape[1]} bodies, expected "
            f"{len(R1_BODY_NAMES)} (R1)"
        )

    mirrored = mirror_r1_motion_file(data)
    torch.save(mirrored, output)
    print(f"Wrote mirrored motion ({len(data['dof_pos'])} frames) to {output}")


if __name__ == "__main__":
    main()
