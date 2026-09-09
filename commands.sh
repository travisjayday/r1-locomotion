# Build R1-retargeted motion library from Kimodo G1 CSVs 
env -u LD_LIBRARY_PATH python data/scripts/convert_g1_csv_to_r1_proto.py \
  --input-dir data/g1-kimodo-generated \
  --output-dir data/r1-kimodo-generated \
  --pyroki-python /home/tjz/secure/miniconda3/envs/pyroki/bin/python \
  --r1-sole-clearance 0.003

# Start Viser motion tuning visualizer
/home/tjz/secure/miniconda3/envs/pyroki/bin/python   pyroki/visualize_g1_r1_retarget.py   --g1 data/r1-kimodo-generated/g1-foot-fixed/output_walk_jog_run.npz   --r1 data/r1-kimodo-generated/r1-retargeted/output_walk_jog_run_r1.npz   --separation 0

# Tune weights for visualizer 
env -u LD_LIBRARY_PATH   /home/tjz/secure/miniconda3/envs/pyroki/bin/python   pyroki/batch_retarget_g1_npz_to_r1.py   --input-file data/r1-kimodo-generated/g1-foot-fixed/output_walk_jog_run.npz   --output-dir data/r1-kimodo-generated/r1-retargeted   --weight landmark=32   --weight foot_anchor=80   --weight foot_penetration=100   --weight foot_height=10 --weight swing_foot_orientation=12  --weight foot_tilt=30 --swing-height-scale=3 --weight swing_foot_height=70 --weight torso_orientation=16 --weight root_orientation=0.5

# Viz in isaacsim
python examples/motion_libs_visualizer.py   --simulator isaaclab   --robot r1   --motion_files data/r1-kimodo-generated/kimodo_r1_motions.pt   

# Extract physical motion from RL policy
env -u LD_LIBRARY_PATH MPLCONFIGDIR=/tmp/matplotlib-r1-teacher /home/tjz/secure/RL/next/IsaacLab/.venv/bin/python \
    protomotions/inference_agent.py   \
    --checkpoint results/r1_walk_jog_run_teacher_omni_action_scale/last.ckpt \
    --simulator isaaclab \
    --motion-file data/r1-kimodo-generated/proto-r1/output_walk_jog_run_r1.motion  \
    --num-envs 1   --headless   --full-eval

# Evaluate physical motion vs kinematic motion visually
python pyroki/visualize_r1_physics_overlay.py \
  --dataset /home/tjz/secure/RL/next/ProtoMotions \
  --physics results/r1_locomotion_5/results/predicted_motion_lib_epoch_7600.pt \
  --separation 0




env -u LD_LIBRARY_PATH MPLCONFIGDIR=/tmp/matplotlib-r1-teacher /home/tjz/secure/RL/next/IsaacLab/.venv/bin/python protomotions/train_agent.py --robot-name r1   \
    --simulator isaaclab \
    --experiment-path examples/experiments/mimic/r1_omniscient_teacher.py  \
    --experiment-name r1_locomotion_run1 \
    --motion-file data/r1_locomotion/r1_locomotion_lib.pt \
    --num-envs 4096  \
    --batch-size 16384   \
    --use-wandb   \
    --wandb-project r1_locomotion \
    --training-max-iterations 10000   \
    --reserve-future-context \
    --motion-id 0 \
    --checkpoint results/r1_walk_jog_run_teacher_omni_action_scale/last.ckpt   

# Generate clean command annotations for a kimodo generation motion library
/home/tjz/secure/RL/next/IsaacLab/.venv/bin/python \
  data/scripts/annotate_motion_commands.py \
  --input results/r1_locomotion_4/results/predicted_motion_lib_epoch_6000.pt \
  --output results/r1_locomotion_4/results/predicted_motion_lib_epoch_6000_commands.pt \
  --source-command-root /home/tjz/secure/RL/next/Kimodo_Locomotion_Dataset/data/g1_locomotion \
  --csv-dir results/r1_locomotion_4/results/predicted_motion_lib_epoch_6000_commands_csv \
  --overwrite

# Run steering AMP policy with keyboard control
env -u LD_LIBRARY_PATH MPLCONFIGDIR=/tmp/matplotlib-r1-steering \
  /home/tjz/secure/RL/next/IsaacLab/.venv/bin/python \
  protomotions/inference_agent.py     \
  --checkpoint results/r1_steering6/last.ckpt    \
  --simulator isaaclab     \
  --num-envs 1 \
  --command-source steering=keyboard

# Run steering policy in IsaacSim
/home/tjz/secure/RL/next/IsaacLab/.venv/bin/python \
  protomotions/inference_agent.py \
  --checkpoint results/r1_steering10/last.ckpt \
  --simulator isaaclab \
  --num-envs 1 \
  --command-source steering=window \
  --disable-camera-follow \
  --kit-args='--/rtx/post/aa/op=0 --/rtx/shadows/enabled=false --/rtx/ambientOcclusion/enabled=false --/rtx/reflections/enabled=false --/rtx/indirectDiffuse/enabled=false --/rtx/translucency/enabled=false'

env -u LD_LIBRARY_PATH   /home/tjz/secure/RL/next/IsaacLab/.venv/bin/python   protomotions/train_agent.py   --robot-name r1   --simulator isaaclab   --num-envs 4096   --batch-size 16384   --motion-file results/r1_locomotion_4/results/predicted_motion_lib_epoch_6000_commands.pt   --experiment-path examples/experiments/steering/r1_steering.py   --experiment-name r1_steering10 --checkpoint results/r1_steering9/last.ckpt     --training-max-iterations 6000   --use-wandb   --wandb-project r1_locomotion   --headless