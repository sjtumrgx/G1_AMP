# Parkour Task

## Basic Usage Guidelines

### Parkour Task

**Task ID:** `Instinct-Parkour-Target-Amp-G1-v0`

1. Go to `config/g1/g1_parkour_target_amp_cfg.py` and set the `path` and `filtered_motion_selection_filepath` in `AmassMotionCfg` to the reference motion you want to use.

2. Train the policy:
```bash
python scripts/instinct_rl/train.py --headless --task=Instinct-Parkour-Target-Amp-G1-v0
```

3. Play trained policy (load_run must be provided, absolute path is recommended, or use `--no_resume` to visualize untrained policy):

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=<run_name>
```

4. Export trained policy (load_run must be provided, absolute path is recommended):

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=<run_name> --exportonnx --useonnx
```

5. Validate the exported ONNX policy in MuJoCo:

```bash
pip install mujoco
python scripts/instinct_rl/play_mujoco.py --load_run=<run_name> --headless --depth-mode=zeros --sim_duration=2.0
```

6. Run the MuJoCo viewer with keyboard override:

```bash
python scripts/instinct_rl/play_mujoco.py --load_run=<run_name> --keyboard_control --depth-mode=mujoco
```

OMNI_KIT_ACCEPT_EULA=YES python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py \
    --task=Instinct-Parkour-Target-Amp-G1-v0 \
    --load_run=20260327_163647 \
    --keyboard_control \
    --num_envs=1 \
    --video \
    --show_depth_window \
    --show_depth_coverage \
    --disable_auto_reset \
    --video_duration_s 300

## Common Options

- `--num_envs`: Number of parallel environments (default varies by task)
- `--keyboard_control`: Enable keyboard control during playing
- `--load_run`: Run name to load checkpoint from for playing
- `--video`: Record training/playback videos
- `--exportonnx`: Export the trained model to ONNX format for onboard deployment during playing
- `--useonnx`: Use the ONNX model for inference during playing (requires `--exportonnx`)

## MuJoCo Sim2Sim Notes

- `scripts/instinct_rl/play_mujoco.py` defaults to the logged parkour URDF, promotes it to a floating-base MuJoCo model, and injects a flat floor plus a head camera at runtime.
- `--depth-mode=zeros` is the debug / bring-up path. It is useful for validating the MuJoCo loop, PD control, and ONNX inference chain before relying on rendered depth.
- `--depth-mode=mujoco` uses an approximate MuJoCo depth pipeline. It is suitable for early sim2sim checks, but it does not yet claim pixel-level parity with the IsaacLab ray-caster camera and noise pipeline.
- Keyboard controls currently mirror the IsaacLab play script:
  - `W`: increase forward command
  - `F`: set positive yaw command
  - `G`: set negative yaw command
  - `S`: reset yaw command to zero
  - `X`: zero all commands
- The MuJoCo path records optional traces through `--record_npz=<path>` for offline comparison.
- Fidelity gaps still remain:
  - the MuJoCo scene is a flat plane, not the full Isaac parkour terrain generator
  - the depth preprocessing is approximate
  - explicit XML overrides passed with `--mjcf` must already contain compatible joints and, if needed, their own camera setup
