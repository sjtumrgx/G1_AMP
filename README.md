# G1_AMP

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Overview

This repository is the environment side of the `G1_AMP` project.

We aim at industralize Reinforcement Learning for Humanoid (legged robots) whole-body control.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.
- `Unified Ecosystem` This repository is a part of the `G1_AMP` ecosystem, which includes the [instinct_rl](https://github.com/project-instinct/instinct_rl) and [instinct_onboard](https://github.com/project-instinct/instinct_onboard) repositories.
    - The core design of this ecosystem is to treat each experiment as a standalone structured folder, which start with a timestamp as a unique identifier.
    - Adding `--exportonnx` flag to the `play.py` script will export the policy as an ONNX model. After that, you should directly copy the logdir to the robot computer and use the `instinct_onboard` workflow to run the policy on the real robot.

**Keywords:** extension, template, isaaclab

## Warning
This codebase is under [CC BY-NC 4.0 license](LICENSE), with inherited license in IsaacLab. You may not use the material for commercial purposes, e.g., to make demos to advertise your commercial products or wrap the code for your own commercial purposes.

## Contributing
See our [Contributor Agreement](CONTRIBUTOR_AGREEMENT.md) for contribution guidelines. By contributing or submitting a pull request, you agree to transfer copyright ownership of your contributions to the project maintainers.

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for a list of acknowledged contributors.

## Installation

This repository is the environment side of the Project Instinct stack. A working setup has five layers:

1. System and GPU driver prerequisites
2. Isaac Sim 5.1.0
3. Isaac Lab pinned to the supported commit
4. `instinct_rl`
5. This repository (`G1_AMP` / `instinctlab`) plus task-specific motion-data configuration

The sequence matters. If Isaac Sim itself does not start cleanly, do not continue to Isaac Lab or task debugging.

### System Requirements

- `x86_64` machine
- Ubuntu `22.04`
- GLIBC `>= 2.35`
- NVIDIA driver `>= 580.65.06` recommended
- CUDA-capable NVIDIA GPU with enough VRAM for Isaac Sim rendering and training

Preflight check:

```bash
uname -m
lsb_release -a
python3 --version
ldd --version
nvidia-smi
```

If these do not match the expected platform, do not continue with the pip-based Isaac Sim 5.1 route below.

### Step 1: Create an Isolated Python Environment

We recommend a clean conda environment for Python isolation only.

```bash
conda create -n instinct51 python=3.11 -y
conda activate instinct51
pip install --upgrade pip
```

### Step 2: Install System Packages

```bash
sudo apt update
sudo apt install -y cmake build-essential libvulkan1 mesa-vulkan-drivers
```

### Step 3: Install Isaac Sim 5.1.0

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

Install PyTorch after Isaac Sim so the CUDA runtime matches the intended stack:

```bash
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

### Step 4: Accept the EULA and Verify Isaac Sim First

```bash
export OMNI_KIT_ACCEPT_EULA=YES

isaacsim isaacsim.exp.compatibility_check
isaacsim
```

Notes:

- `OMNI_KIT_ACCEPT_EULA=YES` avoids interactive terminal prompts.
- The first `isaacsim` launch downloads and initializes extensions and can take a while.
- Do not continue until the simulator itself starts successfully.

### Step 5: Install Isaac Lab

Clone Isaac Lab separately from this repository and pin it to the supported commit:

```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 37ddf626871758333d6ed89cf64ad702aef127d0

./isaaclab.sh --install
```

This repository is developed against Isaac Sim `5.1.0` and Isaac Lab commit `37ddf626871758333d6ed89cf64ad702aef127d0` from January 30, 2026.

### Step 6: Verify Isaac Lab Before Installing Tasks

```bash
cd ~/IsaacLab
python -c "import isaaclab; print('isaaclab ok')"
python -c "from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper; print('isaaclab_rl ok')"
```

If these imports fail, fix the Isaac Sim / Isaac Lab stack before touching task-level configuration.

### Step 7: Install instinct_rl

```bash
cd ~
git clone https://github.com/project-instinct/instinct_rl.git
python -m pip install -e instinct_rl
```

### Step 8: Clone and Install This Repository

Clone this repository outside the Isaac Lab checkout:

```bash
# Option 1: HTTPS
cd ~
git clone https://github.com/sjtumrgx/G1_AMP.git

# Option 2: SSH
git clone git@github.com:sjtumrgx/G1_AMP.git
```

Then install the package:

```bash
cd ~/G1_AMP
python -m pip install -e source/instinctlab
```

If you use Git LFS and run into clone or dirty-worktree issues with large tracked assets:

```bash
git lfs install --skip-smudge
git clone https://github.com/sjtumrgx/G1_AMP.git
cd G1_AMP
git lfs pull
python -m pip install -e source/instinctlab
```

### Step 9: Run a Project-Level Smoke Test

Use a small headless run to confirm the full chain is working:

```bash
cd ~/G1_AMP
python scripts/instinct_rl/train.py \
    --task=Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 \
    --headless \
    --num_envs 16
```

This is a smoke test, not a recommended full training recipe.

### Step 10: Configure Task-Specific Motion Data

Most public tasks in this repository are not zero-data entry points. Before training or playback, verify the task-specific motion or metadata paths.

- `Instinct-BeyondMimic-Plane-G1-v0`
  File: `source/instinctlab/instinctlab/tasks/shadowing/beyondmimic/config/g1/beyondmimic_plane_cfg.py`
  Set `MOTION_NAME`, `_hacked_selected_file_`, and `AmassMotionCfg.path`.
- `Instinct-Shadowing-WholeBody-Plane-G1-v0`
  File: `source/instinctlab/instinctlab/tasks/shadowing/whole_body/config/g1/plane_shadowing_cfg.py`
  Set `MOTION_NAME`, `_path_`, and `_hacked_selected_files_`.
- `Instinct-Perceptive-Shadowing-G1-v0`
  File: `source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py`
  Set `MOTION_FOLDER` and ensure the folder contains motion files plus a readable `metadata.yaml`.
- `Instinct-Parkour-Target-Amp-G1-v0`
  File: `source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py`
  Set `AmassMotionCfg.path` and `filtered_motion_selection_filepath`.

The `scripts/instinct_rl` directory is intentionally kept to user-facing entry scripts and standalone utilities.
Task-specific helper modules now live under `source/instinctlab/instinctlab/tasks/.../scripts` or other source
packages instead of being mixed into the script directory.

### Optional: BeyondMimic Motion Preprocessing and W&B Export Workflow

If your source motions are still in the Unitree-style generalized-coordinate `.csv` format, you can preprocess them
inside this repository and optionally publish the exported motion bundle to W&B Registry. This is adapted from the
`Motion Preprocessing & Registry Setup` flow in `whole_body_tracking`, but aligned with `instinctlab`'s local
`*_retargeted.npz` training path.

- Gather retargeted reference motions and respect the original dataset licenses. The expected CSV convention is the same
  generalized-coordinate layout used by Unitree-style retargeted motion datasets.
- Convert a CSV motion with:

```bash
python scripts/instinct_rl/csv_to_npz.py \
    --input_file /path/to/{motion_name}.csv \
    --input_fps 30 \
    --output_name {motion_name} \
    --output_dir /path/to/motion_outputs \
    --headless
```

This writes two local files:

- `{motion_name}_retargeted.npz`: compatible with `AmassMotionCfg.path` in this repository
- `{motion_name}_motion.npz`: full-body motion export for downstream inspection, interchange, and optional W&B upload

The preprocessing script uploads to W&B by default. To keep the workflow local-only, add `--skip_registry_upload`.

- If you want the W&B registry flow, log in first and ensure the registry path is under your intended entity.
- This repository no longer ships a dedicated motion-replay entry script. Treat `{motion_name}_motion.npz` as an
  export artifact for downstream tooling, ad hoc analysis, or registry archival rather than an in-repo playback format.

- To train `Instinct-BeyondMimic-Plane-G1-v0` or `Instinct-Shadowing-WholeBody-Plane-G1-v0`, point the task config to
  the local `{motion_name}_retargeted.npz` file, not to the W&B registry artifact. Set `AmassMotionCfg.path` (or
  `_path_`) to `/path/to/motion_outputs` and select `{motion_name}_retargeted.npz` in the corresponding config file.
- For terrain-matched perceptive datasets, keep using local folders plus `metadata.yaml`. If needed, generate the YAML
  with `python scripts/motion_matched_metadata_generator.py --path /path/to/dataset_dir`.

Debugging notes:

- If you are uploading to an organization-scoped registry, export `WANDB_ENTITY` to the organization name or pass
  `--wandb_entity`.
- If the default temporary directory is unavailable, pass `--temp_dir /path/to/tmpdir`.
- The current `scripts/instinct_rl/train.py` flow is local-file based; registry upload is optional for sharing and
  archival, not a required training dependency.

If task imports succeed but rollouts fail immediately, separate platform problems from task-data problems:

- Platform problems show up as Isaac Sim, Isaac Lab, Vulkan, CUDA, import, or environment bootstrap failures.
- Task problems usually appear later as missing motion files, bad metadata paths, invalid task names, or checkpoint/data mismatches.

### Installation References

- [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- [Instinct-RL README](https://github.com/project-instinct/instinct_rl/blob/main/README.md)

## Documentation of Critical Components

- [Instinct-RL Documentation](https://github.com/project-instinct/instinct_rl/blob/main/README.md)
- [G1_AMP Documentation](https://github.com/sjtumrgx/G1_AMP/blob/main/DOCS.md)

## Parkour Task

### Basic Usage Guidelines

**Task ID:** `Instinct-Parkour-Target-Amp-G1-v0`

1. Go to `config/g1/g1_parkour_target_amp_cfg.py` and set the `path` and `filtered_motion_selection_filepath` in `AmassMotionCfg` to the reference motion you want to use.

2. Train the policy:

```bash
python scripts/instinct_rl/train.py --headless --task=Instinct-Parkour-Target-Amp-G1-v0
```

3. Play a trained policy:

```bash
python scripts/instinct_rl/play_depth.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=<run_name>
```

4. Export the trained policy to ONNX and validate it through the same play entrypoint:

```bash
python scripts/instinct_rl/play_depth.py \
    --task=Instinct-Parkour-Target-Amp-G1-v0 \
    --load_run=<run_name> \
    --exportonnx \
    --useonnx
```

The exported model is written under the selected run's `exported/` directory. `--exportonnx` writes the ONNX artifact,
and `--useonnx` immediately switches play-time inference to the exported model so you can do a quick sanity check with
the same command.

Recommended interactive depth-debug, route-recording, and video command:

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/instinct_rl/play_depth.py \
    --task=Instinct-Parkour-Target-Amp-G1-v0 \
    --load_run=20260327_163647 \
    --seed=123 \
    --keyboard_control \
    --num_envs=1 \
    --save_route=outputs/parkour/seed123-route.json \
    --video \
    --show_depth_window \
    --show_depth_coverage \
    --disable_auto_reset \
    --video_duration_s 300
```

What this command does:

- `OMNI_KIT_ACCEPT_EULA=YES`: skips the interactive Isaac Sim EULA prompt in terminal runs.
- `--keyboard_control`: lets you steer the robot manually with the keyboard.
- `--num_envs=1`: forces a single robot instance so the play view, keyboard commands, and recording stay easy to inspect.
- `--seed=123`: fixes the terrain generator and reset randomization so the map can be replayed later.
- `--save_route=...`: writes the driven route as a JSON artifact containing the seed and sampled world-frame XY waypoints.
- `--video`: enables video recording.
- `--show_depth_window`: opens a live depth preview window.
- `--show_depth_coverage`: shows the raw camera coverage footprint in the RGB scene.
- `--disable_auto_reset`: prevents termination conditions from immediately resetting the robot during manual debugging.
- `--video_duration_s 300`: records up to 300 seconds of encoded video instead of the shorter default clip.

### Parkour Play Visualization Modules

The parkour play path has a centralized visualization-default block at:

- `source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py`
- class: `ParkourPlayVisualizationCfg`

The seven visualization toggles in that config are:

- `depth_window`
- `depth_coverage`
- `normals_panel`
- `route_overlay`
- `foot_contact_overlay`
- `ghost_reference`
- `obstacle_edges`

At runtime, `scripts/instinct_rl/play_depth.py` resolves these defaults through `resolve_play_visualization_config()` in `source/instinctlab/instinctlab/tasks/parkour/scripts/play_runtime.py`.

Precedence rules:

- First set your default behavior in `ParkourPlayVisualizationCfg`.
- Then override per run from the CLI.
- These flags use `argparse.BooleanOptionalAction`, so each toggle supports both enable and disable forms.

CLI override forms:

- `--show_depth_window` / `--no-show_depth_window`
- `--show_depth_coverage` / `--no-show_depth_coverage`
- `--normals_panel` / `--no-normals_panel`
- `--route_overlay` / `--no-route_overlay`
- `--foot_contact_overlay` / `--no-foot_contact_overlay`
- `--ghost_reference` / `--no-ghost_reference`
- `--obstacle_edges` / `--no-obstacle_edges`

Examples:

```bash
# Enable all seven visualizations for one run
OMNI_KIT_ACCEPT_EULA=YES python scripts/instinct_rl/play_depth.py \
    --task=Instinct-Parkour-Target-Amp-G1-v0 \
    --load_run=20260327_163647 \
    --replay_route=outputs/parkour/routes/Instinct-Parkour-Target-Amp-G1-v0-seed123.json \
    --video \
    --show_depth_window \
    --show_depth_coverage \
    --normals_panel \
    --route_overlay \
    --foot_contact_overlay \
    --ghost_reference \
    --obstacle_edges

# Start from config defaults, but disable just the ghost
OMNI_KIT_ACCEPT_EULA=YES python scripts/instinct_rl/play_depth.py \
    --task=Instinct-Parkour-Target-Amp-G1-v0 \
    --load_run=20260327_163647 \
    --no-ghost_reference
```

#### 1. `depth_window`

- Purpose: opens a live OpenCV depth preview window named `parkour_depth`.
- Enable:
  - config: `depth_window = True`
  - CLI: `--show_depth_window`
- Disable:
  - config: `depth_window = False`
  - CLI: `--no-show_depth_window`
- Where it appears:
  - live interactive preview window only
  - the recorded video always includes the processed depth panel regardless of the separate live window
- Main tunables:
  - preview scale is currently code-level in `build_live_preview_panels(scale=8.0)` in `play_runtime.py`
  - displayed depth range comes from the camera noise pipeline normalization config:
    - `env_cfg.scene.camera.noise_pipeline.depth_normalization.depth_range`
    - if normalization is missing, play falls back to `0.0` to `2.5` meters

#### 2. `depth_coverage`

- Purpose: draws the raw grouped ray-caster field-of-view coverage into the 3D scene.
- Enable:
  - config: `depth_coverage = True`
  - CLI: `--show_depth_coverage`
- Disable:
  - config: `depth_coverage = False`
  - CLI: `--no-show_depth_coverage`
- What it shows:
  - the raw camera coverage footprint, not the final cropped policy depth tensor
- Main tunables:
  - camera ray pattern size from `scene.camera.pattern_cfg.height` and `scene.camera.pattern_cfg.width`
  - policy crop from `scene.camera.noise_pipeline.crop_and_resize.crop_region`
  - camera extrinsics / placement from the parkour scene camera config in `parkour_env_cfg.py`

#### 3. `normals_panel`

- Purpose: shows a false-color surface-normal visualization from the same ray-caster camera.
- Enable:
  - config: `normals_panel = True`
  - CLI: `--normals_panel`
- Disable:
  - config: `normals_panel = False`
  - CLI: `--no-normals_panel`
- Where it appears:
  - live OpenCV window named `parkour_normals`
  - recorded composite video as the fifth panel
- Main tunables:
  - the camera must emit the `normals` data type; play auto-adds it when this toggle is enabled
  - preview scale is shared with the depth window via `build_live_preview_panels(scale=8.0)`
  - false-color conversion is handled by `normalize_normals_frame_for_display()` in `play_runtime.py`

#### 4. `route_overlay`

- Purpose: draws route-following diagnostics in the 3D scene.
- Enable:
  - config: `route_overlay = True`
  - CLI: `--route_overlay`
- Disable:
  - config: `route_overlay = False`
  - CLI: `--no-route_overlay`
- What it shows:
  - the exact saved route polyline in amber
  - a short predicted future trajectory in cyan
- Important behavior:
  - the overlay now draws the original route polyline directly; it does not spline-smooth the saved waypoints
- Main tunables:
  - route geometry itself comes from `--replay_route=...` or from the JSON route artifact you edit/save
  - replay follower behavior is controlled by:
    - `--route_lookahead_m`
    - `--route_goal_tolerance_m`
    - `--route_cruise_speed`
  - predicted overlay horizon is currently code-level in `RouteOverlayDebugDraw`:
    - `prediction_horizon_s=1.0`
    - `prediction_samples=20`

#### 5. `foot_contact_overlay`

- Purpose: visualizes foot contact forces and touchdown events.
- Enable:
  - config: `foot_contact_overlay = True`
  - CLI: `--foot_contact_overlay`
- Disable:
  - config: `foot_contact_overlay = False`
  - CLI: `--no-foot_contact_overlay`
- What it shows:
  - red arrows for active foot contact forces
  - yellow markers for recent touchdown points
- Main tunables:
  - currently code-level in `FootContactOverlayRig` inside `scripts/instinct_rl/play_depth.py`
  - key values are:
    - `touchdown_ttl_steps=300`
    - `force_threshold=1.0`
    - `force_scale=0.0025`

#### 6. `ghost_reference`

- Purpose: shows the motion-reference robot as a side-by-side visual ghost for qualitative comparison.
- Enable:
  - config: `ghost_reference = True`
  - CLI: `--ghost_reference`
- Disable:
  - config: `ghost_reference = False`
  - CLI: `--no-ghost_reference`
- What play mode does when enabled:
  - turns on `motion_reference.debug_vis`
  - switches the preview ghost to `visualizing_robot_from="reference_frame"` when it was still using `aiming_frame`
  - applies a default lateral offset of `(0.0, 1.5, 0.0)` if no offset was configured, so the ghost is readable instead of overlapping the live robot
- Main tunables:
  - `motion_reference.visualizing_robot_from`
  - `motion_reference.visualizing_robot_offset`
  - `motion_reference.reference_prim_path`
  - `motion_reference.visualizing_marker_types`
  - these live in the motion-reference config used by the parkour task

#### 7. `obstacle_edges`

- Purpose: highlights obstacle boundaries and terrain-generated wall edges in the scene.
- Enable:
  - config: `obstacle_edges = True`
  - CLI: `--obstacle_edges`
- Disable:
  - config: `obstacle_edges = False`
  - CLI: `--no-obstacle_edges`
- What it shows:
  - the terrain / obstacle edge debug representation registered from the terrain metadata
- Main tunables:
  - wall generation parameters in the terrain generator config, especially:
    - `wall_prob`
    - `wall_height`
    - `wall_thickness`
  - for replay, the exact wall layout can also come from `tile_wall_edges` embedded inside the saved route artifact

### Recorded Video Layout

When `--video_layout quad` is active, the composite recorder uses a 3x2 grid once extra panels are present.

- Base 4 panels:
  - `Hero`
  - `Side`
  - `Overview`
  - `Depth`
- Fifth panel:
  - `Normals` when `normals_panel` is enabled
- Sixth panel:
  - `Route Map` during replay video capture when a route artifact is loaded

The `Route Map` panel is an automatically generated debug view of the replay route, terrain tile layout, wall edges, map center, and current robot pose. It is not one of the seven top-level play-visualization toggles, but it fills the sixth composite slot during replay recording so the grid is no longer padded with an empty panel.

Notes:

- The red coverage visualization shows the raw grouped ray-caster field of view. The policy depth observation is still a cropped lower-center patch after preprocessing.
- Keyboard commands during play are:
  - `W`: increase forward command
  - `F`: positive yaw
  - `G`: negative yaw
  - `S`: reset yaw to zero
  - `X`: zero all commands

Headless replay from a saved route:

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/instinct_rl/play_depth.py \
    --headless \
    --task=Instinct-Parkour-Target-Amp-G1-v0 \
    --load_run=20260327_163647 \
    --replay_route=outputs/parkour/routes/Instinct-Parkour-Target-Amp-G1-v0-seed123.json \
    --video \
    --show_depth_window \
    --show_depth_coverage \
    --video_duration_s 120
```

Replay notes:

- `--replay_route=...` loads the saved route artifact and follows its waypoints automatically.
- The saved route seed is reused automatically unless you pass `--seed` explicitly.
- Replay forces a deterministic single-env start state at the map center and disables auto reset so the route can be rendered headlessly into a video.
- When `--replay_route` is active, capture stops on route completion or when `--video_duration_s` reaches its maximum, whichever comes first.
- If `--video_duration_s` is omitted during replay, the run stays unbounded and stops only on route completion.
- Replay video filenames get a timestamp suffix so repeated runs do not overwrite earlier captures.
- Play video capture now records at 2K output resolution.
- Route artifacts are plain JSON files with world-frame XY waypoints, so you can also edit them manually if you want to sketch a custom path.

Offline map export and click-to-route tool:

```bash
python scripts/instinct_rl/edit_route_map.py \
    --task=Instinct-Parkour-Target-Amp-G1-v0 \
    --seed=123 \
    --load_run=20260327_163647 \
    --output_png=outputs/parkour/maps/seed123.png \
    --output_route=outputs/parkour/routes/seed123.json
```

What this tool does:

- It uses the lightweight schematic route-editor path by default, loading the logged `env.yaml` and precomputing deterministic wall layouts from the selected seed.
- The saved route JSON now stores `tile_wall_edges` alongside the waypoints so replay can rebuild the exact wall layout instead of sampling again at runtime.
- This default path does not require launching Isaac Sim or accepting the Omniverse EULA.
- If you explicitly want the heavy real-mesh export path, add `--real_mesh` and run with `OMNI_KIT_ACCEPT_EULA=YES`.
- It saves a top-down PNG first.
- `--export_only` also writes a route JSON skeleton with the precomputed `tile_wall_edges`.
- It opens an interactive editor where clicks are already in world coordinates, so the saved `route.json` works directly with `--replay_route`.

Fallback note:

- If you have multiple logged parkour runs and want the fallback schematic map to use a specific run snapshot, add `--load_run=<run_name>`.

Editor controls:

- Left click: add a waypoint
- Right click: remove the last waypoint
- `Backspace` / `Delete` / `U`: undo last waypoint
- `C`: clear all waypoints
- `Enter` / `S`: save `route.json` and an annotated PNG
- `Esc` / `Q`: close without saving

Map export only, without opening the editor:

```bash
python scripts/instinct_rl/edit_route_map.py \
    --task=Instinct-Parkour-Target-Amp-G1-v0 \
    --seed=123 \
    --load_run=20260327_163647 \
    --output_route=outputs/parkour/routes/seed123.json \
    --export_only
```

4. Export a trained policy:

```bash
python scripts/instinct_rl/play_depth.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=<run_name> --exportonnx --useonnx
```

The ONNX export under the run's `exported/` directory is the handoff artifact for downstream deployment or external
runtime validation. This repository no longer includes a MuJoCo-based sim2sim validation path.

### Common Options

- `--num_envs`: Number of parallel environments.
- `--keyboard_control`: Enable keyboard control during play.
- `--seed`: Seed the terrain and reset randomization for deterministic play.
- `--load_run`: Run name to load checkpoint from for play.
- `--save_route`: Save the executed route as a JSON artifact.
- `--replay_route`: Load a saved route and follow it automatically.
- `edit_route_map.py`: Export a fixed-seed top-down terrain map and click waypoints into a compatible `route.json`.
- `--video`: Record playback videos.
- `--exportonnx`: Export the trained model to ONNX for onboard deployment.
- `--useonnx`: Use the ONNX model for inference during play.

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory. The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps in indexing all the python modules for intelligent suggestions while writing code.


## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

To make the `pre-commit` run automatically on every commit, you can use the following command in your repository:

```bash
pre-commit install
```

## Train your own projects

***To preserve your code development and progress. PLEASE create your own repository as an individual project by referring to https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/index.html***

And copy `scripts/instinct_rl` to your own repository.

### Or you are just to stubborn and want to fork and directly modify the code in this repo.

- Please create a new folder in the `source/instinctlab/instinctlab/tasks` directory. The name of the folder should be your project name. Inside the folder, DO add `__init__.py` in each level of the subfolders. (Many people tend to forget this step and could not find the supposely registered tasks.)

- We inherit the manager based RL env from IsaacLab to add new features. DO use `instinctlab.envs:InstinctRlEnv` as the entry_point in the `gym.register` call. For example, if you want to add a new task, you can use the following code:

```python
import gymnasium as gym
from . import agents
task_entry = "instinctlab.tasks.shadowing.perceptive.config.g1"
gym.register(
    id="Instinct-Perceptive-Shadowing-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.perceptive_shadowing_cfg:G1PerceptiveShadowingEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
    },
)
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/instinctlab"
    ]
}
```
