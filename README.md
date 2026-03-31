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

5. Validate the exported ONNX policy in MuJoCo:

```bash
pip install mujoco
python scripts/instinct_rl/play_mujoco.py --load_run=<run_name> --headless --depth-mode=zeros --sim_duration=2.0
```

6. Run the MuJoCo viewer with keyboard override:

```bash
python scripts/instinct_rl/play_mujoco.py --load_run=<run_name> --keyboard_control --depth-mode=mujoco
```

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

### MuJoCo Sim2Sim Notes

- `scripts/instinct_rl/play_mujoco.py` defaults to the logged parkour URDF, promotes it to a floating-base MuJoCo model, and injects a flat floor plus a head camera at runtime.
- `--depth-mode=zeros` is the debug or bring-up path. It is useful for validating the MuJoCo loop, PD control, and ONNX inference chain before relying on rendered depth.
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
