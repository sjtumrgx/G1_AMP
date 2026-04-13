# Parkour Task

## Basic Usage Guidelines

**Task ID:** `Instinct-Parkour-Target-Amp-G1-v0`

User-facing parkour entry scripts live in `scripts/instinct_rl`. Parkour-specific helper modules now live under
`source/instinctlab/instinctlab/tasks/parkour/scripts`.

1. Go to `config/g1/g1_parkour_target_amp_cfg.py` and set the `path` and `filtered_motion_selection_filepath` in `AmassMotionCfg` to the reference motion you want to use.

2. Train the policy:
```bash
python scripts/instinct_rl/train.py --headless --task=Instinct-Parkour-Target-Amp-G1-v0
```

3. Play trained policy (load_run must be provided, absolute path is recommended, or use `--no_resume` to visualize untrained policy):

```bash
python scripts/instinct_rl/play_depth.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=<run_name>
```

Recommended interactive depth-debug and recording command:

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
    --show_elevation_map_window \
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
- `--show_elevation_map_window`: opens a robot-centered rolling elevation-map window that preserves seen terrain and leaves unseen cells blank.
- `--disable_auto_reset`: prevents termination conditions from immediately resetting the robot during manual debugging.
- `--video_duration_s 300`: records up to 300 seconds of encoded video instead of the shorter default clip.

Play-visualization config:

- `source/instinctlab/instinctlab/tasks/parkour/scripts/play_runtime.py` resolves an optional `env_cfg.play_visualization` namespace plus the CLI overrides below.
- This repo's current G1 parkour config does not define play-visualization defaults, so the CLI flags are the normal entry point unless you add your own `play_visualization` block.
- The eight visualization booleans are:
  - `depth_window`
  - `depth_coverage`
  - `elevation_map_window`
  - `normals_panel`
  - `route_overlay`
  - `foot_contact_overlay`
  - `ghost_reference`
  - `obstacle_edges`
- CLI flags can override these defaults at runtime:
  - `--show_depth_window`
  - `--show_depth_coverage`
  - `--show_elevation_map_window`
  - `--normals_panel`
  - `--route_overlay`
  - `--foot_contact_overlay`
  - `--ghost_reference`
  - `--obstacle_edges`

Notes:

- The red coverage visualization shows the raw grouped ray-caster field of view. The policy depth observation is still a cropped lower-center patch after preprocessing.
- The elevation-map window is a world-aligned local memory view centered on the robot: seen cells are retained over time, while unseen cells intentionally remain blank in v1.
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
    --replay_route=outputs/parkour/seed123-route.json \
    --video \
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

4. Export trained policy (load_run must be provided, absolute path is recommended):

```bash
python scripts/instinct_rl/play_depth.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=<run_name> --exportonnx --useonnx
```

The exported ONNX model under the run's `exported/` directory is intended for downstream deployment or external runtime
validation. This repository no longer carries a MuJoCo-based sim2sim bridge.

## Common Options

- `--num_envs`: Number of parallel environments (default varies by task)
- `--keyboard_control`: Enable keyboard control during play
- `--seed`: Seed the terrain and reset randomization for deterministic play
- `--load_run`: Run name to load checkpoint from for play
- `--save_route`: Save the executed route as a JSON artifact
- `--replay_route`: Load a saved route and follow it automatically
- `edit_route_map.py`: Export a fixed-seed top-down terrain map and click waypoints into a compatible `route.json`
- `--video`: Record training/playback videos
- `--exportonnx`: Export the trained model to ONNX format for onboard deployment during play
- `--useonnx`: Use the ONNX model for inference during play (requires `--exportonnx`)
