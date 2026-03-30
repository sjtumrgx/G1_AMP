# Parkour Play Recording And Depth Debug Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the parkour Isaac play flow so it can record a composite video with multiple external RGB cameras plus the live depth view, pin the robot to the map center when requested, optionally disable automatic resets, and visualize the depth camera coverage inside the RGB scene.

**Architecture:** Keep the policy/play loop in [`play_depth.py`](/home/eilab/instinctlab/scripts/instinct_rl/play_depth.py), but move argument parsing helpers, config override logic, camera pose math, depth normalization, and frame composition into a new pure-Python helper module that can be unit-tested without launching Isaac. Use Isaac Lab `Camera` sensors plus existing grouped-ray-caster hit data during integration so the live scene wiring stays thin and the new behavior is built from supported camera/render primitives already present in the codebase.

**Tech Stack:** Python 3.11, Isaac Lab / Isaac Sim camera sensors, Gymnasium play script flow, NumPy, OpenCV, imageio, pytest.

---

### Task 1: Lock the helper contract with failing tests

**Files:**
- Create: `tests/test_parkour_play_runtime.py`
- Create: `source/instinctlab/instinctlab/tasks/parkour/scripts/play_runtime.py`

**Step 1: Write the failing parser and config tests**

Cover:
- new CLI flags for center spawn, disabling auto reset, depth window, depth coverage, and composite video
- selecting the terrain origin closest to the map center
- mutating a fake nested config object to disable terminations and enable debug visuals

Run:
```bash
pytest tests/test_parkour_play_runtime.py -q
```

Expected: FAIL because `play_runtime.py` does not exist yet.

**Step 2: Add the minimal helper module**

Implement:
- `add_play_runtime_args(parser)`
- `select_center_terrain_origin(terrain_origins)`
- `apply_play_runtime_overrides(env_cfg, options)`

**Step 3: Re-run the focused tests**

Run:
```bash
pytest tests/test_parkour_play_runtime.py -q
```

Expected: partial PASS or new focused failures for the still-missing video/depth helpers.

### Task 2: Add tested camera-pose and frame-composition helpers

**Files:**
- Modify: `tests/test_parkour_play_runtime.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/scripts/play_runtime.py`

**Step 1: Extend the failing tests**

Cover:
- orbit/tracking camera eye and target computation from robot pose and yaw
- depth normalization to an 8-bit display image
- composing three RGB panels plus one depth panel into a single recorded frame

Run:
```bash
pytest tests/test_parkour_play_runtime.py -q
```

Expected: FAIL on missing pose/layout helpers.

**Step 2: Implement the minimal pure helpers**

Implement:
- `build_default_tracking_camera_specs()`
- `compute_tracking_camera_views(root_position, root_yaw, specs)`
- `normalize_depth_frame_for_display(depth_frame, depth_range)`
- `compose_recording_frame(rgb_frames, depth_frame, labels)`

**Step 3: Re-run the focused tests**

Run:
```bash
pytest tests/test_parkour_play_runtime.py -q
```

Expected: PASS.

### Task 3: Wire the runtime overrides into the Isaac play script

**Files:**
- Modify: `scripts/instinct_rl/play_depth.py`

**Step 1: Import the new helper surface and add CLI flags**

Use the helper parser extension instead of hard-coding everything directly in `play.py`.

**Step 2: Apply the runtime config overrides before `gym.make(...)`**

Hook in:
- center-spawn option
- disable-auto-reset option
- depth debug visualization option
- depth coverage visualization option

Implementation notes:
- do not replace the user’s existing keyboard-control changes
- keep `num_envs=1` behavior for keyboard control
- only disable reset behavior when the new CLI flag is explicitly set

**Step 3: Pin the live env origin after construction when center spawn is enabled**

Use the terrain importer’s `terrain_origins` / `env_origins` data so reset events continue to reuse the centered origin.

### Task 4: Add multi-camera RGB capture and composite recording

**Files:**
- Modify: `scripts/instinct_rl/play_depth.py`

**Step 1: Replace the legacy single-view recording path with a composite recorder**

Use `imageio` and supported Isaac camera sensors / render products to record a single MP4 containing:
- a wide chase view
- a side view
- a higher overview view
- the normalized depth panel

**Step 2: Keep the cameras tracking the robot every step**

Use the helper camera spec math and Isaac camera `set_world_poses_from_view(...)`.

**Step 3: Preserve video-start and video-length behavior**

Start writing at `--video_start_step` and stop automatically after `--video_length` frames.

### Task 5: Add the live depth window and depth-coverage overlay

**Files:**
- Modify: `scripts/instinct_rl/play_depth.py`

**Step 1: Open a live depth window only when requested**

Use OpenCV to show the latest normalized depth frame during play.

**Step 2: Reuse the grouped ray-caster hits for RGB-visible coverage**

Implementation notes:
- prefer existing sensor hit data over a second ray-casting implementation
- visualize the coverage inside the scene so it appears in the RGB camera recordings
- keep the feature optional behind a CLI flag

### Task 6: Verify the helper layer and script surface

**Files:**
- Modify: none

**Step 1: Run the new focused tests**

Run:
```bash
pytest tests/test_parkour_play_runtime.py -q
```

Expected: PASS.

**Step 2: Run the existing parkour helper tests that should still hold**

Run:
```bash
pytest tests/test_parkour_keyboard_commands.py -q
```

Expected: PASS.

**Step 3: Capture the verification limits honestly**

Record any Isaac-runtime-only gaps explicitly if a full live simulator run is not feasible in this turn.
