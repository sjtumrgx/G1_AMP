# MuJoCo Parkour Sim2Sim Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a MuJoCo-based sim2sim validation script for the exported parkour ONNX policy, with keyboard command override, reproducible runtime checks, and a clear path from first runnable baseline to higher-fidelity validation.

**Architecture:** Add a standalone CLI entrypoint at `scripts/instinct_rl/play_mujoco.py` and keep MuJoCo-specific pure logic in a helper module so path resolution, joint mapping, keyboard state, observation history, depth adaptation, and PD control are unit-testable without launching the viewer. Reuse the exported `actor.onnx` and `0-depth_encoder.onnx`, the logged Isaac config files, and the existing parkour ONNX loading conventions as the source of truth for observation structure and policy I/O.

**Tech Stack:** Python 3.11, MuJoCo Python bindings, onnxruntime, NumPy, pytest, existing InstinctLab YAML config logs, existing `scripts/instinct_rl` CLI patterns.

---

## Context To Preserve

- The exported parkour policy is not proprio-only. `0-depth_encoder.onnx` expects input shape `[1, 8, 18, 32]` and emits `[1, 128]`; `actor.onnx` expects `[1, 896]` and emits `[1, 29]`.
- The parkour policy observation contract comes from `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:416` and includes 8-step histories for `base_ang_vel`, `projected_gravity`, `velocity_commands`, `joint_pos`, `joint_vel`, `actions`, plus delayed `depth_image`.
- The Isaac control asset uses delayed PD actuators from `source/instinctlab/instinctlab/assets/unitree_g1.py:391`.
- The trained parkour run logs point to a shoe-equipped URDF, but the repo’s ready-to-run MuJoCo model is `source/instinctlab/instinctlab/assets/resources/unitree_g1/scene.xml`, which does not currently encode the shoe variant.

## Recommended Delivery Strategy

Use a two-lane rollout inside one implementation:

1. **Bring-up lane**
   Build the MuJoCo loop, ONNX inference path, keyboard control, joint mapping, PD actuation, logging, and a debug `--depth-mode=zeros` fallback first. This gets a runnable sim2sim harness quickly.

2. **Fidelity lane**
   Add a MuJoCo-rendered head-camera depth path that matches the Isaac observation contract closely enough for real validation, and keep `--depth-mode=zeros` as a diagnostic mode only.

Do not block the entire script on perfect depth fidelity or shoe-matched MJCF. Make those explicit knobs:

- `--mjcf`: defaults to the existing `scene.xml`, overrideable later
- `--depth-mode`: `mujoco` or `zeros`
- `--headless`: allow non-viewer verification runs

## Environment Prerequisites

### Task 0: Prepare the runtime environment

**Files:**
- Modify: none

**Step 1: Install the missing MuJoCo runtime**

Run:
```bash
pip install mujoco
```

Expected: `mujoco` imports successfully in the active environment.

**Step 2: Verify runtime libraries before coding**

Run:
```bash
python - <<'PY'
for name in ["mujoco", "onnxruntime", "numpy"]:
    __import__(name)
    print(name, "OK")
PY
```

Expected: all three print `OK`.

**Step 3: Record the current ONNX contract**

Run:
```bash
python - <<'PY'
import onnxruntime as ort
from pathlib import Path
base = Path("logs/instinct_rl/g1_parkour/20260327_163647/exported")
for file in ["0-depth_encoder.onnx", "actor.onnx"]:
    sess = ort.InferenceSession(str(base / file))
    print(file, [i.shape for i in sess.get_inputs()], [o.shape for o in sess.get_outputs()])
PY
```

Expected: depth encoder input `[1, 8, 18, 32]`, actor input `[1, 896]`, actor output `[1, 29]`.

### Task 1: Add a tested runtime-config layer

**Files:**
- Create: `scripts/instinct_rl/parkour_mujoco_runtime.py`
- Create: `tests/test_parkour_mujoco_runtime.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/scripts/config_loading.py` only if a tiny reusable helper is needed

**Step 1: Write the failing tests for path resolution**

Test behaviors:
- resolves `--load_run` into `logs/instinct_rl/g1_parkour/<run>/exported`
- rejects runs missing `actor.onnx`
- defaults `--mjcf` to `source/instinctlab/instinctlab/assets/resources/unitree_g1/scene.xml`
- exposes `--depth-mode` with `mujoco` and `zeros`

Run:
```bash
pytest tests/test_parkour_mujoco_runtime.py -q
```

Expected: FAIL because the helper module does not exist yet.

**Step 2: Implement minimal runtime-resolution helpers**

Implement:
- `resolve_run_paths(load_run: str) -> dataclass`
- `validate_exported_models(...)`
- `resolve_default_mjcf(...)`

**Step 3: Re-run the focused tests**

Run:
```bash
pytest tests/test_parkour_mujoco_runtime.py -q
```

Expected: PASS.

**Step 4: Commit**

```bash
git add scripts/instinct_rl/parkour_mujoco_runtime.py tests/test_parkour_mujoco_runtime.py
git commit -m "Define MuJoCo sim2sim runtime contract

Constraint: Parkour inference depends on exported encoder and actor ONNX files
Rejected: Hard-code absolute export paths | brittle across runs and machines
Confidence: high
Scope-risk: narrow
Directive: Keep path resolution separate from the main simulation loop so runtime checks stay unit-testable
Tested: pytest tests/test_parkour_mujoco_runtime.py -q
" 
```

### Task 2: Lock the joint-order and action-mapping contract

**Files:**
- Modify: `scripts/instinct_rl/parkour_mujoco_runtime.py`
- Create: `tests/test_parkour_mujoco_joint_mapping.py`

**Step 1: Write the failing tests for joint ordering**

Test behaviors:
- parse the logged parkour URDF joint order from `env.yaml`
- parse MuJoCo actuator names from `scene.xml` / `g1.xml`
- verify the 29 policy outputs map to the same actuated joint sequence
- reject models whose actuator set cannot be aligned

Run:
```bash
pytest tests/test_parkour_mujoco_joint_mapping.py -q
```

Expected: FAIL because the parsing and comparison helpers are missing.

**Step 2: Implement minimal joint-order utilities**

Implement:
- `extract_actuated_joint_order_from_urdf(urdf_path)`
- `extract_mujoco_actuator_order(mjcf_path)`
- `build_action_joint_map(...)`

Implementation notes:
- treat the URDF named in the logged run as the policy source-of-truth
- do not assume `g1.xml` actuator order is automatically safe
- fail fast with a descriptive error if the orders diverge

**Step 3: Re-run the focused tests**

Run:
```bash
pytest tests/test_parkour_mujoco_joint_mapping.py -q
```

Expected: PASS.

**Step 4: Commit**

```bash
git add scripts/instinct_rl/parkour_mujoco_runtime.py tests/test_parkour_mujoco_joint_mapping.py
git commit -m "Lock MuJoCo joint mapping to the parkour policy order

Constraint: Actor ONNX emits 29 actions in the training asset's joint order
Rejected: Assume MuJoCo actuator order matches by luck | silent policy corruption risk
Confidence: high
Scope-risk: narrow
Directive: Never bypass the URDF-to-MJCF order check when swapping models
Tested: pytest tests/test_parkour_mujoco_joint_mapping.py -q
"
```

### Task 3: Add keyboard command state and CLI surface

**Files:**
- Create: `scripts/instinct_rl/play_mujoco.py`
- Modify: `scripts/instinct_rl/parkour_mujoco_runtime.py`
- Create: `tests/test_parkour_mujoco_keyboard.py`

**Step 1: Write the failing keyboard-state tests**

Test behaviors:
- `W` increments forward command by `keyboard_linvel_step`
- `F/G` set yaw command to `+/- keyboard_angvel`
- `X` zeros the command
- the current command expands into the 8-step `velocity_commands` observation history

Run:
```bash
pytest tests/test_parkour_mujoco_keyboard.py -q
```

Expected: FAIL because the keyboard state object does not exist yet.

**Step 2: Implement the CLI skeleton**

Add `scripts/instinct_rl/play_mujoco.py` with arguments:
- `--load_run`
- `--mjcf`
- `--headless`
- `--depth-mode`
- `--keyboard_control`
- `--keyboard_linvel_step`
- `--keyboard_angvel`
- `--zero_act_until`
- `--sim_duration`
- `--record_npz`

**Step 3: Implement a viewer-agnostic keyboard state helper**

Implement:
- `KeyboardCommandState`
- `handle_key_event(...)`
- `expand_velocity_command_history(...)`

Use MuJoCo viewer callbacks when running with the interactive viewer. Keep the state object independent of the viewer API so it is unit-testable.

**Step 4: Re-run the focused tests**

Run:
```bash
pytest tests/test_parkour_mujoco_keyboard.py -q
```

Expected: PASS.

### Task 4: Reconstruct the parkour proprio observation stream

**Files:**
- Modify: `scripts/instinct_rl/parkour_mujoco_runtime.py`
- Create: `tests/test_parkour_mujoco_observation.py`

**Step 1: Write the failing observation-builder tests**

Test behaviors:
- maintain 8-step histories for `base_ang_vel`, `projected_gravity`, `velocity_commands`, `joint_pos`, `joint_vel`, `actions`
- flatten the histories in the same order as the Isaac policy group
- produce a proprio vector of width `896 - 128 = 768`

Run:
```bash
pytest tests/test_parkour_mujoco_observation.py -q
```

Expected: FAIL because the history buffer and flattening code are missing.

**Step 2: Implement the minimal observation pipeline**

Implement:
- `HistoryBuffer`
- `compute_base_ang_vel(...)`
- `compute_projected_gravity(...)`
- `compute_joint_pos_rel(...)`
- `compute_joint_vel_rel(...)`
- `assemble_actor_input(...)`

Implementation notes:
- mirror the observation order in `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:416`
- keep the pure observation assembly NumPy-first
- use the previous action as the `actions` history source

**Step 3: Re-run the focused tests**

Run:
```bash
pytest tests/test_parkour_mujoco_observation.py -q
```

Expected: PASS.

### Task 5: Implement the depth adapter with a debug fallback

**Files:**
- Modify: `scripts/instinct_rl/parkour_mujoco_runtime.py`
- Create: `tests/test_parkour_mujoco_depth.py`

**Step 1: Write the failing depth tests**

Test behaviors:
- `depth_mode=zeros` returns a correctly shaped `[1, 8, 18, 32]` tensor
- `depth_mode=mujoco` applies the same crop/resize/normalize contract as the parkour camera config
- delayed frame history honors the same output-frame count as the Isaac config

Run:
```bash
pytest tests/test_parkour_mujoco_depth.py -q
```

Expected: FAIL because the depth adapter does not exist yet.

**Step 2: Implement the debug fallback first**

Implement:
- `ZeroDepthAdapter`
- `DepthHistoryBuffer`

This is only for bring-up and must remain explicitly marked as a debug path.

**Step 3: Implement the MuJoCo depth path**

Implement:
- a programmatic head camera using the Isaac camera pose from `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py:352`
- MuJoCo depth rendering
- resize/crop/normalize to match the encoder input `[8, 18, 32]`

Implementation notes:
- there is no camera in the current MJCF, so the script must own camera placement
- do not silently replace `mujoco` depth with zeros if rendering breaks; fail loudly unless `--depth-mode=zeros` was requested

**Step 4: Re-run the focused tests**

Run:
```bash
pytest tests/test_parkour_mujoco_depth.py -q
```

Expected: PASS for the pure helpers. If a renderer-backed integration test is added, mark it with `@pytest.mark.integration` and skip when MuJoCo GPU/GL context is unavailable.

### Task 6: Implement the ONNX + MuJoCo control loop

**Files:**
- Modify: `scripts/instinct_rl/play_mujoco.py`
- Modify: `scripts/instinct_rl/parkour_mujoco_runtime.py`
- Create: `tests/test_parkour_mujoco_pd.py`

**Step 1: Write the failing control tests**

Test behaviors:
- actor input shape is exactly `[1, 896]`
- actor output shape is exactly `[1, 29]`
- policy actions map into joint targets in the verified joint order
- delayed PD control respects per-group gains and configured delay slots

Run:
```bash
pytest tests/test_parkour_mujoco_pd.py -q
```

Expected: FAIL because the control loop and PD adapter are missing.

**Step 2: Implement the ONNX inference wrapper**

Implement:
- `load_parkour_onnx_sessions(...)`
- `run_parkour_policy(...)`

Reuse the logic pattern from `source/instinctlab/instinctlab/tasks/parkour/scripts/onnxer.py`, but keep this runtime NumPy-first.

**Step 3: Implement the delayed PD adapter**

Implement:
- `DelayedPDController`
- per-joint gains based on `source/instinctlab/instinctlab/assets/unitree_g1.py:391`
- a fixed simulation loop with Isaac-equivalent timing (`sim.dt=0.005`, policy every 4 steps)

**Step 4: Wire the main script loop**

The main loop should:
- load run config and ONNX sessions
- build the joint map
- open MuJoCo model/data/viewer
- update keyboard commands
- step the observation history
- compute ONNX action
- convert action to joint targets
- apply PD torques
- log per-step summary and optional NPZ trace

**Step 5: Re-run the focused tests**

Run:
```bash
pytest tests/test_parkour_mujoco_pd.py -q
```

Expected: PASS.

### Task 7: Add end-to-end validation and usage docs

**Files:**
- Modify: `source/instinctlab/instinctlab/tasks/parkour/README.md`
- Optionally modify: `README.md`

**Step 1: Add a smoke-run command**

Target command:
```bash
python scripts/instinct_rl/play_mujoco.py \
  --load_run=20260327_163647 \
  --keyboard_control \
  --depth-mode=zeros \
  --sim_duration=5
```

Expected: viewer opens, policy loop runs, keyboard commands update the target command, and the robot remains numerically stable for a short debug run.

**Step 2: Add a fidelity run command**

Target command:
```bash
python scripts/instinct_rl/play_mujoco.py \
  --load_run=20260327_163647 \
  --keyboard_control \
  --depth-mode=mujoco
```

Expected: viewer opens, rendered depth is fed through the ONNX encoder, and command changes are visible in the locomotion behavior.

**Step 3: Add trace export validation**

Run:
```bash
python scripts/instinct_rl/play_mujoco.py \
  --load_run=20260327_163647 \
  --headless \
  --depth-mode=zeros \
  --sim_duration=2 \
  --record_npz=/tmp/parkour_mujoco_trace.npz
```

Expected: `/tmp/parkour_mujoco_trace.npz` exists and contains commands, observations, actions, and joint states for offline comparison against IsaacLab.

**Step 4: Update docs**

Document:
- required `pip install mujoco`
- keyboard keys and their effects
- meaning of `--depth-mode=zeros`
- known fidelity risks: no shoe-matched MJCF by default, no terrain parity with IsaacLab rough terrains

**Step 5: Final verification pass**

Run:
```bash
pytest tests/test_parkour_mujoco_runtime.py \
       tests/test_parkour_mujoco_joint_mapping.py \
       tests/test_parkour_mujoco_keyboard.py \
       tests/test_parkour_mujoco_observation.py \
       tests/test_parkour_mujoco_depth.py \
       tests/test_parkour_mujoco_pd.py -q
python -m py_compile scripts/instinct_rl/play_mujoco.py scripts/instinct_rl/parkour_mujoco_runtime.py
```

Expected: all targeted tests pass and both scripts compile.

## Risks To Track Explicitly

- **Depth fidelity risk:** MuJoCo currently has no configured head camera in the checked-in MJCF. The script must create and maintain the camera pose itself.
- **Asset mismatch risk:** the trained run uses `g1_29dof_torsoBase_popsicle_with_shoe.urdf`, while the checked-in MJCF is a generic G1 model. Keep `--mjcf` overrideable and do not present the default model as perfectly matched.
- **Terrain mismatch risk:** `scene.xml` provides only a flat plane, so “policy runs in MuJoCo” and “policy validates parkour transfer” are not the same claim.
- **Control mismatch risk:** Isaac uses delayed PD actuators and policy decimation 4 on top of `sim.dt=0.005`. Do not collapse this to a one-step torque loop without measuring the behavioral change.

## Completion Gate

Do not call the task complete until all of the following are true:

- `scripts/instinct_rl/play_mujoco.py` exists and supports keyboard control
- exported parkour ONNX files run inside the MuJoCo loop
- the script supports both `--depth-mode=zeros` and `--depth-mode=mujoco`
- a short headless smoke run works
- README usage is updated
- targeted unit tests pass
- the remaining fidelity gaps are documented honestly
