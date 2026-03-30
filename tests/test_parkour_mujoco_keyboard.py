import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np


RUNTIME_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "instinct_rl" / "parkour_mujoco_runtime.py"
PLAY_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "instinct_rl" / "play_mujoco.py"


def load_runtime_module():
    spec = importlib.util.spec_from_file_location("parkour_mujoco_runtime", RUNTIME_MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_handle_keyboard_command_updates_state():
    module = load_runtime_module()
    state = module.KeyboardCommandState()

    module.handle_keyboard_command("W", state, linvel_step=0.5, angvel=1.0)
    module.handle_keyboard_command("F", state, linvel_step=0.5, angvel=1.0)

    assert state.command.tolist() == [0.5, 0.0, 1.0]


def test_handle_keyboard_command_resets_yaw_and_zeroes_all():
    module = load_runtime_module()
    state = module.KeyboardCommandState(command=np.array([0.5, 0.0, -1.0], dtype=np.float32))

    module.handle_keyboard_command("S", state, linvel_step=0.5, angvel=1.0)
    assert state.command.tolist() == [0.5, 0.0, 0.0]

    module.handle_keyboard_command("X", state, linvel_step=0.5, angvel=1.0)
    assert state.command.tolist() == [0.0, 0.0, 0.0]


def test_expand_velocity_command_history_repeats_command_across_history():
    module = load_runtime_module()
    command = np.array([0.25, 0.0, -0.5], dtype=np.float32)

    history = module.expand_velocity_command_history(command, history_length=8)

    assert history.shape == (24,)
    assert history[:6].tolist() == [0.25, 0.0, -0.5, 0.25, 0.0, -0.5]
    assert history[-3:].tolist() == [0.25, 0.0, -0.5]


def test_handle_keyboard_command_toggles_elastic_band():
    module = load_runtime_module()
    state = module.KeyboardCommandState()

    module.handle_keyboard_command("9", state, linvel_step=0.5, angvel=1.0)
    assert state.elastic_band_enabled is False

    module.handle_keyboard_command("8", state, linvel_step=0.5, angvel=1.0)
    assert state.elastic_band_enabled is True


def test_play_mujoco_help_exits_successfully():
    result = subprocess.run(
        [sys.executable, str(PLAY_SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--depth-mode" in result.stdout
    assert "--keyboard_control" in result.stdout
