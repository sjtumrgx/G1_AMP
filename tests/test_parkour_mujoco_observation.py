import importlib.util
from pathlib import Path

import numpy as np


RUNTIME_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "instinct_rl" / "parkour_mujoco_runtime.py"


def load_runtime_module():
    spec = importlib.util.spec_from_file_location("parkour_mujoco_runtime", RUNTIME_MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_history_buffer_keeps_latest_samples_in_order():
    module = load_runtime_module()
    history = module.HistoryBuffer(feature_dim=2, history_length=3)

    history.push(np.array([1.0, 2.0], dtype=np.float32))
    history.push(np.array([3.0, 4.0], dtype=np.float32))
    history.push(np.array([5.0, 6.0], dtype=np.float32))

    assert history.flatten().tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_history_buffer_drops_oldest_sample_on_overflow():
    module = load_runtime_module()
    history = module.HistoryBuffer(feature_dim=2, history_length=3)

    history.push(np.array([1.0, 2.0], dtype=np.float32))
    history.push(np.array([3.0, 4.0], dtype=np.float32))
    history.push(np.array([5.0, 6.0], dtype=np.float32))
    history.push(np.array([7.0, 8.0], dtype=np.float32))

    assert history.flatten().tolist() == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def test_parkour_proprio_builder_returns_expected_width():
    module = load_runtime_module()
    builder = module.ParkourProprioObservationBuilder(joint_count=29, history_length=8)

    builder.push(
        base_ang_vel=np.ones(3, dtype=np.float32),
        projected_gravity=np.ones(3, dtype=np.float32) * 2.0,
        velocity_command=np.ones(3, dtype=np.float32) * 3.0,
        joint_pos_rel=np.ones(29, dtype=np.float32) * 4.0,
        joint_vel_rel=np.ones(29, dtype=np.float32) * 5.0,
        action=np.ones(29, dtype=np.float32) * 6.0,
    )
    proprio = builder.build()

    assert proprio.shape == (768,)


def test_parkour_proprio_builder_reports_expected_component_slices():
    module = load_runtime_module()
    builder = module.ParkourProprioObservationBuilder(joint_count=29, history_length=8)

    slices = builder.component_slices()

    assert slices["base_ang_vel"] == slice(0, 24)
    assert slices["projected_gravity"] == slice(24, 48)
    assert slices["velocity_commands"] == slice(48, 72)
    assert slices["joint_pos"] == slice(72, 304)
    assert slices["joint_vel"] == slice(304, 536)
    assert slices["actions"] == slice(536, 768)


def test_resolve_policy_observation_scales_from_logged_env_cfg():
    module = load_runtime_module()
    run_paths = module.resolve_logged_run_paths("20260327_163647")
    env_cfg = module.load_logged_yaml(run_paths.env_cfg)

    scales = module.resolve_policy_observation_scales(env_cfg)

    assert scales["base_ang_vel"] == 0.25
    assert scales["joint_vel"] == 0.05
    assert scales["projected_gravity"] == 1.0


def test_parkour_proprio_builder_applies_component_scales():
    module = load_runtime_module()
    builder = module.ParkourProprioObservationBuilder(
        joint_count=2,
        history_length=2,
        observation_scales={
            "base_ang_vel": 0.25,
            "projected_gravity": 1.0,
            "velocity_commands": 1.0,
            "joint_pos": 1.0,
            "joint_vel": 0.05,
            "actions": 1.0,
        },
    )

    builder.push(
        base_ang_vel=np.array([4.0, 0.0, 0.0], dtype=np.float32),
        projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        velocity_command=np.zeros(3, dtype=np.float32),
        joint_pos_rel=np.zeros(2, dtype=np.float32),
        joint_vel_rel=np.array([10.0, -10.0], dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
    )
    proprio = builder.build()
    slices = builder.component_slices()

    assert np.isclose(proprio[slices["base_ang_vel"]][-3], 1.0)
    assert np.allclose(proprio[slices["joint_vel"]][-2:], np.array([0.5, -0.5], dtype=np.float32))
