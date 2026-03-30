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


def test_resolve_joint_action_scale_from_logged_env_cfg():
    module = load_runtime_module()
    run_paths = module.resolve_logged_run_paths("20260327_163647")
    env_cfg = module.load_logged_yaml(run_paths.env_cfg)
    joint_order = module.extract_actuated_joint_order_from_urdf(module.resolve_default_model_path(run_paths))

    action_scale = module.resolve_joint_action_scale(joint_order, env_cfg)

    assert action_scale.shape == (29,)
    assert np.isclose(action_scale[0], 0.5475464652142303)
    assert np.isclose(action_scale[5], 0.43857731392336724)


def test_resolve_joint_drive_gains_from_logged_env_cfg():
    module = load_runtime_module()
    run_paths = module.resolve_logged_run_paths("20260327_163647")
    env_cfg = module.load_logged_yaml(run_paths.env_cfg)
    joint_order = module.extract_actuated_joint_order_from_urdf(module.resolve_default_model_path(run_paths))

    stiffness, damping = module.resolve_joint_drive_gains(joint_order, env_cfg)

    assert stiffness.shape == (29,)
    assert damping.shape == (29,)
    assert np.all(stiffness > 0.0)
    assert np.all(damping > 0.0)


def test_resolve_joint_effort_limits_from_logged_env_cfg():
    module = load_runtime_module()
    run_paths = module.resolve_logged_run_paths("20260327_163647")
    env_cfg = module.load_logged_yaml(run_paths.env_cfg)
    joint_order = module.extract_actuated_joint_order_from_urdf(module.resolve_default_model_path(run_paths))

    effort_limits = module.resolve_joint_effort_limits(joint_order, env_cfg)

    assert effort_limits.shape == (29,)
    assert np.isclose(effort_limits[0], 88.0)
    assert np.isclose(effort_limits[5], 50.0)


def test_resolve_joint_armature_from_logged_env_cfg():
    module = load_runtime_module()
    run_paths = module.resolve_logged_run_paths("20260327_163647")
    env_cfg = module.load_logged_yaml(run_paths.env_cfg)
    joint_order = module.extract_actuated_joint_order_from_urdf(module.resolve_default_model_path(run_paths))

    armature = module.resolve_joint_armature(joint_order, env_cfg)

    assert armature.shape == (29,)
    assert np.all(armature > 0.0)


def test_resolve_joint_init_targets_from_logged_env_cfg():
    module = load_runtime_module()
    run_paths = module.resolve_logged_run_paths("20260327_163647")
    env_cfg = module.load_logged_yaml(run_paths.env_cfg)
    joint_order = module.extract_actuated_joint_order_from_urdf(module.resolve_default_model_path(run_paths))

    init_targets = module.resolve_joint_init_targets(joint_order, env_cfg)

    assert init_targets.shape == (29,)
    assert np.isclose(init_targets[0], -0.312)
    assert np.isclose(init_targets[3], 0.669)
    assert np.isclose(init_targets[4], -0.363)
    assert np.isclose(init_targets[18], 0.6)


def test_delayed_pd_controller_delays_targets_before_applying_torque():
    module = load_runtime_module()
    controller = module.DelayedPDController(
        stiffness=np.array([10.0], dtype=np.float32),
        damping=np.array([1.0], dtype=np.float32),
        delay_steps=2,
        default_targets=np.array([0.0], dtype=np.float32),
    )
    current_qpos = np.array([0.0], dtype=np.float32)
    current_qvel = np.array([0.0], dtype=np.float32)

    torque_0 = controller.compute_torque(np.array([1.0], dtype=np.float32), current_qpos, current_qvel)
    torque_1 = controller.compute_torque(np.array([1.0], dtype=np.float32), current_qpos, current_qvel)
    torque_2 = controller.compute_torque(np.array([1.0], dtype=np.float32), current_qpos, current_qvel)

    assert torque_0.tolist() == [0.0]
    assert torque_1.tolist() == [0.0]
    assert torque_2.tolist() == [10.0]
