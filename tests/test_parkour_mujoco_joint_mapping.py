import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "instinct_rl" / "parkour_mujoco_runtime.py"


def load_module():
    spec = importlib.util.spec_from_file_location("parkour_mujoco_runtime", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_actuated_joint_order_from_logged_urdf():
    module = load_module()

    joint_order = module.extract_actuated_joint_order_from_urdf(
        Path("source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf")
    )

    assert len(joint_order) == 29
    assert joint_order[:6] == [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ]
    assert joint_order[-4:] == [
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]


def test_build_action_joint_map_accepts_exact_urdf_match():
    module = load_module()
    policy_joint_order = module.extract_actuated_joint_order_from_urdf(
        Path("source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf")
    )

    mapping = module.build_action_joint_map(policy_joint_order, policy_joint_order)

    assert mapping.policy_joint_order == policy_joint_order
    assert mapping.control_joint_order == policy_joint_order
    assert mapping.policy_to_control_indices == list(range(29))
    assert mapping.extra_control_joints == []


def test_extract_mujoco_actuator_order_follows_includes():
    module = load_module()

    joint_order = module.extract_mujoco_actuator_order(
        Path("source/instinctlab/instinctlab/assets/resources/unitree_g1/scene.xml")
    )

    assert "left_hip_pitch_joint" in joint_order
    assert "torso_joint" in joint_order
    assert "left_zero_joint" in joint_order


def test_build_action_joint_map_rejects_missing_policy_joints():
    module = load_module()
    policy_joint_order = module.extract_actuated_joint_order_from_urdf(
        Path("source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf")
    )
    incompatible_mujoco_order = module.extract_mujoco_actuator_order(
        Path("source/instinctlab/instinctlab/assets/resources/unitree_g1/scene.xml")
    )

    with pytest.raises(ValueError, match="waist_yaw_joint|left_elbow_joint"):
        module.build_action_joint_map(policy_joint_order, incompatible_mujoco_order)
