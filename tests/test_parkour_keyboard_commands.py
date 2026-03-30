import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "instinctlab"
    / "instinctlab"
    / "tasks"
    / "parkour"
    / "scripts"
    / "keyboard_commands.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("parkour_keyboard_commands", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_keyboard_command_limits_from_logged_env_cfg():
    module = load_module()
    env_cfg = {
        "commands": {
            "base_velocity": {
                "only_positive_lin_vel_x": True,
                "velocity_ranges": {
                    "perlin_rough": {"lin_vel_x": (0.45, 1.0), "ang_vel_z": (-1.0, 1.0)},
                    "boxes": {"lin_vel_x": (0.45, 0.8), "ang_vel_z": (-1.0, 1.0)},
                    "perlin_rough_stand": {"lin_vel_x": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
                },
            }
        }
    }

    limits = module.resolve_keyboard_command_limits(env_cfg)

    assert limits.lin_vel_x_min == 0.0
    assert limits.lin_vel_x_max == 1.0
    assert limits.ang_vel_z_min == -1.0
    assert limits.ang_vel_z_max == 1.0


def test_keyboard_controller_clamps_forward_command_and_rolls_history():
    module = load_module()
    controller = module.ParkourKeyboardCommandController(
        num_envs=1,
        history_length=4,
        device="cpu",
        limits=module.KeyboardCommandLimits(0.0, 1.0, -1.0, 1.0),
        linvel_step=0.5,
        angvel=1.0,
    )

    controller.forward()
    controller.forward()
    controller.forward()
    obs_1 = controller.build_observation()
    controller.zero_all()
    obs_2 = controller.build_observation()

    assert controller.command[0, 0].item() == 0.0
    assert obs_1.shape == (1, 12)
    assert obs_1[0, -3:].tolist() == [1.0, 0.0, 0.0]
    assert obs_2[0, -6:-3].tolist() == [1.0, 0.0, 0.0]
    assert obs_2[0, -3:].tolist() == [0.0, 0.0, 0.0]


def test_keyboard_controller_sets_yaw_commands_with_limits():
    module = load_module()
    controller = module.ParkourKeyboardCommandController(
        num_envs=1,
        history_length=2,
        device="cpu",
        limits=module.KeyboardCommandLimits(0.0, 1.0, -1.0, 1.0),
        linvel_step=0.5,
        angvel=1.0,
    )

    controller.yaw_positive()
    assert controller.command[0, 2].item() == 1.0

    controller.yaw_negative()
    assert controller.command[0, 2].item() == -1.0

    controller.reset_yaw()
    assert controller.command[0, 2].item() == 0.0
