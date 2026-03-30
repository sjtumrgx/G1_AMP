import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "instinctlab"
    / "instinctlab"
    / "tasks"
    / "parkour"
    / "scripts"
    / "play_runtime.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("parkour_play_runtime", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_fake_env_cfg():
    return SimpleNamespace(
        episode_length_s=20.0,
        scene=SimpleNamespace(
            camera=SimpleNamespace(debug_vis=False),
        ),
        commands=SimpleNamespace(
            base_velocity=SimpleNamespace(
                debug_vis=False,
                goal_vel_visualizer_cfg=SimpleNamespace(markers={"arrow": SimpleNamespace(scale=(1.0, 0.2, 0.2))}),
                current_vel_visualizer_cfg=SimpleNamespace(markers={"arrow": SimpleNamespace(scale=(1.0, 0.2, 0.2))}),
            )
        ),
        events=SimpleNamespace(
            reset_base=SimpleNamespace(
                params={"pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)}}
            )
        ),
        terminations=SimpleNamespace(
            time_out="keep",
            terrain_out_bound="keep",
            base_contact="keep",
            root_height="keep",
        ),
        observations=SimpleNamespace(
            policy=SimpleNamespace(depth_image=SimpleNamespace(params={"debug_vis": False})),
            critic=SimpleNamespace(depth_image=SimpleNamespace(params={"debug_vis": False})),
        ),
    )


def test_add_play_runtime_args_accepts_new_flags():
    module = load_module()
    parser = argparse.ArgumentParser()

    module.add_play_runtime_args(parser)
    args = parser.parse_args(
        [
            "--center_spawn",
            "--disable_auto_reset",
            "--show_depth_window",
            "--show_depth_coverage",
            "--show_command_arrow",
            "--video_layout",
            "quad",
            "--video_duration_s",
            "120",
            "--video_fps",
            "15",
            "--video_frame_stride",
            "4",
        ]
    )

    assert args.center_spawn is True
    assert args.disable_auto_reset is True
    assert args.show_depth_window is True
    assert args.show_depth_coverage is True
    assert args.show_command_arrow is True
    assert args.video_layout == "quad"
    assert args.video_duration_s == 120.0
    assert args.video_fps == 15.0
    assert args.video_frame_stride == 4


def test_select_center_terrain_origin_returns_middle_origin():
    module = load_module()
    terrain_origins = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 20.0, 0.0]],
            [[10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [10.0, 20.0, 0.0]],
            [[20.0, 0.0, 0.0], [20.0, 10.0, 0.0], [20.0, 20.0, 0.0]],
        ],
        dtype=np.float32,
    )

    center_origin = module.select_center_terrain_origin(terrain_origins)

    assert center_origin.tolist() == [10.0, 10.0, 0.0]


def test_apply_play_runtime_overrides_disables_resets_and_enables_depth_debug():
    module = load_module()
    env_cfg = make_fake_env_cfg()
    options = SimpleNamespace(
        disable_auto_reset=True,
        show_depth_window=True,
        show_depth_coverage=True,
        show_command_arrow=True,
        center_spawn=True,
    )

    module.apply_play_runtime_overrides(env_cfg, options)

    assert env_cfg.episode_length_s == module.NO_AUTO_RESET_EPISODE_LENGTH_S
    assert env_cfg.scene.camera.debug_vis is True
    assert env_cfg.observations.policy.depth_image.params["debug_vis"] is False
    assert env_cfg.observations.critic.depth_image.params["debug_vis"] is False
    assert env_cfg.commands.base_velocity.debug_vis is True
    assert env_cfg.commands.base_velocity.goal_vel_visualizer_cfg.markers["arrow"].scale == (1.5, 0.45, 0.45)
    assert env_cfg.events.reset_base.params["pose_range"]["x"] == (0.0, 0.0)
    assert env_cfg.events.reset_base.params["pose_range"]["y"] == (0.0, 0.0)
    assert env_cfg.terminations.time_out is None
    assert env_cfg.terminations.terrain_out_bound is None
    assert env_cfg.terminations.base_contact is None
    assert env_cfg.terminations.root_height is None


def test_build_default_tracking_camera_specs_returns_three_named_views():
    module = load_module()

    specs = module.build_default_tracking_camera_specs()

    assert [spec.name for spec in specs] == ["hero", "side", "overview"]


def test_compute_tracking_camera_views_rotates_offsets_with_yaw():
    module = load_module()
    specs = [
        module.TrackingCameraSpec(
            name="test",
            eye_offset=(2.0, 0.0, 1.5),
            target_offset=(0.5, 0.0, 0.8),
        )
    ]

    views = module.compute_tracking_camera_views(
        root_position=np.array([1.0, 2.0, 0.0], dtype=np.float32),
        root_yaw=np.pi / 2.0,
        specs=specs,
    )

    eye, target = views["test"]
    np.testing.assert_allclose(eye, np.array([1.0, 4.0, 1.5], dtype=np.float32))
    np.testing.assert_allclose(target, np.array([1.0, 2.5, 0.8], dtype=np.float32))


def test_normalize_depth_frame_for_display_returns_rgb_image():
    module = load_module()
    depth_frame = np.array([[0.0, 1.25], [2.5, 0.625]], dtype=np.float32)

    image = module.normalize_depth_frame_for_display(depth_frame, depth_range=(0.0, 2.5))

    assert image.shape == (2, 2, 3)
    assert image.dtype == np.uint8
    assert image[0, 0].tolist() == [0, 0, 0]
    assert image[1, 0].tolist() == [255, 255, 255]


def test_compose_recording_frame_builds_2x2_panel_layout():
    module = load_module()
    rgb_frames = {
        "hero": np.full((4, 6, 3), 10, dtype=np.uint8),
        "side": np.full((4, 6, 3), 20, dtype=np.uint8),
        "overview": np.full((4, 6, 3), 30, dtype=np.uint8),
    }
    depth_frame = np.full((4, 6, 3), 40, dtype=np.uint8)

    composite = module.compose_recording_frame(rgb_frames, depth_frame)

    assert composite.shape == (8, 12, 3)
    assert composite[0, 0].tolist() == [10, 10, 10]
    assert composite[0, 6].tolist() == [20, 20, 20]
    assert composite[4, 0].tolist() == [30, 30, 30]
    assert composite[4, 6].tolist() == [40, 40, 40]


def test_validate_isaacsim_python_environment_rejects_numpy_2():
    module = load_module()

    try:
        module.validate_isaacsim_python_environment(numpy_version="2.4.3")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected numpy 2.x to be rejected.")

    assert "numpy==1.26.0" in message
    assert "numpy 2.4.3" in message


def test_validate_isaacsim_python_environment_accepts_numpy_1_26():
    module = load_module()

    module.validate_isaacsim_python_environment(numpy_version="1.26.0")


def test_ensure_sensor_initialized_invokes_callback_when_needed():
    module = load_module()

    class FakeSensor:
        def __init__(self):
            self.is_initialized = False
            self.calls = 0

        def _initialize_callback(self, event):
            self.calls += 1
            self.is_initialized = True

    sensor = FakeSensor()

    module.ensure_sensor_initialized(sensor)

    assert sensor.is_initialized is True
    assert sensor.calls == 1


def test_ensure_sensor_initialized_raises_if_callback_does_not_initialize():
    module = load_module()

    class FakeSensor:
        def __init__(self):
            self.is_initialized = False

        def _initialize_callback(self, event):
            return None

    try:
        module.ensure_sensor_initialized(FakeSensor(), sensor_name="bad_sensor")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected initialization failure to raise.")

    assert "bad_sensor" in message


def test_create_keyboard_event_subscription_keeps_handle_and_unsubscribes():
    module = load_module()

    class FakeAppWindow:
        def __init__(self):
            self.keyboard = object()

        def get_keyboard(self):
            return self.keyboard

    class FakeInputInterface:
        def __init__(self):
            self.subscribed = []
            self.unsubscribed = []

        def subscribe_to_keyboard_events(self, keyboard, callback):
            token = object()
            self.subscribed.append((keyboard, callback, token))
            return token

        def unsubscribe_to_keyboard_events(self, keyboard, token):
            self.unsubscribed.append((keyboard, token))

    callback = lambda event: None
    app_window = FakeAppWindow()
    input_interface = FakeInputInterface()

    subscription = module.create_keyboard_event_subscription(
        callback=callback,
        app_window=app_window,
        input_interface=input_interface,
    )

    assert subscription.callback is callback
    assert subscription.keyboard is app_window.keyboard
    token = input_interface.subscribed[0][2]
    assert subscription.handle is token

    subscription.close()

    assert input_interface.unsubscribed == [(app_window.keyboard, token)]
    assert subscription.handle is None


def test_select_map_center_xy_prefers_geometric_middle():
    module = load_module()
    terrain_origins = np.array(
        [
            [[-12.0, -36.0, 0.0], [-12.0, -28.0, 0.0], [-12.0, -20.0, 0.0], [-12.0, -12.0, 0.0]],
            [[-4.0, -36.0, 0.0], [-4.0, -28.0, 0.0], [-4.0, -20.0, 0.0], [-4.0, -12.0, 0.0]],
            [[4.0, -36.0, 0.0], [4.0, -28.0, 0.0], [4.0, -20.0, 0.0], [4.0, -12.0, 0.0]],
            [[12.0, -36.0, 0.0], [12.0, -28.0, 0.0], [12.0, -20.0, 0.0], [12.0, -12.0, 0.0]],
        ],
        dtype=np.float32,
    )

    center_xy = module.select_map_center_xy(terrain_origins)

    assert center_xy.tolist() == [0.0, -24.0]


def test_select_center_terrain_origin_uses_map_center_xy_with_nearest_height():
    module = load_module()
    terrain_origins = np.array(
        [
            [[-12.0, -36.0, -0.90], [-12.0, -28.0, -0.85], [-12.0, -20.0, -0.84], [-12.0, -12.0, -0.83]],
            [[-4.0, -36.0, -0.87], [-4.0, -28.0, -0.82], [-4.0, -20.0, -0.81], [-4.0, -12.0, -0.80]],
            [[4.0, -36.0, -0.79], [4.0, -28.0, -0.78], [4.0, -20.0, -0.77], [4.0, -12.0, -0.76]],
            [[12.0, -36.0, -0.75], [12.0, -28.0, -0.74], [12.0, -20.0, -0.73], [12.0, -12.0, -0.72]],
        ],
        dtype=np.float32,
    )

    center_origin = module.select_center_terrain_origin(terrain_origins)

    np.testing.assert_allclose(center_origin, np.array([0.0, -24.0, -0.82], dtype=np.float32))


def test_apply_play_runtime_overrides_disables_built_in_depth_history_debug_window():
    module = load_module()
    env_cfg = make_fake_env_cfg()
    options = SimpleNamespace(
        disable_auto_reset=False,
        show_depth_window=True,
        show_depth_coverage=False,
        show_command_arrow=False,
        center_spawn=False,
    )

    module.apply_play_runtime_overrides(env_cfg, options)

    assert env_cfg.observations.policy.depth_image.params["debug_vis"] is False
    assert env_cfg.observations.critic.depth_image.params["debug_vis"] is False
    assert env_cfg.commands.base_velocity.debug_vis is False


def test_resolve_video_capture_settings_defaults_to_sim_fps_and_frame_count():
    module = load_module()

    settings = module.resolve_video_capture_settings(
        video_length=3000,
        video_duration_s=None,
        step_dt=0.02,
        video_frame_stride=1,
        video_fps=None,
    )

    assert settings.max_frames == 3000
    assert settings.video_fps == 50.0
    assert settings.frame_stride == 1


def test_resolve_video_capture_settings_respects_duration_and_stride():
    module = load_module()

    settings = module.resolve_video_capture_settings(
        video_length=3000,
        video_duration_s=120.0,
        step_dt=0.02,
        video_frame_stride=4,
        video_fps=None,
    )

    assert settings.video_fps == 12.5
    assert settings.max_frames == 1500
    assert settings.frame_stride == 4
