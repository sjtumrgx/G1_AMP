import argparse
from datetime import datetime
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
        seed=7,
        episode_length_s=20.0,
        curriculum=SimpleNamespace(terrain_levels="keep"),
        scene=SimpleNamespace(
            num_envs=8,
            camera=SimpleNamespace(debug_vis=False, data_types=["distance_to_image_plane"]),
            terrain=SimpleNamespace(
                debug_vis=False,
                terrain_generator=SimpleNamespace(
                    seed=0,
                    precomputed_tile_wall_edges=None,
                )
            ),
            motion_reference=SimpleNamespace(debug_vis=False, visualizing_marker_types=[]),
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
                params={
                    "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
                    "velocity_range": {
                        "x": (-0.2, 0.2),
                        "y": (-0.2, 0.2),
                        "z": (-0.2, 0.2),
                        "roll": (-0.2, 0.2),
                        "pitch": (-0.2, 0.2),
                        "yaw": (-0.2, 0.2),
                    },
                }
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
            "--show_elevation_map_window",
            "--show_elevation_viewport",
            "--normals_panel",
            "--route_overlay",
            "--foot_contact_overlay",
            "--ghost_reference",
            "--obstacle_edges",
            "--show_command_arrow",
            "--video_layout",
            "quad",
            "--save_route",
            "route.json",
            "--replay_route",
            "route.json",
            "--route_interval_m",
            "0.35",
            "--route_lookahead_m",
            "0.9",
            "--route_goal_tolerance_m",
            "0.15",
            "--route_cruise_speed",
            "0.7",
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
    assert args.show_elevation_map_window is True
    assert args.show_elevation_viewport is True
    assert args.normals_panel is True
    assert args.route_overlay is True
    assert args.foot_contact_overlay is True
    assert args.ghost_reference is True
    assert args.obstacle_edges is True
    assert args.show_command_arrow is True
    assert args.video_layout == "quad"
    assert args.save_route == "route.json"
    assert args.replay_route == "route.json"
    assert args.route_interval_m == 0.35
    assert args.route_lookahead_m == 0.9
    assert args.route_goal_tolerance_m == 0.15
    assert args.route_cruise_speed == 0.7
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
    assert env_cfg.events.reset_base.params["pose_range"]["yaw"] == (0.0, 0.0)
    assert env_cfg.events.reset_base.params["velocity_range"]["x"] == (0.0, 0.0)
    assert env_cfg.events.reset_base.params["velocity_range"]["yaw"] == (0.0, 0.0)
    assert env_cfg.curriculum.terrain_levels is None
    assert env_cfg.terminations.time_out is None
    assert env_cfg.terminations.terrain_out_bound is None
    assert env_cfg.terminations.base_contact is None
    assert env_cfg.terminations.root_height is None


def test_resolve_play_visualization_config_uses_env_defaults():
    module = load_module()
    env_cfg = make_fake_env_cfg()
    env_cfg.play_visualization = SimpleNamespace(
        depth_window=True,
        depth_coverage=True,
        elevation_map_window=True,
        elevation_viewport=True,
        normals_panel=True,
        route_overlay=True,
        foot_contact_overlay=True,
        ghost_reference=True,
        obstacle_edges=True,
    )
    options = SimpleNamespace(
        show_depth_window=None,
        show_depth_coverage=None,
        show_elevation_map_window=None,
        show_elevation_viewport=None,
        normals_panel=None,
        route_overlay=None,
        foot_contact_overlay=None,
        ghost_reference=None,
        obstacle_edges=None,
    )

    resolved = module.resolve_play_visualization_config(env_cfg, options)

    assert resolved.depth_window is True
    assert resolved.depth_coverage is True
    assert resolved.elevation_map_window is True
    assert resolved.elevation_viewport is True
    assert resolved.normals_panel is True
    assert resolved.route_overlay is True
    assert resolved.foot_contact_overlay is True
    assert resolved.ghost_reference is True
    assert resolved.obstacle_edges is True


def test_resolve_play_visualization_config_applies_cli_overrides():
    module = load_module()
    env_cfg = make_fake_env_cfg()
    env_cfg.play_visualization = SimpleNamespace(
        depth_window=True,
        depth_coverage=True,
        elevation_map_window=False,
        elevation_viewport=False,
        normals_panel=False,
        route_overlay=False,
        foot_contact_overlay=True,
        ghost_reference=False,
        obstacle_edges=True,
    )
    options = SimpleNamespace(
        show_depth_window=False,
        show_depth_coverage=None,
        show_elevation_map_window=True,
        show_elevation_viewport=True,
        normals_panel=True,
        route_overlay=True,
        foot_contact_overlay=False,
        ghost_reference=True,
        obstacle_edges=False,
    )

    resolved = module.resolve_play_visualization_config(env_cfg, options)

    assert resolved.depth_window is False
    assert resolved.depth_coverage is True
    assert resolved.elevation_map_window is True
    assert resolved.elevation_viewport is True
    assert resolved.normals_panel is True
    assert resolved.route_overlay is True
    assert resolved.foot_contact_overlay is False
    assert resolved.ghost_reference is True
    assert resolved.obstacle_edges is False


def test_apply_play_runtime_overrides_reads_visualization_namespace():
    module = load_module()
    env_cfg = make_fake_env_cfg()
    options = SimpleNamespace(
        disable_auto_reset=False,
        show_command_arrow=False,
        center_spawn=False,
        keyboard_control=False,
        replay_route=None,
        visualization=SimpleNamespace(
            depth_window=True,
            depth_coverage=True,
            elevation_map_window=False,
            elevation_viewport=False,
            normals_panel=False,
            route_overlay=False,
            foot_contact_overlay=False,
            ghost_reference=False,
            obstacle_edges=False,
        ),
    )

    module.apply_play_runtime_overrides(env_cfg, options)

    assert env_cfg.scene.camera.debug_vis is True
    assert env_cfg.observations.policy.depth_image.params["debug_vis"] is False
    assert env_cfg.observations.critic.depth_image.params["debug_vis"] is False


def test_apply_play_runtime_overrides_enables_ghost_reference_and_obstacle_edges():
    module = load_module()
    env_cfg = make_fake_env_cfg()
    options = SimpleNamespace(
        disable_auto_reset=False,
        show_command_arrow=False,
        center_spawn=False,
        keyboard_control=False,
        replay_route=None,
        visualization=module.PlayVisualizationConfig(
            ghost_reference=True,
            obstacle_edges=True,
        ),
    )

    module.apply_play_runtime_overrides(env_cfg, options)

    assert env_cfg.scene.motion_reference.debug_vis is True
    assert env_cfg.scene.motion_reference.visualizing_robot_from == "reference_frame"
    assert env_cfg.scene.motion_reference.visualizing_robot_offset == (0.0, 1.5, 0.0)
    assert env_cfg.scene.terrain.debug_vis is True


def test_apply_play_runtime_overrides_treats_keyboard_control_like_center_spawn():
    module = load_module()
    env_cfg = make_fake_env_cfg()
    options = SimpleNamespace(
        disable_auto_reset=False,
        show_depth_window=False,
        show_depth_coverage=False,
        show_command_arrow=False,
        center_spawn=False,
        keyboard_control=True,
    )

    module.apply_play_runtime_overrides(env_cfg, options)

    assert env_cfg.events.reset_base.params["pose_range"]["x"] == (0.0, 0.0)
    assert env_cfg.events.reset_base.params["pose_range"]["y"] == (0.0, 0.0)
    assert env_cfg.events.reset_base.params["pose_range"]["yaw"] == (0.0, 0.0)
    assert env_cfg.events.reset_base.params["velocity_range"]["x"] == (0.0, 0.0)
    assert env_cfg.curriculum.terrain_levels is None


def test_resolve_play_seed_prefers_cli_then_route_then_agent_seed():
    module = load_module()

    assert module.resolve_play_seed(cli_seed=123, route_seed=77, agent_seed=42) == 123
    assert module.resolve_play_seed(cli_seed=None, route_seed=77, agent_seed=42) == 77
    assert module.resolve_play_seed(cli_seed=None, route_seed=None, agent_seed=42) == 42


def test_prepare_play_env_cfg_sets_seed_and_single_env_for_route_replay():
    module = load_module()
    env_cfg = make_fake_env_cfg()
    options = SimpleNamespace(
        disable_auto_reset=False,
        show_depth_window=False,
        show_depth_coverage=False,
        show_command_arrow=False,
        center_spawn=False,
        keyboard_control=False,
        replay_route="saved-route.json",
    )

    module.prepare_play_env_cfg(env_cfg, options, seed=123)

    assert env_cfg.seed == 123
    assert env_cfg.scene.terrain.terrain_generator.seed == 123
    assert env_cfg.scene.num_envs == 1
    assert env_cfg.episode_length_s == module.NO_AUTO_RESET_EPISODE_LENGTH_S
    assert env_cfg.events.reset_base.params["pose_range"]["yaw"] == (0.0, 0.0)
    assert env_cfg.terminations.time_out is None


def test_prepare_play_env_cfg_sets_single_env_for_onnx_play():
    module = load_module()
    env_cfg = make_fake_env_cfg()
    options = SimpleNamespace(
        disable_auto_reset=False,
        show_depth_window=False,
        show_depth_coverage=False,
        show_command_arrow=False,
        center_spawn=False,
        keyboard_control=False,
        replay_route=None,
        exportonnx=True,
        useonnx=True,
    )

    module.prepare_play_env_cfg(env_cfg, options, seed=123)

    assert env_cfg.seed == 123
    assert env_cfg.scene.num_envs == 1
    assert env_cfg.episode_length_s == module.NO_AUTO_RESET_EPISODE_LENGTH_S


def test_prepare_play_env_cfg_injects_precomputed_tile_wall_edges_from_route_artifact():
    module = load_module()
    env_cfg = make_fake_env_cfg()
    options = SimpleNamespace(
        disable_auto_reset=False,
        show_depth_window=False,
        show_depth_coverage=False,
        show_command_arrow=False,
        center_spawn=False,
        keyboard_control=False,
        replay_route="saved-route.json",
    )
    route_artifact = SimpleNamespace(
        tile_wall_edges=[
            [
                [
                    {"side": "left", "xy": (-4.05, -4.0), "width": 0.05, "height": 8.0},
                ]
            ]
        ]
    )

    module.prepare_play_env_cfg(env_cfg, options, seed=123, route_artifact=route_artifact)

    assert env_cfg.scene.terrain.terrain_generator.precomputed_tile_wall_edges == route_artifact.tile_wall_edges


def test_prepare_play_env_cfg_enables_normals_data_type_when_visualization_requests_it():
    module = load_module()
    env_cfg = make_fake_env_cfg()
    options = SimpleNamespace(
        disable_auto_reset=False,
        show_command_arrow=False,
        center_spawn=False,
        keyboard_control=False,
        replay_route=None,
        visualization=module.PlayVisualizationConfig(normals_panel=True),
    )

    module.prepare_play_env_cfg(env_cfg, options, seed=123)

    assert env_cfg.scene.camera.data_types == ["distance_to_image_plane", "normals"]


def test_build_default_tracking_camera_specs_returns_three_named_views():
    module = load_module()

    specs = module.build_default_tracking_camera_specs()

    assert [spec.name for spec in specs] == ["hero", "side", "overview"]


def test_build_default_tracking_camera_specs_biases_hero_and_side_views_downward():
    module = load_module()

    specs = {spec.name: spec for spec in module.build_default_tracking_camera_specs()}
    hero = specs["hero"]
    side = specs["side"]

    assert hero.eye_offset[2] < 1.8
    assert hero.target_offset[2] <= 0.6
    assert side.eye_offset[1] > -4.5
    assert side.eye_offset[2] < 1.6
    assert side.target_offset[2] <= 0.6


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


def test_normalize_normals_frame_for_display_maps_unit_vectors_to_rgb():
    module = load_module()
    normals = np.array(
        [
            [[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    image = module.normalize_normals_frame_for_display(normals)

    assert image.shape == (1, 2, 3)
    assert image.dtype == np.uint8
    assert image[0, 0].tolist() == [0, 128, 255]
    assert image[0, 1].tolist() == [128, 128, 128]


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


def test_compose_recording_frame_expands_to_3x2_layout_when_extra_panel_is_present():
    module = load_module()
    rgb_frames = {
        "hero": np.full((4, 6, 3), 10, dtype=np.uint8),
        "side": np.full((4, 6, 3), 20, dtype=np.uint8),
        "overview": np.full((4, 6, 3), 30, dtype=np.uint8),
    }
    depth_frame = np.full((4, 6, 3), 40, dtype=np.uint8)
    normals_panel = np.full((4, 6, 3), 50, dtype=np.uint8)

    composite = module.compose_recording_frame(rgb_frames, depth_frame, extra_panels=[normals_panel])

    assert composite.shape == (8, 18, 3)
    assert composite[0, 0].tolist() == [10, 10, 10]
    assert composite[0, 6].tolist() == [20, 20, 20]
    assert composite[0, 12].tolist() == [30, 30, 30]
    assert composite[4, 0].tolist() == [40, 40, 40]
    assert composite[4, 6].tolist() == [50, 50, 50]


def test_compose_recording_frame_uses_bottom_right_slot_for_second_extra_panel():
    module = load_module()
    rgb_frames = {
        "hero": np.full((4, 6, 3), 10, dtype=np.uint8),
        "side": np.full((4, 6, 3), 20, dtype=np.uint8),
        "overview": np.full((4, 6, 3), 30, dtype=np.uint8),
    }
    depth_frame = np.full((4, 6, 3), 40, dtype=np.uint8)
    normals_panel = np.full((4, 6, 3), 50, dtype=np.uint8)
    route_map_panel = np.full((4, 6, 3), 60, dtype=np.uint8)

    composite = module.compose_recording_frame(
        rgb_frames,
        depth_frame,
        extra_panels=[normals_panel, route_map_panel],
    )

    assert composite.shape == (8, 18, 3)
    assert composite[4, 12].tolist() == [60, 60, 60]


def test_build_live_preview_panels_returns_depth_elevation_and_normals_windows():
    module = load_module()
    depth_panel = np.full((4, 6, 3), 40, dtype=np.uint8)
    elevation_map_panel = np.full((4, 6, 3), 45, dtype=np.uint8)
    normals_panel = np.full((4, 6, 3), 50, dtype=np.uint8)

    panels = module.build_live_preview_panels(
        depth_panel=depth_panel,
        elevation_map_panel=elevation_map_panel,
        normals_panel=normals_panel,
        show_depth_window=True,
        show_elevation_map_window=True,
        show_normals_window=True,
        scale=2.0,
    )

    assert [name for name, _ in panels] == [
        module.DEPTH_WINDOW_NAME,
        module.ELEVATION_MAP_WINDOW_NAME,
        module.NORMALS_WINDOW_NAME,
    ]
    assert panels[0][1].shape == (8, 12, 3)
    assert panels[1][1].shape == (8, 12, 3)
    assert panels[2][1].shape == (8, 12, 3)


def test_render_route_map_panel_draws_route_state_on_nonempty_canvas():
    module = load_module()
    terrain_origins = np.array(
        [
            [[-4.0, -4.0, 0.0], [-4.0, 4.0, 0.0]],
            [[4.0, -4.0, 0.0], [4.0, 4.0, 0.0]],
        ],
        dtype=np.float32,
    )
    tile_wall_edges = [
        [[{"side": "left", "xy": [-4.05, -4.0], "width": 0.05, "height": 8.0}], []],
        [[], []],
    ]

    panel = module.render_route_map_panel(
        terrain_origins=terrain_origins,
        center_origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        current_position_xy=[0.5, -0.25],
        current_yaw=0.4,
        route_waypoints_xy=[[0.0, 0.0], [2.0, 1.0], [3.0, 2.5]],
        tile_wall_edges=tile_wall_edges,
        image_size=(96, 96),
    )

    assert panel.shape == (96, 96, 3)
    assert panel.dtype == np.uint8
    assert np.unique(panel.reshape(-1, 3), axis=0).shape[0] > 4


def test_render_route_map_panel_ignores_faraway_tiles_outside_route_window():
    module = load_module()
    terrain_origins = np.array(
        [
            [[-12.0, -12.0, 0.0], [-12.0, -4.0, 0.0], [-12.0, 4.0, 0.0], [-12.0, 12.0, 0.0]],
            [[-4.0, -12.0, 0.0], [-4.0, -4.0, 0.0], [-4.0, 4.0, 0.0], [-4.0, 12.0, 0.0]],
            [[4.0, -12.0, 0.0], [4.0, -4.0, 0.0], [4.0, 4.0, 0.0], [4.0, 12.0, 0.0]],
            [[12.0, -12.0, 0.0], [12.0, -4.0, 0.0], [12.0, 4.0, 0.0], [12.0, 12.0, 0.0]],
        ],
        dtype=np.float32,
    )
    focused_origins = terrain_origins[:2, :2]
    route_waypoints = [[-11.0, -11.0], [-8.0, -8.5], [-5.0, -5.0]]

    panel_from_full_grid = module.render_route_map_panel(
        terrain_origins=terrain_origins,
        current_position_xy=[-5.5, -5.25],
        current_yaw=0.25,
        route_waypoints_xy=route_waypoints,
        image_size=(128, 128),
    )
    panel_from_focused_grid = module.render_route_map_panel(
        terrain_origins=focused_origins,
        current_position_xy=[-5.5, -5.25],
        current_yaw=0.25,
        route_waypoints_xy=route_waypoints,
        image_size=(128, 128),
    )

    assert np.array_equal(panel_from_full_grid, panel_from_focused_grid)


def test_build_route_map_recording_panel_centers_square_map_without_resizing():
    module = load_module()
    route_map = np.arange(6 * 6 * 3, dtype=np.uint8).reshape(6, 6, 3)

    panel = module.build_route_map_recording_panel(route_map, output_size=(6, 20))

    assert panel.shape == (6, 20, 3)
    np.testing.assert_array_equal(panel[:, 7:13], route_map)
    assert np.unique(panel[:, :7].reshape(-1, 3), axis=0).shape[0] > 1


def test_resolve_inference_checkpoint_model_state_skips_critic_only_mismatches():
    module = load_module()
    checkpoint_state = {
        "std": np.zeros((29,), dtype=np.float32),
        "actor.gate.0.weight": np.zeros((4, 896), dtype=np.float32),
        "critic.gate.0.weight": np.zeros((4, 920), dtype=np.float32),
    }
    expected_state = {
        "std": np.zeros((29,), dtype=np.float32),
        "actor.gate.0.weight": np.zeros((4, 896), dtype=np.float32),
        "critic.gate.0.weight": np.zeros((4, 950), dtype=np.float32),
    }

    compatible_state, skipped_keys, critical_keys = module.resolve_inference_checkpoint_model_state(
        checkpoint_state_dict=checkpoint_state,
        expected_state_dict=expected_state,
    )

    assert set(compatible_state) == {"std", "actor.gate.0.weight"}
    assert skipped_keys == ["critic.gate.0.weight"]
    assert critical_keys == []


def test_resolve_inference_checkpoint_model_state_flags_actor_mismatches_as_critical():
    module = load_module()
    checkpoint_state = {
        "actor.gate.0.weight": np.zeros((4, 800), dtype=np.float32),
        "critic.gate.0.weight": np.zeros((4, 920), dtype=np.float32),
    }
    expected_state = {
        "actor.gate.0.weight": np.zeros((4, 896), dtype=np.float32),
        "critic.gate.0.weight": np.zeros((4, 950), dtype=np.float32),
    }

    _, skipped_keys, critical_keys = module.resolve_inference_checkpoint_model_state(
        checkpoint_state_dict=checkpoint_state,
        expected_state_dict=expected_state,
    )

    assert skipped_keys == ["actor.gate.0.weight", "critic.gate.0.weight"]
    assert critical_keys == ["actor.gate.0.weight"]


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

    def callback(event):
        return None

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


def test_select_center_terrain_origin_returns_nearest_actual_origin_to_map_center():
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

    np.testing.assert_allclose(center_origin, np.array([-4.0, -28.0, -0.82], dtype=np.float32))
    assert any(np.allclose(center_origin, origin) for origin in terrain_origins.reshape(-1, 3))


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


def test_resolve_video_capture_settings_replay_uses_duration_as_maximum():
    module = load_module()

    settings = module.resolve_video_capture_settings(
        video_length=3000,
        video_duration_s=120.0,
        step_dt=0.02,
        video_frame_stride=2,
        video_fps=None,
        replay_route_active=True,
    )

    assert settings.max_frames == 3000
    assert settings.video_fps == 25.0


def test_resolve_interactive_rendering_gpu_override_prefers_display_gpu_when_env_stays_on_cuda0():
    module = load_module()

    def fake_run(command, capture_output, text, check):
        assert command == [
            "nvidia-smi",
            "--query-gpu=index,display_active",
            "--format=csv,noheader",
        ]
        return SimpleNamespace(
            returncode=0,
            stdout="0, Disabled\n1, Disabled\n2, Enabled\n3, Disabled\n",
            stderr="",
        )

    resolved = module.resolve_interactive_rendering_gpu_override(
        "cuda:0",
        headless=False,
        display=":1",
        command_runner=fake_run,
    )

    assert resolved == 2


def test_resolve_interactive_rendering_gpu_override_keeps_same_gpu_when_display_already_matches_cuda0():
    module = load_module()

    def fake_run(command, capture_output, text, check):
        return SimpleNamespace(
            returncode=0,
            stdout="0, Enabled\n1, Disabled\n2, Disabled\n3, Disabled\n",
            stderr="",
        )

    resolved = module.resolve_interactive_rendering_gpu_override(
        "cuda:0",
        headless=False,
        display=":1",
        command_runner=fake_run,
    )

    assert resolved is None


def test_resolve_interactive_rendering_gpu_override_returns_none_when_display_query_is_ambiguous():
    module = load_module()

    def fake_run(command, capture_output, text, check):
        return SimpleNamespace(
            returncode=0,
            stdout="0, Enabled\n1, Disabled\n2, Enabled\n3, Disabled\n",
            stderr="",
        )

    resolved = module.resolve_interactive_rendering_gpu_override(
        "cuda:0",
        headless=False,
        display=":1",
        command_runner=fake_run,
    )

    assert resolved is None


def test_resolve_interactive_rendering_gpu_override_returns_none_for_nonzero_env_device():
    module = load_module()

    def fake_run(command, capture_output, text, check):
        return SimpleNamespace(
            returncode=0,
            stdout="0, Disabled\n1, Disabled\n2, Enabled\n3, Disabled\n",
            stderr="",
        )

    resolved = module.resolve_interactive_rendering_gpu_override(
        "cuda:2",
        headless=False,
        display=":1",
        command_runner=fake_run,
    )

    assert resolved is None


def test_resolve_video_capture_settings_replay_without_duration_is_unbounded():
    module = load_module()

    settings = module.resolve_video_capture_settings(
        video_length=3000,
        video_duration_s=None,
        step_dt=0.02,
        video_frame_stride=2,
        video_fps=None,
        replay_route_active=True,
    )

    assert settings.max_frames is None
    assert settings.video_fps == 25.0
    assert settings.frame_stride == 2


def test_build_play_video_output_path_appends_timestamp_suffix():
    module = load_module()

    output_path = module.build_play_video_output_path(
        log_dir="/tmp/logs/run",
        resume_path="/tmp/logs/run/model_123.pt",
        video_start_step=0,
        timestamp=datetime(2026, 3, 31, 12, 34, 56),
    )

    assert output_path == "/tmp/logs/run/videos/play/model_123-step-0-20260331_123456.mp4"


def test_build_play_video_output_path_includes_route_stem_for_replay():
    module = load_module()

    output_path = module.build_play_video_output_path(
        log_dir="/tmp/logs/run",
        resume_path="/tmp/logs/run/model_123.pt",
        video_start_step=0,
        replay_route_path="outputs/parkour/routes/my-route.json",
        timestamp=datetime(2026, 3, 31, 12, 34, 56),
    )

    assert output_path == "/tmp/logs/run/videos/play/model_123-step-0-my-route-20260331_123456.mp4"


def test_resolve_recording_camera_resolution_targets_2k_output():
    module = load_module()

    assert module.resolve_recording_camera_resolution("quad") == (1280, 720)
    assert module.resolve_recording_camera_resolution("single") == (2560, 1440)


def test_compute_contact_overlay_state_filters_small_forces_and_scales_active_forces():
    module = load_module()
    positions = np.array([[0.0, 0.0, 0.1], [1.0, 0.0, 0.1]], dtype=np.float32)
    forces = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 10.0]], dtype=np.float32)

    state = module.compute_contact_overlay_state(
        positions,
        forces,
        force_threshold=1.0,
        force_scale=0.1,
    )

    assert state.active_mask.tolist() == [False, True]
    assert state.lengths.tolist() == [0.0, 1.0]
    assert state.directions[1].tolist() == [0.0, 0.0, 1.0]


def test_detect_new_contact_events_flags_only_new_touchdowns():
    module = load_module()

    events = module.detect_new_contact_events(
        previous_contact_mask=np.array([False, True, False]),
        current_contact_mask=np.array([True, True, False]),
    )

    assert events.tolist() == [True, False, False]


def test_get_play_visualization_config_reads_namespace_elevation_window():
    module = load_module()
    config = module._get_play_visualization_config(
        SimpleNamespace(
            visualization=SimpleNamespace(
                depth_window=False,
                depth_coverage=False,
                elevation_map_window=True,
                normals_panel=False,
                route_overlay=False,
                foot_contact_overlay=False,
                ghost_reference=False,
                obstacle_edges=False,
            )
        )
    )

    assert config.elevation_map_window is True


def test_get_play_visualization_config_accepts_dataclass_elevation_window():
    module = load_module()
    config = module._get_play_visualization_config(
        SimpleNamespace(
            visualization=module.PlayVisualizationConfig(elevation_map_window=True)
        )
    )

    assert config.elevation_map_window is True


def test_integrate_elevation_map_observations_keeps_highest_height_per_cell():
    module = load_module()
    state = module.create_rolling_elevation_map(module.RollingElevationMapConfig(resolution_m=1.0, size_m=4.0))

    module.integrate_elevation_map_observations(
        state,
        np.array([[0.1, 0.1, 0.2], [0.4, 0.4, 0.8]], dtype=np.float32),
        robot_position_xy=np.array([0.0, 0.0], dtype=np.float32),
    )

    assert np.isclose(state.observed_height_by_cell[(0, 0)], 0.8)


def test_render_elevation_map_panel_preserves_seen_cells_and_unknown_space():
    module = load_module()
    state = module.create_rolling_elevation_map(
        module.RollingElevationMapConfig(resolution_m=1.0, size_m=4.0, image_size=(40, 40))
    )

    module.integrate_elevation_map_observations(
        state,
        np.array([[0.1, 0.1, 0.4], [1.1, 0.1, 0.8]], dtype=np.float32),
        robot_position_xy=np.array([0.0, 0.0], dtype=np.float32),
    )
    panel = module.render_elevation_map_panel(
        state,
        robot_position_xy=np.array([0.0, 0.0], dtype=np.float32),
        robot_yaw=0.0,
    )

    assert panel.shape == (40, 40, 3)
    assert np.any(np.all(panel == np.array(state.config.unknown_color, dtype=np.uint8), axis=-1))
    assert np.unique(panel.reshape(-1, 3), axis=0).shape[0] > 3


def test_render_elevation_map_panel_keeps_world_aligned_memory_across_yaw_only_rotation():
    module = load_module()
    state = module.create_rolling_elevation_map(
        module.RollingElevationMapConfig(resolution_m=1.0, size_m=4.0, image_size=(40, 40))
    )

    module.integrate_elevation_map_observations(
        state,
        np.array([[-1.6, -1.6, 0.5]], dtype=np.float32),
        robot_position_xy=np.array([0.0, 0.0], dtype=np.float32),
    )
    panel_yaw0 = module.render_elevation_map_panel(
        state,
        robot_position_xy=np.array([0.0, 0.0], dtype=np.float32),
        robot_yaw=0.0,
    )
    panel_yaw90 = module.render_elevation_map_panel(
        state,
        robot_position_xy=np.array([0.0, 0.0], dtype=np.float32),
        robot_yaw=float(np.pi * 0.5),
    )

    cell_bounds = module._cell_bounds_from_index((-2, -2), resolution_m=1.0)
    render_bounds = (-2.0, 2.0, -2.0, 2.0)
    top_left = module._world_xy_to_panel_pixel(
        (cell_bounds[0], cell_bounds[3]),
        bounds=render_bounds,
        image_size=(40, 40),
        margin=0,
    )
    bottom_right = module._world_xy_to_panel_pixel(
        (cell_bounds[1], cell_bounds[2]),
        bounds=render_bounds,
        image_size=(40, 40),
        margin=0,
    )
    left = min(top_left[0], bottom_right[0])
    right = max(top_left[0], bottom_right[0])
    top = min(top_left[1], bottom_right[1])
    bottom = max(top_left[1], bottom_right[1])

    assert np.any(np.any(panel_yaw0[top : bottom + 1, left : right + 1, :] != 0, axis=-1))
    np.testing.assert_array_equal(
        panel_yaw0[top : bottom + 1, left : right + 1, :],
        panel_yaw90[top : bottom + 1, left : right + 1, :],
    )


def test_integrate_elevation_map_observations_prunes_far_cells_outside_retention_window():
    module = load_module()
    state = module.create_rolling_elevation_map(
        module.RollingElevationMapConfig(resolution_m=1.0, size_m=4.0, retention_multiplier=1.0)
    )

    module.integrate_elevation_map_observations(
        state,
        np.array([[0.1, 0.1, 0.4]], dtype=np.float32),
        robot_position_xy=np.array([0.0, 0.0], dtype=np.float32),
    )
    module.integrate_elevation_map_observations(
        state,
        np.empty((0, 3), dtype=np.float32),
        robot_position_xy=np.array([10.0, 10.0], dtype=np.float32),
    )

    assert state.observed_height_by_cell == {}


def test_get_play_visualization_config_reads_namespace_elevation_viewport():
    module = load_module()
    config = module._get_play_visualization_config(
        SimpleNamespace(
            visualization=SimpleNamespace(
                depth_window=False,
                depth_coverage=False,
                elevation_map_window=False,
                elevation_viewport=True,
                normals_panel=False,
                route_overlay=False,
                foot_contact_overlay=False,
                ghost_reference=False,
                obstacle_edges=False,
            )
        )
    )

    assert config.elevation_viewport is True


def test_build_elevation_surface_mesh_data_returns_displaced_geometry():
    module = load_module()
    state = module.create_rolling_elevation_map(module.RollingElevationMapConfig(resolution_m=1.0, size_m=4.0))
    module.integrate_elevation_map_observations(
        state,
        np.array([[0.1, 0.1, 0.2], [1.1, 0.1, 0.8]], dtype=np.float32),
        robot_position_xy=np.array([0.0, 0.0], dtype=np.float32),
    )

    points, face_counts, face_indices, display_colors = module.build_elevation_surface_mesh_data(
        state,
        robot_position_w=np.array([0.0, 0.0, 0.5], dtype=np.float32),
    )

    assert points.shape[1] == 3
    assert face_counts.size > 0
    assert face_indices.size > 0
    assert display_colors.shape[0] == points.shape[0]
    assert np.unique(points[:, 2]).size > 2


def test_build_elevation_surface_mesh_data_omits_unknown_cells():
    module = load_module()
    state = module.create_rolling_elevation_map(module.RollingElevationMapConfig(resolution_m=1.0, size_m=4.0))

    points, face_counts, face_indices, display_colors = module.build_elevation_surface_mesh_data(
        state,
        robot_position_w=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert points.size == 0
    assert face_counts.size == 0
    assert face_indices.size == 0
    assert display_colors.size == 0


def test_build_elevation_surface_mesh_data_bridges_neighbor_height_steps():
    module = load_module()
    state = module.create_rolling_elevation_map(module.RollingElevationMapConfig(resolution_m=1.0, size_m=4.0))
    module.integrate_elevation_map_observations(
        state,
        np.array([[0.1, 0.1, 0.2], [1.1, 0.1, 0.8]], dtype=np.float32),
        robot_position_xy=np.array([0.0, 0.0], dtype=np.float32),
    )

    points, _, _, _ = module.build_elevation_surface_mesh_data(
        state,
        robot_position_w=np.array([0.0, 0.0, 0.5], dtype=np.float32),
    )

    z_values = np.unique(np.round(points[:, 2], decimals=3))
    assert np.any(np.isclose(z_values, 0.6, atol=1e-3))
    assert np.any(np.isclose(z_values, 1.2, atol=1e-3))


def test_compute_preview_relative_body_positions_centers_on_root():
    module = load_module()
    relative = module.compute_preview_relative_body_positions(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=np.float32),
        preview_origin=(0.0, 0.0, 0.9),
    )

    np.testing.assert_allclose(relative[0], np.array([0.0, 0.0, 0.9], dtype=np.float32))
    np.testing.assert_allclose(relative[1], np.array([1.0, 2.0, 3.9], dtype=np.float32))
