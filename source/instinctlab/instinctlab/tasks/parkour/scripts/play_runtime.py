import argparse
from datetime import datetime
import math
from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
from typing import Iterable, Sequence

import cv2
import numpy as np
from packaging.version import Version


NO_AUTO_RESET_EPISODE_LENGTH_S = 1e10
VIDEO_LAYOUT_CHOICES = ("quad", "single")
ISAACSIM_REQUIRED_NUMPY_VERSION = "1.26.0"
DEPTH_WINDOW_NAME = "parkour_depth"
NORMALS_WINDOW_NAME = "parkour_normals"
ELEVATION_MAP_WINDOW_NAME = "parkour_elevation_map"


@dataclass(frozen=True)
class TrackingCameraSpec:
    name: str
    eye_offset: tuple[float, float, float]
    target_offset: tuple[float, float, float]


@dataclass(frozen=True)
class VideoCaptureSettings:
    max_frames: int | None
    video_fps: float
    frame_stride: int


@dataclass(frozen=True)
class PlayVisualizationConfig:
    depth_window: bool = False
    depth_coverage: bool = False
    elevation_map_window: bool = False
    normals_panel: bool = False
    route_overlay: bool = False
    foot_contact_overlay: bool = False
    ghost_reference: bool = False
    obstacle_edges: bool = False


@dataclass(frozen=True)
class RollingElevationMapConfig:
    resolution_m: float = 0.1
    size_m: float = 6.0
    retention_multiplier: float = 2.0
    image_size: tuple[int, int] = (240, 240)
    unknown_color: tuple[int, int, int] = (0, 0, 0)
    robot_color: tuple[int, int, int] = (14, 165, 233)
    border_color: tuple[int, int, int] = (214, 223, 233)


@dataclass
class RollingElevationMapState:
    config: RollingElevationMapConfig = field(default_factory=RollingElevationMapConfig)
    step_count: int = 0
    observed_height_by_cell: dict[tuple[int, int], float] = field(default_factory=dict)
    last_seen_step_by_cell: dict[tuple[int, int], int] = field(default_factory=dict)


@dataclass(frozen=True)
class ContactOverlayState:
    positions: np.ndarray
    directions: np.ndarray
    lengths: np.ndarray
    active_mask: np.ndarray


@dataclass
class KeyboardEventSubscription:
    input_interface: object
    keyboard: object
    handle: object
    callback: object

    def close(self) -> None:
        if self.handle is None:
            return
        self.input_interface.unsubscribe_to_keyboard_events(self.keyboard, self.handle)
        self.handle = None


def add_play_runtime_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--center_spawn", action="store_true", default=False, help="Spawn at the map center.")
    parser.add_argument(
        "--disable_auto_reset",
        action="store_true",
        default=False,
        help="Disable termination-driven auto reset during play.",
    )
    parser.add_argument(
        "--show_depth_window",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show the latest depth frame in a separate OpenCV window.",
    )
    parser.add_argument(
        "--show_depth_coverage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Visualize depth ray hits in the RGB scene.",
    )
    parser.add_argument(
        "--show_elevation_map_window",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show a robot-centered rolling local elevation-map preview window built from observed ray hits.",
    )
    parser.add_argument(
        "--normals_panel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show a false-color normals panel derived from the ray-caster camera.",
    )
    parser.add_argument(
        "--route_overlay",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show the replay route spline and a short future trajectory prediction overlay.",
    )
    parser.add_argument(
        "--foot_contact_overlay",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show foot contact force arrows and touchdown markers.",
    )
    parser.add_argument(
        "--ghost_reference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show the motion-reference ghost overlay.",
    )
    parser.add_argument(
        "--obstacle_edges",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Highlight obstacle and terrain edges with virtual cylinders.",
    )
    parser.add_argument(
        "--video_layout",
        type=str,
        default="quad",
        choices=VIDEO_LAYOUT_CHOICES,
        help="Layout used when recording play videos.",
    )
    parser.add_argument(
        "--show_command_arrow",
        action="store_true",
        default=False,
        help="Show the large command direction arrow above the robot.",
    )
    parser.add_argument(
        "--video_duration_s",
        type=float,
        default=None,
        help="Recorded video duration in seconds. Overrides --video_length when set.",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=None,
        help="Encoded video FPS. Defaults to simulation FPS divided by frame stride.",
    )
    parser.add_argument(
        "--video_frame_stride",
        type=int,
        default=1,
        help="Capture one video frame every N simulation steps to reduce recording overhead.",
    )
    parser.add_argument(
        "--save_route",
        type=str,
        default=None,
        help="Optional JSON output path for saving the executed route as world-frame waypoints.",
    )
    parser.add_argument(
        "--replay_route",
        type=str,
        default=None,
        help="Optional JSON route artifact to replay in headless or interactive runs.",
    )
    parser.add_argument(
        "--route_interval_m",
        type=float,
        default=0.15,
        help="Minimum XY spacing, in meters, between saved route waypoints.",
    )
    parser.add_argument(
        "--route_lookahead_m",
        type=float,
        default=0.6,
        help="Lookahead distance, in meters, used by route replay.",
    )
    parser.add_argument(
        "--route_goal_tolerance_m",
        type=float,
        default=0.2,
        help="Distance threshold, in meters, for advancing or finishing route waypoints.",
    )
    parser.add_argument(
        "--route_cruise_speed",
        type=float,
        default=None,
        help="Forward speed used during route replay. Defaults to the command limit maximum.",
    )
    return parser


def validate_isaacsim_python_environment(numpy_version: str | None = None) -> None:
    current_numpy_version = numpy_version if numpy_version is not None else np.__version__
    if Version(current_numpy_version).major >= 2:
        raise RuntimeError(
            "Incompatible Python environment for Isaac Sim: detected numpy"
            f" {current_numpy_version}, but Isaac Sim 5.1 / isaacsim-kernel requires numpy=={ISAACSIM_REQUIRED_NUMPY_VERSION}."
            " Downgrade numpy in this environment before launching play.py."
        )


def resolve_interactive_rendering_gpu_override(
    device: str,
    *,
    headless: bool,
    display: str | None = None,
    command_runner=subprocess.run,
) -> int | None:
    if headless or not device.startswith("cuda"):
        return None

    active_display = display if display is not None else os.environ.get("DISPLAY")
    if not active_display:
        return None

    display_gpu_index = _discover_single_display_active_gpu(command_runner=command_runner)
    if display_gpu_index is None:
        return None

    device_gpu_index = _extract_cuda_device_index(device)
    if device_gpu_index is None or device_gpu_index != 0 or device_gpu_index == display_gpu_index:
        return None
    return display_gpu_index


def create_keyboard_event_subscription(callback, app_window=None, input_interface=None) -> KeyboardEventSubscription:
    if app_window is None:
        import omni.appwindow

        app_window = omni.appwindow.get_default_app_window()
    if input_interface is None:
        import carb.input

        input_interface = carb.input.acquire_input_interface()
    keyboard = app_window.get_keyboard()
    handle = input_interface.subscribe_to_keyboard_events(keyboard, callback)
    return KeyboardEventSubscription(
        input_interface=input_interface,
        keyboard=keyboard,
        handle=handle,
        callback=callback,
    )


def ensure_sensor_initialized(sensor, sensor_name: str | None = None) -> None:
    if sensor.is_initialized:
        return
    sensor._initialize_callback(None)
    if not sensor.is_initialized:
        raise RuntimeError(f"Failed to initialize sensor: {sensor_name or sensor}.")


def select_map_center_xy(terrain_origins) -> np.ndarray:
    origins = np.asarray(terrain_origins, dtype=np.float32)
    if origins.ndim != 3 or origins.shape[-1] != 3:
        raise ValueError(f"Expected terrain origins with shape (rows, cols, 3), got {origins.shape}.")
    min_xy = origins[..., :2].reshape(-1, 2).min(axis=0)
    max_xy = origins[..., :2].reshape(-1, 2).max(axis=0)
    return ((min_xy + max_xy) * 0.5).astype(np.float32, copy=False)


def select_center_terrain_origin(terrain_origins) -> np.ndarray:
    origins = np.asarray(terrain_origins, dtype=np.float32)
    if origins.ndim != 3 or origins.shape[-1] != 3:
        raise ValueError(f"Expected terrain origins with shape (rows, cols, 3), got {origins.shape}.")
    flattened = origins.reshape(-1, 3)
    center_xy = select_map_center_xy(origins)
    distances = np.sum((flattened[:, :2] - center_xy[None, :]) ** 2, axis=1)
    nearest_origin = flattened[int(np.argmin(distances))]
    return nearest_origin.astype(np.float32, copy=True)


def apply_play_runtime_overrides(env_cfg, options) -> None:
    visualization = _get_play_visualization_config(options)

    if _should_disable_auto_reset(options):
        env_cfg.episode_length_s = NO_AUTO_RESET_EPISODE_LENGTH_S
        for term_name in vars(env_cfg.terminations):
            setattr(env_cfg.terminations, term_name, None)

    if visualization.depth_window:
        _set_depth_debug_vis(env_cfg.observations.policy, enabled=False)
        if hasattr(env_cfg.observations, "critic"):
            _set_depth_debug_vis(env_cfg.observations.critic, enabled=False)

    if visualization.depth_coverage:
        env_cfg.scene.camera.debug_vis = True
    if visualization.ghost_reference:
        motion_reference = getattr(env_cfg.scene, "motion_reference", None)
        if motion_reference is not None:
            _configure_ghost_reference_preview(motion_reference)
    if visualization.obstacle_edges:
        terrain = getattr(env_cfg.scene, "terrain", None)
        if terrain is not None:
            terrain.debug_vis = True

    if _should_enforce_center_spawn(options):
        _lock_center_spawn_pose(env_cfg)
        _disable_terrain_curriculum(env_cfg)

    _configure_command_arrow_visuals(env_cfg, enabled=getattr(options, "show_command_arrow", False))


def resolve_play_seed(*, cli_seed: int | None, route_seed: int | None, agent_seed: int | None) -> int | None:
    if cli_seed is not None:
        return int(cli_seed)
    if route_seed is not None:
        return int(route_seed)
    if agent_seed is not None:
        return int(agent_seed)
    return None


def _apply_terrain_generator_seed(env_cfg, seed: int | None) -> None:
    if seed is None:
        return
    terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    terrain_generator = getattr(terrain, "terrain_generator", None)
    if terrain_generator is not None:
        terrain_generator.seed = int(seed)


def _apply_route_artifact_terrain_layout(env_cfg, route_artifact) -> None:
    if route_artifact is None:
        return
    terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    terrain_generator = getattr(terrain, "terrain_generator", None)
    if terrain_generator is None:
        return
    tile_wall_edges = getattr(route_artifact, "tile_wall_edges", None)
    if tile_wall_edges is not None:
        terrain_generator.precomputed_tile_wall_edges = tile_wall_edges


def prepare_play_env_cfg(env_cfg, options, *, seed: int | None = None, route_artifact=None) -> None:
    if seed is not None:
        env_cfg.seed = int(seed)
    _apply_terrain_generator_seed(env_cfg, seed)
    _apply_route_artifact_terrain_layout(env_cfg, route_artifact)
    _apply_visualization_sensor_requirements(env_cfg, options)
    if _should_use_single_env_play(options):
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = NO_AUTO_RESET_EPISODE_LENGTH_S
    apply_play_runtime_overrides(env_cfg, options)


def resolve_play_visualization_config(env_cfg, options) -> PlayVisualizationConfig:
    defaults = getattr(env_cfg, "play_visualization", None)
    return PlayVisualizationConfig(
        depth_window=_resolve_visualization_flag(
            default=getattr(defaults, "depth_window", False),
            override=getattr(options, "show_depth_window", None),
        ),
        depth_coverage=_resolve_visualization_flag(
            default=getattr(defaults, "depth_coverage", False),
            override=getattr(options, "show_depth_coverage", None),
        ),
        elevation_map_window=_resolve_visualization_flag(
            default=getattr(defaults, "elevation_map_window", False),
            override=getattr(options, "show_elevation_map_window", None),
        ),
        normals_panel=_resolve_visualization_flag(
            default=getattr(defaults, "normals_panel", False),
            override=getattr(options, "normals_panel", None),
        ),
        route_overlay=_resolve_visualization_flag(
            default=getattr(defaults, "route_overlay", False),
            override=getattr(options, "route_overlay", None),
        ),
        foot_contact_overlay=_resolve_visualization_flag(
            default=getattr(defaults, "foot_contact_overlay", False),
            override=getattr(options, "foot_contact_overlay", None),
        ),
        ghost_reference=_resolve_visualization_flag(
            default=getattr(defaults, "ghost_reference", False),
            override=getattr(options, "ghost_reference", None),
        ),
        obstacle_edges=_resolve_visualization_flag(
            default=getattr(defaults, "obstacle_edges", False),
            override=getattr(options, "obstacle_edges", None),
        ),
    )


def build_default_tracking_camera_specs() -> list[TrackingCameraSpec]:
    return [
        TrackingCameraSpec("hero", eye_offset=(-3.4, -1.1, 1.45), target_offset=(0.9, 0.0, 0.32)),
        TrackingCameraSpec("side", eye_offset=(0.15, -3.3, 1.35), target_offset=(0.8, 0.0, 0.4)),
        TrackingCameraSpec("overview", eye_offset=(-1.0, 0.0, 5.5), target_offset=(1.0, 0.0, 0.7)),
    ]


def resolve_recording_camera_resolution(video_layout: str) -> tuple[int, int]:
    if video_layout == "single":
        return (2560, 1440)
    return (1280, 720)


def compute_tracking_camera_views(
    root_position,
    root_yaw: float,
    specs: Iterable[TrackingCameraSpec],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    root = np.asarray(root_position, dtype=np.float32).reshape(3)
    cos_yaw = float(np.cos(root_yaw))
    sin_yaw = float(np.sin(root_yaw))
    rotation = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    views: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spec in specs:
        eye = root + rotation @ np.asarray(spec.eye_offset, dtype=np.float32)
        target = root + rotation @ np.asarray(spec.target_offset, dtype=np.float32)
        views[spec.name] = (eye.astype(np.float32, copy=False), target.astype(np.float32, copy=False))
    return views


def normalize_depth_frame_for_display(
    depth_frame,
    depth_range: tuple[float, float],
) -> np.ndarray:
    depth = np.asarray(depth_frame, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=depth_range[1], posinf=depth_range[1], neginf=depth_range[0])
    low, high = depth_range
    clipped = np.clip(depth, low, high)
    if high <= low:
        normalized = np.zeros_like(clipped, dtype=np.uint8)
    else:
        normalized = np.round(((clipped - low) / (high - low)) * 255.0).astype(np.uint8)
    return np.repeat(normalized[..., None], 3, axis=-1)


def normalize_normals_frame_for_display(normals_frame) -> np.ndarray:
    normals = np.asarray(normals_frame, dtype=np.float32)
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = np.clip(np.round((normals + 1.0) * 127.5), 0.0, 255.0)
    return normalized.astype(np.uint8)


def build_live_preview_panels(
    *,
    depth_panel: np.ndarray,
    elevation_map_panel: np.ndarray | None = None,
    normals_panel: np.ndarray | None = None,
    show_depth_window: bool,
    show_elevation_map_window: bool = False,
    show_normals_window: bool,
    scale: float = 8.0,
) -> list[tuple[str, np.ndarray]]:
    panels: list[tuple[str, np.ndarray]] = []
    if show_depth_window:
        panels.append((DEPTH_WINDOW_NAME, _resize_live_preview_panel(depth_panel, scale=scale)))
    if show_elevation_map_window and elevation_map_panel is not None:
        panels.append((ELEVATION_MAP_WINDOW_NAME, _resize_live_preview_panel(elevation_map_panel, scale=scale)))
    if show_normals_window and normals_panel is not None:
        panels.append((NORMALS_WINDOW_NAME, _resize_live_preview_panel(normals_panel, scale=scale)))
    return panels


def create_rolling_elevation_map(
    config: RollingElevationMapConfig | None = None,
) -> RollingElevationMapState:
    return RollingElevationMapState(config=RollingElevationMapConfig() if config is None else config)


def integrate_elevation_map_observations(
    state: RollingElevationMapState,
    ray_hits_w,
    *,
    robot_position_xy,
) -> None:
    if not isinstance(state, RollingElevationMapState):
        raise TypeError(f"Expected RollingElevationMapState, got {type(state)!r}.")
    config = state.config
    state.step_count += 1

    observed_cells = _collapse_ray_hits_to_height_cells(ray_hits_w, resolution_m=config.resolution_m)
    for cell_key, cell_height in observed_cells.items():
        state.observed_height_by_cell[cell_key] = float(cell_height)
        state.last_seen_step_by_cell[cell_key] = state.step_count

    _prune_rolling_elevation_map(state, robot_position_xy=robot_position_xy)


def render_elevation_map_panel(
    state: RollingElevationMapState,
    *,
    robot_position_xy,
    robot_yaw: float,
) -> np.ndarray:
    if not isinstance(state, RollingElevationMapState):
        raise TypeError(f"Expected RollingElevationMapState, got {type(state)!r}.")

    config = state.config
    image_height, image_width = config.image_size
    panel = np.zeros((int(image_height), int(image_width), 3), dtype=np.uint8)
    panel[...] = np.asarray(config.unknown_color, dtype=np.uint8)

    robot_xy = np.asarray(robot_position_xy, dtype=np.float32).reshape(2)
    half_extent = float(config.size_m) * 0.5
    bounds = (
        float(robot_xy[0] - half_extent),
        float(robot_xy[0] + half_extent),
        float(robot_xy[1] - half_extent),
        float(robot_xy[1] + half_extent),
    )

    visible_cells: list[tuple[tuple[int, int], float]] = []
    for cell_key, cell_height in state.observed_height_by_cell.items():
        cell_bounds = _cell_bounds_from_index(cell_key, resolution_m=config.resolution_m)
        if _cell_intersects_bounds(cell_bounds, bounds):
            visible_cells.append((cell_key, float(cell_height)))

    if visible_cells:
        heights = np.asarray([height for _, height in visible_cells], dtype=np.float32)
        normalized_heights = _normalize_elevation_heights_for_display(heights)
        colors = cv2.applyColorMap(normalized_heights[:, None], cv2.COLORMAP_TURBO)[:, 0, :]
        for (cell_key, _), color in zip(visible_cells, colors, strict=True):
            x_min, x_max, y_min, y_max = _cell_bounds_from_index(cell_key, resolution_m=config.resolution_m)
            top_left = _world_xy_to_panel_pixel(
                (x_min, y_max),
                bounds=bounds,
                image_size=(image_height, image_width),
                margin=0,
            )
            bottom_right = _world_xy_to_panel_pixel(
                (x_max, y_min),
                bounds=bounds,
                image_size=(image_height, image_width),
                margin=0,
            )
            left = min(top_left[0], bottom_right[0])
            right = max(top_left[0], bottom_right[0])
            top = min(top_left[1], bottom_right[1])
            bottom = max(top_left[1], bottom_right[1])
            cv2.rectangle(
                panel,
                (left, top),
                (right, bottom),
                color=tuple(int(channel) for channel in color.tolist()),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    robot_px = _world_xy_to_panel_pixel(robot_xy, bounds=bounds, image_size=(image_height, image_width), margin=0)
    cv2.circle(panel, robot_px, 5, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(panel, robot_px, 3, color=config.robot_color, thickness=-1, lineType=cv2.LINE_AA)
    arrow_length_px = max(10, int(min(image_width, image_height) * 0.08))
    heading_end = (
        int(round(robot_px[0] + math.cos(float(robot_yaw)) * arrow_length_px)),
        int(round(robot_px[1] - math.sin(float(robot_yaw)) * arrow_length_px)),
    )
    cv2.arrowedLine(
        panel,
        robot_px,
        heading_end,
        color=config.robot_color,
        thickness=2,
        tipLength=0.25,
        line_type=cv2.LINE_AA,
    )
    cv2.rectangle(
        panel,
        (0, 0),
        (image_width - 1, image_height - 1),
        color=config.border_color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return panel


def render_route_map_panel(
    *,
    terrain_origins,
    current_position_xy,
    current_yaw: float,
    route_waypoints_xy: list[list[float]] | list[tuple[float, float]] | None = None,
    center_origin=None,
    tile_wall_edges=None,
    image_size: tuple[int, int] = (720, 720),
) -> np.ndarray:
    origins = np.asarray(terrain_origins, dtype=np.float32)
    if origins.ndim != 3 or origins.shape[-1] != 3:
        raise ValueError(f"Expected terrain origins with shape (rows, cols, 3), got {origins.shape}.")

    height = int(image_size[0])
    width = int(image_size[1])

    tile_size_x = _estimate_grid_spacing(origins[..., 0], default=8.0)
    tile_size_y = _estimate_grid_spacing(origins[..., 1], default=8.0)
    origins, tile_wall_edges = _focus_route_map_tiles(
        origins,
        tile_wall_edges=tile_wall_edges,
        current_position_xy=current_position_xy,
        route_waypoints_xy=route_waypoints_xy,
        tile_size_x=tile_size_x,
        tile_size_y=tile_size_y,
    )
    canvas = _build_route_map_canvas(image_size=(height, width))

    min_x = float(origins[..., 0].min() - tile_size_x * 0.5)
    max_x = float(origins[..., 0].max() + tile_size_x * 0.5)
    min_y = float(origins[..., 1].min() - tile_size_y * 0.5)
    max_y = float(origins[..., 1].max() + tile_size_y * 0.5)

    focus_points: list[np.ndarray] = [np.asarray(current_position_xy, dtype=np.float32).reshape(2)]
    if route_waypoints_xy:
        focus_points.append(np.asarray(route_waypoints_xy, dtype=np.float32).reshape(-1, 2))
    stacked_points = np.concatenate([points.reshape(-1, 2) for points in focus_points], axis=0)
    min_x = min(min_x, float(stacked_points[:, 0].min()))
    max_x = max(max_x, float(stacked_points[:, 0].max()))
    min_y = min(min_y, float(stacked_points[:, 1].min()))
    max_y = max(max_y, float(stacked_points[:, 1].max()))

    bounds = _expand_xy_bounds((min_x, max_x, min_y, max_y), min_padding=min(tile_size_x, tile_size_y) * 0.35)
    margin = max(8, int(min(width, height) * 0.06))
    tile_fill_color = (236, 240, 244)
    grid_color = (205, 213, 223)
    wall_color = (15, 23, 42)
    route_color = (245, 158, 11)
    route_glow_color = (255, 225, 168)
    center_color = (225, 29, 72)
    robot_color = (14, 165, 233)

    for origin in origins.reshape(-1, 3):
        top_left = _world_xy_to_panel_pixel(
            (origin[0] - tile_size_x * 0.5, origin[1] + tile_size_y * 0.5),
            bounds=bounds,
            image_size=(height, width),
            margin=margin,
        )
        bottom_right = _world_xy_to_panel_pixel(
            (origin[0] + tile_size_x * 0.5, origin[1] - tile_size_y * 0.5),
            bounds=bounds,
            image_size=(height, width),
            margin=margin,
        )
        cv2.rectangle(canvas, top_left, bottom_right, color=tile_fill_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.rectangle(canvas, top_left, bottom_right, color=grid_color, thickness=1, lineType=cv2.LINE_AA)

    if tile_wall_edges is not None:
        for row in range(origins.shape[0]):
            for col in range(origins.shape[1]):
                for wall in _translate_tile_wall_edges(tile_wall_edges[row][col], origins[row, col, :2]):
                    x0, y0, x1, y1 = _wall_edge_to_segment(wall)
                    start = _world_xy_to_panel_pixel((x0, y0), bounds=bounds, image_size=(height, width), margin=margin)
                    end = _world_xy_to_panel_pixel((x1, y1), bounds=bounds, image_size=(height, width), margin=margin)
                    cv2.line(canvas, start, end, color=wall_color, thickness=2, lineType=cv2.LINE_AA)

    if route_waypoints_xy:
        route_waypoints = np.asarray(route_waypoints_xy, dtype=np.float32).reshape(-1, 2)
        route_pixels = np.asarray(
            [
                _world_xy_to_panel_pixel(point, bounds=bounds, image_size=(height, width), margin=margin)
                for point in route_waypoints
            ],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [route_pixels], False, color=route_glow_color, thickness=6, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [route_pixels], False, color=route_color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, tuple(route_pixels[0]), 6, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, tuple(route_pixels[-1]), 6, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, tuple(route_pixels[0]), 4, color=(34, 197, 94), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, tuple(route_pixels[-1]), 4, color=(239, 68, 68), thickness=-1, lineType=cv2.LINE_AA)

    if center_origin is not None and route_waypoints_xy is None:
        center_xy = np.asarray(center_origin, dtype=np.float32).reshape(-1)[:2]
        center_px = _world_xy_to_panel_pixel(center_xy, bounds=bounds, image_size=(height, width), margin=margin)
        cv2.drawMarker(
            canvas,
            center_px,
            color=center_color,
            markerType=cv2.MARKER_STAR,
            markerSize=12,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

    robot_xy = np.asarray(current_position_xy, dtype=np.float32).reshape(2)
    robot_px = _world_xy_to_panel_pixel(robot_xy, bounds=bounds, image_size=(height, width), margin=margin)
    cv2.circle(canvas, robot_px, 8, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, robot_px, 5, color=robot_color, thickness=-1, lineType=cv2.LINE_AA)

    arrow_length_px = max(10, int(min(width, height) * 0.06))
    heading_end = (
        int(round(robot_px[0] + math.cos(float(current_yaw)) * arrow_length_px)),
        int(round(robot_px[1] - math.sin(float(current_yaw)) * arrow_length_px)),
    )
    cv2.arrowedLine(
        canvas,
        robot_px,
        heading_end,
        color=robot_color,
        thickness=2,
        tipLength=0.25,
        line_type=cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (4, 4), (width - 5, height - 5), color=(214, 223, 233), thickness=1, lineType=cv2.LINE_AA)

    return canvas


def build_route_map_recording_panel(route_map_panel: np.ndarray, *, output_size: tuple[int, int]) -> np.ndarray:
    route_map = _ensure_rgb_uint8(route_map_panel)
    output_height = int(output_size[0])
    output_width = int(output_size[1])
    map_height, map_width = route_map.shape[:2]
    if map_height > output_height or map_width > output_width:
        raise ValueError(
            f"Route map panel with shape {route_map.shape[:2]} does not fit within output size {output_size}."
        )

    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    left_color = np.array([20, 28, 43], dtype=np.float32)
    right_color = np.array([33, 47, 72], dtype=np.float32)
    for column in range(output_width):
        alpha = 0.0 if output_width <= 1 else float(column) / float(output_width - 1)
        canvas[:, column] = np.round((1.0 - alpha) * left_color + alpha * right_color).astype(np.uint8)

    inset_x = (output_width - map_width) // 2
    inset_y = (output_height - map_height) // 2
    canvas[inset_y : inset_y + map_height, inset_x : inset_x + map_width] = route_map

    if inset_x > 0:
        cv2.line(
            canvas,
            (inset_x - 12, 14),
            (inset_x - 12, output_height - 15),
            color=(87, 124, 187),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (inset_x + map_width + 11, 14),
            (inset_x + map_width + 11, output_height - 15),
            color=(87, 124, 187),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    return canvas


def resolve_inference_checkpoint_model_state(
    *,
    checkpoint_state_dict,
    expected_state_dict,
) -> tuple[dict, list[str], list[str]]:
    compatible_state_dict: dict = {}
    skipped_keys: list[str] = []
    critical_keys: list[str] = []

    for key, expected_value in expected_state_dict.items():
        if key not in checkpoint_state_dict:
            if _is_inference_policy_state_key(key):
                critical_keys.append(key)
            continue

        checkpoint_value = checkpoint_state_dict[key]
        expected_shape = tuple(getattr(expected_value, "shape", ()))
        checkpoint_shape = tuple(getattr(checkpoint_value, "shape", ()))
        if expected_shape != checkpoint_shape:
            skipped_keys.append(key)
            if _is_inference_policy_state_key(key):
                critical_keys.append(key)
            continue

        compatible_state_dict[key] = checkpoint_value

    return compatible_state_dict, skipped_keys, critical_keys


def compose_recording_frame(
    rgb_frames: dict[str, np.ndarray],
    depth_frame: np.ndarray,
    extra_panels: list[np.ndarray] | None = None,
    labels: Sequence[str] | None = None,
) -> np.ndarray:
    ordered_panels = [
        _ensure_rgb_uint8(rgb_frames["hero"]),
        _ensure_rgb_uint8(rgb_frames["side"]),
        _ensure_rgb_uint8(rgb_frames["overview"]),
        _ensure_rgb_uint8(depth_frame),
    ]
    if extra_panels:
        ordered_panels.extend(_ensure_rgb_uint8(panel) for panel in extra_panels)
    base_height, base_width = ordered_panels[0].shape[:2]
    resized_panels = [
        _resize_panel(panel, (base_width, base_height)) if panel.shape[:2] != (base_height, base_width) else panel
        for panel in ordered_panels
    ]
    if labels is not None:
        resized_panels = [
            _annotate_panel(panel.copy(), label)
            for panel, label in zip(resized_panels, labels, strict=True)
        ]
    if len(resized_panels) <= 4:
        top_row = np.concatenate(resized_panels[:2], axis=1)
        bottom_row = np.concatenate(resized_panels[2:], axis=1)
        return np.concatenate([top_row, bottom_row], axis=0)

    num_cols = 3
    blank_panel = np.zeros_like(resized_panels[0], dtype=np.uint8)
    while len(resized_panels) % num_cols != 0:
        resized_panels.append(blank_panel.copy())
    rows = [
        np.concatenate(resized_panels[row_start : row_start + num_cols], axis=1)
        for row_start in range(0, len(resized_panels), num_cols)
    ]
    return np.concatenate(rows, axis=0)


def resolve_video_capture_settings(
    *,
    video_length: int,
    video_duration_s: float | None,
    step_dt: float,
    video_frame_stride: int,
    video_fps: float | None,
    replay_route_active: bool = False,
) -> VideoCaptureSettings:
    frame_stride = max(1, int(video_frame_stride))
    default_video_fps = 1.0 / (step_dt * frame_stride)
    resolved_video_fps = default_video_fps if video_fps is None else float(video_fps)
    if resolved_video_fps <= 0.0:
        raise ValueError(f"video_fps must be positive, got {resolved_video_fps}.")
    if replay_route_active and video_duration_s is None:
        max_frames = None
    elif video_duration_s is None:
        max_frames = max(1, int(video_length))
    else:
        max_frames = max(1, int(math.ceil(video_duration_s * resolved_video_fps)))
    return VideoCaptureSettings(
        max_frames=max_frames,
        video_fps=resolved_video_fps,
        frame_stride=frame_stride,
    )


def build_play_video_output_path(
    *,
    log_dir: str | Path,
    resume_path: str | Path,
    video_start_step: int,
    replay_route_path: str | Path | None = None,
    timestamp: datetime | None = None,
) -> str:
    log_dir_path = Path(log_dir)
    resume_path_str = str(resume_path)
    model_token = Path(resume_path_str).stem.split("_")[-1]
    file_stem = f"model_{model_token}-step-{int(video_start_step)}"
    if replay_route_path is not None:
        file_stem += f"-{Path(replay_route_path).stem}"
    resolved_timestamp = datetime.now() if timestamp is None else timestamp
    file_stem += f"-{resolved_timestamp.strftime('%Y%m%d_%H%M%S')}"
    return str(log_dir_path / "videos" / "play" / f"{file_stem}.mp4")


def compute_contact_overlay_state(
    positions_w,
    net_forces_w,
    *,
    force_threshold: float = 1.0,
    force_scale: float = 0.0025,
) -> ContactOverlayState:
    positions = np.asarray(positions_w, dtype=np.float32).reshape(-1, 3)
    forces = np.asarray(net_forces_w, dtype=np.float32).reshape(-1, 3)
    magnitudes = np.linalg.norm(forces, axis=1)
    active_mask = magnitudes > float(force_threshold)
    directions = np.zeros_like(forces, dtype=np.float32)
    if np.any(active_mask):
        directions[active_mask] = forces[active_mask] / magnitudes[active_mask, None]
    lengths = np.where(active_mask, magnitudes * float(force_scale), 0.0).astype(np.float32)
    return ContactOverlayState(
        positions=positions,
        directions=directions,
        lengths=lengths,
        active_mask=active_mask.astype(bool),
    )


def detect_new_contact_events(previous_contact_mask, current_contact_mask) -> np.ndarray:
    previous_mask = np.asarray(previous_contact_mask, dtype=bool).reshape(-1)
    current_mask = np.asarray(current_contact_mask, dtype=bool).reshape(-1)
    if previous_mask.shape != current_mask.shape:
        raise ValueError(
            f"Contact masks must share the same shape, got {previous_mask.shape} and {current_mask.shape}."
        )
    return np.logical_and(~previous_mask, current_mask)


def _configure_ghost_reference_preview(motion_reference) -> None:
    motion_reference.debug_vis = True
    if getattr(motion_reference, "visualizing_robot_from", "aiming_frame") == "aiming_frame":
        motion_reference.visualizing_robot_from = "reference_frame"
    offset = np.asarray(getattr(motion_reference, "visualizing_robot_offset", (0.0, 0.0, 0.0)), dtype=np.float32)
    if offset.shape != (3,) or np.allclose(offset, 0.0):
        motion_reference.visualizing_robot_offset = (0.0, 1.5, 0.0)


def _get_play_visualization_config(options) -> PlayVisualizationConfig:
    visualization = getattr(options, "visualization", None)
    if visualization is None:
        return PlayVisualizationConfig(
            depth_window=bool(getattr(options, "show_depth_window", False)),
            depth_coverage=bool(getattr(options, "show_depth_coverage", False)),
            elevation_map_window=bool(getattr(options, "show_elevation_map_window", False)),
            normals_panel=bool(getattr(options, "normals_panel", False)),
            route_overlay=bool(getattr(options, "route_overlay", False)),
            foot_contact_overlay=bool(getattr(options, "foot_contact_overlay", False)),
            ghost_reference=bool(getattr(options, "ghost_reference", False)),
            obstacle_edges=bool(getattr(options, "obstacle_edges", False)),
        )
    if isinstance(visualization, PlayVisualizationConfig):
        return visualization
    return PlayVisualizationConfig(
        depth_window=bool(getattr(visualization, "depth_window", False)),
        depth_coverage=bool(getattr(visualization, "depth_coverage", False)),
        elevation_map_window=bool(getattr(visualization, "elevation_map_window", False)),
        normals_panel=bool(getattr(visualization, "normals_panel", False)),
        route_overlay=bool(getattr(visualization, "route_overlay", False)),
        foot_contact_overlay=bool(getattr(visualization, "foot_contact_overlay", False)),
        ghost_reference=bool(getattr(visualization, "ghost_reference", False)),
        obstacle_edges=bool(getattr(visualization, "obstacle_edges", False)),
    )


def _resolve_visualization_flag(*, default: bool, override: bool | None) -> bool:
    if override is None:
        return bool(default)
    return bool(override)


def _apply_visualization_sensor_requirements(env_cfg, options) -> None:
    visualization = _get_play_visualization_config(options)
    if visualization.normals_panel:
        _ensure_camera_data_type(env_cfg, "normals")


def _ensure_camera_data_type(env_cfg, data_type: str) -> None:
    camera = getattr(getattr(env_cfg, "scene", None), "camera", None)
    if camera is None:
        return
    data_types = getattr(camera, "data_types", None)
    if data_types is None:
        camera.data_types = [data_type]
        return
    if data_type in data_types:
        return
    if isinstance(data_types, tuple):
        camera.data_types = list(data_types) + [data_type]
        return
    if isinstance(data_types, list):
        camera.data_types.append(data_type)
        return
    camera.data_types = [*list(data_types), data_type]


def _resize_live_preview_panel(panel: np.ndarray, *, scale: float) -> np.ndarray:
    if scale == 1.0:
        return panel
    return cv2.resize(panel, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_NEAREST)


def _estimate_grid_spacing(values: np.ndarray, *, default: float) -> float:
    flattened = np.unique(np.asarray(values, dtype=np.float32).reshape(-1))
    if flattened.size <= 1:
        return float(default)
    diffs = np.diff(np.sort(flattened))
    positive_diffs = diffs[diffs > 1e-5]
    if positive_diffs.size == 0:
        return float(default)
    return float(np.median(positive_diffs))


def _is_inference_policy_state_key(key: str) -> bool:
    return key == "std" or key.startswith("actor.") or key.startswith("encoders.")


def _focus_route_map_tiles(
    origins: np.ndarray,
    *,
    tile_wall_edges,
    current_position_xy,
    route_waypoints_xy: list[list[float]] | list[tuple[float, float]] | None,
    tile_size_x: float,
    tile_size_y: float,
):
    focus_points: list[np.ndarray] = [np.asarray(current_position_xy, dtype=np.float32).reshape(2)]
    if route_waypoints_xy:
        focus_points.append(np.asarray(route_waypoints_xy, dtype=np.float32).reshape(-1, 2))
    if not focus_points:
        return origins, tile_wall_edges

    stacked_points = np.concatenate([points.reshape(-1, 2) for points in focus_points], axis=0)
    focus_bounds = (
        float(stacked_points[:, 0].min() - tile_size_x * 0.55),
        float(stacked_points[:, 0].max() + tile_size_x * 0.55),
        float(stacked_points[:, 1].min() - tile_size_y * 0.55),
        float(stacked_points[:, 1].max() + tile_size_y * 0.55),
    )

    tile_min_x = origins[..., 0] - tile_size_x * 0.5
    tile_max_x = origins[..., 0] + tile_size_x * 0.5
    tile_min_y = origins[..., 1] - tile_size_y * 0.5
    tile_max_y = origins[..., 1] + tile_size_y * 0.5
    active_tiles = (
        (tile_max_x >= focus_bounds[0])
        & (tile_min_x <= focus_bounds[1])
        & (tile_max_y >= focus_bounds[2])
        & (tile_min_y <= focus_bounds[3])
    )
    if not np.any(active_tiles):
        return origins, tile_wall_edges

    active_rows = np.where(active_tiles.any(axis=1))[0]
    active_cols = np.where(active_tiles.any(axis=0))[0]
    row_start = int(active_rows[0])
    row_stop = int(active_rows[-1]) + 1
    col_start = int(active_cols[0])
    col_stop = int(active_cols[-1]) + 1
    focused_origins = origins[row_start:row_stop, col_start:col_stop]
    if tile_wall_edges is None:
        return focused_origins, None
    focused_walls = [list(row[col_start:col_stop]) for row in tile_wall_edges[row_start:row_stop]]
    return focused_origins, focused_walls


def _build_route_map_canvas(*, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    top_color = np.array([247, 249, 252], dtype=np.float32)
    bottom_color = np.array([233, 238, 244], dtype=np.float32)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        alpha = 0.0 if height <= 1 else float(row) / float(height - 1)
        canvas[row, :] = np.round((1.0 - alpha) * top_color + alpha * bottom_color).astype(np.uint8)
    return canvas


def _expand_xy_bounds(bounds: tuple[float, float, float, float], *, min_padding: float) -> tuple[float, float, float, float]:
    min_x, max_x, min_y, max_y = bounds
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    padding = max(min_padding, max(span_x, span_y) * 0.05)
    return (min_x - padding, max_x + padding, min_y - padding, max_y + padding)


def _world_xy_to_panel_pixel(
    xy,
    *,
    bounds: tuple[float, float, float, float],
    image_size: tuple[int, int],
    margin: int,
) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    height, width = image_size
    x = float(np.asarray(xy, dtype=np.float32).reshape(2)[0])
    y = float(np.asarray(xy, dtype=np.float32).reshape(2)[1])
    usable_width = max(width - 1 - margin * 2, 1)
    usable_height = max(height - 1 - margin * 2, 1)
    x_norm = 0.0 if max_x - min_x <= 1e-6 else (x - min_x) / (max_x - min_x)
    y_norm = 0.0 if max_y - min_y <= 1e-6 else (y - min_y) / (max_y - min_y)
    px = int(round(margin + x_norm * usable_width))
    py = int(round(height - 1 - (margin + y_norm * usable_height)))
    return (int(np.clip(px, 0, width - 1)), int(np.clip(py, 0, height - 1)))


def _translate_tile_wall_edges(wall_edges: list[dict] | None, offset_xy) -> list[dict]:
    if not wall_edges:
        return []
    offset = np.asarray(offset_xy, dtype=np.float32).reshape(2)
    return [
        {
            "side": str(wall["side"]),
            "xy": (
                float(wall["xy"][0]) + float(offset[0]),
                float(wall["xy"][1]) + float(offset[1]),
            ),
            "width": float(wall["width"]),
            "height": float(wall["height"]),
        }
        for wall in wall_edges
    ]


def _wall_edge_to_segment(wall: dict) -> tuple[float, float, float, float]:
    x = float(wall["xy"][0])
    y = float(wall["xy"][1])
    width = float(wall["width"])
    height = float(wall["height"])
    side = str(wall["side"])
    if side in {"left", "right"}:
        wall_x = x + width * 0.5
        return wall_x, y, wall_x, y + height
    wall_y = y + height * 0.5
    return x, wall_y, x + width, wall_y


def _collapse_ray_hits_to_height_cells(
    ray_hits_w,
    *,
    resolution_m: float,
) -> dict[tuple[int, int], float]:
    points = np.asarray(ray_hits_w, dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        return {}
    valid_points = points[np.isfinite(points).all(axis=1)]
    if valid_points.size == 0:
        return {}

    cell_heights: dict[tuple[int, int], float] = {}
    inverse_resolution = 1.0 / float(resolution_m)
    for point_x, point_y, point_z in valid_points:
        cell_key = (
            int(np.floor(float(point_x) * inverse_resolution)),
            int(np.floor(float(point_y) * inverse_resolution)),
        )
        previous_height = cell_heights.get(cell_key)
        height_value = float(point_z)
        if previous_height is None or height_value > previous_height:
            cell_heights[cell_key] = height_value
    return cell_heights


def _prune_rolling_elevation_map(
    state: RollingElevationMapState,
    *,
    robot_position_xy,
) -> None:
    if not state.observed_height_by_cell:
        return
    config = state.config
    retention_radius_m = float(config.size_m) * 0.5 * float(config.retention_multiplier)
    min_x = float(robot_position_xy[0]) - retention_radius_m
    max_x = float(robot_position_xy[0]) + retention_radius_m
    min_y = float(robot_position_xy[1]) - retention_radius_m
    max_y = float(robot_position_xy[1]) + retention_radius_m

    keys_to_delete = []
    for cell_key in state.observed_height_by_cell:
        x_min, x_max, y_min, y_max = _cell_bounds_from_index(cell_key, resolution_m=config.resolution_m)
        if x_max < min_x or x_min > max_x or y_max < min_y or y_min > max_y:
            keys_to_delete.append(cell_key)
    for cell_key in keys_to_delete:
        state.observed_height_by_cell.pop(cell_key, None)
        state.last_seen_step_by_cell.pop(cell_key, None)


def _cell_bounds_from_index(
    cell_key: tuple[int, int],
    *,
    resolution_m: float,
) -> tuple[float, float, float, float]:
    cell_x, cell_y = cell_key
    x_min = float(cell_x) * float(resolution_m)
    y_min = float(cell_y) * float(resolution_m)
    return (x_min, x_min + float(resolution_m), y_min, y_min + float(resolution_m))


def _cell_intersects_bounds(
    cell_bounds: tuple[float, float, float, float],
    bounds: tuple[float, float, float, float],
) -> bool:
    x_min, x_max, y_min, y_max = cell_bounds
    bound_min_x, bound_max_x, bound_min_y, bound_max_y = bounds
    return not (x_max < bound_min_x or x_min > bound_max_x or y_max < bound_min_y or y_min > bound_max_y)


def _normalize_elevation_heights_for_display(heights: np.ndarray) -> np.ndarray:
    height_values = np.asarray(heights, dtype=np.float32).reshape(-1)
    if height_values.size == 0:
        return np.zeros((0,), dtype=np.uint8)
    min_height = float(height_values.min())
    max_height = float(height_values.max())
    if max_height - min_height <= 1e-5:
        return np.full(height_values.shape, 160, dtype=np.uint8)
    normalized = (height_values - min_height) / (max_height - min_height)
    return np.clip(np.round(normalized * 255.0), 0.0, 255.0).astype(np.uint8)


def _discover_single_display_active_gpu(*, command_runner) -> int | None:
    try:
        result = command_runner(
            [
                "nvidia-smi",
                "--query-gpu=index,display_active",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None

    enabled_gpu_indices = _parse_display_active_gpu_indices(result.stdout)
    if len(enabled_gpu_indices) != 1:
        return None
    return enabled_gpu_indices[0]


def _extract_cuda_device_index(device: str) -> int | None:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", maxsplit=1)[1])
        except ValueError:
            return None
    return None


def _parse_display_active_gpu_indices(raw_output: str) -> list[int]:
    enabled_gpu_indices: list[int] = []
    for line in raw_output.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue
        index_and_state = [part.strip() for part in stripped_line.split(",", maxsplit=1)]
        if len(index_and_state) != 2:
            continue
        gpu_index_str, display_state = index_and_state
        try:
            gpu_index = int(gpu_index_str)
        except ValueError:
            continue
        if display_state.lower() == "enabled":
            enabled_gpu_indices.append(gpu_index)
    return enabled_gpu_indices


def _set_depth_debug_vis(observation_group, enabled: bool) -> None:
    depth_term = getattr(observation_group, "depth_image", None)
    if depth_term is None or not hasattr(depth_term, "params"):
        return
    depth_term.params["debug_vis"] = enabled


def _configure_command_arrow_visuals(env_cfg, enabled: bool) -> None:
    commands = getattr(env_cfg, "commands", None)
    if commands is None or not hasattr(commands, "base_velocity"):
        return
    base_velocity = commands.base_velocity
    base_velocity.debug_vis = enabled
    if not enabled:
        return
    for attr_name, scale in (
        ("goal_vel_visualizer_cfg", (1.5, 0.45, 0.45)),
        ("current_vel_visualizer_cfg", (1.0, 0.30, 0.30)),
    ):
        marker_cfg = getattr(base_velocity, attr_name, None)
        if marker_cfg is None:
            continue
        markers = getattr(marker_cfg, "markers", None)
        if markers is None or "arrow" not in markers:
            continue
        markers["arrow"].scale = scale


def _lock_center_spawn_pose(env_cfg) -> None:
    events = getattr(env_cfg, "events", None)
    if events is None or not hasattr(events, "reset_base"):
        return
    pose_range = events.reset_base.params.get("pose_range")
    if pose_range is None:
        return
    for axis in ("x", "y", "yaw"):
        pose_range[axis] = (0.0, 0.0)
    velocity_range = events.reset_base.params.get("velocity_range")
    if velocity_range is None:
        return
    for axis in tuple(velocity_range.keys()):
        velocity_range[axis] = (0.0, 0.0)


def _disable_terrain_curriculum(env_cfg) -> None:
    curriculum = getattr(env_cfg, "curriculum", None)
    if curriculum is None:
        return
    if isinstance(curriculum, dict):
        if "terrain_levels" in curriculum:
            curriculum["terrain_levels"] = None
        return
    if hasattr(curriculum, "terrain_levels"):
        curriculum.terrain_levels = None


def _should_enforce_center_spawn(options) -> bool:
    return bool(
        getattr(options, "center_spawn", False)
        or getattr(options, "keyboard_control", False)
        or getattr(options, "replay_route", None)
    )


def _should_disable_auto_reset(options) -> bool:
    return bool(getattr(options, "disable_auto_reset", False) or getattr(options, "replay_route", None))


def _should_use_single_env_play(options) -> bool:
    return bool(
        getattr(options, "keyboard_control", False)
        or getattr(options, "replay_route", None)
        or getattr(options, "exportonnx", False)
        or getattr(options, "useonnx", False)
    )


def _ensure_rgb_uint8(frame) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _resize_panel(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _annotate_panel(frame: np.ndarray, label: str) -> np.ndarray:
    cv2.putText(
        frame,
        label,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame
