import argparse
import math
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from packaging.version import Version


NO_AUTO_RESET_EPISODE_LENGTH_S = 1e10
VIDEO_LAYOUT_CHOICES = ("quad", "single")
ISAACSIM_REQUIRED_NUMPY_VERSION = "1.26.0"


@dataclass(frozen=True)
class TrackingCameraSpec:
    name: str
    eye_offset: tuple[float, float, float]
    target_offset: tuple[float, float, float]


@dataclass(frozen=True)
class VideoCaptureSettings:
    max_frames: int
    video_fps: float
    frame_stride: int


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
        action="store_true",
        default=False,
        help="Show the latest depth frame in a separate OpenCV window.",
    )
    parser.add_argument(
        "--show_depth_coverage",
        action="store_true",
        default=False,
        help="Visualize depth ray hits in the RGB scene.",
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
    return parser


def validate_isaacsim_python_environment(numpy_version: str | None = None) -> None:
    current_numpy_version = numpy_version if numpy_version is not None else np.__version__
    if Version(current_numpy_version).major >= 2:
        raise RuntimeError(
            "Incompatible Python environment for Isaac Sim: detected numpy"
            f" {current_numpy_version}, but Isaac Sim 5.1 / isaacsim-kernel requires numpy=={ISAACSIM_REQUIRED_NUMPY_VERSION}."
            " Downgrade numpy in this environment before launching play.py."
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
    return np.array([center_xy[0], center_xy[1], nearest_origin[2]], dtype=np.float32)


def apply_play_runtime_overrides(env_cfg, options) -> None:
    if getattr(options, "disable_auto_reset", False):
        env_cfg.episode_length_s = NO_AUTO_RESET_EPISODE_LENGTH_S
        for term_name in vars(env_cfg.terminations):
            setattr(env_cfg.terminations, term_name, None)

    if getattr(options, "show_depth_window", False):
        _set_depth_debug_vis(env_cfg.observations.policy, enabled=False)
        if hasattr(env_cfg.observations, "critic"):
            _set_depth_debug_vis(env_cfg.observations.critic, enabled=False)

    if getattr(options, "show_depth_coverage", False):
        env_cfg.scene.camera.debug_vis = True

    if getattr(options, "center_spawn", False):
        _lock_center_spawn_pose(env_cfg)

    _configure_command_arrow_visuals(env_cfg, enabled=getattr(options, "show_command_arrow", False))


def build_default_tracking_camera_specs() -> list[TrackingCameraSpec]:
    return [
        TrackingCameraSpec("hero", eye_offset=(-4.0, -1.5, 1.8), target_offset=(1.0, 0.0, 0.9)),
        TrackingCameraSpec("side", eye_offset=(0.0, -4.5, 1.6), target_offset=(0.7, 0.0, 0.9)),
        TrackingCameraSpec("overview", eye_offset=(-1.0, 0.0, 5.5), target_offset=(1.0, 0.0, 0.7)),
    ]


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


def compose_recording_frame(
    rgb_frames: dict[str, np.ndarray],
    depth_frame: np.ndarray,
    labels: tuple[str, str, str, str] | None = None,
) -> np.ndarray:
    ordered_panels = [
        _ensure_rgb_uint8(rgb_frames["hero"]),
        _ensure_rgb_uint8(rgb_frames["side"]),
        _ensure_rgb_uint8(rgb_frames["overview"]),
        _ensure_rgb_uint8(depth_frame),
    ]
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
    top_row = np.concatenate(resized_panels[:2], axis=1)
    bottom_row = np.concatenate(resized_panels[2:], axis=1)
    return np.concatenate([top_row, bottom_row], axis=0)


def resolve_video_capture_settings(
    *,
    video_length: int,
    video_duration_s: float | None,
    step_dt: float,
    video_frame_stride: int,
    video_fps: float | None,
) -> VideoCaptureSettings:
    frame_stride = max(1, int(video_frame_stride))
    default_video_fps = 1.0 / (step_dt * frame_stride)
    resolved_video_fps = default_video_fps if video_fps is None else float(video_fps)
    if resolved_video_fps <= 0.0:
        raise ValueError(f"video_fps must be positive, got {resolved_video_fps}.")
    if video_duration_s is None:
        max_frames = max(1, int(video_length))
    else:
        max_frames = max(1, int(math.ceil(video_duration_s * resolved_video_fps)))
    return VideoCaptureSettings(
        max_frames=max_frames,
        video_fps=resolved_video_fps,
        frame_stride=frame_stride,
    )


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
    for axis in ("x", "y"):
        pose_range[axis] = (0.0, 0.0)


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
