"""Script to play a checkpoint if an RL agent from Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import subprocess
import sys

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))
sys.path.append(os.path.join(os.getcwd(), "source", "instinctlab", "instinctlab", "tasks", "parkour", "scripts"))

from isaaclab.app import AppLauncher
from play_runtime import (
    DEPTH_WINDOW_NAME,
    NORMALS_WINDOW_NAME,
    add_play_runtime_args,
    build_play_video_output_path,
    build_default_tracking_camera_specs,
    build_live_preview_panels,
    compose_recording_frame,
    compute_tracking_camera_views,
    create_keyboard_event_subscription,
    ensure_sensor_initialized,
    normalize_depth_frame_for_display,
    normalize_normals_frame_for_display,
    prepare_play_env_cfg,
    resolve_play_visualization_config,
    resolve_recording_camera_resolution,
    resolve_interactive_rendering_gpu_override,
    resolve_play_seed,
    resolve_video_capture_settings,
    render_route_map_panel,
    select_center_terrain_origin,
    validate_isaacsim_python_environment,
)

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=3000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_start_step", type=int, default=0, help="Start step for the simulation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--exportonnx", action="store_true", default=False, help="Export policy as ONNX model.")
parser.add_argument("--useonnx", action="store_true", default=False, help="Use the exported ONNX model for inference.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
parser.add_argument("--no_resume", default=None, action="store_true", help="Force play in no resume mode.")
# custom play arguments
parser.add_argument("--env_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--sample", action="store_true", default=False, help="Sample actions instead of using the policy.")
parser.add_argument("--zero_act_until", type=int, default=0, help="Zero actions until this timestep.")
parser.add_argument("--keyboard_control", action="store_true", default=False, help="Enable keyboard control.")
parser.add_argument("--keyboard_linvel_step", type=float, default=0.5, help="Linear velocity change per keyboard step.")
parser.add_argument("--keyboard_angvel", type=float, default=1.0, help="Angular velocity set by keyboard.")
add_play_runtime_args(parser)

# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
validate_isaacsim_python_environment()


class DisplayAwareAppLauncher(AppLauncher):
    def __init__(self, launcher_args=None, *, active_gpu_override: int | None = None, **kwargs):
        self._active_gpu_override = active_gpu_override
        super().__init__(launcher_args, **kwargs)

    def _resolve_device_settings(self, launcher_args: dict):
        super()._resolve_device_settings(launcher_args)
        if self._active_gpu_override is None:
            return
        if "cuda" not in launcher_args.get("device", ""):
            return
        launcher_args["active_gpu"] = int(self._active_gpu_override)
        print(
            "[INFO][AppLauncher]: Using rendering GPU:"
            f" cuda:{self._active_gpu_override} while simulation tensors remain on"
            f" {launcher_args.get('device', f'cuda:{self.device_id}')}"
        )


# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
# Force single-GPU rendering for play/replay. This avoids Isaac Sim mGPU P2P startup failures
# on systems where multi-GPU peer access probing is unstable or already pre-configured.
args_cli.multi_gpu = False
rendering_gpu_override = resolve_interactive_rendering_gpu_override(
    args_cli.device,
    headless=args_cli.headless,
)
if rendering_gpu_override is not None:
    print(
        "[INFO] Interactive X11 display is active on GPU"
        f" {rendering_gpu_override}; rendering will use cuda:{rendering_gpu_override}"
        f" while simulation tensors stay on {args_cli.device}."
    )

# launch omniverse app
app_launcher = DisplayAwareAppLauncher(args_cli, active_gpu_override=rendering_gpu_override)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import carb.input
import cv2
import isaaclab.utils.math as math_utils
from carb.input import KeyboardEventType
from instinct_rl.runners import OnPolicyRunner
from instinct_rl.utils.utils import get_obs_slice, get_subobs_by_components, get_subobs_size

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, SPHERE_MARKER_CFG
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
from instinctlab.tasks.parkour.scripts.keyboard_commands import (
    ParkourKeyboardCommandController,
    resolve_keyboard_command_limits,
)
from instinctlab.tasks.parkour.scripts.config_loading import load_logged_config
from instinctlab.tasks.parkour.scripts.play_route import (
    RouteWaypointFollower,
    RouteWaypointRecorder,
    build_line_strip_segments,
    build_tile_wall_edges_grid,
    build_route_overlay_points,
    detect_new_contact_events,
    load_route_artifact,
    compute_contact_overlay_state,
    predict_future_trajectory_points,
    save_route_artifact,
)
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

# wait for attach if in debug mode
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


def _quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float32)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _read_root_state(env) -> tuple[np.ndarray, float]:
    robot = env.unwrapped.scene["robot"]
    root_position = robot.data.root_pos_w[0].detach().cpu().numpy()
    root_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
    return root_position, _quat_wxyz_to_yaw(root_quat)


def _pin_first_env_to_center_origin(env) -> bool:
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        print("[WARN] Center spawn requested but terrain origins are unavailable.")
        return False

    center_origin_np = select_center_terrain_origin(terrain.terrain_origins.detach().cpu().numpy())
    center_origin = torch.as_tensor(center_origin_np, device=env.device, dtype=terrain.env_origins.dtype)
    terrain.env_origins[0] = center_origin
    print(f"[INFO] Center spawn origin set to: {center_origin.tolist()}")
    return True


def _resolve_depth_display_range(env_cfg, using_noised_output: bool) -> tuple[float, float]:
    if using_noised_output:
        return (0.0, 1.0)
    noise_pipeline = getattr(env_cfg.scene.camera, "noise_pipeline", None)
    if noise_pipeline is None:
        return (0.0, 2.5)
    depth_normalization = noise_pipeline.get("depth_normalization")
    if depth_normalization is None:
        return (0.0, 2.5)
    return tuple(getattr(depth_normalization, "depth_range", (0.0, 2.5)))


def _read_depth_panel(env, env_cfg) -> np.ndarray:
    camera_sensor = env.unwrapped.scene.sensors["camera"]
    outputs = camera_sensor.data.output
    data_type = "distance_to_image_plane_noised" if "distance_to_image_plane_noised" in outputs else "distance_to_image_plane"
    depth_frame = outputs[data_type][0].detach().cpu().numpy()
    if depth_frame.ndim == 3:
        depth_frame = depth_frame[..., 0]
    depth_range = _resolve_depth_display_range(env_cfg, using_noised_output=data_type.endswith("_noised"))
    return normalize_depth_frame_for_display(depth_frame, depth_range)


def _read_normals_panel(env) -> np.ndarray | None:
    camera_sensor = env.unwrapped.scene.sensors["camera"]
    outputs = camera_sensor.data.output
    if "normals" not in outputs:
        return None
    normals_frame = outputs["normals"][0].detach().cpu().numpy()
    return normalize_normals_frame_for_display(normals_frame)


def _log_depth_input_design(env_cfg) -> None:
    camera_cfg = env_cfg.scene.camera
    crop_cfg = None
    if getattr(camera_cfg, "noise_pipeline", None) is not None:
        crop_cfg = camera_cfg.noise_pipeline.get("crop_and_resize")
    if crop_cfg is None:
        return
    raw_height = getattr(camera_cfg.pattern_cfg, "height", None)
    raw_width = getattr(camera_cfg.pattern_cfg, "width", None)
    print(
        "[INFO] Depth coverage note: the red debug points show the raw camera FOV"
        f" ({raw_height}x{raw_width}), while the policy depth uses crop region"
        f" {tuple(crop_cfg.crop_region)}. The effective depth image is intentionally the lower-center cropped patch."
    )


class RouteOverlayDebugDraw:
    def __init__(
        self,
        *,
        enabled: bool,
        route_points: list[tuple[float, float, float]] | None = None,
        prediction_horizon_s: float = 1.0,
        prediction_samples: int = 20,
    ):
        self.enabled = enabled
        self.route_points = route_points or []
        self.prediction_horizon_s = float(prediction_horizon_s)
        self.prediction_samples = int(prediction_samples)
        self._draw = None
        if not self.enabled:
            return
        try:
            from isaacsim.util.debug_draw import _debug_draw

            self._draw = _debug_draw.acquire_debug_draw_interface()
        except Exception as exc:  # pragma: no cover - runtime-only dependency
            print(f"[WARN] Route overlay disabled because debug_draw is unavailable: {exc}")
            self.enabled = False

    def update(self, *, position_xy, yaw: float, command) -> None:
        if not self.enabled or self._draw is None:
            return

        self._draw.clear_lines()
        route_starts, route_ends = build_line_strip_segments(self.route_points)
        if route_starts:
            self._draw.draw_lines(
                route_starts,
                route_ends,
                [(1.0, 0.85, 0.2, 1.0)] * len(route_starts),
                [3.0] * len(route_starts),
            )

        prediction_points = predict_future_trajectory_points(
            position_xy=position_xy,
            yaw=yaw,
            command=command,
            horizon_s=self.prediction_horizon_s,
            num_samples=self.prediction_samples,
            z_height=0.08,
        )
        if len(prediction_points) < 2:
            return
        self._draw.draw_lines(
            prediction_points[:-1],
            prediction_points[1:],
            [(0.15, 0.85, 1.0, 1.0)] * (len(prediction_points) - 1),
            [2.0] * (len(prediction_points) - 1),
        )

    def close(self) -> None:
        if self._draw is not None:
            self._draw.clear_lines()


class FootContactOverlayRig:
    def __init__(
        self,
        *,
        enabled: bool,
        env,
        touchdown_ttl_steps: int = 300,
        force_threshold: float = 1.0,
        force_scale: float = 0.0025,
    ):
        self.enabled = enabled
        self.touchdown_ttl_steps = int(touchdown_ttl_steps)
        self.force_threshold = float(force_threshold)
        self.force_scale = float(force_scale)
        self._footfall_records: list[tuple[int, np.ndarray]] = []
        self._previous_contact_mask = np.zeros(0, dtype=bool)
        if not self.enabled:
            return

        self._robot = env.unwrapped.scene["robot"]
        self._contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
        self._robot_foot_ids, _ = self._robot.find_bodies(".*_ankle_roll_link", preserve_order=True)
        self._sensor_foot_ids, _ = self._contact_sensor.find_bodies(".*_ankle_roll_link", preserve_order=True)
        self._previous_contact_mask = np.zeros(len(self._robot_foot_ids), dtype=bool)

        arrow_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Play/foot_contact_forces")
        arrow_cfg.markers["arrow"].scale = (1.0, 0.05, 0.05)
        marker_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Play/footfall_markers")
        marker_cfg.markers["sphere"].radius = 0.05
        marker_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.8, 0.1))
        self._force_visualizer = VisualizationMarkers(arrow_cfg)
        self._footfall_visualizer = VisualizationMarkers(marker_cfg)
        self._force_visualizer.set_visibility(False)
        self._footfall_visualizer.set_visibility(False)

    def update(self, *, timestep: int) -> None:
        if not self.enabled:
            return

        foot_positions = self._robot.data.body_pos_w[0, self._robot_foot_ids, :].detach().cpu().numpy()
        contact_forces = self._contact_sensor.data.net_forces_w[0, self._sensor_foot_ids, :].detach().cpu().numpy()
        state = compute_contact_overlay_state(
            foot_positions,
            contact_forces,
            force_threshold=self.force_threshold,
            force_scale=self.force_scale,
        )

        new_contact_events = detect_new_contact_events(self._previous_contact_mask, state.active_mask)
        for point in state.positions[new_contact_events]:
            self._footfall_records.append((int(timestep) + self.touchdown_ttl_steps, point.copy()))
        self._previous_contact_mask = state.active_mask.copy()

        self._update_force_markers(state)
        self._update_footfall_markers(timestep=int(timestep))

    def close(self) -> None:
        if not self.enabled:
            return
        self._force_visualizer.set_visibility(False)
        self._footfall_visualizer.set_visibility(False)

    def _update_force_markers(self, state) -> None:
        if not np.any(state.active_mask):
            self._force_visualizer.set_visibility(False)
            return

        active_positions = torch.as_tensor(state.positions[state.active_mask], device=self._robot.device, dtype=torch.float32)
        active_directions = torch.as_tensor(
            state.directions[state.active_mask],
            device=self._robot.device,
            dtype=torch.float32,
        )
        active_lengths = torch.as_tensor(state.lengths[state.active_mask], device=self._robot.device, dtype=torch.float32)

        quaternions = _vectors_to_world_quaternions(active_directions)
        scales = torch.ones((active_positions.shape[0], 3), device=self._robot.device, dtype=torch.float32)
        scales[:, 0] = active_lengths
        scales[:, 1] = 0.05
        scales[:, 2] = 0.05
        self._force_visualizer.set_visibility(True)
        self._force_visualizer.visualize(
            translations=active_positions,
            orientations=quaternions,
            scales=scales,
        )

    def _update_footfall_markers(self, *, timestep: int) -> None:
        self._footfall_records = [
            (expiry_step, point)
            for expiry_step, point in self._footfall_records
            if expiry_step >= timestep
        ]
        if not self._footfall_records:
            self._footfall_visualizer.set_visibility(False)
            return
        points = torch.as_tensor(
            np.stack([point for _, point in self._footfall_records], axis=0),
            device=self._robot.device,
            dtype=torch.float32,
        )
        self._footfall_visualizer.set_visibility(True)
        self._footfall_visualizer.visualize(translations=points)


def _vectors_to_world_quaternions(vectors: torch.Tensor) -> torch.Tensor:
    default_direction = torch.zeros_like(vectors)
    default_direction[:, 0] = 1.0
    normalized_direction = vectors / torch.norm(vectors, dim=-1, keepdim=True).clamp_min(1e-6)
    axis = torch.cross(default_direction, normalized_direction, dim=-1)
    axis = axis / torch.norm(axis, dim=-1, keepdim=True).clamp_min(1e-6)
    angle = torch.acos(torch.clamp(torch.sum(default_direction * normalized_direction, dim=-1), -1.0, 1.0))
    return math_utils.quat_from_angle_axis(angle, axis)


class PlayCaptureRig:
    def __init__(
        self,
        env,
        env_cfg,
        output_path: str | None,
        video_start_step: int,
        video_max_frames: int,
        video_fps: float,
        video_frame_stride: int,
        show_depth_window: bool,
        show_normals_panel: bool,
        video_layout: str,
        route_map_waypoints_xy: list[list[float]] | list[tuple[float, float]] | None = None,
        route_map_tile_wall_edges=None,
    ):
        self.env = env
        self.env_cfg = env_cfg
        self.output_path = output_path
        self.video_start_step = video_start_step
        self.video_max_frames = video_max_frames
        self.video_fps = video_fps
        self.video_frame_stride = video_frame_stride
        self.show_depth_window = show_depth_window
        self.show_normals_panel = show_normals_panel
        self.video_layout = video_layout
        self.frames_written = 0
        self.writer = None
        self._preview_window_names = []
        if self.show_depth_window:
            self._preview_window_names.append(DEPTH_WINDOW_NAME)
        if self.show_normals_panel:
            self._preview_window_names.append(NORMALS_WINDOW_NAME)
        self._disabled_preview_windows: set[str] = set()
        self.route_map_waypoints_xy = [list(point) for point in route_map_waypoints_xy] if route_map_waypoints_xy else None
        self.route_map_tile_wall_edges = route_map_tile_wall_edges
        self._camera_specs = build_default_tracking_camera_specs() if output_path is not None else []
        self._cameras: dict[str, Camera] = {}
        self._camera_resolution = resolve_recording_camera_resolution(video_layout)
        self._camera_warmup_frames = 5
        terrain = getattr(self.env.unwrapped.scene, "terrain", None)
        terrain_origins = getattr(terrain, "terrain_origins", None)
        self._route_map_terrain_origins = (
            terrain_origins.detach().cpu().numpy() if terrain_origins is not None and self.route_map_waypoints_xy is not None else None
        )
        self._route_map_center_origin = (
            select_center_terrain_origin(self._route_map_terrain_origins)
            if self._route_map_terrain_origins is not None
            else None
        )
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self._cameras = self._create_tracking_cameras()
            print(
                f"[INFO] Recording composite play video to {output_path}"
                f" with layout={video_layout}, start_step={video_start_step},"
                f" max_frames={video_max_frames}, fps={video_fps}, stride={video_frame_stride}"
            )

    @property
    def is_complete(self) -> bool:
        return self.output_path is not None and self.video_max_frames is not None and self.frames_written >= self.video_max_frames

    def capture(self, timestep: int) -> None:
        depth_panel = _read_depth_panel(self.env, self.env_cfg)
        normals_panel = _read_normals_panel(self.env) if self.show_normals_panel else None
        self._show_preview_windows(depth_panel, normals_panel)

        if self.output_path is None or timestep < self.video_start_step or self.is_complete:
            return
        if (timestep - self.video_start_step) % self.video_frame_stride != 0:
            return

        if self.writer is None:
            self.writer = imageio.get_writer(self.output_path, fps=self.video_fps)

        frame = self._build_video_frame(depth_panel, normals_panel)
        self.writer.append_data(frame)
        self.frames_written += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        for window_name in self._preview_window_names:
            if window_name in self._disabled_preview_windows:
                continue
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass

    def _build_video_frame(self, depth_panel: np.ndarray, normals_panel: np.ndarray | None) -> np.ndarray:
        rgb_frames = self._capture_tracking_frames()
        if self.video_layout == "single":
            return rgb_frames["hero"]
        extra_panels: list[np.ndarray] = []
        labels = ["Hero", "Side", "Overview", "Depth"]
        if normals_panel is not None:
            extra_panels.append(normals_panel)
            labels.append("Normals")
        route_map_panel = self._build_route_map_panel()
        if route_map_panel is not None:
            extra_panels.append(route_map_panel)
            labels.append("Route Map")
        return compose_recording_frame(
            rgb_frames,
            depth_panel,
            extra_panels=extra_panels or None,
            labels=tuple(labels),
        )

    def _capture_tracking_frames(self) -> dict[str, np.ndarray]:
        robot = self.env.unwrapped.scene["robot"]
        root_position = robot.data.root_pos_w[0].detach().cpu().numpy()
        root_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
        root_yaw = _quat_wxyz_to_yaw(root_quat)
        views = compute_tracking_camera_views(root_position=root_position, root_yaw=root_yaw, specs=self._camera_specs)

        for spec in self._camera_specs:
            eye, target = views[spec.name]
            self._cameras[spec.name].set_world_poses_from_view(
                torch.as_tensor(eye, device=self.env.device, dtype=torch.float32).unsqueeze(0),
                torch.as_tensor(target, device=self.env.device, dtype=torch.float32).unsqueeze(0),
            )

        self.env.unwrapped.sim.render()

        frames: dict[str, np.ndarray] = {}
        for spec in self._camera_specs:
            camera = self._cameras[spec.name]
            camera.update(self.env.unwrapped.step_dt)
            frames[spec.name] = np.ascontiguousarray(camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3])
        return frames

    def _create_tracking_cameras(self) -> dict[str, Camera]:
        width, height = self._camera_resolution
        cameras: dict[str, Camera] = {}
        for spec in self._camera_specs:
            camera_cfg = CameraCfg(
                prim_path=f"/World/PlayCameras/{spec.name}",
                update_period=0.0,
                data_types=["rgb"],
                width=width,
                height=height,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1.0e5),
                ),
            )
            camera = Camera(camera_cfg)
            self._ensure_camera_initialized(camera)
            cameras[spec.name] = camera
        for _ in range(self._camera_warmup_frames):
            self.env.unwrapped.sim.render()
            for camera in cameras.values():
                camera.update(self.env.unwrapped.step_dt, force_recompute=True)
        return cameras

    def _ensure_camera_initialized(self, camera: Camera) -> None:
        if camera.is_initialized:
            return
        self.env.unwrapped.sim.render()
        ensure_sensor_initialized(camera, sensor_name=camera.cfg.prim_path)

    def _show_preview_windows(self, depth_panel: np.ndarray, normals_panel: np.ndarray | None) -> None:
        preview_panels = build_live_preview_panels(
            depth_panel=depth_panel,
            normals_panel=normals_panel,
            show_depth_window=self.show_depth_window,
            show_normals_window=self.show_normals_panel,
        )
        if not preview_panels:
            return
        displayed_any_panel = False
        for window_name, panel in preview_panels:
            if window_name in self._disabled_preview_windows:
                continue
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, panel)
                displayed_any_panel = True
            except cv2.error as exc:
                print(f"[WARN] Disabling preview window '{window_name}' after OpenCV error: {exc}")
                self._disabled_preview_windows.add(window_name)
        if displayed_any_panel:
            cv2.waitKey(1)

    def _build_route_map_panel(self) -> np.ndarray | None:
        if self.route_map_waypoints_xy is None or self._route_map_terrain_origins is None:
            return None
        root_position, root_yaw = _read_root_state(self.env)
        return render_route_map_panel(
            terrain_origins=self._route_map_terrain_origins,
            center_origin=self._route_map_center_origin,
            current_position_xy=root_position[:2],
            current_yaw=root_yaw,
            route_waypoints_xy=self.route_map_waypoints_xy,
            tile_wall_edges=self.route_map_tile_wall_edges,
            image_size=(720, 720),
        )


def main():
    """Play with Instinct-RL agent."""
    env = None
    capture_rig = None
    keyboard_subscription = None
    video_output_path = None
    interrupted = False
    route_artifact = None
    route_recorder = None
    route_overlay = None
    foot_contact_overlay = None
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg.load_run = args_cli.load_run
    if agent_cfg.load_run is not None:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint
            )
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    elif not args_cli.no_resume:
        raise RuntimeError(
            "\033[91m[ERROR] No checkpoint specified and play.py resumes from a checkpoint by default. Please specify"
            " a checkpoint to resume from using --load_run or use --no_resume to disable this behavior.\033[0m"
        )
    else:
        print(f"[INFO] No experiment directory specified. Using default: {log_root_path}")
        log_dir = os.path.join(log_root_path, agent_cfg.run_name + "_play")
        resume_path = "model_scratch.pt"

    if args_cli.replay_route is not None:
        route_artifact = load_route_artifact(args_cli.replay_route)
        if route_artifact.task is not None and route_artifact.task != args_cli.task:
            raise ValueError(
                f"Route task mismatch: route was recorded for {route_artifact.task},"
                f" but current task is {args_cli.task}."
            )
    if args_cli.keyboard_control and route_artifact is not None:
        raise ValueError("--keyboard_control cannot be combined with --replay_route.")

    if args_cli.env_cfg:
        env_cfg = load_logged_config(log_dir, "env")
    args_cli.visualization = resolve_play_visualization_config(env_cfg, args_cli)
    args_cli.show_depth_window = args_cli.visualization.depth_window
    args_cli.show_depth_coverage = args_cli.visualization.depth_coverage
    args_cli.normals_panel = args_cli.visualization.normals_panel
    args_cli.route_overlay = args_cli.visualization.route_overlay
    args_cli.foot_contact_overlay = args_cli.visualization.foot_contact_overlay
    args_cli.ghost_reference = args_cli.visualization.ghost_reference
    args_cli.obstacle_edges = args_cli.visualization.obstacle_edges
    play_seed = resolve_play_seed(
        cli_seed=args_cli.seed,
        route_seed=route_artifact.seed if route_artifact is not None else None,
        agent_seed=getattr(agent_cfg, "seed", None),
    )
    if play_seed is not None:
        agent_cfg.seed = play_seed
        print(f"[INFO] Using play seed: {play_seed}")
    prepare_play_env_cfg(env_cfg, args_cli, seed=play_seed, route_artifact=route_artifact)
    if args_cli.agent_cfg:
        agent_cfg_dict = load_logged_config(log_dir, "agent")
        if isinstance(agent_cfg_dict, dict) and play_seed is not None:
            agent_cfg_dict["seed"] = play_seed
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    try:
        # create isaac environment
        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video and args_cli.video_layout == "single" else None,
        )

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # wrap around environment for instinct-rl
        env = InstinctRlVecEnvWrapper(env)

        # load previously trained model
        ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
        if agent_cfg.load_run is not None:
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            ppo_runner.load(resume_path)

        # obtain the trained policy for inference
        if args_cli.sample:
            policy = ppo_runner.alg.actor_critic.act
        else:
            policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        # export policy to onnx/jit
        if agent_cfg.load_run is not None:
            export_model_dir = os.path.join(log_dir, "exported")
            if args_cli.exportonnx:
                assert env.unwrapped.num_envs == 1, "Exporting to ONNX is only supported for single environment."
                if not os.path.exists(export_model_dir):
                    os.makedirs(export_model_dir)
                obs, _ = env.get_observations()
                ppo_runner.alg.actor_critic.export_as_onnx(obs, export_model_dir)

        # use the exported model for inference
        if args_cli.useonnx:
            from onnxer import load_parkour_onnx_model

            # NOTE: This is only applicable with parkour task
            onnx_policy = load_parkour_onnx_model(
                model_dir=os.path.join(log_dir, "exported"),
                get_subobs_func=lambda obs: get_subobs_by_components(
                    obs,
                    agent_cfg.policy.encoder_configs.depth_encoder.component_names,
                    env.get_obs_segments(),
                    temporal=True,
                ),
                depth_shape=env.get_obs_segments()["depth_image"],
                proprio_slice=slice(
                    0,
                    get_subobs_size(
                        env.get_obs_segments(),
                        [
                            "base_lin_vel",
                            "base_ang_vel",
                            "projected_gravity",
                            "velocity_commands",
                            "joint_pos",
                            "joint_vel",
                            "actions",
                        ],
                    ),
                ),
            )

        command_obs_slice = get_obs_slice(env.get_obs_segments(), "velocity_commands")
        keyboard_history_length = command_obs_slice[1][0] // 3
        keyboard_controller = None
        route_follower = None
        if args_cli.keyboard_control or route_artifact is not None:
            keyboard_controller = ParkourKeyboardCommandController(
                num_envs=env.num_envs,
                history_length=keyboard_history_length,
                device=env.device,
                limits=resolve_keyboard_command_limits(env_cfg),
                linvel_step=args_cli.keyboard_linvel_step,
                angvel=args_cli.keyboard_angvel,
            )
        if route_artifact is not None:
            route_follower = RouteWaypointFollower(
                waypoints_xy=route_artifact.waypoints_xy,
                limits=resolve_keyboard_command_limits(env_cfg),
                lookahead_distance_m=args_cli.route_lookahead_m,
                goal_tolerance_m=args_cli.route_goal_tolerance_m,
                cruise_speed=args_cli.route_cruise_speed,
            )
            print(
                f"[INFO] Loaded route replay from {args_cli.replay_route}"
                f" with {len(route_artifact.waypoints_xy)} waypoints."
            )
        route_overlay_points = None
        if route_artifact is not None:
            route_overlay_points = build_route_overlay_points(waypoints_xy=route_artifact.waypoints_xy, z_height=0.08)
        route_overlay = RouteOverlayDebugDraw(
            enabled=args_cli.route_overlay,
            route_points=route_overlay_points,
        )
        foot_contact_overlay = FootContactOverlayRig(
            enabled=args_cli.foot_contact_overlay,
            env=env,
        )

        def on_keyboard_input(e):
            if keyboard_controller is None:
                return
            if e.type != KeyboardEventType.KEY_PRESS:
                return
            if e.input == carb.input.KeyboardInput.W:
                keyboard_controller.forward()
            if e.input == carb.input.KeyboardInput.S:
                keyboard_controller.reset_yaw()
            if e.input == carb.input.KeyboardInput.F:
                keyboard_controller.yaw_positive()
            if e.input == carb.input.KeyboardInput.G:
                keyboard_controller.yaw_negative()
            if e.input == carb.input.KeyboardInput.X:
                keyboard_controller.zero_all()

        if args_cli.keyboard_control:
            keyboard_subscription = create_keyboard_event_subscription(on_keyboard_input)

        if args_cli.center_spawn or args_cli.keyboard_control or route_artifact is not None:
            if _pin_first_env_to_center_origin(env.unwrapped):
                obs, _ = env.reset()
            else:
                obs, _ = env.get_observations()
        else:
            obs, _ = env.get_observations()

        if args_cli.save_route is not None:
            terrain_generator_cfg = getattr(getattr(env_cfg.scene, "terrain", None), "terrain_generator", None)
            saved_tile_wall_edges = None
            if terrain_generator_cfg is not None:
                num_rows = int(getattr(terrain_generator_cfg, "num_rows"))
                num_cols = int(getattr(terrain_generator_cfg, "num_cols"))
                subterrain_specific_cfgs = getattr(env.unwrapped.scene.terrain, "subterrain_specific_cfgs", None)
                saved_tile_wall_edges = build_tile_wall_edges_grid(
                    subterrain_specific_cfgs=subterrain_specific_cfgs,
                    num_rows=num_rows,
                    num_cols=num_cols,
                )
            route_recorder = RouteWaypointRecorder(
                task=args_cli.task,
                seed=play_seed,
                step_dt=float(env.unwrapped.step_dt),
                waypoint_spacing_m=args_cli.route_interval_m,
                tile_wall_edges=saved_tile_wall_edges,
            )
            root_position, _ = _read_root_state(env)
            route_recorder.record_position(root_position)

        if args_cli.show_depth_coverage:
            _log_depth_input_design(env_cfg)

        if args_cli.video:
            video_output_path = build_play_video_output_path(
                log_dir=log_dir,
                resume_path=resume_path,
                video_start_step=args_cli.video_start_step,
                replay_route_path=args_cli.replay_route,
            )
        capture_settings = resolve_video_capture_settings(
            video_length=args_cli.video_length,
            video_duration_s=args_cli.video_duration_s,
            step_dt=float(env.unwrapped.step_dt),
            video_frame_stride=args_cli.video_frame_stride,
            video_fps=args_cli.video_fps,
            replay_route_active=route_artifact is not None,
        )
        capture_rig = PlayCaptureRig(
            env=env,
            env_cfg=env_cfg,
            output_path=video_output_path,
            video_start_step=args_cli.video_start_step,
            video_max_frames=capture_settings.max_frames,
            video_fps=capture_settings.video_fps,
            video_frame_stride=capture_settings.frame_stride,
            show_depth_window=args_cli.show_depth_window,
            show_normals_panel=args_cli.normals_panel,
            video_layout=args_cli.video_layout,
            route_map_waypoints_xy=None if route_artifact is None else route_artifact.waypoints_xy,
            route_map_tile_wall_edges=None if route_artifact is None else route_artifact.tile_wall_edges,
        )
        capture_rig.capture(0)
        if route_overlay is not None:
            root_position, root_yaw = _read_root_state(env)
            current_command = (
                keyboard_controller.command[0].detach().cpu().numpy()
                if keyboard_controller is not None
                else np.zeros(3, dtype=np.float32)
            )
            route_overlay.update(position_xy=root_position[:2], yaw=root_yaw, command=current_command)
        if foot_contact_overlay is not None:
            foot_contact_overlay.update(timestep=0)
        timestep = 0
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                if route_follower is not None:
                    root_position, root_yaw = _read_root_state(env)
                    replay_command = route_follower.compute_command(position_xy=root_position[:2], yaw=root_yaw)
                    keyboard_controller.set_command(replay_command)
                if args_cli.keyboard_control or route_follower is not None:
                    obs[:, command_obs_slice[0]] = keyboard_controller.build_observation()
                actions = policy(obs)
                if args_cli.useonnx:
                    torch_actions = actions
                    actions = onnx_policy(obs)
                    if (actions - torch_actions).abs().max() > 1e-5:
                        print(
                            "[INFO]: ONNX model and PyTorch model have a difference of"
                            f" {(actions - torch_actions).abs().max()} in actions at joint"
                            f" {((actions - torch_actions).abs() > 1e-5).nonzero(as_tuple=True)[0]}"
                        )
                if timestep < args_cli.zero_act_until:
                    actions[:] = 0.0
                # env stepping
                obs, rewards, dones, infos = env.step(actions)
            timestep += 1
            if route_recorder is not None:
                root_position, _ = _read_root_state(env)
                route_recorder.record_position(root_position)
            if route_overlay is not None:
                root_position, root_yaw = _read_root_state(env)
                current_command = (
                    keyboard_controller.command[0].detach().cpu().numpy()
                    if keyboard_controller is not None
                    else np.zeros(3, dtype=np.float32)
                )
                route_overlay.update(position_xy=root_position[:2], yaw=root_yaw, command=current_command)
            if foot_contact_overlay is not None:
                foot_contact_overlay.update(timestep=timestep)
            capture_rig.capture(timestep)

            if capture_rig.is_complete or (route_follower is not None and route_follower.is_complete):
                break
    except KeyboardInterrupt:
        interrupted = True
        print("[INFO] KeyboardInterrupt received, finalizing partial video before exit.")
    finally:
        if keyboard_subscription is not None:
            keyboard_subscription.close()
        if route_recorder is not None:
            if env is not None:
                root_position, _ = _read_root_state(env)
                route_recorder.record_position(root_position)
            route_path = save_route_artifact(args_cli.save_route, route_recorder.build_artifact())
            print(f"[INFO] Saved route artifact to {route_path}")
        if route_overlay is not None:
            route_overlay.close()
        if foot_contact_overlay is not None:
            foot_contact_overlay.close()
        if capture_rig is not None:
            capture_rig.close()
        if env is not None:
            env.close()

    if args_cli.video and video_output_path is not None and os.path.isfile(video_output_path):
        subprocess.run(
            [
                "code",
                "-r",
                video_output_path,
            ],
            check=False,
        )

    return 130 if interrupted else 0


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
