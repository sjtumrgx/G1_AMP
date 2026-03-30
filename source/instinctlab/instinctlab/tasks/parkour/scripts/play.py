"""Script to play a checkpoint if an RL agent from Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import subprocess
import sys

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher
from play_runtime import (
    add_play_runtime_args,
    apply_play_runtime_overrides,
    build_default_tracking_camera_specs,
    compose_recording_frame,
    compute_tracking_camera_views,
    ensure_sensor_initialized,
    normalize_depth_frame_for_display,
    resolve_video_capture_settings,
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
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import carb.input
import omni.appwindow
import cv2
from carb.input import KeyboardEventType
from instinct_rl.runners import OnPolicyRunner
from instinct_rl.utils.utils import get_obs_slice, get_subobs_by_components, get_subobs_size

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
from instinctlab.tasks.parkour.scripts.keyboard_commands import (
    ParkourKeyboardCommandController,
    resolve_keyboard_command_limits,
)
from instinctlab.tasks.parkour.scripts.config_loading import load_logged_config
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
        video_layout: str,
    ):
        self.env = env
        self.env_cfg = env_cfg
        self.output_path = output_path
        self.video_start_step = video_start_step
        self.video_max_frames = video_max_frames
        self.video_fps = video_fps
        self.video_frame_stride = video_frame_stride
        self.show_depth_window = show_depth_window
        self.video_layout = video_layout
        self.frames_written = 0
        self.writer = None
        self.depth_window_name = "parkour_depth"
        self._depth_window_disabled = False
        self._camera_specs = build_default_tracking_camera_specs() if output_path is not None else []
        self._cameras: dict[str, Camera] = {}
        self._camera_resolution = (640, 360)
        self._camera_warmup_frames = 5
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
        return self.output_path is not None and self.frames_written >= self.video_max_frames

    def capture(self, timestep: int) -> None:
        depth_panel = _read_depth_panel(self.env, self.env_cfg)
        self._show_depth_window(depth_panel)

        if self.output_path is None or timestep < self.video_start_step or self.is_complete:
            return
        if (timestep - self.video_start_step) % self.video_frame_stride != 0:
            return

        if self.writer is None:
            self.writer = imageio.get_writer(self.output_path, fps=self.video_fps)

        frame = self._build_video_frame(depth_panel)
        self.writer.append_data(frame)
        self.frames_written += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.show_depth_window and not self._depth_window_disabled:
            try:
                cv2.destroyWindow(self.depth_window_name)
            except cv2.error:
                pass

    def _build_video_frame(self, depth_panel: np.ndarray) -> np.ndarray:
        rgb_frames = self._capture_tracking_frames()
        if self.video_layout == "single":
            return rgb_frames["hero"]
        return compose_recording_frame(
            rgb_frames,
            depth_panel,
            labels=("Hero", "Side", "Overview", "Depth"),
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

    def _show_depth_window(self, depth_panel: np.ndarray) -> None:
        if not self.show_depth_window or self._depth_window_disabled:
            return
        try:
            cv2.namedWindow(self.depth_window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(
                self.depth_window_name,
                cv2.resize(depth_panel, None, fx=8.0, fy=8.0, interpolation=cv2.INTER_NEAREST),
            )
            cv2.waitKey(1)
        except cv2.error as exc:
            print(f"[WARN] Disabling depth window after OpenCV error: {exc}")
            self._depth_window_disabled = True


def main():
    """Play with Instinct-RL agent."""
    env = None
    capture_rig = None
    video_output_path = None
    interrupted = False
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

    if args_cli.env_cfg:
        env_cfg = load_logged_config(log_dir, "env")
    if args_cli.agent_cfg:
        agent_cfg_dict = load_logged_config(log_dir, "agent")
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    apply_play_runtime_overrides(env_cfg, args_cli)

    if args_cli.keyboard_control:
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1e10

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
        if args_cli.keyboard_control:
            keyboard_controller = ParkourKeyboardCommandController(
                num_envs=env.num_envs,
                history_length=keyboard_history_length,
                device=env.device,
                limits=resolve_keyboard_command_limits(env_cfg),
                linvel_step=args_cli.keyboard_linvel_step,
                angvel=args_cli.keyboard_angvel,
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
            app_window = omni.appwindow.get_default_app_window()
            keyboard = app_window.get_keyboard()
            input = carb.input.acquire_input_interface()
            input.subscribe_to_keyboard_events(keyboard, on_keyboard_input)

        if args_cli.center_spawn or args_cli.keyboard_control:
            if _pin_first_env_to_center_origin(env.unwrapped):
                obs, _ = env.reset()
            else:
                obs, _ = env.get_observations()
        else:
            obs, _ = env.get_observations()

        if args_cli.show_depth_coverage:
            _log_depth_input_design(env_cfg)

        if args_cli.video:
            video_output_path = os.path.join(
                log_dir,
                "videos",
                "play",
                f"model_{resume_path.split('_')[-1].split('.')[0]}-step-{args_cli.video_start_step}.mp4",
            )
        capture_settings = resolve_video_capture_settings(
            video_length=args_cli.video_length,
            video_duration_s=args_cli.video_duration_s,
            step_dt=float(env.unwrapped.step_dt),
            video_frame_stride=args_cli.video_frame_stride,
            video_fps=args_cli.video_fps,
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
            video_layout=args_cli.video_layout,
        )
        capture_rig.capture(0)
        timestep = 0
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                if args_cli.keyboard_control:
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
            capture_rig.capture(timestep)

            if capture_rig.is_complete:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("[INFO] KeyboardInterrupt received, finalizing partial video before exit.")
    finally:
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
