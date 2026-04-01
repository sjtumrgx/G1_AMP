"""Convert Unitree-style retargeted CSV motions into instinctlab-friendly NPZ files.

This adapts the `whole_body_tracking` preprocessing flow to this repository:

- `<output_name>_retargeted.npz` is compatible with instinctlab's `AmassMotionCfg`
- `<output_name>_motion.npz` stores full-body replay data
- optional W&B upload logs an artifact directory that contains `motion.npz`
  plus the local `*_retargeted.npz` file
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

INPUT_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


parser = argparse.ArgumentParser(description="Replay a retargeted CSV motion and export NPZ files.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input motion CSV file.")
parser.add_argument("--input_fps", type=int, default=30, help="FPS of the input CSV motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Optional inclusive 1-based frame range to load from the CSV.",
)
parser.add_argument("--output_name", type=str, required=True, help="Basename for the exported motion files.")
parser.add_argument("--output_fps", type=int, default=50, help="FPS of the exported motion files.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=".",
    help="Directory for the local `<name>_motion.npz` and `<name>_retargeted.npz` outputs.",
)
parser.add_argument(
    "--skip_registry_upload",
    action="store_true",
    default=False,
    help="Skip W&B upload and only write local files.",
)
parser.add_argument(
    "--registry_type",
    type=str,
    default="motions",
    help="W&B artifact/registry type used when uploading replay data.",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="csv_to_npz",
    help="W&B project name used when uploading replay data.",
)
parser.add_argument(
    "--wandb_entity",
    type=str,
    default=None,
    help="Optional W&B entity override. If omitted, wandb's normal resolution is used.",
)
parser.add_argument(
    "--temp_dir",
    type=str,
    default=None,
    help="Optional directory for temporary artifact staging. Use this if `/tmp` is unavailable.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from instinctlab.assets.unitree_g1 import G1_29DOF_TORSOBASE_POPSICLE_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = G1_29DOF_TORSOBASE_POPSICLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class CsvMotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self) -> None:
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        if motion.shape[0] < 2:
            raise ValueError("At least two frames are required to interpolate and estimate velocities.")

        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"[INFO] Loaded motion {self.motion_file} with {self.input_frames} frames ({self.duration:.3f}s).")

    def _interpolate_motion(self) -> None:
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            "[INFO] Interpolated motion from"
            f" {self.input_frames}@{self.input_fps}fps to {self.output_frames}@{self.output_fps}fps."
        )

    @staticmethod
    def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    @staticmethod
    def _slerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for idx in range(a.shape[0]):
            slerped_quats[idx] = quat_slerp(a[idx], b[idx], blend[idx])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.clamp(index_0 + 1, max=self.input_frames - 1)
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self) -> None:
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    @staticmethod
    def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(
        self,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], bool]:
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def _build_motion_payload(
    *,
    log: dict[str, np.ndarray | np.generic],
    output_fps: int,
    joint_names: list[str],
    body_names: list[str],
    root_body_name: str,
    root_body_index: int,
) -> dict[str, np.ndarray | np.generic]:
    motion_payload: dict[str, np.ndarray | np.generic] = {
        "fps": np.asarray(output_fps, dtype=np.float32),
        "framerate": np.asarray(output_fps, dtype=np.float32),
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        "root_body_name": np.asarray(root_body_name),
        "root_body_index": np.asarray(root_body_index, dtype=np.int32),
        "joint_pos": log["joint_pos"],
        "joint_vel": log["joint_vel"],
        "body_pos_w": log["body_pos_w"],
        "body_quat_w": log["body_quat_w"],
        "body_lin_vel_w": log["body_lin_vel_w"],
        "body_ang_vel_w": log["body_ang_vel_w"],
    }
    motion_payload["base_pos_w"] = motion_payload["body_pos_w"][:, root_body_index, :]
    motion_payload["base_quat_w"] = motion_payload["body_quat_w"][:, root_body_index, :]
    motion_payload["base_lin_vel_w"] = motion_payload["body_lin_vel_w"][:, root_body_index, :]
    motion_payload["base_ang_vel_w"] = motion_payload["body_ang_vel_w"][:, root_body_index, :]
    return motion_payload


def _write_outputs(
    *,
    motion_payload: dict[str, np.ndarray | np.generic],
    output_name: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / f"{output_name}_motion.npz"
    retargeted_path = output_dir / f"{output_name}_retargeted.npz"

    np.savez(replay_path, **motion_payload)
    np.savez(
        retargeted_path,
        framerate=motion_payload["framerate"],
        joint_names=motion_payload["joint_names"],
        joint_pos=motion_payload["joint_pos"],
        joint_vel=motion_payload["joint_vel"],
        base_pos_w=motion_payload["base_pos_w"],
        base_quat_w=motion_payload["base_quat_w"],
        base_lin_vel_w=motion_payload["base_lin_vel_w"],
        base_ang_vel_w=motion_payload["base_ang_vel_w"],
    )

    return replay_path, retargeted_path


def _upload_to_wandb(*, motion_payload: dict[str, np.ndarray | np.generic], retargeted_path: Path) -> None:
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "W&B upload requires `wandb` to be installed. Re-run with `--skip_registry_upload` or install wandb."
        ) from exc

    with tempfile.TemporaryDirectory(dir=args_cli.temp_dir) as tmp_dir:
        staging_dir = Path(tmp_dir)
        np.savez(staging_dir / "motion.npz", **motion_payload)
        staged_retargeted_path = staging_dir / retargeted_path.name
        staged_retargeted_path.write_bytes(retargeted_path.read_bytes())

        init_kwargs = dict(project=args_cli.wandb_project, name=args_cli.output_name)
        if args_cli.wandb_entity:
            init_kwargs["entity"] = args_cli.wandb_entity
        run = wandb.init(**init_kwargs)
        if run is None:
            raise RuntimeError("wandb.init() did not return a run handle.")

        print(f"[INFO] Uploading replay artifact `{args_cli.output_name}` to W&B.")
        logged_artifact = run.log_artifact(
            artifact_or_path=str(staging_dir),
            name=args_cli.output_name,
            type=args_cli.registry_type,
        )
        run.link_artifact(
            artifact=logged_artifact,
            target_path=f"wandb-registry-{args_cli.registry_type}/{args_cli.output_name}",
        )
        print(
            "[INFO] W&B registry upload complete:"
            f" wandb-registry-{args_cli.registry_type}/{args_cli.output_name}"
        )
        run.finish()


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    motion = CsvMotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(INPUT_JOINT_NAMES, preserve_order=True)[0]
    _, joint_names = robot.find_joints(".*", preserve_order=True)
    _, body_names = robot.find_bodies(".*", preserve_order=True)
    root_body_index = robot.find_bodies(["torso_link"], preserve_order=True)[0][0]

    log: dict[str, list[np.ndarray] | np.ndarray | np.generic] = {
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        log["joint_pos"].append(robot.data.joint_pos[0, :].detach().cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].detach().cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].detach().cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].detach().cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].detach().cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].detach().cpu().numpy().copy())

        if not reset_flag:
            continue

        stacked_log = {
            key: np.stack(value, axis=0)  # type: ignore[arg-type]
            for key, value in log.items()
        }
        motion_payload = _build_motion_payload(
            log=stacked_log,
            output_fps=args_cli.output_fps,
            joint_names=joint_names,
            body_names=body_names,
            root_body_name="torso_link",
            root_body_index=root_body_index,
        )
        replay_path, retargeted_path = _write_outputs(
            motion_payload=motion_payload,
            output_name=args_cli.output_name,
            output_dir=Path(args_cli.output_dir).expanduser(),
        )
        print(f"[INFO] Wrote replay motion file to {replay_path}")
        print(f"[INFO] Wrote instinctlab retargeted motion file to {retargeted_path}")

        if not args_cli.skip_registry_upload:
            _upload_to_wandb(motion_payload=motion_payload, retargeted_path=retargeted_path)

        break


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO] Setup complete.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
