# pyright: reportAttributeAccessIssue=false

import argparse
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import onnxruntime as ort
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_NAME = "g1_parkour"
DEFAULT_DEPTH_MODE = "mujoco"
SUPPORTED_DEPTH_MODES = ("mujoco", "zeros")
DEFAULT_HEAD_CAMERA_BODY = "torso_link"
DEFAULT_HEAD_CAMERA_NAME = "head_camera"
DEFAULT_HEAD_CAMERA_POS = (0.0487988662332928, 0.01, 0.4378029937970051)
DEFAULT_HEAD_CAMERA_QUAT = (0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0)
DEFAULT_FREE_ROOT_BODY = "robot_root"
DEFAULT_FREE_ROOT_JOINT = "root_freejoint"
DEFAULT_DEPTH_FRAME_SHAPE = (18, 32)
DEFAULT_RAW_RENDER_SHAPE = (36, 64)
DEFAULT_DEPTH_RANGE = (0.0, 2.5)


@dataclass(frozen=True)
class LoggedRunPaths:
    repo_root: Path
    log_dir: Path
    params_dir: Path
    export_dir: Path
    actor_onnx: Path
    depth_encoder_onnx: Path
    agent_cfg: Path
    env_cfg: Path


@dataclass(frozen=True)
class ActionJointMap:
    policy_joint_order: list[str]
    control_joint_order: list[str]
    policy_to_control_indices: list[int]
    extra_control_joints: list[str]


@dataclass
class KeyboardCommandState:
    command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    elastic_band_enabled: bool = True


class HistoryBuffer:
    def __init__(self, feature_dim: int, history_length: int):
        self.feature_dim = feature_dim
        self.history_length = history_length
        self.buffer = np.zeros((history_length, feature_dim), dtype=np.float32)

    def push(self, sample: np.ndarray) -> None:
        sample_array = np.asarray(sample, dtype=np.float32).reshape(self.feature_dim)
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = sample_array

    def flatten(self) -> np.ndarray:
        return self.buffer.reshape(-1).copy()


class ParkourProprioObservationBuilder:
    def __init__(
        self,
        joint_count: int,
        history_length: int = 8,
        observation_scales: dict[str, float] | None = None,
    ):
        self.history_length = history_length
        self.joint_count = joint_count
        self.observation_scales = observation_scales or {
            "base_ang_vel": 1.0,
            "projected_gravity": 1.0,
            "velocity_commands": 1.0,
            "joint_pos": 1.0,
            "joint_vel": 1.0,
            "actions": 1.0,
        }
        self._buffers = {
            "base_ang_vel": HistoryBuffer(feature_dim=3, history_length=history_length),
            "projected_gravity": HistoryBuffer(feature_dim=3, history_length=history_length),
            "velocity_commands": HistoryBuffer(feature_dim=3, history_length=history_length),
            "joint_pos": HistoryBuffer(feature_dim=joint_count, history_length=history_length),
            "joint_vel": HistoryBuffer(feature_dim=joint_count, history_length=history_length),
            "actions": HistoryBuffer(feature_dim=joint_count, history_length=history_length),
        }

    def push(
        self,
        *,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        velocity_command: np.ndarray,
        joint_pos_rel: np.ndarray,
        joint_vel_rel: np.ndarray,
        action: np.ndarray,
    ) -> None:
        self._buffers["base_ang_vel"].push(base_ang_vel)
        self._buffers["projected_gravity"].push(projected_gravity)
        self._buffers["velocity_commands"].push(velocity_command)
        self._buffers["joint_pos"].push(joint_pos_rel)
        self._buffers["joint_vel"].push(joint_vel_rel)
        self._buffers["actions"].push(action)

    def build(self) -> np.ndarray:
        components = [
            self._buffers[name].flatten() * self.observation_scales.get(name, 1.0)
            for name in self.component_slices().keys()
        ]
        return np.concatenate(components, axis=0)

    def component_slices(self) -> dict[str, slice]:
        component_dims = (
            ("base_ang_vel", 3 * self.history_length),
            ("projected_gravity", 3 * self.history_length),
            ("velocity_commands", 3 * self.history_length),
            ("joint_pos", self.joint_count * self.history_length),
            ("joint_vel", self.joint_count * self.history_length),
            ("actions", self.joint_count * self.history_length),
        )
        slices: dict[str, slice] = {}
        offset = 0
        for name, dim in component_dims:
            slices[name] = slice(offset, offset + dim)
            offset += dim
        return slices


class ZeroDepthAdapter:
    def __init__(self, history_length: int, frame_shape: tuple[int, int]):
        self.history_length = history_length
        self.frame_shape = frame_shape

    def build(self) -> np.ndarray:
        return np.zeros((self.history_length, *self.frame_shape), dtype=np.float32)


class MuJoCoDepthAdapter:
    def __init__(
        self,
        *,
        model,
        history_length: int,
        frame_shape: tuple[int, int] = DEFAULT_DEPTH_FRAME_SHAPE,
        raw_shape: tuple[int, int] = DEFAULT_RAW_RENDER_SHAPE,
        camera_name: str = DEFAULT_HEAD_CAMERA_NAME,
        depth_range: tuple[float, float] = DEFAULT_DEPTH_RANGE,
    ):
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.history_length = history_length
        self.frame_shape = frame_shape
        self.raw_shape = raw_shape
        self.camera_name = camera_name
        self.depth_range = depth_range
        self.history = HistoryBuffer(feature_dim=frame_shape[0] * frame_shape[1], history_length=history_length)
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.camera_body_id = model.cam_bodyid[self.camera_id]
        self.camera_local_pos = model.cam_pos[self.camera_id].copy()
        self.camera_local_quat = model.cam_quat[self.camera_id].copy()
        self.geomgroup = np.ones(6, dtype=np.uint8)
        self.deep_max = depth_range[1]
        self.local_ray_dirs = self._build_local_ray_dirs()

    def close(self) -> None:
        return None

    def update(self, data) -> None:
        raw_depth = self._compute_distance_to_image_plane(data)
        processed_depth = self._process_raw_depth(raw_depth)
        self.history.push(processed_depth.reshape(-1))

    def build(self) -> np.ndarray:
        return self.history.buffer.reshape(self.history_length, *self.frame_shape).copy()

    def _build_local_ray_dirs(self) -> np.ndarray:
        focal_length = 1.0
        horizontal_aperture = 2.0 * np.tan(np.radians(89.51) / 2.0)
        vertical_aperture = 2.0 * np.tan(np.radians(58.29) / 2.0)
        h_pixel = horizontal_aperture / self.raw_shape[1]
        v_pixel = vertical_aperture / self.raw_shape[0]
        dirs = np.zeros((self.raw_shape[0] * self.raw_shape[1], 3), dtype=np.float64)
        for i in range(self.raw_shape[0]):
            for j in range(self.raw_shape[1]):
                x = (j + 0.5 - self.raw_shape[1] / 2.0) * h_pixel
                y = (self.raw_shape[0] / 2.0 - i - 0.5) * v_pixel
                z = -focal_length
                vec = np.array([x, y, z], dtype=np.float64)
                dirs[i * self.raw_shape[1] + j] = vec / np.linalg.norm(vec)
        return dirs

    def _compute_distance_to_image_plane(self, data) -> np.ndarray:
        camera_pos = np.zeros(3, dtype=np.float64)
        camera_xmat = np.zeros(9, dtype=np.float64)
        self._mujoco.mj_local2Global(
            data,
            camera_pos,
            camera_xmat,
            self.camera_local_pos,
            self.camera_local_quat,
            int(self.camera_body_id),
            int(self._mujoco.mjtSameFrame.mjSAMEFRAME_BODY),
        )
        rotation = camera_xmat.reshape(3, 3)
        world_dirs = (rotation @ self.local_ray_dirs.T).T.reshape(-1)
        geom_ids = np.full(self.local_ray_dirs.shape[0], -1, dtype=np.int32)
        distances = np.full(self.local_ray_dirs.shape[0], -1.0, dtype=np.float64)
        self._mujoco.mj_multiRay(
            self.model,
            data,
            camera_pos,
            world_dirs,
            self.geomgroup,
            1,
            -1,
            geom_ids,
            distances,
            None,
            self.local_ray_dirs.shape[0],
            self.deep_max,
        )
        plane_dist = distances * np.abs(self.local_ray_dirs[:, 2])
        plane_dist[distances < 0.0] = self.deep_max
        return plane_dist.reshape(self.raw_shape[0], self.raw_shape[1]).astype(np.float32, copy=False)

    def _process_raw_depth(self, raw_depth: np.ndarray) -> np.ndarray:
        clipped = np.clip(raw_depth, self.depth_range[0], self.depth_range[1])
        normalized = (clipped - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0])
        return normalized[::2, ::2].astype(np.float32, copy=False)


class DelayedPDController:
    def __init__(
        self,
        *,
        stiffness: np.ndarray,
        damping: np.ndarray,
        delay_steps: int,
        default_targets: np.ndarray,
    ):
        self.stiffness = np.asarray(stiffness, dtype=np.float32)
        self.damping = np.asarray(damping, dtype=np.float32)
        self.delay_steps = delay_steps
        self.default_targets = np.asarray(default_targets, dtype=np.float32)
        self._target_history = [self.default_targets.copy() for _ in range(delay_steps)]

    def compute_torque(self, target_qpos: np.ndarray, current_qpos: np.ndarray, current_qvel: np.ndarray) -> np.ndarray:
        target_qpos_array = np.asarray(target_qpos, dtype=np.float32).copy()
        if self.delay_steps > 0:
            delayed_target = self._target_history.pop(0)
            self._target_history.append(target_qpos_array)
        else:
            delayed_target = target_qpos_array
        position_error = delayed_target - np.asarray(current_qpos, dtype=np.float32)
        velocity_error = -np.asarray(current_qvel, dtype=np.float32)
        return self.stiffness * position_error + self.damping * velocity_error


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate exported parkour ONNX policy in MuJoCo.")
    parser.add_argument("--load_run", type=str, required=True, help="Parkour run directory under logs/instinct_rl.")
    parser.add_argument("--mjcf", type=str, default=None, help="Optional MuJoCo XML/URDF model override.")
    parser.add_argument(
        "--depth-mode",
        type=str,
        default=DEFAULT_DEPTH_MODE,
        choices=SUPPORTED_DEPTH_MODES,
        help="Depth observation source.",
    )
    parser.add_argument("--headless", action="store_true", default=False, help="Disable the interactive viewer.")
    parser.add_argument("--keyboard_control", action="store_true", default=False, help="Enable keyboard control.")
    parser.add_argument("--keyboard_linvel_step", type=float, default=0.5, help="Forward command delta per key press.")
    parser.add_argument("--keyboard_angvel", type=float, default=1.0, help="Yaw command magnitude.")
    parser.add_argument("--zero_act_until", type=int, default=0, help="Force zero actions until this control step.")
    parser.add_argument("--sim_duration", type=float, default=None, help="Optional simulation duration in seconds.")
    parser.add_argument("--record_npz", type=str, default=None, help="Optional NPZ trace output path.")
    return parser


def resolve_logged_run_paths(
    load_run: str | Path,
    repo_root: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> LoggedRunPaths:
    repo_root_path = Path(repo_root) if repo_root is not None else REPO_ROOT
    load_run_path = Path(load_run)

    if load_run_path.is_absolute():
        log_dir = load_run_path
    else:
        log_dir = repo_root_path / "logs" / "instinct_rl" / experiment_name / load_run_path

    params_dir = log_dir / "params"
    export_dir = log_dir / "exported"
    return LoggedRunPaths(
        repo_root=repo_root_path.resolve(),
        log_dir=log_dir.resolve(),
        params_dir=params_dir.resolve(),
        export_dir=export_dir.resolve(),
        actor_onnx=(export_dir / "actor.onnx").resolve(),
        depth_encoder_onnx=(export_dir / "0-depth_encoder.onnx").resolve(),
        agent_cfg=(params_dir / "agent.yaml").resolve(),
        env_cfg=(params_dir / "env.yaml").resolve(),
    )


def validate_run_artifacts(paths: LoggedRunPaths) -> None:
    required_paths = (
        paths.actor_onnx,
        paths.depth_encoder_onnx,
        paths.agent_cfg,
        paths.env_cfg,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required run artifact not found: {path}")


def load_logged_yaml(path: str | Path) -> dict:
    with Path(path).open() as file:
        data = yaml.unsafe_load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


def resolve_default_model_path(paths: LoggedRunPaths) -> Path:
    env_cfg = load_logged_yaml(paths.env_cfg)
    asset_path = Path(env_cfg["scene"]["robot"]["spawn"]["asset_path"])
    return asset_path.resolve()


def extract_actuated_joint_order_from_urdf(urdf_path: str | Path) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    joint_order: list[str] = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        joint_order.append(joint.attrib["name"])
    return joint_order


def extract_mujoco_actuator_order(mjcf_path: str | Path) -> list[str]:
    return _extract_mujoco_actuator_order(Path(mjcf_path), visited_paths=set())


def build_action_joint_map(policy_joint_order: list[str], control_joint_order: list[str]) -> ActionJointMap:
    control_index_by_name = {joint_name: index for index, joint_name in enumerate(control_joint_order)}
    missing_policy_joints = [joint_name for joint_name in policy_joint_order if joint_name not in control_index_by_name]
    if missing_policy_joints:
        raise ValueError(
            "Control model is missing policy joints: " + ", ".join(missing_policy_joints)
        )

    return ActionJointMap(
        policy_joint_order=list(policy_joint_order),
        control_joint_order=list(control_joint_order),
        policy_to_control_indices=[control_index_by_name[joint_name] for joint_name in policy_joint_order],
        extra_control_joints=[joint_name for joint_name in control_joint_order if joint_name not in policy_joint_order],
    )


def handle_keyboard_command(key: str, state: KeyboardCommandState, linvel_step: float, angvel: float) -> None:
    normalized_key = key.upper()
    if normalized_key == "W":
        state.command[0] += linvel_step
    elif normalized_key == "S":
        state.command[2] = 0.0
    elif normalized_key == "F":
        state.command[2] = angvel
    elif normalized_key == "G":
        state.command[2] = -angvel
    elif normalized_key == "X":
        state.command[:] = 0.0
    elif normalized_key == "8":
        state.elastic_band_enabled = True
    elif normalized_key == "9":
        state.elastic_band_enabled = False


def expand_velocity_command_history(command: np.ndarray, history_length: int) -> np.ndarray:
    return np.tile(np.asarray(command, dtype=np.float32), history_length)


def load_parkour_onnx_sessions(export_dir: str | Path) -> tuple[ort.InferenceSession, ort.InferenceSession]:
    export_dir_path = Path(export_dir)
    providers = ort.get_available_providers()
    encoder_session = ort.InferenceSession(str(export_dir_path / "0-depth_encoder.onnx"), providers=providers)
    actor_session = ort.InferenceSession(str(export_dir_path / "actor.onnx"), providers=providers)
    return encoder_session, actor_session


def run_parkour_policy(
    *,
    encoder_session: ort.InferenceSession,
    actor_session: ort.InferenceSession,
    proprio_observation: np.ndarray,
    depth_observation: np.ndarray,
) -> np.ndarray:
    proprio_batch = np.asarray(proprio_observation, dtype=np.float32).reshape(1, -1)
    depth_batch = np.asarray(depth_observation, dtype=np.float32).reshape(1, *depth_observation.shape)

    depth_latent = np.asarray(
        encoder_session.run(None, {encoder_session.get_inputs()[0].name: depth_batch})[0],
        dtype=np.float32,
    )
    actor_input = np.concatenate([proprio_batch, depth_latent], axis=1)
    action = np.asarray(
        actor_session.run(None, {actor_session.get_inputs()[0].name: actor_input})[0],
        dtype=np.float32,
    )
    return action.astype(np.float32, copy=False).reshape(-1)


def resolve_joint_action_scale(joint_order: list[str], env_cfg: dict) -> np.ndarray:
    action_scale_cfg = env_cfg["actions"]["joint_pos"]["scale"]
    return np.asarray([_resolve_pattern_value(joint_name, action_scale_cfg) for joint_name in joint_order], dtype=np.float32)


def resolve_joint_drive_gains(joint_order: list[str], env_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    actuator_cfgs = env_cfg["scene"]["robot"]["actuators"]
    stiffness = np.zeros(len(joint_order), dtype=np.float32)
    damping = np.zeros(len(joint_order), dtype=np.float32)

    for index, joint_name in enumerate(joint_order):
        stiffness[index] = _resolve_joint_gain(joint_name, actuator_cfgs, "stiffness")
        damping[index] = _resolve_joint_gain(joint_name, actuator_cfgs, "damping")

    return stiffness, damping


def resolve_joint_armature(joint_order: list[str], env_cfg: dict) -> np.ndarray:
    actuator_cfgs = env_cfg["scene"]["robot"]["actuators"]
    armature = np.zeros(len(joint_order), dtype=np.float32)
    for index, joint_name in enumerate(joint_order):
        armature[index] = _resolve_joint_gain(joint_name, actuator_cfgs, "armature")
    return armature


def resolve_joint_init_targets(joint_order: list[str], env_cfg: dict) -> np.ndarray:
    init_joint_pos_cfg = env_cfg["scene"]["robot"]["init_state"]["joint_pos"]
    return np.asarray([_resolve_pattern_value_with_default(joint_name, init_joint_pos_cfg, 0.0) for joint_name in joint_order], dtype=np.float32)


def resolve_joint_effort_limits(joint_order: list[str], env_cfg: dict) -> np.ndarray:
    actuator_cfgs = env_cfg["scene"]["robot"]["actuators"]
    effort_limits = np.zeros(len(joint_order), dtype=np.float32)
    for index, joint_name in enumerate(joint_order):
        effort_limits[index] = _resolve_joint_gain(joint_name, actuator_cfgs, "effort_limit_sim")
    return effort_limits


def resolve_joint_delay_steps(env_cfg: dict) -> int:
    actuator_cfgs = env_cfg["scene"]["robot"]["actuators"]
    return max(int(actuator_cfg.get("max_delay", 0)) for actuator_cfg in actuator_cfgs.values())


def resolve_policy_observation_scales(env_cfg: dict) -> dict[str, float]:
    policy_cfg = env_cfg["observations"]["policy"]
    return {
        "base_ang_vel": float(policy_cfg["base_ang_vel"].get("scale", 1.0) or 1.0),
        "projected_gravity": float(policy_cfg["projected_gravity"].get("scale", 1.0) or 1.0),
        "velocity_commands": float(policy_cfg["velocity_commands"].get("scale", 1.0) or 1.0),
        "joint_pos": float(policy_cfg["joint_pos"].get("scale", 1.0) or 1.0),
        "joint_vel": float(policy_cfg["joint_vel"].get("scale", 1.0) or 1.0),
        "actions": float(policy_cfg["actions"].get("scale", 1.0) or 1.0),
    }


def resolve_nominal_velocity_command(env_cfg: dict) -> np.ndarray:
    velocity_ranges = env_cfg["commands"]["base_velocity"].get("velocity_ranges", {})
    positive_forward_midpoints: list[float] = []
    for terrain_cfg in velocity_ranges.values():
        lin_vel_x = terrain_cfg.get("lin_vel_x")
        if lin_vel_x is None:
            continue
        low, high = float(lin_vel_x[0]), float(lin_vel_x[1])
        if high <= 0.0:
            continue
        positive_forward_midpoints.append(max(0.0, (low + high) * 0.5))

    lin_vel_x_nominal = 0.6 if not positive_forward_midpoints else float(np.median(positive_forward_midpoints))
    return np.array([lin_vel_x_nominal, 0.0, 0.0], dtype=np.float32)


def extract_model_joint_indices(model, joint_order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    qpos_indices = np.zeros(len(joint_order), dtype=np.int32)
    qvel_indices = np.zeros(len(joint_order), dtype=np.int32)
    for index, joint_name in enumerate(joint_order):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in MuJoCo model.")
        qpos_indices[index] = model.jnt_qposadr[joint_id]
        qvel_indices[index] = model.jnt_dofadr[joint_id]
    return qpos_indices, qvel_indices


def resolve_default_joint_targets(model, qpos_indices: np.ndarray) -> np.ndarray:
    return model.qpos0[qpos_indices].astype(np.float32, copy=True)


def compute_projected_gravity(root_quat_wxyz: np.ndarray) -> np.ndarray:
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return rotate_vector_inverse(root_quat_wxyz, gravity_world)


def rotate_vector_inverse(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float32)
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return rotation.T @ np.asarray(vector, dtype=np.float32)


def prepare_mujoco_model_xml(model_path: str | Path, output_dir: str | Path) -> Path:
    import mujoco

    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_path.stem}.floating.xml"

    if model_path.suffix.lower() == ".xml":
        return model_path
    if model_path.suffix.lower() != ".urdf":
        raise ValueError(f"Expected a URDF or MJCF model path, got: {model_path}")

    visual_urdf_path = _build_visual_mujoco_urdf(model_path)
    try:
        compiled_model = mujoco.MjModel.from_xml_path(str(visual_urdf_path))
        mujoco.mj_saveLastXML(str(output_path), compiled_model)
    finally:
        visual_urdf_path.unlink(missing_ok=True)

    tree = ET.parse(output_path)
    xml_root = tree.getroot()
    asset = xml_root.find("asset")
    if asset is None:
        asset = ET.SubElement(xml_root, "asset")
    _ensure_ground_assets(asset)
    _ensure_visual_setup(xml_root)
    worldbody = xml_root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Compiled model is missing a worldbody: {output_path}")

    root_children = list(worldbody)
    for child in root_children:
        worldbody.remove(child)

    floating_root = ET.Element("body", {"name": DEFAULT_FREE_ROOT_BODY, "pos": "0 0 0.755"})
    floating_root.append(ET.Element("freejoint", {"name": DEFAULT_FREE_ROOT_JOINT}))
    for child in root_children:
        floating_root.append(child)
    worldbody.append(floating_root)
    worldbody.append(
        ET.Element(
            "light",
            {
                "name": "scene_light",
                "pos": "1 0 3.5",
                "dir": "0 0 -1",
                "directional": "true",
            },
        )
    )
    worldbody.append(
        ET.Element(
            "geom",
            {
                "name": "floor",
                "type": "plane",
                "size": "0 0 0.05",
                "pos": "0 0 0",
                "material": "groundplane",
            },
        )
    )

    camera_parent = xml_root.find(f".//body[@name='{DEFAULT_HEAD_CAMERA_BODY}']")
    if camera_parent is None:
        camera_parent = xml_root.find(f".//body[@name='{DEFAULT_FREE_ROOT_BODY}']")
    if camera_parent is None:
        raise ValueError(
            f"Could not find camera parent body '{DEFAULT_HEAD_CAMERA_BODY}' or '{DEFAULT_FREE_ROOT_BODY}' in"
            f" {output_path}"
        )
    camera_parent.append(
        ET.Element(
            "camera",
            {
                "name": DEFAULT_HEAD_CAMERA_NAME,
                "pos": " ".join(str(value) for value in DEFAULT_HEAD_CAMERA_POS),
                "quat": " ".join(str(value) for value in DEFAULT_HEAD_CAMERA_QUAT),
            },
        )
    )

    tree.write(output_path)
    return output_path


def settle_root_to_floor(model, data, clearance: float = 0.005) -> None:
    import mujoco

    mujoco.mj_forward(model, data)
    lowest_robot_geom_z = min(
        float(data.geom_xpos[i][2])
        for i in range(model.ngeom)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) != "floor"
    )
    data.qpos[2] -= lowest_robot_geom_z - clearance
    mujoco.mj_forward(model, data)


def _resolve_joint_gain(joint_name: str, actuator_cfgs: dict, gain_key: str) -> float:
    for actuator_cfg in actuator_cfgs.values():
        joint_patterns = actuator_cfg.get("joint_names_expr", [])
        if any(re.fullmatch(pattern, joint_name) for pattern in joint_patterns):
            return _resolve_pattern_value(joint_name, actuator_cfg[gain_key])
    raise ValueError(f"Could not resolve {gain_key} for joint: {joint_name}")


def _resolve_pattern_value(joint_name: str, value_cfg) -> float:
    if isinstance(value_cfg, dict):
        for pattern, value in value_cfg.items():
            if re.fullmatch(pattern, joint_name):
                return float(value)
        raise ValueError(f"Could not resolve configured value for joint: {joint_name}")
    return float(value_cfg)


def _resolve_pattern_value_with_default(joint_name: str, value_cfg, default: float) -> float:
    if isinstance(value_cfg, dict):
        for pattern, value in value_cfg.items():
            if re.fullmatch(pattern, joint_name):
                return float(value)
        return default
    return float(value_cfg)


def _build_visual_mujoco_urdf(model_path: Path) -> Path:
    tree = ET.parse(model_path)
    robot = tree.getroot()
    mujoco_tag = robot.find("mujoco")
    if mujoco_tag is None:
        mujoco_tag = ET.SubElement(robot, "mujoco")
    compiler_tag = mujoco_tag.find("compiler")
    if compiler_tag is None:
        compiler_tag = ET.SubElement(mujoco_tag, "compiler")
    compiler_tag.set("discardvisual", "false")
    compiler_tag.set("meshdir", str((model_path.parent / "../../../assets/resources/unitree_g1/meshes").resolve()))
    compiler_tag.set("balanceinertia", "true")

    fd, temp_path = tempfile.mkstemp(suffix=".urdf")
    Path(temp_path).unlink(missing_ok=True)
    tree.write(temp_path)
    return Path(temp_path)


def _ensure_ground_assets(asset_tag: ET.Element) -> None:
    if asset_tag.find("./texture[@name='groundplane']") is None:
        asset_tag.append(
            ET.Element(
                "texture",
                {
                    "name": "groundplane",
                    "type": "2d",
                    "builtin": "checker",
                    "mark": "edge",
                    "rgb1": "0.2 0.3 0.4",
                    "rgb2": "0.1 0.2 0.3",
                    "markrgb": "0.8 0.8 0.8",
                    "width": "300",
                    "height": "300",
                },
            )
        )
    if asset_tag.find("./material[@name='groundplane']") is None:
        asset_tag.append(
            ET.Element(
                "material",
                {
                    "name": "groundplane",
                    "texture": "groundplane",
                    "texuniform": "true",
                    "texrepeat": "5 5",
                    "reflectance": "0.2",
                },
            )
        )


def _ensure_visual_setup(root_tag: ET.Element) -> None:
    visual_tag = root_tag.find("visual")
    if visual_tag is None:
        visual_tag = ET.SubElement(root_tag, "visual")
    if visual_tag.find("headlight") is None:
        visual_tag.append(
            ET.Element(
                "headlight",
                {
                    "diffuse": "0.6 0.6 0.6",
                    "ambient": "0.1 0.1 0.1",
                    "specular": "0.9 0.9 0.9",
                },
            )
        )
    if visual_tag.find("rgba") is None:
        visual_tag.append(ET.Element("rgba", {"haze": "0.15 0.25 0.35 1"}))
    if visual_tag.find("global") is None:
        visual_tag.append(ET.Element("global", {"azimuth": "-140", "elevation": "-20"}))


def _extract_mujoco_actuator_order(mjcf_path: Path, visited_paths: set[Path]) -> list[str]:
    resolved_path = mjcf_path.resolve()
    if resolved_path in visited_paths:
        return []
    visited_paths.add(resolved_path)

    root = ET.parse(resolved_path).getroot()
    actuator_order: list[str] = []

    for include in root.findall("include"):
        include_path = (resolved_path.parent / include.attrib["file"]).resolve()
        actuator_order.extend(_extract_mujoco_actuator_order(include_path, visited_paths))

    actuator_section = root.find("actuator")
    if actuator_section is not None:
        for actuator in actuator_section:
            joint_name = actuator.attrib.get("joint")
            if joint_name is not None:
                actuator_order.append(joint_name)

    return actuator_order
