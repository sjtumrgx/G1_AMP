from __future__ import annotations
# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np


PREVIEW_ORIGIN = np.array([0.0, 0.0, 0.9], dtype=np.float64)
DEFAULT_WINDOW_SIZE = (1280, 720)
DEFAULT_TITLE_SUFFIX = "mujoco-elevation-viewer"
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "resources"
    / "unitree_g1"
    / "g1.xml"
)
DEFAULT_SURFACE_ROWS = 60
DEFAULT_SURFACE_COLS = 60
DEFAULT_SURFACE_RESOLUTION_M = 0.1
SURFACE_BIN_RGBA = np.asarray(
    [
        [0.23, 0.20, 0.43, 0.95],
        [0.17, 0.45, 0.56, 0.95],
        [0.31, 0.67, 0.42, 0.95],
        [0.95, 0.86, 0.23, 0.95],
    ],
    dtype=np.float32,
)


def _derive_sibling_path(primary_output_path: str | None, suffix: str) -> str | None:
    if primary_output_path is None:
        return None
    output_path = Path(primary_output_path)
    return str(output_path.with_name(f"{output_path.stem}-{DEFAULT_TITLE_SUFFIX}{suffix}"))


def build_mujoco_elevation_capture_path(primary_output_path: str | None) -> str | None:
    return _derive_sibling_path(primary_output_path, ".mp4")


def build_mujoco_elevation_screenshot_path(primary_output_path: str | None) -> str | None:
    return _derive_sibling_path(primary_output_path, ".png")


def build_mujoco_elevation_log_path(primary_output_path: str | None) -> str | None:
    return _derive_sibling_path(primary_output_path, ".log")


def build_mujoco_elevation_frame_dir(primary_output_path: str | None) -> str | None:
    sibling = _derive_sibling_path(primary_output_path, "")
    if sibling is None:
        return None
    return str(Path(sibling).with_name(f"{Path(sibling).name}_frames"))


def _load_mujoco_assets(model_path: Path) -> dict[str, bytes]:
    asset_bytes: dict[str, bytes] = {}
    search_roots = [model_path.parent]
    parent_root = model_path.parent.parent
    if parent_root.exists() and parent_root != model_path.parent:
        search_roots.append(parent_root)
    for search_root in search_roots:
        for asset_path in search_root.rglob("*"):
            if not asset_path.is_file():
                continue
            relative_key = Path(os.path.relpath(asset_path, start=model_path.parent)).as_posix()
            asset_bytes.setdefault(relative_key, asset_path.read_bytes())
    return asset_bytes


def _build_surface_mesh_obj(*, rows: int, cols: int, resolution_m: float) -> bytes:
    half_extent_x = cols * float(resolution_m) * 0.5
    half_extent_y = rows * float(resolution_m) * 0.5
    lines: list[str] = []
    for row in range(rows + 1):
        for col in range(cols + 1):
            x = -half_extent_x + col * float(resolution_m) + float(PREVIEW_ORIGIN[0])
            y = -half_extent_y + row * float(resolution_m) + float(PREVIEW_ORIGIN[1])
            z = float(PREVIEW_ORIGIN[2])
            lines.append(f"v {x:.8f} {y:.8f} {z:.8f}")
    for row in range(rows + 1):
        for col in range(cols + 1):
            u = 0.0 if cols <= 0 else float(col) / float(cols)
            v = 0.0 if rows <= 0 else float(row) / float(rows)
            lines.append(f"vt {u:.8f} {v:.8f}")
    for row in range(rows):
        for col in range(cols):
            top_left = row * (cols + 1) + col + 1
            top_right = top_left + 1
            bottom_left = (row + 1) * (cols + 1) + col + 1
            bottom_right = bottom_left + 1
            lines.append(f"f {top_left}/{top_left} {top_right}/{top_right} {bottom_left}/{bottom_left}")
            lines.append(f"f {top_right}/{top_right} {bottom_right}/{bottom_right} {bottom_left}/{bottom_left}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _compile_g1_viewer_model(
    *,
    model_path: Path,
    surface_rows: int,
    surface_cols: int,
    surface_resolution_m: float,
) -> tuple[mujoco.MjModel, list[int]]:
    spec = mujoco.MjSpec.from_file(str(model_path))
    spec.assets.update(_load_mujoco_assets(model_path))
    spec.assets["elevation_surface.obj"] = _build_surface_mesh_obj(
        rows=surface_rows,
        cols=surface_cols,
        resolution_m=surface_resolution_m,
    )

    if not list(spec.lights):
        light = spec.worldbody.add_light()
        light.name = "viewer_key_light"
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        light.pos = [0.0, 0.0, 4.5]
        light.dir = [0.1, 0.2, -1.0]
        light.diffuse = [0.9, 0.9, 0.9]
        light.ambient = [0.35, 0.35, 0.35]
        light.specular = [0.25, 0.25, 0.25]
        light.castshadow = True

    if spec.worldbody.bodies:
        root_body = spec.worldbody.bodies[0]
        if not list(root_body.joints):
            root_body.add_freejoint()

    mesh_names: list[str] = []
    for bin_index, bin_rgba in enumerate(SURFACE_BIN_RGBA):
        mesh_name = f"elevation_surface_mesh_{bin_index}"
        mesh = spec.add_mesh(name=mesh_name)
        mesh.file = "elevation_surface.obj"
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        geom = spec.worldbody.add_geom()
        geom.name = f"elevation_surface_{bin_index}"
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = mesh_name
        geom.rgba = bin_rgba.tolist()
        geom.conaffinity = 0
        geom.contype = 0
        geom.group = 1
        mesh_names.append(mesh_name)

    model = spec.compile()
    mesh_ids = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)) for mesh_name in mesh_names]
    return model, mesh_ids


@dataclass
class MuJoCoElevationViewerConfig:
    model_path: str | None = None
    surface_rows: int = DEFAULT_SURFACE_ROWS
    surface_cols: int = DEFAULT_SURFACE_COLS
    surface_resolution_m: float = DEFAULT_SURFACE_RESOLUTION_M
    capture_path: str | None = None
    frame_dir: str | None = None
    screenshot_path: str | None = None
    log_path: str | None = None
    window_size: tuple[int, int] = DEFAULT_WINDOW_SIZE
    video_fps: float = 30.0
    video_frame_stride: int = 1


class MuJoCoElevationViewerRig:
    def __init__(self, *, config: MuJoCoElevationViewerConfig):
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._log_handle = None
        self._launch()

    def _launch(self) -> None:
        script_path = Path(__file__).resolve()
        command = [
            sys.executable,
            str(script_path),
            "--sidecar",
            "--model-path",
            str(Path(self.config.model_path or DEFAULT_MODEL_PATH).expanduser().resolve()),
            "--surface-rows",
            str(int(self.config.surface_rows)),
            "--surface-cols",
            str(int(self.config.surface_cols)),
            "--surface-resolution-m",
            str(float(self.config.surface_resolution_m)),
            "--window-width",
            str(int(self.config.window_size[0])),
            "--window-height",
            str(int(self.config.window_size[1])),
            "--video-fps",
            str(float(self.config.video_fps)),
            "--video-frame-stride",
            str(max(1, int(self.config.video_frame_stride))),
        ]
        if self.config.capture_path is not None:
            command.extend(["--capture-path", self.config.capture_path])
        if self.config.frame_dir is not None:
            command.extend(["--frame-dir", self.config.frame_dir])
        if self.config.screenshot_path is not None:
            command.extend(["--screenshot-path", self.config.screenshot_path])

        stdout_target = subprocess.DEVNULL
        if self.config.log_path is not None:
            log_path = Path(self.config.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(log_path, "w", encoding="utf-8")
            stdout_target = self._log_handle

        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=stdout_target,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(Path(__file__).resolve().parents[6]),
        )

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def update(
        self,
        *,
        timestep: int,
        root_position_w,
        root_quat_w,
        joint_positions_by_name: dict[str, float],
        surface_vertices,
        surface_confidence,
        surface_valid,
    ) -> None:
        if not self.is_running or self._process is None or self._process.stdin is None:
            return
        packet = {
            "type": "state",
            "timestep": int(timestep),
            "root_position_w": np.asarray(root_position_w, dtype=np.float64).reshape(3).tolist(),
            "root_quaternion_wxyz": np.asarray(root_quat_w, dtype=np.float64).reshape(4).tolist(),
            "joint_positions_by_name": {
                str(name): float(value) for name, value in joint_positions_by_name.items()
            },
            "surface_vertices": np.asarray(surface_vertices, dtype=np.float64).reshape(-1, 3).tolist(),
            "surface_confidence": np.asarray(surface_confidence, dtype=np.float32).reshape(-1).tolist(),
            "surface_valid": np.asarray(surface_valid, dtype=bool).reshape(-1).tolist(),
        }
        try:
            self._process.stdin.write(json.dumps(packet, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
        except BrokenPipeError:
            pass

    def close(self) -> None:
        if self._process is None:
            return
        try:
            if self._process.stdin is not None:
                self._process.stdin.write('{"type":"close"}\n')
                self._process.stdin.flush()
                self._process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self._process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self._process = None


class _SidecarRuntime:
    def __init__(
        self,
        *,
        model_path: Path,
        surface_rows: int,
        surface_cols: int,
        surface_resolution_m: float,
        capture_path: str | None,
        frame_dir: str | None,
        screenshot_path: str | None,
        window_size: tuple[int, int],
        video_fps: float,
        video_frame_stride: int,
    ):
        self.surface_rows = int(surface_rows)
        self.surface_cols = int(surface_cols)
        self.surface_resolution_m = float(surface_resolution_m)
        self.model, self.surface_mesh_ids = _compile_g1_viewer_model(
            model_path=model_path,
            surface_rows=self.surface_rows,
            surface_cols=self.surface_cols,
            surface_resolution_m=self.surface_resolution_m,
        )
        self.surface_geom_ids = [
            int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"elevation_surface_{bin_index}"))
            for bin_index in range(len(self.surface_mesh_ids))
        ]
        for mesh_id, geom_id in zip(self.surface_mesh_ids, self.surface_geom_ids, strict=True):
            self.model.mesh_pos[mesh_id] = np.zeros((3,), dtype=np.float64)
            self.model.mesh_quat[mesh_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            self.model.geom_pos[geom_id] = np.zeros((3,), dtype=np.float64)
            self.model.geom_quat[geom_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data = mujoco.MjData(self.model)
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), int(window_size[0]))
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), int(window_size[1]))
        self.renderer = mujoco.Renderer(self.model, height=int(window_size[1]), width=int(window_size[0]))
        self.capture_path = Path(capture_path).expanduser().resolve() if capture_path is not None else None
        self.frame_dir = Path(frame_dir).expanduser().resolve() if frame_dir is not None else None
        self.screenshot_path = Path(screenshot_path).expanduser().resolve() if screenshot_path is not None else None
        self.video_fps = float(video_fps)
        self.video_frame_stride = max(1, int(video_frame_stride))
        self.window_size = tuple(int(v) for v in window_size)
        self._writer = None
        self._latest_packet: dict | None = None
        self._latest_packet_version = 0
        self._reader_closed = False
        self._lock = threading.Lock()
        self._frame_index = 0
        self._screenshot_written = False
        self._logged_first_update = False
        self._latest_active_faces = 0
        self._latest_active_tiles = 0
        self._joint_qpos_addr_by_name = self._build_joint_qpos_addr_by_name()
        self._prepare_capture_outputs()

    def _build_joint_qpos_addr_by_name(self) -> dict[str, int]:
        joint_qpos_addr_by_name: dict[str, int] = {}
        for joint_index in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_index)
            if joint_name is None:
                continue
            joint_qpos_addr_by_name[joint_name] = int(self.model.jnt_qposadr[joint_index])
        return joint_qpos_addr_by_name

    def _prepare_capture_outputs(self) -> None:
        if self.capture_path is not None:
            self.capture_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(self.capture_path, fps=self.video_fps)
        if self.frame_dir is not None:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
        if self.screenshot_path is not None:
            self.screenshot_path.parent.mkdir(parents=True, exist_ok=True)

    def _reader_loop(self) -> None:
        for line in sys.stdin:
            stripped = line.strip()
            if not stripped:
                continue
            packet = json.loads(stripped)
            if packet.get("type") == "close":
                with self._lock:
                    self._reader_closed = True
                return
            with self._lock:
                self._latest_packet = packet
                self._latest_packet_version += 1
        with self._lock:
            self._reader_closed = True

    def _snapshot_packet(self) -> tuple[dict | None, int, bool]:
        with self._lock:
            return self._latest_packet, self._latest_packet_version, self._reader_closed

    def _apply_packet(self, packet: dict) -> tuple[int, int]:
        self.data.qpos[:3] = PREVIEW_ORIGIN
        self.data.qpos[3:7] = np.asarray(packet["root_quaternion_wxyz"], dtype=np.float64).reshape(4)
        self.data.qvel[:] = 0.0
        for joint_name, joint_value in packet.get("joint_positions_by_name", {}).items():
            qpos_addr = self._joint_qpos_addr_by_name.get(joint_name)
            if qpos_addr is None:
                continue
            self.data.qpos[qpos_addr] = float(joint_value)

        surface_vertices = np.asarray(packet.get("surface_vertices", []), dtype=np.float32).reshape(-1, 3)
        surface_confidence = np.asarray(packet.get("surface_confidence", []), dtype=np.float32).reshape(self.surface_rows, self.surface_cols)
        surface_valid = np.asarray(packet.get("surface_valid", []), dtype=bool).reshape(self.surface_rows, self.surface_cols)
        mesh_vertex_count = int(self.model.mesh_vertnum[self.surface_mesh_ids[0]])
        mesh_face_count = int(self.model.mesh_facenum[self.surface_mesh_ids[0]])

        face_templates = np.zeros((mesh_face_count, 3), dtype=np.int32)
        face_slot = 0
        for cell_row in range(self.surface_rows):
            for cell_col in range(self.surface_cols):
                top_left = cell_row * (self.surface_cols + 1) + cell_col
                top_right = top_left + 1
                bottom_left = (cell_row + 1) * (self.surface_cols + 1) + cell_col
                bottom_right = bottom_left + 1
                face_templates[face_slot] = np.array([top_left, top_right, bottom_left], dtype=np.int32)
                face_templates[face_slot + 1] = np.array([top_right, bottom_right, bottom_left], dtype=np.int32)
                face_slot += 2

        bin_edges = np.linspace(0.0, 1.0, len(self.surface_mesh_ids) + 1)
        active_face_count = 0
        active_tile_count = int(np.count_nonzero(surface_valid))
        for bin_index, mesh_id in enumerate(self.surface_mesh_ids):
            mesh_vert_adr = int(self.model.mesh_vertadr[mesh_id])
            mesh_face_adr = int(self.model.mesh_faceadr[mesh_id])
            self.model.mesh_vert[mesh_vert_adr : mesh_vert_adr + mesh_vertex_count] = surface_vertices[:mesh_vertex_count]
            bin_faces = np.zeros((mesh_face_count, 3), dtype=np.int32)
            face_slot = 0
            bin_min = float(bin_edges[bin_index])
            bin_max = float(bin_edges[bin_index + 1])
            for cell_row in range(self.surface_rows):
                for cell_col in range(self.surface_cols):
                    confidence_value = float(surface_confidence[cell_row, cell_col])
                    valid_cell = bool(surface_valid[cell_row, cell_col])
                    in_bin = confidence_value >= bin_min and (confidence_value < bin_max or bin_index == len(self.surface_mesh_ids) - 1)
                    if valid_cell and in_bin:
                        bin_faces[face_slot] = face_templates[face_slot]
                        bin_faces[face_slot + 1] = face_templates[face_slot + 1]
                        active_face_count += 2
                    face_slot += 2
            self.model.mesh_face[mesh_face_adr : mesh_face_adr + mesh_face_count] = bin_faces
        mujoco.mj_forward(self.model, self.data)
        for mesh_id in self.surface_mesh_ids:
            mujoco.mjr_uploadMesh(self.model, self.renderer._mjr_context, mesh_id)
        self._latest_active_faces = active_face_count
        self._latest_active_tiles = active_tile_count
        if not self._logged_first_update:
            self._logged_first_update = True
            print(
                "[INFO] MuJoCo elevation viewer received first state update:"
                f" vertices={mesh_vertex_count}, faces={mesh_face_count}, active_faces={active_face_count},"
                f" active_tiles={active_tile_count}, joints={len(packet.get('joint_positions_by_name', {}))}",
                flush=True,
            )
        return mesh_vertex_count, mesh_face_count

    def _capture_frame(self, *, camera, timestep: int) -> None:
        if timestep % self.video_frame_stride != 0:
            return
        if self._writer is None and self.frame_dir is None and (self._screenshot_written or self.screenshot_path is None):
            return
        self.renderer.update_scene(self.data, camera=camera)
        frame = self.renderer.render()
        if (
            self.screenshot_path is not None
            and not self._screenshot_written
            and self._frame_index >= 2
            and self._latest_active_tiles > 0
        ):
            imageio.imwrite(self.screenshot_path, frame)
            self._screenshot_written = True
            print(f"[INFO] MuJoCo elevation viewer wrote screenshot: {self.screenshot_path}", flush=True)
        if self.frame_dir is not None:
            imageio.imwrite(self.frame_dir / f"frame_{self._frame_index:05d}.png", frame)
        if self._writer is not None:
            self._writer.append_data(frame)
        self._frame_index += 1

    def run(self) -> None:
        reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        reader_thread.start()
        last_version = -1
        with mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False) as viewer:
            viewer.cam.lookat[:] = PREVIEW_ORIGIN
            viewer.cam.distance = 3.4
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0
            print("[INFO] MuJoCo elevation viewer window opened.", flush=True)
            while viewer.is_running():
                packet, version, reader_closed = self._snapshot_packet()
                if version != last_version and packet is not None:
                    with viewer.lock():
                        self._apply_packet(packet)
                    for mesh_id in self.surface_mesh_ids:
                        viewer.update_mesh(mesh_id)
                    self._capture_frame(camera=viewer.cam, timestep=int(packet.get("timestep", 0)))
                    last_version = version
                viewer.sync()
                if reader_closed and version == last_version:
                    break
                time.sleep(1.0 / 120.0)
        self.close()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.renderer.close()
        if self.capture_path is not None and self.capture_path.exists():
            print(f"[INFO] MuJoCo elevation viewer wrote video: {self.capture_path}", flush=True)
        if self.frame_dir is not None and self.frame_dir.exists():
            print(f"[INFO] MuJoCo elevation viewer wrote frames: {self.frame_dir}", flush=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuJoCo elevation viewer sidecar.")
    parser.add_argument("--sidecar", action="store_true", default=False)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--surface-rows", type=int, default=DEFAULT_SURFACE_ROWS)
    parser.add_argument("--surface-cols", type=int, default=DEFAULT_SURFACE_COLS)
    parser.add_argument("--surface-resolution-m", type=float, default=DEFAULT_SURFACE_RESOLUTION_M)
    parser.add_argument("--capture-path", type=str, default=None)
    parser.add_argument("--frame-dir", type=str, default=None)
    parser.add_argument("--screenshot-path", type=str, default=None)
    parser.add_argument("--window-width", type=int, default=DEFAULT_WINDOW_SIZE[0])
    parser.add_argument("--window-height", type=int, default=DEFAULT_WINDOW_SIZE[1])
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-frame-stride", type=int, default=1)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if not args.sidecar:
        raise SystemExit("Use --sidecar to run the MuJoCo elevation viewer subprocess.")
    runtime = _SidecarRuntime(
        model_path=Path(args.model_path).expanduser().resolve(),
        surface_rows=max(1, int(args.surface_rows)),
        surface_cols=max(1, int(args.surface_cols)),
        surface_resolution_m=float(args.surface_resolution_m),
        capture_path=args.capture_path,
        frame_dir=args.frame_dir,
        screenshot_path=args.screenshot_path,
        window_size=(int(args.window_width), int(args.window_height)),
        video_fps=float(args.video_fps),
        video_frame_stride=max(1, int(args.video_frame_stride)),
    )
    runtime.run()


if __name__ == "__main__":
    main()
