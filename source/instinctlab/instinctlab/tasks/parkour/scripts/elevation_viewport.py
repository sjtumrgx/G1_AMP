from __future__ import annotations

from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np

import carb
import omni.kit.app
import omni.usd
from omni.kit.viewport.utility import create_viewport_window
from omni.kit.widget.viewport.capture import FileCapture
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

from play_runtime import build_elevation_surface_mesh_data, compute_preview_relative_body_positions


class _FrameFileCapture(FileCapture):
    def __init__(self, frame_path: Path):
        self.frame_path = Path(frame_path)
        self.completed = False
        super().__init__(str(self.frame_path))

    def _set_completed(self, value=True):  # pragma: no cover - runtime-only capture path
        self.completed = True
        super()._set_completed(value)


class ElevationViewportRig:
    WINDOW_NAME = "parkour_elevation_viewport"
    ROOT_PATH = Sdf.Path("/World/ElevationViewport")
    ROBOT_ROOT_PATH = ROOT_PATH.AppendChild("Robot")
    SURFACE_PATH = ROOT_PATH.AppendChild("Surface")
    LOOKS_PATH = ROOT_PATH.AppendChild("Looks")
    CAMERA_PATH = ROOT_PATH.AppendChild("Camera")
    LIGHT_PATH = ROOT_PATH.AppendChild("KeyLight")
    WORLD_ANCHOR = np.array([320.0, 320.0, 36.0], dtype=np.float32)
    PREVIEW_ORIGIN = np.array([0.0, 0.0, 0.9], dtype=np.float32)

    def __init__(
        self,
        *,
        body_names,
        source_robot_prim_path: str,
        elevation_map_state=None,
        output_path: str | None = None,
        video_fps: float = 0.0,
        video_frame_stride: int = 1,
        window_size: tuple[int, int] = (1280, 720),
        window_position: tuple[int, int] = (80, 120),
    ):
        self._stage = omni.usd.get_context().get_stage()
        if self._stage is None:
            raise RuntimeError("Main stage is unavailable; cannot build elevation viewport preview scene.")

        self.body_names = list(body_names)
        if not self.body_names:
            raise RuntimeError("No robot body names provided for elevation viewport preview.")

        self._source_robot_prim_path = Sdf.Path(source_robot_prim_path)
        if not self._stage.GetPrimAtPath(self._source_robot_prim_path).IsValid():
            raise RuntimeError(
                f"Elevation viewport source robot prim does not exist: {self._source_robot_prim_path}."
            )

        self.elevation_map_state = elevation_map_state
        self.window_size = tuple(int(v) for v in window_size)
        self.window_position = tuple(int(v) for v in window_position)
        self.output_path: str | None = None
        self.video_fps = 0.0
        self.video_frame_stride = 1
        self._capture_paths: list[Path] = []
        self._capture_frames_dir: Path | None = None
        self._next_capture_index = 0
        self._pending_capture: _FrameFileCapture | None = None
        self._body_prims: dict[str, object] = {}
        self.viewport_window = None
        self.viewport_api = None

        self._clear_preview_prims()
        self._build_preview_prims()

        self.viewport_window = create_viewport_window(
            name=self.WINDOW_NAME,
            width=self.window_size[0],
            height=self.window_size[1],
            position_x=self.window_position[0],
            position_y=self.window_position[1],
            camera_path=self.CAMERA_PATH,
        )
        if self.viewport_window is None:
            raise RuntimeError("Failed to create elevation viewport window.")
        self.viewport_window.visible = True
        self.viewport_api = self.viewport_window.viewport_api
        self.viewport_api.camera_path = self.CAMERA_PATH
        self.viewport_api.resolution = self.window_size
        app = omni.kit.app.get_app()
        app.update()
        app.update()

        self._set_initial_camera_view()
        self.configure_capture(output_path, video_fps=video_fps, video_frame_stride=video_frame_stride)
        print(
            "[INFO] Elevation viewport window created:"
            f" {self.WINDOW_NAME} using source robot {self._source_robot_prim_path}"
        )

    def configure_capture(self, output_path: str | None, *, video_fps: float, video_frame_stride: int) -> None:
        self._service_pending_capture(force_flush=True)
        self.output_path = output_path
        self.video_fps = float(video_fps)
        self.video_frame_stride = max(1, int(video_frame_stride))
        self._capture_paths.clear()
        self._capture_frames_dir = None
        self._next_capture_index = 0
        if self.output_path is not None:
            output_path_obj = Path(self.output_path)
            self._capture_frames_dir = output_path_obj.with_suffix("")
            self._capture_frames_dir = self._capture_frames_dir.parent / f"{self._capture_frames_dir.name}_frames"
            self._capture_frames_dir.mkdir(parents=True, exist_ok=True)
            for stale_frame in self._capture_frames_dir.glob("frame_*.png"):
                try:
                    stale_frame.unlink()
                except OSError:
                    pass

    def _clear_preview_prims(self) -> None:
        if self._stage.GetPrimAtPath(self.ROOT_PATH).IsValid():
            self._stage.RemovePrim(self.ROOT_PATH)

    def _build_preview_prims(self) -> None:
        root = UsdGeom.Xform.Define(self._stage, self.ROOT_PATH)
        self._set_pose(root.GetPrim(), self.WORLD_ANCHOR, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        UsdGeom.Xform.Define(self._stage, self.ROBOT_ROOT_PATH)
        self._build_robot_fallback_material()
        self._build_surface_prim()
        self._build_robot_visual_prims()
        self._build_camera_and_light()

    def _build_robot_fallback_material(self) -> None:
        material_path = self.LOOKS_PATH.AppendChild("RobotFallback")
        shader_path = material_path.AppendChild("Shader")
        material = UsdShade.Material.Define(self._stage, material_path)
        shader = UsdShade.Shader.Define(self._stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.92, 0.98))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.22)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self._robot_fallback_material = material

    def _build_surface_prim(self) -> None:
        mesh = UsdGeom.Mesh.Define(self._stage, self.SURFACE_PATH)
        mesh.GetDoubleSidedAttr().Set(True)
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        material_path = self.LOOKS_PATH.AppendChild("ElevationSurface")
        shader_path = material_path.AppendChild("Shader")
        material = UsdShade.Material.Define(self._stage, material_path)
        shader = UsdShade.Shader.Define(self._stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.26, 0.64, 0.95))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.08)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
        self._surface_mesh = mesh
        self._surface_color_primvar = mesh.CreateDisplayColorPrimvar(interpolation=UsdGeom.Tokens.vertex)

    def _build_robot_visual_prims(self) -> None:
        missing_visuals: list[str] = []
        for body_name in self.body_names:
            body_path = self.ROBOT_ROOT_PATH.AppendChild(body_name)
            body_prim = UsdGeom.Xform.Define(self._stage, body_path).GetPrim()
            self._body_prims[body_name] = body_prim

            source_visual_path = self._source_robot_prim_path.AppendChild(body_name).AppendChild("visuals")
            if self._stage.GetPrimAtPath(source_visual_path).IsValid():
                visuals_ref_prim = UsdGeom.Xform.Define(self._stage, body_path.AppendChild("VisualsRef")).GetPrim()
                visuals_ref_prim.GetReferences().AddInternalReference(source_visual_path)
                continue

            missing_visuals.append(body_name)
            fallback = UsdGeom.Sphere.Define(self._stage, body_path.AppendChild("Proxy"))
            fallback.CreateRadiusAttr().Set(0.055 if body_name == "pelvis" else 0.035)
            UsdShade.MaterialBindingAPI(fallback.GetPrim()).Bind(self._robot_fallback_material)

        if missing_visuals:
            carb.log_warn(
                "Elevation viewport fell back to proxy geometry for bodies without a visuals subtree:"
                f" {', '.join(missing_visuals)}"
            )

    def _build_camera_and_light(self) -> None:
        camera = UsdGeom.Camera.Define(self._stage, self.CAMERA_PATH)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 120.0))
        camera.CreateFocalLengthAttr().Set(24.0)
        camera.CreateFocusDistanceAttr().Set(400.0)

        light = UsdLux.DistantLight.Define(self._stage, self.LIGHT_PATH)
        light.CreateIntensityAttr().Set(3500.0)
        light.CreateAngleAttr().Set(0.53)
        self._set_pose(
            light.GetPrim(),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.8924, 0.2391, 0.3696, 0.0990], dtype=np.float32),
        )

    @staticmethod
    def _ensure_pose_ops(prim):
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        if len(ops) >= 3:
            return ops[0], ops[1], ops[2]
        xformable.ClearXformOpOrder()
        xformable.SetResetXformStack(True)
        translate_op = xformable.AddTranslateOp()
        orient_op = xformable.AddOrientOp()
        scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))
        return translate_op, orient_op, scale_op

    @classmethod
    def _set_pose(cls, prim, position: np.ndarray, quat_wxyz: np.ndarray) -> None:
        translate_op, orient_op, scale_op = cls._ensure_pose_ops(prim)
        translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        orient_op.Set(Gf.Quatf(float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])))
        scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))

    def _set_initial_camera_view(self) -> None:
        from isaacsim.core.utils.viewports import set_camera_view

        eye = self.WORLD_ANCHOR + self.PREVIEW_ORIGIN + np.asarray([-4.5, -3.2, 2.5], dtype=np.float32)
        target = self.WORLD_ANCHOR + self.PREVIEW_ORIGIN + np.asarray([0.0, 0.0, 0.35], dtype=np.float32)
        set_camera_view(eye=eye, target=target, camera_prim_path=str(self.CAMERA_PATH), viewport_api=self.viewport_api)

    def update(self, timestep: int, *, robot_root_position_w, robot_body_positions_w, robot_body_quats_w) -> None:
        relative_positions = compute_preview_relative_body_positions(
            robot_root_position_w,
            robot_body_positions_w,
            preview_origin=self.PREVIEW_ORIGIN,
        )
        for body_index, body_name in enumerate(self.body_names):
            body_prim = self._body_prims.get(body_name)
            if body_prim is None:
                continue
            self._set_pose(body_prim, relative_positions[body_index], np.asarray(robot_body_quats_w[body_index], dtype=np.float32))
        self._update_surface_mesh(robot_root_position_w)
        self._capture_frame(timestep)

    def _update_surface_mesh(self, robot_root_position_w) -> None:
        points, face_counts, face_indices, display_colors = build_elevation_surface_mesh_data(
            self.elevation_map_state,
            robot_position_w=robot_root_position_w,
            preview_origin=self.PREVIEW_ORIGIN,
        )
        self._surface_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32, copy=False)))
        self._surface_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(face_counts.astype(np.int32, copy=False)))
        self._surface_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(face_indices.astype(np.int32, copy=False)))
        if display_colors.size == 0:
            self._surface_color_primvar.Set(Vt.Vec3fArray())
        else:
            self._surface_color_primvar.Set(Vt.Vec3fArray.FromNumpy(display_colors.astype(np.float32, copy=False)))

    def _capture_frame(self, timestep: int) -> None:
        self._service_pending_capture(force_flush=False)
        if self.output_path is None or self._capture_frames_dir is None:
            return
        if self._pending_capture is not None:
            return
        if timestep % self.video_frame_stride != 0:
            return
        frame_path = self._capture_frames_dir / f"frame_{self._next_capture_index:05d}.png"
        self._next_capture_index += 1
        capture = _FrameFileCapture(frame_path)
        self.viewport_api.schedule_capture(capture)
        self._pending_capture = capture

    def _service_pending_capture(self, *, force_flush: bool) -> None:
        if self._pending_capture is None:
            return
        app = omni.kit.app.get_app()
        if force_flush:
            deadline = time.time() + 5.0
            while self._pending_capture is not None and time.time() < deadline:
                frame_ready = (
                    self._pending_capture.completed
                    and self._pending_capture.frame_path.exists()
                    and self._pending_capture.frame_path.stat().st_size > 0
                )
                if frame_ready:
                    break
                app.update()
                time.sleep(0.01)
        if self._pending_capture is None or not self._pending_capture.completed:
            return
        if not self._pending_capture.frame_path.exists() or self._pending_capture.frame_path.stat().st_size <= 0:
            if force_flush:
                carb.log_warn(
                    f"Elevation viewport capture completed but frame file is missing: {self._pending_capture.frame_path}"
                )
                self._pending_capture = None
            return
        self._capture_paths.append(self._pending_capture.frame_path)
        self._pending_capture = None

    def close(self) -> None:
        self._service_pending_capture(force_flush=True)
        self._finalize_capture_video()
        if self.viewport_window is not None:
            self.viewport_window.destroy()
            self.viewport_window = None
        self._clear_preview_prims()

    def _finalize_capture_video(self) -> None:
        if self.output_path is None or self._capture_frames_dir is None or not self._capture_paths:
            return
        existing_paths = [path for path in self._capture_paths if path.exists() and path.stat().st_size > 0]
        if not existing_paths:
            carb.log_warn("Elevation viewport capture produced no completed frame files.")
            return
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(output_path, fps=self.video_fps) as writer:
            for frame_path in existing_paths:
                writer.append_data(imageio.imread(frame_path))


def build_elevation_viewport_output_path(primary_video_output_path: str | None) -> str | None:
    if primary_video_output_path is None:
        return None
    output_path = Path(primary_video_output_path)
    return str(output_path.with_name(f"{output_path.stem}-elevation-viewport{output_path.suffix}"))
