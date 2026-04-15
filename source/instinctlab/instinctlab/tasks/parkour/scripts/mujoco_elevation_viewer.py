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
import cv2
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
SURFACE_PACKET_SCHEMA_VERSION = 2
SURFACE_MESH_NAME = "elevation_surface_mesh"
SURFACE_GEOM_NAME = "elevation_surface"
SURFACE_TEXTURE_NAME = "elevation_surface_texture"
SURFACE_MATERIAL_NAME = "elevation_surface_material"
SURFACE_BACKGROUND_TEXTURE_NAME = "elevation_background_texture"
SURFACE_BACKGROUND_MATERIAL_NAME = "elevation_background_material"
SURFACE_BACKGROUND_PLANE_NAME = "elevation_background_plane"
SURFACE_GRID_TEXELS_PER_CELL = 16
SURFACE_GRID_LINE_WIDTH_TEXELS = 1
INVALID_SURFACE_RGB = np.array([255, 255, 255], dtype=np.uint8)
GRID_LINE_RGB = np.array([232, 236, 242], dtype=np.uint8)
FRAME_RGBA = np.array([0.96, 0.97, 0.99, 1.0], dtype=np.float32)
FRAME_EDGE_ORDER = ("south", "north", "west", "east")
HIDDEN_HAND_BODY_NAMES = {
    "left_zero_link",
    "left_one_link",
    "left_two_link",
    "left_three_link",
    "left_four_link",
    "left_five_link",
    "left_six_link",
    "right_zero_link",
    "right_one_link",
    "right_two_link",
    "right_three_link",
    "right_four_link",
    "right_five_link",
    "right_six_link",
}


@dataclass(frozen=True)
class SurfaceViewerHandles:
    surface_mesh_id: int
    surface_geom_id: int
    surface_texture_id: int
    surface_material_id: int
    background_plane_geom_id: int | None
    background_texture_id: int | None
    background_material_id: int | None
    frame_geom_ids: dict[str, int]


@dataclass(frozen=True)
class DecodedSurfacePacket:
    schema_version: int
    surface_vertices: np.ndarray
    surface_scalar: np.ndarray
    surface_confidence: np.ndarray
    surface_valid: np.ndarray


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


def _load_editable_model_spec(model_path: Path) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(model_path))
    if model_path.suffix.lower() != ".xml":
        spec = mujoco.MjSpec.from_string(spec.to_xml())
    spec.assets.update(_load_mujoco_assets(model_path))
    return spec


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


def _surface_texture_dimensions(
    rows: int,
    cols: int,
    *,
    texels_per_cell: int = SURFACE_GRID_TEXELS_PER_CELL,
) -> tuple[int, int]:
    return max(1, int(cols) * int(texels_per_cell)), max(1, int(rows) * int(texels_per_cell))


def _build_region_frame_specs(*, rows: int, cols: int, resolution_m: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    half_extent_x = cols * float(resolution_m) * 0.5
    half_extent_y = rows * float(resolution_m) * 0.5
    half_thickness = max(float(resolution_m) * 0.04, 0.008)
    half_height = max(float(resolution_m) * 0.02, 0.006)
    frame_z = float(PREVIEW_ORIGIN[2]) + half_height
    return {
        "south": (
            np.array([float(PREVIEW_ORIGIN[0]), -half_extent_y, frame_z], dtype=np.float64),
            np.array([half_extent_x, half_thickness, half_height], dtype=np.float64),
        ),
        "north": (
            np.array([float(PREVIEW_ORIGIN[0]), half_extent_y, frame_z], dtype=np.float64),
            np.array([half_extent_x, half_thickness, half_height], dtype=np.float64),
        ),
        "west": (
            np.array([-half_extent_x, float(PREVIEW_ORIGIN[1]), frame_z], dtype=np.float64),
            np.array([half_thickness, half_extent_y, half_height], dtype=np.float64),
        ),
        "east": (
            np.array([half_extent_x, float(PREVIEW_ORIGIN[1]), frame_z], dtype=np.float64),
            np.array([half_thickness, half_extent_y, half_height], dtype=np.float64),
        ),
    }


def _normalize_surface_scalar(surface_scalar: np.ndarray, surface_valid: np.ndarray) -> np.ndarray:
    scalar = np.asarray(surface_scalar, dtype=np.float32)
    valid = np.asarray(surface_valid, dtype=bool)
    normalized = np.zeros_like(scalar, dtype=np.float32)
    if not np.any(valid):
        return normalized
    valid_values = scalar[valid]
    min_value = float(valid_values.min())
    max_value = float(valid_values.max())
    if max_value - min_value <= 1.0e-6:
        normalized[valid] = 0.55
        return normalized
    normalized[valid] = (valid_values - min_value) / (max_value - min_value)
    return np.clip(normalized, 0.0, 1.0)


def _sample_surface_colormap(normalized_scalar: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(normalized_scalar, dtype=np.float32), 0.0, 1.0)
    colormap_input = np.clip(np.round(values * 255.0), 0.0, 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(colormap_input, cv2.COLORMAP_TURBO)
    return colored_bgr[..., ::-1].astype(np.uint8)


def _build_surface_texture_rgb(
    *,
    surface_scalar: np.ndarray,
    surface_valid: np.ndarray,
    surface_confidence: np.ndarray | None = None,
    texels_per_cell: int = SURFACE_GRID_TEXELS_PER_CELL,
    grid_line_width: int = SURFACE_GRID_LINE_WIDTH_TEXELS,
) -> np.ndarray:
    scalar = np.asarray(surface_scalar, dtype=np.float32)
    valid = np.asarray(surface_valid, dtype=bool)
    if scalar.shape != valid.shape:
        raise ValueError(f"Expected scalar and valid masks to share a shape, got {scalar.shape} and {valid.shape}.")
    rows, cols = valid.shape
    normalized = _normalize_surface_scalar(scalar, valid)
    texture_height = rows * int(texels_per_cell)
    texture_width = cols * int(texels_per_cell)
    scalar_texture = cv2.resize(normalized, (texture_width, texture_height), interpolation=cv2.INTER_CUBIC)
    texture_rgb = np.broadcast_to(INVALID_SURFACE_RGB, (texture_height, texture_width, 3)).copy()
    valid_texture = cv2.resize(valid.astype(np.uint8), (texture_width, texture_height), interpolation=cv2.INTER_NEAREST) > 0
    if np.any(valid_texture):
        texture_rgb[valid_texture] = _sample_surface_colormap(scalar_texture)[valid_texture]
    if surface_confidence is not None:
        confidence = np.clip(np.asarray(surface_confidence, dtype=np.float32), 0.0, 1.0)
        if confidence.shape != valid.shape:
            raise ValueError(f"Expected confidence shape {valid.shape}, got {confidence.shape}.")
        confidence_texture = cv2.resize(confidence, (texture_width, texture_height), interpolation=cv2.INTER_LINEAR)
        brightness = 0.82 + 0.18 * confidence_texture[..., None]
        shaded = np.clip(np.round(texture_rgb.astype(np.float32) * brightness), 0.0, 255.0).astype(np.uint8)
        texture_rgb[valid_texture] = shaded[valid_texture]

    line_width = max(1, int(grid_line_width))
    for row in range(rows + 1):
        y_start = min(texture_height, row * texels_per_cell)
        y_end = min(texture_height, y_start + line_width)
        if y_start < y_end:
            texture_rgb[y_start:y_end, :, :] = GRID_LINE_RGB
    for col in range(cols + 1):
        x_start = min(texture_width, col * texels_per_cell)
        x_end = min(texture_width, x_start + line_width)
        if x_start < x_end:
            texture_rgb[:, x_start:x_end, :] = GRID_LINE_RGB

    if texture_height > 0 and texture_width > 0:
        texture_rgb[-line_width:, :, :] = GRID_LINE_RGB
        texture_rgb[:, -line_width:, :] = GRID_LINE_RGB
    return texture_rgb


def _build_background_grid_texture_rgb(
    *,
    rows: int,
    cols: int,
    texels_per_cell: int = SURFACE_GRID_TEXELS_PER_CELL,
    grid_line_width: int = SURFACE_GRID_LINE_WIDTH_TEXELS,
) -> np.ndarray:
    texture_height = rows * int(texels_per_cell)
    texture_width = cols * int(texels_per_cell)
    texture_rgb = np.broadcast_to(INVALID_SURFACE_RGB, (texture_height, texture_width, 3)).copy()
    line_width = max(1, int(grid_line_width))
    for row in range(rows + 1):
        y_start = min(texture_height, row * texels_per_cell)
        y_end = min(texture_height, y_start + line_width)
        if y_start < y_end:
            texture_rgb[y_start:y_end, :, :] = GRID_LINE_RGB
    for col in range(cols + 1):
        x_start = min(texture_width, col * texels_per_cell)
        x_end = min(texture_width, x_start + line_width)
        if x_start < x_end:
            texture_rgb[:, x_start:x_end, :] = GRID_LINE_RGB
    if texture_height > 0 and texture_width > 0:
        texture_rgb[-line_width:, :, :] = GRID_LINE_RGB
        texture_rgb[:, -line_width:, :] = GRID_LINE_RGB
    return texture_rgb


def _build_surface_face_indices(surface_valid: np.ndarray, *, render_all_cells: bool = True) -> tuple[np.ndarray, int]:
    valid = np.asarray(surface_valid, dtype=bool)
    rows, cols = valid.shape
    face_indices = np.zeros((rows * cols * 2, 3), dtype=np.int32)
    active_face_count = 0
    face_slot = 0
    for cell_row in range(rows):
        for cell_col in range(cols):
            top_left = cell_row * (cols + 1) + cell_col
            top_right = top_left + 1
            bottom_left = (cell_row + 1) * (cols + 1) + cell_col
            bottom_right = bottom_left + 1
            if render_all_cells or valid[cell_row, cell_col]:
                face_indices[face_slot] = np.array([top_left, top_right, bottom_left], dtype=np.int32)
                face_indices[face_slot + 1] = np.array([top_right, bottom_right, bottom_left], dtype=np.int32)
                active_face_count += 2
            face_slot += 2
    return face_indices, active_face_count


def _derive_surface_scalar_from_vertices(*, surface_vertices: np.ndarray, rows: int, cols: int) -> np.ndarray:
    expected_vertex_count = (rows + 1) * (cols + 1)
    vertices = np.asarray(surface_vertices, dtype=np.float32).reshape(-1, 3)
    if vertices.shape[0] < expected_vertex_count:
        padded_vertices = np.zeros((expected_vertex_count, 3), dtype=np.float32)
        padded_vertices[: vertices.shape[0]] = vertices
        vertices = padded_vertices
    vertex_heights = vertices[:expected_vertex_count, 2].reshape(rows + 1, cols + 1)
    return 0.25 * (
        vertex_heights[:-1, :-1]
        + vertex_heights[:-1, 1:]
        + vertex_heights[1:, :-1]
        + vertex_heights[1:, 1:]
    )


def _decode_surface_packet(packet: dict, *, rows: int, cols: int) -> DecodedSurfacePacket:
    surface_vertices = np.asarray(packet.get("surface_vertices", []), dtype=np.float32).reshape(-1, 3)
    surface_valid = np.asarray(packet.get("surface_valid", []), dtype=bool).reshape(rows, cols)

    confidence_raw = packet.get("surface_confidence", [])
    if confidence_raw:
        surface_confidence = np.asarray(confidence_raw, dtype=np.float32).reshape(rows, cols)
    else:
        surface_confidence = np.ones((rows, cols), dtype=np.float32)

    scalar_raw = packet.get("surface_scalar")
    if scalar_raw is None:
        surface_scalar = _derive_surface_scalar_from_vertices(surface_vertices=surface_vertices, rows=rows, cols=cols)
    else:
        surface_scalar = np.asarray(scalar_raw, dtype=np.float32).reshape(rows, cols)

    return DecodedSurfacePacket(
        schema_version=int(packet.get("schema_version", 1)),
        surface_vertices=surface_vertices,
        surface_scalar=surface_scalar,
        surface_confidence=surface_confidence,
        surface_valid=surface_valid,
    )


def _set_texture_rgb_data(*, model: mujoco.MjModel, texture_id: int, texture_rgb: np.ndarray) -> None:
    rgb = np.asarray(texture_rgb, dtype=np.uint8)
    texture_width = int(model.tex_width[texture_id])
    texture_height = int(model.tex_height[texture_id])
    nchannel = int(model.tex_nchannel[texture_id])
    if rgb.shape != (texture_height, texture_width, 3):
        raise ValueError(
            f"Expected texture RGB image with shape {(texture_height, texture_width, 3)}, got {rgb.shape}."
        )
    texture_adr = int(model.tex_adr[texture_id])
    flipped_rgb = np.ascontiguousarray(rgb[::-1, :, :])
    if nchannel == 3:
        model.tex_data[texture_adr : texture_adr + flipped_rgb.size] = flipped_rgb.reshape(-1)
        return
    if nchannel == 4:
        alpha = np.full((texture_height, texture_width, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([flipped_rgb, alpha], axis=-1)
        model.tex_data[texture_adr : texture_adr + rgba.size] = rgba.reshape(-1)
        return
    raise ValueError(f"Unsupported texture channel count {nchannel}; expected 3 or 4 channels.")


def _sync_viewer_assets(viewer, *, mesh_id: int | None = None, texture_ids: tuple[int, ...] = ()) -> None:
    if mesh_id is not None:
        viewer.update_mesh(mesh_id)
    for texture_id in texture_ids:
        viewer.update_texture(texture_id)


def _hide_extra_hand_geoms(model: mujoco.MjModel) -> None:
    hidden_body_ids = {
        body_id
        for body_id in range(model.nbody)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)) in HIDDEN_HAND_BODY_NAMES
    }
    if not hidden_body_ids:
        return
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) in hidden_body_ids:
            model.geom_rgba[geom_id, 3] = 0.0


def _compile_g1_viewer_model(
    *,
    model_path: Path,
    surface_rows: int,
    surface_cols: int,
    surface_resolution_m: float,
    include_background_plane: bool = False,
) -> tuple[mujoco.MjModel, SurfaceViewerHandles]:
    spec = _load_editable_model_spec(model_path)
    spec.assets["elevation_surface.obj"] = _build_surface_mesh_obj(
        rows=surface_rows,
        cols=surface_cols,
        resolution_m=surface_resolution_m,
    )

    skybox_texture = spec.add_texture()
    skybox_texture.name = "viewer_white_skybox"
    skybox_texture.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    skybox_texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
    skybox_texture.width = 64
    skybox_texture.height = 64
    skybox_texture.nchannel = 3
    skybox_texture.rgb1 = [1.0, 1.0, 1.0]
    skybox_texture.rgb2 = [1.0, 1.0, 1.0]

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

    texture_width, texture_height = _surface_texture_dimensions(surface_rows, surface_cols)
    texture = spec.add_texture()
    texture.name = SURFACE_TEXTURE_NAME
    texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
    texture.width = texture_width
    texture.height = texture_height
    texture.nchannel = 3
    texture.rgb1 = [float(INVALID_SURFACE_RGB[0]) / 255.0, float(INVALID_SURFACE_RGB[1]) / 255.0, float(INVALID_SURFACE_RGB[2]) / 255.0]
    texture.rgb2 = texture.rgb1

    material = spec.add_material()
    material.name = SURFACE_MATERIAL_NAME
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = SURFACE_TEXTURE_NAME
    material.texuniform = False
    material.reflectance = 0.12
    material.specular = 0.08
    material.shininess = 0.12
    material.roughness = 0.76
    material.rgba = [1.0, 1.0, 1.0, 1.0]

    if include_background_plane:
        background_texture = spec.add_texture()
        background_texture.name = SURFACE_BACKGROUND_TEXTURE_NAME
        background_texture.type = mujoco.mjtTexture.mjTEXTURE_2D
        background_texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
        background_texture.width = texture_width
        background_texture.height = texture_height
        background_texture.nchannel = 3
        background_texture.rgb1 = [1.0, 1.0, 1.0]
        background_texture.rgb2 = [1.0, 1.0, 1.0]

        background_material = spec.add_material()
        background_material.name = SURFACE_BACKGROUND_MATERIAL_NAME
        background_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = SURFACE_BACKGROUND_TEXTURE_NAME
        background_material.texuniform = False
        background_material.reflectance = 0.02
        background_material.specular = 0.01
        background_material.shininess = 0.02
        background_material.roughness = 0.95
        background_material.rgba = [1.0, 1.0, 1.0, 1.0]

    mesh = spec.add_mesh(name=SURFACE_MESH_NAME)
    mesh.file = "elevation_surface.obj"
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL

    surface_geom = spec.worldbody.add_geom()
    surface_geom.name = SURFACE_GEOM_NAME
    surface_geom.type = mujoco.mjtGeom.mjGEOM_MESH
    surface_geom.meshname = SURFACE_MESH_NAME
    surface_geom.material = SURFACE_MATERIAL_NAME
    surface_geom.rgba = [1.0, 1.0, 1.0, 1.0]
    surface_geom.conaffinity = 0
    surface_geom.contype = 0
    surface_geom.group = 1

    if include_background_plane:
        background_plane = spec.worldbody.add_geom()
        background_plane.name = SURFACE_BACKGROUND_PLANE_NAME
        background_plane.type = mujoco.mjtGeom.mjGEOM_BOX
        background_plane.size = [
            surface_cols * float(surface_resolution_m) * 0.5,
            surface_rows * float(surface_resolution_m) * 0.5,
            max(surface_resolution_m * 0.03, 0.01),
        ]
        background_plane.pos = [0.0, 0.0, float(PREVIEW_ORIGIN[2]) - max(surface_resolution_m * 0.08, 0.04)]
        background_plane.material = SURFACE_BACKGROUND_MATERIAL_NAME
        background_plane.rgba = [1.0, 1.0, 1.0, 1.0]
        background_plane.conaffinity = 0
        background_plane.contype = 0
        background_plane.group = 1

    for edge_name, (frame_pos, frame_size) in _build_region_frame_specs(
        rows=surface_rows,
        cols=surface_cols,
        resolution_m=surface_resolution_m,
    ).items():
        frame_geom = spec.worldbody.add_geom()
        frame_geom.name = f"elevation_map_region_frame_{edge_name}"
        frame_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        frame_geom.size = frame_size.tolist()
        frame_geom.pos = frame_pos.tolist()
        frame_geom.rgba = FRAME_RGBA.tolist()
        frame_geom.conaffinity = 0
        frame_geom.contype = 0
        frame_geom.group = 1

    model = spec.compile()
    frame_geom_ids = {
        edge_name: int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"elevation_map_region_frame_{edge_name}")
        )
        for edge_name in FRAME_EDGE_ORDER
    }
    return model, SurfaceViewerHandles(
        surface_mesh_id=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, SURFACE_MESH_NAME)),
        surface_geom_id=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, SURFACE_GEOM_NAME)),
        surface_texture_id=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, SURFACE_TEXTURE_NAME)),
        surface_material_id=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, SURFACE_MATERIAL_NAME)),
        background_plane_geom_id=(
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, SURFACE_BACKGROUND_PLANE_NAME))
            if include_background_plane
            else None
        ),
        background_texture_id=(
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, SURFACE_BACKGROUND_TEXTURE_NAME))
            if include_background_plane
            else None
        ),
        background_material_id=(
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, SURFACE_BACKGROUND_MATERIAL_NAME))
            if include_background_plane
            else None
        ),
        frame_geom_ids=frame_geom_ids,
    )


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
    show_background_plane: bool = False


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
        command.extend(["--show-background-plane" if self.config.show_background_plane else "--hide-background-plane"])
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
        surface_scalar=None,
    ) -> None:
        if not self.is_running or self._process is None or self._process.stdin is None:
            return
        packet = {
            "type": "state",
            "schema_version": SURFACE_PACKET_SCHEMA_VERSION,
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
        if surface_scalar is not None:
            packet["surface_scalar"] = np.asarray(surface_scalar, dtype=np.float32).reshape(-1).tolist()
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
        show_background_plane: bool,
    ):
        self.surface_rows = int(surface_rows)
        self.surface_cols = int(surface_cols)
        self.surface_resolution_m = float(surface_resolution_m)
        self.model, self.surface_handles = _compile_g1_viewer_model(
            model_path=model_path,
            surface_rows=self.surface_rows,
            surface_cols=self.surface_cols,
            surface_resolution_m=self.surface_resolution_m,
            include_background_plane=show_background_plane,
        )
        self.surface_mesh_id = int(self.surface_handles.surface_mesh_id)
        self.surface_geom_id = int(self.surface_handles.surface_geom_id)
        self.surface_texture_id = int(self.surface_handles.surface_texture_id)
        self.background_texture_id = (
            None if self.surface_handles.background_texture_id is None else int(self.surface_handles.background_texture_id)
        )
        self.background_plane_geom_id = (
            None if self.surface_handles.background_plane_geom_id is None else int(self.surface_handles.background_plane_geom_id)
        )
        self.surface_frame_geom_ids = dict(self.surface_handles.frame_geom_ids)
        _hide_extra_hand_geoms(self.model)
        self.model.mesh_pos[self.surface_mesh_id] = np.zeros((3,), dtype=np.float64)
        self.model.mesh_quat[self.surface_mesh_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.model.geom_pos[self.surface_geom_id] = np.zeros((3,), dtype=np.float64)
        self.model.geom_quat[self.surface_geom_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._frame_specs = _build_region_frame_specs(
            rows=self.surface_rows,
            cols=self.surface_cols,
            resolution_m=self.surface_resolution_m,
        )
        self._background_plane_half_height = max(self.surface_resolution_m * 0.03, 0.01)
        self.data = mujoco.MjData(self.model)
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), int(window_size[0]))
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), int(window_size[1]))
        self.renderer = mujoco.Renderer(self.model, height=int(window_size[1]), width=int(window_size[0]))
        if self.background_texture_id is not None:
            _set_texture_rgb_data(
                model=self.model,
                texture_id=self.background_texture_id,
                texture_rgb=_build_background_grid_texture_rgb(
                    rows=self.surface_rows,
                    cols=self.surface_cols,
                ),
            )
            mujoco.mjr_uploadTexture(self.model, self.renderer._mjr_context, self.background_texture_id)
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
        self._mesh_vertex_count = int(self.model.mesh_vertnum[self.surface_mesh_id])
        self._mesh_face_count = int(self.model.mesh_facenum[self.surface_mesh_id])
        self._mesh_vert_adr = int(self.model.mesh_vertadr[self.surface_mesh_id])
        self._mesh_face_adr = int(self.model.mesh_faceadr[self.surface_mesh_id])
        self._texture_width = int(self.model.tex_width[self.surface_texture_id])
        self._texture_height = int(self.model.tex_height[self.surface_texture_id])
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

    def _update_region_helper_geoms(self, *, surface_scalar: np.ndarray, surface_valid: np.ndarray) -> None:
        if np.any(surface_valid):
            frame_z = float(np.min(surface_scalar[surface_valid])) + max(self.surface_resolution_m * 0.04, 0.01)
        else:
            frame_z = float(PREVIEW_ORIGIN[2]) + max(self.surface_resolution_m * 0.04, 0.01)
        if self.background_plane_geom_id is not None:
            self.model.geom_pos[self.background_plane_geom_id] = np.array(
                [
                    0.0,
                    0.0,
                    frame_z - (self._background_plane_half_height + max(self.surface_resolution_m * 0.02, 0.01)),
                ],
                dtype=np.float64,
            )
        for edge_name, geom_id in self.surface_frame_geom_ids.items():
            frame_pos, _ = self._frame_specs[edge_name]
            self.model.geom_pos[geom_id] = np.array([frame_pos[0], frame_pos[1], frame_z], dtype=np.float64)

    def _apply_packet(self, packet: dict) -> tuple[int, int]:
        self.data.qpos[:3] = PREVIEW_ORIGIN
        self.data.qpos[3:7] = np.asarray(packet["root_quaternion_wxyz"], dtype=np.float64).reshape(4)
        self.data.qvel[:] = 0.0
        for joint_name, joint_value in packet.get("joint_positions_by_name", {}).items():
            qpos_addr = self._joint_qpos_addr_by_name.get(joint_name)
            if qpos_addr is None:
                continue
            self.data.qpos[qpos_addr] = float(joint_value)

        decoded_packet = _decode_surface_packet(packet, rows=self.surface_rows, cols=self.surface_cols)
        surface_vertices = decoded_packet.surface_vertices
        if surface_vertices.shape[0] < self._mesh_vertex_count:
            padded_vertices = np.zeros((self._mesh_vertex_count, 3), dtype=np.float32)
            padded_vertices[: surface_vertices.shape[0]] = surface_vertices
            surface_vertices = padded_vertices
        self.model.mesh_vert[self._mesh_vert_adr : self._mesh_vert_adr + self._mesh_vertex_count] = surface_vertices[
            : self._mesh_vertex_count
        ]

        face_indices, active_face_count = _build_surface_face_indices(
            decoded_packet.surface_valid,
            render_all_cells=False,
        )
        self.model.mesh_face[self._mesh_face_adr : self._mesh_face_adr + self._mesh_face_count] = face_indices

        surface_texture_rgb = _build_surface_texture_rgb(
            surface_scalar=decoded_packet.surface_scalar,
            surface_valid=decoded_packet.surface_valid,
            surface_confidence=decoded_packet.surface_confidence,
        )
        if surface_texture_rgb.shape[:2] != (self._texture_height, self._texture_width):
            raise ValueError(
                f"Expected texture image shape {(self._texture_height, self._texture_width)}, got {surface_texture_rgb.shape[:2]}."
            )
        _set_texture_rgb_data(model=self.model, texture_id=self.surface_texture_id, texture_rgb=surface_texture_rgb)
        self._update_region_helper_geoms(
            surface_scalar=decoded_packet.surface_scalar,
            surface_valid=decoded_packet.surface_valid,
        )

        mujoco.mj_forward(self.model, self.data)
        mujoco.mjr_uploadMesh(self.model, self.renderer._mjr_context, self.surface_mesh_id)
        mujoco.mjr_uploadTexture(self.model, self.renderer._mjr_context, self.surface_texture_id)
        self._latest_active_faces = active_face_count
        self._latest_active_tiles = int(np.count_nonzero(decoded_packet.surface_valid))
        if not self._logged_first_update:
            self._logged_first_update = True
            print(
                "[INFO] MuJoCo elevation viewer received first state update:"
                f" schema={decoded_packet.schema_version}, vertices={self._mesh_vertex_count},"
                f" faces={self._mesh_face_count}, active_faces={active_face_count},"
                f" active_tiles={self._latest_active_tiles}, joints={len(packet.get('joint_positions_by_name', {}))}",
                flush=True,
            )
        return self._mesh_vertex_count, self._mesh_face_count

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
            if self.background_texture_id is not None:
                _sync_viewer_assets(viewer, texture_ids=(self.background_texture_id,))
            print("[INFO] MuJoCo elevation viewer window opened.", flush=True)
            while viewer.is_running():
                packet, version, reader_closed = self._snapshot_packet()
                if version != last_version and packet is not None:
                    with viewer.lock():
                        self._apply_packet(packet)
                    _sync_viewer_assets(viewer, mesh_id=self.surface_mesh_id, texture_ids=(self.surface_texture_id,))
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
    parser.add_argument("--show-background-plane", action="store_true", dest="show_background_plane", default=False)
    parser.add_argument("--hide-background-plane", action="store_false", dest="show_background_plane")
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
        show_background_plane=bool(args.show_background_plane),
    )
    runtime.run()


if __name__ == "__main__":
    main()
