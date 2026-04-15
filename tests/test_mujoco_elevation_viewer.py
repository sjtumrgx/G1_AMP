import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "instinctlab"
    / "instinctlab"
    / "tasks"
    / "parkour"
    / "scripts"
)
MODULE_PATH = SCRIPTS_DIR / "mujoco_elevation_viewer.py"


sys.path.append(str(SCRIPTS_DIR))


def load_module():
    spec = importlib.util.spec_from_file_location("parkour_mujoco_elevation_viewer", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_mujoco_elevation_capture_related_paths_suffix_primary_video():
    module = load_module()

    capture_path = module.build_mujoco_elevation_capture_path("/tmp/play.mp4")
    screenshot_path = module.build_mujoco_elevation_screenshot_path("/tmp/play.mp4")
    log_path = module.build_mujoco_elevation_log_path("/tmp/play.mp4")
    frame_dir = module.build_mujoco_elevation_frame_dir("/tmp/play.mp4")

    assert capture_path == "/tmp/play-mujoco-elevation-viewer.mp4"
    assert screenshot_path == "/tmp/play-mujoco-elevation-viewer.png"
    assert log_path == "/tmp/play-mujoco-elevation-viewer.log"
    assert frame_dir == "/tmp/play-mujoco-elevation-viewer_frames"


def test_build_surface_mesh_obj_contains_vertices_texcoords_and_faces():
    module = load_module()
    obj_bytes = module._build_surface_mesh_obj(rows=2, cols=3, resolution_m=0.5)
    lines = obj_bytes.decode("utf-8").splitlines()

    assert sum(line.startswith("v ") for line in lines) == (2 + 1) * (3 + 1)
    assert sum(line.startswith("vt ") for line in lines) == (2 + 1) * (3 + 1)
    assert sum(line.startswith("f ") for line in lines) == 2 * 3 * 2


def test_surface_texture_dimensions_scale_with_cells():
    module = load_module()
    width, height = module._surface_texture_dimensions(3, 4, texels_per_cell=6)

    assert width == 24
    assert height == 18


def test_build_surface_texture_rgb_creates_rich_texture_with_grid_lines():
    module = load_module()
    surface_scalar = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
    surface_valid = np.array([[True, True], [False, True]], dtype=bool)
    surface_confidence = np.array([[1.0, 0.8], [0.0, 0.6]], dtype=np.float32)

    texture_rgb = module._build_surface_texture_rgb(
        surface_scalar=surface_scalar,
        surface_valid=surface_valid,
        surface_confidence=surface_confidence,
        texels_per_cell=4,
        grid_line_width=1,
    )

    assert texture_rgb.shape == (8, 8, 3)
    assert texture_rgb.dtype == np.uint8
    assert np.array_equal(texture_rgb[0, 0], module.GRID_LINE_RGB)
    assert np.array_equal(texture_rgb[6, 2], module.INVALID_SURFACE_RGB)
    assert not np.array_equal(texture_rgb[2, 2], texture_rgb[2, 6])


def test_build_background_grid_texture_rgb_creates_white_background_and_visible_grid():
    module = load_module()
    texture_rgb = module._build_background_grid_texture_rgb(rows=2, cols=2, texels_per_cell=4, grid_line_width=1)

    assert texture_rgb.shape == (8, 8, 3)
    assert np.array_equal(texture_rgb[0, 0], module.GRID_LINE_RGB)
    assert np.array_equal(texture_rgb[2, 2], module.INVALID_SURFACE_RGB)


def test_decode_surface_packet_falls_back_to_vertex_derived_scalar_and_defaults_schema_version():
    module = load_module()
    rows = cols = 2
    vertex_heights = np.array(
        [
            [0.2, 0.4, 0.6],
            [0.4, 0.8, 1.0],
            [0.6, 1.0, 1.2],
        ],
        dtype=np.float32,
    )
    vertices = []
    for row in range(rows + 1):
        for col in range(cols + 1):
            vertices.append([float(col), float(row), float(vertex_heights[row, col])])
    packet = {
        "surface_vertices": vertices,
        "surface_valid": [True, False, True, True],
    }

    decoded = module._decode_surface_packet(packet, rows=rows, cols=cols)

    assert decoded.schema_version == 1
    assert decoded.surface_scalar.shape == (rows, cols)
    assert decoded.surface_confidence.shape == (rows, cols)
    assert decoded.surface_valid.shape == (rows, cols)
    assert np.isclose(decoded.surface_scalar[0, 0], 0.45)
    assert np.isclose(decoded.surface_scalar[1, 1], 1.0)


def test_build_surface_face_indices_respects_valid_mask():
    module = load_module()
    face_indices, active_face_count = module._build_surface_face_indices(
        np.array([[True, False], [True, True]], dtype=bool),
        render_all_cells=False,
    )

    assert face_indices.shape == (8, 3)
    assert active_face_count == 6
    assert np.array_equal(face_indices[2], np.zeros(3, dtype=np.int32))
    assert np.array_equal(face_indices[3], np.zeros(3, dtype=np.int32))


def test_build_surface_face_indices_can_render_full_white_background_plane_mesh():
    module = load_module()
    face_indices, active_face_count = module._build_surface_face_indices(
        np.array([[False, False], [False, False]], dtype=bool),
        render_all_cells=True,
    )

    assert face_indices.shape == (8, 3)
    assert active_face_count == 8
    assert np.any(face_indices)


def test_compile_g1_viewer_model_can_omit_background_plane():
    module = load_module()
    model, handles = module._compile_g1_viewer_model(
        model_path=Path(module.DEFAULT_MODEL_PATH),
        surface_rows=4,
        surface_cols=4,
        surface_resolution_m=0.25,
        include_background_plane=False,
    )

    assert handles.background_plane_geom_id is None
    assert handles.background_texture_id is None
    assert handles.background_material_id is None
    assert handles.surface_geom_id >= 0
    assert int(model.ngeom) >= 1


def test_compile_g1_viewer_model_supports_repo_urdf_source():
    module = load_module()
    urdf_path = (
        Path(module.DEFAULT_MODEL_PATH).resolve().parent
        / "urdf"
        / "g1_29dof_torsobase_popsicle.urdf"
    )
    model, handles = module._compile_g1_viewer_model(
        model_path=urdf_path,
        surface_rows=4,
        surface_cols=4,
        surface_resolution_m=0.25,
        include_background_plane=False,
    )

    assert handles.surface_mesh_id >= 0
    assert handles.surface_geom_id >= 0
    assert int(model.njnt) > 0


def test_hide_extra_hand_geoms_hides_default_g1_finger_geometry():
    module = load_module()
    model = module.mujoco.MjModel.from_xml_path(str(module.DEFAULT_MODEL_PATH))

    module._hide_extra_hand_geoms(model)

    hidden_body_ids = {
        body_id
        for body_id in range(model.nbody)
        if module.mujoco.mj_id2name(model, module.mujoco.mjtObj.mjOBJ_BODY, body_id) in module.HIDDEN_HAND_BODY_NAMES
    }
    hidden_geom_count = sum(
        1
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in hidden_body_ids and np.isclose(model.geom_rgba[geom_id, 3], 0.0)
    )

    assert hidden_body_ids
    assert hidden_geom_count > 0


def test_sync_viewer_assets_updates_mesh_then_requested_textures():
    module = load_module()
    calls: list[tuple[str, int]] = []

    class FakeViewer:
        def update_mesh(self, mesh_id: int) -> None:
            calls.append(("mesh", mesh_id))

        def update_texture(self, texture_id: int) -> None:
            calls.append(("texture", texture_id))

    module._sync_viewer_assets(FakeViewer(), mesh_id=7, texture_ids=(11, 13))

    assert calls == [("mesh", 7), ("texture", 11), ("texture", 13)]


def test_sync_viewer_assets_supports_texture_only_updates():
    module = load_module()
    calls: list[tuple[str, int]] = []

    class FakeViewer:
        def update_mesh(self, mesh_id: int) -> None:
            calls.append(("mesh", mesh_id))

        def update_texture(self, texture_id: int) -> None:
            calls.append(("texture", texture_id))

    module._sync_viewer_assets(FakeViewer(), texture_ids=(5,))

    assert calls == [("texture", 5)]


def test_compile_g1_viewer_model_adds_single_surface_mesh_texture_and_frame_geoms():
    module = load_module()
    model, handles = module._compile_g1_viewer_model(
        model_path=Path(module.DEFAULT_MODEL_PATH),
        surface_rows=4,
        surface_cols=4,
        surface_resolution_m=0.25,
        include_background_plane=True,
    )

    assert handles.surface_mesh_id >= 0
    assert handles.surface_geom_id >= 0
    assert handles.surface_texture_id >= 0
    assert handles.surface_material_id >= 0
    assert handles.background_plane_geom_id >= 0
    assert handles.background_texture_id >= 0
    assert handles.background_material_id >= 0
    assert set(handles.frame_geom_ids.keys()) == set(module.FRAME_EDGE_ORDER)
    assert all(geom_id >= 0 for geom_id in handles.frame_geom_ids.values())
    texture_role = int(module.mujoco.mjtTextureRole.mjTEXROLE_RGB)
    assert model.mat_texid[handles.surface_material_id, texture_role] == handles.surface_texture_id
    assert model.mat_texid[handles.background_material_id, texture_role] == handles.background_texture_id
    assert int(model.ntex) >= 3
