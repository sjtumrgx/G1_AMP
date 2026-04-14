import importlib.util
from pathlib import Path
import sys


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


def test_compile_g1_viewer_model_adds_requested_surface_mesh_and_texture():
    module = load_module()
    model, mesh_ids = module._compile_g1_viewer_model(
        model_path=Path(module.DEFAULT_MODEL_PATH),
        surface_rows=4,
        surface_cols=4,
        surface_resolution_m=0.25,
    )

    assert model.ngeom >= len(module.SURFACE_BIN_RGBA)
    assert len(mesh_ids) == len(module.SURFACE_BIN_RGBA)
    assert all(mesh_id >= 0 for mesh_id in mesh_ids)
