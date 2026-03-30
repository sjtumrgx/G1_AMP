# pyright: reportAttributeAccessIssue=false

import importlib.util
from pathlib import Path

import mujoco
import xml.etree.ElementTree as ET


RUNTIME_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "instinct_rl" / "parkour_mujoco_runtime.py"


def load_runtime_module():
    spec = importlib.util.spec_from_file_location("parkour_mujoco_runtime", RUNTIME_MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_mujoco_model_xml_promotes_urdf_to_floating_model(tmp_path):
    module = load_runtime_module()
    urdf_path = Path("source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf")

    generated_xml = module.prepare_mujoco_model_xml(urdf_path, output_dir=tmp_path)
    model = mujoco.MjModel.from_xml_path(str(generated_xml))

    assert generated_xml.is_file()
    assert model.nq == 36
    assert model.nv == 35
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_freejoint") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera") >= 0


def test_prepare_mujoco_model_xml_preserves_visual_mesh_assets_and_ground_material(tmp_path):
    module = load_runtime_module()
    urdf_path = Path("source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf")

    generated_xml = module.prepare_mujoco_model_xml(urdf_path, output_dir=tmp_path)
    root = ET.parse(generated_xml).getroot()
    asset = root.find("asset")
    assert asset is not None

    mesh_names = [mesh.attrib.get("name") for mesh in asset.findall("mesh")]
    material_names = [material.attrib.get("name") for material in asset.findall("material")]
    floor = root.find(".//geom[@name='floor']")

    assert "pelvis" in mesh_names
    assert "torso_link_rev_1_0" in mesh_names
    assert "groundplane" in material_names
    assert floor is not None
    assert floor.attrib.get("material") == "groundplane"


def test_prepare_mujoco_model_xml_adds_light_and_headlight_visuals(tmp_path):
    module = load_runtime_module()
    urdf_path = Path("source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf")

    generated_xml = module.prepare_mujoco_model_xml(urdf_path, output_dir=tmp_path)
    root = ET.parse(generated_xml).getroot()

    visual = root.find("visual")
    light = root.find(".//light[@name='scene_light']")

    assert visual is not None
    assert visual.find("headlight") is not None
    assert light is not None
    assert light.attrib.get("directional") == "true"


def test_settle_root_to_floor_brings_lowest_robot_geom_close_to_ground(tmp_path):
    module = load_runtime_module()
    urdf_path = Path("source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf")

    generated_xml = module.prepare_mujoco_model_xml(urdf_path, output_dir=tmp_path)
    model = mujoco.MjModel.from_xml_path(str(generated_xml))
    data = mujoco.MjData(model)

    module.settle_root_to_floor(model, data, clearance=0.005)

    lowest = min(
        float(data.geom_xpos[i][2])
        for i in range(model.ngeom)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) != "floor"
    )
    assert lowest <= 0.01
