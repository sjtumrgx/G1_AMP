import argparse
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "instinct_rl" / "parkour_mujoco_runtime.py"


def load_module():
    spec = importlib.util.spec_from_file_location("parkour_mujoco_runtime", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_logged_run_paths_for_parkour_export():
    module = load_module()

    paths = module.resolve_logged_run_paths("20260327_163647")

    assert paths.log_dir.name == "20260327_163647"
    assert paths.export_dir == paths.log_dir / "exported"
    assert paths.actor_onnx == paths.export_dir / "actor.onnx"
    assert paths.depth_encoder_onnx == paths.export_dir / "0-depth_encoder.onnx"
    assert paths.agent_cfg == paths.log_dir / "params" / "agent.yaml"
    assert paths.env_cfg == paths.log_dir / "params" / "env.yaml"


def test_validate_run_artifacts_rejects_missing_actor(tmp_path):
    module = load_module()

    run_dir = tmp_path / "run"
    export_dir = run_dir / "exported"
    params_dir = run_dir / "params"
    export_dir.mkdir(parents=True)
    params_dir.mkdir(parents=True)
    (export_dir / "0-depth_encoder.onnx").write_bytes(b"encoder")
    (params_dir / "agent.yaml").write_text("policy: {}\n")
    (params_dir / "env.yaml").write_text("scene: {}\n")

    paths = module.LoggedRunPaths(
        repo_root=tmp_path,
        log_dir=run_dir,
        params_dir=params_dir,
        export_dir=export_dir,
        actor_onnx=export_dir / "actor.onnx",
        depth_encoder_onnx=export_dir / "0-depth_encoder.onnx",
        agent_cfg=params_dir / "agent.yaml",
        env_cfg=params_dir / "env.yaml",
    )

    with pytest.raises(FileNotFoundError, match="actor.onnx"):
        module.validate_run_artifacts(paths)


def test_resolve_default_model_path_uses_logged_robot_asset():
    module = load_module()
    paths = module.resolve_logged_run_paths("20260327_163647")

    default_model = module.resolve_default_model_path(paths)

    assert default_model == Path(
        "/home/eilab/instinctlab/source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"
    )


def test_build_cli_parser_accepts_depth_mode_choices():
    module = load_module()

    parser = module.build_cli_parser()
    args = parser.parse_args(["--load_run", "20260327_163647", "--depth-mode", "zeros"])

    assert isinstance(parser, argparse.ArgumentParser)
    assert args.load_run == "20260327_163647"
    assert args.depth_mode == "zeros"


def test_resolve_nominal_velocity_command_prefers_positive_forward_motion():
    module = load_module()
    env_cfg = module.load_logged_yaml("logs/instinct_rl/g1_parkour/20260327_163647/params/env.yaml")

    command = module.resolve_nominal_velocity_command(env_cfg)

    assert command.shape == (3,)
    assert command[0] > 0.0
    assert command[1] == 0.0
