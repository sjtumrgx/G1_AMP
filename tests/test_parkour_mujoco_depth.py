import importlib.util
from pathlib import Path

import numpy as np


RUNTIME_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "instinct_rl" / "parkour_mujoco_runtime.py"


def load_runtime_module():
    spec = importlib.util.spec_from_file_location("parkour_mujoco_runtime", RUNTIME_MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zero_depth_adapter_returns_expected_shape():
    module = load_runtime_module()
    adapter = module.ZeroDepthAdapter(history_length=8, frame_shape=(18, 32))

    depth = adapter.build()

    assert depth.shape == (8, 18, 32)
    assert np.count_nonzero(depth) == 0


def test_run_parkour_policy_accepts_zero_depth_and_returns_29_actions():
    module = load_runtime_module()
    run_paths = module.resolve_logged_run_paths("20260327_163647")
    encoder_session, actor_session = module.load_parkour_onnx_sessions(run_paths.export_dir)

    proprio = np.zeros(768, dtype=np.float32)
    depth = module.ZeroDepthAdapter(history_length=8, frame_shape=(18, 32)).build()
    action = module.run_parkour_policy(
        encoder_session=encoder_session,
        actor_session=actor_session,
        proprio_observation=proprio,
        depth_observation=depth,
    )

    assert action.shape == (29,)
    assert action.dtype == np.float32
