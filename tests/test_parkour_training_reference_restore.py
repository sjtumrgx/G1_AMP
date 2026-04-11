from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "instinct_rl" / "train.py"
PARKOUR_ENV_CFG = (
    REPO_ROOT / "source" / "instinctlab" / "instinctlab" / "tasks" / "parkour" / "config" / "parkour_env_cfg.py"
)
PARKOUR_TARGET_CFG = (
    REPO_ROOT
    / "source"
    / "instinctlab"
    / "instinctlab"
    / "tasks"
    / "parkour"
    / "config"
    / "g1"
    / "g1_parkour_target_amp_cfg.py"
)


def test_train_script_uses_reference_seed_flow_for_parkour_training():
    source = TRAIN_SCRIPT.read_text()

    assert "terrain_generator.seed = agent_cfg.seed" not in source
    assert "terrain_generator.seed = env_cfg.seed" not in source


def test_parkour_env_cfg_uses_reference_terrain_generator_stack():
    source = PARKOUR_ENV_CFG.read_text()

    assert "TerrainGeneratorCfg" in source
    assert "FiledTerrainGeneratorCfg" not in source
    assert "DepthArtifactNoiseCfg" in source
    assert "RandomGaussianNoiseCfg" in source
    assert "RangeBasedGaussianNoiseCfg" in source
    assert "robot_reference: ArticulationCfg = None" not in source


def test_parkour_target_cfg_matches_reference_symmetric_training_setup():
    source = PARKOUR_TARGET_CFG.read_text()

    assert 'path = os.path.expanduser("~/Datasets")' in source
    assert 'filtered_motion_selection_filepath = os.path.expanduser("~/Datasets/parkour_motion_without_run.yaml")' in source
    assert 'mp_split_method="Even"' in source
    assert 'mp_split_method="None"' not in source
    assert "G1_REFERENCE_CFG" not in source
    assert "play_visualization" not in source
    assert "self.scene.robot_reference" not in source
