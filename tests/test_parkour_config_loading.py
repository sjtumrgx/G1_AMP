import importlib.util
import pickle
from pathlib import Path

import pytest
import yaml

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "instinctlab"
    / "instinctlab"
    / "tasks"
    / "parkour"
    / "scripts"
    / "config_loading.py"
)


def load_logged_config(log_dir, config_name):
    spec = importlib.util.spec_from_file_location("parkour_config_loading", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_logged_config(log_dir, config_name)


def test_load_logged_config_prefers_yaml(tmp_path):
    log_dir = tmp_path / "run"
    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True)
    (params_dir / "env.yaml").write_text("source: yaml\nvalue: 1\n")
    with (params_dir / "env.pkl").open("wb") as file:
        pickle.dump({"source": "pickle", "value": 2}, file)

    loaded = load_logged_config(log_dir, "env")

    assert loaded == {"source": "yaml", "value": 1}


def test_load_logged_config_falls_back_to_pickle(tmp_path):
    log_dir = tmp_path / "run"
    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True)
    expected = {"source": "pickle", "value": 2}
    with (params_dir / "env.pkl").open("wb") as file:
        pickle.dump(expected, file)

    loaded = load_logged_config(log_dir, "env")

    assert loaded == expected


def test_load_logged_config_raises_for_missing_file(tmp_path):
    log_dir = tmp_path / "run"
    (log_dir / "params").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="env.yaml|env.pkl"):
        load_logged_config(log_dir, "env")


def test_load_logged_config_supports_python_yaml_tags(tmp_path):
    log_dir = tmp_path / "run"
    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True)
    expected = {"segment": slice(0, 4, 2)}
    with (params_dir / "env.yaml").open("w") as file:
        yaml.dump(expected, file)

    loaded = load_logged_config(log_dir, "env")

    assert loaded == expected
