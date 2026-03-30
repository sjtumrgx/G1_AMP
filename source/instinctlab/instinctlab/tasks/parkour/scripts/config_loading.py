"""Helpers for loading logged parkour play configs."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import yaml


def load_logged_config(log_dir: str | Path, config_name: str) -> Any:
    """Load a config from ``log_dir/params``.

    Current runs store configs as YAML. Older runs may still have PKL files, so
    keep a pickle fallback for backwards compatibility.
    """

    params_dir = Path(log_dir) / "params"
    yaml_path = params_dir / f"{config_name}.yaml"
    if yaml_path.is_file():
        with yaml_path.open() as file:
            # Training artifacts may include Python-specific tags such as
            # ``slice`` that are not supported by ``yaml.full_load``.
            return yaml.unsafe_load(file)

    pickle_path = params_dir / f"{config_name}.pkl"
    if pickle_path.is_file():
        with pickle_path.open("rb") as file:
            return pickle.load(file)

    raise FileNotFoundError(f"Could not find {yaml_path} or {pickle_path}")
