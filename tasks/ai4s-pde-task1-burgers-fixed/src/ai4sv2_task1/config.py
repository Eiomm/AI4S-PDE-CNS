from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import resolve_path, task_root


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    payload["_config_path"] = str(config_path)
    return payload


def default_input_for_split(split: str) -> Path:
    if split == "val":
        return task_root() / "data" / "task1_val.hdf5"
    if split == "test":
        return task_root() / "data" / "task1_test.hdf5"
    raise ValueError(f"unsupported split: {split}")


def default_run_label(config: dict[str, Any], split: str) -> str:
    config_path = Path(str(config.get("_config_path", "task1")))
    route = str(config.get("route", config_path.stem)).replace("_", "-")
    return f"{split}__{route}"
