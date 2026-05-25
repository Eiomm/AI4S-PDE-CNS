from __future__ import annotations

import os
from pathlib import Path


def task_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base or task_root()) / path


def _root_from_env(name: str, fallback: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return fallback
    path = Path(raw)
    return path if path.is_absolute() else task_root() / path


def data_path(name: str = "task1_test.hdf5") -> Path:
    return task_root() / "data" / name


def checkpoint_path(name: str) -> Path:
    return task_root() / "checkpoints" / "official" / name


def runs_root() -> Path:
    return _root_from_env("AI4S_TASK1_RUNS_ROOT", task_root() / "runs" / "task1")


def submissions_root() -> Path:
    return _root_from_env("AI4S_TASK1_SUBMISSIONS_ROOT", task_root() / "submissions")
