from __future__ import annotations

from pathlib import Path


def task_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base or task_root()) / path


def data_path(name: str = "task1_test.hdf5") -> Path:
    return task_root() / "data" / name


def checkpoint_path(name: str) -> Path:
    return task_root() / "checkpoints" / "official" / name


def runs_root() -> Path:
    return task_root() / "runs" / "task1"


def submissions_root() -> Path:
    return task_root() / "submissions"
