from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py

from .baseline_reader import summarize_baseline_context
from .pde_method_library import select_method_candidates


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _hdf5_summary(path: Path, root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": _rel(path, root),
        "exists": path.is_file(),
        "datasets": {},
    }
    if not path.is_file():
        return summary
    try:
        with h5py.File(path, "r") as h5:
            for key, value in h5.items():
                if hasattr(value, "shape"):
                    summary["datasets"][key] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
    except OSError as exc:
        summary["error"] = str(exc)
    return summary


def _knowledge_base(root: Path, *, max_files: int = 8, max_chars: int = 800) -> list[dict[str, str]]:
    directory = root / "data" / "knowledge_base"
    if not directory.is_dir():
        return []
    records: list[dict[str, str]] = []
    for path in sorted(directory.glob("*.md"))[:max_files]:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        records.append(
            {
                "path": _rel(path, root),
                "preview": text[:max_chars],
            }
        )
    return records


def observe_research_context(project_root: str | Path = ".") -> dict[str, Any]:
    """Build a compact factual context for the PDE research planner."""
    root = Path(project_root).resolve()
    task1_dir = root / "data" / "Task1"
    task2_dir = root / "data" / "Task2"
    official_bundle = root / "data" / "data_and_sample_submission"
    return {
        "project_root": str(root),
        "official_bundle": {
            "path": _rel(official_bundle, root),
            "exists": official_bundle.is_dir(),
            "train_val_test_init_exists": (official_bundle / "train_val_test_init").is_dir(),
            "sample_submission_exists": (official_bundle / "sample_submission" / "sample_submission").is_dir(),
        },
        "task1": {
            "validation": _hdf5_summary(task1_dir / "task1_val.hdf5", root),
            "test": _hdf5_summary(task1_dir / "task1_test.hdf5", root),
            "official_fno_checkpoint": {
                "path": "checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt",
                "exists": (root / "checkpoints" / "extracted" / "1D_Burgers_Sols_Nu0.001_FNO.pt").is_file(),
            },
            "official_unet_checkpoint": {
                "path": "checkpoints/extracted/1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt",
                "exists": (root / "checkpoints" / "extracted" / "1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt").is_file(),
            },
        },
        "task2": {
            "validation": _hdf5_summary(task2_dir / "task2_val.h5", root),
            "test": _hdf5_summary(task2_dir / "task2_test.h5", root),
        },
        "knowledge_base": _knowledge_base(root),
        "baseline_context": summarize_baseline_context(root / "third_party" / "baseline"),
        "method_candidates": select_method_candidates(task="task1", metrics={}),
        "planner_hints": [
            "Task 1 prediction must preserve the first 10 frames exactly.",
            "Official PDEBench FNO checkpoint used reduced_resolution_t=5; fine-tune temporal_stride should be treated as a controlled experiment variable.",
            "Do not call numerical solvers or generate extra data.",
        ],
    }
