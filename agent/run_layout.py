from __future__ import annotations

from datetime import datetime
from pathlib import Path


def safe_study_name(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in name.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "study"


def classified_study_dir(
    *,
    project_root: str | Path = ".",
    task: str,
    category: str,
    study_name: str,
    date: str | None = None,
) -> Path:
    """Return the normalized run directory for new experiments.

    New autonomous studies live under:
    runs/<task>/<category>/<YYYYMMDD>/<study_name>

    Existing legacy run folders are intentionally left in place.
    """

    root = Path(project_root).resolve()
    day = date or datetime.now().strftime("%Y%m%d")
    return root / "runs" / safe_study_name(task) / safe_study_name(category) / day / safe_study_name(study_name)


def registry_path(*, project_root: str | Path = ".") -> Path:
    return Path(project_root).resolve() / "runs" / "experiment_registry.jsonl"
