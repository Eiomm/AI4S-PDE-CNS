from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping


DEFAULT_HWPYTORCH_PYTHON = Path("D:/Junao/ProgramData/anaconda3/envs/Hwpytorch/python.exe")


def resolve_project_python(
    *,
    explicit_python: str | Path | None = None,
    default_python: str | Path = DEFAULT_HWPYTORCH_PYTHON,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the Python executable used by local project helper scripts."""
    values = os.environ if env is None else env
    raw_path = explicit_python or values.get("AI4S_PROJECT_PYTHON") or default_python
    path = Path(raw_path).expanduser()
    if path.exists():
        return path.resolve()
    return Path(sys.executable).resolve()
