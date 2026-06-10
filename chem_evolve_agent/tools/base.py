from __future__ import annotations

import shutil
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    OK = "ok"
    SKIPPED = "skipped"
    ERROR = "error"


class ToolResult(BaseModel):
    tool_name: str
    status: ToolStatus
    command: list[str] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = 0.0
    reason: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status == ToolStatus.OK

    @property
    def skipped(self) -> bool:
        return self.status == ToolStatus.SKIPPED

    def to_log_fields(self) -> dict[str, Any]:
        return {
            "tool": self.tool_name,
            "status": self.status.value,
            "reason": self.reason,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "warnings": self.warnings,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def find_executable(*names: str) -> str | None:
    env_bin = Path(sys.executable).resolve().parent
    for name in names:
        path = shutil.which(name)
        if path:
            return path
        env_path = env_bin / name
        if env_path.exists() and env_path.is_file():
            return str(env_path)
    return None


def has_python_module(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


def run_command(command: list[str], timeout: int = 120, cwd: Path | None = None) -> ToolResult:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
    except subprocess.TimeoutExpired as exc:
        return ToolResult(
            tool_name=Path(command[0]).name,
            status=ToolStatus.ERROR,
            command=command,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            elapsed_seconds=time.monotonic() - start,
            reason="timeout",
        )
    except Exception as exc:
        return ToolResult(
            tool_name=Path(command[0]).name,
            status=ToolStatus.ERROR,
            command=command,
            elapsed_seconds=time.monotonic() - start,
            reason=str(exc),
        )
    return ToolResult(
        tool_name=Path(command[0]).name,
        status=ToolStatus.OK if proc.returncode == 0 else ToolStatus.ERROR,
        command=command,
        stdout=proc.stdout,
        stderr=proc.stderr,
        elapsed_seconds=time.monotonic() - start,
        reason=None if proc.returncode == 0 else f"return_code_{proc.returncode}",
    )


def safe_name(text: str, limit: int = 48) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in text)
    cleaned = cleaned.strip("._-") or "item"
    return cleaned[:limit]
