from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .submission import validate_submission, SubmissionError


class ToolError(RuntimeError):
    pass


def _resolve(path: str | Path) -> Path:
    return Path(path).resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _command_name(command: str | Path) -> str:
    name = Path(command).name.lower()
    return name[:-4] if name.endswith(".exe") else name


@dataclass
class ToolRunner:
    project_root: Path
    allowed_roots: list[Path] = field(default_factory=list)
    allowed_shell_commands: list[str] = field(
        default_factory=lambda: ["python", "pytest", "git"]
    )
    command_aliases: dict[str, str] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=lambda: {"writes": [], "shell": []})

    def __post_init__(self) -> None:
        self.project_root = _resolve(self.project_root)
        self.allowed_roots = [_resolve(root) for root in ([self.project_root] + self.allowed_roots)]

    def ensure_allowed_path(self, path: str | Path) -> Path:
        resolved = _resolve(path)
        if not any(_is_relative_to(resolved, root) or resolved == root for root in self.allowed_roots):
            raise ToolError(f"{resolved} is outside allowed roots")
        return resolved

    def read_file(self, path: str | Path) -> str:
        target = self.ensure_allowed_path(path)
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str | Path, content: str, *, reason: str) -> None:
        target = self.ensure_allowed_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self.manifest["writes"].append({"path": str(target), "reason": reason})

    def run_shell(self, args: list[str], *, timeout_seconds: int = 120) -> dict[str, Any]:
        if not args:
            raise ToolError("Shell command cannot be empty")
        original_args = [str(arg) for arg in args]
        command_key = _command_name(original_args[0])
        allowed = {_command_name(cmd) for cmd in self.allowed_shell_commands}
        if command_key not in allowed:
            raise ToolError(f"Shell command '{args[0]}' is not allowed")
        resolved_args = list(original_args)
        if command_key in self.command_aliases:
            resolved_args[0] = self.command_aliases[command_key]
        started = time.perf_counter()
        result = subprocess.run(
            resolved_args,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed_seconds = round(time.perf_counter() - started, 6)
        record = {
            "args": original_args,
            "resolved_args": resolved_args,
            "returncode": result.returncode,
            "elapsed_seconds": elapsed_seconds,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }
        self.manifest["shell"].append(record)
        return record

    def save_manifest(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        target = self.ensure_allowed_path(path)
        payload = dict(self.manifest)
        if extra:
            payload.update(extra)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def analyze_result(self, path: str | Path) -> dict[str, Any]:
        """Read and format experiment results (metrics.json or prediction file)."""
        target = self.ensure_allowed_path(path)
        if not target.exists():
            return {"ok": False, "error": f"File not found: {target}"}

        if target.suffix == ".json":
            try:
                data = json.loads(target.read_text(encoding="utf-8"))
                return {"ok": True, "type": "metrics", "data": data}
            except json.JSONDecodeError as e:
                return {"ok": False, "error": f"Invalid JSON: {e}"}

        if target.suffix in (".hdf5", ".h5"):
            try:
                import h5py
                import numpy as np
                with h5py.File(target, "r") as f:
                    info = {}
                    for key in f.keys():
                        ds = f[key]
                        info[key] = {
                            "shape": list(ds.shape),
                            "dtype": str(ds.dtype),
                            "min": float(np.min(ds[:])) if ds.size < 1e8 else "too_large",
                            "max": float(np.max(ds[:])) if ds.size < 1e8 else "too_large",
                            "mean": float(np.mean(ds[:])) if ds.size < 1e8 else "too_large",
                        }
                return {"ok": True, "type": "hdf5", "file": str(target), "datasets": info}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        return {"ok": False, "error": f"Unsupported file type: {target.suffix}"}

    def validate_submission(self, path: str | Path) -> dict[str, Any]:
        """Validate a submission directory."""
        try:
            report = validate_submission(path)
            return {"ok": True, "valid": report.valid, "tasks": report.tasks, "messages": report.messages}
        except (SubmissionError, Exception) as e:
            return {"ok": False, "error": str(e)}
