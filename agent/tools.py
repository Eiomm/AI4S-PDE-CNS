from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


@dataclass
class ToolRunner:
    project_root: Path
    allowed_roots: list[Path] = field(default_factory=list)
    allowed_shell_commands: list[str] = field(
        default_factory=lambda: ["python", "pytest", "git"]
    )
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
        executable = Path(args[0]).name.lower()
        allowed = {cmd.lower() for cmd in self.allowed_shell_commands}
        if executable not in allowed:
            raise ToolError(f"Shell command '{args[0]}' is not allowed")
        result = subprocess.run(
            args,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        record = {
            "args": args,
            "returncode": result.returncode,
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
