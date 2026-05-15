from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .logging import utc_now_iso


@dataclass(frozen=True)
class CodePatchSnapshot:
    code_root: Path
    files: list[dict[str, str]]
    manifest_path: Path


def _safe_relative_path(raw_path: str) -> Path:
    path = Path(str(raw_path).replace("\\", "/"))
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"code patch path is outside code root: {raw_path}")
    if path.parts and path.parts[0] == "code":
        path = Path(*path.parts[1:]) if len(path.parts) > 1 else Path()
    if not path.parts:
        raise ValueError("code patch path must name a file")
    return path


def apply_agent_code_patch(
    *,
    code_root: str | Path,
    files: list[dict[str, str]],
    provenance_record: dict[str, Any] | None = None,
) -> CodePatchSnapshot:
    root = Path(code_root)
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for item in files:
        relative = _safe_relative_path(str(item.get("path", "")))
        content = item.get("content")
        if not isinstance(content, str):
            raise ValueError(f"code patch content for {relative.as_posix()} must be a string")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    manifest_path = root / "code_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": utc_now_iso(),
                "provenance": dict(provenance_record or {}),
                "files": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return CodePatchSnapshot(code_root=root, files=records, manifest_path=manifest_path)
