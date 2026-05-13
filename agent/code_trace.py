from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .logging import read_jsonl, utc_now_iso


def _iter_traceable_files(code_dir: Path) -> list[Path]:
    ignored_suffixes = {".pyc", ".pyo"}
    files: list[Path] = []
    for path in code_dir.rglob("*"):
        if path.is_dir():
            continue
        if "__pycache__" in path.parts or path.suffix in ignored_suffixes:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(code_dir).as_posix())


def _file_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_code_trace_records(code_dir: str | Path, *, code_root_name: str = "code") -> list[dict[str, Any]]:
    root = Path(code_dir)
    records: list[dict[str, Any]] = []
    for path in _iter_traceable_files(root):
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel_path = f"{code_root_name}/{path.relative_to(root).as_posix()}"
        records.append(
            {
                "timestamp": utc_now_iso(),
                "elapsed_seconds": 0.0,
                "provider": "codex",
                "model": "gpt-5",
                "messages": [
                    {
                        "role": "system",
                        "content": "Trace submitted source code content for code-log consistency.",
                    }
                ],
                "response": {
                    "action": "write_code_file",
                    "path": rel_path,
                    "sha256": _file_sha256(content),
                    "content": content,
                },
            }
        )
    return records


def append_code_trace_log(log_path: str | Path, code_dir: str | Path, *, code_root_name: str = "code") -> None:
    records = build_code_trace_records(code_dir, code_root_name=code_root_name)
    with Path(log_path).open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_code_log_consistency(
    *,
    code_dir: str | Path,
    log_paths: list[str | Path],
    code_root_name: str = "code",
) -> None:
    code_root = Path(code_dir)
    traced: dict[str, dict[str, Any]] = {}
    for log_path in log_paths:
        for record in read_jsonl(log_path):
            response = record.get("response")
            if not isinstance(response, dict):
                continue
            if response.get("action") != "write_code_file":
                continue
            path = response.get("path")
            if isinstance(path, str):
                traced[path] = response

    missing: list[str] = []
    mismatched: list[str] = []
    for path in _iter_traceable_files(code_root):
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel_path = f"{code_root_name}/{path.relative_to(code_root).as_posix()}"
        trace = traced.get(rel_path)
        if trace is None:
            missing.append(rel_path)
            continue
        if trace.get("sha256") != _file_sha256(content) or trace.get("content") != content:
            mismatched.append(rel_path)
    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing trace: {', '.join(missing[:5])}")
        if mismatched:
            details.append(f"mismatched trace: {', '.join(mismatched[:5])}")
        raise ValueError("; ".join(details))
