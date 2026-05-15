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


SYNTHETIC_TRACE_PROVIDERS = {
    "codex",
    "Task1FNOWorkflow",
    "Task2PersistenceWorkflow",
    "bootstrap",
    "mock",
    "static",
    "recording",
    "sequence",
}


def _normal_code_trace_path(raw_path: str, *, code_root_name: str) -> str:
    path = Path(str(raw_path).replace("\\", "/"))
    parts = list(path.parts)
    if parts and parts[0] == code_root_name:
        parts = parts[1:]
    if not parts:
        raise ValueError("empty code trace path")
    return f"{code_root_name}/{Path(*parts).as_posix()}"


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _trace_from_code_patch_record(record: dict[str, Any], *, code_root_name: str) -> list[dict[str, Any]]:
    response = record.get("response")
    payload: dict[str, Any] | None = None
    if isinstance(response, dict):
        if isinstance(response.get("action"), dict):
            payload = response["action"]
        elif isinstance(response.get("content"), str):
            payload = _extract_json_payload(response["content"])
    elif isinstance(response, str):
        payload = _extract_json_payload(response)
    if not isinstance(payload, dict) or payload.get("action_type") != "code_patch":
        return []
    params = payload.get("params")
    if not isinstance(params, dict):
        return []
    files = params.get("files")
    if not isinstance(files, list):
        return []
    traces: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        content = item.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            continue
        traces.append(
            {
                "path": _normal_code_trace_path(path, code_root_name=code_root_name),
                "sha256": _file_sha256(content),
                "content": content,
                "source": "llm_code_patch",
                "provider": record.get("provider"),
                "model": record.get("model"),
            }
        )
    return traces


def collect_code_traces(
    log_paths: list[str | Path],
    *,
    code_root_name: str = "code",
    require_real_llm: bool = False,
) -> dict[str, dict[str, Any]]:
    traced: dict[str, dict[str, Any]] = {}
    for log_path in log_paths:
        for record in read_jsonl(log_path):
            provider = str(record.get("provider", ""))
            synthetic = provider in SYNTHETIC_TRACE_PROVIDERS
            response = record.get("response")
            if isinstance(response, dict) and response.get("action") == "write_code_file":
                if require_real_llm and synthetic:
                    continue
                path = response.get("path")
                if isinstance(path, str):
                    traced[_normal_code_trace_path(path, code_root_name=code_root_name)] = {
                        **response,
                        "source": "write_code_file",
                        "provider": provider,
                        "model": record.get("model"),
                    }
            if require_real_llm and synthetic:
                continue
            for trace in _trace_from_code_patch_record(record, code_root_name=code_root_name):
                traced[trace["path"]] = trace
    return traced


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
    require_real_llm: bool = False,
) -> None:
    code_root = Path(code_dir)
    traced = collect_code_traces(log_paths, code_root_name=code_root_name, require_real_llm=require_real_llm)

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
