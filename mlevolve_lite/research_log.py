from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def new_response_id(node_id: str) -> str:
    return f"{node_id}__resp_{uuid.uuid4().hex[:10]}"


def append_research_event(path: Path | str, event: dict[str, Any], lock: threading.Lock | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("timestamp", utc_now())
    payload.setdefault("elapsed_seconds", 0.0)

    def write() -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    if lock is None:
        write()
    else:
        with lock:
            write()


def append_research_event_many(
    paths: list[Path | str],
    event: dict[str, Any],
    lock: threading.Lock | None = None,
) -> None:
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)

    if lock is None:
        for path in unique_paths:
            append_research_event(path, event)
        return

    with lock:
        for path in unique_paths:
            append_research_event(path, event, lock=None)


def file_record(path: Path | str, *, root: Path | str | None = None) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    display_path = path
    if root is not None:
        try:
            display_path = path.resolve().relative_to(Path(root).resolve())
        except ValueError:
            display_path = path
    return {
        "path": display_path.as_posix(),
        "sha256": text_sha256(text),
        "bytes": len(text.encode("utf-8")),
    }
