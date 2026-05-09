from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


@dataclass
class LLMCallLogger:
    path: Path

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_call(
        self,
        *,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        response: Any,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        record = {
            "timestamp": utc_now_iso(),
            "elapsed_seconds": elapsed_seconds,
            "provider": provider,
            "model": model,
            "messages": messages,
            "response": response,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
    return records


def log_span_seconds(path: str | Path) -> float:
    records = read_jsonl(path)
    if not records:
        return 0.0
    first = datetime.fromisoformat(records[0]["timestamp"])
    last = datetime.fromisoformat(records[-1]["timestamp"])
    return (last - first).total_seconds()
