from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()

    def write(self, response: dict[str, Any], *, tool_calls: list[dict[str, Any]] | None = None) -> None:
        record: dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "elapsed_seconds": float(time.perf_counter() - self.started),
            "response": response,
        }
        if tool_calls is not None:
            record["tool_calls"] = tool_calls
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
