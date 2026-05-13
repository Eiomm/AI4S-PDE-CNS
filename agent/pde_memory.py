from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .logging import utc_now_iso


class ExperimentMemory:
    """Small append-only memory for one experiment directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return payload["records"]
        raise ValueError(f"{self.path} must contain a records list")

    def write(self, records: list[dict[str, Any]]) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "records": records}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return self.path

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        records = self.read()
        next_record = dict(record)
        next_record.setdefault("created_at", utc_now_iso())
        records.append(next_record)
        self.write(records)
        return next_record

    def best(self, *, metric: str = "mse", lower_is_better: bool = True) -> dict[str, Any] | None:
        records = [record for record in self.read() if metric in record.get("metrics", {})]
        if not records:
            return None
        key = lambda record: float(record["metrics"][metric])
        return min(records, key=key) if lower_is_better else max(records, key=key)
