from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    task_id: str
    run_dir: Path
    metrics: dict[str, float]
    prediction_path: Path | None
    zip_path: Path | None
    train_time: float
    inference_time: float
    success: bool
    error: str | None = None
    weights: dict[str, float] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_dir": str(self.run_dir),
            "metrics": self.metrics,
            "prediction_path": str(self.prediction_path) if self.prediction_path is not None else None,
            "zip_path": str(self.zip_path) if self.zip_path is not None else None,
            "train_time": float(self.train_time),
            "inference_time": float(self.inference_time),
            "success": bool(self.success),
            "error": self.error,
            "weights": self.weights,
            "command": self.command,
        }


def write_run_result_json(run_dir: str | Path, result: RunResult) -> Path:
    path = Path(run_dir) / "run_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
