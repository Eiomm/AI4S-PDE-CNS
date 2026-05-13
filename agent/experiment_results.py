from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .logging import utc_now_iso


@dataclass(frozen=True)
class ExperimentRecord:
    run_dir: Path
    relative_run_dir: str
    study_name: str
    model_name: str
    metrics: dict[str, float]
    config: dict[str, Any] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    success: bool | None = None
    error: str | None = None
    prediction_path: str | None = None
    zip_path: str | None = None

    def to_json_dict(self, *, rank: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_dir": str(self.run_dir),
            "relative_run_dir": self.relative_run_dir,
            "study_name": self.study_name,
            "model_name": self.model_name,
            "metrics": self.metrics,
            "config": self.config,
            "command": self.command,
            "success": self.success,
            "error": self.error,
            "prediction_path": self.prediction_path,
            "zip_path": self.zip_path,
        }
        if rank is not None:
            payload["rank"] = rank
        return payload


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _float_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            metrics[key] = float(value)
    return metrics


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _compact_json(value: Any, *, max_chars: int = 220) -> str:
    if not value:
        return ""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _sort_key(record: ExperimentRecord) -> tuple[int, float, float, str]:
    if "competition_score_proxy" in record.metrics:
        return (0, -float(record.metrics["competition_score_proxy"]), float(record.metrics.get("mse", 1.0e9)), record.relative_run_dir)
    if "mse" in record.metrics:
        return (1, 0.0, float(record.metrics["mse"]), record.relative_run_dir)
    return (2, 0.0, 1.0e9, record.relative_run_dir)


def collect_experiment_records(runs_root: str | Path = "runs") -> list[ExperimentRecord]:
    root = Path(runs_root)
    if not root.exists():
        return []
    records: list[ExperimentRecord] = []
    for metrics_path in root.rglob("metrics.json"):
        run_dir = metrics_path.parent
        metrics = _float_metrics(_read_json(metrics_path))
        run_result = _read_json(run_dir / "run_result.json")
        manifest = _read_json(run_dir / "baseline_manifest.json")
        baseline = manifest.get("baseline", {}) if isinstance(manifest.get("baseline"), dict) else {}
        result = manifest.get("result", {}) if isinstance(manifest.get("result"), dict) else {}
        config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
        relative = _relative(run_dir, root)
        parts = Path(relative).parts
        study_name = parts[0] if parts else run_dir.name
        model_name = str(baseline.get("name") or run_dir.name)
        command = run_result.get("command") or result.get("command") or []
        if not isinstance(command, list):
            command = [str(command)]
        success_raw = run_result.get("success", result.get("success"))
        success = bool(success_raw) if success_raw is not None else None
        records.append(
            ExperimentRecord(
                run_dir=run_dir,
                relative_run_dir=relative,
                study_name=study_name,
                model_name=model_name,
                metrics=metrics,
                config=config,
                command=[str(item) for item in command],
                success=success,
                error=run_result.get("error") or result.get("error"),
                prediction_path=run_result.get("prediction_path") or result.get("prediction_path"),
                zip_path=run_result.get("zip_path") or result.get("zip_path"),
            )
        )
    return sorted(records, key=_sort_key)


def write_experiment_results(
    records: list[ExperimentRecord],
    *,
    output_path: str | Path,
    csv_path: str | Path | None = None,
    json_path: str | Path | None = None,
    top_n: int = 80,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Task 1 Experiment Results",
        "",
        f"- Updated: {utc_now_iso()}",
        f"- Records scanned: {len(records)}",
        "- Sort: competition_score_proxy desc, then mse asc.",
        "",
        "| Rank | Run | Model | Proxy | MSE | Forecast MSE | Long MSE | Segment3 RMSE | Status | Params |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rank, record in enumerate(records[:top_n], start=1):
        metrics = record.metrics
        status = "ok" if record.success is not False else f"failed: {record.error or ''}".strip()
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    _markdown_cell(record.relative_run_dir),
                    _markdown_cell(record.model_name),
                    _markdown_cell(_format_float(metrics.get("competition_score_proxy"))),
                    _markdown_cell(_format_float(metrics.get("mse"))),
                    _markdown_cell(_format_float(metrics.get("forecast_mse"))),
                    _markdown_cell(_format_float(metrics.get("long_horizon_mse"))),
                    _markdown_cell(_format_float(metrics.get("segment3_rmse"))),
                    _markdown_cell(status),
                    _markdown_cell(_compact_json(record.config)),
                ]
            )
            + " |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_path is not None:
        _write_csv(records, Path(csv_path))
    if json_path is not None:
        payload = {
            "version": 1,
            "updated_at": utc_now_iso(),
            "records": [record.to_json_dict(rank=idx) for idx, record in enumerate(records, start=1)],
        }
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.10g}"


def _write_csv(records: list[ExperimentRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "run_dir",
                "model",
                "competition_score_proxy",
                "mse",
                "forecast_mse",
                "long_horizon_mse",
                "segment3_rmse",
                "success",
                "error",
                "config",
                "command",
            ],
        )
        writer.writeheader()
        for rank, record in enumerate(records, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "run_dir": record.relative_run_dir,
                    "model": record.model_name,
                    "competition_score_proxy": record.metrics.get("competition_score_proxy"),
                    "mse": record.metrics.get("mse"),
                    "forecast_mse": record.metrics.get("forecast_mse"),
                    "long_horizon_mse": record.metrics.get("long_horizon_mse"),
                    "segment3_rmse": record.metrics.get("segment3_rmse"),
                    "success": record.success,
                    "error": record.error,
                    "config": _compact_json(record.config, max_chars=1000),
                    "command": " ".join(record.command),
                }
            )


def _inside_root(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
    except OSError:
        return False
    return resolved == root_resolved or root_resolved in resolved.parents


def cleanup_experiment_artifacts(
    runs_root: str | Path = "runs",
    *,
    cleanup_zips: bool = False,
    keep_zips: list[str | Path] | None = None,
    delete_partials: list[str | Path] | None = None,
) -> dict[str, Any]:
    root = Path(runs_root)
    keep = {Path(path).resolve() for path in keep_zips or [] if Path(path).exists()}
    manifest: dict[str, Any] = {"version": 1, "updated_at": utc_now_iso(), "deleted": [], "kept": []}
    if cleanup_zips and root.exists():
        for zip_path in root.rglob("pred.zip"):
            resolved = zip_path.resolve()
            if resolved in keep:
                manifest["kept"].append({"kind": "zip", "path": str(zip_path)})
                continue
            zip_path.unlink()
            manifest["deleted"].append({"kind": "zip", "path": str(zip_path)})
    for item in delete_partials or []:
        raw = Path(item)
        target = raw if raw.is_absolute() else root / raw
        if not target.exists() or not _inside_root(target, root):
            continue
        if target.is_dir():
            shutil.rmtree(target)
            manifest["deleted"].append({"kind": "partial_run", "path": str(target)})
        else:
            target.unlink()
            manifest["deleted"].append({"kind": "partial_file", "path": str(target)})
    return manifest
