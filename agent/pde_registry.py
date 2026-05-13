from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .logging import utc_now_iso
from .pde_journal import CandidateNode, ExperimentJournal


COMPARISON_COLUMNS = [
    "study_name",
    "node_id",
    "step",
    "parent_id",
    "status",
    "action_type",
    "intent",
    "metric_name",
    "metric_value",
    "best_candidate_name",
    "run_dir",
    "prediction_path",
    "zip_path",
    "review_next_intent",
    "error",
    "hypothesis",
]

CANDIDATE_COMPARISON_COLUMNS = [
    "study_name",
    "node_id",
    "step",
    "candidate_name",
    "metric_name",
    "metric_value",
    "mse",
    "forecast_mse",
    "competition_score_proxy",
    "success",
    "run_dir",
    "prediction_path",
    "zip_path",
    "weights",
    "error",
]


def _best_candidate(artifacts: dict[str, Any]) -> dict[str, Any]:
    candidate = artifacts.get("best_candidate")
    return candidate if isinstance(candidate, dict) else {}


def _record_from_node(node: CandidateNode, *, study_name: str, metric: str) -> dict[str, Any]:
    candidate = _best_candidate(node.artifacts)
    return {
        "study_name": study_name,
        "node_id": node.id,
        "step": node.step,
        "parent_id": node.parent_id,
        "status": node.status,
        "action_type": node.plan.action_type,
        "intent": node.plan.intent,
        "hypothesis": node.plan.hypothesis,
        "expected_effect": node.plan.expected_effect,
        "risk": node.plan.risk,
        "metric_name": metric,
        "metric_value": node.metrics.get(metric),
        "metrics": dict(node.metrics),
        "best_candidate_name": candidate.get("name"),
        "weights": candidate.get("weights"),
        "run_dir": node.artifacts.get("run_dir"),
        "prediction_path": node.artifacts.get("prediction_path"),
        "zip_path": node.artifacts.get("zip_path"),
        "artifacts": dict(node.artifacts),
        "review_next_intent": node.review.get("next_intent"),
        "review_analysis": node.review.get("analysis"),
        "error": node.error,
        "created_at": node.created_at,
        "updated_at": node.updated_at,
    }


def records_from_journal(journal: ExperimentJournal, *, study_name: str, metric: str) -> list[dict[str, Any]]:
    return [_record_from_node(node, study_name=study_name, metric=metric) for node in journal.read()]


def candidate_records_from_journal(journal: ExperimentJournal, *, study_name: str, metric: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for node in journal.read():
        candidate_results = node.artifacts.get("candidate_results", [])
        if not isinstance(candidate_results, list):
            continue
        for candidate in candidate_results:
            if not isinstance(candidate, dict):
                continue
            metrics = candidate.get("metrics", {})
            metrics = metrics if isinstance(metrics, dict) else {}
            weights = candidate.get("weights", {})
            records.append(
                {
                    "study_name": study_name,
                    "node_id": node.id,
                    "step": node.step,
                    "candidate_name": candidate.get("name"),
                    "metric_name": metric,
                    "metric_value": metrics.get(metric),
                    "mse": metrics.get("mse"),
                    "forecast_mse": metrics.get("forecast_mse"),
                    "competition_score_proxy": metrics.get("competition_score_proxy"),
                    "success": candidate.get("success"),
                    "run_dir": candidate.get("run_dir"),
                    "prediction_path": candidate.get("prediction_path"),
                    "zip_path": candidate.get("zip_path"),
                    "weights": weights if isinstance(weights, dict) else {},
                    "error": candidate.get("error"),
                }
            )
    return records


def load_registry_records(path: str | Path) -> list[dict[str, Any]]:
    registry_path = Path(path)
    if not registry_path.exists():
        return []
    records = []
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        line = line.lstrip("\ufeff")
        if line.strip():
            records.append(json.loads(line))
    return records


def rank_experiment_records(
    records: list[dict[str, Any]],
    *,
    metric: str,
    maximize: bool,
) -> list[dict[str, Any]]:
    candidates = [
        record
        for record in records
        if record.get("metric_name") == metric and record.get("metric_value") is not None
    ]
    return sorted(candidates, key=lambda record: float(record["metric_value"]), reverse=maximize)


def _write_json(path: Path, *, records: list[dict[str, Any]], best: dict[str, Any] | None, metric: str, maximize: bool) -> None:
    payload = {
        "version": 1,
        "updated_at": utc_now_iso(),
        "metric": metric,
        "maximize": maximize,
        "best": best,
        "records": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in COMPARISON_COLUMNS})


def _write_candidate_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATE_COMPARISON_COLUMNS)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in CANDIDATE_COMPARISON_COLUMNS}
            row["weights"] = json.dumps(row["weights"] or {}, ensure_ascii=False, sort_keys=True)
            writer.writerow(row)


def _upsert_registry(path: Path, records: list[dict[str, Any]]) -> None:
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.lstrip("\ufeff")
            if not line.strip():
                continue
            payload = json.loads(line)
            existing[(str(payload.get("study_name")), str(payload.get("node_id")))] = payload
    for record in records:
        existing[(str(record.get("study_name")), str(record.get("node_id")))] = record
    ordered = sorted(existing.values(), key=lambda item: (str(item.get("study_name")), int(item.get("step", 0))))
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered),
        encoding="utf-8",
    )


def export_experiment_records(
    journal: ExperimentJournal,
    *,
    study_dir: str | Path,
    study_name: str,
    runs_root: str | Path,
    metric: str,
    maximize: bool,
) -> dict[str, Path]:
    study_path = Path(study_dir)
    study_path.mkdir(parents=True, exist_ok=True)
    records = records_from_journal(journal, study_name=study_name, metric=metric)
    candidate_records = candidate_records_from_journal(journal, study_name=study_name, metric=metric)
    best_node = journal.best(metric=metric, maximize=maximize)
    best = None
    if best_node is not None:
        best_records = [record for record in records if record["node_id"] == best_node.id]
        best = best_records[0] if best_records else None

    json_path = study_path / "experiment_results.json"
    csv_path = study_path / "experiment_comparison.csv"
    candidate_csv_path = study_path / "candidate_comparison.csv"
    registry_path = Path(runs_root) / "experiment_registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(json_path, records=records, best=best, metric=metric, maximize=maximize)
    _write_csv(csv_path, records)
    _write_candidate_csv(candidate_csv_path, candidate_records)
    _upsert_registry(registry_path, records)
    return {"json": json_path, "csv": csv_path, "candidate_csv": candidate_csv_path, "registry": registry_path}
