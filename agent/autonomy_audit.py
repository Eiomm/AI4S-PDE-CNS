from __future__ import annotations

from pathlib import Path
from typing import Any

from .code_trace import SYNTHETIC_TRACE_PROVIDERS
from .logging import read_jsonl
from .pde_journal import CandidateNode, ExperimentJournal


class AutonomyAuditError(RuntimeError):
    pass


EXPERIMENT_ACTIONS_WITH_METRICS = {
    "weight_search",
    "postprocess_search",
    "finetune_checkpoint",
    "finetune",
    "baseline_train",
    "baseline_validate",
    "baseline_ensemble",
    "baseline_refine",
    "task2_train_model",
    "evaluate_candidate",
}


def _failures_from_candidates(node: CandidateNode) -> bool:
    candidates = node.artifacts.get("candidate_results") if isinstance(node.artifacts, dict) else None
    if not isinstance(candidates, list):
        return False
    return any(isinstance(item, dict) and item.get("success") is False for item in candidates)


def _has_baseline_source_trace(nodes: list[CandidateNode]) -> bool:
    for node in nodes:
        params = node.plan.params
        source_files = params.get("source_files")
        if isinstance(source_files, list) and source_files:
            return True
        source_method = params.get("source_method")
        if isinstance(source_method, str) and source_method.strip():
            return True
        if node.plan.action_type == "inspect_data" and node.status == "completed":
            return True
    return False


def _read_planner_records(study_dir: Path) -> list[dict[str, Any]]:
    path = study_dir / "planner_logs.log"
    if not path.is_file():
        raise AutonomyAuditError(f"planner_logs.log not found: {path}")
    records = read_jsonl(path)
    if not records:
        raise AutonomyAuditError("planner_logs.log is empty")
    for index, record in enumerate(records, start=1):
        if "timestamp" not in record or "elapsed_seconds" not in record:
            raise AutonomyAuditError(f"planner_logs.log:{index} missing timestamp or elapsed_seconds")
    return records


def audit_autonomous_study(
    study_dir: str | Path,
    *,
    task: str,
    metric: str,
    min_llm_calls: int = 3,
    min_metric_experiments: int = 2,
    require_real_llm: bool = True,
    require_code_patch: bool = True,
    require_failed_experiment: bool = True,
) -> dict[str, Any]:
    """Check whether an autonomous study is strong enough for strict submission provenance."""
    study = Path(study_dir)
    if not study.is_dir():
        raise AutonomyAuditError(f"study_dir not found: {study}")
    records = _read_planner_records(study)
    providers = [str(record.get("provider", "")) for record in records]
    if require_real_llm:
        synthetic = sorted({provider for provider in providers if provider in SYNTHETIC_TRACE_PROVIDERS})
        if synthetic:
            raise AutonomyAuditError(f"synthetic planner provider is not allowed in strict autonomy: {synthetic}")
    if len(records) < min_llm_calls:
        raise AutonomyAuditError(f"too few LLM calls: {len(records)} < {min_llm_calls}")

    journal = ExperimentJournal(study / "journal.json")
    nodes = journal.read()
    if not nodes:
        raise AutonomyAuditError("journal has no experiment nodes")
    if not _has_baseline_source_trace(nodes):
        raise AutonomyAuditError("missing baseline/source reading trace")

    metric_nodes = [
        node
        for node in nodes
        if node.status == "completed"
        and node.plan.action_type in EXPERIMENT_ACTIONS_WITH_METRICS
        and metric in node.metrics
    ]
    if len(metric_nodes) < min_metric_experiments:
        raise AutonomyAuditError(
            f"too few completed metric experiments: {len(metric_nodes)} < {min_metric_experiments}"
        )
    if require_code_patch and not any(node.status == "completed" and node.plan.action_type == "code_patch" for node in nodes):
        raise AutonomyAuditError("missing completed code_patch node")
    if require_failed_experiment and not any(node.status == "failed" or node.error or _failures_from_candidates(node) for node in nodes):
        raise AutonomyAuditError("missing failed experiment or failed-candidate analysis")

    if task == "task1":
        finetunes = [node for node in nodes if node.plan.action_type == "finetune_checkpoint"]
        official_origin = any(
            "1D_Burgers_Sols_Nu0.001_FNO.pt" in str(node.plan.params.get("base_checkpoint", ""))
            or "1D_Burgers_Sols_Nu0.001_FNO.pt" in str(node.artifacts.get("base_checkpoint", ""))
            for node in finetunes
        )
        stride5_seen = any(int(node.plan.params.get("temporal_stride", 0) or 0) == 5 for node in finetunes)
        if finetunes and not official_origin:
            raise AutonomyAuditError("Task1 finetune nodes must trace back to the official Nu0.001 FNO checkpoint")
        if finetunes and not stride5_seen:
            raise AutonomyAuditError("Task1 finetune study never selected temporal_stride=5")
    if task == "task2":
        forbidden = [
            node
            for node in nodes
            if "task1" in " ".join(str(value).lower() for value in node.plan.params.values())
            or "checkpoints/extracted" in " ".join(str(value).lower() for value in node.plan.params.values())
        ]
        if forbidden:
            raise AutonomyAuditError("Task2 study references Task1 data or checkpoints")

    return {
        "ok": True,
        "task": task,
        "metric": metric,
        "llm_call_count": len(records),
        "providers": providers,
        "node_count": len(nodes),
        "metric_experiment_count": len(metric_nodes),
        "has_code_patch": any(node.plan.action_type == "code_patch" for node in nodes),
        "has_failed_experiment": any(node.status == "failed" or node.error or _failures_from_candidates(node) for node in nodes),
    }
