from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .code_trace import append_code_trace_log
from .logging import read_jsonl, utc_now_iso
from .pde_journal import ExperimentJournal
from .pde_method_library import select_method_candidates


def _record(section: str, content: str, *, action: str = "research_trace") -> dict[str, Any]:
    return {
        "timestamp": utc_now_iso(),
        "elapsed_seconds": 0.0,
        "provider": "pde-research-agent",
        "model": "task-log-exporter",
        "messages": [
            {
                "role": "system",
                "content": "Export official AI4S Agent research log with trace and experiment tracking.",
            }
        ],
        "response": {
            "action": action,
            "section": section,
            "content": content,
        },
    }


def _best_node(nodes: list[Any], *, metric: str, maximize: bool) -> Any | None:
    candidates = [node for node in nodes if node.status == "completed" and metric in node.metrics and node.error is None]
    if not candidates:
        return None
    return (max if maximize else min)(candidates, key=lambda node: float(node.metrics[metric]))


def _problem_understanding(task: str, best: Any | None, *, metric: str) -> str:
    lines = [
        f"Task: {task}",
        "Input: first 10 frames from task test HDF5; output: 200 frames with first 10 preserved.",
        "Rule guard: no numerical solver generated data; experiments use provided PDEBench data/checkpoints and controlled model evolution.",
    ]
    if best is not None:
        lines.append(f"Current best {metric}: {best.metrics.get(metric)} from node {best.id}.")
        for key in ("forecast_mse", "long_horizon_mse", "segment1_rel_mse", "segment2_rel_mse", "segment3_rmse"):
            if key in best.metrics:
                lines.append(f"{key}: {best.metrics[key]}")
    return "\n".join(lines)


def _method_trace(task: str, best: Any | None) -> str:
    metrics = best.metrics if best is not None else {}
    methods = select_method_candidates(task=task, metrics=metrics)
    lines = ["Agent literature/method candidates:"]
    for item in methods:
        lines.append(
            f"- {item['name']} ({item['source']}): {item['reason']} "
            f"Knobs={json.dumps(item['implementation_knobs'], ensure_ascii=False)} Risk={item['risk']}"
        )
    return "\n".join(lines)


def _experiment_tracking(nodes: list[Any], *, metric: str) -> str:
    lines = ["Experiment trajectory:"]
    for node in nodes:
        value = node.metrics.get(metric)
        conclusion = node.review.get("analysis") if isinstance(node.review, dict) else None
        lines.append(
            f"- step={node.step} id={node.id[:8]} status={node.status} action={node.plan.action_type} "
            f"metric={value} hypothesis={node.plan.hypothesis} params={json.dumps(node.plan.params, ensure_ascii=False)} "
            f"error={node.error} conclusion={conclusion}"
        )
    return "\n".join(lines)


def _bottleneck_diagnosis(nodes: list[Any], *, metric: str) -> str:
    lines = ["Bottleneck diagnosis from validation metrics and reviews:"]
    completed = [node for node in nodes if node.status == "completed"]
    if not completed:
        lines.append("- No completed experiments yet; Agent must run baseline validation first.")
        return "\n".join(lines)
    for node in completed:
        metrics = node.metrics
        signals = []
        if "long_horizon_mse" in metrics and "forecast_mse" in metrics:
            if float(metrics["long_horizon_mse"]) > float(metrics["forecast_mse"]):
                signals.append("long-horizon drift")
        if "segment2_rel_mse" in metrics and "segment1_rel_mse" in metrics:
            if float(metrics["segment2_rel_mse"]) > float(metrics["segment1_rel_mse"]):
                signals.append("rollout accumulation")
        if "segment3_rmse" in metrics:
            signals.append(f"late-segment RMSE={metrics['segment3_rmse']}")
        if not signals:
            signals.append("metric comparison did not isolate a single bottleneck")
        review = node.review.get("analysis") if isinstance(node.review, dict) else None
        lines.append(
            f"- step={node.step} id={node.id[:8]} action={node.plan.action_type} "
            f"{metric}={metrics.get(metric)} signals={'; '.join(signals)} review={review}"
        )
    return "\n".join(lines)


def _code_evolution(nodes: list[Any]) -> str:
    lines = ["Code evolution trace:"]
    touched = False
    for node in nodes:
        code_snapshot = node.artifacts.get("code_snapshot_dir") if isinstance(node.artifacts, dict) else None
        patched = node.artifacts.get("patched_files") if isinstance(node.artifacts, dict) else None
        if node.plan.action_type != "code_patch" and not code_snapshot and not patched:
            continue
        touched = True
        lines.append(
            f"- step={node.step} id={node.id[:8]} action={node.plan.action_type} "
            f"hypothesis={node.plan.hypothesis} patched_files={patched} code_snapshot_dir={code_snapshot} "
            f"params={json.dumps(node.plan.params, ensure_ascii=False)}"
        )
    if not touched:
        lines.append("- No code_patch node was accepted in this study; submitted code must be traced separately before strict submission.")
    return "\n".join(lines)


def _planner_trace(study_dir: Path) -> str:
    log_path = study_dir / "planner_logs.log"
    if not log_path.exists():
        return "No planner_logs.log found."
    lines = ["Raw LLM planner trace summary:"]
    for index, record in enumerate(read_jsonl(log_path), start=1):
        response = record.get("response", {})
        content = response.get("content") if isinstance(response, dict) else None
        lines.append(
            f"- call={index} provider={record.get('provider')} model={record.get('model')} "
            f"elapsed={record.get('elapsed_seconds')} response={str(content)[:1200]}"
        )
    return "\n".join(lines)


def export_task_research_log(
    *,
    study_dir: str | Path,
    output_path: str | Path,
    task: str = "task1",
    metric: str = "competition_score_proxy",
    maximize: bool = True,
    code_dir: str | Path | None = None,
) -> Path:
    study = Path(study_dir)
    if not study.is_dir():
        raise FileNotFoundError(f"study_dir not found: {study}")
    journal = ExperimentJournal(study / "journal.json")
    nodes = journal.read()
    best = _best_node(nodes, metric=metric, maximize=maximize)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [
        _record("problem_understanding", _problem_understanding(task, best, metric=metric)),
        _record("literature_method_trace", _method_trace(task, best)),
        _record("llm_thinking_trace", _planner_trace(study)),
        _record("bottleneck_diagnosis", _bottleneck_diagnosis(nodes, metric=metric)),
        _record("code_evolution", _code_evolution(nodes)),
        _record("experiment_tracking", _experiment_tracking(nodes, metric=metric)),
    ]
    output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    if code_dir is not None:
        append_code_trace_log(output, code_dir)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export official AI4S task research log from autonomous study artifacts.")
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--task", default="task1")
    parser.add_argument("--metric", default="competition_score_proxy")
    parser.add_argument("--minimize", action="store_true")
    parser.add_argument("--code-dir", default=None)
    args = parser.parse_args()
    path = export_task_research_log(
        study_dir=args.study_dir,
        output_path=args.output_path,
        task=args.task,
        metric=args.metric,
        maximize=not args.minimize,
        code_dir=args.code_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
