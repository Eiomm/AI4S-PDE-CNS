from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from .pde_executor import ControlledExperimentExecutor
from .pde_journal import ExperimentJournal
from .pde_mcts import PDEMCTSRunner
from .pde_report import write_journal_report
from .pde_reviewer import ExperimentReviewer
from .pde_workflow import Task1FNOWorkflow


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MCTS config must be a YAML mapping")
    return payload


def run_task1_mcts_experiment(
    *,
    config_path: str | Path,
    project_root: str | Path = ".",
    max_steps: int | None = None,
    reset: bool = False,
) -> Path:
    config = _load_config(config_path)
    study_name = str(config.get("study_name", "task1-mcts-mock"))
    execution = str(config.get("execution", "mock"))
    metric_config = dict(config.get("metric", {}))
    metric = str(metric_config.get("name", config.get("metric_name", "mse")))
    maximize = bool(metric_config.get("maximize", config.get("maximize", False)))
    actions = list(config.get("actions", []))
    if max_steps is not None:
        actions = actions[:max_steps]
    if not all(isinstance(action, dict) for action in actions):
        raise ValueError("MCTS config actions must be a list of mappings")

    root = Path(project_root).resolve()
    study_dir = root / "runs" / study_name
    if reset and study_dir.exists():
        shutil.rmtree(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    journal = ExperimentJournal(study_dir / "journal.json")
    executor = None
    reviewer = None
    if execution == "controlled":
        workflow = Task1FNOWorkflow(
            project_root=root,
            run_root=study_dir / "nodes",
            code_dir=root / "code",
            methodology_path=root / "docs" / "methodology.pdf",
        )
        executor = ControlledExperimentExecutor(
            project_root=root,
            code_dir=root / "code",
            workflow=workflow,
            journal=journal,
            metric=metric,
            maximize=maximize,
            require_code_patch_validation=True,
        )
        reviewer = ExperimentReviewer(journal=journal, metric=metric, maximize=maximize)
    runner = PDEMCTSRunner(
        journal=journal,
        metric=metric,
        maximize=maximize,
        study_name=study_name,
        max_children=int(config.get("max_children", 2)),
        exploration_constant=float(config.get("exploration_constant", 1.414)),
        execution=execution,
        executor=executor,
        reviewer=reviewer,
    )
    summary = runner.run(actions=actions)
    submission_node = _find_submission_node(summary)
    summary["study_name"] = study_name
    summary["execution"] = execution
    summary["config_path"] = str(Path(config_path))
    summary["max_steps"] = max_steps
    summary["reset"] = reset
    summary["journal_path"] = str(journal.path)
    if submission_node is not None:
        summary["submission_node_id"] = submission_node["id"]
        summary["submission_zip_path"] = submission_node.get("artifacts", {}).get("zip_path")
        summary["submission_prediction_path"] = submission_node.get("artifacts", {}).get("prediction_path")
        summary["submission_run_dir"] = submission_node.get("artifacts", {}).get("run_dir")
    report_path = write_journal_report(
        journal,
        study_dir / "journal_report.md",
        title=f"Task 1 MCTS Report: {study_name}",
        metric=metric,
        maximize=maximize,
    )
    summary["journal_report_path"] = str(report_path)
    summary_path = study_dir / "mcts_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def _find_submission_node(summary: dict[str, Any]) -> dict[str, Any] | None:
    for node in reversed(summary.get("nodes", [])):
        if node.get("action_type") != "submit_best":
            continue
        if node.get("status") != "completed" or node.get("error") is not None:
            continue
        artifacts = node.get("artifacts", {})
        if not isinstance(artifacts, dict) or artifacts.get("success") is not True:
            continue
        zip_path = artifacts.get("zip_path")
        if zip_path and Path(zip_path).is_file():
            return node
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Task 1 MCTS experiments in mock or controlled execution mode.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--reset", action="store_true", help="Delete the existing study directory before running.")
    args = parser.parse_args(argv)
    try:
        summary_path = run_task1_mcts_experiment(
            config_path=args.config,
            project_root=args.project_root,
            max_steps=args.max_steps,
            reset=args.reset,
        )
    except Exception as exc:
        print(f"MCTS experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
