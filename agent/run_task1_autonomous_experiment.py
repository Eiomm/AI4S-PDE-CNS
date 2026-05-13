from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .llm import LLMClient
from .llm import build_llm_client
from .logging import LLMCallLogger
from .pde_autonomous import AutonomousExperimentRunner
from .pde_executor import ControlledExperimentExecutor
from .pde_journal import ExperimentJournal
from .pde_planner import ExperimentPlanner
from .pde_registry import export_experiment_records
from .pde_reviewer import ExperimentReviewer
from .pde_workflow import Task1FNOWorkflow
from .run import load_config, load_project_env
from .run_task1_weight_search import parse_checkpoint_overrides


TASK1_CURRENT_FINAL_WEIGHTS: dict[str, float] = {
    "nu0.001": 0.0,
    "nu0.01": 0.085,
    "nu0.1": 0.915,
    "nu1.0": 0.0,
}


def _rounded_weights(weights: dict[str, float]) -> dict[str, float]:
    rounded = {key: round(float(value), 6) for key, value in weights.items()}
    total = round(sum(rounded.values()), 6)
    if total != 1.0:
        rounded["nu0.1"] = round(rounded.get("nu0.1", 0.0) + (1.0 - total), 6)
    return rounded


def _delta_candidate_name(offset: int, delta: float) -> str:
    if offset == 0:
        return "current-final-proxy"
    direction = "pos" if delta > 0 else "neg"
    return f"grid-delta-{direction}{abs(delta):.3f}"


def task1_local_weight_grid_candidates(
    *,
    base_weights: dict[str, float] | None = None,
    grid_step: float = 0.01,
    grid_radius: int = 2,
) -> list[dict[str, Any]]:
    """Build a local line-search grid by shifting mass between nu0.01 and nu0.1."""
    if grid_step <= 0:
        raise ValueError("grid_step must be positive")
    if grid_radius < 0:
        raise ValueError("grid_radius must be non-negative")
    base = dict(base_weights or TASK1_CURRENT_FINAL_WEIGHTS)
    candidates: list[dict[str, Any]] = []
    for offset in range(-grid_radius, grid_radius + 1):
        delta = round(offset * grid_step, 6)
        weights = dict(base)
        weights["nu0.01"] = round(float(weights.get("nu0.01", 0.0)) + delta, 6)
        weights["nu0.1"] = round(float(weights.get("nu0.1", 0.0)) - delta, 6)
        if any(value < 0.0 or value > 1.0 for value in weights.values()):
            continue
        candidates.append({"name": _delta_candidate_name(offset, delta), "weights": _rounded_weights(weights)})
    return candidates


def task1_bootstrap_weight_search_plan(
    *,
    metric: str = "mse",
    maximize: bool = False,
    grid_step: float = 0.01,
    grid_radius: int = 2,
) -> dict[str, Any]:
    return {
        "intent": "draft",
        "hypothesis": (
            "Run a conservative Task 1 autonomous weight-search node around the current "
            "fine-tuned nu=0.1 ensemble baseline before attempting riskier code patches."
        ),
        "action_type": "weight_search",
        "params": {
            "metric": metric,
            "maximize": maximize,
            "make_submission": False,
            "grid_step": grid_step,
            "grid_radius": grid_radius,
            "candidates": task1_local_weight_grid_candidates(grid_step=grid_step, grid_radius=grid_radius),
        },
        "expected_effect": "Establish that the autonomous loop can run real Task 1 validation and rank candidates.",
        "risk": "Low; only validation inference is run and no submission package is produced.",
    }


class BootstrapPlanClient:
    provider = "bootstrap"
    model = "task1-bootstrap-weight-search"

    def __init__(self, delegate: LLMClient, plans: list[dict[str, Any]]):
        self.delegate = delegate
        self.plans = list(plans)
        if not self.plans:
            self.provider = delegate.provider
            self.model = delegate.model

    def complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if self.plans:
            plan = self.plans.pop(0)
            return {"content": json.dumps(plan, ensure_ascii=False)}
        return self.delegate.complete(messages)


def run_autonomous_task1(
    *,
    config_path: str | Path,
    project_root: str | Path = ".",
    study_name: str = "task1-autonomous",
    max_iterations: int = 3,
    metric: str = "mse",
    maximize: bool = False,
    time_budget_seconds: float | None = None,
    checkpoint_overrides: dict[str, Path] | None = None,
    bootstrap_weight_search: bool = False,
    bootstrap_grid_step: float = 0.01,
    bootstrap_grid_radius: int = 2,
) -> Path:
    root = Path(project_root).resolve()
    config = load_config(config_path)
    loaded_env_keys = load_project_env(config, root)
    study_dir = root / "runs" / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    journal = ExperimentJournal(study_dir / "journal.json")
    client = build_llm_client(config)
    if bootstrap_weight_search:
        client = BootstrapPlanClient(
            client,
            [
                task1_bootstrap_weight_search_plan(
                    metric=metric,
                    maximize=maximize,
                    grid_step=bootstrap_grid_step,
                    grid_radius=bootstrap_grid_radius,
                )
            ],
        )
    planner = ExperimentPlanner(
        client=client,
        logger=LLMCallLogger(study_dir / "planner_logs.log"),
        journal=journal,
        metric=metric,
        maximize=maximize,
    )
    workflow = Task1FNOWorkflow(
        project_root=root,
        run_root=study_dir / "nodes",
        checkpoint_paths=checkpoint_overrides,
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
    runner = AutonomousExperimentRunner(planner=planner, executor=executor, reviewer=reviewer)
    summary = runner.run(
        context={
            "task": "task1",
            "study_name": study_name,
            "loaded_env_keys": loaded_env_keys,
            "allowed_actions": [
                "weight_search",
                "finetune",
                "code_patch",
                "baseline_train",
                "baseline_validate",
                "baseline_ensemble",
                "baseline_refine",
                "submit_best",
                "stop",
            ],
            "submission_rule": "final package must be runs/<experiment>/pred.zip",
            "bootstrap_weight_search": bootstrap_weight_search,
            "bootstrap_grid_step": bootstrap_grid_step,
            "bootstrap_grid_radius": bootstrap_grid_radius,
            "checkpoint_overrides": {key: str(value) for key, value in (checkpoint_overrides or {}).items()},
        },
        max_iterations=max_iterations,
        time_budget_seconds=time_budget_seconds,
        report_path=study_dir / "journal_report.md",
    )
    record_outputs = export_experiment_records(
        journal,
        study_dir=study_dir,
        study_name=study_name,
        runs_root=root / "runs",
        metric=metric,
        maximize=maximize,
    )
    summary["experiment_results_path"] = str(record_outputs["json"])
    summary["experiment_comparison_path"] = str(record_outputs["csv"])
    summary["candidate_comparison_path"] = str(record_outputs["candidate_csv"])
    summary["experiment_registry_path"] = str(record_outputs["registry"])
    summary_path = study_dir / "autonomous_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 1 AIDE-style autonomous experiment planning.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--study-name", default="task1-autonomous")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--metric", default="mse")
    parser.add_argument("--maximize", action="store_true")
    parser.add_argument("--time-budget-seconds", type=float, default=None)
    parser.add_argument("--bootstrap-weight-search", action="store_true")
    parser.add_argument("--bootstrap-grid-step", type=float, default=0.01)
    parser.add_argument("--bootstrap-grid-radius", type=int, default=2)
    parser.add_argument(
        "--checkpoint-override",
        action="append",
        default=[],
        help="Override an FNO checkpoint path with KEY=PATH, e.g. nu0.1=runs/finetune/best.pt",
    )
    args = parser.parse_args()
    path = run_autonomous_task1(
        config_path=args.config,
        project_root=args.project_root,
        study_name=args.study_name,
        max_iterations=args.max_iterations,
        metric=args.metric,
        maximize=args.maximize,
        time_budget_seconds=args.time_budget_seconds,
        checkpoint_overrides=parse_checkpoint_overrides(args.checkpoint_override),
        bootstrap_weight_search=args.bootstrap_weight_search,
        bootstrap_grid_step=args.bootstrap_grid_step,
        bootstrap_grid_radius=args.bootstrap_grid_radius,
    )
    print(path)


if __name__ == "__main__":
    main()
