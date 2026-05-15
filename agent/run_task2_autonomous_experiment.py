from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .llm import LLMClient, build_llm_client
from .logging import LLMCallLogger
from .pde_autonomous import AutonomousExperimentRunner
from .pde_executor import ControlledExperimentExecutor
from .pde_journal import ExperimentJournal
from .pde_observer import observe_research_context
from .pde_planner import ExperimentPlanner
from .pde_registry import export_experiment_records
from .pde_reviewer import ExperimentReviewer
from .run import load_config, load_project_env
from .run_layout import classified_study_dir


def task2_bootstrap_train_plan(*, study_name: str, model: str = "minifno_nu") -> dict[str, Any]:
    return {
        "intent": "improve",
        "hypothesis": (
            "Task2 must train from scratch on multi-Nu data. Start with a small Nu-aux MiniFNO "
            "smoke run to verify data loading, validation metrics, and checkpoint promotion before code evolution."
        ),
        "action_type": "task2_train_model",
        "params": {
            "model": model,
            "output_dir": f"runs/task2/bootstrap/{study_name}/{model}",
            "epochs": 1,
            "sample_limit": 64,
            "val_sample_limit": 16,
            "batch_size": 8,
            "hidden_channels": 16,
            "modes": 12,
            "lr": 1.0e-3,
            "device": "cpu",
            "nu_aux_weight": 0.02,
        },
        "expected_effect": "Establish a traceable Task2 train/eval node and compare against persistence.",
        "risk": "A one-epoch smoke run may be worse than persistence; use it only to validate the loop.",
    }


class BootstrapPlanClient:
    provider = "bootstrap"
    model = "task2-bootstrap"

    def __init__(self, delegate: LLMClient, plans: list[dict[str, Any]]):
        self.delegate = delegate
        self.plans = list(plans)
        if not self.plans:
            self.provider = delegate.provider
            self.model = delegate.model

    def complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if self.plans:
            return {"content": json.dumps(self.plans.pop(0), ensure_ascii=False)}
        return self.delegate.complete(messages)


def run_autonomous_task2(
    *,
    config_path: str | Path,
    project_root: str | Path = ".",
    study_name: str = "task2-autonomous",
    max_iterations: int = 3,
    metric: str = "forecast_mse",
    maximize: bool = False,
    time_budget_seconds: float | None = None,
    bootstrap_train: bool = False,
    strict_autonomy: bool = False,
) -> Path:
    root = Path(project_root).resolve()
    if strict_autonomy and bootstrap_train:
        raise ValueError("strict_autonomy forbids bootstrap_train; the LLM planner must choose every Task2 experiment")
    config = load_config(config_path)
    loaded_env_keys = load_project_env(config, root)
    study_dir = classified_study_dir(project_root=root, task="task2", category="autonomous", study_name=study_name)
    study_dir.mkdir(parents=True, exist_ok=True)

    journal = ExperimentJournal(study_dir / "journal.json")
    client = build_llm_client(config)
    if bootstrap_train:
        client = BootstrapPlanClient(client, [task2_bootstrap_train_plan(study_name=study_name)])
    planner = ExperimentPlanner(
        client=client,
        logger=LLMCallLogger(study_dir / "planner_logs.log"),
        journal=journal,
        metric=metric,
        maximize=maximize,
    )
    executor = ControlledExperimentExecutor(
        project_root=root,
        code_dir=root / "code",
        journal=journal,
        metric=metric,
        maximize=maximize,
        require_code_patch_validation=True,
    )
    reviewer = ExperimentReviewer(journal=journal, metric=metric, maximize=maximize)
    runner = AutonomousExperimentRunner(
        planner=planner,
        executor=executor,
        reviewer=reviewer,
        observer=lambda: observe_research_context(root),
    )
    summary = runner.run(
        context={
            "task": "task2",
            "study_name": study_name,
            "loaded_env_keys": loaded_env_keys,
            "submission_rule": "Task2 final package must be runs/<task2-study>/pred.zip",
            "task2_rules": [
                "train from scratch only",
                "use data/Task2 files only",
                "do not use Task1 checkpoints or Task1 data",
                "test Nu is hidden; infer any latent Nu from the first 10 frames",
                "plateau should trigger code_patch for reusable Task2 capability evolution",
            ],
            "bootstrap_train": bootstrap_train,
            "strict_autonomy": strict_autonomy,
            "strict_autonomy_rules": [
                "No bootstrap/preset Task2 plan is active.",
                "Planner must choose model family, hyperparameters, and follow-up experiments from observed metrics.",
                "Task2 code evolution must remain train-from-scratch and must not reference Task1 assets.",
                "Before final packaging, run an autonomy audit over planner_logs.log and journal.json.",
            ],
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
    parser = argparse.ArgumentParser(description="Run Task 2 autonomous train-from-scratch experiment planning.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--study-name", default="task2-autonomous")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--metric", default="forecast_mse")
    parser.add_argument("--maximize", action="store_true")
    parser.add_argument("--time-budget-seconds", type=float, default=None)
    parser.add_argument("--bootstrap-train", action="store_true")
    parser.add_argument("--strict-autonomy", action="store_true")
    args = parser.parse_args()
    path = run_autonomous_task2(
        config_path=args.config,
        project_root=args.project_root,
        study_name=args.study_name,
        max_iterations=args.max_iterations,
        metric=args.metric,
        maximize=args.maximize,
        time_budget_seconds=args.time_budget_seconds,
        bootstrap_train=args.bootstrap_train,
        strict_autonomy=args.strict_autonomy,
    )
    print(path)


if __name__ == "__main__":
    main()
