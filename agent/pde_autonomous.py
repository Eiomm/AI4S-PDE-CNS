from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .pde_executor import ControlledExperimentExecutor
from .pde_planner import ExperimentPlanner
from .pde_report import write_journal_report
from .pde_reviewer import ExperimentReviewer


class AutonomousExperimentRunner:
    """Minimal AIDE-style loop: plan one node, execute it, review it, repeat."""

    def __init__(
        self,
        *,
        planner: ExperimentPlanner,
        executor: ControlledExperimentExecutor,
        reviewer: ExperimentReviewer,
        observer: Any | None = None,
    ):
        self.planner = planner
        self.executor = executor
        self.reviewer = reviewer
        self.observer = observer

    def run(
        self,
        *,
        context: dict[str, Any],
        max_iterations: int,
        time_budget_seconds: float | None = None,
        report_path: str | Path | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        completed = []
        stopped = False
        stop_reason = None

        for _ in range(max_iterations):
            elapsed = time.perf_counter() - started
            if time_budget_seconds is not None and elapsed >= time_budget_seconds:
                stop_reason = "time_budget_exhausted"
                break

            step_context = {**context, "elapsed_seconds": elapsed}
            if self.observer is not None:
                step_context["observer"] = self.observer()
            node = self.planner.plan_next(step_context)
            self.planner.journal.mark_running(node.id)
            execution = self.executor.execute(node)
            reviewed = self.reviewer.review_execution(
                node,
                success=execution.success,
                metrics=execution.metrics,
                artifacts=execution.artifacts,
                error=execution.error,
            )
            completed.append(reviewed.id)
            if node.plan.action_type == "stop":
                stopped = True
                params = node.plan.params
                stop_reason = params.get("reason") if isinstance(params, dict) else None
                break

        summary = {
            "iterations": len(completed),
            "node_ids": completed,
            "stopped": stopped,
            "stop_reason": stop_reason,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if report_path is not None:
            written = write_journal_report(
                self.planner.journal,
                report_path,
                metric=self.reviewer.metric,
                maximize=self.reviewer.maximize,
            )
            summary["report_path"] = str(written)
        return summary
