from __future__ import annotations

from typing import Any

from .pde_journal import CandidateNode, ExperimentJournal


class ExperimentReviewer:
    """Turn execution outcomes into AIDE-style node reviews."""

    def __init__(self, *, journal: ExperimentJournal, metric: str = "mse", maximize: bool = False):
        self.journal = journal
        self.metric = metric
        self.maximize = maximize

    def review_execution(
        self,
        node: CandidateNode,
        *,
        success: bool,
        metrics: dict[str, float],
        artifacts: dict[str, Any],
        error: str | None,
    ) -> CandidateNode:
        if not success:
            review = {
                "analysis": f"Execution failed: {error or 'unknown error'}",
                "next_intent": "debug",
                "metric": self.metric,
                "metric_value": None,
            }
            return self.journal.update_result(
                node.id,
                success=False,
                metrics=metrics,
                artifacts=artifacts,
                error=error,
                review=review,
            )

        best_before = self._best_parent_or_prior(node)
        current_value = metrics.get(self.metric)
        if current_value is None:
            next_intent = "improve"
            analysis = "Execution succeeded but did not produce the target metric; run validation next."
        elif best_before is None or self._is_better(float(current_value), float(best_before)):
            next_intent = "improve"
            analysis = f"Execution improved or established {self.metric}={current_value}."
        else:
            next_intent = "debug"
            analysis = f"Execution completed but did not improve {self.metric}={current_value}."

        review = {
            "analysis": analysis,
            "next_intent": next_intent,
            "metric": self.metric,
            "metric_value": current_value,
        }
        return self.journal.update_result(
            node.id,
            success=True,
            metrics=metrics,
            artifacts=artifacts,
            error=None,
            review=review,
        )

    def _best_parent_or_prior(self, node: CandidateNode) -> float | None:
        nodes = [candidate for candidate in self.journal.read() if candidate.id != node.id]
        values = [
            float(candidate.metrics[self.metric])
            for candidate in nodes
            if candidate.status == "completed" and self.metric in candidate.metrics and candidate.error is None
        ]
        if not values:
            return None
        return max(values) if self.maximize else min(values)

    def _is_better(self, value: float, best: float) -> bool:
        return value > best if self.maximize else value < best
