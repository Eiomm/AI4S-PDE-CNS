from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .pde_results import RunResult
from .pde_tasks import DEFAULT_TASK1_FNO_WEIGHTS
from .pde_workflow import Task1FNOWorkflow


@dataclass(frozen=True)
class Candidate:
    name: str
    weights: dict[str, float]


@dataclass
class SearchResult:
    search_name: str
    candidate_results: list[tuple[Candidate, RunResult]] = field(default_factory=list)
    best_candidate: Candidate | None = None
    best_validation_result: RunResult | None = None
    best_submission_result: RunResult | None = None
    metric: str = "mse"
    maximize: bool = False


class WeightedEnsembleSearch:
    def __init__(
        self,
        *,
        workflow: Task1FNOWorkflow,
        candidates: list[Candidate] | None = None,
        search_name: str = "task1-weight-search",
        metric: str = "mse",
        maximize: bool = False,
    ):
        self.workflow = workflow
        self.candidates = candidates or [Candidate(name="default-weighted-fno", weights=dict(DEFAULT_TASK1_FNO_WEIGHTS))]
        self.search_name = search_name
        self.metric = metric
        self.maximize = maximize

    def run(self, *, make_submission: bool = True) -> SearchResult:
        result = SearchResult(search_name=self.search_name, metric=self.metric, maximize=self.maximize)
        for candidate in self.candidates:
            run_name = str(Path(self.search_name) / self._safe_name(candidate.name))
            run_result = self.workflow.run_validation(candidate.weights, run_name=run_name)
            result.candidate_results.append((candidate, run_result))

        successful = [
            (candidate, run_result)
            for candidate, run_result in result.candidate_results
            if run_result.success and self.metric in run_result.metrics
        ]
        if not successful:
            return result

        rank = max if self.maximize else min
        best_candidate, best_result = rank(successful, key=lambda item: float(item[1].metrics[self.metric]))
        result.best_candidate = best_candidate
        result.best_validation_result = best_result
        if make_submission:
            result.best_submission_result = self.workflow.run_test_submission(
                best_candidate.weights,
                run_name=str(Path(self.search_name) / "best"),
            )
        return result

    @staticmethod
    def _safe_name(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-")
        return cleaned or "candidate"
