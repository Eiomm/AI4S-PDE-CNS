from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class SearchNode(BaseModel):
    node_id: str
    strategy: Dict[str, object] = Field(default_factory=dict)
    parent_ids: List[str] = Field(default_factory=list)
    candidate_ids: List[str] = Field(default_factory=list)
    mean_score: float = 0.0
    best_score: float = 0.0
    cost_seconds: float = 0.0
    status: Literal["running", "complete", "failed"] = "running"


def budget_phase(fraction_used: float) -> str:
    if fraction_used < 0.30:
        return "diversity_exploration"
    if fraction_used < 0.80:
        return "mixed_ucb"
    return "exploit_route_feasibility"
