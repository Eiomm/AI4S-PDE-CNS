from __future__ import annotations

from typing import Any, Dict, List, Set

from pydantic import BaseModel, Field

from chem_evolve_agent.models import Candidate


class AgentBudget(BaseModel):
    rounds: int
    per_round: int
    mode: str = "proxy"
    docking_limit: int = 0


class AgentState(BaseModel):
    target_id: str
    pocket_summary: str
    round_index: int = 0
    candidates: List[Candidate] = Field(default_factory=list)
    seen_smiles: Set[str] = Field(default_factory=set)
    history: List[Dict[str, Any]] = Field(default_factory=list)

    def ranked_candidates(self) -> List[Candidate]:
        return sorted(self.candidates, key=lambda item: item.score.total, reverse=True)

    def best_score(self) -> float:
        ranked = self.ranked_candidates()
        return ranked[0].score.total if ranked else 0.0
