from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Route(BaseModel):
    steps: List[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return ",".join(self.steps)


class Score(BaseModel):
    molecule_score: float
    route_score: float
    docking_energy: Optional[float] = None
    qed: Optional[float] = None
    sa: Optional[float] = None
    penalties: List[str] = Field(default_factory=list)

    @property
    def total(self) -> float:
        return 0.60 * self.molecule_score + 0.40 * self.route_score


class Candidate(BaseModel):
    mol_smiles: str
    route: Route
    score: Score
    metadata: Dict[str, Any] = Field(default_factory=dict)
