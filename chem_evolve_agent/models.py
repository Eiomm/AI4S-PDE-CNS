from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Route(BaseModel):
    steps: List[str] = Field(default_factory=list)
    starting_materials: List[str] = Field(default_factory=list)
    intermediates: List[str] = Field(default_factory=list)
    source: str = "unknown"
    confidence: str = "unknown"
    solved: Optional[bool] = None

    @property
    def text(self) -> str:
        return ",".join(self.steps)


class Score(BaseModel):
    molecule_score: float
    route_score: float
    binding_score: Optional[float] = None
    binding_source: str = "proxy"
    validity_score: Optional[float] = None
    route_validity_score: Optional[float] = None
    starting_material_availability_score: Optional[float] = None
    step_penalty_score: Optional[float] = None
    convergence_score: Optional[float] = None
    balance_score: Optional[float] = None
    docking_energy: Optional[float] = None
    qed: Optional[float] = None
    sa: Optional[float] = None
    property_prior_score: Optional[float] = None
    penalties: List[str] = Field(default_factory=list)

    @property
    def total(self) -> float:
        return round(0.60 * _clamp01(self.molecule_score) + 0.40 * _clamp01(self.route_score), 4)


class Candidate(BaseModel):
    mol_smiles: str
    route: Route
    score: Score
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
