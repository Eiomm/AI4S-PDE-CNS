from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from chem_evolve_agent.chemistry.smiles import _rdkit
from chem_evolve_agent.tools.rdkit_property import run_rdkit_property_tool


class PropertyFilterResult(BaseModel):
    valid: bool
    metrics: Dict[str, float] = Field(default_factory=dict)
    penalties: List[str] = Field(default_factory=list)


def evaluate_properties(smiles: str) -> PropertyFilterResult:
    Chem = _rdkit()
    if Chem is None:
        if not smiles or " " in smiles or ">>" in smiles or "not-a" in smiles:
            return PropertyFilterResult(valid=False, penalties=["invalid_smiles"])
        heavy_atoms = sum(1 for char in smiles if char.isalpha() and char.isupper())
        penalties = []
        if heavy_atoms > 70:
            penalties.append("extreme_size")
        return PropertyFilterResult(
            valid=not penalties,
            metrics={"heavy_atoms": float(heavy_atoms)},
            penalties=penalties,
        )

    result = run_rdkit_property_tool(smiles)
    if not result.success:
        reason = result.reason or "rdkit_property_failed"
        penalty = "invalid_smiles" if reason == "invalid_smiles" else reason
        return PropertyFilterResult(valid=False, metrics=result.metrics, penalties=[penalty])

    numeric_metrics = {
        key: float(value)
        for key, value in result.metrics.items()
        if isinstance(value, (int, float))
    }
    return PropertyFilterResult(valid=not result.warnings, metrics=numeric_metrics, penalties=list(result.warnings))
