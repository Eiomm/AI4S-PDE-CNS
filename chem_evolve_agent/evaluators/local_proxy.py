from __future__ import annotations

from chem_evolve_agent.chemistry.filters import evaluate_properties
from chem_evolve_agent.chemistry.smiles import is_valid_smiles
from chem_evolve_agent.chemistry.scoring import clamp01
from chem_evolve_agent.models import Score


def score_smiles_proxy(smiles: str) -> Score:
    if not is_valid_smiles(smiles):
        return Score(molecule_score=0.0, route_score=0.0, penalties=["invalid_smiles"])
    heavy_atom_proxy = sum(1 for char in smiles if char.isalpha() and char.isupper())
    size_score = min(heavy_atom_proxy / 35.0, 1.0)
    simplicity_bonus = max(0.0, 1.0 - max(0, len(smiles) - 80) / 80.0)
    molecule_score = round(0.65 * size_score + 0.35 * simplicity_bonus, 4)

    properties = evaluate_properties(smiles)
    penalties = list(properties.penalties)
    qed = properties.metrics.get("qed")
    if qed is not None:
        molecule_score = round(0.65 * molecule_score + 0.35 * qed, 4)
    logp = properties.metrics.get("logp")
    if logp is not None:
        cns_logp_score = clamp01(1.0 - abs(logp - 2.4) / 4.0)
        molecule_score = round(0.8 * molecule_score + 0.2 * cns_logp_score, 4)
    if penalties:
        molecule_score = round(max(0.0, molecule_score - 0.05 * len(penalties)), 4)
    return Score(molecule_score=molecule_score, route_score=0.0, qed=qed, penalties=penalties)
