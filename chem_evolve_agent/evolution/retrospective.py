from __future__ import annotations

from chem_evolve_agent.models import Candidate


def summarize_candidates(candidates: list[Candidate]) -> dict[str, object]:
    if not candidates:
        return {"top_score": 0.0, "penalties": []}
    top = max(candidates, key=lambda item: item.score.total)
    penalties = sorted({penalty for item in candidates for penalty in item.score.penalties})
    return {
        "top_smiles": top.mol_smiles,
        "top_score": top.score.total,
        "penalties": penalties,
        "candidate_count": len(candidates),
    }
