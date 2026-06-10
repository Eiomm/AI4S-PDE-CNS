from __future__ import annotations

from typing import Tuple

from pydantic import BaseModel

from chem_evolve_agent.target.pdb_loader import PdbTarget


class PocketSummary(BaseModel):
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    box_size: Tuple[float, float, float] = (22.0, 22.0, 22.0)
    summary: str = "conservative_fallback_box"
    method: str = "fallback"


def summarize_pocket(target: PdbTarget) -> PocketSummary:
    if target.atom_count == 0:
        return PocketSummary(summary="smoke_mode_no_atoms", method="empty_target")
    center = tuple(round(value, 3) for value in target.center)
    bounds = target.bounds
    if bounds is None:
        return PocketSummary(center=center, summary=f"fallback_box_atoms_{target.atom_count}")
    (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
    span = (max_x - min_x, max_y - min_y, max_z - min_z)
    # No bound ligand is present in the official preliminary target, so use a
    # conservative local box around the protein centroid rather than docking
    # over the whole protein envelope.
    box_size = tuple(round(min(max(axis * 0.35, 18.0), 28.0), 3) for axis in span)
    return PocketSummary(
        center=center,
        box_size=box_size,
        summary=f"centroid_fallback_atoms_{target.atom_count}_residues_{len(target.residues)}",
        method="centroid_fallback_no_ligand",
    )
