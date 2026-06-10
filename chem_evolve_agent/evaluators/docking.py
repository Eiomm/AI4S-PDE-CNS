from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from chem_evolve_agent.tools.preparation import prepare_ligand_pdbqt, prepare_receptor_pdbqt
from chem_evolve_agent.tools.vina import run_vina_docking_tool


class DockingResult(BaseModel):
    attempted: bool
    success: bool
    docking_energy: Optional[float] = None
    output_path: Optional[Path] = None
    command: List[str] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = 0.0
    penalties: List[str] = Field(default_factory=list)
    reason: Optional[str] = None


def dock_smiles_with_vina(
    smiles: str,
    receptor_pdb: Path,
    out_dir: Path,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    box_size: Tuple[float, float, float] = (22.0, 22.0, 22.0),
) -> DockingResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = out_dir / "prepared"
    receptor_result = prepare_receptor_pdbqt(receptor_pdb, prep_dir / "receptor", name=receptor_pdb.stem)
    if not receptor_result.success:
        return DockingResult(
            attempted=False,
            success=False,
            command=receptor_result.command,
            stdout=receptor_result.stdout,
            stderr=receptor_result.stderr,
            elapsed_seconds=receptor_result.elapsed_seconds,
            penalties=["docking_skipped"],
            reason=receptor_result.reason,
        )

    ligand_result = prepare_ligand_pdbqt(smiles, prep_dir / "ligands")
    if not ligand_result.success:
        return DockingResult(
            attempted=False,
            success=False,
            command=ligand_result.command,
            stdout=ligand_result.stdout,
            stderr=ligand_result.stderr,
            elapsed_seconds=receptor_result.elapsed_seconds + ligand_result.elapsed_seconds,
            penalties=["docking_skipped"],
            reason=ligand_result.reason,
        )

    receptor_pdbqt = Path(receptor_result.artifacts["pdbqt"])
    ligand_pdbqt = Path(ligand_result.artifacts["pdbqt"])
    result = run_vina_docking_tool(
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=ligand_pdbqt,
        out_dir=out_dir / "vina",
        center=center,
        box_size=box_size,
        score_only=False,
    )
    return DockingResult(
        attempted=not result.skipped,
        success=result.success,
        docking_energy=result.metrics.get("docking_energy"),
        output_path=Path(result.artifacts["pose_pdbqt"]) if "pose_pdbqt" in result.artifacts else None,
        command=result.command,
        stdout=result.stdout,
        stderr=result.stderr,
        elapsed_seconds=receptor_result.elapsed_seconds + ligand_result.elapsed_seconds + result.elapsed_seconds,
        penalties=[] if result.success else ["docking_skipped" if result.skipped else "docking_failed"],
        reason=result.reason,
    )


def _parse_vina_energy(text: str) -> Optional[float]:
    patterns = [
        r"Estimated Free Energy of Binding\s*:\s*(-?\d+(?:\.\d+)?)",
        r"Affinity:\s*(-?\d+(?:\.\d+)?)",
        r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return float(match.group(1))
    return None
