from __future__ import annotations

import time
from typing import Any

from chem_evolve_agent.chemistry.smiles import _rdkit
from chem_evolve_agent.tools.base import ToolResult, ToolStatus


def run_rdkit_property_tool(smiles: str) -> ToolResult:
    start = time.monotonic()
    Chem = _rdkit()
    if Chem is None:
        return ToolResult(
            tool_name="rdkit_property_tool",
            status=ToolStatus.SKIPPED,
            elapsed_seconds=time.monotonic() - start,
            reason="rdkit_not_installed",
        )

    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ToolResult(
            tool_name="rdkit_property_tool",
            status=ToolStatus.ERROR,
            elapsed_seconds=time.monotonic() - start,
            reason="invalid_smiles",
            metrics={"input_smiles": smiles},
        )
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return ToolResult(
            tool_name="rdkit_property_tool",
            status=ToolStatus.ERROR,
            elapsed_seconds=time.monotonic() - start,
            reason=f"sanitize_failed:{exc}",
            metrics={"input_smiles": smiles},
        )

    metrics: dict[str, Any] = {
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "qed": float(QED.qed(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
    }
    warnings = _druglike_warnings(metrics)
    return ToolResult(
        tool_name="rdkit_property_tool",
        status=ToolStatus.OK,
        elapsed_seconds=time.monotonic() - start,
        metrics=metrics,
        warnings=warnings,
    )


def _druglike_warnings(metrics: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if metrics["mw"] < 120 or metrics["mw"] > 650:
        warnings.append("mw_out_of_range")
    if metrics["logp"] < -1.5 or metrics["logp"] > 6.0:
        warnings.append("logp_out_of_range")
    if metrics["hbd"] > 5:
        warnings.append("hbd_high")
    if metrics["hba"] > 10:
        warnings.append("hba_high")
    if metrics["rotatable_bonds"] > 12:
        warnings.append("too_flexible")
    if abs(metrics["formal_charge"]) > 1:
        warnings.append("large_formal_charge")
    return warnings
