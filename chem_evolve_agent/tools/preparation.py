from __future__ import annotations

from pathlib import Path

from chem_evolve_agent.chemistry.smiles import _rdkit
from chem_evolve_agent.tools.base import ToolResult, ToolStatus, find_executable, run_command, safe_name


def run_obabel_meeko_prepare_tool(
    *,
    kind: str,
    out_dir: Path,
    smiles: str | None = None,
    input_path: Path | None = None,
    name: str | None = None,
    timeout: int = 180,
) -> ToolResult:
    if kind == "ligand":
        if smiles is None:
            return ToolResult(tool_name="obabel_meeko_prepare_tool", status=ToolStatus.ERROR, reason="missing_smiles")
        return prepare_ligand_pdbqt(smiles=smiles, out_dir=out_dir, name=name, timeout=timeout)
    if kind == "receptor":
        if input_path is None:
            return ToolResult(tool_name="obabel_meeko_prepare_tool", status=ToolStatus.ERROR, reason="missing_input_path")
        return prepare_receptor_pdbqt(receptor_path=input_path, out_dir=out_dir, name=name, timeout=timeout)
    return ToolResult(
        tool_name="obabel_meeko_prepare_tool",
        status=ToolStatus.ERROR,
        reason=f"unsupported_kind:{kind}",
    )


def prepare_ligand_pdbqt(smiles: str, out_dir: Path, name: str | None = None, timeout: int = 180) -> ToolResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(name or smiles)
    sdf_path = out_dir / f"{stem}.sdf"
    pdbqt_path = out_dir / f"{stem}.pdbqt"

    sdf_result = _write_ligand_sdf(smiles, sdf_path)
    if not sdf_result.success:
        return sdf_result

    meeko = find_executable("mk_prepare_ligand.py", "mk_prepare_ligand")
    if meeko:
        result = run_command([meeko, "-i", str(sdf_path), "-o", str(pdbqt_path)], timeout=timeout)
        result.tool_name = "obabel_meeko_prepare_tool"
        result.artifacts.update({"sdf": str(sdf_path), "pdbqt": str(pdbqt_path)})
        if result.success and not pdbqt_path.exists():
            result.status = ToolStatus.ERROR
            result.reason = "ligand_pdbqt_not_written"
        if result.success:
            return result
        meeko_warning = f"meeko_ligand_failed:{result.reason}"
    else:
        meeko_warning = ""

    obabel = find_executable("obabel", "babel")
    if obabel:
        result = run_command(
            [obabel, "-isdf", str(sdf_path), "-opdbqt", "-O", str(pdbqt_path), "--partialcharge", "gasteiger"],
            timeout=timeout,
        )
        result.tool_name = "obabel_meeko_prepare_tool"
        result.artifacts.update({"sdf": str(sdf_path), "pdbqt": str(pdbqt_path)})
        if result.success and not pdbqt_path.exists():
            result.status = ToolStatus.ERROR
            result.reason = "ligand_pdbqt_not_written"
        if meeko_warning:
            result.warnings.append(meeko_warning)
        return result

    return ToolResult(
        tool_name="obabel_meeko_prepare_tool",
        status=ToolStatus.SKIPPED,
        reason="ligand_preparation_tool_not_installed",
        artifacts={"sdf": str(sdf_path)},
        warnings=[warning for warning in [meeko_warning, "install_meeko_or_openbabel_for_ligand_pdbqt"] if warning],
    )


def prepare_receptor_pdbqt(receptor_path: Path, out_dir: Path, name: str | None = None, timeout: int = 240) -> ToolResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not receptor_path.exists():
        return ToolResult(
            tool_name="obabel_meeko_prepare_tool",
            status=ToolStatus.ERROR,
            reason=f"receptor_not_found:{receptor_path}",
        )
    if receptor_path.suffix.lower() == ".pdbqt":
        return ToolResult(
            tool_name="obabel_meeko_prepare_tool",
            status=ToolStatus.OK,
            reason="already_pdbqt",
            artifacts={"pdbqt": str(receptor_path)},
        )

    stem = safe_name(name or receptor_path.stem)
    pdbqt_path = out_dir / f"{stem}.pdbqt"
    meeko = find_executable("mk_prepare_receptor.py", "mk_prepare_receptor")
    if meeko:
        prefix = out_dir / stem
        strict_result = _run_receptor_meeko(
            [meeko, "-i", str(receptor_path), "-o", str(prefix), "-p"],
            pdbqt_path=pdbqt_path,
            timeout=timeout,
        )
        if strict_result.success:
            return strict_result

        tolerant_result = _run_receptor_meeko(
            [meeko, "-i", str(receptor_path), "-o", str(prefix), "-p", "-a", "--default_altloc", "A"],
            pdbqt_path=pdbqt_path,
            timeout=timeout,
        )
        tolerant_result.warnings.append(f"strict_meeko_receptor_failed:{strict_result.reason}")
        if tolerant_result.success:
            return tolerant_result
        meeko_warning = f"meeko_receptor_failed:{tolerant_result.reason}"
    else:
        meeko_warning = ""

    obabel = find_executable("obabel", "babel")
    if obabel:
        result = run_command([obabel, "-ipdb", str(receptor_path), "-opdbqt", "-O", str(pdbqt_path), "-xr"], timeout=timeout)
        result.tool_name = "obabel_meeko_prepare_tool"
        result.artifacts["pdbqt"] = str(pdbqt_path)
        if result.success and not pdbqt_path.exists():
            result.status = ToolStatus.ERROR
            result.reason = "receptor_pdbqt_not_written"
        if meeko_warning:
            result.warnings.append(meeko_warning)
        return result

    return ToolResult(
        tool_name="obabel_meeko_prepare_tool",
        status=ToolStatus.SKIPPED,
        reason="receptor_preparation_tool_not_installed",
        warnings=[warning for warning in [meeko_warning, "install_meeko_or_openbabel_for_receptor_pdbqt"] if warning],
    )


def _run_receptor_meeko(command: list[str], *, pdbqt_path: Path, timeout: int) -> ToolResult:
    result = run_command(command, timeout=timeout)
    result.tool_name = "obabel_meeko_prepare_tool"
    result.artifacts["pdbqt"] = str(pdbqt_path)
    if result.success and not pdbqt_path.exists():
        result.status = ToolStatus.ERROR
        result.reason = "receptor_pdbqt_not_written"
    return result


def _write_ligand_sdf(smiles: str, sdf_path: Path) -> ToolResult:
    Chem = _rdkit()
    if Chem is None:
        return ToolResult(
            tool_name="obabel_meeko_prepare_tool",
            status=ToolStatus.SKIPPED,
            reason="rdkit_not_installed_for_ligand_sdf",
        )

    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ToolResult(
            tool_name="obabel_meeko_prepare_tool",
            status=ToolStatus.ERROR,
            reason="invalid_smiles",
        )
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 17
    embed_code = AllChem.EmbedMolecule(mol, params)
    if embed_code != 0:
        return ToolResult(
            tool_name="obabel_meeko_prepare_tool",
            status=ToolStatus.ERROR,
            reason=f"rdkit_embed_failed:{embed_code}",
        )
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()
    return ToolResult(
        tool_name="obabel_meeko_prepare_tool",
        status=ToolStatus.OK,
        artifacts={"sdf": str(sdf_path)},
    )
