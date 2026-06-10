from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Tuple

from chem_evolve_agent.tools.base import ToolResult, ToolStatus, find_executable, has_python_module, run_command, safe_name


def run_vina_docking_tool(
    *,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    out_dir: Path,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    box_size: Tuple[float, float, float] = (22.0, 22.0, 22.0),
    score_only: bool = True,
    exhaustiveness: int = 8,
    timeout: int = 180,
) -> ToolResult:
    vina = find_executable("vina")
    if not receptor_pdbqt.exists():
        return ToolResult(
            tool_name="vina_docking_tool",
            status=ToolStatus.ERROR,
            reason=f"receptor_pdbqt_not_found:{receptor_pdbqt}",
        )
    if not ligand_pdbqt.exists():
        return ToolResult(
            tool_name="vina_docking_tool",
            status=ToolStatus.ERROR,
            reason=f"ligand_pdbqt_not_found:{ligand_pdbqt}",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pose_path = out_dir / f"{safe_name(ligand_pdbqt.stem)}_vina.pdbqt"
    if vina is None:
        if not has_python_module("vina"):
            return ToolResult(tool_name="vina_docking_tool", status=ToolStatus.SKIPPED, reason="vina_not_installed")
        return _run_vina_python_api(
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=ligand_pdbqt,
            pose_path=pose_path,
            center=center,
            box_size=box_size,
            score_only=score_only,
            exhaustiveness=exhaustiveness,
        )

    command = [
        vina,
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(box_size[0]),
        "--size_y",
        str(box_size[1]),
        "--size_z",
        str(box_size[2]),
        "--exhaustiveness",
        str(exhaustiveness),
    ]
    if score_only:
        command.append("--score_only")
    else:
        command.extend(["--out", str(pose_path)])
    result = run_command(command, timeout=timeout)
    result.tool_name = "vina_docking_tool"
    result.metrics["docking_energy"] = _parse_vina_energy(result.stdout + "\n" + result.stderr)
    if pose_path.exists():
        result.artifacts["pose_pdbqt"] = str(pose_path)
    if result.success and result.metrics["docking_energy"] is None:
        result.warnings.append("vina_energy_not_parsed")
    return result


def _run_vina_python_api(
    *,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    pose_path: Path,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
    score_only: bool,
    exhaustiveness: int,
) -> ToolResult:
    start = time.monotonic()
    try:
        from vina import Vina

        v = Vina(sf_name="vina", verbosity=0)
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_file(str(ligand_pdbqt))
        v.compute_vina_maps(center=list(center), box_size=list(box_size))
        if score_only:
            energy = float(v.score()[0])
            artifacts: dict[str, str] = {}
        else:
            v.dock(exhaustiveness=exhaustiveness, n_poses=1)
            energy = float(v.energies(n_poses=1)[0][0])
            v.write_pose(str(pose_path), overwrite=True)
            artifacts = {"pose_pdbqt": str(pose_path)}
    except Exception as exc:
        return ToolResult(
            tool_name="vina_docking_tool",
            status=ToolStatus.ERROR,
            command=["python:vina"],
            elapsed_seconds=time.monotonic() - start,
            reason=str(exc),
        )
    return ToolResult(
        tool_name="vina_docking_tool",
        status=ToolStatus.OK,
        command=["python:vina"],
        elapsed_seconds=time.monotonic() - start,
        metrics={"docking_energy": energy},
        artifacts=artifacts,
    )


def _parse_vina_energy(text: str) -> float | None:
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
