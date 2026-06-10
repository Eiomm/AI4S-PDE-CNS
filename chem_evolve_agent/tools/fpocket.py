from __future__ import annotations

import re
from pathlib import Path

from chem_evolve_agent.tools.base import ToolResult, ToolStatus, find_executable, run_command


def run_fpocket_tool(receptor_pdb: Path, out_dir: Path, timeout: int = 300) -> ToolResult:
    fpocket = find_executable("fpocket")
    if fpocket is None:
        return ToolResult(
            tool_name="fpocket_tool",
            status=ToolStatus.SKIPPED,
            reason="fpocket_not_installed",
        )
    if not receptor_pdb.exists():
        return ToolResult(
            tool_name="fpocket_tool",
            status=ToolStatus.ERROR,
            reason=f"receptor_not_found:{receptor_pdb}",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_command([fpocket, "-f", str(receptor_pdb.resolve())], timeout=timeout, cwd=out_dir)
    result.tool_name = "fpocket_tool"
    output_dir = _find_fpocket_output(out_dir, receptor_pdb)
    if output_dir:
        result.artifacts["fpocket_output_dir"] = str(output_dir)
        info_file = _find_info_file(output_dir)
        if info_file:
            result.artifacts["pocket_info"] = str(info_file)
            result.metrics.update(_parse_pocket_info(info_file))
    if result.success and not output_dir:
        result.warnings.append("fpocket_output_not_found")
    return result


def _find_fpocket_output(out_dir: Path, receptor_pdb: Path) -> Path | None:
    candidates = sorted(out_dir.glob(f"{receptor_pdb.stem}_out"))
    if not candidates:
        candidates = sorted(out_dir.glob("*_out"))
    return candidates[0] if candidates else None


def _find_info_file(output_dir: Path) -> Path | None:
    files = sorted(output_dir.glob("*_info.txt"))
    if not files:
        files = sorted(output_dir.rglob("*info*.txt"))
    return files[0] if files else None


def _parse_pocket_info(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    metrics: dict[str, float | int] = {}
    pocket_match = re.search(r"Pocket\s+(\d+)", text, re.IGNORECASE)
    if pocket_match:
        metrics["best_pocket_id"] = int(pocket_match.group(1))
    patterns = {
        "fpocket_score": r"Score\s*:\s*(-?\d+(?:\.\d+)?)",
        "drug_score": r"Drug\s*Score\s*:\s*(-?\d+(?:\.\d+)?)",
        "volume": r"Volume\s*:\s*(-?\d+(?:\.\d+)?)",
        "hydrophobicity_score": r"Hydrophobicity\s*Score\s*:\s*(-?\d+(?:\.\d+)?)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics[key] = float(match.group(1))
    return metrics
