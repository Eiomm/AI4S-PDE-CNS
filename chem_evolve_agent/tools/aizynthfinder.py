from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

from chem_evolve_agent.tools.base import ToolResult, ToolStatus, find_executable, run_command, safe_name


def run_aizynthfinder_route_tool(
    smiles: str,
    out_dir: Path,
    *,
    config_path: Path | None = None,
    timeout: int = 300,
) -> ToolResult:
    command_path = find_executable("aizynthcli", "aizynthfinder")
    if command_path is None:
        return ToolResult(
            tool_name="aizynthfinder_route_tool",
            status=ToolStatus.SKIPPED,
            reason="aizynthfinder_not_installed",
        )

    resolved_config = config_path or _config_from_env()
    if resolved_config is None or not resolved_config.exists():
        return ToolResult(
            tool_name="aizynthfinder_route_tool",
            status=ToolStatus.SKIPPED,
            reason="aizynthfinder_config_not_found",
            warnings=["set_AIZYNTHFINDER_CONFIG_to_enable_real_retrosynthesis"],
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_json = out_dir / f"{safe_name(smiles)}_aizynth.json.gz"
    command = [
        command_path,
        "--config",
        str(resolved_config),
        "--smiles",
        smiles,
        "--output",
        str(output_json),
    ]
    result = run_command(command, timeout=timeout)
    result.tool_name = "aizynthfinder_route_tool"
    if output_json.exists():
        result.artifacts["routes_json"] = str(output_json)
        result.metrics.update(_extract_route_metrics(output_json))
    return result


def _config_from_env() -> Path | None:
    value = os.getenv("AIZYNTHFINDER_CONFIG")
    return Path(value) if value else None


def _extract_route_metrics(path: Path) -> dict[str, Any]:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
                data = json.load(handle)
        else:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {"routes_parsed": False}

    metrics: dict[str, Any] = {"routes_parsed": True}
    if isinstance(data, dict):
        if "solved" in data:
            metrics["solved"] = bool(data["solved"])
        if "top_score" in data:
            metrics["top_score"] = data["top_score"]
        if "number_of_routes" in data:
            metrics["number_of_routes"] = data["number_of_routes"]
        if "trees" in data and isinstance(data["trees"], list):
            metrics["number_of_routes"] = len(data["trees"])
    elif isinstance(data, list):
        metrics["number_of_routes"] = len(data)
    return metrics
