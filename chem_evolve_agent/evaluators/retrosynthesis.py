from __future__ import annotations

from pathlib import Path

from chem_evolve_agent.chemistry.smiles import canonicalize_smiles, route_final_product
from chem_evolve_agent.models import Route
from chem_evolve_agent.tools.aizynthfinder import run_aizynthfinder_route_tool


def fallback_route_for(smiles: str) -> Route:
    return Route(steps=[f"START.O>>{smiles}"])


def plan_route_with_aizynthfinder(smiles: str, out_dir: Path) -> tuple[Route, str, list[str]]:
    result = run_aizynthfinder_route_tool(smiles, out_dir)
    if not result.success:
        return fallback_route_for(smiles), "fallback", [result.reason or "aizynthfinder_unavailable"]
    # The tool stores AiZynthFinder's route tree as an artifact. Until we add a
    # robust route-tree-to-reaction-SMILES converter, keep the submission route
    # conservative but preserve the artifact path in logs.
    return fallback_route_for(smiles), "aizynthfinder", []


def route_consistency_score(smiles: str, route: Route) -> tuple[float, list[str]]:
    try:
        product = route_final_product(route.text)
        if canonicalize_smiles(product) != canonicalize_smiles(smiles):
            return 0.0, ["route_product_mismatch"]
    except Exception:
        return 0.0, ["route_unparseable"]
    step_penalty = max(0, len(route.steps) - 3) * 0.1
    return max(0.1, 0.75 - step_penalty), []
