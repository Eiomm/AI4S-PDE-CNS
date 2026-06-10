from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chem_evolve_agent.chemistry.filters import evaluate_properties
from chem_evolve_agent.chemistry.smiles import canonicalize_smiles, route_final_product
from chem_evolve_agent.evaluators.local_proxy import score_smiles_proxy
from chem_evolve_agent.evaluators.retrosynthesis import route_consistency_score


@dataclass
class RunMetrics:
    run_dir: str
    status: str
    candidate_count: int = 0
    unique_smiles: int = 0
    valid_smiles: int = 0
    route_consistent: int = 0
    scaffold_count: int = 0
    best_total: float = 0.0
    avg_total: float = 0.0
    best_molecule_score: float = 0.0
    best_route_score: float = 0.0
    best_smiles: str = ""
    invalid_events: int = 0
    duplicate_events: int = 0
    penalty_events: int = 0
    llm_calls: int = 0
    inspect_ok: bool = False
    generated_by: dict[str, int] = field(default_factory=dict)
    objective: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_run_directory(run_dir: Path) -> RunMetrics:
    run_dir = run_dir.resolve()
    done = _read_done(run_dir / "harness.done")
    metrics = RunMetrics(run_dir=str(run_dir), status="pass" if done.get("exit_code") == "0" else "fail")

    rows = _read_candidates(run_dir)
    totals: list[float] = []
    seen: set[str] = set()
    scaffolds: set[str] = set()
    for row in rows:
        smiles = str(row.get("mol_smiles", "")).strip()
        route = str(row.get("route", "")).strip()
        if not smiles:
            continue
        metrics.candidate_count += 1
        try:
            canonical = canonicalize_smiles(smiles)
        except Exception:
            continue
        metrics.valid_smiles += 1
        seen.add(canonical)
        scaffold = _scaffold(canonical)
        if scaffold:
            scaffolds.add(scaffold)

        route_score = 0.0
        try:
            product = canonicalize_smiles(route_final_product(route))
            if product == canonical:
                metrics.route_consistent += 1
            route_score, _ = route_consistency_score(canonical, _RouteText(route))
        except Exception:
            route_score = 0.0

        score = score_smiles_proxy(canonical)
        score.route_score = route_score
        total = score.total
        totals.append(total)
        if total > metrics.best_total:
            metrics.best_total = round(total, 4)
            metrics.best_molecule_score = round(score.molecule_score, 4)
            metrics.best_route_score = round(score.route_score, 4)
            metrics.best_smiles = canonical

    metrics.unique_smiles = len(seen)
    metrics.scaffold_count = len(scaffolds)
    metrics.avg_total = round(sum(totals) / len(totals), 4) if totals else 0.0

    events = _read_jsonl_events(run_dir / "pipeline.log")
    for event in events:
        name = event.get("event")
        if name == "skip_invalid_smiles":
            metrics.invalid_events += 1
        elif name == "skip_duplicate_smiles":
            metrics.duplicate_events += 1
        elif name == "evaluate" and event.get("penalties"):
            metrics.penalty_events += len(event.get("penalties") or [])
        elif name == "generate":
            generator = str(event.get("generator", "unknown"))
            metrics.generated_by[generator] = metrics.generated_by.get(generator, 0) + int(event.get("count") or 0)

    metrics.llm_calls = _count_lines(run_dir / "llm_io")
    metrics.inspect_ok = "OK " in (run_dir / "inspect.log").read_text(encoding="utf-8", errors="ignore") if (run_dir / "inspect.log").exists() else False
    metrics.objective = _objective(metrics)
    return metrics


class _RouteText:
    def __init__(self, text: str):
        self.text = text
        self.steps = text.split(",") if text else []


def _objective(metrics: RunMetrics) -> float:
    if metrics.candidate_count == 0 or not metrics.inspect_ok:
        return 0.0
    validity = metrics.valid_smiles / max(1, metrics.candidate_count)
    route_ok = metrics.route_consistent / max(1, metrics.candidate_count)
    diversity = min(metrics.scaffold_count / max(1, metrics.candidate_count), 1.0)
    penalty = min(metrics.penalty_events / max(1, metrics.candidate_count * 4), 1.0)
    return round(0.55 * metrics.best_total + 0.20 * metrics.avg_total + 0.10 * validity + 0.10 * route_ok + 0.10 * diversity - 0.05 * penalty, 4)


def _read_done(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def _read_candidates(run_dir: Path) -> list[dict[str, str]]:
    for name in ("candidates.csv", "result.csv"):
        path = run_dir / name
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
    return []


def _read_jsonl_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _count_lines(directory: Path) -> int:
    if not directory.exists():
        return 0
    count = 0
    for path in directory.glob("*.jsonl"):
        with path.open(encoding="utf-8", errors="ignore") as handle:
            count += sum(1 for _ in handle)
    return count


def _scaffold(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        return smiles[:16]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        props = evaluate_properties(smiles).metrics
        return f"acyclic:{int(props.get('mw', 0) // 50)}:{int(props.get('hba', 0))}:{int(props.get('hbd', 0))}"
    return Chem.MolToSmiles(scaffold, canonical=True)
