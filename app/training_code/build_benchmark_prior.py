#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chem_evolve_agent.chem_ops import canonicalize_smiles, property_metrics


FEATURES = ["mw", "logp", "qed", "tpsa", "hbd", "hba", "rotatable_bonds", "heavy_atoms", "ring_count", "aromatic_rings", "sascore_raw"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aggregate benchmark property priors without storing benchmark SMILES.")
    parser.add_argument("--manifest", default="data/benchmarks/manifest.yaml")
    parser.add_argument("--output", default="data/benchmarks/benchmark_prior.json")
    args = parser.parse_args()

    manifest_path = (ROOT / args.manifest).resolve()
    output_path = (ROOT / args.output).resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    benchmark_root = Path(manifest.get("root", manifest_path.parent))
    if not benchmark_root.is_absolute():
        benchmark_root = (manifest_path.parent / benchmark_root).resolve()

    targets: list[dict[str, Any]] = []
    active_rows: list[dict[str, float]] = []
    decoy_rows: list[dict[str, float]] = []
    for benchmark in manifest.get("benchmarks", []):
        for target in benchmark.get("targets", []):
            active_metrics = _load_metrics(benchmark_root / target["actives_smi"])
            decoy_metrics = _load_metrics(benchmark_root / target["decoys_smi"])
            active_rows.extend(active_metrics)
            decoy_rows.extend(decoy_metrics)
            targets.append(
                {
                    "id": target["id"],
                    "active_count": len(active_metrics),
                    "decoy_count": len(decoy_metrics),
                    "active_summary": _summarize(active_metrics),
                    "decoy_summary": _summarize(decoy_metrics),
                }
            )

    output = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path.relative_to(ROOT)),
        "feature_names": FEATURES,
        "active_summary": _summarize(active_rows),
        "decoy_summary": _summarize(decoy_rows),
        "targets": targets,
        "leakage_guard": [
            "This file stores aggregate descriptor statistics only.",
            "It intentionally does not store benchmark molecular strings or target-specific final molecules.",
            "Use it as a property prior for generation and scoring, not as a fixed answer library.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE {output_path}")


def _load_metrics(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        smiles = line.strip().split()[0] if line.strip() else ""
        if not smiles:
            continue
        try:
            canonical = canonicalize_smiles(smiles)
            metrics, _ = property_metrics(canonical)
        except Exception:
            continue
        rows.append({feature: round(float(metrics[feature]), 4) for feature in FEATURES if feature in metrics})
    return rows


def _summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for feature in FEATURES:
        values = sorted(row[feature] for row in rows if feature in row)
        if not values:
            continue
        summary[feature] = {
            "median": round(statistics.median(values), 4),
            "q1": round(_quantile(values, 0.25), 4),
            "q3": round(_quantile(values, 0.75), 4),
        }
    return summary


def _quantile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


if __name__ == "__main__":
    main()
