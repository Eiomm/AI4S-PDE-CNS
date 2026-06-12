from __future__ import annotations

import zipfile
import csv
import shutil
from pathlib import Path

from chem_evolve_agent.chem_ops import canonicalize_smiles, validate_route
from chem_evolve_agent.models import Candidate, Route


def clean_managed_outputs(out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    removed: list[str] = []
    for pattern in ("result*.csv", "result*.log", "result*.zip"):
        for path in sorted(out_dir.glob(pattern)):
            if path.is_file():
                path.unlink()
                removed.append(path.name)
    for name in ("generation", "routes", "docking", "docking_feedback", "llm_io", "work"):
        path = out_dir / name
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(f"{name}/")
    return removed


def write_single_target_result(
    out_dir: Path,
    stem: str,
    candidates: list[Candidate],
    log_lines: list[str],
    zip_name: str | None = None,
    write_zip: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    log_path = out_dir / f"{stem}.log"
    zip_path = out_dir / (zip_name or f"{stem}.zip")

    if not candidates:
        raise ValueError("at least one candidate is required")

    rows = []
    for candidate in sorted(candidates, key=lambda item: item.score.total, reverse=True):
        rows.append(_submission_row(candidate))
    if not rows:
        raise ValueError("at least one candidate with a non-empty route is required")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mol_smiles", "route"])
        writer.writeheader()
        writer.writerows(rows)
    log_path.write_text("\n".join(log_lines) + "\n")

    if not write_zip:
        return csv_path

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=csv_path.name)
        zf.write(log_path, arcname=log_path.name)
    return zip_path


def write_final_result_zip(out_dir: Path, stems: list[str], zip_name: str = "result.zip") -> Path:
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for stem in stems:
            csv_path = out_dir / f"{stem}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"missing final CSV for {stem}: {csv_path}")
            _validate_submission_csv(csv_path)
            zf.write(csv_path, arcname=csv_path.name)
    return zip_path


def _submission_row(candidate: Candidate) -> dict[str, str]:
    route_text = candidate.route.text.strip()
    if not route_text:
        raise ValueError(f"candidate has empty route: {candidate.mol_smiles}")
    validation = validate_route(candidate.mol_smiles, candidate.route)
    if validation.route_score <= 0:
        penalties = ",".join(validation.penalties) or "unknown"
        raise ValueError(f"candidate route is invalid for submission: {candidate.mol_smiles}; penalties={penalties}")
    if candidate.score.molecule_score <= 0 or candidate.score.route_score <= 0:
        raise ValueError(f"candidate score is not submission-eligible: {candidate.mol_smiles}")
    return {"mol_smiles": canonicalize_smiles(candidate.mol_smiles), "route": route_text}


def _validate_submission_csv(csv_path: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"submission csv is empty: {csv_path}")
    if list(rows[0].keys()) != ["mol_smiles", "route"]:
        raise ValueError(f"submission csv has invalid columns: {csv_path}")
    for row_index, row in enumerate(rows):
        route_text = str(row["route"]).strip()
        if not route_text:
            raise ValueError(f"{csv_path}:{row_index}: empty route")
        route = Route(steps=[step.strip() for step in route_text.split(",") if step.strip()])
        validation = validate_route(str(row["mol_smiles"]), route)
        if validation.route_score <= 0:
            penalties = ",".join(validation.penalties) or "unknown"
            raise ValueError(f"{csv_path}:{row_index}: invalid route; penalties={penalties}")
