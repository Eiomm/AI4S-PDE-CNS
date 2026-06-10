from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from chem_evolve_agent.models import Candidate


def write_single_target_result(
    out_dir: Path,
    stem: str,
    candidates: list[Candidate],
    log_lines: list[str],
    zip_name: str | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    log_path = out_dir / f"{stem}.log"
    zip_path = out_dir / (zip_name or f"{stem}.zip")

    if not candidates:
        raise ValueError("at least one candidate is required")

    rows = [
        {"mol_smiles": candidate.mol_smiles, "route": candidate.route.text}
        for candidate in sorted(candidates, key=lambda item: item.score.total, reverse=True)
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log_path.write_text("\n".join(log_lines) + "\n")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=csv_path.name)
        zf.write(log_path, arcname=log_path.name)
    return zip_path


def write_final_result_zip(out_dir: Path, stems: list[str], zip_name: str = "result.zip") -> Path:
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for stem in stems:
            csv_path = out_dir / f"{stem}.csv"
            if csv_path.exists():
                zf.write(csv_path, arcname=csv_path.name)
    return zip_path
