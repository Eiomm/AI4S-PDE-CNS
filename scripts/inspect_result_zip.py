#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chem_evolve_agent.chemistry.smiles import canonicalize_smiles, route_final_product


def inspect(path: Path, expected_csvs: list[str] | None = None) -> None:
    if not path.exists():
        raise SystemExit(f"missing zip: {path}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(path) as zf:
            names = sorted(zf.namelist())
            zf.extractall(tmp_path)
        csv_names = [name for name in names if name.endswith(".csv")]
        if not csv_names:
            raise SystemExit("zip has no csv files")
        if expected_csvs:
            missing = sorted(set(expected_csvs) - set(csv_names))
            extra = sorted(set(csv_names) - set(expected_csvs))
            if missing or extra:
                raise SystemExit(f"zip csv mismatch: missing={missing}, extra={extra}")
        for name in csv_names:
            df = pd.read_csv(tmp_path / name)
            if list(df.columns) != ["mol_smiles", "route"]:
                raise SystemExit(f"{name}: expected mol_smiles,route columns")
            if df.empty:
                raise SystemExit(f"{name}: empty csv")
            for row_index, row in df.iterrows():
                product = route_final_product(str(row["route"]))
                try:
                    if canonicalize_smiles(product) != canonicalize_smiles(str(row["mol_smiles"])):
                        raise SystemExit(f"{name}:{row_index}: route product mismatch")
                except ValueError as exc:
                    raise SystemExit(f"{name}:{row_index}: {exc}") from exc
    print(f"OK {path} contains {', '.join(csv_names)}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: inspect_result_zip.py result.zip [expected.csv ...]")
    inspect(Path(sys.argv[1]), sys.argv[2:] or None)


if __name__ == "__main__":
    main()
