#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import zipfile
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chem_evolve_agent.chem_ops import canonicalize_smiles, route_final_product, validate_route
from chem_evolve_agent.models import Route


def inspect(path: Path, expected_csvs: list[str] | None = None) -> None:
    if not path.exists():
        raise SystemExit(f"缺少 zip 文件：{path}")
    expected_members = sorted(expected_csvs) if expected_csvs else ["result.csv", "result.log"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(path) as zf:
            names = sorted(zf.namelist())
            missing = sorted(set(expected_members) - set(names))
            extra = sorted(set(names) - set(expected_members))
            if missing or extra:
                raise SystemExit(f"zip 内容不匹配：缺少={missing}，多余={extra}")
            zf.extractall(tmp_path)
        csv_names = [name for name in names if name.endswith(".csv")]
        if not csv_names:
            raise SystemExit("zip 里没有 csv 文件")
        for name in csv_names:
            with (tmp_path / name).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise SystemExit(f"{name}: CSV 为空")
            if list(rows[0].keys()) != ["mol_smiles", "route"]:
                raise SystemExit(f"{name}: 需要 mol_smiles,route 两列")
            for row_index, row in enumerate(rows):
                product = route_final_product(str(row["route"]))
                try:
                    if canonicalize_smiles(product) != canonicalize_smiles(str(row["mol_smiles"])):
                        raise SystemExit(f"{name}:{row_index}: route 终产物和 mol_smiles 不一致")
                    route = Route(steps=[step.strip() for step in str(row["route"]).split(",") if step.strip()])
                    validation = validate_route(str(row["mol_smiles"]), route)
                    if validation.route_score <= 0:
                        penalties = ",".join(validation.penalties) or "unknown"
                        raise SystemExit(f"{name}:{row_index}: 路线得分无效；惩罚项={penalties}")
                except ValueError as exc:
                    raise SystemExit(f"{name}:{row_index}: {exc}") from exc
    print(f"OK 检查通过：{path} 包含 {', '.join(csv_names)}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("用法：inspect_result_zip.py result.zip [expected.csv ...]")
    inspect(Path(sys.argv[1]), sys.argv[2:] or None)


if __name__ == "__main__":
    main()
