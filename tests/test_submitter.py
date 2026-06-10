import zipfile
from pathlib import Path

import pandas as pd

from chem_evolve_agent.models import Candidate, Route, Score
from chem_evolve_agent.submitter import write_final_result_zip, write_single_target_result


def test_write_single_target_result_creates_csv_log_zip(tmp_path: Path):
    candidate = Candidate(
        mol_smiles="CCO",
        route=Route(steps=["CCBr.O>>CCO"]),
        score=Score(molecule_score=0.8, route_score=0.7),
        metadata={"source": "unit"},
    )
    zip_path = write_single_target_result(tmp_path, "result", [candidate], ["event one"])
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        assert sorted(zf.namelist()) == ["result.csv", "result.log"]
        zf.extractall(tmp_path / "unzipped")
    df = pd.read_csv(tmp_path / "unzipped" / "result.csv")
    assert list(df.columns) == ["mol_smiles", "route"]
    assert df.iloc[0]["mol_smiles"] == "CCO"
    assert (tmp_path / "unzipped" / "result.log").read_text().strip() == "event one"


def test_write_final_result_zip_contains_three_final_csvs(tmp_path: Path):
    for index in range(1, 4):
        (tmp_path / f"result{index}.csv").write_text("mol_smiles,route\nCCO,CCBr.O>>CCO\n")
    zip_path = write_final_result_zip(tmp_path, ["result1", "result2", "result3"])
    with zipfile.ZipFile(zip_path) as zf:
        assert sorted(zf.namelist()) == ["result1.csv", "result2.csv", "result3.csv"]
