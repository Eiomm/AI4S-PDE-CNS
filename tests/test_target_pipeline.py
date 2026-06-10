from pathlib import Path

from chem_evolve_agent.target.pdb_loader import load_pdb_target
from chem_evolve_agent.target.pocket import summarize_pocket
from chem_evolve_agent.workflow.runner import run_target_pipeline


PDB_TEXT = """\
ATOM      1  N   SER A 580      34.249   8.089  34.272  1.00 60.99           N
ATOM      2  CA  SER A 580      33.880   7.175  35.352  1.00 61.41           C
ATOM      3  C   PHE A 581      32.359   7.154  35.663  1.00 63.84           C
END
"""


def test_load_pdb_target_extracts_atoms_and_center(tmp_path: Path):
    pdb_path = tmp_path / "target.pdb"
    pdb_path.write_text(PDB_TEXT)
    target = load_pdb_target(pdb_path)
    assert target.atom_count == 3
    assert len(target.residues) == 2
    assert round(target.center[0], 3) == 33.496


def test_pocket_summary_uses_centroid_fallback(tmp_path: Path):
    pdb_path = tmp_path / "target.pdb"
    pdb_path.write_text(PDB_TEXT)
    pocket = summarize_pocket(load_pdb_target(pdb_path))
    assert pocket.method == "centroid_fallback_no_ligand"
    assert pocket.summary.startswith("centroid_fallback_atoms_3")


def test_run_target_pipeline_uses_target_and_logs_stages(tmp_path: Path):
    pdb_path = tmp_path / "target.pdb"
    pdb_path.write_text(PDB_TEXT)
    candidates, logs = run_target_pipeline(
        target_path=pdb_path,
        out_dir=tmp_path / "out",
        rounds=1,
        per_round=4,
        mode="docking",
        docking_limit=2,
    )
    joined = "\n".join(logs)
    assert candidates
    assert '"event": "target_loaded"' in joined
    assert '"event": "pocket_summary"' in joined
    assert '"event": "evaluate"' in joined
    assert '"event": "retrosynthesis"' in joined
    assert '"event": "docking"' in joined
