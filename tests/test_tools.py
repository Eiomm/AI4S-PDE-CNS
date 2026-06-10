from pathlib import Path

from chem_evolve_agent.tools.base import ToolResult, ToolStatus
from chem_evolve_agent.tools.preparation import prepare_ligand_pdbqt, prepare_receptor_pdbqt
from chem_evolve_agent.tools.rdkit_property import run_rdkit_property_tool
from chem_evolve_agent.tools.registry import list_tool_specs
from chem_evolve_agent.tools.vina import run_vina_docking_tool


def test_tool_registry_contains_five_tools():
    names = {spec.name for spec in list_tool_specs()}
    assert {
        "rdkit_property_tool",
        "fpocket_tool",
        "obabel_meeko_prepare_tool",
        "vina_docking_tool",
        "aizynthfinder_route_tool",
    }.issubset(names)


def test_rdkit_property_tool_reports_valid_smiles():
    result = run_rdkit_property_tool("CCO")
    assert result.status in {ToolStatus.OK, ToolStatus.SKIPPED}
    if result.success:
        assert result.metrics["canonical_smiles"] == "CCO"
        assert result.metrics["mw"] > 40


def test_prepare_ligand_tool_is_structured_when_external_tools_missing(tmp_path: Path):
    result = prepare_ligand_pdbqt("CCO", tmp_path)
    assert result.status in {ToolStatus.OK, ToolStatus.SKIPPED, ToolStatus.ERROR}
    assert result.tool_name == "obabel_meeko_prepare_tool"
    if result.skipped:
        assert result.reason


def test_prepare_receptor_retries_tolerant_meeko(monkeypatch, tmp_path: Path):
    receptor = tmp_path / "target.pdb"
    receptor.write_text("HEADER target\nEND\n")
    calls = []

    def fake_find_executable(*names):
        if "mk_prepare_receptor.py" in names:
            return "/fake/bin/mk_prepare_receptor.py"
        return None

    def fake_run_command(command, timeout=120, cwd=None):
        calls.append(command)
        if "-a" in command:
            (tmp_path / "target.pdbqt").write_text("PDBQT\n")
            return ToolResult(tool_name="mk_prepare_receptor.py", status=ToolStatus.OK, command=command)
        return ToolResult(
            tool_name="mk_prepare_receptor.py",
            status=ToolStatus.ERROR,
            command=command,
            reason="return_code_1",
        )

    monkeypatch.setattr("chem_evolve_agent.tools.preparation.find_executable", fake_find_executable)
    monkeypatch.setattr("chem_evolve_agent.tools.preparation.run_command", fake_run_command)

    result = prepare_receptor_pdbqt(receptor, tmp_path, name="target")

    assert result.success
    assert len(calls) == 2
    assert "-a" in calls[1]
    assert "--default_altloc" in calls[1]
    assert result.warnings == ["strict_meeko_receptor_failed:return_code_1"]


def test_vina_tool_reports_missing_inputs_or_binary(tmp_path: Path):
    result = run_vina_docking_tool(
        receptor_pdbqt=tmp_path / "missing_receptor.pdbqt",
        ligand_pdbqt=tmp_path / "missing_ligand.pdbqt",
        out_dir=tmp_path,
    )
    assert result.status in {ToolStatus.SKIPPED, ToolStatus.ERROR}
    assert result.reason
