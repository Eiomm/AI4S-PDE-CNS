import json

from agent.code_generation_workspace import apply_agent_code_patch
from agent.pde_executor import ControlledExperimentExecutor
from agent.pde_journal import CandidatePlan, ExperimentJournal


def test_apply_agent_code_patch_writes_snapshot_and_manifest(tmp_path):
    result = apply_agent_code_patch(
        code_root=tmp_path / "runs" / "task1" / "study" / "code",
        files=[{"path": "model.py", "content": "class Model:\n    pass\n"}],
        provenance_record={"provider": "hkustgz_gpt", "model": "gpt-5.3-chat"},
    )

    assert (result.code_root / "model.py").read_text(encoding="utf-8") == "class Model:\n    pass\n"
    manifest = json.loads((result.code_root / "code_manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"][0]["path"] == "model.py"
    assert manifest["provenance"]["provider"] == "hkustgz_gpt"


def test_code_patch_executor_records_per_node_code_snapshot(tmp_path):
    journal = ExperimentJournal(tmp_path / "runs" / "task1" / "autonomous" / "20260515" / "study" / "journal.json")
    node = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="generate task code snapshot",
            action_type="code_patch",
            params={"files": [{"path": "infer.py", "content": "print('agent')\n"}]},
        )
    )
    executor = ControlledExperimentExecutor(project_root=tmp_path, code_dir=tmp_path / "code", journal=journal)

    execution = executor.execute(node)

    snapshot = tmp_path / "runs" / "task1" / "autonomous" / "20260515" / "study" / "nodes" / node.id / "code"
    assert execution.success is True
    assert execution.artifacts["code_snapshot_dir"] == str(snapshot)
    assert (snapshot / "infer.py").read_text(encoding="utf-8") == "print('agent')\n"
    assert (snapshot / "code_manifest.json").is_file()
