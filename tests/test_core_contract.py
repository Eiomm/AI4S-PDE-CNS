from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

from chem_evolve_agent.chem_ops import (
    benchmark_prior_property_score,
    canonicalize_smiles,
    generate_evolved_smiles,
    molecule_score,
    property_metrics,
    route_score,
    validate_route,
)
from chem_evolve_agent.cli import (
    _append_final_submit_logs,
    _apply_mode_defaults,
    _configure_agent_memory_file,
    _configure_run_log_dir,
    _runtime_audit_event,
    _validate_run_args,
    _validate_runtime_requirements,
    main as cli_main,
)
from chem_evolve_agent.core import (
    AgentRunError,
    _agent_context,
    _screen_generated_smiles,
    _target_seed_offset,
    append_agent_experience,
    run_agent_for_target,
)
from chem_evolve_agent.models import Route
from chem_evolve_agent.runtime_tools import ToolError, list_tool_specs, load_target, run_sota_sbdd_generator, _select_best_aizynth_route
from chem_evolve_agent.submitter import clean_managed_outputs, write_final_result_zip, write_single_target_result
from scripts.inspect_result_zip import inspect


def _fake_aizynth_route(smiles: str) -> Route:
    target = canonicalize_smiles(smiles)
    stock = ["C" * 64, *["N"] * 12, *["O"] * 12, *["F"] * 8, *["Cl"] * 8, *["Br"] * 4, *["S"] * 4, *["P"] * 2, *["I"] * 2, *["B"] * 2]
    return Route(
        steps=[f"{'.'.join(stock)}>>{target}"],
        starting_materials=stock,
        source="aizynthfinder",
        confidence="high",
        solved=True,
    )


@pytest.fixture(autouse=True)
def _fast_aizynthfinder_route(monkeypatch) -> None:
    monkeypatch.setattr("chem_evolve_agent.core.run_aizynthfinder_route", lambda smiles, out_dir: _fake_aizynth_route(smiles))


def test_smiles_canonicalization_uses_rdkit() -> None:
    assert canonicalize_smiles("OCC") == "CCO"


def test_competition_scoring_hard_zero_rules() -> None:
    assert molecule_score(binding_score=1.0, validity_score=0.0, sa_score_value=1.0) == 0.0
    assert (
        route_score(
            route_validity_score=1.0,
            starting_material_availability_score=1.0,
            step_penalty_score=1.0,
            convergence_score=1.0,
            balance_score=1.0,
            final_product_matches=False,
            has_self_reaction=False,
        )
        == 0.0
    )


def test_route_tool_specs_expose_real_route_planner_only() -> None:
    smiles = canonicalize_smiles("CC(=O)Nc1ccccc1Cl")
    route = _fake_aizynth_route(smiles)
    validation = validate_route(smiles, route)
    names = {spec.name for spec in list_tool_specs()}

    assert "aizynthfinder_route_tool" in names
    assert "template_route_tool" not in names
    assert route.source == "aizynthfinder"
    assert validation.final_product_matches
    assert validation.route_score > 0
    assert "route_self_reaction" not in validation.penalties


def test_aizynthfinder_solved_route_uses_stock_availability_evidence() -> None:
    target = canonicalize_smiles("CCO")
    stock_route = Route(
        steps=[f"{canonicalize_smiles('CCCCCCCCCCCCCCCCO')}>>{target}"],
        starting_materials=[canonicalize_smiles("CCCCCCCCCCCCCCCCO")],
        source="aizynthfinder",
        confidence="high",
        solved=True,
    )
    unsolved_route = stock_route.model_copy(update={"source": "manual", "solved": None})

    solved_validation = validate_route(target, stock_route)
    unsolved_validation = validate_route(target, unsolved_route)

    assert solved_validation.final_product_matches
    assert solved_validation.starting_material_availability_score == 1.0
    assert unsolved_validation.starting_material_availability_score < solved_validation.starting_material_availability_score


def test_aizynthfinder_selects_later_route_tree_that_passes_hard_rules() -> None:
    target = canonicalize_smiles("CCO")
    self_reaction_tree = {
        "smiles": target,
        "children": [
            {
                "type": "reaction",
                "children": [{"smiles": target}],
            }
        ],
    }
    valid_tree = {
        "smiles": target,
        "children": [
            {
                "type": "reaction",
                "children": [{"smiles": canonicalize_smiles("CCCO")}],
            }
        ],
    }

    route = _select_best_aizynth_route(target, [self_reaction_tree, valid_tree])
    validation = validate_route(target, route)

    assert route.source == "aizynthfinder"
    assert route.steps == [f"{canonicalize_smiles('CCCO')}>>{target}"]
    assert validation.route_score > 0
    assert "route_self_reaction" not in validation.penalties


def test_sbdd_generator_adapter_returns_valid_canonical_smiles(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "fake_sbdd.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "test -f \"$AI4S_TARGET_PDB\"\n"
        "test -f \"$AI4S_POCKET_JSON\"\n"
        "test \"$AI4S_LIMIT\" = \"3\"\n"
        "printf 'smiles\\nOCC\\nnot_a_smiles\\nCCO\\n' > \"$AI4S_OUTPUT_SMILES\"\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("AI4S_SBDD_GENERATOR_CMD", str(script))
    target = load_target(Path("examples/target.pdb"))

    generated = run_sota_sbdd_generator(target, tmp_path / "sbdd_out", limit=3)

    assert generated == ["CCO"]


def test_sbdd_generator_adapter_rejects_invalid_outputs(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "fake_bad_sbdd.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'smiles\\nnot_a_smiles\\n' > \"$AI4S_OUTPUT_SMILES\"\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("AI4S_SBDD_GENERATOR_CMD", str(script))
    target = load_target(Path("examples/target.pdb"))

    with pytest.raises(ToolError, match="no RDKit-valid SMILES"):
        run_sota_sbdd_generator(target, tmp_path / "sbdd_bad_out", limit=3)


def test_sbdd_tool_spec_requires_existing_executable(tmp_path: Path, monkeypatch) -> None:
    def sbdd_available() -> bool:
        return next(spec.available for spec in list_tool_specs() if spec.name == "sota_sbdd_generator_tool")

    monkeypatch.setenv("AI4S_SBDD_GENERATOR_CMD", str(tmp_path / "missing_sbdd"))
    assert sbdd_available() is False

    script = tmp_path / "fake_sbdd.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("AI4S_SBDD_GENERATOR_CMD", str(script))
    assert sbdd_available() is True


def test_check_tools_require_sbdd_exits_when_missing(monkeypatch) -> None:
    check_tools = importlib.import_module("scripts.check_tools")
    monkeypatch.setattr(check_tools, "_load_dotenv", lambda: None)
    monkeypatch.delenv("AI4S_SBDD_GENERATOR_CMD", raising=False)
    monkeypatch.setattr(sys, "argv", ["check_tools.py", "--require-sbdd"])

    with pytest.raises(SystemExit, match="sota_sbdd_generator_tool"):
        check_tools.main()


def test_check_tools_require_sbdd_runs_probe(tmp_path: Path, monkeypatch, capsys) -> None:
    check_tools = importlib.import_module("scripts.check_tools")
    script = tmp_path / "fake_sbdd.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "test -f \"$AI4S_TARGET_PDB\"\n"
        "test -f \"$AI4S_POCKET_JSON\"\n"
        "test \"$AI4S_LIMIT\" = \"1\"\n"
        "printf 'smiles\\nOCC\\n' > \"$AI4S_OUTPUT_SMILES\"\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setattr(check_tools, "_load_dotenv", lambda: None)
    monkeypatch.setenv("AI4S_SBDD_GENERATOR_CMD", str(script))
    monkeypatch.setattr(sys, "argv", ["check_tools.py", "--require-sbdd", "--probe-limit", "1"])

    check_tools.main()

    assert "PROBE_OK SBDD 生成器探测通过：生成 1 个有效 SMILES" in capsys.readouterr().out


def test_check_tools_require_sbdd_rejects_failed_probe(tmp_path: Path, monkeypatch) -> None:
    check_tools = importlib.import_module("scripts.check_tools")
    script = tmp_path / "fake_bad_sbdd.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setattr(check_tools, "_load_dotenv", lambda: None)
    monkeypatch.setenv("AI4S_SBDD_GENERATOR_CMD", str(script))
    monkeypatch.setattr(sys, "argv", ["check_tools.py", "--require-sbdd"])

    with pytest.raises(SystemExit, match="SBDD_PROBE_FAILED"):
        check_tools.main()


def test_evolution_generator_makes_valid_variants() -> None:
    parent = canonicalize_smiles("COc1ccc(NC(C)=O)cc1")
    variants = generate_evolved_smiles([parent], limit=4, seed=0)
    assert variants
    assert parent not in variants
    assert all(canonicalize_smiles(item) == item for item in variants)


def test_benchmark_prior_guides_property_score() -> None:
    active_like_metrics, _ = property_metrics("COc1ccc(Cl)c(N)c1OC")
    tiny_metrics, _ = property_metrics("CCO")
    active_like = benchmark_prior_property_score(active_like_metrics)
    tiny = benchmark_prior_property_score(tiny_metrics)
    assert 0.0 <= tiny <= active_like <= 1.0


def test_competition_defaults_use_aizynthfinder_without_env(monkeypatch) -> None:
    monkeypatch.delenv("AI4S_ROUTE_ENGINE", raising=False)
    monkeypatch.delenv("AIZYNTHFINDER_CONFIG", raising=False)
    monkeypatch.delenv("AI4S_ROUTE_LIMIT_PER_ROUND", raising=False)
    monkeypatch.delenv("AI4S_VINA_FEEDBACK_PER_ROUND", raising=False)
    _apply_mode_defaults("competition")
    assert os.environ["AI4S_ROUTE_ENGINE"] == "aizynthfinder"
    assert os.environ["AIZYNTHFINDER_CONFIG"].endswith("data/aizynthfinder/config.yml")
    assert os.environ["AI4S_ROUTE_LIMIT_PER_ROUND"] == "10"
    assert os.environ["AI4S_VINA_FEEDBACK_PER_ROUND"] == "1"


def test_runtime_rejects_obsolete_route_engine(monkeypatch) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "0")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "template")
    _apply_mode_defaults("competition")
    assert os.environ["AI4S_ROUTE_ENGINE"] == "template"
    with pytest.raises(ValueError, match="未知 AI4S_ROUTE_ENGINE：template"):
        _validate_runtime_requirements(argparse.Namespace(mode="proxy"))


def test_runtime_audit_event_records_config_without_secret(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "1")
    monkeypatch.setenv("AI4S_AGENT_MODEL", "openai/test-model")
    monkeypatch.setenv("AI4S_AGENT_PROVIDER", "openai")
    monkeypatch.setenv("AI4S_AGENT_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("AI4S_AGENT_LLM_LOG_DIR", str(tmp_path / "llm_io"))
    monkeypatch.delenv("CHEM_EVOLVE_LLM_LOG_DIR", raising=False)
    monkeypatch.setenv("APIFOX_GPT_GE_API_KEY", "SECRET_SHOULD_NOT_APPEAR")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")
    monkeypatch.setenv("AI4S_ROUTE_LIMIT_PER_ROUND", "10")
    monkeypatch.setenv("AI4S_VINA_FEEDBACK_PER_ROUND", "1")
    args = argparse.Namespace(
        rounds=8,
        per_round=64,
        top_k=20,
        mode="competition",
        docking_limit=20,
        run_seed=7,
    )

    event_text = _runtime_audit_event(args, tmp_path, target_count=3, removed_outputs=["result.zip"])
    event = json.loads(event_text)

    assert event["event"] == "agent_runtime_config"
    assert event["target_count"] == 3
    assert event["mode"] == "competition"
    assert event["route_engine"] == "aizynthfinder"
    assert event["route_limit_per_round"] == "10"
    assert event["vina_feedback_per_round"] == "1"
    assert event["llm_enabled"] is True
    assert event["llm_model"] == "openai/test-model"
    assert event["llm_api_key_set"] is True
    assert event["llm_log_dir"] == str(tmp_path / "llm_io")
    assert "SECRET_SHOULD_NOT_APPEAR" not in event_text


def test_configure_run_log_dir_defaults_to_output_without_overriding(monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    monkeypatch.delenv("AI4S_AGENT_LLM_LOG_DIR", raising=False)
    monkeypatch.delenv("CHEM_EVOLVE_LLM_LOG_DIR", raising=False)

    _configure_run_log_dir(out_dir)

    assert os.environ["AI4S_AGENT_LLM_LOG_DIR"] == str(out_dir / "llm_io")
    assert os.environ["CHEM_EVOLVE_LLM_LOG_DIR"] == str(out_dir / "llm_io")

    monkeypatch.setenv("AI4S_AGENT_LLM_LOG_DIR", "runs/llm_io")
    monkeypatch.setenv("CHEM_EVOLVE_LLM_LOG_DIR", "runs/llm_io")

    _configure_run_log_dir(tmp_path / "legacy_out")

    assert os.environ["AI4S_AGENT_LLM_LOG_DIR"] == str(tmp_path / "legacy_out" / "llm_io")
    assert os.environ["CHEM_EVOLVE_LLM_LOG_DIR"] == str(tmp_path / "legacy_out" / "llm_io")

    monkeypatch.setenv("AI4S_AGENT_LLM_LOG_DIR", str(tmp_path / "custom_agent_logs"))
    monkeypatch.setenv("CHEM_EVOLVE_LLM_LOG_DIR", str(tmp_path / "custom_legacy_logs"))

    _configure_run_log_dir(tmp_path / "other")

    assert os.environ["AI4S_AGENT_LLM_LOG_DIR"] == str(tmp_path / "custom_agent_logs")
    assert os.environ["CHEM_EVOLVE_LLM_LOG_DIR"] == str(tmp_path / "custom_legacy_logs")


def test_configure_agent_memory_file_defaults_to_outputs_root(monkeypatch, tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    out_dir = outputs_root / "demo" / "20260612_120000"
    monkeypatch.setenv("AI4S_OUTPUTS_DIR", str(outputs_root))
    monkeypatch.delenv("AI4S_AGENT_MEMORY_FILE", raising=False)

    memory_file = _configure_agent_memory_file(out_dir, explicit_path=None)

    assert memory_file == outputs_root / "agent_experience.jsonl"
    assert os.environ["AI4S_AGENT_MEMORY_FILE"] == str(memory_file)

    explicit = tmp_path / "custom_memory.jsonl"
    assert _configure_agent_memory_file(out_dir, explicit_path=str(explicit)) == explicit
    assert os.environ["AI4S_AGENT_MEMORY_FILE"] == str(explicit)


def test_configure_agent_memory_file_keeps_temp_runs_isolated(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AI4S_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.delenv("AI4S_AGENT_MEMORY_FILE", raising=False)
    out_dir = tmp_path / "scratch_run"

    memory_file = _configure_agent_memory_file(out_dir, explicit_path=None)

    assert memory_file == out_dir / "agent_experience.jsonl"
    assert os.environ["AI4S_AGENT_MEMORY_FILE"] == str(memory_file)


def test_cli_rejects_invalid_run_parameters() -> None:
    valid = argparse.Namespace(rounds=1, per_round=1, top_k=1, mode="proxy", docking_limit=0)
    _validate_run_args(valid)

    with pytest.raises(ValueError, match="--rounds 必须是正数"):
        _validate_run_args(valid.__class__(rounds=0, per_round=1, top_k=1, mode="proxy", docking_limit=0))
    with pytest.raises(ValueError, match="--per-round 必须是正数"):
        _validate_run_args(valid.__class__(rounds=1, per_round=0, top_k=1, mode="proxy", docking_limit=0))
    with pytest.raises(ValueError, match="--top-k 必须是正数"):
        _validate_run_args(valid.__class__(rounds=1, per_round=1, top_k=0, mode="proxy", docking_limit=0))
    with pytest.raises(ValueError, match="--docking-limit 不能为负数"):
        _validate_run_args(valid.__class__(rounds=1, per_round=1, top_k=1, mode="proxy", docking_limit=-1))
    with pytest.raises(ValueError, match="docking 模式下 --docking-limit 必须是正数"):
        _validate_run_args(valid.__class__(rounds=1, per_round=1, top_k=1, mode="docking", docking_limit=0))
    with pytest.raises(ValueError, match="competition 模式下 --docking-limit 必须是正数"):
        _validate_run_args(valid.__class__(rounds=1, per_round=1, top_k=1, mode="competition", docking_limit=0))


def test_runtime_requirements_reject_enabled_llm_without_api(monkeypatch) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "1")
    monkeypatch.setenv("AI4S_AGENT_API_KEY_ENVS", "AI4S_TEST_MISSING_KEY")
    for name in (
        "AI4S_TEST_MISSING_KEY",
        "CHEM_EVOLVE_LLM_BASE_URL",
        "AI4S_AGENT_BASE_URL",
        "OPENAI_API_BASE",
        "AI4S_ROUTE_ENGINE",
        "AI4S_SBDD_GENERATOR_CMD",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="CHEM_EVOLVE_LLM_ENABLED=1"):
        _validate_runtime_requirements(argparse.Namespace(mode="proxy"))


def test_cli_rejects_invalid_parameters_before_cleanup(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale = out_dir / "result.csv"
    stale.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chem_evolve_agent",
            "--targets",
            "examples/target.pdb",
            "--out",
            str(out_dir),
            "--rounds",
            "0",
            "--per-round",
            "1",
            "--top-k",
            "1",
            "--mode",
            "proxy",
        ],
    )

    with pytest.raises(SystemExit):
        cli_main()

    assert stale.exists()


def test_cli_rejects_missing_llm_before_cleanup(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale = out_dir / "result.csv"
    stale.write_text("stale", encoding="utf-8")
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "1")
    monkeypatch.setenv("AI4S_AGENT_API_KEY_ENVS", "AI4S_TEST_MISSING_KEY")
    for name in (
        "AI4S_TEST_MISSING_KEY",
        "CHEM_EVOLVE_LLM_BASE_URL",
        "AI4S_AGENT_BASE_URL",
        "OPENAI_API_BASE",
        "AI4S_ROUTE_ENGINE",
        "AI4S_SBDD_GENERATOR_CMD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chem-evolve",
            "--targets",
            "examples/target.pdb",
            "--out",
            str(out_dir),
            "--rounds",
            "1",
            "--per-round",
            "1",
            "--top-k",
            "1",
            "--mode",
            "proxy",
            "--docking-limit",
            "0",
        ],
    )

    with pytest.raises(SystemExit):
        cli_main()

    assert stale.exists()


def test_agent_writes_valid_single_target_submission(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "0")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")
    candidates, logs = run_agent_for_target(
        target_path=Path("examples/target.pdb"),
        out_dir=tmp_path,
        rounds=1,
        per_round=8,
        mode="proxy",
        docking_limit=0,
    )
    zip_path = write_single_target_result(tmp_path, "result", candidates[:5], logs)
    assert zip_path.exists()
    assert any('"event": "agent_strategy"' in line for line in logs)
    generate_event = next(json.loads(line) for line in logs if json.loads(line)["event"] == "generate")
    assert generate_event["seed"] == generate_event["target_seed_offset"]
    round_summary = next(json.loads(line) for line in logs if json.loads(line)["event"] == "agent_round_summary")
    assert round_summary["generated_count"] == 8
    assert round_summary["accepted_count"] == len(candidates)
    assert round_summary["best_overall_smiles"] == candidates[0].mol_smiles
    assert any('"property_prior_score"' in line for line in logs)
    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == ["result.csv", "result.log"]


def test_agent_loads_persistent_experience_across_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "0")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")
    memory_file = tmp_path / "agent_experience.jsonl"

    first_candidates, _ = run_agent_for_target(
        target_path=Path("examples/target.pdb"),
        out_dir=tmp_path / "run1",
        rounds=1,
        per_round=8,
        mode="proxy",
        docking_limit=0,
        experience_file=memory_file,
    )
    saved = append_agent_experience(memory_file, Path("examples/target.pdb"), first_candidates, top_k=3, run_dir=tmp_path / "run1")

    assert saved == 3
    assert memory_file.exists()

    _, logs = run_agent_for_target(
        target_path=Path("examples/target.pdb"),
        out_dir=tmp_path / "run2",
        rounds=1,
        per_round=4,
        mode="proxy",
        docking_limit=0,
        experience_file=memory_file,
    )

    loaded_event = next(json.loads(line) for line in logs if json.loads(line)["event"] == "agent_experience_loaded")
    plan_event = next(json.loads(line) for line in logs if json.loads(line)["event"] == "agent_plan")
    memory_event = next(json.loads(line) for line in logs if json.loads(line)["event"] == "agent_memory")

    assert loaded_event["loaded_count"] == 3
    assert "evolution_generator" in plan_event["actions"]
    assert memory_event["elites"][0]["origin"] == "long_term_memory"


def test_agent_loads_global_experience_seed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "0")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")
    memory_file = tmp_path / "agent_experience.jsonl"
    memory_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event": "agent_experience",
                "scope": "global",
                "target_id": "seed",
                "target_signature": "global",
                "run_dir": "unit_test",
                "rank": 1,
                "created_at": "2026-06-12T00:00:00+00:00",
                "candidate": {
                    "smiles": "CC(=O)Nc1ccccc1Cl",
                    "total": 0.7906,
                    "molecule": 0.686,
                    "route": 0.9475,
                    "binding": 0.6216,
                    "binding_source": "proxy",
                    "route_source": "aizynthfinder",
                    "penalties": [],
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    _, logs = run_agent_for_target(
        target_path=Path("examples/target.pdb"),
        out_dir=tmp_path / "run",
        rounds=1,
        per_round=4,
        mode="proxy",
        docking_limit=0,
        experience_file=memory_file,
    )

    loaded_event = next(json.loads(line) for line in logs if json.loads(line)["event"] == "agent_experience_loaded")
    plan_event = next(json.loads(line) for line in logs if json.loads(line)["event"] == "agent_plan")
    memory_event = next(json.loads(line) for line in logs if json.loads(line)["event"] == "agent_memory")

    assert loaded_event["loaded_count"] == 1
    assert "evolution_generator" in plan_event["actions"]
    assert memory_event["elites"][0]["origin"] == "global_experience"


def test_long_term_memory_exact_smiles_are_not_resubmitted() -> None:
    logs: list[str] = []
    rejection_memory: list[dict[str, object]] = []

    screened = _screen_generated_smiles(
        ["CC(=O)Nc1ccccc1Cl", "CC(=O)Nc1ccccc1F"],
        seen=set(),
        rejection_memory=rejection_memory,
        logs=logs,
        target_id="target",
        round_index=0,
        forbidden_smiles={canonicalize_smiles("CC(=O)Nc1ccccc1Cl")},
    )

    assert screened == [canonicalize_smiles("CC(=O)Nc1ccccc1F")]
    assert any("long_term_memory_exact_reuse" in line for line in logs)


def test_agent_prints_live_progress(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "0")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")
    monkeypatch.delenv("AI4S_PROGRESS_STDERR", raising=False)

    run_agent_for_target(
        target_path=Path("examples/target.pdb"),
        out_dir=tmp_path,
        rounds=1,
        per_round=2,
        mode="proxy",
        docking_limit=0,
    )

    stderr = capsys.readouterr().err
    assert "[agent][运行] 开始运行 | event=start" in stderr
    assert "[agent][轮次] 开始新一轮 | event=round_start" in stderr
    assert "[agent][路线] 开始路线规划 | event=route_start" in stderr
    assert "[agent][路线] 路线规划通过 | event=route_accept" in stderr
    assert "[agent][轮次] 本轮完成 | event=round_done" in stderr
    assert "[agent][运行] 运行完成 | event=done" in stderr


def test_target_seed_offset_uses_pocket_geometry(tmp_path: Path) -> None:
    def write_target(path: Path, x: float) -> None:
        path.write_text(
            f"ATOM      1  C   ALA A   1    {x:8.3f}   1.000   2.000  1.00 20.00           C\n"
            "END\n",
            encoding="utf-8",
        )

    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    left_target = left_dir / "target.pdb"
    right_target = right_dir / "target.pdb"
    write_target(left_target, 0.0)
    write_target(right_target, 9.0)

    assert load_target(left_target).target_id == load_target(right_target).target_id == "target"
    assert _target_seed_offset(load_target(left_target)) != _target_seed_offset(load_target(right_target))


def test_submitter_rejects_invalid_candidate_route(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "0")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")
    candidates, logs = run_agent_for_target(
        target_path=Path("examples/target.pdb"),
        out_dir=tmp_path,
        rounds=1,
        per_round=4,
        mode="proxy",
        docking_limit=0,
    )
    bad = candidates[0].model_copy(deep=True)
    bad.route = Route(steps=[f"{bad.mol_smiles}>>{bad.mol_smiles}"])

    with pytest.raises(ValueError, match="candidate route is invalid"):
        write_single_target_result(tmp_path, "bad", [bad], logs)


def test_final_zip_revalidates_existing_csv(tmp_path: Path) -> None:
    (tmp_path / "result1.csv").write_text("mol_smiles,route\nCCO,CCO>>CCO\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid route"):
        write_final_result_zip(tmp_path, ["result1"])


def test_llm_round_planner_reads_context_and_supplies_candidates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "1")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")
    monkeypatch.delenv("AI4S_SBDD_GENERATOR_CMD", raising=False)

    class FakeLlmClient:
        calls: list[list[dict[str, str]]] = []

        def complete_json(self, messages: list[dict[str, str]]) -> dict:
            self.calls.append(messages)
            return {
                "strategy": "llm_route_valid_acylation",
                "focus": ["route-valid chloro acetanilide analog"],
                "avoid": ["route product mismatch"],
                "candidates": [
                    {
                        "smiles": "CC(=O)Nc1ccccc1Cl",
                        "rationale": "AiZynthFinder-solvable scaffold with CNS-like size",
                    }
                ],
            }

    monkeypatch.setattr("chem_evolve_agent.core.LiteLlmClient", FakeLlmClient)

    candidates, logs = run_agent_for_target(
        target_path=Path("examples/target.pdb"),
        out_dir=tmp_path,
        rounds=1,
        per_round=1,
        mode="proxy",
        docking_limit=0,
    )

    prompt = FakeLlmClient.calls[0][1]["content"]
    assert "AI Agent for Targeted Molecule Discovery and Synthesis Planning" in prompt
    assert candidates
    assert any(candidate.metadata["plan"][0] == "llm_generator" for candidate in candidates)
    assert any('"source": "llm"' in line and "llm_route_valid_acylation" in line for line in logs)


def test_llm_round_planner_reports_bad_json_cleanly(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "1")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")
    monkeypatch.delenv("AI4S_SBDD_GENERATOR_CMD", raising=False)

    class FakeSettings:
        log_dir = tmp_path / "llm_io"

    class FakeLlmClient:
        settings = FakeSettings()

        def complete_json(self, messages: list[dict[str, str]]) -> dict:
            raise ValueError("Expecting ',' delimiter: line 65 column 6")

    monkeypatch.setattr("chem_evolve_agent.core.LiteLlmClient", FakeLlmClient)

    with pytest.raises(AgentRunError, match="LLM 返回的 JSON 无法解析"):
        run_agent_for_target(
            target_path=Path("examples/target.pdb"),
            out_dir=tmp_path,
            rounds=1,
            per_round=1,
            mode="proxy",
            docking_limit=0,
        )

    stderr = capsys.readouterr().err
    assert "[agent][LLM] LLM 输出解析失败 | event=llm_reject" in stderr
    assert "日志目录=" in stderr


def test_per_round_is_global_candidate_pool(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "1")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")
    monkeypatch.delenv("AI4S_SBDD_GENERATOR_CMD", raising=False)

    class FakeLlmClient:
        def complete_json(self, messages: list[dict[str, str]]) -> dict:
            return {
                "strategy": "fill_pool_from_llm_first",
                "focus": ["global per-round budget"],
                "avoid": ["tool budget multiplication"],
                "candidates": [
                    {"smiles": "CC(=O)Nc1ccccc1Cl", "rationale": "AiZynthFinder route"},
                    {"smiles": "CC(=O)Nc1ccccc1F", "rationale": "AiZynthFinder route"},
                    {"smiles": "CC(=O)Nc1ccccc1C", "rationale": "over budget"},
                ],
            }

    monkeypatch.setattr("chem_evolve_agent.core.LiteLlmClient", FakeLlmClient)

    _, logs = run_agent_for_target(
        target_path=Path("examples/target.pdb"),
        out_dir=tmp_path,
        rounds=1,
        per_round=2,
        mode="proxy",
        docking_limit=0,
    )

    generate_event = next(json.loads(line) for line in logs if json.loads(line)["event"] == "generate")
    assert generate_event["count"] == 2


def test_llm_round_planner_receives_rejection_memory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "1")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")
    monkeypatch.delenv("AI4S_SBDD_GENERATOR_CMD", raising=False)
    rejected = canonicalize_smiles("c1ccc2[nH]c(-c3ccncc3)nc2c1")

    def fake_route_with_one_failure(smiles, out_dir):
        if canonicalize_smiles(smiles) == rejected:
            raise ToolError("AiZynthFinder found no solved route")
        return _fake_aizynth_route(smiles)

    monkeypatch.setattr("chem_evolve_agent.core.run_aizynthfinder_route", fake_route_with_one_failure)

    class FakeLlmClient:
        calls: list[list[dict[str, str]]] = []

        def complete_json(self, messages: list[dict[str, str]]) -> dict:
            self.calls.append(messages)
            if len(self.calls) == 1:
                return {
                    "strategy": "try_uncovered_scaffold",
                    "focus": ["novel scaffold"],
                    "avoid": ["none yet"],
                    "candidates": [{"smiles": "c1ccc2[nH]c(-c3ccncc3)nc2c1", "rationale": "will need route feedback"}],
                }
            return {
                "strategy": "repair_route_after_rejection",
                "focus": ["AiZynthFinder-solvable analog"],
                "avoid": ["AiZynthFinder found no solved route"],
                "candidates": [{"smiles": "CC(=O)Nc1ccccc1Cl", "rationale": "route-solvable analog"}],
            }

    monkeypatch.setattr("chem_evolve_agent.core.LiteLlmClient", FakeLlmClient)

    _, logs = run_agent_for_target(
        target_path=Path("examples/target.pdb"),
        out_dir=tmp_path,
        rounds=2,
        per_round=2,
        mode="proxy",
        docking_limit=0,
    )

    second_prompt = FakeLlmClient.calls[1][1]["content"]
    assert "Recent rejected candidates and reasons" in second_prompt
    assert "AiZynthFinder found no solved route" in second_prompt
    assert "c1ccc2[nH]c(-c3ccncc3)nc2c1" in second_prompt
    assert any('"event": "agent_rejection_memory"' in line for line in logs)


def test_llm_round_planner_receives_vina_feedback_memory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "1")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")
    monkeypatch.setenv("AI4S_VINA_FEEDBACK_PER_ROUND", "1")
    monkeypatch.delenv("AI4S_SBDD_GENERATOR_CMD", raising=False)
    monkeypatch.setattr("chem_evolve_agent.core.run_vina_binding_score", lambda smiles, target, out_dir: (0.75, -10.0))

    class FakeLlmClient:
        calls: list[list[dict[str, str]]] = []

        def complete_json(self, messages: list[dict[str, str]]) -> dict:
            self.calls.append(messages)
            if len(self.calls) == 1:
                return {
                    "strategy": "dock_first_aizynth_candidate",
                    "focus": ["real Vina feedback"],
                    "avoid": ["proxy-only memory"],
                    "candidates": [{"smiles": "CC(=O)Nc1ccccc1Cl", "rationale": "route-solvable scaffold"}],
                }
            return {
                "strategy": "use_vina_feedback",
                "focus": ["preserve docked analogs"],
                "avoid": ["losing route score"],
                "candidates": [{"smiles": "CC(=O)Nc1ccccc1F", "rationale": "small analog after Vina feedback"}],
            }

    monkeypatch.setattr("chem_evolve_agent.core.LiteLlmClient", FakeLlmClient)

    _, logs = run_agent_for_target(
        target_path=Path("examples/target.pdb"),
        out_dir=tmp_path,
        rounds=2,
        per_round=1,
        mode="competition",
        docking_limit=1,
    )

    second_prompt = FakeLlmClient.calls[1][1]["content"]
    assert "binding_source=vina" in second_prompt
    assert "docking_energy=-10.0" in second_prompt
    assert "round_vina_feedback=True" in second_prompt
    assert "route_validity=" in second_prompt
    assert any('"event": "competition_feedback_dock"' in line for line in logs)


def test_llm_round_planner_receives_failed_vina_feedback_memory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "1")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")
    monkeypatch.setenv("AI4S_VINA_FEEDBACK_PER_ROUND", "1")
    monkeypatch.delenv("AI4S_SBDD_GENERATOR_CMD", raising=False)

    dock_calls = {"count": 0}

    def fake_vina(smiles, target, out_dir):
        dock_calls["count"] += 1
        if dock_calls["count"] == 1:
            raise RuntimeError("docking failed")
        return (0.75, -10.0)

    monkeypatch.setattr("chem_evolve_agent.core.run_vina_binding_score", fake_vina)

    class FakeLlmClient:
        calls: list[list[dict[str, str]]] = []

        def complete_json(self, messages: list[dict[str, str]]) -> dict:
            self.calls.append(messages)
            if len(self.calls) == 1:
                return {
                    "strategy": "first_candidate_for_feedback",
                    "focus": ["test feedback failure"],
                    "avoid": ["none yet"],
                    "candidates": [{"smiles": "CC(=O)Nc1ccccc1Cl", "rationale": "route-solvable scaffold"}],
                }
            return {
                "strategy": "repair_after_failed_docking",
                "focus": ["avoid failed feedback"],
                "avoid": ["round_vina_feedback_failed"],
                "candidates": [{"smiles": "CC(=O)Nc1ccccc1F", "rationale": "small route-valid analog"}],
            }

    monkeypatch.setattr("chem_evolve_agent.core.LiteLlmClient", FakeLlmClient)

    candidates, logs = run_agent_for_target(
        target_path=Path("examples/target.pdb"),
        out_dir=tmp_path,
        rounds=2,
        per_round=1,
        mode="competition",
        docking_limit=1,
    )

    second_prompt = FakeLlmClient.calls[1][1]["content"]
    assert candidates
    assert "round_vina_feedback_failed:RuntimeError:docking failed" in second_prompt
    assert any('"event": "agent_rejection_memory"' in line and "round_vina_feedback_failed" in line for line in logs)
    assert any('"event": "agent_round_summary"' in line and '"failed_vina_feedback_count": 1' in line for line in logs)


def test_route_planning_limit_prefilters_before_routes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "0")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")
    monkeypatch.setenv("AI4S_ROUTE_LIMIT_PER_ROUND", "2")

    _, logs = run_agent_for_target(
        target_path=Path("examples/target.pdb"),
        out_dir=tmp_path,
        rounds=1,
        per_round=8,
        mode="proxy",
        docking_limit=0,
    )

    route_prefilter = next(json.loads(line) for line in logs if json.loads(line)["event"] == "route_prefilter")
    assert route_prefilter["route_limit"] == 2
    assert route_prefilter["selected_count"] == 2
    assert len(route_prefilter["selected"]) == 2


def test_route_planning_limit_rejects_invalid_config(monkeypatch) -> None:
    from chem_evolve_agent.core import _route_planning_limit

    monkeypatch.setenv("AI4S_ROUTE_LIMIT_PER_ROUND", "0")
    with pytest.raises(AgentRunError, match="must be positive"):
        _route_planning_limit("proxy", valid_count=4, docking_limit=0)


def test_llm_context_paths_are_required(monkeypatch) -> None:
    monkeypatch.setenv("AI4S_COMPETITION_CONTEXT_PATHS", "docs/does_not_exist.md")
    with pytest.raises(AgentRunError, match="competition context file missing"):
        _agent_context()


def test_llm_context_paths_are_resolved_from_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI4S_COMPETITION_CONTEXT_PATHS", "docs/competition_race5_description.md")

    context = _agent_context()

    assert "AI Agent for Targeted Molecule Discovery and Synthesis Planning" in context


def test_write_single_target_can_skip_intermediate_zip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "0")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")
    candidates, logs = run_agent_for_target(
        target_path=Path("examples/target.pdb"),
        out_dir=tmp_path,
        rounds=1,
        per_round=4,
        mode="proxy",
        docking_limit=0,
    )
    path = write_single_target_result(tmp_path, "result1", candidates[:2], logs, write_zip=False)
    assert path == tmp_path / "result1.csv"
    assert path.exists()
    assert (tmp_path / "result1.log").exists()
    assert not (tmp_path / "result1.zip").exists()


def test_final_submit_event_is_appended_to_multitarget_logs(tmp_path: Path) -> None:
    stems = ["result1", "result2", "result3"]
    zip_path = tmp_path / "result.zip"
    for stem in stems:
        (tmp_path / f"{stem}.log").write_text('{"event":"submit"}\n', encoding="utf-8")

    _append_final_submit_logs(tmp_path, stems, zip_path)

    for stem in stems:
        events = [json.loads(line) for line in (tmp_path / f"{stem}.log").read_text(encoding="utf-8").splitlines()]
        final_event = events[-1]
        assert final_event["event"] == "final_submit"
        assert final_event["zip_path"] == str(zip_path)
        assert final_event["members"] == ["result1.csv", "result2.csv", "result3.csv"]
        assert final_event["member_count"] == 3


def test_clean_managed_outputs_removes_only_agent_artifacts(tmp_path: Path) -> None:
    for name in ("result.csv", "result1.log", "result.zip"):
        (tmp_path / name).write_text("stale", encoding="utf-8")
    for name in ("generation", "routes", "docking", "docking_feedback", "llm_io", "work"):
        path = tmp_path / name
        path.mkdir()
        (path / "stale.txt").write_text("stale", encoding="utf-8")
    keep = tmp_path / "notes.txt"
    keep.write_text("keep", encoding="utf-8")

    removed = clean_managed_outputs(tmp_path)

    assert "result.csv" in removed
    assert "generation/" in removed
    assert "docking_feedback/" in removed
    assert "llm_io/" in removed
    assert "work/" in removed
    assert keep.exists()
    assert not (tmp_path / "result.zip").exists()
    assert not (tmp_path / "routes").exists()
    assert not (tmp_path / "docking_feedback").exists()
    assert not (tmp_path / "llm_io").exists()
    assert not (tmp_path / "work").exists()


def test_cli_preflights_all_targets_before_generation(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "result1.csv").write_text("stale", encoding="utf-8")
    keep = out_dir / "notes.txt"
    keep.write_text("keep", encoding="utf-8")
    missing_target = tmp_path / "missing_target.pdb"
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "0")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("run_agent_for_target should not be called before all targets pass preflight")

    monkeypatch.setattr("chem_evolve_agent.cli.run_agent_for_target", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chem_evolve_agent",
            "--targets",
            "examples/target.pdb",
            str(missing_target),
            "--out",
            str(out_dir),
            "--rounds",
            "1",
            "--per-round",
            "1",
            "--top-k",
            "1",
            "--mode",
            "proxy",
        ],
    )

    with pytest.raises(SystemExit):
        cli_main()

    assert (out_dir / "result1.csv").exists()
    assert keep.exists()
    assert not (out_dir / "result2.csv").exists()


def test_cli_reports_agent_run_error_cleanly(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "0")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")

    def fail_agent(*args, **kwargs):
        raise AgentRunError("agent produced no candidates with valid score and route")

    monkeypatch.setattr("chem_evolve_agent.cli.run_agent_for_target", fail_agent)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chem_evolve_agent",
            "--targets",
            "examples/target.pdb",
            "--out",
            str(out_dir),
            "--rounds",
            "1",
            "--per-round",
            "1",
            "--top-k",
            "1",
            "--mode",
            "proxy",
        ],
    )

    with pytest.raises(SystemExit, match="AGENT_RUN_FAILED: 靶点运行失败：examples/target.pdb: agent produced no candidates"):
        cli_main()


def test_cli_isolates_multitarget_work_directories(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    target1 = tmp_path / "target1.pdb"
    target2 = tmp_path / "target2.pdb"
    source = Path("examples/target.pdb").read_text(encoding="utf-8")
    target1.write_text(source, encoding="utf-8")
    target2.write_text(source, encoding="utf-8")
    agent_calls: list[Path] = []
    write_calls: list[tuple[Path, str, list[str]]] = []

    def fake_run_agent_for_target(*, target_path, out_dir, rounds, per_round, mode, docking_limit, run_seed, experience_file):
        agent_calls.append(out_dir)
        return [], ['{"event":"agent_rank","best_score":1.0}']

    def fake_write_single_target_result(out_dir_arg, stem, candidates, logs, write_zip=True):
        write_calls.append((out_dir_arg, stem, logs))
        (out_dir_arg / f"{stem}.csv").write_text("mol_smiles,route\nCCO,CCCO>>CCO\n", encoding="utf-8")
        (out_dir_arg / f"{stem}.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
        return out_dir_arg / f"{stem}.csv"

    def fake_write_final_result_zip(out_dir_arg, stems):
        zip_path = out_dir_arg / "result.zip"
        zip_path.write_text("fake zip placeholder", encoding="utf-8")
        return zip_path

    monkeypatch.setattr("chem_evolve_agent.cli.run_agent_for_target", fake_run_agent_for_target)
    monkeypatch.setattr("chem_evolve_agent.cli.write_single_target_result", fake_write_single_target_result)
    monkeypatch.setattr("chem_evolve_agent.cli.write_final_result_zip", fake_write_final_result_zip)
    monkeypatch.setattr("chem_evolve_agent.cli._append_final_submit_logs", lambda out_dir, stems, zip_path: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chem_evolve_agent",
            "--targets",
            str(target1),
            str(target2),
            "--out",
            str(out_dir),
            "--rounds",
            "1",
            "--per-round",
            "1",
            "--top-k",
            "1",
            "--mode",
            "proxy",
        ],
    )

    cli_main()

    assert agent_calls == [out_dir / "work" / "result1", out_dir / "work" / "result2"]
    assert (out_dir / "work" / "result1").is_dir()
    assert (out_dir / "work" / "result2").is_dir()
    assert [call[0] for call in write_calls] == [out_dir, out_dir]
    assert [call[1] for call in write_calls] == ["result1", "result2"]
    assert '"work_dir": "' in write_calls[0][2][-1]


def test_competition_mode_reranks_top_candidates_with_vina(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "0")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")
    monkeypatch.setattr("chem_evolve_agent.core.run_vina_binding_score", lambda smiles, target, out_dir: (0.75, -10.0))
    candidates, logs = run_agent_for_target(
        target_path=Path("examples/target.pdb"),
        out_dir=tmp_path,
        rounds=1,
        per_round=8,
        mode="competition",
        docking_limit=3,
    )
    assert len(candidates) == 3
    assert all(candidate.score.binding_source == "vina" for candidate in candidates)
    assert any('"event": "competition_dock"' in line for line in logs)


def test_competition_mode_uses_round_vina_feedback_in_memory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHEM_EVOLVE_LLM_ENABLED", "0")
    monkeypatch.setenv("AI4S_ROUTE_ENGINE", "aizynthfinder")
    monkeypatch.setenv("AI4S_VINA_FEEDBACK_PER_ROUND", "1")
    monkeypatch.setattr("chem_evolve_agent.core.run_vina_binding_score", lambda smiles, target, out_dir: (0.75, -10.0))

    candidates, logs = run_agent_for_target(
        target_path=Path("examples/target.pdb"),
        out_dir=tmp_path,
        rounds=1,
        per_round=8,
        mode="competition",
        docking_limit=1,
    )

    assert len(candidates) == 1
    assert candidates[0].metadata["round_vina_feedback"] is True
    assert candidates[0].score.binding_source == "vina"
    assert any('"event": "competition_feedback_dock"' in line for line in logs)
    assert any('"event": "competition_dock"' in line and '"reused_vina_score": true' in line for line in logs)


def test_vina_feedback_limit_rejects_invalid_config(monkeypatch) -> None:
    from chem_evolve_agent.core import _vina_feedback_per_round

    monkeypatch.setenv("AI4S_VINA_FEEDBACK_PER_ROUND", "-1")
    with pytest.raises(AgentRunError, match="must be non-negative"):
        _vina_feedback_per_round("competition")


def test_result_inspector_rejects_self_reaction(tmp_path: Path) -> None:
    csv_path = tmp_path / "result.csv"
    csv_path.write_text("mol_smiles,route\nCCO,CCO>>CCO\n", encoding="utf-8")
    zip_path = tmp_path / "result.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(csv_path, arcname="result.csv")
        archive.writestr("result.log", "log\n")
    with pytest.raises(SystemExit, match="路线得分无效"):
        inspect(zip_path)


def test_result_inspector_requires_exact_single_target_zip_members(tmp_path: Path) -> None:
    csv_path = tmp_path / "result.csv"
    csv_path.write_text("mol_smiles,route\nCCO,CCCO>>CCO\n", encoding="utf-8")
    zip_path = tmp_path / "result.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(csv_path, arcname="result.csv")
        archive.writestr("result.log", "log\n")
        archive.writestr("debug.json", "{}\n")

    with pytest.raises(SystemExit, match="zip 内容不匹配"):
        inspect(zip_path)


def test_result_inspector_requires_exact_semifinal_zip_members(tmp_path: Path) -> None:
    expected = ["result1.csv", "result2.csv", "result3.csv"]
    for name in expected:
        (tmp_path / name).write_text("mol_smiles,route\nCCO,CCCO>>CCO\n", encoding="utf-8")
    zip_path = tmp_path / "result.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name in expected:
            archive.write(tmp_path / name, arcname=name)
        archive.writestr("result1.log", "extra audit file must stay outside semifinal zip\n")

    with pytest.raises(SystemExit, match="zip 内容不匹配"):
        inspect(zip_path, expected)


def test_benchmark_prior_is_aggregate_only() -> None:
    payload = json.loads(Path("data/benchmarks/benchmark_prior.json").read_text(encoding="utf-8"))
    assert "active_summary" in payload
    assert "decoy_summary" in payload
    assert "smiles" not in json.dumps(payload).lower()


def test_data_check_covers_submission_review_artifacts(capsys) -> None:
    check_data = importlib.import_module("scripts.check_data")

    assert check_data.main() == 0

    output = capsys.readouterr().out
    assert "submission_entrypoint" in output
    assert "submission_entrypoint:targets" in output
    assert "submission_docker_build_script" in output
    assert "training_code_readme" in output
    assert "training_code_prior_script" in output
    assert "DATA_CHECK_OK" in output


def test_competition_ready_script_covers_core_gates() -> None:
    script = Path("scripts/check_competition_ready.sh")
    text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert "scripts/check_data.py" in text
    assert "scripts/check_tools.py --require-sbdd" in text
    assert "AI4S_REQUIRE_SBDD" in text
    assert "scripts/check_llm_connectivity.py" in text
    assert "chem_evolve_agent.cli" in text
    assert "scripts/inspect_result_zip.py" in text
    assert "AI4S_RUN_REAL_COMPETITION_SMOKE" in text
    assert "COMPETITION_READY_CHECK_OK" in text


def test_harness_memory_pass_requires_final_artifacts() -> None:
    text = Path("scripts/run_harness_once.sh").read_text(encoding="utf-8")

    assert "submission_artifacts_complete()" in text
    assert 'grep -q \'^OK \' "$RUN_ROOT/inspect.log"' in text
    assert 'status="incomplete"' in text
    assert 'effective_rc=3' in text
    assert "artifact_status=$artifact_status" in text
    assert '[[ "$rc" == "0" && "$artifact_status" == "complete" ]]' in text


def test_docker_defaults_match_competition_entrypoint() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")
    docker_build = Path("docker_build.sh").read_text(encoding="utf-8")
    real_smoke = Path("scripts/run_real_competition_smoke.sh").read_text(encoding="utf-8")
    code_readme = Path("Code/README.md").read_text(encoding="utf-8")
    env_example = Path(".env.example").read_text(encoding="utf-8")
    assert "AGENT_MODE=competition" in dockerfile
    assert "CHEM_EVOLVE_LLM_ENABLED=1" in dockerfile
    assert "AI4S_ROUTE_ENGINE=aizynthfinder" in dockerfile
    assert "AIZYNTHFINDER_CONFIG=/workspace/data/aizynthfinder/config.yml" in dockerfile
    assert "AI4S_AGENT_MEMORY_FILE=/workspace/data/agent_experience.jsonl" in dockerfile
    assert "SAISDATA_DIR=/saisdata" in dockerfile
    assert "CONDA_CHANNEL_ALIAS=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" in dockerfile
    assert "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple" in dockerfile
    assert "channel_alias: ${CONDA_CHANNEL_ALIAS}" in dockerfile
    assert 'python -m pip config set global.index-url "${PIP_INDEX_URL}"' in dockerfile
    assert "AI4S_ROUTE_LIMIT_PER_ROUND=10" in dockerfile
    assert "AGENT_ROUNDS=8" in dockerfile
    assert "AGENT_PER_ROUND=32" in dockerfile
    assert "AGENT_TOP_K=10" in dockerfile
    assert "AGENT_DOCKING_LIMIT=10" in dockerfile
    assert "AI4S_AGENT_MEMORY_LIMIT=20" in dockerfile
    assert "AI4S_VINA_FEEDBACK_PER_ROUND=1" in dockerfile
    assert "cp -a /workspace/app/training_code /app/training_code" in dockerfile
    assert "cp /workspace/run.sh /app/run.sh" in dockerfile
    assert "test -f /app/training_code/README.md" in dockerfile
    assert "test -f /workspace/configs/docker_llm.env" in dockerfile
    assert "! grep -q '<your-docker-llm-api-key>' /workspace/configs/docker_llm.env" in dockerfile
    assert "chmod +x /app/run.sh" in dockerfile
    assert "chmod +x /workspace/docker_build.sh" in dockerfile
    assert 'CMD ["bash", "/app/run.sh"]' in dockerfile
    assert "source /workspace/configs/docker_llm.env" in Path("run.sh").read_text(encoding="utf-8")
    assert "DOCKER_REGISTRY" in docker_build
    assert "scripts/docker_build_push.sh" in docker_build
    assert "data/benchmarks/dude/" in dockerignore
    assert "--mode competition" in real_smoke
    assert 'CHEM_EVOLVE_LLM_ENABLED="${CHEM_EVOLVE_LLM_ENABLED:-1}"' in real_smoke
    assert "AI4S_ROUTE_ENGINE" in real_smoke and "aizynthfinder" in real_smoke
    assert "competition_dock" in real_smoke
    assert "AI4S_AGENT_BASE_URL" in code_readme
    assert "OPENAI_BASE_URL" not in code_readme
    assert "AI4S_AGENT_LLM_LOG_DIR=runs/llm_io" not in env_example


def test_code_main_inspects_final_semifinal_zip(monkeypatch, tmp_path: Path) -> None:
    code_main = importlib.import_module("Code.main")
    saisdata = tmp_path / "saisdata"
    saisresult = tmp_path / "saisresult"
    calls: list[list[str]] = []
    inspections: list[tuple[Path, list[str]]] = []

    def fake_run_cli() -> None:
        calls.append(list(sys.argv))

    def fake_inspect(path: Path, expected_csvs: list[str]) -> None:
        inspections.append((path, expected_csvs))

    monkeypatch.setattr(code_main, "run_cli", fake_run_cli)
    monkeypatch.setattr(code_main, "inspect", fake_inspect)
    monkeypatch.setenv("SAISDATA_DIR", str(saisdata))
    monkeypatch.setenv("SAISRESULT_DIR", str(saisresult))
    monkeypatch.setenv("AGENT_ROUNDS", "1")
    monkeypatch.setenv("AGENT_PER_ROUND", "2")
    monkeypatch.setenv("AGENT_TOP_K", "3")
    monkeypatch.setenv("AGENT_MODE", "proxy")

    code_main.main()

    assert calls
    assert str(saisdata / "target1.pdb") in calls[0]
    assert str(saisdata / "target2.pdb") in calls[0]
    assert str(saisdata / "target3.pdb") in calls[0]
    assert inspections == [(saisresult / "result.zip", ["result1.csv", "result2.csv", "result3.csv"])]


def test_code_main_accepts_legacy_nested_saisdata(monkeypatch, tmp_path: Path) -> None:
    code_main = importlib.import_module("Code.main")
    saisdata = tmp_path / "saisdata"
    nested = saisdata / "37"
    saisresult = tmp_path / "saisresult"
    calls: list[list[str]] = []

    nested.mkdir(parents=True)
    for index in range(1, 4):
        (nested / f"target{index}.pdb").write_text("HEADER target\n", encoding="utf-8")

    monkeypatch.setattr(code_main, "run_cli", lambda: calls.append(list(sys.argv)))
    monkeypatch.setattr(code_main, "inspect", lambda path, expected_csvs: None)
    monkeypatch.setenv("SAISDATA_DIR", str(saisdata))
    monkeypatch.setenv("SAISRESULT_DIR", str(saisresult))
    monkeypatch.setenv("AGENT_MODE", "proxy")

    code_main.main()

    assert calls
    assert str(nested / "target1.pdb") in calls[0]
    assert str(nested / "target2.pdb") in calls[0]
    assert str(nested / "target3.pdb") in calls[0]
