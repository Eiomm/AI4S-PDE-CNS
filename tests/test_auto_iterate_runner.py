import importlib.util
import sys
from pathlib import Path


def _load_auto_iterate_module():
    path = Path("scripts/auto_iterate.py")
    spec = importlib.util.spec_from_file_location("auto_iterate_for_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_strategy_runner_defaults_to_legacy():
    module = _load_auto_iterate_module()
    strategy = module.Strategy.from_dict({"name": "baseline"})
    assert strategy.runner == "legacy"


def test_strategy_runner_can_select_agent():
    module = _load_auto_iterate_module()
    strategy = module.Strategy.from_dict({"name": "agent_main", "runner": "agent"})
    assert strategy.runner == "agent"


def test_agent_config_includes_agent_main_runner():
    module = _load_auto_iterate_module()
    strategies = module.load_strategies(Path("configs/strategies/agent.yaml"), ["agent_main"])
    assert len(strategies) == 1
    assert strategies[0].name == "agent_main"
    assert strategies[0].runner == "agent"


def test_session_root_groups_runs_under_experiment(tmp_path):
    module = _load_auto_iterate_module()

    session_root = module.make_session_root(tmp_path, "goal_agent", "session_a")

    assert session_root == tmp_path / "goal_agent" / "session_a"


def test_run_seed_varies_by_iteration_and_strategy():
    module = _load_auto_iterate_module()

    first = module.run_seed_for(1234, iteration=1, strategy_index=1)
    second = module.run_seed_for(1234, iteration=2, strategy_index=1)
    other_strategy = module.run_seed_for(1234, iteration=1, strategy_index=2)

    assert len({first, second, other_strategy}) == 3


def test_update_session_best_links_and_copies_artifacts(tmp_path):
    module = _load_auto_iterate_module()
    session_root = tmp_path / "goal" / "session"
    run_dir = session_root / "runs" / "i01_agent_main" / "20260611_010000"
    run_dir.mkdir(parents=True)
    (run_dir / "result.zip").write_bytes(b"zip")
    (run_dir / "result.csv").write_text("mol_smiles,route\n", encoding="utf-8")
    record = {"objective": 0.5, "run_dir": str(run_dir)}

    module.update_session_best(session_root, run_dir, record)

    assert (session_root / "best").resolve() == run_dir
    assert (session_root / "best_result.zip").read_bytes() == b"zip"
    assert (session_root / "best_record.json").exists()
