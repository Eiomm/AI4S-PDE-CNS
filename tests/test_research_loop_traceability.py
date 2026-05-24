import json
from argparse import Namespace
from pathlib import Path

from mlevolve_lite import scheduler
from mlevolve_lite.code_extractor import extract_python_code
from mlevolve_lite.evaluator import Evaluator
from mlevolve_lite.llm_backend import LLMBackend
from mlevolve_lite.node_schema import Metrics, Node, Operator
from mlevolve_lite.operators import apply_llm_to_child, create_child_workspace


class FakeLLM(LLMBackend):
    def __init__(self, response: str):
        self.model = "fake-research-model"
        self.response = response

    def chat(self, messages):  # noqa: ANN001
        assert messages
        return self.response


def _parent(tmp_path: Path) -> Node:
    code_dir = tmp_path / "parent" / "code"
    artifact_dir = tmp_path / "parent" / "artifacts"
    code_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "train.py").write_text("print('parent')\n", encoding="utf-8")
    return Node(
        node_id="multi_nu_fno_baseline",
        signature="multi_nu_fno_baseline",
        parent_ids=[],
        operator=Operator.DRAFT,
        hypothesis="Spectral baseline",
        code_dir=str(code_dir),
        artifact_dir=str(artifact_dir),
        lineage=["multi_nu_fno_baseline"],
    )


def test_child_workspace_uses_readable_research_node_names(tmp_path: Path) -> None:
    child = create_child_workspace(
        tmp_path,
        _parent(tmp_path),
        Operator.IMPROVE,
        round_index=7,
        branch_index=3,
        node_slug="Long rollout loss + residual delta",
    )

    assert child.node_id.startswith("r007_b03_long_rollout_loss_residual_delta")
    assert child.signature == "long_rollout_loss_residual_delta"
    assert Path(child.code_dir, "train.py").exists()


def test_parent_selection_uses_best_node_per_readable_signature(tmp_path: Path) -> None:
    weak = _parent(tmp_path)
    weak.node_id = "r001_b01_long_rollout_stability"
    weak.signature = "long_rollout_stability"
    weak.status = "cheap_probe_passed"
    weak.metrics = Metrics(reward=-0.1, compliance_pass=True)
    weak.mean_score = -0.1

    strong = _parent(tmp_path)
    strong.node_id = "r002_b01_long_rollout_stability"
    strong.signature = "long_rollout_stability"
    strong.status = "cheap_probe_passed"
    strong.metrics = Metrics(reward=0.7, compliance_pass=True)
    strong.mean_score = 0.7

    diverse = _parent(tmp_path)
    diverse.node_id = "conditional_fno_marginalized_nu"
    diverse.signature = "conditional_fno_marginalized_nu"
    diverse.status = "cheap_probe_passed"
    diverse.metrics = Metrics(reward=0.2, compliance_pass=True)
    diverse.mean_score = 0.2

    selected = scheduler._select_parents([weak, strong, diverse], k=2)

    assert [node.node_id for node in selected] == [
        "r002_b01_long_rollout_stability",
        "conditional_fno_marginalized_nu",
    ]


def test_llm_code_write_is_traceable_to_response_field(tmp_path: Path) -> None:
    child = create_child_workspace(
        tmp_path,
        _parent(tmp_path),
        Operator.IMPROVE,
        round_index=1,
        branch_index=1,
        node_slug="rollout stability",
    )
    response = """### Hypothesis name
rollout stability

### What changes
Use a longer rollout loss and internal AutoML trial list.

```python
AUTO_ML_SEARCH_SPACE = {"lr": [1e-3, 5e-4], "rollout": [10, 20]}
print("traceable child")
```
"""

    ok, err = apply_llm_to_child(
        child=child,
        parent=_parent(tmp_path),
        all_nodes=[],
        llm=FakeLLM(response),
        prompts_dir=tmp_path,
        research_log_path=tmp_path / "task2_logs.jsonl",
    )

    assert ok, err
    entries = [json.loads(line) for line in (tmp_path / "task2_logs.jsonl").read_text(encoding="utf-8").splitlines()]
    local_entries = [
        json.loads(line)
        for line in (Path(child.code_dir).parent / "logs" / "research_loop.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    code_events = [entry for entry in entries if entry["event"] == "code_written"]
    local_code_events = [entry for entry in local_entries if entry["event"] == "code_written"]
    assert code_events
    assert local_code_events
    event = code_events[0]
    local_event = local_code_events[0]
    assert event["node_id"] == child.node_id
    assert event["response"] == response
    assert event["response_id"] == child.response_id
    assert event["files_written"][0]["path"].endswith("train.py")
    assert event["files_written"][0]["sha256"]
    assert local_event["response"] == response
    assert local_event["response_id"] == child.response_id
    assert "AUTO_ML_SEARCH_SPACE" in Path(child.code_dir, "train.py").read_text(encoding="utf-8")


def test_parallel_scheduler_records_research_loop_events(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    class SchedulerFakeLLM(LLMBackend):
        def __init__(self, **kwargs):  # noqa: ANN003
            self.model = kwargs.get("model", "fake-scheduler-model")

        def chat(self, messages):  # noqa: ANN001
            assert messages
            return """### Hypothesis name
Long rollout stability

### Why this path
Let the candidate tune its own small search space instead of hardcoding scheduler params.

```python
AUTO_ML_SEARCH_SPACE = {"lr": [1e-3, 5e-4], "rollout_steps": [8, 16]}
print("scheduler traceable child")
```
"""

    class FakeEvaluator:
        def __init__(self, data_dir):  # noqa: ANN001
            self.data_dir = Path(data_dir)

        def static_check(self, code_dir):  # noqa: ANN001
            assert (Path(code_dir) / "train.py").exists()
            return True, []

        def validation_metrics(self, val_pred_path, runtime_sec=None):  # noqa: ANN001
            return Metrics(
                overall_mse=0.25,
                runtime_sec=runtime_sec,
                shape_pass=True,
                first_10_pass=True,
                compliance_pass=True,
                official_score_estimate=75.0,
                reward=0.5,
            )

        def shape_check(self, pred_path):  # noqa: ANN001
            return Metrics(shape_pass=True, first_10_pass=True, compliance_pass=True)

        @staticmethod
        def compute_reward(metrics: Metrics) -> float:
            return metrics.reward

    class FakeGPUQueue:
        def __init__(self, workspace):  # noqa: ANN001
            self.workspace = Path(workspace)

        def run(self, node_id, cmd, log_path, timeout_sec):  # noqa: ANN001
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("fake training completed\n", encoding="utf-8")
            return {"status": "completed", "returncode": 0, "runtime_sec": 0.01, "cmd": cmd}

    monkeypatch.setattr(scheduler, "LLMBackend", SchedulerFakeLLM)
    monkeypatch.setattr(scheduler, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(scheduler, "GPUQueue", FakeGPUQueue)

    args = Namespace(
        data_dir=str(tmp_path / "data"),
        cheap_epochs=1,
        timeout_sec=30,
        use_llm=True,
        llm_base_url=None,
        llm_api_key="fake-key",
        llm_model="fake-scheduler-model",
        llm_reasoning_effort=None,
        children_per_round=2,
        parallel_llm=True,
        parallel_gpu=2,
    )
    db = scheduler.GraphDB(tmp_path)
    graph = scheduler.initialize_graph(db)

    children = scheduler.run_round(args, db, graph, round_index=1)

    assert len(children) == 2
    assert {child.node_id[:8] for child in children} == {"r001_b01", "r001_b02"}
    assert all(child.response_id for child in children)

    entries = [json.loads(line) for line in (tmp_path / "task2_logs.jsonl").read_text(encoding="utf-8").splitlines()]
    run_dirs = sorted((tmp_path / "runs").glob("run_*"))
    assert len(run_dirs) == 1
    run_entries = [json.loads(line) for line in (run_dirs[0] / "task2_logs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert entries
    assert run_entries
    assert {entry["run_id"] for entry in run_entries} == {run_dirs[0].name}
    assert all("timestamp" in entry for entry in entries)
    assert all("elapsed_seconds" in entry for entry in entries)
    assert all("response" in entry or "tool_calls" in entry for entry in entries)

    events_by_node: dict[str, set[str]] = {}
    response_ids_by_node: dict[str, set[str]] = {}
    for entry in entries:
        node_id = entry.get("node_id")
        if not node_id:
            continue
        events_by_node.setdefault(node_id, set()).add(entry["event"])
        if entry.get("response_id"):
            response_ids_by_node.setdefault(node_id, set()).add(entry["response_id"])

    for child in children:
        node_log = tmp_path / "nodes" / child.node_id / "logs" / "research_loop.jsonl"
        node_entries = [json.loads(line) for line in node_log.read_text(encoding="utf-8").splitlines()]
        run_node_dir = run_dirs[0] / "nodes" / child.node_id
        run_node_log = run_node_dir / "research_loop.jsonl"
        run_node_entries = [json.loads(line) for line in run_node_log.read_text(encoding="utf-8").splitlines()]
        node_events = {entry["event"] for entry in node_entries}
        run_node_events = {entry["event"] for entry in run_node_entries}
        assert {
            "llm_response",
            "code_written",
            "trial_started",
            "trial_finished",
            "metrics_recorded",
        }.issubset(events_by_node[child.node_id])
        assert {
            "child_created",
            "llm_response",
            "code_written",
            "trial_started",
            "trial_finished",
            "metrics_recorded",
        }.issubset(node_events)
        assert node_events.issubset(run_node_events)
        assert all(entry.get("node_id") == child.node_id for entry in node_entries)
        assert all(entry.get("node_id") == child.node_id for entry in run_node_entries)
        assert (run_node_dir / "node.json").exists()
        assert (run_node_dir / "code" / "train.py").exists()
        assert response_ids_by_node[child.node_id] == {child.response_id}


def test_prompts_start_from_hypothesis_and_delegate_automl() -> None:
    prompts_dir = Path(scheduler.__file__).parent / "prompts"
    improve_prompt = (prompts_dir / "improve.md").read_text(encoding="utf-8")
    draft_prompt = (prompts_dir / "draft.md").read_text(encoding="utf-8")

    assert "### Hypothesis name" in improve_prompt
    assert "### Hypothesis name" in draft_prompt
    assert "AUTO_ML_SEARCH_SPACE" in improve_prompt
    assert "AUTO_ML_SEARCH_SPACE" in draft_prompt
    assert "hidden_channels ≥ 128" not in improve_prompt
    assert "fno_modes ≥ 32" not in improve_prompt


def test_full_train_uses_same_metrics_log_without_scheduler_trial_params(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    commands: list[list[str]] = []

    class FakeEvaluator:
        def __init__(self, data_dir):  # noqa: ANN001
            self.data_dir = Path(data_dir)

        def validation_metrics(self, val_pred_path, runtime_sec=None):  # noqa: ANN001
            return Metrics(
                overall_mse=0.2,
                runtime_sec=runtime_sec,
                shape_pass=True,
                first_10_pass=True,
                compliance_pass=True,
                official_score_estimate=80.0,
                reward=0.6,
            )

        def shape_check(self, pred_path):  # noqa: ANN001
            return Metrics(shape_pass=True, first_10_pass=True, compliance_pass=True)

        @staticmethod
        def compute_reward(metrics: Metrics) -> float:
            return metrics.reward

    class FakeGPUQueue:
        def __init__(self, workspace):  # noqa: ANN001
            self.workspace = Path(workspace)

        def run(self, node_id, cmd, log_path, timeout_sec):  # noqa: ANN001
            commands.append(cmd)
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("fake full train completed\n", encoding="utf-8")
            return {"status": "completed", "returncode": 0, "runtime_sec": 0.02, "cmd": cmd}

    monkeypatch.setattr(scheduler, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(scheduler, "GPUQueue", FakeGPUQueue)

    code_dir = tmp_path / "nodes" / "n1" / "code"
    artifact_dir = tmp_path / "nodes" / "n1" / "artifacts"
    code_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    (code_dir / "train.py").write_text("print('full train')\n", encoding="utf-8")
    node = Node(
        node_id="n1",
        signature="long_rollout_stability",
        parent_ids=[],
        operator=Operator.IMPROVE,
        hypothesis="Long rollout stability",
        code_dir=str(code_dir),
        artifact_dir=str(artifact_dir),
        lineage=["n1"],
        log_path=str(tmp_path / "nodes" / "n1" / "logs" / "train.log"),
        response_id="n1__resp_test",
    )
    args = Namespace(
        data_dir=str(tmp_path / "data"),
        full_epochs=3,
        timeout_sec=30,
    )
    db = scheduler.GraphDB(tmp_path)
    graph = db.load_graph()
    db.upsert_node(graph, node)
    db.save_graph(graph)

    scheduler._run_full_train(args, db, graph, "n1")

    assert commands
    assert "--batch-size" not in commands[0]
    assert "--lr" not in commands[0]

    entries = [json.loads(line) for line in (tmp_path / "task2_logs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert entries
    assert all("elapsed_seconds" in entry for entry in entries)
    assert all("response" in entry or "tool_calls" in entry for entry in entries)
    assert any(entry["event"] == "trial_started" and entry.get("stage") == "full_train" for entry in entries)
    assert any(entry["event"] == "trial_finished" and entry.get("stage") == "full_train" for entry in entries)
    assert any(entry["event"] == "metrics_recorded" and entry.get("stage") == "full_train" for entry in entries)

    node_entries = [
        json.loads(line)
        for line in (tmp_path / "nodes" / "n1" / "logs" / "research_loop.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [entry["event"] for entry in node_entries] == ["trial_started", "trial_finished", "metrics_recorded"]


def test_auto_full_train_runs_after_each_round_and_logs_feedback(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    full_trained: list[str] = []

    def fake_run_round(args, db, graph, round_index=None, run_context=None):  # noqa: ANN001
        node_id = f"probe_{round_index}"
        code_dir = tmp_path / "nodes" / node_id / "code"
        artifact_dir = tmp_path / "nodes" / node_id / "artifacts"
        code_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "train.py").write_text("print('probe')\n", encoding="utf-8")
        node = Node(
            node_id=node_id,
            signature=f"probe-{round_index}",
            parent_ids=[],
            operator=Operator.IMPROVE,
            hypothesis=f"Probe {round_index}",
            code_dir=str(code_dir),
            artifact_dir=str(artifact_dir),
            log_path=str(tmp_path / "nodes" / node_id / "logs" / "train.log"),
            response_id=f"{node_id}__resp",
            metrics=Metrics(
                overall_mse=0.3 - (0.05 * round_index),
                worst_nu_mse=0.5,
                compliance_pass=True,
                shape_pass=True,
                first_10_pass=True,
                reward=0.2 + (0.1 * round_index),
            ),
            status="cheap_probe_passed",
        )
        db.upsert_node(graph, node)
        db.save_graph(graph)
        return [node]

    def fake_run_full_train(args, db, graph, node_id, run_context=None):  # noqa: ANN001
        full_trained.append(node_id)
        node = db.get_node(graph, node_id)
        node.metrics = Metrics(
            overall_mse=0.1,
            worst_nu_mse=0.2,
            compliance_pass=True,
            shape_pass=True,
            first_10_pass=True,
            reward=0.9,
        )
        node.update_status("full_train_passed")
        db.upsert_node(graph, node)
        db.save_graph(graph)
        return node.to_dict()

    monkeypatch.setattr(scheduler, "run_round", fake_run_round)
    monkeypatch.setattr(scheduler, "_run_full_train", fake_run_full_train)

    rc = scheduler.main(
        [
            "--workspace",
            str(tmp_path),
            "--data-dir",
            str(tmp_path / "data"),
            "--rounds",
            "2",
            "--auto-full-train",
            "--full-train-top-k",
            "1",
            "--run-id",
            "autonomous_test",
        ]
    )

    assert rc == 0
    assert full_trained == ["probe_1", "probe_2"]

    entries = [json.loads(line) for line in (tmp_path / "runs" / "autonomous_test" / "task2_logs.jsonl").read_text(encoding="utf-8").splitlines()]
    feedback = [entry for entry in entries if entry["event"] == "llm_feedback_reflection"]
    assert len(feedback) == 2
    assert all("response" in entry for entry in feedback)
    assert all("next LLM round" in entry["response"] for entry in feedback)


def test_auto_full_train_convergence_patience_stops_loop(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    rounds: list[int] = []

    def fake_run_round(args, db, graph, round_index=None, run_context=None):  # noqa: ANN001
        rounds.append(round_index)
        node_id = f"probe_{round_index}"
        code_dir = tmp_path / "nodes" / node_id / "code"
        artifact_dir = tmp_path / "nodes" / node_id / "artifacts"
        code_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "train.py").write_text("print('probe')\n", encoding="utf-8")
        node = Node(
            node_id=node_id,
            signature=f"probe-{round_index}",
            parent_ids=[],
            operator=Operator.IMPROVE,
            hypothesis=f"Probe {round_index}",
            code_dir=str(code_dir),
            artifact_dir=str(artifact_dir),
            log_path=str(tmp_path / "nodes" / node_id / "logs" / "train.log"),
            metrics=Metrics(compliance_pass=True, shape_pass=True, first_10_pass=True, reward=0.5),
            status="cheap_probe_passed",
        )
        db.upsert_node(graph, node)
        db.save_graph(graph)
        return [node]

    def fake_run_full_train(args, db, graph, node_id, run_context=None):  # noqa: ANN001
        node = db.get_node(graph, node_id)
        node.metrics = Metrics(compliance_pass=True, shape_pass=True, first_10_pass=True, reward=0.5)
        node.update_status("full_train_passed")
        db.upsert_node(graph, node)
        db.save_graph(graph)
        return node.to_dict()

    monkeypatch.setattr(scheduler, "run_round", fake_run_round)
    monkeypatch.setattr(scheduler, "_run_full_train", fake_run_full_train)

    scheduler.main(
        [
            "--workspace",
            str(tmp_path),
            "--data-dir",
            str(tmp_path / "data"),
            "--rounds",
            "5",
            "--auto-full-train",
            "--convergence-patience",
            "1",
            "--min-reward-delta",
            "0.01",
            "--run-id",
            "convergence_test",
        ]
    )

    assert rounds == [1, 2]
    entries = [json.loads(line) for line in (tmp_path / "runs" / "convergence_test" / "task2_logs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(entry["event"] == "convergence_stopped" for entry in entries)


def test_extract_python_code_accepts_unclosed_final_fence() -> None:
    response = """### Hypothesis name
Nu search branch

```python
AUTO_ML_SEARCH_SPACE = {"hidden": [64, 128]}
print("still useful even if the model output was truncated")
"""

    code = extract_python_code(response)

    assert code is not None
    assert "AUTO_ML_SEARCH_SPACE" in code
    assert "```" not in code


def test_static_check_rejects_truncated_generated_python(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "train.py").write_text("def main():\n    message = 'truncated\n", encoding="utf-8")

    ok, reasons = Evaluator(tmp_path / "data").static_check(code_dir)

    assert not ok
    assert any(reason.startswith("syntax_error:train.py") for reason in reasons)
