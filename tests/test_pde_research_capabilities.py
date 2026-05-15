from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from agent.pde_executor import ControlledExperimentExecutor, ExperimentExecution
from agent.pde_autonomous import AutonomousExperimentRunner
from agent.pde_journal import CandidatePlan, ExperimentJournal
from agent.pde_observer import observe_research_context
from agent.pde_planner import ALLOWED_EXPERIMENT_ACTIONS, ExperimentPlanner, parse_candidate_plan
from agent.pde_reviewer import ExperimentReviewer
from agent.logging import LLMCallLogger
from agent.run_task1_autonomous_experiment import task1_bootstrap_finetune_stride5_plan


_FINETUNE_SPEC = importlib.util.spec_from_file_location(
    "train_task1_fno_finetune",
    Path(__file__).resolve().parents[1] / "code" / "train_task1_fno_finetune.py",
)
assert _FINETUNE_SPEC is not None and _FINETUNE_SPEC.loader is not None
_FINETUNE_MODULE = importlib.util.module_from_spec(_FINETUNE_SPEC)
_FINETUNE_SPEC.loader.exec_module(_FINETUNE_MODULE)
_training_loss = _FINETUNE_MODULE._training_loss


class _RecordingClient:
    provider = "recording"
    model = "recording"

    def __init__(self):
        self.messages = []

    def complete(self, messages):
        self.messages = messages
        return {
            "content": json.dumps(
                {
                    "intent": "stop",
                    "hypothesis": "observer inspected",
                    "action_type": "stop",
                    "params": {"reason": "done"},
                    "expected_effect": "none",
                    "risk": "none",
                }
            )
        }


class _NoopExecutor:
    def execute(self, node):
        from agent.pde_executor import ExperimentExecution

        return ExperimentExecution(success=True, artifacts={"stopped": True})


class _FakeWorkflow:
    def __init__(self):
        self.checkpoint_paths = {"nu0.001": Path("official.pt")}
        self.calls = []

    def run_test_submission(self, weights, *, run_name=None, train_time=0.0, extra_inference_args=None):
        from agent.pde_results import RunResult

        self.calls.append(
            {
                "weights": dict(weights),
                "checkpoint_paths": dict(self.checkpoint_paths),
                "extra_inference_args": list(extra_inference_args or []),
            }
        )
        return RunResult(
            task_id="task1",
            run_dir=Path("runs/fake"),
            metrics={},
            prediction_path=Path("runs/fake/task1_pred.hdf5"),
            zip_path=Path("runs/fake/pred.zip"),
            train_time=float(train_time),
            inference_time=0.1,
            success=True,
            weights=dict(weights),
            command=["fake"],
        )


class _FakePostprocessWorkflow:
    def __init__(self, prediction: np.ndarray, run_root: Path):
        self.prediction = prediction
        self.run_root = run_root
        self.checkpoint_paths = {"nu0.001": Path("official.pt")}
        self.calls = []

    def run_validation(self, weights, *, run_name=None):
        from agent.pde_results import RunResult

        self.calls.append({"weights": dict(weights), "checkpoint_paths": dict(self.checkpoint_paths)})
        run_dir = self.run_root / str(run_name or "validation")
        run_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = run_dir / "task1_val_pred.hdf5"
        with h5py.File(prediction_path, "w") as h5:
            h5.create_dataset("prediction", data=self.prediction.astype(np.float32))
        return RunResult(
            task_id="task1",
            run_dir=run_dir,
            metrics={},
            prediction_path=prediction_path,
            zip_path=None,
            train_time=0.0,
            inference_time=0.0,
            success=True,
            weights=dict(weights),
            command=["fake-validation"],
        )


class _FakeCheckpointSearchWorkflow:
    def __init__(self):
        self.checkpoint_paths = {"nu0.001": Path("official.pt")}
        self.calls = []

    def run_validation(self, weights, *, run_name=None):
        from agent.pde_results import RunResult

        checkpoint = str(self.checkpoint_paths["nu0.001"])
        score = 79.0 if "stride5" in checkpoint else 5.0
        self.calls.append({"weights": dict(weights), "checkpoint_paths": dict(self.checkpoint_paths), "run_name": run_name})
        return RunResult(
            task_id="task1",
            run_dir=Path("runs") / str(run_name or "validation"),
            metrics={"competition_score_proxy": score, "mse": 1.0 / score},
            prediction_path=Path("runs") / str(run_name or "validation") / "task1_val_pred.hdf5",
            zip_path=None,
            train_time=0.0,
            inference_time=0.0,
            success=True,
            weights=dict(weights),
            command=["fake-validation"],
        )


def _write_task_hdf5(path: Path, shape: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("tensor", data=np.zeros(shape, dtype=np.float32))
        h5.create_dataset("x-coordinate", data=np.linspace(0.0, 1.0, shape[-1], dtype=np.float32))
        h5.create_dataset("t-coordinate", data=np.linspace(0.0, 1.0, shape[1], dtype=np.float32))


def test_observer_summarizes_pdebench_data_and_knowledge_base(tmp_path):
    _write_task_hdf5(tmp_path / "data" / "Task1" / "task1_val.hdf5", (3, 200, 256))
    _write_task_hdf5(tmp_path / "data" / "Task1" / "task1_test.hdf5", (5, 10, 256))
    knowledge = tmp_path / "data" / "knowledge_base"
    knowledge.mkdir(parents=True)
    (knowledge / "fno.md").write_text("# FNO\nUse temporal stride 5 for PDEBench checkpoint alignment.\n", encoding="utf-8")

    state = observe_research_context(tmp_path)

    assert state["task1"]["validation"]["datasets"]["tensor"]["shape"] == [3, 200, 256]
    assert state["task1"]["test"]["datasets"]["tensor"]["shape"] == [5, 10, 256]
    assert state["knowledge_base"][0]["path"].endswith("data/knowledge_base/fno.md")
    assert "temporal stride 5" in state["knowledge_base"][0]["preview"]


def test_planner_accepts_research_agent_action_types():
    assert {"inspect_data", "finetune_checkpoint", "evaluate_candidate", "validate_submission"} <= ALLOWED_EXPERIMENT_ACTIONS

    plan = parse_candidate_plan(
        {
            "content": json.dumps(
                {
                    "intent": "improve",
                    "hypothesis": "fine-tune the official FNO checkpoint with temporal stride 5 to match PDEBench training scale",
                    "action_type": "finetune_checkpoint",
                    "params": {
                        "temporal_stride": 5,
                        "trainable": "all",
                        "lr": 3e-6,
                        "steps": 3000,
                    },
                    "expected_effect": "improve long-horizon stability",
                    "risk": "training may overfit validation windows",
                }
            )
        }
    )

    assert plan.action_type == "finetune_checkpoint"
    assert plan.params["temporal_stride"] == 5


def test_finetune_checkpoint_action_builds_official_checkpoint_training_command(tmp_path):
    executor = ControlledExperimentExecutor(project_root=tmp_path)
    plan = CandidatePlan(
        intent="improve",
        hypothesis="stride5 fine-tune should align with official reduced_resolution_t",
        action_type="finetune_checkpoint",
        params={
            "run_dir": "runs/stride5",
            "temporal_stride": 5,
            "trainable": "all",
            "lr": 3e-6,
            "steps": 3000,
            "rollout_steps": 1,
        },
    )

    command = executor.build_finetune_checkpoint_command(plan)

    assert command[:2] == ["python", "code/train_task1_fno_finetune.py"]
    assert command[command.index("--base-checkpoint") + 1].endswith("checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt")
    assert command[command.index("--temporal-stride") + 1] == "5"
    assert command[command.index("--trainable") + 1] == "all"
    assert command[command.index("--lr") + 1] == "3e-06"
    assert command[command.index("--steps") + 1] == "3000"


def test_bootstrap_finetune_stride5_plan_starts_from_official_checkpoint():
    plan = task1_bootstrap_finetune_stride5_plan(steps=3000, lr=3e-6)

    assert plan["action_type"] == "finetune_checkpoint"
    assert plan["params"]["base_checkpoint"].endswith("checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt")
    assert plan["params"]["temporal_stride"] == 5
    assert plan["params"]["trainable"] == "all"
    assert plan["params"]["lr"] == 3e-6
    assert plan["params"]["steps"] == 3000


def test_evaluate_candidate_action_computes_task1_metrics(tmp_path):
    target = np.zeros((2, 200, 256), dtype=np.float32)
    prediction = target.copy()
    prediction[:, 10:, :] = 0.1
    pred_path = tmp_path / "runs" / "candidate.hdf5"
    target_path = tmp_path / "data" / "Task1" / "task1_val.hdf5"
    _write_task_hdf5(pred_path, prediction.shape)
    with h5py.File(pred_path, "r+") as h5:
        del h5["tensor"]
        h5.create_dataset("prediction", data=prediction)
    _write_task_hdf5(target_path, target.shape)

    plan = CandidatePlan(
        intent="improve",
        hypothesis="evaluate candidate validation prediction",
        action_type="evaluate_candidate",
        params={"prediction_path": str(pred_path), "target_path": str(target_path)},
    )
    execution = ControlledExperimentExecutor(project_root=tmp_path).execute(type("Node", (), {"plan": plan})())

    assert execution.success is True
    assert execution.metrics["mse"] > 0.0
    assert execution.artifacts["prediction_shape"] == [2, 200, 256]


def test_evaluate_candidate_action_accepts_checkpoint_path(tmp_path):
    workflow = _FakeCheckpointSearchWorkflow()
    plan = CandidatePlan(
        intent="draft",
        hypothesis="evaluate official checkpoint before choosing a fine-tune direction",
        action_type="evaluate_candidate",
        params={
            "checkpoint_path": "checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt",
            "task1_weights": {"nu0.001": 1.0},
        },
    )

    execution = ControlledExperimentExecutor(project_root=tmp_path, workflow=workflow).execute(
        type("Node", (), {"id": "node-a", "plan": plan})()
    )

    assert execution.success is True
    assert workflow.calls[0]["checkpoint_paths"]["nu0.001"] == Path(
        "checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt"
    )
    assert execution.metrics["competition_score_proxy"] == 5.0
    assert execution.artifacts["best_candidate"]["checkpoint_overrides"] == {
        "nu0.001": "checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt"
    }


def test_runner_passes_observer_context_to_planner(tmp_path):
    client = _RecordingClient()
    journal = ExperimentJournal(tmp_path / "journal.json")
    runner = AutonomousExperimentRunner(
        planner=ExperimentPlanner(
            client=client,
            logger=LLMCallLogger(tmp_path / "planner.log"),
            journal=journal,
        ),
        executor=_NoopExecutor(),
        reviewer=ExperimentReviewer(journal=journal),
        observer=lambda: {"task1": {"validation": {"datasets": {"tensor": {"shape": [3, 200, 256]}}}}},
    )

    runner.run(context={"task": "task1"}, max_iterations=1)

    payload = json.loads(client.messages[-1]["content"])
    assert payload["context"]["observer"]["task1"]["validation"]["datasets"]["tensor"]["shape"] == [3, 200, 256]


def test_submit_best_uses_finetuned_checkpoint_override_from_best_candidate(tmp_path):
    workflow = _FakeWorkflow()
    journal = ExperimentJournal(tmp_path / "journal.json")
    best = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="fine-tuned checkpoint wins validation",
            action_type="finetune_checkpoint",
            params={},
        )
    )
    journal.update_result(
        best.id,
        success=True,
        metrics={"competition_score_proxy": 79.0},
        artifacts={
            "best_candidate": {
                "name": "stride5-finetuned-fno",
                "task1_weights": {"nu0.001": 1.0},
                "checkpoint_overrides": {"nu0.001": "runs/stride5/best.pt"},
            }
        },
    )
    submit = journal.append_plan(
        CandidatePlan(
            intent="submit",
            hypothesis="submit best fine-tuned checkpoint",
            action_type="submit_best",
            params={},
        )
    )

    execution = ControlledExperimentExecutor(
        project_root=tmp_path,
        workflow=workflow,
        journal=journal,
        metric="competition_score_proxy",
        maximize=True,
    ).execute(submit)

    assert execution.success is True
    assert workflow.calls[0]["weights"] == {"nu0.001": 1.0}
    assert workflow.calls[0]["checkpoint_paths"]["nu0.001"] == Path("runs/stride5/best.pt")


def test_finetune_checkpoint_action_reads_nested_best_metrics(tmp_path):
    run_dir = tmp_path / "runs" / "finetune"
    result_path = run_dir / "finetune_result.json"
    command = [
        "python",
        "-c",
        (
            "import json, pathlib; "
            f"p=pathlib.Path(r'{result_path}'); "
            "p.parent.mkdir(parents=True, exist_ok=True); "
            "(p.parent/'best.pt').write_text('checkpoint'); "
            "p.write_text(json.dumps({'best_metrics': {'competition_score_proxy': 79.0, 'mse': 0.001}}))"
        ),
    ]
    plan = CandidatePlan(
        intent="improve",
        hypothesis="fine-tune checkpoint and parse nested metrics",
        action_type="finetune_checkpoint",
        params={
            "command": command,
            "run_dir": str(run_dir),
            "metrics_path": str(result_path),
        },
    )

    execution = ControlledExperimentExecutor(project_root=tmp_path).execute(type("Node", (), {"plan": plan})())

    assert execution.success is True
    assert execution.metrics == {"competition_score_proxy": 79.0, "mse": 0.001}
    assert execution.artifacts["best_checkpoint"].endswith("best.pt")
    assert execution.artifacts["best_candidate"]["checkpoint_overrides"]["nu0.001"].endswith("best.pt")


def test_finetune_checkpoint_action_assigns_unique_run_dir_when_planner_omits_it(tmp_path, monkeypatch):
    captured = {}

    def fake_execute_command(self, node, *, required_label):
        captured["params"] = dict(node.plan.params)
        return ExperimentExecution(success=True, metrics={}, artifacts={})

    monkeypatch.setattr(ControlledExperimentExecutor, "_execute_command_node", fake_execute_command)
    plan = CandidatePlan(
        intent="improve",
        hypothesis="planner omitted run_dir during a long autonomous run",
        action_type="finetune_checkpoint",
        params={"steps": 2000, "temporal_stride": 5, "trainable": "all"},
    )

    execution = ControlledExperimentExecutor(project_root=tmp_path).execute(
        type("Node", (), {"id": "node-xyz", "plan": plan})()
    )

    assert execution.success is True
    assert captured["params"]["run_dir"] == "runs/autonomous/node-xyz/finetune_checkpoint"
    assert captured["params"]["metrics_path"] == "runs/autonomous/node-xyz/finetune_checkpoint/finetune_result.json"


def test_finetune_checkpoint_action_rejects_reused_run_dir(tmp_path, monkeypatch):
    def fail_if_called(self, node, *, required_label):
        raise AssertionError("duplicate run_dir should be rejected before execution")

    monkeypatch.setattr(ControlledExperimentExecutor, "_execute_command_node", fail_if_called)
    journal = ExperimentJournal(tmp_path / "journal.json")
    prior = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="first run owns this directory",
            action_type="finetune_checkpoint",
            params={"run_dir": "runs/reused-finetune"},
        )
    )
    journal.update_result(prior.id, success=True, metrics={"competition_score_proxy": 1.0}, artifacts={})
    plan = CandidatePlan(
        intent="improve",
        hypothesis="planner accidentally repeats the same directory",
        action_type="finetune_checkpoint",
        params={"run_dir": "runs/reused-finetune", "steps": 2000, "temporal_stride": 5, "trainable": "all"},
    )

    execution = ControlledExperimentExecutor(project_root=tmp_path, journal=journal).execute(
        type("Node", (), {"id": "new-node", "plan": plan})()
    )

    assert execution.success is False
    assert "already used by node" in execution.error


def test_postprocess_search_uses_base_candidate_checkpoint_overrides(tmp_path):
    target = np.zeros((2, 200, 256), dtype=np.float32)
    base_prediction = target.copy()
    base_prediction[:, 10:, :] = 0.1
    target_path = tmp_path / "data" / "Task1" / "task1_val.hdf5"
    _write_task_hdf5(target_path, target.shape)
    workflow = _FakePostprocessWorkflow(base_prediction, tmp_path / "runs")
    plan = CandidatePlan(
        intent="improve",
        hypothesis="postprocess the fine-tuned FNO validation prediction",
        action_type="postprocess_search",
        params={
            "base_candidate": {
                "name": "finetuned-fno",
                "task1_weights": {"nu0.001": 1.0},
                "checkpoint_overrides": {"nu0.001": "runs/stride5/best.pt"},
            },
            "target_path": str(target_path),
            "persistence_alpha_grid": [0.0, 1.0],
        },
    )

    execution = ControlledExperimentExecutor(project_root=tmp_path, workflow=workflow).execute(type("Node", (), {"plan": plan, "id": "abc"})())

    assert execution.success is True
    assert workflow.calls[0]["weights"] == {"nu0.001": 1.0}
    assert workflow.calls[0]["checkpoint_paths"]["nu0.001"] == Path("runs/stride5/best.pt")
    assert execution.artifacts["best_candidate"]["checkpoint_overrides"] == {"nu0.001": "runs/stride5/best.pt"}
    assert execution.artifacts["best_candidate"]["task1_extra_inference_args"] == [
        "--persistence-segment-alpha",
        "0.0",
        "0.0",
        "0.0",
    ]
    assert execution.metrics["mse"] == 0.0


def test_postprocess_search_accepts_gpt_checkpoint_params(tmp_path):
    target = np.zeros((2, 200, 256), dtype=np.float32)
    base_prediction = target.copy()
    base_prediction[:, 10:, :] = 0.1
    target_path = tmp_path / "data" / "Task1" / "task1_val.hdf5"
    _write_task_hdf5(target_path, target.shape)
    workflow = _FakePostprocessWorkflow(base_prediction, tmp_path / "runs")
    plan = CandidatePlan(
        intent="improve",
        hypothesis="postprocess the checkpoint named by GPT",
        action_type="postprocess_search",
        params={
            "candidate_name": "finetuned-fno-stride5",
            "checkpoint": "runs/task1-agent-finetune-stride5-all-lr3e-6/best.pt",
            "target_path": str(target_path),
            "alpha_grid": [0.0, 1.0],
        },
    )

    execution = ControlledExperimentExecutor(project_root=tmp_path, workflow=workflow).execute(type("Node", (), {"plan": plan, "id": "abc"})())

    assert execution.success is True
    assert workflow.calls[0]["checkpoint_paths"]["nu0.001"] == Path("runs/task1-agent-finetune-stride5-all-lr3e-6/best.pt")
    assert execution.artifacts["best_candidate"]["checkpoint_overrides"] == {
        "nu0.001": "runs/task1-agent-finetune-stride5-all-lr3e-6/best.pt"
    }
    assert execution.metrics["mse"] == 0.0


def test_postprocess_search_defaults_weights_for_checkpoint_base_candidate(tmp_path):
    target = np.zeros((2, 200, 256), dtype=np.float32)
    base_prediction = target.copy()
    target_path = tmp_path / "data" / "Task1" / "task1_val.hdf5"
    _write_task_hdf5(target_path, target.shape)
    workflow = _FakePostprocessWorkflow(base_prediction, tmp_path / "runs")
    plan = CandidatePlan(
        intent="improve",
        hypothesis="postprocess fine-tuned checkpoint from GPT planner",
        action_type="postprocess_search",
        params={
            "base_candidate": {
                "name": "finetuned-fno",
                "checkpoint_overrides": {"nu0.001": "runs/stride5/best.pt"},
            },
            "target_path": str(target_path),
            "blend_alpha_grid": [0.0, 1.0],
        },
    )

    execution = ControlledExperimentExecutor(project_root=tmp_path, workflow=workflow).execute(
        type("Node", (), {"plan": plan, "id": "abc"})()
    )

    assert execution.success is True
    assert workflow.calls[0]["weights"] == {"nu0.001": 1.0}
    assert workflow.calls[0]["checkpoint_paths"]["nu0.001"] == Path("runs/stride5/best.pt")
    assert execution.artifacts["best_candidate"]["task1_weights"] == {"nu0.001": 1.0}


def test_postprocess_search_resolves_source_candidate_from_journal(tmp_path):
    target = np.zeros((2, 200, 256), dtype=np.float32)
    base_prediction = target.copy()
    base_prediction[:, 10:, :] = 0.1
    target_path = tmp_path / "data" / "Task1" / "task1_val.hdf5"
    _write_task_hdf5(target_path, target.shape)
    workflow = _FakePostprocessWorkflow(base_prediction, tmp_path / "runs")
    journal = ExperimentJournal(tmp_path / "journal.json")
    best = journal.append_plan(
        CandidatePlan(
            intent="improve",
            hypothesis="fine-tuned source",
            action_type="finetune_checkpoint",
            params={},
        )
    )
    journal.update_result(
        best.id,
        success=True,
        metrics={"competition_score_proxy": 79.0},
        artifacts={
            "best_candidate": {
                "name": "finetuned-fno",
                "task1_weights": {"nu0.001": 1.0},
                "checkpoint_overrides": {"nu0.001": "runs/stride5/best.pt"},
            }
        },
    )
    plan = CandidatePlan(
        intent="improve",
        hypothesis="postprocess source candidate by name",
        action_type="postprocess_search",
        params={
            "source_candidate": "finetuned-fno",
            "target_path": str(target_path),
            "blend_alpha_grid": [0.0, 1.0],
        },
    )

    execution = ControlledExperimentExecutor(project_root=tmp_path, workflow=workflow, journal=journal).execute(
        type("Node", (), {"plan": plan, "id": "abc"})()
    )

    assert execution.success is True
    assert workflow.calls[0]["checkpoint_paths"]["nu0.001"] == Path("runs/stride5/best.pt")
    assert execution.artifacts["best_candidate"]["checkpoint_overrides"] == {"nu0.001": "runs/stride5/best.pt"}
    assert execution.metrics["mse"] == 0.0


def test_weight_search_accepts_checkpoint_override_candidates(tmp_path):
    workflow = _FakeCheckpointSearchWorkflow()
    plan = CandidatePlan(
        intent="improve",
        hypothesis="compare fine-tuned and official FNO checkpoints",
        action_type="weight_search",
        params={
            "metric": "competition_score_proxy",
            "maximize": True,
            "candidates": [
                {
                    "name": "finetuned_fno_stride5",
                    "checkpoint_overrides": {"nu0.001": "runs/stride5/best.pt"},
                },
                {
                    "name": "official_fno",
                    "checkpoint_overrides": {"nu0.001": "checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt"},
                },
            ],
        },
    )

    execution = ControlledExperimentExecutor(project_root=tmp_path, workflow=workflow).execute(type("Node", (), {"plan": plan, "id": "abc"})())

    assert execution.success is True
    assert execution.metrics["competition_score_proxy"] == 79.0
    assert execution.artifacts["best_candidate"]["name"] == "finetuned_fno_stride5"
    assert execution.artifacts["best_candidate"]["checkpoint_overrides"] == {"nu0.001": "runs/stride5/best.pt"}
    assert workflow.calls[0]["checkpoint_paths"]["nu0.001"] == Path("runs/stride5/best.pt")


def test_weight_search_accepts_single_checkpoint_candidates(tmp_path):
    workflow = _FakeCheckpointSearchWorkflow()
    plan = CandidatePlan(
        intent="improve",
        hypothesis="compare checkpoints emitted by the LLM planner",
        action_type="weight_search",
        params={
            "metric": "competition_score_proxy",
            "maximize": True,
            "candidates": [
                {
                    "name": "official_fno",
                    "checkpoint": "checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt",
                },
                {
                    "name": "finetuned_stride5",
                    "checkpoint": "runs/stride5/best.pt",
                },
            ],
        },
    )

    execution = ControlledExperimentExecutor(project_root=tmp_path, workflow=workflow).execute(
        type("Node", (), {"plan": plan, "id": "abc"})()
    )

    assert execution.success is True
    assert execution.metrics["competition_score_proxy"] == 79.0
    assert execution.artifacts["best_candidate"]["name"] == "finetuned_stride5"
    assert execution.artifacts["best_candidate"]["checkpoint_overrides"] == {"nu0.001": "runs/stride5/best.pt"}
    assert workflow.calls[0]["checkpoint_paths"]["nu0.001"] == Path("checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt")
    assert workflow.calls[1]["checkpoint_paths"]["nu0.001"] == Path("runs/stride5/best.pt")


def test_training_loss_supports_scientific_stability_terms():
    class ShiftModel(torch.nn.Module):
        def forward(self, features, grid):
            del grid
            return features[:, :, -1:] + 0.5

    inputs = torch.zeros((2, 10, 8), dtype=torch.float32)
    targets = torch.zeros((2, 8), dtype=torch.float32)

    base = _training_loss(
        ShiftModel(),
        inputs,
        targets,
        rollout_steps=1,
        gradient_loss_weight=0.0,
        spectral_loss_weight=0.0,
    )
    stabilized = _training_loss(
        ShiftModel(),
        inputs,
        targets,
        rollout_steps=1,
        gradient_loss_weight=0.25,
        spectral_loss_weight=0.25,
    )

    assert stabilized > base


def test_training_loss_supports_burgers_physics_residual():
    class ShiftModel(torch.nn.Module):
        def forward(self, features, grid):
            del grid
            return features[:, :, -1:] + 0.5

    inputs = torch.zeros((2, 10, 8), dtype=torch.float32)
    targets = torch.zeros((2, 8), dtype=torch.float32)

    base = _training_loss(
        ShiftModel(),
        inputs,
        targets,
        rollout_steps=1,
        gradient_loss_weight=0.0,
        spectral_loss_weight=0.0,
        physics_loss_weight=0.0,
    )
    physics_regularized = _training_loss(
        ShiftModel(),
        inputs,
        targets,
        rollout_steps=1,
        gradient_loss_weight=0.0,
        spectral_loss_weight=0.0,
        physics_loss_weight=0.25,
        physics_nu=0.001,
        physics_dt=0.025,
        physics_dx=1.0 / 256.0,
    )

    assert physics_regularized > base


def test_finetune_checkpoint_command_exposes_scientific_loss_knobs(tmp_path):
    plan = CandidatePlan(
        intent="improve",
        hypothesis="reduce long-horizon drift with Sobolev and spectral regularization",
        action_type="finetune_checkpoint",
        params={
            "gradient_loss_weight": 0.05,
            "spectral_loss_weight": 0.001,
            "physics_loss_weight": 0.002,
            "physics_nu": 0.001,
            "physics_dt": 0.025,
            "physics_dx": 1.0 / 256.0,
            "horizon_loss_gamma": 1.1,
            "architecture": "residual-corrected-fno",
        },
    )

    command = ControlledExperimentExecutor(project_root=tmp_path).build_finetune_checkpoint_command(plan)

    assert "--gradient-loss-weight" in command
    assert command[command.index("--gradient-loss-weight") + 1] == "0.05"
    assert "--spectral-loss-weight" in command
    assert command[command.index("--spectral-loss-weight") + 1] == "0.001"
    assert "--physics-loss-weight" in command
    assert command[command.index("--physics-loss-weight") + 1] == "0.002"
    assert "--physics-nu" in command
    assert command[command.index("--physics-nu") + 1] == "0.001"
    assert "--physics-dt" in command
    assert command[command.index("--physics-dt") + 1] == "0.025"
    assert "--physics-dx" in command
    assert command[command.index("--physics-dx") + 1] == str(1.0 / 256.0)
    assert "--horizon-loss-gamma" in command
    assert command[command.index("--horizon-loss-gamma") + 1] == "1.1"
    assert "--architecture" in command
    assert command[command.index("--architecture") + 1] == "residual-corrected-fno"


def test_planner_prompt_exposes_physics_loss_knobs(tmp_path):
    client = _RecordingClient()
    journal = ExperimentJournal(tmp_path / "journal.json")
    planner = ExperimentPlanner(
        client=client,
        logger=LLMCallLogger(tmp_path / "planner.log"),
        journal=journal,
        metric="competition_score_proxy",
        maximize=True,
    )

    planner.plan_next({"task": "task1", "best_metric": 79.0168964})

    prompt = "\n".join(message["content"] for message in client.messages)
    assert "gradient_loss_weight" in prompt
    assert "spectral_loss_weight" in prompt
    assert "physics_loss_weight" in prompt
    assert "residual-corrected-fno" in prompt
    assert "horizon_loss_gamma" in prompt
