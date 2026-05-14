import json

import h5py
import numpy as np
import pytest

from agent.pde_baseline_losses import initial_consistency_mse, spectral_mse
from agent.run_task1_baseline_zoo import (
    build_task1_fno_workflow,
    run_task1_baseline_zoo,
    run_validation_ensembles,
)
from agent.physicsnemo_adapter import PhysicsNeMoStatus
from agent.task1_baseline_train import build_model, normalize_loss_window
from agent.task1_baseline_train import torch_burgers_residual_mse, torch_spectral_mse


def _write_hdf5(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("tensor", data=data.astype(np.float32))


def test_spectral_and_initial_consistency_losses_are_zero_for_matching_inputs():
    prediction = np.zeros((2, 200, 256), dtype=np.float32)
    target = prediction.copy()
    initial = prediction[:, :10, :].copy()

    assert spectral_mse(prediction, target) == 0.0
    assert initial_consistency_mse(prediction, initial) == 0.0


def test_normalize_loss_window_rejects_invalid_tail_range():
    assert normalize_loss_window(200, 120, None) == (120, 200)
    assert normalize_loss_window(200, 10, 160) == (10, 160)
    try:
        normalize_loss_window(200, 200, None)
    except ValueError as exc:
        assert "loss_start_step" in str(exc)
    else:
        raise AssertionError("expected invalid loss window to fail")


def test_run_task1_baseline_zoo_fake_model_writes_journal_and_summary(tmp_path):
    data_dir = tmp_path / "data" / "Task1"
    initial = np.zeros((2, 10, 256), dtype=np.float32)
    target = np.concatenate([initial, np.ones((2, 190, 256), dtype=np.float32)], axis=1)
    _write_hdf5(data_dir / "task1_test.hdf5", initial)
    _write_hdf5(data_dir / "task1_val.hdf5", target)

    summary_path = run_task1_baseline_zoo(
        project_root=tmp_path,
        study_name="zoo-smoke",
        models=["fake"],
        max_samples=2,
        steps=1,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    study_dir = tmp_path / "runs" / "zoo-smoke"
    assert summary["study_name"] == "zoo-smoke"
    assert summary["results"][0]["model"] == "fake"
    assert (study_dir / "journal.json").is_file()
    assert (study_dir / "experiment_comparison.csv").is_file()
    assert (study_dir / "candidate_comparison.csv").is_file()


def test_build_task1_fno_workflow_applies_checkpoint_override(tmp_path):
    checkpoint = tmp_path / "runs" / "finetune" / "best.pt"

    workflow = build_task1_fno_workflow(
        project_root=tmp_path,
        study_dir=tmp_path / "runs" / "study",
        checkpoint_overrides={"nu0.001": checkpoint},
    )

    assert workflow.checkpoint_paths["nu0.001"] == checkpoint


def test_run_task1_baseline_zoo_train_config_is_recorded_for_fake(tmp_path):
    data_dir = tmp_path / "data" / "Task1"
    initial = np.zeros((2, 10, 256), dtype=np.float32)
    target = np.concatenate([initial, np.ones((2, 190, 256), dtype=np.float32)], axis=1)
    _write_hdf5(data_dir / "task1_test.hdf5", initial)
    _write_hdf5(data_dir / "task1_val.hdf5", target)

    summary_path = run_task1_baseline_zoo(
        project_root=tmp_path,
        study_name="zoo-config",
        models=["fake"],
        max_samples=7,
        steps=11,
        batch_size=3,
        lr=2.0e-4,
        hidden=12,
        device="cpu",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["train_config"] == {
        "max_samples": 7,
        "steps": 11,
        "batch_size": 3,
        "lr": 2.0e-4,
        "hidden": 12,
        "device": "cpu",
        "loss_start_step": 10,
        "loss_end_step": None,
        "base_train_hdf5": [],
        "base_validation_prediction_path": None,
        "initial_loss_weight": 0.05,
        "spectral_loss_weight": 0.0,
        "spectral_high_weight": 2.0,
        "physics_loss_weight": 0.0,
        "physics_nu": 0.001,
        "physics_dt": 0.05,
        "physics_dx": 1.0 / 256.0,
    }


def test_torch_physics_and_spectral_losses_are_zero_for_constant_solution():
    torch = pytest.importorskip("torch")
    trajectory = torch.ones((2, 20, 32), dtype=torch.float32)

    assert float(torch_burgers_residual_mse(trajectory, start_step=10, end_step=20)) == 0.0
    assert float(torch_spectral_mse(trajectory, trajectory)) == 0.0


def test_run_task1_baseline_zoo_records_tail_loss_window(tmp_path):
    data_dir = tmp_path / "data" / "Task1"
    initial = np.zeros((2, 10, 256), dtype=np.float32)
    target = np.concatenate([initial, np.ones((2, 190, 256), dtype=np.float32)], axis=1)
    _write_hdf5(data_dir / "task1_test.hdf5", initial)
    _write_hdf5(data_dir / "task1_val.hdf5", target)

    summary_path = run_task1_baseline_zoo(
        project_root=tmp_path,
        study_name="zoo-tail-loss",
        models=["fake"],
        max_samples=2,
        steps=1,
        loss_start_step=120,
        loss_end_step=200,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["train_config"]["loss_start_step"] == 120
    assert summary["train_config"]["loss_end_step"] == 200


def test_residual_refiner_requires_base_and_zero_correction_preserves_base():
    torch = pytest.importorskip("torch")

    model = build_model("residual_refiner", spatial_size=256, output_steps=200, hidden=4)
    for parameter in model.parameters():
        parameter.data.zero_()

    initial = torch.zeros((2, 10, 256), dtype=torch.float32)
    base = torch.ones((2, 200, 256), dtype=torch.float32)
    base[:, :10, :] = initial

    try:
        model(initial)
    except ValueError as exc:
        assert "base" in str(exc)
    else:
        raise AssertionError("residual_refiner must require a base prediction")

    prediction = model(initial, base=base)

    assert torch.allclose(prediction, base)


def test_residual_refiner_initializes_as_base_identity():
    torch = pytest.importorskip("torch")

    model = build_model("residual_refiner", spatial_size=256, output_steps=200, hidden=4)
    initial = torch.zeros((2, 10, 256), dtype=torch.float32)
    base = torch.ones((2, 200, 256), dtype=torch.float32)
    base[:, :10, :] = initial

    prediction = model(initial, base=base)

    assert torch.allclose(prediction, base)


def test_run_task1_baseline_zoo_passes_refiner_base_paths_to_trainer(tmp_path, monkeypatch):
    import agent.run_task1_baseline_zoo as zoo

    captured = {}

    def fake_train_task1_baseline(**kwargs):
        captured.update(kwargs)
        run_dir = kwargs["run_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = run_dir / "task1_val_pred.hdf5"
        _write_hdf5(prediction_path, np.zeros((2, 200, 256), dtype=np.float32))
        from agent.pde_results import RunResult

        return RunResult(
            task_id="task1",
            run_dir=run_dir,
            metrics={"mse": 0.1, "competition_score_proxy": 1.0},
            prediction_path=prediction_path,
            success=True,
            command=["fake-train"],
        )

    monkeypatch.setattr(zoo, "train_task1_baseline", fake_train_task1_baseline, raising=False)
    monkeypatch.setattr(zoo, "_torch_available", lambda: True)
    base_train = tmp_path / "base_train.hdf5"
    base_val = tmp_path / "base_val.hdf5"
    _write_hdf5(base_train, np.zeros((2, 200, 256), dtype=np.float32))
    _write_hdf5(base_val, np.zeros((2, 200, 256), dtype=np.float32))

    run_task1_baseline_zoo(
        project_root=tmp_path,
        study_name="zoo-refiner-base",
        models=["residual_refiner"],
        max_samples=2,
        steps=1,
        base_train_hdf5=[base_train],
        base_validation_prediction_path=base_val,
        physics_loss_weight=0.002,
        spectral_loss_weight=0.003,
    )

    assert captured["model_name"] == "residual_refiner"
    assert captured["base_train_hdf5"] == [base_train]
    assert captured["base_validation_prediction_path"] == base_val
    assert captured["physics_loss_weight"] == 0.002
    assert captured["spectral_loss_weight"] == 0.003


def test_run_task1_baseline_zoo_skips_physicsnemo_when_adapter_unusable(tmp_path, monkeypatch):
    import agent.run_task1_baseline_zoo as zoo

    monkeypatch.setattr(
        zoo,
        "physicsnemo_status",
        lambda: PhysicsNeMoStatus(
            installed=False,
            usable=False,
            import_name="physicsnemo",
            package_name="nvidia-physicsnemo",
            current_python="3.10.18",
            latest_package_python_supported=False,
            version=None,
            reason="physicsnemo is not installed; latest nvidia-physicsnemo requires Python >=3.11",
            recommendation="Use an isolated Python 3.11 environment, or pin nvidia-physicsnemo==1.3.* for Hwpytorch.",
        ),
    )

    summary_path = run_task1_baseline_zoo(
        project_root=tmp_path,
        study_name="zoo-physicsnemo-skip",
        models=["physicsnemo_fno"],
        max_samples=2,
        steps=1,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = summary["results"][0]
    assert result["model"] == "physicsnemo_fno"
    assert result["success"] is False
    assert result["skipped"] is True
    assert "Python >=3.11" in result["error"]


def test_run_validation_ensembles_writes_global_and_cluster_candidates(tmp_path):
    target = np.zeros((4, 200, 256), dtype=np.float32)
    target[2:, :, :] = 3.0
    low = np.zeros_like(target)
    high = np.full_like(target, 3.0)
    high[:2, :, :] = 3.0
    low[2:, :, :] = 0.0

    results = run_validation_ensembles(
        study_dir=tmp_path / "study",
        target=target,
        predictions={"low": low, "high": high},
        grid_step=0.5,
    )

    assert {result["name"] for result in results} == {"global_ensemble", "cluster_em_ensemble"}
    assert (tmp_path / "study" / "global_ensemble" / "metrics.json").is_file()
    assert (tmp_path / "study" / "cluster_em_ensemble" / "baseline_manifest.json").is_file()
