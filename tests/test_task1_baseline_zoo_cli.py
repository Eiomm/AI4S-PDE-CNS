import json

import h5py
import numpy as np

from agent.pde_baseline_losses import initial_consistency_mse, spectral_mse
from agent.run_task1_baseline_zoo import (
    build_task1_fno_workflow,
    run_task1_baseline_zoo,
    run_validation_ensembles,
)


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
        checkpoint_overrides={"nu0.1": checkpoint},
    )

    assert workflow.checkpoint_paths["nu0.1"] == checkpoint


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
    }


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
