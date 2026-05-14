import json
from pathlib import Path

import h5py
import numpy as np

from agent.task1_auto_explorer import run_task1_auto_explorer
from agent.task1_combo_space import ComboSearchConfig, search_task1_combinations


def _write_prediction(path: Path, data: np.ndarray, *, key: str = "prediction") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset(key, data=data.astype(np.float32))


def test_dense_temporal_tail_blend_finds_non_handcoded_cut():
    target = np.zeros((2, 200, 8), dtype=np.float32)
    base = target.copy()
    base[:, 113:, :] = 1.0
    tail = target.copy()
    tail[:, 105:113, :] = 1.0

    results = search_task1_combinations(
        predictions={"fno_ensemble": base, "deeponet_lite": tail},
        target=target,
        config=ComboSearchConfig(
            include_global=False,
            include_cluster=False,
            include_piecewise=False,
            temporal_cut_min=105,
            temporal_cut_max=120,
            temporal_cut_stride=1,
            temporal_weight_step=1.0,
            top_k=5,
        ),
    )

    best = results[0]
    assert best.kind == "temporal_tail_blend"
    assert best.config["cut"] == 113
    assert best.config["tail_weight"] == 1.0
    assert best.metrics["mse"] == 0.0


def test_piecewise_temporal_blend_can_use_different_tail_weights_by_horizon():
    target = np.zeros((2, 200, 8), dtype=np.float32)
    base = target.copy()
    base[:, 105:, :] = 1.0
    tail = target.copy()
    tail[:, 105:140, :] = 0.0
    tail[:, 140:, :] = -1.0

    results = search_task1_combinations(
        predictions={"fno_ensemble": base, "tail_expert": tail},
        target=target,
        config=ComboSearchConfig(
            include_global=False,
            include_cluster=False,
            include_temporal=False,
            include_piecewise=True,
            piecewise_split_candidates=(140,),
            temporal_cut_min=105,
            temporal_cut_max=105,
            temporal_weight_step=0.5,
            top_k=3,
        ),
    )

    best = results[0]
    assert best.kind == "piecewise_temporal_blend"
    assert best.config["split"] == 140
    assert best.config["early_tail_weight"] == 1.0
    assert best.config["late_tail_weight"] == 0.5
    assert best.metrics["mse"] == 0.0


def test_cross_piecewise_temporal_blend_can_use_two_different_tail_experts():
    target = np.zeros((2, 200, 8), dtype=np.float32)
    base = target.copy()
    base[:, 95:, :] = 1.0
    early_tail = base.copy()
    early_tail[:, 95:140, :] = 0.0
    late_tail = base.copy()
    late_tail[:, 140:, :] = 0.0

    results = search_task1_combinations(
        predictions={"fno_ensemble": base, "early_expert": early_tail, "late_expert": late_tail},
        target=target,
        config=ComboSearchConfig(
            include_global=False,
            include_cluster=False,
            include_temporal=False,
            include_piecewise=False,
            piecewise_split_candidates=(140,),
            temporal_cut_min=95,
            temporal_cut_max=95,
            temporal_weight_step=1.0,
            top_k=3,
        ),
    )

    best = results[0]
    assert best.kind == "cross_piecewise_temporal_blend"
    assert best.config["early_tail_name"] == "early_expert"
    assert best.config["late_tail_name"] == "late_expert"
    assert best.config["early_tail_weight"] == 1.0
    assert best.config["late_tail_weight"] == 1.0
    assert best.metrics["mse"] == 0.0


def test_auto_explorer_writes_ranked_summary_journal_and_training_queue(tmp_path):
    study_dir = tmp_path / "study"
    target = np.zeros((2, 200, 8), dtype=np.float32)
    base = target.copy()
    base[:, 113:, :] = 1.0
    tail = target.copy()
    tail[:, 105:113, :] = 1.0
    _write_prediction(study_dir / "fno_ensemble" / "task1_val_pred.hdf5", base)
    _write_prediction(study_dir / "deeponet_lite" / "task1_val_pred.hdf5", tail)
    target_path = tmp_path / "target.hdf5"
    _write_prediction(target_path, target, key="tensor")

    summary_path = run_task1_auto_explorer(
        study_dir=study_dir,
        output_dir=tmp_path / "auto",
        target_path=target_path,
        config=ComboSearchConfig(
            include_global=False,
            include_cluster=False,
            include_piecewise=False,
            temporal_cut_min=105,
            temporal_cut_max=120,
            temporal_cut_stride=1,
            temporal_weight_step=1.0,
            top_k=5,
        ),
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["best"]["config"]["cut"] == 113
    assert summary["best"]["metrics"]["mse"] == 0.0
    assert Path(summary["best"]["prediction_path"]).is_file()
    assert (tmp_path / "auto" / "journal.json").is_file()
    assert {item["action_type"] for item in summary["training_queue"]} == {"baseline_train"}
    assert {"deeponet_lite", "residual_refiner", "pino_fno"}.issubset(
        {item["params"]["model"] for item in summary["training_queue"]}
    )


def test_training_queue_contains_policy_metadata_and_does_not_retrain_fno(tmp_path):
    study_dir = tmp_path / "study"
    target = np.zeros((2, 200, 8), dtype=np.float32)
    base = target.copy()
    tail = target.copy()
    tail[:, 140:, :] = 1.0
    _write_prediction(study_dir / "fno_ensemble" / "task1_val_pred.hdf5", base)
    _write_prediction(study_dir / "deeponet_lite" / "task1_val_pred.hdf5", tail)
    target_path = tmp_path / "target.hdf5"
    _write_prediction(target_path, target, key="tensor")

    summary_path = run_task1_auto_explorer(
        study_dir=study_dir,
        output_dir=tmp_path / "auto",
        target_path=target_path,
        config=ComboSearchConfig(include_global=False, include_cluster=False, top_k=2),
    )

    queue = json.loads(summary_path.read_text(encoding="utf-8"))["training_queue"]
    models = {item["params"]["model"] for item in queue}
    tags = {tag for item in queue for tag in item["params"]["policy"]["tags"]}
    statuses = {item["params"]["policy"]["status"] for item in queue}

    assert "fno_ensemble" not in models
    assert {"stronger-backbone", "long-horizon", "physics-loss", "spectral-loss", "multi-seed"}.issubset(tags)
    assert {"requires_base_train_predictions", "requires_trainer_knob", "requires_dependency"}.issubset(statuses)
    assert all(item["params"]["command"] for item in queue if item["params"]["policy"]["status"] == "ready")


def test_auto_explorer_residual_refiner_ready_plan_uses_fno_base_prediction(tmp_path):
    study_dir = tmp_path / "study"
    target = np.zeros((2, 200, 8), dtype=np.float32)
    fno_path = study_dir / "fno_ensemble" / "task1_val_pred.hdf5"
    _write_prediction(fno_path, target)
    _write_prediction(study_dir / "deeponet_lite" / "task1_val_pred.hdf5", target + 1.0)
    target_path = tmp_path / "target.hdf5"
    _write_prediction(target_path, target, key="tensor")

    summary_path = run_task1_auto_explorer(
        study_dir=study_dir,
        output_dir=tmp_path / "auto",
        target_path=target_path,
        config=ComboSearchConfig(include_global=False, include_cluster=False, top_k=1),
    )

    queue = json.loads(summary_path.read_text(encoding="utf-8"))["training_queue"]
    refiner = next(item for item in queue if item["params"]["model"] == "residual_refiner")

    assert refiner["params"]["base_validation_prediction_path"] == str(fno_path)
    assert "--base-validation-prediction-path" in refiner["params"]["command"]
    assert str(fno_path) in refiner["params"]["command"]


def test_auto_explorer_can_execute_ready_refiner_and_research_combinations(tmp_path):
    study_dir = tmp_path / "study"
    target = np.zeros((2, 200, 8), dtype=np.float32)
    fno_path = study_dir / "fno_ensemble" / "task1_val_pred.hdf5"
    base_train = tmp_path / "base_train.hdf5"
    _write_prediction(fno_path, target + 0.5)
    _write_prediction(study_dir / "deeponet_lite" / "task1_val_pred.hdf5", target + 1.0)
    _write_prediction(base_train, target + 0.5, key="tensor")
    target_path = tmp_path / "target.hdf5"
    _write_prediction(target_path, target, key="tensor")
    executed = []

    def fake_runner(command, *, cwd, timeout):
        executed.append(command)
        prediction_path = tmp_path / "runs" / "residual-refiner-long-horizon" / "residual_refiner" / "task1_val_pred.hdf5"
        _write_prediction(prediction_path, target)
        return {"returncode": 0, "stdout_tail": "ok", "stderr_tail": "", "command": command, "timeout": timeout}

    summary_path = run_task1_auto_explorer(
        study_dir=study_dir,
        output_dir=tmp_path / "auto",
        target_path=target_path,
        config=ComboSearchConfig(include_global=False, include_cluster=False, top_k=3),
        base_train_hdf5=[base_train],
        execute_ready=True,
        project_root=tmp_path,
        command_runner=fake_runner,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    refiner = next(item for item in summary["training_queue"] if item["params"]["model"] == "residual_refiner")

    assert refiner["params"]["policy"]["status"] == "ready"
    assert "--base-train-hdf5" in refiner["params"]["command"]
    assert executed
    assert summary["post_training_best"]["kind"] == "single_model"
    assert summary["post_training_best"]["name"] == "residual_refiner"
