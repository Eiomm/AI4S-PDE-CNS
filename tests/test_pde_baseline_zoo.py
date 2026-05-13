import json

import h5py
import numpy as np

from agent.pde_baselines import (
    BaselineSpec,
    FakeTask1BaselineWorkflow,
    build_default_task1_baseline_registry,
)
from agent.pde_executor import ControlledExperimentExecutor
from agent.pde_gating import (
    extract_task1_initial_features,
    fit_cluster_em_ensemble,
    fit_global_convex_ensemble,
    fit_temporal_tail_blend,
    softmax_rows,
)
from agent.pde_journal import CandidatePlan
from agent.pde_tasks import TaskSpec
from agent.task1_trajectory_data import (
    Task1TrajectoryConfig,
    read_task1_trajectory_sample,
    task1_trajectory_length,
)


def _write_hdf5(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("tensor", data=data.astype(np.float32))
        h5.create_dataset("x-coordinate", data=np.linspace(0.0, 1.0, data.shape[2], endpoint=False, dtype=np.float32))
        h5.create_dataset("t-coordinate", data=np.linspace(0.0, 1.0, data.shape[1], dtype=np.float32))


def _fake_task1_spec(root):
    data_dir = root / "data"
    init = np.zeros((2, 10, 256), dtype=np.float32)
    target = np.concatenate([init, np.ones((2, 190, 256), dtype=np.float32)], axis=1)
    _write_hdf5(data_dir / "task1_test.hdf5", init)
    _write_hdf5(data_dir / "task1_val.hdf5", target)
    return TaskSpec(
        task_id="task1",
        test_input_path=data_dir / "task1_test.hdf5",
        validation_target_path=data_dir / "task1_val.hdf5",
        initial_condition_path=data_dir / "task1_test.hdf5",
        output_shape=(None, 200, 256),
        prediction_name="task1_pred.hdf5",
        time_budget_seconds=3600,
    )


def test_default_task1_baseline_registry_contains_planned_models():
    registry = build_default_task1_baseline_registry()

    assert registry.names() == [
        "fno_ensemble",
        "tfno",
        "unet1d",
        "deeponet_lite",
        "pino_fno",
        "residual_refiner",
    ]
    assert registry.get("fno_ensemble").family == "FNO"
    assert registry.get("residual_refiner").trainable is True


def test_fake_baseline_workflow_writes_standard_validation_artifacts(tmp_path):
    spec = _fake_task1_spec(tmp_path)
    baseline = FakeTask1BaselineWorkflow(
        BaselineSpec(name="fake", family="test", trainable=False),
        spec=spec,
        run_root=tmp_path / "runs",
        fill_value=1.0,
    )

    result = baseline.run_validation({"note": "unit"}, run_name="fake-val")

    run_dir = tmp_path / "runs" / "fake-val"
    assert result.success is True
    assert result.prediction_path == run_dir / "task1_val_pred.hdf5"
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "run_result.json").is_file()
    assert (run_dir / "experiment_memory.json").is_file()
    manifest = json.loads((run_dir / "baseline_manifest.json").read_text(encoding="utf-8"))
    assert manifest["baseline"]["name"] == "fake"
    with h5py.File(result.prediction_path, "r") as h5:
        pred = h5["prediction"][:]
    assert pred.shape == (2, 200, 256)
    assert np.allclose(pred[:, :10, :], 0.0)


def test_task1_trajectory_reader_downsamples_full_trajectory(tmp_path):
    data = np.arange(3 * 201 * 512, dtype=np.float32).reshape(3, 201, 512)
    path = tmp_path / "burgers.hdf5"
    _write_hdf5(path, data)
    config = Task1TrajectoryConfig(hdf5_paths=[path], max_samples_per_file=2, spatial_size=256, output_steps=200)

    sample = read_task1_trajectory_sample(config, 1)

    assert task1_trajectory_length(config) == 2
    assert sample.initial.shape == (10, 256)
    assert sample.target.shape == (200, 256)
    assert np.allclose(sample.initial, data[1, :10, ::2])
    assert np.allclose(sample.target, data[1, :200, ::2])
    assert sample.nu is None


def test_global_convex_ensemble_selects_better_prediction():
    target = np.zeros((2, 200, 4), dtype=np.float32)
    good = target.copy()
    bad = np.ones_like(target)

    result = fit_global_convex_ensemble({"bad": bad, "good": good}, target, grid_step=0.5)

    assert result.weights["good"] == 1.0
    assert result.weights["bad"] == 0.0
    assert result.metrics["mse"] == 0.0
    assert np.allclose(result.prediction, target)


def test_global_convex_ensemble_computes_full_metrics_once(monkeypatch):
    import agent.pde_gating as pde_gating

    calls = {"count": 0}

    def fake_metrics(prediction, target):
        calls["count"] += 1
        mse = float(np.mean((prediction - target) ** 2))
        return {"mse": mse, "forecast_mse": mse, "competition_score_proxy": 100.0 / (1.0 + mse)}

    monkeypatch.setattr(pde_gating, "compute_task1_metrics", fake_metrics)
    target = np.zeros((2, 200, 256), dtype=np.float32)
    good = target.copy()
    bad = np.ones_like(target)

    result = fit_global_convex_ensemble({"bad": bad, "good": good}, target, grid_step=0.5)

    assert result.weights["good"] == 1.0
    assert calls["count"] == 1


def test_cluster_em_ensemble_uses_different_experts_by_feature_cluster():
    initial = np.zeros((4, 10, 8), dtype=np.float32)
    initial[2:, :, :] = 2.0
    target = np.zeros((4, 200, 8), dtype=np.float32)
    target[2:, :, :] = 5.0
    expert_low = np.zeros_like(target)
    expert_high = np.full_like(target, 5.0)

    result = fit_cluster_em_ensemble(
        initial,
        {"low": expert_low, "high": expert_high},
        target,
        n_clusters=2,
    )

    assert set(result.cluster_weights) == {0, 1}
    assert np.allclose(result.prediction, target)
    assert any(weights["low"] == 1.0 for weights in result.cluster_weights.values())
    assert any(weights["high"] == 1.0 for weights in result.cluster_weights.values())


def test_temporal_tail_blend_selects_late_horizon_expert():
    target = np.zeros((2, 200, 256), dtype=np.float32)
    base = target.copy()
    base[:, 120:, :] = 1.0
    tail = target.copy()
    tail[:, :120, :] = 1.0

    result = fit_temporal_tail_blend(
        {"base": base, "tail": tail},
        target,
        base_name="base",
        tail_name="tail",
        cut_candidates=[120],
        tail_weights=[1.0],
    )

    assert result.config["cut"] == 120
    assert result.config["tail_weight"] == 1.0
    assert result.metrics["mse"] == 0.0
    assert np.allclose(result.prediction, target)


def test_initial_features_and_softmax_are_well_formed():
    initial = np.zeros((3, 10, 16), dtype=np.float32)
    initial[:, :, 8:] = 1.0

    features = extract_task1_initial_features(initial)
    weights = softmax_rows(np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32))

    assert features.shape == (3, 6)
    assert np.isfinite(features).all()
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert weights[0, 1] > weights[0, 0]


def test_executor_accepts_baseline_action_types(tmp_path):
    executor = ControlledExperimentExecutor(project_root=tmp_path)

    for action_type in ("baseline_train", "baseline_validate", "baseline_ensemble", "baseline_refine"):
        plan = CandidatePlan(
            intent=f"run {action_type}",
            hypothesis="baseline zoo action is routed through a controlled command",
            action_type=action_type,
            params={"command": ["python", "-c", "print('ok')"], "timeout_seconds": 30},
        )
        execution = executor.execute(type("Node", (), {"plan": plan})())
        assert execution.success is True
        assert execution.artifacts["returncode"] == 0
