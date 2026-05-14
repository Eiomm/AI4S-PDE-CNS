import json
import zipfile

import h5py
import numpy as np

from agent.pde_memory import ExperimentMemory
from agent.pde_results import RunResult, write_run_result_json
from agent.pde_search import Candidate, WeightedEnsembleSearch
from agent.pde_tasks import DEFAULT_TASK1_FNO_WEIGHTS, TaskSpec, task1_spec
from agent.pde_workflow import Task1FNOWorkflow
from agent.run_task1_weight_search import parse_checkpoint_overrides


def _write_hdf5(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("tensor", data=data.astype(np.float32))


def _minimal_code_and_methodology(root):
    code_dir = root / "code"
    code_dir.mkdir()
    (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
    methodology_path = root / "methodology.pdf"
    methodology_path.write_bytes(b"%PDF-1.4\n% placeholder\n")
    return code_dir, methodology_path


def _fake_spec(root, *, val_offset=0.0):
    data_dir = root / "data"
    init = np.zeros((2, 10, 256), dtype=np.float32)
    future = np.ones((2, 190, 256), dtype=np.float32) + val_offset
    val = np.concatenate([init, future], axis=1)
    _write_hdf5(data_dir / "task1_test.hdf5", init)
    _write_hdf5(data_dir / "task1_val.hdf5", val)
    return TaskSpec(
        task_id="task1",
        test_input_path=data_dir / "task1_test.hdf5",
        validation_target_path=data_dir / "task1_val.hdf5",
        initial_condition_path=data_dir / "task1_test.hdf5",
        output_shape=(None, 200, 256),
        prediction_name="task1_pred.hdf5",
        time_budget_seconds=3600,
    )


def test_task1_spec_defaults_to_competition_paths():
    spec = task1_spec(project_root="D:/Study/AI4S-PDE-CNS")

    assert spec.task_id == "task1"
    assert spec.output_shape == (None, 200, 256)
    assert spec.test_input_path.as_posix().endswith("data/Task1/task1_test.hdf5")
    assert spec.validation_target_path.as_posix().endswith("data/Task1/task1_val.hdf5")
    assert DEFAULT_TASK1_FNO_WEIGHTS == {"nu0.001": 0.12, "unet_pf20_nu0.001": 0.88}


def test_run_result_writes_json_round_trip(tmp_path):
    result = RunResult(
        task_id="task1",
        run_dir=tmp_path,
        metrics={"mse": 0.25, "forecast_mse": 0.3},
        prediction_path=tmp_path / "task1_pred.hdf5",
        zip_path=tmp_path / "pred.zip",
        train_time=1.0,
        inference_time=2.0,
        success=True,
        error=None,
    )

    path = write_run_result_json(tmp_path, result)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["success"] is True
    assert loaded["metrics"]["mse"] == 0.25
    assert loaded["zip_path"].endswith("pred.zip")


def test_experiment_memory_appends_and_reads_records(tmp_path):
    memory = ExperimentMemory(tmp_path / "experiment_memory.json")
    memory.append(
        {
            "model": "weighted_fno",
            "weights": {"nu0.001": 1.0},
            "metrics": {"mse": 0.2},
            "conclusion": "baseline",
        }
    )
    memory.append(
        {
            "model": "weighted_fno",
            "weights": {"unet_pf20_nu0.001": 1.0},
            "metrics": {"mse": 0.1},
            "conclusion": "better",
        }
    )

    records = memory.read()

    assert [record["conclusion"] for record in records] == ["baseline", "better"]
    assert memory.best(metric="mse")["conclusion"] == "better"


def test_task1_workflow_runs_validation_and_submission(tmp_path):
    spec = _fake_spec(tmp_path)
    code_dir, methodology_path = _minimal_code_and_methodology(tmp_path)

    def provider(input_path, weights, output_steps):
        with h5py.File(input_path, "r") as h5:
            source = h5["tensor"][:]
        if source.shape[1] == output_steps:
            return source.copy()
        pred = np.zeros((source.shape[0], output_steps, source.shape[2]), dtype=np.float32)
        pred[:, : source.shape[1], :] = source
        pred[:, source.shape[1] :, :] = source[:, -1:, :]
        return pred

    workflow = Task1FNOWorkflow(
        spec=spec,
        run_root=tmp_path / "runs",
        code_dir=code_dir,
        methodology_path=methodology_path,
        prediction_provider=provider,
    )

    val_result = workflow.run_validation({"nu0.001": 1.0}, run_name="val")
    submit_result = workflow.run_test_submission({"nu0.001": 1.0}, run_name="submission", train_time=12.5)

    assert val_result.success is True
    assert val_result.metrics["mse"] == 0.0
    assert (tmp_path / "runs" / "val" / "metrics.json").is_file()
    assert (tmp_path / "runs" / "val" / "experiment_memory.json").is_file()
    assert submit_result.success is True
    assert submit_result.train_time == 12.5
    assert submit_result.zip_path == tmp_path / "runs" / "submission" / "pred.zip"
    assert "12.500000" in (tmp_path / "runs" / "submission" / "task1_time.csv").read_text(encoding="utf-8")
    with h5py.File(submit_result.prediction_path, "r") as h5:
        assert h5["tensor"].shape == (2, 200, 256)
        assert np.allclose(h5["tensor"][:, :10, :], 0.0)
    with zipfile.ZipFile(submit_result.zip_path) as zf:
        names = set(zf.namelist())
    assert {
        "task1_pred.hdf5",
        "task1_time.csv",
        "task1_logs.log",
        "submission.json",
        "methodology.pdf",
        "code/train.py",
    } <= names


def test_task1_workflow_accepts_checkpoint_path_overrides(tmp_path):
    spec = _fake_spec(tmp_path)
    code_dir, methodology_path = _minimal_code_and_methodology(tmp_path)
    custom_checkpoint = tmp_path / "runs" / "finetune" / "best.pt"

    workflow = Task1FNOWorkflow(
        spec=spec,
        run_root=tmp_path / "runs",
        code_dir=code_dir,
        methodology_path=methodology_path,
        checkpoint_paths={"nu0.001": custom_checkpoint},
        project_root=tmp_path,
    )

    command = workflow._fno_ensemble_command(spec.test_input_path, {"nu0.001": 1.0}, tmp_path / "pred.hdf5")

    assert str(custom_checkpoint) in " ".join(command)


def test_task1_workflow_defaults_to_official_task1_checkpoint_whitelist(tmp_path):
    spec = _fake_spec(tmp_path)
    code_dir, methodology_path = _minimal_code_and_methodology(tmp_path)

    workflow = Task1FNOWorkflow(
        spec=spec,
        run_root=tmp_path / "runs",
        code_dir=code_dir,
        methodology_path=methodology_path,
        project_root=tmp_path,
    )

    command = workflow._fno_ensemble_command(spec.test_input_path, DEFAULT_TASK1_FNO_WEIGHTS, tmp_path / "pred.hdf5")

    assert list(workflow.checkpoint_paths) == ["nu0.001", "unet_pf20_nu0.001"]
    assert "1D_Burgers_Sols_Nu0.001_FNO.pt" in " ".join(command)
    assert "1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt" in " ".join(command)
    assert "0.12" in command
    assert "0.88" in command
    assert "Nu0.01_FNO" not in " ".join(command)
    assert "Nu0.1_FNO" not in " ".join(command)
    assert "Nu1.0_FNO" not in " ".join(command)


def test_task1_workflow_can_combine_official_fno_and_unet_predictions(tmp_path):
    spec = _fake_spec(tmp_path)
    code_dir, methodology_path = _minimal_code_and_methodology(tmp_path)
    workflow = Task1FNOWorkflow(
        spec=spec,
        run_root=tmp_path / "runs",
        code_dir=code_dir,
        methodology_path=methodology_path,
        project_root=tmp_path,
    )

    command = workflow._fno_ensemble_command(
        spec.test_input_path,
        {"nu0.001": 0.7, "unet_pf20_nu0.001": 0.3},
        tmp_path / "pred.hdf5",
    )
    joined = " ".join(command)

    assert "official_checkpoint_ensemble.py" in joined
    assert "fno=" in joined
    assert "unet_pf20=" in joined
    assert "1D_Burgers_Sols_Nu0.001_FNO.pt" in joined
    assert "1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt" in joined
    assert "0.7" in command
    assert "0.3" in command


def test_parse_checkpoint_overrides_maps_key_value_pairs(tmp_path):
    path = tmp_path / "best.pt"

    overrides = parse_checkpoint_overrides([f"nu0.001={path}"])

    assert overrides == {"nu0.001": path}


def test_parse_checkpoint_overrides_rejects_legacy_non_task1_official_nu_key(tmp_path):
    path = tmp_path / "best.pt"

    try:
        parse_checkpoint_overrides([f"nu0.1={path}"])
    except ValueError as exc:
        assert "unknown checkpoint key 'nu0.1'" in str(exc)
        assert "nu0.001" in str(exc)
    else:
        raise AssertionError("nu0.1 checkpoint override must be rejected in compliant mode")


def test_weighted_ensemble_search_selects_lowest_mse_and_packs_best(tmp_path):
    spec = _fake_spec(tmp_path)
    code_dir, methodology_path = _minimal_code_and_methodology(tmp_path)

    def provider(input_path, weights, output_steps):
        with h5py.File(input_path, "r") as h5:
            source = h5["tensor"][:]
        if source.shape[1] == output_steps:
            target = source.copy()
        else:
            target = np.zeros((source.shape[0], output_steps, source.shape[2]), dtype=np.float32)
            target[:, : source.shape[1], :] = source
        penalty = float(weights.get("penalty", 0.0))
        pred = target + penalty
        pred[:, :10, :] = target[:, :10, :]
        return pred

    workflow = Task1FNOWorkflow(
        spec=spec,
        run_root=tmp_path / "runs",
        code_dir=code_dir,
        methodology_path=methodology_path,
        prediction_provider=provider,
    )
    search = WeightedEnsembleSearch(
        workflow=workflow,
        candidates=[
            Candidate(name="worse", weights={"penalty": 0.2}),
            Candidate(name="better", weights={"penalty": 0.05}),
        ],
        search_name="grid",
    )

    result = search.run()

    assert result.best_candidate.name == "better"
    assert result.best_validation_result.metrics["mse"] < 0.01
    assert result.best_submission_result.zip_path.name == "pred.zip"
    assert result.best_submission_result.zip_path.is_file()


def test_weighted_ensemble_search_can_select_highest_score_metric(tmp_path):
    spec = _fake_spec(tmp_path)
    code_dir, methodology_path = _minimal_code_and_methodology(tmp_path)

    def provider(input_path, weights, output_steps):
        with h5py.File(input_path, "r") as h5:
            source = h5["tensor"][:]
        if source.shape[1] == output_steps:
            target = source.copy()
        else:
            target = np.zeros((source.shape[0], output_steps, source.shape[2]), dtype=np.float32)
            target[:, : source.shape[1], :] = source
        pred = target + float(weights.get("penalty", 0.0))
        pred[:, :10, :] = target[:, :10, :]
        return pred

    workflow = Task1FNOWorkflow(
        spec=spec,
        run_root=tmp_path / "runs",
        code_dir=code_dir,
        methodology_path=methodology_path,
        prediction_provider=provider,
    )
    search = WeightedEnsembleSearch(
        workflow=workflow,
        candidates=[
            Candidate(name="lower-score", weights={"penalty": 0.2}),
            Candidate(name="higher-score", weights={"penalty": 0.05}),
        ],
        search_name="score-grid",
        metric="competition_score_proxy",
        maximize=True,
    )

    result = search.run(make_submission=False)

    assert result.best_candidate.name == "higher-score"
