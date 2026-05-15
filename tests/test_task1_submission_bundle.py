import json

import h5py
import numpy as np

from agent.submission import validate_submission
from agent.task1_submission import create_task1_submission_bundle


def _write_hdf5(path, key, data):
    with h5py.File(path, "w") as h5:
        h5.create_dataset(key, data=data.astype(np.float32))


def test_create_task1_submission_bundle_validates_outputs(tmp_path):
    initial = np.random.default_rng(0).normal(size=(2, 10, 256)).astype(np.float32)
    prediction = np.zeros((2, 200, 256), dtype=np.float32)
    prediction[:, :10, :] = initial
    prediction[:, 10:, :] = initial[:, 9:10, :]

    initial_path = tmp_path / "task1_test.hdf5"
    prediction_path = tmp_path / "candidate.hdf5"
    log_path = tmp_path / "task1_logs.log"
    methodology_path = tmp_path / "methodology.pdf"
    code_dir = tmp_path / "code-src"
    output_dir = tmp_path / "submission-run"
    _write_hdf5(initial_path, "tensor", initial)
    _write_hdf5(prediction_path, "prediction", prediction)
    log_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-12T19:00:00+08:00",
                "elapsed_seconds": 1.25,
                "response": {"content": "generated prediction"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    methodology_path.write_bytes(b"%PDF-1.4\n% methodology\n")
    code_dir.mkdir()
    (code_dir / "infer.py").write_text("print('ok')\n", encoding="utf-8")

    result = create_task1_submission_bundle(
        prediction_path=prediction_path,
        initial_path=initial_path,
        output_dir=output_dir,
        code_dir=code_dir,
        log_path=log_path,
        methodology_path=methodology_path,
        train_time=12.5,
        inference_time=3.25,
    )

    assert result == output_dir
    assert (output_dir / "task1_pred.hdf5").exists()
    with h5py.File(output_dir / "task1_pred.hdf5", "r") as h5:
        assert list(h5.keys()) == ["tensor"]
        np.testing.assert_allclose(h5["tensor"][:], prediction)
    log_lines = (output_dir / "task1_logs.log").read_text(encoding="utf-8").splitlines()
    assert json.loads(log_lines[0]) == json.loads(log_path.read_text(encoding="utf-8"))
    trace_record = json.loads(log_lines[-1])
    assert trace_record["response"]["action"] == "write_code_file"
    assert trace_record["response"]["path"] == "code/infer.py"
    assert trace_record["response"]["content"] == "print('ok')\n"
    assert (output_dir / "methodology.pdf").read_bytes() == methodology_path.read_bytes()
    assert (output_dir / "code" / "infer.py").read_text(encoding="utf-8") == "print('ok')\n"
    assert "12.500000,3.250000" in (output_dir / "task1_time.csv").read_text(encoding="utf-8")
    report = validate_submission(output_dir)
    assert report.valid is True
    assert report.tasks == ["task1"]


def test_create_task1_submission_bundle_can_append_real_llm_provenance_log(tmp_path):
    initial = np.random.default_rng(1).normal(size=(1, 10, 256)).astype(np.float32)
    prediction = np.zeros((1, 200, 256), dtype=np.float32)
    prediction[:, :10, :] = initial
    initial_path = tmp_path / "task1_test.hdf5"
    prediction_path = tmp_path / "candidate.hdf5"
    log_path = tmp_path / "task1_logs.log"
    provenance_log = tmp_path / "planner_logs.log"
    methodology_path = tmp_path / "methodology.pdf"
    code_dir = tmp_path / "code-src"
    output_dir = tmp_path / "submission-run"
    _write_hdf5(initial_path, "tensor", initial)
    _write_hdf5(prediction_path, "prediction", prediction)
    log_path.write_text(
        json.dumps({"timestamp": "2026-05-12T19:00:00+08:00", "elapsed_seconds": 1.0, "response": {"content": "run"}})
        + "\n",
        encoding="utf-8",
    )
    methodology_path.write_bytes(b"%PDF-1.4\n% methodology\n")
    code_dir.mkdir()
    (code_dir / "infer.py").write_text("print('agent code')\n", encoding="utf-8")
    provenance_log.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-12T19:00:02+08:00",
                "elapsed_seconds": 2.0,
                "provider": "hkustgz_gpt",
                "model": "gpt-5.3-chat",
                "response": {
                    "content": json.dumps(
                        {
                            "intent": "improve",
                            "hypothesis": "generate traceable inference code",
                            "action_type": "code_patch",
                            "params": {
                                "files": [{"path": "infer.py", "content": "print('agent code')\n"}],
                                "validation_command": ["python", "-m", "pytest", "-q"],
                            },
                            "expected_effect": "strict provenance passes",
                            "risk": "none",
                        }
                    )
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    create_task1_submission_bundle(
        prediction_path=prediction_path,
        initial_path=initial_path,
        output_dir=output_dir,
        code_dir=code_dir,
        log_path=log_path,
        methodology_path=methodology_path,
        train_time=1.0,
        inference_time=1.0,
        require_llm_code_trace=True,
        provenance_log_paths=[provenance_log],
    )

    report = validate_submission(output_dir)
    assert report.valid is True
    assert json.loads((output_dir / "submission.json").read_text(encoding="utf-8"))["require_llm_code_trace"] is True


def test_strict_task1_bundle_does_not_append_synthetic_code_trace(tmp_path):
    initial = np.random.default_rng(2).normal(size=(1, 10, 256)).astype(np.float32)
    prediction = np.zeros((1, 200, 256), dtype=np.float32)
    prediction[:, :10, :] = initial
    initial_path = tmp_path / "task1_test.hdf5"
    prediction_path = tmp_path / "candidate.hdf5"
    log_path = tmp_path / "task1_logs.log"
    provenance_log = tmp_path / "planner_logs.log"
    methodology_path = tmp_path / "methodology.pdf"
    code_dir = tmp_path / "code-src"
    output_dir = tmp_path / "submission-run"
    _write_hdf5(initial_path, "tensor", initial)
    _write_hdf5(prediction_path, "prediction", prediction)
    log_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-15T00:00:00+08:00",
                "elapsed_seconds": 1.0,
                "provider": "hkustgz_gpt",
                "model": "gpt-5.3-chat",
                "messages": [{"role": "user", "content": "plan task1 run"}],
                "response": {"content": "planned traceable task1 experiment"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    methodology_path.write_bytes(b"%PDF-1.4\n% methodology\n")
    code_dir.mkdir()
    (code_dir / "infer.py").write_text("print('agent code')\n", encoding="utf-8")
    provenance_log.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-15T00:00:02+08:00",
                "elapsed_seconds": 2.0,
                "provider": "hkustgz_gpt",
                "model": "gpt-5.3-chat",
                "messages": [{"role": "user", "content": "generate final code"}],
                "response": {
                    "content": json.dumps(
                        {
                            "intent": "improve",
                            "hypothesis": "generate final traceable inference code",
                            "action_type": "code_patch",
                            "params": {
                                "files": [{"path": "infer.py", "content": "print('agent code')\n"}],
                                "validation_command": ["python", "-m", "pytest", "-q"],
                            },
                            "expected_effect": "code can be traced to an LLM response",
                            "risk": "none",
                        }
                    )
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    create_task1_submission_bundle(
        prediction_path=prediction_path,
        initial_path=initial_path,
        output_dir=output_dir,
        code_dir=code_dir,
        log_path=log_path,
        methodology_path=methodology_path,
        train_time=1.0,
        inference_time=1.0,
        require_llm_code_trace=True,
        provenance_log_paths=[provenance_log],
    )

    log_text = (output_dir / "task1_logs.log").read_text(encoding="utf-8")
    assert '"provider": "codex"' not in log_text
    assert validate_submission(output_dir).valid is True
