import csv
import json

import h5py
import numpy as np
import pytest

from agent.submission import validate_submission
from agent.submission_workspace import SubmissionWorkspaceError, build_submission_workspace


def _write_prediction(path, samples=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("tensor", data=np.zeros((samples, 200, 256), dtype=np.float32))


def _write_task_run(root, task, *, code_files):
    root.mkdir(parents=True, exist_ok=True)
    _write_prediction(root / f"{task}_pred.hdf5")
    with (root / f"{task}_time.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["train_time", "inference_time"])
        writer.writeheader()
        writer.writerow({"train_time": "1.0", "inference_time": "0.5"})
    (root / f"{task}_logs.log").write_text(
        json.dumps(
            {
                "timestamp": "2026-05-15T00:00:00+08:00",
                "elapsed_seconds": 1.0,
                "response": {"content": f"{task} trace"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    code_dir = root / "code"
    code_dir.mkdir()
    for relative_path, content in code_files.items():
        target = code_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def test_submission_workspace_merges_task_runs_with_shared_code(tmp_path):
    task1 = tmp_path / "runs" / "task1-study"
    task2 = tmp_path / "runs" / "task2-study"
    _write_task_run(task1, "task1", code_files={"task1_infer.py": "print('task1')\n", "shared.py": "VALUE = 1\n"})
    _write_task_run(task2, "task2", code_files={"task2_infer.py": "print('task2')\n", "shared.py": "VALUE = 1\n"})
    (task1 / "code" / "code_manifest.json").write_text("{}\n", encoding="utf-8")
    methodology = tmp_path / "docs" / "methodology.pdf"
    methodology.parent.mkdir()
    methodology.write_bytes(b"%PDF-1.4\n% methodology\n")

    output = build_submission_workspace(
        output_dir=tmp_path / "runs" / "combined",
        task_runs={"task1": task1, "task2": task2},
        methodology_path=methodology,
    )

    assert (output / "task1_pred.hdf5").is_file()
    assert (output / "task2_pred.hdf5").is_file()
    assert (output / "code" / "task1_infer.py").read_text(encoding="utf-8") == "print('task1')\n"
    assert (output / "code" / "task2_infer.py").read_text(encoding="utf-8") == "print('task2')\n"
    assert (output / "code" / "shared.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (output / "code" / "code_manifest.json").exists()
    assert validate_submission(output).tasks == ["task1", "task2"]


def test_submission_workspace_rejects_conflicting_shared_code(tmp_path):
    task1 = tmp_path / "runs" / "task1-study"
    task2 = tmp_path / "runs" / "task2-study"
    _write_task_run(task1, "task1", code_files={"model.py": "VERSION = 'task1'\n"})
    _write_task_run(task2, "task2", code_files={"model.py": "VERSION = 'task2'\n"})
    methodology = tmp_path / "docs" / "methodology.pdf"
    methodology.parent.mkdir()
    methodology.write_bytes(b"%PDF-1.4\n% methodology\n")

    with pytest.raises(SubmissionWorkspaceError, match="Shared code collision"):
        build_submission_workspace(
            output_dir=tmp_path / "runs" / "combined",
            task_runs={"task1": task1, "task2": task2},
            methodology_path=methodology,
        )
