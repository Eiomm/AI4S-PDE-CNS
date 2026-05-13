import json

import h5py
import numpy as np

from agent.run_task1_combo_search import run_task1_combo_search


def _write_prediction(path, prediction):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("prediction", data=prediction.astype(np.float32))


def test_combo_search_writes_validation_summary_without_packaging(tmp_path):
    study_dir = tmp_path / "runs" / "study"
    target = np.zeros((2, 200, 256), dtype=np.float32)
    fno = target.copy()
    fno[:, 120:, :] = 1.0
    tail = target.copy()
    tail[:, :120, :] = 1.0
    _write_prediction(tmp_path / "data" / "task1_val.hdf5", target)
    _write_prediction(study_dir / "fno_ensemble" / "task1_val_pred.hdf5", fno)
    _write_prediction(study_dir / "deeponet_lite" / "task1_val_pred.hdf5", tail)

    summary_path = run_task1_combo_search(study_dir=study_dir, target_path=tmp_path / "data" / "task1_val.hdf5")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["best"]["name"] == "temporal_tail_blend_deeponet_lite"
    assert payload["best"]["metrics"]["mse"] == 0.0
    assert (study_dir / "temporal_tail_blend_deeponet_lite" / "task1_val_pred.hdf5").is_file()
    assert not list(study_dir.rglob("pred.zip"))
