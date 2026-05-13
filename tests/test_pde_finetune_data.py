import h5py
import numpy as np

from agent.pde_finetune import (
    BURGERS_NU_FILES,
    build_finetune_command,
    is_better_metric,
    discover_pdebench_burgers_files,
    pdebench_burgers_filename,
)
from agent.pde_finetune_data import HDF5WindowDatasetConfig, index_to_sample_and_target, read_training_window, spatial_indices
from agent.pde_finetune_data import index_to_sample_and_rollout_start, read_rollout_window, rollout_dataset_length


def _write_fake_burgers(path, *, samples=2, time_steps=6, spatial=512):
    values = np.arange(samples * time_steps * spatial, dtype=np.float32).reshape(samples, time_steps, spatial)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("tensor", data=values)
        h5.create_dataset("x-coordinate", data=np.linspace(0.0, 1.0, spatial, endpoint=False, dtype=np.float32))
        h5.create_dataset("t-coordinate", data=np.linspace(0.0, 0.5, time_steps, dtype=np.float32))
    return values


def test_pdebench_burgers_filename_maps_supported_nu_values():
    assert pdebench_burgers_filename("0.01") == "1D_Burgers_Sols_Nu0.01.hdf5"
    assert pdebench_burgers_filename(0.1) == "1D_Burgers_Sols_Nu0.1.hdf5"
    assert set(BURGERS_NU_FILES) >= {"0.001", "0.01", "0.1", "1.0"}


def test_discover_pdebench_burgers_files_returns_existing_files(tmp_path):
    path = tmp_path / "1D_Burgers_Sols_Nu0.1.hdf5"
    path.write_bytes(b"placeholder")

    files = discover_pdebench_burgers_files(tmp_path)

    assert files == {"0.1": path}


def test_spatial_indices_downsamples_evenly():
    indices = spatial_indices(source_size=512, target_size=256)

    assert indices.shape == (256,)
    assert indices[0] == 0
    assert indices[1] == 2
    assert indices[-1] == 510


def test_read_training_window_returns_initial_frames_and_next_target(tmp_path):
    data_path = tmp_path / "burgers.hdf5"
    values = _write_fake_burgers(data_path)
    config = HDF5WindowDatasetConfig(
        hdf5_path=data_path,
        initial_step=3,
        spatial_size=256,
        max_samples=None,
        max_time_steps=None,
    )

    window = read_training_window(config, sample_index=1, target_time_index=4)

    expected_indices = spatial_indices(source_size=512, target_size=256)
    assert window.input_frames.shape == (3, 256)
    assert window.target_frame.shape == (256,)
    assert np.allclose(window.input_frames, values[1, 1:4][:, expected_indices])
    assert np.allclose(window.target_frame, values[1, 4][expected_indices])
    assert window.x_coords.shape == (256,)
    assert window.t_coords.shape == (6,)


def test_read_rollout_window_returns_initial_frames_and_future_targets(tmp_path):
    data_path = tmp_path / "burgers.hdf5"
    values = _write_fake_burgers(data_path, samples=2, time_steps=8, spatial=512)
    config = HDF5WindowDatasetConfig(
        hdf5_path=data_path,
        initial_step=3,
        spatial_size=256,
        max_samples=None,
        max_time_steps=None,
    )

    window = read_rollout_window(config, sample_index=1, start_time_index=3, rollout_steps=4)

    expected_indices = spatial_indices(source_size=512, target_size=256)
    assert window.input_frames.shape == (3, 256)
    assert window.target_frames.shape == (4, 256)
    assert np.allclose(window.input_frames, values[1, 0:3][:, expected_indices])
    assert np.allclose(window.target_frames, values[1, 3:7][:, expected_indices])
    assert window.start_time_index == 3


def test_rollout_dataset_length_accounts_for_future_horizon(tmp_path):
    data_path = tmp_path / "burgers.hdf5"
    _write_fake_burgers(data_path, samples=5, time_steps=8, spatial=512)
    config = HDF5WindowDatasetConfig(
        hdf5_path=data_path,
        initial_step=3,
        spatial_size=256,
        max_samples=2,
        sample_start=1,
        max_time_steps=None,
    )

    assert rollout_dataset_length(config, rollout_steps=4) == 2 * 2
    assert index_to_sample_and_rollout_start(config, 0, rollout_steps=4) == (1, 3)
    assert index_to_sample_and_rollout_start(config, 3, rollout_steps=4) == (2, 4)


def test_dataset_config_supports_sample_start_for_holdout_splits(tmp_path):
    data_path = tmp_path / "burgers.hdf5"
    values = _write_fake_burgers(data_path, samples=5, time_steps=6)
    config = HDF5WindowDatasetConfig(
        hdf5_path=data_path,
        initial_step=3,
        spatial_size=256,
        max_samples=2,
        sample_start=2,
        max_time_steps=None,
    )

    sample_index, target_time_index = index_to_sample_and_target(config, 0)
    window = read_training_window(config, sample_index=sample_index, target_time_index=target_time_index)

    expected_indices = spatial_indices(source_size=512, target_size=256)
    assert sample_index == 2
    assert target_time_index == 3
    assert np.allclose(window.input_frames, values[2, 0:3][:, expected_indices])


def test_build_finetune_command_points_to_run_dir_and_checkpoint(tmp_path):
    command = build_finetune_command(
        train_hdf5=tmp_path / "raw.hdf5",
        base_checkpoint=tmp_path / "base.pt",
        run_dir=tmp_path / "runs" / "finetune",
        val_hdf5=tmp_path / "val.hdf5",
        steps=25,
    )

    assert "code/train_task1_fno_finetune.py" in command
    assert "--steps" in command
    assert "25" in command
    assert str(tmp_path / "runs" / "finetune") in command


def test_is_better_metric_requires_strict_improvement_past_margin():
    assert is_better_metric({"mse": 0.9}, None)
    assert is_better_metric({"mse": 0.89}, {"mse": 0.9}, min_improvement=0.005)
    assert not is_better_metric({"mse": 0.899}, {"mse": 0.9}, min_improvement=0.005)


def test_is_better_metric_can_maximize_score_metrics():
    assert is_better_metric({"competition_score_proxy": 58.0}, None, metric="competition_score_proxy", maximize=True)
    assert is_better_metric(
        {"competition_score_proxy": 58.2},
        {"competition_score_proxy": 58.0},
        metric="competition_score_proxy",
        min_improvement=0.1,
        maximize=True,
    )
    assert not is_better_metric(
        {"competition_score_proxy": 58.05},
        {"competition_score_proxy": 58.0},
        metric="competition_score_proxy",
        min_improvement=0.1,
        maximize=True,
    )
