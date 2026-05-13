import math

import numpy as np

from agent.pde_metrics import compute_task1_metrics


def test_task1_metrics_exclude_initial_frames_for_competition_proxy():
    target = np.ones((1, 200, 256), dtype=np.float32)
    prediction = target.copy()
    prediction[:, :10, :] = 99.0

    metrics = compute_task1_metrics(prediction, target)

    assert metrics["initial_mse"] > 0.0
    assert metrics["forecast_mse"] == 0.0
    assert metrics["segment1_rel_mse"] == 0.0
    assert metrics["segment2_rel_mse"] == 0.0
    assert metrics["segment3_rmse"] == 0.0
    assert metrics["competition_score_proxy"] == 100.0


def test_task1_metrics_use_segmented_relative_mse_and_lorentzian_score():
    target = np.ones((1, 200, 256), dtype=np.float32)
    prediction = target.copy()
    prediction[:, 10:57, :] = 2.0

    metrics = compute_task1_metrics(prediction, target)

    assert metrics["segment1_rel_mse"] == 1.0
    assert metrics["segment2_rel_mse"] == 0.0
    assert metrics["segment3_rmse"] == 0.0
    expected = 0.25 * (100.0 * math.exp(-20.0)) + 0.25 * 100.0 + 0.5 * 100.0
    assert math.isclose(metrics["competition_score_proxy"], expected)


def test_task1_relative_mse_caps_each_sample_at_five():
    target = np.ones((1, 200, 256), dtype=np.float32)
    prediction = target.copy()
    prediction[:, 10:57, :] = 100.0

    metrics = compute_task1_metrics(prediction, target)

    assert metrics["segment1_rel_mse"] == 5.0
