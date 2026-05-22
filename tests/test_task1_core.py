from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai4sv2_task1.metrics import compute_task1_metrics
from ai4sv2_task1.predict import combine_predictions


def test_combine_predictions_normalizes_weights():
    first = np.zeros((1, 2, 3), dtype=np.float32)
    second = np.full((1, 2, 3), 10.0, dtype=np.float32)
    combined = combine_predictions([first, second], [1.0, 3.0])
    np.testing.assert_allclose(combined, np.full((1, 2, 3), 7.5, dtype=np.float32))


def test_task1_metrics_basic_splits():
    target = np.zeros((2, 200, 256), dtype=np.float32)
    prediction = np.zeros_like(target)
    prediction[:, :10, :] = 1.0
    prediction[:, 10:105, :] = 2.0
    prediction[:, 105:, :] = 3.0
    metrics = compute_task1_metrics(prediction, target)
    assert metrics["initial_mse"] == 1.0
    assert metrics["forecast_mse"] == 6.5
    assert metrics["long_horizon_mse"] == 9.0
    assert "segment3_frechet_distance_proxy" in metrics
    assert "segment3_score_proxy" in metrics


def test_task1_metrics_identical_long_horizon_gets_frechet_credit():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(4, 200, 256)).astype(np.float32)
    prediction = target.copy()
    metrics = compute_task1_metrics(prediction, target)
    assert metrics["segment3_frechet_distance_proxy"] < 1.0e-4
    assert metrics["segment3_frechet_score_proxy"] > 49.0
    assert metrics["competition_score_proxy"] > 99.0
