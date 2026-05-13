import numpy as np

from agent.run_task1_physics_search import score_physics_candidates


def test_score_physics_candidates_scores_top_mse_candidates_only():
    target = np.zeros((1, 4, 8), dtype=np.float32)
    single_predictions = {
        "a": np.zeros_like(target),
        "b": np.ones_like(target),
    }
    x_coords = np.linspace(0.0, 1.0, 8, endpoint=False)
    t_coords = np.linspace(0.0, 0.1, 4)
    candidates = [
        (0.0, {"a": 1.0, "b": 0.0}),
        (1.0, {"a": 0.0, "b": 1.0}),
    ]

    records = score_physics_candidates(
        single_predictions,
        target,
        x_coords,
        t_coords,
        candidates,
        physics_top_k=1,
        continuation=0,
        time_stride=1,
        spatial_margin=0,
    )

    assert len(records) == 1
    assert records[0]["weights"] == {"a": 1.0, "b": 0.0}
    assert records[0]["mse"] == 0.0
    assert "physics_mse" in records[0]
