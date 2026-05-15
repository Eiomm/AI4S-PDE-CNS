import argparse
import json
from pathlib import Path

from agent.run_task1_finetune_agent import (
    FinetuneExperiment,
    _command_for_experiment,
    _summary_row,
    _write_summary,
)


def test_finetune_agent_command_includes_strategy_knobs(tmp_path):
    args = argparse.Namespace(
        train_hdf5="train.hdf5",
        base_checkpoint="base.pt",
        val_hdf5="val.hdf5",
        eval_batch_size=25,
        device="cpu",
    )
    experiment = FinetuneExperiment(
        name="multi",
        hypothesis="test",
        trainable="last-block-head",
        rollout_steps=5,
        steps=123,
        lr=3.0e-6,
        batch_size=4,
    )

    command = _command_for_experiment(args, experiment, tmp_path / "run")

    assert "--rollout-steps" in command
    assert command[command.index("--rollout-steps") + 1] == "5"
    assert "--trainable" in command
    assert command[command.index("--trainable") + 1] == "last-block-head"
    assert "--device" in command
    assert command[command.index("--device") + 1] == "cpu"


def test_finetune_agent_writes_ranked_summary(tmp_path):
    experiments = [
        FinetuneExperiment(name="weak", hypothesis="weak hypothesis"),
        FinetuneExperiment(name="strong", hypothesis="strong hypothesis", rollout_steps=5),
    ]
    results = [
        {
            "best_metrics": {"competition_score_proxy": 1.0, "mse": 0.2, "forecast_mse": 0.3, "long_horizon_mse": 0.4},
            "base_metrics": {"competition_score_proxy": 0.5},
            "history": [{"step": 10, "competition_score_proxy": 1.0}],
            "elapsed_seconds": 2.0,
        },
        {
            "best_metrics": {"competition_score_proxy": 3.0, "mse": 0.1, "forecast_mse": 0.2, "long_horizon_mse": 0.3},
            "base_metrics": {"competition_score_proxy": 0.5},
            "history": [{"step": 20, "competition_score_proxy": 3.0}],
            "elapsed_seconds": 2.0,
        },
    ]
    rows = [
        _summary_row(experiments[0], tmp_path / "weak", results[0]),
        _summary_row(experiments[1], tmp_path / "strong", results[1]),
    ]

    _write_summary(tmp_path / "summary", rows)

    summary = json.loads((tmp_path / "summary" / "summary.json").read_text(encoding="utf-8"))
    assert summary[0]["name"] == "strong"
    assert Path(tmp_path / "summary" / "summary.csv").is_file()
    assert "`strong`" in (tmp_path / "summary" / "summary.md").read_text(encoding="utf-8")
