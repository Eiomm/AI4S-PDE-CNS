from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FinetuneExperiment:
    name: str
    hypothesis: str
    trainable: str = "all"
    rollout_steps: int = 1
    temporal_stride: int = 1
    steps: int = 250
    lr: float = 1.0e-5
    batch_size: int = 8
    val_every: int = 250
    log_every: int = 100
    weight_decay: float = 0.0
    grad_clip: float = 0.1
    max_samples: int = 2048
    sample_start: int = 0
    val_max_samples: int = 100
    seed: int = 0


def quick_nu001_experiments() -> list[FinetuneExperiment]:
    return [
        FinetuneExperiment(
            name="one_step_lr1e-5_steps5000",
            hypothesis="Long one-step training checks whether more local fitting improves rollout.",
            rollout_steps=1,
            trainable="all",
            steps=5000,
            lr=1.0e-5,
            batch_size=8,
            val_every=250,
        ),
        FinetuneExperiment(
            name="multi_step5_lr3e-6_steps1000",
            hypothesis="Five-step rollout loss better aligns training with autoregressive validation.",
            rollout_steps=5,
            trainable="all",
            steps=1000,
            lr=3.0e-6,
            batch_size=4,
            val_every=250,
        ),
        FinetuneExperiment(
            name="multi_step10_lr1e-6_steps600",
            hypothesis="Ten-step rollout loss with lower LR tests stability-focused fine-tuning.",
            rollout_steps=10,
            trainable="all",
            steps=600,
            lr=1.0e-6,
            batch_size=2,
            val_every=150,
        ),
        FinetuneExperiment(
            name="head_only_lr1e-5_steps1000",
            hypothesis="Head-only tuning limits damage to the pretrained dynamics.",
            rollout_steps=1,
            trainable="head",
            steps=1000,
            lr=1.0e-5,
            batch_size=8,
            val_every=250,
        ),
        FinetuneExperiment(
            name="last_block_head_lr3e-6_steps1000",
            hypothesis="Last spectral block plus head allows adaptation while keeping early operator layers fixed.",
            rollout_steps=1,
            trainable="last-block-head",
            steps=1000,
            lr=3.0e-6,
            batch_size=8,
            val_every=250,
        ),
    ]


def _command_for_experiment(args: argparse.Namespace, experiment: FinetuneExperiment, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "code/train_task1_fno_finetune.py",
        "--train-hdf5",
        str(args.train_hdf5),
        "--base-checkpoint",
        str(args.base_checkpoint),
        "--run-dir",
        str(run_dir),
        "--val-hdf5",
        str(args.val_hdf5),
        "--steps",
        str(experiment.steps),
        "--batch-size",
        str(experiment.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--max-samples",
        str(experiment.max_samples),
        "--sample-start",
        str(experiment.sample_start),
        "--val-max-samples",
        str(experiment.val_max_samples),
        "--val-every",
        str(experiment.val_every),
        "--log-every",
        str(experiment.log_every),
        "--lr",
        str(experiment.lr),
        "--weight-decay",
        str(experiment.weight_decay),
        "--grad-clip",
        str(experiment.grad_clip),
        "--rollout-steps",
        str(experiment.rollout_steps),
        "--temporal-stride",
        str(experiment.temporal_stride),
        "--trainable",
        experiment.trainable,
        "--seed",
        str(experiment.seed),
    ]
    if args.device:
        command.extend(["--device", args.device])
    return command


def _load_result(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "finetune_result.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _best_step(result: dict[str, Any]) -> int | None:
    history = result.get("history") or []
    scored = [item for item in history if "competition_score_proxy" in item]
    if not scored:
        return None
    return max(scored, key=lambda item: float(item["competition_score_proxy"])).get("step")


def _summary_row(experiment: FinetuneExperiment, run_dir: Path, result: dict[str, Any] | None) -> dict[str, Any]:
    metrics = (result or {}).get("best_metrics") or {}
    base_metrics = (result or {}).get("base_metrics") or {}
    return {
        "name": experiment.name,
        "status": "done" if result else "missing",
        "hypothesis": experiment.hypothesis,
        "run_dir": str(run_dir),
        "trainable": experiment.trainable,
        "rollout_steps": experiment.rollout_steps,
        "temporal_stride": experiment.temporal_stride,
        "steps": experiment.steps,
        "lr": experiment.lr,
        "batch_size": experiment.batch_size,
        "val_every": experiment.val_every,
        "base_proxy": base_metrics.get("competition_score_proxy"),
        "best_proxy": metrics.get("competition_score_proxy"),
        "best_step": _best_step(result) if result else None,
        "best_mse": metrics.get("mse"),
        "best_forecast_mse": metrics.get("forecast_mse"),
        "best_long_horizon_mse": metrics.get("long_horizon_mse"),
        "elapsed_seconds": (result or {}).get("elapsed_seconds"),
    }


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda row: float(row["best_proxy"] or float("-inf")), reverse=True)
    (output_dir / "summary.json").write_text(json.dumps(ranked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0].keys()) if ranked else ["name"])
        writer.writeheader()
        writer.writerows(ranked)

    lines = [
        "# Task1 Nu0.001 Fine-tune Agent Summary",
        "",
        "| rank | name | strategy | rollout_steps | lr | steps | best_proxy | best_step | forecast_mse | long_horizon_mse |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            "| {rank} | {name} | {trainable} | {rollout_steps} | {lr:.1e} | {steps} | {best_proxy:.6f} | {best_step} | {forecast:.8f} | {long:.8f} |".format(
                rank=rank,
                name=row["name"],
                trainable=row["trainable"],
                rollout_steps=row["rollout_steps"],
                lr=float(row["lr"]),
                steps=row["steps"],
                best_proxy=float(row["best_proxy"] or 0.0),
                best_step=row["best_step"],
                forecast=float(row["best_forecast_mse"] or 0.0),
                long=float(row["best_long_horizon_mse"] or 0.0),
            )
        )
    lines.extend(["", "## Experiment Hypotheses", ""])
    for row in ranked:
        lines.append(f"- `{row['name']}`: {row['hypothesis']}")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_agent(args: argparse.Namespace) -> Path:
    experiments = quick_nu001_experiments()
    output_dir = Path(args.output_dir)
    runs_root = Path(args.runs_root)
    rows: list[dict[str, Any]] = []

    for experiment in experiments:
        run_dir = runs_root / f"task1-nu001-agent-{experiment.name}"
        existing = _load_result(run_dir)
        if existing and not args.force:
            rows.append(_summary_row(experiment, run_dir, existing))
            print(json.dumps({"event": "skip_existing", "run_dir": str(run_dir)}, ensure_ascii=False))
            continue

        command = _command_for_experiment(args, experiment, run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        print(json.dumps({"event": "start", "name": experiment.name, "command": command}, ensure_ascii=False))
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout_seconds,
        )
        elapsed = time.perf_counter() - started
        (run_dir / "agent_stdout.log").write_text(completed.stdout, encoding="utf-8")
        (run_dir / "agent_stderr.log").write_text(completed.stderr, encoding="utf-8")
        (run_dir / "agent_command.json").write_text(
            json.dumps({"command": command, "elapsed_seconds": elapsed, "returncode": completed.returncode}, indent=2) + "\n",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{experiment.name} failed with return code {completed.returncode}; see {run_dir}")
        rows.append(_summary_row(experiment, run_dir, _load_result(run_dir)))

    _write_summary(output_dir, rows)
    print(json.dumps({"event": "summary", "path": str(output_dir / "summary.md")}, ensure_ascii=False))
    return output_dir / "summary.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small autonomous Nu0.001 fine-tune strategy sweep.")
    parser.add_argument("--train-hdf5", default="data/pdebench_burgers/raw/1D_Burgers_Sols_Nu0.001.hdf5")
    parser.add_argument("--base-checkpoint", default="checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt")
    parser.add_argument("--val-hdf5", default="data/Task1/task1_val.hdf5")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-dir", default="runs/task1-nu001-finetune-agent")
    parser.add_argument("--eval-batch-size", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_agent(args)


if __name__ == "__main__":
    main()
