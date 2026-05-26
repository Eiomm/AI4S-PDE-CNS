"""Per-task submission packager (Task 1 / 2 / 3).

Generic version that supersedes build_task{1,2,3}_submission.py. Selects
task layout via `--task {1,2,3}`.

Inputs (resolved automatically from --run-root or each task's default
`outputs/aide_task{N}_claude/latest`):
  - llm_io/llm-*.jsonl                     (proxy log → task{N}_logs.log)
  - workspace/.../sandbox/task{N}_pred.hdf5  (agent prediction file)
  - workspace/.../sandbox/task{N}_inference_time.txt (inference timer)
  - aide_stdout_stderr.log                  (stdout INFERENCE_TIME= line)

Outputs (default submission/task{N}/):
  - task{N}_pred.hdf5
  - task{N}_logs.log
  - task{N}_time.csv  (train_time = LLM-call wall-clock span,
                       inference_time = agent-measured rollout seconds)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = Path(os.environ.get("AI4S_OUTPUTS_DIR", REPO / "outputs"))

TASK_CONFIG: dict[int, dict] = {
    1: {
        "task_dir": "tasks/ai4s-pde-task1-burgers-fixed",
        "run_root_default": "outputs/aide_task1_claude/latest",
        "pred_name": "task1_pred.hdf5",
        "time_txt_name": "task1_inference_time.txt",
        "logs_name": "task1_logs.log",
        "time_csv_name": "task1_time.csv",
        "test_rel": "data/task1_test.hdf5",
        "expected_shape": (1000, 200, 256),
        "first_n": 10,
    },
    2: {
        "task_dir": "tasks/ai4s-pde-task2-burgers-multinu",
        "run_root_default": "outputs/aide_task2_claude/latest",
        "pred_name": "task2_pred.hdf5",
        "time_txt_name": "task2_inference_time.txt",
        "logs_name": "task2_logs.log",
        "time_csv_name": "task2_time.csv",
        "test_rel": "data/task2_test.h5",
        "expected_shape": (1000, 200, 256),
        "first_n": 10,
    },
    3: {
        "task_dir": "tasks/ai4s-pde-task3-ks-multiparam",
        "run_root_default": "outputs/aide_task3_claude/latest",
        "pred_name": "task3_pred.hdf5",
        "time_txt_name": "task3_inference_time.txt",
        "logs_name": "task3_logs.log",
        "time_csv_name": "task3_time.csv",
        "test_rel": "data/KS_test.hdf5",
        "expected_shape": (100, 400, 256),
        "first_n": 20,
    },
}


# ----------------------------------------------------------------------------
# Discovery helpers
# ----------------------------------------------------------------------------

def find_pred_in_sandbox(run_root: Path, pred_name: str) -> Path:
    cands = list(run_root.glob(f"workspace/*/*/sandbox/{pred_name}"))
    if not cands:
        raise FileNotFoundError(f"No {pred_name} under {run_root}/workspace/**/sandbox/")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def find_best_pred_candidate(run_root: Path, pred_name: str) -> Path | None:
    manifest_path = run_root / "best_candidates" / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    best = manifest.get("best_candidate")
    if not isinstance(best, dict):
        return None
    pred_path = best.get("pred_path")
    if not pred_path:
        return None
    pred = Path(pred_path)
    if pred.exists() and pred.name == pred_name:
        return pred
    return None


def find_proxy_log(run_root: Path) -> Path:
    cands = list(run_root.glob("llm_io/llm-*.jsonl"))
    if not cands:
        raise FileNotFoundError(f"No llm-*.jsonl under {run_root}/llm_io/")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def validate_log(path: Path) -> tuple[int, float, datetime, datetime]:
    required = {"timestamp", "elapsed_seconds"}
    n, sum_elapsed = 0, 0.0
    timestamps: list[datetime] = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            missing = required - obj.keys()
            if missing:
                raise ValueError(f"line {i}: missing {missing}")
            if "response" not in obj and "tool_calls" not in obj:
                raise ValueError(f"line {i}: missing both response and tool_calls")
            n += 1
            sum_elapsed += float(obj["elapsed_seconds"])
            timestamps.append(datetime.fromisoformat(obj["timestamp"]))
    if not timestamps:
        raise ValueError("log is empty")
    timestamps.sort()
    return n, sum_elapsed, timestamps[0], timestamps[-1]


def extract_aide_train_time(run_root: Path) -> float | None:
    log = run_root / "aide_stdout_stderr.log"
    if not log.exists():
        return None
    line_re = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]")
    starts, ends = [], []
    with log.open() as f:
        for line in f:
            m = line_re.match(line)
            if not m:
                continue
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
            if "Starting evaluation" in line or "AIDEWorkflow starting" in line:
                starts.append(ts)
            if "Task " in line and "LLM cost" in line:
                ends.append(ts)
    if not starts or not ends:
        return None
    return (max(ends) - min(starts)).total_seconds()


def extract_inference_time_from_txt(pred_path: Path, txt_name: str) -> float | None:
    txt = pred_path.parent / txt_name
    if not txt.exists():
        return None
    try:
        content = txt.read_text().strip()
    except Exception:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", content)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_inference_time_from_log(run_root: Path) -> float | None:
    log = run_root / "aide_stdout_stderr.log"
    if not log.exists():
        return None
    pattern = re.compile(r"INFERENCE_TIME=([0-9.eE+-]+)")
    values: list[float] = []
    with log.open() as f:
        for line in f:
            for m in pattern.finditer(line):
                try:
                    values.append(float(m.group(1)))
                except ValueError:
                    pass
    return values[-1] if values else None


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def package_task(task: int, run_root: Path | None, out_dir: Path | None) -> dict:
    cfg = TASK_CONFIG[task]
    run_root_default = OUTPUTS_DIR / f"aide_task{task}_claude/latest"
    run_root = (run_root or run_root_default).resolve()
    out_dir  = (out_dir  or (REPO / f"submission/task{task}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_src = find_best_pred_candidate(run_root, cfg["pred_name"])
    if pred_src is None:
        pred_src = find_pred_in_sandbox(run_root, cfg["pred_name"])
    log_src  = find_proxy_log(run_root)

    n, sum_elapsed, first_ts, last_ts = validate_log(log_src)
    aide_wall = extract_aide_train_time(run_root)
    log_wall  = (last_ts - first_ts).total_seconds()
    train_time = aide_wall if aide_wall is not None else max(log_wall, sum_elapsed)

    inf_txt = extract_inference_time_from_txt(pred_src, cfg["time_txt_name"])
    inf_log = extract_inference_time_from_log(run_root)
    if inf_txt is not None:
        inference_time = inf_txt
        inf_src = f"{cfg['time_txt_name']}"
    elif inf_log is not None:
        inference_time = inf_log
        inf_src = "agent stdout"
    else:
        inference_time = log_wall  # last-resort upper bound
        inf_src = "wall-clock fallback"

    shutil.copy2(pred_src, out_dir / cfg["pred_name"])
    shutil.copy2(log_src, out_dir / cfg["logs_name"])
    with (out_dir / cfg["time_csv_name"]).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_time", "inference_time"])
        w.writerow([f"{train_time:.1f}", f"{inference_time:.1f}"])

    return {
        "task": task,
        "run_root": str(run_root),
        "pred_src": str(pred_src),
        "log_entries": n,
        "train_time": train_time,
        "inference_time": inference_time,
        "inference_source": inf_src,
        "out_dir": str(out_dir),
        "files": sorted([p.name for p in out_dir.iterdir()]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default submission/task{N})")
    args = parser.parse_args()

    info = package_task(args.task, args.run_root, args.out)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
