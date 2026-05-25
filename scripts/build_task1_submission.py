"""Assemble Task-1 submission files.

Inputs (all relative to repo root):
  - outputs/aide_task1_claude/latest/llm_io/llm-*.jsonl  (or any AIDE run)
  - outputs/aide_task1_claude/latest/workspace/.../sandbox/task1_pred.hdf5
  - tasks/ai4s-pde-task1-burgers-fixed/data/task1_test.hdf5

Outputs (written into --out, default submission/task1/):
  - task1_pred.hdf5       (copied from sandbox)
  - task1_logs.log        (proxy jsonl, contest-format-compliant)
  - task1_time.csv        (train_time = LLM/session wall-clock, inference_time = measured)

Usage:
  python scripts/build_task1_submission.py \
    --run-root outputs/aide_task1_claude/latest \
    --out submission/task1

If --skip-inference-measure is passed, inference_time is taken from the AIDE
sandbox script's wall-clock instead of doing a fresh standalone rollout.
"""

import argparse
import csv
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


def find_pred_in_sandbox(run_root: Path) -> Path:
    candidates = list(run_root.glob("workspace/*/*/sandbox/task1_pred.hdf5"))
    if not candidates:
        raise FileNotFoundError(f"No task1_pred.hdf5 under {run_root}/workspace/**/sandbox/")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_proxy_log(run_root: Path) -> Path:
    candidates = list(run_root.glob("llm_io/llm-*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No llm-*.jsonl under {run_root}/llm_io/")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def validate_log(path: Path) -> tuple[int, float, datetime, datetime]:
    """Returns (n_entries, sum_elapsed, first_ts, last_ts). Raises on bad format."""
    required = {"timestamp", "elapsed_seconds"}
    n = 0
    sum_elapsed = 0.0
    timestamps: list[datetime] = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            missing = required - obj.keys()
            if missing:
                raise ValueError(f"line {i} missing {missing}")
            if "response" not in obj and "tool_calls" not in obj:
                raise ValueError(f"line {i} missing both response and tool_calls")
            n += 1
            sum_elapsed += float(obj["elapsed_seconds"])
            timestamps.append(datetime.fromisoformat(obj["timestamp"]))
    if not timestamps:
        raise ValueError("log is empty")
    timestamps.sort()
    return n, sum_elapsed, timestamps[0], timestamps[-1]


def extract_aide_train_time(run_root: Path) -> float | None:
    """Wall-clock between AIDE start and end, as a more accurate train_time."""
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


def extract_inference_time_from_txt(pred_path: Path) -> float | None:
    """Read inference_time from `task1_inference_time.txt` written by the
    agent in the same sandbox directory as the prediction file.

    Accepts either `<float>` or `INFERENCE_TIME=<float>` content. Returns
    the first parseable float found. Primary source — survives stdout
    truncation, log rotation, and re-runs.
    """
    txt = pred_path.parent / "task1_inference_time.txt"
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
    """Parse the agent's stdout for `INFERENCE_TIME=<float>` lines.

    Secondary source if the HDF5 attribute is missing. If multiple lines
    exist (e.g. val + test rollouts), take the last one — that is the
    test-set rollout.
    """
    log = run_root / "aide_stdout_stderr.log"
    if not log.exists():
        return None
    pattern = re.compile(r"INFERENCE_TIME=([0-9.eE+-]+)")
    values: list[float] = []
    with log.open() as f:
        for line in f:
            for match in pattern.finditer(line):
                try:
                    values.append(float(match.group(1)))
                except ValueError:
                    pass
    if not values:
        return None
    return values[-1]


def measure_inference_time(repo_root: Path) -> float:
    """Fallback: reproduce the test rollout standalone and time it.

    Only used when the agent's stdout did not contain a parseable
    INFERENCE_TIME line.
    """
    import numpy as np
    import h5py
    import torch
    sys.path.insert(0, str(repo_root / "src"))
    from ai4sv2_task1.models.fno import load_fno_checkpoint, rollout_fno  # type: ignore

    task_dir = repo_root / "tasks" / "ai4s-pde-task1-burgers-fixed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fno_checkpoint(
        str(task_dir / "burgers_FNO" / "1D_Burgers_Sols_Nu0.001_FNO.pt"),
        device,
    )
    with h5py.File(task_dir / "data" / "task1_test.hdf5", "r") as f:
        initial = f["tensor"][:].astype(np.float32)
        x = f["x-coordinate"][:].astype(np.float32)
    t_full = np.arange(200, dtype=np.float32)

    t0 = time.perf_counter()
    pred = rollout_fno(model, initial, x, t_full, device, batch_size=100)
    elapsed = time.perf_counter() - t0
    assert pred.shape == (1000, 200, 256)
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("outputs/aide_task1_claude/latest"))
    parser.add_argument("--out", type=Path, default=Path("submission/task1"))
    parser.add_argument("--skip-inference-measure", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    run_root = (repo_root / args.run_root).resolve() if not args.run_root.is_absolute() else args.run_root
    out = (repo_root / args.out).resolve() if not args.out.is_absolute() else args.out
    out.mkdir(parents=True, exist_ok=True)

    pred_src = find_pred_in_sandbox(run_root)
    log_src = find_proxy_log(run_root)

    print(f"run_root      : {run_root}")
    print(f"prediction    : {pred_src} ({pred_src.stat().st_size/1e6:.1f} MB)")
    print(f"proxy log     : {log_src} ({log_src.stat().st_size/1e3:.1f} KB)")

    n_entries, sum_elapsed, first_ts, last_ts = validate_log(log_src)
    print(f"log validates : {n_entries} entries; first {first_ts}; last {last_ts}")

    aide_wall = extract_aide_train_time(run_root)
    log_wall = (last_ts - first_ts).total_seconds()
    train_time = aide_wall if aide_wall is not None else max(log_wall, sum_elapsed)
    print(f"train_time    : {train_time:.1f}s (aide_wall={aide_wall}, log_wall={log_wall:.1f}, sum_elapsed={sum_elapsed:.1f})")

    # Primary:   task1_inference_time.txt (agent writes it next to pred file).
    # Secondary: agent's stdout INFERENCE_TIME=<sec> line.
    # Fallback:  re-measure locally.
    inference_time_from_txt = extract_inference_time_from_txt(pred_src)
    inference_time_from_log = extract_inference_time_from_log(run_root)
    if inference_time_from_txt is not None:
        inference_time = inference_time_from_txt
        print(f"inference_time: {inference_time:.3f}s (read from task1_inference_time.txt)")
    elif inference_time_from_log is not None:
        inference_time = inference_time_from_log
        print(f"inference_time: {inference_time:.3f}s (parsed from agent stdout)")
    elif args.skip_inference_measure:
        inference_time = log_wall
        print(f"inference_time: {inference_time:.1f}s (fallback to session wall-clock)")
    else:
        print("No inference_time in txt or stdout; reproducing rollout standalone ...")
        inference_time = measure_inference_time(repo_root)
        print(f"inference_time: {inference_time:.3f}s (re-measured fallback)")

    shutil.copy2(pred_src, out / "task1_pred.hdf5")
    shutil.copy2(log_src, out / "task1_logs.log")
    with (out / "task1_time.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_time", "inference_time"])
        w.writerow([f"{train_time:.1f}", f"{inference_time:.1f}"])

    print()
    print(f"written to    : {out}/")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}  {p.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
