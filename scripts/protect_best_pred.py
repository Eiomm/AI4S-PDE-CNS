#!/usr/bin/env python3
"""Snapshot per-step prediction files and track the best validation result.

The AIDE sandbox writes every attempt to the same taskN_pred.hdf5 path. This
helper watches that file, copies each new version to best_candidates/, and
updates a small manifest once the corresponding validation score appears in
the AIDE log. The packager can then use the best saved candidate instead of
blindly taking the newest sandbox file.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py


TASKS: dict[int, dict[str, Any]] = {
    1: {
        "pred_name": "task1_pred.hdf5",
        "time_name": "task1_inference_time.txt",
        "score_name": "validation_score",
        "shape": (1000, 200, 256),
        "lower_is_better": False,
    },
    2: {
        "pred_name": "task2_pred.hdf5",
        "time_name": "task2_inference_time.txt",
        "score_name": "validation_score",
        "shape": (1000, 200, 256),
        "lower_is_better": False,
    },
    3: {
        "pred_name": "task3_pred.hdf5",
        "time_name": "task3_inference_time.txt",
        "score_name": "validation_score",
        "shape": (100, 400, 256),
        "lower_is_better": False,
    },
}

TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]")
STEP_RE = re.compile(
    r"Step\s+(\d+)\s+complete\.\s+Buggy:\s+False(?:\.\s+Metric:\s+Metric([↑↓])\(([0-9.eE+-]+)\))?"
)
SCORE_PATTERNS = [
    re.compile(r"Best validation score:\s*([0-9.eE+-]+)"),
    re.compile(r"official_like_segment_score:\s*([0-9.eE+-]+)"),
    re.compile(r"Task3_prediction_score:\s*([0-9.eE+-]+)"),
    re.compile(r"Task completed successfully\. Best validation score:\s*([0-9.eE+-]+)"),
    re.compile(r"Final validation weighted score:\s*([0-9.eE+-]+)"),
    re.compile(r"Validation weighted score:\s*([0-9.eE+-]+)"),
]


@dataclass
class StepRecord:
    step: int
    score: float
    success_ts: float | None
    step_ts: float | None


def parse_log_records(log_path: Path) -> list[StepRecord]:
    if not log_path.exists():
        return []

    records: list[StepRecord] = []
    last_score: float | None = None
    last_success_ts: float | None = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_ts: float | None = None
            ts_match = TS_RE.match(line)
            if ts_match:
                dt = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S,%f")
                line_ts = dt.timestamp()

            if "Script execution finished: succeeded" in line:
                last_success_ts = line_ts

            for pattern in SCORE_PATTERNS:
                score_match = pattern.search(line)
                if score_match:
                    try:
                        last_score = float(score_match.group(1))
                    except ValueError:
                        pass

            step_match = STEP_RE.search(line)
            if step_match:
                step_score: float | None = None
                if step_match.group(3) is not None:
                    try:
                        step_score = float(step_match.group(3))
                    except ValueError:
                        step_score = None
                if step_score is None:
                    step_score = last_score
                if step_score is None:
                    continue
                records.append(
                    StepRecord(
                        step=int(step_match.group(1)),
                        score=step_score,
                        success_ts=last_success_ts,
                        step_ts=line_ts,
                    )
                )
                last_score = None

    return records


def find_current_pred(run_root: Path, pred_name: str) -> Path | None:
    cands = list(run_root.glob(f"workspace/*/*/sandbox/{pred_name}"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def valid_prediction_file(path: Path, expected_shape: tuple[int, int, int]) -> tuple[bool, str | None]:
    if not path.exists():
        return False, "missing"
    try:
        with h5py.File(path, "r") as f:
            if "tensor" not in f:
                return False, "missing tensor dataset"
            shape = tuple(f["tensor"].shape)
            if shape != expected_shape:
                return False, f"wrong tensor shape {shape}, expected {expected_shape}"
    except Exception as exc:
        return False, f"cannot read hdf5: {exc}"
    return True, None


def load_manifest(path: Path, task: int, run_root: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {
        "task": task,
        "run_root": str(run_root.resolve()),
        "lower_is_better": bool(TASKS[task].get("lower_is_better", False)),
        "score_name": TASKS[task]["score_name"],
        "candidates": [],
        "best_candidate": None,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def snapshot_current_pred(task: int, run_root: Path, manifest: dict[str, Any]) -> bool:
    cfg = TASKS[task]
    pred = find_current_pred(run_root, cfg["pred_name"])
    if pred is None:
        return False

    ok, reason = valid_prediction_file(pred, cfg["shape"])
    if not ok:
        manifest["last_skipped"] = {
            "source_pred": str(pred),
            "reason": reason,
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }
        return True

    stat = pred.stat()
    source_key = f"{stat.st_mtime_ns}:{stat.st_size}"
    for cand in manifest["candidates"]:
        if cand.get("source_key") == source_key:
            return False

    out_root = run_root / "best_candidates"
    out_root.mkdir(parents=True, exist_ok=True)
    snap_id = f"snapshot_{len(manifest['candidates']) + 1:03d}_{stat.st_mtime_ns}"
    snap_dir = out_root / snap_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    pred_dest = snap_dir / cfg["pred_name"]
    shutil.copy2(pred, pred_dest)

    time_src = pred.parent / cfg["time_name"]
    time_dest = snap_dir / cfg["time_name"]
    if time_src.exists():
        shutil.copy2(time_src, time_dest)

    manifest["candidates"].append(
        {
            "snapshot_id": snap_id,
            "source_key": source_key,
            "source_pred": str(pred),
            "source_pred_mtime": stat.st_mtime,
            "source_pred_size": stat.st_size,
            "pred_path": str(pred_dest),
            "time_path": str(time_dest) if time_dest.exists() else None,
            "valid": True,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "step": None,
            "score": None,
            "score_name": cfg["score_name"],
        }
    )
    return True


def attach_scores(manifest: dict[str, Any], records: list[StepRecord]) -> bool:
    changed = False
    candidates = [cand for cand in manifest["candidates"] if cand.get("valid", True)]
    used_steps = {cand.get("step") for cand in candidates if cand.get("step") is not None}

    for cand in candidates:
        if cand.get("step") is not None:
            continue

        source_ts = float(cand.get("source_pred_mtime") or 0.0)
        eligible = [
            rec
            for rec in records
            if rec.step not in used_steps
            and (rec.success_ts is None or rec.success_ts + 10.0 >= source_ts)
        ]
        if not eligible:
            continue

        eligible.sort(key=lambda rec: abs(((rec.success_ts or rec.step_ts or source_ts) - source_ts)))
        rec = eligible[0]
        cand["step"] = rec.step
        cand["score"] = rec.score
        cand["score_recorded_at"] = datetime.now().isoformat(timespec="seconds")
        used_steps.add(rec.step)
        changed = True

    scored = [cand for cand in candidates if cand.get("score") is not None]
    if scored:
        lower_is_better = bool(manifest.get("lower_is_better", False))
        key_fn = lambda cand: float(cand["score"])
        best = min(scored, key=key_fn) if lower_is_better else max(scored, key=key_fn)
        if manifest.get("best_candidate") != best:
            manifest["best_candidate"] = best
            changed = True

    return changed


def validate_existing_candidates(task: int, manifest: dict[str, Any]) -> bool:
    cfg = TASKS[task]
    changed = False
    for cand in manifest.get("candidates", []):
        pred_path = cand.get("pred_path")
        if not pred_path:
            continue
        ok, reason = valid_prediction_file(Path(pred_path), cfg["shape"])
        if cand.get("valid") != ok:
            cand["valid"] = ok
            changed = True
        if not ok and cand.get("invalid_reason") != reason:
            cand["invalid_reason"] = reason
            changed = True
        if ok and cand.get("invalid_reason") is not None:
            cand.pop("invalid_reason", None)
            changed = True

    best = manifest.get("best_candidate")
    if isinstance(best, dict):
        pred_path = best.get("pred_path")
        ok = False
        if pred_path:
            ok, _ = valid_prediction_file(Path(pred_path), cfg["shape"])
        if not ok:
            manifest["best_candidate"] = None
            changed = True
    return changed


def copy_best_alias(task: int, run_root: Path, manifest: dict[str, Any]) -> bool:
    best = manifest.get("best_candidate")
    if not best:
        return False
    cfg = TASKS[task]
    pred_path = Path(best["pred_path"])
    ok, _ = valid_prediction_file(pred_path, cfg["shape"])
    if not ok:
        return False

    best_dir = run_root / "best_candidates" / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pred_path, best_dir / cfg["pred_name"])

    time_path = best.get("time_path")
    if time_path and Path(time_path).exists():
        shutil.copy2(Path(time_path), best_dir / cfg["time_name"])
    return True


def update_once(task: int, run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "best_candidates" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path, task, run_root)
    expected_lower = bool(TASKS[task].get("lower_is_better", False))
    if manifest.get("lower_is_better") != expected_lower:
        manifest["lower_is_better"] = expected_lower

    changed = validate_existing_candidates(task, manifest)
    changed = snapshot_current_pred(task, run_root, manifest) or changed
    records = parse_log_records(run_root / "aide_stdout_stderr.log")
    changed = attach_scores(manifest, records) or changed
    if copy_best_alias(task, run_root, manifest):
        changed = True
    if changed or not manifest_path.exists():
        write_manifest(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=sorted(TASKS), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    if args.watch:
        while True:
            update_once(args.task, run_root)
            time.sleep(args.poll_seconds)

    manifest = update_once(args.task, run_root)
    best = manifest.get("best_candidate")
    if best:
        print(
            f"task {args.task}: best step={best.get('step')} "
            f"score={best.get('score')} pred={best.get('pred_path')}"
        )
    else:
        print(f"task {args.task}: no scored candidate yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
