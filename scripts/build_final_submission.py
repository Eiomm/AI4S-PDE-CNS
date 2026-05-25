"""Assemble the FINAL submission/ directory from all per-task runs.

Produces:

    submission/
    ├── submission.json
    ├── methodology.pdf
    ├── task1_pred.hdf5        (if --skip-task1 not set)
    ├── task1_time.csv
    ├── task1_logs.log
    ├── task2_pred.hdf5
    ├── task2_time.csv
    ├── task2_logs.log
    ├── task3_pred.hdf5
    ├── task3_time.csv
    ├── task3_logs.log
    └── code/
        ├── task1/_sandbox_script_<hash>.py   (every attempt, in step order)
        ├── task2/...
        └── task3/...

Source of truth for each component:
  - Per-task pred/log/csv come from scripts/build_task_submission.py
  - code/task{N}/ collects EVERY sandbox script from the AIDE run for
    audit traceability (the official rule requires code/ to match the
    log's LLM call history)
  - submission.json is copied verbatim from scripts/submission.json
  - methodology.pdf is copied from scripts/methodology.pdf

Usage:
    python scripts/build_final_submission.py              # all 3 tasks, default run-roots
    python scripts/build_final_submission.py --tasks 1 2  # only tasks 1, 2
    python scripts/build_final_submission.py --zip        # also create submission.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

sys.path.insert(0, str(SCRIPTS))
from build_task_submission import package_task, TASK_CONFIG  # noqa: E402


def collect_sandbox_scripts(run_root: Path, code_dest: Path) -> int:
    """Copy every `_sandbox_script_*.py` from the AIDE run into code_dest,
    renumbered in mtime order as step01_, step02_, ...

    Audit rationale: the contest requires code/ to be traceable to the
    LLM call log. Each sandbox script is the exact program the LLM
    produced; preserving the full history (not just the final attempt)
    lets the audit correlate scripts ↔ JSONL entries.
    """
    code_dest.mkdir(parents=True, exist_ok=True)
    # The agent's per-attempt scripts live in artifacts/sandbox_scripts/.
    # The currently-running one is also in run/, but it's a duplicate.
    cands = list(run_root.glob("workspace/*/*/sandbox/artifacts/sandbox_scripts/_sandbox_script_*.py"))
    cands.sort(key=lambda p: p.stat().st_mtime)  # oldest -> newest
    for i, src in enumerate(cands, 1):
        # Original name: _sandbox_script_<hex>.py — keep the hex for traceability.
        hex_part = src.stem.replace("_sandbox_script_", "")
        dst = code_dest / f"step{i:02d}_{hex_part}.py"
        shutil.copy2(src, dst)
    return len(cands)


def write_submission_zip(submission_dir: Path, zip_path: Path) -> int:
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(submission_dir.rglob("*")):
            if p.is_file():
                arc = "submission/" + str(p.relative_to(submission_dir))
                zf.write(p, arcname=arc)
                n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--out", type=Path, default=REPO / "submission")
    parser.add_argument("--zip", action="store_true",
                        help="also produce submission.zip alongside submission/")
    args = parser.parse_args()

    out = args.out.resolve()
    code_root = out / "code"
    out.mkdir(parents=True, exist_ok=True)
    code_root.mkdir(parents=True, exist_ok=True)

    # 1) Run per-task packager into a TEMP sub-dir, then flatten into out/.
    summaries = []
    for task in args.tasks:
        cfg = TASK_CONFIG[task]
        tmp = out / f"_tmp_task{task}"
        try:
            info = package_task(task, run_root=None, out_dir=tmp)
        except FileNotFoundError as e:
            print(f"[skip task {task}] {e}")
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        summaries.append(info)

        # Move flattened outputs to submission root
        for name in (cfg["pred_name"], cfg["logs_name"], cfg["time_csv_name"]):
            src = tmp / name
            if src.exists():
                shutil.move(str(src), str(out / name))
        shutil.rmtree(tmp, ignore_errors=True)

        # Collect sandbox scripts into code/task{N}/
        n_scripts = collect_sandbox_scripts(Path(info["run_root"]), code_root / f"task{task}")
        info["sandbox_scripts_collected"] = n_scripts

    # 2) submission.json + methodology.pdf
    src_meta = SCRIPTS / "submission.json"
    src_pdf  = SCRIPTS / "methodology.pdf"
    if src_meta.exists():
        shutil.copy2(src_meta, out / "submission.json")
    else:
        print(f"[warn] missing {src_meta}; skipping")
    if src_pdf.exists():
        shutil.copy2(src_pdf, out / "methodology.pdf")
    else:
        print(f"[warn] missing {src_pdf}; consider running scripts/build_methodology_pdf.py")

    # 3) Print summary
    print()
    print("=" * 72)
    print(f"submission/  →  {out}")
    print("=" * 72)
    print(f"{'file':<32}{'size':>12}")
    print("-" * 44)
    for p in sorted(out.iterdir()):
        if p.is_file():
            print(f"  {p.name:<30}{p.stat().st_size:>12,} B")
        elif p.is_dir():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"  {p.name + '/':<30}{n:>10} files")
    print()
    print("Per-task summary:")
    for s in summaries:
        print(f"  task {s['task']}: "
              f"train={s['train_time']:.0f}s  inference={s['inference_time']:.1f}s  "
              f"({s['inference_source']})  "
              f"log_entries={s['log_entries']}  "
              f"sandbox_scripts={s.get('sandbox_scripts_collected', 0)}")

    # 4) Optional zip
    if args.zip:
        zip_path = out.parent / "submission.zip"
        n_files = write_submission_zip(out, zip_path)
        print()
        print(f"submission.zip: {n_files} files, {zip_path.stat().st_size:,} bytes -> {zip_path}")

    print()
    print("Submission ready. Review the files above before uploading.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
