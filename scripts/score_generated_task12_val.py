#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


OFFICIAL_HELPERS = r'''

def _official_like_task12_score(pred, true):
    import math
    import numpy as np

    pred = pred.astype(np.float64, copy=False)
    true = true.astype(np.float64, copy=False)
    seg1 = (slice(None), slice(10, 57), slice(None))
    seg2 = (slice(None), slice(57, 105), slice(None))
    seg3 = (slice(None), slice(105, 200), slice(None))

    def rel_mse(index):
        err = np.sum((pred[index] - true[index]) ** 2)
        denom = np.sum(true[index] ** 2) + 1e-12
        return float(err / denom)

    rel1 = rel_mse(seg1)
    rel2 = rel_mse(seg2)
    mse3 = float(np.mean((pred[seg3] - true[seg3]) ** 2))
    rmse3 = math.sqrt(mse3)
    score1 = 100.0 * math.exp(-20.0 * rel1)
    score2 = 100.0 * math.exp(-10.0 * rel2)
    lorentzian3 = 100.0 / (1.0 + 10.0 * rmse3)
    segmented_lorentz_only = 0.25 * score1 + 0.25 * score2 + 0.50 * lorentzian3
    print("OFFICIAL_LIKE_TASK12_LORENTZ_ONLY")
    print(f"  rel_mse_1={rel1:.8g} score1={score1:.6f}")
    print(f"  rel_mse_2={rel2:.8g} score2={score2:.6f}")
    print(f"  mse3={mse3:.8g} rmse3={rmse3:.8g} lorentzian3={lorentzian3:.6f}")
    print(f"  segmented_score_lorentz_only={segmented_lorentz_only:.6f}")
    print("  note=Frechet branch not included because official FD implementation is not available locally")

'''


def prepare_scratch(task_root: Path, scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    for name in ("data", "burgers_FNO"):
        src = task_root / name
        dst = scratch / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src, target_is_directory=src.is_dir())


def instrument_source(task: str, source: str) -> str:
    source = OFFICIAL_HELPERS + "\n" + source
    if task == "task1":
        marker = 'print("\\nLoading test data...")'
        injection = (
            'print("\\nScoring official-like validation with selected weights...")\n'
            'val_pred_best = rollout_fno(model, val_true[:, :10, :], val_x, val_t, device, batch_size=50)\n'
            '_official_like_task12_score(val_pred_best, val_true)\n'
            'print("\\nLoading test data...")'
        )
    else:
        marker = 'print(f"Validation weighted score: {val_score:.6f}")'
        injection = (
            'print(f"Validation weighted score: {val_score:.6f}")\n'
            '_official_like_task12_score(val_pred, val_true_raw)'
        )
    if marker not in source:
        raise SystemExit(f"Could not find injection marker for {task}: {marker}")
    return source.replace(marker, injection, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--scratch", required=True)
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    task_root = Path(args.task_root).resolve()
    scratch = Path(args.scratch).resolve()
    prepare_scratch(task_root, scratch)

    code = instrument_source(args.task, source_path.read_text())
    os.chdir(scratch)
    print(f"RUN_DIR={scratch}")
    print(f"SOURCE={source_path}")
    exec(compile(code, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})


if __name__ == "__main__":
    main()
