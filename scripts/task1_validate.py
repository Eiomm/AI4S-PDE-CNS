#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai4sv2_task1.metrics import compute_task1_metrics
from ai4sv2_task1.hdf5_io import read_named_or_single
from ai4sv2_task1.validate import validate_task1_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Task1 prediction HDF5.")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--input", default=str(ROOT / "data" / "task1_test.hdf5"))
    parser.add_argument("--target", default=None, help="Optional validation target HDF5.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = {"validation": validate_task1_prediction(args.prediction, args.input)}
    if args.target:
        report["metrics"] = compute_task1_metrics(read_named_or_single(args.prediction, "tensor"), read_named_or_single(args.target, "tensor"))
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
