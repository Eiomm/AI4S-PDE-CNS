from __future__ import annotations

import sys
from pathlib import Path


BURGERS_NU_FILES: dict[str, str] = {
    "0.001": "1D_Burgers_Sols_Nu0.001.hdf5",
}


BURGERS_CHECKPOINTS: dict[str, str] = {
    "0.001": "checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt",
}


def normalize_nu(nu: str | float) -> str:
    text = str(nu).strip()
    aliases = {
        "1": "1.0",
        "0.100": "0.1",
        "0.010": "0.01",
        "0.0010": "0.001",
    }
    text = aliases.get(text, text)
    if text not in BURGERS_NU_FILES:
        raise ValueError(f"Unsupported Burgers nu {nu!r}; supported values: {sorted(BURGERS_NU_FILES)}")
    return text


def pdebench_burgers_filename(nu: str | float) -> str:
    return BURGERS_NU_FILES[normalize_nu(nu)]


def discover_pdebench_burgers_files(raw_dir: str | Path) -> dict[str, Path]:
    root = Path(raw_dir)
    found: dict[str, Path] = {}
    for nu, filename in BURGERS_NU_FILES.items():
        path = root / filename
        if path.is_file():
            found[nu] = path
    return found


def is_better_metric(
    candidate: dict[str, float],
    current_best: dict[str, float] | None,
    *,
    metric: str = "mse",
    min_improvement: float = 0.0,
    maximize: bool = False,
) -> bool:
    if current_best is None:
        return True
    if maximize:
        return float(candidate[metric]) > float(current_best[metric]) + float(min_improvement)
    return float(candidate[metric]) < float(current_best[metric]) - float(min_improvement)


def build_download_command(
    *,
    nu_values: list[str | float],
    local_dir: str | Path = "data/pdebench_burgers/raw",
) -> list[str]:
    filenames = [pdebench_burgers_filename(nu) for nu in nu_values]
    return [
        "huggingface-cli",
        "download",
        "pdebench/Burgers",
        *filenames,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(local_dir),
    ]


def build_finetune_command(
    *,
    train_hdf5: str | Path,
    base_checkpoint: str | Path,
    run_dir: str | Path,
    val_hdf5: str | Path = "data/Task1/task1_val.hdf5",
    steps: int = 500,
    batch_size: int = 8,
    lr: float = 1.0e-4,
    spatial_size: int = 256,
    max_samples: int | None = 2048,
) -> list[str]:
    command = [
        sys.executable,
        "code/train_task1_fno_finetune.py",
        "--train-hdf5",
        str(train_hdf5),
        "--base-checkpoint",
        str(base_checkpoint),
        "--run-dir",
        str(run_dir),
        "--val-hdf5",
        str(val_hdf5),
        "--steps",
        str(int(steps)),
        "--batch-size",
        str(int(batch_size)),
        "--lr",
        str(float(lr)),
        "--spatial-size",
        str(int(spatial_size)),
    ]
    if max_samples is not None:
        command.extend(["--max-samples", str(int(max_samples))])
    return command
