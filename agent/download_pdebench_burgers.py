from __future__ import annotations

import argparse

from .pde_finetune import build_download_command, pdebench_burgers_filename


def download_burgers_files(*, nu_values: list[str], local_dir: str) -> str:
    from huggingface_hub import snapshot_download

    filenames = [pdebench_burgers_filename(nu) for nu in nu_values]
    return snapshot_download(
        repo_id="pdebench/Burgers",
        repo_type="dataset",
        allow_patterns=filenames,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected PDEBench 1D Burgers files from Hugging Face.")
    parser.add_argument("--nu", nargs="+", default=["0.01", "0.1"])
    parser.add_argument("--local-dir", default="data/pdebench_burgers/raw")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = build_download_command(nu_values=args.nu, local_dir=args.local_dir)
    if args.dry_run:
        print(" ".join(command))
        return
    print(download_burgers_files(nu_values=args.nu, local_dir=args.local_dir))


if __name__ == "__main__":
    main()
