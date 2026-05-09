from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from typing import Any


def _nvidia_smi() -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def collect_environment() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "nvidia_smi": _nvidia_smi(),
        "torch_available": False,
        "cuda_available": False,
        "cuda_device_count": 0,
    }
    try:
        import torch  # type: ignore
    except Exception as exc:
        report["torch_import_error"] = str(exc)
        return report
    report["torch_available"] = True
    report["torch_version"] = getattr(torch, "__version__", "unknown")
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["cuda_device_count"] = int(torch.cuda.device_count()) if report["cuda_available"] else 0
    if report["cuda_available"]:
        report["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(report["cuda_device_count"])
        ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Print local AI4S environment details.")
    parser.parse_args()
    print(json.dumps(collect_environment(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
