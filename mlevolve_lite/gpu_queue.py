from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from .node_schema import utc_now


class GPUQueue:
    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace)
        self.queue_path = self.workspace / "gpu_queue.jsonl"

    def run(self, node_id: str, cmd: list[str], log_path: Path | str, timeout_sec: int) -> dict:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        job = {
            "job_id": f"{node_id}-{int(time.time())}",
            "node_id": node_id,
            "cmd": cmd,
            "status": "running",
            "start_time": utc_now(),
            "end_time": None,
            "runtime_sec": None,
            "returncode": None,
            "log_path": str(log_path),
        }
        self._append(job)
        start = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            try:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_sec, check=False)
                job["returncode"] = proc.returncode
                job["status"] = "completed" if proc.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                job["returncode"] = 124
                job["status"] = "failed"
        job["end_time"] = utc_now()
        job["runtime_sec"] = time.time() - start
        self._append(job)
        return job

    def _append(self, job: dict) -> None:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        with self.queue_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(job, sort_keys=True) + "\n")
