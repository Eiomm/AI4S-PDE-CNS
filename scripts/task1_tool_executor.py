#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import html
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, unquote, urlparse, parse_qs
from urllib.request import Request, urlopen

import h5py

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai4sv2_task1.memory import add_record_to_registry, export_run_memory, promote_candidate
from ai4sv2_task1.predict import run_prediction


CONFIG_BY_REPLAY_TARGET = {
    "official_fno": "configs/official_fno.yaml",
    "official_fno_baseline": "configs/official_fno.yaml",
    "official_fno_checkpoint_replay": "configs/official_fno.yaml",
    "fno": "configs/official_fno.yaml",
    "fno_baseline": "configs/official_fno.yaml",
    "nu0.001_fno.pt": "configs/official_fno.yaml",
    "checkpoints/official/nu0.001_fno.pt": "configs/official_fno.yaml",
    "official_unet_pf20": "configs/official_unet_pf20.yaml",
    "official_unet_pf20_baseline": "configs/official_unet_pf20.yaml",
    "official_unet_pf20_checkpoint_replay": "configs/official_unet_pf20.yaml",
    "unet_pf20": "configs/official_unet_pf20.yaml",
    "unet_pf20_baseline": "configs/official_unet_pf20.yaml",
    "nu0.001_unet_pf20.pt": "configs/official_unet_pf20.yaml",
    "checkpoints/official/nu0.001_unet_pf20.pt": "configs/official_unet_pf20.yaml",
    "official_ensemble": "configs/official_ensemble.yaml",
    "official_ensemble_postprocess": "configs/official_ensemble_postprocess.yaml",
}

FNO_TRAINABLE_MODULES = {"fc0", "conv0", "w0", "conv1", "w1", "conv2", "w2", "conv3", "w3", "fc1", "fc2"}


def utc_stamp() -> str:
    """生成 executor 日志时间戳。"""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_log_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def resolve_root_path(raw_path: str | None, fallback: Path) -> Path:
    if not raw_path:
        return fallback
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def tool_output_root() -> Path:
    return resolve_root_path(os.environ.get("AI4S_TASK1_TOOL_OUTPUT_ROOT"), ROOT / "agent_workspace" / "tool_outputs")


def executor_log_root() -> Path:
    return resolve_root_path(os.environ.get("AI4S_TASK1_EXECUTOR_LOG_ROOT"), ROOT / "agent_workspace" / "executor_logs")


def experiment_runs_root() -> Path:
    return resolve_root_path(os.environ.get("AI4S_TASK1_RUNS_ROOT"), ROOT / "runs" / "task1")


def experiment_dir() -> Path | None:
    raw = os.environ.get("AI4S_TASK1_EXPERIMENT_DIR")
    if not raw:
        return None
    return resolve_root_path(raw, ROOT / "agent_workspace" / "experiments")


def experiment_state_path() -> Path | None:
    raw = os.environ.get("AI4S_TASK1_STATE_PATH")
    if raw:
        return resolve_root_path(raw, ROOT / "memory" / "working" / "task1_state.json")
    exp = experiment_dir()
    return exp / "state.json" if exp is not None else None


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def resolve_any_path(raw: Any) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else ROOT / path


def require_current_experiment_artifact(raw: Any, *, purpose: str) -> Path:
    path = resolve_any_path(raw)
    exp = experiment_dir()
    if exp is None:
        return path
    if not is_relative_to(path, exp):
        raise ValueError(
            f"{purpose} 必须来自当前实验目录 {relative_to_root(exp)}；"
            f"历史 memory/runs 只能作为策略提示，不能直接作为当前 submission artifact: {raw}"
        )
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_state() -> dict[str, Any]:
    path = experiment_state_path()
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def strategy_score(program: dict[str, Any] | None) -> float:
    if not isinstance(program, dict):
        return float("-inf")
    metrics = program.get("metrics") if isinstance(program.get("metrics"), dict) else {}
    for key in ("combined_score", "competition_score_proxy", "best_score", "score"):
        value = metrics.get(key, program.get(key))
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
    return float("-inf")


def strategy_db(state: dict[str, Any]) -> dict[str, Any]:
    db = state.setdefault("strategy_db", {})
    db.setdefault("schema", "task1_strategy_db_v1")
    db.setdefault("programs", {})
    db.setdefault("archive", [])
    db.setdefault("best_strategy_id", None)
    db.setdefault("num_islands", 3)
    db.setdefault("current_island", 0)
    db.setdefault("islands", {str(index): [] for index in range(int(db.get("num_islands") or 3))})
    db.setdefault("feature_map", {})
    db.setdefault("active_parent_id", None)
    db.setdefault("active_inspiration_ids", [])
    return db


def _stdout_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = extract_json_payload(str(result.get("stdout") or "")) if result.get("stdout") else None
    return payload if isinstance(payload, dict) else {}


def _metric_float(raw: Any) -> float | None:
    try:
        if raw is not None:
            return float(raw)
    except (TypeError, ValueError):
        return None
    return None


def _merge_best_metric(metrics: dict[str, Any], key: str, value: Any) -> None:
    parsed = _metric_float(value)
    if parsed is None:
        return
    current = _metric_float(metrics.get(key))
    if current is None or parsed > current:
        metrics[key] = parsed


def _sample_budget(value: Any) -> str:
    parsed = _metric_float(value)
    if parsed is None:
        return "unknown"
    if parsed <= 2048:
        return "small"
    if parsed < 10000:
        return "medium"
    return "full"


def _strategy_feature_key(record: dict[str, Any]) -> str:
    request = record.get("selected_request") if isinstance(record.get("selected_request"), dict) else {}
    return "|".join(
        [
            str(record.get("workflow_module") or "unknown"),
            str(request.get("trainable") or "none"),
            _sample_budget(request.get("max_samples")),
            str(request.get("rollout_steps") or "none"),
        ]
    )


def summarize_strategy_results(results: list[dict[str, Any]]) -> tuple[str, dict[str, Any], dict[str, Any], str | None]:
    metrics: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    errors: list[str] = []
    planned_only = True
    elapsed_total = 0.0
    elapsed_seen = False

    for result in results:
        if not isinstance(result, dict):
            continue
        if result.get("status") not in {"planned", "recorded", "deferred"}:
            planned_only = False
        payload = _stdout_payload(result)
        result_metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        payload_metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        for source in (result_metrics, payload_metrics):
            for key, value in source.items():
                if _metric_float(value) is not None:
                    _merge_best_metric(metrics, key, value)

        _merge_best_metric(metrics, "best_score", payload.get("best_score"))
        if "competition_score_proxy" not in metrics and metrics.get("best_score") is not None:
            metrics["competition_score_proxy"] = metrics["best_score"]

        elapsed = _metric_float(payload.get("elapsed_seconds") or result.get("elapsed_seconds"))
        if elapsed is not None:
            elapsed_total += elapsed
            elapsed_seen = True

        run_dir = result.get("run_dir") or payload.get("run_dir")
        prediction = result.get("prediction_path") or payload.get("prediction_path")
        checkpoint = payload.get("best_checkpoint")
        if run_dir:
            artifacts["run_dir"] = relative_to_root(resolve_any_path(run_dir))
        if prediction:
            artifacts["prediction"] = relative_to_root(resolve_any_path(prediction))
        if checkpoint:
            artifacts["checkpoint"] = relative_to_root(resolve_any_path(checkpoint))

        if result.get("returncode") not in (None, 0):
            errors.append(f"returncode={result.get('returncode')} command={_command_text(result.get('command'))}")
        if result.get("status") in {"failed", "rejected"}:
            errors.append(str(result.get("error") or result.get("reason") or result.get("status")))
        retry = result.get("retry")
        if isinstance(retry, dict) and retry.get("returncode") not in (None, 0):
            errors.append(f"retry_returncode={retry.get('returncode')}")

    if elapsed_seen:
        metrics["elapsed_seconds"] = elapsed_total
    if "combined_score" not in metrics:
        for key in ("competition_score_proxy", "best_score", "score"):
            if metrics.get(key) is not None:
                metrics["combined_score"] = metrics[key]
                break

    status = "planned" if planned_only and not metrics and not artifacts and not errors else ("failed" if errors else "evaluated")
    return status, metrics, artifacts, "; ".join(errors) if errors else None


def update_strategy_db_from_results(
    state: dict[str, Any],
    *,
    execution_id: str,
    results: list[dict[str, Any]],
    summary_payload: dict[str, Any] | None,
) -> None:
    if not isinstance(summary_payload, dict):
        return
    candidates = summary_payload.get("strategy_candidates")
    selected_local_id = str(summary_payload.get("selected_strategy_id") or "").strip()
    if not isinstance(candidates, list) or not selected_local_id:
        return

    db = strategy_db(state)
    programs = db.setdefault("programs", {})
    parent_id = db.get("active_parent_id")
    parent = programs.get(parent_id) if parent_id else None
    generation = int(parent.get("generation", -1)) + 1 if isinstance(parent, dict) else 0
    island = int(parent.get("island", db.get("current_island", 0))) if isinstance(parent, dict) else int(db.get("current_island", 0))
    island_key = str(island % int(db.get("num_islands") or 3))
    tool_requests = summary_payload.get("tool_requests") if isinstance(summary_payload.get("tool_requests"), list) else []
    selected_request = next((item for item in tool_requests if isinstance(item, dict) and str(item.get("tool") or "") == "finetune_local"), {})

    selected_strategy_id = f"{execution_id}::{selected_local_id}"
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        local_id = str(candidate.get("id") or "").strip()
        if not local_id:
            continue
        strategy_id = f"{execution_id}::{local_id}"
        record = programs.get(strategy_id) if isinstance(programs.get(strategy_id), dict) else {}
        record.update(
            {
                "id": strategy_id,
                "local_id": local_id,
                "run_id": execution_id,
                "parent_id": parent_id,
                "generation": generation,
                "timestamp": utc_now_iso(),
                "route": str(candidate.get("route") or ""),
                "hypothesis": str(candidate.get("hypothesis") or ""),
                "expected_gain": str(candidate.get("expected_gain") or ""),
                "risk": str(candidate.get("risk") or ""),
                "estimated_cost": str(candidate.get("estimated_cost") or ""),
                "execute_now": bool(candidate.get("execute_now")),
                "selected": local_id == selected_local_id,
                "workflow_module": (summary_payload.get("experiment_plan") or {}).get("workflow_module")
                if isinstance(summary_payload.get("experiment_plan"), dict)
                else None,
                "stage": (summary_payload.get("experiment_plan") or {}).get("stage")
                if isinstance(summary_payload.get("experiment_plan"), dict)
                else None,
                "island": island,
            }
        )
        if local_id == selected_local_id:
            status, metrics, artifacts, error = summarize_strategy_results(results)
            record.update(
                {
                    "status": status,
                    "metrics": metrics,
                    "artifacts": artifacts,
                    "error": error,
                    "selected_request": selected_request,
                    "tool_results": [_compact_tool_result(item) for item in results if isinstance(item, dict)],
                    "updated_at": utc_now_iso(),
                }
            )
        else:
            record.setdefault("status", "planned")
        programs[strategy_id] = record

    island_ids = db.setdefault("islands", {}).setdefault(island_key, [])
    if selected_strategy_id not in island_ids:
        island_ids.append(selected_strategy_id)
    selected = programs.get(selected_strategy_id)
    if isinstance(selected, dict):
        feature_key = _strategy_feature_key(selected)
        feature_map = db.setdefault("feature_map", {})
        incumbent_id = feature_map.get(feature_key)
        selected_score = strategy_score(selected)
        if selected_score > float("-inf"):
            if not incumbent_id or selected_score >= strategy_score(programs.get(incumbent_id)):
                feature_map[feature_key] = selected_strategy_id
            best_id = db.get("best_strategy_id")
            if not best_id or selected_score >= strategy_score(programs.get(best_id)):
                db["best_strategy_id"] = selected_strategy_id
            evaluated_ids = [
                strategy_id
                for strategy_id, program in programs.items()
                if isinstance(program, dict) and program.get("status") == "evaluated" and strategy_score(program) > float("-inf")
            ]
            evaluated_ids.sort(key=lambda item: strategy_score(programs.get(item)), reverse=True)
            db["archive"] = evaluated_ids[:20]

    state["strategy_db"] = db


def update_state_from_results(
    *,
    execution_id: str,
    source: str,
    results: list[dict[str, Any]],
    summary_payload: dict[str, Any] | None,
) -> None:
    path = experiment_state_path()
    if path is None:
        return
    state = load_state()
    if not state:
        return

    artifacts = list(state.get("local_artifacts") or [])
    best = state.get("best_local_candidate")
    for item in results:
        if not isinstance(item, dict):
            continue
        artifact: dict[str, Any] = {
            "tool": item.get("tool"),
            "status": item.get("status"),
            "recorded_at": utc_now_iso(),
        }
        stdout_payload = extract_json_payload(str(item.get("stdout") or "")) if item.get("stdout") else None
        run_dir = item.get("run_dir") or (stdout_payload or {}).get("run_dir")
        prediction = item.get("prediction_path") or (stdout_payload or {}).get("prediction_path")
        checkpoint = (stdout_payload or {}).get("best_checkpoint")
        metrics = item.get("metrics") or (stdout_payload or {}).get("metrics") or {}
        score = (stdout_payload or {}).get("best_score")
        if isinstance(metrics, dict):
            score = score if score is not None else metrics.get("competition_score_proxy")
        if run_dir:
            artifact["run_dir"] = relative_to_root(resolve_any_path(run_dir))
        if prediction:
            artifact["prediction"] = relative_to_root(resolve_any_path(prediction))
        if checkpoint:
            artifact["checkpoint"] = relative_to_root(resolve_any_path(checkpoint))
        if metrics:
            artifact["metrics"] = metrics
        if score is not None:
            artifact["score"] = score
            candidate = {
                "source": "current_experiment",
                "tool": item.get("tool"),
                "score": score,
                "run_dir": artifact.get("run_dir"),
                "checkpoint": artifact.get("checkpoint"),
                "prediction": artifact.get("prediction"),
                "recorded_at": utc_now_iso(),
            }
            try:
                if best is None or float(score) >= float(best.get("score", float("-inf"))):
                    best = candidate
            except Exception:
                best = candidate
        if any(key in artifact for key in ("run_dir", "prediction", "checkpoint", "metrics", "score")):
            artifacts.append(artifact)

    state["local_artifacts"] = artifacts[-30:]
    state["best_local_candidate"] = best
    state["last_executor"] = {
        "execution_id": execution_id,
        "source": source,
        "summary_run_id": (summary_payload or {}).get("run_id"),
        "updated_at": utc_now_iso(),
    }
    if best:
        blockers = [item for item in state.get("blockers", []) if item != "need_current_experiment_candidate"]
        state["blockers"] = blockers
    update_strategy_db_from_results(
        state,
        execution_id=execution_id,
        results=results,
        summary_payload=summary_payload,
    )
    state["updated_at"] = utc_now_iso()
    write_json(path, state)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _json_call(name: str, payload: dict[str, Any]) -> str:
    return f"{name}({json.dumps(payload, ensure_ascii=False, separators=(',', ':'))})"


def _command_text(command: Any) -> str:
    if isinstance(command, list):
        return " ".join(str(item) for item in command)
    return str(command or "")


def _compact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "tool": result.get("tool"),
        "status": result.get("status"),
        "returncode": result.get("returncode"),
        "run_dir": result.get("run_dir"),
        "prediction_path": result.get("prediction_path"),
        "promoted_to": result.get("promoted_to"),
        "error": result.get("error") or result.get("reason"),
    }
    command = result.get("command")
    if command:
        compact["command"] = _command_text(command)
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        compact["metrics"] = metrics
    retry = result.get("retry")
    if isinstance(retry, dict):
        compact["retry"] = _compact_tool_result(retry)
    return {key: value for key, value in compact.items() if value not in (None, "", [])}


def append_executor_agent_log(
    agent_log: Path | None,
    *,
    execution_id: str,
    source: str,
    dry_run: bool,
    requests: list[Any],
    results: list[dict[str, Any]],
    elapsed_seconds: float,
) -> None:
    if agent_log is None:
        return
    tool_calls: list[str] = []
    for index, request in enumerate(requests):
        request_payload = request if isinstance(request, dict) else {"request": request}
        result = results[index] if index < len(results) and isinstance(results[index], dict) else {}
        command = result.get("command") if isinstance(result, dict) else None
        if command:
            tool_calls.append(_json_call("bash", {"command": _command_text(command)}))
        else:
            tool_name = str(request_payload.get("tool") or "tool_request")
            tool_calls.append(_json_call(tool_name, request_payload))

    response = {
        "execution_id": execution_id,
        "source": source,
        "dry_run": bool(dry_run),
        "results": [_compact_tool_result(item) for item in results if isinstance(item, dict)],
    }
    record: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "response": json.dumps(response, ensure_ascii=False, separators=(",", ":")),
    }
    if tool_calls:
        record["tool_calls"] = "\n".join(tool_calls)
    append_jsonl(agent_log, record)


def normalize_execution_id(raw: Any | None = None) -> str:
    text = str(raw or "").strip()
    if re.fullmatch(r"\d{8}T\d{6}\d{6}Z", text):
        return f"agent_{text}"
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", text):
        return text
    return f"agent_{utc_stamp()}"


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件，空文件或非对象会直接报错。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 必须是 JSON object")
    return payload


def extract_json_payload(text: str) -> dict[str, Any] | None:
    """Extract the last JSON object from mixed command stdout."""

    decoder = json.JSONDecoder()
    best: tuple[int, dict[str, Any]] | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and (best is None or index + end > best[0]):
            best = (index + end, payload)
    return best[1] if best else None


def mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def latest_agent_summary() -> Path:
    """寻找最近一次 Agent runner 的 summary.json。"""

    candidates = sorted(
        list((ROOT / "agent_workspace" / "logs").glob("agent_*/summary.json"))
        + list((ROOT / "agent_workspace" / "experiments").glob("*/logs/turn_*/summary.json")),
        key=lambda item: item.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("没有找到 Agent summary.json")
    return candidates[-1]


def run_command(command: list[str], *, timeout: int) -> dict[str, Any]:
    """执行白名单命令，并捕获 stdout/stderr。

    executor 不接受任意 shell 字符串，只运行本文件构造出来的参数列表。
    """

    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


@contextmanager
def time_limit(seconds: int):
    """Apply executor timeout to in-process Python tools."""

    if seconds <= 0:
        yield
        return

    def handle_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"tool exceeded timeout={seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _relative_tool_output(*parts: str) -> Path:
    return tool_output_root() / Path(*parts)


def hdf5_shape_summary(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    datasets: dict[str, Any] = {}
    with h5py.File(resolved, "r") as h5:
        for key in h5.keys():
            value = h5[key]
            shape = list(value.shape) if hasattr(value, "shape") else None
            dtype = str(value.dtype) if hasattr(value, "dtype") else None
            datasets[key] = {"shape": shape, "dtype": dtype}
    return {"path": str(resolved), "datasets": datasets}


def data_shape_check(request: dict[str, Any], *, execution_id: str, index: int) -> dict[str, Any]:
    """检查 Task1 关键 HDF5 数据形状，并落盘为可追踪 tool output。"""

    raw_paths = request.get("paths")
    if raw_paths is None:
        paths = [
            "data/task1_test.hdf5",
            "data/task1_val.hdf5",
            "data/pdebench_burgers/raw/1D_Burgers_Sols_Nu0.001.hdf5",
        ]
    elif isinstance(raw_paths, list):
        paths = [str(item) for item in raw_paths]
    else:
        paths = [str(raw_paths)]
    summaries = [hdf5_shape_summary(Path(item)) for item in paths]
    output = _relative_tool_output("data_shape", f"{execution_id}__tool{index:02d}__data_shape.json")
    payload = {
        "tool": "data_shape_check",
        "status": "completed",
        "summaries": summaries,
        "task1_required_output": {"dataset_key": "tensor", "shape": [1000, 200, 256], "copied_initial_frames": 10},
        "official_scale_alignment": {
            "reduced_resolution_t": 5,
            "reduced_resolution": 4,
            "raw_spatial_size": 1024,
            "reduced_spatial_size": 256,
            "first_observed_raw_indices": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
            "first_supervised_target_raw_index": 50,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"tool": "data_shape_check", "status": "completed", "output": str(output), "summaries": summaries}


def parse_budget_int(raw_budget: Any, key: str, default: int) -> int:
    """从 `budget` 字符串里解析整数，例如 `max_trials=8 timeout=2h`。

    Agent 输出的预算是文本，为了不执行任意表达式，这里只支持简单的
    `key=整数` 模式。
    """

    text = str(raw_budget or "")
    match = re.search(rf"\b{re.escape(key)}\s*=\s*(\d+)\b", text)
    return int(match.group(1)) if match else default


def parse_budget_float(raw_budget: Any, key: str, default: float) -> float:
    """从 `budget` 字符串里解析浮点数。"""

    text = str(raw_budget or "")
    match = re.search(rf"\b{re.escape(key)}\s*=\s*([0-9]*\.?[0-9]+)\b", text)
    return float(match.group(1)) if match else default


def extract_duckduckgo_url(raw_url: str) -> str:
    """把 DuckDuckGo redirect URL 还原成真实 URL。"""

    parsed = urlparse(raw_url)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    return raw_url


def web_search(request: dict[str, Any], *, timeout: int, execution_id: str, index: int) -> dict[str, Any]:
    """执行轻量 web search，并把摘要写入 executor log 目录。

    这里不用浏览器，也不依赖额外搜索 SDK。结果只作为 Agent 下一轮决策的
    外部证据入口，不能替代本地 validation。
    """

    query = str(request.get("query") or request.get("purpose") or "").strip()
    if not query:
        raise ValueError("web_search 请求必须提供 query 或 purpose")
    top_k = max(1, min(parse_budget_int(request.get("budget"), "top_k", 5), 10))
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    http_request = Request(url, headers={"User-Agent": "Mozilla/5.0 Task1AgentExecutor/1.0"})
    with urlopen(http_request, timeout=min(timeout, 30)) as response:
        body = response.read().decode("utf-8", errors="replace")

    pattern = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
    results: list[dict[str, str]] = []
    for raw_href, raw_title in pattern.findall(body):
        title = re.sub(r"<.*?>", "", raw_title)
        title = html.unescape(title).strip()
        href = html.unescape(raw_href)
        results.append({"title": title, "url": extract_duckduckgo_url(href)})
        if len(results) >= top_k:
            break

    output_dir = tool_output_root() / "web_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{execution_id}__tool{index:02d}__web_search.json"
    payload = {
        "tool": "web_search",
        "query": query,
        "top_k": top_k,
        "results": results,
        "note": "检索结果只供 Agent 参考，最终方案必须通过本地 validation。",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"tool": "web_search", "status": "completed", "output": str(output), "results": results}


def replay_command(request: dict[str, Any], index: int, *, execution_id: str) -> list[str]:
    """把 checkpoint_replay 请求转换成受控 workflow 命令。"""

    target = str(request.get("target") or request.get("route") or request.get("name") or "official_fno")
    config = CONFIG_BY_REPLAY_TARGET.get(target) or CONFIG_BY_REPLAY_TARGET.get(Path(target).name)
    if not config:
        raise ValueError(f"不支持的 checkpoint_replay target: {target}")
    split = str(request.get("split") or "val")
    if split not in {"val", "test"}:
        raise ValueError(f"checkpoint_replay split 只能是 val/test: {split}")
    hypothesis = str(request.get("purpose") or f"Replay {target} checkpoint baseline.")
    command = [
        "bash",
        "scripts/run_in_env.sh",
        "scripts/task1_run_workflow.py",
        "--config",
        config,
        "--split",
        split,
        "--run-name",
        f"{execution_id}__tool{index:02d}",
        "--hypothesis",
        hypothesis,
        "--decision",
        "baseline" if split == "test" else "keep",
        "--tag",
        target,
        "--tag",
        split,
    ]
    if request.get("promote_slot"):
        command.extend(["--promote-slot", str(request["promote_slot"])])
    return command


def finetuned_checkpoint_replay(request: dict[str, Any], index: int, *, execution_id: str, timeout: int) -> dict[str, Any]:
    """用 finetune_local 产出的 FNO checkpoint 做标准 val/test replay。"""

    checkpoint = request.get("checkpoint") or request.get("checkpoint_path")
    if not checkpoint:
        run_dir = request.get("run_dir")
        if run_dir:
            require_current_experiment_artifact(run_dir, purpose="finetuned_checkpoint_replay.run_dir")
            checkpoint = str(Path(str(run_dir)) / "best.pt")
        else:
            for previous in range(index - 1, -1, -1):
                candidate = experiment_runs_root() / f"{execution_id}__tool{previous:02d}" / "best.pt"
                if candidate.is_file():
                    checkpoint = str(candidate)
                    break
        if not checkpoint:
            raise ValueError(
                "finetuned_checkpoint_replay 需要 checkpoint/checkpoint_path/run_dir，"
                "或在同一轮较早 tool 中先成功执行 finetune_local。"
            )
    require_current_experiment_artifact(checkpoint, purpose="finetuned_checkpoint_replay.checkpoint")
    split = str(request.get("split") or "val")
    if split not in {"val", "test"}:
        raise ValueError(f"finetuned_checkpoint_replay split 只能是 val/test: {split}")
    run_name = request.get("run_name")
    if not run_name and request.get("run_dir"):
        run_name = Path(str(request["run_dir"])).name
    run_name = str(run_name or f"{execution_id}__tool{index:02d}")
    config = {
        "route": "finetune_fno",
        "_config_path": "finetune_local_tool",
        "device": request.get("device") or "auto",
        "batch_size": int(request.get("batch_size") or 50),
        "val_input_path": str(request.get("val_hdf5") or "data/task1_val.hdf5"),
        "input_path": str(request.get("input_hdf5") or "data/task1_test.hdf5"),
        "full_t_path": str(request.get("full_t_path") or "data/task1_val.hdf5"),
        "train_time": float(request.get("train_time") or 0.0),
        "models": [{"kind": "fno", "checkpoint": str(checkpoint), "weight": 1.0}],
    }
    with time_limit(timeout):
        result = run_prediction(config, split=split, run_name=run_name)
    decision = str(request.get("decision") or ("promote_candidate" if split == "val" else "baseline"))
    hypothesis = str(request.get("purpose") or f"Replay finetuned FNO checkpoint on Task1 {split} split.")
    tags = ["finetune_fno", split, "scale_alignment"]
    if request.get("tag"):
        raw_tags = request["tag"] if isinstance(request["tag"], list) else [request["tag"]]
        tags.extend(str(item) for item in raw_tags)
    memory_export = export_run_memory(result["run_dir"], hypothesis=hypothesis, decision=decision, tags=tags)
    registry = add_record_to_registry(memory_export)
    promoted = None
    metrics = result.get("metrics") or {}
    promote_slot = request.get("promote_slot")
    if promote_slot is None and split == "val":
        promote_slot = "best_finetuned_standard_val_replay"
    metric = str(request.get("metric") or "competition_score_proxy")
    if promote_slot and metrics and metric in metrics:
        promoted = promote_candidate(
            f"task1:{result['run_name']}",
            slot=str(promote_slot),
            metric=metric,
            value=float(metrics[metric]),
            blockers=["需要 test replay、合规 LLM log 和 submission_package 复核"] if split == "val" else [],
        )
    return {
        "tool": "finetuned_checkpoint_replay",
        "status": "completed",
        "run_dir": result["run_dir"],
        "prediction_path": result["prediction_path"],
        "metrics": result.get("metrics"),
        "validation": result["validation"],
        "memory_export": str(memory_export),
        "memory_registry": str(registry),
        "promoted_to": str(promoted) if promoted else None,
    }


def latest_prediction_from_results(results: list[dict[str, Any]]) -> str | None:
    for result in reversed(results):
        prediction = result.get("prediction_path")
        if prediction:
            return str(prediction)
        retry = result.get("retry")
        if isinstance(retry, dict) and retry.get("prediction_path"):
            return str(retry["prediction_path"])
    return None


def execute_finetune_local(request: dict[str, Any], index: int, *, execution_id: str, timeout: int) -> dict[str, Any]:
    """Run controlled fine-tune and always register the checkpoint probe result.

    A fine-tune run can discover a strong checkpoint even when the Agent forgets
    to request `finetuned_checkpoint_replay` in the same round. Recording the
    probe result keeps later Agent rounds from reasoning against a stale
    leaderboard while still marking replay/submission as blockers.
    """

    result = run_finetune_local_raw(request, index, execution_id=execution_id, timeout=timeout)
    return finalize_finetune_result(request, result)


def run_finetune_local_raw(request: dict[str, Any], index: int, *, execution_id: str, timeout: int) -> dict[str, Any]:
    return run_command(finetune_local_command(request, index, execution_id=execution_id), timeout=timeout)


def finalize_finetune_result(request: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    if result.get("returncode") != 0:
        return result

    payload = extract_json_payload(str(result.get("stdout") or ""))
    if not payload:
        result["memory_status"] = "skipped_no_json_stdout"
        return result

    run_dir = payload.get("run_dir")
    best_score = payload.get("best_score")
    if not run_dir:
        result["memory_status"] = "skipped_no_run_dir"
        return result

    hypothesis = str(request.get("purpose") or "Controlled local FNO checkpoint fine-tune probe.")
    tags = ["finetune_fno", "checkpoint_probe", "scale_alignment"]
    if request.get("tag"):
        raw_tags = request["tag"] if isinstance(request["tag"], list) else [request["tag"]]
        tags.extend(str(item) for item in raw_tags)

    memory_export = export_run_memory(run_dir, hypothesis=hypothesis, decision="promote_candidate", tags=tags)
    registry = add_record_to_registry(memory_export)
    result["memory_export"] = str(memory_export)
    result["memory_registry"] = str(registry)

    promoted = None
    if best_score is not None:
        promoted = promote_candidate(
            f"task1:{Path(str(run_dir)).name}",
            slot="best_finetune_checkpoint_probe",
            metric=str(request.get("metric") or "competition_score_proxy"),
            value=float(best_score),
            blockers=[
                "需要 finetuned_checkpoint_replay 标准验证",
                "需要 test replay、合规 LLM log 和 submission_package 复核",
            ],
        )
    result["promoted_to"] = str(promoted) if promoted else None
    return result


def execute_finetune_group(
    group: list[tuple[int, dict[str, Any]]],
    *,
    timeout: int,
    execution_id: str,
    max_workers: int,
) -> list[dict[str, Any]]:
    """Run independent finetune_local trials concurrently, then register memory sequentially."""

    if len(group) == 1 or max_workers <= 1:
        return [
            execute_finetune_local(request, index, execution_id=execution_id, timeout=timeout)
            for index, request in group
        ]

    raw_results: dict[int, dict[str, Any]] = {}
    workers = min(max_workers, len(group))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_finetune_local_raw, request, index, execution_id=execution_id, timeout=timeout): (index, request)
            for index, request in group
        }
        for future in as_completed(futures):
            index, request = futures[future]
            try:
                raw_results[index] = future.result()
            except Exception as exc:
                raw_results[index] = {
                    "tool": "finetune_local",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "request": request,
                }

    results: list[dict[str, Any]] = []
    for index, request in group:
        result = raw_results[index]
        if result.get("status") == "failed":
            results.append(result)
        else:
            results.append(finalize_finetune_result(request, result))
    return results


def memory_query_command(request: dict[str, Any]) -> list[str]:
    """构造 memory_query 命令。"""

    command = ["bash", "scripts/run_in_env.sh", "scripts/memory_query.py"]
    if request.get("route"):
        command.extend(["--route", str(request["route"])])
    if request.get("max_chars"):
        command.extend(["--max-chars", str(request["max_chars"])])
    return command


def validation_command(request: dict[str, Any]) -> list[str]:
    """构造 HDF5 validation 命令。"""

    prediction = request.get("prediction")
    if not prediction:
        raise ValueError("validation 请求必须提供 prediction")
    require_current_experiment_artifact(prediction, purpose="validation.prediction")
    command = [
        "bash",
        "scripts/run_in_env.sh",
        "scripts/task1_validate.py",
        "--prediction",
        str(prediction),
        "--input",
        str(request.get("input") or "data/task1_val.hdf5"),
    ]
    target_hdf5 = request.get("target_hdf5")
    input_hdf5 = str(request.get("input") or "data/task1_val.hdf5")
    if target_hdf5 and "task1_test" not in str(target_hdf5) and "task1_test" not in input_hdf5:
        command.extend(["--target", str(target_hdf5)])
    if request.get("output"):
        command.extend(["--output", str(request["output"])])
    return command


def llm_log_prepare_command(
    request: dict[str, Any],
    index: int,
    *,
    execution_id: str,
    summary_payload: dict[str, Any] | None = None,
    fallback_start_timestamp: str | None = None,
    fallback_end_timestamp: str | None = None,
) -> list[str]:
    """构造日志整理命令。

    优先使用当前实验的 per-experiment task1_logs.log；只有旧代理模式
    没有显式日志时才回退到 logs/openai_proxy_*.jsonl。
    """

    raw_inputs = request.get("input_logs")
    if raw_inputs is None and request.get("llm_log"):
        raw_inputs = [request["llm_log"]]
    if raw_inputs is None and isinstance(summary_payload, dict) and summary_payload.get("agent_log"):
        raw_inputs = [summary_payload["agent_log"]]
    if raw_inputs is None:
        candidates = sorted((ROOT / "logs").glob("openai_proxy_*.jsonl"))
        if not candidates:
            raise FileNotFoundError("没有找到 logs/openai_proxy_*.jsonl")
        input_logs = [candidates[-1]]
    elif isinstance(raw_inputs, list):
        input_logs = [Path(str(item)) for item in raw_inputs]
    else:
        input_logs = [Path(str(raw_inputs))]
    output = request.get("output") or _relative_tool_output("llm_logs", f"{execution_id}__tool{index:02d}__task1_logs.log")
    command = ["bash", "scripts/run_in_env.sh", "scripts/task1_prepare_llm_log.py"]
    for item in input_logs:
        command.extend(["--input", str(item)])
    command.extend(["--output", str(output)])
    start_timestamp = request.get("start_timestamp") or fallback_start_timestamp
    end_timestamp = request.get("end_timestamp") or fallback_end_timestamp
    if (not start_timestamp or not end_timestamp) and summary_payload:
        runner_log_dir = summary_payload.get("runner_log_dir")
        if runner_log_dir:
            log_dir = Path(str(runner_log_dir))
            request_context = log_dir / "request_context.json"
            summary_json = log_dir / "summary.json"
            if not start_timestamp and request_context.is_file():
                start_timestamp = mtime_iso(request_context)
            if not end_timestamp and summary_json.is_file():
                end_timestamp = mtime_iso(summary_json)
    if start_timestamp:
        command.extend(["--start-timestamp", str(start_timestamp)])
    if end_timestamp:
        command.extend(["--end-timestamp", str(end_timestamp)])
    return command


def _run_metadata(run_dir: Any) -> dict[str, Any]:
    path = require_current_experiment_artifact(run_dir, purpose="submission_package.run_dir")
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _single_checkpoint_from_metadata(metadata: dict[str, Any]) -> str | None:
    checkpoints = metadata.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        return None
    first = checkpoints[0]
    if not isinstance(first, dict):
        return None
    path = first.get("path")
    return str(path) if path else None


def execute_submission_package(
    request: dict[str, Any],
    index: int,
    *,
    execution_id: str,
    timeout: int,
    agent_log_path: str | None = None,
    summary_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build submission; auto-create test replay if Agent passed a val run."""

    run_dir = request.get("run_dir")
    if not run_dir:
        return {"tool": "submission_package", "status": "failed", "error": "submission_package 请求必须提供 run_dir"}
    metadata = _run_metadata(run_dir)
    split = str(metadata.get("split") or "")
    effective_request = dict(request)
    test_replay_result: dict[str, Any] | None = None
    if split and split != "test":
        checkpoint = _single_checkpoint_from_metadata(metadata)
        if not checkpoint:
            return {
                "tool": "submission_package",
                "status": "failed",
                "error": "submission_package 不能使用 val run_dir；请先运行 split=test 的 replay，再用 test run_dir 打包。",
                "run_dir": str(run_dir),
                "detected_split": split,
            }
        replay_request = {
            "tool": "finetuned_checkpoint_replay",
            "purpose": "submission_package 自动补跑 test replay；提交包必须使用 1000 条 test prediction。",
            "checkpoint": checkpoint,
            "split": "test",
            "run_name": f"{execution_id}__tool{index:02d}__test_replay",
            "decision": "submission_test_replay",
        }
        test_replay_result = finetuned_checkpoint_replay(
            replay_request,
            index,
            execution_id=execution_id,
            timeout=timeout,
        )
        if test_replay_result.get("status") != "completed":
            return {
                "tool": "submission_package",
                "status": "failed",
                "error": "自动 test replay 失败，无法打包 submission。",
                "detected_split": split,
                "test_replay": test_replay_result,
            }
        effective_request["run_dir"] = test_replay_result["run_dir"]
    command = submission_package_command(
        effective_request,
        index,
        execution_id=execution_id,
        agent_log_path=agent_log_path,
        summary_payload=summary_payload,
    )
    package_result = run_command(command, timeout=timeout)
    if test_replay_result is None:
        return package_result
    return {
        "tool": "submission_package",
        "status": "completed" if package_result.get("returncode") == 0 else "failed",
        "auto_test_replay": test_replay_result,
        "package": package_result,
    }


def submission_package_command(
    request: dict[str, Any],
    index: int,
    *,
    execution_id: str,
    agent_log_path: str | None = None,
    summary_payload: dict[str, Any] | None = None,
) -> list[str]:
    """构造 Task1 submission 打包命令。"""

    run_dir = request.get("run_dir")
    if not run_dir:
        raise ValueError("submission_package 请求必须提供 run_dir")
    require_current_experiment_artifact(run_dir, purpose="submission_package.run_dir")
    if os.environ.get("AI4S_TASK1_SUBMISSIONS_ROOT"):
        submission_name = "submission"
    else:
        submission_name = str(request.get("submission_name") or f"{execution_id}__tool{index:02d}_submission")
    command = [
        "bash",
        "scripts/run_in_env.sh",
        "scripts/task1_make_submission.py",
        "--run-dir",
        str(run_dir),
        "--submission-name",
        submission_name,
    ]
    if request.get("submission_id"):
        command.extend(["--submission-id", str(request["submission_id"])])
    code_dir = request.get("code_dir")
    if not code_dir and isinstance(summary_payload, dict):
        code_dir = summary_payload.get("generated_code_root")
    if code_dir:
        require_current_experiment_artifact(code_dir, purpose="submission_package.code_dir")
        command.extend(["--code-dir", str(code_dir)])
    llm_log = request.get("llm_log") or agent_log_path
    if llm_log:
        require_current_experiment_artifact(llm_log, purpose="submission_package.llm_log")
        command.extend(["--llm-log", str(llm_log)])
    return command


def _bounded_int(request: dict[str, Any], key: str, default: int, *, minimum: int, maximum: int) -> int:
    value = int(request.get(key, default))
    if value < minimum or value > maximum:
        raise ValueError(f"{key} must be in [{minimum}, {maximum}], got {value}")
    return value


def _capped_int(request: dict[str, Any], key: str, default: int, *, minimum: int, maximum: int) -> int:
    value = int(request.get(key, default))
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value}")
    return min(value, maximum)


def _bounded_float(request: dict[str, Any], key: str, default: float, *, minimum: float, maximum: float) -> float:
    value = float(request.get(key, default))
    if value < minimum or value > maximum:
        raise ValueError(f"{key} must be in [{minimum}, {maximum}], got {value}")
    return value


def _choice_str(request: dict[str, Any], key: str, default: str, allowed: set[str]) -> str:
    raw = request.get(key, default)
    if isinstance(raw, bool):
        value = "true" if raw else "false"
    else:
        value = str(raw)
    if value not in allowed:
        raise ValueError(f"{key} must be one of {sorted(allowed)}, got {value}")
    return value


def _safe_run_dir(raw: Any, index: int, *, execution_id: str) -> str:
    del raw
    path = experiment_runs_root() / f"{execution_id}__tool{index:02d}"
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe finetune_local run_dir: {path}")
    return relative_to_root(path)


def _trainable_modules(request: dict[str, Any]) -> list[str]:
    raw = request.get("trainable_modules")
    if raw in (None, ""):
        return []
    if isinstance(raw, str):
        candidates = raw.split(",")
    elif isinstance(raw, list):
        candidates = []
        for item in raw:
            candidates.extend(str(item).split(","))
    else:
        raise ValueError("finetune_local trainable_modules must be a list or comma-separated string")
    modules = [item.strip() for item in candidates if item.strip()]
    unknown = sorted(set(modules) - FNO_TRAINABLE_MODULES)
    if unknown:
        raise ValueError(f"unsupported trainable_modules: {unknown}; allowed={sorted(FNO_TRAINABLE_MODULES)}")
    return list(dict.fromkeys(modules))


def finetune_local_command(request: dict[str, Any], index: int, *, execution_id: str) -> list[str]:
    """把 finetune_local 请求转换成受控微调命令。

    这个工具故意不暴露 temporal_stride / spatial_downsample 给 Agent 修改；
    executor 固定传入 5 和 4，保证官方 checkpoint 的 reduced-scale 对齐。
    """

    modules = _trainable_modules(request)
    trainable = str(request.get("trainable") or ("custom" if modules else "last-block-head"))
    if trainable not in {"head", "last-block-head", "all", "custom"}:
        raise ValueError(f"unsupported trainable mode: {trainable}")
    if trainable == "custom" and not modules:
        raise ValueError("custom trainable mode requires trainable_modules")
    device = request.get("device")
    base_checkpoint = request.get("base_checkpoint") or request.get("checkpoint_path") or request.get("checkpoint")
    if not base_checkpoint:
        base_checkpoint = "checkpoints/official/nu0.001_fno.pt"
    if "unet" in Path(str(base_checkpoint)).name.lower():
        raise ValueError(
            "finetune_local 当前只支持 FNO checkpoint 微调；"
            "不要把 Unet-PF checkpoint 传给该工具。"
        )
    command = [
        "bash",
        "scripts/run_in_env.sh",
        "scripts/task1_finetune_local.py",
        "--run-dir",
        _safe_run_dir(request.get("run_dir"), index, execution_id=execution_id),
        "--train-hdf5",
        str(request.get("train_hdf5") or "data/pdebench_burgers/raw/1D_Burgers_Sols_Nu0.001.hdf5"),
        "--val-hdf5",
        str(request.get("val_hdf5") or "data/task1_val.hdf5"),
        "--base-checkpoint",
        str(base_checkpoint),
        "--steps",
        str(_bounded_int(request, "steps", 300, minimum=1, maximum=20000)),
        "--batch-size",
        str(_bounded_int(request, "batch_size", 8, minimum=1, maximum=128)),
        "--eval-batch-size",
        str(_bounded_int(request, "eval_batch_size", 50, minimum=1, maximum=256)),
        "--lr",
        str(_bounded_float(request, "lr", 2.0e-5, minimum=1.0e-7, maximum=1.0e-3)),
        "--weight-decay",
        str(_bounded_float(request, "weight_decay", 1.0e-4, minimum=0.0, maximum=1.0e-1)),
        "--rollout-steps",
        str(_bounded_int(request, "rollout_steps", 2, minimum=1, maximum=8)),
        "--horizon-gamma",
        str(_bounded_float(request, "horizon_gamma", 0.9, minimum=0.1, maximum=1.0)),
        "--max-samples",
        str(_bounded_int(request, "max_samples", 2048, minimum=1, maximum=10000)),
        "--val-max-samples",
        str(_capped_int(request, "val_max_samples", 100, minimum=1, maximum=100)),
        "--val-every",
        str(_bounded_int(request, "val_every", 100, minimum=1, maximum=5000)),
        "--log-every",
        str(_bounded_int(request, "log_every", 20, minimum=1, maximum=1000)),
        "--num-workers",
        str(_bounded_int(request, "num_workers", 4, minimum=0, maximum=16)),
        "--prefetch-factor",
        str(_bounded_int(request, "prefetch_factor", 2, minimum=1, maximum=16)),
        "--pin-memory",
        _choice_str(request, "pin_memory", "auto", {"auto", "true", "false"}),
        "--persistent-workers",
        _choice_str(request, "persistent_workers", "auto", {"auto", "true", "false"}),
        "--trainable",
        trainable,
        "--temporal-stride",
        "5",
        "--spatial-downsample",
        "4",
    ]
    for module in modules:
        command.extend(["--trainable-module", module])
    if device:
        command.extend(["--device", str(device)])
    return command


def execute_request(
    request: dict[str, Any],
    index: int,
    *,
    timeout: int,
    execution_id: str,
    summary_payload: dict[str, Any] | None = None,
    log_start_timestamp: str | None = None,
    log_end_timestamp: str | None = None,
) -> dict[str, Any]:
    """执行单个 tool_request。

    目前本地 executor 执行确定性白名单工具和轻量 web search。
    Optuna 是 Agent 生成代码可以直接 import 的 Python 库，不在 executor
    中代替 Agent 执行。`ray_tune` 和 `wandb` 暂不执行。
    """

    tool = str(request.get("tool") or "")
    if tool == "checkpoint_replay":
        return run_command(replay_command(request, index, execution_id=execution_id), timeout=timeout)
    if tool == "finetuned_checkpoint_replay":
        return finetuned_checkpoint_replay(request, index, execution_id=execution_id, timeout=timeout)
    if tool == "memory_query":
        return run_command(memory_query_command(request), timeout=timeout)
    if tool == "validation":
        return run_command(validation_command(request), timeout=timeout)
    if tool == "finetune_local":
        return execute_finetune_local(request, index, execution_id=execution_id, timeout=timeout)
    if tool == "data_shape_check":
        return data_shape_check(request, execution_id=execution_id, index=index)
    if tool == "llm_log_prepare":
        return run_command(
            llm_log_prepare_command(
                request,
                index,
                execution_id=execution_id,
                summary_payload=summary_payload,
                fallback_start_timestamp=log_start_timestamp,
                fallback_end_timestamp=log_end_timestamp,
            ),
            timeout=timeout,
        )
    if tool == "submission_package":
        agent_log_path = str(summary_payload.get("agent_log")) if isinstance(summary_payload, dict) and summary_payload.get("agent_log") else None
        return execute_submission_package(
            request,
            index,
            execution_id=execution_id,
            timeout=timeout,
            agent_log_path=agent_log_path,
            summary_payload=summary_payload,
        )
    if tool == "web_search":
        return web_search(request, timeout=timeout, execution_id=execution_id, index=index)
    if tool == "optuna":
        return {
            "tool": tool,
            "status": "recorded",
            "reason": "Optuna 是 Python 库，应由 Agent 在生成代码中直接 import optuna 并实现 study/objective；executor 不代替 Agent 使用该库。",
            "request": request,
        }
    if tool in {"ray_tune", "wandb"}:
        return {
            "tool": tool,
            "status": "deferred",
            "reason": "该工具当前按用户要求暂不执行，只记录请求。",
            "request": request,
        }
    return {"tool": tool, "status": "rejected", "reason": "不在 executor 白名单内", "request": request}


def plan_request(
    request: dict[str, Any],
    index: int,
    *,
    execution_id: str,
    summary_payload: dict[str, Any] | None = None,
    log_start_timestamp: str | None = None,
    log_end_timestamp: str | None = None,
) -> dict[str, Any]:
    """解析单个 tool_request，并返回将要执行的受控命令。"""

    tool = str(request.get("tool") or "")
    if tool == "checkpoint_replay":
        return {"tool": tool, "status": "planned", "command": replay_command(request, index, execution_id=execution_id)}
    if tool == "finetuned_checkpoint_replay":
        checkpoint = request.get("checkpoint") or request.get("checkpoint_path")
        if not checkpoint and request.get("run_dir"):
            checkpoint = str(Path(str(request["run_dir"])) / "best.pt")
        if not checkpoint:
            checkpoint = str(experiment_runs_root() / f"{execution_id}__tool<previous_finetune>" / "best.pt")
        return {
            "tool": tool,
            "status": "planned",
            "checkpoint": checkpoint,
            "split": str(request.get("split") or "val"),
            "run_name": f"{execution_id}__tool{index:02d}",
        }
    if tool == "memory_query":
        return {"tool": tool, "status": "planned", "command": memory_query_command(request)}
    if tool == "validation":
        return {"tool": tool, "status": "planned", "command": validation_command(request)}
    if tool == "finetune_local":
        return {"tool": tool, "status": "planned", "command": finetune_local_command(request, index, execution_id=execution_id)}
    if tool == "data_shape_check":
        paths = request.get("paths") or [
            "data/task1_test.hdf5",
            "data/task1_val.hdf5",
            "data/pdebench_burgers/raw/1D_Burgers_Sols_Nu0.001.hdf5",
        ]
        output = _relative_tool_output("data_shape", f"{execution_id}__tool{index:02d}__data_shape.json")
        return {"tool": tool, "status": "planned", "paths": paths, "output": str(output)}
    if tool == "llm_log_prepare":
        return {
            "tool": tool,
            "status": "planned",
            "command": llm_log_prepare_command(
                request,
                index,
                execution_id=execution_id,
                summary_payload=summary_payload,
                fallback_start_timestamp=log_start_timestamp,
                fallback_end_timestamp=log_end_timestamp,
            ),
        }
    if tool == "submission_package":
        agent_log_path = str(summary_payload.get("agent_log")) if isinstance(summary_payload, dict) and summary_payload.get("agent_log") else None
        return {
            "tool": tool,
            "status": "planned",
            "command": submission_package_command(
                request,
                index,
                execution_id=execution_id,
                agent_log_path=agent_log_path,
                summary_payload=summary_payload,
            ),
        }
    if tool == "web_search":
        query = str(request.get("query") or request.get("purpose") or "").strip()
        if not query:
            raise ValueError("web_search 请求必须提供 query 或 purpose")
        output = tool_output_root() / "web_search" / f"{execution_id}__tool{index:02d}__web_search.json"
        return {"tool": tool, "status": "planned", "query": query, "output": str(output)}
    if tool == "optuna":
        return {
            "tool": tool,
            "status": "recorded",
            "reason": "Optuna 应由 Agent 在生成代码中直接 import 并实现 study/objective；executor 不代替 Agent 执行。",
            "request": request,
        }
    if tool in {"ray_tune", "wandb"}:
        return {
            "tool": tool,
            "status": "deferred",
            "reason": "该工具当前按用户要求暂不执行，只记录请求。",
            "request": request,
        }
    return {"tool": tool, "status": "rejected", "reason": "不在 executor 白名单内", "request": request}


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute whitelisted Task1 Agent tool requests.")
    parser.add_argument("--summary", default=None, help="Agent runner summary.json；默认使用最新一次。")
    parser.add_argument("--requests-json", default=None, help="直接提供 tool_requests JSON 文件。")
    parser.add_argument("--execution-id", default=None, help="统一本次 executor/log/runs 命名；默认取 Agent summary.run_id。")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--log-start-timestamp", default=None, help="llm_log_prepare 默认时间窗起点；loop 会传入整轮 loop 开始时间。")
    parser.add_argument("--log-end-timestamp", default=None, help="llm_log_prepare 默认时间窗终点；不填则使用当前 runner summary 时间。")
    parser.add_argument("--dry-run", action="store_true", help="只解析请求，不执行命令。")
    parser.add_argument("--agent-log", default=None, help="本次实验专属 task1_logs.log；为空时尝试使用 summary.agent_log。")
    parser.add_argument("--runs-root", default=None, help="本次实验 run 产物根目录。")
    parser.add_argument("--tool-output-root", default=None, help="本次实验轻量工具输出根目录。")
    parser.add_argument("--executor-log-dir", default=None, help="本次 executor JSON 日志目录。")
    parser.add_argument("--submission-root", default=None, help="本次实验 submission 根目录的父目录。")
    parser.add_argument("--experiment-dir", default=None, help="本次实验根目录；提供后禁止直接引用旧实验 artifact。")
    parser.add_argument("--state-path", default=None, help="当前实验 state.json，executor 会追加本轮 artifact 摘要。")
    parser.add_argument("--max-parallel-finetunes", type=int, default=4, help="同一批连续 finetune_local 请求的最大并发数。")
    args = parser.parse_args()
    executor_started_at = datetime.now(timezone.utc)

    if args.runs_root:
        os.environ["AI4S_TASK1_RUNS_ROOT"] = relative_to_root(resolve_root_path(args.runs_root, ROOT / "runs" / "task1"))
    if args.tool_output_root:
        os.environ["AI4S_TASK1_TOOL_OUTPUT_ROOT"] = relative_to_root(resolve_root_path(args.tool_output_root, ROOT / "agent_workspace" / "tool_outputs"))
    if args.executor_log_dir:
        os.environ["AI4S_TASK1_EXECUTOR_LOG_ROOT"] = relative_to_root(resolve_root_path(args.executor_log_dir, ROOT / "agent_workspace" / "executor_logs"))
    if args.submission_root:
        os.environ["AI4S_TASK1_SUBMISSIONS_ROOT"] = relative_to_root(resolve_root_path(args.submission_root, ROOT / "submissions"))
    if args.experiment_dir:
        os.environ["AI4S_TASK1_EXPERIMENT_DIR"] = relative_to_root(resolve_root_path(args.experiment_dir, ROOT / "agent_workspace" / "experiments"))
    if args.state_path:
        os.environ["AI4S_TASK1_STATE_PATH"] = relative_to_root(resolve_root_path(args.state_path, ROOT / "memory" / "working" / "task1_state.json"))

    if args.requests_json:
        payload = load_json(Path(args.requests_json))
        requests = payload.get("tool_requests", [])
        source = str(Path(args.requests_json))
        execution_id = normalize_execution_id(args.execution_id)
        agent_log = resolve_log_path(args.agent_log)
    else:
        summary_path = Path(args.summary) if args.summary else latest_agent_summary()
        payload = load_json(summary_path)
        requests = payload.get("tool_requests", [])
        source = str(summary_path)
        execution_id = normalize_execution_id(args.execution_id or payload.get("run_id") or summary_path.parent.name)
        agent_log = resolve_log_path(args.agent_log or payload.get("agent_log"))

    if not isinstance(requests, list):
        raise ValueError("tool_requests 必须是 list")

    if args.max_parallel_finetunes < 1:
        raise ValueError("--max-parallel-finetunes must be >= 1")

    results: list[dict[str, Any]] = []
    index = 0
    while index < len(requests):
        request = requests[index]
        if not isinstance(request, dict):
            results.append({"status": "rejected", "reason": "tool_request 必须是 object", "request": request})
            index += 1
            continue
        try:
            if args.dry_run:
                results.append(
                    plan_request(
                        request,
                        index,
                        execution_id=execution_id,
                        summary_payload=payload,
                        log_start_timestamp=args.log_start_timestamp,
                        log_end_timestamp=args.log_end_timestamp,
                    )
                )
                index += 1
            else:
                if str(request.get("tool") or "") == "finetune_local":
                    group: list[tuple[int, dict[str, Any]]] = []
                    cursor = index
                    while cursor < len(requests):
                        next_request = requests[cursor]
                        if not isinstance(next_request, dict) or str(next_request.get("tool") or "") != "finetune_local":
                            break
                        group.append((cursor, next_request))
                        cursor += 1
                    results.extend(
                        execute_finetune_group(
                            group,
                            timeout=args.timeout,
                            execution_id=execution_id,
                            max_workers=args.max_parallel_finetunes,
                        )
                    )
                    index = cursor
                    continue
                result = execute_request(
                    request,
                    index,
                    timeout=args.timeout,
                    execution_id=execution_id,
                    summary_payload=payload,
                    log_start_timestamp=args.log_start_timestamp,
                    log_end_timestamp=args.log_end_timestamp,
                )
                if (
                    str(request.get("tool") or "") == "validation"
                    and isinstance(result, dict)
                    and result.get("returncode") not in (None, 0)
                ):
                    fallback_prediction = latest_prediction_from_results(results)
                    if fallback_prediction:
                        retry_request = dict(request)
                        retry_request["prediction"] = fallback_prediction
                        retry_request["output"] = str(Path(fallback_prediction).with_name("validation_report.json"))
                        retry = execute_request(
                            retry_request,
                            index,
                            timeout=args.timeout,
                            execution_id=execution_id,
                            summary_payload=payload,
                            log_start_timestamp=args.log_start_timestamp,
                            log_end_timestamp=args.log_end_timestamp,
                        )
                        if isinstance(retry, dict) and retry.get("returncode") in (None, 0):
                            result = {
                                "tool": "validation",
                                "status": "completed_with_fallback_prediction",
                                "fallback_prediction": fallback_prediction,
                                "initial_failure": result,
                                "retry": retry,
                            }
                results.append(result)
                index += 1
        except Exception as exc:
            results.append(
                {
                    "tool": str(request.get("tool") or ""),
                    "status": "rejected" if args.dry_run else "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "request": request,
                }
            )
            index += 1

    log_dir = executor_log_root()
    log_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{execution_id}__dryrun_{utc_stamp()}.json" if args.dry_run else f"{execution_id}.json"
    output = log_dir / output_name
    report = {
        "execution_id": execution_id,
        "source": source,
        "dry_run": bool(args.dry_run),
        "results": results,
        "agent_log": str(agent_log) if agent_log else None,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_state_from_results(
        execution_id=execution_id,
        source=source,
        results=results,
        summary_payload=payload,
    )
    append_executor_agent_log(
        agent_log,
        execution_id=execution_id,
        source=source,
        dry_run=bool(args.dry_run),
        requests=requests,
        results=results,
        elapsed_seconds=(datetime.now(timezone.utc) - executor_started_at).total_seconds(),
    )
    print(json.dumps({"executor_log": str(output), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
