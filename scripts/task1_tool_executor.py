#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import contextmanager
import html
import json
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
    "fno": "configs/official_fno.yaml",
    "official_unet_pf20": "configs/official_unet_pf20.yaml",
    "unet_pf20": "configs/official_unet_pf20.yaml",
    "official_ensemble": "configs/official_ensemble.yaml",
    "official_ensemble_postprocess": "configs/official_ensemble_postprocess.yaml",
}

FNO_TRAINABLE_MODULES = {"fc0", "conv0", "w0", "conv1", "w1", "conv2", "w2", "conv3", "w3", "fc1", "fc2"}


def utc_stamp() -> str:
    """生成 executor 日志时间戳。"""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def normalize_execution_id(raw: Any | None = None) -> str:
    text = str(raw or "").strip()
    if re.fullmatch(r"agent_\d{8}T\d{6}\d{6}Z", text):
        return text
    if re.fullmatch(r"\d{8}T\d{6}\d{6}Z", text):
        return f"agent_{text}"
    return f"agent_{utc_stamp()}"


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件，空文件或非对象会直接报错。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 必须是 JSON object")
    return payload


def mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def latest_agent_summary() -> Path:
    """寻找最近一次 Agent runner 的 summary.json。"""

    candidates = sorted((ROOT / "agent_workspace" / "logs").glob("agent_*/summary.json"))
    if not candidates:
        raise FileNotFoundError("没有找到 agent_workspace/logs/agent_*/summary.json")
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
    return ROOT / "agent_workspace" / "tool_outputs" / Path(*parts)


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

    output_dir = ROOT / "agent_workspace" / "tool_outputs" / "web_search"
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
    config = CONFIG_BY_REPLAY_TARGET.get(target)
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
            checkpoint = str(Path(str(run_dir)) / "best.pt")
        else:
            for previous in range(index - 1, -1, -1):
                candidate = ROOT / "runs" / "task1" / f"{execution_id}__tool{previous:02d}" / "best.pt"
                if candidate.is_file():
                    checkpoint = str(candidate)
                    break
        if not checkpoint:
            raise ValueError(
                "finetuned_checkpoint_replay 需要 checkpoint/checkpoint_path/run_dir，"
                "或在同一轮较早 tool 中先成功执行 finetune_local。"
            )
    split = str(request.get("split") or "val")
    if split not in {"val", "test"}:
        raise ValueError(f"finetuned_checkpoint_replay split 只能是 val/test: {split}")
    run_name = f"{execution_id}__tool{index:02d}"
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
    command = [
        "bash",
        "scripts/run_in_env.sh",
        "scripts/task1_validate.py",
        "--prediction",
        str(prediction),
        "--input",
        str(request.get("input") or "data/task1_val.hdf5"),
    ]
    if request.get("target_hdf5"):
        command.extend(["--target", str(request["target_hdf5"])])
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
    """构造官方 proxy 日志转换命令。"""

    raw_inputs = request.get("input_logs")
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


def submission_package_command(request: dict[str, Any], index: int, *, execution_id: str) -> list[str]:
    """构造 Task1 submission 打包命令。"""

    run_dir = request.get("run_dir")
    if not run_dir:
        raise ValueError("submission_package 请求必须提供 run_dir")
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
    if request.get("code_dir"):
        command.extend(["--code-dir", str(request["code_dir"])])
    if request.get("llm_log"):
        command.extend(["--llm-log", str(request["llm_log"])])
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


def _safe_run_dir(raw: Any, index: int, *, execution_id: str) -> str:
    run_dir = f"runs/task1/{execution_id}__tool{index:02d}"
    path = Path(run_dir)
    if path.is_absolute():
        raise ValueError("finetune_local run_dir must be relative to Task1 root")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe finetune_local run_dir: {run_dir}")
    if len(path.parts) < 2 or path.parts[0] != "runs" or path.parts[1] != "task1":
        raise ValueError("finetune_local run_dir must be under runs/task1/")
    return path.as_posix()


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
        str(request.get("base_checkpoint") or "checkpoints/official/nu0.001_fno.pt"),
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
        return run_command(finetune_local_command(request, index, execution_id=execution_id), timeout=timeout)
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
        return run_command(submission_package_command(request, index, execution_id=execution_id), timeout=timeout)
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
            checkpoint = f"runs/task1/{execution_id}__tool<previous_finetune>/best.pt"
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
        return {"tool": tool, "status": "planned", "command": submission_package_command(request, index, execution_id=execution_id)}
    if tool == "web_search":
        query = str(request.get("query") or request.get("purpose") or "").strip()
        if not query:
            raise ValueError("web_search 请求必须提供 query 或 purpose")
        output = ROOT / "agent_workspace" / "tool_outputs" / "web_search" / f"{execution_id}__tool{index:02d}__web_search.json"
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
    args = parser.parse_args()

    if args.requests_json:
        payload = load_json(Path(args.requests_json))
        requests = payload.get("tool_requests", [])
        source = str(Path(args.requests_json))
        execution_id = normalize_execution_id(args.execution_id)
    else:
        summary_path = Path(args.summary) if args.summary else latest_agent_summary()
        payload = load_json(summary_path)
        requests = payload.get("tool_requests", [])
        source = str(summary_path)
        execution_id = normalize_execution_id(args.execution_id or payload.get("run_id") or summary_path.parent.name)

    if not isinstance(requests, list):
        raise ValueError("tool_requests 必须是 list")

    results: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            results.append({"status": "rejected", "reason": "tool_request 必须是 object", "request": request})
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
            else:
                results.append(
                    execute_request(
                        request,
                        index,
                        timeout=args.timeout,
                        execution_id=execution_id,
                        summary_payload=payload,
                        log_start_timestamp=args.log_start_timestamp,
                        log_end_timestamp=args.log_end_timestamp,
                    )
                )
        except Exception as exc:
            results.append(
                {
                    "tool": str(request.get("tool") or ""),
                    "status": "rejected" if args.dry_run else "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "request": request,
                }
            )

    log_dir = ROOT / "agent_workspace" / "executor_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{execution_id}__dryrun_{utc_stamp()}.json" if args.dry_run else f"{execution_id}.json"
    output = log_dir / output_name
    report = {
        "execution_id": execution_id,
        "source": source,
        "dry_run": bool(args.dry_run),
        "results": results,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"executor_log": str(output), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
