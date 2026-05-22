#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "agent_workspace" / "code"
LOG_ROOT = ROOT / "agent_workspace" / "logs"
DEFAULT_CONFIG = ROOT / "configs" / "agent_gpt55.yaml"
PYTHON_BIN = "/root/miniconda3/envs/ai4s-pde-cns/bin/python"


def progress(message: str) -> None:
    """打印 runner 进度；flush 避免长请求期间终端没有反馈。"""

    print(f"[loop] {message}", flush=True)


def utc_stamp() -> str:
    """生成稳定 UTC 时间戳，用于 run id。"""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def load_dotenv(path: Path) -> None:
    """读取 `.env` 到当前进程环境，不打印任何 secret。"""

    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置，空文件返回空字典。"""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def first_existing_env(names: list[str]) -> tuple[str, str]:
    """选择第一个可用 API key 环境变量。"""

    for name in names:
        value = os.environ.get(name)
        if value:
            return name, value
    raise RuntimeError(f"未找到 API key 环境变量：{', '.join(names)}")


def extract_first_json_object(text: str) -> dict[str, Any]:
    """从可能含 `<think>` 或说明文字的响应里提取第一个完整 JSON object。"""

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("LLM 响应中没有可解析的 JSON object。")


def guarded_relative_path(raw_path: str) -> str:
    """校验 Agent 生成文件路径只能是干净相对路径。"""

    normalized = raw_path.replace("\\", "/").strip()
    path = Path(normalized)
    if path.is_absolute() or not normalized:
        raise ValueError(f"生成文件路径不能是绝对路径或空路径：{raw_path}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"生成文件路径不能包含空段或 ..：{raw_path}")
    if any(part.startswith(".") for part in path.parts):
        raise ValueError(f"生成文件路径不能写隐藏文件：{raw_path}")
    return path.as_posix()


def load_small_text(path: Path, max_chars: int = 6000) -> str:
    """读取小型上下文文件；过长时截断，避免 prompt 变重。"""

    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]\n"


def build_context(
    run_id: str,
    *,
    split: str,
    limit: int | None,
    goal: str,
    mode: str,
    history: list[dict[str, Any]],
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造给 Agent 的轻量上下文。

    只提供接口事实、当前目标和少量 memory 摘要，不把 harness 源码塞进 prompt。
    """

    if os.environ.get("TASK1_LOOP_VERBOSE_CONTEXT") != "1":
        return {
            "task_root": str(ROOT),
            "run_id": run_id,
            "goal": goal,
            "mode": mode,
            "split": split,
            "limit": limit,
            "limit_rule": "if limit is null, run full split and omit --limit from predict_cli",
            "must_generate": "files.run_experiment.py only",
            "must_write_observation_env": "TASK1_LOOP_OBSERVATION_PATH",
            "must_write_progress_env": "TASK1_LOOP_PROGRESS_PATH",
            "allowed_action": "use subprocess to call existing harness CLI; do not copy harness source",
            "predict_cli": "bash scripts/run_in_env.sh scripts/task1_predict.py --config <yaml> --split <split> [--limit N]",
            "run_output": "runs/task1/<UTC timestamp>/task1_pred.hdf5 and metadata.json",
            "history": compact_history(history),
            "config_yaml_required_fields": {
                "route": "official_ensemble_postprocess",
                "device": "auto",
                "batch_size": 50,
                "input_path": "data/task1_test.hdf5",
                "val_input_path": "data/task1_val.hdf5",
                "full_t_path": "data/task1_val.hdf5",
                "models": [
                    {"kind": "fno", "checkpoint": "checkpoints/official/nu0.001_fno.pt", "weight": 0.12},
                    {"kind": "unet_pf20", "checkpoint": "checkpoints/official/nu0.001_unet_pf20.pt", "weight": 0.88},
                ],
                "segment_fno_weights": [0.17, 0.03, 0.11],
                "persistence_segment_alpha": [0.89, 0.95, 0.41],
            },
            "allowed_experiments_when_mode_explore": [
                "official_fno baseline replay",
                "official_unet_pf20 baseline replay",
                "official_ensemble replay",
                "official_ensemble_postprocess replay",
                "small ensemble/postprocess parameter probe using explicit models weights and 3 segment arrays",
            ],
            "exploration_rules": [
                "Each turn must test exactly one clear hypothesis.",
                "Do not repeat a run/config already shown in history.",
                "Use validation split for exploration; do not run test except final replay.",
                "Prefer cheap limit runs before full validation.",
                "Record score, run_dir, prediction, metadata, and next_recommendation.",
            ],
            "observation_minimum": ["summary", "metrics", "artifacts", "next_recommendation"],
            "previous_failure": previous,
        }

    return {
        "task_root": str(ROOT),
        "generated_code_dir": "agent_workspace/code",
        "entrypoint": "run_experiment.py",
        "run_id": run_id,
        "goal": goal,
        "mode": mode,
        "split": split,
        "limit": limit,
        "history": compact_history(history),
        "required_behavior": [
            "只生成 run_experiment.py 一个文件。",
            "脚本从 Task1 根目录运行，可以调用现有 harness CLI。",
            "第一轮目标是 replay 官方 FNO、官方 Unet-PF、official_ensemble_postprocess，并写 observation。",
            "预测必须来自神经网络 checkpoint，不得调用数值求解器。",
            "必须把 JSON observation 写到环境变量 TASK1_LOOP_OBSERVATION_PATH。",
            "建议把 JSONL progress 写到 TASK1_LOOP_PROGRESS_PATH，终端会实时打印。",
            "所有代码包含详细中文注释。",
        ],
        "harness": {
            "python": PYTHON_BIN,
            "predict_command": "bash scripts/run_in_env.sh scripts/task1_predict.py --config <yaml> --split <split> [--limit N]",
            "memory_export_command": "bash scripts/run_in_env.sh scripts/memory_export.py --run-dir <run_dir> --hypothesis <text> --decision <decision> --append-registry",
            "config_schema": {
                "route": "official_ensemble_postprocess",
                "device": "auto",
                "batch_size": 50,
                "input_path": "data/task1_test.hdf5",
                "val_input_path": "data/task1_val.hdf5",
                "full_t_path": "data/task1_val.hdf5",
                "models": [
                    {"kind": "fno", "checkpoint": "checkpoints/official/nu0.001_fno.pt", "weight": 0.12},
                    {"kind": "unet_pf20", "checkpoint": "checkpoints/official/nu0.001_unet_pf20.pt", "weight": 0.88},
                ],
                "segment_fno_weights": [0.17, 0.03, 0.11],
                "persistence_segment_alpha": [0.89, 0.95, 0.41],
            },
            "output_paths": {
                "prediction": "runs/task1/<UTC timestamp>/task1_pred.hdf5",
                "time": "runs/task1/<UTC timestamp>/task1_time.csv",
                "log": "runs/task1/<UTC timestamp>/task1_logs.log",
                "metadata": "runs/task1/<UTC timestamp>/metadata.json",
                "metrics": "runs/task1/<run_name>/metrics.json for validation split",
            },
        },
        "prediction_contract": {
            "input_test": "data/task1_test.hdf5",
            "input_val": "data/task1_val.hdf5",
            "dataset_key": "tensor",
            "test_shape": [1000, 200, 256],
            "observed_steps": 10,
            "first_10_frames_equal_input": True,
            "finite_required": True,
        },
        "scale_alignment_hard_rule": {
            "reduced_resolution_t": 5,
            "reduced_resolution": 4,
            "model_step_meaning": "one model step equals 5 raw PDEBench time indices",
            "raw_spatial_size": 1024,
            "reduced_spatial_size": 256,
            "fine_tune_must_use_temporal_stride": 5,
            "fine_tune_must_use_spatial_downsample": 4,
            "forbid_raw_adjacent_temporal_stride_1": True,
        },
        "current_memory": {
            "rules": load_small_text(ROOT / "memory" / "contract" / "task1_rules.yaml", 3000),
            "leaderboard": load_small_text(ROOT / "memory" / "findings" / "metric_leaderboard.csv", 2000),
            "strategy": load_small_text(ROOT / "memory" / "wisdom" / "strategy_summary.md", 3000),
        },
        "previous_failure": previous,
    }


def compact_history(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把已有 turn 压缩成下一轮可用的探索记忆。

    这里只保留少量字段，避免多轮后 prompt 膨胀。完整记录仍在每个 turn 目录。
    """

    compact: list[dict[str, Any]] = []
    for turn in turns[-6:]:
        observation = turn.get("observation") if isinstance(turn, dict) else {}
        execution = turn.get("execution") if isinstance(turn, dict) else {}
        artifacts = observation.get("artifacts") if isinstance(observation, dict) else {}
        metrics = observation.get("metrics") if isinstance(observation, dict) else {}
        if not isinstance(artifacts, dict):
            artifacts = {}
        if not isinstance(metrics, dict):
            metrics = {}
        compact.append(
            {
                "turn_index": turn.get("turn_index"),
                "returncode": execution.get("returncode"),
                "summary": str(observation.get("summary", ""))[:240] if isinstance(observation, dict) else "",
                "run_dir": artifacts.get("run_dir"),
                "prediction": artifacts.get("prediction") or artifacts.get("pred") or artifacts.get("best_prediction"),
                "score": metrics.get("competition_score_proxy"),
                "mse": metrics.get("mse"),
                "forecast_mse": metrics.get("forecast_mse"),
                "long_horizon_mse": metrics.get("long_horizon_mse"),
            }
        )
    return compact


def build_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    """构造短 prompt，明确要求单文件代码包。"""

    system = "Output JSON only. First char must be {. Top-level keys exactly: think, files, entrypoint."
    if context.get("mode") == "explore":
        action_note = (
            "Explore mode: choose one new validation experiment based on context.history. "
            "Do not repeat the same route/config. You may change route, model weights, "
            "segment_fno_weights, or persistence_segment_alpha within the official checkpoint harness."
        )
    else:
        action_note = "Replay mode: run official_ensemble_postprocess for this turn."

    user = {
        "context": context,
        "output_schema": {
            "think": "one short public reason",
            "files": {"run_experiment.py": "完整 Python 源码字符串"},
            "entrypoint": "run_experiment.py",
        },
        "implementation_notes": [
            "Keep code under 120 lines.",
            "Use Chinese comments.",
            "Write YAML config to agent_workspace/code/tmp_configs.",
            action_note,
            "If context.limit is null, do not pass --limit and name run with _full.",
            "Append JSONL progress before and after subprocess.",
            "Always write observation JSON, even on failure.",
            "Do not use numerical solvers.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def call_llm(config: dict[str, Any], messages: list[dict[str, str]], *, heartbeat_seconds: float = 15.0) -> dict[str, Any]:
    """调用 LLM，并在等待时输出 heartbeat。"""

    api_key_env, api_key = first_existing_env(list(config.get("api_key_env") or ["OPENAI_API_KEY"]))
    client = OpenAI(
        api_key=api_key,
        base_url=str(config["base_url"]),
        timeout=float(config.get("timeout_seconds", 600)),
        max_retries=int(config.get("max_retries", 0)),
    )
    request_options = dict(config.get("request_options") or {})
    request_options["stream"] = False
    request_options["max_completion_tokens"] = min(int(request_options.get("max_completion_tokens") or 4096), 4096)
    request_options["response_format"] = {"type": "json_object"}
    progress(f"调用 LLM：model={config['model']} base_url={config['base_url']} key_env={api_key_env}")

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}
    done = threading.Event()
    started = time.perf_counter()

    def worker() -> None:
        try:
            response = client.chat.completions.create(
                model=str(config["model"]),
                messages=messages,
                temperature=float(config.get("temperature", 0.2)),
                **request_options,
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("LLM 返回空 content。")
            result["raw"] = content
            result["raw_response"] = response.model_dump(mode="json") if hasattr(response, "model_dump") else None
        except BaseException as exc:  # noqa: BLE001 - 需要跨线程传回原始异常
            error["exc"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while not done.wait(float(heartbeat_seconds)):
        progress(f"等待 LLM 返回 elapsed={time.perf_counter() - started:.0f}s")
    thread.join(timeout=1.0)
    if "exc" in error:
        raise error["exc"]
    progress(f"LLM 已返回 elapsed={time.perf_counter() - started:.1f}s")
    raw = str(result["raw"])
    return {"raw": raw, "payload": extract_first_json_object(raw), "raw_response": result.get("raw_response")}


def parse_agent_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """校验 Agent 输出为单文件 files+entrypoint 格式。"""

    if set(payload) != {"think", "files", "entrypoint"}:
        raise ValueError(f"Agent JSON 顶层字段必须只包含 think/files/entrypoint，实际为：{sorted(payload)}")
    files = payload["files"]
    if not isinstance(files, dict) or set(files) != {"run_experiment.py"}:
        raise ValueError("files 必须且只能包含 run_experiment.py")
    entrypoint = guarded_relative_path(str(payload["entrypoint"]))
    if entrypoint != "run_experiment.py":
        raise ValueError("entrypoint 必须是 run_experiment.py")
    content = files["run_experiment.py"]
    if not isinstance(content, str) or "TASK1_LOOP_OBSERVATION_PATH" not in content:
        raise ValueError("run_experiment.py 必须是源码字符串，并写 TASK1_LOOP_OBSERVATION_PATH")
    return {
        "think": str(payload["think"]),
        "files": {"run_experiment.py": content},
        "entrypoint": entrypoint,
    }


def write_agent_files(parsed: dict[str, Any], turn_dir: Path) -> None:
    """同时写入最终 code 目录和 turn 备份目录。"""

    CODE_ROOT.mkdir(parents=True, exist_ok=True)
    generated_dir = turn_dir / "generated_files"
    generated_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in parsed["files"].items():
        safe = guarded_relative_path(relative_path)
        (CODE_ROOT / safe).write_text(content, encoding="utf-8")
        (generated_dir / safe).write_text(content, encoding="utf-8")


class TailBuffer:
    """线程安全尾部缓冲，避免完整 stdout/stderr 过大。"""

    def __init__(self, limit: int = 8000) -> None:
        self.limit = int(limit)
        self.text = ""
        self.lock = threading.Lock()

    def append(self, value: str) -> None:
        with self.lock:
            self.text = (self.text + value)[-self.limit :]

    def get(self) -> str:
        with self.lock:
            return self.text


def drain_stream(stream: Any, sink: TailBuffer) -> threading.Thread:
    """后台读取子进程输出，避免 pipe 阻塞。"""

    def run() -> None:
        if stream is None:
            return
        for line in stream:
            sink.append(str(line))
        try:
            stream.close()
        except Exception:
            pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def emit_new_progress(progress_path: Path, seen_lines: int) -> int:
    """读取 Agent 代码写出的 JSONL progress，并打印新增行。"""

    if not progress_path.is_file():
        return seen_lines
    lines = progress_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for raw in lines[seen_lines:]:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            progress(f"[live] {raw[:240]}")
            continue
        name = event.get("event") or event.get("type") or "event"
        status = event.get("status")
        run_name = event.get("run_name")
        score = event.get("competition_score_proxy")
        pieces = [f"[live] {name}"]
        if status is not None:
            pieces.append(f"status={status}")
        if run_name is not None:
            pieces.append(f"run={run_name}")
        if score is not None:
            pieces.append(f"score={score}")
        progress(" ".join(pieces))
    return len(lines)


def execute_entrypoint(turn_dir: Path, *, timeout_seconds: float) -> dict[str, Any]:
    """执行 Agent 生成的 run_experiment.py，并读取 observation。"""

    observation_path = turn_dir / "observation.json"
    progress_path = turn_dir / "live_progress.jsonl"
    observation_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "TASK1_ROOT": str(ROOT),
            "TASK1_LOOP_TURN_DIR": str(turn_dir),
            "TASK1_LOOP_OBSERVATION_PATH": str(observation_path),
            "TASK1_LOOP_PROGRESS_PATH": str(progress_path),
            "PYTHONUNBUFFERED": "1",
        }
    )
    pythonpath = [str(ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    stdout_tail = TailBuffer()
    stderr_tail = TailBuffer()
    process = subprocess.Popen(
        [PYTHON_BIN, str(CODE_ROOT / "run_experiment.py")],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    stdout_thread = drain_stream(process.stdout, stdout_tail)
    stderr_thread = drain_stream(process.stderr, stderr_tail)
    started = time.perf_counter()
    seen_lines = 0
    timed_out = False
    progress(f"执行 Agent 代码：{CODE_ROOT / 'run_experiment.py'}")
    progress(f"live_progress={progress_path}")
    while process.poll() is None:
        elapsed = time.perf_counter() - started
        if elapsed > float(timeout_seconds):
            timed_out = True
            process.kill()
            break
        seen_lines = emit_new_progress(progress_path, seen_lines)
        time.sleep(3.0)
    returncode = process.wait()
    stdout_thread.join(timeout=2.0)
    stderr_thread.join(timeout=2.0)
    seen_lines = emit_new_progress(progress_path, seen_lines)
    elapsed = time.perf_counter() - started
    observation: dict[str, Any]
    if observation_path.is_file():
        try:
            observation = json.loads(observation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            observation = {"summary": "observation 非法 JSON", "error": str(exc)}
    else:
        observation = {"summary": "Agent 代码没有写 observation.json"}
    execution = {
        "returncode": int(returncode),
        "timed_out": bool(timed_out),
        "elapsed_seconds": elapsed,
        "stdout_tail": stdout_tail.get(),
        "stderr_tail": stderr_tail.get(),
        "observation_path": str(observation_path),
        "progress_path": str(progress_path),
    }
    return {"execution": execution, "observation": observation}


def auto_export_memory(observation: dict[str, Any]) -> list[dict[str, Any]]:
    """从 observation 中发现 run_dir，并自动追加 compact memory。"""

    artifacts = observation.get("artifacts") if isinstance(observation, dict) else {}
    candidates: list[str] = []
    if isinstance(artifacts, dict):
        for key in ("run_dir", "best_run_dir"):
            value = artifacts.get(key)
            if isinstance(value, str) and value:
                candidates.append(value)
        runs = artifacts.get("runs")
        if isinstance(runs, list):
            for item in runs:
                if isinstance(item, dict) and isinstance(item.get("run_dir"), str):
                    candidates.append(item["run_dir"])
    exported: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_run_dir in candidates:
        run_dir = str(raw_run_dir)
        if run_dir in seen:
            continue
        seen.add(run_dir)
        command = [
            "bash",
            "scripts/run_in_env.sh",
            "scripts/memory_export.py",
            "--run-dir",
            run_dir,
            "--hypothesis",
            _compact_hypothesis(observation),
            "--decision",
            "baseline",
            "--tag",
            "agent_loop",
            "--append-registry",
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
        exported.append(
            {
                "run_dir": run_dir,
                "returncode": result.returncode,
                "stdout": result.stdout[-1000:],
                "stderr": result.stderr[-1000:],
            }
        )
    return exported


def turn_success(record: dict[str, Any]) -> bool:
    """判断一轮闭环是否成功完成。"""

    execution = record.get("execution") or {}
    observation = record.get("observation") or {}
    artifacts = observation.get("artifacts") if isinstance(observation, dict) else {}
    if execution.get("returncode") != 0:
        return False
    if not isinstance(artifacts, dict):
        return False
    prediction = artifacts.get("prediction") or artifacts.get("best_prediction") or artifacts.get("pred")
    if isinstance(prediction, str) and (ROOT / prediction).is_file():
        return True
    if isinstance(prediction, str) and Path(prediction).is_file():
        return True
    return bool(observation.get("metrics"))


def _compact_hypothesis(observation: dict[str, Any]) -> str:
    """把 observation summary 压成适合 memory 的一句话。"""

    summary = observation.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()[:240]
    if isinstance(summary, dict):
        run_name = summary.get("run_name")
        success = summary.get("success")
        if run_name:
            return f"agent checkpoint replay run={run_name} success={success}"
    return "agent generated checkpoint replay"


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    """运行轻量 code loop，失败时把错误反馈给下一轮 Agent 修复。"""

    # 旧仓库 .env 里可能保存 APIFOX_GPT_GE_API_KEY；先读取它作为候选来源。
    # 随后读取 Task1 本地 .env，保留本目录优先配置，但不会覆盖已经存在的 key。
    load_dotenv(Path("/autodl-fs/data/AI4S-PDE-CNS/.env"))
    load_dotenv(ROOT / ".env")
    config = load_yaml(args.config)
    run_id = args.run_id or f"task1_loop_{utc_stamp()}"
    sample_limit = None if args.limit is not None and args.limit <= 0 else args.limit
    loop_dir = LOG_ROOT / run_id
    loop_dir.mkdir(parents=True, exist_ok=True)
    previous: dict[str, Any] | None = None
    turns: list[dict[str, Any]] = []
    for turn_index in range(1, int(args.max_turns) + 1):
        turn_dir = loop_dir / f"turn_{turn_index:03d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        progress(f"Turn {turn_index}/{args.max_turns} 开始，目录={turn_dir}")
        context = build_context(
            run_id,
            split=args.split,
            limit=sample_limit,
            goal=args.goal,
            mode=args.mode,
            history=turns,
            previous=previous,
        )
        messages = build_messages(context)
        (turn_dir / "request_context.json").write_text(json.dumps({"messages": messages}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        llm_result = call_llm(config, messages, heartbeat_seconds=args.llm_heartbeat_seconds)
        (turn_dir / "raw_response.txt").write_text(llm_result["raw"], encoding="utf-8")
        if llm_result.get("raw_response") is not None:
            (turn_dir / "raw_response_api.json").write_text(json.dumps(llm_result["raw_response"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        parsed = parse_agent_payload(llm_result["payload"])
        (turn_dir / "parsed_response.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_agent_files(parsed, turn_dir)
        executed = execute_entrypoint(turn_dir, timeout_seconds=args.execution_timeout_seconds)
        memory_exports = auto_export_memory(executed["observation"])
        record = {
            "turn_index": turn_index,
            "think": parsed["think"],
            "execution": executed["execution"],
            "observation": executed["observation"],
            "memory_exports": memory_exports,
            "turn_dir": str(turn_dir),
        }
        (turn_dir / "turn_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        turns.append(record)
        progress(
            "Turn 完成："
            f"returncode={record['execution']['returncode']} "
            f"elapsed={record['execution']['elapsed_seconds']:.1f}s "
            f"summary={str(record['observation'].get('summary', ''))[:120]}"
        )
        if turn_success(record):
            progress("闭环成功：Agent 代码已执行并产生可追踪产物。")
            if not args.continue_after_success:
                break
        previous = {
            "summary": record["observation"].get("summary"),
            "execution": record["execution"],
            "observation": record["observation"],
        }
    summary = {
        "run_id": run_id,
        "loop_dir": str(loop_dir),
        "turns": len(turns),
        "success": bool(turns and turn_success(turns[-1])),
        "latest_turn": turns[-1] if turns else None,
        "generated_code": str(CODE_ROOT / "run_experiment.py"),
    }
    (loop_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight Task1 Agent code loop.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=1, help="默认先用 1 条样本做前缀验证；传 0 表示完整 split，不加 --limit。")
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--mode", choices=["replay", "explore"], default="replay", help="replay 成功即停；explore 可连续测试不同假设。")
    parser.add_argument("--continue-after-success", action="store_true", help="成功后继续下一轮，用于真正多轮探索。")
    parser.add_argument("--llm-heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("--execution-timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--goal",
        default="生成并执行 Task1 官方 checkpoint replay 的最小实验脚本，完成 prediction、校验、metrics、memory 自动记录。",
    )
    args = parser.parse_args()
    summary = run_loop(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
