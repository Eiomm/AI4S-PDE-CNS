#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "agent_workspace" / "code"
LOG_ROOT = ROOT / "agent_workspace" / "logs"
DEFAULT_CONFIG = ROOT / "configs" / "agent_gpt55.yaml"


def progress(message: str) -> None:
    """打印分文件生成进度，避免长时间 API 请求看起来像卡住。"""

    print(f"[chunk-agent] {message}", flush=True)


def utc_stamp() -> str:
    """生成 UTC 时间戳，保证日志目录稳定可排序。"""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def load_dotenv(path: Path) -> None:
    """读取 Task1 `.env`，只写入当前进程环境，不打印密钥。"""

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


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 配置；空配置返回空字典。"""

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def first_existing_env(names: list[str]) -> tuple[str, str]:
    """从候选环境变量里选择第一个可用 API key。"""

    for name in names:
        value = os.environ.get(name)
        if value:
            return name, value
    raise RuntimeError(f"未找到 API key 环境变量：{', '.join(names)}")


def extract_first_json_object(text: str) -> str:
    """从模型文本中提取第一个完整 JSON object。"""

    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    start = stripped.find("{")
    if start < 0:
        raise ValueError("Agent 响应中没有 JSON 起始 `{`。")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    raise ValueError("Agent 响应中的 JSON 没有完整闭合。")


def guarded_code_path(relative_path: str) -> Path:
    """限制 Agent 只能写入 agent_workspace/code 内部的普通相对路径。"""

    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError(f"不能写绝对路径：{relative_path}")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"路径不能包含空段或 ..：{relative_path}")
    if any(part.startswith(".") for part in candidate.parts):
        raise ValueError(f"路径不能写隐藏文件：{relative_path}")
    resolved = (CODE_ROOT / candidate).resolve()
    if not resolved.is_relative_to(CODE_ROOT.resolve()):
        raise ValueError(f"路径越界：{relative_path}")
    return resolved


def call_agent(client: OpenAI, config: dict[str, Any], messages: list[dict[str, str]]) -> str:
    """通过官方 proxy 调用模型，并兼容 stream / non-stream 两种模式。"""

    request_options = dict(config.get("request_options") or {})
    request_options["response_format"] = {"type": "json_object"}
    request_options["max_completion_tokens"] = int(request_options.get("max_completion_tokens") or 4096)
    response = client.chat.completions.create(
        model=str(config["model"]),
        messages=messages,
        temperature=float(config.get("temperature", 0.2)),
        **request_options,
    )
    if request_options.get("stream"):
        parts: list[str] = []
        for chunk in response:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                print(content, end="", flush=True)
                parts.append(content)
        print("", flush=True)
        return "".join(parts)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Agent 返回为空。")
    return content


def base_context() -> dict[str, Any]:
    """给每个小 Agent 调用的共享上下文，只保留必要接口事实。"""

    return {
        "task_root": str(ROOT),
        "write_root": "agent_workspace/code",
        "fixed_python_env": "/root/miniconda3/envs/ai4s-pde-cns/bin/python",
        "harness_command": "scripts/task1_predict.py --config <yaml> --split <val|test> [--limit N]",
        "harness_config_schema": {
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
        "harness_outputs": {
            "prediction": "runs/task1/<UTC timestamp>/task1_pred.hdf5",
            "time": "runs/task1/<UTC timestamp>/task1_time.csv",
            "log": "runs/task1/<UTC timestamp>/task1_logs.log",
            "metadata": "runs/task1/<UTC timestamp>/metadata.json",
            "metrics": "runs/task1/<run_name>/metrics.json when split=val",
        },
        "prediction_contract": {
            "dataset_key": "tensor",
            "test_shape": [1000, 200, 256],
            "observed_steps": 10,
            "finite_required": True,
            "first_10_frames_equal_input": True,
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
        "current_best": {
            "route": "official_ensemble_postprocess",
            "competition_score_proxy": 18.8111937324,
            "models": "FNO 0.12 + Unet-PF20 0.88, segment_fno_weights [0.17,0.03,0.11], persistence alpha [0.89,0.95,0.41]",
        },
        "rules": [
            "最终 code 文件必须由 Agent 生成，不要复制 harness src 源码。",
            "可以调用公开的本仓库 harness 命令完成 checkpoint replay，因为预测来自神经网络 checkpoint。",
            "如果生成 fine-tune 代码，必须使用 reduced_resolution_t=5 和 reduced_resolution=4；禁止用 raw adjacent frames temporal_stride=1 训练官方 checkpoint descendant。",
            "所有代码写详细中文注释。",
            "不要使用数值求解器生成预测。",
            "Optuna 是 Python 库，可以在生成代码里 import optuna；executor 不需要替 Agent 封装 optuna。",
        ],
    }


def file_tasks() -> list[dict[str, str]]:
    """定义三个小文件生成任务，避免一个响应里塞入多个长文件。"""

    return [
        {
            "path": "task1_routes.py",
            "purpose": "生成路线配置工具：定义 ModelSpec、RouteConfig、构建 official_ensemble_postprocess YAML dict、写 YAML、规范 run_name 和输出路径。必须使用显式 models 列表，不允许 weight_fno/weight_unet 这种旧字段。",
        },
        {
            "path": "task1_replay.py",
            "purpose": "生成命令行 replay 脚本：解析 split/run-name/limit/权重/分段后处理参数，通过 task1_routes 构造本轮配置，再运行 bash scripts/run_in_env.sh scripts/task1_predict.py，终端实时打印进展，最后检查 task1_pred.hdf5 的 shape、finite、前 10 帧一致，并输出 metrics 路径。",
        },
        {
            "path": "task1_optuna_search.py",
            "purpose": "生成 Optuna 搜索脚本：直接 import optuna，搜索 FNO/Unet 权重、segment_fno_weights、persistence_segment_alpha；每个 trial 调用 task1_replay.py 做 val；读取 metrics.json 的 competition_score_proxy；打印 trial/best 进展；写 best_params.json。预算默认小，便于先 cheap probe。",
        },
    ]


def build_messages(task: dict[str, str], context: dict[str, Any]) -> list[dict[str, str]]:
    """为单个文件构造短 prompt，要求 JSON 只包含一个文件。"""

    system = (
        "你是 Task1 PDE_Burgers 的代码生成 Agent。你正在官方 OpenAI-compatible proxy 记录下生成最终 code 文件。"
        "只输出 JSON 对象，不输出 Markdown。JSON schema: "
        "{\"path\": string, \"content\": string, \"notes\": string}。"
        "content 必须是完整 Python 文件源码，包含详细中文注释。"
    )
    user = {
        "target_file": task["path"],
        "purpose": task["purpose"],
        "shared_context": context,
        "implementation_constraints": [
            "只能生成 target_file 这一个文件。",
            "代码要简单可运行，优先调用已有 harness CLI，而不是复制 harness 源码。",
            "脚本路径假设从 Task1 根目录执行。",
            "如果导入同目录文件，使用 sys.path 插入当前文件目录，保证直接 python agent_workspace/code/xxx.py 可运行。",
            "检查 HDF5 时兼容数据集 key: tensor, data, u。",
            "所有异常消息要明确，便于终端定位。",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunked Agent code writer for Task1.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Agent API YAML。")
    parser.add_argument("--only", choices=[task["path"] for task in file_tasks()], default=None, help="只生成单个文件。")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = load_yaml(Path(args.config))
    api_key_env, api_key = first_existing_env(list(config.get("api_key_env") or ["OPENAI_API_KEY"]))
    progress(f"使用 API key 环境变量：{api_key_env}")
    progress(f"模型：{config['model']}，base_url={config['base_url']}")

    client = OpenAI(
        api_key=api_key,
        base_url=str(config["base_url"]),
        timeout=float(config.get("timeout_seconds", 600)),
        max_retries=int(config.get("max_retries", 0)),
    )
    run_id = f"agent_chunked_{utc_stamp()}"
    log_dir = LOG_ROOT / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    context = base_context()
    tasks = [task for task in file_tasks() if args.only is None or task["path"] == args.only]
    written: list[str] = []

    for task in tasks:
        progress(f"开始生成：{task['path']}")
        task_log = log_dir / task["path"].replace("/", "__")
        task_log.mkdir(parents=True, exist_ok=True)
        messages = build_messages(task, context)
        (task_log / "request_context.json").write_text(json.dumps({"messages": messages}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raw = call_agent(client, config, messages)
        (task_log / "raw_response.txt").write_text(raw, encoding="utf-8")
        payload = json.loads(extract_first_json_object(raw))
        if payload.get("path") != task["path"]:
            raise ValueError(f"Agent 返回 path 不匹配：expected={task['path']} actual={payload.get('path')}")
        output_path = guarded_code_path(str(payload["path"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(payload["content"]), encoding="utf-8")
        (task_log / "parsed_response.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(payload["path"])
        progress(f"已写入：{output_path}")

    summary = {
        "run_id": run_id,
        "model": config.get("model"),
        "base_url": config.get("base_url"),
        "written_files": written,
        "log_dir": str(log_dir),
        "reason": "按文件拆分生成，规避第三方流式接口长 JSON 截断。",
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
