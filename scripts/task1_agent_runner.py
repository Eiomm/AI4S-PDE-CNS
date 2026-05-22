#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai4sv2_task1.memory import query_memory


PROMPT_PATH = ROOT / "agent_workspace" / "prompts" / "task1_planner.md"
SCHEMA_PATH = ROOT / "agent_workspace" / "prompts" / "action_schema.json"
CONFIG_PATH = ROOT / "configs" / "agent_gpt55.yaml"
CODE_ROOT = ROOT / "agent_workspace" / "code"
RUNNER_LOG_ROOT = ROOT / "agent_workspace" / "logs"

ALL_TOOL_NAMES = [
    "memory_query",
    "data_shape_check",
    "checkpoint_replay",
    "finetune_local",
    "finetuned_checkpoint_replay",
    "validation",
    "llm_log_prepare",
    "submission_package",
    "optuna",
    "web_search",
    "ray_tune",
    "wandb",
]

WORKFLOW_TOOL_WHITELIST = {
    "rules_read": {"memory_query", "data_shape_check", "web_search"},
    "data_shape_check": {"data_shape_check", "memory_query"},
    "baseline_replay": {"checkpoint_replay", "validation", "memory_query", "data_shape_check"},
    "checkpoint_finetune": {
        "checkpoint_replay",
        "finetune_local",
        "finetuned_checkpoint_replay",
        "validation",
        "optuna",
        "memory_query",
        "data_shape_check",
        "web_search",
        "ray_tune",
        "wandb",
    },
    "prediction_validation": {"finetuned_checkpoint_replay", "validation", "memory_query", "data_shape_check"},
    "log_compliance": {"llm_log_prepare", "validation", "memory_query", "data_shape_check"},
    "submission_packaging": {
        "submission_package",
        "llm_log_prepare",
        "validation",
        "finetuned_checkpoint_replay",
        "memory_query",
        "data_shape_check",
    },
}


def progress(message: str) -> None:
    """向终端打印 Agent runner 阶段进度。

    这里使用 `flush=True`，避免长时间 API 请求前的提示被缓冲，看起来像卡住。
    """

    print(f"[agent] {message}", flush=True)


def utc_stamp() -> str:
    """生成稳定的 UTC 时间戳，用于 runner 日志目录命名。"""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_log_path(raw_path: str | None, *, fallback: Path) -> Path:
    if raw_path:
        path = Path(raw_path)
        return path if path.is_absolute() else ROOT / path
    return fallback


def resolve_optional_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_experiment_state(raw_path: str | None, experiment_dir: Path | None) -> dict[str, Any] | None:
    path = resolve_optional_path(raw_path)
    if path is None and experiment_dir is not None:
        path = experiment_dir / "state.json"
    if path is None or not path.is_file():
        return None
    state = load_json_file(path)
    compact = {
        "experiment_id": state.get("experiment_id"),
        "stage": state.get("stage"),
        "current_round": state.get("current_round"),
        "max_rounds": state.get("max_rounds"),
        "paths": state.get("paths", {}),
        "artifact_policy": state.get("artifact_policy", {}),
        "best_local_candidate": state.get("best_local_candidate"),
        "blockers": state.get("blockers", []),
        "history_hints": state.get("history_hints", [])[:3],
        "recent_rounds": state.get("rounds", [])[-3:],
        "local_artifacts": state.get("local_artifacts", [])[-6:],
    }
    return compact


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def api_status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def is_transient_api_error(exc: Exception) -> bool:
    status = api_status_code(exc)
    if status in {408, 409, 425, 429}:
        return True
    if status is not None and status >= 500:
        return True
    name = type(exc).__name__.lower()
    return any(marker in name for marker in ("timeout", "connection", "server", "ratelimit"))


def append_api_error(path: Path, *, attempt: int, max_attempts: int, exc: Exception, retrying: bool) -> None:
    append_jsonl(
        path,
        {
            "timestamp": utc_now_iso(),
            "attempt": attempt,
            "max_attempts": max_attempts,
            "retrying": retrying,
            "error_type": type(exc).__name__,
            "status_code": api_status_code(exc),
            "message": str(exc),
        },
    )


def _relative_display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def build_run_paths(args: argparse.Namespace) -> dict[str, Any]:
    experiment_dir = resolve_optional_path(args.experiment_dir)
    if experiment_dir is None:
        run_id = f"agent_{utc_stamp()}"
        code_root = CODE_ROOT / run_id
        log_dir = RUNNER_LOG_ROOT / run_id
        agent_log = resolve_log_path(args.agent_log, fallback=log_dir / "task1_logs.log")
        return {
            "experiment_dir": None,
            "experiment_id": None,
            "round_name": None,
            "run_id": run_id,
            "code_root": code_root,
            "log_dir": log_dir,
            "agent_log": agent_log,
            "state_path": resolve_optional_path(args.state_path),
        }

    round_name = f"turn_{int(args.round_index):03d}" if args.round_index is not None else f"turn_{utc_stamp()}"
    run_id = f"{experiment_dir.name}__{round_name}"
    code_root = experiment_dir / "code"
    log_dir = experiment_dir / "logs" / round_name
    agent_log = resolve_log_path(args.agent_log, fallback=experiment_dir / "logs" / "task1_logs.log")
    return {
        "experiment_dir": experiment_dir,
        "experiment_id": experiment_dir.name,
        "round_name": round_name,
        "run_id": run_id,
        "code_root": code_root,
        "log_dir": log_dir,
        "agent_log": agent_log,
        "state_path": resolve_optional_path(args.state_path) or experiment_dir / "state.json",
    }


def _json_call(name: str, payload: dict[str, Any]) -> str:
    return f"{name}({json.dumps(payload, ensure_ascii=False, separators=(',', ':'))})"


def _agent_log_response(payload: dict[str, Any]) -> str:
    response = {
        "decision": payload.get("decision"),
        "experiment_plan": payload.get("experiment_plan"),
        "planned_commands": payload.get("planned_commands", []),
        "memory_update": payload.get("memory_update", {}),
    }
    return json.dumps(response, ensure_ascii=False, separators=(",", ":"))


def append_agent_response_log(raw: str, agent_log: Path, code_root: Path, elapsed_seconds: float) -> None:
    """Append one competition-style Agent log record without prompt content."""

    record: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
    }
    try:
        payload = json.loads(strip_json_fence(raw))
    except Exception:
        record["response"] = raw
        append_jsonl(agent_log, record)
        return

    record["response"] = _agent_log_response(payload)
    tool_calls: list[str] = []
    for item in payload.get("files") or []:
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("path") or "")
        if not raw_path:
            continue
        tool_calls.append(
            _json_call(
                "write",
                {
                    "filePath": _relative_display_path(code_root / raw_path),
                    "content": str(item.get("content") or ""),
                },
            )
        )
    for request in payload.get("tool_requests") or []:
        if isinstance(request, dict):
            tool_calls.append(_json_call("tool_request", request))
    if tool_calls:
        record["tool_calls"] = "\n".join(tool_calls)
    append_jsonl(agent_log, record)


def load_dotenv(path: Path) -> None:
    """读取 Task1 根目录 `.env`，只写入当前进程环境，不打印任何 secret。

    这里不用 `python-dotenv`，避免给 runner 增加额外依赖。格式只支持最常见
    的 `KEY=value`，足够覆盖当前第三方 API key 配置。
    """

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
    """读取 YAML 配置；空文件返回空 dict。"""

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def first_existing_env(names: list[str]) -> tuple[str, str]:
    """从候选环境变量里选择第一个可用 API key。"""

    for name in names:
        value = os.environ.get(name)
        if value:
            return name, value
    joined = ", ".join(names)
    raise RuntimeError(f"未找到 API key 环境变量，请在 .env 或 shell 中设置：{joined}")


def strip_json_fence(text: str) -> str:
    """兼容模型偶尔返回 ```json fenced block 的情况。

    第三方 OpenAI-compatible API 有时会把 `<think>...</think>` 也放入
    message.content。runner 需要从这类文本中提取第一个完整 JSON 对象，
    否则会在第一个字符处 JSONDecodeError。
    """

    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    stripped = match.group(1).strip() if match else stripped
    if stripped.startswith("{"):
        return stripped
    return extract_first_valid_json_object(stripped)


def extract_first_valid_json_object(text: str) -> str:
    """Find the first syntactically valid JSON object in mixed model text.

    Some OpenAI-compatible providers may still return `<think>...</think>` text
    even when `response_format=json_object` is requested. The older balanced
    brace extractor can grab Python snippets inside thinking text. This scanner
    tries every `{` position with JSONDecoder and returns the first object that
    looks like the action schema payload.
    """

    decoder = json.JSONDecoder()
    fallback: str | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        candidate = text[index : index + end]
        if fallback is None:
            fallback = candidate
        if "decision" in payload and "files" in payload:
            return candidate
    if fallback is not None:
        return fallback
    return extract_first_json_object(text)


def extract_first_json_object(text: str) -> str:
    """从任意模型文本中提取第一个完整 JSON object。

    这个函数按字符扫描，理解 JSON 字符串里的转义和引号，因此不会被代码内容
    里的 `{` / `}` 干扰。找不到完整对象时抛出明确错误，便于排查 prompt 或
    第三方 API 返回格式问题。
    """

    start = text.find("{")
    if start < 0:
        raise ValueError("Agent 响应中没有找到 JSON 对象起始 `{`。")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
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
                return text[start : index + 1]

    raise ValueError("Agent 响应中 JSON 对象没有完整闭合。")


def build_context(args: argparse.Namespace, generated_code_root: Path, agent_log_path: Path) -> dict[str, Any]:
    """构造给 Agent 的最小上下文。

    关键原则：只放 memory 的小型 retrieval packet、路径摘要和目标，不把 harness
    源码或旧仓库源码塞进 prompt。
    """

    memory_packet = query_memory(route=args.route, tags=args.tag, limit=args.memory_limit, max_chars=args.max_memory_chars)
    memory_packet["retrieved_records"] = (memory_packet.get("retrieved_records") or [])[:4]
    memory_packet["leaderboard_rows"] = (memory_packet.get("leaderboard_rows") or [])[:8]
    if isinstance(memory_packet.get("rules"), str) and len(memory_packet["rules"]) > 2400:
        memory_packet["rules"] = memory_packet["rules"][:2400] + "\n...[truncated]"
    if isinstance(memory_packet.get("strategy"), str) and len(memory_packet["strategy"]) > 1800:
        memory_packet["strategy"] = memory_packet["strategy"][:1800] + "\n...[truncated]"
    experiment_dir = resolve_optional_path(args.experiment_dir)
    experiment_context = None
    if experiment_dir is not None:
        experiment_context = {
            "experiment_id": experiment_dir.name,
            "experiment_dir": relative_to_root(experiment_dir),
            "round_name": f"turn_{int(args.round_index):03d}" if args.round_index is not None else None,
            "code_dir": relative_to_root(experiment_dir / "code"),
            "logs_dir": relative_to_root(experiment_dir / "logs"),
            "runs_dir": relative_to_root(experiment_dir / "runs"),
            "metrics_dir": relative_to_root(experiment_dir / "metrics"),
            "submission_dir": relative_to_root(experiment_dir / "submission"),
            "policy": "One experiment owns one code directory. Later rounds modify the same code directory and write runs/logs under this experiment directory.",
        }
    experiment_state = load_experiment_state(args.state_path, experiment_dir)
    return {
        "task_root": str(ROOT),
        "generated_code_root": generated_code_root.relative_to(ROOT).as_posix(),
        "experiment": experiment_context,
        "experiment_state": experiment_state,
        "agent_log_path": _relative_display_path(agent_log_path),
        "logging_policy": {
            "main_agent_log": _relative_display_path(agent_log_path),
            "main_agent_log_is_per_experiment": True,
            "do_not_append_to_shared_data_task1_logs": True,
            "do_not_use_raw_proxy_log_as_submission_log": True,
        },
        "current_goal": args.goal,
        "candidate_routes": [
            "official_fno",
            "official_unet_pf20",
            "official_ensemble",
            "official_ensemble_postprocess",
            "finetune_fno",
        ],
        "required_first_baselines": [
            "official_fno_checkpoint_replay",
            "official_unet_pf20_checkpoint_replay",
        ],
        "main_optimization_direction": "fine_tune_official_checkpoints",
        "scale_alignment_hard_rule": {
            "official_checkpoint_reduced_resolution_t": 5,
            "official_checkpoint_reduced_resolution": 4,
            "model_step_meaning": "one model step equals 5 raw PDEBench time indices",
            "raw_spatial_size": 1024,
            "reduced_spatial_size": 256,
            "finetune_temporal_stride": 5,
            "finetune_spatial_downsample": 4,
            "forbid_finetune_temporal_stride_1": True,
            "first_observed_raw_indices": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
            "first_supervised_target_raw_index": 50,
            "note": "Validation/test HDF5 files are already on reduced scale; raw PDEBench fine-tune windows must be reduced before supervision.",
        },
        "data": {
            "test_hdf5": "data/task1_test.hdf5",
            "val_hdf5": "data/task1_val.hdf5",
            "finetune_train_manifest": "data/pdebench_burgers/manifest.json",
            "finetune_train_hdf5": "data/pdebench_burgers/raw/1D_Burgers_Sols_Nu0.001.hdf5",
            "output_dataset_key": "tensor",
            "output_shape": [1000, 200, 256],
            "observed_steps": 10,
            "reduced_resolution_t": 5,
            "reduced_resolution": 4,
        },
        "scoring": {
            "task_total_max": 150.0,
            "prediction_component_max": 75.0,
            "train_time_component_max": 35.0,
            "inference_time_component_max": 40.0,
            "prediction_score_formula": "segmented_prediction_score * 0.75",
            "segmented_prediction_score_max": 100.0,
            "train_time_seconds_tiers": [
                {"max_seconds": 3600, "score": 35},
                {"max_seconds": 7200, "score": 25},
                {"max_seconds": 18000, "score": 20},
                {"max_seconds": 30000, "score": 10},
                {"above_seconds": 30000, "score": 0},
            ],
            "inference_time_policy": {
                "max_seconds_for_nonzero_task": 120,
                "score": "40 linearly decreases to 0 from 0s to 120s; >120s makes Task1 score 0",
            },
            "agent_budget_policy": {
                "optimize_total_score_not_accuracy_only": True,
                "prefer_train_time_under_seconds_for_full_time_score": 3600,
                "keep_final_test_inference_under_seconds": 120,
                "reason": "Task1 score includes prediction accuracy, training time including Agent thinking time, and inference time.",
            },
        },
        "checkpoints": {
            "fno": "checkpoints/official/nu0.001_fno.pt",
            "unet_pf20": "checkpoints/official/nu0.001_unet_pf20.pt",
        },
        "baseline_knowledge_paths": {
            "neuraloperator": "/autodl-fs/data/AI4Sv2/Task1/data/baselinecode/neuraloperator",
            "deeponet": "/autodl-fs/data/AI4Sv2/Task1/data/baselinecode/deeponet",
            "physics_informed_deeponets": "/autodl-fs/data/AI4Sv2/Task1/data/baselinecode/Physics-informed-DeepONets",
        },
        "available_tools": {
            "web_search": "查找 FNO / Burgers / neural operator / checkpoint fine-tune 相关公开资料，辅助决策。",
            "optuna": "搜索 fine-tune 超参、ensemble 权重和后处理参数。",
            "ray_tune": "在预算允许时运行并发 trial，建议使用 ASHA 或 early stopping。",
            "wandb": "长训练或多 trial 时记录曲线和候选对比。",
            "data_shape_check": "检查 Task1 test/val/raw train HDF5 的 key、shape、dtype 和官方 reduced-scale 约束。",
            "validation": "校验 shape、finite、前 10 帧一致和 validation metric。",
            "checkpoint_replay": "运行官方 FNO / Unet-PF checkpoint baseline。",
            "finetuned_checkpoint_replay": "把 finetune_local 产出的 best.pt 接入标准 val/test prediction、validation、memory 和 leaderboard 链路；同一轮前序 finetune_local 成功时可自动接上 best.pt。",
            "finetune_local": (
                "运行受控本地 FNO checkpoint 微调工具；executor 固定 temporal_stride=5、spatial_downsample=4。"
                "该工具只支持 FNO checkpoint，不要把 Unet-PF checkpoint 传给 finetune_local。"
                "Agent 可决策 steps/lr/rollout/trainable，也可用 trainable_modules 自由选择 FNO 模块："
                "fc0, conv0, w0, conv1, w1, conv2, w2, conv3, w3, fc1, fc2。"
                "Agent 触发的实验产物由 executor 统一写入当前 experiment/runs/<turn>__toolXX；旧模式才写入 runs/task1/<agent_run_id>__toolXX。"
            ),
            "llm_log_prepare": "整理比赛要求的 task1_logs.log JSONL；direct API 模式优先使用当前实验 logs/task1_logs.log，旧代理模式才转换 openai_proxy JSONL。",
            "submission_package": "从已通过校验的 test split run 目录生成 submission 目录；不能使用 val run_dir，且当前实验 code/ 需要有真实代码快照。",
            "memory_query": "读取 4 个 compact memory 入口。",
        },
        "workflow_modules": [
            {
                "name": "rules_read",
                "purpose": "读取赛题硬规则、checkpoint/数据/提交约束。",
                "tool": "memory_query",
                "expected_artifact": "compact memory packet",
            },
            {
                "name": "data_shape_check",
                "purpose": "确认 test/val/raw train HDF5 shape，并确认 reduced_resolution_t=5、reduced_resolution=4。",
                "tool": "data_shape_check",
                "expected_artifact": "当前 experiment/runs/tool_outputs/data_shape/*.json",
            },
            {
                "name": "baseline_replay",
                "purpose": "运行官方 FNO 和 Unet-PF baseline，建立对照分数。",
                "tool": "checkpoint_replay",
                "expected_artifact": "当前 experiment/runs/<turn>__toolXX/",
            },
            {
                "name": "checkpoint_finetune",
                "purpose": "在官方尺度对齐下微调 FNO checkpoint。",
                "tool": "finetune_local",
                "expected_artifact": "best.pt, last.pt, metadata.json, task1_time.csv",
            },
            {
                "name": "finetuned_checkpoint_replay",
                "purpose": "对 finetune_local 产出的 best.pt 做标准 val/test replay，并写入 memory。",
                "tool": "finetuned_checkpoint_replay",
                "expected_artifact": "task1_pred.hdf5, metrics.json, metadata.json, memory_export.json",
            },
            {
                "name": "prediction_validation",
                "purpose": "校验 prediction shape、finite、前 10 帧一致和 validation metric。",
                "tool": "validation",
                "expected_artifact": "validation JSON report",
            },
            {
                "name": "log_compliance",
                "purpose": "确认本次实验专属 Agent JSONL 日志可用于提交；必要时再整理 proxy 日志。",
                "tool": "llm_log_prepare",
                "expected_artifact": _relative_display_path(agent_log_path),
            },
            {
                "name": "submission_packaging",
                "purpose": "把通过校验的 run、合规 log、本轮 Agent code 快照打包。",
                "tool": "submission_package",
                "expected_artifact": "submissions/<name>/",
            },
        ],
        "code_artifact_contract": {
            "purpose": "只有提出新代码时才校验 code_artifacts；常规策略 mutation 可只调用受控工具。",
            "required_top_level_field": "code_artifacts only when files are generated",
            "entrypoint_paths_must_match_files": True,
            "stage_requirements": {
                "checkpoint_finetune": {
                    "required_role": "train",
                    "required_executable_python": True,
                    "must_record": [
                        "train_hdf5",
                        "val_hdf5",
                        "base_checkpoint",
                        "temporal_stride=5",
                        "spatial_downsample=4",
                        "trainable or trainable_modules",
                        "checkpoint/metadata output",
                    ],
                },
                "prediction_validation": {
                    "required_role": "validate",
                    "required_executable_python": True,
                    "must_record": ["prediction_path", "input_path", "shape", "first_ten_match", "finite"],
                },
                "submission_packaging": {
                    "required_role_any_of": ["package", "predict"],
                    "required_executable_python": True,
                    "must_record": ["run_dir", "llm_log", "code_dir", "submission output"],
                },
            },
            "planning_only_allowed_role": "research_planning",
            "readme_manifest_only_is_not_enough_for": ["checkpoint_finetune", "prediction_validation", "submission_packaging"],
        },
        "decision_policy": {
            "toolbox_is_not_exhaustive": True,
            "agent_internal_knowledge_allowed": True,
            "web_search_is_optional_for_external_evidence": True,
            "final_decisions_require_local_validation": True,
            "hard_constraints_scope": [
                "competition rules",
                "output format and shape",
                "LLM log compliance",
                "scale alignment",
                "time budget and score accounting",
                "traceable Agent-generated code artifacts",
            ],
            "exploration_freedom": (
                "Outside hard constraints, Agent may choose model family, loss, fine-tune scope, "
                "ensemble, postprocess, experiment order, rollback, and repair strategy."
            ),
        },
        "exploration_policy": {
            "stages": ["baseline", "cheap_probe", "search", "promotion", "submission"],
            "avoid_naive_full_parallel_training": True,
            "prefer_budgeted_search": True,
            "cheap_probe_first": True,
            "full_validation_only_for_promoted_candidates": True,
            "parallelism": {
                "cheap_probe_trials": "allowed",
                "ensemble_weight_scan": "allowed",
                "full_finetune_trials": "limited",
            },
        },
        "allowed_next_commands": [
            "bash scripts/run_in_env.sh scripts/task1_predict.py --config <config> --split val",
            "bash scripts/run_in_env.sh scripts/task1_validate.py --prediction <pred> --input data/task1_val.hdf5 --target data/task1_val.hdf5 --output <metrics>",
            "bash scripts/run_in_env.sh scripts/task1_finetune_local.py --steps <N> --temporal-stride 5 --spatial-downsample 4 [--trainable-module conv3 --trainable-module fc2]",
            "bash scripts/run_in_env.sh scripts/task1_prepare_llm_log.py --input logs/openai_proxy_*.jsonl --output <task1_logs.log>",
            "bash scripts/run_in_env.sh scripts/task1_make_submission.py --run-dir <run_dir> --submission-name submission --llm-log <task1_logs.log>",
            "bash scripts/run_in_env.sh scripts/memory_export.py --run-dir <run_dir> --hypothesis <text> --decision <decision>",
            "bash scripts/run_in_env.sh scripts/memory_promote.py --record-id <id> --slot <slot> --metric <metric> --value <value>",
        ],
        "memory_packet": memory_packet,
    }


def build_messages(prompt: str, schema: dict[str, Any], context: dict[str, Any]) -> list[dict[str, str]]:
    """组装 OpenAI-compatible chat messages。"""

    user_payload = {
        "action_schema": schema,
        "context": context,
    }
    return [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
        },
    ]


def _truncate_text(value: Any, limit: int) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "\n...[truncated]"
    return value


def _compact_memory_packet(packet: dict[str, Any]) -> dict[str, Any]:
    task_rules = packet.get("task_rules") if isinstance(packet.get("task_rules"), dict) else {}
    compact_rules = {
        "task": task_rules.get("task"),
        "problem_id": task_rules.get("problem_id"),
        "input": {
            "observed_steps": (task_rules.get("input") or {}).get("observed_steps"),
            "official_reduced_resolution": (task_rules.get("input") or {}).get("official_reduced_resolution"),
            "finetune_scale_hard_rule": (task_rules.get("input") or {}).get("finetune_scale_hard_rule"),
        },
        "output": task_rules.get("output"),
        "validation": task_rules.get("validation"),
    }
    best_candidates = []
    for item in (packet.get("best_candidates") or [])[:3]:
        if not isinstance(item, dict):
            continue
        best_candidates.append(
            {
                "slot": item.get("slot"),
                "record_id": item.get("record_id"),
                "metric": item.get("metric"),
                "value": item.get("value"),
                "submit_ready": item.get("submit_ready"),
                "blockers": _truncate_text(item.get("blockers"), 160),
                "strategy_hint_only": True,
                "usable_as_current_artifact": False,
            }
        )
    retrieved_records = []
    for item in (packet.get("retrieved_records") or [])[:2]:
        if isinstance(item, dict):
            retrieved_records.append(
                {
                    "id": item.get("id") or item.get("record_id"),
                    "route": item.get("route"),
                    "tags": item.get("tags"),
                    "summary": _truncate_text(item.get("summary") or item.get("content"), 500),
                }
            )
    return {
        "task_rules_digest": compact_rules,
        "strategy_summary": _truncate_text(packet.get("strategy_summary") or packet.get("strategy"), 900),
        "best_candidates": best_candidates,
        "relevant_experiments": (packet.get("relevant_experiments") or [])[:2],
        "retrieval_budget": packet.get("retrieval_budget"),
        "memory_sources": packet.get("memory_sources"),
        "retrieved_records": retrieved_records,
    }


def infer_context_stage(context: dict[str, Any]) -> str:
    state = context.get("experiment_state")
    if isinstance(state, dict):
        if state.get("best_local_candidate"):
            return "validation_submission"
        stage = str(state.get("stage") or "").strip()
        if stage:
            return stage
    return "fresh_experiment"


def _compact_experiment_state(state: Any) -> Any:
    if not isinstance(state, dict):
        return state
    compact = {
        "schema": state.get("schema"),
        "experiment_id": state.get("experiment_id"),
        "stage": state.get("stage"),
        "current_round": state.get("current_round"),
        "best_local_candidate": state.get("best_local_candidate"),
        "blockers": state.get("blockers"),
        "local_artifacts_tail": (state.get("local_artifacts") or [])[-5:],
    }
    db = state.get("strategy_db") if isinstance(state.get("strategy_db"), dict) else {}
    programs = db.get("programs") if isinstance(db.get("programs"), dict) else {}
    keep_ids = []
    for key in ("active_parent_id", "best_strategy_id"):
        value = db.get(key)
        if value:
            keep_ids.append(str(value))
    keep_ids.extend(str(item) for item in db.get("active_inspiration_ids") or [])
    keep_ids.extend(str(item) for item in (db.get("archive") or [])[:5])
    seen: set[str] = set()
    selected_programs: list[dict[str, Any]] = []
    for strategy_id in keep_ids:
        if strategy_id in seen:
            continue
        seen.add(strategy_id)
        item = programs.get(strategy_id)
        if isinstance(item, dict):
            selected_programs.append(
                {
                    "id": item.get("id"),
                    "parent_id": item.get("parent_id"),
                    "generation": item.get("generation"),
                    "status": item.get("status"),
                    "route": item.get("route"),
                    "workflow_module": item.get("workflow_module"),
                    "hypothesis": _truncate_text(item.get("hypothesis"), 240),
                    "metrics": item.get("metrics") or {},
                    "artifacts": item.get("artifacts") or {},
                    "error": _truncate_text(item.get("error"), 220),
                }
            )
    compact["strategy_db"] = {
        "schema": db.get("schema"),
        "strategy_pool_size": len(programs),
        "active_parent_id": db.get("active_parent_id"),
        "active_inspiration_ids": db.get("active_inspiration_ids") or [],
        "best_strategy_id": db.get("best_strategy_id"),
        "last_sampling_mode": db.get("last_sampling_mode"),
        "archive": (db.get("archive") or [])[:8],
        "selected_programs": selected_programs,
        "feature_map": db.get("feature_map") or {},
    }
    return compact


def compact_context_for_prompt(context: dict[str, Any]) -> dict[str, Any]:
    """Remove duplicated long-form guidance before sending the request.

    The system prompt already carries the policy. This payload should only carry
    current facts, compact state, and short tool hints so each round focuses on
    the current experiment instead of repeatedly re-reading the whole workflow.
    """

    stage = infer_context_stage(context)
    available_tools = {
        "memory_query": "读取 compact memory；历史结果只能作策略提示。",
        "data_shape_check": "检查 Task1 HDF5 shape、dtype、key 和 reduced-scale 约束。",
        "checkpoint_replay": "运行官方 checkpoint baseline，建立当前实验对照。",
        "finetune_local": "受控本地 FNO 微调；executor 固定 stride=5/downsample=4。不要传 Unet-PF checkpoint。Agent 可规划 max_samples、steps、batch_size、num_workers、prefetch、trainable 等速度/精度参数。",
        "finetuned_checkpoint_replay": "把当前实验 best.pt 接入标准 val/test replay、metrics、memory。",
        "validation": "校验 prediction shape、finite、前 10 帧和 validation metric。",
        "llm_log_prepare": "整理当前实验窗口内的合规 JSONL 日志；direct API 模式使用当前实验 logs/task1_logs.log。",
        "submission_package": "只基于当前实验已验证 test run、log、code 生成 submission；不能拿 val run 打包。",
        "optuna": "在 Agent 生成代码中做受控超参/后处理搜索；不要替代验证。",
        "web_search": "可选外部证据；最终结论必须回到本地 validation。",
    }
    stage_tool_hints = {
        "fresh_experiment": ["memory_query", "data_shape_check", "checkpoint_replay", "finetune_local"],
        "round_completed": ["finetuned_checkpoint_replay", "validation", "finetune_local", "submission_package"],
        "tool_failed": ["validation", "data_shape_check", "finetune_local", "finetuned_checkpoint_replay"],
        "validation_submission": ["finetuned_checkpoint_replay", "validation", "llm_log_prepare", "submission_package"],
        "runner_failed": ["memory_query", "data_shape_check"],
    }
    workflow_modules = [
        "rules_read",
        "data_shape_check",
        "baseline_replay",
        "checkpoint_finetune",
        "prediction_validation",
        "log_compliance",
        "submission_packaging",
    ]
    code_artifact_contract = {
        "default": "strategy-only output is allowed when using existing controlled tools",
        "new_code": "if files are generated, code_artifacts must describe executable entrypoints and traceability",
        "readme_manifest_only_forbidden_when_generating_code": [
            "checkpoint_finetune",
            "prediction_validation",
            "submission_packaging",
        ],
    }
    return {
        "task_root": context.get("task_root"),
        "generated_code_root": context.get("generated_code_root"),
        "experiment": context.get("experiment"),
        "experiment_state": _compact_experiment_state(context.get("experiment_state")),
        "context_stage": stage,
        "agent_log_path": context.get("agent_log_path"),
        "logging_policy": context.get("logging_policy"),
        "current_goal": context.get("current_goal"),
        "scale_alignment_hard_rule": context.get("scale_alignment_hard_rule"),
        "data": context.get("data"),
        "scoring": context.get("scoring"),
        "checkpoints": context.get("checkpoints"),
        "baseline_knowledge_paths": context.get("baseline_knowledge_paths"),
        "available_tools": available_tools,
        "stage_tool_hints": {
            "current_stage": stage,
            "recommended_tools": stage_tool_hints.get(stage, stage_tool_hints["fresh_experiment"]),
            "note": "Hints are not a fixed route. Stay within hard rules and current-experiment artifact policy.",
        },
        "workflow_modules": workflow_modules,
        "code_artifact_contract": code_artifact_contract,
        "decision_policy": {
            "optimize_total_score": True,
            "current_experiment_artifacts_only": True,
            "historical_memory_strategy_only": True,
            "do_not_stop_before_max_rounds": True,
            "free_to_explore_within_hard_rules": True,
        },
        "memory_packet": _compact_memory_packet(context.get("memory_packet") or {}),
    }


def compact_action_schema_for_prompt(schema: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Send a stage-aware schema card instead of the full verbose schema.

    The full schema file remains the source of truth for humans and docs. The
    runner still performs Python-side validation after the model returns JSON.
    This compact version reduces repeated prompt tokens while preserving the
    fields that matter for contract enforcement.
    """

    stage_hints = context.get("stage_tool_hints") if isinstance(context.get("stage_tool_hints"), dict) else {}
    recommended_tools = stage_hints.get("recommended_tools") or []
    workflow_modules = context.get("workflow_modules") or []
    return {
        "type": "object",
        "required": schema.get("required") or ["decision", "files", "code_artifacts", "experiment_plan"],
        "additionalProperties": False,
        "properties": {
            "decision": {
                "type": "object",
                "required": ["route", "reason", "expected_gain", "risk"],
                "properties": {
                    "route": "string",
                    "reason": "string; cite current experiment state, memory hints, and hard constraints",
                    "expected_gain": "string; expected accuracy/time/reliability gain",
                    "risk": "string; main failure or compliance risk",
                },
            },
            "strategy_candidates": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "description": "List 2-4 candidate strategies before choosing one to execute.",
                "items": {
                    "required": ["id", "route", "hypothesis", "expected_gain", "risk", "estimated_cost", "execute_now"],
                    "properties": {
                        "id": "unique short string",
                        "route": "strategy route name",
                        "hypothesis": "string; what this strategy tests",
                        "expected_gain": "string; expected score/time/reliability gain",
                        "risk": "string; key failure or compliance risk",
                        "estimated_cost": "string; train/inference/API/file cost estimate",
                        "execute_now": "boolean; true for selected_strategy_id",
                    },
                },
            },
            "selected_strategy_id": "string; must match one strategy_candidates[].id and explain the executed tool_requests",
            "experiment_plan": {
                "type": "object",
                "required": ["stage", "workflow_module", "budget", "time_budget", "score_tradeoff"],
                "properties": {
                    "stage": {"enum": ["baseline", "cheap_probe", "search", "promotion", "submission"]},
                    "workflow_module": {"enum": workflow_modules},
                    "parallel_trials": "integer >= 1",
                    "budget": "string",
                    "time_budget": "string; include train<=3600s preference and inference<120s hard limit",
                    "score_tradeoff": "string; optimize Task1 total score, not validation only",
                    "early_stop": "string",
                    "promotion_rule": "string",
                },
            },
            "files": {
                "type": "array",
                "description": "Optional. Omit when this round only mutates strategy and calls existing controlled tools.",
                "items": {
                    "path": "relative path under current experiment code/, no absolute path, no ..",
                    "content": "complete Agent-generated file content",
                },
            },
            "code_artifacts": {
                "type": "object",
                "description": "Optional unless files are generated. Required when introducing new executable code.",
                "required": ["primary_role", "entrypoints", "experiment_traceability", "limitations"],
                "properties": {
                    "primary_role": {
                        "enum": [
                            "research_planning",
                            "baseline_replay",
                            "checkpoint_finetune",
                            "prediction_validation",
                            "log_compliance",
                            "submission_packaging",
                        ]
                    },
                    "entrypoints": {
                        "type": "array",
                        "items": {
                            "path": "must match files[].path",
                            "role": {"enum": ["train", "predict", "validate", "package", "analyze", "manifest"]},
                            "is_executable": "boolean",
                            "linked_tool": {"enum": ["none", *ALL_TOOL_NAMES]},
                        },
                    },
                    "experiment_traceability": "string; connect code to tool_requests, checkpoint, data scale, parameters",
                    "limitations": "string",
                },
            },
            "tool_requests": {
                "type": "array",
                "description": (
                    "Tool use must match experiment_plan.workflow_module. "
                    f"Current stage recommended tools: {recommended_tools}. "
                    "The runner will reject tool_requests outside the workflow whitelist."
                ),
                "items": {
                    "required": ["tool", "purpose"],
                    "properties": {
                        "tool": {"enum": ALL_TOOL_NAMES},
                        "purpose": "string",
                        "split": {"enum": ["val", "test"]},
                        "target": "checkpoint_replay target",
                        "run_dir": "current experiment run directory only when referencing artifacts",
                        "checkpoint": "current experiment checkpoint path for finetuned replay",
                        "checkpoint_path": "alias of checkpoint",
                        "prediction": "current experiment prediction path for validation",
                        "input": "validation input hdf5",
                        "target_hdf5": "validation target hdf5",
                        "output": "validation/log output path",
                        "llm_log": "current experiment task1_logs.log",
                        "code_dir": "current experiment code dir",
                        "submission_name": "submission directory name",
                        "steps": "integer for finetune_local",
                        "batch_size": "integer; try 8/16/32 when GPU memory allows",
                        "eval_batch_size": "integer for validation rollout",
                        "lr": "number for finetune_local",
                        "rollout_steps": "integer for finetune_local",
                        "max_samples": "integer; cheap probe 2048/5000, promotion up to 10000",
                        "val_max_samples": "integer <= 100",
                        "val_every": "integer validation interval",
                        "log_every": "integer logging interval",
                        "num_workers": "integer 0-16; current machine smoke test suggests 4 as a good default",
                        "prefetch_factor": "integer 1-16 when num_workers>0",
                        "pin_memory": {"enum": ["auto", "true", "false"]},
                        "persistent_workers": {"enum": ["auto", "true", "false"]},
                        "trainable": {"enum": ["head", "last-block-head", "all", "custom"]},
                        "trainable_modules": "array of FNO module names",
                        "budget": "string",
                        "query": "web_search query",
                        "objective": "optuna objective",
                    },
                },
            },
            "planned_commands": {"type": "array", "items": "string"},
            "memory_update": {
                "type": "object",
                "properties": {"hypothesis": "string", "tags": "array[string]"},
            },
        },
    }


def validate_tool_requests_for_workflow(payload: dict[str, Any]) -> dict[str, Any]:
    plan = payload.get("experiment_plan") if isinstance(payload.get("experiment_plan"), dict) else {}
    workflow_module = str(plan.get("workflow_module") or "")
    allowed = WORKFLOW_TOOL_WHITELIST.get(workflow_module)
    if allowed is None:
        raise ValueError(f"未知 workflow_module，无法校验 tool 白名单：{workflow_module}")
    requested: list[str] = []
    rejected: list[str] = []
    for request in payload.get("tool_requests") or []:
        if not isinstance(request, dict):
            raise ValueError("tool_requests[] 必须是 object。")
        tool = str(request.get("tool") or "")
        requested.append(tool)
        if tool not in allowed:
            rejected.append(tool)
    if rejected:
        raise ValueError(
            f"tool_requests 与 workflow_module={workflow_module} 不匹配；"
            f"rejected={sorted(set(rejected))}；allowed={sorted(allowed)}"
        )
    return {
        "workflow_module": workflow_module,
        "allowed_tools": sorted(allowed),
        "requested_tools": requested,
    }


def guarded_code_path(relative_path: str, code_root: Path) -> Path:
    """把 Agent 返回路径限制在本轮专属 code 目录内。

    这一步是安全边界：不允许绝对路径、不允许 `..`，也不允许写隐藏文件。
    """

    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError(f"Agent 文件路径不能是绝对路径：{relative_path}")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"Agent 文件路径不能包含空段或 ..：{relative_path}")
    if any(part.startswith(".") for part in candidate.parts):
        raise ValueError(f"Agent 文件路径不能写隐藏文件：{relative_path}")
    resolved = (code_root / candidate).resolve()
    if not resolved.is_relative_to(code_root.resolve()):
        raise ValueError(f"Agent 文件路径越界：{relative_path}")
    return resolved


def write_agent_files(files: list[dict[str, str]], code_root: Path) -> list[str]:
    """写入 Agent 生成的文件，并返回相对路径清单。"""

    written: list[str] = []
    code_root.mkdir(parents=True, exist_ok=True)
    for item in files:
        path = guarded_code_path(str(item["path"]), code_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(item["content"]), encoding="utf-8")
        written.append(path.relative_to(code_root).as_posix())
    return written


def _looks_executable_python(path: str, content: str) -> bool:
    if not path.endswith(".py"):
        return False
    markers = ("def main(", "argparse", "if __name__ ==", "click.command", "typer.")
    return any(marker in content for marker in markers)


def validate_strategy_selection(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("strategy_candidates")
    if not isinstance(candidates, list):
        raise ValueError("Agent 响应必须包含 strategy_candidates list。")
    if len(candidates) < 2 or len(candidates) > 4:
        raise ValueError("strategy_candidates 必须包含 2-4 个候选策略。")

    required_fields = ["id", "route", "hypothesis", "expected_gain", "risk", "estimated_cost", "execute_now"]
    seen_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValueError("strategy_candidates[] 必须是 object。")
        missing = [field for field in required_fields if field not in candidate]
        if missing:
            raise ValueError(f"strategy_candidates[{index}] 缺少字段：{missing}")
        strategy_id = str(candidate.get("id") or "").strip()
        if not strategy_id:
            raise ValueError(f"strategy_candidates[{index}].id 不能为空。")
        if strategy_id in seen_ids:
            raise ValueError(f"strategy_candidates id 重复：{strategy_id}")
        seen_ids.add(strategy_id)
        normalized.append(
            {
                "id": strategy_id,
                "route": str(candidate.get("route") or ""),
                "execute_now": bool(candidate.get("execute_now")),
            }
        )

    selected_strategy_id = str(payload.get("selected_strategy_id") or "").strip()
    if not selected_strategy_id:
        raise ValueError("Agent 响应必须包含 selected_strategy_id。")
    if selected_strategy_id not in seen_ids:
        raise ValueError(f"selected_strategy_id 未命中 strategy_candidates：{selected_strategy_id}")

    selected = next(item for item in normalized if item["id"] == selected_strategy_id)
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    decision_route = str(decision.get("route") or "")
    selected_route = str(selected.get("route") or "")
    route_matches = bool(
        decision_route
        and selected_route
        and (decision_route in selected_route or selected_route in decision_route)
    )
    return {
        "selected_strategy_id": selected_strategy_id,
        "selected_route": selected_route,
        "candidate_count": len(normalized),
        "execute_now_ids": [item["id"] for item in normalized if item["execute_now"]],
        "decision_route_matches_selected": route_matches,
    }


def _has_real_code_file(path: Path | None) -> bool:
    if path is None or not path.is_dir():
        return False
    for item in path.rglob("*"):
        if item.is_file() and not item.name.startswith("."):
            return True
    return False


def validate_code_artifacts(payload: dict[str, Any], *, code_root: Path | None = None) -> dict[str, Any]:
    """校验 Agent 生成代码是否满足当前 workflow 的最低产物契约。

    这不是代码质量评审，只防止 `code/` 退化成 README/manifest 集合。
    更深入的训练正确性仍由 tool executor、validation 和 memory 记录来判断。
    """

    experiment_plan = payload.get("experiment_plan")
    if not isinstance(experiment_plan, dict):
        raise ValueError("Agent 响应必须包含 experiment_plan，记录阶段、workflow、时间预算和精度/耗时权衡。")
    required_plan_fields = ["stage", "workflow_module", "budget", "time_budget", "score_tradeoff"]
    missing_plan_fields = [field for field in required_plan_fields if not str(experiment_plan.get(field) or "").strip()]
    if missing_plan_fields:
        raise ValueError(f"experiment_plan 缺少必填字段：{missing_plan_fields}")
    allowed_stages = {"baseline", "cheap_probe", "search", "promotion", "submission"}
    allowed_modules = {
        "rules_read",
        "data_shape_check",
        "baseline_replay",
        "checkpoint_finetune",
        "prediction_validation",
        "log_compliance",
        "submission_packaging",
    }
    if str(experiment_plan.get("stage")) not in allowed_stages:
        raise ValueError(f"experiment_plan.stage 不在允许范围内：{experiment_plan.get('stage')}")
    if str(experiment_plan.get("workflow_module")) not in allowed_modules:
        raise ValueError(f"experiment_plan.workflow_module 不在允许范围内：{experiment_plan.get('workflow_module')}")

    files = payload.get("files")
    artifacts = payload.get("code_artifacts")
    if files in (None, []) and artifacts in (None, {}):
        requested_tools = [
            str(request.get("tool") or "")
            for request in (payload.get("tool_requests") or [])
            if isinstance(request, dict)
        ]
        if "submission_package" in requested_tools:
            if not _has_real_code_file(code_root):
                raise ValueError(
                    "submission_package 需要当前实验 code/ 中有真实代码文件；"
                    "本轮必须输出 files 和 code_artifacts，生成一个 role=package 或 role=predict 的可执行 Python 入口。"
                )
        return {
            "primary_role": "strategy_only",
            "workflow_module": experiment_plan.get("workflow_module"),
            "enforced_modules": [],
            "entrypoints": [],
            "readme_manifest_only": False,
            "code_required": False,
            "note": "No code generated; this round uses existing controlled tools as evaluator.",
        }
    if not isinstance(files, list) or not files:
        raise ValueError("如果输出 code_artifacts，则必须同时包含非空 files。")
    if not isinstance(artifacts, dict):
        raise ValueError("如果输出 files，则必须同时包含 code_artifacts，声明本轮 code/ 的核心入口。")
    file_map: dict[str, str] = {}
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("files[] 必须是 object。")
        path = str(item.get("path") or "")
        content = str(item.get("content") or "")
        if path:
            file_map[path] = content
    entrypoints = artifacts.get("entrypoints")
    if not isinstance(entrypoints, list) or not entrypoints:
        raise ValueError("code_artifacts.entrypoints 必须是非空 list。")

    normalized_entries: list[dict[str, Any]] = []
    for entry in entrypoints:
        if not isinstance(entry, dict):
            raise ValueError("code_artifacts.entrypoints[] 必须是 object。")
        path = str(entry.get("path") or "")
        if path not in file_map:
            raise ValueError(f"code_artifacts entrypoint 未在 files 中生成：{path}")
        role = str(entry.get("role") or "")
        executable = bool(entry.get("is_executable"))
        content = file_map[path]
        normalized_entries.append(
            {
                "path": path,
                "role": role,
                "is_executable": executable,
                "looks_executable_python": _looks_executable_python(path, content),
            }
        )

    declared_module = str(experiment_plan.get("workflow_module") or "")
    primary_role = str(artifacts.get("primary_role") or "")
    primary_role_modules = {
        "checkpoint_finetune": "checkpoint_finetune",
        "prediction_validation": "prediction_validation",
        "submission_packaging": "submission_packaging",
    }
    inferred_modules = {declared_module} if declared_module else set()
    if primary_role in primary_role_modules:
        inferred_modules.add(primary_role_modules[primary_role])

    required_roles = {
        "checkpoint_finetune": {"train"},
        "prediction_validation": {"validate"},
        "submission_packaging": {"package", "predict"},
    }
    modules_to_check = sorted(module for module in inferred_modules if module in required_roles)
    for module in modules_to_check:
        candidates = [
            item
            for item in normalized_entries
            if item["role"] in required_roles[module] and item["is_executable"] and item["looks_executable_python"]
        ]
        if not candidates:
            roles = ", ".join(sorted(required_roles[module]))
            raise ValueError(
                f"{module} 阶段必须生成 role in {{{roles}}} 且 is_executable=true 的 Python 入口文件，"
                "不能只写 README/manifest/helper。"
            )
        if module == "checkpoint_finetune":
            train_content = "\n".join(file_map[item["path"]] for item in candidates)
            required_markers = ["temporal_stride", "spatial_downsample", "base_checkpoint"]
            missing = [marker for marker in required_markers if marker not in train_content]
            if missing:
                raise ValueError(f"checkpoint_finetune 训练入口缺少关键尺度/ckpt 标记：{missing}")

    readme_manifest_only = all(item["role"] == "manifest" for item in normalized_entries)
    return {
        "primary_role": artifacts.get("primary_role"),
        "workflow_module": declared_module,
        "enforced_modules": modules_to_check,
        "entrypoints": normalized_entries,
        "readme_manifest_only": readme_manifest_only,
    }


def process_raw_response(
    raw: str,
    config: dict[str, Any],
    log_dir: Path,
    run_id: str,
    code_root: Path,
    agent_log_path: Path,
    *,
    experiment_dir: Path | None = None,
    round_name: str | None = None,
) -> dict[str, Any]:
    """解析 Agent 原始响应、写入代码并生成 summary。"""

    payload = json.loads(strip_json_fence(raw))
    strategy_selection_review = validate_strategy_selection(payload)
    code_artifact_review = validate_code_artifacts(payload, code_root=code_root)
    tool_request_review = validate_tool_requests_for_workflow(payload)
    files = payload.get("files") or []
    written: list[str] = []
    if files:
        progress(f"开始写入 Agent 生成代码：{code_root}")
        written = write_agent_files(files, code_root)
    else:
        progress("本轮未生成新代码；将执行策略候选对应的受控工具。")
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id,
        "model": config.get("model"),
        "base_url": config.get("base_url"),
        "decision": payload.get("decision"),
        "strategy_candidates": payload.get("strategy_candidates", []),
        "selected_strategy_id": payload.get("selected_strategy_id"),
        "written_files": written,
        "planned_commands": payload.get("planned_commands", []),
        "tool_requests": payload.get("tool_requests", []),
        "experiment_plan": payload.get("experiment_plan", {}),
        "code_artifacts": payload.get("code_artifacts", {}),
        "strategy_selection_review": strategy_selection_review,
        "code_artifact_review": code_artifact_review,
        "tool_request_review": tool_request_review,
        "memory_update": payload.get("memory_update", {}),
        "runner_log_dir": str(log_dir),
        "generated_code_root": str(code_root),
        "agent_log": str(agent_log_path),
    }
    if experiment_dir is not None:
        summary["experiment_id"] = experiment_dir.name
        summary["experiment_dir"] = str(experiment_dir)
        summary["round_name"] = round_name
        summary["experiment_runs_dir"] = str(experiment_dir / "runs")
        summary["experiment_metrics_dir"] = str(experiment_dir / "metrics")
        summary["experiment_submission_dir"] = str(experiment_dir / "submission")
        summary["experiment_state"] = str(experiment_dir / "state.json")
    (log_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    progress(f"summary 已写入：{log_dir / 'summary.json'}")
    return summary


def call_agent(config: dict[str, Any], messages: list[dict[str, str]], *, api_error_log: Path | None = None) -> str:
    """通过官方 logging proxy 调用 GPT-5.5。

    `base_url` 应指向 `http://127.0.0.1:8080/v1`，proxy 再转发到第三方
    OpenAI-compatible API。真实请求内容会由 proxy 写入 `logs/`。
    """

    api_key_env, api_key = first_existing_env(list(config.get("api_key_env") or ["OPENAI_API_KEY"]))
    progress(f"使用 API key 环境变量：{api_key_env}")
    progress(f"请求模型：{config['model']}，base_url={config['base_url']}")
    client = OpenAI(
        api_key=api_key,
        base_url=str(config["base_url"]),
        timeout=float(config.get("timeout_seconds", 600)),
        max_retries=int(config.get("max_retries", 0)),
    )
    request_options = dict(config.get("request_options") or {})
    original_max_tokens = request_options.get("max_completion_tokens")
    retry_count = int(config.get("api_retries", config.get("transient_error_retries", 3)))
    max_attempts = max(1, retry_count + 1)
    retry_initial = float(config.get("api_retry_initial_seconds", 5.0))
    retry_max = float(config.get("api_retry_max_seconds", 45.0))
    response = None
    for attempt in range(1, max_attempts + 1):
        attempt_options = dict(request_options)
        if isinstance(original_max_tokens, int) and attempt > 1:
            attempt_options["max_completion_tokens"] = max(2048, int(original_max_tokens * (0.65 ** (attempt - 1))))
        try:
            progress(f"开始调用 LLM；如果这里等待较久，通常是代理/API 正在返回。attempt={attempt}/{max_attempts}")
            response = client.chat.completions.create(
                model=str(config["model"]),
                messages=messages,
                temperature=float(config.get("temperature", 0.2)),
                **attempt_options,
            )
            break
        except Exception as exc:
            retrying = attempt < max_attempts and is_transient_api_error(exc)
            if api_error_log is not None:
                append_api_error(api_error_log, attempt=attempt, max_attempts=max_attempts, exc=exc, retrying=retrying)
            if not retrying:
                raise
            sleep_seconds = min(retry_max, retry_initial * (2 ** (attempt - 1)))
            progress(
                f"LLM API 暂时失败：{type(exc).__name__} status={api_status_code(exc)}；"
                f"{sleep_seconds:.1f}s 后重试。"
            )
            time.sleep(sleep_seconds)
    if response is None:
        raise RuntimeError("LLM API 未返回 response。")
    if request_options.get("stream"):
        progress("LLM stream 已建立，下面会实时打印模型增量输出。")
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
        progress("LLM stream 已结束，开始解析。")
        content = "".join(parts)
        if not content:
            raise RuntimeError(f"Agent stream 返回为空，使用的 key 环境变量：{api_key_env}")
        return content
    progress("LLM 响应已返回，开始解析。")
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"Agent 返回为空，使用的 key 环境变量：{api_key_env}")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT-5.5 Agent once and write generated Task1 code.")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Agent API 配置 YAML。")
    parser.add_argument("--prompt", default=str(PROMPT_PATH), help="Agent system prompt。")
    parser.add_argument("--schema", default=str(SCHEMA_PATH), help="Agent JSON 输出 schema。")
    parser.add_argument("--goal", default="选择 Task1 当前最有希望的神经网络路线，并生成可运行的最小提交代码。")
    parser.add_argument("--route", default=None, help="可选 memory 检索路线过滤。默认让 Agent 自己决策。")
    parser.add_argument("--tag", action="append", default=[], help="可重复传入的 memory tag。")
    parser.add_argument("--memory-limit", type=int, default=8)
    parser.add_argument("--max-memory-chars", type=int, default=3500)
    parser.add_argument("--dry-run", action="store_true", help="只生成 prompt 上下文，不调用 API，不写 code。")
    parser.add_argument("--from-raw", default=None, help="复用已有 raw_response.txt 解析并写入 code，不重新调用 API。")
    parser.add_argument("--agent-log", default=None, help="本次实验专属 task1_logs.log；loop 会传入，runner/executor 共同追加。")
    parser.add_argument("--experiment-dir", default=None, help="本次多轮实验根目录；提供后所有轮次共用其中的 code/。")
    parser.add_argument("--round-index", type=int, default=None, help="本次实验轮次，用于 logs/turn_XXX 和 run_id。")
    parser.add_argument("--state-path", default=None, help="当前实验 state.json；用于约束历史 memory 只能作为策略提示。")
    args = parser.parse_args()

    progress("加载 .env 和 Agent 配置。")
    load_dotenv(ROOT / ".env")
    config = load_yaml(Path(args.config))
    paths = build_run_paths(args)

    if args.from_raw:
        raw_path = Path(args.from_raw)
        log_dir = paths["log_dir"] if args.experiment_dir else raw_path.parent
        run_id = str(paths["run_id"] if args.experiment_dir else log_dir.name)
        code_root = paths["code_root"] if args.experiment_dir else CODE_ROOT / run_id
        agent_log = paths["agent_log"] if args.experiment_dir else resolve_log_path(args.agent_log, fallback=log_dir / "task1_logs.log")
        progress(f"复用已有 Agent 原始响应：{raw_path}")
        raw = raw_path.read_text(encoding="utf-8")
        summary = process_raw_response(
            raw,
            config,
            log_dir,
            run_id,
            code_root,
            agent_log,
            experiment_dir=paths["experiment_dir"],
            round_name=paths["round_name"],
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    run_id = str(paths["run_id"])
    code_root = paths["code_root"]
    log_dir = paths["log_dir"]
    agent_log = paths["agent_log"]
    progress(f"读取 prompt：{args.prompt}")
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    progress("构造 compact memory 上下文。")
    context = compact_context_for_prompt(build_context(args, code_root, agent_log))
    schema = compact_action_schema_for_prompt(schema, context)
    messages = build_messages(prompt, schema, context)

    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "request_context.json").write_text(json.dumps({"messages": messages}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    progress(f"请求上下文已写入：{log_dir / 'request_context.json'}")

    if args.dry_run:
        print(json.dumps({"run_id": run_id, "dry_run": True, "request_context": str(log_dir / "request_context.json")}, ensure_ascii=False, indent=2))
        return

    started_at = datetime.now(timezone.utc)
    raw = call_agent(config, messages, api_error_log=log_dir / "api_errors.jsonl")
    elapsed_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
    (log_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
    progress(f"原始响应已写入：{log_dir / 'raw_response.txt'}")
    append_agent_response_log(raw, agent_log, code_root, elapsed_seconds)
    progress(f"本次实验 Agent 主日志已追加：{agent_log}")
    summary = process_raw_response(
        raw,
        config,
        log_dir,
        run_id,
        code_root,
        agent_log,
        experiment_dir=paths["experiment_dir"],
        round_name=paths["round_name"],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
