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
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai4sv2_task1.memory import query_memory


PROMPT_PATH = ROOT / "agent_workspace" / "prompts" / "task1_planner.md"
SCHEMA_PATH = ROOT / "agent_workspace" / "prompts" / "action_schema.json"
CONFIG_PATH = ROOT / "configs" / "agent_gpt55.yaml"
CODE_ROOT = ROOT / "agent_workspace" / "code"
RUNNER_LOG_ROOT = ROOT / "agent_workspace" / "logs"


def progress(message: str) -> None:
    """向终端打印 Agent runner 阶段进度。

    这里使用 `flush=True`，避免长时间 API 请求前的提示被缓冲，看起来像卡住。
    """

    print(f"[agent] {message}", flush=True)


def utc_stamp() -> str:
    """生成稳定的 UTC 时间戳，用于 runner 日志目录命名。"""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


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
    return extract_first_json_object(stripped)


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


def build_context(args: argparse.Namespace, generated_code_root: Path) -> dict[str, Any]:
    """构造给 Agent 的最小上下文。

    关键原则：只放 memory 的小型 retrieval packet、路径摘要和目标，不把 harness
    源码或旧仓库源码塞进 prompt。
    """

    memory_packet = query_memory(route=args.route, tags=args.tag, limit=args.memory_limit, max_chars=args.max_memory_chars)
    return {
        "task_root": str(ROOT),
        "generated_code_root": generated_code_root.relative_to(ROOT).as_posix(),
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
                "Agent 可决策 steps/lr/rollout/trainable，也可用 trainable_modules 自由选择 FNO 模块："
                "fc0, conv0, w0, conv1, w1, conv2, w2, conv3, w3, fc1, fc2。"
                "Agent 触发的实验目录统一写入 runs/task1/<agent_run_id>__toolXX。"
            ),
            "llm_log_prepare": "把官方 proxy JSONL 转换为比赛要求的 task1_logs.log JSONL。",
            "submission_package": "从已通过校验的 run 目录生成 submission 目录，默认复制本轮 Agent code 快照。",
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
                "expected_artifact": "agent_workspace/tool_outputs/data_shape/*.json",
            },
            {
                "name": "baseline_replay",
                "purpose": "运行官方 FNO 和 Unet-PF baseline，建立对照分数。",
                "tool": "checkpoint_replay",
                "expected_artifact": "runs/task1/<agent_run_id>__toolXX/",
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
                "purpose": "整理官方 proxy LLM 调用日志为比赛 JSONL。",
                "tool": "llm_log_prepare",
                "expected_artifact": "task1_logs.log",
            },
            {
                "name": "submission_packaging",
                "purpose": "把通过校验的 run、合规 log、本轮 Agent code 快照打包。",
                "tool": "submission_package",
                "expected_artifact": "submissions/<name>/",
            },
        ],
        "code_artifact_contract": {
            "purpose": "防止 Agent 只生成 README/manifest，而没有与实验阶段对应的核心代码。",
            "required_top_level_field": "code_artifacts",
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
            "bash scripts/run_in_env.sh scripts/task1_make_submission.py --run-dir <run_dir> --submission-name <name> --llm-log <task1_logs.log>",
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


def validate_code_artifacts(payload: dict[str, Any]) -> dict[str, Any]:
    """校验 Agent 生成代码是否满足当前 workflow 的最低产物契约。

    这不是代码质量评审，只防止 `code/` 退化成 README/manifest 集合。
    更深入的训练正确性仍由 tool executor、validation 和 memory 记录来判断。
    """

    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Agent 响应必须包含非空 files。")
    file_map: dict[str, str] = {}
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("files[] 必须是 object。")
        path = str(item.get("path") or "")
        content = str(item.get("content") or "")
        if path:
            file_map[path] = content

    artifacts = payload.get("code_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Agent 响应必须包含 code_artifacts，声明本轮 code/ 的核心入口。")
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
    tool_modules = {
        "finetune_local": "checkpoint_finetune",
        "validation": "prediction_validation",
        "submission_package": "submission_packaging",
    }
    inferred_modules = {declared_module} if declared_module else set()
    if primary_role in primary_role_modules:
        inferred_modules.add(primary_role_modules[primary_role])
    for request in payload.get("tool_requests") or []:
        if isinstance(request, dict):
            module_from_tool = tool_modules.get(str(request.get("tool") or ""))
            if module_from_tool:
                inferred_modules.add(module_from_tool)

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


def process_raw_response(raw: str, config: dict[str, Any], log_dir: Path, run_id: str, code_root: Path) -> dict[str, Any]:
    """解析 Agent 原始响应、写入代码并生成 summary。"""

    payload = json.loads(strip_json_fence(raw))
    code_artifact_review = validate_code_artifacts(payload)
    progress(f"开始写入 Agent 生成代码：{code_root}")
    written = write_agent_files(payload["files"], code_root)
    summary = {
        "run_id": run_id,
        "model": config.get("model"),
        "base_url": config.get("base_url"),
        "decision": payload.get("decision"),
        "written_files": written,
        "planned_commands": payload.get("planned_commands", []),
        "tool_requests": payload.get("tool_requests", []),
        "experiment_plan": payload.get("experiment_plan", {}),
        "code_artifacts": payload.get("code_artifacts", {}),
        "code_artifact_review": code_artifact_review,
        "memory_update": payload.get("memory_update", {}),
        "runner_log_dir": str(log_dir),
        "generated_code_root": str(code_root),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    progress(f"summary 已写入：{log_dir / 'summary.json'}")
    return summary


def call_agent(config: dict[str, Any], messages: list[dict[str, str]]) -> str:
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
    progress("开始调用 LLM；如果这里等待较久，通常是代理/API 正在返回。")
    response = client.chat.completions.create(
        model=str(config["model"]),
        messages=messages,
        temperature=float(config.get("temperature", 0.2)),
        **request_options,
    )
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
    parser.add_argument("--max-memory-chars", type=int, default=6000)
    parser.add_argument("--dry-run", action="store_true", help="只生成 prompt 上下文，不调用 API，不写 code。")
    parser.add_argument("--from-raw", default=None, help="复用已有 raw_response.txt 解析并写入 code，不重新调用 API。")
    args = parser.parse_args()

    progress("加载 .env 和 Agent 配置。")
    load_dotenv(ROOT / ".env")
    config = load_yaml(Path(args.config))

    if args.from_raw:
        raw_path = Path(args.from_raw)
        log_dir = raw_path.parent
        run_id = log_dir.name
        code_root = CODE_ROOT / run_id
        progress(f"复用已有 Agent 原始响应：{raw_path}")
        raw = raw_path.read_text(encoding="utf-8")
        summary = process_raw_response(raw, config, log_dir, run_id, code_root)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    run_id = f"agent_{utc_stamp()}"
    code_root = CODE_ROOT / run_id
    progress(f"读取 prompt：{args.prompt}")
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    progress("构造 compact memory 上下文。")
    context = build_context(args, code_root)
    messages = build_messages(prompt, schema, context)

    log_dir = RUNNER_LOG_ROOT / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "request_context.json").write_text(json.dumps({"messages": messages}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    progress(f"请求上下文已写入：{log_dir / 'request_context.json'}")

    if args.dry_run:
        print(json.dumps({"run_id": run_id, "dry_run": True, "request_context": str(log_dir / "request_context.json")}, ensure_ascii=False, indent=2))
        return

    raw = call_agent(config, messages)
    (log_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
    progress(f"原始响应已写入：{log_dir / 'raw_response.txt'}")
    summary = process_raw_response(raw, config, log_dir, run_id, code_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
