#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def progress(message: str) -> None:
    print(f"[agent-loop] {message}", flush=True)


def run_stream(command: list[str], *, cwd: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)
    returncode = process.wait()
    return {
        "command": command,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "returncode": returncode,
        "output": "".join(output_lines),
    }


def latest_summary(experiment_dir: Path | None = None) -> Path | None:
    if experiment_dir is not None:
        candidates = sorted((experiment_dir / "logs").glob("turn_*/summary.json"))
        return candidates[-1] if candidates else None
    candidates = sorted((ROOT / "agent_workspace" / "logs").glob("agent_*/summary.json"))
    return candidates[-1] if candidates else None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return load_json(path)


def write_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, state)


def build_initial_state(
    *,
    experiment_id: str,
    experiment_dir: Path,
    max_rounds: int,
    goal: str,
    started_at: str,
) -> dict[str, Any]:
    best = best_leaderboard()
    history_hints = []
    if best:
        history_hints.append(
            {
                "source": "memory/findings/metric_leaderboard.csv",
                "slot": best.get("slot"),
                "record_id": best.get("record_id"),
                "metric": best.get("metric"),
                "value": best.get("value"),
                "strategy_hint_only": True,
                "usable_as_current_artifact": False,
                "reason": (
                    "历史结果不能直接作为当前 submission 的 prediction/log；"
                    "高分历史 FNO checkpoint 可作为 finetune_local.base_checkpoint warm-start，"
                    "但必须在当前实验内继续训练并产出新 best.pt 后再 replay/test/package。"
                ),
            }
        )
    return {
        "schema": "task1_experiment_state_v1",
        "experiment_id": experiment_id,
        "stage": "fresh_experiment",
        "created_at": started_at,
        "updated_at": started_at,
        "max_rounds": max_rounds,
        "current_round": 0,
        "goal": goal,
        "paths": {
            "experiment_dir": relative_to_root(experiment_dir),
            "code_dir": relative_to_root(experiment_dir / "code"),
            "logs_dir": relative_to_root(experiment_dir / "logs"),
            "runs_dir": relative_to_root(experiment_dir / "runs"),
            "metrics_dir": relative_to_root(experiment_dir / "metrics"),
            "submission_dir": relative_to_root(experiment_dir / "submission"),
        },
        "artifact_policy": {
            "current_experiment_only": True,
            "historical_memory_as_strategy_only": True,
            "forbidden_as_submission_artifacts": ["historical runs/", "historical agent_workspace/experiments/"],
            "allowed_external_inputs": ["data/", "checkpoints/official/"],
        },
        "history_hints": history_hints,
        "local_artifacts": [],
        "best_local_candidate": None,
        "strategy_db": {
            "schema": "task1_strategy_db_v1",
            "programs": {},
            "archive": [],
            "best_strategy_id": None,
            "num_islands": 3,
            "current_island": 0,
            "islands": {"0": [], "1": [], "2": []},
            "active_parent_id": None,
            "active_inspiration_ids": [],
            "sampling": {
                "exploration_ratio": 0.35,
                "exploitation_ratio": 0.45,
                "random_ratio": 0.20,
            },
            "feature_dimensions": ["workflow_module", "trainable", "sample_budget", "rollout_steps"],
            "feature_map": {},
        },
        "blockers": ["need_current_experiment_candidate", "need_validation", "need_submission_package"],
        "rounds": [],
    }


def record_round_state(state_path: Path, record: dict[str, Any], *, stage: str | None = None) -> None:
    state = load_state(state_path)
    state.setdefault("rounds", []).append(
        {
            "round": record.get("round"),
            "run_id": record.get("run_id"),
            "summary": record.get("summary"),
            "executor_log": record.get("executor_log"),
            "executor_failed": record.get("executor_failed"),
            "decision": record.get("decision"),
            "selected_strategy_id": record.get("selected_strategy_id"),
            "tool_count": record.get("tool_count"),
        }
    )
    state["current_round"] = record.get("round", state.get("current_round", 0))
    if stage:
        state["stage"] = stage
    write_state(state_path, state)


def _strategy_score(program: dict[str, Any]) -> float:
    metrics = program.get("metrics") if isinstance(program.get("metrics"), dict) else {}
    for key in ("combined_score", "competition_score_proxy", "best_score", "score"):
        value = metrics.get(key, program.get(key))
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
    return float("-inf")


def _strategy_brief(program: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": program.get("id"),
        "parent_id": program.get("parent_id"),
        "generation": program.get("generation"),
        "status": program.get("status"),
        "route": program.get("route"),
        "workflow_module": program.get("workflow_module"),
        "hypothesis": program.get("hypothesis"),
        "metrics": program.get("metrics") or {},
        "artifacts": program.get("artifacts") or {},
        "error": program.get("error"),
    }


def _strategy_db(state: dict[str, Any]) -> dict[str, Any]:
    db = state.setdefault("strategy_db", {})
    db.setdefault("schema", "task1_strategy_db_v1")
    db.setdefault("programs", {})
    db.setdefault("archive", [])
    db.setdefault("best_strategy_id", None)
    db.setdefault("num_islands", 3)
    db.setdefault("current_island", 0)
    db.setdefault("islands", {str(index): [] for index in range(int(db.get("num_islands") or 3))})
    db.setdefault("active_parent_id", None)
    db.setdefault("active_inspiration_ids", [])
    db.setdefault(
        "sampling",
        {"exploration_ratio": 0.35, "exploitation_ratio": 0.45, "random_ratio": 0.20},
    )
    db.setdefault("feature_map", {})
    return db


def sample_strategy_context(state_path: Path, *, round_index: int) -> dict[str, Any]:
    state = load_state(state_path)
    db = _strategy_db(state)
    programs = db.get("programs") if isinstance(db.get("programs"), dict) else {}
    evaluated = [
        item
        for item in programs.values()
        if isinstance(item, dict) and item.get("status") in {"evaluated", "failed"}
    ]
    top = sorted(evaluated, key=_strategy_score, reverse=True)[:5]

    parent: dict[str, Any] | None = None
    sampling_mode = "seed"
    if evaluated:
        sampling = db.get("sampling") if isinstance(db.get("sampling"), dict) else {}
        draw = random.random()
        exploration_ratio = float(sampling.get("exploration_ratio", 0.35))
        exploitation_ratio = float(sampling.get("exploitation_ratio", 0.45))
        best_id = db.get("best_strategy_id")
        if draw < exploitation_ratio and best_id in programs:
            parent = programs[str(best_id)]
            sampling_mode = "exploitation_best"
        elif draw < exploitation_ratio + exploration_ratio:
            island_key = str(db.get("current_island", 0))
            island_ids = [item for item in db.get("islands", {}).get(island_key, []) if item in programs]
            if island_ids:
                parent = programs[random.choice(island_ids)]
                sampling_mode = f"exploration_island_{island_key}"
        if parent is None:
            parent = random.choice(evaluated)
            sampling_mode = "random"

    parent_id = str(parent.get("id")) if parent else None
    inspirations = [item for item in top if item.get("id") != parent_id][:3]
    db["active_parent_id"] = parent_id
    db["active_inspiration_ids"] = [str(item.get("id")) for item in inspirations if item.get("id")]
    db["last_sampling_mode"] = sampling_mode
    db["current_island"] = int(round_index % int(db.get("num_islands") or 3))
    state["strategy_db"] = db
    write_state(state_path, state)
    return {
        "schema": "task1_strategy_context_v1",
        "sampling_mode": sampling_mode,
        "parent": _strategy_brief(parent) if parent else None,
        "top_strategies": [_strategy_brief(item) for item in top[:3]],
        "inspirations": [_strategy_brief(item) for item in inspirations],
        "strategy_pool_size": len(programs),
        "best_strategy_id": db.get("best_strategy_id"),
    }


def parse_stdout_json(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
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


def compact_executor_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("tool") == "finetuned_checkpoint_replay":
        metrics = result.get("metrics") or {}
        return {
            "tool": "finetuned_checkpoint_replay",
            "status": result.get("status"),
            "run_dir": result.get("run_dir"),
            "score": metrics.get("competition_score_proxy"),
            "prediction_path": result.get("prediction_path"),
            "promoted_to": result.get("promoted_to"),
        }
    if result.get("tool") in {"data_shape_check", "web_search"}:
        return {"tool": result.get("tool"), "status": result.get("status"), "output": result.get("output")}
    if result.get("status") in {"failed", "rejected"}:
        return {"tool": result.get("tool"), "status": result.get("status"), "error": result.get("error") or result.get("reason")}

    command = result.get("command")
    stdout_payload = parse_stdout_json(result.get("stdout")) if isinstance(result.get("stdout"), str) else None
    command_name = ""
    if isinstance(command, list) and command:
        command_name = " ".join(str(item) for item in command[:4])
    if stdout_payload:
        metrics = stdout_payload.get("metrics") or {}
        return {
            "command": command_name,
            "returncode": result.get("returncode"),
            "run_dir": stdout_payload.get("run_dir"),
            "score": stdout_payload.get("best_score") or metrics.get("competition_score_proxy"),
            "best_checkpoint": stdout_payload.get("best_checkpoint"),
            "elapsed_seconds": stdout_payload.get("elapsed_seconds"),
            "memory_export": result.get("memory_export"),
            "promoted_to": result.get("promoted_to"),
        }
    return {"command": command_name, "returncode": result.get("returncode")}


def summarize_executor_log(log_path: Path, *, max_chars: int = 2400) -> str:
    if not log_path.is_file():
        return "executor_log 不存在。"
    try:
        payload = load_json(log_path)
    except Exception as exc:
        return f"executor_log 读取失败：{type(exc).__name__}: {exc}"
    compact = [compact_executor_result(item) for item in payload.get("results") or [] if isinstance(item, dict)]
    text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return text


def tail_text(text: str | None, *, max_chars: int = 1800) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[-max_chars:]


def best_leaderboard() -> dict[str, Any] | None:
    path = ROOT / "memory" / "findings" / "metric_leaderboard.csv"
    if not path.is_file():
        return None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                row["_value_float"] = float(row.get("value") or "nan")
            except ValueError:
                continue
            rows.append(row)
    if not rows:
        return None
    rows.sort(key=lambda item: float(item["_value_float"]), reverse=True)
    best = rows[0]
    return {
        "slot": best.get("slot"),
        "record_id": best.get("record_id"),
        "metric": best.get("metric"),
        "value": float(best["_value_float"]),
        "submit_ready": best.get("submit_ready"),
        "blockers": best.get("blockers"),
        "updated_at": best.get("updated_at"),
    }


def extract_json_payload(text: str) -> dict[str, Any] | None:
    """Extract the last complete JSON object from mixed runner stdout."""

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


def summary_from_runner_output(output: str) -> Path | None:
    payload = extract_json_payload(output)
    if not payload:
        return None
    runner_log_dir = payload.get("runner_log_dir")
    if runner_log_dir:
        candidate = Path(str(runner_log_dir))
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        summary = candidate / "summary.json"
        return summary if summary.is_file() else None
    run_id = payload.get("run_id")
    if not run_id:
        return None
    candidate = ROOT / "agent_workspace" / "logs" / str(run_id) / "summary.json"
    return candidate if candidate.is_file() else None


def score_band(
    score: float | None,
    *,
    baseline_score: float,
    prior_probe_score: float,
    strong_score: float,
    excellent_score: float,
    full_score: float,
) -> str:
    if score is None:
        return "no_validated_score"
    if score < baseline_score:
        return "below_unet_baseline"
    if score < prior_probe_score:
        return "above_baseline_below_prior_probe"
    if score < strong_score:
        return "promising_below_strong_score"
    if score < excellent_score:
        return "strong_candidate_below_excellent"
    if score < full_score:
        return "excellent_candidate_below_full_score"
    return "near_or_above_full_score"


def build_round_goal(
    args: argparse.Namespace,
    round_index: int,
    previous: dict[str, Any] | None,
    *,
    strategy_context: dict[str, Any] | None = None,
) -> str:
    best = best_leaderboard()
    score = float(best["value"]) if best else None
    band = score_band(
        score,
        baseline_score=args.baseline_score,
        prior_probe_score=args.prior_probe_score,
        strong_score=args.strong_score,
        excellent_score=args.excellent_score,
        full_score=args.full_score,
    )
    previous_text = "无上一轮。"
    if previous:
        executor_summary = summarize_executor_log(Path(str(previous.get("executor_log") or "")))
        previous_text = (
            f"上一轮 run_id={previous.get('run_id')}，summary={previous.get('summary')}，"
            f"executor_log={previous.get('executor_log')}，executor_failed={previous.get('executor_failed')}，"
            f"executor_results={executor_summary}。"
        )
    strategy_text = json.dumps(strategy_context or {}, ensure_ascii=False, separators=(",", ":"))

    return (
        f"{args.goal}\n"
        f"\n[自动科研闭环控制]\n"
        f"- 当前是第 {round_index}/{args.max_rounds} 轮。必须继续执行本轮；不要因为 submit_ready 或当前已有高分候选而停止。\n"
        f"- loop 的硬停止条件只有 max_rounds 跑满，或 API/文件系统/工具执行出现不可恢复错误。\n"
        f"- 当前 leaderboard best={score if score is not None else 'NA'}，score_band={band}。\n"
        f"- Task1 最终目标是在合规约束下逼近满分：总分上限 {args.task_total_full_score}，"
        f"其中预测分来自分段预测满分 {args.full_score}，同时训练耗时和推理耗时也计入总分。"
        f"不是满足某个中间阈值。"
        f"strong_score/excellent_score 只是证据强度分层线，不是路线指令、不是优化终点、也不作为提前停止条件："
        f"official_unet_baseline={args.baseline_score}；prior_probe={args.prior_probe_score}；"
        f"strong_score={args.strong_score}；excellent_score={args.excellent_score}；full_score={args.full_score}。\n"
        f"- 时间预算是评分约束：训练耗时 <= {args.train_full_score_seconds}s 可拿满训练耗时分；"
        f"推理耗时必须控制在 {args.inference_hard_limit_seconds}s 内，否则 Task1 该任务得 0 分。"
        f"你可以自由探索模型路线，但每轮实验都要显式权衡精度收益、训练成本和推理成本。\n"
        f"- score_band 只提供当前证据成熟度。你需要基于边际收益、失败风险、验证可信度和合规风险，"
        f"自主选择继续探索、利用当前候选、强化验证、修复工程链路或准备提交；不要机械套用固定流程。\n"
        f"- 本系统现在按 OpenEvolve 式策略池运行：每轮从 strategy_db 采样 parent/inspirations，"
        f"你输出的 strategy_candidates 是 child strategy mutations；executor 评估后会把 metrics/artifacts/error 写回池子。\n"
        f"- 本轮 strategy_context={strategy_text}\n"
        f"- 常规 FNO 微调、replay、validation 优先使用白名单工具和已有脚本；只有提出新训练/推理/打包代码时才生成 files/code_artifacts。\n"
        f"- 如果本轮使用 finetune_local，请优先同时规划 finetuned_checkpoint_replay，把 best.pt 接到标准 prediction/memory 链路。\n"
        f"- 一轮可以先 checkpoint_replay 建当前实验 baseline，再 finetune_local 做 selected_strategy_id 对应的 cheap probe；不要因为 workflow 标签机械拒绝合理组合。\n"
        f"- 进入 submission_package 前必须先有当前实验 split=test 的 replay run；val run 只能验证分数，不能直接打包提交。\n"
        f"- direct API 模式下合规 LLM 日志优先使用当前实验 logs/task1_logs.log，不要依赖 8080 proxy 日志。\n"
        f"- {previous_text}"
    )


def executor_log_for(summary_path: Path) -> Path:
    payload = load_json(summary_path)
    run_id = str(payload.get("run_id") or summary_path.parent.name)
    experiment_dir = payload.get("experiment_dir")
    if experiment_dir:
        base = Path(str(experiment_dir))
        if not base.is_absolute():
            base = ROOT / base
        return base / "logs" / "executor" / f"{run_id}.json"
    return ROOT / "agent_workspace" / "executor_logs" / f"{run_id}.json"


def executor_failed(log_path: Path) -> bool:
    if not log_path.is_file():
        return True
    try:
        payload = load_json(log_path)
    except Exception:
        return True
    for result in payload.get("results") or []:
        if not isinstance(result, dict):
            return True
        if result.get("returncode") not in (None, 0):
            return True
        if result.get("status") in {"failed", "rejected"}:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-round Task1 Agent research loop.")
    parser.add_argument("--config", default="configs/agent_gpt55.yaml")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--executor-timeout", type=int, default=3600)
    parser.add_argument("--goal", default="自动推进 Task1 PDE_Burgers 科研闭环：根据 memory 和上一轮反馈选择下一步，生成必要代码，调用白名单工具，并把结果写入 memory。")
    parser.add_argument("--baseline-score", type=float, default=13.3233129395)
    parser.add_argument("--prior-probe-score", type=float, default=18.8862194195)
    parser.add_argument("--task-total-full-score", type=float, default=150.0, help="Task1 总分满分。")
    parser.add_argument("--full-score", type=float, default=100.0, help="Task1 validation proxy 的满分目标；loop 不会因达到该分数提前停止。")
    parser.add_argument("--target-score", type=float, default=None, help="兼容旧参数；提供时等价于覆盖 --full-score。")
    parser.add_argument("--strong-score", type=float, default=None, help="证据强度分层线；默认 max(prior_probe_score, 0.5 * full_score)。")
    parser.add_argument("--excellent-score", type=float, default=None, help="高置信候选分层线；默认 0.9 * full_score。")
    parser.add_argument("--train-full-score-seconds", type=float, default=3600.0, help="Task1 训练耗时满分线，默认 60 分钟。")
    parser.add_argument("--inference-hard-limit-seconds", type=float, default=120.0, help="Task1 推理硬上限，默认 2 分钟。")
    parser.add_argument("--runner-retries", type=int, default=2, help="LLM 未返回合格 JSON 时的同轮重试次数。")
    parser.add_argument("--stop-on-executor-failure", action="store_true", help="工具失败时立即停止；默认继续到 max-rounds，让 Agent 基于失败日志修复。")
    args = parser.parse_args()

    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be >= 1")
    if args.runner_retries < 0:
        raise ValueError("--runner-retries must be >= 0")
    if args.target_score is not None:
        args.full_score = args.target_score
    args.strong_score = args.strong_score if args.strong_score is not None else max(args.prior_probe_score, 0.5 * args.full_score)
    args.excellent_score = args.excellent_score if args.excellent_score is not None else 0.9 * args.full_score
    if args.full_score <= 0:
        raise ValueError("--full-score must be > 0")
    if args.strong_score > args.full_score:
        raise ValueError("--strong-score must be <= --full-score")
    if args.excellent_score > args.full_score:
        raise ValueError("--excellent-score must be <= --full-score")
    if args.strong_score > args.excellent_score:
        raise ValueError("--strong-score must be <= --excellent-score")

    loop_id = f"task1_{utc_stamp()}"
    experiment_dir = ROOT / "agent_workspace" / "experiments" / loop_id
    for child in ("code", "logs", "runs", "metrics", "submission"):
        (experiment_dir / child).mkdir(parents=True, exist_ok=True)
    loop_dir = experiment_dir
    rounds_path = experiment_dir / "logs" / "rounds.jsonl"
    agent_log_path = experiment_dir / "logs" / "task1_logs.log"
    agent_log_path.write_text("", encoding="utf-8")
    loop_started_at = datetime.now(timezone.utc).isoformat()
    state_path = experiment_dir / "state.json"
    experiment_config = {
        "experiment_id": loop_id,
        "task": "task1",
        "created_at": loop_started_at,
        "max_rounds": args.max_rounds,
        "config": args.config,
        "goal": args.goal,
        "layout": {
            "code": "code/",
            "logs": "logs/",
            "runs": "runs/",
            "metrics": "metrics/",
            "submission": "submission/",
        },
        "policy": "One experiment owns one code directory; every round modifies code/ and writes logs/runs under this experiment.",
    }
    (experiment_dir / "experiment.yaml").write_text(
        yaml.safe_dump(experiment_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    write_state(
        state_path,
        build_initial_state(
            experiment_id=loop_id,
            experiment_dir=experiment_dir,
            max_rounds=args.max_rounds,
            goal=args.goal,
            started_at=loop_started_at,
        ),
    )
    progress(f"experiment_id={loop_id}, max_rounds={args.max_rounds}")
    progress(f"experiment dir: {experiment_dir}")
    progress(f"state: {state_path}")
    progress(f"round log: {rounds_path}")
    progress(f"agent main log: {agent_log_path}")

    previous: dict[str, Any] | None = None
    round_records: list[dict[str, Any]] = []
    for round_index in range(1, args.max_rounds + 1):
        state = load_state(state_path)
        state["current_round"] = round_index
        state["stage"] = state.get("stage") or "fresh_experiment"
        write_state(state_path, state)
        strategy_context = sample_strategy_context(state_path, round_index=round_index)
        progress(f"开始第 {round_index}/{args.max_rounds} 轮")
        goal = build_round_goal(args, round_index, previous, strategy_context=strategy_context)
        before_summary = latest_summary(experiment_dir)
        runner: dict[str, Any] | None = None
        after_summary: Path | None = None
        for attempt in range(args.runner_retries + 1):
            attempt_goal = goal
            if attempt:
                previous_runner_output = tail_text(runner.get("output") if isinstance(runner, dict) else "")
                attempt_goal = (
                    goal
                    + "\n\n[格式重试]\n"
                    + "- 上一次 LLM 响应没有生成符合 action_schema / code_artifact_contract / tool whitelist 的可执行结果。\n"
                    + f"- 上一次 runner 错误尾部如下，请针对性修复，不要重复同类错误：\n{previous_runner_output}\n"
                    + "- 本次必须只输出一个 JSON object；不要输出 <think>、Markdown、解释性文字或代码块围栏。\n"
                    + "- 如果本轮 workflow_module=checkpoint_finetune，训练入口源码必须显式包含 base_checkpoint、temporal_stride、spatial_downsample 这三个标记。\n"
                )
                progress(f"第 {round_index} 轮 runner 第 {attempt + 1} 次尝试：强制 JSON-only 重试。")
            runner = run_stream(
                [
                    "bash",
                    "scripts/run_in_env.sh",
                    "scripts/task1_agent_runner.py",
                    "--config",
                    args.config,
                    "--goal",
                    attempt_goal,
                    "--agent-log",
                    str(agent_log_path.relative_to(ROOT)),
                    "--experiment-dir",
                    str(experiment_dir.relative_to(ROOT)),
                    "--round-index",
                    str(round_index),
                    "--state-path",
                    str(state_path.relative_to(ROOT)),
                ],
                cwd=ROOT,
            )
            after_summary = summary_from_runner_output(runner["output"]) or latest_summary(experiment_dir)
            if runner["returncode"] == 0 and after_summary is not None and after_summary != before_summary:
                break
            progress(f"第 {round_index} 轮 runner 尝试 {attempt + 1} 未生成新 summary。")
        assert runner is not None
        if runner["returncode"] != 0 or after_summary is None or after_summary == before_summary:
            record = {
                "round": round_index,
                "runner": runner,
                "summary": str(after_summary) if after_summary else None,
                "error": "runner_failed_or_no_new_summary",
            }
            rounds_path.open("a", encoding="utf-8").write(json.dumps(record, ensure_ascii=False) + "\n")
            round_records.append(record)
            record_round_state(state_path, record, stage="runner_failed")
            progress("runner 未生成新 summary，停止 loop。")
            break

        progress(f"执行第 {round_index} 轮 tool_requests: {after_summary}")
        executor = run_stream(
            [
                "bash",
                "scripts/run_in_env.sh",
                "scripts/task1_tool_executor.py",
                "--summary",
                str(after_summary.relative_to(ROOT)),
                "--timeout",
                str(args.executor_timeout),
                "--log-start-timestamp",
                loop_started_at,
                "--agent-log",
                str(agent_log_path.relative_to(ROOT)),
                "--runs-root",
                str((experiment_dir / "runs").relative_to(ROOT)),
                "--tool-output-root",
                str((experiment_dir / "runs" / "tool_outputs").relative_to(ROOT)),
                "--executor-log-dir",
                str((experiment_dir / "logs" / "executor").relative_to(ROOT)),
                "--submission-root",
                str(experiment_dir.relative_to(ROOT)),
                "--experiment-dir",
                str(experiment_dir.relative_to(ROOT)),
                "--state-path",
                str(state_path.relative_to(ROOT)),
            ],
            cwd=ROOT,
        )
        log_path = executor_log_for(after_summary)
        failed = executor["returncode"] != 0 or executor_failed(log_path)
        summary_payload = load_json(after_summary)
        best = best_leaderboard()
        record = {
            "round": round_index,
            "run_id": summary_payload.get("run_id"),
            "summary": str(after_summary),
            "executor_log": str(log_path),
            "runner_returncode": runner["returncode"],
            "executor_returncode": executor["returncode"],
            "executor_failed": failed,
            "decision": summary_payload.get("decision"),
            "selected_strategy_id": summary_payload.get("selected_strategy_id"),
            "strategy_selection_review": summary_payload.get("strategy_selection_review"),
            "experiment_plan": summary_payload.get("experiment_plan"),
            "tool_count": len(summary_payload.get("tool_requests") or []),
            "agent_log": str(agent_log_path),
            "experiment_dir": str(experiment_dir),
            "best_leaderboard": best,
        }
        rounds_path.open("a", encoding="utf-8").write(json.dumps(record, ensure_ascii=False) + "\n")
        round_records.append(record)
        record_round_state(state_path, record, stage="tool_failed" if failed else "round_completed")
        previous = record
        progress(f"第 {round_index} 轮完成；executor_failed={failed}；best={best}")
        if failed and args.stop_on_executor_failure:
            progress("工具执行失败，且启用了 --stop-on-executor-failure，停止 loop。")
            break

    final_report = {
        "loop_id": loop_id,
        "experiment_id": loop_id,
        "experiment_dir": str(experiment_dir),
        "max_rounds": args.max_rounds,
        "loop_started_at": loop_started_at,
        "attempted_rounds": len(round_records),
        "completed_rounds": len([item for item in round_records if item.get("run_id")]),
        "rounds_path": str(rounds_path),
        "agent_log": str(agent_log_path),
        "final_best": best_leaderboard(),
        "rounds": round_records,
    }
    leaderboard = ROOT / "memory" / "findings" / "metric_leaderboard.csv"
    if leaderboard.is_file():
        shutil.copy2(leaderboard, experiment_dir / "metrics" / "leaderboard.csv")
    (experiment_dir / "metrics" / "summary.json").write_text(json.dumps(final_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (experiment_dir / "logs" / "summary.json").write_text(json.dumps(final_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
