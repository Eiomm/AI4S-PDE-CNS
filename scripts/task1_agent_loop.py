#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def latest_summary() -> Path | None:
    candidates = sorted((ROOT / "agent_workspace" / "logs").glob("agent_*/summary.json"))
    return candidates[-1] if candidates else None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def build_round_goal(args: argparse.Namespace, round_index: int, previous: dict[str, Any] | None) -> str:
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
        previous_text = (
            f"上一轮 run_id={previous.get('run_id')}，summary={previous.get('summary')}，"
            f"executor_log={previous.get('executor_log')}，executor_failed={previous.get('executor_failed')}。"
        )

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
        f"- 每轮必须输出 code_artifacts；进入 checkpoint_finetune 时必须生成可执行 train 入口，不能只写 README/manifest。\n"
        f"- 如果本轮使用 finetune_local，请优先同时规划 finetuned_checkpoint_replay，把 best.pt 接到标准 prediction/memory 链路。\n"
        f"- {previous_text}"
    )


def executor_log_for(summary_path: Path) -> Path:
    payload = load_json(summary_path)
    run_id = str(payload.get("run_id") or summary_path.parent.name)
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
    parser.add_argument("--stop-on-executor-failure", action="store_true", help="工具失败时立即停止；默认继续到 max-rounds，让 Agent 基于失败日志修复。")
    args = parser.parse_args()

    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be >= 1")
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

    loop_id = f"loop_{utc_stamp()}"
    loop_dir = ROOT / "agent_workspace" / "loop_logs" / loop_id
    loop_dir.mkdir(parents=True, exist_ok=True)
    rounds_path = loop_dir / "rounds.jsonl"
    progress(f"loop_id={loop_id}, max_rounds={args.max_rounds}")
    progress(f"loop log: {rounds_path}")

    previous: dict[str, Any] | None = None
    round_records: list[dict[str, Any]] = []
    loop_started_at = datetime.now(timezone.utc).isoformat()
    for round_index in range(1, args.max_rounds + 1):
        progress(f"开始第 {round_index}/{args.max_rounds} 轮")
        goal = build_round_goal(args, round_index, previous)
        before_summary = latest_summary()
        runner = run_stream(
            [
                "bash",
                "scripts/run_in_env.sh",
                "scripts/task1_agent_runner.py",
                "--config",
                args.config,
                "--goal",
                goal,
            ],
            cwd=ROOT,
        )
        after_summary = summary_from_runner_output(runner["output"]) or latest_summary()
        if runner["returncode"] != 0 or after_summary is None or after_summary == before_summary:
            record = {
                "round": round_index,
                "runner": runner,
                "summary": str(after_summary) if after_summary else None,
                "error": "runner_failed_or_no_new_summary",
            }
            rounds_path.open("a", encoding="utf-8").write(json.dumps(record, ensure_ascii=False) + "\n")
            round_records.append(record)
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
            "experiment_plan": summary_payload.get("experiment_plan"),
            "tool_count": len(summary_payload.get("tool_requests") or []),
            "best_leaderboard": best,
        }
        rounds_path.open("a", encoding="utf-8").write(json.dumps(record, ensure_ascii=False) + "\n")
        round_records.append(record)
        previous = record
        progress(f"第 {round_index} 轮完成；executor_failed={failed}；best={best}")
        if failed and args.stop_on_executor_failure:
            progress("工具执行失败，且启用了 --stop-on-executor-failure，停止 loop。")
            break

    final_report = {
        "loop_id": loop_id,
        "max_rounds": args.max_rounds,
        "loop_started_at": loop_started_at,
        "completed_rounds": len(round_records),
        "rounds_path": str(rounds_path),
        "final_best": best_leaderboard(),
        "rounds": round_records,
    }
    (loop_dir / "summary.json").write_text(json.dumps(final_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
