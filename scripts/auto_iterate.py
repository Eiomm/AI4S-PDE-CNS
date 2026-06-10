#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chem_evolve_agent.experiments import RunMetrics, analyze_run_directory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "strategies" / "initial.yaml"
DEFAULT_OUTPUTS = ROOT / "outputs"
REPO_SKILL = ROOT / "skills" / "ai4s-chem-evolve" / "SKILL.md"
USER_SKILL = Path.home() / ".codex" / "skills" / "ai4s-chem-evolve" / "SKILL.md"


@dataclass
class Strategy:
    name: str
    description: str
    technique: str
    rounds: int
    per_round: int
    top_k: int
    mode: str
    docking_limit: int
    runner: str
    llm_enabled: bool
    env: dict[str, str]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Strategy":
        return cls(
            name=str(raw["name"]),
            description=str(raw.get("description", "")),
            technique=str(raw.get("technique", "inference")),
            rounds=int(raw.get("rounds", 4)),
            per_round=int(raw.get("per_round", 12)),
            top_k=int(raw.get("top_k", 20)),
            mode=str(raw.get("mode", "proxy")),
            docking_limit=int(raw.get("docking_limit", 6)),
            runner=str(raw.get("runner", "legacy")),
            llm_enabled=bool(raw.get("llm_enabled", False)),
            env={str(k): str(v) for k, v in dict(raw.get("env") or {}).items()},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 AI4S 小分子生成策略实验，并记录可复用的 agent 经验。")
    parser.add_argument("--goal", default="通过推理阶段的策略搜索，提升 AI4S 小分子生成效果。")
    parser.add_argument("--experiment", default="goal_iter")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--target", type=Path, default=ROOT / "target.pdb")
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--strategies", nargs="*", help="只运行指定策略名。")
    parser.add_argument("--session-id", help="本次实验启动的目录名；默认使用当前时间戳。")
    parser.add_argument("--seed", type=int, help="本次实验的基础随机种子；默认从 session-id 派生。")
    parser.add_argument("--skip-tests", action="store_true", help="跳过每轮内部 pytest，加快迭代。")
    parser.add_argument("--patience", type=int, default=0, help="连续 N 次没有提升就停止。0 表示不启用。")
    parser.add_argument("--tail-events", type=int, default=0)
    parser.add_argument("--no-stream", action="store_true", help="运行时隐藏 harness 输出，只显示汇总。")
    parser.add_argument(
        "--view",
        choices=["compact", "normal", "debug"],
        default="compact",
        help="终端显示密度。compact 最清爽；normal 显示关键事件；debug 显示 harness 原始细节。",
    )
    parser.add_argument("--lang", choices=["zh", "en"], default="zh", help="终端显示语言，默认中文。")
    parser.add_argument("--no-translate", action="store_true", help="显示 harness 原始输出，不做中文压缩。")
    args = parser.parse_args()

    strategies = load_strategies(args.config, args.strategies)
    if not strategies:
        raise SystemExit("没有选中任何策略。")

    memory_dir = args.outputs / "strategy_memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    index_jsonl = memory_dir / "experiment_index.jsonl"
    index_csv = memory_dir / "experiment_index.csv"
    best_md = memory_dir / "best_strategies.md"
    failed_md = memory_dir / "failed_strategies.md"

    records = load_records(index_jsonl)
    session_records: list[dict[str, Any]] = []

    # 一次启动 auto_iterate 就是一场实验 session；同一 session 下可以跑多轮 strategy。
    # 这样 outputs 不会被 i01/i02/... 顶层目录打散，也方便保留本次最好的提交包。
    session_root = make_session_root(args.outputs, args.experiment, args.session_id)
    session_runs_dir = session_root / "runs"
    session_runs_dir.mkdir(parents=True, exist_ok=True)
    session_jsonl = session_root / "experiment_index.jsonl"
    session_csv = session_root / "experiment_index.csv"

    # base_seed 固定一场实验的可复现性；每个 run 再派生自己的 run_seed，避免重复重放同一路径。
    base_seed = args.seed if args.seed is not None else seed_from_text(session_root.name)
    best_so_far = max((float(record.get("objective", 0.0)) for record in records), default=0.0)
    session_best = -1.0
    no_improve = 0

    print_header(args.goal, args.experiment, args.target, strategies, memory_dir, session_root, args.view)
    try:
        for iteration in range(1, args.iterations + 1):
            for strategy_index, strategy in enumerate(strategies, start=1):
                if strategy.technique != "inference":
                    print(f"[跳过] {strategy.name}: technique={strategy.technique} 需要训练数据/真实评估标签后再启用。")
                    continue

                run_name = f"i{iteration:02d}_{strategy.name}"
                run_seed = run_seed_for(base_seed, iteration, strategy_index)
                print_strategy_start(iteration, args.iterations, strategy_index, len(strategies), strategy, run_name, args.view)
                rc = run_harness(strategy, run_name, args, session_runs_dir, run_seed)
                run_dir = latest_run_dir(session_runs_dir / run_name)
                metrics = analyze_run_directory(run_dir) if run_dir else RunMetrics(run_dir="", status="fail")
                if rc != 0:
                    metrics.status = "fail"

                # improved 面向全局历史；best_in_session 只比较本次启动里的 run。
                # 前者更新 strategy_memory，后者更新 session 根目录下的 best 链接和 best_result.zip。
                improved = metrics.objective > best_so_far
                best_in_session = metrics.status == "pass" and metrics.inspect_ok and metrics.objective > session_best
                if improved:
                    best_so_far = metrics.objective
                    no_improve = 0
                else:
                    no_improve += 1
                if best_in_session:
                    session_best = metrics.objective

                record = make_record(args, iteration, strategy, metrics, improved, best_in_session, session_root, run_seed)
                records.append(record)
                session_records.append(record)
                append_jsonl(index_jsonl, record)
                append_jsonl(session_jsonl, record)
                write_csv(index_csv, records)
                write_csv(session_csv, session_records)
                append_memory(best_md if improved else failed_md, record)
                if best_in_session and run_dir is not None:
                    update_session_best(session_root, run_dir, record)
                if should_update_skill(args.outputs):
                    update_skill(records, best_md)

                print_run_summary(record, best_so_far)
                if args.view in {"normal", "debug"}:
                    print_event_tail(run_dir, args.tail_events)
                    print_dashboard(session_records[-12:], best_so_far)

                if args.patience and no_improve >= args.patience:
                    print(f"[停止] patience={args.patience}；最近 {no_improve} 轮没有提升。")
                    print_outputs(memory_dir, session_root, session_records)
                    return
    except KeyboardInterrupt:
        print("\n[停止] 收到 Ctrl-C，已结束当前自动迭代。已完成的 run 已写入记录。")
        print_outputs(memory_dir, session_root, session_records)
        return

    print_outputs(memory_dir, session_root, session_records)


def load_strategies(path: Path, selected: list[str] | None) -> list[Strategy]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    strategies = [Strategy.from_dict(item) for item in data.get("strategies", [])]
    if selected:
        wanted = set(selected)
        strategies = [strategy for strategy in strategies if strategy.name in wanted]
    return strategies


def should_update_skill(outputs: Path) -> bool:
    try:
        return outputs.resolve() == DEFAULT_OUTPUTS.resolve()
    except FileNotFoundError:
        return outputs.absolute() == DEFAULT_OUTPUTS.absolute()


def make_session_root(outputs: Path, experiment: str, session_id: str | None = None) -> Path:
    # session_id 可手工指定，适合复现实验；默认时间戳适合日常连续试验。
    session_name = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = outputs / experiment / session_name
    if not root.exists():
        return root
    for suffix in range(2, 1000):
        candidate = outputs / experiment / f"{session_name}_{suffix:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate a unique session directory under {outputs / experiment}")


def seed_from_text(text: str) -> int:
    # 不依赖 Python 内置 hash；内置 hash 默认带随机盐，不适合跨进程复现。
    seed = 0
    for char in text:
        seed = (seed * 131 + ord(char)) % 2_147_483_647
    return seed or 1


def run_seed_for(base_seed: int, iteration: int, strategy_index: int) -> int:
    return (base_seed + iteration * 1009 + strategy_index * 37) % 2_147_483_647 or 1


def strategy_display(strategy: Strategy) -> str:
    mapping = {
        "seed_baseline": "固定种子基线",
        "llm_diverse": "LLM 多样性探索",
        "docking_probe": "Docking 探测",
        "agent_main": "Agent 主循环",
    }
    return mapping.get(strategy.name, strategy.name)


def description_zh(strategy: Strategy) -> str:
    mapping = {
        "seed_baseline": "不调用 LLM，用固定种子、突变和片段组合建立稳定基线。",
        "llm_diverse": "在一轮中调用 LLM，让它提出更多样的合法 SMILES。",
        "docking_probe": "如果本地 docking 工具可用，对 top 分子尝试 docking 排序。",
        "agent_main": "使用 planner/action/memory loop 自动选择下一步生成动作。",
    }
    return mapping.get(strategy.name, strategy.description)


def run_harness(strategy: Strategy, run_name: str, args: argparse.Namespace, outputs_dir: Path, run_seed: int) -> int:
    env = os.environ.copy()
    env.update(strategy.env)

    # harness 仍然只知道一个 outputs 根目录；这里把根目录缩到 session/runs，
    # 让 run_harness_once.sh 自己继续维护 <run_name>/<timestamp>/latest。
    env["AI4S_OUTPUTS_DIR"] = str(outputs_dir)
    env["CHEM_EVOLVE_RUN_SEED"] = str(run_seed)
    env["CHEM_EVOLVE_LLM_ENABLED"] = "1" if strategy.llm_enabled else "0"
    env.setdefault("PYTHONUNBUFFERED", "1")

    command = [
        "bash",
        "scripts/run_harness_once.sh",
        "--name",
        run_name,
        "--target",
        str(args.target),
        "--rounds",
        str(strategy.rounds),
        "--per-round",
        str(strategy.per_round),
        "--top-k",
        str(strategy.top_k),
        "--mode",
        strategy.mode,
        "--docking-limit",
        str(strategy.docking_limit),
        "--runner",
        strategy.runner,
    ]
    if args.skip_tests:
        command.append("--skip-tests")

    if args.view == "debug" and args.lang == "zh":
        print("[执行命令] " + " ".join(command))
    elif args.view == "debug":
        print("[cmd] " + " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            if should_print_harness_line(args):
                rendered = render_harness_line(line, strategy, args)
                if rendered:
                    print(rendered)
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise


def should_print_harness_line(args: argparse.Namespace) -> bool:
    return not args.no_stream


def render_harness_line(line: str, strategy: Strategy, args: argparse.Namespace) -> str | None:
    if args.view == "debug" or args.no_translate or args.lang == "en":
        return f"[{strategy.name}] {line.rstrip()}"
    if args.view == "normal":
        translated = translate_harness_line(line, compact=False)
        return f"[{strategy_display(strategy)}] {translated}" if translated else None
    translated = translate_harness_line(line, compact=True)
    return f"  {translated}" if translated else None


def translate_harness_line(line: str, compact: bool = False) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if compact:
        if "step 1/3: pytest skipped" in stripped:
            return "测试：跳过"
        if "step 1/3: pytest" in stripped:
            return "测试：运行中"
        if "passed in" in stripped and "==" in stripped:
            return "测试：通过"
        if "step 2/3: generate candidates" in stripped:
            return "生成：运行中"
        if "step 3/3: inspect submission" in stripped:
            return "检查：提交包格式"
        if stripped.startswith("OK ") and "contains" in stripped:
            return "检查：通过"
        if "[harness] run root:" in stripped:
            return "目录：" + stripped.split("run root:", 1)[-1].strip()
        if "[harness] result:" in stripped:
            return "提交包：" + stripped.split("result:", 1)[-1].strip()
        if "FAILED" in stripped or "ERROR" in stripped:
            return stripped
        return None
    if "[harness] root:" in stripped:
        return "项目目录：" + stripped.split("root:", 1)[-1].strip()
    if "[harness] python:" in stripped:
        return "Python 环境：" + stripped.split("python:", 1)[-1].strip()
    if "[harness] target:" in stripped:
        return "输入靶点：" + stripped.split("target:", 1)[-1].strip()
    if "[harness] output:" in stripped:
        return "本轮输出目录：" + stripped.split("output:", 1)[-1].strip()
    if "[harness] mode:" in stripped:
        return "运行模式：" + stripped.split("mode:", 1)[-1].strip()
    if "step 1/3: pytest skipped" in stripped:
        return "步骤 1/3：跳过测试。"
    if "step 1/3: pytest" in stripped:
        return "步骤 1/3：正在运行单元测试。"
    if "passed in" in stripped and "==" in stripped:
        return "单元测试通过：" + stripped.strip("=")
    if "step 2/3: generate candidates" in stripped:
        return "步骤 2/3：正在生成候选小分子。"
    if "step 3/3: inspect submission" in stripped:
        return "步骤 3/3：正在检查提交压缩包格式。"
    if stripped.startswith("OK ") and "contains" in stripped:
        return "提交包检查通过：" + stripped
    if "[harness] artifacts:" in stripped:
        return "本轮产物："
    if "candidates:" in stripped:
        return "候选分子表：" + stripped.split("candidates:", 1)[-1].strip()
    if "pipeline:" in stripped:
        return "流水线日志：" + stripped.split("pipeline:", 1)[-1].strip()
    if "submission:" in stripped:
        return "提交包：" + stripped.split("submission:", 1)[-1].strip()
    if "[harness] exit=" in stripped:
        return "本轮退出码：" + stripped.split("exit=", 1)[-1].strip()
    if "[harness] run root:" in stripped:
        return "本轮完整目录：" + stripped.split("run root:", 1)[-1].strip()
    if "[harness] latest:" in stripped:
        return "latest 链接：" + stripped.split("latest:", 1)[-1].strip()
    if "[harness] result:" in stripped:
        return "最终提交包：" + stripped.split("result:", 1)[-1].strip()
    if "FAILED" in stripped or "ERROR" in stripped:
        return stripped
    return None


def latest_run_dir(parent: Path) -> Path | None:
    latest = parent / "latest"
    if not latest.exists():
        return None
    return latest.resolve()


def make_record(
    args: argparse.Namespace,
    iteration: int,
    strategy: Strategy,
    metrics: RunMetrics,
    improved: bool,
    best_in_session: bool,
    session_root: Path,
    run_seed: int,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "goal": args.goal,
        "experiment": args.experiment,
        "session_dir": str(session_root),
        "iteration": iteration,
        "strategy": strategy.name,
        "description": strategy.description,
        "technique": strategy.technique,
        "rounds": strategy.rounds,
        "per_round": strategy.per_round,
        "top_k": strategy.top_k,
        "mode": strategy.mode,
        "docking_limit": strategy.docking_limit,
        "runner": strategy.runner,
        "llm_enabled": strategy.llm_enabled,
        "run_seed": run_seed,
        "improved": improved,
        "best_in_session": best_in_session,
        **metrics.to_dict(),
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    keys = [
        "timestamp",
        "experiment",
        "iteration",
        "strategy",
        "status",
        "improved",
        "best_in_session",
        "objective",
        "best_total",
        "avg_total",
        "candidate_count",
        "valid_smiles",
        "unique_smiles",
        "scaffold_count",
        "route_consistent",
        "llm_calls",
        "mode",
        "llm_enabled",
        "runner",
        "run_seed",
        "session_dir",
        "run_dir",
        "best_smiles",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in keys})


def append_memory(path: Path, record: dict[str, Any]) -> None:
    marker = "IMPROVED" if record.get("improved") else "NO_IMPROVEMENT"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {record['timestamp']} - {record['strategy']} - {marker}\n")
        handle.write(f"- 综合指标 objective: {record['objective']}\n")
        handle.write(f"- 最高分 best_total: {record['best_total']}, 平均分 avg_total: {record['avg_total']}\n")
        handle.write(f"- 合法/去重/骨架数: {record['valid_smiles']}/{record['unique_smiles']}/{record['scaffold_count']}\n")
        handle.write(
            f"- LLM 调用次数: {record['llm_calls']}, 模式: {record['mode']}, runner: {record.get('runner', 'legacy')}, "
            f"run_seed: {record.get('run_seed', '')}, 运行目录: {record['run_dir']}\n"
        )
        handle.write(f"- 最佳 SMILES: `{record.get('best_smiles', '')}`\n")


def update_session_best(session_root: Path, run_dir: Path, record: dict[str, Any]) -> None:
    # best 是给人看的快捷入口；best_result.* 是给后续脚本直接拿来比较/提交的稳定文件名。
    best_link = session_root / "best"
    if best_link.is_symlink() or best_link.exists() and best_link.is_file():
        best_link.unlink()
    if not best_link.exists():
        best_link.symlink_to(run_dir, target_is_directory=True)

    (session_root / "best_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_map = {
        "result.csv": "best_result.csv",
        "result.log": "best_result.log",
        "result.zip": "best_result.zip",
        "candidates.csv": "best_candidates.csv",
        "pipeline.log": "best_pipeline.log",
        "submission.zip": "best_submission.zip",
    }
    for source_name, dest_name in artifact_map.items():
        source = run_dir / source_name
        if source.exists():
            shutil.copy2(source, session_root / dest_name)


def update_skill(records: list[dict[str, Any]], best_md: Path) -> None:
    best = sorted(
        [record for record in records if record.get("status") == "pass" and record.get("inspect_ok")],
        key=lambda item: float(item.get("objective", 0.0)),
        reverse=True,
    )[:5]
    lines = [
        "---",
        "name: ai4s-chem-evolve",
        "description: Use when iterating the AI4S chemistry molecule-generation pipeline, running inference experiments, comparing strategies, logging results, and reusing the best strategy memory.",
        "---",
        "",
        "# AI4S Chem Evolve",
        "",
        "Use this skill when improving `/data/wangjunao/AI4S` for the AI4S small-molecule generation task.",
        "",
        "## Operating Loop",
        "",
        "1. Keep the submission contract intact: `result.csv`, `result.log`, `result.zip` with `mol_smiles,route` columns.",
        "2. Prefer inference-time strategy experiments first: prompt changes, generator mix, filtering, scoring, docking probes, and retrosynthesis checks.",
        "3. Change one main variable per run, then execute `scripts/run_harness_once.sh` or `scripts/auto_iterate.py`.",
        "4. Judge progress by objective, best score, average score, valid SMILES, route consistency, scaffold diversity, and penalties.",
        "5. Record every run under `outputs/strategy_memory`; promote only strategies that improve objective or reveal a reusable failure rule.",
        "6. If LLM is enabled, use the OpenAI-compatible Apifox/GPT.GE route from `.env`: model `openai/claude-opus-4-8`, provider `openai`, base URL `https://api.gpt.ge/v1`.",
        "",
        "## Current Best Strategies",
        "",
    ]
    if best:
        for index, record in enumerate(best, start=1):
            lines.append(
                f"{index}. `{record['strategy']}` objective={record['objective']} best={record['best_total']} "
                f"avg={record['avg_total']} scaffolds={record['scaffold_count']} llm_calls={record['llm_calls']} "
                f"runner={record.get('runner', 'legacy')}"
            )
            lines.append(f"   Run: `{record['run_dir']}`")
    else:
        lines.append("- No successful strategy has been recorded yet. Run `scripts/auto_iterate.py` first.")
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```bash",
            "python scripts/auto_iterate.py --experiment goal_iter --iterations 1 --skip-tests",
            "bash scripts/run_harness_once.sh --name official --target target.pdb",
            "python scripts/check_llm_connectivity.py",
            "```",
            "",
            "## Memory Files",
            "",
            "- `outputs/strategy_memory/experiment_index.csv`: compact experiment table.",
            "- `outputs/strategy_memory/experiment_index.jsonl`: full machine-readable records.",
            "- `outputs/strategy_memory/best_strategies.md`: promoted strategy notes.",
            "- `outputs/strategy_memory/failed_strategies.md`: reusable failure notes.",
        ]
    )
    content = "\n".join(lines) + "\n"
    for path in (REPO_SKILL, USER_SKILL):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    best_md.parent.mkdir(parents=True, exist_ok=True)


def print_header(
    goal: str,
    experiment: str,
    target: Path,
    strategies: list[Strategy],
    memory_dir: Path,
    session_root: Path,
    view: str,
) -> None:
    if view == "compact":
        print("AI4S 自动实验")
        print(f"实验：{experiment} | 靶点：{target}")
        print(f"本次目录：{session_root}")
        print("策略：" + " / ".join(f"{strategy_display(strategy)}" for strategy in strategies))
        print(f"记录：{memory_dir / 'experiment_index.csv'}")
        return
    print("=" * 88)
    print("AI4S 自动科研迭代")
    print("=" * 88)
    print(f"目标：     {goal}")
    print(f"实验名：   {experiment}")
    print(f"本次目录： {session_root}")
    print(f"输入靶点： {target}")
    print(f"记忆目录： {memory_dir}")
    print("将要尝试的策略：" + "，".join(f"{strategy_display(strategy)}({strategy.name})" for strategy in strategies))
    print("=" * 88)


def print_strategy_start(
    iteration: int,
    total_iterations: int,
    strategy_index: int,
    total_strategies: int,
    strategy: Strategy,
    run_name: str,
    view: str,
) -> None:
    if view == "compact":
        print()
        print(
            f"[{iteration}/{total_iterations} · {strategy_index}/{total_strategies}] "
            f"{strategy_display(strategy)} 开始 "
            f"(rounds={strategy.rounds}, per_round={strategy.per_round}, mode={strategy.mode}, "
            f"runner={strategy.runner}, llm={strategy.llm_enabled})"
        )
        return
    print()
    print("-" * 88)
    print(f"第 {iteration} 轮 | 策略：{strategy_display(strategy)} ({strategy.name}) | 运行名：{run_name}")
    print(f"策略说明：{description_zh(strategy)}")
    print(
        "参数："
        f"生成轮数={strategy.rounds}，每轮数量={strategy.per_round}，保留 top_k={strategy.top_k}，"
        f"模式={strategy.mode}，runner={strategy.runner}，docking 数={strategy.docking_limit}，是否调用 LLM={strategy.llm_enabled}"
    )
    print("-" * 88)


def print_run_summary(record: dict[str, Any], best_so_far: float) -> None:
    status = "通过" if record.get("status") == "pass" else "失败"
    improved = "提升" if record.get("improved") else "未提升"
    run_dir = str(record.get("run_dir", ""))
    short_dir = run_dir.replace(str(ROOT) + "/", "") if run_dir else ""
    print(
        f"  结果：{status} | {improved} | 综合={float(record.get('objective', 0.0)):.4f} "
        f"| 最高={float(record.get('best_total', 0.0)):.4f} "
        f"| 平均={float(record.get('avg_total', 0.0)):.4f} "
        f"| 合法={record.get('valid_smiles', 0)}/{record.get('candidate_count', 0)} "
        f"| 骨架={record.get('scaffold_count', 0)} | LLM={record.get('llm_calls', 0)}"
    )
    print(f"  最佳：{record.get('best_smiles', '')}")
    print(f"  当前最好综合={best_so_far:.4f} | 目录：{short_dir}")


def print_event_tail(run_dir: Path | None, limit: int) -> None:
    if not run_dir or limit <= 0:
        return
    log_path = run_dir / "pipeline.log"
    if not log_path.exists():
        return
    events: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") in {"generate", "rank", "pipeline_complete", "llm_skipped", "docking"}:
            events.append(event)
    if not events:
        return
    print("最近关键事件：")
    for event in events[-limit:]:
        name = event.get("event")
        if name == "generate":
            print(f"  生成：第 {event.get('round')} 轮，生成器={event.get('generator')}，数量={event.get('count')}")
        elif name == "docking":
            print(f"  docking：成功={event.get('success')}，能量={event.get('docking_energy')}，原因={event.get('reason')}")
        elif name == "pipeline_complete":
            print(f"  流水线完成：候选数={event.get('candidate_count')}，最高分={event.get('best_score')}，模式={event.get('mode')}")
        elif name == "rank":
            print(f"  排序完成：候选数={event.get('candidate_count')}，最高分={event.get('best_score')}")
        elif name == "llm_skipped":
            print(f"  LLM 跳过：原因={event.get('reason')}")
        else:
            print(f"  {name}: {json.dumps(event, ensure_ascii=False, sort_keys=True)}")


def print_dashboard(records: list[dict[str, Any]], best_so_far: float) -> None:
    print()
    print(f"实验结果看板：当前最好综合指标={best_so_far:.4f}")
    columns = [
        ("轮次", 4),
        ("策略", 18),
        ("状态", 7),
        ("提升", 5),
        ("综合", 7),
        ("最高", 7),
        ("平均", 7),
        ("合法", 8),
        ("去重", 5),
        ("骨架", 5),
        ("LLM", 4),
    ]
    header = " ".join(name.ljust(width) for name, width in columns)
    print(header)
    print("-" * len(header))
    for record in records:
        values = [
            str(record.get("iteration", "")),
            strategy_display(Strategy.from_dict({
                "name": str(record.get("strategy", "")),
                "rounds": record.get("rounds", 0),
                "per_round": record.get("per_round", 0),
                "top_k": record.get("top_k", 0),
                "mode": record.get("mode", "proxy"),
                "docking_limit": record.get("docking_limit", 0),
                "llm_enabled": record.get("llm_enabled", False),
            }))[:18],
            "通过" if record.get("status") == "pass" else "失败",
            "是" if record.get("improved") else "否",
            f"{float(record.get('objective', 0.0)):.4f}",
            f"{float(record.get('best_total', 0.0)):.4f}",
            f"{float(record.get('avg_total', 0.0)):.4f}",
            f"{record.get('valid_smiles', 0)}/{record.get('candidate_count', 0)}",
            str(record.get("unique_smiles", 0)),
            str(record.get("scaffold_count", 0)),
            str(record.get("llm_calls", 0)),
        ]
        print(" ".join(value.ljust(width) for value, (_, width) in zip(values, columns)))


def print_outputs(memory_dir: Path, session_root: Path, session_records: list[dict[str, Any]] | None = None) -> None:
    print()
    print("自动迭代完成")
    print(f"本次实验目录： {session_root}")
    print(f"本次汇总表：   {session_root / 'experiment_index.csv'}")
    if (session_root / "best").exists():
        print(f"本次 best：    {session_root / 'best'}")
    if (session_root / "best_result.zip").exists():
        print(f"本次 best zip：{session_root / 'best_result.zip'}")
    print(f"实验汇总表：   {memory_dir / 'experiment_index.csv'}")
    print(f"最佳策略记录： {memory_dir / 'best_strategies.md'}")
    if session_records:
        best = max(session_records, key=lambda item: float(item.get("objective", 0.0)))
        print(
            f"本次最好：{best.get('strategy')} | 综合={float(best.get('objective', 0.0)):.4f} "
            f"| 最高={float(best.get('best_total', 0.0)):.4f}"
        )


if __name__ == "__main__":
    main()
