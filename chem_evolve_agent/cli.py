from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from chem_evolve_agent.core import AgentRunError, append_agent_experience, run_agent_for_target
from chem_evolve_agent.llm import LiteLlmClient, LlmSettings
from chem_evolve_agent.submitter import clean_managed_outputs, write_final_result_zip, write_single_target_result
from chem_evolve_agent.runtime_tools import find_executable, has_python_module, json_event, load_target


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _apply_mode_defaults(mode: str) -> None:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("AI4S_ROUTE_ENGINE", "aizynthfinder")
    os.environ.setdefault("AIZYNTHFINDER_CONFIG", str(root / "data/aizynthfinder/config.yml"))
    if mode != "competition":
        return
    os.environ.setdefault("AI4S_ROUTE_LIMIT_PER_ROUND", "10")
    os.environ.setdefault("AI4S_VINA_FEEDBACK_PER_ROUND", "1")
    os.environ.setdefault("CHEM_EVOLVE_LLM_ENABLED", "1")
    os.environ.setdefault("AI4S_AGENT_BYPASS_PROXY", "1")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _runtime_audit_event(args: argparse.Namespace, out_dir: Path, target_count: int, removed_outputs: list[str]) -> str:
    settings = LlmSettings.from_env()
    return json_event(
        "agent_runtime_config",
        cwd=str(Path.cwd()),
        python=sys.version.split()[0],
        out_dir=str(out_dir),
        target_count=target_count,
        rounds=args.rounds,
        per_round=args.per_round,
        top_k=args.top_k,
        mode=args.mode,
        docking_limit=args.docking_limit,
        run_seed=args.run_seed,
        route_engine=os.getenv("AI4S_ROUTE_ENGINE", "aizynthfinder"),
        route_limit_per_round=os.getenv("AI4S_ROUTE_LIMIT_PER_ROUND") or os.getenv("AGENT_ROUTE_LIMIT_PER_ROUND"),
        vina_feedback_per_round=os.getenv("AI4S_VINA_FEEDBACK_PER_ROUND") or os.getenv("AGENT_VINA_FEEDBACK_PER_ROUND"),
        aizynthfinder_config=os.getenv("AIZYNTHFINDER_CONFIG"),
        sbdd_generator_configured=bool(os.getenv("AI4S_SBDD_GENERATOR_CMD")),
        llm_enabled=settings.enabled,
        llm_model=settings.model,
        llm_provider=settings.provider,
        llm_api_base_set=bool(settings.api_base),
        llm_api_key_set=bool(settings.api_key),
        llm_temperature=settings.temperature,
        llm_max_tokens=settings.max_tokens,
        llm_log_dir=str(settings.log_dir),
        agent_memory_file=os.getenv("AI4S_AGENT_MEMORY_FILE"),
        agent_memory_limit=os.getenv("AI4S_AGENT_MEMORY_LIMIT", "10"),
        removed_outputs=removed_outputs,
    )


def _validated_target_paths(targets: list[str]) -> list[Path]:
    paths = [Path(target) for target in targets]
    for path in paths:
        load_target(path)
    return paths


def _configure_run_log_dir(out_dir: Path) -> None:
    log_dir = str(out_dir / "llm_io")
    legacy_default = Path("runs/llm_io")
    for key in ("AI4S_AGENT_LLM_LOG_DIR", "CHEM_EVOLVE_LLM_LOG_DIR"):
        current = os.getenv(key)
        if not current or Path(current) == legacy_default:
            os.environ[key] = log_dir


def _configure_agent_memory_file(out_dir: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        memory_file = Path(explicit_path)
    elif os.getenv("AI4S_AGENT_MEMORY_FILE"):
        memory_file = Path(os.environ["AI4S_AGENT_MEMORY_FILE"])
    else:
        memory_file = _default_agent_memory_file(out_dir)
    os.environ["AI4S_AGENT_MEMORY_FILE"] = str(memory_file)
    return memory_file


def _default_agent_memory_file(out_dir: Path) -> Path:
    outputs_root = Path(os.getenv("AI4S_OUTPUTS_DIR", "outputs"))
    try:
        if out_dir.resolve().is_relative_to(outputs_root.resolve()):
            return outputs_root / "agent_experience.jsonl"
    except OSError:
        pass
    return out_dir / "agent_experience.jsonl"


def _validate_run_args(args: argparse.Namespace) -> None:
    positive_fields = {
        "rounds": args.rounds,
        "per_round": args.per_round,
        "top_k": args.top_k,
    }
    for name, value in positive_fields.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须是正数")
    if args.docking_limit < 0:
        raise ValueError("--docking-limit 不能为负数")
    if args.mode in {"docking", "competition"} and args.docking_limit <= 0:
        raise ValueError(f"{args.mode} 模式下 --docking-limit 必须是正数")


def _validate_runtime_requirements(args: argparse.Namespace) -> None:
    settings = LlmSettings.from_env()
    if settings.enabled:
        if not settings.api_key:
            raise ValueError("CHEM_EVOLVE_LLM_ENABLED=1 需要 API key；请先运行 scripts/check_llm_connectivity.py")
        if not LiteLlmClient(settings).available:
            raise ValueError("CHEM_EVOLVE_LLM_ENABLED=1 需要 LiteLLM；请先运行 scripts/check_llm_connectivity.py")

    route_engine = os.getenv("AI4S_ROUTE_ENGINE", "aizynthfinder").strip().lower()
    if route_engine != "aizynthfinder":
        raise ValueError(f"未知 AI4S_ROUTE_ENGINE：{route_engine}")
    if find_executable("aizynthcli", "aizynthfinder") is None:
        raise ValueError("AI4S_ROUTE_ENGINE=aizynthfinder 需要 aizynthcli")
    config = os.getenv("AIZYNTHFINDER_CONFIG")
    if not config or not Path(config).exists():
        raise ValueError("AI4S_ROUTE_ENGINE=aizynthfinder 需要存在的 AIZYNTHFINDER_CONFIG")

    if args.mode in {"docking", "competition"}:
        if not has_python_module("vina"):
            raise ValueError(f"{args.mode} 模式需要 Python 包 vina")
        if find_executable("obabel", "babel") is None:
            raise ValueError(f"{args.mode} 模式需要 obabel 或 babel 可执行文件")

    sbdd_command = os.getenv("AI4S_SBDD_GENERATOR_CMD")
    if sbdd_command:
        try:
            command = shlex.split(sbdd_command)
        except ValueError as exc:
            raise ValueError(f"AI4S_SBDD_GENERATOR_CMD 不是合法 shell 命令：{exc}") from exc
        if not command:
            raise ValueError("AI4S_SBDD_GENERATOR_CMD 为空")
        if find_executable(command[0]) is None:
            raise ValueError(f"找不到 AI4S_SBDD_GENERATOR_CMD 可执行文件：{command[0]}")


def main() -> None:
    _load_dotenv_if_available()
    parser = argparse.ArgumentParser(description="生成 AI4S 小分子发现任务的提交文件。", add_help=False)
    parser._optionals.title = "参数"
    parser.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    parser.add_argument("--targets", nargs="+", required=True, help="一个或多个 PDB 靶点路径")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--rounds", type=int, default=3, help="生成轮数")
    parser.add_argument("--per-round", type=int, default=16, help="每轮请求的候选数")
    parser.add_argument("--top-k", type=int, default=20, help="结果 CSV 保留的候选数")
    parser.add_argument("--mode", choices=["proxy", "docking", "competition"], default="proxy", help="评分模式")
    parser.add_argument("--docking-limit", type=int, default=8, help="Vina docking 或终局复排预算")
    parser.add_argument("--runner", choices=["agent"], default="agent", help="运行器，目前固定为 agent")
    parser.add_argument("--run-seed", type=int, default=0, help="随机种子偏移")
    parser.add_argument("--memory-file", default=None, help="持久化 agent 经验 JSONL 路径。")
    args = parser.parse_args()
    _apply_mode_defaults(args.mode)
    out_dir = Path(args.out)
    _configure_run_log_dir(out_dir)
    experience_file = _configure_agent_memory_file(out_dir, args.memory_file)
    try:
        _validate_run_args(args)
        _validate_runtime_requirements(args)
        target_paths = _validated_target_paths(args.targets)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    removed_outputs = clean_managed_outputs(out_dir)
    stems: list[str] = []
    for index, target in enumerate(target_paths, start=1):
        stem = "result" if len(args.targets) == 1 else f"result{index}"
        work_dir = out_dir if len(target_paths) == 1 else out_dir / "work" / stem
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            candidates, logs = run_agent_for_target(
                target_path=target,
                out_dir=work_dir,
                rounds=args.rounds,
                per_round=args.per_round,
                mode=args.mode,
                docking_limit=args.docking_limit,
                run_seed=args.run_seed + index,
                experience_file=experience_file,
            )
        except AgentRunError as exc:
            raise SystemExit(f"AGENT_RUN_FAILED: 靶点运行失败：{target}: {exc}") from None
        logs.insert(0, json_event("output_cleanup", out_dir=str(out_dir), removed=removed_outputs))
        logs.insert(1, _runtime_audit_event(args, out_dir, target_count=len(target_paths), removed_outputs=removed_outputs))
        logs.append(json_event("submit", stem=stem, top_k=args.top_k, work_dir=str(work_dir)))
        write_single_target_result(out_dir, stem, candidates[: args.top_k], logs, write_zip=len(args.targets) == 1)
        append_agent_experience(experience_file, target, candidates, args.top_k, work_dir)
        stems.append(stem)

    if len(stems) > 1:
        zip_path = write_final_result_zip(out_dir, stems)
        _append_final_submit_logs(out_dir, stems, zip_path)


def _append_final_submit_logs(out_dir: Path, stems: list[str], zip_path: Path) -> None:
    members = [f"{stem}.csv" for stem in stems]
    event = json_event(
        "final_submit",
        zip_path=str(zip_path),
        members=members,
        member_count=len(members),
    )
    for stem in stems:
        log_path = out_dir / f"{stem}.log"
        if not log_path.exists():
            raise FileNotFoundError(f"缺少最终审计日志 {stem}: {log_path}")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(event + "\n")


if __name__ == "__main__":
    main()
