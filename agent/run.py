from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .llm import build_llm_client, logged_completion
from .logging import LLMCallLogger
from .tools import ToolRunner


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    profile_name = config.get("llm_profile")
    if not profile_name:
        return config

    profiles_path = Path(
        config.get("llm_profiles_path", config_path.parent / "llm_providers.yaml")
    )
    if not profiles_path.is_absolute():
        profiles_path = config_path.parent / profiles_path
    with profiles_path.open("r", encoding="utf-8") as fh:
        profiles_doc = yaml.safe_load(fh) or {}
    profiles = profiles_doc.get("profiles", {})
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown llm_profile '{profile_name}'. Available profiles: {available}")
    merged = dict(profiles[profile_name])
    merged.update(config)
    return merged


def observe(project_root: Path, task: str) -> dict[str, Any]:
    race_summary = project_root / "docs" / "race_summary.md"
    return {
        "task": task,
        "project_root": str(project_root),
        "race_summary": race_summary.read_text(encoding="utf-8") if race_summary.exists() else "",
        "top_level_files": sorted(path.name for path in project_root.iterdir()),
    }


def _run_action(action: dict[str, Any], tools: ToolRunner, run_dir: Path) -> dict[str, Any]:
    tool = action.get("tool")
    args = action.get("args", {})
    if tool == "record_note":
        note_path = run_dir / "notes.md"
        existing = note_path.read_text(encoding="utf-8") if note_path.exists() else ""
        note = str(args.get("note", "")).strip()
        tools.write_file(note_path, existing + f"- {note}\n", reason="record Agent note")
        return {"ok": True, "path": str(note_path)}
    if tool == "read_file":
        content = tools.read_file(args["path"])
        return {"ok": True, "content": content[-4000:]}
    if tool == "run_shell":
        return tools.run_shell(args["args"], timeout_seconds=int(args.get("timeout_seconds", 120)))
    return {"ok": False, "error": f"Unsupported tool: {tool}"}


def run_agent(task: str, config_path: str | Path, project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root else Path.cwd()
    root = root.resolve()
    config = load_config(config_path)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / "runs" / f"{task}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = LLMCallLogger(run_dir / f"{task}_logs.log")
    client = build_llm_client(config)
    tools = ToolRunner(
        project_root=root,
        allowed_roots=[run_dir],
        allowed_shell_commands=list(config.get("allowed_shell_commands", ["python", "pytest", "git"])),
    )
    started = time.perf_counter()
    iterations: list[dict[str, Any]] = []
    max_iterations = int(config.get("max_iterations", 1))
    budget = float(config.get("time_budget_seconds", 12 * 60 * 60))

    for index in range(max_iterations):
        if time.perf_counter() - started > budget:
            break
        observation = observe(root, task)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an auditable AI4S PDE research agent. "
                    "Return a concise action JSON-like object when possible."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"observation": observation}, ensure_ascii=False),
            },
        ]
        response = logged_completion(client, logger, messages)
        action = response.get("action") or {
            "tool": "record_note",
            "args": {"note": str(response.get("content", "No content returned."))[:500]},
        }
        result = _run_action(action, tools, run_dir)
        iterations.append(
            {
                "index": index,
                "observation": observation,
                "action": action,
                "result": result,
            }
        )

    tools.save_manifest(
        run_dir / "manifest.json",
        {
            "task": task,
            "config_path": str(Path(config_path).resolve()),
            "run_dir": str(run_dir),
            "iterations": iterations,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI4S auditable Agent loop.")
    parser.add_argument("--task", choices=["task1", "task2"], required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_agent(task=args.task, config_path=args.config))


if __name__ == "__main__":
    main()
