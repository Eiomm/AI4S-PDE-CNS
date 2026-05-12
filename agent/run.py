"""AI4S PDE Agent main loop: observe-plan-act-record."""

from __future__ import annotations

import argparse
import json
import shlex
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .llm import build_llm_client, load_env_file, logged_completion
from .logging import LLMCallLogger
from .prompts import SYSTEM_PROMPT, parse_action, build_messages
from .task1_submission import create_task1_submission_bundle
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


def load_project_env(config: dict[str, Any], project_root: str | Path) -> list[str]:
    """Load a configured .env file relative to the project root."""

    env_file = config.get("env_file")
    if not env_file:
        return []
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = Path(project_root) / env_path
    return load_env_file(env_path)


def read_data_info(data_dir: Path) -> dict[str, Any]:
    """Read shapes and keys of HDF5 data files."""
    import h5py
    info = {}
    if not data_dir.exists():
        return info
    for hdf_file in data_dir.rglob("*.hdf5"):
        try:
            with h5py.File(hdf_file, "r") as f:
                shapes = {}
                for key in f.keys():
                    ds = f[key]
                    shapes[key] = {"shape": list(ds.shape), "dtype": str(ds.dtype)}
                info[str(hdf_file.relative_to(data_dir))] = shapes
        except Exception:
            pass
    for hdf_file in data_dir.rglob("*.h5"):
        try:
            with h5py.File(hdf_file, "r") as f:
                shapes = {}
                for key in f.keys():
                    ds = f[key]
                    shapes[key] = {"shape": list(ds.shape), "dtype": str(ds.dtype)}
                info[str(hdf_file.relative_to(data_dir))] = shapes
        except Exception:
            pass
    return info


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """List existing run directories and their metrics."""
    runs = []
    if not runs_dir.exists():
        return runs
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        entry = {"name": run_dir.name, "files": []}
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            try:
                entry["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        for f in run_dir.iterdir():
            if f.is_file():
                entry["files"].append(f.name)
        runs.append(entry)
    return runs


def list_code_files(code_dir: Path) -> list[str]:
    """List Python files in the code directory."""
    if not code_dir.exists():
        return []
    return sorted(f.name for f in code_dir.glob("*.py"))


def observe(project_root: Path, task: str, run_dir: Path) -> dict[str, Any]:
    """Collect rich observation of the current project state."""
    data_dir = project_root / "data"
    code_dir = project_root / "code"
    runs_dir = project_root / "runs"
    checkpoints_dir = project_root / "checkpoints"

    observation = {
        "task": task,
        "project_root": str(project_root),
        "data_info": read_data_info(data_dir),
        "code_files": list_code_files(code_dir),
        "existing_runs": list_runs(runs_dir),
        "checkpoints": [],
    }

    # List available checkpoints
    if checkpoints_dir.exists():
        for f in checkpoints_dir.iterdir():
            if f.suffix in (".tar", ".pt", ".pth"):
                observation["checkpoints"].append(f.name)

    # Check for extracted checkpoints
    extracted = checkpoints_dir / "extracted"
    if extracted.exists():
        observation["extracted_checkpoints"] = [f.name for f in extracted.glob("*.pt")]

    # Read last metrics if this run already has results
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            observation["last_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Read notes from previous iterations
    notes_path = run_dir / "notes.md"
    if notes_path.exists():
        observation["notes"] = notes_path.read_text(encoding="utf-8")

    return observation


def execute_action(
    action: dict[str, Any],
    tools: ToolRunner,
    run_dir: Path,
    project_root: Path,
    *,
    agent_started: float | None = None,
) -> dict[str, Any]:
    """Execute an action using the appropriate tool."""
    tool = action.get("tool")
    args = action.get("args", {})

    if tool == "read_file":
        try:
            content = tools.read_file(args["path"])
            # Truncate large files for context
            if len(content) > 8000:
                content = content[:4000] + "\n... [truncated] ...\n" + content[-4000:]
            return {"ok": True, "content": content}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif tool == "write_file":
        try:
            path = args["path"]
            content = args["content"]
            tools.write_file(path, content, reason="Agent generated code")
            return {"ok": True, "path": path, "lines": content.count("\n") + 1}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif tool == "run_shell":
        try:
            if "args" in args:
                cmd_args = [str(arg) for arg in args["args"]]
            else:
                command = args.get("command", "")
                cmd_args = shlex.split(command)
            timeout = int(args.get("timeout", 300))
            result = tools.run_shell(cmd_args, timeout_seconds=timeout)
            result["ok"] = result.get("returncode") == 0
            return result
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif tool == "analyze_result":
        try:
            return tools.analyze_result(args["path"])
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif tool == "validate_submission":
        try:
            return tools.validate_submission(args["path"])
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif tool == "create_task1_submission":
        try:
            inference_time = float(args["inference_time"])
            train_time_value = args.get("train_time", 0.0)
            if train_time_value == "elapsed_without_inference" and agent_started is not None:
                train_time = max(time.perf_counter() - agent_started - inference_time, 0.0)
            else:
                train_time = float(train_time_value)
            log_path = run_dir / "task1_logs.log"
            output_dir = create_task1_submission_bundle(
                prediction_path=args["prediction_path"],
                initial_path=args["initial_path"],
                output_dir=args["output_dir"],
                code_dir=args.get("code_dir", "code"),
                log_path=log_path,
                train_time=train_time,
                inference_time=inference_time,
            )
            return {"ok": True, "path": str(output_dir)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif tool == "record_note":
        note = args.get("note", "")
        notes_path = run_dir / "notes.md"
        existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
        tools.write_file(notes_path, existing + f"\n- {note}\n", reason="Agent note")
        return {"ok": True, "note": note}

    elif tool == "stop":
        return {"ok": True, "stopped": True, "reason": args.get("reason", "Agent decided to stop")}

    else:
        return {"ok": False, "error": f"Unknown tool: {tool}"}


def run_agent(
    task: str,
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> Path:
    """Run the full Agent loop: observe-plan-act-record."""
    root = Path(project_root) if project_root else Path.cwd()
    root = root.resolve()
    config = load_config(config_path)
    loaded_env_keys = load_project_env(config, root)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / "runs" / f"{task}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = LLMCallLogger(run_dir / f"{task}_logs.log")
    client = build_llm_client(config)
    command_aliases = dict(config.get("command_aliases", {}))
    if config.get("python_executable"):
        command_aliases.setdefault("python", str(config["python_executable"]))
    tools = ToolRunner(
        project_root=root,
        allowed_roots=[run_dir],
        allowed_shell_commands=list(config.get("allowed_shell_commands", ["python", "pytest", "git"])),
        command_aliases=command_aliases,
    )

    started = time.perf_counter()
    iterations: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    max_iterations = int(config.get("max_iterations", 10))
    budget = float(config.get("time_budget_seconds", 3600))

    print(f"[Agent] Starting {task} agent loop (max {max_iterations} iterations)")
    print(f"[Agent] Run directory: {run_dir}")
    print(f"[Agent] LLM provider: {client.provider}/{client.model}")

    for index in range(max_iterations):
        if time.perf_counter() - started > budget:
            print(f"[Agent] Time budget exhausted ({budget}s)")
            break

        print(f"\n[Agent] === Iteration {index + 1}/{max_iterations} ===")

        # 1. OBSERVE
        observation = observe(root, task, run_dir)
        print(f"[Agent] Observed: {len(observation.get('code_files', []))} code files, "
              f"{len(observation.get('existing_runs', []))} existing runs")

        # 2. PLAN — call LLM
        messages = build_messages(SYSTEM_PROMPT, observation, history)
        print(f"[Agent] Calling LLM...")
        try:
            response = logged_completion(client, logger, messages)
        except Exception as e:
            print(f"[Agent] LLM call failed: {e}")
            # Record failure and continue
            iterations.append({
                "index": index,
                "observation": observation,
                "error": str(e),
            })
            continue

        # 3. Parse action
        action = parse_action(response)
        thinking = response.get("content", "")[:200]
        print(f"[Agent] Action: {action.get('tool')} — {thinking}")

        # 4. ACT
        result = execute_action(action, tools, run_dir, root, agent_started=started)
        print(f"[Agent] Result: ok={result.get('ok')}")

        # 5. RECORD
        entry = {
            "index": index,
            "observation": observation,
            "action": action,
            "action_response": response,
            "result": result,
            "elapsed": round(time.perf_counter() - started, 2),
        }
        iterations.append(entry)
        history.append(entry)

        # 6. Check stop condition
        if action.get("tool") == "stop":
            print(f"[Agent] Agent stopped: {action.get('args', {}).get('reason', '')}")
            break

    # Save manifest
    elapsed = round(time.perf_counter() - started, 2)
    tools.save_manifest(
        run_dir / "manifest.json",
        {
            "task": task,
            "config_path": str(Path(config_path).resolve()),
            "run_dir": str(run_dir),
            "loaded_env_keys": loaded_env_keys,
            "iterations": iterations,
            "iteration_count": len(iterations),
            "elapsed_seconds": elapsed,
            "provider": client.provider,
            "model": client.model,
        },
    )
    print(f"\n[Agent] Completed {len(iterations)} iterations in {elapsed}s")
    print(f"[Agent] Results saved to {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI4S PDE Agent loop.")
    parser.add_argument("--task", choices=["task1", "task2"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()
    print(run_agent(task=args.task, config_path=args.config, project_root=args.project_root))


if __name__ == "__main__":
    main()
