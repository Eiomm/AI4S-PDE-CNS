#!/usr/bin/env python3
"""Run an AI4S PDE task through the vendored DSLighting AIDE workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_PROVIDER_RAW_DEBUG = os.getenv("AI4S_PROVIDER_RAW_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

if _PROVIDER_RAW_DEBUG:
    os.environ.setdefault("LITELLM_LOG", "DEBUG")
    os.environ.setdefault("LITELLM_SET_VERBOSE", "True")
else:
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    os.environ.setdefault("LITELLM_SET_VERBOSE", "False")

if _PROVIDER_RAW_DEBUG:
    import litellm
else:
    logging.disable(logging.WARNING)
    try:
        import litellm
    finally:
        logging.disable(logging.NOTSET)
import h5py
import numpy as np
import yaml
from dotenv import load_dotenv

litellm.telemetry = False
litellm.suppress_debug_info = True
litellm.set_verbose = _PROVIDER_RAW_DEBUG

from dslighting.config import (
    AgentSearchConfig,
    DSLightingConfig,
    DataAnalysisConfig,
    LLMConfig,
    RunConfig,
    SandboxConfig,
    WorkflowConfig,
)
from dslighting.core.types import TaskDefinition
from dslighting.runner import DSLightingRunner


TASKS_ROOT = PROJECT_ROOT / "tasks"

TASK_DIRS = {
    "1": "ai4s-pde-task1-burgers-fixed",
    "task1": "ai4s-pde-task1-burgers-fixed",
    "ai4s-pde-task1-burgers-fixed": "ai4s-pde-task1-burgers-fixed",
    "2": "ai4s-pde-task2-burgers-multinu",
    "task2": "ai4s-pde-task2-burgers-multinu",
    "ai4s-pde-task2-burgers-multinu": "ai4s-pde-task2-burgers-multinu",
    "3": "ai4s-pde-task3-ks-multiparam",
    "task3": "ai4s-pde-task3-ks-multiparam",
    "ai4s-pde-task3-ks-multiparam": "ai4s-pde-task3-ks-multiparam",
}

OBSERVED_INPUTS = {
    "ai4s-pde-task1-burgers-fixed": ("data/task1_test.hdf5", 10, (1000, 200, 256)),
    "ai4s-pde-task2-burgers-multinu": ("data/task2_test.h5", 10, (1000, 200, 256)),
    "ai4s-pde-task3-ks-multiparam": ("data/KS_test.hdf5", 20, (100, 400, 256)),
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_task_dir(task: str) -> Path:
    key = task.strip()
    if key not in TASK_DIRS:
        allowed = ", ".join(["1", "2", "3", *sorted(set(TASK_DIRS.values()))])
        raise SystemExit(f"Unknown task '{task}'. Use one of: {allowed}")
    task_dir = TASKS_ROOT / TASK_DIRS[key]
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    return task_dir


def _resolve_api_key(explicit: str | None, env_names: list[str]) -> str:
    if explicit:
        return explicit
    for name in env_names:
        value = os.getenv(name)
        if value:
            return value
    raise SystemExit(
        "No API key found. Set OPENAI_API_KEY, APIFOX_GPT_GE_API_KEY, "
        "VAPI_API_KEY, or pass --api-key-env."
    )


def _load_task_spec(task_dir: Path, output_dir: Path) -> dict[str, Any]:
    config = _read_yaml(task_dir / "config.yaml")
    description_path = PROJECT_ROOT / str(config.get("description", task_dir / "description.md"))
    if not description_path.exists():
        description_path = task_dir / "description.md"
    description = description_path.read_text(encoding="utf-8").strip()

    required_outputs = (
        config.get("submission", {}).get("agent_required_outputs")
        if isinstance(config.get("submission"), dict)
        else None
    )
    if not required_outputs:
        raise ValueError(f"No submission.agent_required_outputs configured in {task_dir / 'config.yaml'}")
    output_name = str(required_outputs[0])
    output_path = (output_dir / task_dir.name / output_name).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    io_instructions = (
        "All task files are visible in the current working directory. "
        "Use the `data/` directory for input HDF5 files. "
        f"Save the final prediction file exactly as `{output_name}` in the current working directory. "
        "Do not create logs, time CSVs, methodology files, or a submission zip for this task run."
    )

    return {
        "task_id": task_dir.name,
        "description": description,
        "data_dir": task_dir.resolve(),
        "output_name": output_name,
        "output_path": output_path,
        "io_instructions": io_instructions,
        "config": config,
    }


def _build_config(args: argparse.Namespace, task_id: str) -> DSLightingConfig:
    existing_pythonpath = os.getenv("PYTHONPATH")
    pythonpath = str(PROJECT_ROOT) if not existing_pythonpath else f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}"

    config = DSLightingConfig(
        run=RunConfig(
            run_name=f"ai4s_aide_{task_id}",
            keep_all_workspaces=args.keep_workspace,
            keep_workspace_on_failure=True,
            parameters={
                "sandbox_env": {
                    "PYTHONPATH": pythonpath,
                    "MPLBACKEND": "Agg",
                    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            },
        ),
        workflow=WorkflowConfig(
            name="aide",
            params={"workspace_base_dir": str(args.workspace_dir.resolve())},
        ),
        llm=LLMConfig(
            model=args.model,
            provider=args.provider,
            api_key=args.api_key,
            api_base=args.api_base,
            temperature=args.temperature,
            max_retries=args.max_retries,
            max_concurrent_per_key=args.max_concurrent_per_key,
        ),
        sandbox=SandboxConfig(
            timeout=args.sandbox_timeout,
            backend=args.sandbox_backend,
            max_cpu_seconds=args.sandbox_timeout,
            max_memory_mb=args.sandbox_memory_mb,
        ),
        data_analysis=DataAnalysisConfig(enabled=args.enable_data_analysis),
    )
    config.agent.search = AgentSearchConfig(
        num_drafts=args.num_drafts,
        debug_prob=args.debug_prob,
        max_iterations=args.max_iterations,
        max_debug_depth=args.max_debug_depth,
    )
    config.run.total_steps = args.max_iterations
    return config


def _build_task(spec: dict[str, Any]) -> TaskDefinition:
    return TaskDefinition(
        task_id=spec["task_id"],
        task_type="kaggle",
        payload={
            "description": spec["description"],
            "io_instructions": spec["io_instructions"],
            "agent_visible_data_dir": str(spec["data_dir"]),
            "public_data_dir": str(spec["data_dir"]),
            "output_submission_path": str(spec["output_path"]),
            "submission_filename": spec["output_name"],
            "submission_format": "hdf5",
            "entries": [
                {
                    "relative_path": spec["output_name"],
                    "kind": "file",
                    "format": "hdf5",
                    "required": True,
                }
            ],
            "metric_name": "AI4S external leaderboard score",
            "lower_is_better": False,
        },
    )


async def _run_one(args: argparse.Namespace, task_dir: Path) -> dict[str, Any]:
    spec = _load_task_spec(task_dir, args.output_dir)
    config = _build_config(args, spec["task_id"])
    task = _build_task(spec)

    if args.dry_run:
        return {
            "task_id": spec["task_id"],
            "data_dir": str(spec["data_dir"]),
            "output_path": str(spec["output_path"]),
            "model": config.llm.model,
            "api_base": config.llm.api_base,
            "provider": config.llm.provider,
            "max_iterations": config.agent.search.max_iterations,
            "dry_run": True,
        }

    runner = DSLightingRunner(config)

    async def _no_local_grade(*_args: Any, **_kwargs: Any) -> float:
        return 0.0

    runner.registry_grader.grade = _no_local_grade
    result, cost, usage = await runner.get_eval_function()(task)
    if not spec["output_path"].exists():
        raise RuntimeError(
            f"AIDE workflow finished without creating the required output file: {spec['output_path']}. "
            f"Workflow result: {result}"
        )
    validation = None if args.skip_validate else validate_prediction(spec["task_id"], spec["output_path"])
    return {
        "task_id": spec["task_id"],
        "output_path": str(spec["output_path"]),
        "result": str(result),
        "cost": cost,
        "usage": usage,
        "validation": validation,
    }


def validate_prediction(task_id: str, output_path: Path) -> dict[str, Any]:
    if task_id not in OBSERVED_INPUTS:
        raise ValueError(f"No validation rule for task: {task_id}")
    input_rel, observed_steps, expected_shape = OBSERVED_INPUTS[task_id]
    input_path = TASKS_ROOT / task_id / input_rel
    if not output_path.exists():
        raise FileNotFoundError(f"Expected prediction file was not created: {output_path}")

    with h5py.File(input_path, "r") as input_h5, h5py.File(output_path, "r") as output_h5:
        if "tensor" not in output_h5:
            raise ValueError(f"{output_path} is missing dataset 'tensor'")
        pred = output_h5["tensor"]
        if tuple(pred.shape) != expected_shape:
            raise ValueError(f"Wrong tensor shape: got {tuple(pred.shape)}, expected {expected_shape}")
        observed = input_h5["tensor"]
        first = pred[:, :observed_steps, :]
        max_observed_error = float(np.max(np.abs(first - observed[:])))
        if max_observed_error > 1e-3:
            raise ValueError(
                f"Observed prefix mismatch: max abs error {max_observed_error:.6g} > 1e-3"
            )
        for start in range(0, pred.shape[0], 32):
            chunk = pred[start : start + 32]
            if not np.isfinite(chunk).all():
                raise ValueError(f"Non-finite values found in prediction chunk starting at sample {start}")
    return {
        "shape": list(expected_shape),
        "observed_steps_checked": observed_steps,
        "max_observed_error": max_observed_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Task selector: 1, 2, 3, task1, task2, task3, or all")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "aide")
    parser.add_argument("--workspace-dir", type=Path, default=PROJECT_ROOT / "runs" / "dslighting_workspace")
    parser.add_argument("--model", default=os.getenv("AI4S_AGENT_MODEL") or "gpt-5.5")
    parser.add_argument("--provider", default=os.getenv("AI4S_AGENT_PROVIDER") or "openai")
    parser.add_argument("--api-base", default=os.getenv("AI4S_AGENT_BASE_URL") or os.getenv("OPENAI_API_BASE") or "http://127.0.0.1:8080/v1")
    parser.add_argument("--api-key-env", action="append", default=["OPENAI_API_KEY", "APIFOX_GPT_GE_API_KEY", "VAPI_API_KEY"])
    parser.add_argument("--temperature", type=float, default=float(os.getenv("AI4S_AGENT_TEMPERATURE", "0.2")))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-concurrent-per-key", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--num-drafts", type=int, default=3)
    parser.add_argument("--debug-prob", type=float, default=0.8)
    parser.add_argument("--max-debug-depth", type=int, default=6)
    parser.add_argument("--sandbox-timeout", type=int, default=12 * 60 * 60)
    parser.add_argument("--sandbox-memory-mb", type=int, default=32768)
    parser.add_argument("--sandbox-backend", default="local", choices=["local", "e2b", "ds_sandbox"])
    parser.add_argument("--enable-data-analysis", action="store_true")
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--llm-debug", action="store_true", help="Enable DSLighting structured LLM debug archive.")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Directory for DSLighting structured debug session.")
    parser.add_argument("--debug-console", action="store_true", help="Print DSLighting structured debug blocks to stdout.")
    parser.add_argument("--provider-raw-debug", action="store_true", help="Enable raw LiteLLM provider debug output.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


async def main_async() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    if args.provider_raw_debug or os.getenv("AI4S_PROVIDER_RAW_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
        os.environ["AI4S_PROVIDER_RAW_DEBUG"] = "1"
        os.environ["LITELLM_LOG"] = "DEBUG"
        os.environ["LITELLM_SET_VERBOSE"] = "True"
        litellm.suppress_debug_info = True
        litellm.set_verbose = True
        litellm._turn_on_debug()
    elif args.llm_debug or os.getenv("AI4S_DSLIGHTING_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
        os.environ["LITELLM_LOG"] = "ERROR"
        os.environ["LITELLM_SET_VERBOSE"] = "False"
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.output_dir = args.output_dir.resolve()
    args.workspace_dir = args.workspace_dir.resolve()
    args.api_key = _resolve_api_key(None, args.api_key_env)

    debug_session = None
    if args.llm_debug or os.getenv("AI4S_DSLIGHTING_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
        from dslighting.debug import init_debug

        debug_output_dir = (args.debug_dir or (args.output_dir.parent / "dslighting_debug")).resolve()
        debug_console = args.debug_console or os.getenv("AI4S_DSLIGHTING_DEBUG_CONSOLE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        debug_session = init_debug(
            enabled=True,
            profile="full",
            output_dir=str(debug_output_dir),
            console_output=debug_console,
        )

    if args.task == "all":
        task_dirs = [_resolve_task_dir("1"), _resolve_task_dir("2"), _resolve_task_dir("3")]
    else:
        task_dirs = [_resolve_task_dir(args.task)]

    try:
        results = []
        for task_dir in task_dirs:
            results.append(await _run_one(args, task_dir))
        if debug_session is not None and debug_session.output_dir is not None:
            results.append({"dslighting_debug_dir": str(debug_session.output_dir)})
        print(json.dumps(results, ensure_ascii=False, indent=2))
    finally:
        if debug_session is not None:
            await debug_session.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
