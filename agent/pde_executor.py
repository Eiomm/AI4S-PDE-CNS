from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .pde_journal import CandidateNode, ExperimentJournal
from .pde_results import RunResult
from .pde_search import Candidate, WeightedEnsembleSearch
from .pde_workflow import Task1FNOWorkflow
from .submission import SubmissionError, validate_submission


@dataclass
class ExperimentExecution:
    success: bool
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    elapsed_seconds: float = 0.0


class ControlledExperimentExecutor:
    """Execute one journal node through a small whitelist of PDE experiment actions."""

    def __init__(
        self,
        *,
        project_root: str | Path = ".",
        code_dir: str | Path = "code",
        workflow: Task1FNOWorkflow | None = None,
        journal: ExperimentJournal | None = None,
        metric: str = "mse",
        maximize: bool = False,
        allowed_validation_commands: tuple[str, ...] = ("python", "pytest"),
        require_code_patch_validation: bool = False,
    ):
        self.project_root = Path(project_root).resolve()
        self.code_dir = self._resolve_code_dir(code_dir)
        self.workflow = workflow
        self.journal = journal
        self.metric = metric
        self.maximize = maximize
        self.allowed_validation_commands = allowed_validation_commands
        self.require_code_patch_validation = require_code_patch_validation

    def execute(self, node: CandidateNode) -> ExperimentExecution:
        started = time.perf_counter()
        try:
            if node.plan.action_type == "code_patch":
                execution = self._execute_code_patch(node)
            elif node.plan.action_type == "weight_search":
                execution = self._execute_weight_search(node)
            elif node.plan.action_type == "finetune":
                execution = self._execute_finetune(node)
            elif node.plan.action_type in {"baseline_train", "baseline_validate", "baseline_ensemble", "baseline_refine"}:
                execution = self._execute_baseline_command(node)
            elif node.plan.action_type == "submit_best":
                execution = self._execute_submit_best(node)
            elif node.plan.action_type == "stop":
                execution = ExperimentExecution(success=True, artifacts={"stopped": True})
            else:
                execution = ExperimentExecution(success=False, error=f"unsupported action_type: {node.plan.action_type}")
        except Exception as exc:
            execution = ExperimentExecution(success=False, error=f"{type(exc).__name__}: {exc}")
        execution.elapsed_seconds = time.perf_counter() - started
        return execution

    def _resolve_code_dir(self, code_dir: str | Path) -> Path:
        path = Path(code_dir)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def _target_for_code_path(self, raw_path: str) -> tuple[Path, str]:
        path = Path(raw_path)
        if path.is_absolute():
            raise ValueError(f"code_patch path is outside code_dir: {raw_path}")
        parts = path.parts
        if parts and parts[0] == self.code_dir.name:
            path = Path(*parts[1:]) if len(parts) > 1 else Path()
        if not path.parts:
            raise ValueError("code_patch path must name a file")
        target = (self.code_dir / path).resolve()
        try:
            rel = target.relative_to(self.code_dir)
        except ValueError as exc:
            raise ValueError(f"code_patch path is outside code_dir: {raw_path}") from exc
        return target, f"{self.code_dir.name}/{rel.as_posix()}"

    def _execute_code_patch(self, node: CandidateNode) -> ExperimentExecution:
        files = node.plan.params.get("files", [])
        if not isinstance(files, list) or not files:
            return ExperimentExecution(success=False, error="code_patch requires a non-empty files list")
        if (
            self.require_code_patch_validation
            and "validation_command" not in node.plan.params
            and "submission_validation_path" not in node.plan.params
        ):
            return ExperimentExecution(
                success=False,
                error="code_patch requires validation_command or submission_validation_path in strict mode",
            )
        patched: list[str] = []
        hashes: dict[str, str] = {}
        for item in files:
            if not isinstance(item, dict):
                return ExperimentExecution(success=False, error="code_patch files entries must be objects")
            raw_path = str(item.get("path", ""))
            content = item.get("content")
            if not isinstance(content, str):
                return ExperimentExecution(success=False, error=f"code_patch content for {raw_path!r} must be a string")
            target, trace_path = self._target_for_code_path(raw_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            patched.append(trace_path)
            hashes[trace_path] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        validation = self._run_optional_validation_command(node)
        artifacts = {
            "patched_files": patched,
            "sha256": hashes,
            "requires_validation": True,
        }
        if validation is not None:
            artifacts["validation"] = validation
            if validation["returncode"] != 0:
                return ExperimentExecution(
                    success=False,
                    artifacts=artifacts,
                    error=f"validation command failed with return code {validation['returncode']}",
                )
        submission_validation = self._run_optional_submission_validation(node)
        if submission_validation is not None:
            artifacts["submission_validation"] = submission_validation
            if not submission_validation["valid"]:
                return ExperimentExecution(
                    success=False,
                    artifacts=artifacts,
                    error=f"submission validation failed: {submission_validation['error']}",
                )
        return ExperimentExecution(
            success=True,
            artifacts=artifacts,
        )

    def _run_optional_validation_command(self, node: CandidateNode) -> dict[str, Any] | None:
        command = node.plan.params.get("validation_command")
        if command is None:
            return None
        if not isinstance(command, list) or not command:
            raise ValueError("validation_command must be a non-empty command list")
        command = [str(arg) for arg in command]
        executable = command[0]
        if executable not in self.allowed_validation_commands:
            raise ValueError(f"validation command is not allowed: {executable}")
        command = [sys.executable if arg == "python" else arg for arg in command]
        timeout = int(node.plan.params.get("validation_timeout_seconds", node.plan.params.get("timeout_seconds", 300)))
        result = subprocess.run(
            command,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }

    def _run_optional_submission_validation(self, node: CandidateNode) -> dict[str, Any] | None:
        raw_path = node.plan.params.get("submission_validation_path")
        if raw_path is None:
            return None
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("submission_validation_path must be a non-empty string")
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.project_root / path
        path = path.resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(f"submission_validation_path is outside project_root: {raw_path}") from exc
        try:
            report = validate_submission(path)
        except (SubmissionError, ValueError, OSError) as exc:
            return {
                "path": str(path),
                "valid": False,
                "tasks": [],
                "messages": [],
                "error": str(exc),
            }
        return {
            "path": str(path),
            "valid": report.valid,
            "tasks": report.tasks,
            "messages": report.messages,
            "error": None,
        }

    def _execute_weight_search(self, node: CandidateNode) -> ExperimentExecution:
        if self.workflow is None:
            return ExperimentExecution(success=False, error="weight_search requires a Task1FNOWorkflow")
        candidates_payload = node.plan.params.get("candidates", [])
        if not isinstance(candidates_payload, list) or not candidates_payload:
            return ExperimentExecution(success=False, error="weight_search requires candidate weights")
        candidates = []
        for index, item in enumerate(candidates_payload, start=1):
            if not isinstance(item, dict) or not isinstance(item.get("weights"), dict):
                return ExperimentExecution(success=False, error="each weight_search candidate needs weights")
            candidates.append(Candidate(name=str(item.get("name", f"candidate-{index}")), weights=dict(item["weights"])))
        search = WeightedEnsembleSearch(
            workflow=self.workflow,
            candidates=candidates,
            search_name=str(Path("autonomous") / node.id),
            metric=str(node.plan.params.get("metric", self.metric)),
            maximize=bool(node.plan.params.get("maximize", self.maximize)),
        )
        search_result = search.run(make_submission=bool(node.plan.params.get("make_submission", False)))
        if search_result.best_validation_result is None or search_result.best_candidate is None:
            return ExperimentExecution(success=False, error="weight_search produced no successful validation result")
        best = search_result.best_validation_result
        return ExperimentExecution(
            success=best.success,
            metrics=dict(best.metrics),
            artifacts={
                "best_candidate": {
                    "name": search_result.best_candidate.name,
                    "weights": search_result.best_candidate.weights,
                },
                "run_dir": str(best.run_dir),
                "prediction_path": str(best.prediction_path) if best.prediction_path else None,
                "zip_path": str(search_result.best_submission_result.zip_path)
                if search_result.best_submission_result and search_result.best_submission_result.zip_path
                else None,
                "candidate_results": [
                    self._candidate_result_artifact(candidate, run_result)
                    for candidate, run_result in search_result.candidate_results
                ],
            },
            error=best.error,
        )

    @staticmethod
    def _candidate_result_artifact(candidate: Candidate, run_result: RunResult) -> dict[str, Any]:
        return {
            "name": candidate.name,
            "weights": dict(candidate.weights),
            "metrics": dict(run_result.metrics),
            "run_dir": str(run_result.run_dir),
            "prediction_path": str(run_result.prediction_path) if run_result.prediction_path else None,
            "zip_path": str(run_result.zip_path) if run_result.zip_path else None,
            "success": run_result.success,
            "error": run_result.error,
        }

    def _execute_finetune(self, node: CandidateNode) -> ExperimentExecution:
        return self._execute_command_node(node, required_label="finetune")

    def _execute_baseline_command(self, node: CandidateNode) -> ExperimentExecution:
        return self._execute_command_node(node, required_label=node.plan.action_type)

    def _execute_command_node(self, node: CandidateNode, *, required_label: str) -> ExperimentExecution:
        command = node.plan.params.get("command")
        if not isinstance(command, list) or not command:
            return ExperimentExecution(success=False, error=f"{required_label} requires a command list")
        command = [sys.executable if arg == "python" else str(arg) for arg in command]
        timeout = int(node.plan.params.get("timeout_seconds", 3600))
        result = subprocess.run(
            command,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        artifacts = {
            "command": command,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
        metrics_path = node.plan.params.get("metrics_path")
        metrics: dict[str, float] = {}
        if isinstance(metrics_path, str):
            path = (self.project_root / metrics_path).resolve()
            if path.exists():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                metrics = {key: float(value) for key, value in loaded.items() if isinstance(value, int | float)}
        return ExperimentExecution(
            success=result.returncode == 0,
            metrics=metrics,
            artifacts=artifacts,
            error=None if result.returncode == 0 else f"{required_label} command failed with return code {result.returncode}",
        )

    def _execute_submit_best(self, node: CandidateNode) -> ExperimentExecution:
        if self.workflow is None:
            return ExperimentExecution(success=False, error="submit_best requires a Task1FNOWorkflow")
        weights = node.plan.params.get("weights")
        if not isinstance(weights, dict) and self.journal is not None:
            best = self.journal.best(metric=self.metric, maximize=self.maximize)
            if best is not None:
                candidate = best.artifacts.get("best_candidate")
                if isinstance(candidate, dict) and isinstance(candidate.get("weights"), dict):
                    weights = candidate["weights"]
        if not isinstance(weights, dict):
            return ExperimentExecution(success=False, error="submit_best requires weights or a prior best_candidate")
        result: RunResult = self.workflow.run_test_submission(
            weights,
            run_name=str(Path("autonomous") / node.id / "submission"),
            train_time=float(node.plan.params.get("train_time", 0.0)),
        )
        return ExperimentExecution(
            success=result.success,
            metrics=dict(result.metrics),
            artifacts=result.to_json_dict(),
            error=result.error,
        )
