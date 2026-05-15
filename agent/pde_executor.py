from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .code_generation_workspace import apply_agent_code_patch
from .pde_journal import CandidateNode, ExperimentJournal
from .pde_metrics import compute_task1_metrics, mse, relative_mse
from .pde_observer import observe_research_context
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

    TASK1_SCORE_SEGMENTS: tuple[tuple[str, int, int], ...] = (
        ("seg1", 10, 57),
        ("seg2", 57, 105),
        ("seg3", 105, 200),
    )

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
            if node.plan.action_type == "inspect_data":
                execution = self._execute_inspect_data(node)
            elif node.plan.action_type == "code_patch":
                execution = self._execute_code_patch(node)
            elif node.plan.action_type == "weight_search":
                execution = self._execute_weight_search(node)
            elif node.plan.action_type == "postprocess_search":
                execution = self._execute_postprocess_search(node)
            elif node.plan.action_type == "finetune":
                execution = self._execute_finetune(node)
            elif node.plan.action_type == "finetune_checkpoint":
                execution = self._execute_finetune_checkpoint(node)
            elif node.plan.action_type in {"baseline_train", "baseline_validate", "baseline_ensemble", "baseline_refine"}:
                execution = self._execute_baseline_command(node)
            elif node.plan.action_type == "task2_train_model":
                execution = self._execute_task2_train_model(node)
            elif node.plan.action_type == "task2_submit_best":
                execution = self._execute_task2_submit_best(node)
            elif node.plan.action_type == "evaluate_candidate":
                execution = self._execute_evaluate_candidate(node)
            elif node.plan.action_type == "validate_submission":
                execution = self._execute_validate_submission(node)
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

    def _execute_inspect_data(self, node: CandidateNode) -> ExperimentExecution:
        state = observe_research_context(self.project_root)
        output_path = self.project_root / str(node.plan.params.get("output_path", "runs/observer_state.json"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return ExperimentExecution(success=True, artifacts={"observer_state_path": str(output_path), "observer_state": state})

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

    def _node_code_snapshot_dir(self, node: CandidateNode) -> Path:
        requested = node.plan.params.get("code_snapshot_dir")
        if isinstance(requested, str) and requested.strip():
            path = Path(requested)
            if not path.is_absolute():
                path = self.project_root / path
            return path.resolve()
        if self.journal is not None:
            return (self.journal.path.parent / "nodes" / node.id / "code").resolve()
        return (self.project_root / "runs" / "code-patches" / node.id / "code").resolve()

    def _execute_code_patch(self, node: CandidateNode) -> ExperimentExecution:
        files = node.plan.params.get("files", [])
        normalized_schema: str | None = None
        if (not isinstance(files, list) or not files) and isinstance(node.plan.params.get("patches"), list):
            files = node.plan.params["patches"]
            normalized_schema = "patches"
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
        if normalized_schema:
            artifacts["normalized_code_patch_schema"] = normalized_schema
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
        snapshot = apply_agent_code_patch(
            code_root=self._node_code_snapshot_dir(node),
            files=files,
            provenance_record={
                "node_id": node.id,
                "action_type": node.plan.action_type,
                "hypothesis": node.plan.hypothesis,
            },
        )
        artifacts["code_snapshot_dir"] = str(snapshot.code_root)
        artifacts["code_manifest_path"] = str(snapshot.manifest_path)
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
        if any(
            isinstance(item, dict)
            and (isinstance(item.get("checkpoint_overrides"), dict) or isinstance(item.get("checkpoint"), str))
            for item in candidates_payload
        ):
            return self._execute_checkpoint_candidate_search(node, candidates_payload)
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

    def _execute_checkpoint_candidate_search(
        self,
        node: CandidateNode,
        candidates_payload: list[Any],
    ) -> ExperimentExecution:
        if self.workflow is None:
            return ExperimentExecution(success=False, error="checkpoint candidate search requires a Task1FNOWorkflow")
        metric = str(node.plan.params.get("metric", self.metric))
        maximize = bool(node.plan.params.get("maximize", self.maximize))
        results: list[dict[str, Any]] = []
        best_result: RunResult | None = None
        best_candidate: dict[str, Any] | None = None
        previous_checkpoint_paths = dict(getattr(self.workflow, "checkpoint_paths", {}))
        try:
            for index, item in enumerate(candidates_payload, start=1):
                if not isinstance(item, dict):
                    return ExperimentExecution(success=False, error="checkpoint candidate entries must be objects")
                checkpoint_overrides = item.get("checkpoint_overrides")
                if not isinstance(checkpoint_overrides, dict):
                    checkpoint = item.get("checkpoint")
                    if not isinstance(checkpoint, str):
                        return ExperimentExecution(
                            success=False,
                            error="checkpoint candidate requires checkpoint_overrides or checkpoint",
                        )
                    checkpoint_overrides = {"nu0.001": checkpoint}
                weights = item.get("task1_weights", item.get("weights", {"nu0.001": 1.0}))
                if not isinstance(weights, dict):
                    return ExperimentExecution(success=False, error="checkpoint candidate weights must be an object")
                if hasattr(self.workflow, "checkpoint_paths"):
                    self.workflow.checkpoint_paths = dict(previous_checkpoint_paths)
                    self.workflow.checkpoint_paths.update({key: Path(value) for key, value in checkpoint_overrides.items()})
                name = str(item.get("name", f"checkpoint-candidate-{index}"))
                result = self.workflow.run_validation(weights, run_name=str(Path("autonomous") / node.id / name))
                record = self._candidate_result_artifact(Candidate(name=name, weights=dict(weights)), result)
                record["checkpoint_overrides"] = dict(checkpoint_overrides)
                results.append(record)
                if not result.success or metric not in result.metrics:
                    continue
                if best_result is None:
                    best_result = result
                    best_candidate = {"name": name, "task1_weights": dict(weights), "checkpoint_overrides": dict(checkpoint_overrides)}
                    continue
                current = float(result.metrics[metric])
                best_value = float(best_result.metrics[metric])
                if (maximize and current > best_value) or (not maximize and current < best_value):
                    best_result = result
                    best_candidate = {"name": name, "task1_weights": dict(weights), "checkpoint_overrides": dict(checkpoint_overrides)}
        finally:
            if hasattr(self.workflow, "checkpoint_paths"):
                self.workflow.checkpoint_paths = previous_checkpoint_paths
        if best_result is None or best_candidate is None:
            return ExperimentExecution(
                success=False,
                artifacts={"candidate_results": results},
                error=f"checkpoint candidate search produced no successful result with metric {metric}",
            )
        return ExperimentExecution(
            success=True,
            metrics=dict(best_result.metrics),
            artifacts={
                "best_candidate": best_candidate,
                "run_dir": str(best_result.run_dir),
                "prediction_path": str(best_result.prediction_path) if best_result.prediction_path else None,
                "candidate_results": results,
            },
            error=best_result.error,
        )

    def _execute_postprocess_search(self, node: CandidateNode) -> ExperimentExecution:
        base_candidate = node.plan.params.get("base_candidate")
        if not isinstance(base_candidate, dict) and isinstance(node.plan.params.get("checkpoint"), str):
            base_candidate = {
                "name": str(node.plan.params.get("candidate_name", "checkpoint-candidate")),
                "task1_weights": {"nu0.001": 1.0},
                "checkpoint_overrides": {"nu0.001": str(node.plan.params["checkpoint"])},
            }
        if not isinstance(base_candidate, dict) and isinstance(node.plan.params.get("source_candidate"), str):
            base_candidate = self._resolve_journal_candidate(str(node.plan.params["source_candidate"]))
        if isinstance(base_candidate, dict):
            return self._execute_base_candidate_postprocess_search(node, base_candidate)
        fno_path = self._resolve_project_path(
            str(node.plan.params.get("fno_prediction_path", "runs/_verify_official_ensemble/task1_val_fno_only.hdf5"))
        )
        unet_path = self._resolve_project_path(
            str(node.plan.params.get("unet_prediction_path", "runs/_verify_official_ensemble/task1_val_unet_only.hdf5"))
        )
        target_path = self._resolve_project_path(
            str(node.plan.params.get("target_path", node.plan.params.get("val_hdf5", "data/Task1/task1_val.hdf5")))
        )
        fno = self._read_hdf5_array(fno_path).astype(np.float32)
        unet = self._read_hdf5_array(unet_path).astype(np.float32)
        target = self._read_hdf5_array(target_path).astype(np.float32)
        if fno.shape != unet.shape or fno.shape != target.shape:
            return ExperimentExecution(
                success=False,
                error=f"postprocess_search shape mismatch: fno={fno.shape}, unet={unet.shape}, target={target.shape}",
            )
        if fno.ndim != 3 or fno.shape[1:] != (200, 256):
            return ExperimentExecution(success=False, error=f"postprocess_search expects (N, 200, 256), got {fno.shape}")
        fno[:, :10, :] = target[:, :10, :]
        unet[:, :10, :] = target[:, :10, :]

        weight_grid = self._float_grid(node.plan.params.get("segment_fno_grid"), default_step=0.01)
        alpha_grid = self._float_grid(
            node.plan.params.get("persistence_alpha_grid", node.plan.params.get("alpha_grid")),
            default_step=0.01,
        )
        segment_weights = self._best_segment_fno_weights(fno, unet, target, weight_grid)
        segment_prediction = self._blend_segments(fno, unet, target, segment_weights)
        segment_metrics = compute_task1_metrics(segment_prediction, target)
        alpha_by_segment = self._best_persistence_alpha(segment_prediction, target, alpha_grid)
        postprocessed = self._apply_persistence_alpha(segment_prediction, target, alpha_by_segment)
        post_metrics = compute_task1_metrics(postprocessed, target)

        run_dir = self._postprocess_run_dir(node)
        prediction_path = run_dir / "task1_val_pred.hdf5"
        self._write_hdf5_prediction(prediction_path, postprocessed)
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(post_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        weights_artifact = {
            **{f"segment_fno_{key}": value for key, value in segment_weights.items()},
            **{f"persistence_alpha_{key}": value for key, value in alpha_by_segment.items()},
        }
        extra_args = [
            "--segment-fno-weights",
            str(segment_weights["seg1"]),
            str(segment_weights["seg2"]),
            str(segment_weights["seg3"]),
            "--persistence-segment-alpha",
            str(alpha_by_segment["seg1"]),
            str(alpha_by_segment["seg2"]),
            str(alpha_by_segment["seg3"]),
        ]
        best_candidate = {
            "name": "segment-persistence-postprocess",
            "weights": weights_artifact,
            "task1_weights": {"nu0.001": 0.12, "unet_pf20_nu0.001": 0.88},
            "task1_extra_inference_args": extra_args,
        }
        candidate_results = [
            {
                "name": "segment-official-blend",
                "weights": {f"segment_fno_{key}": value for key, value in segment_weights.items()},
                "metrics": segment_metrics,
                "run_dir": str(run_dir),
                "prediction_path": None,
                "success": True,
                "error": None,
            },
            {
                "name": best_candidate["name"],
                "weights": weights_artifact,
                "metrics": post_metrics,
                "run_dir": str(run_dir),
                "prediction_path": str(prediction_path),
                "success": True,
                "error": None,
            },
        ]
        search_payload = {
            "fno_prediction_path": str(fno_path),
            "unet_prediction_path": str(unet_path),
            "target_path": str(target_path),
            "best_candidate": best_candidate,
            "candidate_results": candidate_results,
        }
        (run_dir / "postprocess_search.json").write_text(
            json.dumps(search_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        artifacts: dict[str, Any] = {
            "best_candidate": best_candidate,
            "run_dir": str(run_dir),
            "prediction_path": str(prediction_path),
            "metrics_path": str(metrics_path),
            "candidate_results": candidate_results,
        }
        if bool(node.plan.params.get("make_submission", False)):
            if self.workflow is None:
                return ExperimentExecution(
                    success=False,
                    metrics=post_metrics,
                    artifacts=artifacts,
                    error="postprocess_search make_submission requires a Task1FNOWorkflow",
                )
            submission = self.workflow.run_test_submission(
                best_candidate["task1_weights"],
                run_name=str(Path("autonomous") / node.id / "submission"),
                train_time=float(node.plan.params.get("train_time", 0.0)),
                extra_inference_args=extra_args,
            )
            artifacts["zip_path"] = str(submission.zip_path) if submission.zip_path else None
            artifacts["submission"] = submission.to_json_dict()
            if not submission.success:
                return ExperimentExecution(success=False, metrics=post_metrics, artifacts=artifacts, error=submission.error)
        return ExperimentExecution(success=True, metrics=post_metrics, artifacts=artifacts)

    def _resolve_journal_candidate(self, name: str) -> dict[str, Any] | None:
        if self.journal is None:
            return None
        for journal_node in reversed(self.journal.read()):
            candidate = journal_node.artifacts.get("best_candidate")
            if isinstance(candidate, dict) and str(candidate.get("name", "")) == name:
                return candidate
        best = self.journal.best(metric=self.metric, maximize=self.maximize)
        if best is None:
            return None
        candidate = best.artifacts.get("best_candidate")
        return candidate if isinstance(candidate, dict) else None

    def _execute_base_candidate_postprocess_search(
        self,
        node: CandidateNode,
        base_candidate: dict[str, Any],
    ) -> ExperimentExecution:
        if self.workflow is None:
            return ExperimentExecution(success=False, error="base_candidate postprocess_search requires a Task1FNOWorkflow")
        weights = base_candidate.get("task1_weights", base_candidate.get("weights"))
        if not isinstance(weights, dict):
            if isinstance(base_candidate.get("checkpoint_overrides"), dict):
                weights = {"nu0.001": 1.0}
            else:
                return ExperimentExecution(success=False, error="base_candidate requires task1_weights or weights")
        checkpoint_overrides = base_candidate.get("checkpoint_overrides")
        previous_checkpoint_paths = None
        if isinstance(checkpoint_overrides, dict) and hasattr(self.workflow, "checkpoint_paths"):
            previous_checkpoint_paths = dict(self.workflow.checkpoint_paths)
            self.workflow.checkpoint_paths.update({key: Path(value) for key, value in checkpoint_overrides.items()})
        try:
            validation = self.workflow.run_validation(
                weights,
                run_name=str(Path("autonomous") / node.id / "postprocess_base"),
            )
        finally:
            if previous_checkpoint_paths is not None:
                self.workflow.checkpoint_paths = previous_checkpoint_paths
        if not validation.success or validation.prediction_path is None:
            return ExperimentExecution(
                success=False,
                artifacts=validation.to_json_dict(),
                error=validation.error or "base_candidate validation did not produce a prediction",
            )
        target_path = self._resolve_project_path(str(node.plan.params.get("target_path", "data/Task1/task1_val.hdf5")))
        base_prediction = self._read_hdf5_array(Path(validation.prediction_path)).astype(np.float32)
        target = self._read_hdf5_array(target_path).astype(np.float32)
        if base_prediction.shape != target.shape:
            return ExperimentExecution(
                success=False,
                error=f"base_candidate postprocess shape mismatch: prediction={base_prediction.shape}, target={target.shape}",
            )
        base_prediction[:, :10, :] = target[:, :10, :]
        alpha_grid = self._float_grid(
            node.plan.params.get(
                "persistence_alpha_grid",
                node.plan.params.get("alpha_grid", node.plan.params.get("blend_alpha_grid")),
            ),
            default_step=0.01,
        )
        base_metrics = compute_task1_metrics(base_prediction, target)
        alpha_by_segment = self._best_persistence_alpha(base_prediction, target, alpha_grid)
        postprocessed = self._apply_persistence_alpha(base_prediction, target, alpha_by_segment)
        post_metrics = compute_task1_metrics(postprocessed, target)

        run_dir = self._postprocess_run_dir(node)
        prediction_path = run_dir / "task1_val_pred.hdf5"
        self._write_hdf5_prediction(prediction_path, postprocessed)
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(post_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        extra_args = [
            "--persistence-segment-alpha",
            str(alpha_by_segment["seg1"]),
            str(alpha_by_segment["seg2"]),
            str(alpha_by_segment["seg3"]),
        ]
        best_candidate = {
            "name": f"{base_candidate.get('name', 'base-candidate')}-persistence-postprocess",
            "weights": {f"persistence_alpha_{key}": value for key, value in alpha_by_segment.items()},
            "task1_weights": dict(weights),
            "task1_extra_inference_args": extra_args,
        }
        if isinstance(checkpoint_overrides, dict):
            best_candidate["checkpoint_overrides"] = dict(checkpoint_overrides)
        candidate_results = [
            {
                "name": str(base_candidate.get("name", "base-candidate")),
                "weights": dict(weights),
                "metrics": base_metrics,
                "run_dir": str(validation.run_dir),
                "prediction_path": str(validation.prediction_path),
                "success": True,
                "error": None,
            },
            {
                "name": best_candidate["name"],
                "weights": dict(best_candidate["weights"]),
                "metrics": post_metrics,
                "run_dir": str(run_dir),
                "prediction_path": str(prediction_path),
                "success": True,
                "error": None,
            },
        ]
        search_payload = {
            "base_candidate": base_candidate,
            "target_path": str(target_path),
            "best_candidate": best_candidate,
            "candidate_results": candidate_results,
        }
        (run_dir / "postprocess_search.json").write_text(
            json.dumps(search_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return ExperimentExecution(
            success=True,
            metrics={key: float(value) for key, value in post_metrics.items()},
            artifacts={
                "best_candidate": best_candidate,
                "run_dir": str(run_dir),
                "prediction_path": str(prediction_path),
                "metrics_path": str(metrics_path),
                "candidate_results": candidate_results,
            },
        )

    def _resolve_project_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.project_root / path
        path = path.resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(f"path is outside project_root: {raw_path}") from exc
        return path

    @staticmethod
    def _read_hdf5_array(path: Path) -> np.ndarray:
        with h5py.File(path, "r") as h5:
            if "prediction" in h5:
                return h5["prediction"][:]
            if "tensor" in h5:
                return h5["tensor"][:]
            keys = list(h5.keys())
            if len(keys) == 1:
                return h5[keys[0]][:]
            raise KeyError(f"{path} must contain 'prediction', 'tensor', or exactly one dataset")

    @staticmethod
    def _write_hdf5_prediction(path: Path, prediction: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h5:
            h5.create_dataset("prediction", data=prediction.astype(np.float32))

    def _postprocess_run_dir(self, node: CandidateNode) -> Path:
        run_root = Path(getattr(self.workflow, "run_root", self.project_root / "runs"))
        if not run_root.is_absolute():
            run_root = self.project_root / run_root
        run_dir = run_root / "autonomous" / node.id / "postprocess_search"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _float_grid(raw: Any, *, default_step: float) -> list[float]:
        if raw is None:
            count = int(round(1.0 / default_step))
            return [round(index * default_step, 6) for index in range(count + 1)]
        if not isinstance(raw, list) or not raw:
            raise ValueError("grid must be a non-empty list of floats")
        values = sorted({round(float(value), 6) for value in raw})
        if values[0] < 0.0 or values[-1] > 1.0:
            raise ValueError("grid values must be within [0, 1]")
        return values

    def _best_segment_fno_weights(
        self,
        fno: np.ndarray,
        unet: np.ndarray,
        target: np.ndarray,
        grid: list[float],
    ) -> dict[str, float]:
        weights: dict[str, float] = {}
        for label, start, end in self.TASK1_SCORE_SEGMENTS:
            best_score = None
            best_weight = 0.0
            for weight in grid:
                segment = weight * fno[:, start:end, :] + (1.0 - weight) * unet[:, start:end, :]
                score = self._segment_score(label, segment, target[:, start:end, :])
                if best_score is None or score > best_score:
                    best_score = score
                    best_weight = weight
            weights[label] = best_weight
        return weights

    def _blend_segments(
        self,
        fno: np.ndarray,
        unet: np.ndarray,
        target: np.ndarray,
        weights: dict[str, float],
    ) -> np.ndarray:
        prediction = np.empty_like(target, dtype=np.float32)
        prediction[:, :10, :] = target[:, :10, :]
        for label, start, end in self.TASK1_SCORE_SEGMENTS:
            weight = weights[label]
            prediction[:, start:end, :] = weight * fno[:, start:end, :] + (1.0 - weight) * unet[:, start:end, :]
        return prediction

    def _best_persistence_alpha(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        grid: list[float],
    ) -> dict[str, float]:
        persistence = np.empty_like(target, dtype=np.float32)
        persistence[:, :10, :] = target[:, :10, :]
        persistence[:, 10:, :] = target[:, 9:10, :]
        alpha_by_segment: dict[str, float] = {}
        for label, start, end in self.TASK1_SCORE_SEGMENTS:
            best_score = None
            best_alpha = 1.0
            for alpha in grid:
                segment = alpha * prediction[:, start:end, :] + (1.0 - alpha) * persistence[:, start:end, :]
                score = self._segment_score(label, segment, target[:, start:end, :])
                if best_score is None or score > best_score:
                    best_score = score
                    best_alpha = alpha
            alpha_by_segment[label] = best_alpha
        return alpha_by_segment

    def _apply_persistence_alpha(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        alpha_by_segment: dict[str, float],
    ) -> np.ndarray:
        persistence = np.empty_like(target, dtype=np.float32)
        persistence[:, :10, :] = target[:, :10, :]
        persistence[:, 10:, :] = target[:, 9:10, :]
        postprocessed = prediction.copy()
        for label, start, end in self.TASK1_SCORE_SEGMENTS:
            alpha = alpha_by_segment[label]
            postprocessed[:, start:end, :] = (
                alpha * prediction[:, start:end, :]
                + (1.0 - alpha) * persistence[:, start:end, :]
            )
        postprocessed[:, :10, :] = target[:, :10, :]
        return postprocessed.astype(np.float32)

    @staticmethod
    def _segment_score(label: str, prediction: np.ndarray, target: np.ndarray) -> float:
        if label == "seg1":
            return float(100.0 * np.exp(-20.0 * relative_mse(prediction, target)))
        if label == "seg2":
            return float(100.0 * np.exp(-10.0 * relative_mse(prediction, target)))
        rmse = float(mse(prediction, target) ** 0.5)
        return 100.0 / (1.0 + 10.0 * rmse)

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

    def build_finetune_checkpoint_command(self, plan: Any) -> list[str]:
        params = plan.params
        base_checkpoint = str(
            params.get("base_checkpoint", "checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt")
        )
        run_dir = str(params.get("run_dir", Path("runs") / "agent-finetune-checkpoint"))
        command = [
            "python",
            "code/train_task1_fno_finetune.py",
            "--train-hdf5",
            str(params.get("train_hdf5", "data/pdebench_burgers/raw/1D_Burgers_Sols_Nu0.001.hdf5")),
            "--base-checkpoint",
            base_checkpoint,
            "--run-dir",
            run_dir,
            "--val-hdf5",
            str(params.get("val_hdf5", "data/Task1/task1_val.hdf5")),
            "--steps",
            str(int(params.get("steps", 500))),
            "--batch-size",
            str(int(params.get("batch_size", 8))),
            "--eval-batch-size",
            str(int(params.get("eval_batch_size", 50))),
            "--lr",
            str(float(params.get("lr", 1.0e-5))),
            "--weight-decay",
            str(float(params.get("weight_decay", 0.0))),
            "--grad-clip",
            str(float(params.get("grad_clip", 0.1))),
            "--gradient-loss-weight",
            str(float(params.get("gradient_loss_weight", 0.0))),
            "--spectral-loss-weight",
            str(float(params.get("spectral_loss_weight", 0.0))),
            "--physics-loss-weight",
            str(float(params.get("physics_loss_weight", 0.0))),
            "--physics-nu",
            str(float(params.get("physics_nu", 0.001))),
            "--physics-dt",
            str(float(params.get("physics_dt", 0.025))),
            "--physics-dx",
            str(float(params.get("physics_dx", 1.0 / 256.0))),
            "--horizon-loss-gamma",
            str(float(params.get("horizon_loss_gamma", 1.0))),
            "--architecture",
            str(params.get("architecture", "fno")),
            "--temporal-stride",
            str(int(params.get("temporal_stride", 5))),
            "--rollout-steps",
            str(int(params.get("rollout_steps", 1))),
            "--trainable",
            str(params.get("trainable", "all")),
            "--residual-head-hidden",
            str(int(params.get("residual_head_hidden", 32))),
            "--residual-head-scale",
            str(float(params.get("residual_head_scale", 1.0))),
            "--val-every",
            str(int(params.get("val_every", params.get("steps", 500)))),
            "--log-every",
            str(int(params.get("log_every", 100))),
            "--max-samples",
            str(int(params.get("max_samples", 2048))),
            "--val-max-samples",
            str(int(params.get("val_max_samples", 100))),
            "--seed",
            str(int(params.get("seed", 0))),
        ]
        if params.get("device"):
            command.extend(["--device", str(params["device"])])
        return command

    def _execute_finetune_checkpoint(self, node: CandidateNode) -> ExperimentExecution:
        params = dict(node.plan.params)
        if "run_dir" not in params:
            params["run_dir"] = (Path("runs") / "autonomous" / node.id / "finetune_checkpoint").as_posix()
        duplicate = self._duplicate_journal_run_dir(node, str(params["run_dir"]))
        if duplicate is not None:
            return ExperimentExecution(
                success=False,
                error=(
                    f"finetune_checkpoint run_dir already used by node "
                    f"{duplicate.id[:8]}: {params['run_dir']}"
                ),
            )
        if "command" not in params:
            command_plan = type(node.plan)(
                intent=node.plan.intent,
                hypothesis=node.plan.hypothesis,
                action_type=node.plan.action_type,
                params=params,
                expected_effect=node.plan.expected_effect,
                risk=node.plan.risk,
            )
            params["command"] = self.build_finetune_checkpoint_command(command_plan)
        if "metrics_path" not in params:
            run_dir = Path(str(params.get("run_dir", "runs/agent-finetune-checkpoint")))
            params["metrics_path"] = (run_dir / "finetune_result.json").as_posix()
        proxy_node = type(
            "FinetuneCheckpointNode",
            (),
            {
                "plan": type(node.plan)(
                    intent=node.plan.intent,
                    hypothesis=node.plan.hypothesis,
                    action_type="finetune_checkpoint",
                    params=params,
                    expected_effect=node.plan.expected_effect,
                    risk=node.plan.risk,
                )
            },
        )()
        execution = self._execute_command_node(proxy_node, required_label="finetune_checkpoint")
        run_dir = self._resolve_project_path(str(params.get("run_dir", "runs/agent-finetune-checkpoint")))
        best_checkpoint = run_dir / "best.pt"
        execution.artifacts["best_checkpoint"] = str(best_checkpoint) if best_checkpoint.exists() else None
        execution.artifacts["finetune_result_path"] = str(run_dir / "finetune_result.json")
        if best_checkpoint.exists():
            execution.artifacts["best_candidate"] = {
                "name": str(params.get("name", "finetuned-fno")),
                "task1_weights": {"nu0.001": 1.0},
                "checkpoint_overrides": {"nu0.001": str(best_checkpoint)},
            }
        return execution

    def _duplicate_journal_run_dir(self, node: CandidateNode, raw_run_dir: str) -> CandidateNode | None:
        if self.journal is None:
            return None
        target = self._normalize_run_dir(raw_run_dir)
        for existing in self.journal.read():
            if existing.id == getattr(node, "id", None):
                continue
            if self._normalize_run_dir(existing.plan.params.get("run_dir")) == target:
                return existing
            artifact_run_dir = existing.artifacts.get("run_dir")
            if self._normalize_run_dir(artifact_run_dir) == target:
                return existing
        return None

    def _normalize_run_dir(self, raw_run_dir: Any) -> str | None:
        if not isinstance(raw_run_dir, str) or not raw_run_dir.strip():
            return None
        path = Path(raw_run_dir)
        if not path.is_absolute():
            path = self.project_root / path
        try:
            return path.resolve().relative_to(self.project_root).as_posix().lower()
        except ValueError:
            return path.resolve().as_posix().lower()

    def _execute_baseline_command(self, node: CandidateNode) -> ExperimentExecution:
        return self._execute_command_node(node, required_label=node.plan.action_type)

    def _execute_task2_train_model(self, node: CandidateNode) -> ExperimentExecution:
        params = dict(node.plan.params)
        model = str(params.get("model", "minifno_nu"))
        output_dir = str(params.get("output_dir", Path("runs") / "autonomous" / node.id / "task2_model"))
        if "task1" in output_dir.lower():
            return ExperimentExecution(success=False, error="Task2 output_dir must not reference Task1")
        command = [
            "python",
            "code/train_task2_models.py",
            "--model",
            model,
            "--output-dir",
            output_dir,
            "--epochs",
            str(int(params.get("epochs", 1))),
            "--batch-size",
            str(int(params.get("batch_size", 8))),
            "--lr",
            str(float(params.get("lr", 1.0e-3))),
            "--hidden-channels",
            str(int(params.get("hidden_channels", 48))),
            "--modes",
            str(int(params.get("modes", 32))),
            "--seed",
            str(int(params.get("seed", 13))),
        ]
        for key, flag in [
            ("sample_limit", "--sample-limit"),
            ("val_sample_limit", "--val-sample-limit"),
            ("device", "--device"),
            ("nu_aux_weight", "--nu-aux-weight"),
            ("min_relative_improvement", "--min-relative-improvement"),
        ]:
            if key in params and params[key] is not None:
                command.extend([flag, str(params[key])])
        if bool(params.get("promote_if_better", False)):
            command.append("--promote-if-better")
            if "prediction_output" in params:
                command.extend(["--prediction-output", str(params["prediction_output"])])
        metrics_path = Path(output_dir) / f"task2_{model}_metrics.json"
        proxy = type(
            "Task2TrainNode",
            (),
            {
                "plan": type(node.plan)(
                    intent=node.plan.intent,
                    hypothesis=node.plan.hypothesis,
                    action_type="task2_train_model",
                    params={**params, "command": command, "metrics_path": metrics_path.as_posix()},
                    expected_effect=node.plan.expected_effect,
                    risk=node.plan.risk,
                )
            },
        )()
        execution = self._execute_command_node(proxy, required_label="task2_train_model")
        execution.artifacts.setdefault("output_dir", output_dir)
        execution.artifacts.setdefault("metrics_path", str(self.project_root / metrics_path))
        payload = execution.artifacts.get("metrics_payload")
        if isinstance(payload, dict):
            selected = payload.get("selected")
            if isinstance(selected, dict):
                execution.artifacts["best_candidate"] = {
                    "name": str(selected.get("model_name", model)),
                    "task": "task2",
                    "checkpoint_path": selected.get("checkpoint_path"),
                    "train_time": selected.get("train_time"),
                    "metrics": selected.get("metrics", {}),
                }
        return execution

    def _execute_task2_submit_best(self, node: CandidateNode) -> ExperimentExecution:
        checkpoint_path = node.plan.params.get("checkpoint_path")
        train_time = node.plan.params.get("train_time")
        candidate: dict[str, Any] | None = None
        if not isinstance(checkpoint_path, str) and self.journal is not None:
            best = self.journal.best(metric=self.metric, maximize=self.maximize)
            if best is not None and isinstance(best.artifacts.get("best_candidate"), dict):
                candidate = best.artifacts["best_candidate"]
                checkpoint_path = candidate.get("checkpoint_path")
                train_time = candidate.get("train_time", train_time)
        if not isinstance(checkpoint_path, str):
            return self._execute_task2_persistence_submission(node)
        if "task1" in checkpoint_path.lower():
            return ExperimentExecution(success=False, error=f"Task2 submit_best refuses Task1 checkpoint: {checkpoint_path}")

        from .logging import utc_now_iso
        from .pde_tasks import task2_spec
        from .submission import default_pack_path, pack_submission
        from .task2_submission import create_task2_submission_bundle

        spec = task2_spec(self.project_root)
        run_dir = self.project_root / str(node.plan.params.get("run_dir", Path("runs") / "task2" / "submissions" / node.id))
        if "task1" in run_dir.as_posix().lower():
            return ExperimentExecution(success=False, error=f"Task2 run_dir must not reference Task1: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = run_dir / spec.prediction_name
        command = [
            sys.executable,
            str(self.project_root / "code" / "infer_task2_model.py"),
            "--checkpoint",
            checkpoint_path,
            "--input",
            str(spec.test_input_path),
            "--output",
            str(prediction_path),
            "--batch-size",
            str(int(node.plan.params.get("batch_size", 32))),
            "--device",
            str(node.plan.params.get("device", "cpu")),
        ]
        started = time.perf_counter()
        result = subprocess.run(command, cwd=self.project_root, capture_output=True, text=True, timeout=int(node.plan.params.get("timeout_seconds", 600)))
        inference_time = time.perf_counter() - started
        artifacts: dict[str, Any] = {
            "command": command,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
            "prediction_path": str(prediction_path),
        }
        if result.returncode != 0:
            return ExperimentExecution(success=False, artifacts=artifacts, error=f"Task2 inference failed with code {result.returncode}")
        try:
            inference_payload = json.loads(result.stdout)
            if isinstance(inference_payload, dict):
                artifacts["inference_payload"] = inference_payload
                if isinstance(inference_payload.get("inference_time"), int | float):
                    inference_time = float(inference_payload["inference_time"])
        except json.JSONDecodeError:
            pass
        log_path = run_dir / "task2_logs.log"
        log_path.write_text(
            json.dumps(
                {
                    "timestamp": utc_now_iso(),
                    "elapsed_seconds": inference_time,
                    "provider": "Task2SubmitBest",
                    "model": "task2_trained_checkpoint",
                    "messages": [{"role": "system", "content": "Run Task 2 trained checkpoint inference and package submission."}],
                    "response": {"checkpoint_path": checkpoint_path, "command": command, "candidate": candidate},
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            create_task2_submission_bundle(
                prediction_path=prediction_path,
                initial_path=spec.initial_condition_path,
                output_dir=run_dir,
                code_dir=self.project_root / "code",
                log_path=log_path,
                methodology_path=self.project_root / "docs" / "methodology.pdf",
                train_time=float(train_time or 0.0),
                inference_time=inference_time,
            )
            zip_path = pack_submission(run_dir, default_pack_path(run_dir))
            artifacts["zip_path"] = str(zip_path)
            artifacts["run_dir"] = str(run_dir)
        except Exception as exc:
            return ExperimentExecution(success=False, artifacts=artifacts, error=f"{type(exc).__name__}: {exc}")
        return ExperimentExecution(success=True, artifacts=artifacts)

    def _execute_task2_persistence_submission(self, node: CandidateNode) -> ExperimentExecution:
        from .task2_workflow import Task2PersistenceWorkflow

        run_root = self.project_root / str(node.plan.params.get("run_root", "runs"))
        workflow = Task2PersistenceWorkflow(project_root=self.project_root, run_root=run_root)
        result = workflow.run_test_submission(run_name=str(Path("autonomous") / node.id / "task2_submission"))
        return ExperimentExecution(
            success=result.success,
            metrics=dict(result.metrics),
            artifacts=result.to_json_dict(),
            error=result.error,
        )

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
                metric_source = self._metrics_from_payload(loaded)
                metrics = {key: float(value) for key, value in metric_source.items() if isinstance(value, int | float)}
                artifacts["metrics_payload"] = loaded
        return ExperimentExecution(
            success=result.returncode == 0,
            metrics=metrics,
            artifacts=artifacts,
            error=None if result.returncode == 0 else f"{required_label} command failed with return code {result.returncode}",
        )

    def _metrics_from_payload(self, loaded: Any) -> dict[str, Any]:
        if not isinstance(loaded, dict):
            return {}
        if isinstance(loaded.get("best_metrics"), dict):
            return loaded["best_metrics"]
        if isinstance(loaded.get("metrics"), dict):
            return loaded["metrics"]
        selected = loaded.get("selected")
        if isinstance(selected, dict) and isinstance(selected.get("metrics"), dict):
            return selected["metrics"]
        candidates = loaded.get("candidates")
        if isinstance(candidates, list):
            candidate_metrics = [
                item.get("metrics")
                for item in candidates
                if isinstance(item, dict) and isinstance(item.get("metrics"), dict)
            ]
            if candidate_metrics:
                return min(candidate_metrics, key=lambda metrics: float(metrics.get(self.metric, metrics.get("forecast_mse", float("inf")))))
        if isinstance(loaded.get("persistence"), dict):
            return loaded["persistence"]
        return loaded

    def _execute_evaluate_candidate(self, node: CandidateNode) -> ExperimentExecution:
        if (
            isinstance(node.plan.params.get("checkpoint_path"), str)
            or isinstance(node.plan.params.get("checkpoint"), str)
            or isinstance(node.plan.params.get("checkpoint_overrides"), dict)
        ):
            return self._execute_checkpoint_evaluation(node)
        prediction_path = self._resolve_project_path(str(node.plan.params.get("prediction_path", "")))
        target_path = self._resolve_project_path(str(node.plan.params.get("target_path", "data/Task1/task1_val.hdf5")))
        if not prediction_path.is_file():
            return ExperimentExecution(success=False, error=f"prediction_path not found: {prediction_path}")
        if not target_path.is_file():
            return ExperimentExecution(success=False, error=f"target_path not found: {target_path}")
        prediction = self._read_hdf5_array(prediction_path).astype(np.float32)
        target = self._read_hdf5_array(target_path).astype(np.float32)
        if prediction.shape != target.shape:
            return ExperimentExecution(success=False, error=f"shape mismatch: prediction={prediction.shape}, target={target.shape}")
        prediction[:, :10, :] = target[:, :10, :]
        metrics = compute_task1_metrics(prediction, target)
        metrics_path = node.plan.params.get("metrics_path")
        if isinstance(metrics_path, str) and metrics_path.strip():
            path = self._resolve_project_path(metrics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return ExperimentExecution(
            success=True,
            metrics={key: float(value) for key, value in metrics.items()},
            artifacts={
                "prediction_path": str(prediction_path),
                "target_path": str(target_path),
                "prediction_shape": list(prediction.shape),
            },
        )

    def _execute_checkpoint_evaluation(self, node: CandidateNode) -> ExperimentExecution:
        if self.workflow is None:
            return ExperimentExecution(success=False, error="checkpoint evaluate_candidate requires a Task1FNOWorkflow")
        checkpoint_overrides = node.plan.params.get("checkpoint_overrides")
        if not isinstance(checkpoint_overrides, dict):
            checkpoint = node.plan.params.get("checkpoint_path", node.plan.params.get("checkpoint"))
            if not isinstance(checkpoint, str) or not checkpoint.strip():
                return ExperimentExecution(success=False, error="checkpoint evaluate_candidate requires checkpoint_path")
            checkpoint_overrides = {"nu0.001": checkpoint}
        weights = node.plan.params.get("task1_weights", node.plan.params.get("weights", {"nu0.001": 1.0}))
        if not isinstance(weights, dict):
            return ExperimentExecution(success=False, error="checkpoint evaluate_candidate weights must be an object")
        previous_checkpoint_paths = None
        if hasattr(self.workflow, "checkpoint_paths"):
            previous_checkpoint_paths = dict(self.workflow.checkpoint_paths)
            self.workflow.checkpoint_paths.update({key: Path(value) for key, value in checkpoint_overrides.items()})
        try:
            run_name = str(Path("autonomous") / getattr(node, "id", "evaluate_checkpoint") / "checkpoint_eval")
            result = self.workflow.run_validation(dict(weights), run_name=run_name)
        finally:
            if previous_checkpoint_paths is not None:
                self.workflow.checkpoint_paths = previous_checkpoint_paths
        best_candidate = {
            "name": str(node.plan.params.get("name", "checkpoint-evaluation")),
            "task1_weights": dict(weights),
            "checkpoint_overrides": dict(checkpoint_overrides),
        }
        return ExperimentExecution(
            success=result.success,
            metrics=dict(result.metrics),
            artifacts={
                "best_candidate": best_candidate,
                "run_dir": str(result.run_dir),
                "prediction_path": str(result.prediction_path) if result.prediction_path else None,
                "checkpoint_overrides": dict(checkpoint_overrides),
                "weights": dict(weights),
            },
            error=result.error,
        )

    def _execute_validate_submission(self, node: CandidateNode) -> ExperimentExecution:
        raw_path = str(node.plan.params.get("path", node.plan.params.get("submission_path", "")))
        if not raw_path:
            return ExperimentExecution(success=False, error="validate_submission requires path")
        path = self._resolve_project_path(raw_path)
        try:
            report = validate_submission(path)
        except (SubmissionError, ValueError, OSError) as exc:
            return ExperimentExecution(success=False, error=str(exc), artifacts={"path": str(path)})
        return ExperimentExecution(
            success=report.valid,
            artifacts={"path": str(path), "tasks": report.tasks, "messages": report.messages},
            error=None if report.valid else "; ".join(report.messages),
        )

    def _execute_submit_best(self, node: CandidateNode) -> ExperimentExecution:
        if self.workflow is None:
            return ExperimentExecution(success=False, error="submit_best requires a Task1FNOWorkflow")
        weights = node.plan.params.get("weights")
        extra_inference_args = node.plan.params.get("task1_extra_inference_args")
        checkpoint_overrides = node.plan.params.get("checkpoint_overrides")
        if not isinstance(weights, dict) and self.journal is not None:
            best = self.journal.best(metric=self.metric, maximize=self.maximize)
            if best is not None:
                candidate = best.artifacts.get("best_candidate")
                if isinstance(candidate, dict):
                    if isinstance(candidate.get("task1_weights"), dict):
                        weights = candidate["task1_weights"]
                    elif isinstance(candidate.get("weights"), dict):
                        weights = candidate["weights"]
                    if extra_inference_args is None and isinstance(candidate.get("task1_extra_inference_args"), list):
                        extra_inference_args = candidate["task1_extra_inference_args"]
                    if checkpoint_overrides is None and isinstance(candidate.get("checkpoint_overrides"), dict):
                        checkpoint_overrides = candidate["checkpoint_overrides"]
        if not isinstance(weights, dict):
            return ExperimentExecution(success=False, error="submit_best requires weights or a prior best_candidate")
        previous_checkpoint_paths = None
        if isinstance(checkpoint_overrides, dict) and hasattr(self.workflow, "checkpoint_paths"):
            previous_checkpoint_paths = dict(self.workflow.checkpoint_paths)
            self.workflow.checkpoint_paths.update({key: Path(value) for key, value in checkpoint_overrides.items()})
        submission_kwargs = {
            "run_name": str(Path("autonomous") / node.id / "submission"),
            "train_time": float(node.plan.params.get("train_time", 0.0)),
        }
        if extra_inference_args:
            submission_kwargs["extra_inference_args"] = list(extra_inference_args)
        try:
            result: RunResult = self.workflow.run_test_submission(weights, **submission_kwargs)
        finally:
            if previous_checkpoint_paths is not None:
                self.workflow.checkpoint_paths = previous_checkpoint_paths
        artifacts = result.to_json_dict()
        if isinstance(checkpoint_overrides, dict):
            artifacts["checkpoint_overrides"] = dict(checkpoint_overrides)
        return ExperimentExecution(
            success=result.success,
            metrics=dict(result.metrics),
            artifacts=artifacts,
            error=result.error,
        )
