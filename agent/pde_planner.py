from __future__ import annotations

import json
import re
from typing import Any

from .llm import LLMClient, logged_completion
from .logging import LLMCallLogger
from .pde_journal import CandidateNode, CandidatePlan, ExperimentJournal
from .pde_method_library import select_method_candidates


ALLOWED_EXPERIMENT_ACTIONS = {
    "inspect_data",
    "weight_search",
    "postprocess_search",
    "finetune_checkpoint",
    "finetune",
    "code_patch",
    "baseline_train",
    "baseline_validate",
    "baseline_ensemble",
    "baseline_refine",
    "task2_train_model",
    "task2_submit_best",
    "evaluate_candidate",
    "submit_best",
    "validate_submission",
    "stop",
}

TASK1_BASELINE_VALIDATE_DEFAULT_COMMAND = [
    "python",
    "-m",
    "pytest",
    "tests/test_pde_autonomous.py::test_bootstrap_postprocess_search_plan_uses_official_cache_paths",
    "-q",
]


PLANNER_SYSTEM_PROMPT = """You are an AIDE-style PDE experiment planner.
Generate exactly one atomic experiment plan as JSON.
Allowed action_type values: inspect_data, weight_search, postprocess_search, finetune_checkpoint, finetune, code_patch, baseline_train, baseline_validate, baseline_ensemble, baseline_refine, task2_train_model, task2_submit_best, evaluate_candidate, submit_best, validate_submission, stop.
Use prompt_payload.method_candidates as a method tool library: first name the selected paper-inspired method in params.selected_method/source_method when it guides the experiment, then map its implementation_knobs to concrete safe parameters.
code_patch is allowed, including large rewrites under code/, but every patch must be scoped to submitted code files and followed by validation.
For code_patch, include params.files as a list of {"path": "relative/file.py", "content": "..."} objects. If you accidentally call it patches, the executor may normalize it, but files is the canonical schema.
For code_patch, include validation_command or submission_validation_path so the executor can verify the patch before accepting the node.
For code_patch, finetune_checkpoint, task2_train_model, train_refiner-like experiments, include params.source_method and params.source_files when baseline context is available. source_files should cite relevant files under third_party/baseline, for example third_party/baseline/PDEBench/pdebench/models/fno/fno.py.
finetune_checkpoint should start from the official Task 1 FNO checkpoint unless params.base_checkpoint explicitly overrides it. It must state temporal_stride, trainable, lr, steps, and rollout_steps. To test physical or mathematical bottlenecks, it may also set gradient_loss_weight for Sobolev/local-slope stability, spectral_loss_weight for spectral loss / high-frequency dissipation control, physics_loss_weight plus physics_nu/physics_dt/physics_dx for Burgers residual loss, and horizon_loss_gamma > 1.0 to emphasize later rollout steps. To test architecture bottlenecks, it may set architecture="residual-corrected-fno" and trainable="residual-head" or "last-block-head".
baseline_validate requires params.command as a non-empty command list. baseline_train, baseline_ensemble, baseline_refine, and finetune also require params.command.
If the task is Task 1 and you only want to validate the current baseline scaffolding, baseline_validate may use params {"task": "task1"} and the Rule Guard will attach a safe pytest command.
Do not use Task 1 checkpoints or data for Task 2. Do not use numerical solvers to generate extra training data.
For Task 2, train from scratch on data/Task2 only. The test set has no Nu values, so any Nu-aware method must infer latent Nu from the first 10 frames or use Nu only as a training auxiliary target.

Champion-route priorities for this repository:
1. Stabilize the autonomous loop before expensive experiments.
2. Prefer low-cost Task 1 experiments that can be validated immediately.
3. Diagnose the current best node before proposing a run: compare forecast_mse, long_horizon_mse, segment1_rel_mse, segment2_rel_mse, and segment3_rmse, then explain whether the bottleneck is early accuracy, accumulated rollout drift, excessive smoothing, or high-frequency instability.
4. Prefer scientific fine-tune variants that can beat the current best score: stride-5 checkpoint fine-tune with multi-step rollout loss, gradient_loss_weight, spectral_loss_weight, physics_loss_weight, horizon_loss_gamma, trainable=last-block-head/all/residual-head comparisons, architecture=residual-corrected-fno, and controlled validation sample splits.
5. Then try ensemble weight search, postprocess_search with segment persistence stabilization, U-Net, DeepONetLite, TFNO, PINO / physics residual, conservation diagnostics, and PDE-Refiner correction-head ideas.
6. Keep Task 2 isolated and train from scratch.

Task 1 autonomous research policy:
- For the official Nu0.001 FNO checkpoint, temporal_stride=5 follows from the competition/PDEBench reduced_resolution_t=5 description and should be treated as the default time-scale alignment. Use temporal_stride=1 only as an explicit negative-control experiment.
- Treat trainable, steps, rollout_steps, learning rate, and stability losses as controlled experiment variables, not hard-coded answers.
- Low-cost probes such as last-block-head or short-step fine-tuning are allowed for diagnosis. If a probe improves but remains clearly weaker than the journal's current best, propose a higher-capacity follow-up such as all-parameter fine-tuning, longer training, or continuing from the best checkpoint, with explicit metric-based justification.
- Every finetune_checkpoint node must include a unique run_dir. Use names that encode the scientific choice, e.g. runs/task1-agent-finetune-stride5-rollout2-gamma105.
- Do not hard-code a known best hyperparameter tuple. The Agent must justify each move from the observed validation metrics and experiment history.
- If repeated fine-tune variants plateau, the Agent may propose a scoped code_patch that adds a reusable research capability, such as a new diagnostic, training objective, postprocess operator, model component, or evaluation helper. The patch must include validation_command or submission_validation_path and must be justified by the observed failure mode.
- If prompt_payload.research_state.capability_evolution_required is true, choose code_patch unless the only safe next step is submit_best or stop. The patch should add a reusable capability that changes the future search space, not another one-off hyperparameter setting.
"""


def _apply_rule_guard_defaults(action_type: str, params: dict[str, Any]) -> dict[str, Any]:
    completed = dict(params)
    if (
        action_type == "baseline_validate"
        and completed.get("task") == "task1"
        and "command" not in completed
    ):
        completed["command"] = list(TASK1_BASELINE_VALIDATE_DEFAULT_COMMAND)
        completed.setdefault("timeout_seconds", 120)
        completed["auto_filled_by_rule_guard"] = "task1_baseline_validate_default"
    return completed


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError("planner response does not contain a JSON object")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("planner response JSON must be an object")
    return payload


def parse_candidate_plan(response: dict[str, Any]) -> CandidatePlan:
    payload = response.get("action")
    if not isinstance(payload, dict):
        content = response.get("content", "")
        if not isinstance(content, str):
            raise ValueError("planner response must contain string content or dict action")
        payload = _extract_json_object(content)

    action_type = str(payload.get("action_type", "")).strip()
    if action_type not in ALLOWED_EXPERIMENT_ACTIONS:
        raise ValueError(f"unsupported experiment action_type: {action_type!r}")
    hypothesis = str(payload.get("hypothesis", "")).strip()
    if not hypothesis and action_type != "stop":
        raise ValueError("planner hypothesis is required")
    params = payload.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("planner params must be an object")
    params = _apply_rule_guard_defaults(action_type, params)
    return CandidatePlan(
        intent=str(payload.get("intent", "improve")).strip() or "improve",
        hypothesis=hypothesis,
        action_type=action_type,
        params=params,
        expected_effect=str(payload.get("expected_effect", "")).strip(),
        risk=str(payload.get("risk", "")).strip(),
    )


def summarize_research_state(
    journal: ExperimentJournal,
    *,
    metric: str = "mse",
    maximize: bool = False,
    plateau_window: int = 5,
    improvement_epsilon: float = 1.0e-6,
) -> dict[str, Any]:
    nodes = journal.read()
    scored: list[CandidateNode] = [
        node
        for node in nodes
        if node.status == "completed" and metric in node.metrics and node.error is None
    ]
    best_value: float | None = None
    non_improving_streak = 0
    for node in scored:
        value = float(node.metrics[metric])
        improved = (
            best_value is None
            or (value > best_value + improvement_epsilon if maximize else value < best_value - improvement_epsilon)
        )
        if improved:
            best_value = value
            non_improving_streak = 0
        else:
            non_improving_streak += 1
    recent_nodes = nodes[-max(plateau_window, 1) :]
    recent_code_patch = any(node.plan.action_type == "code_patch" for node in recent_nodes)
    recent_actions: dict[str, int] = {}
    recent_run_dirs: list[str] = []
    for node in recent_nodes:
        recent_actions[node.plan.action_type] = recent_actions.get(node.plan.action_type, 0) + 1
        run_dir = node.plan.params.get("run_dir")
        if isinstance(run_dir, str):
            recent_run_dirs.append(run_dir)
    capability_required = non_improving_streak >= plateau_window and not recent_code_patch
    return {
        "best_value": best_value,
        "non_improving_streak": non_improving_streak,
        "plateau_window": plateau_window,
        "improvement_epsilon": improvement_epsilon,
        "capability_evolution_required": capability_required,
        "recent_actions": recent_actions,
        "recent_run_dirs": recent_run_dirs,
    }


class ExperimentPlanner:
    def __init__(
        self,
        *,
        client: LLMClient,
        logger: LLMCallLogger,
        journal: ExperimentJournal,
        metric: str = "mse",
        maximize: bool = False,
    ):
        self.client = client
        self.logger = logger
        self.journal = journal
        self.metric = metric
        self.maximize = maximize

    def plan_next(self, context: dict[str, Any]) -> CandidateNode:
        best = self.journal.best(metric=self.metric, maximize=self.maximize)
        task = str(context.get("task", "task1"))
        method_candidates = select_method_candidates(
            task=task,
            metrics=best.metrics if best is not None else {},
        )
        prompt_payload = {
            "context": context,
            "metric": self.metric,
            "maximize": self.maximize,
            "best_node": best.to_dict() if best else None,
            "recent_journal": self.journal.summary(metric=self.metric),
            "research_state": summarize_research_state(self.journal, metric=self.metric, maximize=self.maximize),
            "method_candidates": method_candidates,
            "required_json_schema": {
                "intent": "draft|improve|debug|submit|stop",
                "hypothesis": "one atomic scientific reason",
                "action_type": sorted(ALLOWED_EXPERIMENT_ACTIONS),
                "params": {},
                "expected_effect": "metric or reliability change",
                "risk": "what can go wrong",
            },
        }
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ]
        response = logged_completion(self.client, self.logger, messages)
        plan = parse_candidate_plan(response)
        parent_id = response.get("parent_id")
        if not isinstance(parent_id, str):
            parent_id = best.id if best and plan.intent in {"improve", "debug"} else None
        return self.journal.append_plan(plan, parent_id=parent_id)
