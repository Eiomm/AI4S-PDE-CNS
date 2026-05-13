from __future__ import annotations

import json
import re
from typing import Any

from .llm import LLMClient, logged_completion
from .logging import LLMCallLogger
from .pde_journal import CandidateNode, CandidatePlan, ExperimentJournal


ALLOWED_EXPERIMENT_ACTIONS = {
    "weight_search",
    "finetune",
    "code_patch",
    "baseline_train",
    "baseline_validate",
    "baseline_ensemble",
    "baseline_refine",
    "submit_best",
    "stop",
}


PLANNER_SYSTEM_PROMPT = """You are an AIDE-style PDE experiment planner.
Generate exactly one atomic experiment plan as JSON.
Allowed action_type values: weight_search, finetune, code_patch, baseline_train, baseline_validate, baseline_ensemble, baseline_refine, submit_best, stop.
code_patch is allowed, including large rewrites under code/, but every patch must be scoped to submitted code files and followed by validation.
For code_patch, include validation_command or submission_validation_path so the executor can verify the patch before accepting the node.
Do not use Task 1 checkpoints or data for Task 2. Do not use numerical solvers to generate extra training data.

Champion-route priorities for this repository:
1. Stabilize the autonomous loop before expensive experiments.
2. Prefer low-cost Task 1 experiments that can be validated immediately.
3. First try ensemble weight search, short fine-tune variants, and controlled Baseline Zoo prototype runs.
4. Then try U-Net, DeepONetLite, TFNO, multi-step rollout loss, PINO / physics residual, spectral loss, and conservation/long-horizon diagnostics.
5. Treat PDE-Refiner or correction-head ideas as high-priority branches once a validation cache exists.
6. Keep Task 2 isolated and train from scratch.
"""


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
    return CandidatePlan(
        intent=str(payload.get("intent", "improve")).strip() or "improve",
        hypothesis=hypothesis,
        action_type=action_type,
        params=params,
        expected_effect=str(payload.get("expected_effect", "")).strip(),
        risk=str(payload.get("risk", "")).strip(),
    )


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
        prompt_payload = {
            "context": context,
            "metric": self.metric,
            "maximize": self.maximize,
            "best_node": best.to_dict() if best else None,
            "recent_journal": self.journal.summary(metric=self.metric),
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
