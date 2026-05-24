from __future__ import annotations

from pathlib import Path

from .node_schema import Node


def summarize_nodes(nodes: list[Node], limit: int = 8) -> str:
    rows = []
    for node in sorted(nodes, key=lambda item: item.updated_at, reverse=True)[:limit]:
        reward = None if node.metrics is None else node.metrics.reward
        rows.append(
            f"Attempt {node.node_id}: operator={node.operator}, status={node.status}, "
            f"reward={reward}, hypothesis={node.hypothesis}, failure={node.failure_reason}"
        )
    return "\n".join(rows)


def build_prompt_context(
    operator: str,
    parent_node: Node | None,
    nodes: list[Node],
    code_dir: Path | str,
    prompt_template_path: Path | str | None = None,
) -> list[dict[str, str]]:
    """Build OpenAI-compatible messages list for LLM code generation."""
    code_dir = Path(code_dir)
    train_py = code_dir / "train.py"
    current_code = train_py.read_text(encoding="utf-8") if train_py.exists() else "# No existing code."

    if prompt_template_path and prompt_template_path.exists():
        system_prompt = prompt_template_path.read_text(encoding="utf-8")
    else:
        system_prompt = (
            "You are an expert deep-learning researcher tasked with improving a PDE surrogate model.\n"
            "Constraints:\n"
            "- Use only data from data/Task2.\n"
            "- Do not read test nu; task2_test.h5 does not provide it.\n"
            "- Do not use numerical PDE solvers to generate future test trajectories.\n"
            "- Do not load public pretrained weights or external checkpoints.\n"
            "- Produce task2_pred.hdf5 with dataset tensor shaped (1000, 200, 256), float32.\n"
            "- Copy task2_test.h5/tensor into tensor[:, :10, :] within 1e-3.\n"
            "- Support: python train.py --data-dir ../../../data/Task2 --out-dir ../artifacts --epochs 1 --cheap-probe\n"
            "- Log scientific reasoning as JSONL with timestamp, elapsed_seconds, and response or tool_calls.\n"
            "- End by printing: Final Validation Score: <overall_mse>\n"
        )

    history = summarize_nodes(nodes)

    # Build parent metrics summary for targeted improvement
    parent_metrics_str = "No parent metrics available."
    if parent_node and parent_node.metrics:
        m = parent_node.metrics
        parent_metrics_str = (
            f"overall_mse={m.overall_mse}, short_mse={m.short_mse}, "
            f"worst_nu_mse={m.worst_nu_mse}, long_stat_error={m.long_stat_error}, "
            f"reward={m.reward}, runtime_sec={m.runtime_sec}"
        )

    parent_info = ""
    if parent_node:
        parent_info = (
            f"\nParent node: {parent_node.node_id}\n"
            f"Hypothesis: {parent_node.hypothesis}\n"
            f"Status: {parent_node.status}\n"
            f"Parent Metrics: {parent_metrics_str}\n"
        )

    user_content = (
        f"Operator: {operator}\n"
        f"{parent_info}\n"
        f"Recent attempts:\n{history}\n\n"
        f"Current train.py:\n```python\n{current_code}\n```\n\n"
        f"YOUR TASK: Propose one research hypothesis, generate a COMPLETE NEW train.py, "
        f"and make the code responsible for its own lightweight AutoML/search choices.\n"
        f"Use the parent metrics and recent attempts as feedback. Pick a meaningfully different path "
        f"from prior attempts rather than only changing scalar hyperparameters.\n\n"
        f"Your response must start with:\n"
        f"### Hypothesis name\n"
        f"<short human-readable node name>\n\n"
        f"Then explain why this hypothesis is worth trying, what feedback it uses, and what metric movement "
        f"would support or falsify it.\n\n"
        f"The generated train.py should expose an internal AUTO_ML_SEARCH_SPACE and, when practical, "
        f"a small --automl-trials control so the candidate can tune architecture/training parameters itself. "
        f"The scheduler will only run the experiment and record metrics; it should not hardcode detailed trial params.\n\n"
        f"Output the complete new train.py inside a single ```python ... ``` code block."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
