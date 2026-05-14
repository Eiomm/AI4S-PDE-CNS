from __future__ import annotations

from pathlib import Path

from .pde_journal import ExperimentJournal


def _format_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.10g}"


def _artifact_value(node_artifacts: dict, key: str) -> str:
    value = node_artifacts.get(key)
    return str(value).replace("|", "\\|") if value else ""


def render_journal_report(
    journal: ExperimentJournal,
    *,
    title: str = "Autonomous PDE Experiment Report",
    metric: str = "mse",
    maximize: bool = False,
) -> str:
    nodes = journal.read()
    best = journal.best(metric=metric, maximize=maximize)
    lines = [
        f"# {title}",
        "",
        f"- Nodes: {len(nodes)}",
        f"- Metric: `{metric}` ({'maximize' if maximize else 'minimize'})",
    ]
    if best is not None:
        lines.append(f"- Best node: `{best.id}` with `{metric}={_format_metric(best.metrics.get(metric))}`")
    else:
        lines.append("- Best node: none yet")
    submission_nodes = [
        node
        for node in nodes
        if node.plan.action_type == "submit_best" and node.artifacts.get("zip_path")
    ]
    if submission_nodes:
        latest_submission = submission_nodes[-1]
        lines.append(f"- Submission zip: `{latest_submission.artifacts['zip_path']}`")

    lines.extend(
        [
            "",
            "## Journal",
            "",
            "| Step | Node | Parent | Status | Action | Metric | Run Dir | Zip Path | Hypothesis | Decision |",
            "|---:|---|---|---|---|---:|---|---|---|---|",
        ]
    )
    for node in nodes:
        decision = node.review.get("next_intent", "")
        hypothesis = node.plan.hypothesis.replace("|", "\\|")
        lines.append(
            "| "
            f"{node.step} | "
            f"`{node.id[:8]}` | "
            f"{'`' + node.parent_id[:8] + '`' if node.parent_id else ''} | "
            f"{node.status} | "
            f"{node.plan.action_type} | "
            f"{_format_metric(node.metrics.get(metric))} | "
            f"{_artifact_value(node.artifacts, 'run_dir')} | "
            f"{_artifact_value(node.artifacts, 'zip_path')} | "
            f"{hypothesis} | "
            f"{decision} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_journal_report(
    journal: ExperimentJournal,
    path: str | Path,
    *,
    title: str = "Autonomous PDE Experiment Report",
    metric: str = "mse",
    maximize: bool = False,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_journal_report(journal, title=title, metric=metric, maximize=maximize),
        encoding="utf-8",
    )
    return output
