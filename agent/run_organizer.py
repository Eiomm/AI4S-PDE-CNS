from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


CANONICAL_ROOTS = {"agent", "archive", "final", "task1", "task2"}


@dataclass(frozen=True)
class RunClassification:
    category_parts: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class MovePlanItem:
    source: Path
    destination: Path
    classification: RunClassification
    size_bytes: int
    last_write_time: datetime


@dataclass(frozen=True)
class OrganizationPlan:
    runs_dir: Path
    items: tuple[MovePlanItem, ...]
    skipped: tuple[Path, ...]


@dataclass(frozen=True)
class OrganizationReport:
    plan: OrganizationPlan
    applied: bool
    index_path: Path | None


def classify_run_dir(name: str) -> RunClassification:
    lowered = name.lower()

    if lowered.startswith("final-") or lowered.startswith("e2e-final-"):
        return RunClassification(("final",), "final submission candidate")

    if lowered == "code-patches" or "code-patch" in lowered:
        return RunClassification(("agent", "code_patches"), "agent generated code patches")

    if any(token in lowered for token in ("gpt", "gemini", "claude", "api-check", "proposal")):
        return RunClassification(("agent", "llm_proposals"), "LLM/API proposal or connectivity run")

    if "agent" in lowered or "chain" in lowered or "autonomous" in lowered:
        return RunClassification(("agent", "chains"), "agent workflow chain")

    if lowered.startswith("task1"):
        if any(token in lowered for token in ("nu0.01", "nu0.1", "finetune", "refiner", "weight-search")):
            return RunClassification(("task1", "risky_or_legacy"), "Task1 legacy run with checkpoint/Nu compliance risk")
        return RunClassification(("task1", "experiments"), "Task1 experiment")

    if lowered.startswith("task2"):
        if any(token in lowered for token in ("smoke", "short")):
            return RunClassification(("task2", "smoke"), "Task2 smoke or short run")
        if any(token in lowered for token in ("full", "train", "minifno", "unet", "submission")):
            return RunClassification(("task2", "full_train"), "Task2 full training/submission run")
        return RunClassification(("task2", "baselines"), "Task2 baseline or uncategorized run")

    return RunClassification((), "unclassified")


def build_organization_plan(runs_dir: str | Path, *, include_unknown: bool = False) -> OrganizationPlan:
    root = Path(runs_dir).resolve()
    items: list[MovePlanItem] = []
    skipped: list[Path] = []

    for child in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if not child.is_dir():
            continue
        if child.name in CANONICAL_ROOTS:
            skipped.append(child)
            continue

        classification = classify_run_dir(child.name)
        if not classification.category_parts:
            if not include_unknown:
                skipped.append(child)
                continue
            archive_day = datetime.fromtimestamp(child.stat().st_mtime).strftime("%Y-%m-%d")
            classification = RunClassification(("archive", archive_day), "unknown run archived by date")

        destination = root.joinpath(*classification.category_parts, child.name)
        if destination == child:
            skipped.append(child)
            continue

        stat = child.stat()
        items.append(
            MovePlanItem(
                source=child,
                destination=destination,
                classification=classification,
                size_bytes=_directory_size(child),
                last_write_time=datetime.fromtimestamp(stat.st_mtime),
            )
        )

    return OrganizationPlan(runs_dir=root, items=tuple(items), skipped=tuple(skipped))


def organize_runs(
    runs_dir: str | Path,
    *,
    apply: bool = False,
    write_index: bool = True,
    include_unknown: bool = False,
) -> OrganizationReport:
    plan = build_organization_plan(runs_dir, include_unknown=include_unknown)

    if apply:
        for item in plan.items:
            destination = _available_destination(item.destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item.source), str(destination))

    index_path = None
    if write_index:
        index_path = plan.runs_dir / "INDEX.md"
        index_path.write_text(render_index(plan, applied=apply), encoding="utf-8")

    return OrganizationReport(plan=plan, applied=apply, index_path=index_path)


def render_index(plan: OrganizationPlan, *, applied: bool) -> str:
    status = "applied" if applied else "dry-run"
    lines = [
        "# Runs Index",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Mode: {status}",
        "",
        "## Move Plan",
        "",
        "| Run | Destination | Size MB | Last Write | Reason |",
        "| --- | --- | ---: | --- | --- |",
    ]

    if plan.items:
        for item in sorted(plan.items, key=lambda entry: entry.last_write_time, reverse=True):
            lines.append(
                "| "
                f"{item.source.name} | "
                f"{_relative_to(item.destination, plan.runs_dir)} | "
                f"{item.size_bytes / (1024 * 1024):.2f} | "
                f"{item.last_write_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"{item.classification.reason} |"
            )
    else:
        lines.append("| No moves planned |  |  |  |  |")

    lines.extend(["", "## Skipped", ""])
    if plan.skipped:
        for path in sorted(plan.skipped, key=lambda entry: entry.name.lower()):
            lines.append(f"- `{_relative_to(path, plan.runs_dir)}`")
    else:
        lines.append("- None")

    lines.append("")
    return "\n".join(lines)


def _directory_size(path: Path) -> int:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def _available_destination(destination: Path) -> Path:
    if not destination.exists():
        return destination

    parent = destination.parent
    stem = destination.name
    index = 2
    while True:
        candidate = parent / f"{stem}__{index}"
        if not candidate.exists():
            return candidate
        index += 1


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Organize legacy runs into a normalized runs/ layout.")
    parser.add_argument("--runs-dir", default="runs", help="Path to the runs directory.")
    parser.add_argument("--apply", action="store_true", help="Actually move directories. Default is dry-run.")
    parser.add_argument("--include-unknown", action="store_true", help="Archive unclassified run directories by date.")
    parser.add_argument("--no-index", action="store_true", help="Do not write runs/INDEX.md.")
    args = parser.parse_args(argv)

    report = organize_runs(
        args.runs_dir,
        apply=args.apply,
        write_index=not args.no_index,
        include_unknown=args.include_unknown,
    )
    print(render_index(report.plan, applied=report.applied))
    if report.index_path:
        print(f"Index written to: {report.index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
