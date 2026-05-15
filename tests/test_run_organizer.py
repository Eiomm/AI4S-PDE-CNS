from pathlib import Path

from agent.run_organizer import build_organization_plan, classify_run_dir, organize_runs


def test_classify_run_dir_groups_common_legacy_names():
    assert classify_run_dir("final-task1best-task2-full-minifno").category_parts == ("final",)
    assert classify_run_dir("task1-finetune-nu0.1-short-proxy-weight-search").category_parts == (
        "task1",
        "risky_or_legacy",
    )
    assert classify_run_dir("task2-hwpytorch-full-submission").category_parts == ("task2", "full_train")
    assert classify_run_dir("task2-short-submission").category_parts == ("task2", "smoke")
    assert classify_run_dir("code-patches").category_parts == ("agent", "code_patches")
    assert classify_run_dir("gpt53-task2-proposals-20260514-205219").category_parts == (
        "agent",
        "llm_proposals",
    )
    assert classify_run_dir("compliant-agent-research-chain").category_parts == ("agent", "chains")


def test_build_organization_plan_skips_canonical_roots(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for name in ["task1", "task2", "final", "archive", "task2-short-submission"]:
        (runs_dir / name).mkdir()

    plan = build_organization_plan(runs_dir)

    assert [item.source.name for item in plan.items] == ["task2-short-submission"]
    assert plan.items[0].destination == runs_dir / "task2" / "smoke" / "task2-short-submission"


def test_organize_runs_dry_run_does_not_move_directories(tmp_path):
    runs_dir = tmp_path / "runs"
    source = runs_dir / "task2-short-submission"
    source.mkdir(parents=True)
    (source / "metrics.json").write_text("{}", encoding="utf-8")

    report = organize_runs(runs_dir, apply=False, write_index=True)

    assert source.exists()
    assert not (runs_dir / "task2" / "smoke" / "task2-short-submission").exists()
    assert report.applied is False
    assert (runs_dir / "INDEX.md").exists()
    assert "task2-short-submission" in (runs_dir / "INDEX.md").read_text(encoding="utf-8")


def test_organize_runs_apply_moves_without_deleting_files(tmp_path):
    runs_dir = tmp_path / "runs"
    source = runs_dir / "task2-hwpytorch-full-submission"
    source.mkdir(parents=True)
    artifact = source / "submission.zip"
    artifact.write_text("zip", encoding="utf-8")

    report = organize_runs(runs_dir, apply=True, write_index=True)
    destination = runs_dir / "task2" / "full_train" / "task2-hwpytorch-full-submission"

    assert report.applied is True
    assert not source.exists()
    assert destination.exists()
    assert (destination / "submission.zip").read_text(encoding="utf-8") == "zip"
    assert "task2-hwpytorch-full-submission" in (runs_dir / "INDEX.md").read_text(encoding="utf-8")
