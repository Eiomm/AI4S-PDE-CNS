import json
from pathlib import Path

from agent.experiment_results import (
    cleanup_experiment_artifacts,
    collect_experiment_records,
    write_experiment_results,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_collect_experiment_records_sorts_by_proxy_and_preserves_config(tmp_path):
    runs_root = tmp_path / "runs"
    good = runs_root / "study-a" / "candidate-good"
    bad = runs_root / "study-a" / "candidate-bad"
    _write_json(good / "metrics.json", {"competition_score_proxy": 58.9, "mse": 0.0015})
    _write_json(
        good / "baseline_manifest.json",
        {
            "baseline": {"name": "temporal_tail_blend_deeponet_lite", "family": "ensemble"},
            "config": {"cut": 120, "tail_weight": 0.6},
        },
    )
    _write_json(good / "run_result.json", {"success": True, "command": ["baseline_ensemble"], "zip_path": None})
    _write_json(bad / "metrics.json", {"competition_score_proxy": 12.0, "mse": 0.02})

    records = collect_experiment_records(runs_root)

    assert [record.run_dir.name for record in records] == ["candidate-good", "candidate-bad"]
    assert records[0].model_name == "temporal_tail_blend_deeponet_lite"
    assert records[0].config["tail_weight"] == 0.6
    assert records[0].success is True


def test_write_experiment_results_creates_markdown_csv_and_json(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "study-b" / "candidate"
    _write_json(run_dir / "metrics.json", {"competition_score_proxy": 42.0, "mse": 0.01})

    output = tmp_path / "docs" / "results.md"
    csv_path = runs_root / "summary.csv"
    json_path = runs_root / "summary.json"
    records = collect_experiment_records(runs_root)

    write_experiment_results(records, output_path=output, csv_path=csv_path, json_path=json_path, top_n=10)

    markdown = output.read_text(encoding="utf-8")
    assert "| Rank | Run | Model | Proxy | MSE |" in markdown
    assert "candidate" in markdown
    assert csv_path.read_text(encoding="utf-8").startswith("rank,run_dir")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["records"][0]["metrics"]["competition_score_proxy"] == 42.0


def test_cleanup_experiment_artifacts_deletes_only_requested_paths(tmp_path):
    runs_root = tmp_path / "runs"
    keep_zip = runs_root / "best" / "pred.zip"
    old_zip = runs_root / "old" / "pred.zip"
    partial_file = runs_root / "partial-submit" / "task1_pred.hdf5"
    keep_zip.parent.mkdir(parents=True)
    old_zip.parent.mkdir(parents=True)
    partial_file.parent.mkdir(parents=True)
    keep_zip.write_bytes(b"keep")
    old_zip.write_bytes(b"delete")
    partial_file.write_bytes(b"partial")

    manifest = cleanup_experiment_artifacts(
        runs_root,
        cleanup_zips=True,
        keep_zips=[keep_zip],
        delete_partials=["partial-submit"],
    )

    assert keep_zip.exists()
    assert not old_zip.exists()
    assert not partial_file.parent.exists()
    deleted = {Path(item["path"]).name for item in manifest["deleted"]}
    assert "pred.zip" in deleted
    assert "partial-submit" in deleted
