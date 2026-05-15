from agent.run_layout import classified_study_dir, safe_study_name


def test_safe_study_name_removes_path_separators_and_spaces():
    assert safe_study_name(" Task 2 / smoke run ") == "Task-2-smoke-run"


def test_classified_study_dir_groups_by_task_category_and_date(tmp_path):
    path = classified_study_dir(
        project_root=tmp_path,
        task="task2",
        category="autonomous",
        study_name="smoke",
        date="20260515",
    )

    assert path == tmp_path / "runs" / "task2" / "autonomous" / "20260515" / "smoke"
