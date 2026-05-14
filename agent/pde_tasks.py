from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_TASK1_FNO_WEIGHTS: dict[str, float] = {
    "nu0.001": 0.12,
    "unet_pf20_nu0.001": 0.88,
}


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    test_input_path: Path
    validation_target_path: Path | None
    initial_condition_path: Path
    output_shape: tuple[int | None, int, int]
    prediction_name: str
    time_budget_seconds: int

    @property
    def output_steps(self) -> int:
        return self.output_shape[1]

    @property
    def spatial_size(self) -> int:
        return self.output_shape[2]


def task1_spec(project_root: str | Path = ".") -> TaskSpec:
    root = Path(project_root)
    data_dir = root / "data" / "Task1"
    test_input = data_dir / "task1_test.hdf5"
    return TaskSpec(
        task_id="task1",
        test_input_path=test_input,
        validation_target_path=data_dir / "task1_val.hdf5",
        initial_condition_path=test_input,
        output_shape=(None, 200, 256),
        prediction_name="task1_pred.hdf5",
        time_budget_seconds=6 * 60 * 60,
    )


def task2_spec(project_root: str | Path = ".") -> TaskSpec:
    root = Path(project_root)
    data_dir = root / "data" / "Task2"
    test_input = data_dir / "task2_test.h5"
    return TaskSpec(
        task_id="task2",
        test_input_path=test_input,
        validation_target_path=data_dir / "task2_val.h5",
        initial_condition_path=test_input,
        output_shape=(None, 200, 256),
        prediction_name="task2_pred.hdf5",
        time_budget_seconds=6 * 60 * 60,
    )
