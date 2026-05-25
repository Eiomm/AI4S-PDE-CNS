from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .hdf5_io import write_prediction, read_named_or_single
from .paths import submissions_root, task_root
from .validate import validate_task1_prediction


def _copy_code_snapshot(source_code_dir: Path, target_code_dir: Path) -> None:
    """复制提交代码目录。

    提交规则要求 `code/` 必须是 Agent 独立生成或修改的代码。这里默认只复制
    调用方显式提供的 Agent 代码目录，不再默认把当前 harness 的 `src/`
    塞进最终 submission。
    """

    if target_code_dir.exists():
        shutil.rmtree(target_code_dir)
    shutil.copytree(source_code_dir, target_code_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _has_real_code(path: Path) -> bool:
    """判断 Agent code 目录里是否真的有代码文件。

    `.gitkeep` 这类占位文件不能算作 Agent 生成代码，否则会绕过提交规则。
    """

    if not path.is_dir():
        return False
    for item in path.rglob("*"):
        if item.is_file() and not item.name.startswith("."):
            return True
    return False


def _latest_agent_code_dir() -> Path:
    summaries = sorted(
        list((task_root() / "agent_workspace" / "experiments").glob("*/logs/turn_*/summary.json"))
        + list((task_root() / "agent_workspace" / "logs").glob("agent_*/summary.json")),
        key=lambda item: item.stat().st_mtime,
    )
    for summary_path in reversed(summaries):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        raw_code_root = payload.get("generated_code_root")
        if not raw_code_root:
            continue
        code_root = Path(str(raw_code_root))
        if not code_root.is_absolute():
            code_root = task_root() / code_root
        if _has_real_code(code_root):
            return code_root
    raise FileNotFoundError(
        "没有找到可用的本轮 Agent code 目录。请先运行 task1_agent_runner.py，"
        "或用 --code-dir 显式指定某次 Agent run 的 code 目录。"
    )


def make_submission(
    run_dir: str | Path,
    *,
    submission_name: str,
    submission_id: str = "AI4S-PDE-CNS",
    code_dir: str | Path | None = None,
    llm_log_path: str | Path | None = None,
) -> dict[str, Any]:
    """根据一次 run 生成 Task1 提交目录。

    设计时参考了旧仓库 `data/Task1/sample_submission`：
    - 根目录包含 `task1_pred.hdf5`、`task1_time.csv`、`task1_logs.log`；
    - `task1_pred.hdf5` 里使用官方 dataset key `tensor`；
    - `submission.json` 至少包含 `submission_id`、`problem_id`、`code_path`；
    - sample 里还出现了 `methodology` 和 `submission` 字段，因此这里保留
      兼容字段，方便后续如果官方校验器需要它们时无需再改结构。

    默认读取最新 `summary.generated_code_root` 指向的单次 Agent code 快照。
    这个目录应该由 GPT-5.5 Agent 通过官方 proxy 生成或修改。
    """

    source = Path(run_dir)
    if not source.is_absolute():
        source = task_root() / source
    output = submissions_root() / submission_name
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    prediction_path = source / "task1_pred.hdf5"
    input_path = task_root() / "data" / "task1_test.hdf5"
    validate_task1_prediction(prediction_path, input_path)
    prediction = read_named_or_single(prediction_path, "tensor")
    write_prediction(output / "task1_pred.hdf5", prediction, dataset_key="tensor")
    shutil.copy2(source / "task1_time.csv", output / "task1_time.csv")
    if llm_log_path is None:
        shutil.copy2(source / "task1_logs.log", output / "task1_logs.log")
        log_policy = {
            "source": str(source / "task1_logs.log"),
            "llm_proxy_log_used": False,
            "warning": "该日志是 harness 行为日志；正式提交建议使用 --llm-log 指向 proxy 转换后的 JSONL。",
        }
    else:
        resolved_log = Path(llm_log_path)
        if not resolved_log.is_absolute():
            resolved_log = task_root() / resolved_log
        shutil.copy2(resolved_log, output / "task1_logs.log")
        log_policy = {
            "source": str(resolved_log),
            "llm_proxy_log_used": True,
        }

    source_code_dir = Path(code_dir) if code_dir is not None else _latest_agent_code_dir()
    if not source_code_dir.is_absolute():
        source_code_dir = task_root() / source_code_dir
    code_target = output / "code"
    if _has_real_code(source_code_dir):
        _copy_code_snapshot(source_code_dir, code_target)
        code_policy = {
            "source": str(source_code_dir),
            "agent_generated_required": True,
        }
    else:
        raise FileNotFoundError(
            f"缺少 Agent 生成代码目录：{source_code_dir}。"
            "最终提交不能默认复制当前 src/。请先让 Agent 通过官方 proxy 生成或修改 code/。"
        )
    (output / "submission.json").write_text(
        json.dumps(
            {
                "submission_id": submission_id,
                "problem_id": "PDE_Burgers",
                "code_path": "code",
                "methodology": "./methodology.pdf",
                "submission": "./submission.zip",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    zip_base = submissions_root() / submission_name
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=output))
    report = {
        "submission_dir": str(output),
        "submission_zip": str(zip_path),
        "prediction": str(output / "task1_pred.hdf5"),
        "time_csv": str(output / "task1_time.csv"),
        "logs": str(output / "task1_logs.log"),
        "code_dir": str(code_target),
        "code_policy": code_policy,
        "log_policy": log_policy,
        "submission_json": str(output / "submission.json"),
    }
    (output / "bundle_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
