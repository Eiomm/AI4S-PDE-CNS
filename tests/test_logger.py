import json
from datetime import datetime

from agent.logging import LLMCallLogger, log_span_seconds


def test_logger_writes_jsonl_with_required_fields(tmp_path):
    log_path = tmp_path / "task1_logs.log"
    logger = LLMCallLogger(log_path)

    logger.write_call(
        provider="mock",
        model="mock-model",
        messages=[{"role": "user", "content": "hello"}],
        response={"content": "world"},
        elapsed_seconds=1.25,
    )

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["provider"] == "mock"
    assert record["model"] == "mock-model"
    assert record["elapsed_seconds"] == 1.25
    assert record["messages"][0]["content"] == "hello"
    assert record["response"]["content"] == "world"
    datetime.fromisoformat(record["timestamp"])


def test_log_span_seconds_reads_first_and_last_timestamp(tmp_path):
    log_path = tmp_path / "task2_logs.log"
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"timestamp": "2026-05-09T10:00:00+08:00", "elapsed_seconds": 1}),
                json.dumps({"timestamp": "2026-05-09T11:30:30+08:00", "elapsed_seconds": 2}),
            ]
        ),
        encoding="utf-8",
    )

    assert log_span_seconds(log_path) == 5430
