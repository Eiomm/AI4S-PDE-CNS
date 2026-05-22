#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _assistant_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    assistant = record.get("assistant_message")
    if isinstance(assistant, dict):
        return assistant
    response = record.get("response")
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
                return message
    return None


def convert_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one proxy record to the competition JSONL contract.

    The competition requires timestamp, elapsed_seconds, and at least one of
    response/tool_calls. The official proxy has richer HTTP/request metadata;
    final submission logs should keep the assistant output and optional tool
    calls while avoiding unnecessary transport details.
    """

    if record.get("path") not in {"/v1/chat/completions", "/chat/completions"}:
        return None
    timestamp = record.get("timestamp")
    elapsed_seconds = record.get("elapsed_seconds")
    if timestamp is None or elapsed_seconds is None:
        return None
    assistant = _assistant_payload(record)
    if not assistant:
        return None

    out: dict[str, Any] = {
        "timestamp": timestamp,
        "elapsed_seconds": float(elapsed_seconds),
    }
    content = assistant.get("content")
    tool_calls = assistant.get("tool_calls")
    if content not in (None, ""):
        out["response"] = content
    if tool_calls:
        out["tool_calls"] = tool_calls
    if "response" not in out and "tool_calls" not in out:
        return None
    return out


def parse_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def convert_files(
    input_paths: list[Path],
    output_path: Path,
    *,
    start_timestamp: str | None = None,
    end_timestamp: str | None = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_dt = parse_timestamp(start_timestamp)
    end_dt = parse_timestamp(end_timestamp)
    read_count = 0
    written_count = 0
    skipped_count = 0
    with output_path.open("w", encoding="utf-8") as target:
        for input_path in input_paths:
            with input_path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    text = line.strip()
                    if not text:
                        continue
                    read_count += 1
                    try:
                        record = json.loads(text)
                    except json.JSONDecodeError:
                        skipped_count += 1
                        continue
                    record_dt = parse_timestamp(record.get("timestamp"))
                    if start_dt is not None and (record_dt is None or record_dt < start_dt):
                        skipped_count += 1
                        continue
                    if end_dt is not None and (record_dt is None or record_dt > end_dt):
                        skipped_count += 1
                        continue
                    converted = convert_record(record)
                    if converted is None:
                        skipped_count += 1
                        continue
                    target.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    written_count += 1
    if written_count == 0:
        joined_inputs = ", ".join(str(path) for path in input_paths)
        raise RuntimeError(f"no LLM assistant records were written from {joined_inputs}")
    return {
        "inputs": [str(path) for path in input_paths],
        "output": str(output_path),
        "read_records": read_count,
        "written_records": written_count,
        "skipped_records": skipped_count,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OpenAI proxy JSONL to competition task1_logs.log JSONL.")
    parser.add_argument("--input", action="append", required=True, help="logs/openai_proxy_*.jsonl；可重复传入多个文件")
    parser.add_argument("--output", required=True, help="output task1_logs.log path")
    parser.add_argument("--start-timestamp", default=None, help="可选 ISO 8601 起始时间，过滤本轮之前的 proxy 记录")
    parser.add_argument("--end-timestamp", default=None, help="可选 ISO 8601 结束时间，过滤本轮之后的 proxy 记录")
    args = parser.parse_args()
    report = convert_files(
        [Path(item) for item in args.input],
        Path(args.output),
        start_timestamp=args.start_timestamp,
        end_timestamp=args.end_timestamp,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
