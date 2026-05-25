#!/usr/bin/env python3
"""Export DSLighting debug events into normalized LLM input/output JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_payloads(path: Path) -> dict[str, Any]:
    payloads: dict[str, Any] = {}
    for row in read_jsonl(path):
        ref = row.get("ref")
        if isinstance(ref, str):
            payloads[ref] = row.get("body")
    return payloads


def payload_for(event: dict[str, Any], payloads: dict[str, Any], label: str) -> Any:
    ref_info = (event.get("payload_refs") or {}).get(label) or {}
    ref = ref_info.get("ref")
    return payloads.get(ref) if isinstance(ref, str) else None


def assistant_text(response_body: Any) -> str | None:
    if not isinstance(response_body, dict):
        return None
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def tool_calls(response_body: Any) -> Any:
    if not isinstance(response_body, dict):
        return None
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None
    return message.get("tool_calls")


def event_call_id(event: dict[str, Any]) -> str | None:
    llm = event.get("llm")
    if isinstance(llm, dict):
        call_id = llm.get("logical_call_id")
        if isinstance(call_id, str):
            return call_id
    return None


def normalize(debug_dir: Path) -> list[dict[str, Any]]:
    events = read_jsonl(debug_dir / "events.jsonl")
    payloads = load_payloads(debug_dir / "payloads.jsonl")
    calls: dict[str, dict[str, Any]] = {}

    for event in events:
        call_id = event_call_id(event)
        if not call_id:
            continue
        call = calls.setdefault(
            call_id,
            {
                "call_id": call_id,
                "timestamp": event.get("timestamp_utc"),
                "model": (event.get("llm") or {}).get("model"),
                "provider": (event.get("llm") or {}).get("provider"),
                "input": {},
                "output": {},
                "metrics": {},
                "events": [],
            },
        )
        call["events"].append(
            {
                "timestamp": event.get("timestamp_utc"),
                "type": event.get("event_type"),
                "summary": event.get("summary"),
                "error": event.get("error"),
            }
        )
        call["timestamp"] = call.get("timestamp") or event.get("timestamp_utc")

        request_messages = payload_for(event, payloads, "request_messages")
        if request_messages is not None:
            call["input"]["messages"] = request_messages
            call["input"]["tags"] = event.get("tags") or {}

        response_body = payload_for(event, payloads, "response_body")
        if response_body is not None:
            call["output"]["response_body"] = response_body
            text = assistant_text(response_body)
            if text is not None:
                call["output"]["response"] = text
            tc = tool_calls(response_body)
            if tc is not None:
                call["output"]["tool_calls"] = tc

        provider_response = payload_for(event, payloads, "provider_response_body")
        if provider_response is not None and "response_body" not in call["output"]:
            call["output"]["provider_response_body"] = provider_response

        if event.get("error"):
            call["error"] = event.get("error")
        metrics = event.get("metrics") or {}
        if metrics:
            call["metrics"].update(metrics)

    return list(calls.values())


def latest_debug_session(root: Path) -> Path:
    if root.name.startswith("debug_session_"):
        return root
    sessions = sorted(p for p in root.glob("debug_session_*") if p.is_dir())
    if not sessions:
        raise FileNotFoundError(f"No debug_session_* directory found under {root}")
    return sessions[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("debug_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    debug_dir = latest_debug_session(args.debug_dir.resolve())
    output = args.output or (debug_dir.parent / "llm_io_normalized.jsonl")
    rows = normalize(debug_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"debug_dir": str(debug_dir), "output": str(output), "records": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
