"""
openai-log: A local proxy server that forwards requests to an OpenAI-compatible API
and logs all requests along with their complete LLM responses.

Streaming responses are collected in full before logging — only one log entry per request.
Each JSONL entry includes the request payload under `request` and the assistant
output under `response`, `tool_calls`, and/or `think`.

Usage:
    python proxy.py [--port PORT] [--target TARGET_URL] [--log-dir LOG_DIR]

Example:
    python proxy.py --port 8080 --target https://api.openai.com --log-dir ./logs
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI-compatible logging proxy")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PROXY_PORT", 8080)),
                        help="Local port to listen on (default: 8080, env: PROXY_PORT)")
    parser.add_argument("--target", type=str,
                        default=os.environ.get("PROXY_TARGET", "https://api.openai.com"),
                        help="Target base URL to forward requests to (default: https://api.openai.com, env: PROXY_TARGET)")
    parser.add_argument("--log-dir", type=str,
                        default=os.environ.get("PROXY_LOG_DIR", "./logs"),
                        help="Directory to write log files (default: ./logs, env: PROXY_LOG_DIR)")
    parser.add_argument("--log-level", type=str,
                        default=os.environ.get("PROXY_LOG_LEVEL", "INFO"),
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Console log level (default: INFO)")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str, level: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("openai-log")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — one file per day
    log_file = Path(log_dir) / f"proxy-{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

# ---------------------------------------------------------------------------
# LLM response log writer
# ---------------------------------------------------------------------------

def write_llm_log(log_dir: str, entry: dict) -> None:
    """Append a single JSON log entry to the LLM log file."""
    log_file = Path(log_dir) / f"llm-{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _extract_request_payload(body_bytes: bytes, url: str, method: str) -> dict:
    """Return the LLM input payload for JSONL auditing."""
    request_payload: dict = {"method": method, "path": url}
    if not body_bytes:
        return request_payload

    try:
        body = json.loads(body_bytes)
    except (json.JSONDecodeError, ValueError):
        request_payload["body"] = body_bytes.decode("utf-8", errors="replace")
        return request_payload

    if not isinstance(body, dict):
        request_payload["body"] = body
        return request_payload

    for key in ("model", "stream", "temperature", "max_tokens", "response_format", "tools", "tool_choice"):
        if key in body:
            request_payload[key] = body[key]
    if "messages" in body:
        request_payload["messages"] = body["messages"]
    elif "input" in body:
        request_payload["input"] = body["input"]
    else:
        request_payload["body"] = body
    return request_payload

# ---------------------------------------------------------------------------
# SSE / streaming helpers
# ---------------------------------------------------------------------------

def parse_sse_chunks(raw_bytes: bytes) -> dict:
    """
    Parse Server-Sent Events bytes and reconstruct the full assistant message.
    Handles text content, reasoning_content (think), and tool_calls deltas.
    Returns a dict with keys: 'think', 'response', 'tool_calls' (all Optional[str]).
    """
    lines = raw_bytes.decode("utf-8", errors="replace").splitlines()
    text_parts: list[str] = []
    think_parts: list[str] = []
    # tool_calls accumulator: index -> {name, arguments}
    tool_call_acc: dict[int, dict] = {}

    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})

                # Reasoning / think content (DeepSeek, Qwen, etc.)
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    think_parts.append(reasoning)

                # Text content
                content = delta.get("content")
                if content:
                    text_parts.append(content)

                # Tool calls
                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in tool_call_acc:
                        tool_call_acc[idx] = {"name": "", "arguments": ""}
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        tool_call_acc[idx]["name"] += fn["name"]
                    if fn.get("arguments"):
                        tool_call_acc[idx]["arguments"] += fn["arguments"]
        except json.JSONDecodeError:
            pass

    result: dict = {}

    if think_parts:
        result["think"] = "".join(think_parts)

    if text_parts:
        # Strip embedded <think>...</think> tags if present (some models inline them)
        full_text = "".join(text_parts)
        think_tag = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL)
        if think_tag and "think" not in result:
            result["think"] = think_tag.group(1).strip()
        clean_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
        if clean_text:
            result["response"] = clean_text

    if tool_call_acc:
        parts = []
        for idx in sorted(tool_call_acc):
            tc = tool_call_acc[idx]
            parts.append(f"{tc['name']}({tc['arguments']})")
        result["tool_calls"] = "\n".join(parts)

    return result

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(target: str, log_dir: str, logger: logging.Logger) -> FastAPI:
    app = FastAPI(title="openai-log proxy")

    # Shared async HTTP client — reused across requests
    # We disable SSL verification warnings but keep verification on by default.
    # Set verify=False if your target uses a self-signed cert.
    http_client = httpx.AsyncClient(
        base_url=target,
        timeout=httpx.Timeout(120.0),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )

    @app.on_event("startup")
    async def startup():
        logger.info(f"Proxy started — forwarding to {target}")

    @app.on_event("shutdown")
    async def shutdown():
        await http_client.aclose()
        logger.info("Proxy stopped")

    # ------------------------------------------------------------------
    # Catch-all route — forward every method / path
    # ------------------------------------------------------------------
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def proxy(path: str, request: Request) -> Response:
        start_ts = time.time()
        request_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")

        # ---- Build forwarded request ----
        # Strip hop-by-hop headers
        skip_headers = {
            "host", "content-length", "transfer-encoding",
            "connection", "keep-alive", "upgrade", "proxy-connection",
        }
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in skip_headers
        }

        body_bytes = await request.body()

        # Detect streaming without storing full request body
        try:
            request_body = json.loads(body_bytes) if body_bytes else None
        except (json.JSONDecodeError, ValueError):
            request_body = None

        is_stream = isinstance(request_body, dict) and request_body.get("stream", False)

        url = f"/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"

        logger.info(f"[{request_id}] {request.method} {url} stream={is_stream}")

        # ---- Forward ----
        try:
            if is_stream:
                return await _handle_stream(
                    http_client, request, headers,
                    body_bytes, url, request_id, start_ts, log_dir, logger
                )
            else:
                return await _handle_normal(
                    http_client, request, headers,
                    body_bytes, url, request_id, start_ts, log_dir, logger
                )
        except httpx.RequestError as exc:
            logger.error(f"[{request_id}] Upstream request failed: {exc}")
            return Response(
                content=json.dumps({"error": {"message": str(exc), "type": "proxy_error"}}),
                status_code=502,
                media_type="application/json",
            )

    return app


# ---------------------------------------------------------------------------
# Normal (non-streaming) request handler
# ---------------------------------------------------------------------------

async def _handle_normal(
    client: httpx.AsyncClient,
    request: Request,
    headers: dict,
    body_bytes: bytes,
    url: str,
    request_id: str,
    start_ts: float,
    log_dir: str,
    logger: logging.Logger,
) -> Response:
    resp = await client.request(
        method=request.method,
        url=url,
        headers=headers,
        content=body_bytes,
    )

    elapsed = time.time() - start_ts
    resp_body_bytes = resp.content

    # Parse response
    try:
        response_body = json.loads(resp_body_bytes)
    except (json.JSONDecodeError, ValueError):
        response_body = resp_body_bytes.decode("utf-8", errors="replace")

    # Extract assistant message for clean logging
    extracted = _extract_assistant_message(response_body)

    if extracted:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "request": _extract_request_payload(body_bytes, url, request.method),
        }
        log_entry.update(extracted)
        write_llm_log(log_dir, log_entry)
    logger.info(f"[{request_id}] {resp.status_code} elapsed={elapsed:.3f}s")

    # Forward response back to caller
    # httpx may transparently decode compressed upstream bodies. Forwarding the
    # original content-length then makes h11 reject the response because the
    # decoded byte length no longer matches the upstream header.
    skip_resp_headers = {"content-encoding", "transfer-encoding", "connection", "content-length"}
    resp_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in skip_resp_headers
    }
    return Response(
        content=resp_body_bytes,
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=resp.headers.get("content-type"),
    )


# ---------------------------------------------------------------------------
# Streaming request handler
# ---------------------------------------------------------------------------

async def _handle_stream(
    client: httpx.AsyncClient,
    request: Request,
    headers: dict,
    body_bytes: bytes,
    url: str,
    request_id: str,
    start_ts: float,
    log_dir: str,
    logger: logging.Logger,
) -> StreamingResponse:
    """
    Stream the response to the caller while simultaneously collecting all chunks.
    After the stream ends, write a single complete log entry.
    """
    collected_chunks: list[bytes] = []
    upstream_status: list[int] = [200]
    upstream_headers: list[dict] = [{}]

    async def generate():
        nonlocal collected_chunks
        try:
            async with client.stream(
                method=request.method,
                url=url,
                headers=headers,
                content=body_bytes,
            ) as resp:
                upstream_status[0] = resp.status_code
                upstream_headers[0] = dict(resp.headers)

                async for chunk in resp.aiter_bytes():
                    collected_chunks.append(chunk)
                    yield chunk

        except httpx.RequestError as exc:
            err_payload = json.dumps(
                {"error": {"message": str(exc), "type": "proxy_error"}}
            ).encode()
            collected_chunks.append(err_payload)
            yield err_payload
        finally:
            # Log once the stream is fully consumed
            elapsed = time.time() - start_ts
            raw_response = b"".join(collected_chunks)

            # Reconstruct full assistant text from SSE chunks
            extracted = parse_sse_chunks(raw_response)

            if extracted:
                log_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": round(elapsed, 3),
                    "request": _extract_request_payload(body_bytes, url, request.method),
                }
                log_entry.update(extracted)
                write_llm_log(log_dir, log_entry)
            logger.info(
                f"[{request_id}] stream complete "
                f"status={upstream_status[0]} elapsed={elapsed:.3f}s"
            )

    skip_resp_headers = {"content-encoding", "transfer-encoding", "connection", "content-length"}
    # We can't read upstream_headers until the first chunk arrives inside generate(),
    # so we set basic streaming headers here.
    response_headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(
        generate(),
        status_code=200,  # will be overridden by actual upstream status via headers
        headers=response_headers,
        media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# Helper: extract plain text or tool_calls from a non-streaming response
# ---------------------------------------------------------------------------

def _extract_assistant_message(response_body) -> dict:
    """
    Returns a dict with keys: 'think', 'response', 'tool_calls' (all Optional[str]).
    """
    if not isinstance(response_body, dict):
        return {}
    result: dict = {}
    try:
        message = response_body.get("choices", [{}])[0].get("message", {})

        # Reasoning / think (DeepSeek, Qwen, etc.)
        reasoning = message.get("reasoning_content")
        if reasoning:
            result["think"] = reasoning

        content = message.get("content") or ""

        # Strip inline <think> tags
        think_tag = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_tag and "think" not in result:
            result["think"] = think_tag.group(1).strip()
        clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if clean_content:
            result["response"] = clean_content

        tool_calls = message.get("tool_calls")
        if tool_calls:
            parts = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", "")
                parts.append(f"{name}({args})")
            result["tool_calls"] = "\n".join(parts)

    except (KeyError, IndexError, TypeError):
        pass
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger(args.log_dir, args.log_level)

    logger.info(f"openai-log proxy")
    logger.info(f"  Listen port : {args.port}")
    logger.info(f"  Target      : {args.target}")
    logger.info(f"  Log dir     : {args.log_dir}")

    app = create_app(
        target=args.target.rstrip("/"),
        log_dir=args.log_dir,
        logger=logger,
    )

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
