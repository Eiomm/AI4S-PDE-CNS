from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_assistant_message(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            return first.get("message") or first.get("delta") or first.get("text")
    return None


def _extract_assistant_message_from_sse(text: str) -> dict[str, str] | None:
    """从 OpenAI-compatible SSE 文本中提取 assistant content。

    官方样例里提到 stream 需要单独解析。这里只做我们需要的最小解析：
    遍历 `data: {...}` 行，拼接 `choices[0].delta.content`。如果第三方
    服务把 reasoning 也放进 content，它会被原样记录，保证日志可追溯。
    """

    parts: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        data = stripped.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        first = choices[0]
        if not isinstance(first, dict):
            continue
        delta = first.get("delta") or first.get("message") or {}
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                parts.append(content)
    if not parts:
        return None
    return {"role": "assistant", "content": "".join(parts)}


def create_app(*, target: str, log_dir: Path, trust_env: bool = False) -> FastAPI:
    app = FastAPI()
    log_dir.mkdir(parents=True, exist_ok=True)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def proxy(path: str, request: Request) -> Response:
        started = time.perf_counter()
        body = await request.body()
        headers = {key: value for key, value in request.headers.items() if key.lower() != "host"}
        url = f"{target.rstrip('/')}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        try:
            parsed_request = json.loads(body.decode("utf-8")) if body else None
        except Exception:
            parsed_request = None

        is_stream = bool(isinstance(parsed_request, dict) and parsed_request.get("stream"))

        if is_stream:
            async def stream_upstream():
                status_code = 200
                chunks: list[bytes] = []
                error_payload = None
                try:
                    async with httpx.AsyncClient(timeout=None, trust_env=trust_env) as client:
                        async with client.stream(request.method, url, content=body, headers=headers) as upstream:
                            status_code = upstream.status_code
                            async for chunk in upstream.aiter_bytes():
                                chunks.append(chunk)
                                yield chunk
                except Exception as exc:
                    status_code = 502
                    error_payload = {
                        "error": {
                            "message": f"proxy upstream stream failed: {exc.__class__.__name__}: {exc}",
                            "type": "proxy_upstream_error",
                        }
                    }
                    encoded = f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n".encode("utf-8")
                    chunks.append(encoded)
                    yield encoded
                finally:
                    elapsed = time.perf_counter() - started
                    stream_text = b"".join(chunks).decode("utf-8", errors="replace")
                    assistant_message = _extract_assistant_message_from_sse(stream_text)
                    record = {
                        "timestamp": _now(),
                        "elapsed_seconds": elapsed,
                        "method": request.method,
                        "path": "/" + path,
                        "status_code": status_code,
                        "request": parsed_request,
                        "assistant_message": assistant_message,
                        "response": error_payload if error_payload is not None else {"stream": True, "raw_sse": stream_text},
                    }
                    log_path = log_dir / f"openai_proxy_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            return StreamingResponse(stream_upstream(), media_type="text/event-stream")

        # 默认不继承当前 shell 的 HTTP_PROXY/HTTPS_PROXY。少数上游域名在当前
        # 服务器直连 DNS 不可用时，可以通过 --trust-env 显式允许环境代理。
        try:
            async with httpx.AsyncClient(timeout=None, trust_env=trust_env) as client:
                upstream = await client.request(request.method, url, content=body, headers=headers)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            error_payload = {
                "error": {
                    "message": f"proxy upstream request failed: {exc.__class__.__name__}: {exc}",
                    "type": "proxy_upstream_error",
                }
            }
            record = {
                "timestamp": _now(),
                "elapsed_seconds": elapsed,
                "method": request.method,
                "path": "/" + path,
                "status_code": 502,
                "request": parsed_request,
                "assistant_message": None,
                "response": error_payload,
            }
            log_path = log_dir / f"openai_proxy_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return Response(
                content=json.dumps(error_payload, ensure_ascii=False),
                status_code=502,
                media_type="application/json",
            )

        elapsed = time.perf_counter() - started
        response_body = upstream.content
        parsed_response = None
        assistant_message = None
        try:
            parsed_response = upstream.json()
            assistant_message = _extract_assistant_message(parsed_response)
        except Exception:
            pass
        record = {
            "timestamp": _now(),
            "elapsed_seconds": elapsed,
            "method": request.method,
            "path": "/" + path,
            "status_code": upstream.status_code,
            "request": parsed_request,
            "assistant_message": assistant_message,
            "response": parsed_response,
        }
        log_path = log_dir / f"openai_proxy_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return Response(
            content=response_body,
            status_code=upstream.status_code,
            headers={
                key: value
                for key, value in upstream.headers.items()
                if key.lower() not in {"content-encoding", "transfer-encoding", "content-length", "connection"}
            },
            media_type=upstream.headers.get("content-type"),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible logging proxy.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--target", default="https://api.openai.com")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--trust-env", action="store_true", help="allow upstream requests to use HTTP_PROXY/HTTPS_PROXY")
    args = parser.parse_args()
    uvicorn.run(create_app(target=args.target, log_dir=Path(args.log_dir), trust_env=args.trust_env), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
