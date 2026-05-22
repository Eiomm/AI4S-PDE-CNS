#!/usr/bin/env python3
"""检查当前 GPT-5.5 API 与官方日志代理是否连通。

这个脚本只用于本地运维检查，不属于最终提交的 `code/`。它会：
1. 从 Task1 根目录的 `.env` 加载密钥和代理配置；
2. 通过本地 OpenAI-compatible proxy 发送一个最小 chat completion 请求；
3. 打印模型返回内容、耗时和当天 proxy jsonl 日志行数，方便确认真实调用已落盘。
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    """加载简单 KEY=VALUE 格式的 .env；不打印任何密钥值。"""

    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_api_key() -> str:
    """按项目配置优先级查找 API key 环境变量。"""

    for name in ("APIFOX_GPT_GE_API_KEY", "VAPI_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(name)
        if value:
            print(f"[api-test] 使用密钥环境变量：{name}")
            return value
    raise RuntimeError("没有找到 APIFOX_GPT_GE_API_KEY / VAPI_API_KEY / OPENAI_API_KEY")


def main() -> None:
    load_dotenv(ROOT / ".env")

    port = os.getenv("AI4S_PROXY_PORT", "8080")
    base_url = os.getenv("AI4S_AGENT_BASE_URL", f"http://127.0.0.1:{port}/v1")
    model = os.getenv("AI4S_AGENT_MODEL", "gpt-5.5")
    log_dir = ROOT / os.getenv("AI4S_PROXY_LOG_DIR", "./logs")
    log_path = log_dir / f"openai_proxy_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"

    print(f"[api-test] base_url={base_url}")
    print(f"[api-test] model={model}")

    client = OpenAI(api_key=get_api_key(), base_url=base_url, timeout=120)
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_completion_tokens=16,
    )
    elapsed = time.perf_counter() - started
    content = (response.choices[0].message.content or "").strip()

    print(f"[api-test] status=ok elapsed_seconds={elapsed:.3f}")
    print(f"[api-test] response_model={response.model}")
    print(f"[api-test] content={content}")
    if log_path.exists():
        line_count = sum(1 for _ in log_path.open("r", encoding="utf-8"))
        print(f"[api-test] proxy_log={log_path}")
        print(f"[api-test] proxy_log_lines={line_count}")
    else:
        print(f"[api-test] proxy_log_missing={log_path}")


if __name__ == "__main__":
    main()
