#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chem_evolve_agent.llm import LiteLlmClient, LlmSettings


def scrub(text: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-***", text)
    text = re.sub(r"(?i)(api[_-]?key[=: ]+)[^\s,]+", r"\1***", text)
    text = re.sub(r"(?i)(authorization[=: ]+bearer )[A-Za-z0-9_\-.]+", r"\1***", text)
    text = re.sub(r"(?i)(x-api-key[=: ]+)[A-Za-z0-9_\-.]+", r"\1***", text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 LiteLLM 连通性，不打印 API key 明文。", add_help=False)
    parser._optionals.title = "参数"
    parser.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    parser.add_argument("--max-tokens", type=int, default=64, help="本次探测允许的最大输出 token 数")
    parser.add_argument("--log-dir", help="可选：临时 LiteLLM 审计日志目录")
    args = parser.parse_args()

    settings = LlmSettings.from_env()
    settings.enabled = True
    settings.max_tokens = args.max_tokens
    settings.temperature = 0
    if args.log_dir:
        settings.log_dir = Path(args.log_dir)
        _run_check(settings)
        return

    with tempfile.TemporaryDirectory(prefix="ai4s_llm_check_") as tmp:
        settings.log_dir = Path(tmp)
        _run_check(settings)


def _run_check(settings: LlmSettings) -> None:
    print(f"模型={settings.model}")
    print(f"服务商={settings.provider or '自动'}")
    print(f"base_url={'已设置' if settings.api_base else '未设置'}")
    print(f"api_key={'已设置' if settings.api_key else '未设置'}")

    client = LiteLlmClient(settings)
    if not client.available:
        raise SystemExit("CONNECTIVITY_FAILED: LLM 客户端不可用；没有检测到 API key")

    try:
        response = client.complete(
            [
                {"role": "system", "content": "You are a connectivity test. Return only compact JSON."},
                {"role": "user", "content": 'Return exactly this JSON object: {"ok":true}'},
            ]
        )
    except Exception as exc:
        raise SystemExit("CONNECTIVITY_FAILED: " + scrub(str(exc))) from None

    print("CONNECTIVITY_OK 连通性检查通过")
    print("响应=" + response.text.strip()[:200])


if __name__ == "__main__":
    main()
