#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
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
    parser = argparse.ArgumentParser(description="Check LiteLLM connectivity without printing API keys.")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    settings = LlmSettings.from_env()
    settings.enabled = True
    settings.max_tokens = args.max_tokens
    settings.temperature = 0

    print(f"model={settings.model}")
    print(f"provider={settings.provider or 'auto'}")
    print(f"base_url={'set' if settings.api_base else 'not_set'}")
    print(f"api_key={'set' if settings.api_key else 'not_set'}")

    client = LiteLlmClient(settings)
    if not client.available:
        raise SystemExit("CONNECTIVITY_FAILED: LLM client is not available; key or base_url not detected")

    try:
        response = client.complete(
            [
                {"role": "system", "content": "You are a connectivity test. Return only compact JSON."},
                {"role": "user", "content": 'Return exactly this JSON object: {"ok":true}'},
            ]
        )
    except Exception as exc:
        raise SystemExit("CONNECTIVITY_FAILED: " + scrub(str(exc))) from None

    print("CONNECTIVITY_OK")
    print("response=" + response.text.strip()[:200])


if __name__ == "__main__":
    main()
