from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from .logging import LLMCallLogger


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        ...


@dataclass
class MockLLMClient:
    model: str = "mock-planner"
    provider: str = "mock"

    def complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "content": "Record the current observation and stop after this smoke-test cycle.",
            "action": {
                "tool": "record_note",
                "args": {
                    "note": "Mock Agent observed the project state and recorded a traceable step."
                },
            },
        }


@dataclass
class OpenAICompatibleClient:
    provider: str
    model: str
    api_key_env: str | list[str]
    base_url: str
    request_options: dict[str, Any] | None = None
    timeout_seconds: int = 120

    def api_key(self) -> str:
        env_names = self.api_key_env if isinstance(self.api_key_env, list) else [self.api_key_env]
        for env_name in env_names:
            value = os.getenv(env_name)
            if value:
                return value
        raise LLMError(f"Missing API key environment variable: {', '.join(env_names)}")

    def complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        api_key = self.api_key()
        payload = {
            "model": self.model,
            "messages": messages,
            **(self.request_options or {}),
        }
        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise LLMError(f"{self.provider} API error {response.status_code}: {response.text[:500]}")
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        return {"content": content, "raw": payload}


def build_llm_client(config: dict[str, Any]) -> LLMClient:
    provider = str(config.get("provider", "mock")).lower()
    model = str(config.get("model", "mock-planner"))
    request_options = config.get("request_options")
    timeout_seconds = int(config.get("timeout_seconds", 120))
    if provider == "mock":
        return MockLLMClient(model=model)
    if provider == "deepseek":
        return OpenAICompatibleClient(
            provider="deepseek",
            model=model,
            api_key_env=config.get("api_key_env", "DEEPSEEK_API_KEY"),
            base_url=str(config.get("base_url", "https://api.deepseek.com")),
            request_options=request_options,
            timeout_seconds=timeout_seconds,
        )
    if provider == "openai":
        return OpenAICompatibleClient(
            provider="openai",
            model=model,
            api_key_env=config.get("api_key_env", "OPENAI_API_KEY"),
            base_url=str(config.get("base_url", "https://api.openai.com/v1")),
            request_options=request_options,
            timeout_seconds=timeout_seconds,
        )
    if provider == "kimi":
        return OpenAICompatibleClient(
            provider="kimi",
            model=model,
            api_key_env=config.get("api_key_env", ["KIMI_CODE_API_KEY", "KIMI_API_KEY"]),
            base_url=str(config.get("base_url", "https://api.kimi.com/coding/v1")),
            request_options=request_options,
            timeout_seconds=timeout_seconds,
        )
    if provider == "siliconflow":
        return OpenAICompatibleClient(
            provider="siliconflow",
            model=model,
            api_key_env=config.get("api_key_env", "SILICONFLOW_API_KEY"),
            base_url=str(config.get("base_url", "https://api.siliconflow.cn/v1")),
            request_options=request_options,
            timeout_seconds=timeout_seconds,
        )
    raise LLMError(f"Unsupported provider: {provider}")


def logged_completion(
    client: LLMClient,
    logger: LLMCallLogger,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.complete(messages)
    elapsed = time.perf_counter() - started
    logger.write_call(
        provider=client.provider,
        model=client.model,
        messages=messages,
        response=response,
        elapsed_seconds=round(elapsed, 6),
    )
    return response
