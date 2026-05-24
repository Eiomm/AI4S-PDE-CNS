from __future__ import annotations

import os
from pathlib import Path

import requests


class LLMBackend:
    """OpenAI-compatible chat backend supporting DeepSeek, gpt.ge, etc."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: float = 300.0,
        reasoning_effort: str | None = None,
    ):
        self.base_url = (base_url or "https://api.deepseek.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("VAPI_API_KEY") or ""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            raise RuntimeError("LLMBackend: api_key is empty")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
