from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

_DEFAULT_API_KEY_ENVS = [
    "APIFOX_GPT_GE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "VAPI_API_KEY",
    "AIGC_API_KEY",
    "HKUSTGZ_AIGC_API_KEY",
    "KIMI_CODE_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
    "DEEPSEEK_API_KEY",
    "SILICONFLOW_API_KEY",
]


class LlmSettings(BaseModel):
    enabled: bool = False
    model: str = "openai/claude-opus-4-8"
    provider: Optional[str] = "openai"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    api_key_envs: List[str] = Field(default_factory=lambda: list(_DEFAULT_API_KEY_ENVS))
    temperature: float = 0.4
    max_tokens: int = 700
    timeout: int = 60
    max_retries: int = 3
    log_dir: Path = Path("runs/llm_io")
    raw_debug: bool = False

    @classmethod
    def from_env(cls) -> "LlmSettings":
        _load_dotenv_if_available()
        enabled = _env_bool("CHEM_EVOLVE_LLM_ENABLED", default=False)
        api_key_envs = _split_env_list(os.getenv("AI4S_AGENT_API_KEY_ENVS")) or list(_DEFAULT_API_KEY_ENVS)
        api_key = _resolve_api_key(api_key_envs)
        return cls(
            enabled=enabled,
            model=os.getenv("CHEM_EVOLVE_LLM_MODEL") or os.getenv("AI4S_AGENT_MODEL", "openai/claude-opus-4-8"),
            provider=os.getenv("CHEM_EVOLVE_LLM_PROVIDER") or os.getenv("AI4S_AGENT_PROVIDER"),
            api_base=os.getenv("CHEM_EVOLVE_LLM_BASE_URL") or os.getenv("AI4S_AGENT_BASE_URL") or os.getenv("OPENAI_API_BASE") or None,
            api_key=api_key,
            api_key_envs=api_key_envs,
            temperature=float(os.getenv("CHEM_EVOLVE_LLM_TEMPERATURE") or os.getenv("AI4S_AGENT_TEMPERATURE", "0.4")),
            max_tokens=int(os.getenv("CHEM_EVOLVE_LLM_MAX_TOKENS") or os.getenv("AI4S_AGENT_MAX_TOKENS", "700")),
            timeout=int(os.getenv("CHEM_EVOLVE_LLM_TIMEOUT") or os.getenv("AI4S_AGENT_TIMEOUT", "60")),
            max_retries=int(os.getenv("CHEM_EVOLVE_LLM_MAX_RETRIES") or os.getenv("AI4S_AGENT_MAX_RETRIES", "3")),
            log_dir=Path(os.getenv("CHEM_EVOLVE_LLM_LOG_DIR") or os.getenv("AI4S_AGENT_LLM_LOG_DIR", "runs/llm_io")),
            raw_debug=_env_bool("AI4S_PROVIDER_RAW_DEBUG", default=False),
        )


class LlmResponse(BaseModel):
    text: str
    usage: Dict[str, Any] = Field(default_factory=dict)
    model: str
    raw: Dict[str, Any] = Field(default_factory=dict)


class LiteLlmClient:
    def __init__(self, settings: Optional[LlmSettings] = None):
        self.settings = settings or LlmSettings.from_env()
        self._configure_litellm()

    @property
    def available(self) -> bool:
        if not self.settings.enabled:
            return False
        try:
            import litellm  # noqa: F401
        except Exception:
            return False
        return bool(self.settings.api_key or self.settings.api_base)

    def complete(self, messages: List[Dict[str, str]]) -> LlmResponse:
        if not self.available:
            raise RuntimeError("LiteLLM is disabled or not installed")

        from litellm import completion

        kwargs = self._completion_kwargs(messages)
        started = time.perf_counter()
        last_error: Optional[Exception] = None
        for attempt in range(1, max(1, self.settings.max_retries) + 1):
            try:
                response = completion(**kwargs)
                elapsed = time.perf_counter() - started
                llm_response = self._normalize_response(response)
                if not llm_response.text.strip():
                    raise RuntimeError("LLM returned an empty response")
                self._write_llm_log(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": round(elapsed, 3),
                        "attempt": attempt,
                        "request": _audit_request_payload(kwargs),
                        **_extract_response_payload(llm_response),
                    }
                )
                return llm_response
            except Exception as exc:
                last_error = exc
                self._write_llm_log(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "attempt": attempt,
                        "request": _audit_request_payload(kwargs),
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                )
                if attempt >= max(1, self.settings.max_retries):
                    break
                time.sleep(min(8.0, 1.5 * attempt))
        raise RuntimeError(f"LiteLLM call failed after {self.settings.max_retries} attempt(s): {last_error}") from last_error

    def _completion_kwargs(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "timeout": self.settings.timeout,
        }
        if self.settings.api_key:
            kwargs["api_key"] = self.settings.api_key
        if self.settings.api_base:
            kwargs["api_base"] = self.settings.api_base
        if self.settings.provider:
            kwargs["custom_llm_provider"] = self.settings.provider
        return kwargs

    def _normalize_response(self, response: Any) -> LlmResponse:
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return LlmResponse(
            text=text,
            usage=usage.model_dump() if hasattr(usage, "model_dump") else dict(usage or {}),
            model=self.settings.model,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    def complete_json(self, messages: List[Dict[str, str]]) -> Any:
        response = self.complete(messages)
        return _parse_json_object_or_array(response.text)

    def _configure_litellm(self) -> None:
        try:
            import litellm
        except Exception:
            return
        litellm.telemetry = False
        litellm.suppress_debug_info = True
        litellm.set_verbose = self.settings.raw_debug
        os.environ.setdefault("LITELLM_LOG", "DEBUG" if self.settings.raw_debug else "ERROR")
        os.environ.setdefault("LITELLM_SET_VERBOSE", "True" if self.settings.raw_debug else "False")

    def _write_llm_log(self, entry: Dict[str, Any]) -> None:
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.settings.log_dir / f"llm-{datetime.now().strftime('%Y%m%d')}.jsonl"
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _parse_json_object_or_array(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith(("\"", "'")):
        stripped = stripped.strip("\"'")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1).strip())

    start_positions = [pos for pos in [text.find("["), text.find("{")] if pos >= 0]
    if not start_positions:
        raise ValueError("LLM response did not contain JSON")
    start = min(start_positions)
    end = max(text.rfind("]"), text.rfind("}"))
    if end <= start:
        raise ValueError("LLM response contained incomplete JSON")
    return json.loads(text[start : end + 1])


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_env_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_api_key(env_names: List[str]) -> Optional[str]:
    for name in env_names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _audit_request_payload(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": kwargs.get("model"),
        "temperature": kwargs.get("temperature"),
        "max_tokens": kwargs.get("max_tokens"),
        "api_base": kwargs.get("api_base"),
        "custom_llm_provider": kwargs.get("custom_llm_provider"),
        "messages": kwargs.get("messages", []),
    }
    if kwargs.get("api_key"):
        payload["api_key"] = "***"
    return {key: value for key, value in payload.items() if value is not None}


def _extract_response_payload(response: LlmResponse) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    think_match = re.search(r"<think>(.*?)</think>", response.text, flags=re.DOTALL)
    text = response.text
    if think_match:
        result["think"] = think_match.group(1).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text:
        result["response"] = text
    if response.usage:
        result["usage"] = response.usage
    return result
