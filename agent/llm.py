from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests

from .logging import LLMCallLogger


class LLMError(RuntimeError):
    pass


def load_env_file(path: str | Path, *, override: bool = False) -> list[str]:
    """Load key-value pairs from a .env file and return loaded key names only."""

    env_path = Path(path)
    if not env_path.exists():
        return []
    loaded: list[str] = []
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or not os.getenv(key):
            os.environ[key] = value
            loaded.append(key)
    return loaded


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        ...


@dataclass
class MockLLMClient:
    model: str = "mock-planner"
    provider: str = "mock"
    _call_count: int = 0

    def _latest_elapsed_seconds(
        self,
        messages: list[dict[str, Any]],
        command_fragment: str,
    ) -> float:
        for message in reversed(messages):
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            result = payload.get("result")
            if not isinstance(result, dict):
                continue
            args = " ".join(str(arg) for arg in result.get("args", []))
            if command_fragment in args and "elapsed_seconds" in result:
                return float(result["elapsed_seconds"])
        return 0.0

    def complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self._call_count += 1
        system_text = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "system"
        )
        if "AIDE-style PDE experiment planner" in system_text:
            if self._call_count == 1:
                return {
                    "content": json.dumps(
                        {
                            "intent": "improve",
                            "hypothesis": "Run a safe autonomous smoke code patch before expensive PDE experiments.",
                            "action_type": "code_patch",
                            "params": {
                                "files": [
                                    {
                                        "path": "mock_autonomous_smoke.py",
                                        "content": "VALUE = 1\n",
                                    }
                                ],
                                "validation_command": ["python", "-c", "print('autonomous mock validation ok')"],
                            },
                            "expected_effect": "verify journal, patch, validation, and report plumbing",
                            "risk": "low",
                        },
                        ensure_ascii=False,
                    )
                }
            return {
                "content": json.dumps(
                    {
                        "intent": "stop",
                        "hypothesis": "Mock autonomous smoke completed.",
                        "action_type": "stop",
                        "params": {"reason": "Mock autonomous smoke test completed."},
                        "expected_effect": "stop without extra compute",
                        "risk": "none",
                    },
                    ensure_ascii=False,
                )
            }

        # Simulate a multi-step research workflow
        if self._call_count == 1:
            return {
                "content": "分析项目状态：有官方 Task 1 FNO/Unet-PF checkpoint 和推理脚本，先跑一次合规 baseline 推理获取指标。",
                "action": {
                    "tool": "record_note",
                    "args": {"note": "Step 1: 读取数据和代码，准备用官方 FNO/Unet-PF checkpoint ensemble 跑 baseline 推理"},
                },
            }
        elif self._call_count == 2:
            return {
                "content": "运行官方 Task 1 checkpoint 推理生成 Task 1 test 预测。",
                "action": {
                    "tool": "run_shell",
                    "args": {
                        "args": [
                            "python",
                            "code/official_checkpoint_ensemble.py",
                            "--input",
                            "data/Task1/task1_test.hdf5",
                            "--output",
                            "runs/mock-test/task1_pred.hdf5",
                            "--batch-size",
                            "64",
                            "--models",
                            "fno=checkpoints/extracted/1D_Burgers_Sols_Nu0.001_FNO.pt",
                            "unet_pf20=checkpoints/extracted/1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt",
                            "--weights",
                            "0.12",
                            "0.88",
                        ],
                        "timeout": 300,
                    },
                },
            }
        elif self._call_count == 3:
            inference_time = self._latest_elapsed_seconds(messages, "code/official_checkpoint_ensemble.py")
            return {
                "content": "生成并校验 Task 1-only 提交目录。",
                "action": {
                    "tool": "create_task1_submission",
                    "args": {
                        "prediction_path": "runs/mock-test/task1_pred.hdf5",
                        "initial_path": "data/data_and_sample_submission/data_and_sample_submission/train_val_test_init/task1_test.hdf5",
                        "output_dir": "runs/mock-test/submission",
                        "code_dir": "code",
                        "train_time": "elapsed_without_inference",
                        "inference_time": inference_time,
                    },
                },
            }
        elif self._call_count == 4:
            return {
                "content": "再次调用提交校验器确认目录完整。",
                "action": {
                    "tool": "validate_submission",
                    "args": {"path": "runs/mock-test/submission"},
                },
            }
        else:
            return {
                "content": "Mock 测试完成，Agent 框架运行正常。",
                "action": {
                    "tool": "stop",
                    "args": {"reason": "Mock agent smoke test completed successfully."},
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
