import pytest

from agent.env_check import collect_environment
from agent.llm import LLMError, build_llm_client


def test_openai_compatible_client_reports_missing_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = build_llm_client({"provider": "deepseek", "model": "deepseek-v4-pro"})

    with pytest.raises(LLMError, match="DEEPSEEK_API_KEY"):
        client.complete([{"role": "user", "content": "hello"}])


def test_environment_check_reports_python_and_torch_fields():
    report = collect_environment()

    assert "python_version" in report
    assert "torch_available" in report
    assert "cuda_available" in report
