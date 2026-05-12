import pytest
import json

from agent.env_check import collect_environment
from agent.llm import LLMError, MockLLMClient, build_llm_client, load_env_file


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


def test_load_env_file_sets_missing_keys_without_exposing_values(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# local secrets",
                "DEEPSEEK_API_KEY='secret-value'",
                'KIMI_API_KEY="kimi-secret"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    loaded = load_env_file(env_path)

    assert loaded == ["DEEPSEEK_API_KEY", "KIMI_API_KEY"]
    assert "secret-value" not in loaded
    assert "kimi-secret" not in loaded
    assert build_llm_client({"provider": "deepseek", "model": "deepseek-v4-pro"}).api_key() == "secret-value"


def test_mock_llm_reaches_task1_submission_action_with_measured_inference_time():
    client = MockLLMClient()

    first = client.complete([])
    second = client.complete([])
    assert first["action"]["tool"] == "record_note"
    assert second["action"]["tool"] == "run_shell"
    assert "--weights" in second["action"]["args"]["args"]
    assert second["action"]["args"]["args"][-4:] == ["0.01", "0.31", "0.66", "0.02"]

    messages = [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "result": {
                        "args": ["python", "code/fno_ensemble.py"],
                        "returncode": 0,
                        "elapsed_seconds": 22.4,
                    }
                }
            ),
        }
    ]
    third = client.complete(messages)

    assert third["action"]["tool"] == "create_task1_submission"
    assert third["action"]["args"]["inference_time"] == 22.4
