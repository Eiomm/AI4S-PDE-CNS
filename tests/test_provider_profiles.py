import pytest

from agent.llm import LLMError, OpenAICompatibleClient, build_llm_client
from agent.run import load_config


def test_load_config_merges_named_llm_profile(tmp_path):
    profile_file = tmp_path / "providers.yaml"
    profile_file.write_text(
        "\n".join(
            [
                "profiles:",
                "  kimi:",
                "    provider: kimi",
                "    model: kimi-for-coding",
                "    api_key_env:",
                "      - KIMI_CODE_API_KEY",
                "      - KIMI_API_KEY",
                "    base_url: https://api.kimi.com/coding/v1",
                "    request_options:",
                "      temperature: 0.2",
            ]
        ),
        encoding="utf-8",
    )
    config_file = tmp_path / "task.yaml"
    config_file.write_text(
        "\n".join(
            [
                "llm_profile: kimi",
                f"llm_profiles_path: {profile_file.as_posix()}",
                "max_iterations: 2",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config["provider"] == "kimi"
    assert config["model"] == "kimi-for-coding"
    assert config["api_key_env"] == ["KIMI_CODE_API_KEY", "KIMI_API_KEY"]
    assert config["request_options"]["temperature"] == 0.2
    assert config["max_iterations"] == 2


def test_build_llm_client_accepts_kimi_and_siliconflow_profiles():
    kimi = build_llm_client(
        {
            "provider": "kimi",
            "model": "kimi-for-coding",
            "api_key_env": ["KIMI_CODE_API_KEY", "KIMI_API_KEY"],
            "base_url": "https://api.kimi.com/coding/v1",
        }
    )
    siliconflow = build_llm_client(
        {
            "provider": "siliconflow",
            "model": "Pro/zai-org/GLM-4.7",
            "api_key_env": "SILICONFLOW_API_KEY",
            "base_url": "https://api.siliconflow.cn/v1",
        }
    )

    assert isinstance(kimi, OpenAICompatibleClient)
    assert kimi.provider == "kimi"
    assert kimi.api_key_env == ["KIMI_CODE_API_KEY", "KIMI_API_KEY"]
    assert isinstance(siliconflow, OpenAICompatibleClient)
    assert siliconflow.provider == "siliconflow"


def test_api_key_lookup_accepts_env_fallback_list(monkeypatch):
    monkeypatch.delenv("KIMI_CODE_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.setenv("KIMI_API_KEY", "kimi-secret")
    client = OpenAICompatibleClient(
        provider="kimi",
        model="kimi-for-coding",
        api_key_env=["KIMI_CODE_API_KEY", "KIMI_API_KEY"],
        base_url="https://api.kimi.com/coding/v1",
    )

    assert client.api_key() == "kimi-secret"


def test_api_key_lookup_reports_all_missing_env_names(monkeypatch):
    monkeypatch.delenv("KIMI_CODE_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    client = OpenAICompatibleClient(
        provider="kimi",
        model="kimi-for-coding",
        api_key_env=["KIMI_CODE_API_KEY", "KIMI_API_KEY"],
        base_url="https://api.kimi.com/coding/v1",
    )

    with pytest.raises(LLMError, match="KIMI_CODE_API_KEY, KIMI_API_KEY"):
        client.api_key()
