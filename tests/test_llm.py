from chem_evolve_agent.generators.base import GenerationContext
from chem_evolve_agent.generators.llm_generator import LlmGenerator
from chem_evolve_agent.llm import (
    LlmSettings,
    LiteLlmClient,
    _audit_request_payload,
    _extract_response_payload,
    _parse_json_object_or_array,
)
from chem_evolve_agent.llm import LlmResponse


def test_parse_json_from_fenced_response():
    assert _parse_json_object_or_array('```json\n["CCO"]\n```') == ["CCO"]


def test_llm_generator_disabled_returns_empty():
    client = LiteLlmClient(LlmSettings(enabled=False))
    generator = LlmGenerator(client=client)
    context = GenerationContext(target_id="target", pocket_summary="none", round_index=0)
    assert generator.generate(context, limit=3) == []


def test_audit_request_masks_api_key():
    payload = _audit_request_payload(
        {
            "model": "openai/gpt-4o-mini",
            "api_key": "secret",
            "api_base": "http://127.0.0.1:8080/v1",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert payload["api_key"] == "***"
    assert payload["api_base"] == "http://127.0.0.1:8080/v1"


def test_response_payload_extracts_think_tag():
    payload = _extract_response_payload(
        LlmResponse(text="<think>hidden</think>\nanswer", model="m", usage={"total_tokens": 3})
    )
    assert payload["think"] == "hidden"
    assert payload["response"] == "answer"
    assert payload["usage"]["total_tokens"] == 3
