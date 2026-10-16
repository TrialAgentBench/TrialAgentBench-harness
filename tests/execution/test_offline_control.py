"""Public offline controls execute participant tools without a provider SDK."""

import json

import pytest

from trialagentbench.adapters.llm_providers import ProviderRouting, get_provider
from trialagentbench.ports.llm_provider import JsonObject


@pytest.mark.parametrize("language,filename", [("python", "analysis.py"), ("r", "analysis.R")])
def test_offline_control_saves_reproducible_script_without_credentials(monkeypatch, language, filename):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = get_provider(
        f"smoke-zero-{language}",
        routing=ProviderRouting("offline_control"),
        send_temperature=False,
        timeout_s=1,
    )
    tools: list[JsonObject] = [{"function": {"name": "execute_code"}}, {"function": {"name": "submit_response"}}]
    response = provider.generate_turn([], tools, temperature=0, max_tokens=100)
    call = response.tool_calls[0]
    assert call.name == "execute_code"
    assert filename in json.loads(call.arguments)["code"]
    assert response.metadata.reported_cost_usd == 0
    assert response.metadata.request_attempts == 1
    messages: list[JsonObject] = [
        {"role": "assistant", "tool_calls": [{"function": {"name": call.name}}]},
        {"role": "tool", "content": "Saved analysis"},
    ]
    response = provider.generate_turn(messages, tools, temperature=0, max_tokens=100)
    assert response.tool_calls[0].name == "submit_response"
    assert "not clinical or model-performance evidence" in response.tool_calls[0].arguments


def test_offline_control_rejects_unknown_models_and_missing_tools():
    with pytest.raises(ValueError, match="Unknown offline"):
        get_provider("real-model", routing=ProviderRouting("offline_control"), send_temperature=False, timeout_s=1)
    provider = get_provider(
        "smoke-zero-python",
        routing=ProviderRouting("offline_control"),
        send_temperature=False,
        timeout_s=1,
    )
    with pytest.raises(ValueError, match="unavailable"):
        provider.generate_turn([], [], temperature=0, max_tokens=100)
