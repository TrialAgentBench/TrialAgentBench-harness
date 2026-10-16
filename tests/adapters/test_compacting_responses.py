"""Tests for OpenAI Agents SDK Responses compaction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from trialagentbench.adapters.compacting_responses import OpenAIAgentsCompactingProvider
from trialagentbench.ports import LLMResponse
from trialagentbench.util.runtime_context import provider_context


class _FakeCompactEndpoint:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def compact(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(
            output=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Retained result: adjusted effect = 7."}],
                }
            ],
            usage=SimpleNamespace(
                input_tokens=12,
                output_tokens=3,
                total_tokens=15,
                input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
                output_tokens_details={"reasoning_tokens": 0},
            ),
        )


class _FakeClient:
    def __init__(self) -> None:
        self.responses = _FakeCompactEndpoint()


class _RecordingTransport:
    model = "gpt-5.6"

    def __init__(self) -> None:
        self.inputs: list[list[dict[str, object]]] = []

    def generate_responses_turn(
        self,
        input_items: object,
        tools: object,
        **kwargs: object,
    ) -> LLMResponse:
        del tools, kwargs
        assert isinstance(input_items, list)
        self.inputs.append(input_items)
        return LLMResponse(
            content="Use the retained adjusted effect of 7.",
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        )

    def close(self) -> None:
        """Close the fake transport."""

        return None


def test_forced_compaction_preserves_prior_analysis_and_accounts_for_usage() -> None:
    transport = _RecordingTransport()
    client = _FakeClient()
    provider = OpenAIAgentsCompactingProvider(
        transport,
        client=cast(Any, client),
        minimum_candidate_items=1,
        session_id="continuity-test",
    )

    response = provider.generate_turn(
        [
            {"role": "system", "content": "Analyse the trial."},
            {"role": "user", "content": "The adjusted effect was 7. Retain it for the decision."},
        ],
        temperature=0.0,
        max_tokens=128,
    )

    assert len(client.responses.requests) == 1
    assert "Retained result: adjusted effect = 7." in str(transport.inputs[0])
    assert response.content == "Use the retained adjusted effect of 7."
    assert response.usage == {
        "prompt_tokens": 17,
        "completion_tokens": 5,
        "total_tokens": 22,
        "model_prompt_tokens": 5,
        "model_completion_tokens": 2,
        "compaction_events": 1,
        "compaction_requests": 1,
        "compaction_input_tokens": 12,
        "compaction_output_tokens": 3,
    }
    provider.close()
    provider.close()
    with pytest.raises(RuntimeError, match="closed"):
        provider.generate_turn(
            [{"role": "user", "content": "Continue."}],
            temperature=0.0,
            max_tokens=16,
        )


def test_compacting_provider_rejects_reuse_for_another_conversation() -> None:
    provider = OpenAIAgentsCompactingProvider(
        _RecordingTransport(),
        client=cast(Any, _FakeClient()),
        minimum_candidate_items=10,
        session_id="single-conversation-test",
    )
    provider.generate_turn(
        [{"role": "system", "content": "First."}, {"role": "user", "content": "Begin."}],
        temperature=0.0,
        max_tokens=16,
    )

    with pytest.raises(RuntimeError, match="divergent conversations"):
        provider.generate_turn(
            [{"role": "system", "content": "Second."}, {"role": "user", "content": "Begin."}],
            temperature=0.0,
            max_tokens=16,
        )
    provider.close()


def test_managed_context_bypasses_local_character_truncation() -> None:
    provider = OpenAIAgentsCompactingProvider(
        _RecordingTransport(),
        client=cast(Any, _FakeClient()),
        minimum_candidate_items=10,
        session_id="context-owner-test",
    )
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Prompt"},
        {"role": "assistant", "content": "Earlier reasoning"},
    ]

    assert (
        provider_context(
            messages,
            provider=provider,
            session=SimpleNamespace(),  # unused when the provider owns context
            active_prompt_index=1,
            max_chars=1,
        )
        == messages
    )
    provider.close()
