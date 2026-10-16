"""Tests for the benchmark-neutral agent turn runtime."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest

from trialagentbench.ports import CodeExecutionResultV1, LLMResponse, LLMResponseMetadata, ToolCall
from trialagentbench.ports.llm_provider import JsonObject
from trialagentbench.util.agent_runtime import assistant_message_v1, offered_tool_names_v1, run_agent_turn_v1


def test_assistant_message_preserves_tool_calls_without_provider_raw_state() -> None:
    response = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="call-1", name="submit", arguments='{"content":"done"}')],
        usage={},
        raw=None,
    )

    assert assistant_message_v1(response) == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "submit", "arguments": '{"content":"done"}'},
            }
        ],
    }


def test_offered_tool_names_rejects_duplicate_names() -> None:
    tool = {"type": "function", "function": {"name": "inspect", "parameters": {}}}

    with pytest.raises(RuntimeError, match="repeats 'inspect'"):
        offered_tool_names_v1([tool, tool])


def test_agent_turn_records_local_context_archiving_and_completion_outcome(tmp_path: Path) -> None:
    class Session:
        def execute_result(self, code: str) -> CodeExecutionResultV1:
            return CodeExecutionResultV1(status="success", output="", elapsed_seconds=0.0)

    class Provider:
        model = "test-model"
        telemetry_route = "test-route"

        def generate_turn(
            self,
            messages: Sequence[JsonObject],
            tools: Sequence[JsonObject] | None = None,
            *,
            temperature: float,
            max_tokens: int,
            timeout_seconds: float | None = None,
            tool_choice: Literal["auto", "required"] = "auto",
        ) -> LLMResponse:
            return LLMResponse(
                content="done",
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                metadata=LLMResponseMetadata(finish_reason="stop"),
            )

    log_path = tmp_path / "provider_responses.jsonl"
    run_agent_turn_v1(
        provider=Provider(),
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "active prompt"},
            {"role": "assistant", "content": "old analysis " * 1_000},
            {"role": "user", "content": "continue"},
        ],
        session=Session(),  # type: ignore[arg-type]
        active_prompt_index=1,
        max_context_chars=100,
        tools=[],
        temperature=0.0,
        max_tokens=256,
        timeout_seconds=30.0,
        tool_choice="auto",
        provider_log_path=log_path,
        benchmark="trialeval",
        unit_id="task-1",
        phase_id="task",
        step_id="analysis",
        turn_index=1,
    )

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["context_management"] == "local_archived"
    assert events[-1]["source_context_characters"] > events[-1]["transmitted_context_characters"]
    assert events[-1]["requested_completion_tokens"] == 256
    assert events[-1]["completion_limit_reached"] is False
