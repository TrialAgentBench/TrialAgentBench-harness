from __future__ import annotations

import ast
import base64
import json
import re
import time
from typing import Literal

import pytest

from trialagentbench_harness.ports.code_execution import CodeExecutionResultV1
from trialagentbench_harness.util.runtime_context import (
    RuntimeDeadline,
    bounded_provider_context,
    persist_bulky_tool_output,
    turn_budget_tag,
)


class _Session:
    def __init__(
        self,
        *,
        status: Literal["success", "session_terminated"] = "success",
    ) -> None:
        self.files: dict[str, str] = {}
        self.status = status

    def execute_result(self, code: str) -> CodeExecutionResultV1:
        expression = ast.parse(code).body[0]
        assert isinstance(expression, ast.Expr)
        assert isinstance(expression.value, ast.Call)
        script = ast.literal_eval(expression.value.args[0])
        path_match = re.search(r"Path\('scratch'\) / ('.*?')", script)
        content_match = re.search(r"base64\.b64decode\(('.*?')\)", script)
        if self.status == "success":
            assert path_match is not None
            assert content_match is not None
            path = ast.literal_eval(path_match.group(1))
            encoded = ast.literal_eval(content_match.group(1))
            self.files[path] = base64.b64decode(encoded).decode("utf-8")
        return CodeExecutionResultV1(
            status=self.status,
            output="" if self.status == "success" else "session unavailable",
            elapsed_seconds=0.0,
        )

    def execute(self, code: str) -> str:
        return self.execute_result(code).output

    def close(self) -> None:
        return None


def test_context_archives_complete_assistant_tool_pair_without_summary() -> None:
    session = _Session()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old step"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "execute_code", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "old evidence"},
        {"role": "user", "content": "current step"},
        {"role": "assistant", "content": "current analysis"},
    ]

    context = bounded_provider_context(
        messages,
        session=session,
        active_prompt_index=4,
        max_chars=140,
    )

    assert context[0] == messages[0]
    assert messages[4] in context
    assert messages[2] not in context
    assert messages[3] not in context
    archive = json.loads(session.files["context_archive.json"])
    assert archive["messages"] == messages[1:4]
    assert any("context_archive.json" in str(message.get("content")) for message in context)
    assert not any("scratch/context_archive.json" in str(message.get("content")) for message in context)
    assert not any(message.get("role") == "assistant" and "summary" in message for message in context)


def test_context_rejects_unmatched_or_duplicate_tool_call_ids() -> None:
    session = _Session()
    unmatched = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "current"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "tool", "arguments": "{}"},
                }
            ],
        },
    ]
    with pytest.raises(ValueError, match="do not match"):
        bounded_provider_context(
            unmatched,
            session=session,
            active_prompt_index=1,
            max_chars=1000,
        )

    duplicate = [
        *unmatched[:2],
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "tool", "arguments": "{}"},
                },
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "tool", "arguments": "{}"},
                },
            ],
        },
    ]
    with pytest.raises(ValueError, match="Duplicate assistant tool_call_id"):
        bounded_provider_context(
            duplicate,
            session=session,
            active_prompt_index=1,
            max_chars=1000,
        )


def test_context_retains_required_terminal_message_beyond_soft_budget() -> None:
    session = _Session()
    final_reminder = {"role": "user", "content": "submit now"}
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "active prompt " * 100},
        {"role": "assistant", "content": "recent analysis " * 100},
        final_reminder,
    ]

    context = bounded_provider_context(
        messages,
        session=session,
        active_prompt_index=1,
        max_chars=20,
        required_message_indices=(3,),
    )

    assert final_reminder in context
    assert messages[2] not in context


def test_bulky_output_is_losslessly_persisted() -> None:
    session = _Session()
    output = "evidence-" * 100

    rendered = persist_bulky_tool_output(
        output,
        session=session,
        artifact_id="phase-turn-call",
        inline_chars=20,
    )

    retained_path, retained_content = next(iter(session.files.items()))
    assert retained_content == output
    assert retained_path in rendered
    assert f"scratch/{retained_path}" not in rendered


def test_runtime_workspace_persistence_fails_loudly() -> None:
    session = _Session(status="session_terminated")

    with pytest.raises(RuntimeError, match="Unable to persist runtime workspace artifact"):
        persist_bulky_tool_output(
            "evidence-" * 100,
            session=session,
            artifact_id="phase-turn-call",
            inline_chars=20,
        )


def test_turn_budget_tag_is_bounded_and_suite_independent() -> None:
    assert turn_budget_tag(turn=1, maximum=3) == "\n[turn 1/3, 2 left]"
    assert turn_budget_tag(turn=3, maximum=3) == "\n[turn 3/3, 0 left]"

    with pytest.raises(ValueError, match="within"):
        turn_budget_tag(turn=4, maximum=3)


def test_blocking_operation_is_capped_to_remaining_deadline() -> None:
    deadline = RuntimeDeadline.after(0.02, label="test operation")
    timed_out: list[bool] = []

    with pytest.raises(TimeoutError, match="blocking"):
        deadline.run_blocking(
            lambda: time.sleep(0.2),
            operation_name="slow tool",
            on_timeout=lambda: timed_out.append(True),
        )

    assert timed_out == [True]
