from __future__ import annotations

import json
from pathlib import Path

import pytest
from trialagentbench_test_helpers import (
    minimal_participant_output_contract,
    write_minimal_trialeval_release_dictionaries,
)

from trialagentbench_harness.ports import CodeExecutionResultV1, LLMResponse, ToolCall
from trialagentbench_harness.trialeval import agent
from trialagentbench_harness.trialeval.schema import BenchmarkItem
from trialagentbench_harness.util.runtime_context import FINAL_TURN_SUBMISSION_REMINDER


class _Session:
    def execute(self, code: str) -> str:
        return ""

    def execute_result(self, code: str) -> CodeExecutionResultV1:
        return CodeExecutionResultV1(status="success", output="", elapsed_seconds=0.0)

    def close(self) -> None:
        return None


class _FailingProvider:
    model = "test"
    telemetry_route = "test"

    def generate_turn(self, **_: object) -> LLMResponse:
        raise TimeoutError("provider timed out")


class _NoSubmissionProvider:
    model = "test"
    telemetry_route = "test"

    def __init__(self) -> None:
        self.timeout_seconds: float | None = None
        self.messages: list[dict[str, object]] = []
        self.tools: list[list[dict[str, object]]] = []
        self.tool_choices: list[object] = []

    def generate_turn(self, **kwargs: object) -> LLMResponse:
        timeout = kwargs.get("timeout_seconds")
        assert timeout is None or isinstance(timeout, float)
        self.timeout_seconds = timeout
        messages = kwargs.get("messages")
        assert isinstance(messages, list)
        self.messages = messages
        tools = kwargs.get("tools")
        assert isinstance(tools, list)
        self.tools.append(tools)
        self.tool_choices.append(kwargs.get("tool_choice"))
        return LLMResponse(content="analysis without submission")


class _ToolThenNoSubmissionProvider(_NoSubmissionProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def generate_turn(self, **kwargs: object) -> LLMResponse:
        self.calls += 1
        super().generate_turn(**kwargs)
        if self.calls == 1:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="execute",
                        name="execute_code",
                        arguments='{"code":"print(42)"}',
                    )
                ]
            )
        return LLMResponse(content="analysis without submission")


class _InvalidSubmissionProvider(_NoSubmissionProvider):
    def generate_turn(self, **kwargs: object) -> LLMResponse:
        super().generate_turn(**kwargs)
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="submit",
                    name="submit_response",
                    arguments='{"task_id":"TASK1"}',
                )
            ]
        )


def _item(tmp_path: Path) -> BenchmarkItem:
    visible = tmp_path / "visible"
    data = visible / "data"
    hidden = tmp_path / "hidden"
    data.mkdir(parents=True)
    hidden.mkdir()
    write_minimal_trialeval_release_dictionaries(tmp_path)
    return BenchmarkItem(
        item_id="TASK1",
        task_id="TASK1",
        trial_name="fixture",
        design_tier="D1",
        design_subtype="individual_randomized",
        assumption_tier="A1",
        context_tier="C4",
        data_preparation="raw_domains",
        analysis_specification="protocol_only",
        visible_dir=visible,
        data_dir=data,
        task={},
        submission_contract=minimal_participant_output_contract(
            "TASK1",
            data_preparation="raw_domains",
        ),
        suite_dir=tmp_path,
    )


def test_trialeval_provider_failure_finalizes_request_and_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "DockerPythonSession", lambda **_: _Session())
    provider_path = tmp_path / "TASK1_provider_responses.jsonl"
    event_path = tmp_path / "TASK1_events.jsonl"

    with pytest.raises(TimeoutError, match="provider timed out"):
        agent.run_agent(
            _item(tmp_path),
            _FailingProvider(),
            max_turns=1,
            analysis_specification="protocol_only",
            verbose=False,
            provider_log_path=provider_path,
            event_log_path=event_path,
        )

    provider_events = [json.loads(line) for line in provider_path.read_text(encoding="utf-8").splitlines()]
    assert [event["status"] for event in provider_events] == ["started", "failed"]
    assert provider_events[-1]["failure_type"] == "timeout"
    runtime_events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event_id"] for event in runtime_events] == [
        f"trialeval:TASK1:{index:06d}" for index in range(len(runtime_events))
    ]
    assert all(event["timestamp"].endswith("Z") for event in runtime_events)
    assert all(event["source_artifact_path"] for event in runtime_events)
    assert all(len(event["source_payload_sha256"]) == 64 for event in runtime_events)
    terminals = [event for event in runtime_events if event["event_type"] == "step_terminal"]
    assert len(terminals) == 1
    assert terminals[0]["terminal_status"] == "failed"
    assert terminals[0]["failure_type"] == "TimeoutError"


def test_trialeval_executor_initialization_failure_still_finalizes_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_session(**_: object) -> _Session:
        raise RuntimeError("executor unavailable")

    monkeypatch.setattr(agent, "DockerPythonSession", fail_session)
    event_path = tmp_path / "TASK1_events.jsonl"

    with pytest.raises(RuntimeError, match="executor unavailable"):
        agent.run_agent(
            _item(tmp_path),
            _FailingProvider(),
            max_turns=1,
            analysis_specification="protocol_only",
            verbose=False,
            event_log_path=event_path,
        )

    runtime_events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    terminals = [event for event in runtime_events if event["event_type"] == "step_terminal"]
    assert len(terminals) == 1
    assert terminals[0]["terminal_status"] == "failed"
    assert terminals[0]["failure_type"] == "RuntimeError"


def test_trialeval_turn_limit_is_failed_terminal_and_workspace_is_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "DockerPythonSession", lambda **_: _Session())
    workspace = tmp_path / "item-workspace"
    event_path = tmp_path / "events.jsonl"
    provider = _NoSubmissionProvider()
    item = _item(tmp_path)

    output = agent.run_agent(
        item,
        provider,
        max_turns=1,
        analysis_specification="protocol_only",
        verbose=False,
        event_log_path=event_path,
        item_workspace=workspace,
        max_elapsed_seconds=10.0,
    )

    assert output["status"] == "max_turns_reached"
    assert provider.messages[-1] == {
        "role": "user",
        "content": FINAL_TURN_SUBMISSION_REMINDER,
    }
    assert provider.timeout_seconds is not None
    assert 0.0 < provider.timeout_seconds <= 10.0
    terminals = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "step_terminal"
    ]
    assert terminals == [
        {
            **terminals[0],
            "terminal_status": "failed",
            "failure_type": "turn_limit_no_submission",
        }
    ]
    prompt_events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "prompt"
    ]
    assert len(prompt_events) == 2
    assert (workspace / "item").is_dir()
    audit_marker = workspace / "item" / "scratch" / "audit-marker.txt"
    audit_marker.parent.mkdir(parents=True)
    audit_marker.write_text("retained", encoding="utf-8")

    second = agent.run_agent(
        item,
        _NoSubmissionProvider(),
        max_turns=1,
        analysis_specification="protocol_only",
        verbose=False,
        item_workspace=workspace,
    )
    assert second["status"] == "max_turns_reached"
    assert audit_marker.read_text(encoding="utf-8") == "retained"


def test_trialeval_tool_reply_exposes_remaining_turn_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "DockerPythonSession", lambda **_: _Session())
    provider = _ToolThenNoSubmissionProvider()

    output = agent.run_agent(
        _item(tmp_path),
        provider,
        max_turns=2,
        analysis_specification="protocol_only",
        verbose=False,
        item_workspace=tmp_path / "workspace",
    )

    assert output["status"] == "max_turns_reached"
    tool_messages = [message for message in provider.messages if message.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "was not offered on this turn" in str(tool_messages[0]["content"])
    assert str(tool_messages[0]["content"]).endswith("[turn 1/2, 1 left]")


def test_trialeval_terminal_turn_exposes_only_submission_transports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "DockerPythonSession", lambda **_: _Session())
    provider = _NoSubmissionProvider()

    output = agent.run_agent(
        _item(tmp_path),
        provider,
        max_turns=3,
        analysis_specification="protocol_only",
        verbose=False,
        item_workspace=tmp_path / "workspace",
    )

    assert output["status"] == "max_turns_reached"
    assert len(provider.tools) == 3
    assert {tool["function"]["name"] for tool in provider.tools[0]} > {
        "submit_response",
        "submit_response_file",
    }
    assert all(
        [tool["function"]["name"] for tool in request_tools] == ["submit_response", "submit_response_file"]
        for request_tools in provider.tools[1:]
    )
    assert provider.tool_choices == ["auto", "required", "required"]


def test_trialeval_trace_retains_parseable_rejected_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "DockerPythonSession", lambda **_: _Session())
    conversation_path = tmp_path / "conversation.json"

    output = agent.run_agent(
        _item(tmp_path),
        _InvalidSubmissionProvider(),
        max_turns=1,
        analysis_specification="protocol_only",
        verbose=False,
        conversation_log_path=conversation_path,
        item_workspace=tmp_path / "workspace",
    )

    assert output["status"] == "max_turns_reached"
    conversation = json.loads(conversation_path.read_text(encoding="utf-8"))
    rejected = [message for message in conversation if message.get("role") == "tool"]
    assert len(rejected) == 1
    assert rejected[0]["args"] == {"task_id": "TASK1"}
    assert "structured payload is invalid" in rejected[0]["output"]


def test_trialeval_runner_uses_explicit_item_specification_when_not_overridden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct callers must not inherit a locked-SAP default on protocol-only tasks."""

    monkeypatch.setattr(agent, "DockerPythonSession", lambda **_: _Session())
    output = agent.run_agent(
        _item(tmp_path),
        _NoSubmissionProvider(),
        max_turns=1,
        verbose=False,
        item_workspace=tmp_path / "workspace",
    )

    assert output["status"] == "max_turns_reached"
    assert output["condition_provenance"]["analysis_specification"] == "protocol_only"


def test_trialeval_runner_rejects_explicit_specification_mismatch(tmp_path: Path) -> None:
    """An explicit override cannot contradict the immutable participant surface."""

    with pytest.raises(ValueError, match="immutable participant evidence surface"):
        agent.run_agent(
            _item(tmp_path),
            _NoSubmissionProvider(),
            max_turns=1,
            analysis_specification="locked_sap",
            verbose=False,
        )
