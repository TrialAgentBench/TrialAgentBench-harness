from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.core.runs import (
    TrialDevMaterializationUsageV1,
)
from trialagentbench_harness.contracts.trialdev.run_checkpoint import (
    TrialDevRunCheckpointPayloadV1,
    TrialDevRunCheckpointV1,
)
from trialagentbench_harness.contracts.trialdev.runtime_checkpoint import (
    TrialDevCheckpointPhaseSummaryV1,
    TrialDevCheckpointViolationV1,
    TrialDevContinuationCheckpointV1,
    TrialDevContinuationPayloadV1,
)
from trialagentbench_harness.ports import (
    CodeExecutionLimitsV1,
    CodeExecutionResultV1,
    LLMResponse,
    ToolCall,
)
from trialagentbench_harness.trialdev import agent, runner
from trialagentbench_harness.util.provider_telemetry import (
    start_provider_request_v1,
    succeed_provider_request_v1,
)
from trialagentbench_harness.util.runtime_context import FINAL_TURN_SUBMISSION_REMINDER


def _write_path_stats_conversation(path: Path, *, phase_id: str, tool_name: str) -> None:
    path.write_text(
        json.dumps(
            [
                {"role": "user", "content": f"PHASE: {phase_id}"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": tool_name}}],
                },
            ]
        ),
        encoding="utf-8",
    )


def test_phase_path_stats_prefer_active_checkpoint_conversation(tmp_path: Path) -> None:
    _write_path_stats_conversation(
        tmp_path / "conversation.json",
        phase_id="phase1",
        tool_name="inspect_parquet",
    )
    _write_path_stats_conversation(
        tmp_path / "checkpoint_conversation.json",
        phase_id="phase2",
        tool_name="execute_code",
    )

    stats = runner._phase_path_stats_from_program_dir(tmp_path)

    assert stats == {
        "phase2": {
            "turns": 1,
            "execute_code": 1,
            "inspect_parquet": 0,
        }
    }


def test_phase_path_stats_use_standard_conversation_without_checkpoint(tmp_path: Path) -> None:
    _write_path_stats_conversation(
        tmp_path / "conversation.json",
        phase_id="phase3",
        tool_name="inspect_parquet",
    )

    stats = runner._phase_path_stats_from_program_dir(tmp_path)

    assert stats == {
        "phase3": {
            "turns": 1,
            "execute_code": 0,
            "inspect_parquet": 1,
        }
    }


class _Session:
    def __init__(self, *, scratch_root: Path = Path(".")) -> None:
        self.scratch_root = scratch_root

    def execute(self, code: str) -> str:
        return "executed"

    def execute_result(self, code: str) -> CodeExecutionResultV1:
        return CodeExecutionResultV1(status="success", output="executed", elapsed_seconds=0.1)

    def snapshot_scratch(self) -> Path:
        return Path(self.scratch_root)

    def close(self) -> None:
        return None


class _Provider:
    model = "test"
    telemetry_route = "test"

    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[dict[str, object]] = []
        self.tools: list[dict[str, object]] = []
        self.tool_choices: list[str] = []

    def generate_turn(
        self,
        messages,
        tools=None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        self.calls += 1
        self.messages = list(messages)
        self.tools = list(tools or [])
        self.tool_choices.append(tool_choice)
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="submit-1",
                    name="submit_phase_analysis",
                    arguments=json.dumps({"primary_method_id": "rmst"}),
                )
            ]
        )


class _ScalarSubmissionProvider:
    model = "test"
    telemetry_route = "test"

    def generate_turn(
        self,
        messages,
        tools=None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        return LLMResponse(
            tool_calls=[ToolCall(id="submit-1", name="submit_phase_analysis", arguments='["not", "an", "object"]')]
        )


class _ToolThenSubmitProvider:
    model = "test"
    telemetry_route = "test"

    def __init__(self) -> None:
        self.calls = 0

    def generate_turn(
        self,
        messages,
        tools=None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="code-1",
                        name="execute_code",
                        arguments=json.dumps({"code": "print('analysis')"}),
                    )
                ]
            )
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="submit-1",
                    name="submit_phase_analysis",
                    arguments=json.dumps({"primary_method_id": "rmst"}),
                )
            ]
        )


class _FileSession(_Session):
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def execute_result(self, code: str) -> CodeExecutionResultV1:
        return CodeExecutionResultV1(
            status="success",
            output=base64.b64encode(self.payload.encode("utf-8")).decode("ascii"),
            elapsed_seconds=0.1,
        )


class _FileSubmissionProvider:
    model = "test"
    telemetry_route = "test"

    def generate_turn(
        self,
        messages,
        tools=None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="submit-file-1",
                    name="submit_phase_analysis_file",
                    arguments=json.dumps({"path": "analysis.json"}),
                )
            ]
        )


class _FailingProvider:
    model = "test"
    telemetry_route = "test"

    def generate_turn(
        self,
        messages,
        tools=None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        raise TimeoutError("provider timed out")


def test_agent_loop_emits_phase_linked_runtime_events(tmp_path: Path, monkeypatch) -> None:
    session_arguments: dict[str, object] = {}

    def session_factory(**kwargs: object) -> _Session:
        session_arguments.update(kwargs)
        return _Session()

    monkeypatch.setattr(agent, "DockerPythonSession", session_factory)
    limits = CodeExecutionLimitsV1(timeout_seconds=37)
    provider = _Provider()
    loop = agent.AgentLoop(
        provider=provider,
        workdir=tmp_path / "work",
        system_prompt="system",
        conversation_log_path=tmp_path / "conversation.json",
        event_log_path=tmp_path / "events.jsonl",
        provider_log_path=tmp_path / "provider_responses.jsonl",
        program_id="program-1",
        scenario_id="01",
        objective_id="benefit_risk",
        executor_limits=limits,
    )
    loop.begin_step(phase_id="phase2", step_id="trial_analysis")
    loop.append_user_message("analyse")
    capture = loop.run_until_submit(
        tools=[],
        submit_tool_names={"submit_phase_analysis"},
    )
    loop.close()

    assert capture.payload == {"primary_method_id": "rmst"}
    assert provider.tool_choices == ["auto"]
    assert session_arguments["limits"] == limits
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [event["event_index"] for event in events] == list(range(len(events)))
    assert [event["event_id"] for event in events] == [
        f"trialdev:program-1:{index:06d}" for index in range(len(events))
    ]
    assert all(event["timestamp"].endswith("Z") for event in events)
    assert all(event["source_artifact_path"] == (tmp_path / "conversation.json").as_posix() for event in events)
    assert all(len(event["source_payload_sha256"]) == 64 for event in events)
    assert {event["phase_id"] for event in events} == {"phase2"}
    assert events[-2]["event_type"] == "submission"
    assert events[-2]["conversation_message_index"] == 2
    assert events[-1]["event_type"] == "step_terminal"
    assert events[-1]["terminal_status"] == "completed"
    telemetry = [
        json.loads(line) for line in (tmp_path / "provider_responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["status"] for event in telemetry] == ["started", "succeeded"]
    assert telemetry[0]["unit_id"] == "program-1"
    assert telemetry[0]["phase_id"] == "phase2"
    assert telemetry[0]["provider_route"] == "test"


def test_successful_retry_inside_exception_handler_is_not_marked_failed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """An outer handled rejection cannot contaminate nested step telemetry."""

    monkeypatch.setattr(
        agent,
        "DockerPythonSession",
        lambda **kwargs: _Session(scratch_root=Path(kwargs["cwd"]) / "scratch"),
    )
    loop = agent.AgentLoop(
        provider=_Provider(),
        workdir=tmp_path / "work",
        system_prompt="system",
        event_log_path=tmp_path / "events.jsonl",
        program_id="program-1",
        scenario_id="01",
        objective_id="benefit_risk",
    )
    loop.begin_step(phase_id="phase2", step_id="trial_design_request")
    loop.append_user_message("revise request")

    try:
        raise ValueError("handled materialization rejection")
    except ValueError:
        capture = loop.run_until_submit(
            tools=[],
            submit_tool_names={"submit_phase_analysis"},
        )
    loop.close()

    assert capture.payload == {"primary_method_id": "rmst"}
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    terminals = [event for event in events if event["event_type"] == "step_terminal"]
    assert len(terminals) == 1
    assert terminals[0]["terminal_status"] == "completed"
    assert terminals[0]["failure_type"] is None


def test_step_turn_budget_is_cumulative_across_validation_resubmission(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(agent, "DockerPythonSession", lambda **_: _Session())
    provider = _Provider()
    loop = agent.AgentLoop(
        provider=provider,
        workdir=tmp_path / "work",
        system_prompt="system",
        max_turns_per_step=1,
        event_log_path=tmp_path / "events.jsonl",
        program_id="program-1",
        scenario_id="01",
        objective_id="benefit_risk",
    )
    loop.begin_step(phase_id="phase2", step_id="trial_analysis")
    loop.append_user_message("analyse")

    loop.run_until_submit(tools=[], submit_tool_names={"submit_phase_analysis"})
    with pytest.raises(RuntimeError, match="within 1 turns"):
        loop.run_until_submit(tools=[], submit_tool_names={"submit_phase_analysis"})

    assert provider.calls == 1
    assert provider.messages[-1] == {
        "role": "user",
        "content": FINAL_TURN_SUBMISSION_REMINDER,
    }
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["terminal_status"] == "failed"
    assert events[-1]["failure_type"] == "turn_limit_no_submission"
    loop.close()


def test_final_turn_offers_only_submission_tools(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(agent, "DockerPythonSession", lambda **_: _Session())
    provider = _Provider()
    loop = agent.AgentLoop(
        provider=provider,
        workdir=tmp_path / "work",
        system_prompt="system",
        max_turns_per_step=1,
        tool_choice="required",
        program_id="program-1",
        scenario_id="01",
        objective_id="benefit_risk",
    )
    loop.begin_step(phase_id="phase2", step_id="trial_analysis")
    loop.append_user_message("analyse")

    loop.run_until_submit(
        tools=agent.tools_for_phase_analysis(),
        submit_tool_names={"submit_phase_analysis", "submit_phase_analysis_file"},
    )

    assert [tool["function"]["name"] for tool in provider.tools] == [
        "submit_phase_analysis",
        "submit_phase_analysis_file",
    ]
    assert provider.tool_choices == ["required"]
    loop.close()


def test_program_deadline_applies_before_provider_execution(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(agent, "DockerPythonSession", lambda **_: _Session())
    provider = _Provider()
    loop = agent.AgentLoop(
        provider=provider,
        workdir=tmp_path / "work",
        system_prompt="system",
        deadline_monotonic=0.0,
        program_id="program-1",
        scenario_id="01",
        objective_id="benefit_risk",
    )
    loop.begin_step(phase_id="phase2", step_id="trial_analysis")
    loop.append_user_message("analyse")

    with pytest.raises(TimeoutError, match="wall-clock budget"):
        loop.run_until_submit(tools=[], submit_tool_names={"submit_phase_analysis"})

    assert provider.calls == 0
    loop.close()


def test_provider_failure_finalizes_request_and_step_once(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(agent, "DockerPythonSession", lambda **_: _Session())
    loop = agent.AgentLoop(
        provider=_FailingProvider(),
        workdir=tmp_path / "work",
        system_prompt="system",
        event_log_path=tmp_path / "events.jsonl",
        provider_log_path=tmp_path / "provider_responses.jsonl",
        program_id="program-1",
        scenario_id="01",
        objective_id="benefit_risk",
    )
    loop.begin_step(phase_id="phase2", step_id="trial_analysis")
    loop.append_user_message("analyse")

    with pytest.raises(TimeoutError, match="provider timed out"):
        loop.run_until_submit(tools=[], submit_tool_names={"submit_phase_analysis"})
    loop.close()

    provider_events = [
        json.loads(line) for line in (tmp_path / "provider_responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["status"] for event in provider_events] == ["started", "failed"]
    assert provider_events[-1]["failure_type"] == "timeout"
    runtime_events = [
        json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    terminals = [event for event in runtime_events if event["event_type"] == "step_terminal"]
    assert len(terminals) == 1
    assert terminals[0]["terminal_status"] == "failed"
    assert terminals[0]["failure_type"] == "TimeoutError"


def test_agent_loop_reads_file_submission_through_same_payload_boundary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        agent,
        "DockerPythonSession",
        lambda **_: _FileSession('{"primary_method_id":"rmst"}'),
    )
    loop = agent.AgentLoop(
        provider=_FileSubmissionProvider(),
        workdir=tmp_path / "work",
        system_prompt="system",
        event_log_path=tmp_path / "events.jsonl",
        program_id="program-1",
        scenario_id="01",
        objective_id="benefit_risk",
    )
    loop.begin_step(phase_id="phase2", step_id="trial_analysis")
    loop.append_user_message("analyse")

    capture = loop.run_until_submit(
        tools=agent.tools_for_phase_analysis(),
        submit_tool_names={"submit_phase_analysis", "submit_phase_analysis_file"},
    )
    loop.close()

    assert capture.name == "submit_phase_analysis_file"
    assert capture.payload == {"primary_method_id": "rmst"}
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-2]["event_type"] == "submission"
    assert events[-2]["tool_name"] == "submit_phase_analysis_file"
    assert events[-1]["event_type"] == "step_terminal"


def test_agent_loop_rejects_duplicate_fields_in_file_submission(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        agent,
        "DockerPythonSession",
        lambda **_: _FileSession('{"primary_method_id":"rmst","primary_method_id":"cox"}'),
    )
    loop = agent.AgentLoop(
        provider=_FileSubmissionProvider(),
        workdir=tmp_path / "work",
        system_prompt="system",
        max_turns_per_step=1,
        event_log_path=tmp_path / "events.jsonl",
        program_id="program-1",
        scenario_id="01",
        objective_id="benefit_risk",
    )
    loop.begin_step(phase_id="phase2", step_id="trial_analysis")
    loop.append_user_message("analyse")

    with pytest.raises(RuntimeError, match="did not emit"):
        loop.run_until_submit(
            tools=agent.tools_for_phase_analysis(),
            submit_tool_names={"submit_phase_analysis", "submit_phase_analysis_file"},
        )
    assert "Duplicate JSON field" in loop.messages[-1]["content"]
    loop.close()


def test_agent_loop_rejects_events_without_step_context(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(agent, "DockerPythonSession", lambda **_: _Session())
    loop = agent.AgentLoop(
        provider=_Provider(),
        workdir=tmp_path / "work",
        system_prompt="system",
        event_log_path=tmp_path / "events.jsonl",
    )

    try:
        loop.append_user_message("unscoped")
    except RuntimeError as exc:
        assert "trace context" in str(exc)
    else:
        raise AssertionError("Unscoped runtime events must fail.")
    finally:
        loop.close()


def test_agent_loop_rejects_non_object_submission_payload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(agent, "DockerPythonSession", lambda **_: _Session())
    loop = agent.AgentLoop(
        provider=_ScalarSubmissionProvider(),
        workdir=tmp_path / "work",
        system_prompt="system",
        max_turns_per_step=1,
        event_log_path=tmp_path / "events.jsonl",
        program_id="program-1",
        scenario_id="01",
        objective_id="benefit_risk",
    )
    loop.begin_step(phase_id="phase2", step_id="trial_analysis")
    loop.append_user_message("analyse")

    with pytest.raises(RuntimeError, match="did not emit"):
        loop.run_until_submit(tools=[], submit_tool_names={"submit_phase_analysis"})
    assert loop.messages[-1]["content"].startswith("Tool input rejected:")
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-3]["status"] == "invalid"
    assert events[-2]["event_type"] == "tool_result"
    assert events[-2]["status"] == "invalid"
    assert events[-2]["tool_name"] == "submit_phase_analysis"
    assert events[-1]["event_type"] == "step_terminal"
    assert events[-1]["terminal_status"] == "failed"
    loop.close()


def test_agent_loop_records_local_execution_result(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(agent, "DockerPythonSession", lambda **_: _Session())
    loop = agent.AgentLoop(
        provider=_ToolThenSubmitProvider(),
        workdir=tmp_path / "work",
        system_prompt="system",
        event_log_path=tmp_path / "events.jsonl",
        program_id="program-1",
        scenario_id="01",
        objective_id="benefit_risk",
    )
    loop.begin_step(phase_id="phase2", step_id="trial_analysis")
    loop.append_user_message("analyse")
    loop.run_until_submit(tools=[], submit_tool_names={"submit_phase_analysis"})
    loop.close()

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    result_event = next(event for event in events if event["event_type"] == "tool_result")
    assert result_event["tool_name"] == "execute_code"
    assert result_event["execution_status"] == "success"
    assert result_event["elapsed_seconds"] == 0.1
    assert result_event["output_truncated"] is False


def _pending_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    agent.TrialDevContinuationCheckpointV1,
    Path,
    Path,
    agent.AgentLoop,
]:
    monkeypatch.setattr(
        agent,
        "DockerPythonSession",
        lambda **kwargs: _Session(scratch_root=Path(kwargs["cwd"]) / "scratch"),
    )
    custody_root = tmp_path / "program"
    workdir = custody_root / "agent_workdir"
    conversation_path = custody_root / "conversation.json"
    current_state = custody_root / "states" / "state_after_phase1.json"
    current_state.parent.mkdir(parents=True)
    current_state.write_text('{"current_phase_id":"phase2"}\n', encoding="utf-8")
    loop = agent.AgentLoop(
        provider=_Provider(),
        workdir=workdir,
        system_prompt="system",
        max_turns_per_step=4,
        tool_choice="required",
        conversation_log_path=conversation_path,
        program_id="program-1",
        scenario_id="01",
        objective_id="benefit_risk",
    )
    loop.begin_step(phase_id="phase2", step_id="trial_analysis")
    loop.append_user_message("analyse phase2")
    tool_response = LLMResponse(
        tool_calls=[
            ToolCall(
                id="code-1",
                name="execute_code",
                arguments='{"code":"print(1)"}',
            )
        ]
    )
    loop.messages.append(loop._compose_assistant_message(tool_response))
    loop.step_turns_used = 1
    loop.append_tool_reply(
        "code-1",
        "1",
        tool_name="execute_code",
        execution=CodeExecutionResultV1(
            status="success",
            output="1",
            elapsed_seconds=0.1,
        ),
    )
    scratch_note = workdir / "scratch" / "phase2.txt"
    scratch_note.parent.mkdir(parents=True, exist_ok=True)
    scratch_note.write_text("durable evidence", encoding="utf-8")
    loop._persist_log()
    checkpoint = loop.create_checkpoint(
        custody_root=custody_root,
        current_state_path=current_state,
        materialization_usage=TrialDevMaterializationUsageV1(materialize_calls_by_phase={"phase1": 1}),
        completed_phase_summaries=[
            TrialDevCheckpointPhaseSummaryV1(
                phase_id="phase1",
                advance=True,
                primary_effect={"estimate": 0.25, "effect_scale": "risk_difference"},
                safety_estimate={"estimate": 0.02},
            )
        ],
        violations=[
            TrialDevCheckpointViolationV1(
                phase_id="phase1",
                kind="materialize_rejection",
                error="insufficient_enrollment_window_support",
                artifact_relative_path="materialization_attempts/phase1/attempt-1",
            )
        ],
    )
    return checkpoint, custody_root, conversation_path, loop


def test_agent_loop_restores_exact_pending_step_with_monotone_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, custody_root, conversation_path, original = _pending_checkpoint(tmp_path, monkeypatch)
    assert checkpoint.payload.materialization_usage.materialize_calls_by_phase == {"phase1": 1}
    assert checkpoint.payload.completed_phase_summaries[0].phase_id == "phase1"
    assert checkpoint.payload.completed_phase_summaries[0].primary_effect == {
        "estimate": 0.25,
        "effect_scale": "risk_difference",
    }
    assert checkpoint.payload.violations[0].kind == "materialize_rejection"
    assert checkpoint.payload.violations[0].artifact_relative_path == ("materialization_attempts/phase1/attempt-1")
    original.session.close()  # type: ignore[union-attr]
    original.session = None

    restored = agent.AgentLoop.restore_from_checkpoint(
        checkpoint,
        provider=_Provider(),
        custody_root=custody_root,
        system_prompt="system",
        conversation_log_path=conversation_path,
    )
    assert restored.tool_choice == "required"

    assert restored.trace_phase_id == "phase2"
    assert restored.trace_step_id == "trial_analysis"
    assert restored.step_turns_used == 1
    assert restored.messages == [message.to_message() for message in checkpoint.payload.conversation]
    capture = restored.run_until_submit(
        tools=[],
        submit_tool_names={"submit_phase_analysis"},
    )
    assert capture.name == "submit_phase_analysis"
    assert restored.step_turns_used == 2
    assert (restored.workdir / "scratch" / "phase2.txt").read_text(encoding="utf-8") == "durable evidence"
    restored.close()


def test_agent_loop_can_reset_context_at_pre_response_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, custody_root, conversation_path, original = _pending_checkpoint(tmp_path, monkeypatch)
    payload = checkpoint.payload.model_dump(mode="json")
    prompt_index = int(payload["pending_step"]["active_prompt_index"])
    payload["conversation"] = payload["conversation"][: prompt_index + 1]
    payload["pending_step"]["turns_used"] = 0
    reset_checkpoint = TrialDevContinuationCheckpointV1.create(TrialDevContinuationPayloadV1.model_validate(payload))
    conversation_path.write_text(
        json.dumps(
            [message.to_message() for message in reset_checkpoint.payload.conversation],
            indent=2,
        ),
        encoding="utf-8",
    )
    reset_conversation = custody_root / "checkpoint_conversation.json"
    original.close()

    restored = agent.AgentLoop.restore_from_checkpoint(
        reset_checkpoint,
        provider=_Provider(),
        custody_root=custody_root,
        system_prompt="system",
        conversation_log_path=conversation_path,
        context_mode="active_step_only",
        reset_conversation_log_path=reset_conversation,
        reset_event_log_path=custody_root / "checkpoint_events.jsonl",
        reset_provider_log_path=custody_root / "checkpoint_provider_responses.jsonl",
        checkpoint_deadline_monotonic=1.0e12,
    )

    assert restored.messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "analyse phase2"},
    ]
    assert restored.active_prompt_index == 1
    assert restored.step_turns_used == 0
    assert restored.event_index == 0
    assert restored.runtime_deadline.monotonic == 1.0e12
    assert json.loads(reset_conversation.read_text(encoding="utf-8")) == restored.messages
    assert (restored.workdir / "scratch" / "phase2.txt").read_text(encoding="utf-8") == "durable evidence"
    restored.close()


def test_active_step_only_restore_is_provider_neutral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, custody_root, conversation_path, original = _pending_checkpoint(tmp_path, monkeypatch)
    payload = checkpoint.payload.model_dump(mode="json")
    prompt_index = int(payload["pending_step"]["active_prompt_index"])
    payload["conversation"] = payload["conversation"][: prompt_index + 1]
    payload["pending_step"]["turns_used"] = 0
    reset_checkpoint = TrialDevContinuationCheckpointV1.create(TrialDevContinuationPayloadV1.model_validate(payload))
    conversation_path.write_text(
        json.dumps([message.to_message() for message in reset_checkpoint.payload.conversation]),
        encoding="utf-8",
    )
    different_provider = _Provider()
    different_provider.model = "evaluated-model"
    different_provider.telemetry_route = "evaluated-route"
    original.close()

    restored = agent.AgentLoop.restore_from_checkpoint(
        reset_checkpoint,
        provider=different_provider,
        custody_root=custody_root,
        system_prompt="system",
        conversation_log_path=conversation_path,
        context_mode="active_step_only",
        reset_conversation_log_path=custody_root / "checkpoint_conversation.json",
    )

    assert restored.provider is different_provider
    assert restored.messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "analyse phase2"},
    ]
    restored.close()


def test_agent_loop_rejects_context_reset_after_model_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, custody_root, conversation_path, original = _pending_checkpoint(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="pre-response checkpoint"):
        agent.AgentLoop.restore_from_checkpoint(
            checkpoint,
            provider=_Provider(),
            custody_root=custody_root,
            system_prompt="system",
            conversation_log_path=conversation_path,
            context_mode="active_step_only",
            reset_conversation_log_path=custody_root / "checkpoint_conversation.json",
        )
    original.close()


def test_checkpoint_rejects_nonmonotone_turn_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, _, _, original = _pending_checkpoint(tmp_path, monkeypatch)
    payload = checkpoint.payload.model_dump(mode="json")
    payload["pending_step"]["turns_used"] = 0

    with pytest.raises(ValidationError, match="turns_used"):
        TrialDevContinuationPayloadV1.model_validate(payload)
    original.session.close()  # type: ignore[union-attr]


def test_checkpoint_rejects_envelope_checksum_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, _, _, original = _pending_checkpoint(tmp_path, monkeypatch)
    envelope = checkpoint.model_dump(mode="json")
    envelope["payload"]["objective_id"] = "tampered"

    with pytest.raises(ValidationError, match="payload checksum mismatch"):
        TrialDevContinuationCheckpointV1.model_validate(envelope)
    original.session.close()  # type: ignore[union-attr]


def test_runner_checkpoint_requires_authoritative_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, _, _, original = _pending_checkpoint(tmp_path, monkeypatch)

    with pytest.raises(ValidationError, match="run_identity_sha256"):
        TrialDevRunCheckpointPayloadV1.model_validate(
            {
                "sequence": 0,
                "pending_operation": "observational_review",
                "continuation": checkpoint.model_dump(mode="json"),
            }
        )
    original.session.close()  # type: ignore[union-attr]


def test_runner_checkpoint_chain_rejects_identity_and_predecessor_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    continuation, custody_root, _, original = _pending_checkpoint(tmp_path, monkeypatch)
    checkpoint_dir = custody_root / "checkpoints"
    checkpoint_dir.mkdir()
    first = TrialDevRunCheckpointV1.create(
        TrialDevRunCheckpointPayloadV1(
            sequence=0,
            run_identity_sha256="a" * 64,
            pending_operation="observational_review",
            continuation=continuation,
        )
    )
    (checkpoint_dir / "00000000.json").write_text(
        first.model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run identity mismatch"):
        runner._load_checkpoint_chain(custody_root, run_identity_sha256="b" * 64)

    second = TrialDevRunCheckpointV1.create(
        TrialDevRunCheckpointPayloadV1(
            sequence=1,
            previous_checkpoint_sha256="c" * 64,
            run_identity_sha256="a" * 64,
            pending_operation="observational_review",
            continuation=continuation,
        )
    )
    (checkpoint_dir / "00000001.json").write_text(
        second.model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="predecessor hash mismatch"):
        runner._load_checkpoint_chain(custody_root, run_identity_sha256="a" * 64)
    original.session.close()  # type: ignore[union-attr]


@pytest.mark.parametrize("artifact", ["current_state", "scratch"])
def test_checkpoint_restore_rejects_artifact_checksum_drift(
    artifact: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, custody_root, conversation_path, original = _pending_checkpoint(tmp_path, monkeypatch)
    if artifact == "current_state":
        (custody_root / checkpoint.payload.current_state.relative_path).write_text(
            '{"current_phase_id":"phase3"}\n',
            encoding="utf-8",
        )
    else:
        (custody_root / checkpoint.payload.scratch_workspace.relative_path / "phase2.txt").write_text(
            "mutated",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        agent.AgentLoop.restore_from_checkpoint(
            checkpoint,
            provider=_Provider(),
            custody_root=custody_root,
            system_prompt="system",
            conversation_log_path=conversation_path,
        )
    original.session.close()  # type: ignore[union-attr]


def test_checkpoint_restore_rejects_provider_or_conversation_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, custody_root, conversation_path, original = _pending_checkpoint(tmp_path, monkeypatch)
    wrong_provider = _Provider()
    wrong_provider.model = "other-model"
    with pytest.raises(ValueError, match="provider identity mismatch"):
        agent.AgentLoop.restore_from_checkpoint(
            checkpoint,
            provider=wrong_provider,
            custody_root=custody_root,
            system_prompt="system",
            conversation_log_path=conversation_path,
        )

    conversation = json.loads(conversation_path.read_text(encoding="utf-8"))
    conversation.append({"role": "user", "content": "advanced"})
    conversation_path.write_text(json.dumps(conversation), encoding="utf-8")
    with pytest.raises(ValueError, match="conversation log has advanced"):
        agent.AgentLoop.restore_from_checkpoint(
            checkpoint,
            provider=_Provider(),
            custody_root=custody_root,
            system_prompt="system",
            conversation_log_path=conversation_path,
        )
    original.session.close()  # type: ignore[union-attr]


def test_checkpoint_restore_rejects_advanced_provider_turn_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, custody_root, conversation_path, original = _pending_checkpoint(tmp_path, monkeypatch)
    provider_path = custody_root / "provider_responses.jsonl"
    for turn in (1, 2):
        handle = start_provider_request_v1(
            path=provider_path,
            benchmark="trialdev",
            unit_id="program-1",
            phase_id="phase2",
            step_id="trial_analysis",
            turn_index=turn,
            requested_model="test",
            provider_route="test",
        )
        succeed_provider_request_v1(
            handle,
            elapsed_seconds=0.1,
            response=LLMResponse(content="ok"),
        )

    with pytest.raises(ValueError, match="turn counts have advanced"):
        agent.AgentLoop.restore_from_checkpoint(
            checkpoint,
            provider=_Provider(),
            custody_root=custody_root,
            system_prompt="system",
            conversation_log_path=conversation_path,
            provider_log_path=provider_path,
        )
    original.session.close()  # type: ignore[union-attr]


def test_checkpoint_restore_rejects_pending_step_with_terminal_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, custody_root, conversation_path, original = _pending_checkpoint(tmp_path, monkeypatch)
    event_path = custody_root / "events.jsonl"
    events = [
        {
            "benchmark": "trialdev",
            "program_id": "program-1",
            "scenario_id": "01",
            "objective_id": "benefit_risk",
            "event_index": 0,
            "phase_id": "phase2",
            "step_id": "trial_analysis",
            "event_type": "step_started",
            "status": "observed",
        },
        {
            "benchmark": "trialdev",
            "program_id": "program-1",
            "scenario_id": "01",
            "objective_id": "benefit_risk",
            "event_index": 1,
            "phase_id": "phase2",
            "step_id": "trial_analysis",
            "event_type": "step_terminal",
            "status": "observed",
            "terminal_status": "failed",
            "failure_type": "worker_exit",
        },
    ]
    for event in events:
        event_index = event["event_index"]
        event["event_id"] = f"trialdev:program-1:{event_index:06d}"
        event["timestamp"] = "2026-01-01T00:00:00Z"
        event["source_artifact_path"] = conversation_path.as_posix()
        event["source_payload_sha256"] = "a" * 64
    event_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    payload = checkpoint.payload.model_dump(mode="json")
    payload["pending_step"]["next_event_index"] = 2
    advanced = TrialDevContinuationCheckpointV1.create(TrialDevContinuationPayloadV1.model_validate(payload))

    with pytest.raises(ValueError, match="already has a terminal"):
        agent.AgentLoop.restore_from_checkpoint(
            advanced,
            provider=_Provider(),
            custody_root=custody_root,
            system_prompt="system",
            conversation_log_path=conversation_path,
            event_log_path=event_path,
        )
    original.session.close()  # type: ignore[union-attr]
