"""Agent-time accounting must survive interruption without buying extra time."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from trialagentbench_verifier.numerical import NumericalReportingContext

from trialagentbench.io import read_json_model
from trialagentbench.ports import CodeExecutionResultV1, LLMResponse, ToolCall
from trialagentbench.trialdev import task_agent as owner
from trialagentbench.util import runtime_context
from trialagentbench.util.agent_runtime import assistant_message_v1
from trialagentbench.util.runtime_context import RuntimeDeadline, RuntimeDeadlineExceeded


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(runtime_context.time, "monotonic", lambda: value.now)
    return value


def test_nested_harness_work_and_offline_downtime_do_not_refill_or_spend_budget(clock):
    deadline = RuntimeDeadline.after(100, label="test")
    clock.advance(7)  # Provider request.
    with deadline.harness_work():
        clock.advance(300)
        with deadline.harness_work():
            clock.advance(400)
            assert deadline.snapshot_remaining() == 93
    clock.advance(6)  # Interactive tool.
    saved = deadline.snapshot_remaining()
    assert saved == 87
    clock.advance(8_000)  # No running process.
    resumed = RuntimeDeadline.after(100, label="test")
    resumed.restore_remaining(saved)
    assert resumed.remaining() == 87
    clock.advance(87)
    assert resumed.snapshot_remaining() == 0
    with pytest.raises(RuntimeDeadlineExceeded):
        resumed.remaining()


@pytest.mark.parametrize("seconds", [-1, float("inf"), float("nan")])
def test_saved_time_rejects_nonfinite_or_negative_values(seconds):
    with pytest.raises(ValidationError):
        owner.TrialDevAgentTimeCheckpoint(remaining_seconds=seconds)
    with pytest.raises(ValueError):
        RuntimeDeadline.after(100, label="test").restore_remaining(seconds)


def _arguments(tmp_path, monkeypatch, clock):
    monkeypatch.setattr(owner, "DockerCodeSession", lambda **kwargs: Mock(image_id="image-sha"))
    return dict(
        provider=Mock(),
        workdir=tmp_path / "workspace",
        system_prompt="Analyse the evidence.",
        program_id="programme",
        scenario_id="case",
        objective_id="evidence_analysis",
        max_turns_per_step=5,
        deadline_monotonic=clock.now + 100,
        agent_time_budget_seconds=100,
        reporting_context=NumericalReportingContext(
            checkpoint_id="initial", target_ids=("effect",), evidence_artifacts=()
        ),
    )


def test_native_resume_retains_charged_time_turns_and_model_visible_allowance(tmp_path, monkeypatch, clock):
    arguments = _arguments(tmp_path, monkeypatch, clock)
    agent = owner.TrialDevTaskAgent(**arguments)
    agent.begin_step(phase_id="analysis", step_id="initial")
    agent.append_user_message("Estimate effect.")
    notices = []
    calls = [
        ToolCall(id="code", name="execute_code", arguments='{"language":"python","code":"pass"}'),
        ToolCall(id="check", name="check_analysis", arguments="{}"),
    ]

    def provider_turn(**kwargs):
        notices.append(agent.messages[-1]["content"])
        if kwargs["turn"] == 2:
            clock.advance(7)
            raise OSError("provider interrupted")
        clock.advance(5)
        response = LLMResponse(content=None, tool_calls=calls, usage={})
        return SimpleNamespace(response=response, message=assistant_message_v1(response))

    def execute(call):
        duration = 12 if call.name == "execute_code" else 500
        clock.advance(duration)
        # A checkpoint written inside excluded work must use the paused clock.
        agent._persist_checkpoint()
        return CodeExecutionResultV1(status="success", output="done", elapsed_seconds=duration)

    monkeypatch.setattr(agent, "_provider_turn", provider_turn)
    monkeypatch.setattr(agent, "_execute_checkpointed_tool", execute)
    with pytest.raises(OSError, match="provider interrupted"):
        agent.run_until_report()
    agent.close()
    assert "100 seconds total; 100 seconds remaining" in notices[0]
    assert "100 seconds total; 83 seconds remaining" in notices[1]
    assert "Model requests and agent-run tools consume this time" in notices[0]
    assert "do not" in notices[0] and "simulated programme days" in notices[0]
    checkpoint = read_json_model(owner.TrialDevAgentCheckpoint, agent._checkpoint_path())
    assert checkpoint.agent_time.remaining_seconds == 76
    assert checkpoint.agent_time.charged_operation_in_progress is False
    assert checkpoint.turns_used == 1 and checkpoint.pending_calls == ()

    clock.advance(8_000)
    resumed = owner.TrialDevTaskAgent(**(arguments | {"deadline_monotonic": clock.now + 100}), resume=True)
    assert resumed.deadline.remaining() == 76
    assert resumed.turns_used == 1

    def fail_again(**kwargs):
        assert kwargs["turn"] == 2
        assert kwargs["timeout_seconds"] == 76
        assert "76 seconds remaining" in resumed.messages[-1]["content"]
        clock.advance(3)
        raise OSError("second interruption")

    monkeypatch.setattr(resumed, "_provider_turn", fail_again)
    with pytest.raises(OSError, match="second interruption"):
        resumed.run_until_report()
    resumed.close()
    assert read_json_model(owner.TrialDevAgentCheckpoint, agent._checkpoint_path()).agent_time.remaining_seconds == 73


@pytest.mark.parametrize("defect", ["legacy", "unclean", "changed_limit"])
def test_resume_refuses_unknown_consumed_time_or_changed_limit(tmp_path, monkeypatch, clock, defect):
    arguments = _arguments(tmp_path, monkeypatch, clock)
    agent = owner.TrialDevTaskAgent(**arguments)
    agent.begin_step(phase_id="analysis", step_id="initial")
    agent.append_user_message("Estimate effect.")
    agent.close()
    path = agent._checkpoint_path()
    payload = json.loads(path.read_text())
    if defect == "legacy":
        del payload["agent_time"]
        error = "no saved agent-time budget"
    elif defect == "unclean":
        payload["agent_time"]["charged_operation_in_progress"] = True
        error = "unclean interruption"
    else:
        arguments["agent_time_budget_seconds"] = 200
        error = "execution limits"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=error):
        owner.TrialDevTaskAgent(**arguments, resume=True)


def test_exhausted_checkpoint_cannot_request_another_turn(tmp_path, monkeypatch, clock):
    arguments = _arguments(tmp_path, monkeypatch, clock)
    agent = owner.TrialDevTaskAgent(**arguments)
    agent.begin_step(phase_id="analysis", step_id="initial")
    agent.append_user_message("Estimate effect.")
    clock.advance(101)
    agent.close()
    clock.advance(1_000)
    resumed = owner.TrialDevTaskAgent(**(arguments | {"deadline_monotonic": clock.now + 100}), resume=True)
    provider = Mock(side_effect=AssertionError("expired task must not call provider"))
    monkeypatch.setattr(resumed, "_provider_turn", provider)
    with pytest.raises(RuntimeDeadlineExceeded):
        resumed.run_until_report()
    resumed.close()
    provider.assert_not_called()
