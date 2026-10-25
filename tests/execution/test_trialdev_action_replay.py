"""Authored regrading must reconstruct actions, not merely trust a valid ledger."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from trialagentbench.io import canonical_payload_sha256, read_json_model, write_json_model
from trialagentbench.trialdev.actions import TrialDevActionSession
from trialagentbench.trialdev.errors import TrialDevActionRejectedError
from trialagentbench.trialdev.regrade import replay_trialdev_programme_actions
from trialagentbench.trialdev.release import TrialDevAuthoredProgrammeConfiguration
from trialagentbench.trialdev.resources import TrialDevBudgetLedger
from trialagentbench.trialdev.task_agent import TrialDevAgentCheckpoint


def _history(root: Path) -> tuple[Mock, TrialDevAgentCheckpoint]:
    configuration = Mock(
        spec=TrialDevAuthoredProgrammeConfiguration,
        budget_usd=15_000_000,
        pool_manifest_sha256="a" * 64,
        recruitment_seed=7,
    )
    budget = root / "private/budget.json"
    write_json_model(
        budget,
        TrialDevBudgetLedger(
            budget_usd=configuration.budget_usd,
            pool_manifest_sha256=configuration.pool_manifest_sha256,
            recruitment_seed=configuration.recruitment_seed,
        ),
    )
    session = TrialDevActionSession(
        pool_root=root / "pool",
        budget_path=budget,
        evidence_root=root / "workspace/evidence",
        commitment_root=root / "private/commitments",
        recruitment_seed=7,
    )
    checkpoint = TrialDevAgentCheckpoint(
        identity="test",
        messages=[],
        phase_id="development",
        step_id="programme",
        active_prompt_index=0,
        event_index=0,
        report_index=0,
        turns_used=3,
        pending_calls=(),
        pending_message_index=0,
        pending_tool_index=0,
        executed_code=[],
        execution_outputs=[],
        reporting_context=None,
    )
    for index, (name, arguments) in enumerate(
        (
            ("wait_until", {"programme_day": 90}),
            ("wait_until", {"programme_day": 50}),
            ("stop_development", {}),
        )
    ):
        call_id = f"call-{index}"
        identity = "action-" + canonical_payload_sha256({"step": "programme", "call": call_id})[:24]
        checkpoint.messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
            }
        )
        try:
            session.execute(session.decode(name, arguments).action, action_id=identity)
        except TrialDevActionRejectedError:
            assert index == 1
            reply = "Rejected: cannot move the programme clock backwards."
        else:
            reply = "Completed"
        checkpoint.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply})
    write_json_model(root / "private/agent_state.json", checkpoint)
    return configuration, checkpoint


def test_action_replay_preserves_rejected_requests_and_terminal_time(tmp_path: Path) -> None:
    configuration, _ = _history(tmp_path)
    replay_trialdev_programme_actions(task_root=tmp_path, pool_root=tmp_path / "pool", configuration=configuration)
    assert read_json_model(TrialDevBudgetLedger, tmp_path / "private/budget.json").resource_day == 90
    assert len(list((tmp_path / "private/commitments").glob("*.request.json"))) == 3


@pytest.mark.parametrize("defect", ["omit_action", "reorder_actions", "change_argument", "change_ledger"])
def test_action_replay_rejects_valid_looking_but_unreproducible_history(tmp_path: Path, defect: str) -> None:
    configuration, checkpoint = _history(tmp_path)
    if defect == "omit_action":
        checkpoint.messages.pop(0)
    elif defect == "reorder_actions":
        checkpoint.messages[0], checkpoint.messages[4] = checkpoint.messages[4], checkpoint.messages[0]
    elif defect == "change_argument":
        checkpoint.messages[0] = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-0",
                    "type": "function",
                    "function": {"name": "wait_until", "arguments": '{"programme_day":91}'},
                }
            ],
        }
    else:
        ledger_path = tmp_path / "private/budget.json"
        ledger = read_json_model(TrialDevBudgetLedger, ledger_path)
        assert ledger.terminal is not None
        write_json_model(
            ledger_path,
            ledger.model_copy(
                update={"resource_day": 100, "terminal": ledger.terminal.model_copy(update={"programme_day": 100})}
            ),
        )
    write_json_model(tmp_path / "private/agent_state.json", checkpoint)
    with pytest.raises(ValueError):
        replay_trialdev_programme_actions(task_root=tmp_path, pool_root=tmp_path / "pool", configuration=configuration)


def test_action_replay_rejects_erased_requests_even_with_a_matching_forged_ledger(tmp_path: Path) -> None:
    configuration, _ = _history(tmp_path)
    for call_id in ("call-0", "call-1"):
        identity = "action-" + canonical_payload_sha256({"step": "programme", "call": call_id})[:24]
        path = tmp_path / "private/commitments" / f"{identity}.request.json"
        path.rename(tmp_path / path.name)
    ledger_path = tmp_path / "private/budget.json"
    ledger = read_json_model(TrialDevBudgetLedger, ledger_path)
    assert ledger.terminal is not None
    write_json_model(
        ledger_path,
        ledger.model_copy(
            update={
                "resource_day": 0,
                "terminal": ledger.terminal.model_copy(update={"programme_day": 0}),
            }
        ),
    )
    with pytest.raises(ValueError, match="Acknowledged programme action lacks its committed request"):
        replay_trialdev_programme_actions(task_root=tmp_path, pool_root=tmp_path / "pool", configuration=configuration)


@pytest.mark.parametrize(
    "malformed_arguments",
    [
        '{"cost":0}',
        '{"accept_incomplete":true,"accept_incomplete":true}',
        '{"programme_day":10}',
        "[]",
        "{",
    ],
)
def test_action_replay_preserves_malformed_and_unexecuted_calls(tmp_path: Path, malformed_arguments: str) -> None:
    configuration, checkpoint = _history(tmp_path)
    checkpoint.messages.extend(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "malformed", "function": {"name": "stop_development", "arguments": malformed_arguments}},
                    {"id": "unexecuted", "function": {"name": "stop_development", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "malformed", "content": "Tool input rejected: invalid action"},
        ]
    )
    write_json_model(tmp_path / "private/agent_state.json", checkpoint)
    replay_trialdev_programme_actions(task_root=tmp_path, pool_root=tmp_path / "pool", configuration=configuration)


@pytest.mark.parametrize("applied", [False, True])
@pytest.mark.parametrize("corrupt_queue", [False, True])
def test_action_replay_preserves_pending_commit_boundary_and_rejects_queue_drift(
    tmp_path: Path, applied: bool, corrupt_queue: bool
) -> None:
    from trialagentbench.ports import ToolCall

    configuration, checkpoint = _history(tmp_path)
    checkpoint.messages.pop()  # Stop has no delivered tool reply yet.
    pending = (
        ToolCall(id="call-2", name="wait_until", arguments='{"programme_day":90}')
        if corrupt_queue
        else ToolCall(id="call-2", name="stop_development", arguments="{}")
    )
    checkpoint = checkpoint.model_copy(
        update={
            "pending_calls": (pending,),
            "pending_message_index": 4,
            "pending_tool_index": 0,
        }
    )
    if not applied:
        ledger_path = tmp_path / "private/budget.json"
        ledger = read_json_model(TrialDevBudgetLedger, ledger_path)
        write_json_model(ledger_path, ledger.model_copy(update={"terminal": None}))
    write_json_model(tmp_path / "private/agent_state.json", checkpoint)
    retained = {path: path.read_bytes() for path in (tmp_path / "private").rglob("*.json")}
    if corrupt_queue:
        with pytest.raises(ValueError, match="Pending programme action differs"):
            replay_trialdev_programme_actions(
                task_root=tmp_path, pool_root=tmp_path / "pool", configuration=configuration
            )
    else:
        replay_trialdev_programme_actions(task_root=tmp_path, pool_root=tmp_path / "pool", configuration=configuration)
    assert all(path.read_bytes() == content for path, content in retained.items())


@pytest.mark.parametrize("rejected_before_commit", [False, True])
def test_action_replay_distinguishes_result_rejection_from_erased_action(
    tmp_path: Path, rejected_before_commit: bool
) -> None:
    configuration, checkpoint = _history(tmp_path)
    checkpoint.messages[:0] = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "cutoff", "function": {"name": "stop_development", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "cutoff",
            "content": (
                "Tool input rejected: Invalid scratch/estimates.csv: unknown numerical target: bad"
                if rejected_before_commit
                else "Completed"
            ),
        },
    ]
    write_json_model(tmp_path / "private/agent_state.json", checkpoint)
    if rejected_before_commit:
        replay_trialdev_programme_actions(task_root=tmp_path, pool_root=tmp_path / "pool", configuration=configuration)
    else:
        with pytest.raises(ValueError, match="Acknowledged programme action lacks its committed request"):
            replay_trialdev_programme_actions(
                task_root=tmp_path, pool_root=tmp_path / "pool", configuration=configuration
            )
