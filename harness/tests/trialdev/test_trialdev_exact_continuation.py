"""Exact-once runner continuation tests for TrialDev."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import pytest
from pydantic import ValidationError

from trialagentbench_harness.adapters.trialdev_share import TrialDevelopmentRequestV1
from trialagentbench_harness.contracts.trialdev.run_checkpoint import (
    TrialDevRunCheckpointPhaseV1,
)
from trialagentbench_harness.contracts.trialdev.runtime_checkpoint import TrialDevCheckpointMessageV1
from trialagentbench_harness.trialdev import runner
from trialagentbench_harness.trialdev.runner import RunOptions
from trialagentbench_harness.trialdev.schema import MaterializationUsage, PhaseAttempt, Program


def _request() -> TrialDevelopmentRequestV1:
    return TrialDevelopmentRequestV1(
        scenario_id="s01",
        phase_id="phase2",
        candidate_drug_ids=("drug_a",),
        target_sample_size=80,
        endpoint_id="E1",
        follow_up_days=90,
        enrollment_window_days=42,
        site_count_budget=8,
        allocation_ratio="1:1",
        design_cell_id="trialdev.phase2.fixed_final_operating_characteristics.v1",
        treatment_discontinuation_strategy="treatment_policy",
        interim_policy="fixed_final",
        site_strategy="high_enrolling",
        selection_objective="benefit_risk",
    )


def test_checkpoint_log_custody_separates_exact_and_reset_context(tmp_path: Path) -> None:
    assert runner._checkpoint_reset_log_paths(tmp_path, context_mode="exact") == (
        None,
        None,
        None,
    )
    assert runner._checkpoint_reset_log_paths(tmp_path, context_mode="active_step_only") == (
        tmp_path / "checkpoint_conversation.json",
        tmp_path / "checkpoint_events.jsonl",
        tmp_path / "checkpoint_provider_responses.jsonl",
    )


def test_checkpoint_message_round_trips_responses_provider_state() -> None:
    message = TrialDevCheckpointMessageV1.model_validate(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "execute_code", "arguments": '{"code":"print(1)"}'},
                }
            ],
            "provider_state": [
                {
                    "id": "reasoning-1",
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "opaque-reasoning-state",
                },
                {
                    "id": "function-1",
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "execute_code",
                    "arguments": '{"code":"print(1)"}',
                    "status": "completed",
                },
            ],
        }
    )

    restored = message.to_message()

    assert restored["provider_state"][0]["encrypted_content"] == "opaque-reasoning-state"
    assert restored["tool_calls"][0]["id"] == "call-1"


def test_empty_checkpoint_provider_state_is_absent_from_legacy_payload() -> None:
    message = TrialDevCheckpointMessageV1(role="user", content="Continue.")

    assert "provider_state" not in message.model_dump(mode="json")


def test_kill_restart_does_not_repeat_request_or_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restart at analysis resumes after the exactly-once materialization."""

    phase_dir = tmp_path / "work" / "phase_phase2"
    phase_dir.mkdir(parents=True)
    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    request = _request()
    provider_steps: list[str] = []
    materializations = 0
    captured: PhaseAttempt | None = None
    interrupt = True

    def drive_request(**kwargs: Any) -> tuple[TrialDevelopmentRequestV1, Path]:
        del kwargs
        provider_steps.append("request")
        path = phase_dir / "request.json"
        path.write_text(request.model_dump_json(), encoding="utf-8")
        return request, path

    def materialize_phase(**kwargs: Any) -> None:
        nonlocal materializations
        materializations += 1
        Path(kwargs["out_dir"]).mkdir(parents=True)

    def summarize_output(path: Path, **kwargs: Any) -> dict[str, Any]:
        del path, kwargs
        return {
            "request_checksum": "a" * 64,
            "trial_output_checksum": "b" * 64,
            "trial_output_relpath": "phase_phase2/trial_output",
            "n_participants": 80,
        }

    def drive_analysis(
        *,
        step_checkpoint: Any,
        start_step: bool,
        **kwargs: Any,
    ) -> Path:
        nonlocal interrupt
        del kwargs
        assert start_step is interrupt
        step_checkpoint()
        if interrupt:
            interrupt = False
            raise SystemExit("simulated kill after durable materialization")
        provider_steps.append("analysis")
        path = phase_dir / "analysis_submission.json"
        path.write_text("{}", encoding="utf-8")
        return path

    def drive_decision(**kwargs: Any) -> tuple[Path, str, bool, str]:
        del kwargs
        provider_steps.append("decision")
        path = phase_dir / "decision_submission.json"
        path.write_text("{}", encoding="utf-8")
        return path, "declare_success", False, "drug_a"

    def checkpoint(operation: str, attempt: PhaseAttempt) -> None:
        nonlocal captured
        if operation == "phase_analysis":
            custody = TrialDevRunCheckpointPhaseV1.model_validate_json(
                runner._checkpoint_phase(
                    attempt,
                    program_dir=tmp_path,
                ).model_dump_json()
            )
            captured = runner._restore_phase_attempt(custody, program_dir=tmp_path)

    monkeypatch.setattr(runner, "_drive_phase_request", drive_request)
    monkeypatch.setattr(runner.trialdev_upstream, "materialize_phase", materialize_phase)
    monkeypatch.setattr(runner.bridge, "summarize_program_state_for_agent", lambda path: {})
    monkeypatch.setattr(runner.bridge, "summarize_trial_output_for_agent", summarize_output)
    monkeypatch.setattr(runner.prompts, "get_phase_module", lambda *args: {})
    monkeypatch.setattr(runner, "_drive_phase_analysis", drive_analysis)
    monkeypatch.setattr(runner, "_drive_phase_decision", drive_decision)
    monkeypatch.setattr(
        "trialagentbench_harness.adapters.trialdev_share.sha256_file_hex",
        lambda path: "c" * 64,
    )

    program = Program(
        program_id="s01__benefit_risk",
        scenario_id="s01",
        objective_id="benefit_risk",
        items_by_phase={},
    )
    loop = cast(runner.agent_mod.AgentLoop, object())
    usage = MaterializationUsage()
    options = RunOptions(
        bundle_root=tmp_path,
        output_root=tmp_path,
        model="model",
        run_identity_sha256="r" * 64,
    )
    action_policy = cast(runner.TrialDevelopmentPhaseActionPolicyV1, object())
    action_spec = cast(runner.TrialDevelopmentPhaseActionSpecV1, object())

    def execute(
        *,
        resume_operation: Literal[
            "phase_request",
            "materialize",
            "phase_analysis",
            "phase_decision",
            "advance_state",
        ] = "phase_request",
        restored_attempt: PhaseAttempt | None = None,
    ) -> PhaseAttempt:
        return runner._run_one_phase(
            phase_id="phase2",
            public_dir=tmp_path / "work",
            phase_dir=phase_dir,
            state_path=state_path,
            program=program,
            loop=loop,
            usage=usage,
            options=options,
            src_root=tmp_path,
            prior_phase_summaries=[],
            violations=[],
            action_policy=action_policy,
            action_spec=action_spec,
            checkpoint=checkpoint,
            resume_operation=resume_operation,
            restored_attempt=restored_attempt,
        )

    with pytest.raises(SystemExit, match="simulated kill"):
        execute()

    assert captured is not None
    result = execute(
        resume_operation="phase_analysis",
        restored_attempt=captured,
    )

    assert provider_steps == ["request", "analysis", "decision"]
    assert materializations == 1
    assert result.decision_action == "declare_success"


def test_phase_checkpoint_rejects_partial_materialization_custody() -> None:
    """A trial-output artifact cannot be restored without its exact receipt."""

    with pytest.raises(ValidationError, match="materialization custody must be complete"):
        TrialDevRunCheckpointPhaseV1.model_validate(
            {
                "phase_id": "phase2",
                "trial_output": {
                    "relative_path": "agent_workdir/phase_phase2/trial_output",
                    "kind": "directory",
                    "sha256": "a" * 64,
                },
            }
        )
