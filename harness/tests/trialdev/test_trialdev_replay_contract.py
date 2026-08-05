"""Behavioral tests for TrialDev fixed-evidence replay boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from trialagentbench_harness.trialdev.bridge import parse_phase_decision
from trialagentbench_harness.trialdev.grading.sequential import materialize_phase_v1
from trialagentbench_harness.trialdev.prompts import build_obs_review_block
from trialagentbench_harness.trialdev.runner import _fixed_phase_replay_available, _run_obs_review
from trialagentbench_harness.trialdev.schema import MaterializationUsage, Program
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentRequestV1
from trialagentbench_harness.trialdev.share.sequential import (
    TrialDevelopmentObservationalReviewSubmissionV1,
    TrialDevelopmentPhaseActionPolicyV1,
    TrialDevelopmentPhaseActionSpecV1,
    validate_trial_output_bundle_v1,
)


def _request(*, sample_size: int, discontinuation_strategy: str) -> TrialDevelopmentRequestV1:
    return TrialDevelopmentRequestV1(
        scenario_id="s01",
        phase_id="phase2",
        candidate_drug_ids=("drug_a",),
        target_sample_size=sample_size,
        endpoint_id="E1",
        follow_up_days=90,
        enrollment_window_days=42,
        site_count_budget=8,
        allocation_ratio="1:1",
        design_cell_id="trialdev.phase2.fixed_final_operating_characteristics.v1",
        treatment_discontinuation_strategy=discontinuation_strategy,
        interim_policy="fixed_final",
        site_strategy="high_enrolling",
        selection_objective="benefit_risk",
    )


def _action_policy() -> TrialDevelopmentPhaseActionPolicyV1:
    return TrialDevelopmentPhaseActionPolicyV1(
        scenario_id="s01",
        phase_policy_checksum="a" * 64,
        decision_charter_checksum="b" * 64,
        action_specs=(
            TrialDevelopmentPhaseActionSpecV1(
                phase_id="phase1",
                allowed_action_ids=("advance_to_proof_of_concept", "stop_development"),
                stop_action_ids=("stop_development",),
                advance_action_ids=("advance_to_proof_of_concept",),
            ),
            TrialDevelopmentPhaseActionSpecV1(
                phase_id="phase2",
                allowed_action_ids=("advance_to_confirmation", "stop_development"),
                stop_action_ids=("stop_development",),
                advance_action_ids=("advance_to_confirmation",),
            ),
        ),
    )


def test_phase_action_policy_rejects_mutation_after_checksum_binding() -> None:
    policy = _action_policy()
    payload = policy.model_dump(mode="json")
    payload["scenario_id"] = "mutated"

    with pytest.raises(ValidationError, match="checksum"):
        TrialDevelopmentPhaseActionPolicyV1.model_validate(payload)


def test_observational_prompt_distinguishes_low_utility_from_failed_identification() -> None:
    prompt = build_obs_review_block()
    prose = " ".join(prompt.split())

    assert "none meets the entry criterion" in prose
    assert "report each candidate's estimate and ranking" in prose
    assert "comparison is not identified" in prose


def test_estimable_observational_branch_requires_structured_candidate_results() -> None:
    with pytest.raises(ValidationError, match="candidate estimates and ranking"):
        TrialDevelopmentObservationalReviewSubmissionV1.model_validate(
            {
                "response_branch": "estimable",
                "primary_resolution_evidence_class": "empirical_diagnosis",
                "ranked_drug_ids": [],
                "candidate_utility_estimates": [],
                "supporting_evidence_ids": ["analysis"],
                "candidate_drug_id": None,
                "decision_action": "withhold_nomination",
                "decision_rationale": "No candidate meets the declared entry criterion.",
            }
        )


def test_runtime_rejects_obsolete_single_asset_reallocation_fields() -> None:
    decision, error = parse_phase_decision(
        {
            "decision_action": "advance_to_confirmation",
            "supporting_evidence_ids": ["effect"],
            "candidate_drug_id": "drug_a",
            "dropped_candidate_drug_ids": ["drug_b"],
        },
        action_policy=_action_policy(),
        scenario_id="s01",
        phase_id="phase2",
        request_checksum="a" * 64,
        analysis_checksum="b" * 64,
    )

    assert decision is None
    assert error is not None
    assert "Extra inputs are not permitted" in error


def test_phase_decision_rejects_participant_supplied_custody() -> None:
    decision, error = parse_phase_decision(
        {
            "request_checksum": "a" * 64,
            "decision_action": "advance_to_confirmation",
            "supporting_evidence_ids": ["effect"],
            "candidate_drug_id": "drug_a",
        },
        action_policy=_action_policy(),
        scenario_id="s01",
        phase_id="phase2",
        request_checksum="c" * 64,
        analysis_checksum="b" * 64,
    )

    assert decision is None
    assert error is not None
    assert "harness-owned fields" in error


def test_fixed_replay_availability_is_bound_to_scenario_phase_and_candidate(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario_s01"
    cases_path = tmp_path / "fixed_trajectories" / "cases.jsonl"
    cases_path.parent.mkdir()
    case = {
        "scenario_root": "scenario_s01",
        "world_seed": 101,
        "program_objective_ids": ["benefit_risk"],
        "request": _request(
            sample_size=120,
            discontinuation_strategy="composite_discontinuation",
        ).model_dump(mode="json", exclude_none=True),
    }
    cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")

    assert _fixed_phase_replay_available(
        scenario_root=scenario,
        phase_id="phase2",
        candidate_drug_id="drug_a",
    )
    assert not _fixed_phase_replay_available(
        scenario_root=scenario,
        phase_id="phase1",
        candidate_drug_id="drug_a",
    )
    assert not _fixed_phase_replay_available(
        scenario_root=scenario,
        phase_id="phase2",
        candidate_drug_id="drug_b",
    )


def test_unavailable_submitted_nomination_is_recorded_without_continuation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import trialagentbench_harness.trialdev.runner as runner

    replies = []

    class Loop:
        def begin_step(self, **_kwargs) -> None:
            return None

        def append_user_message(self, _message: str) -> None:
            return None

        def run_until_submit(self, **_kwargs):
            return SimpleNamespace(payload={}, tool_call_id="call", name="submit")

        def append_tool_reply(self, _tool_call_id: str, content: str, **_kwargs) -> None:
            replies.append(json.loads(content))

    submission = SimpleNamespace(
        analysis_report=SimpleNamespace(candidate_utility_estimates=(), ranked_drug_ids=()),
        program_decision=SimpleNamespace(
            decision_action="nominate_for_early_study",
            recommended_drug_id="drug_a",
        ),
    )
    monkeypatch.setattr(runner, "_build_obs_review_submission_for_grader", lambda **_: submission)
    monkeypatch.setattr(runner, "_fixed_phase_replay_available", lambda **_: False)
    monkeypatch.setattr(
        runner.bridge,
        "observational_source_artifact_checksums",
        lambda *_args, **_kwargs: {"observational_extract.parquet": "a" * 64},
    )
    monkeypatch.setattr(
        runner.bridge,
        "observational_identification_artifact_checksums",
        lambda *_args: {"public/assignment_mechanism.json": "b" * 64},
    )
    program = Program(
        program_id="s01__benefit_risk",
        scenario_id="s01",
        objective_id="benefit_risk",
        items_by_phase={},
    )
    result = _run_obs_review(
        program=program,
        program_dir=tmp_path,
        loop=Loop(),
        usage=MaterializationUsage(),
        src_root=tmp_path / "scenario_s01",
        checkpoint=lambda: None,
    )

    assert result is submission
    assert replies == [
        {
            "status": "obs_review recorded",
            "decision_action": "nominate_for_early_study",
            "randomized_continuation_available": False,
        }
    ]


def test_fixed_replay_preserves_evidence_design_and_binds_agent_proposal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import trialagentbench_harness.trialdev.grading.sequential as sequential

    scenario = tmp_path / "scenario_s01"
    trajectory_root = tmp_path / "fixed_trajectories"
    trajectory_root.mkdir()
    (trajectory_root / "cases.jsonl").write_text("{}\n", encoding="utf-8")
    proposal = _request(sample_size=180, discontinuation_strategy="treatment_policy")
    evidence_request = _request(sample_size=120, discontinuation_strategy="composite_discontinuation")
    request_path = tmp_path / "proposal.json"
    request_path.write_text(proposal.model_dump_json(exclude_none=True), encoding="utf-8")
    state = SimpleNamespace(
        checksum="c" * 64,
        programme_id="programme",
        scenario_id="s01",
        stream_id="single_asset_development",
        current_checkpoint_id="proof_of_concept",
        active_asset_id="drug_a",
        retired_asset_ids=(),
        terminal_disposition="active",
        model_dump=lambda **_: {"version": "v1"},
    )

    def copy_evidence(**kwargs) -> tuple[str, int, int]:
        output = Path(kwargs["out_dir"])
        output.mkdir(parents=True)
        (output / "request.json").write_text(evidence_request.model_dump_json(exclude_none=True), encoding="utf-8")
        (output / "execution_summary.json").write_text("{}\n", encoding="utf-8")
        (output / "arm_mapping.json").write_text(
            json.dumps(
                {
                    "control_arm_id": "control",
                    "candidate_arm_ids": ["candidate"],
                    "drug_id_by_arm": {"control": "usual_care", "candidate": "drug_a"},
                    "arm_role_by_id": {"control": "control", "candidate": "candidate"},
                    "request_candidate_drug_ids": ["drug_a"],
                }
            ),
            encoding="utf-8",
        )
        common = {"USUBJID": ["01", "02"], "ARM": ["control", "candidate"]}
        pd.DataFrame(common).to_parquet(output / "participants.parquet", index=False)
        pd.DataFrame(
            {
                **common,
                "EVENT": [0, 1],
                "TIME": [90, 70],
                "FOLLOW_UP_DAYS": [90, 90],
                "TREATMENT_DISCONTINUATION_STRATEGY": ["composite_discontinuation"] * 2,
            }
        ).to_parquet(output / "endpoints.parquet", index=False)
        pd.DataFrame(common).to_parquet(output / "safety.parquet", index=False)
        return evidence_request.checksum(), 101, 202

    monkeypatch.setattr(sequential, "validate_program_state_file_v1", lambda **_: state)
    monkeypatch.setattr(sequential, "_validate_phase_request", lambda **_: None)
    monkeypatch.setattr(sequential, "_copy_fixed_phase_evidence_v1", copy_evidence)
    monkeypatch.setattr(sequential, "phase_summary_v1", lambda **_: {})

    output = tmp_path / "output"
    materialize_phase_v1(
        scenario_root=scenario,
        state_path=tmp_path / "state.json",
        request_path=request_path,
        out_dir=output,
        seed=303,
    )

    manifest = validate_trial_output_bundle_v1(trial_output_root=output)
    realized = TrialDevelopmentRequestV1.model_validate_json((output / "request.json").read_text(encoding="utf-8"))
    submitted = TrialDevelopmentRequestV1.model_validate_json(
        (output / "agent_request.json").read_text(encoding="utf-8")
    )
    assert realized.treatment_discontinuation_strategy == "composite_discontinuation"
    assert submitted.treatment_discontinuation_strategy == "treatment_policy"
    assert manifest.evidence_request_checksum == evidence_request.checksum()
    assert manifest.request_checksum == proposal.checksum()
    assert {"request.json", "agent_request.json"} <= set(manifest.metadata_files)

    altered_proposal = proposal.model_copy(update={"follow_up_days": 91})
    (output / "agent_request.json").write_text(
        altered_proposal.model_dump_json(exclude_none=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="agent_request.json does not match"):
        validate_trial_output_bundle_v1(trial_output_root=output)

    (output / "agent_request.json").write_text(proposal.model_dump_json(exclude_none=True), encoding="utf-8")
    altered_evidence = evidence_request.model_copy(update={"follow_up_days": 91})
    (output / "request.json").write_text(
        altered_evidence.model_dump_json(exclude_none=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="request.json does not match"):
        validate_trial_output_bundle_v1(trial_output_root=output)
