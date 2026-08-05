"""Behavioral tests for canonical TrialDev programme contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevCheckpointOutcomeV1,
    TrialDevEvidenceReferenceV1,
    TrialDevPolicyBindingV1,
    TrialDevPortfolioActionSelectionV1,
    TrialDevPortfolioCheckpointHistoryEntryV1,
    TrialDevPortfolioProgrammeStateV1,
    TrialDevResourceScheduleV1,
    TrialDevSingleAssetProgrammeStateV1,
    TrialDevSupportedActionSetV1,
    TrialDevSupportedActionV1,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


def _policy(*, stream_id: str, budget: int | None = None) -> TrialDevPolicyBindingV1:
    return TrialDevPolicyBindingV1(
        stream_id=stream_id,
        objective_id="maximise_benefit",
        objective_policy_checksum=_SHA_A,
        action_policy_checksum=_SHA_B,
        design_menu_checksum=_SHA_C,
        resource_schedule=TrialDevResourceScheduleV1() if budget is not None else None,
        resource_budget_units=budget,
    )


def _outcome() -> TrialDevCheckpointOutcomeV1:
    return TrialDevCheckpointOutcomeV1(
        reach_status="reached",
        submission_status="accepted",
        analysis_status="estimable",
        execution_status="completed",
    )


def _evidence(*, evidence_id: str, checkpoint_id: str, asset_id: str) -> TrialDevEvidenceReferenceV1:
    return TrialDevEvidenceReferenceV1(
        evidence_id=evidence_id,
        evidence_kind="dataset",
        checkpoint_id=checkpoint_id,
        asset_id=asset_id,
        evidence_protocol_id="fixed_protocol",
        evidence_protocol_checksum=_SHA_A,
        source_family_id="source_family",
        world_id="world",
        generation_seed=1,
        relative_path=f"evidence/{evidence_id}.json",
        artifact_sha256=_SHA_B,
    )


def test_single_asset_state_is_locked_to_one_nominated_asset() -> None:
    evidence = _evidence(evidence_id="early-A", checkpoint_id="early_safety_study", asset_id="A")
    state = TrialDevSingleAssetProgrammeStateV1(
        programme_id="scenario__maximise_benefit",
        scenario_id="scenario",
        stream_id="single_asset_development",
        current_checkpoint_id="early_safety_study",
        candidate_asset_ids=("A", "B"),
        nominated_asset_id="A",
        active_asset_id="A",
        retired_asset_ids=("B",),
        policy_binding=_policy(stream_id="single_asset_development"),
        evidence=(evidence,),
    )

    assert state.checksum is not None
    payload = state.model_dump(mode="json")
    payload["checksum"] = None
    payload["active_asset_id"] = "B"
    with pytest.raises(ValidationError, match="cannot be retired|nominated single asset"):
        TrialDevSingleAssetProgrammeStateV1.model_validate(payload)


def test_portfolio_state_preserves_irreversible_history_and_budget() -> None:
    lead_early = _evidence(evidence_id="lead-safety", checkpoint_id="joint_early_study_review", asset_id="A")
    reserve_early = _evidence(evidence_id="reserve-safety", checkpoint_id="joint_early_study_review", asset_id="B")
    reserve_poc = _evidence(
        evidence_id="reserve-poc",
        checkpoint_id="promoted_reserve_proof_of_concept_review",
        asset_id="B",
    )
    action = TrialDevPortfolioActionSelectionV1(
        state_checksum=_SHA_A,
        checkpoint_id="joint_early_study_review",
        action_id="promote_reserve_to_proof_of_concept",
        analysis_method_id="published_method",
        supporting_evidence_ids=("lead-safety", "reserve-safety"),
        justification="The lead fails the safety rule and the reserve remains supported.",
    )
    history = TrialDevPortfolioCheckpointHistoryEntryV1(
        state_index=0,
        checkpoint_id="joint_early_study_review",
        evidence_reference_checksums=(str(lead_early.checksum), str(reserve_early.checksum)),
        selected_action=action,
        outcome=_outcome(),
        active_asset_id="B",
        lead_asset_id="A",
        reserve_asset_id="B",
        retired_asset_ids=("C",),
        resources_spent_units=4,
    )
    state = TrialDevPortfolioProgrammeStateV1(
        programme_id="portfolio__maximise_benefit__budget_8",
        scenario_id="portfolio",
        stream_id="bounded_portfolio_reallocation",
        current_checkpoint_id="promoted_reserve_proof_of_concept_review",
        candidate_asset_ids=("A", "B", "C"),
        lead_asset_id="A",
        reserve_asset_id="B",
        active_asset_id="B",
        retired_asset_ids=("C",),
        policy_binding=_policy(stream_id="bounded_portfolio_reallocation", budget=8),
        evidence=(lead_early, reserve_early, reserve_poc),
        history=(history,),
        resource_spent_units=4,
        switch_count=1,
        previous_state_checksum=_SHA_D,
    )

    assert state.active_asset_id == state.reserve_asset_id
    over_budget = state.model_dump(mode="json")
    over_budget["checksum"] = None
    over_budget["resource_spent_units"] = 9
    with pytest.raises(ValidationError, match="must not exceed"):
        TrialDevPortfolioProgrammeStateV1.model_validate(over_budget)


@pytest.mark.parametrize(
    ("reach", "submission", "analysis", "execution"),
    [
        ("structural_nonreach", "not_applicable", "not_applicable", "not_applicable"),
        ("reached", "missing", "missing", "model_noncompletion"),
        ("reached", "missing", "missing", "infrastructure_failure"),
        ("reached", "invalid", "invalid", "completed"),
        ("reached", "accepted", "invalid", "completed"),
        ("reached", "accepted", "non_estimable", "completed"),
        ("terminal", "not_applicable", "not_applicable", "not_applicable"),
    ],
)
def test_checkpoint_outcomes_keep_failure_classes_distinct(
    reach: str,
    submission: str,
    analysis: str,
    execution: str,
) -> None:
    outcome = TrialDevCheckpointOutcomeV1(
        reach_status=reach,
        submission_status=submission,
        analysis_status=analysis,
        execution_status=execution,
    )

    assert outcome.model_dump() == {
        "reach_status": reach,
        "submission_status": submission,
        "analysis_status": analysis,
        "execution_status": execution,
        "asset_eligibility": (),
    }


def test_supported_action_set_uses_public_evidence_without_entering_public_state_schema() -> None:
    supported = TrialDevSupportedActionSetV1(
        state_checksum=_SHA_A,
        checkpoint_id="proof_of_concept",
        submitted_analysis_method_id="published_method",
        policy_binding_checksum=_SHA_B,
        legal_actions=(
            TrialDevSupportedActionV1(action_id="advance_to_confirmation"),
            TrialDevSupportedActionV1(action_id="stop_development"),
        ),
        supported_actions=(
            TrialDevSupportedActionV1(action_id="advance_to_confirmation"),
            TrialDevSupportedActionV1(action_id="stop_development"),
        ),
        public_evidence_checksums=(_SHA_C,),
        sensitivity_policy_id="primary",
    )
    public_schema = TrialDevSingleAssetProgrammeStateV1.model_json_schema()
    public_schema_text = str(public_schema)

    assert {item.action_id for item in supported.supported_actions} == {
        "advance_to_confirmation",
        "stop_development",
    }
    assert "supported_action_ids" not in public_schema_text
    assert "submitted_analysis_method_id" not in public_schema_text
    assert "lead_asset_id" not in public_schema_text
    assert "reserve_asset_id" not in public_schema_text
    assert "switch_count" not in public_schema_text
    assert "resource_spent_units" not in public_schema_text


def test_checksum_mismatch_fails_loudly() -> None:
    policy = _policy(stream_id="single_asset_development")
    payload = policy.model_dump(mode="json")
    payload["objective_id"] = "changed_after_signing"

    with pytest.raises(ValidationError, match="checksum"):
        TrialDevPolicyBindingV1.model_validate(payload)
