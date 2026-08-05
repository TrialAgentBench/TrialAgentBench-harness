"""Behavioral tests for portfolio decisions under observational uncertainty."""

from __future__ import annotations

import json

import pytest

from trialagentbench_harness.contracts.trialdev.portfolio_submission import (
    TrialDevPortfolioCheckpointGradeV1,
    TrialDevPortfolioRunSummaryV1,
    TrialDevPortfolioSubmissionAttemptV1,
)
from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevCheckpointOutcomeV1,
    TrialDevDecisionRuleEvidenceV1,
    TrialDevEvidenceReferenceV1,
    TrialDevObservationalCandidateEvidenceV1,
    TrialDevObservationalDecisionEvidenceV1,
    TrialDevPairContrastEvidenceV1,
    TrialDevPolicyBindingV1,
    TrialDevPortfolioProgrammeStateV1,
    TrialDevRandomizedDecisionEvidenceV1,
    TrialDevResourceScheduleV1,
)
from trialagentbench_harness.contracts.trialdev.scientific_grade import (
    TrialDevScientificAssessmentV1,
    TrialDevScientificEnvelopeV1,
)
from trialagentbench_harness.trialdev.agent import AgentTurnLimitExceeded
from trialagentbench_harness.trialdev.participant_submission import (
    _normalize_implicit_action_assets_v1,
    build_portfolio_checkpoint_v1,
    participant_payload_v1,
)
from trialagentbench_harness.trialdev.policy import derive_supported_action_set_v1
from trialagentbench_harness.trialdev.portfolio_grading import (
    _evidence_provenance_valid,
    _observational_agreement,
    _observational_disagreement_paths,
    _observational_scientific_agreement,
    _randomized_agreement,
    _randomized_disagreement_paths,
)
from trialagentbench_harness.trialdev.portfolio_runtime import (
    _participant_submission_schema,
    _require_submission_attempt_available,
    _system_prompt,
    resume_portfolio_programme_v1,
    run_portfolio_programme_v1,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def test_portfolio_prompt_presents_an_ordinary_clinical_decision_task() -> None:
    prompt = _system_prompt(
        programme_id="programme-01",
        objective_id="benefit_risk",
        budget=10,
        procedure_assistance="output_contract_only",
    )

    assert prompt.startswith("You are the statistical lead for a clinical development portfolio.")
    assert "CLINICAL QUESTION" in prompt
    assert "EVIDENCE AVAILABLE" in prompt
    assert "WORK REQUIRED" in prompt
    assert "CONCLUSION" in prompt
    assert "Choose and execute a defensible analysis" in prompt
    assert "synthetic" not in prompt.lower()


def test_portfolio_prompt_applies_the_selected_workflow_support() -> None:
    prompt = _system_prompt(
        programme_id="programme-01",
        objective_id="benefit_risk",
        budget=10,
        procedure_assistance="ordered_sop",
    )

    assert "Complete the same operations in this required order" in prompt
    assert "Choose and execute a defensible analysis" not in prompt


@pytest.mark.parametrize(
    "operation",
    [run_portfolio_programme_v1, resume_portfolio_programme_v1],
)
def test_portfolio_runtime_rejects_an_empty_submission_budget(operation, tmp_path) -> None:
    with pytest.raises(ValueError, match="max_submission_attempts"):
        operation(
            release_root=tmp_path / "release",
            programme_id="programme-01",
            output_root=tmp_path / "output",
            provider=object(),
            max_turns_per_checkpoint=5,
            max_tokens=4096,
            max_context_characters=120_000,
            watchdog_seconds=1800,
            max_submission_attempts=0,
            procedure_assistance="output_contract_only",
        )


def test_portfolio_runtime_stops_before_requesting_an_excess_submission() -> None:
    _require_submission_attempt_available(
        attempts_used=9,
        maximum_attempts=10,
        checkpoint="observational_review",
    )

    with pytest.raises(AgentTurnLimitExceeded, match="exceeded 10 submission attempts"):
        _require_submission_attempt_available(
            attempts_used=10,
            maximum_attempts=10,
            checkpoint="observational_review",
        )


def test_portfolio_submission_schema_omits_computed_checksums() -> None:
    schema = _participant_submission_schema()

    assert all(
        "checksum" not in definition.get("properties", {})
        for definition in schema.get("$defs", {}).values()
        if isinstance(definition, dict)
    )
    serialized = json.dumps(schema)
    assert "state_checksum" not in serialized
    assert "evidence_reference_checksums" not in serialized
    assert "schema_id" not in serialized
    assert "methods[].method_route_id" in serialized
    assert "distinct from the practical-equivalence margin" in serialized
    assert "two phase-1 studies for lead-reserve selection" in serialized


def test_participant_projection_can_remove_harness_bound_root_fields() -> None:
    projected = participant_payload_v1(
        _state(),
        root_fields=frozenset({"programme_id", "current_checkpoint_id"}),
    )

    assert "programme_id" not in projected
    assert "current_checkpoint_id" not in projected
    assert "evidence" in projected


def test_portfolio_builder_binds_state_and_evidence_custody() -> None:
    state = _state()
    evidence = participant_payload_v1(_identified_evidence(state))
    for candidate in evidence["candidates"]:
        candidate["evidence_ids"] = ["observational_extract"]
    submission = build_portfolio_checkpoint_v1(
        {
            "decision_evidence": evidence,
            "selected_action": {
                "action_id": "select_lead_and_reserve",
                "target_asset_id": "A",
                "reserve_asset_id": "B",
                "justification": "Both assets satisfy the stated entry criteria and remain practically equivalent.",
            },
            "scheduled_studies": [
                {"asset_id": "A", "phase_id": "phase1", "design_cell_id": "phase1_standard"},
                {"asset_id": "B", "phase_id": "phase1", "design_cell_id": "phase1_standard"},
            ],
        },
        state=state,
    )

    assert submission.state_checksum == state.checksum
    assert submission.selected_action.checkpoint_id == state.current_checkpoint_id
    assert submission.selected_action.supporting_evidence_ids == ("observational_extract",)
    assert isinstance(submission.decision_evidence, TrialDevObservationalDecisionEvidenceV1)
    assert all(
        candidate.evidence_reference_checksums == (str(state.evidence[0].checksum),)
        for candidate in submission.decision_evidence.candidates
    )


def test_portfolio_builder_removes_a_redundant_implicit_lead_name() -> None:
    state = _state().model_copy(
        update={
            "current_checkpoint_id": "joint_early_study_review",
            "lead_asset_id": "A",
            "reserve_asset_id": "B",
            "active_asset_id": "A",
        }
    )
    action = {
        "action_id": "advance_lead_to_proof_of_concept",
        "target_asset_id": "A",
    }

    _normalize_implicit_action_assets_v1(action=action, state=state)

    assert action == {"action_id": "advance_lead_to_proof_of_concept"}


def test_portfolio_builder_rejects_a_contradictory_implicit_asset_name() -> None:
    state = _state().model_copy(
        update={
            "current_checkpoint_id": "joint_early_study_review",
            "lead_asset_id": "A",
            "reserve_asset_id": "B",
            "active_asset_id": "A",
        }
    )
    action = {
        "action_id": "advance_lead_to_proof_of_concept",
        "target_asset_id": "B",
    }

    with pytest.raises(ValueError, match="current implied asset 'A'"):
        _normalize_implicit_action_assets_v1(action=action, state=state)


def test_incomplete_portfolio_summary_can_record_one_pending_transition() -> None:
    summary = TrialDevPortfolioRunSummaryV1(
        programme_id="programme",
        scenario_id="scenario",
        objective_id="benefit_risk",
        resource_budget_units=8,
        participant_view_checksum=_SHA_A,
        release_source_identity=_SHA_B,
        execution_status="infrastructure_failure",
        reached_checkpoint_ids=("observational_review",),
        state_relative_paths=("states/000_observational_review.json",),
        submission_relative_paths=("submissions/000_observational_review.json",),
        grade_relative_paths=("grades/000_observational_review.json",),
        wall_seconds_total=1.0,
        submission_attempts=1,
        correction_count=0,
        agent_turns=1,
        execute_code_calls=0,
        inspect_data_calls=0,
        provider_calls=1,
        provider_elapsed_seconds=1.0,
        prompt_tokens=1,
        completion_tokens=1,
        error="OSError: interrupted state write",
    )

    assert len(summary.state_relative_paths) == len(summary.grade_relative_paths)


def test_completed_portfolio_summary_requires_every_post_decision_state() -> None:
    with pytest.raises(ValueError, match="initial and post-decision states"):
        TrialDevPortfolioRunSummaryV1(
            programme_id="programme",
            scenario_id="scenario",
            objective_id="benefit_risk",
            resource_budget_units=8,
            participant_view_checksum=_SHA_A,
            release_source_identity=_SHA_B,
            execution_status="completed",
            terminal_disposition="success",
            reached_checkpoint_ids=("confirmation",),
            state_relative_paths=("states/000_confirmation.json",),
            submission_relative_paths=("submissions/000_confirmation.json",),
            grade_relative_paths=("grades/000_confirmation.json",),
            wall_seconds_total=1.0,
            submission_attempts=1,
            correction_count=0,
            agent_turns=1,
            execute_code_calls=0,
            inspect_data_calls=0,
            provider_calls=1,
            provider_elapsed_seconds=1.0,
            prompt_tokens=1,
            completion_tokens=1,
        )


def test_scientifically_complete_portfolio_grade_does_not_require_exact_reproduction() -> None:
    state = _state()
    supported = derive_supported_action_set_v1(
        state=state,
        evidence=_identified_evidence(state),
    )
    grade = TrialDevPortfolioCheckpointGradeV1(
        checkpoint_id="observational_review",
        state_checksum=str(state.checksum),
        evidence_numeric_agreement=False,
        numeric_disagreement_paths=("decision_evidence.candidates[A].utility_estimate",),
        provenance_valid=True,
        supported_action_set=supported,
        selected_action_supported=True,
        scheduled_designs_valid=True,
        scientific_assessment=TrialDevScientificAssessmentV1(
            execution="passed",
            question_estimand="passed",
            design="passed",
            assumptions="passed",
            analysis_classification="uncertainty_qualified",
            scientific_agreement="passed",
            exact_reproduction="failed",
            uncertainty="passed",
            action_admissibility="passed",
            evidential_support="passed",
            sequential_coherence="passed",
            scientific_envelope=TrialDevScientificEnvelopeV1(
                envelope_id="declared-margin",
                basis="declared_practical_equivalence_margin",
                absolute_margin=0.02,
                exact_reproduction_tolerance=0.0005,
            ),
            decision_complete=True,
        ),
        outcome=TrialDevCheckpointOutcomeV1(
            reach_status="reached",
            submission_status="accepted",
            analysis_status="estimable",
            execution_status="completed",
        ),
    )

    assert grade.evidence_numeric_agreement is False
    assert grade.scientific_assessment.decision_complete is True


def test_scientific_disagreement_is_an_accepted_measured_attempt() -> None:
    state = _state()
    supported = derive_supported_action_set_v1(
        state=state,
        evidence=_identified_evidence(state),
    )
    path = "decision_evidence.candidates[A].utility_estimate"
    grade = TrialDevPortfolioCheckpointGradeV1(
        checkpoint_id="observational_review",
        state_checksum=str(state.checksum),
        evidence_numeric_agreement=False,
        numeric_disagreement_paths=(path,),
        provenance_valid=True,
        supported_action_set=supported,
        selected_action_supported=True,
        scheduled_designs_valid=True,
        scientific_assessment=TrialDevScientificAssessmentV1(
            execution="passed",
            question_estimand="passed",
            design="passed",
            assumptions="passed",
            analysis_classification="uncertainty_qualified",
            scientific_agreement="failed",
            exact_reproduction="failed",
            uncertainty="passed",
            action_admissibility="passed",
            evidential_support="passed",
            sequential_coherence="passed",
            scientific_envelope=TrialDevScientificEnvelopeV1(
                envelope_id="declared-margin",
                basis="declared_practical_equivalence_margin",
                absolute_margin=0.02,
                exact_reproduction_tolerance=0.0005,
            ),
            failure_reasons=("scientific_disagreement",),
            decision_complete=False,
        ),
        outcome=TrialDevCheckpointOutcomeV1(
            reach_status="reached",
            submission_status="accepted",
            analysis_status="estimable",
            execution_status="completed",
        ),
    )

    attempt = TrialDevPortfolioSubmissionAttemptV1(
        checkpoint_id="observational_review",
        attempt_index=1,
        transport_name="submit_portfolio_checkpoint",
        status="accepted",
        submitted_payload={},
        grade=grade,
    )

    assert attempt.status == "accepted"
    assert attempt.grade is not None
    assert attempt.grade.scientific_assessment.decision_complete is False


def _state() -> TrialDevPortfolioProgrammeStateV1:
    evidence = TrialDevEvidenceReferenceV1(
        evidence_id="observational_extract",
        evidence_kind="dataset",
        checkpoint_id="observational_review",
        evidence_protocol_id="observational_review_v1",
        evidence_protocol_checksum=_SHA_A,
        source_family_id=_SHA_C,
        world_id="world",
        generation_seed=12,
        relative_path="worlds/world/public/observational_extract.parquet",
        artifact_sha256=_SHA_B,
    )
    return TrialDevPortfolioProgrammeStateV1(
        programme_id="world__benefit_risk__budget_10",
        scenario_id="world",
        current_checkpoint_id="observational_review",
        candidate_asset_ids=("A", "B", "C"),
        policy_binding=TrialDevPolicyBindingV1(
            stream_id="bounded_portfolio_reallocation",
            objective_id="benefit_risk",
            objective_policy_checksum=_SHA_A,
            action_policy_checksum=_SHA_B,
            design_menu_checksum=_SHA_C,
            resource_schedule=TrialDevResourceScheduleV1(),
            resource_budget_units=10,
        ),
        evidence=(evidence,),
    )


def _identified_evidence(state: TrialDevPortfolioProgrammeStateV1) -> TrialDevObservationalDecisionEvidenceV1:
    values = {
        "A": (1.00, 0.08),
        "B": (0.97, 0.07),
        "C": (0.40, 0.01),
    }
    candidates = tuple(
        TrialDevObservationalCandidateEvidenceV1(
            asset_id=asset_id,
            utility_estimate=utility,
            utility_lower_bound=utility - 0.1,
            utility_upper_bound=utility + 0.1,
            efficacy_estimate=efficacy,
            efficacy_lower_bound=efficacy - 0.02,
            efficacy_upper_bound=efficacy + 0.02,
            evidence_reference_checksums=(str(state.evidence[0].checksum),),
        )
        for asset_id, (utility, efficacy) in values.items()
    )
    pairs = tuple(
        TrialDevPairContrastEvidenceV1(
            lead_asset_id=first,
            reserve_asset_id=second,
            confidence_half_width=0.05,
        )
        for index, first in enumerate(sorted(state.candidate_asset_ids))
        for second in sorted(state.candidate_asset_ids)[index + 1 :]
    )
    return TrialDevObservationalDecisionEvidenceV1(
        state_checksum=str(state.checksum),
        analysis_method_id="declared_observational_method",
        identification_status="identified",
        minimum_efficacy_gain=0.05,
        practical_equivalence_margin=0.05,
        candidates=candidates,
        pair_contrasts=pairs,
    )


def test_identified_portfolio_allows_every_evidence_supported_ordered_pair() -> None:
    state = _state()

    supported = derive_supported_action_set_v1(
        state=state,
        evidence=_identified_evidence(state),
    )

    signatures = {
        (action.action_id, action.target_asset_id, action.reserve_asset_id) for action in supported.supported_actions
    }
    assert signatures == {
        ("select_lead_and_reserve", "A", "B"),
        ("select_lead_and_reserve", "B", "A"),
    }


def test_candidate_labels_do_not_change_the_supported_decision_structure() -> None:
    state = _state()
    evidence = _identified_evidence(state)
    label_map = {"A": "asset_3", "B": "asset_1", "C": "asset_2"}
    state_payload = state.model_dump(mode="python", exclude={"checksum"})
    state_payload["candidate_asset_ids"] = tuple(label_map[item] for item in state.candidate_asset_ids)
    relabelled_state = TrialDevPortfolioProgrammeStateV1.model_validate(state_payload)
    candidates = tuple(
        type(row).model_validate(
            {
                **row.model_dump(mode="python", exclude={"checksum", "asset_id"}),
                "asset_id": label_map[row.asset_id],
            }
        )
        for row in evidence.candidates
    )
    pairs = tuple(
        type(row).model_validate(
            {
                **row.model_dump(
                    mode="python",
                    exclude={"checksum", "lead_asset_id", "reserve_asset_id"},
                ),
                "lead_asset_id": sorted((label_map[row.lead_asset_id], label_map[row.reserve_asset_id]))[0],
                "reserve_asset_id": sorted((label_map[row.lead_asset_id], label_map[row.reserve_asset_id]))[1],
            }
        )
        for row in evidence.pair_contrasts
    )
    evidence_payload = evidence.model_dump(
        mode="python",
        exclude={"checksum", "state_checksum", "candidates", "pair_contrasts"},
    )
    relabelled_evidence = TrialDevObservationalDecisionEvidenceV1.model_validate(
        {
            **evidence_payload,
            "state_checksum": str(relabelled_state.checksum),
            "candidates": candidates,
            "pair_contrasts": pairs,
        }
    )

    original = derive_supported_action_set_v1(state=state, evidence=evidence)
    relabelled = derive_supported_action_set_v1(
        state=relabelled_state,
        evidence=relabelled_evidence,
    )
    inverse = {value: key for key, value in label_map.items()}
    relabelled_signatures = {
        (
            action.action_id,
            inverse.get(action.target_asset_id, action.target_asset_id),
            inverse.get(action.reserve_asset_id, action.reserve_asset_id),
        )
        for action in relabelled.supported_actions
    }

    assert relabelled_signatures == {
        (action.action_id, action.target_asset_id, action.reserve_asset_id) for action in original.supported_actions
    }


def test_reserve_is_compared_with_alternatives_not_required_to_match_the_lead() -> None:
    state = _state()
    base = _identified_evidence(state)
    utilities = {"A": 1.0, "B": 0.60, "C": 0.20}
    efficacy = {"A": 0.08, "B": 0.07, "C": 0.01}
    candidates = tuple(
        row.model_copy(
            update={
                "utility_estimate": utilities[row.asset_id],
                "utility_lower_bound": utilities[row.asset_id] - 0.1,
                "utility_upper_bound": utilities[row.asset_id] + 0.1,
                "efficacy_estimate": efficacy[row.asset_id],
                "efficacy_lower_bound": efficacy[row.asset_id] - 0.02,
                "efficacy_upper_bound": efficacy[row.asset_id] + 0.02,
            }
        )
        for row in base.candidates
    )
    evidence = base.model_copy(update={"candidates": candidates})

    supported = derive_supported_action_set_v1(state=state, evidence=evidence)

    signatures = {
        (action.action_id, action.target_asset_id, action.reserve_asset_id) for action in supported.supported_actions
    }
    assert signatures == {("select_lead_and_reserve", "A", "B")}


def test_nonidentified_portfolio_requires_qualified_nonallocation() -> None:
    state = _state()
    evidence = TrialDevObservationalDecisionEvidenceV1(
        state_checksum=str(state.checksum),
        analysis_method_id="declared_observational_method",
        identification_status="not_identified",
        minimum_efficacy_gain=0.05,
        practical_equivalence_margin=0.05,
        identification_evidence_reference_checksums=(str(state.evidence[0].checksum),),
    )

    supported = derive_supported_action_set_v1(state=state, evidence=evidence)

    assert tuple(action.action_id for action in supported.supported_actions) == ("withhold_selection",)


def test_nonidentified_evidence_reference_error_is_provenance_not_numeric() -> None:
    state = _state()
    expected = TrialDevObservationalDecisionEvidenceV1(
        state_checksum=str(state.checksum),
        analysis_method_id="declared_observational_method",
        identification_status="not_identified",
        minimum_efficacy_gain=0.05,
        practical_equivalence_margin=0.05,
        identification_evidence_reference_checksums=(str(state.evidence[0].checksum),),
    )
    observed = TrialDevObservationalDecisionEvidenceV1(
        state_checksum=str(state.checksum),
        analysis_method_id=expected.analysis_method_id,
        identification_status=expected.identification_status,
        minimum_efficacy_gain=expected.minimum_efficacy_gain,
        practical_equivalence_margin=expected.practical_equivalence_margin,
        identification_evidence_reference_checksums=(_SHA_A,),
    )

    assert _observational_agreement(observed, expected, tolerance=0.0005)
    assert not _evidence_provenance_valid(observed, expected)


def test_identification_disagreement_does_not_invalidate_current_evidence_provenance() -> None:
    state = _state()
    expected = _identified_evidence(state)
    observed = TrialDevObservationalDecisionEvidenceV1(
        state_checksum=str(state.checksum),
        analysis_method_id=expected.analysis_method_id,
        identification_status="not_identified",
        minimum_efficacy_gain=expected.minimum_efficacy_gain,
        practical_equivalence_margin=expected.practical_equivalence_margin,
        identification_evidence_reference_checksums=(str(state.evidence[0].checksum),),
    )

    assert _evidence_provenance_valid(observed, expected)


def test_observational_disagreement_paths_name_only_the_incorrect_field() -> None:
    state = _state()
    expected = _identified_evidence(state)
    candidates = tuple(
        row.model_copy(update={"utility_lower_bound": row.utility_lower_bound - 0.01}) if row.asset_id == "A" else row
        for row in expected.candidates
    )
    observed = expected.model_copy(update={"candidates": candidates})

    assert _observational_disagreement_paths(observed, expected, tolerance=0.0005) == (
        "decision_evidence.candidates[A].utility_lower_bound",
    )


def _observational_scientific_agreement_for_test(
    *,
    state: TrialDevPortfolioProgrammeStateV1,
    observed: TrialDevObservationalDecisionEvidenceV1,
    expected: TrialDevObservationalDecisionEvidenceV1,
) -> bool:
    return _observational_scientific_agreement(
        state=state,
        observed=observed,
        expected=expected,
        envelope=TrialDevScientificEnvelopeV1(
            envelope_id="declared-margin",
            basis="declared_practical_equivalence_margin",
            absolute_margin=expected.practical_equivalence_margin,
            exact_reproduction_tolerance=0.0005,
        ),
        exact_tolerance=0.0005,
    )


@pytest.mark.parametrize(
    ("offset", "agrees"),
    [(0.0499, True), (0.0501, False)],
)
def test_observational_scientific_agreement_uses_the_declared_utility_margin(
    offset: float,
    agrees: bool,
) -> None:
    state = _state()
    expected = _identified_evidence(state)
    candidates = tuple(
        row.model_copy(update={"utility_estimate": row.utility_estimate + offset}) for row in expected.candidates
    )
    observed = expected.model_copy(update={"candidates": candidates})

    assert (
        _observational_scientific_agreement_for_test(
            state=state,
            observed=observed,
            expected=expected,
        )
        is agrees
    )


@pytest.mark.parametrize(
    ("efficacy_lower_bound", "agrees"),
    [(0.0501, True), (0.0499, False)],
)
def test_observational_scientific_agreement_uses_the_efficacy_threshold(
    efficacy_lower_bound: float,
    agrees: bool,
) -> None:
    state = _state()
    expected = _identified_evidence(state)
    candidates = tuple(
        row.model_copy(update={"efficacy_lower_bound": efficacy_lower_bound}) if row.asset_id == "A" else row
        for row in expected.candidates
    )
    observed = expected.model_copy(update={"candidates": candidates})

    assert (
        _observational_scientific_agreement_for_test(
            state=state,
            observed=observed,
            expected=expected,
        )
        is agrees
    )


@pytest.mark.parametrize(
    ("confidence_half_width", "agrees"),
    [(0.0999, True), (0.1001, False)],
)
def test_observational_scientific_agreement_uses_pairwise_uncertainty(
    confidence_half_width: float,
    agrees: bool,
) -> None:
    state = _state()
    expected = _identified_evidence(state)
    pairs = tuple(
        (
            row.model_copy(update={"confidence_half_width": confidence_half_width})
            if (row.lead_asset_id, row.reserve_asset_id) == ("A", "B")
            else row
        )
        for row in expected.pair_contrasts
    )
    observed = expected.model_copy(update={"pair_contrasts": pairs})

    assert (
        _observational_scientific_agreement_for_test(
            state=state,
            observed=observed,
            expected=expected,
        )
        is agrees
    )


def test_observational_scientific_agreement_rejects_a_changed_supported_action_set() -> None:
    state = _state()
    expected = _identified_evidence(state)
    candidates = tuple(
        (
            row.model_copy(
                update={
                    "utility_estimate": 0.96,
                    "utility_lower_bound": 0.86,
                    "utility_upper_bound": 1.06,
                    "efficacy_estimate": 0.07,
                    "efficacy_lower_bound": 0.05,
                    "efficacy_upper_bound": 0.09,
                }
            )
            if row.asset_id == "C"
            else row
        )
        for row in expected.candidates
    )
    observed = expected.model_copy(update={"candidates": candidates})

    assert not _observational_scientific_agreement_for_test(
        state=state,
        observed=observed,
        expected=expected,
    )


def test_evidence_reference_order_does_not_change_provenance() -> None:
    state = _state()
    expected = TrialDevObservationalDecisionEvidenceV1(
        state_checksum=str(state.checksum),
        analysis_method_id="declared_observational_method",
        identification_status="not_identified",
        minimum_efficacy_gain=0.05,
        practical_equivalence_margin=0.05,
        identification_evidence_reference_checksums=(_SHA_A, _SHA_B),
    )
    observed = TrialDevObservationalDecisionEvidenceV1(
        state_checksum=str(state.checksum),
        analysis_method_id=expected.analysis_method_id,
        identification_status=expected.identification_status,
        minimum_efficacy_gain=expected.minimum_efficacy_gain,
        practical_equivalence_margin=expected.practical_equivalence_margin,
        identification_evidence_reference_checksums=(_SHA_B, _SHA_A),
    )

    assert _evidence_provenance_valid(observed, expected)


def test_randomized_evidence_reference_error_is_provenance_not_numeric() -> None:
    expected_rule = TrialDevDecisionRuleEvidenceV1(
        rule_id="A:efficacy",
        asset_id="A",
        domain="efficacy",
        direction="minimum",
        estimate=0.10,
        lower_bound=0.06,
        upper_bound=0.14,
        threshold=0.05,
        evidence_reference_checksums=(_SHA_A,),
    )
    observed_rule = TrialDevDecisionRuleEvidenceV1(
        **expected_rule.model_dump(mode="python", exclude={"checksum", "evidence_reference_checksums"}),
        evidence_reference_checksums=(_SHA_B,),
    )
    expected = TrialDevRandomizedDecisionEvidenceV1(
        state_checksum=_SHA_C,
        analysis_method_id="declared_randomized_method",
        rules=(expected_rule,),
    )
    observed = TrialDevRandomizedDecisionEvidenceV1(
        state_checksum=_SHA_C,
        analysis_method_id=expected.analysis_method_id,
        rules=(observed_rule,),
    )

    assert _randomized_agreement(observed, expected, tolerance=0.0005)
    assert not _evidence_provenance_valid(observed, expected)


def test_randomized_disagreement_paths_name_only_the_incorrect_field() -> None:
    expected_rule = TrialDevDecisionRuleEvidenceV1(
        rule_id="A:efficacy",
        asset_id="A",
        domain="efficacy",
        direction="minimum",
        estimate=0.10,
        lower_bound=0.06,
        upper_bound=0.14,
        threshold=0.05,
        evidence_reference_checksums=(_SHA_A,),
    )
    observed_rule = expected_rule.model_copy(update={"lower_bound": 0.05})
    expected = TrialDevRandomizedDecisionEvidenceV1(
        state_checksum=_SHA_C,
        analysis_method_id="declared_randomized_method",
        rules=(expected_rule,),
    )
    observed = expected.model_copy(update={"rules": (observed_rule,)})

    assert _randomized_disagreement_paths(observed, expected, tolerance=0.0005) == (
        "decision_evidence.rules[A:efficacy].lower_bound",
    )


def test_portfolio_allocation_fails_when_one_candidate_contrast_is_missing() -> None:
    state = _state()
    complete = _identified_evidence(state)
    incomplete = complete.model_copy(update={"pair_contrasts": complete.pair_contrasts[:-1]})

    with pytest.raises(ValueError, match="every candidate pair"):
        derive_supported_action_set_v1(state=state, evidence=incomplete)


def test_candidate_contrast_identifier_order_does_not_change_the_policy() -> None:
    state = _state()
    complete = _identified_evidence(state)
    first = complete.pair_contrasts[0]
    reversed_first = TrialDevPairContrastEvidenceV1(
        lead_asset_id=first.reserve_asset_id,
        reserve_asset_id=first.lead_asset_id,
        confidence_half_width=first.confidence_half_width,
    )
    reordered = complete.model_copy(update={"pair_contrasts": (reversed_first, *complete.pair_contrasts[1:])})

    assert derive_supported_action_set_v1(state=state, evidence=reordered) == derive_supported_action_set_v1(
        state=state,
        evidence=complete,
    )


def test_candidate_contrast_rejects_a_symmetric_duplicate() -> None:
    complete = _identified_evidence(_state())
    first = complete.pair_contrasts[0]
    reverse = TrialDevPairContrastEvidenceV1(
        lead_asset_id=first.reserve_asset_id,
        reserve_asset_id=first.lead_asset_id,
        confidence_half_width=first.confidence_half_width,
    )
    payload = complete.model_dump(mode="python", exclude={"checksum", "pair_contrasts"})

    with pytest.raises(ValueError, match="unique regardless of identifier order"):
        TrialDevObservationalDecisionEvidenceV1.model_validate(
            {**payload, "pair_contrasts": (*complete.pair_contrasts, reverse)}
        )


def test_identified_but_unqualified_candidates_withhold_without_pairwise_ranking() -> None:
    state = _state()
    candidates = tuple(
        TrialDevObservationalCandidateEvidenceV1(
            asset_id=asset_id,
            utility_estimate=0.1,
            utility_lower_bound=0.0,
            utility_upper_bound=0.2,
            efficacy_estimate=0.01,
            efficacy_lower_bound=-0.01,
            efficacy_upper_bound=0.03,
            evidence_reference_checksums=(str(state.evidence[0].checksum),),
        )
        for asset_id in state.candidate_asset_ids
    )
    evidence = TrialDevObservationalDecisionEvidenceV1(
        state_checksum=str(state.checksum),
        analysis_method_id="declared_observational_method",
        identification_status="identified",
        minimum_efficacy_gain=0.05,
        practical_equivalence_margin=0.05,
        candidates=candidates,
    )

    supported = derive_supported_action_set_v1(state=state, evidence=evidence)

    assert tuple(action.action_id for action in supported.supported_actions) == ("withhold_selection",)
