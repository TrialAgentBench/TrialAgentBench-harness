"""Tests for source-grounded narrative transcription."""

from __future__ import annotations

import hashlib
from typing import cast

import pytest
from pydantic import ValidationError

from trialagentbench_harness.analysis.experiments.trialeval_endpoint_rows import (
    join_ablation_evaluator_labels_v1,
)
from trialagentbench_harness.analysis.experiments.trialeval_transcription import (
    evaluate_representation_fixtures_v1,
    validate_narrative_transcription_v1,
)
from trialagentbench_harness.contracts.experiments import (
    TrialEvalAblationEndpointRowV1,
    TrialEvalAblationEvaluatorLabelsV1,
    TrialEvalNarrativeTranscriptionV1,
    TrialEvalRepresentationFixtureV1,
)


def _submission() -> dict[str, object]:
    return {
        "task_id": "TASK1001",
        "primary_analysis": {
            "declared_primary": True,
            "estimand": {
                "estimand_id": "primary_itt_rmst",
                "population_id": "randomized_participants",
                "treatment_id": "active",
                "comparator_id": "control",
                "endpoint_id": "time_to_event",
                "intercurrent_event_strategy_ids": ["rescue_therapy:treatment_policy"],
                "horizon": {"value": 365, "unit": "days"},
            },
            "estimator": {
                "analysis_method_id": "km_rmst_greenwood",
                "implementation": "Kaplan-Meier integration",
                "qualifications": ["independent_censoring"],
            },
            "result_kind": "numeric_point",
            "result": {
                "kind": "scalar",
                "value": 18.0,
                "effect_scale": "rmst_difference_tau",
                "unit": "days",
                "interval": {"lower": 4.0, "upper": 32.0, "confidence_level": 0.95},
            },
            "favorable_direction": "higher",
        },
        "limitations": ["The horizon is protocol-defined."],
    }


def _claim(
    report: str,
    *,
    claim_id: str,
    field_path: str,
    text: str,
    parsed_value: object,
    role: str = "primary",
    result_shape: str | None = None,
    unit: str | None = None,
    orientation: str | None = None,
) -> dict[str, object]:
    start = report.index(text)
    return {
        "claim_id": claim_id,
        "field_path": field_path,
        "claim_role": role,
        "evidence_level": "executed",
        "spans": [{"start": start, "end": start + len(text), "text": text}],
        "raw_value": text,
        "parsed_value": parsed_value,
        "result_shape": result_shape,
        "unit": unit,
        "orientation": orientation,
    }


def _transcription(report: str) -> TrialEvalNarrativeTranscriptionV1:
    primary = cast(dict[str, object], _submission()["primary_analysis"])
    return TrialEvalNarrativeTranscriptionV1.model_validate(
        {
            "assignment_id": "assignment-1",
            "report_sha256": hashlib.sha256(report.encode("utf-8")).hexdigest(),
            "source": "manual_masked",
            "source_identity": "masked-panel-1",
            "transcriber_identities": ["transcriber-A", "transcriber-B"],
            "transcription_disposition": "independent_exact_agreement",
            "blinded_to_model_identity": True,
            "blinded_to_evaluator_reference": True,
            "status": "complete",
            "submission": _submission(),
            "claims": [
                _claim(
                    report,
                    claim_id="estimand",
                    field_path="primary_analysis.estimand",
                    text="365-day ITT contrast",
                    parsed_value=primary["estimand"],
                ),
                _claim(
                    report,
                    claim_id="estimator",
                    field_path="primary_analysis.estimator",
                    text="Kaplan-Meier integration",
                    parsed_value=primary["estimator"],
                ),
                _claim(
                    report,
                    claim_id="result",
                    field_path="primary_analysis.result",
                    text="18 days (95% CI 4 to 32)",
                    parsed_value=primary["result"],
                    result_shape="scalar",
                    unit="days",
                ),
                _claim(
                    report,
                    claim_id="result-kind",
                    field_path="primary_analysis.result_kind",
                    text="primary numeric point result",
                    parsed_value="numeric_point",
                ),
                _claim(
                    report,
                    claim_id="direction",
                    field_path="primary_analysis.favorable_direction",
                    text="Higher is favorable",
                    parsed_value="higher",
                    orientation="higher",
                ),
                _claim(
                    report,
                    claim_id="limitations",
                    field_path="limitations",
                    text="limited to 365 days",
                    parsed_value=["The horizon is protocol-defined."],
                    role="limitation",
                ),
            ],
        }
    )


def test_complete_transcription_requires_exact_report_support() -> None:
    report = (
        "The 365-day ITT contrast used Kaplan-Meier integration with Greenwood uncertainty under "
        "independent censoring: 18 days (95% CI 4 to 32), reported as the primary numeric point result. "
        "Higher is favorable; interpretation is limited to 365 days."
    )
    transcription = _transcription(report)
    validate_narrative_transcription_v1(
        transcription=transcription,
        frozen_report=report,
        expected_assignment_id="assignment-1",
        expected_task_id="TASK1001",
    )

    unsupported_payload = transcription.model_dump(mode="json")
    span = unsupported_payload["claims"][1]["spans"][0]
    span["text"] = "X" * (span["end"] - span["start"])
    unsupported_payload["claims"][1]["raw_value"] = span["text"]
    unsupported = TrialEvalNarrativeTranscriptionV1.model_validate(unsupported_payload)
    with pytest.raises(ValueError, match="does not match the frozen report"):
        validate_narrative_transcription_v1(
            transcription=unsupported,
            frozen_report=report,
            expected_assignment_id="assignment-1",
            expected_task_id="TASK1001",
        )


def test_complete_transcription_submission_must_equal_its_claims() -> None:
    report = (
        "The 365-day ITT contrast used Kaplan-Meier integration with Greenwood uncertainty under "
        "independent censoring: 18 days (95% CI 4 to 32), reported as the primary numeric point result. "
        "Higher is favorable; interpretation is limited to 365 days."
    )
    payload = _transcription(report).model_dump(mode="json")
    payload["submission"]["primary_analysis"]["result"]["value"] = 19.0

    with pytest.raises(ValidationError, match="differs from its source-grounded narrative claims"):
        TrialEvalNarrativeTranscriptionV1.model_validate(payload)


def test_term_mention_alone_cannot_form_a_complete_transcription() -> None:
    payload = {
        "assignment_id": "assignment-1",
        "report_sha256": "a" * 64,
        "source": "manual_masked",
        "source_identity": "masked-panel-1",
        "transcriber_identities": ["transcriber-A", "transcriber-B"],
        "transcription_disposition": "independent_exact_agreement",
        "blinded_to_model_identity": True,
        "blinded_to_evaluator_reference": True,
        "status": "complete",
        "submission": _submission(),
        "claims": [
            {
                "claim_id": "estimator",
                "field_path": "primary_analysis.estimator",
                "claim_role": "primary",
                "evidence_level": "executed",
                "spans": [{"start": 0, "end": 4, "text": "RMST"}],
                "raw_value": "RMST",
                "parsed_value": {
                    "analysis_method_id": "km_rmst_greenwood",
                    "implementation": "Kaplan-Meier integration",
                },
            },
        ],
    }
    with pytest.raises(ValidationError, match="lacks source-grounded claims"):
        TrialEvalNarrativeTranscriptionV1.model_validate(payload)


@pytest.mark.parametrize(
    ("claim_role", "evidence_level"),
    (("rejected", "executed"), ("hypothetical", "declared"), ("primary", "mentioned")),
)
def test_nonexecuted_or_nonprimary_claim_cannot_support_primary_credit(
    claim_role: str,
    evidence_level: str,
) -> None:
    report = (
        "The 365-day ITT contrast used Kaplan-Meier integration with Greenwood uncertainty under "
        "independent censoring: 18 days (95% CI 4 to 32), reported as the primary numeric point result. "
        "Higher is favorable; interpretation is limited to 365 days."
    )
    payload = _transcription(report).model_dump(mode="json")
    payload["claims"][1]["claim_role"] = claim_role
    payload["claims"][1]["evidence_level"] = evidence_level

    with pytest.raises(ValidationError, match="lacks source-grounded claims"):
        TrialEvalNarrativeTranscriptionV1.model_validate(payload)


@pytest.mark.parametrize(
    "transcriber_identities",
    ([], ["transcriber-A"], ["transcriber-A", "transcriber-A"]),
)
def test_manual_transcription_requires_two_independent_transcriber_identities(
    transcriber_identities: list[str],
) -> None:
    report = (
        "The 365-day ITT contrast used Kaplan-Meier integration with Greenwood uncertainty under "
        "independent censoring: 18 days (95% CI 4 to 32), reported as the primary numeric point result. "
        "Higher is favorable; interpretation is limited to 365 days."
    )
    payload = _transcription(report).model_dump(mode="json")
    payload["transcriber_identities"] = transcriber_identities

    with pytest.raises(ValidationError, match="at least two unique transcriber"):
        TrialEvalNarrativeTranscriptionV1.model_validate(payload)


def test_automated_import_has_no_implicit_authority() -> None:
    payload = {
        "assignment_id": "assignment-1",
        "report_sha256": "a" * 64,
        "source": "automated_importer",
        "source_identity": "provider/model/version",
        "blinded_to_model_identity": True,
        "blinded_to_evaluator_reference": True,
        "status": "abstain",
        "abstention_reason": "The report did not contain a complete numerical primary analysis.",
    }
    with pytest.raises(ValidationError, match="requires immutable prompt, schema, and response hashes"):
        TrialEvalNarrativeTranscriptionV1.model_validate(payload)


def test_fixed_answer_fixture_requires_exact_structured_semantic_parity() -> None:
    report = (
        "The 365-day ITT contrast used Kaplan-Meier integration with Greenwood uncertainty under "
        "independent censoring: 18 days (95% CI 4 to 32), reported as the primary numeric point result. "
        "Higher is favorable; interpretation is limited to 365 days."
    )
    fixture = TrialEvalRepresentationFixtureV1.model_validate(
        {
            "fixture_id": "scalar-rmst",
            "task_id": "TASK1001",
            "structured_submission": _submission(),
            "narrative_report": report,
            "manual_transcription": _transcription(report),
        }
    )
    rows = evaluate_representation_fixtures_v1((fixture,))
    assert rows[0].manual_exact_match is True
    assert rows[0].automated_status == "not_run"

    altered = fixture.model_dump(mode="json")
    altered["structured_submission"]["primary_analysis"]["result"]["value"] = 19.0
    mismatched = TrialEvalRepresentationFixtureV1.model_validate(altered)
    with pytest.raises(ValueError, match="identical structured semantics"):
        evaluate_representation_fixtures_v1((mismatched,))


def _evaluator_labels() -> TrialEvalAblationEvaluatorLabelsV1:
    return TrialEvalAblationEvaluatorLabelsV1.model_validate(
        {
            "evaluator_release_sha256": "e" * 64,
            "task_identities": [
                {
                    "task_id": "TASK1001",
                    "base_trial_id": "trial-1",
                    "regime_cell_id": "family-1",
                    "evaluation_series_id": "randomized",
                    "design_tier": "D1",
                    "design_subtype": "individual_randomized",
                    "assumption_tier": "A1",
                    "context_tier": "C1",
                    "data_preparation": "analysis_ready",
                    "analysis_specification": "locked_sap",
                }
            ],
            "labels": [
                {
                    "task_id": "TASK1001",
                    "prompt_condition": condition,
                    "applicability": "applicable" if condition == "targeted_covariate_structure" else "mismatched",
                    "evidence_basis": ["fixture_evaluator_fact"],
                }
                for condition in (
                    "targeted_covariate_structure",
                    "targeted_survival_assumptions",
                    "targeted_design_structure",
                    "targeted_data_integrity",
                )
            ],
        }
    )


def test_applicability_is_joined_only_after_scoring() -> None:
    endpoint = TrialEvalAblationEndpointRowV1.model_validate(
        {
            "assignment_id": "assignment-targeted",
            "task_id": "TASK1001",
            "context_tier": "C1",
            "data_preparation": "analysis_ready",
            "analysis_specification": "locked_sap",
            "model_id": "model-a",
            "replicate_id": "seed-1",
            "procedure_assistance": "output_contract_only",
            "prompt_condition": "targeted_covariate_structure",
            "submission_interface": "structured",
            "normalization_source": "direct_structured",
            "usable_primary": True,
            "route_match": True,
            "obligations_met": True,
            "credit_eligible_route_count": 1,
            "numeric_result_available": True,
            "result_match": True,
            "primary_analysis_conforms": True,
            "planning_applicable": False,
        }
    )
    joined = join_ablation_evaluator_labels_v1(
        endpoints=(endpoint,),
        evaluator_labels=_evaluator_labels(),
        design="targeted_control",
    )
    assert joined[0].base_trial_id == "trial-1"
    assert joined[0].targeted_applicability == "applicable"

    missing_task = endpoint.model_copy(update={"task_id": "TASK9999"})
    with pytest.raises(ValueError, match="do not contain task"):
        join_ablation_evaluator_labels_v1(
            endpoints=(missing_task,),
            evaluator_labels=_evaluator_labels(),
            design="targeted_control",
        )


def test_factorial_join_requires_base_trial_identity_but_not_applicability() -> None:
    endpoint = TrialEvalAblationEndpointRowV1.model_validate(
        {
            "assignment_id": "assignment-neutral",
            "task_id": "TASK2F35B95DDAC94E8BA70C06C554D995DC",
            "context_tier": "C1",
            "data_preparation": "analysis_ready",
            "analysis_specification": "locked_sap",
            "model_id": "model-a",
            "replicate_id": "seed-1",
            "procedure_assistance": "output_contract_only",
            "prompt_condition": "neutral",
            "submission_interface": "structured",
            "normalization_source": "direct_structured",
            "usable_primary": True,
            "route_match": True,
            "obligations_met": True,
            "credit_eligible_route_count": 1,
            "numeric_result_available": True,
            "result_match": True,
            "primary_analysis_conforms": True,
            "planning_applicable": False,
        }
    )
    evaluator = TrialEvalAblationEvaluatorLabelsV1.model_validate(
        {
            "evaluator_release_sha256": "e" * 64,
            "task_identities": [
                {
                    "task_id": "TASK2F35B95DDAC94E8BA70C06C554D995DC",
                    "base_trial_id": "exact-evaluator-trial",
                    "regime_cell_id": "family-1",
                    "evaluation_series_id": "response_adaptive",
                    "design_tier": "D4",
                    "design_subtype": "group_sequential",
                    "assumption_tier": "A4",
                    "context_tier": "C1",
                    "data_preparation": "analysis_ready",
                    "analysis_specification": "locked_sap",
                }
            ],
            "labels": [],
        }
    )

    joined = join_ablation_evaluator_labels_v1(
        endpoints=(endpoint,),
        evaluator_labels=evaluator,
        design="factorial_interface",
    )
    assert joined[0].base_trial_id == "exact-evaluator-trial"
    assert joined[0].regime_cell_id == "family-1"
    assert joined[0].design_tier == "D4"
    assert joined[0].design_subtype == "group_sequential"
    assert joined[0].assumption_tier == "A4"
    assert joined[0].targeted_applicability is None


def test_evaluator_label_join_preserves_practical_consequence_vector() -> None:
    endpoint = TrialEvalAblationEndpointRowV1.model_validate(
        {
            "assignment_id": "assignment-neutral",
            "task_id": "TASK1001",
            "context_tier": "C1",
            "data_preparation": "analysis_ready",
            "analysis_specification": "locked_sap",
            "model_id": "model-a",
            "replicate_id": "seed-1",
            "procedure_assistance": "output_contract_only",
            "prompt_condition": "neutral",
            "submission_interface": "structured",
            "normalization_source": "direct_structured",
            "usable_primary": True,
            "route_match": True,
            "obligations_met": True,
            "credit_eligible_route_count": 1,
            "numeric_result_available": True,
            "result_match": True,
            "primary_analysis_conforms": True,
            "planning_applicable": True,
            "planning_valid": True,
            "planning_usable_with_primary": True,
            "planning_achieved_power": 0.74,
            "planning_power_shortfall": 0.06,
            "planning_underpowered": True,
            "planning_proportional_participant_deviation": -0.2,
            "planning_log_sample_size_ratio": -0.223143551,
            "planning_event_shortage": 12,
            "planning_excess_events": 0,
            "planning_excess_participants": 0,
            "planning_participant_shortage": 25,
        }
    )

    joined = join_ablation_evaluator_labels_v1(
        endpoints=(endpoint,),
        evaluator_labels=_evaluator_labels(),
        design="factorial_interface",
    )

    assert joined[0].planning_proportional_participant_deviation == -0.2
    assert joined[0].planning_log_sample_size_ratio == pytest.approx(-0.223143551)
    assert joined[0].planning_event_shortage == 12
    assert joined[0].planning_excess_events == 0


def test_targeted_join_rejects_missing_applicability_block() -> None:
    endpoint = TrialEvalAblationEndpointRowV1.model_validate(
        {
            "assignment_id": "assignment-targeted",
            "task_id": "TASK1001",
            "context_tier": "C1",
            "data_preparation": "analysis_ready",
            "analysis_specification": "locked_sap",
            "model_id": "model-a",
            "replicate_id": "seed-1",
            "procedure_assistance": "output_contract_only",
            "prompt_condition": "targeted_covariate_structure",
            "submission_interface": "structured",
            "normalization_source": "direct_structured",
            "usable_primary": True,
            "route_match": True,
            "obligations_met": True,
            "credit_eligible_route_count": 1,
            "numeric_result_available": True,
            "result_match": True,
            "primary_analysis_conforms": True,
            "planning_applicable": False,
        }
    )
    evaluator = TrialEvalAblationEvaluatorLabelsV1.model_validate(
        {
            "evaluator_release_sha256": "e" * 64,
            "task_identities": [
                {
                    "task_id": "TASK1001",
                    "base_trial_id": "trial-1",
                    "regime_cell_id": "family-1",
                    "evaluation_series_id": "randomized",
                    "design_tier": "D1",
                    "design_subtype": "individual_randomized",
                    "assumption_tier": "A1",
                    "context_tier": "C1",
                    "data_preparation": "analysis_ready",
                    "analysis_specification": "locked_sap",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="labels are incomplete"):
        join_ablation_evaluator_labels_v1(
            endpoints=(endpoint,),
            evaluator_labels=evaluator,
            design="targeted_control",
        )
