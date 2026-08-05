"""Adversarial tests for the narrow deterministic grading boundary."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trialagentbench_harness.grading import (
    CanonicalSubmissionV1,
    GradeRecordV1,
    ValidatedScoringKeyV1,
    grade,
)


def test_grade_record_rejects_inconsistent_noncompensatory_state() -> None:
    passed = {
        "release_id": "candidate.v1",
        "item_id": "item.nph.001",
        "usable_primary": True,
        "route_match": True,
        "obligations_met": True,
        "result_match": True,
        "passed": True,
        "gates": [
            {"gate_id": "submission", "status": "passed"},
            {"gate_id": "question", "status": "passed"},
            {"gate_id": "route", "status": "passed"},
            {"gate_id": "evidence", "status": "passed"},
            {"gate_id": "integrity", "status": "not_applicable"},
            {"gate_id": "result", "status": "passed"},
            {"gate_id": "conformance", "status": "passed"},
            {"gate_id": "decision", "status": "not_applicable"},
        ],
        "components": [
            {"component_id": "submission", "status": "passed"},
            {"component_id": "question", "status": "passed"},
            {"component_id": "method", "status": "passed"},
            {"component_id": "evidence", "status": "passed"},
            {"component_id": "integrity", "status": "not_applicable"},
            {"component_id": "result_structure", "status": "passed"},
            {"component_id": "route_comparison", "status": "passed"},
        ],
        "matched_route_id": "route.rmst",
        "failure_codes": (),
        "absolute_error": 0.001,
        "tolerance_ratio": 0.1,
    }
    GradeRecordV1.model_validate(passed)

    with pytest.raises(ValidationError, match="Conformance must equal"):
        GradeRecordV1.model_validate(
            {
                **passed,
                "passed": False,
                "failure_codes": ("numeric_result_outside_tolerance",),
            }
        )
    with pytest.raises(ValidationError, match="reported together"):
        GradeRecordV1.model_validate(
            {
                **passed,
                "absolute_error": None,
            }
        )
    with pytest.raises(ValidationError, match="exactly when route_match"):
        GradeRecordV1.model_validate(
            {
                **passed,
                "matched_route_id": None,
            }
        )


def _key() -> ValidatedScoringKeyV1:
    return ValidatedScoringKeyV1.model_validate(
        {
            "schema_id": "trialagentbench.scoring_key/v1",
            "release_id": "candidate.v1",
            "item_id": "item.nph.001",
            "question_id": "q.survival",
            "context_tier": "C1",
            "credit_eligible_routes": [
                {
                    "route_id": "route.rmst",
                    "signature": {
                        "analysis_population_id": "intention_to_treat",
                        "estimand_id": "rmst_difference_365d",
                        "intercurrent_event_strategy_ids": ["treatment_policy"],
                        "assessment_horizon_days": 365,
                        "treatment_id": "treated",
                        "comparator_id": "control",
                        "endpoint_id": "overall_survival",
                        "effect_scale": "days",
                        "analysis_method_id": "kaplan_meier_rmst",
                    },
                    "method": {
                        "analysis_method_id": "kaplan_meier_rmst",
                        "estimator_family": "kaplan_meier_rmst",
                        "result_kind": "numeric_point",
                        "uncertainty_method": "participant_bootstrap",
                        "design_modifiers": [],
                    },
                    "required_identification_assumptions": [
                        "independent_censoring",
                        "randomization_exchangeability",
                    ],
                    "required_diagnostics": ["follow_up_support"],
                    "planning_calculator_id": "rmst_variance_planning",
                    "target": {
                        "kind": "numeric_point",
                        "value": 12.5,
                        "result_unit": "days",
                        "acceptance_envelope": {
                            "schema_id": "trialagentbench.numerical_acceptance_envelope/v1",
                            "reporting_decimal_places": 0,
                            "independent_max_abs_difference": 0.5,
                            "public_verification_id": "public-replay-1",
                            "independent_verification_ids": ["independent-replay-1"],
                        },
                        "require_confidence_interval": True,
                        "confidence_interval_lower": 8.0,
                        "confidence_interval_upper": 17.0,
                    },
                }
            ],
        }
    )


def _submission(**overrides: object) -> CanonicalSubmissionV1:
    payload: dict[str, object] = {
        "item_id": "item.nph.001",
        "primary": {
            "analysis_population_id": "intention_to_treat",
            "estimand_id": "rmst_difference_365d",
            "intercurrent_event_strategy_ids": ["treatment_policy"],
            "assessment_horizon_days": 365,
            "treatment_id": "treated",
            "comparator_id": "control",
            "endpoint_id": "overall_survival",
            "effect_scale": "days",
            "analysis_method_id": "kaplan_meier_rmst",
        },
        "diagnostic_ids": ["follow_up_support"],
        "result": {
            "kind": "numeric_point",
            "value": 12.7,
            "result_unit": "days",
            "confidence_interval_lower": 8.0,
            "confidence_interval_upper": 17.0,
        },
    }
    payload.update(overrides)
    return CanonicalSubmissionV1.model_validate(payload)


def test_exact_primary_route_and_numeric_result_pass() -> None:
    result = grade(_key(), _submission())

    assert result.passed
    assert result.matched_route_id == "route.rmst"
    assert result.tolerance_ratio == 0.1999999999999993


def test_point_estimate_cannot_rescue_wrong_confidence_interval() -> None:
    submission = _submission(result=_submission().result.model_copy(update={"confidence_interval_lower": 6.0}))

    result = grade(_key(), submission)

    assert not result.passed
    assert result.failure_codes == ("confidence_interval_outside_tolerance",)
    assert result.absolute_error == 2.0
    assert result.tolerance_ratio == 2.0


def test_point_estimate_failure_precedes_confidence_interval_failure_within_route() -> None:
    submission = _submission(
        result=_submission().result.model_copy(update={"value": 15.0, "confidence_interval_lower": 5.0})
    )

    result = grade(_key(), submission)

    assert result.failure_codes == ("numeric_result_outside_tolerance",)
    assert result.absolute_error == 3.0


def test_term_mention_cannot_rescue_wrong_primary_route() -> None:
    primary = _submission().primary.model_copy(update={"analysis_method_id": "cox_proportional_hazards"})
    result = grade(_key(), _submission(primary=primary))

    assert not result.passed
    assert result.failure_codes == ("unrecognized_primary_route",)


def test_reduced_context_accepts_each_qualified_route_for_the_same_estimand() -> None:
    key_payload = _key().model_dump(mode="json")
    key_payload["context_tier"] = "C4"
    alternative = {
        **key_payload["credit_eligible_routes"][0],
        "route_id": "route.rmst_pseudo_observation",
    }
    alternative["signature"] = {
        **alternative["signature"],
        "analysis_method_id": "pseudo_observation_rmst",
    }
    alternative["method"] = {
        **alternative["method"],
        "analysis_method_id": "pseudo_observation_rmst",
        "estimator_family": "pseudo_observation_rmst",
        "uncertainty_method": "sandwich_covariance",
    }
    alternative["target"] = {
        **alternative["target"],
        "value": 13.1,
        "acceptance_envelope": {
            **alternative["target"]["acceptance_envelope"],
            "public_verification_id": "public-replay-2",
            "independent_verification_ids": ["independent-replay-2"],
        },
    }
    key_payload["credit_eligible_routes"].append(alternative)
    key = ValidatedScoringKeyV1.model_validate(key_payload)

    default_submission = _submission()
    assert grade(key, default_submission).matched_route_id == "route.rmst"

    alternative_submission = default_submission.model_copy(
        update={
            "primary": key.credit_eligible_routes[1].signature,
            "result": default_submission.result.model_copy(update={"value": 13.1}),
        }
    )
    alternative_grade = grade(key, alternative_submission)
    assert alternative_grade.passed
    assert alternative_grade.matched_route_id == "route.rmst_pseudo_observation"

    unqualified_hybrid = alternative_submission.model_copy(
        update={
            "primary": alternative_submission.primary.model_copy(update={"analysis_method_id": "unregistered_hybrid"})
        }
    )
    assert grade(key, unqualified_hybrid).failure_codes == ("unrecognized_primary_route",)


def test_plural_routes_cannot_mix_estimands_or_effect_scales() -> None:
    key_payload = _key().model_dump(mode="json")
    alternative = {
        **key_payload["credit_eligible_routes"][0],
        "route_id": "route.hazard_ratio",
    }
    alternative["signature"] = {
        **alternative["signature"],
        "estimand_id": "marginal_hazard_ratio",
        "effect_scale": "hazard_ratio",
    }
    key_payload["credit_eligible_routes"].append(alternative)

    with pytest.raises(ValidationError, match="one precise estimand"):
        ValidatedScoringKeyV1.model_validate(key_payload)


def test_missing_required_diagnostic_fails_before_numeric_credit() -> None:
    result = grade(_key(), _submission(diagnostic_ids=()))

    assert not result.passed
    assert result.route_match
    assert not result.obligations_met
    assert result.failure_codes == ("missing_required_diagnostic",)


def test_primary_failure_and_independent_components_are_both_reported() -> None:
    result = grade(
        _key(),
        _submission(
            diagnostic_ids=(),
            result=_submission().result.model_copy(update={"value": 15.0}),
        ),
    )
    components = {row.component_id: row for row in result.components}

    assert result.first_failure_gate == "evidence"
    assert components["evidence"].failure_code == "missing_required_diagnostic"
    assert components["result_structure"].status == "passed"
    assert components["route_comparison"].failure_code == "numeric_result_outside_tolerance"
    assert next(row for row in result.gates if row.gate_id == "conformance").status == "not_reached"


def test_lucky_numeric_coincidence_on_wrong_scale_is_rejected() -> None:
    primary = _submission().primary.model_copy(update={"effect_scale": "log_hazard_ratio"})
    result = grade(_key(), _submission(primary=primary))

    assert not result.passed
    assert not result.route_match


def test_lucky_numeric_coincidence_for_wrong_population_is_rejected() -> None:
    primary = _submission().primary.model_copy(update={"analysis_population_id": "per_protocol"})
    result = grade(_key(), _submission(primary=primary))

    assert not result.passed
    assert result.failure_codes == ("unrecognized_primary_question",)


def test_numeric_result_outside_prespecified_tolerance_fails() -> None:
    result_payload = _submission().result.model_copy(update={"value": 15.0})
    result = grade(_key(), _submission(result=result_payload))

    assert not result.passed
    assert result.obligations_met
    assert result.failure_codes == ("numeric_result_outside_tolerance",)


def test_vector_route_requires_exact_components_and_values() -> None:
    key_payload = _key().model_dump(mode="json")
    key_payload["credit_eligible_routes"][0]["signature"].update(
        {
            "estimand_id": "piecewise_log_hazard_ratio",
            "effect_scale": "log_hazard_ratio_by_interval",
            "analysis_method_id": "piecewise_exponential",
        }
    )
    key_payload["credit_eligible_routes"][0]["method"].update(
        {
            "analysis_method_id": "piecewise_exponential",
            "estimator_family": "piecewise_exponential",
            "result_kind": "numeric_vector",
            "uncertainty_method": "sandwich_covariance",
        }
    )
    key_payload["credit_eligible_routes"][0]["target"] = {
        "kind": "numeric_vector",
        "components": [
            {"name": "early", "value": -0.4},
            {"name": "late", "value": -0.1},
        ],
        "result_unit": "log_hazard_ratio",
        "acceptance_envelope": {
            "schema_id": "trialagentbench.numerical_acceptance_envelope/v1",
            "reporting_decimal_places": 1,
            "independent_max_abs_difference": 0.0,
            "public_verification_id": "public-replay-vector",
            "independent_verification_ids": ["independent-replay-vector"],
        },
    }
    key = ValidatedScoringKeyV1.model_validate(key_payload)
    submission = CanonicalSubmissionV1.model_validate(
        {
            "item_id": key.item_id,
            "primary": key.credit_eligible_routes[0].signature,
            "diagnostic_ids": ["follow_up_support"],
            "result": {
                "kind": "numeric_vector",
                "components": [
                    {"name": "early", "value": -0.39},
                    {"name": "late", "value": -0.11},
                ],
                "result_unit": "log_hazard_ratio",
            },
        }
    )

    assert grade(key, submission).passed
    missing = submission.model_copy(
        update={"result": submission.result.model_copy(update={"components": (submission.result.components[0],)})}
    )
    assert grade(key, missing).failure_codes == ("vector_components_mismatch",)


def test_sensitivity_set_grid_is_fixed_by_the_selected_method() -> None:
    key_payload = _key().model_dump(mode="json")
    route = key_payload["credit_eligible_routes"][0]
    route["signature"].update(
        {
            "estimand_id": "risk_difference_365d",
            "effect_scale": "risk_difference_tau",
            "analysis_method_id": "bounds_delta_grid",
        }
    )
    route["method"].update(
        {
            "analysis_method_id": "bounds_delta_grid",
            "estimator_family": "bounds",
            "result_kind": "sensitivity_set",
            "uncertainty_method": "identified_set",
            "sensitivity_parameters": [0.05, 0.10, 0.20],
        }
    )
    components = [
        {"name": "delta_0.05_lower", "value": -0.10},
        {"name": "delta_0.05_upper", "value": 0.02},
        {"name": "delta_0.10_lower", "value": -0.15},
        {"name": "delta_0.10_upper", "value": 0.07},
        {"name": "delta_0.20_lower", "value": -0.25},
        {"name": "delta_0.20_upper", "value": 0.17},
    ]
    route["target"] = {
        "kind": "numeric_vector",
        "components": components,
        "result_unit": "probability_difference",
        "acceptance_envelope": {
            "schema_id": "trialagentbench.numerical_acceptance_envelope/v1",
            "reporting_decimal_places": 3,
            "independent_max_abs_difference": 1e-12,
            "public_verification_id": "public-replay-sensitivity-set",
            "independent_verification_ids": ["independent-replay-sensitivity-set"],
        },
    }
    key = ValidatedScoringKeyV1.model_validate(key_payload)
    submission = CanonicalSubmissionV1.model_validate(
        {
            "item_id": key.item_id,
            "primary": key.credit_eligible_routes[0].signature,
            "diagnostic_ids": ["follow_up_support"],
            "result": {
                "kind": "numeric_vector",
                "components": components,
                "result_unit": "probability_difference",
            },
        }
    )

    assert grade(key, submission).passed

    wrong_method = submission.primary.model_copy(update={"analysis_method_id": "bounds_other_grid"})
    wrong_route = grade(key, submission.model_copy(update={"primary": wrong_method}))
    assert wrong_route.failure_codes == ("unrecognized_primary_route",)

    key_payload["credit_eligible_routes"][0]["method"]["sensitivity_parameters"] = []
    with pytest.raises(ValidationError, match="required exactly"):
        ValidatedScoringKeyV1.model_validate(key_payload)


def test_hypothesis_test_route_requires_decision_and_p_value() -> None:
    key_payload = _key().model_dump(mode="json")
    key_payload["credit_eligible_routes"][0]["signature"].update(
        {
            "estimand_id": "survival_curve_equality",
            "effect_scale": "p_value",
            "analysis_method_id": "weighted_logrank",
        }
    )
    key_payload["credit_eligible_routes"][0]["method"].update(
        {
            "analysis_method_id": "weighted_logrank",
            "estimator_family": "weighted_logrank",
            "result_kind": "statistical_test",
            "uncertainty_method": "asymptotic_null",
        }
    )
    key_payload["credit_eligible_routes"][0]["target"] = {
        "kind": "statistical_test",
        "p_value": 0.02,
        "reject_null": True,
        "acceptance_envelope": {
            "schema_id": "trialagentbench.numerical_acceptance_envelope/v1",
            "reporting_decimal_places": 2,
            "independent_max_abs_difference": 0.005,
            "public_verification_id": "public-replay-test",
            "independent_verification_ids": ["independent-replay-test"],
        },
    }
    key = ValidatedScoringKeyV1.model_validate(key_payload)
    submission = CanonicalSubmissionV1.model_validate(
        {
            "item_id": key.item_id,
            "primary": key.credit_eligible_routes[0].signature,
            "diagnostic_ids": ["follow_up_support"],
            "result": {
                "kind": "statistical_test",
                "p_value": 0.021,
                "reject_null": True,
            },
        }
    )

    assert grade(key, submission).passed
    wrong_decision = submission.model_copy(
        update={"result": submission.result.model_copy(update={"reject_null": False})}
    )
    assert grade(key, wrong_decision).failure_codes == ("test_decision_mismatch",)


def _c5_key_and_submission() -> tuple[ValidatedScoringKeyV1, CanonicalSubmissionV1]:
    key_payload = _key().model_dump(mode="json")
    key_payload["context_tier"] = "C5"
    checksum = "a" * 64
    integrity = {
        "condition_id": "exact_transport_row_duplication_v1",
        "affected_domain": "data/raw/endpoint_reports.parquet",
        "compound_key_fields": ["USUBJID", "REPORT_DTC", "ENDPOINT_TERM"],
        "observed_duplicate_group_count": 2,
        "observed_extra_row_count": 2,
        "repair_action": "remove_one_exact_duplicate_copy",
        "repair_status": "repaired",
        "post_repair_data_checksum": checksum,
        "analysis_input_data_checksum": checksum,
    }
    key_payload["data_integrity_target"] = integrity
    key = ValidatedScoringKeyV1.model_validate(key_payload)
    submission = CanonicalSubmissionV1.model_validate(
        {
            "item_id": key.item_id,
            "primary": key.credit_eligible_routes[0].signature,
            "diagnostic_ids": ["follow_up_support"],
            "data_integrity_record": integrity,
            "result": {
                "kind": "numeric_point",
                "value": 12.5,
                "result_unit": "days",
                "confidence_interval_lower": 8.0,
                "confidence_interval_upper": 17.0,
            },
        }
    )
    return key, submission


def test_c5_integrity_is_noncompensatory_before_numeric_comparison() -> None:
    key, submission = _c5_key_and_submission()
    assert grade(key, submission).passed

    missing = submission.model_copy(update={"data_integrity_record": None})
    missing_grade = grade(key, missing)
    assert missing_grade.failure_codes == ("missing_data_integrity_record",)
    assert missing_grade.route_match is True
    assert missing_grade.obligations_met is False
    assert missing_grade.result_match is False

    wrong_count = submission.model_copy(
        update={
            "data_integrity_record": submission.data_integrity_record.model_copy(
                update={"observed_extra_row_count": 1}
            )
        }
    )
    assert grade(key, wrong_count).failure_codes == ("data_integrity_counts_mismatch",)


def test_non_c5_submission_cannot_add_an_unscored_integrity_claim() -> None:
    _key_c5, submission = _c5_key_and_submission()
    key = _key()
    unexpected = submission.model_copy(
        update={
            "item_id": key.item_id,
            "primary": key.credit_eligible_routes[0].signature,
        }
    )
    assert grade(key, unexpected).failure_codes == ("unexpected_data_integrity_record",)
