"""Behavior tests for canonical method-neutral submission contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast, get_args

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError

from trialagentbench_harness.contracts.scoring.diagnostic_registry import diagnostic_key_by_assumption_id_v1
from trialagentbench_harness.contracts.submission import (
    TrialEvalSubmissionV1,
    trialeval_submission_schema,
    trialeval_submission_shape_catalogue,
    validate_trialeval_required_deliverables_v1,
)
from trialagentbench_harness.contracts.submission.models import DiagnosticIdV1
from trialagentbench_harness.contracts.submission.schema import (
    trialdev_randomized_phase_analysis_schema,
    write_public_submission_contracts,
)
from trialagentbench_harness.trialdev.bridge import parse_phase_analysis, parse_request
from trialagentbench_harness.trialdev.participant_submission import (
    build_observational_review_v1,
    build_phase_decision_v1,
)


def _scalar_submission() -> dict[str, object]:
    return {
        "task_id": "TASK1",
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
                "analysis_method_id": "km_rmst_bootstrap",
                "implementation": "area under Kaplan-Meier curve",
                "qualifications": [
                    "independent_censoring",
                    "randomization_exchangeability",
                ],
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
            "evidence_ids": ["diag-1"],
        },
        "evidence": [
            {
                "evidence_id": "diag-1",
                "evidence_type": "diagnostic",
                "principle": "proportional_hazards",
                "operation": "assessment",
                "diagnostic_id": "proportional_hazards_public",
                "target": "proportional_hazards",
                "result": {
                    "kind": "diagnostic_test",
                    "statistic": {
                        "metric_id": "scaled_schoenfeld_rank_slope",
                        "value": -0.8,
                        "unit": "scaled_residual",
                        "decimal_places": 1,
                    },
                    "p_value": {
                        "metric_id": "schoenfeld_rank_test_p_value",
                        "value": 0.09,
                        "unit": "probability",
                        "decimal_places": 2,
                    },
                },
                "interpretation": "a constant effect model is inadequate",
                "source_artifacts": ["data/adtte.parquet"],
            }
        ],
        "limitations": ["The horizon is protocol-defined."],
    }


def test_scalar_submission_round_trips() -> None:
    submission = TrialEvalSubmissionV1.model_validate(_scalar_submission())

    assert submission.primary_analysis.result.kind == "scalar"


def test_trialeval_schema_and_balanced_shape_catalogue_share_one_model() -> None:
    schema = trialeval_submission_schema()
    catalogue = trialeval_submission_shape_catalogue()

    assert "examples" not in schema
    assert set(catalogue.primary_submissions) == {
        "abstention:non_identification",
        "identification_bound:identified_interval",
        "limitation:non_identification",
        "numeric_point:scalar",
        "numeric_vector:vector",
        "sensitivity_set:vector",
        "statistical_test:statistical_test",
    }
    Draft202012Validator.check_schema(schema)
    for example in catalogue.primary_submissions.values():
        payload = example.model_dump(mode="json")
        TrialEvalSubmissionV1.model_validate(payload)
        Draft202012Validator(schema).validate(payload)


def test_diagnostic_probability_constraints_are_visible_in_json_schema() -> None:
    payload = _scalar_submission()
    schema = trialeval_submission_schema()
    Draft202012Validator(schema).validate(payload)
    definitions = schema["$defs"]
    assert definitions["DiagnosticTestResultV1"]["properties"]["p_value"] == {"$ref": "#/$defs/ProbabilityMeasureV1"}
    assert definitions["ProbabilityMeasureV1"]["properties"]["unit"]["const"] == "probability"
    assert definitions["ProbabilityMeasureV1"]["properties"]["value"]["minimum"] == 0.0
    assert definitions["ProbabilityMeasureV1"]["properties"]["value"]["maximum"] == 1.0
    assert (
        "Required for sensitivity and supporting-analysis"
        in definitions["EvidenceRecordV1"]["properties"]["estimator"]["description"]
    )

    invalid_unit = copy.deepcopy(payload)
    invalid_unit["evidence"][0]["result"]["p_value"]["unit"] = "p_value"  # type: ignore[index]
    with pytest.raises(ValidationError, match="probability"):
        TrialEvalSubmissionV1.model_validate(invalid_unit)
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schema).validate(invalid_unit)

    invalid_value = copy.deepcopy(payload)
    invalid_value["evidence"][0]["result"]["p_value"]["value"] = 1.1  # type: ignore[index]
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        TrialEvalSubmissionV1.model_validate(invalid_value)
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schema).validate(invalid_value)


@pytest.mark.parametrize(
    ("mutation", "runtime_message"),
    (
        ("diagnostic_without_id", "require diagnostic_id"),
        ("diagnostic_with_estimator", "must omit it"),
        ("supporting_without_estimator", "require estimator"),
        ("supporting_with_diagnostic_result", "require a diagnostic result shape"),
    ),
)
def test_evidence_type_constraints_match_runtime_and_json_schema(
    mutation: str,
    runtime_message: str,
) -> None:
    payload = _scalar_submission()
    evidence = cast(dict[str, object], cast(list[object], payload["evidence"])[0])
    primary_analysis = cast(dict[str, object], payload["primary_analysis"])
    if mutation == "diagnostic_without_id":
        evidence.pop("diagnostic_id")
    elif mutation == "diagnostic_with_estimator":
        evidence["estimator"] = {
            "analysis_method_id": "km_rmst_bootstrap",
            "implementation": "executed estimator",
            "qualifications": ["independent_censoring"],
        }
    elif mutation == "supporting_without_estimator":
        evidence["evidence_type"] = "supporting_analysis"
        evidence.pop("diagnostic_id")
        evidence["result"] = copy.deepcopy(primary_analysis["result"])
    else:
        evidence["evidence_type"] = "supporting_analysis"
        evidence.pop("diagnostic_id")
        evidence["estimator"] = {
            "analysis_method_id": "km_rmst_bootstrap",
            "implementation": "executed estimator",
            "qualifications": ["independent_censoring"],
        }

    with pytest.raises(ValidationError, match=runtime_message):
        TrialEvalSubmissionV1.model_validate(payload)
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(trialeval_submission_schema()).validate(payload)


def test_semantic_requirements_reject_missing_context_specific_deliverables() -> None:
    submission = TrialEvalSubmissionV1.model_validate(_scalar_submission())

    with pytest.raises(ValueError, match="reconstruction"):
        validate_trialeval_required_deliverables_v1(
            submission,
            required_deliverables=("primary_analysis", "evidence", "limitations", "reconstruction"),
        )

    with pytest.raises(ValueError, match="data_integrity_record"):
        validate_trialeval_required_deliverables_v1(
            submission,
            required_deliverables=("primary_analysis", "evidence", "limitations", "data_integrity_record"),
        )


def test_semantic_requirements_accept_canonical_core_submission() -> None:
    submission = TrialEvalSubmissionV1.model_validate(_scalar_submission())

    assert (
        validate_trialeval_required_deliverables_v1(
            submission,
            required_deliverables=("primary_analysis", "evidence", "limitations"),
        )
        is submission
    )


def test_trialeval_submission_rejects_orphan_evidence() -> None:
    payload = _scalar_submission()
    primary = payload["primary_analysis"]
    assert isinstance(primary, dict)
    primary["evidence_ids"] = []

    with pytest.raises(ValidationError, match="evidence records must be linked"):
        TrialEvalSubmissionV1.model_validate(payload)


def test_submission_diagnostic_ids_match_the_score_registry() -> None:
    """Every passed public diagnostic identity must be executable by the scorer."""

    assert set(get_args(DiagnosticIdV1)) == set(diagnostic_key_by_assumption_id_v1().values())


def test_diagnostic_evidence_rejects_primary_result_shapes() -> None:
    payload = _scalar_submission()
    evidence = payload["evidence"]
    assert isinstance(evidence, list)
    record = evidence[0]
    assert isinstance(record, dict)
    record["result"] = {
        "kind": "identified_interval",
        "lower": 0.2,
        "upper": 0.8,
        "effect_scale": "diagnostic_summary",
        "unit": "dimensionless",
        "interpretation": "departure is material",
    }

    with pytest.raises(ValidationError, match="require a diagnostic result shape"):
        TrialEvalSubmissionV1.model_validate(payload)


def test_event_driven_planning_is_bound_to_the_scalar_log_hr_primary() -> None:
    payload = _scalar_submission()
    primary = payload["primary_analysis"]
    assert isinstance(primary, dict)
    primary["estimand"]["estimand_id"] = "primary_itt"  # type: ignore[index]
    primary["result"] = {
        "kind": "scalar",
        "value": -0.4,
        "effect_scale": "log_hr",
        "unit": "log_hazard_ratio",
        "interval": {"lower": -0.6, "upper": -0.2, "confidence_level": 0.95},
    }
    payload["planning"] = {
        "method_id": "schoenfeld_logrank_v1",
        "estimand_id": "primary_itt",
        "alpha_two_sided": 0.05,
        "power": 0.90,
        "treated_allocation_fraction": 0.50,
        "event_probability": 0.40,
        "followup_horizon_dy": 365.0,
        "multiplicity_adjustment": "none",
        "required_events": 263,
        "target_sample_size": 658,
        "sensitivity": [
            {"event_probability": 0.35, "target_sample_size": 752},
            {"event_probability": 0.45, "target_sample_size": 585},
        ],
    }
    assert TrialEvalSubmissionV1.model_validate(payload).planning is not None

    payload["planning"]["estimand_id"] = "different"  # type: ignore[index]
    with pytest.raises(ValidationError, match="planning estimand_id"):
        TrialEvalSubmissionV1.model_validate(payload)


def test_checked_in_public_contracts_equal_generated_contracts(tmp_path: Path) -> None:
    write_public_submission_contracts(tmp_path)
    package_root = Path(__file__).resolve().parents[2]

    for name in (
        "agent_output_schema.json",
        "eval_spec.json",
        "examples/submissions/trialeval_shapes.json",
    ):
        assert (tmp_path / name).read_bytes() == (package_root / name).read_bytes()


def test_randomized_phase_analysis_contract_excludes_observational_utility() -> None:
    schema = trialdev_randomized_phase_analysis_schema()
    properties = schema["properties"]

    assert isinstance(properties, dict)
    assert "candidate_utility_estimates" not in properties
    assert properties["analysis_rationale"]["maxLength"] == 2000


def test_public_submission_examples_execute_against_runtime_contracts() -> None:
    root = Path(__file__).resolve().parents[2]
    examples = root / "examples" / "submissions"
    public_schemas = json.loads((root / "agent_output_schema.json").read_text(encoding="utf-8"))

    trialeval_shapes = json.loads((examples / "trialeval_shapes.json").read_text(encoding="utf-8"))
    for trialeval in trialeval_shapes["primary_submissions"].values():
        TrialEvalSubmissionV1.model_validate(trialeval)
        Draft202012Validator(public_schemas["trialeval"]).validate(trialeval)

    observational = json.loads((examples / "trialdev_observational_review.json").read_text(encoding="utf-8"))
    build_observational_review_v1(
        observational,
        source_artifact_checksums={"public/observational_extract.parquet": "a" * 64},
        identification_artifact_checksums={},
    )
    Draft202012Validator(public_schemas["trialdev_observational_review"]).validate(observational)

    request = json.loads((examples / "trialdev_phase_request.json").read_text(encoding="utf-8"))
    parsed_request, request_error = parse_request(
        request,
        scenario_id="example_scenario",
        phase_id="phase2",
    )
    assert request_error is None
    assert parsed_request is not None
    Draft202012Validator(public_schemas["trialdev_phase_request"]).validate(request)

    analysis = json.loads((examples / "trialdev_phase_analysis.json").read_text(encoding="utf-8"))
    parsed_analysis, analysis_error = parse_phase_analysis(
        analysis,
        scenario_id="example_scenario",
        phase_id="phase2",
        request_checksum="a" * 64,
        trial_output_checksum="b" * 64,
        effect_source_artifact_checksums={"trial_output/endpoints.parquet": "c" * 64},
        safety_source_artifact_checksums={"trial_output/safety.parquet": "d" * 64},
    )
    assert analysis_error is None
    assert parsed_analysis is not None
    Draft202012Validator(public_schemas["trialdev_phase_analysis"]).validate(analysis)

    decision = json.loads((examples / "trialdev_phase_decision.json").read_text(encoding="utf-8"))
    build_phase_decision_v1(
        decision,
        scenario_id="example_scenario",
        phase_id="phase2",
        request_checksum="a" * 64,
        analysis_checksum="d" * 64,
    )
    Draft202012Validator(public_schemas["trialdev_phase_decision"]).validate(decision)


def test_phase_analysis_rejects_participant_supplied_custody() -> None:
    root = Path(__file__).resolve().parents[2]
    analysis = json.loads((root / "examples" / "submissions" / "trialdev_phase_analysis.json").read_text())

    analysis["request_checksum"] = "f" * 64
    parsed, error = parse_phase_analysis(
        analysis,
        scenario_id="example_scenario",
        phase_id="phase2",
        request_checksum="a" * 64,
        trial_output_checksum="b" * 64,
        effect_source_artifact_checksums={"trial_output/endpoints.parquet": "c" * 64},
        safety_source_artifact_checksums={"trial_output/safety.parquet": "d" * 64},
    )

    assert parsed is None
    assert error is not None
    assert "harness-owned fields" in error


@pytest.mark.parametrize(
    ("result", "result_kind", "expected_kind"),
    [
        (
            {
                "kind": "identified_interval",
                "lower": -0.1,
                "upper": 0.3,
                "effect_scale": "risk_difference",
                "unit": "proportion",
                "interpretation": "identified set",
            },
            "identification_bound",
            "identified_interval",
        ),
        (
            {
                "kind": "vector",
                "points": [
                    {"component_id": "day30", "index": 30, "value": 0.9},
                    {"component_id": "day60", "index": 60, "value": 0.8},
                ],
                "index_unit": "days",
                "effect_scale": "survival_probability",
                "unit": "proportion",
            },
            "numeric_vector",
            "vector",
        ),
        (
            {
                "kind": "statistical_test",
                "statistic": 2.4,
                "p_value": 0.02,
                "reject_null": True,
                "effect_scale": "weighted_logrank_test",
                "unit": "standardized_statistic",
                "rho": 0,
                "gamma": 1,
            },
            "statistical_test",
            "statistical_test",
        ),
        (
            {
                "kind": "non_identification",
                "conclusion_code": "effect_not_identified",
                "effect_scale": "risk_difference",
                "unit": "proportion",
                "reason": "unmeasured confounding is unrestricted",
                "additional_assumption_required": "a bounded confounding-strength assumption",
            },
            "limitation",
            "non_identification",
        ),
    ],
)
def test_all_non_scalar_primary_shapes_validate(
    result: dict[str, object],
    result_kind: str,
    expected_kind: str,
) -> None:
    payload = _scalar_submission()
    primary = payload["primary_analysis"]
    assert isinstance(primary, dict)
    primary["result_kind"] = result_kind
    primary["result"] = result

    submission = TrialEvalSubmissionV1.model_validate(payload)

    assert submission.primary_analysis.result.kind == expected_kind


def test_sensitivity_set_uses_a_complete_method_identity_not_a_submitted_grid() -> None:
    payload = _scalar_submission()
    primary = cast(dict[str, object], payload["primary_analysis"])
    estimator = cast(dict[str, object], primary["estimator"])
    estimator["analysis_method_id"] = "bounds_delta_005_010_020"
    primary["result_kind"] = "sensitivity_set"
    primary["result"] = {
        "kind": "vector",
        "points": [
            {"component_id": "delta_0.05_lower", "index": 0, "value": -0.10},
            {"component_id": "delta_0.05_upper", "index": 1, "value": 0.02},
            {"component_id": "delta_0.10_lower", "index": 2, "value": -0.15},
            {"component_id": "delta_0.10_upper", "index": 3, "value": 0.07},
            {"component_id": "delta_0.20_lower", "index": 4, "value": -0.25},
            {"component_id": "delta_0.20_upper", "index": 5, "value": 0.17},
        ],
        "index_unit": "sensitivity_set_component",
        "effect_scale": "risk_difference_tau",
        "unit": "probability_difference",
    }

    submission = TrialEvalSubmissionV1.model_validate(payload)
    assert submission.primary_analysis.estimator.analysis_method_id == "bounds_delta_005_010_020"

    estimator["sensitivity_parameters"] = [
        {"value": 0.05, "unit": "probability"},
    ]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrialEvalSubmissionV1.model_validate(payload)


def test_submission_rejects_unresolved_evidence_reference() -> None:
    payload = _scalar_submission()
    primary = payload["primary_analysis"]
    assert isinstance(primary, dict)
    primary["evidence_ids"] = ["missing"]

    with pytest.raises(ValidationError, match="unknown evidence IDs"):
        TrialEvalSubmissionV1.model_validate(payload)


def test_submission_rejects_ambiguous_horizon() -> None:
    payload = _scalar_submission()
    primary = payload["primary_analysis"]
    assert isinstance(primary, dict)
    estimand = primary["estimand"]
    assert isinstance(estimand, dict)
    estimand["horizon_not_applicable_reason"] = "not applicable"

    with pytest.raises(ValidationError, match="declare a horizon or explain"):
        TrialEvalSubmissionV1.model_validate(payload)


def test_submission_rejects_point_outside_interval() -> None:
    payload = _scalar_submission()
    primary = payload["primary_analysis"]
    assert isinstance(primary, dict)
    result = primary["result"]
    assert isinstance(result, dict)
    result["value"] = 100.0

    with pytest.raises(ValidationError, match="must lie within"):
        TrialEvalSubmissionV1.model_validate(payload)


def test_submission_rejects_answer_specific_extra_field() -> None:
    payload = _scalar_submission()
    payload["assessed_ph_assumption"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrialEvalSubmissionV1.model_validate(payload)


def test_submission_rejects_free_form_analysis_method_identity() -> None:
    """Canonical grading identities cannot be supplied as free-form method prose."""

    payload = _scalar_submission()
    primary = payload["primary_analysis"]
    assert isinstance(primary, dict)
    estimator = primary["estimator"]
    assert isinstance(estimator, dict)
    estimator["analysis_method_id"] = "Kaplan-Meier RMST"

    with pytest.raises(ValidationError, match="String should match pattern"):
        TrialEvalSubmissionV1.model_validate(payload)


def test_submission_rejects_nonpositive_hazard_ratio() -> None:
    payload = _scalar_submission()
    primary = payload["primary_analysis"]
    assert isinstance(primary, dict)
    result = primary["result"]
    assert isinstance(result, dict)
    result.update(
        {
            "value": 0.0,
            "effect_scale": "hazard_ratio",
            "unit": "hazard_ratio",
            "interval": {"lower": 0.0, "upper": 1.2, "confidence_level": 0.95},
        }
    )

    with pytest.raises(ValidationError, match="hazard-ratio"):
        TrialEvalSubmissionV1.model_validate(payload)
