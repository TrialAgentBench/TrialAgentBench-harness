"""Public schemas and examples generated from canonical submission models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import JsonValue

from trialagentbench_harness.contracts.submission.models import (
    DataIntegrityRecordV1,
    EvidenceRecordV1,
    PlanningResultV1,
    PrimaryResultKindV1,
    ReconstructionSummaryV1,
    TrialEvalSubmissionV1,
    validate_trialeval_required_deliverables_v1,
)
from trialagentbench_harness.io.checksums import canonical_payload_sha256


class TrialEvalOptionalSubmissionFragmentsV1(BaseModel):
    """Task-dependent structured deliverables omitted from ordinary tasks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reconstruction: ReconstructionSummaryV1
    data_integrity_record: DataIntegrityRecordV1
    planning: PlanningResultV1


class TrialEvalSubmissionShapeCatalogueV1(BaseModel):
    """Method-neutral examples generated from the canonical submission models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialeval_submission_shapes/v1"] = (
        "trialagentbench.trialeval_submission_shapes/v1"
    )
    purpose: Literal["illustrate_global_wire_shapes_without_selecting_an_item_answer"] = (
        "illustrate_global_wire_shapes_without_selecting_an_item_answer"
    )
    primary_submissions: dict[str, TrialEvalSubmissionV1]
    evidence_records: dict[str, EvidenceRecordV1]
    optional_fragments: TrialEvalOptionalSubmissionFragmentsV1
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _complete_and_checksummed(self) -> Self:
        primary = tuple(sorted(self.primary_submissions))
        expected_primary = (
            "abstention:non_identification",
            "identification_bound:identified_interval",
            "limitation:non_identification",
            "numeric_point:scalar",
            "numeric_vector:vector",
            "sensitivity_set:vector",
            "statistical_test:statistical_test",
        )
        if primary != expected_primary:
            raise ValueError("TrialEval primary shape catalogue is incomplete")
        observed_result_kinds = {
            submission.primary_analysis.result_kind for submission in self.primary_submissions.values()
        }
        if observed_result_kinds != set(get_args(PrimaryResultKindV1)):
            raise ValueError("TrialEval shape catalogue and public result-kind vocabulary disagree")
        expected_evidence = {
            "diagnostic_summary",
            "diagnostic_test",
            "factual_premise",
            "identified_interval",
            "non_identification",
            "scalar",
            "statistical_test",
            "vector",
        }
        observed_evidence = {record.result.kind for record in self.evidence_records.values()}
        if observed_evidence != expected_evidence or set(self.evidence_records) != expected_evidence:
            raise ValueError("TrialEval evidence shape catalogue is incomplete")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        digest = canonical_payload_sha256(payload)
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("TrialEval shape catalogue checksum does not match its payload")
        object.__setattr__(self, "checksum", digest)
        return self


def _trialeval_scalar_example() -> dict[str, JsonValue]:
    """Return a valid, deliberately non-task-specific TrialEval example."""

    example = TrialEvalSubmissionV1.model_validate(
        {
            "task_id": "TASK_EXAMPLE",
            "primary_analysis": {
                "declared_primary": True,
                "estimand": {
                    "estimand_id": "example_estimand",
                    "population_id": "example_population",
                    "treatment_id": "example_treatment",
                    "comparator_id": "example_comparator",
                    "endpoint_id": "example_endpoint",
                    "intercurrent_event_strategy_ids": ["example_strategy"],
                    "horizon": {"value": 1.0, "unit": "days"},
                },
                "estimator": {
                    "analysis_method_id": "example_analysis_method",
                    "implementation": "executed example estimator",
                },
                "result_kind": "numeric_point",
                "result": {
                    "kind": "scalar",
                    "value": 0.0,
                    "effect_scale": "log_hr",
                    "unit": "log_hazard_ratio",
                    "interval": {"lower": -1.0, "upper": 1.0, "confidence_level": 0.95},
                },
                "favorable_direction": "neither",
                "evidence_ids": ["example_evidence"],
            },
            "evidence": [
                {
                    "evidence_id": "example_evidence",
                    "evidence_type": "supporting_analysis",
                    "principle": "uncertainty",
                    "operation": "estimation",
                    "estimator": {
                        "analysis_method_id": "example_analysis_method",
                        "implementation": "executed example supporting estimator",
                    },
                    "target": "example supporting target",
                    "result": {
                        "kind": "scalar",
                        "value": 0.0,
                        "effect_scale": "log_hr",
                        "unit": "log_hazard_ratio",
                        "interval": {"lower": -1.0, "upper": 1.0, "confidence_level": 0.95},
                    },
                    "interpretation": "Example structure only; replace every value with executed evidence.",
                    "source_artifacts": ["data/example.parquet"],
                }
            ],
            "limitations": ["Example structure only; report limitations supported by participant evidence."],
        }
    )
    return cast(dict[str, JsonValue], example.model_dump(mode="json", exclude_none=True))


def _trialeval_nonidentification_example(
    *,
    result_kind: Literal["limitation", "abstention"],
) -> dict[str, JsonValue]:
    """Return a valid shape example for a qualified non-identification result."""

    example = TrialEvalSubmissionV1.model_validate(
        {
            "task_id": "TASK_EXAMPLE_A4",
            "primary_analysis": {
                "declared_primary": True,
                "estimand": {
                    "estimand_id": "example_fixed_estimand",
                    "population_id": "all_randomized",
                    "treatment_id": "example_treatment",
                    "comparator_id": "example_control",
                    "endpoint_id": "example_endpoint",
                    "intercurrent_event_strategy_ids": ["example_event:treatment_policy"],
                    "horizon": {"value": 365.0, "unit": "days"},
                },
                "estimator": {
                    "analysis_method_id": "example_qualified_nonidentification_method",
                },
                "result_kind": result_kind,
                "result": {
                    "kind": "non_identification",
                    "conclusion_code": "point_not_identified_due_to_censoring_or_support_failure",
                    "effect_scale": "risk_difference_tau",
                    "unit": "probability_difference",
                    "reason": "Example shape only; the released evidence must establish this condition.",
                    "additional_assumption_required": (
                        "A justified restriction on informative censoring would be required for a unique point."
                    ),
                },
                "favorable_direction": "higher",
            },
        }
    )
    return cast(dict[str, JsonValue], example.model_dump(mode="json", exclude_none=True))


def trialeval_submission_shape_catalogue() -> TrialEvalSubmissionShapeCatalogueV1:
    """Return balanced, model-valid examples for every public wire shape."""

    primary_payloads = {
        "numeric_point:scalar": _trialeval_scalar_example(),
        "identification_bound:identified_interval": _submission_shape_payload(
            result_kind="identification_bound",
            result={
                "kind": "identified_interval",
                "lower": -0.20,
                "upper": 0.30,
                "effect_scale": "bounds_interval",
                "unit": "probability_difference",
                "interpretation": "Example identified interval; replace with the executed analysis.",
            },
            estimator={
                "analysis_method_id": "example_identification_bound_method",
            },
        ),
        "numeric_vector:vector": _submission_shape_payload(
            result_kind="numeric_vector",
            result=_example_vector_result(),
            estimator={
                "analysis_method_id": "example_numeric_vector_method",
            },
        ),
        "statistical_test:statistical_test": _submission_shape_payload(
            result_kind="statistical_test",
            result={
                "kind": "statistical_test",
                "statistic": 0.0,
                "p_value": 1.0,
                "reject_null": False,
                "effect_scale": "weighted_logrank_test",
                "unit": "standardized_statistic",
                "alternative": "two_sided",
                "rho": 0.0,
                "gamma": 1.0,
            },
            estimator={
                "analysis_method_id": "example_statistical_test_method",
            },
        ),
        "sensitivity_set:vector": _submission_shape_payload(
            result_kind="sensitivity_set",
            result=_example_sensitivity_result(),
            estimator={
                "analysis_method_id": "example_sensitivity_set_method",
            },
        ),
        "limitation:non_identification": _trialeval_nonidentification_example(result_kind="limitation"),
        "abstention:non_identification": _trialeval_nonidentification_example(result_kind="abstention"),
    }
    for payload in primary_payloads.values():
        primary = cast(dict[str, JsonValue], payload["primary_analysis"])
        if not payload.get("evidence"):
            payload["evidence"] = [
                _supporting_evidence_payload(
                    result=primary["result"],
                    estimator=primary["estimator"],
                )
            ]
        primary["evidence_ids"] = ["example_evidence"]
        if not payload.get("limitations"):
            payload["limitations"] = [
                "Example structure only; replace with limitations supported by participant-visible evidence."
            ]
        submission = TrialEvalSubmissionV1.model_validate(payload)
        validate_trialeval_required_deliverables_v1(
            submission,
            required_deliverables=("evidence", "limitations", "primary_analysis"),
        )

    evidence_payloads: dict[str, dict[str, JsonValue]] = {
        "scalar": _supporting_evidence_payload(
            result=cast(dict[str, JsonValue], primary_payloads["numeric_point:scalar"]["primary_analysis"])["result"],
            estimator=cast(dict[str, JsonValue], primary_payloads["numeric_point:scalar"]["primary_analysis"])[
                "estimator"
            ],
        ),
        "identified_interval": _supporting_evidence_payload(
            result=cast(
                dict[str, JsonValue],
                primary_payloads["identification_bound:identified_interval"]["primary_analysis"],
            )["result"],
            estimator=cast(
                dict[str, JsonValue],
                primary_payloads["identification_bound:identified_interval"]["primary_analysis"],
            )["estimator"],
        ),
        "vector": _supporting_evidence_payload(
            result=_example_vector_result(),
            estimator={
                "analysis_method_id": "example_numeric_vector_method",
            },
        ),
        "statistical_test": _supporting_evidence_payload(
            result=cast(
                dict[str, JsonValue],
                primary_payloads["statistical_test:statistical_test"]["primary_analysis"],
            )["result"],
            estimator=cast(
                dict[str, JsonValue],
                primary_payloads["statistical_test:statistical_test"]["primary_analysis"],
            )["estimator"],
        ),
        "non_identification": _supporting_evidence_payload(
            result=cast(
                dict[str, JsonValue],
                primary_payloads["limitation:non_identification"]["primary_analysis"],
            )["result"],
            estimator=cast(
                dict[str, JsonValue],
                primary_payloads["limitation:non_identification"]["primary_analysis"],
            )["estimator"],
        ),
        "diagnostic_test": _diagnostic_evidence_payload(
            result={
                "kind": "diagnostic_test",
                "statistic": {
                    "metric_id": "example_statistic",
                    "value": 0.0,
                    "unit": "example_unit",
                    "decimal_places": 3,
                },
                "p_value": {
                    "metric_id": "example_p_value",
                    "value": 1.0,
                    "unit": "probability",
                    "decimal_places": 3,
                },
                "alternative": "two_sided",
            }
        ),
        "diagnostic_summary": _diagnostic_evidence_payload(
            result={
                "kind": "diagnostic_summary",
                "measures": [
                    {
                        "metric_id": "example_measure",
                        "value": 0.0,
                        "unit": "example_unit",
                        "decimal_places": 3,
                    }
                ],
            }
        ),
        "factual_premise": _diagnostic_evidence_payload(
            result={
                "kind": "factual_premise",
                "premise_id": "randomized_assignment_declared",
                "conclusion": "supported",
            }
        ),
    }
    optional = TrialEvalOptionalSubmissionFragmentsV1.model_validate(
        {
            "reconstruction": {
                "n_subjects": 1,
                "n_primary_population": 1,
                "n_events": 0,
                "n_censored": 1,
                "checks_performed": ["example_key_check"],
                "notes": "Example structure only; replace with executed reconstruction.",
                "source_artifacts": ["data/example.parquet"],
            },
            "data_integrity_record": {
                "condition_id": "exact_transport_row_duplication_v1",
                "affected_domain": "example_domain",
                "compound_key_fields": ["example_id"],
                "observed_duplicate_group_count": 1,
                "observed_extra_row_count": 1,
                "repair_action": "remove_one_exact_duplicate_copy",
                "repair_status": "repaired",
                "post_repair_data_checksum": "0" * 64,
                "analysis_input_data_checksum": "0" * 64,
            },
            "planning": {
                "method_id": "schoenfeld_logrank_v1",
                "estimand_id": "example_estimand",
                "alpha_two_sided": 0.05,
                "power": 0.90,
                "treated_allocation_fraction": 0.50,
                "event_probability": 0.50,
                "followup_horizon_dy": 365.0,
                "multiplicity_adjustment": "none",
                "required_events": 100,
                "target_sample_size": 200,
                "sensitivity": [
                    {"event_probability": 0.40, "target_sample_size": 250},
                    {"event_probability": 0.60, "target_sample_size": 167},
                ],
            },
        }
    )
    return TrialEvalSubmissionShapeCatalogueV1(
        primary_submissions={
            key: TrialEvalSubmissionV1.model_validate(payload) for key, payload in primary_payloads.items()
        },
        evidence_records={key: EvidenceRecordV1.model_validate(payload) for key, payload in evidence_payloads.items()},
        optional_fragments=optional,
    )


def _submission_shape_payload(
    *,
    result_kind: str,
    result: dict[str, JsonValue],
    estimator: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    return {
        "task_id": "TASK_EXAMPLE",
        "primary_analysis": {
            "declared_primary": True,
            "estimand": {
                "estimand_id": "example_estimand",
                "population_id": "example_population",
                "treatment_id": "example_treatment",
                "comparator_id": "example_comparator",
                "endpoint_id": "example_endpoint",
                "intercurrent_event_strategy_ids": ["example_strategy"],
                "horizon": {"value": 365.0, "unit": "days"},
            },
            "estimator": estimator,
            "result_kind": result_kind,
            "result": result,
            "favorable_direction": "neither",
            "evidence_ids": ["example_evidence"],
        },
        "evidence": [_supporting_evidence_payload(result=result, estimator=estimator)],
        "limitations": ["Example structure only; replace with supported limitations."],
    }


def _supporting_evidence_payload(
    *,
    result: JsonValue,
    estimator: JsonValue,
) -> dict[str, JsonValue]:
    return {
        "evidence_id": "example_evidence",
        "evidence_type": "supporting_analysis",
        "principle": "uncertainty",
        "operation": "estimation",
        "estimator": estimator,
        "target": "example supporting target",
        "result": result,
        "interpretation": "Example structure only; replace with executed evidence.",
        "source_artifacts": ["data/example.parquet"],
    }


def _diagnostic_evidence_payload(*, result: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "evidence_id": f"example_{result['kind']}",
        "evidence_type": "diagnostic",
        "principle": "design_validity",
        "operation": "assessment",
        "diagnostic_id": "randomization_integrity_public",
        "target": "example public diagnostic",
        "result": result,
        "interpretation": "Example structure only; replace with an executed public diagnostic.",
        "source_artifacts": ["task.json"],
    }


def _example_vector_result() -> dict[str, JsonValue]:
    return {
        "kind": "vector",
        "points": [
            {"component_id": "example_early", "index": 1.0, "value": 0.0},
            {"component_id": "example_late", "index": 2.0, "value": 0.0},
        ],
        "index_unit": "example_index",
        "effect_scale": "piecewise_log_hr_vector",
        "unit": "log_hazard_ratio",
    }


def _example_sensitivity_result() -> dict[str, JsonValue]:
    return {
        "kind": "vector",
        "points": [
            {"component_id": "delta_0.05_lower", "index": 1.0, "value": -0.10},
            {"component_id": "delta_0.05_upper", "index": 2.0, "value": 0.10},
            {"component_id": "delta_0.10_lower", "index": 3.0, "value": -0.20},
            {"component_id": "delta_0.10_upper", "index": 4.0, "value": 0.20},
            {"component_id": "delta_0.20_lower", "index": 5.0, "value": -0.40},
            {"component_id": "delta_0.20_upper", "index": 6.0, "value": 0.40},
        ],
        "index_unit": "sensitivity_set_component",
        "effect_scale": "risk_difference_tau",
        "unit": "probability_difference",
    }


def trialeval_submission_schema() -> dict[str, JsonValue]:
    """Return the canonical TrialEval schema without answer-shape priming."""

    return cast(dict[str, JsonValue], TrialEvalSubmissionV1.model_json_schema())


def trialdev_randomized_phase_analysis_schema() -> dict[str, JsonValue]:
    """Return the exact randomized-phase analysis submission schema."""

    from trialagentbench_harness.adapters.trialdev_share import (
        TrialDevelopmentPhaseAnalysisSubmissionV1,
    )

    schema = TrialDevelopmentPhaseAnalysisSubmissionV1.model_json_schema()
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("Phase-analysis model schema must define object properties.")
    if "candidate_utility_estimates" not in properties:
        raise ValueError("Phase-analysis model schema is missing candidate_utility_estimates.")
    del properties["candidate_utility_estimates"]
    properties["analysis_rationale"] = {
        "type": "string",
        "maxLength": 2000,
        "description": "Optional explanation retained outside deterministic scoring.",
    }
    return schema


def trialdev_phase_request_schema() -> dict[str, JsonValue]:
    """Return the phase-request base schema including its audit rationale."""

    from trialagentbench_harness.adapters.trialdev_share import TrialDevelopmentRequestV1

    schema = TrialDevelopmentRequestV1.model_json_schema()
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("Phase-request model schema must define object properties.")
    properties["request_rationale"] = {
        "type": "string",
        "maxLength": 2000,
        "description": "Optional explanation retained outside deterministic scoring.",
    }
    return schema


def write_public_submission_contracts(root: Path) -> None:
    """Write public JSON contracts from the canonical runtime models."""

    from trialagentbench_harness.adapters.trialdev_share import (
        TrialDevelopmentObservationalReviewSubmissionV1,
        TrialDevelopmentPhaseDecisionSubmissionV1,
    )
    from trialagentbench_harness.trialdev.participant_submission import participant_schema_v1

    root.mkdir(parents=True, exist_ok=True)
    output_schema = {
        "schema_id": "trialagentbench.public_submission_schemas/v1",
        "trialeval": trialeval_submission_schema(),
        "trialdev_observational_review": participant_schema_v1(TrialDevelopmentObservationalReviewSubmissionV1),
        "trialdev_phase_request": participant_schema_v1(
            trialdev_phase_request_schema(),
            root_fields=frozenset({"scenario_id", "phase_id", "version"}),
        ),
        "trialdev_phase_analysis": participant_schema_v1(
            trialdev_randomized_phase_analysis_schema(),
            root_fields=frozenset({"scenario_id", "phase_id", "version"}),
        ),
        "trialdev_phase_decision": participant_schema_v1(
            TrialDevelopmentPhaseDecisionSubmissionV1,
            root_fields=frozenset({"scenario_id", "phase_id", "version"}),
        ),
    }
    evaluation_spec = {
        "schema_id": "trialagentbench.evaluation_specification/v1",
        "suites": {
            "trialeval": {
                "purpose": "Execute an estimand-compatible analysis from participant-visible evidence.",
                "evaluation_rule": (
                    "validated structured primary analysis and evidence-supported, independently verified "
                    "route-specific reference result"
                ),
                "capability_stages": [
                    "completion",
                    "reconstruction",
                    "estimand",
                    "method_route",
                    "numeric_result",
                    "uncertainty",
                ],
            },
            "trialdev": {
                "purpose": "Integrate sequential evidence and choose an action under a declared public policy.",
                "evaluation_rule": "evidence-linked action membership in the policy-derived acceptable set",
                "capability_stages": ["completion", "evidence", "policy_action", "route"],
            },
        },
        "submission_schema": "agent_output_schema.json",
    }
    outputs = (
        (root / "agent_output_schema.json", output_schema),
        (root / "eval_spec.json", evaluation_spec),
        (
            root / "examples" / "submissions" / "trialeval_shapes.json",
            trialeval_submission_shape_catalogue().model_dump(mode="json"),
        ),
    )
    for path, payload in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
