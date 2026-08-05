"""Deterministic grading functions for TrialDev submissions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from trialagentbench_harness.contracts.trialdev.scientific_grade import (
    TrialDevAnalysisClassificationV1,
    TrialDevScientificAssessmentV1,
    TrialDevScientificEnvelopeV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_evaluation_target_register import (
    required_trialdev_lanes_v1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_public_recoverability import (
    TrialDevPublicRecoverabilityReportV1,
)
from trialagentbench_harness.numeric_policy import (
    FLOAT_EQUALITY_ABS_TOLERANCE_V1,
    MINIMUM_TWO_SIDED_CONFIDENCE_LEVEL_V1,
    SHA256_HEX_DIGEST_LENGTH,
)
from trialagentbench_harness.trialdev.grading.analysis_evidence import (
    derive_effect_references_v1,
    interval_equivalence_score_v1,
    point_equivalence_score_v1,
    point_interval_equivalence_score_v1,
    reporting_tolerance_v1,
)
from trialagentbench_harness.trialdev.grading.decision_evidence import (
    TrialDevPhaseDesignWitnessV1,
    derive_phase_decision_witness_v1,
    derive_phase_design_witness_v1,
)
from trialagentbench_harness.trialdev.grading.design_frontier import derive_phase_design_efficiency_v1
from trialagentbench_harness.trialdev.grading.evaluation_target_register import (
    TrialDevEvaluationTargetRegisterIndex,
    TrialDevEvaluationTargetRegisterRecordV1,
    load_evaluation_target_index,
    score_evaluation_target,
)
from trialagentbench_harness.trialdev.grading.hashing import compute_sha256_hex, sha256_file_hex
from trialagentbench_harness.trialdev.grading.io import read_json, write_json
from trialagentbench_harness.trialdev.grading.method_design import (
    effect_submission_matches_cell_v1,
    load_public_method_design_contracts_v1,
    observational_submission_matches_cell_v1,
    required_phase_cells_v1,
    resolve_observational_method_route_v1,
    safety_submission_matches_cell_v1,
    submitted_analysis_cell_id_v1,
)
from trialagentbench_harness.trialdev.grading.models import (
    TrialDevDesignEfficiencyV1,
    TrialDevelopmentAnalysisQualityV1,
    TrialDevelopmentAuditGateReportV1,
    TrialDevelopmentCandidateEligibilityRecordV1,
    TrialDevelopmentGradeGateRecordV1,
    TrialDevelopmentGradeReportV1,
    TrialDevelopmentLaneScoreRecordV1,
    TrialDevelopmentSubmissionV1,
    TrialDevelopmentValidityReportV1,
)
from trialagentbench_harness.trialdev.grading.scientific_assessment import (
    scientific_bundle_agreement_v1,
)
from trialagentbench_harness.trialdev.grading.validate import validate_release_v1, validate_submission_v1
from trialagentbench_harness.trialdev.share.public_method_design import (
    TrialDevPublicObservationalMethodCatalogV1,
)
from trialagentbench_harness.trialdev.share.sequential import TrialDevelopmentSafetyEstimateV1
from trialagentbench_harness.trialdev.share.validate import (
    candidate_ids_by_role_v1,
    validate_request_against_scenario_v1,
)

__all__ = [
    "PublicCandidateUtilityReferenceV1",
    "PublicObservationalReferenceV1",
    "grade_bundle_v1",
    "grade_item_v1",
    "grade_report_payload_v1",
    "load_public_observational_reference_v1",
]

_REPORT_MODES = {"score", "audit"}
_ACTIVE_LANES = (
    "trial_design",
    "trial_evaluation",
    "program_decision",
    "drug_ranking",
)
_ADVANCING_ACTIONS = {
    "advance_to_proof_of_concept",
    "advance_to_confirmation",
    "declare_success",
}
_STOPPING_ACTIONS = {
    "withhold_nomination",
    "stop_development",
    "declare_failure",
    "declare_inconclusive",
}


def _ordered_grade_gates_v1(
    *,
    question_valid: bool,
    route_valid: bool,
    evidence_valid: bool,
    result_valid: bool,
    conformance_valid: bool,
    decision_valid: bool,
) -> tuple[tuple[TrialDevelopmentGradeGateRecordV1, ...], str | None]:
    """Build the first-failure TrialDev grading cascade."""

    checks: tuple[tuple[str, bool | None, str], ...] = (
        ("submission", True, "submission_invalid"),
        ("question", question_valid, "question_or_candidate_set_invalid"),
        ("route", route_valid, "analysis_or_design_route_not_qualified"),
        ("evidence", evidence_valid, "required_public_evidence_incomplete"),
        ("integrity", None, "not_applicable"),
        ("result", result_valid, "submitted_result_not_equivalent"),
        ("conformance", conformance_valid, "required_analysis_lane_not_conformant"),
        ("decision", decision_valid, "decision_not_supported_by_public_policy"),
    )
    failed = False
    first_failure: str | None = None
    records: list[TrialDevelopmentGradeGateRecordV1] = []
    for gate_id, passed, failure_code in checks:
        if failed:
            status = "not_reached"
            code = None
        elif passed is None:
            status = "not_applicable"
            code = None
        elif passed:
            status = "passed"
            code = None
        else:
            status = "failed"
            code = failure_code
            failed = True
            first_failure = gate_id
        records.append(
            TrialDevelopmentGradeGateRecordV1.model_validate(
                {
                    "gate_id": gate_id,
                    "status": status,
                    "failure_code": code,
                }
            )
        )
    return tuple(records), first_failure


def grade_report_payload_v1(*, report: TrialDevelopmentGradeReportV1, report_mode: str = "score") -> dict[str, object]:
    """Return a score-mode or audit-mode JSON payload for one grade report."""
    mode = str(report_mode)
    if mode not in _REPORT_MODES:
        raise ValueError(f"report_mode must be one of {sorted(_REPORT_MODES)!r}.")
    payload = cast(dict[str, object], report.model_dump(mode="json", exclude_none=True))
    # Applicability is part of the public grading contract: ineligible
    # components must retain explicit nulls rather than become missing fields.
    payload["analysis_quality"] = report.analysis_quality.model_dump(mode="json")
    if mode == "audit":
        return payload
    report_payload = payload.get("payload", {})
    evaluation_target_checksums = {}
    if isinstance(report_payload, dict):
        raw_checksums = report_payload.get("evaluation_target_checksums", {})
        if isinstance(raw_checksums, dict):
            evaluation_target_checksums = raw_checksums
    analysis_method_diagnostics = (
        report_payload.get("analysis_method_diagnostics", {}) if isinstance(report_payload, dict) else {}
    )
    numeric_score_components = (
        report_payload.get("numeric_score_components", {}) if isinstance(report_payload, dict) else {}
    )
    return {
        "schema_id": payload["schema_id"],
        "version": payload["version"],
        "scenario_id": payload["scenario_id"],
        "phase_id": payload["phase_id"],
        "objective_id": payload["objective_id"],
        "program_objective_id": payload["program_objective_id"],
        "phase_scoring_objective_id": payload["phase_scoring_objective_id"],
        "primary_score": payload["primary_score"],
        "design_score": payload["design_score"],
        "evaluation_score": payload["evaluation_score"],
        "program_score": payload["program_score"],
        "ranking_score": payload["ranking_score"],
        "analysis_quality": payload["analysis_quality"],
        "scientific_assessment": payload["scientific_assessment"],
        "gates": payload["gates"],
        "first_failure_gate": payload.get("first_failure_gate"),
        "lane_status": payload.get("lane_status", {}),
        "active_lane_scores": payload.get("active_lane_scores", {}),
        "lane_scores": payload.get("lane_scores", []),
        "validity": payload.get("validity", {}),
        "audit_gates": payload.get("audit_gates", {}),
        "design_efficiency": payload.get("design_efficiency"),
        "selected_winner_drug_id": payload.get("selected_winner_drug_id"),
        "feasibility_failures": payload.get("feasibility_failures", []),
        "lane_breakdown": payload.get("lane_breakdown", {}),
        "evaluation_target_checksums": evaluation_target_checksums,
        "analysis_method_diagnostics": analysis_method_diagnostics,
        "numeric_score_components": numeric_score_components,
        "checksum": payload.get("checksum"),
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _interval_contains_threshold(*, lower: float, upper: float, threshold: object) -> bool:
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        return False
    return float(lower) <= float(threshold) <= float(upper)


def _safety_uncertainty_support_v1(
    *,
    safety_estimate: TrialDevelopmentSafetyEstimateV1,
    safety_diagnostics: dict[str, object],
) -> bool:
    """Return whether submitted intervals evidence an indeterminate hard gate."""

    crossings = [
        _interval_contains_threshold(
            lower=safety_estimate.serious_ae_treatment_lower,
            upper=safety_estimate.serious_ae_treatment_upper,
            threshold=safety_diagnostics.get("absolute_limit"),
        ),
        _interval_contains_threshold(
            lower=safety_estimate.serious_ae_excess_lower,
            upper=safety_estimate.serious_ae_excess_upper,
            threshold=safety_diagnostics.get("excess_limit"),
        ),
    ]
    discontinuation = safety_diagnostics.get("discontinuation")
    if isinstance(discontinuation, dict) and discontinuation.get("role") == "hard_gate":
        crossings.extend(
            (
                _interval_contains_threshold(
                    lower=safety_estimate.discontinuation_treatment_lower,
                    upper=safety_estimate.discontinuation_treatment_upper,
                    threshold=discontinuation.get("absolute_limit"),
                ),
                _interval_contains_threshold(
                    lower=safety_estimate.discontinuation_excess_lower,
                    upper=safety_estimate.discontinuation_excess_upper,
                    threshold=discontinuation.get("excess_limit"),
                ),
            )
        )
    return any(crossings)


def _records_by_context(
    records: list[dict[str, object]],
    *,
    phase_id: str,
    objective_id: str,
    lane_id: str | None = None,
) -> list[dict[str, object]]:
    matches = [
        record
        for record in records
        if str(record.get("phase_id")) == str(phase_id) and str(record.get("objective_id")) == str(objective_id)
    ]
    if lane_id is None:
        return matches
    return [record for record in matches if str(record.get("lane_id")) == str(lane_id)]


def _public_candidate_ids(*, scenario_root: Path) -> set[str]:
    """Return public non-control candidate IDs for a TrialDev scenario."""

    return set(candidate_ids_by_role_v1(scenario_root=Path(scenario_root))["investigational"])


def _resolve_candidate_eligibility(
    *,
    scenario_root: Path,
    scenario_id: str,
    phase_id: str,
    candidate_drug_ids: tuple[str, ...],
    eligible_candidate_drug_ids: tuple[str, ...] | None = None,
    terminal_status: str | None = None,
) -> tuple[TrialDevelopmentCandidateEligibilityRecordV1, ...]:
    """Resolve candidate validity from the public catalog and optional programme state."""

    public_candidates = _public_candidate_ids(scenario_root=Path(scenario_root))
    eligible_candidates = (
        None if eligible_candidate_drug_ids is None else {str(v) for v in eligible_candidate_drug_ids}
    )
    terminal = None if terminal_status is None else str(terminal_status)
    rows: list[TrialDevelopmentCandidateEligibilityRecordV1] = []
    for candidate_id in candidate_drug_ids:
        candidate = str(candidate_id)
        catalog_member = candidate in public_candidates
        if not catalog_member:
            sequentially_eligible = False
            source = "public_candidate_catalog"
            failure = "not_in_public_catalog"
        elif terminal not in {None, "active"}:
            sequentially_eligible = False
            source = "materialized_program_state"
            failure = "program_already_terminal"
        elif eligible_candidates is not None and candidate not in eligible_candidates:
            sequentially_eligible = False
            source = "eval_contract_current_state"
            failure = "not_in_current_eligible_set"
        else:
            sequentially_eligible = True
            source = "eval_contract_current_state" if eligible_candidates is not None else "public_candidate_catalog"
            failure = "not_applicable"
        rows.append(
            TrialDevelopmentCandidateEligibilityRecordV1(
                scenario_id=str(scenario_id),
                phase_id=str(phase_id),
                candidate_drug_id=candidate,
                catalog_member=bool(catalog_member),
                sequentially_eligible=bool(sequentially_eligible),
                eligibility_source=source,
                failure_reason=failure,
            )
        )
    return tuple(rows)


def _record_candidates(record: dict[str, object]) -> tuple[str, ...]:
    raw = record.get("candidate_drug_ids")
    if not isinstance(raw, list | tuple):
        raise ValueError("Evaluator reference record candidate_drug_ids must be a list or tuple.")
    candidates = tuple(str(value) for value in raw)
    if not candidates or any(not value for value in candidates) or len(set(candidates)) != len(candidates):
        raise ValueError("Evaluator reference record candidate_drug_ids must be non-empty and unique.")
    return candidates


def _record_numeric_value(record: dict[str, object]) -> float:
    raw = record.get("value")
    if isinstance(raw, bool) or not isinstance(raw, int | float | str):
        raise ValueError("Evaluator reference record value must be numeric.")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("Evaluator reference record value must be numeric.") from exc
    if not math.isfinite(value):
        raise ValueError("Evaluator reference record value must be finite.")
    return value


def _single_record_candidate(record: dict[str, object]) -> str:
    candidates = _record_candidates(record)
    if len(candidates) != 1:
        raise ValueError("Evaluator reference score records must identify exactly one candidate drug.")
    return candidates[0]


def _best_in_set(*, scores_by_drug: dict[str, float], candidate_drug_ids: tuple[str, ...]) -> tuple[str | None, float]:
    filtered = {
        str(drug): float(scores_by_drug[str(drug)]) for drug in candidate_drug_ids if str(drug) in scores_by_drug
    }
    if not filtered:
        return None, 0.0
    best = sorted(filtered.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return str(best[0]), float(best[1])


def _payload_float(payload: object, key: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get(str(key))
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(float(raw)):
        raise ValueError(f"Public decision evidence {key!r} must be a finite number.")
    return float(raw)


@dataclass(frozen=True)
class PublicCandidateUtilityReferenceV1:
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class PublicObservationalReferenceV1:
    method_route_id: str
    estimator_id: str
    effect_scale_id: str
    confidence_level: float
    absolute_tolerance: float
    scientific_absolute_margin: float
    adjustment_covariates: frozenset[str]
    source_artifact_checksums: dict[str, str]
    candidate_utilities: dict[str, PublicCandidateUtilityReferenceV1]
    reference_target_ids: tuple[str, ...]
    credit_eligible_target_ids: tuple[str, ...]
    recoverability_policy_id: str


@dataclass(frozen=True)
class _GradeFailure:
    """One grading failure with validity semantics fixed at its origin."""

    code: str
    invalid: bool = False


def _qualified_observational_evidence_failures(
    *,
    scenario_root: Path,
    submission: TrialDevelopmentSubmissionV1,
) -> tuple[str, ...]:
    """Verify a qualified non-nomination from participant-visible provenance."""

    evidence = tuple(submission.analysis_report.identification_evidence)
    if submission.analysis_report.response_branch != "qualified_non_nomination" or not evidence:
        return ("qualified_non_nomination_evidence_missing",)
    catalog_path = Path(scenario_root) / "public" / "observational_method_catalog.json"
    extract_path = Path(scenario_root) / "public" / "observational_extract.parquet"
    catalog = TrialDevPublicObservationalMethodCatalogV1.model_validate(read_json(catalog_path))
    recoverability = TrialDevPublicRecoverabilityReportV1.model_validate(
        read_json(Path(scenario_root) / "grader" / "public_recoverability_report.json")
    )
    report_checksums = {record.path: record.sha256 for record in recoverability.public_input_checksums}
    expected_checksums = {
        "public/observational_method_catalog.json": sha256_file_hex(catalog_path),
        "public/observational_extract.parquet": sha256_file_hex(extract_path),
    }
    for path, expected_checksum in expected_checksums.items():
        if report_checksums.get(path) != expected_checksum:
            raise ValueError(f"Public recoverability input checksum disagrees with released artifact: {path}.")
    factors = {factor.factor_id: factor for factor in catalog.assignment_prognostic_factors}
    methods = {str(method.method_route_id): method for method in catalog.methods}
    objective_id = submission.program_decision.objective_id
    if objective_id is None:
        return ("qualified_non_nomination_objective_missing",)
    failures: list[str] = []
    valid_failed_premise_ids: set[str] = set()
    for record in evidence:
        artifact_checksum = expected_checksums.get(record.public_artifact_path)
        if artifact_checksum is None:
            failures.append(f"unsupported_identification_evidence_path:{record.evidence_id}")
            continue
        if record.public_artifact_sha256 != artifact_checksum:
            failures.append(f"identification_evidence_checksum_mismatch:{record.evidence_id}")
            continue
        if record.public_artifact_path == "public/observational_method_catalog.json":
            factor = factors.get(record.source_record_id)
            if record.premise_id != "measured_conditional_exchangeability" or factor is None:
                failures.append(f"unsupported_identification_evidence_record:{record.evidence_id}")
                continue
            expected_failure = bool(
                factor.used_in_treatment_assignment
                and factor.prognostic_for_primary_endpoint
                and not factor.recorded_in_observational_extract
            )
            if record.evidence_kind != "factual_provenance" or record.premise_state != (
                "failed" if expected_failure else "satisfied"
            ):
                failures.append(f"identification_evidence_state_mismatch:{record.evidence_id}")
                continue
            if expected_failure:
                valid_failed_premise_ids.add(record.premise_id)
            continue
        method = methods.get(record.source_record_id)
        method_results = tuple(
            result for result in recoverability.method_results if result.method_route_id == record.source_record_id
        )
        if method is None or len(method_results) != 1:
            failures.append(f"unsupported_identification_evidence_record:{record.evidence_id}")
            continue
        comparisons = tuple(
            comparison
            for comparison in method_results[0].estimator_comparisons
            if str(comparison.objective_id) == objective_id and comparison.estimator_id == method.primary_estimator_id
        )
        if len(comparisons) != 1 or comparisons[0].status != "not_estimable":
            failures.append(f"identification_evidence_state_mismatch:{record.evidence_id}")
            continue
        expected_premise_id = (
            "practical_positivity"
            if comparisons[0].failure_reason == "empirical_positivity_violation"
            else "method_estimability"
        )
        if (
            record.evidence_kind != "empirical_diagnostic"
            or record.premise_state != "failed"
            or record.premise_id != expected_premise_id
        ):
            failures.append(f"identification_evidence_state_mismatch:{record.evidence_id}")
            continue
        valid_failed_premise_ids.add(record.premise_id)
    if not valid_failed_premise_ids:
        failures.append("qualified_non_nomination_lacks_failed_identification_premise")
    return tuple(sorted(set(failures)))


def _public_observational_method_supports_point_result(
    *,
    scenario_root: Path,
    objective_id: str,
    method_route_id: str,
) -> bool:
    """Return whether one declared public method has a point-result policy."""

    report = TrialDevPublicRecoverabilityReportV1.model_validate(
        read_json(Path(scenario_root) / "grader" / "public_recoverability_report.json")
    )
    methods = tuple(result for result in report.method_results if result.method_route_id == method_route_id)
    if len(methods) != 1:
        raise ValueError(f"Public recoverability requires one result for method_route_id={method_route_id!r}.")
    policies = tuple(policy for policy in methods[0].objective_policies if str(policy.objective_id) == objective_id)
    if len(policies) != 1:
        raise ValueError(f"Public recoverability requires one policy for objective_id={objective_id!r}.")
    return bool(policies[0].policy != "insufficient_recoverability")


def _numeric_equivalence_tolerance(*, scenario_root: Path) -> float:
    charter = read_json(Path(scenario_root) / "public" / "objective_charter.json")
    decimal_places = charter.get("numeric_reporting_decimal_places") if isinstance(charter, dict) else None
    if isinstance(decimal_places, bool) or not isinstance(decimal_places, int):
        raise ValueError("Public objective charter requires integer numeric_reporting_decimal_places.")
    return float(reporting_tolerance_v1(decimal_places=decimal_places))


def _method_conditioned_asset_reference(
    *,
    evaluation_target: TrialDevEvaluationTargetRegisterRecordV1,
    reference: PublicObservationalReferenceV1,
) -> TrialDevEvaluationTargetRegisterRecordV1:
    """Bind asset credit to the submitted method's public action policy."""

    if evaluation_target.lane_id != "asset_nomination":
        raise ValueError("Method-conditioned asset reference requires the asset-nomination lane.")
    payload = evaluation_target.model_dump(mode="json", exclude={"checksum"})
    payload.update(
        {
            "scoring_policy_id": "trialdev_method_conditioned_public_action_v1",
            "reference_target_ids": reference.reference_target_ids,
            "credit_eligible_target_ids": reference.credit_eligible_target_ids,
            "recoverability_policy_id": reference.recoverability_policy_id,
            "target_resolution": "release_static",
            "value_payload": {
                **evaluation_target.value_payload,
                "method_route_id": reference.method_route_id,
            },
        }
    )
    payload["checksum"] = compute_sha256_hex(payload)
    return TrialDevEvaluationTargetRegisterRecordV1.model_validate(payload)


def _qualified_non_nomination_asset_reference(
    *,
    evaluation_target: TrialDevEvaluationTargetRegisterRecordV1,
) -> TrialDevEvaluationTargetRegisterRecordV1:
    """Bind verified non-estimability to the supported no-nomination action."""

    if evaluation_target.lane_id != "asset_nomination":
        raise ValueError("Qualified non-nomination reference requires the asset-nomination lane.")
    payload = evaluation_target.model_dump(mode="json", exclude={"checksum"})
    payload.update(
        {
            "scoring_policy_id": "trialdev_evidence_qualified_non_nomination_v1",
            "reference_target_ids": ("withhold_nomination",),
            "credit_eligible_target_ids": (),
            "recoverability_policy_id": "insufficient_recoverability",
            "target_resolution": "release_static",
        }
    )
    payload["checksum"] = compute_sha256_hex(payload)
    return TrialDevEvaluationTargetRegisterRecordV1.model_validate(payload)


def _qualified_non_nomination_analysis_reference(
    *,
    evaluation_target: TrialDevEvaluationTargetRegisterRecordV1,
) -> TrialDevEvaluationTargetRegisterRecordV1:
    """Bind verified non-estimability to its nonnumeric analysis result."""

    if evaluation_target.lane_id != "phase_analysis":
        raise ValueError("Qualified non-nomination analysis reference requires the phase-analysis lane.")
    payload = evaluation_target.model_dump(mode="json", exclude={"checksum"})
    payload.update(
        {
            "scoring_policy_id": "trialdev_evidence_qualified_non_nomination_v1",
            "reference_target_ids": ("qualified_nonidentification",),
            "credit_eligible_target_ids": (),
            "recoverability_policy_id": "insufficient_recoverability",
            "target_resolution": "release_static",
        }
    )
    payload["checksum"] = compute_sha256_hex(payload)
    return TrialDevEvaluationTargetRegisterRecordV1.model_validate(payload)


def load_public_observational_reference_v1(
    *,
    scenario_root: Path,
    objective_id: str,
    method_route_id: str,
) -> PublicObservationalReferenceV1:
    """Load the observational estimand and utilities reconstructed from public evidence."""

    path = Path(scenario_root) / "grader" / "public_recoverability_report.json"
    if not path.is_file():
        raise FileNotFoundError(f"TrialDev public recoverability report is missing: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("public_recoverability_report.json must contain a JSON object.")
    if payload.get("schema_id") != "trialdev_public_recoverability_report_v1":
        raise ValueError(
            "public_recoverability_report.json must use schema_id='trialdev_public_recoverability_report_v1'."
        )
    raw_results = payload.get("method_results")
    if not isinstance(raw_results, list):
        raise ValueError("public_recoverability_report.json must contain method_results.")
    matching_results = [
        result
        for result in raw_results
        if isinstance(result, dict) and str(result.get("method_route_id")) == method_route_id
    ]
    if len(matching_results) != 1:
        raise ValueError(f"public recoverability requires one result for method_route_id={method_route_id!r}.")
    method_result = matching_results[0]
    raw_scores = method_result.get("candidate_scores")
    policies = method_result.get("objective_policies")
    if not isinstance(raw_scores, list) or not isinstance(policies, list):
        raise ValueError("method-specific recoverability requires candidate scores and objective policies.")
    matching_policies = [
        policy
        for policy in policies
        if isinstance(policy, dict) and str(policy.get("objective_id")) == str(objective_id)
    ]
    if len(matching_policies) != 1:
        raise ValueError(f"public recoverability requires one policy for objective_id={objective_id!r}.")
    if str(matching_policies[0].get("policy")) == "insufficient_recoverability":
        raise ValueError(
            f"objective_id={objective_id!r} is not scoreable because public recoverability is insufficient."
        )
    raw_action_policies = method_result.get("observational_action_policies")
    if not isinstance(raw_action_policies, list):
        raise ValueError("method-specific recoverability requires observational action policies.")
    matching_action_policies = [
        policy
        for policy in raw_action_policies
        if isinstance(policy, dict) and str(policy.get("objective_id")) == str(objective_id)
    ]
    if len(matching_action_policies) != 1:
        raise ValueError(f"public recoverability requires one action policy for objective_id={objective_id!r}.")
    action_policy = matching_action_policies[0]
    raw_reference_targets = action_policy.get("reference_target_ids")
    raw_acceptable_targets = action_policy.get("credit_eligible_target_ids")
    if (
        not isinstance(raw_reference_targets, list)
        or not raw_reference_targets
        or not isinstance(raw_acceptable_targets, list)
        or not raw_acceptable_targets
    ):
        raise ValueError("method-specific observational action policy requires non-empty target sets.")
    reference_target_ids = tuple(str(value) for value in raw_reference_targets)
    credit_eligible_target_ids = tuple(
        sorted(set(str(value) for value in raw_acceptable_targets) - set(reference_target_ids))
    )
    objective_spec_path = Path(scenario_root) / "public" / "objective_charter.json"
    if not objective_spec_path.is_file():
        raise FileNotFoundError(f"TrialDev public objective specification is missing: {objective_spec_path}")
    objective_spec = read_json(objective_spec_path)
    if not isinstance(objective_spec, dict):
        raise ValueError("objective_charter.json must contain a JSON object.")
    method_catalog = read_json(Path(scenario_root) / "public" / "observational_method_catalog.json")
    raw_methods = method_catalog.get("methods") if isinstance(method_catalog, dict) else None
    if not isinstance(raw_methods, list):
        raise ValueError("observational_method_catalog.json must declare executable public methods.")
    matching_methods = [
        method
        for method in raw_methods
        if isinstance(method, dict) and str(method.get("method_route_id")) == method_route_id
    ]
    if len(matching_methods) != 1:
        raise ValueError(f"observational method catalog has no unique cell {method_route_id!r}.")
    analysis_spec = matching_methods[0]
    if str(method_result.get("estimator_id") or "") != str(analysis_spec.get("primary_estimator_id") or ""):
        raise ValueError("method-specific recoverability estimator disagrees with its public method route.")
    objective_rows = objective_spec.get("objectives")
    if not isinstance(objective_rows, list):
        raise ValueError("objective_charter.json must declare objectives.")
    matching_objectives = [
        row for row in objective_rows if isinstance(row, dict) and str(row.get("objective_id")) == str(objective_id)
    ]
    if len(matching_objectives) != 1:
        raise ValueError(f"objective_charter requires one objective_id={objective_id!r}.")
    raw_scientific_margin = matching_objectives[0].get("indifference_margin")
    if (
        isinstance(raw_scientific_margin, bool)
        or not isinstance(raw_scientific_margin, int | float)
        or float(raw_scientific_margin) < 0.0
    ):
        raise ValueError("Public objective requires a non-negative indifference_margin.")
    raw_evidence_basis = matching_objectives[0].get("public_evidence_basis")
    if (
        not isinstance(raw_evidence_basis, list)
        or not raw_evidence_basis
        or any(not isinstance(path, str) or not path for path in raw_evidence_basis)
    ):
        raise ValueError("Public objective requires a non-empty public_evidence_basis.")
    evidence_basis = tuple(str(path) for path in raw_evidence_basis)
    if len(evidence_basis) != len(set(evidence_basis)):
        raise ValueError("Public objective evidence paths must be unique.")
    estimator_id = str(analysis_spec.get("primary_estimator_id") or "")
    raw_covariates = analysis_spec.get("adjustment_covariates")
    confidence_level = objective_spec.get("confidence_level")
    decimal_places = objective_spec.get("numeric_reporting_decimal_places")
    if not estimator_id or not isinstance(raw_covariates, list) or not raw_covariates:
        raise ValueError("Public observational analysis requires an estimator and adjustment covariates.")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, int | float)
        or not MINIMUM_TWO_SIDED_CONFIDENCE_LEVEL_V1 < float(confidence_level) < 1.0
    ):
        raise ValueError("Public objective charter requires a valid confidence_level.")
    if isinstance(decimal_places, bool) or not isinstance(decimal_places, int):
        raise ValueError("Public objective charter requires integer numeric_reporting_decimal_places.")
    adjustment_covariates = frozenset(str(value) for value in raw_covariates)
    if len(adjustment_covariates) != len(raw_covariates):
        raise ValueError("Public observational adjustment covariates must be unique.")

    raw_checksums = payload.get("public_input_checksums")
    if not isinstance(raw_checksums, list) or not raw_checksums:
        raise ValueError("public_recoverability_report.json must contain public_input_checksums.")
    available_checksums: dict[str, str] = {}
    for raw_checksum in raw_checksums:
        if not isinstance(raw_checksum, dict):
            raise ValueError("public_input_checksums entries must be objects.")
        source_path = str(raw_checksum.get("path") or "")
        checksum = str(raw_checksum.get("sha256") or "")
        if not source_path or len(checksum) != SHA256_HEX_DIGEST_LENGTH or source_path in available_checksums:
            raise ValueError("public_input_checksums require unique paths and SHA-256 values.")
        available_checksums[source_path] = checksum
    missing_evidence = sorted(set(evidence_basis) - set(available_checksums))
    if missing_evidence:
        raise ValueError(f"Public objective evidence is absent from recoverability checksums: {missing_evidence!r}.")
    source_checksums = {path: available_checksums[path] for path in evidence_basis}

    scores: dict[str, PublicCandidateUtilityReferenceV1] = {}
    for raw_score in raw_scores:
        if not isinstance(raw_score, dict) or str(raw_score.get("objective_id")) != str(objective_id):
            continue
        candidate_id = str(raw_score.get("candidate_drug_id") or "")
        utility = raw_score.get("adjusted_utility")
        lower = raw_score.get("ci_low")
        upper = raw_score.get("ci_high")
        if (
            not candidate_id
            or raw_score.get("point_estimable") is not True
            or raw_score.get("inference_estimable") is not True
            or isinstance(utility, bool)
            or not isinstance(utility, int | float)
            or not math.isfinite(float(utility))
            or isinstance(lower, bool)
            or not isinstance(lower, int | float)
            or not math.isfinite(float(lower))
            or isinstance(upper, bool)
            or not isinstance(upper, int | float)
            or not math.isfinite(float(upper))
        ):
            raise ValueError("Public candidate scores require estimable utility and uncertainty.")
        if float(lower) > float(utility) or float(utility) > float(upper):
            raise ValueError("Public candidate utility must lie within its confidence interval.")
        if candidate_id in scores:
            raise ValueError(
                "public_recoverability_report.json has duplicate candidate scores for "
                f"objective_id={objective_id!r}, candidate_drug_id={candidate_id!r}."
            )
        scores[candidate_id] = PublicCandidateUtilityReferenceV1(
            estimate=float(utility),
            lower=float(lower),
            upper=float(upper),
        )
    if not scores:
        raise ValueError(
            f"public_recoverability_report.json has no candidate scores for objective_id={objective_id!r}."
        )
    return PublicObservationalReferenceV1(
        method_route_id=str(analysis_spec.get("method_route_id") or ""),
        estimator_id=estimator_id,
        effect_scale_id=str(analysis_spec.get("effect_scale_id") or ""),
        confidence_level=float(confidence_level),
        absolute_tolerance=reporting_tolerance_v1(decimal_places=decimal_places),
        scientific_absolute_margin=float(raw_scientific_margin),
        adjustment_covariates=adjustment_covariates,
        source_artifact_checksums=source_checksums,
        candidate_utilities=scores,
        reference_target_ids=reference_target_ids,
        credit_eligible_target_ids=credit_eligible_target_ids,
        recoverability_policy_id=str(matching_policies[0].get("policy")),
    )


def _score_observational_analysis(
    *,
    submission: TrialDevelopmentSubmissionV1,
    reference: PublicObservationalReferenceV1,
    objective_id: str,
    required_candidate_ids: frozenset[str],
    supporting_evidence_ids: set[str],
) -> tuple[float, bool, tuple[str, ...]]:
    """Score a complete observational analysis against its public-evidence replay."""

    analysis_report = submission.analysis_report
    estimates = {str(estimate.candidate_drug_id): estimate for estimate in analysis_report.candidate_utility_estimates}
    failures: list[str] = []
    if set(estimates) != set(required_candidate_ids):
        failures.append("observational_candidate_utility_coverage_mismatch")
    ranked = tuple(str(value) for value in analysis_report.ranked_drug_ids)
    if len(ranked) != len(set(ranked)) or set(ranked) != set(required_candidate_ids):
        failures.append("observational_ranking_not_full_candidate_permutation")
    selected = submission.program_decision.recommended_drug_id
    action = str(submission.program_decision.decision_action or "")
    if action == "nominate_for_early_study":
        if selected is None or str(selected) not in required_candidate_ids:
            failures.append("observational_selected_candidate_missing_or_invalid")
        elif str(selected) in estimates and estimates[str(selected)].evidence_id not in supporting_evidence_ids:
            failures.append("observational_selected_candidate_evidence_not_cited")
    elif action == "withhold_nomination":
        if selected is not None:
            failures.append("observational_decline_declares_candidate")
    else:
        failures.append("observational_invalid_decision_action")

    components: list[float] = []
    scientific_components: list[bool] = []
    scientific_envelope = TrialDevScientificEnvelopeV1(
        envelope_id="trialdev_declared_utility_indifference_margin_v1",
        basis="declared_practical_equivalence_margin",
        absolute_margin=reference.scientific_absolute_margin,
        exact_reproduction_tolerance=reference.absolute_tolerance,
    )
    for candidate_id in sorted(required_candidate_ids):
        estimate = estimates.get(candidate_id)
        expected = reference.candidate_utilities.get(candidate_id)
        if estimate is None or expected is None:
            continue
        if str(estimate.objective_id) != str(objective_id):
            failures.append(f"observational_objective_mismatch:{candidate_id}")
        if estimate.method_route_id is not None and estimate.method_route_id != reference.method_route_id:
            failures.append(f"observational_method_route_mismatch:{candidate_id}")
        if str(estimate.estimator_id) != reference.estimator_id:
            failures.append(f"observational_estimator_mismatch:{candidate_id}")
        if estimate.utility_unit != reference.effect_scale_id:
            failures.append(f"observational_effect_scale_mismatch:{candidate_id}")
        if frozenset(str(value) for value in estimate.analysis_covariate_ids) != reference.adjustment_covariates:
            failures.append(f"observational_adjustment_covariates_mismatch:{candidate_id}")
        if not math.isclose(
            float(estimate.confidence_level),
            reference.confidence_level,
            rel_tol=0.0,
            abs_tol=FLOAT_EQUALITY_ABS_TOLERANCE_V1,
        ):
            failures.append(f"observational_confidence_level_mismatch:{candidate_id}")
        if dict(estimate.source_artifact_checksums) != reference.source_artifact_checksums:
            failures.append(f"observational_source_artifact_checksums_mismatch:{candidate_id}")
        components.append(
            point_interval_equivalence_score_v1(
                estimate=float(estimate.estimate),
                lower=float(estimate.lower),
                upper=float(estimate.upper),
                reference_estimate=expected.estimate,
                reference_lower=expected.lower,
                reference_upper=expected.upper,
                absolute_tolerance=reference.absolute_tolerance,
            )
        )
        scientific_components.append(
            scientific_bundle_agreement_v1(
                estimate=float(estimate.estimate),
                lower=float(estimate.lower),
                upper=float(estimate.upper),
                reference_estimate=expected.estimate,
                reference_lower=expected.lower,
                reference_upper=expected.upper,
                envelope=scientific_envelope,
            )
        )
    if failures or len(components) != len(required_candidate_ids):
        return 0.0, False, tuple(sorted(set(failures)))
    return float(min(components)), all(scientific_components), tuple()


def _evaluation_target_register_checksums(
    *,
    scenario_root: Path,
    phase_id: str,
    program_objective_id: str,
    phase_scoring_objective_id: str,
) -> dict[str, str]:
    path = Path(scenario_root) / "grader" / "evaluation_target_register.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"TrialDev evaluation-target register is missing: {path}")
    required_lanes = ["decision_action"]
    if str(phase_id) == "observational_review":
        required_lanes = ["asset_nomination", "phase_analysis"]
    elif str(phase_id) == "phase1":
        required_lanes = ["safety_gate", "decision_action"]
    out: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"evaluation_target_register.jsonl row {line_number} must be an object.")
        if (
            str(payload.get("phase_id")) == str(phase_id)
            and str(payload.get("program_objective_id")) == str(program_objective_id)
            and str(payload.get("phase_scoring_objective_id")) == str(phase_scoring_objective_id)
        ):
            lane_id = str(payload.get("lane_id", ""))
            if lane_id:
                checksum = payload.get("checksum")
                if not isinstance(checksum, str) or len(checksum) != SHA256_HEX_DIGEST_LENGTH:
                    raise ValueError(f"evaluation_target_register.jsonl row {line_number} lacks a valid checksum.")
                out[lane_id] = checksum
    missing = sorted(set(required_lanes) - set(out))
    if missing:
        raise ValueError(
            "evaluation_target_register.jsonl lacks required records for "
            f"phase_id={phase_id!r}, program_objective_id={program_objective_id!r}, "
            f"phase_scoring_objective_id={phase_scoring_objective_id!r}: {missing!r}"
        )
    return out


def _lane_score(
    *,
    register_index: TrialDevEvaluationTargetRegisterIndex,
    scenario_id: str,
    phase_id: str,
    program_objective_id: str,
    phase_scoring_objective_id: str,
    lane_id: str,
    submitted_target_id: str | None,
    score_override: float | None = None,
    score_derivation: (
        Literal[
            "literal_target",
            "numeric_diagnostic",
            "public_evidence_action",
        ]
        | None
    ) = None,
    artifact_status: Literal["present", "missing", "invalid"] = "present",
    failure_reason: str | None = None,
) -> TrialDevelopmentLaneScoreRecordV1:
    evaluation_target = register_index.require(
        phase_id=phase_id,
        program_objective_id=program_objective_id,
        phase_scoring_objective_id=phase_scoring_objective_id,
        lane_id=lane_id,
    )
    return cast(
        TrialDevelopmentLaneScoreRecordV1,
        score_evaluation_target(
            scenario_id=scenario_id,
            phase_id=phase_id,
            program_objective_id=program_objective_id,
            phase_scoring_objective_id=phase_scoring_objective_id,
            lane_id=lane_id,
            submitted_target_id=submitted_target_id,
            evaluation_target=evaluation_target,
            artifact_status=artifact_status,
            failure_reason=failure_reason,
            score_override=score_override,
            score_derivation=(
                score_derivation
                if score_derivation is not None
                else "numeric_diagnostic" if score_override is not None else None
            ),
        ),
    )


def _allowed_actions_for_phase(*, scenario_root: Path, phase_id: str) -> set[str] | None:
    path = Path(scenario_root) / "public" / "phase_action_policy.json"
    if not path.is_file():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("phase_action_policy.json must contain a JSON object.")
    raw_specs = payload.get("action_specs", payload.get("phase_actions", payload.get("phase_action_specs", [])))
    if not isinstance(raw_specs, list | tuple):
        return None
    for record in raw_specs:
        if not isinstance(record, dict) or str(record.get("phase_id", "")) != str(phase_id):
            continue
        raw = record.get("allowed_action_ids", ())
        if not isinstance(raw, list | tuple):
            raise ValueError("phase_action_policy.json allowed_action_ids must be an array.")
        return {str(value) for value in raw}
    return None


def _ranking_concordance(
    *,
    ranked_drug_ids: tuple[str, ...],
    scores_by_drug: dict[str, float],
    candidate_drug_ids: tuple[str, ...],
) -> float:
    relevant = [drug_id for drug_id in ranked_drug_ids if str(drug_id) in set(candidate_drug_ids)]
    if len(relevant) < 2:
        return 1.0 if relevant else 0.0
    missing_scores = sorted({str(drug_id) for drug_id in candidate_drug_ids} - set(scores_by_drug))
    if missing_scores:
        raise ValueError(
            "Policy reference ranking is incomplete for eligible candidates: " + ", ".join(missing_scores)
        )
    reference_rank = {
        drug_id: rank
        for rank, (drug_id, _) in enumerate(
            sorted(
                (
                    (drug_id, float(scores_by_drug[drug_id]))
                    for drug_id in candidate_drug_ids
                    if drug_id in scores_by_drug
                ),
                key=lambda kv: (-kv[1], kv[0]),
            ),
            start=1,
        )
    }
    wins = 0
    total = 0
    for idx, lhs in enumerate(relevant):
        for rhs in relevant[idx + 1 :]:
            lhs_rank = int(reference_rank[str(lhs)])
            rhs_rank = int(reference_rank[str(rhs)])
            total += 1
            if lhs_rank < rhs_rank:
                wins += 1
    return float(wins) / float(total)


def grade_item_v1(
    *,
    scenario_root: Path,
    submission_path: Path,
    write_path: Path | None = None,
    report_mode: str = "audit",
    assigned_objective_id: str | None = None,
    program_objective_id: str | None = None,
    eligible_candidate_drug_ids: tuple[str, ...] | None = None,
    terminal_status: str | None = None,
    trial_output_root: Path | None = None,
) -> TrialDevelopmentGradeReportV1:
    """Grade one submission against one scenario grader bundle."""
    validate_release_v1(scenario_root=scenario_root)
    submission = validate_submission_v1(submission_path=submission_path)
    method_design_contracts = load_public_method_design_contracts_v1(Path(scenario_root))
    numeric_tolerance = _numeric_equivalence_tolerance(scenario_root=Path(scenario_root))
    grader_dir = Path(scenario_root) / "grader"
    register_index = load_evaluation_target_index(Path(scenario_root))

    phase_id = str(submission.request.phase_id)
    request_objective = (
        None if submission.request.selection_objective is None else str(submission.request.selection_objective)
    )
    submitted_objective = (
        None if submission.program_decision.objective_id is None else str(submission.program_decision.objective_id)
    )
    if assigned_objective_id is None:
        objective_id = str(request_objective or submitted_objective or "benefit_risk")
        if submitted_objective is not None and submitted_objective != objective_id:
            raise ValueError(
                "program_decision.objective_id must match request.selection_objective for direct grading."
            )
        resolved_program_objective_id = str(program_objective_id or objective_id)
    else:
        objective_id = str(assigned_objective_id)
        resolved_program_objective_id = str(program_objective_id or objective_id)
        if submitted_objective is not None and submitted_objective != objective_id:
            raise ValueError("program_decision.objective_id cannot override assigned objective.")
        if request_objective is not None and request_objective not in {objective_id, resolved_program_objective_id}:
            raise ValueError("request.selection_objective is inconsistent with assigned scoring context.")
    if phase_id in {"phase1", "phase2", "phase3"}:
        validate_request_against_scenario_v1(
            scenario_root=Path(scenario_root),
            request=submission.request,
        )
    candidate_set = tuple(str(value) for value in submission.request.candidate_drug_ids)
    candidate_eligibility = _resolve_candidate_eligibility(
        scenario_root=Path(scenario_root),
        scenario_id=str(submission.scenario_id),
        phase_id=phase_id,
        candidate_drug_ids=candidate_set,
        eligible_candidate_drug_ids=eligible_candidate_drug_ids,
        terminal_status=terminal_status,
    )
    ranking_reference = list(read_json(grader_dir / "drug_ranking_reference_manifest.json").get("records", []) or [])
    allowed_actions = _allowed_actions_for_phase(scenario_root=Path(scenario_root), phase_id=phase_id)
    for lane_id in required_trialdev_lanes_v1(phase_id):
        register_index.require(
            phase_id=phase_id,
            program_objective_id=resolved_program_objective_id,
            phase_scoring_objective_id=objective_id,
            lane_id=lane_id,
        )
    evaluation_target_checksums = register_index.checksums_for(
        phase_id=phase_id,
        program_objective_id=resolved_program_objective_id,
        phase_scoring_objective_id=objective_id,
    )

    ranking_records = _records_by_context(ranking_reference, phase_id=phase_id, objective_id=objective_id)
    objective_scores = {
        _single_record_candidate(record): _record_numeric_value(record)
        for record in ranking_records
        if str(record.get("metric")) == "objective_score"
    }

    feasibility_failures = [
        _GradeFailure(code=str(detail), invalid=True)
        for record in candidate_eligibility
        for detail in (record.failure_reason_detail,)
        if detail is not None
    ]
    decision_action = submission.program_decision.decision_action
    if allowed_actions is not None and decision_action is not None and str(decision_action) not in allowed_actions:
        feasibility_failures.append(_GradeFailure(code=f"invalid_action:{decision_action}", invalid=True))
    if phase_id == "observational_review":
        selected_winner = submission.program_decision.recommended_drug_id
    elif submission.program_decision.recommended_drug_id is not None:
        selected_winner = submission.program_decision.recommended_drug_id
    elif len(candidate_set) == 1:
        selected_winner = candidate_set[0]
    else:
        selected_winner = None
    qualified_observational_response = bool(
        phase_id == "observational_review" and submission.analysis_report.response_branch == "qualified_non_nomination"
    )
    qualified_observational_failures = (
        _qualified_observational_evidence_failures(
            scenario_root=Path(scenario_root),
            submission=submission,
        )
        if qualified_observational_response
        else tuple()
    )
    resolved_observational_method = (
        resolve_observational_method_route_v1(
            estimates=submission.analysis_report.candidate_utility_estimates,
            contracts=method_design_contracts,
        )
        if phase_id == "observational_review" and not qualified_observational_response
        else None
    )
    observational_reference_method = resolved_observational_method
    if (
        phase_id == "observational_review"
        and not qualified_observational_response
        and observational_reference_method is None
    ):
        declared_method_ids = {
            estimate.method_route_id
            for estimate in submission.analysis_report.candidate_utility_estimates
            if estimate.method_route_id is not None
        }
        declared_matches = tuple(
            method
            for method in method_design_contracts.observational_methods.methods
            if declared_method_ids == {method.method_route_id}
        )
        if len(declared_matches) == 1:
            observational_reference_method = declared_matches[0]
    observational_reference = (
        load_public_observational_reference_v1(
            scenario_root=Path(scenario_root),
            objective_id=objective_id,
            method_route_id=observational_reference_method.method_route_id,
        )
        if observational_reference_method is not None
        and _public_observational_method_supports_point_result(
            scenario_root=Path(scenario_root),
            objective_id=objective_id,
            method_route_id=observational_reference_method.method_route_id,
        )
        else None
    )
    scoring_reference_scores = (
        {
            candidate_id: reference.estimate
            for candidate_id, reference in observational_reference.candidate_utilities.items()
        }
        if observational_reference is not None
        else ({} if phase_id == "observational_review" else objective_scores)
    )
    global_best, global_best_score = _best_in_set(
        scores_by_drug=scoring_reference_scores,
        candidate_drug_ids=tuple(scoring_reference_scores),
    )
    in_set_best, in_set_best_score = _best_in_set(
        scores_by_drug=scoring_reference_scores,
        candidate_drug_ids=candidate_set,
    )
    scoring_drug = str(selected_winner or "")
    selected_is_eligible = any(
        record.candidate_drug_id == scoring_drug and record.sequentially_eligible for record in candidate_eligibility
    )
    has_selected_scoring_drug = bool(
        scoring_drug
        and selected_is_eligible
        and (
            scoring_drug in scoring_reference_scores
            or (phase_id in {"phase1", "phase2", "phase3"} and scoring_drug in set(candidate_set))
        )
    )

    if not has_selected_scoring_drug and phase_id != "observational_review":
        feasibility_failures.append(_GradeFailure(code="missing_selected_asset"))
    design_score = 1.0
    design_witness: TrialDevPhaseDesignWitnessV1 | None = None
    design_efficiency: TrialDevDesignEfficiencyV1 | None = None
    if phase_id in {"phase1", "phase2", "phase3"}:
        if not has_selected_scoring_drug:
            design_score = 0.0
        elif trial_output_root is None:
            raise ValueError("Materialized phase grading requires trial_output_root for design evidence.")
        else:
            design_witness = derive_phase_design_witness_v1(
                scenario_root=Path(scenario_root),
                request=submission.request,
                trial_output_root=Path(trial_output_root),
                phase_id=phase_id,
            )
            design_score = float(design_witness.adequate)
            design_efficiency = derive_phase_design_efficiency_v1(
                scenario_root=Path(scenario_root),
                request=submission.request,
                design_witness=design_witness,
            )
            feasibility_failures.extend(_GradeFailure(code=f"design:{failure}") for failure in design_witness.failures)

    decision_witness = None
    if str(phase_id) in {"phase1", "phase2", "phase3"}:
        if trial_output_root is None:
            raise ValueError("Materialized phase grading requires trial_output_root for public-evidence reference.")
        decision_witness = derive_phase_decision_witness_v1(
            scenario_root=Path(scenario_root),
            trial_output_root=Path(trial_output_root),
            phase_id=phase_id,
        )
    serious_reference: dict[str, float] = {}
    discontinuation_reference: dict[str, float] = {}
    safety_state_by_candidate: dict[str, str] = {}
    safety_state_diagnostics_by_candidate: dict[str, object] = {}
    efficacy_diagnostics_by_candidate: dict[str, object] = {}
    if decision_witness is not None and trial_output_root is not None:
        candidate_evidence = decision_witness.evidence.get("candidates")
        if not isinstance(candidate_evidence, dict):
            raise ValueError("Public decision witness lacks candidate-bound safety evidence.")
        for candidate_id, raw_evidence in candidate_evidence.items():
            if not isinstance(raw_evidence, dict):
                raise ValueError("Public decision witness candidate evidence must be an object.")
            safety_payload = raw_evidence.get("safety")
            if not str(candidate_id) or not isinstance(safety_payload, dict):
                raise ValueError("Public decision witness lacks candidate safety evidence.")
            candidate_key = str(candidate_id)
            serious_reference[candidate_key] = float(safety_payload["treated_serious_rate"])
            discontinuation_payload = safety_payload.get("discontinuation")
            if isinstance(discontinuation_payload, dict):
                discontinuation_reference[candidate_key] = float(discontinuation_payload["treated_rate"])
            safety_state_by_candidate[candidate_key] = str(raw_evidence.get("safety_state"))
            safety_state_diagnostics_by_candidate[candidate_key] = safety_payload
            efficacy_diagnostics_by_candidate[candidate_key] = raw_evidence.get("efficacy")

    primary_effect = submission.analysis_report.primary_effect
    safety_estimate = submission.analysis_report.safety_estimate
    submitted_analysis_cell_id = (
        "qualified_nonidentification"
        if qualified_observational_response
        else (
            resolved_observational_method.method_route_id
            if resolved_observational_method is not None
            else submitted_analysis_cell_id_v1(
                phase_id=phase_id,
                utility_estimates=submission.analysis_report.candidate_utility_estimates,
                effect=primary_effect,
                safety=safety_estimate,
            )
        )
    )
    submitted_design_cell_id = submission.request.design_cell_id
    analysis_method_valid = False
    effect_valid = False
    safety_valid = False
    design_cell_membership_valid = False
    analysis_method_diagnostics: dict[str, object] = {
        "submitted_method_route_id": submitted_analysis_cell_id,
        "resolved_method_route_id": None,
        "required_method_route_id": None,
        "observational_contract_match": None,
        "effect_contract_match": None,
        "effect_source_checksums_match": None,
        "safety_contract_match": None,
        "safety_source_checksums_match": None,
    }
    if phase_id == "observational_review":
        credit_eligible_method_ids = tuple(
            method.method_route_id for method in method_design_contracts.observational_methods.methods
        )
        analysis_method_diagnostics["required_method_route_id"] = credit_eligible_method_ids
        analysis_method_valid = (
            not qualified_observational_failures
            if qualified_observational_response
            else observational_submission_matches_cell_v1(
                estimates=submission.analysis_report.candidate_utility_estimates,
                contracts=method_design_contracts,
                source_checksums=(
                    {} if observational_reference is None else observational_reference.source_artifact_checksums
                ),
            )
        )
        analysis_method_diagnostics["observational_contract_match"] = analysis_method_valid
        analysis_method_diagnostics["qualified_non_nomination_evidence_failures"] = qualified_observational_failures
        if analysis_method_valid:
            analysis_method_diagnostics["resolved_method_route_id"] = submitted_analysis_cell_id
    elif phase_id in {"phase1", "phase2", "phase3"}:
        phase_method, phase_design = required_phase_cells_v1(
            contracts=method_design_contracts,
            phase_id=phase_id,
        )
        analysis_method_diagnostics["required_method_route_id"] = phase_method.method_route_id
        design_cell_membership_valid = submitted_design_cell_id == phase_design.design_cell_id
        if not design_cell_membership_valid:
            feasibility_failures.append(_GradeFailure(code="unaccepted_design_cell"))
        if decision_witness is None or submission.request.follow_up_days is None:
            raise ValueError("Randomized method resolution requires decision evidence and a horizon.")
        source_checksums = decision_witness.evidence.get("source_checksums")
        if not isinstance(source_checksums, dict) or any(
            not isinstance(path, str) or not isinstance(checksum, str) for path, checksum in source_checksums.items()
        ):
            raise ValueError("Public decision witness lacks exact source checksums.")
        safety_valid = bool(
            safety_estimate is not None
            and safety_submission_matches_cell_v1(
                safety=safety_estimate,
                cell=phase_method,
                phase_id=phase_id,
                horizon_days=int(submission.request.follow_up_days),
                source_checksums={str(path): str(checksum) for path, checksum in source_checksums.items()},
            )
        )
        expected_safety_checksums = {str(path): str(checksum) for path, checksum in source_checksums.items()}
        analysis_method_diagnostics["safety_contract_match"] = safety_valid
        analysis_method_diagnostics["safety_source_checksums_match"] = bool(
            safety_estimate is not None and safety_estimate.source_artifact_checksums == expected_safety_checksums
        )
        effect_valid = phase_id == "phase1"
        if phase_id == "phase1":
            analysis_method_diagnostics["effect_contract_match"] = True
        if phase_id in {"phase2", "phase3"}:
            effect_source_checksums = (
                {}
                if primary_effect is None
                else {
                    "trial_output/arm_mapping.json": str(source_checksums.get("trial_output/arm_mapping.json", "")),
                    "trial_output/endpoints.parquet": str(source_checksums.get("trial_output/endpoints.parquet", "")),
                    "trial_output/request.json": str(source_checksums.get("trial_output/request.json", "")),
                }
            )
            effect_valid = bool(
                primary_effect is not None
                and submission.request.treatment_discontinuation_strategy is not None
                and effect_submission_matches_cell_v1(
                    effect=primary_effect,
                    cell=phase_method,
                    phase_id=phase_id,
                    treatment_discontinuation_strategy=str(submission.request.treatment_discontinuation_strategy),
                    horizon_days=int(submission.request.follow_up_days),
                    source_checksums=effect_source_checksums,
                )
            )
            analysis_method_diagnostics["effect_contract_match"] = effect_valid
            analysis_method_diagnostics["effect_source_checksums_match"] = bool(
                primary_effect is not None and primary_effect.source_artifact_checksums == effect_source_checksums
            )
        analysis_method_valid = safety_valid and effect_valid
        if analysis_method_valid:
            analysis_method_diagnostics["resolved_method_route_id"] = phase_method.method_route_id
    if not analysis_method_valid:
        feasibility_failures.append(_GradeFailure(code="unaccepted_analysis_method_route"))
    supporting_evidence_ids = set(submission.program_decision.supporting_evidence_ids)
    if scoring_drug and primary_effect is not None and str(primary_effect.candidate_drug_id) != scoring_drug:
        feasibility_failures.append(_GradeFailure(code="effect_candidate_mismatch", invalid=True))
    if scoring_drug and safety_estimate is not None and str(safety_estimate.candidate_drug_id) != scoring_drug:
        feasibility_failures.append(_GradeFailure(code="safety_candidate_mismatch", invalid=True))

    winner_score = float(scoring_drug in set(candidate_set))
    if str(phase_id) == "observational_review":
        if qualified_observational_response:
            winner_score = float(not qualified_observational_failures)
        else:
            observational_target = str(selected_winner) if selected_winner is not None else str(decision_action or "")
            asset_lane = _lane_score(
                register_index=register_index,
                scenario_id=str(submission.scenario_id),
                phase_id=phase_id,
                program_objective_id=resolved_program_objective_id,
                phase_scoring_objective_id=objective_id,
                lane_id="asset_nomination",
                submitted_target_id=observational_target or None,
            )
            winner_score = float(asset_lane.score)
    ranking_lane_active = (
        len(candidate_set) > 1 and len({drug for drug in candidate_set if drug in scoring_reference_scores}) > 1
    )
    ranking_score = (
        float(not qualified_observational_failures)
        if qualified_observational_response
        else (
            _ranking_concordance(
                ranked_drug_ids=tuple(submission.analysis_report.ranked_drug_ids),
                scores_by_drug=scoring_reference_scores,
                candidate_drug_ids=candidate_set,
            )
            if ranking_lane_active
            else 1.0
        )
    )
    shallow_diagnostics: dict[str, object] = {
        "single_candidate_only": bool(len(candidate_set) < 2),
        "no_uncertainty": bool(primary_effect is None),
        "no_sensitivity": bool(
            str(phase_id) in {"phase2", "phase3"}
            and not (
                primary_effect is not None
                and any(
                    artifact.metric_family == "sensitivity"
                    and artifact.artifact_id in set(primary_effect.diagnostic_evidence_ids)
                    for artifact in submission.analysis_report.diagnostic_artifacts
                )
            )
        ),
        "no_returned_data_summary": bool(
            not submission.analysis_report.diagnostic_artifacts and primary_effect is None and safety_estimate is None
        ),
        "no_multi_candidate_ranking": bool(len(tuple(submission.analysis_report.ranked_drug_ids)) < 2),
        "ceiling_applied": None,
    }
    effect_score = 0.0
    endpoint_consistency_score = 1.0
    reported_effect_endpoint_id = None if primary_effect is None else primary_effect.endpoint_id
    phase_effect_scored = bool(str(phase_id) in {"phase2", "phase3"})
    if primary_effect is not None and phase_effect_scored:
        request_endpoint_id = submission.request.endpoint_id
        if request_endpoint_id is not None and reported_effect_endpoint_id is not None:
            endpoint_consistency_score = 1.0 if str(reported_effect_endpoint_id) == str(request_endpoint_id) else 0.0
            if endpoint_consistency_score == 0.0:
                feasibility_failures.append(_GradeFailure(code="effect_endpoint_mismatch", invalid=True))
        elif request_endpoint_id is not None and reported_effect_endpoint_id is None:
            endpoint_consistency_score = 0.0
            feasibility_failures.append(_GradeFailure(code="effect_endpoint_missing", invalid=True))
    effect_reference = None
    if primary_effect is not None and trial_output_root is not None and phase_id in {"phase2", "phase3"}:
        references = derive_effect_references_v1(
            scenario_root=Path(scenario_root),
            trial_output_root=Path(trial_output_root),
        )
        effect_reference = next(
            (
                reference
                for reference in references
                if reference.candidate_drug_id == str(primary_effect.candidate_drug_id)
                and reference.endpoint_id == str(primary_effect.endpoint_id)
                and reference.estimand_id == str(primary_effect.estimand_id)
                and reference.estimator_id == str(primary_effect.estimator_id)
                and reference.effect_scale_id == str(primary_effect.effect_scale_id)
                and reference.horizon_days == int(primary_effect.horizon_days or 0)
            ),
            None,
        )
        if effect_reference is None:
            feasibility_failures.append(_GradeFailure(code="unaccepted_effect_method_route"))
        else:
            if not analysis_method_valid:
                feasibility_failures.append(_GradeFailure(code="unaccepted_effect_method_route"))
        if effect_reference is not None and not math.isclose(
            float(primary_effect.confidence_level),
            float(effect_reference.confidence_level),
            rel_tol=0.0,
            abs_tol=FLOAT_EQUALITY_ABS_TOLERANCE_V1,
        ):
            feasibility_failures.append(_GradeFailure(code="effect_confidence_level_mismatch"))
        elif effect_reference is not None:
            effect_score = point_equivalence_score_v1(
                observed=float(primary_effect.estimate),
                expected=float(effect_reference.estimate),
                absolute_tolerance=effect_reference.absolute_tolerance,
            )
            effect_score = float(effect_score) * float(endpoint_consistency_score)
    interval_calibration_score = 0.0
    randomized_scientific_components: list[bool] = []
    randomized_scientific_thresholds: list[float] = []
    if primary_effect is not None and effect_reference is not None:
        lo = float(primary_effect.lower)
        hi = float(primary_effect.upper)
        interval_calibration_score = interval_equivalence_score_v1(
            lower=lo,
            upper=hi,
            reference_lower=float(effect_reference.lower),
            reference_upper=float(effect_reference.upper),
            endpoint_tolerance=effect_reference.absolute_tolerance,
        )
        efficacy_reference = efficacy_diagnostics_by_candidate.get(scoring_drug)
        if isinstance(efficacy_reference, dict):
            minimum_benefit = efficacy_reference.get("minimum_benefit")
            if isinstance(minimum_benefit, bool) or not isinstance(minimum_benefit, int | float):
                raise ValueError("Public efficacy evidence lacks its minimum-benefit threshold.")
            threshold = float(minimum_benefit)
            randomized_scientific_thresholds.append(threshold)
            randomized_scientific_components.append(
                scientific_bundle_agreement_v1(
                    estimate=float(primary_effect.estimate),
                    lower=float(primary_effect.lower),
                    upper=float(primary_effect.upper),
                    reference_estimate=float(effect_reference.estimate),
                    reference_lower=float(effect_reference.lower),
                    reference_upper=float(effect_reference.upper),
                    envelope=TrialDevScientificEnvelopeV1(
                        envelope_id="trialdev_declared_efficacy_threshold_v1",
                        basis="declared_decision_thresholds",
                        decision_thresholds=(threshold,),
                        exact_reproduction_tolerance=effect_reference.absolute_tolerance,
                    ),
                )
            )
    required_evidence_flags: list[str] = []
    effect_required = bool(str(phase_id) in {"phase2", "phase3"})
    interval_required = bool(str(phase_id) in {"phase2", "phase3"})
    safety_components: list[float] = []
    if safety_estimate is not None and scoring_drug in safety_state_by_candidate:
        witness_confidence = (
            None if decision_witness is None else _payload_float(decision_witness.evidence, "confidence_level")
        )
        if witness_confidence is None or not math.isclose(
            float(safety_estimate.confidence_level),
            float(witness_confidence),
            rel_tol=0.0,
            abs_tol=FLOAT_EQUALITY_ABS_TOLERANCE_V1,
        ):
            feasibility_failures.append(_GradeFailure(code="safety_confidence_level_mismatch"))
    if safety_estimate is not None and scoring_drug in serious_reference:
        safety_reference = safety_state_diagnostics_by_candidate[scoring_drug]
        if not isinstance(safety_reference, dict):
            raise ValueError("Public serious-AE reference must be an object.")
        serious_references = (
            (
                safety_estimate.serious_ae_treatment_rate,
                safety_estimate.serious_ae_treatment_lower,
                safety_estimate.serious_ae_treatment_upper,
                safety_reference.get("treated_serious_rate"),
                safety_reference.get("treated_serious_rate_interval"),
            ),
            (
                safety_estimate.serious_ae_control_rate,
                safety_estimate.serious_ae_control_lower,
                safety_estimate.serious_ae_control_upper,
                safety_reference.get("control_serious_rate"),
                safety_reference.get("control_serious_rate_interval"),
            ),
            (
                safety_estimate.serious_ae_excess,
                safety_estimate.serious_ae_excess_lower,
                safety_estimate.serious_ae_excess_upper,
                safety_reference.get("serious_rate_excess"),
                safety_reference.get("serious_rate_excess_interval"),
            ),
        )
        for estimate, lower, upper, reference_estimate, reference_interval in serious_references:
            if (
                isinstance(reference_estimate, bool)
                or not isinstance(reference_estimate, int | float)
                or not isinstance(reference_interval, list | tuple)
                or len(reference_interval) != 2
            ):
                raise ValueError("Public serious-AE reference lacks its complete safety bundle.")
            safety_components.append(
                point_interval_equivalence_score_v1(
                    estimate=float(estimate),
                    lower=float(lower),
                    upper=float(upper),
                    reference_estimate=float(reference_estimate),
                    reference_lower=float(reference_interval[0]),
                    reference_upper=float(reference_interval[1]),
                    absolute_tolerance=numeric_tolerance,
                )
            )
    elif scoring_drug in serious_reference:
        required_evidence_flags.append("missing_required_component:serious_ae_bundle")
        safety_components.append(0.0)
    if safety_estimate is not None and scoring_drug in discontinuation_reference:
        safety_reference = safety_state_diagnostics_by_candidate[scoring_drug]
        discontinuation_bundle = (
            safety_reference.get("discontinuation") if isinstance(safety_reference, dict) else None
        )
        if not isinstance(discontinuation_bundle, dict):
            raise ValueError("Public discontinuation reference must be an object.")
        discontinuation_references = (
            (
                safety_estimate.discontinuation_treatment_rate,
                safety_estimate.discontinuation_treatment_lower,
                safety_estimate.discontinuation_treatment_upper,
                discontinuation_bundle.get("treated_rate"),
                discontinuation_bundle.get("treated_rate_interval"),
            ),
            (
                safety_estimate.discontinuation_control_rate,
                safety_estimate.discontinuation_control_lower,
                safety_estimate.discontinuation_control_upper,
                discontinuation_bundle.get("control_rate"),
                discontinuation_bundle.get("control_rate_interval"),
            ),
            (
                safety_estimate.discontinuation_excess,
                safety_estimate.discontinuation_excess_lower,
                safety_estimate.discontinuation_excess_upper,
                discontinuation_bundle.get("rate_excess"),
                discontinuation_bundle.get("rate_excess_interval"),
            ),
        )
        for estimate, lower, upper, reference_estimate, reference_interval in discontinuation_references:
            if (
                isinstance(reference_estimate, bool)
                or not isinstance(reference_estimate, int | float)
                or not isinstance(reference_interval, list | tuple)
                or len(reference_interval) != 2
            ):
                raise ValueError("Public discontinuation reference lacks its complete safety bundle.")
            safety_components.append(
                point_interval_equivalence_score_v1(
                    estimate=float(estimate),
                    lower=float(lower),
                    upper=float(upper),
                    reference_estimate=float(reference_estimate),
                    reference_lower=float(reference_interval[0]),
                    reference_upper=float(reference_interval[1]),
                    absolute_tolerance=numeric_tolerance,
                )
            )
    elif scoring_drug in discontinuation_reference:
        required_evidence_flags.append("missing_required_component:discontinuation_bundle")
        safety_components.append(0.0)
    if safety_estimate is not None and scoring_drug in safety_state_diagnostics_by_candidate:
        safety_reference = safety_state_diagnostics_by_candidate[scoring_drug]
        if not isinstance(safety_reference, dict):
            raise ValueError("Public safety reference must be an object.")
        scientific_safety_bundles: list[tuple[str, float, float, float, object, object, object]] = [
            (
                "serious_absolute",
                safety_estimate.serious_ae_treatment_rate,
                safety_estimate.serious_ae_treatment_lower,
                safety_estimate.serious_ae_treatment_upper,
                safety_reference.get("treated_serious_rate"),
                safety_reference.get("treated_serious_rate_interval"),
                safety_reference.get("absolute_limit"),
            ),
            (
                "serious_excess",
                safety_estimate.serious_ae_excess,
                safety_estimate.serious_ae_excess_lower,
                safety_estimate.serious_ae_excess_upper,
                safety_reference.get("serious_rate_excess"),
                safety_reference.get("serious_rate_excess_interval"),
                safety_reference.get("excess_limit"),
            ),
        ]
        discontinuation_reference_bundle = safety_reference.get("discontinuation")
        if isinstance(discontinuation_reference_bundle, dict):
            scientific_safety_bundles.extend(
                [
                    (
                        "discontinuation_absolute",
                        safety_estimate.discontinuation_treatment_rate,
                        safety_estimate.discontinuation_treatment_lower,
                        safety_estimate.discontinuation_treatment_upper,
                        discontinuation_reference_bundle.get("treated_rate"),
                        discontinuation_reference_bundle.get("treated_rate_interval"),
                        discontinuation_reference_bundle.get("absolute_limit"),
                    ),
                    (
                        "discontinuation_excess",
                        safety_estimate.discontinuation_excess,
                        safety_estimate.discontinuation_excess_lower,
                        safety_estimate.discontinuation_excess_upper,
                        discontinuation_reference_bundle.get("rate_excess"),
                        discontinuation_reference_bundle.get("rate_excess_interval"),
                        discontinuation_reference_bundle.get("excess_limit"),
                    ),
                ]
            )
        for (
            bundle_id,
            estimate,
            lower,
            upper,
            reference_estimate,
            reference_interval,
            raw_threshold,
        ) in scientific_safety_bundles:
            if (
                isinstance(reference_estimate, bool)
                or not isinstance(reference_estimate, int | float)
                or not isinstance(reference_interval, list | tuple)
                or len(reference_interval) != 2
                or isinstance(raw_threshold, bool)
                or not isinstance(raw_threshold, int | float)
            ):
                raise ValueError(f"Public {bundle_id} evidence lacks its decision boundary.")
            threshold = float(raw_threshold)
            randomized_scientific_thresholds.append(threshold)
            randomized_scientific_components.append(
                scientific_bundle_agreement_v1(
                    estimate=float(estimate),
                    lower=float(lower),
                    upper=float(upper),
                    reference_estimate=float(reference_estimate),
                    reference_lower=float(reference_interval[0]),
                    reference_upper=float(reference_interval[1]),
                    envelope=TrialDevScientificEnvelopeV1(
                        envelope_id=f"trialdev_declared_{bundle_id}_threshold_v1",
                        basis="declared_decision_thresholds",
                        decision_thresholds=(threshold,),
                        exact_reproduction_tolerance=numeric_tolerance,
                    ),
                )
            )
    evaluation_components: list[float] = []
    observational_score = 0.0
    observational_scientific_agreement = False
    observational_analysis_failures: tuple[str, ...] = tuple()
    if phase_id == "observational_review":
        if qualified_observational_response:
            observational_analysis_failures = qualified_observational_failures
            observational_score = 0.0 if qualified_observational_failures else 1.0
            observational_scientific_agreement = not qualified_observational_failures
            evaluation_components.append(observational_score)
        elif observational_reference is None:
            observational_analysis_failures = ("observational_method_not_resolved",)
            evaluation_components.append(0.0)
        else:
            public_candidates = frozenset(_public_candidate_ids(scenario_root=Path(scenario_root)))
        if observational_reference is not None and frozenset(candidate_set) != public_candidates:
            observational_analysis_failures = ("observational_request_candidate_coverage_mismatch",)
            evaluation_components.append(0.0)
        elif observational_reference is not None:
            (
                observational_score,
                observational_scientific_agreement,
                observational_analysis_failures,
            ) = _score_observational_analysis(
                submission=submission,
                reference=observational_reference,
                objective_id=objective_id,
                required_candidate_ids=public_candidates,
                supporting_evidence_ids=supporting_evidence_ids,
            )
            evaluation_components.append(observational_score)
        globally_invalid_observational_failures = {
            "observational_request_candidate_coverage_mismatch",
            "observational_selected_candidate_missing_or_invalid",
            "observational_decline_declares_candidate",
            "observational_invalid_decision_action",
        }
        feasibility_failures.extend(
            _GradeFailure(
                code=failure,
                invalid=failure in globally_invalid_observational_failures,
            )
            for failure in observational_analysis_failures
        )
    if primary_effect is not None and phase_effect_scored:
        evaluation_components.append(float(min(effect_score, interval_calibration_score)))
    elif effect_required:
        required_evidence_flags.append("missing_required_component:effect_estimate")
        evaluation_components.append(0.0)
    if primary_effect is None and interval_required:
        required_evidence_flags.append("missing_required_component:effect_interval")
    subgroup_claims_valid = True
    if submission.analysis_report.claimed_subgroup_variables:
        claimed = {str(value) for value in submission.analysis_report.claimed_subgroup_variables}
        unsupported = sorted(claimed - set(submission.request.subgroup_variables))
        if unsupported:
            subgroup_claims_valid = False
            feasibility_failures.extend(
                _GradeFailure(code=f"unsupported_subgroup_claim:{value}", invalid=True) for value in unsupported
            )
            evaluation_components.append(0.0)
    evaluation_components.extend(safety_components)
    numeric_evaluation_score = float(min(evaluation_components)) if evaluation_components else 1.0
    scientific_evaluation_agreement = (
        observational_scientific_agreement
        if phase_id == "observational_review"
        else bool(randomized_scientific_components) and all(randomized_scientific_components)
    )
    evaluation_score = float(scientific_evaluation_agreement and not required_evidence_flags and analysis_method_valid)

    observational_eligible = phase_id == "observational_review"
    randomized_effect_eligible = phase_id in {"phase2", "phase3"}
    safety_eligible = phase_id in {"phase1", "phase2", "phase3"}
    observational_valid = bool(analysis_method_valid and not observational_analysis_failures and subgroup_claims_valid)
    randomized_effect_valid = bool(
        effect_valid
        and primary_effect is not None
        and effect_reference is not None
        and "missing_required_component:effect_estimate" not in required_evidence_flags
        and "missing_required_component:effect_interval" not in required_evidence_flags
        and subgroup_claims_valid
    )
    safety_evidence_valid = bool(
        safety_valid
        and safety_components
        and not any(flag.endswith("_bundle") for flag in required_evidence_flags)
        and subgroup_claims_valid
    )
    analysis_quality = TrialDevelopmentAnalysisQualityV1(
        observational_analysis_eligible=observational_eligible,
        observational_analysis_valid=observational_valid if observational_eligible else None,
        observational_analysis_score=(
            (float(observational_score) if observational_eligible and observational_valid else 0.0)
            if observational_eligible
            else None
        ),
        randomized_primary_effect_eligible=randomized_effect_eligible,
        randomized_primary_effect_valid=(randomized_effect_valid if randomized_effect_eligible else None),
        randomized_primary_effect_point_agreement=(
            (float(effect_score) if randomized_effect_valid else 0.0) if randomized_effect_eligible else None
        ),
        randomized_primary_effect_interval_agreement=(
            (float(interval_calibration_score) if randomized_effect_valid else 0.0)
            if randomized_effect_eligible
            else None
        ),
        safety_evidence_eligible=safety_eligible,
        safety_evidence_valid=safety_evidence_valid if safety_eligible else None,
        safety_evidence_agreement=(
            (float(min(safety_components)) if safety_evidence_valid else 0.0) if safety_eligible else None
        ),
        phase_evaluation_valid=(
            (not observational_eligible or observational_valid)
            and (not randomized_effect_eligible or randomized_effect_valid)
            and (not safety_eligible or safety_evidence_valid)
        ),
    )

    decision_action_score: float | None = None
    if decision_action is None:
        initial_action_reason = "missing_decision_action"
    elif allowed_actions is not None and str(decision_action) not in allowed_actions:
        initial_action_reason = "action_not_allowed_for_phase_policy"
    else:
        initial_action_reason = "not_applicable_without_materialized_public_evidence"
    decision_action_diagnostics: dict[str, object] = {
        "decision_action": decision_action,
        "allowed_decision_actions": sorted(allowed_actions or set()),
        "decision_action_score_reason": initial_action_reason,
    }
    public_safety_action_score: float | None = None
    uncertainty_required = False
    uncertainty_evidence_present = False
    if str(phase_id) in {"phase1", "phase2", "phase3"}:
        if decision_witness is None:
            raise ValueError("Materialized phase grading requires a public decision witness.")
        effect_uncertainty_support = False
        efficacy_diagnostics = efficacy_diagnostics_by_candidate.get(scoring_drug)
        if (
            primary_effect is not None
            and primary_effect.evidence_id in supporting_evidence_ids
            and isinstance(efficacy_diagnostics, dict)
        ):
            thresholds = [efficacy_diagnostics.get("minimum_benefit")]
            effect_uncertainty_support = any(
                threshold is not None
                and float(primary_effect.lower) <= float(threshold) <= float(primary_effect.upper)
                for threshold in thresholds
            )
        safety_uncertainty_support = False
        safety_diagnostics = safety_state_diagnostics_by_candidate.get(scoring_drug)
        if (
            safety_estimate is not None
            and safety_estimate.evidence_id in supporting_evidence_ids
            and isinstance(safety_diagnostics, dict)
            and safety_state_by_candidate.get(scoring_drug) == "indeterminate"
        ):
            safety_uncertainty_support = _safety_uncertainty_support_v1(
                safety_estimate=safety_estimate,
                safety_diagnostics=safety_diagnostics,
            )
        uncertainty_evidence_present = bool(effect_uncertainty_support or safety_uncertainty_support)
        action_score = (
            0.0
            if decision_action is None
            else float(
                decision_witness.action_is_acceptable(
                    action_id=str(decision_action),
                    candidate_drug_id=scoring_drug or None,
                )
            )
        )
        uncertainty_required = bool(
            decision_action is not None
            and decision_witness.action_requires_uncertainty(
                action_id=str(decision_action),
                candidate_drug_id=scoring_drug or None,
            )
        )
        if action_score > 0.0 and uncertainty_required:
            action_score = 1.0 if uncertainty_evidence_present else 0.0
        decision_action_score = action_score
        if initial_action_reason != "action_not_allowed_for_phase_policy":
            decision_action_diagnostics["decision_action_score_reason"] = "public_evidence_action_set_membership"
        public_safety_action_score = (
            0.0
            if decision_action is None
            else float(
                decision_witness.safety_action_is_acceptable(
                    action_id=str(decision_action),
                    candidate_drug_id=scoring_drug or None,
                )
            )
        )
        if public_safety_action_score > 0.0 and uncertainty_required and not uncertainty_evidence_present:
            public_safety_action_score = 0.0
        decision_action_diagnostics.update(
            {
                "public_evidence_action_checksum": decision_witness.checksum,
                "acceptable_decision_actions": list(decision_witness.acceptable_action_ids),
                "candidate_bound_decision_actions": list(
                    decision_witness.relevant_action_ids(candidate_drug_id=scoring_drug or None)
                ),
                "decision_recoverability_class": decision_witness.recoverability_class,
                "uncertainty_evidence_required": uncertainty_required,
                "uncertainty_evidence_present": uncertainty_evidence_present,
                "ordinary_efficacy_action_set": list(decision_witness.efficacy_action_ids),
                "safety_action_set": list(decision_witness.safety_action_ids),
                "public_evidence_decision_witness": decision_witness.evidence,
            }
        )
    program_score = float(decision_action_score) if decision_action_score is not None else float(winner_score)

    question_valid = all(record.sequentially_eligible for record in candidate_eligibility)
    if phase_id == "observational_review":
        if qualified_observational_response and not qualified_observational_failures:
            analysis_classification: TrialDevAnalysisClassificationV1 = "uncertainty_qualified"
        elif submission.analysis_report.candidate_utility_estimates and all(
            not estimate.analysis_covariate_ids for estimate in submission.analysis_report.candidate_utility_estimates
        ):
            analysis_classification = "descriptive"
        elif submission.analysis_report.candidate_utility_estimates:
            analysis_classification = "uncertainty_qualified" if analysis_method_valid else "adjusted"
        else:
            analysis_classification = "invalid"
        if qualified_observational_response:
            scientific_envelope: TrialDevScientificEnvelopeV1 | None = TrialDevScientificEnvelopeV1(
                envelope_id="trialdev_qualified_nonidentification_v1",
                basis="qualified_nonidentification",
                exact_reproduction_tolerance=numeric_tolerance,
            )
        elif observational_reference is not None:
            scientific_envelope = TrialDevScientificEnvelopeV1(
                envelope_id="trialdev_declared_utility_indifference_margin_v1",
                basis="declared_practical_equivalence_margin",
                absolute_margin=observational_reference.scientific_absolute_margin,
                exact_reproduction_tolerance=observational_reference.absolute_tolerance,
            )
        else:
            scientific_envelope = None
        if qualified_observational_response:
            required_decision_evidence_ids = {
                evidence.evidence_id for evidence in submission.analysis_report.identification_evidence
            }
        elif decision_action == "withhold_nomination":
            required_decision_evidence_ids = {
                estimate.evidence_id for estimate in submission.analysis_report.candidate_utility_estimates
            }
        else:
            required_decision_evidence_ids = {
                estimate.evidence_id
                for estimate in submission.analysis_report.candidate_utility_estimates
                if selected_winner is not None and estimate.candidate_drug_id == selected_winner
            }
    else:
        analysis_classification = "uncertainty_qualified" if analysis_method_valid else "invalid"
        scientific_envelope = (
            TrialDevScientificEnvelopeV1(
                envelope_id="trialdev_randomized_declared_decision_thresholds_v1",
                basis="declared_decision_thresholds",
                decision_thresholds=tuple(sorted(set(randomized_scientific_thresholds))),
                exact_reproduction_tolerance=numeric_tolerance,
            )
            if randomized_scientific_thresholds
            else None
        )
        required_decision_evidence_ids = {
            evidence_id
            for evidence_id in (
                None if primary_effect is None else primary_effect.evidence_id,
                None if safety_estimate is None else safety_estimate.evidence_id,
            )
            if evidence_id is not None
        }
    uncertainty_valid = bool(
        (phase_id == "observational_review" and analysis_classification != "invalid")
        or (
            phase_id in {"phase1", "phase2", "phase3"}
            and not required_evidence_flags
            and bool(randomized_scientific_components)
        )
    )
    evidence_support_valid = bool(
        required_decision_evidence_ids
        and required_decision_evidence_ids <= supporting_evidence_ids
        and analysis_method_valid
    )
    action_admissible = math.isclose(program_score, 1.0)
    sequentially_coherent = bool(
        question_valid
        and decision_action is not None
        and (allowed_actions is None or str(decision_action) in allowed_actions)
    )
    scientific_status = (
        "not_assessed"
        if scientific_envelope is None or not analysis_method_valid
        else "passed" if scientific_evaluation_agreement else "failed"
    )
    exact_reproduction_status = (
        "not_assessed"
        if scientific_envelope is None
        else "passed" if math.isclose(numeric_evaluation_score, 1.0) else "failed"
    )
    design_status = (
        "not_applicable"
        if phase_id == "observational_review"
        else "passed" if math.isclose(design_score, 1.0) else "failed"
    )
    responsibility_failures = tuple(
        reason
        for failed, reason in (
            (not question_valid, "question_or_candidate_ineligible"),
            (design_status == "failed", "design_not_adequate"),
            (not analysis_method_valid, "analysis_method_not_accepted"),
            (scientific_status == "failed", "scientific_disagreement"),
            (not uncertainty_valid, "uncertainty_not_supported"),
            (not action_admissible, "action_not_admissible"),
            (not evidence_support_valid, "decision_evidence_not_supported"),
            (not sequentially_coherent, "decision_not_sequentially_coherent"),
        )
        if failed
    )
    scientific_assessment = TrialDevScientificAssessmentV1(
        execution="passed",
        question_estimand="passed" if question_valid else "failed",
        design=design_status,
        assumptions="passed" if analysis_method_valid else "failed",
        analysis_classification=analysis_classification,
        scientific_agreement=scientific_status,
        exact_reproduction=exact_reproduction_status,
        uncertainty="passed" if uncertainty_valid else "failed",
        action_admissibility="passed" if action_admissible else "failed",
        evidential_support="passed" if evidence_support_valid else "failed",
        sequential_coherence="passed" if sequentially_coherent else "failed",
        scientific_envelope=scientific_envelope,
        failure_reasons=responsibility_failures,
        decision_complete=bool(
            question_valid
            and design_status in {"passed", "not_applicable"}
            and analysis_classification == "uncertainty_qualified"
            and analysis_method_valid
            and scientific_status == "passed"
            and uncertainty_valid
            and action_admissible
            and evidence_support_valid
            and sequentially_coherent
        ),
    )

    lane_breakdown = {
        "trial_design": float(design_score),
        "trial_evaluation": float(evaluation_score),
        "program_decision": float(program_score),
        "drug_ranking": float(ranking_score),
    }
    if phase_id == "observational_review":
        active_lanes = {"trial_evaluation", "program_decision", "drug_ranking"}
    elif phase_id in {"phase1", "phase2", "phase3"}:
        active_lanes = {"trial_design", "trial_evaluation", "program_decision"}
    else:
        active_lanes = set()
    invalid_reasons = tuple(sorted({failure.code for failure in feasibility_failures if failure.invalid}))
    lane_status = {lane: ("active" if lane in active_lanes else "not_applicable") for lane in _ACTIVE_LANES}
    if invalid_reasons:
        lane_status = {lane: ("invalid" if status == "active" else status) for lane, status in lane_status.items()}
    active_lane_scores = {
        lane: _clamp01(float(lane_breakdown[lane]))
        for lane in sorted(active_lanes)
        if lane_status.get(lane) == "active"
    }
    artifact_status: Literal["present", "missing", "invalid"] = "invalid" if invalid_reasons else "present"
    artifact_failure_reason = ";".join(invalid_reasons) if invalid_reasons else None
    lane_scores: list[TrialDevelopmentLaneScoreRecordV1] = []
    for register_lane in required_trialdev_lanes_v1(phase_id):
        registered_evaluation_target = register_index.require(
            phase_id=phase_id,
            program_objective_id=resolved_program_objective_id,
            phase_scoring_objective_id=objective_id,
            lane_id=register_lane,
        )
        evaluation_target = registered_evaluation_target
        if register_lane == "asset_nomination" and observational_reference is not None:
            evaluation_target = _method_conditioned_asset_reference(
                evaluation_target=evaluation_target,
                reference=observational_reference,
            )
        elif (
            register_lane == "asset_nomination"
            and qualified_observational_response
            and not qualified_observational_failures
        ):
            evaluation_target = _qualified_non_nomination_asset_reference(
                evaluation_target=evaluation_target,
            )
        elif (
            register_lane == "phase_analysis"
            and qualified_observational_response
            and not qualified_observational_failures
        ):
            evaluation_target = _qualified_non_nomination_analysis_reference(
                evaluation_target=evaluation_target,
            )
        score_override: float | None = None
        score_derivation: Literal["numeric_diagnostic", "public_evidence_action"] | None = None
        if register_lane == "asset_nomination":
            submitted_target = (
                str(selected_winner) if selected_winner is not None else str(decision_action or "") or None
            )
        elif register_lane == "decision_action" and decision_witness is not None:
            submitted_target = None if decision_action is None else str(decision_action)
            score_override = float(decision_action_score or 0.0)
            score_derivation = "public_evidence_action"
        elif register_lane == "safety_gate" and decision_witness is not None:
            submitted_target = None if decision_action is None else str(decision_action)
            score_override = float(public_safety_action_score or 0.0)
            score_derivation = "public_evidence_action"
        elif register_lane == "route_timing" and decision_witness is not None:
            submitted_target = None if decision_action is None else str(decision_action)
            score_override = float(decision_action_score or 0.0)
            score_derivation = "public_evidence_action"
        elif register_lane in {"decision_action", "safety_gate", "route_timing", "final_recommendation"}:
            submitted_target = None if decision_action is None else str(decision_action)
        elif register_lane == "phase_design":
            submitted_target = submitted_design_cell_id
            accepted_targets = {
                *evaluation_target.reference_target_ids,
                *evaluation_target.credit_eligible_target_ids,
            }
            if design_cell_membership_valid:
                score_override = float(design_score)
                score_derivation = "numeric_diagnostic"
            elif submitted_target in accepted_targets:
                score_override = 0.0
                score_derivation = "numeric_diagnostic"
        elif register_lane == "phase_analysis":
            submitted_target = submitted_analysis_cell_id
            accepted_targets = {
                *evaluation_target.reference_target_ids,
                *evaluation_target.credit_eligible_target_ids,
            }
            if analysis_method_valid:
                score_override = float(evaluation_score)
                score_derivation = "numeric_diagnostic"
            elif submitted_target in accepted_targets:
                score_override = 0.0
                score_derivation = "numeric_diagnostic"
        else:
            submitted_target = None
        lane_score = score_evaluation_target(
            scenario_id=str(submission.scenario_id),
            phase_id=phase_id,
            program_objective_id=resolved_program_objective_id,
            phase_scoring_objective_id=objective_id,
            lane_id=register_lane,
            submitted_target_id=submitted_target,
            evaluation_target=evaluation_target,
            evaluation_target_register_checksum=str(registered_evaluation_target.checksum),
            artifact_status=artifact_status,
            failure_reason=artifact_failure_reason,
            score_override=score_override,
            score_derivation=score_derivation,
            derived_from_trajectory_metric=False,
        )
        if register_lane in {"decision_action", "safety_gate", "route_timing"} and decision_witness is not None:
            lane_payload = lane_score.model_dump(mode="json", exclude={"checksum"})
            runtime_targets = (
                decision_witness.relevant_action_ids(candidate_drug_id=scoring_drug or None)
                if register_lane == "decision_action"
                else (
                    tuple(
                        action_id
                        for action_id in decision_witness.relevant_action_ids(candidate_drug_id=scoring_drug or None)
                        if decision_witness.safety_action_is_acceptable(
                            action_id=action_id,
                            candidate_drug_id=scoring_drug or None,
                        )
                    )
                    if register_lane == "safety_gate"
                    else decision_witness.relevant_action_ids(candidate_drug_id=scoring_drug or None)
                )
            )
            lane_payload.update(
                {
                    "scoring_policy_id": "trialdev_phase_decision_evidence_policy_v1",
                    "recoverability_policy_id": decision_witness.recoverability_class,
                    "reference_target_ids": runtime_targets,
                    "credit_eligible_target_ids": (),
                }
            )
            lane_score = TrialDevelopmentLaneScoreRecordV1.model_validate(lane_payload)
            decision_action_diagnostics[f"public_evidence_{register_lane}_status"] = str(lane_score.status)
        lane_scores.append(lane_score)

    register_lane_scores = {str(record.lane_id): float(record.score) for record in lane_scores}
    required_register_lanes = set(required_trialdev_lanes_v1(phase_id))
    primary_score = float(min(register_lane_scores[lane] for lane in required_register_lanes))
    if invalid_reasons:
        primary_score = 0.0
    required_analysis_lanes = required_register_lanes & {"phase_design", "phase_analysis"}
    required_decision_lanes = required_register_lanes & {
        "asset_nomination",
        "decision_action",
        "safety_gate",
        "route_timing",
        "final_recommendation",
    }
    grade_gates, first_failure_gate = _ordered_grade_gates_v1(
        question_valid=all(record.sequentially_eligible for record in candidate_eligibility),
        route_valid=bool(
            analysis_method_valid and (phase_id not in {"phase1", "phase2", "phase3"} or design_cell_membership_valid)
        ),
        evidence_valid=bool(not required_evidence_flags and not qualified_observational_failures),
        result_valid=bool(
            math.isclose(evaluation_score, 1.0)
            and (phase_id not in {"phase1", "phase2", "phase3"} or math.isclose(design_score, 1.0))
        ),
        conformance_valid=all(math.isclose(register_lane_scores[lane], 1.0) for lane in required_analysis_lanes),
        decision_valid=all(math.isclose(register_lane_scores[lane], 1.0) for lane in required_decision_lanes),
    )

    selected_reference_score = (
        scoring_reference_scores.get(str(selected_winner)) if selected_winner is not None else None
    )
    policy_reference_regret = (
        float(global_best_score - selected_reference_score)
        if global_best is not None and selected_reference_score is not None
        else None
    )
    in_set_regret = (
        float(in_set_best_score - selected_reference_score)
        if in_set_best is not None and selected_reference_score is not None
        else None
    )

    report = TrialDevelopmentGradeReportV1(
        scenario_id=str(submission.scenario_id),
        phase_id=phase_id,
        objective_id=objective_id,
        program_objective_id=resolved_program_objective_id,
        phase_scoring_objective_id=objective_id,
        primary_score=_clamp01(float(primary_score)),
        design_score=float(design_score),
        evaluation_score=float(evaluation_score),
        program_score=float(program_score),
        ranking_score=float(ranking_score),
        analysis_quality=analysis_quality,
        scientific_assessment=scientific_assessment,
        lane_status=lane_status,
        active_lane_scores=active_lane_scores,
        gates=grade_gates,
        first_failure_gate=cast(
            Literal[
                "submission",
                "question",
                "route",
                "evidence",
                "integrity",
                "result",
                "conformance",
                "decision",
            ]
            | None,
            first_failure_gate,
        ),
        validity=TrialDevelopmentValidityReportV1(
            valid=not invalid_reasons,
            invalid_reasons=invalid_reasons,
            warnings=tuple(sorted(set(required_evidence_flags))),
        ),
        audit_gates=TrialDevelopmentAuditGateReportV1(
            gates_triggered=tuple(),
            diagnostic_alignment_score=None,
        ),
        design_efficiency=design_efficiency,
        policy_reference_regret=policy_reference_regret,
        in_set_regret=in_set_regret,
        selected_winner_drug_id=None if selected_winner is None else str(selected_winner),
        best_candidate_drug_id=None if in_set_best is None else str(in_set_best),
        feasibility_failures=tuple(sorted({failure.code for failure in feasibility_failures})),
        lane_breakdown=lane_breakdown,
        lane_scores=tuple(lane_scores),
        payload={
            "global_best_candidate_drug_id": global_best,
            "global_best_score": float(global_best_score),
            "in_set_best_score": float(in_set_best_score),
            "submitted_candidate_drug_ids": list(candidate_set),
            "candidate_eligibility": [
                record.model_dump(mode="json", exclude_none=True) for record in candidate_eligibility
            ],
            "effect_reference": (
                None
                if effect_reference is None
                else {
                    "checksum": effect_reference.checksum,
                    "estimand_id": effect_reference.estimand_id,
                    "estimator_id": effect_reference.estimator_id,
                    "effect_scale_id": effect_reference.effect_scale_id,
                    "horizon_days": effect_reference.horizon_days,
                    "estimate": effect_reference.estimate,
                    "standard_error": effect_reference.standard_error,
                    "lower": effect_reference.lower,
                    "upper": effect_reference.upper,
                    "confidence_level": effect_reference.confidence_level,
                    "source_checksums": dict(effect_reference.source_checksums),
                }
            ),
            "analysis_method_diagnostics": analysis_method_diagnostics,
            "numeric_score_components": {
                "effect_point_score": float(effect_score),
                "effect_interval_score": float(interval_calibration_score),
                "safety_component_scores": [float(value) for value in safety_components],
                "score_before_method_and_completeness_gates": float(numeric_evaluation_score),
                "score_after_method_and_completeness_gates": float(evaluation_score),
            },
            "decision_action_diagnostics": decision_action_diagnostics,
            "phase_design_witness": (
                None
                if design_witness is None
                else {
                    "adequate": design_witness.adequate,
                    "achieved_power": design_witness.achieved_power,
                    "achieved_safety_absolute_risk_power": (design_witness.achieved_safety_absolute_risk_power),
                    "achieved_safety_excess_risk_power": (design_witness.achieved_safety_excess_risk_power),
                    "target_power": design_witness.target_power,
                    "target_safety_decision_power": design_witness.target_safety_decision_power,
                    "failures": list(design_witness.failures),
                    "evidence": design_witness.evidence,
                }
            ),
            "decision_action_score": decision_action_score,
            "required_evidence_flags": sorted(set(required_evidence_flags)),
            "lane_activation": {
                "drug_ranking": {
                    "active": bool(ranking_lane_active),
                    "reason": (
                        "candidate_comparison_available" if ranking_lane_active else "single_asset_or_no_comparator"
                    ),
                },
                "active_lanes": sorted(active_lanes),
            },
            "shallow_trajectory_diagnostics": shallow_diagnostics,
            "evaluation_target_checksums": evaluation_target_checksums,
            "lane_scores": [record.model_dump(mode="json", exclude_none=True) for record in lane_scores],
        },
    )
    if write_path is not None:
        write_json(write_path, grade_report_payload_v1(report=report, report_mode=report_mode))
    return report


def grade_bundle_v1(
    *,
    scenario_root: Path,
    submissions_root: Path,
    out_path: Path | None = None,
    report_mode: str = "audit",
) -> dict[str, object]:
    """Grade every JSON submission in a directory."""
    root = Path(submissions_root)
    reports = []
    for submission_path in sorted(root.glob("*.json")):
        report = grade_item_v1(scenario_root=scenario_root, submission_path=submission_path)
        reports.append(grade_report_payload_v1(report=report, report_mode=report_mode))
    primary_values: list[float] = []
    for payload in reports:
        primary = payload.get("primary_score")
        if isinstance(primary, int | float):
            primary_values.append(float(primary))
    summary = {
        "version": "v1",
        "n_submissions": int(len(reports)),
        "mean_primary_score": (float(sum(primary_values) / len(primary_values)) if primary_values else 0.0),
        "reports": reports,
    }
    if out_path is not None:
        write_json(out_path, summary)
    return summary
