"""Pure deterministic grader for one canonical primary submission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from trialagentbench_harness.grading.models import (
    GRADE_COMPONENT_ORDER_V1,
    GRADE_GATE_ORDER_V1,
    CanonicalDataIntegrityRecordV1,
    CanonicalSubmissionV1,
    CategoricalSubmissionV1,
    CategoricalTargetV1,
    DataIntegrityTargetV1,
    FailureCode,
    GradeComponentIdV1,
    GradeComponentRecordV1,
    GradeComponentStatusV1,
    GradeGateIdV1,
    GradeGateRecordV1,
    GradeRecordV1,
    NumericIntervalSubmissionV1,
    NumericIntervalTargetV1,
    NumericPointSubmissionV1,
    NumericPointTargetV1,
    NumericVectorSubmissionV1,
    NumericVectorTargetV1,
    RouteSignatureV1,
    StatisticalTestSubmissionV1,
    StatisticalTestTargetV1,
    ValidatedScoringKeyV1,
)


def grade(
    scoring_key: ValidatedScoringKeyV1,
    submission: CanonicalSubmissionV1,
) -> GradeRecordV1:
    """Grade one explicit primary analysis without inference or route search."""

    return _grade_from_assessment(scoring_key, _assess(scoring_key, submission))


@dataclass(frozen=True)
class _ResultAssessment:
    structure_failure: FailureCode | None = None
    comparison_failure: FailureCode | None = None
    absolute_error: float | None = None
    tolerance_ratio: float | None = None


@dataclass(frozen=True)
class _GradeAssessment:
    components: tuple[GradeComponentRecordV1, ...]
    first_failure_gate: GradeGateIdV1 | None
    first_failure_code: FailureCode | None
    matched_route_id: str | None = None
    absolute_error: float | None = None
    tolerance_ratio: float | None = None


def _assess(scoring_key: ValidatedScoringKeyV1, submission: CanonicalSubmissionV1) -> _GradeAssessment:
    """Evaluate every available component while preserving one causal primary cascade."""

    component_rows: dict[GradeComponentIdV1, GradeComponentRecordV1] = {}
    item_matches = submission.item_id == scoring_key.item_id
    component_rows["submission"] = _component(
        "submission",
        None if item_matches else "item_mismatch",
    )
    route = None
    result_assessment: _ResultAssessment | None = None
    if not item_matches:
        for component_id in ("question", "method", "evidence", "integrity", "result_structure", "route_comparison"):
            component_rows[component_id] = _component(component_id, status="not_evaluable")
    else:
        question_routes = tuple(
            candidate
            for candidate in scoring_key.credit_eligible_routes
            if _question_binding(candidate.signature) == _question_binding(submission.primary)
        )
        if not question_routes:
            component_rows["question"] = _component("question", "unrecognized_primary_question")
            component_rows["method"] = _component("method", status="not_evaluable")
        else:
            component_rows["question"] = _component("question")
            route = next(
                (candidate for candidate in question_routes if candidate.signature == submission.primary),
                None,
            )
            component_rows["method"] = _component(
                "method",
                None if route is not None else "unrecognized_primary_route",
            )
        if route is None:
            component_rows["evidence"] = _component("evidence", status="not_evaluable")
            component_rows["result_structure"] = _component("result_structure", status="not_evaluable")
            component_rows["route_comparison"] = _component("route_comparison", status="not_evaluable")
        else:
            missing_diagnostics = set(route.required_diagnostics) - set(submission.diagnostic_ids)
            component_rows["evidence"] = _component(
                "evidence",
                "missing_required_diagnostic" if missing_diagnostics else None,
            )
            result_assessment = _assess_result(route.target, submission.result)
            component_rows["result_structure"] = _component(
                "result_structure",
                result_assessment.structure_failure,
            )
            if result_assessment.structure_failure is not None:
                component_rows["route_comparison"] = _component("route_comparison", status="not_evaluable")
            else:
                component_rows["route_comparison"] = _component(
                    "route_comparison",
                    result_assessment.comparison_failure,
                )
        integrity_failure = _integrity_failure_code(
            target=scoring_key.data_integrity_target,
            observed=submission.data_integrity_record,
        )
        if scoring_key.data_integrity_target is None and submission.data_integrity_record is None:
            component_rows["integrity"] = _component("integrity", status="not_applicable")
        else:
            component_rows["integrity"] = _component("integrity", integrity_failure)

    components = tuple(component_rows[component_id] for component_id in GRADE_COMPONENT_ORDER_V1)
    gate_components: tuple[tuple[GradeGateIdV1, GradeComponentIdV1], ...] = (
        ("submission", "submission"),
        ("question", "question"),
        ("route", "method"),
        ("evidence", "evidence"),
        ("integrity", "integrity"),
        ("result", "result_structure"),
        ("conformance", "route_comparison"),
    )
    first_failure_gate = None
    first_failure_code = None
    for gate_id, component_id in gate_components:
        component = component_rows[component_id]
        if component.status == "failed":
            first_failure_gate = gate_id
            first_failure_code = component.failure_code
            break
    comparison_reached = first_failure_gate in {None, "conformance"}
    return _GradeAssessment(
        components=components,
        first_failure_gate=first_failure_gate,
        first_failure_code=first_failure_code,
        matched_route_id=None if route is None else route.route_id,
        absolute_error=(
            None if result_assessment is None or not comparison_reached else result_assessment.absolute_error
        ),
        tolerance_ratio=(
            None if result_assessment is None or not comparison_reached else result_assessment.tolerance_ratio
        ),
    )


def _assess_result(target: object, result: object) -> _ResultAssessment:
    """Evaluate result structure and route-native numerical conformance once."""

    if isinstance(target, NumericPointTargetV1) and isinstance(result, NumericPointSubmissionV1):
        if result.result_unit != target.result_unit:
            return _ResultAssessment(structure_failure="result_unit_mismatch")
        if target.require_confidence_interval and (
            result.confidence_interval_lower is None or result.confidence_interval_upper is None
        ):
            return _ResultAssessment(structure_failure="missing_confidence_interval")
        error = abs(result.value - target.value)
        tolerance = target.acceptance_envelope.absolute_tolerance
        failure: FailureCode | None = "numeric_result_outside_tolerance" if error > tolerance else None
        if target.require_confidence_interval:
            if target.confidence_interval_lower is None or target.confidence_interval_upper is None:
                raise ValueError("validated scoring target is missing required confidence interval endpoints")
            if result.confidence_interval_lower is None or result.confidence_interval_upper is None:
                raise ValueError("validated submission is missing required confidence interval endpoints")
            interval_error = max(
                abs(result.confidence_interval_lower - target.confidence_interval_lower),
                abs(result.confidence_interval_upper - target.confidence_interval_upper),
            )
            if interval_error > tolerance and failure is None:
                failure = "confidence_interval_outside_tolerance"
            error = max(error, interval_error)
        return _ResultAssessment(
            comparison_failure=failure,
            absolute_error=error,
            tolerance_ratio=error / tolerance,
        )
    if isinstance(target, NumericIntervalTargetV1) and isinstance(result, NumericIntervalSubmissionV1):
        if result.result_unit != target.result_unit:
            return _ResultAssessment(structure_failure="result_unit_mismatch")
        error = max(abs(result.lower - target.lower), abs(result.upper - target.upper))
        tolerance = target.acceptance_envelope.absolute_tolerance
        return _ResultAssessment(
            comparison_failure="interval_result_outside_tolerance" if error > tolerance else None,
            absolute_error=error,
            tolerance_ratio=error / tolerance,
        )
    if isinstance(target, NumericVectorTargetV1) and isinstance(result, NumericVectorSubmissionV1):
        if result.result_unit != target.result_unit:
            return _ResultAssessment(structure_failure="result_unit_mismatch")
        target_values = {component.name: component.value for component in target.components}
        result_values = {component.name: component.value for component in result.components}
        if set(result_values) != set(target_values):
            return _ResultAssessment(structure_failure="vector_components_mismatch")
        error = max(abs(result_values[name] - value) for name, value in target_values.items())
        tolerance = target.acceptance_envelope.absolute_tolerance
        return _ResultAssessment(
            comparison_failure="vector_result_outside_tolerance" if error > tolerance else None,
            absolute_error=error,
            tolerance_ratio=error / tolerance,
        )
    if isinstance(target, StatisticalTestTargetV1) and isinstance(result, StatisticalTestSubmissionV1):
        error = abs(result.p_value - target.p_value)
        tolerance = target.acceptance_envelope.absolute_tolerance
        if result.reject_null != target.reject_null:
            failure = "test_decision_mismatch"
        elif error > tolerance:
            failure = "test_p_value_outside_tolerance"
        else:
            failure = None
        return _ResultAssessment(
            comparison_failure=failure,
            absolute_error=error,
            tolerance_ratio=error / tolerance,
        )
    if isinstance(target, CategoricalTargetV1) and isinstance(result, CategoricalSubmissionV1):
        return _ResultAssessment(
            comparison_failure=(
                None if result.code in target.credit_eligible_codes else "categorical_result_not_credit_eligible"
            )
        )
    return _ResultAssessment(structure_failure="result_kind_mismatch")


def _integrity_failure_code(
    *,
    target: DataIntegrityTargetV1 | None,
    observed: CanonicalDataIntegrityRecordV1 | None,
) -> FailureCode | None:
    if target is None:
        return None if observed is None else "unexpected_data_integrity_record"
    if observed is None:
        return "missing_data_integrity_record"
    if observed.condition_id != target.condition_id:
        return "data_integrity_condition_mismatch"
    if observed.affected_domain != target.affected_domain:
        return "data_integrity_domain_mismatch"
    if observed.compound_key_fields != target.compound_key_fields:
        return "data_integrity_compound_key_mismatch"
    if (
        observed.observed_duplicate_group_count != target.observed_duplicate_group_count
        or observed.observed_extra_row_count != target.observed_extra_row_count
    ):
        return "data_integrity_counts_mismatch"
    if observed.repair_action != target.repair_action or observed.repair_status != target.repair_status:
        return "data_integrity_repair_mismatch"
    if (
        observed.post_repair_data_checksum != target.post_repair_data_checksum
        or observed.analysis_input_data_checksum != target.analysis_input_data_checksum
    ):
        return "data_integrity_checksum_mismatch"
    return None


def grade_missing_submission(scoring_key: ValidatedScoringKeyV1) -> GradeRecordV1:
    """Record denominator-preserving failure when no primary was submitted."""

    return _grade_submission_failure(scoring_key, "missing_primary_submission")


def grade_missing_required_deliverable(scoring_key: ValidatedScoringKeyV1) -> GradeRecordV1:
    """Record failure when a typed submission omits an item-required deliverable."""

    return _grade_submission_failure(scoring_key, "missing_required_deliverable")


def _grade_submission_failure(
    scoring_key: ValidatedScoringKeyV1,
    failure_code: Literal["missing_primary_submission", "missing_required_deliverable"],
) -> GradeRecordV1:
    """Build one denominator-preserving submission-gate failure."""

    components = tuple(
        _component(
            component_id,
            failure_code if component_id == "submission" else None,
            status=None if component_id == "submission" else "not_evaluable",
        )
        for component_id in GRADE_COMPONENT_ORDER_V1
    )
    return _grade_from_assessment(
        scoring_key,
        _GradeAssessment(
            components=components,
            first_failure_gate="submission",
            first_failure_code=failure_code,
        ),
    )


def _grade_from_assessment(
    scoring_key: ValidatedScoringKeyV1,
    assessment: _GradeAssessment,
) -> GradeRecordV1:
    """Project component assessment into the noncompensatory primary record."""

    gates = _gate_records(
        failure_gate=assessment.first_failure_gate,
        failure_code=assessment.first_failure_code,
        integrity_applicable=scoring_key.data_integrity_target is not None,
    )
    statuses = {record.gate_id: record.status for record in gates}
    passed = assessment.first_failure_gate is None
    return GradeRecordV1.model_validate(
        {
            "release_id": scoring_key.release_id,
            "item_id": scoring_key.item_id,
            "usable_primary": statuses["submission"] == "passed",
            "route_match": statuses["question"] == statuses["route"] == "passed",
            "obligations_met": (
                statuses["evidence"] == "passed" and statuses["integrity"] in {"passed", "not_applicable"}
            ),
            "result_match": statuses["result"] == statuses["conformance"] == "passed",
            "passed": passed,
            "gates": gates,
            "components": assessment.components,
            "first_failure_gate": assessment.first_failure_gate,
            "matched_route_id": assessment.matched_route_id,
            "failure_codes": (() if assessment.first_failure_code is None else (assessment.first_failure_code,)),
            "absolute_error": assessment.absolute_error,
            "tolerance_ratio": assessment.tolerance_ratio,
        }
    )


def _component(
    component_id: GradeComponentIdV1,
    failure_code: FailureCode | None = None,
    *,
    status: GradeComponentStatusV1 | None = None,
) -> GradeComponentRecordV1:
    """Build one complete descriptive component outcome."""

    return GradeComponentRecordV1(
        component_id=component_id,
        status=status or ("failed" if failure_code is not None else "passed"),
        failure_code=failure_code,
    )


def _question_binding(signature: RouteSignatureV1) -> tuple[object, ...]:
    """Return fields defining the fixed statistical question."""

    return (
        signature.analysis_population_id,
        signature.estimand_id,
        signature.intercurrent_event_strategy_ids,
        signature.assessment_horizon_days,
        signature.treatment_id,
        signature.comparator_id,
        signature.endpoint_id,
        signature.effect_scale,
    )


def _gate_records(
    *,
    failure_gate: GradeGateIdV1 | None,
    failure_code: FailureCode | None,
    integrity_applicable: bool,
) -> tuple[GradeGateRecordV1, ...]:
    """Build the complete ordered gate cascade."""

    failure_index = None if failure_gate is None else GRADE_GATE_ORDER_V1.index(failure_gate)
    records: list[GradeGateRecordV1] = []
    for index, gate_id in enumerate(GRADE_GATE_ORDER_V1):
        if failure_index is not None and index > failure_index:
            status = "not_reached"
        elif gate_id == failure_gate:
            status = "failed"
        elif gate_id == "integrity" and not integrity_applicable:
            status = "not_applicable"
        elif gate_id == "decision":
            status = "not_applicable"
        else:
            status = "passed"
        records.append(
            GradeGateRecordV1(
                gate_id=gate_id,
                status=status,
                failure_code=failure_code if gate_id == failure_gate else None,
            )
        )
    return tuple(records)


__all__ = ["grade", "grade_missing_submission"]
