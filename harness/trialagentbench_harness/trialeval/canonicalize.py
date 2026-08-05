"""Compile a typed TrialEval submission into the narrow grader contract."""

from __future__ import annotations

from trialagentbench_harness.contracts.submission.models import (
    IdentifiedIntervalV1,
    QualifiedNonIdentificationV1,
    ScalarEstimateV1,
    TestResultV1,
    TrialEvalSubmissionV1,
    VectorEstimateV1,
)
from trialagentbench_harness.grading.models import (
    CanonicalDataIntegrityRecordV1,
    CanonicalSubmissionV1,
    CategoricalSubmissionV1,
    NamedNumericValueV1,
    NumericIntervalSubmissionV1,
    NumericPointSubmissionV1,
    NumericVectorSubmissionV1,
    PrimaryResultV1,
    RouteSignatureV1,
    StatisticalTestSubmissionV1,
)


def canonicalize_trialeval_submission_v1(
    submission: TrialEvalSubmissionV1,
    *,
    validated_diagnostic_ids: frozenset[str],
) -> CanonicalSubmissionV1:
    """Return the exact score-bearing primary declaration.

    The conversion depends only on the validated participant submission. It
    never inspects evaluator routes or scoring targets.
    """

    primary = submission.primary_analysis
    estimand = primary.estimand
    estimator = primary.estimator
    declared_diagnostic_ids = {
        str(record.diagnostic_id)
        for record in submission.evidence
        if record.evidence_id in primary.evidence_ids and record.diagnostic_id is not None
    }
    unknown = validated_diagnostic_ids - declared_diagnostic_ids
    if unknown:
        raise ValueError(f"validated diagnostics are absent from the linked primary evidence: {sorted(unknown)!r}")
    diagnostic_ids = tuple(sorted(validated_diagnostic_ids))
    signature = RouteSignatureV1(
        analysis_population_id=str(estimand.population_id),
        estimand_id=str(estimand.estimand_id),
        intercurrent_event_strategy_ids=tuple(str(value) for value in estimand.intercurrent_event_strategy_ids),
        assessment_horizon_days=(None if estimand.horizon is None else float(estimand.horizon.value)),
        treatment_id=str(estimand.treatment_id),
        comparator_id=str(estimand.comparator_id),
        endpoint_id=str(estimand.endpoint_id),
        effect_scale=str(primary.result.effect_scale),
        analysis_method_id=str(estimator.analysis_method_id),
    )
    return CanonicalSubmissionV1(
        item_id=submission.task_id,
        primary=signature,
        diagnostic_ids=diagnostic_ids,
        data_integrity_record=(
            None
            if submission.data_integrity_record is None
            else CanonicalDataIntegrityRecordV1.model_validate(
                submission.data_integrity_record.model_dump(mode="json")
            )
        ),
        result=_canonical_result(submission),
    )


def _canonical_result(submission: TrialEvalSubmissionV1) -> PrimaryResultV1:
    result = submission.primary_analysis.result
    if isinstance(result, ScalarEstimateV1):
        return NumericPointSubmissionV1(
            kind="numeric_point",
            value=float(result.value),
            result_unit=str(result.unit),
            confidence_interval_lower=float(result.interval.lower),
            confidence_interval_upper=float(result.interval.upper),
        )
    if isinstance(result, IdentifiedIntervalV1):
        return NumericIntervalSubmissionV1(
            kind="numeric_interval",
            lower=float(result.lower),
            upper=float(result.upper),
            result_unit=str(result.unit),
        )
    if isinstance(result, VectorEstimateV1):
        return NumericVectorSubmissionV1(
            kind="numeric_vector",
            components=tuple(
                NamedNumericValueV1(name=str(point.component_id), value=float(point.value)) for point in result.points
            ),
            result_unit=str(result.unit),
        )
    if isinstance(result, TestResultV1):
        return StatisticalTestSubmissionV1(
            kind="statistical_test",
            p_value=float(result.p_value),
            reject_null=bool(result.reject_null),
        )
    if isinstance(result, QualifiedNonIdentificationV1):
        return CategoricalSubmissionV1(
            kind="categorical",
            code=str(result.conclusion_code),
        )
    raise AssertionError(f"unsupported validated primary result: {type(result)!r}")


__all__ = ["canonicalize_trialeval_submission_v1"]
