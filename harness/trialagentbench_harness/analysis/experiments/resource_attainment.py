"""Build cumulative terminal-outcome curves from observed benchmark resources."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import TypeVar

from trialagentbench_harness.contracts.experiments.procedure_assistance import (
    ProcedureAssistanceV1,
    TrialEvalAnalysisSpecificationV1,
    TrialEvalPromptConditionV1,
    TrialEvalSubmissionInterfaceV1,
)
from trialagentbench_harness.contracts.experiments.resource_attainment import (
    BenchmarkResourceAttainmentPointV1,
    ResourceAttainmentStageV1,
    validate_resource_attainment_v1,
)
from trialagentbench_harness.contracts.experiments.trialeval_ablation import (
    TrialEvalAblationGradeRowV1,
    TrialEvalAblationObservableRowV1,
)

_Row = TypeVar("_Row")


def resource_attainment_curve_points_v1(
    *,
    rows: tuple[_Row, ...],
    resource: str,
    resource_value: Callable[[_Row], int],
    stage_values: dict[ResourceAttainmentStageV1, Callable[[_Row], float]],
    binary_stages: frozenset[ResourceAttainmentStageV1],
    point_kwargs: Mapping[str, object],
) -> list[BenchmarkResourceAttainmentPointV1]:
    denominator = len(rows)
    budgets = sorted({resource_value(row) for row in rows})
    points: list[BenchmarkResourceAttainmentPointV1] = []
    for budget in budgets:
        attained = tuple(row for row in rows if resource_value(row) <= budget)
        for stage, value_for in stage_values.items():
            values = tuple(value_for(row) for row in attained)
            points.append(
                BenchmarkResourceAttainmentPointV1.model_validate(
                    {
                        **point_kwargs,
                        "resource": resource,
                        "stage": stage,
                        "budget": budget,
                        "denominator": denominator,
                        "attained_units": len(attained),
                        "successful_units": int(sum(values)) if stage in binary_stages else None,
                        "cumulative_yield": sum(values) / denominator,
                    }
                )
            )
    return points


def trialeval_resource_attainment_v1(
    *,
    grades: tuple[TrialEvalAblationGradeRowV1, ...],
    observables: tuple[TrialEvalAblationObservableRowV1, ...],
) -> tuple[BenchmarkResourceAttainmentPointV1, ...]:
    """Build TrialEval terminal score-cascade curves over turns and tokens."""

    canonical_grades = {
        row.assignment_id: row for row in grades if row.normalization_source in {"direct_structured", "manual_masked"}
    }
    if len(canonical_grades) != len({row.assignment_id for row in grades}):
        raise ValueError("TrialEval resource attainment requires one canonical grade per assignment.")
    observable_index = {row.assignment_id: row for row in observables}
    if len(observable_index) != len(observables) or set(observable_index) != set(canonical_grades):
        raise ValueError("TrialEval resource attainment requires exact grade/observable assignment coverage.")

    grouped: dict[
        tuple[
            str,
            TrialEvalAnalysisSpecificationV1,
            ProcedureAssistanceV1,
            TrialEvalPromptConditionV1,
            TrialEvalSubmissionInterfaceV1,
        ],
        list[tuple[TrialEvalAblationGradeRowV1, TrialEvalAblationObservableRowV1]],
    ] = defaultdict(list)
    for assignment_id, grade in canonical_grades.items():
        observable = observable_index[assignment_id]
        dimensions = (
            grade.model_id,
            grade.analysis_specification,
            grade.procedure_assistance,
            grade.prompt_condition,
            grade.submission_interface,
        )
        if dimensions != (
            observable.model_id,
            observable.analysis_specification,
            observable.procedure_assistance,
            observable.prompt_condition,
            observable.submission_interface,
        ):
            raise ValueError(f"TrialEval resource-attainment treatment drift: {assignment_id!r}.")
        grouped[dimensions].append((grade, observable))

    points: list[BenchmarkResourceAttainmentPointV1] = []
    for dimensions, raw_rows in grouped.items():
        rows = tuple(raw_rows)
        model_id, specification, assistance, prompt_condition, interface = dimensions
        kwargs = {
            "benchmark": "trialeval",
            "model_id": model_id,
            "analysis_specification": specification,
            "procedure_assistance": assistance,
            "prompt_condition": prompt_condition,
            "submission_interface": interface,
        }
        stages: dict[
            ResourceAttainmentStageV1,
            Callable[[tuple[TrialEvalAblationGradeRowV1, TrialEvalAblationObservableRowV1]], float],
        ] = {
            "trialeval_answer_submitted": lambda row: float(row[1].answer_submitted),
            "trialeval_usable_primary": lambda row: float(row[0].usable_primary),
            "trialeval_route_match": lambda row: float(row[0].route_match),
            "trialeval_numeric_result_available": lambda row: float(row[0].numeric_result_available),
            "trialeval_primary_analysis_conforms": lambda row: float(row[0].primary_analysis_conforms),
        }
        points.extend(
            resource_attainment_curve_points_v1(
                rows=rows,
                resource="turns",
                resource_value=lambda row: row[1].assistant_turns,
                stage_values=stages,
                binary_stages=frozenset(
                    {
                        "trialeval_answer_submitted",
                        "trialeval_usable_primary",
                        "trialeval_route_match",
                        "trialeval_numeric_result_available",
                        "trialeval_primary_analysis_conforms",
                    }
                ),
                point_kwargs=kwargs,
            )
        )
        if any(row[1].provider_responses_with_usage != row[1].provider_response_count for row in rows):
            raise ValueError("TrialEval token attainment requires reported usage for every provider response.")
        points.extend(
            resource_attainment_curve_points_v1(
                rows=rows,
                resource="tokens",
                resource_value=lambda row: row[1].total_tokens,
                stage_values=stages,
                binary_stages=frozenset(
                    {
                        "trialeval_answer_submitted",
                        "trialeval_usable_primary",
                        "trialeval_route_match",
                        "trialeval_numeric_result_available",
                        "trialeval_primary_analysis_conforms",
                    }
                ),
                point_kwargs=kwargs,
            )
        )
    return validate_resource_attainment_v1(points)


__all__ = [
    "resource_attainment_curve_points_v1",
    "trialeval_resource_attainment_v1",
]
