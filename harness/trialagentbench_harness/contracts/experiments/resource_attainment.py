"""Contracts for terminal benchmark attainment over observed resource use."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.contracts.experiments.procedure_assistance import (
    ProcedureAssistanceV1,
    TrialEvalAnalysisSpecificationV1,
    TrialEvalPromptConditionV1,
    TrialEvalSubmissionInterfaceV1,
)

ResourceAttainmentStageV1: TypeAlias = Literal[  # noqa: UP040
    "trialeval_answer_submitted",
    "trialeval_usable_primary",
    "trialeval_route_match",
    "trialeval_numeric_result_available",
    "trialeval_primary_analysis_conforms",
    "trialdev_programme_completion",
    "trialdev_programme_design_validity",
    "trialdev_trajectory_decision_score",
    "trialdev_trajectory_primary_score",
]

_TRIALEVAL_STAGES = frozenset(
    {
        "trialeval_answer_submitted",
        "trialeval_usable_primary",
        "trialeval_route_match",
        "trialeval_numeric_result_available",
        "trialeval_primary_analysis_conforms",
    }
)
_TRIALDEV_STAGES = frozenset(
    {
        "trialdev_programme_completion",
        "trialdev_programme_design_validity",
        "trialdev_trajectory_decision_score",
        "trialdev_trajectory_primary_score",
    }
)
_BINARY_STAGES = frozenset(
    {
        "trialeval_answer_submitted",
        "trialeval_usable_primary",
        "trialeval_route_match",
        "trialeval_numeric_result_available",
        "trialeval_primary_analysis_conforms",
        "trialdev_programme_completion",
        "trialdev_programme_design_validity",
    }
)


class BenchmarkResourceAttainmentPointV1(BaseModel):
    """One cumulative terminal-outcome point at an observed resource budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: Literal["trialeval", "trialdev"]
    model_id: str = Field(..., min_length=1)
    procedure_assistance: ProcedureAssistanceV1
    analysis_specification: TrialEvalAnalysisSpecificationV1 | None = None
    prompt_condition: TrialEvalPromptConditionV1 | None = None
    submission_interface: TrialEvalSubmissionInterfaceV1 | None = None
    objective_id: str | None = Field(default=None, min_length=1)
    resource: Literal["turns", "tokens"]
    stage: ResourceAttainmentStageV1
    budget: int = Field(..., ge=0)
    denominator: int = Field(..., gt=0)
    attained_units: int = Field(..., ge=0)
    successful_units: int | None = Field(default=None, ge=0)
    cumulative_yield: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_suite_and_counts(self) -> BenchmarkResourceAttainmentPointV1:
        if self.attained_units > self.denominator:
            raise ValueError("Resource-attainment attained_units cannot exceed denominator.")
        if self.successful_units is not None and self.successful_units > self.attained_units:
            raise ValueError("Resource-attainment successes cannot exceed attained units.")
        expected_stages = _TRIALEVAL_STAGES if self.benchmark == "trialeval" else _TRIALDEV_STAGES
        if self.stage not in expected_stages:
            raise ValueError(f"Resource-attainment stage {self.stage!r} does not belong to {self.benchmark}.")
        if (self.successful_units is not None) != (self.stage in _BINARY_STAGES):
            raise ValueError("Resource-attainment successful_units is required exactly for binary stages.")
        if self.successful_units is not None and self.cumulative_yield != self.successful_units / self.denominator:
            raise ValueError("Binary resource-attainment yield must equal successful_units / denominator.")
        trialeval_dimensions = (
            self.analysis_specification,
            self.prompt_condition,
            self.submission_interface,
        )
        if self.benchmark == "trialeval":
            if any(value is None for value in trialeval_dimensions) or self.objective_id is not None:
                raise ValueError("TrialEval resource attainment requires TrialEval dimensions and no objective_id.")
        elif any(value is not None for value in trialeval_dimensions):
            raise ValueError("TrialDev resource attainment cannot contain TrialEval dimensions.")
        return self


def validate_resource_attainment_v1(
    points: Iterable[BenchmarkResourceAttainmentPointV1],
) -> tuple[BenchmarkResourceAttainmentPointV1, ...]:
    """Validate complete cumulative curves and return canonical row order."""

    ordered = tuple(
        sorted(
            points,
            key=lambda row: (
                row.benchmark,
                row.model_id,
                row.analysis_specification or "",
                row.procedure_assistance,
                row.prompt_condition or "",
                row.submission_interface or "",
                row.objective_id or "",
                row.resource,
                row.stage,
                row.budget,
            ),
        )
    )
    if not ordered:
        raise ValueError("Resource-attainment evidence cannot be empty.")
    groups: dict[tuple[object, ...], list[BenchmarkResourceAttainmentPointV1]] = defaultdict(list)
    for row in ordered:
        groups[
            (
                row.benchmark,
                row.model_id,
                row.analysis_specification,
                row.procedure_assistance,
                row.prompt_condition,
                row.submission_interface,
                row.objective_id,
                row.resource,
                row.stage,
            )
        ].append(row)
    for key, group in groups.items():
        if len({row.budget for row in group}) != len(group):
            raise ValueError(f"Resource-attainment curve has duplicate budgets for stratum {key!r}.")
        if len({row.denominator for row in group}) != 1:
            raise ValueError(f"Resource-attainment denominator changes within stratum {key!r}.")
        if any(
            current.attained_units < prior.attained_units or current.cumulative_yield < prior.cumulative_yield
            for prior, current in zip(group, group[1:], strict=False)
        ):
            raise ValueError(f"Resource-attainment curve is not cumulative for stratum {key!r}.")
        if group[-1].attained_units != group[-1].denominator:
            raise ValueError(f"Resource-attainment curve does not reach its denominator for stratum {key!r}.")
    return ordered


__all__ = [
    "BenchmarkResourceAttainmentPointV1",
    "ResourceAttainmentStageV1",
    "validate_resource_attainment_v1",
]
