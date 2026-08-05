"""Paired error accounting for narrative normalization and direct judging."""

from __future__ import annotations

from statistics import fmean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InterfaceCalibrationUnitV1(BaseModel):
    """One report paired to a masked reference, normalizer, and direct judge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.interface_calibration_unit/v1"] = (
        "trialagentbench.interface_calibration_unit/v1"
    )
    assignment_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    reference_conforms: bool
    automated_normalization_status: Literal["complete", "abstain", "failed"]
    automated_normalization_conforms: bool | None = None
    omitted_score_fields: tuple[str, ...] = Field(default_factory=tuple)
    ambiguous_score_fields: tuple[str, ...] = Field(default_factory=tuple)
    normalization_failure_reason: str | None = None
    normalization_elapsed_seconds: float = Field(..., ge=0.0)
    normalization_cost_usd: float | None = Field(default=None, ge=0.0)
    direct_judge_status: Literal["completed", "invalid_response"]
    direct_judge_conforms: bool | None = None
    direct_judge_failure_reason: str | None = None
    direct_judge_elapsed_seconds: float = Field(..., ge=0.0)
    direct_judge_cost_usd: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_dispositions(self) -> InterfaceCalibrationUnitV1:
        """Require outcomes exactly when each interface completed."""

        normalization_complete = self.automated_normalization_status == "complete"
        if normalization_complete != (self.automated_normalization_conforms is not None):
            raise ValueError("Normalizer conformance is required exactly for complete normalization.")
        if normalization_complete == (self.normalization_failure_reason is not None):
            raise ValueError("Normalizer failure reason is required exactly for abstained or failed output.")
        judge_complete = self.direct_judge_status == "completed"
        if judge_complete != (self.direct_judge_conforms is not None):
            raise ValueError("Direct-judge conformance is required exactly for a completed judgement.")
        if judge_complete == (self.direct_judge_failure_reason is not None):
            raise ValueError("Direct-judge failure reason is required exactly for an invalid response.")
        return self


class InterfaceErrorRatesV1(BaseModel):
    """Denominator-preserving binary error estimates for one interface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluable_units: int = Field(..., ge=0)
    agreement_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    false_acceptance_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    false_rejection_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    failure_rate: float = Field(..., ge=0.0, le=1.0)
    mean_latency_seconds: float = Field(..., ge=0.0)
    total_reported_cost_usd: float = Field(..., ge=0.0)


class InterfaceCalibrationReportV1(BaseModel):
    """Complete paired interface-ablation summary and retained unit records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.interface_calibration_report/v1"] = (
        "trialagentbench.interface_calibration_report/v1"
    )
    unit_count: int = Field(..., ge=1)
    normalizer: InterfaceErrorRatesV1
    direct_judge: InterfaceErrorRatesV1
    omission_rate: float = Field(..., ge=0.0, le=1.0)
    ambiguity_rate: float = Field(..., ge=0.0, le=1.0)
    units: tuple[InterfaceCalibrationUnitV1, ...] = Field(..., min_length=1)


def _error_rates(
    units: tuple[InterfaceCalibrationUnitV1, ...],
    *,
    outcome_name: Literal["automated_normalization_conforms", "direct_judge_conforms"],
    latency_name: Literal["normalization_elapsed_seconds", "direct_judge_elapsed_seconds"],
    cost_name: Literal["normalization_cost_usd", "direct_judge_cost_usd"],
) -> InterfaceErrorRatesV1:
    outcomes = [(unit.reference_conforms, getattr(unit, outcome_name)) for unit in units]
    evaluable = [(reference, observed) for reference, observed in outcomes if observed is not None]
    agreements = [reference == observed for reference, observed in evaluable]
    reference_negative = [bool(observed) for reference, observed in evaluable if not reference]
    reference_positive = [not bool(observed) for reference, observed in evaluable if reference]
    denominator = len(evaluable)
    costs = [getattr(unit, cost_name) for unit in units]
    return InterfaceErrorRatesV1(
        evaluable_units=denominator,
        agreement_rate=fmean(agreements) if agreements else None,
        false_acceptance_rate=fmean(reference_negative) if reference_negative else None,
        false_rejection_rate=fmean(reference_positive) if reference_positive else None,
        failure_rate=1.0 - (denominator / len(units)),
        mean_latency_seconds=fmean(getattr(unit, latency_name) for unit in units),
        total_reported_cost_usd=sum(float(value) for value in costs if value is not None),
    )


def analyse_interface_calibration_v1(
    units: tuple[InterfaceCalibrationUnitV1, ...],
) -> InterfaceCalibrationReportV1:
    """Estimate paired interface error rates."""

    if not units:
        raise ValueError("Interface calibration requires at least one paired unit.")
    keys = [(unit.assignment_id, unit.task_id) for unit in units]
    if len(keys) != len(set(keys)):
        raise ValueError("Interface calibration units must have unique assignment and task identities.")
    ordered = tuple(sorted(units, key=lambda unit: (unit.assignment_id, unit.task_id)))
    return InterfaceCalibrationReportV1(
        unit_count=len(ordered),
        normalizer=_error_rates(
            ordered,
            outcome_name="automated_normalization_conforms",
            latency_name="normalization_elapsed_seconds",
            cost_name="normalization_cost_usd",
        ),
        direct_judge=_error_rates(
            ordered,
            outcome_name="direct_judge_conforms",
            latency_name="direct_judge_elapsed_seconds",
            cost_name="direct_judge_cost_usd",
        ),
        omission_rate=fmean(bool(unit.omitted_score_fields) for unit in ordered),
        ambiguity_rate=fmean(bool(unit.ambiguous_score_fields) for unit in ordered),
        units=ordered,
    )


__all__ = [
    "InterfaceCalibrationReportV1",
    "InterfaceCalibrationUnitV1",
    "InterfaceErrorRatesV1",
    "analyse_interface_calibration_v1",
]
