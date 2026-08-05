"""Independently verify A4 identification and sequential-analysis reliability."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import stats

from trialagentbench_validation.statistics.operating_characteristics import (
    proportion_interval,
)

_A4_CELLS = frozenset({"TE-S04-A4", "TE-S06-A4", "TE-S09-A4"})
_SEQUENTIAL_ESTIMATOR = "observed:group_sequential_adjusted"
_IDENTIFIED_ANALYSES = {
    "TE-S04-A4": {
        "required_primary": "observed:tau_bounds_bounded_deviation",
        "credit_eligible_primary_alternative": "observed:tau_bounds_worst_case",
    },
    "TE-S06-A4": {
        "required_primary": "observed:validated_endpoint_bounded_deviation",
        "credit_eligible_primary_alternative": "observed:validated_endpoint_worst_case",
    },
}
_IdentifiedAnalysisRole = Literal[
    "required_primary", "credit_eligible_primary_alternative"
]
_PublicAnalysisRole = Literal[
    "prespecified_bounded_departure", "unrestricted_worst_case", "repeated_monitoring"
]


class _SourceContract(BaseModel):
    """Subset of a qualification artifact required by this verifier."""

    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False)


class _SetWorldRecord(_SourceContract):
    regime_cell_id: str
    method_signature_id: str = Field(min_length=1)
    analysis_role: _IdentifiedAnalysisRole
    estimator_id: str = Field(min_length=1)
    sensitivity_parameter: float | None = Field(default=None, ge=0, le=1)
    reference_value: float
    status: Literal["success", "fit_failure"]
    set_low: float | None = None
    set_high: float | None = None

    @model_validator(mode="after")
    def _valid_result(self) -> _SetWorldRecord:
        if self.status == "success":
            if self.set_low is None or self.set_high is None:
                raise ValueError(
                    "Successful identified-set records require both limits."
                )
            if self.set_low > self.set_high:
                raise ValueError("Identified-set limits are reversed.")
        elif self.set_low is not None or self.set_high is not None:
            raise ValueError("Failed identified-set records cannot contain limits.")
        return self


class _PointWorldRecord(_SourceContract):
    regime_cell_id: str
    method_signature_id: str = Field(min_length=1)
    estimator_id: str = Field(min_length=1)
    component_id: str | None = None
    reference_value: float
    status: Literal["success", "fit_failure"]
    estimate: float | None = None
    interval_low: float | None = None
    interval_high: float | None = None

    @model_validator(mode="after")
    def _valid_result(self) -> _PointWorldRecord:
        values = (self.estimate, self.interval_low, self.interval_high)
        if self.status == "success":
            if any(value is None for value in values):
                raise ValueError(
                    "Successful point records require an estimate and interval."
                )
            assert self.interval_low is not None
            assert self.interval_high is not None
            if self.interval_low > self.interval_high:
                raise ValueError("Point-estimate interval limits are reversed.")
        elif any(value is not None for value in values):
            raise ValueError("Failed point records cannot contain numerical results.")
        return self


class _MonitoringRecord(_SourceContract):
    regime_cell_id: str
    analysis_look_index: int = Field(ge=0)
    stopped_early: bool
    stopping_regime: Literal["efficacy_stop", "final_analysis"]

    @model_validator(mode="after")
    def _consistent_path(self) -> _MonitoringRecord:
        if self.stopped_early != (self.stopping_regime == "efficacy_stop"):
            raise ValueError("Sequential stopping flag and regime disagree.")
        return self


class _WorldRecord(_SourceContract):
    world_id: str = Field(min_length=1)
    regime_cell_id: str
    point_records: tuple[_PointWorldRecord, ...] = ()
    identified_set_records: tuple[_SetWorldRecord, ...] = ()
    group_sequential_monitoring: _MonitoringRecord | None = None


class _ReportedSet(_SourceContract):
    regime_cell_id: str
    method_signature_id: str
    analysis_role: _IdentifiedAnalysisRole
    estimator_id: str
    sensitivity_parameter: float | None = Field(default=None, ge=0, le=1)
    n_worlds: int = Field(gt=0)
    n_success: int = Field(ge=0)
    failure_rate: float = Field(ge=0, le=1)
    conditional_set_coverage: float = Field(ge=0, le=1)
    unconditional_set_coverage: float = Field(ge=0, le=1)
    mean_set_width: float = Field(ge=0)


class _ReportedPoint(_SourceContract):
    regime_cell_id: str
    method_signature_id: str
    estimator_id: str
    component_id: str | None = None
    n_worlds: int = Field(gt=0)
    n_success: int = Field(ge=0)
    failure_rate: float = Field(ge=0, le=1)
    bias: float
    rmse: float = Field(ge=0)
    conditional_coverage: float = Field(ge=0, le=1)
    unconditional_coverage: float = Field(ge=0, le=1)
    mean_interval_width: float = Field(ge=0)


class _ReportedMonitoring(_SourceContract):
    regime_cell_id: str
    n_worlds: int = Field(gt=0)
    efficacy_stop_worlds: int = Field(ge=0)
    final_analysis_worlds: int = Field(ge=0)


class _OperatingCharacteristics(_SourceContract):
    identified_sets: tuple[_ReportedSet, ...] = ()
    point_methods: tuple[_ReportedPoint, ...] = ()
    group_sequential_monitoring: tuple[_ReportedMonitoring, ...] = ()


class IdentificationReliabilityResultV1(BaseModel):
    """Independent operating characteristics for one A4 conclusion."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    series_id: str = Field(pattern=r"^TE-S0[469]$")
    regime_cell_id: str = Field(pattern=r"^TE-S0[469]-A4$")
    conclusion_type: Literal["identified_range", "repeated_interval"]
    analysis_role: _PublicAnalysisRole
    estimator_id: str = Field(min_length=1)
    effect_scale: Literal["risk_difference_tau"]
    sensitivity_parameter: float | None = Field(default=None, ge=0, le=1)
    sensitivity_parameter_unit: Literal["risk_probability_difference"] | None = None
    independent_trials: int = Field(gt=0)
    successful_analyses: int = Field(ge=0)
    fit_failures: int = Field(ge=0)
    coverage: float = Field(ge=0, le=1)
    coverage_low: float = Field(ge=0, le=1)
    coverage_high: float = Field(ge=0, le=1)
    conditional_coverage: float = Field(ge=0, le=1)
    mean_width: float = Field(ge=0)
    mean_width_low: float = Field(ge=0)
    mean_width_high: float = Field(ge=0)
    fit_failure_rate: float = Field(ge=0, le=1)
    fit_failure_rate_low: float = Field(ge=0, le=1)
    fit_failure_rate_high: float = Field(ge=0, le=1)
    bias: float | None = None
    bias_low: float | None = None
    bias_high: float | None = None
    rmse: float | None = Field(default=None, ge=0)
    rmse_low: float | None = Field(default=None, ge=0)
    rmse_high: float | None = Field(default=None, ge=0)
    early_stop_rate: float | None = Field(default=None, ge=0, le=1)
    early_stop_rate_low: float | None = Field(default=None, ge=0, le=1)
    early_stop_rate_high: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _valid_result(self) -> IdentificationReliabilityResultV1:
        if self.successful_analyses + self.fit_failures != self.independent_trials:
            raise ValueError(
                "Successful and failed analyses must equal the trial denominator."
            )
        if not self.coverage_low <= self.coverage <= self.coverage_high:
            raise ValueError("Coverage lies outside its interval.")
        if not self.mean_width_low <= self.mean_width <= self.mean_width_high:
            raise ValueError("Mean width lies outside its interval.")
        if (
            not self.fit_failure_rate_low
            <= self.fit_failure_rate
            <= self.fit_failure_rate_high
        ):
            raise ValueError("Failure rate lies outside its interval.")
        sequential = (
            self.bias,
            self.bias_low,
            self.bias_high,
            self.rmse,
            self.rmse_low,
            self.rmse_high,
            self.early_stop_rate,
            self.early_stop_rate_low,
            self.early_stop_rate_high,
        )
        if self.conclusion_type == "identified_range":
            if any(value is not None for value in sequential):
                raise ValueError(
                    "Identified-range results cannot contain sequential-analysis metrics."
                )
            if self.analysis_role == "repeated_monitoring":
                raise ValueError(
                    "Identified-range results require an identified-set analysis role."
                )
        else:
            if self.analysis_role != "repeated_monitoring":
                raise ValueError(
                    "Repeated intervals require the repeated-monitoring analysis role."
                )
            if any(value is None for value in sequential):
                raise ValueError(
                    "Repeated-interval results require bias, RMSE, and stopping metrics."
                )
            assert self.bias is not None
            assert self.bias_low is not None
            assert self.bias_high is not None
            assert self.rmse is not None
            assert self.rmse_low is not None
            assert self.rmse_high is not None
            assert self.early_stop_rate is not None
            assert self.early_stop_rate_low is not None
            assert self.early_stop_rate_high is not None
            if not self.bias_low <= self.bias <= self.bias_high:
                raise ValueError("Bias lies outside its interval.")
            if not self.rmse_low <= self.rmse <= self.rmse_high:
                raise ValueError("RMSE lies outside its interval.")
            if (
                not self.early_stop_rate_low
                <= self.early_stop_rate
                <= self.early_stop_rate_high
            ):
                raise ValueError("Early-stop rate lies outside its interval.")
        if (self.sensitivity_parameter is None) != (
            self.sensitivity_parameter_unit is None
        ):
            raise ValueError(
                "Sensitivity-parameter values and units must be present together."
            )
        if (
            self.analysis_role == "prespecified_bounded_departure"
            and self.sensitivity_parameter is None
        ):
            raise ValueError(
                "The prespecified bounded-departure analysis requires its risk-departure limit."
            )
        if (
            self.analysis_role != "prespecified_bounded_departure"
            and self.sensitivity_parameter is not None
        ):
            raise ValueError(
                "Only the bounded-departure analysis has a sensitivity parameter."
            )
        return self


def verify_identification_reliability(
    *,
    world_records_path: Path,
    operating_characteristics_path: Path,
    bootstrap_replicates: int = 10_000,
    seed: int = 20_260_803,
) -> tuple[IdentificationReliabilityResultV1, ...]:
    """Recompute A4 identification and sequential reliability from trial records."""

    if bootstrap_replicates < 1_000:
        raise ValueError("At least 1,000 bootstrap replicates are required.")
    worlds = _read_worlds(world_records_path)
    operating = _OperatingCharacteristics.model_validate_json(
        operating_characteristics_path.read_text(encoding="utf-8")
    )
    identified_ranges = [
        _verify_identified_range(regime_cell_id=cell, worlds=worlds, reported=reported)
        for cell in ("TE-S04-A4", "TE-S06-A4")
        for reported in _reported_sets(operating, cell)
    ]
    _verify_bounded_ranges_are_informative(
        worlds=worlds,
        results=identified_ranges,
    )
    results = [
        *identified_ranges,
        _verify_repeated_interval(
            worlds=worlds,
            reported=_one_reported_point(operating, "TE-S09-A4"),
            monitoring=_one_reported_monitoring(operating, "TE-S09-A4"),
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        ),
    ]
    return tuple(results)


def write_identification_reliability_csv(
    *,
    path: Path,
    results: tuple[IdentificationReliabilityResultV1, ...],
) -> None:
    """Write verified A4 operating characteristics as a tidy CSV table."""

    if {row.regime_cell_id for row in results} != _A4_CELLS:
        raise ValueError(
            "Identification reliability output must contain all three A4 cells."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(IdentificationReliabilityResultV1.model_fields)
        )
        writer.writeheader()
        writer.writerows(row.model_dump(mode="json") for row in results)


def _read_worlds(path: Path) -> tuple[_WorldRecord, ...]:
    worlds: list[_WorldRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"Blank qualification world at line {line_number}.")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid world record on line {line_number}."
                ) from error
            world = _WorldRecord.model_validate(payload)
            if world.regime_cell_id in _A4_CELLS:
                worlds.append(world)
    if {world.regime_cell_id for world in worlds} != _A4_CELLS:
        raise ValueError("World records must contain all three A4 cells.")
    identities = [(world.regime_cell_id, world.world_id) for world in worlds]
    if len(identities) != len(set(identities)):
        raise ValueError("A4 world records contain duplicate trial identities.")
    return tuple(worlds)


def _reported_sets(
    operating: _OperatingCharacteristics, cell: str
) -> tuple[_ReportedSet, ...]:
    rows = [row for row in operating.identified_sets if row.regime_cell_id == cell]
    expected = _IDENTIFIED_ANALYSES[cell]
    observed = {(row.analysis_role, row.estimator_id) for row in rows}
    expected_identities = set(expected.items())
    if observed != expected_identities or len(rows) != len(expected_identities):
        raise ValueError(
            f"A4 cell {cell!r} must report its prespecified and unrestricted identified ranges exactly once."
        )
    by_role = {row.analysis_role: row for row in rows}
    primary = by_role["required_primary"]
    alternative = by_role["credit_eligible_primary_alternative"]
    if primary.sensitivity_parameter is None:
        raise ValueError(
            f"A4 cell {cell!r} must state its bounded-departure sensitivity parameter."
        )
    if alternative.sensitivity_parameter is not None:
        raise ValueError(
            f"A4 cell {cell!r} unrestricted range cannot have a bounded-departure parameter."
        )
    return primary, alternative


def _one_reported_point(
    operating: _OperatingCharacteristics, cell: str
) -> _ReportedPoint:
    rows = [
        row
        for row in operating.point_methods
        if row.regime_cell_id == cell
        and row.estimator_id == _SEQUENTIAL_ESTIMATOR
        and row.component_id is None
    ]
    if len(rows) != 1:
        raise ValueError(
            f"A4 cell {cell!r} must have exactly one reported primary sequential result."
        )
    return rows[0]


def _one_reported_monitoring(
    operating: _OperatingCharacteristics,
    cell: str,
) -> _ReportedMonitoring:
    rows = [
        row
        for row in operating.group_sequential_monitoring
        if row.regime_cell_id == cell
    ]
    if len(rows) != 1:
        raise ValueError(f"A4 cell {cell!r} must have exactly one monitoring summary.")
    return rows[0]


def _verify_identified_range(
    *,
    regime_cell_id: str,
    worlds: tuple[_WorldRecord, ...],
    reported: _ReportedSet,
) -> IdentificationReliabilityResultV1:
    selected = [world for world in worlds if world.regime_cell_id == regime_cell_id]
    records: list[_SetWorldRecord] = []
    for world in selected:
        matching = [
            row
            for row in world.identified_set_records
            if row.method_signature_id == reported.method_signature_id
            and row.analysis_role == reported.analysis_role
            and row.estimator_id == reported.estimator_id
            and row.sensitivity_parameter == reported.sensitivity_parameter
        ]
        if len(matching) != 1:
            raise ValueError(
                f"Each {regime_cell_id} trial must contain one {reported.analysis_role} identified-set result."
            )
        records.append(matching[0])
    successful = [row for row in records if row.status == "success"]
    if not successful:
        raise ValueError(
            f"A4 cell {regime_cell_id!r} has no successful identified ranges."
        )
    covered = sum(
        bool(
            row.set_low is not None
            and row.set_high is not None
            and row.set_low <= row.reference_value <= row.set_high
        )
        for row in successful
    )
    widths = np.asarray(
        [
            float(row.set_high - row.set_low)
            for row in successful
            if row.set_low is not None and row.set_high is not None
        ],
        dtype=np.float64,
    )
    return _assemble_result(
        regime_cell_id=regime_cell_id,
        conclusion_type="identified_range",
        analysis_role=(
            "prespecified_bounded_departure"
            if reported.analysis_role == "required_primary"
            else "unrestricted_worst_case"
        ),
        estimator_id=reported.estimator_id,
        sensitivity_parameter=reported.sensitivity_parameter,
        total=len(records),
        successful=len(successful),
        covered=covered,
        widths=widths,
        reported={
            "n_worlds": reported.n_worlds,
            "n_success": reported.n_success,
            "failure_rate": reported.failure_rate,
            "conditional_coverage": reported.conditional_set_coverage,
            "unconditional_coverage": reported.unconditional_set_coverage,
            "mean_width": reported.mean_set_width,
        },
    )


def _verify_bounded_ranges_are_informative(
    *,
    worlds: tuple[_WorldRecord, ...],
    results: list[IdentificationReliabilityResultV1],
) -> None:
    """Require bounded departures to narrow matched unrestricted ranges."""

    for regime_cell_id in ("TE-S04-A4", "TE-S06-A4"):
        by_role = {
            row.analysis_role: row
            for row in results
            if row.regime_cell_id == regime_cell_id
        }
        bounded = by_role["prespecified_bounded_departure"]
        unrestricted = by_role["unrestricted_worst_case"]
        if bounded.mean_width >= unrestricted.mean_width:
            raise ValueError(
                f"A4 cell {regime_cell_id!r} has a prespecified departure range "
                "that is not narrower than its unrestricted worst-case range."
            )
        paired_successes = 0
        for world in worlds:
            if world.regime_cell_id != regime_cell_id:
                continue
            records = {row.analysis_role: row for row in world.identified_set_records}
            if set(records) != {
                "required_primary",
                "credit_eligible_primary_alternative",
            }:
                raise ValueError(
                    f"A4 cell {regime_cell_id!r} requires one bounded and one unrestricted range per trial."
                )
            bounded_record = records["required_primary"]
            unrestricted_record = records["credit_eligible_primary_alternative"]
            if (
                bounded_record.status != "success"
                or unrestricted_record.status != "success"
            ):
                continue
            assert bounded_record.set_low is not None
            assert bounded_record.set_high is not None
            assert unrestricted_record.set_low is not None
            assert unrestricted_record.set_high is not None
            paired_successes += 1
            bounded_width = bounded_record.set_high - bounded_record.set_low
            unrestricted_width = (
                unrestricted_record.set_high - unrestricted_record.set_low
            )
            if bounded_width >= unrestricted_width:
                raise ValueError(
                    f"A4 cell {regime_cell_id!r} contains a trial where the prespecified "
                    "departure limit does not narrow the unrestricted range."
                )
        if paired_successes == 0:
            raise ValueError(
                f"A4 cell {regime_cell_id!r} has no successful matched range pair."
            )


def _verify_repeated_interval(
    *,
    worlds: tuple[_WorldRecord, ...],
    reported: _ReportedPoint,
    monitoring: _ReportedMonitoring,
    bootstrap_replicates: int,
    seed: int,
) -> IdentificationReliabilityResultV1:
    selected = [world for world in worlds if world.regime_cell_id == "TE-S09-A4"]
    records: list[_PointWorldRecord] = []
    stopped_early = 0
    for world in selected:
        matching = [
            row
            for row in world.point_records
            if row.method_signature_id == reported.method_signature_id
            and row.estimator_id == reported.estimator_id
            and row.component_id is None
        ]
        if len(matching) != 1:
            raise ValueError(
                "Each TE-S09-A4 trial must contain one primary repeated-interval result."
            )
        if world.group_sequential_monitoring is None:
            raise ValueError(
                "Each TE-S09-A4 trial must contain its realized monitoring path."
            )
        records.append(matching[0])
        stopped_early += int(world.group_sequential_monitoring.stopped_early)
    successful = [row for row in records if row.status == "success"]
    if not successful:
        raise ValueError("TE-S09-A4 has no successful repeated-interval analyses.")
    covered = sum(
        bool(
            row.interval_low is not None
            and row.interval_high is not None
            and row.interval_low <= row.reference_value <= row.interval_high
        )
        for row in successful
    )
    widths = np.asarray(
        [
            float(row.interval_high - row.interval_low)
            for row in successful
            if row.interval_low is not None and row.interval_high is not None
        ],
        dtype=np.float64,
    )
    errors = np.asarray(
        [
            float(row.estimate - row.reference_value)
            for row in successful
            if row.estimate is not None
        ],
        dtype=np.float64,
    )
    bias, bias_low, bias_high = _mean_interval(errors)
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    rmse_low, rmse_high = _rmse_interval(
        errors,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    result = _assemble_result(
        regime_cell_id="TE-S09-A4",
        conclusion_type="repeated_interval",
        analysis_role="repeated_monitoring",
        estimator_id=reported.estimator_id,
        sensitivity_parameter=None,
        total=len(records),
        successful=len(successful),
        covered=covered,
        widths=widths,
        reported={
            "n_worlds": reported.n_worlds,
            "n_success": reported.n_success,
            "failure_rate": reported.failure_rate,
            "conditional_coverage": reported.conditional_coverage,
            "unconditional_coverage": reported.unconditional_coverage,
            "mean_width": reported.mean_interval_width,
            "bias": reported.bias,
            "rmse": reported.rmse,
        },
        bias=(bias, bias_low, bias_high),
        rmse=(rmse, rmse_low, rmse_high),
        stopped_early=stopped_early,
    )
    _assert_close("monitoring n_worlds", len(records), monitoring.n_worlds)
    _assert_close(
        "efficacy_stop_worlds", stopped_early, monitoring.efficacy_stop_worlds
    )
    _assert_close(
        "final_analysis_worlds",
        len(records) - stopped_early,
        monitoring.final_analysis_worlds,
    )
    return result


def _assemble_result(
    *,
    regime_cell_id: str,
    conclusion_type: Literal["identified_range", "repeated_interval"],
    analysis_role: _PublicAnalysisRole,
    estimator_id: str,
    sensitivity_parameter: float | None,
    total: int,
    successful: int,
    covered: int,
    widths: NDArray[np.float64],
    reported: dict[str, float | int],
    bias: tuple[float, float, float] | None = None,
    rmse: tuple[float, float, float] | None = None,
    stopped_early: int | None = None,
) -> IdentificationReliabilityResultV1:
    failures = total - successful
    conditional_coverage = covered / successful
    coverage = covered / total
    failure_rate = failures / total
    width, width_low, width_high = _mean_interval(widths)
    width_low = max(0.0, width_low)
    recomputed: dict[str, float | int] = {
        "n_worlds": total,
        "n_success": successful,
        "failure_rate": failure_rate,
        "conditional_coverage": conditional_coverage,
        "unconditional_coverage": coverage,
        "mean_width": width,
    }
    if bias is not None and rmse is not None:
        recomputed.update({"bias": bias[0], "rmse": rmse[0]})
    for field, value in recomputed.items():
        _assert_close(field, value, reported[field])
    coverage_low, coverage_high = proportion_interval(covered, total)
    failure_low, failure_high = proportion_interval(failures, total)
    early_rate = early_low = early_high = None
    if stopped_early is not None:
        early_rate = stopped_early / total
        early_low, early_high = proportion_interval(stopped_early, total)
    return IdentificationReliabilityResultV1(
        series_id=regime_cell_id.rsplit("-", 1)[0],
        regime_cell_id=regime_cell_id,
        conclusion_type=conclusion_type,
        analysis_role=analysis_role,
        estimator_id=estimator_id,
        effect_scale="risk_difference_tau",
        sensitivity_parameter=sensitivity_parameter,
        sensitivity_parameter_unit=(
            None if sensitivity_parameter is None else "risk_probability_difference"
        ),
        independent_trials=total,
        successful_analyses=successful,
        fit_failures=failures,
        coverage=coverage,
        coverage_low=coverage_low,
        coverage_high=coverage_high,
        conditional_coverage=conditional_coverage,
        mean_width=width,
        mean_width_low=width_low,
        mean_width_high=width_high,
        fit_failure_rate=failure_rate,
        fit_failure_rate_low=failure_low,
        fit_failure_rate_high=failure_high,
        bias=None if bias is None else bias[0],
        bias_low=None if bias is None else bias[1],
        bias_high=None if bias is None else bias[2],
        rmse=None if rmse is None else rmse[0],
        rmse_low=None if rmse is None else rmse[1],
        rmse_high=None if rmse is None else rmse[2],
        early_stop_rate=early_rate,
        early_stop_rate_low=early_low,
        early_stop_rate_high=early_high,
    )


def _mean_interval(values: NDArray[np.float64]) -> tuple[float, float, float]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("A mean interval requires a non-empty finite vector.")
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, mean, mean
    standard_error = float(stats.sem(values))
    if standard_error == 0.0:
        return mean, mean, mean
    low, high = stats.t.interval(
        confidence=0.95,
        df=int(values.size - 1),
        loc=mean,
        scale=standard_error,
    )
    return mean, min(float(low), mean), max(float(high), mean)


def _rmse_interval(
    errors: NDArray[np.float64],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[float, float]:
    """Return a deterministic trial-bootstrap interval for RMSE."""

    if errors.ndim != 1 or errors.size < 2 or not np.isfinite(errors).all():
        raise ValueError("RMSE uncertainty requires at least two finite trial errors.")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, errors.size, size=(bootstrap_replicates, errors.size)
    )
    bootstrap = np.sqrt(np.mean(np.square(errors[indices]), axis=1))
    point = float(np.sqrt(np.mean(np.square(errors))))
    low, high = (float(value) for value in np.quantile(bootstrap, (0.025, 0.975)))
    return min(low, point), max(high, point)


def _assert_close(label: str, observed: float | int, reported: float | int) -> None:
    if not math.isclose(float(observed), float(reported), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"A4 reliability summary disagrees with trial records: "
            f"field={label!r} observed={observed!r} reported={reported!r}."
        )


__all__ = [
    "IdentificationReliabilityResultV1",
    "verify_identification_reliability",
    "write_identification_reliability_csv",
]
