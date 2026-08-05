"""Independently compare routine and prespecified analyses across repeated trials."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import stats

from trialagentbench_validation.statistics.operating_characteristics import (
    proportion_interval,
)


class _SourceContract(BaseModel):
    """Subset of one public qualification artifact used by this analysis."""

    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False)


class _ConsequenceSpec(_SourceContract):
    qualification_id: str = Field(min_length=1)
    regime_cell_id: str = Field(pattern=r"^TE-S0[1-9]-A3$")
    assumption_id: str = Field(min_length=1)
    comparison_mode: Literal["interval_coverage"]
    default_estimator_id: str = Field(min_length=1)
    corrected_estimator_id: str = Field(min_length=1)


class _WorldConsequence(_SourceContract):
    qualification_spec: _ConsequenceSpec
    status: Literal["success", "fit_failure"]
    reference_value: float | None = None
    default_estimate: float | None = None
    corrected_estimate: float | None = None
    default_interval_low: float | None = None
    default_interval_high: float | None = None
    corrected_interval_low: float | None = None
    corrected_interval_high: float | None = None
    consequence_observed: bool | None = None

    @model_validator(mode="after")
    def _complete_result(self) -> _WorldConsequence:
        values = (
            self.reference_value,
            self.default_estimate,
            self.corrected_estimate,
            self.default_interval_low,
            self.default_interval_high,
            self.corrected_interval_low,
            self.corrected_interval_high,
        )
        if self.status == "fit_failure":
            if (
                any(value is not None for value in values)
                or self.consequence_observed is not None
            ):
                raise ValueError(
                    "Failed analysis comparisons cannot contain numerical results."
                )
            return self
        if any(value is None for value in values) or self.consequence_observed is None:
            raise ValueError(
                "Successful analysis comparisons require estimates, intervals, and a reference."
            )
        assert self.default_interval_low is not None
        assert self.default_interval_high is not None
        assert self.corrected_interval_low is not None
        assert self.corrected_interval_high is not None
        if self.default_interval_low > self.default_interval_high:
            raise ValueError("Default interval limits are reversed.")
        if self.corrected_interval_low > self.corrected_interval_high:
            raise ValueError("Qualified interval limits are reversed.")
        assert self.reference_value is not None
        assert self.consequence_observed is not None
        expected_consequence = bool(
            not self.default_interval_low
            <= self.reference_value
            <= self.default_interval_high
            and self.corrected_interval_low
            <= self.reference_value
            <= self.corrected_interval_high
        )
        if self.consequence_observed != expected_consequence:
            raise ValueError(
                "Stored analysis-recovery status disagrees with the default and qualified intervals."
            )
        return self


class _WorldRecord(_SourceContract):
    world_id: str = Field(min_length=1)
    practical_consequence_records: tuple[_WorldConsequence, ...] = ()


class _ReportedConsequence(_SourceContract):
    qualification_spec: _ConsequenceSpec
    n_worlds: int = Field(gt=0)
    n_success: int = Field(ge=0)
    n_consequences: int = Field(ge=0)
    failure_rate: float = Field(ge=0, le=1)
    paired_recovery_rate: float = Field(ge=0, le=1)
    default_coverage: float = Field(ge=0, le=1)
    corrected_coverage: float = Field(ge=0, le=1)
    default_rmse: float = Field(ge=0)
    corrected_rmse: float = Field(ge=0)
    relative_rmse_reduction: float


class _OperatingCharacteristics(_SourceContract):
    practical_consequences: tuple[_ReportedConsequence, ...] = Field(min_length=1)


class AnalysisReliabilityResultV1(BaseModel):
    """Independent operating characteristics for one same-estimand comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    series_id: str = Field(pattern=r"^TE-S0[1-9]$")
    regime_cell_id: str = Field(pattern=r"^TE-S0[1-9]-A3$")
    assumption: str = Field(min_length=1)
    default_estimator_id: str = Field(min_length=1)
    qualified_estimator_id: str = Field(min_length=1)
    independent_trials: int = Field(gt=0)
    successful_pairs: int = Field(ge=0)
    fit_failures: int = Field(ge=0)
    default_coverage: float = Field(ge=0, le=1)
    default_coverage_low: float = Field(ge=0, le=1)
    default_coverage_high: float = Field(ge=0, le=1)
    qualified_coverage: float = Field(ge=0, le=1)
    qualified_coverage_low: float = Field(ge=0, le=1)
    qualified_coverage_high: float = Field(ge=0, le=1)
    default_bias: float
    default_bias_low: float
    default_bias_high: float
    qualified_bias: float
    qualified_bias_low: float
    qualified_bias_high: float
    default_rmse: float = Field(ge=0)
    qualified_rmse: float = Field(ge=0)
    qualified_to_default_rmse_ratio: float = Field(ge=0)
    rmse_ratio_low: float = Field(ge=0)
    rmse_ratio_high: float = Field(ge=0)
    relative_rmse_reduction: float
    paired_recovery_rate: float = Field(ge=0, le=1)
    paired_recovery_rate_low: float = Field(ge=0, le=1)
    paired_recovery_rate_high: float = Field(ge=0, le=1)
    paired_loss_rate: float = Field(ge=0, le=1)
    paired_loss_rate_low: float = Field(ge=0, le=1)
    paired_loss_rate_high: float = Field(ge=0, le=1)
    fit_failure_rate: float = Field(ge=0, le=1)
    fit_failure_rate_low: float = Field(ge=0, le=1)
    fit_failure_rate_high: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _valid_denominators(self) -> AnalysisReliabilityResultV1:
        if self.successful_pairs + self.fit_failures != self.independent_trials:
            raise ValueError(
                "Successful and failed comparisons must equal the trial denominator."
            )
        if (
            not self.default_coverage_low
            <= self.default_coverage
            <= self.default_coverage_high
        ):
            raise ValueError("Default coverage lies outside its interval.")
        if (
            not self.qualified_coverage_low
            <= self.qualified_coverage
            <= self.qualified_coverage_high
        ):
            raise ValueError("Qualified coverage lies outside its interval.")
        if not self.default_bias_low <= self.default_bias <= self.default_bias_high:
            raise ValueError("Default bias lies outside its interval.")
        if (
            not self.qualified_bias_low
            <= self.qualified_bias
            <= self.qualified_bias_high
        ):
            raise ValueError("Qualified bias lies outside its interval.")
        if (
            not self.rmse_ratio_low
            <= self.qualified_to_default_rmse_ratio
            <= self.rmse_ratio_high
        ):
            raise ValueError("RMSE ratio lies outside its interval.")
        if (
            not self.paired_recovery_rate_low
            <= self.paired_recovery_rate
            <= self.paired_recovery_rate_high
        ):
            raise ValueError("Paired recovery rate lies outside its interval.")
        if (
            not self.paired_loss_rate_low
            <= self.paired_loss_rate
            <= self.paired_loss_rate_high
        ):
            raise ValueError("Paired loss rate lies outside its interval.")
        if (
            not self.fit_failure_rate_low
            <= self.fit_failure_rate
            <= self.fit_failure_rate_high
        ):
            raise ValueError("Fit failure rate lies outside its interval.")
        return self


@dataclass(frozen=True)
class _CompleteConsequence:
    reference: float
    default_estimate: float
    qualified_estimate: float
    default_low: float
    default_high: float
    qualified_low: float
    qualified_high: float
    consequence_observed: bool


def verify_analysis_reliability(
    *,
    world_records_path: Path,
    operating_characteristics_path: Path,
    bootstrap_replicates: int = 10_000,
    seed: int = 20_260_802,
) -> tuple[AnalysisReliabilityResultV1, ...]:
    """Recompute A3 analysis reliability from public repeated-trial records."""

    if bootstrap_replicates < 1_000:
        raise ValueError("At least 1,000 bootstrap replicates are required.")
    if seed < 0:
        raise ValueError("Bootstrap seed must be non-negative.")
    worlds = _read_worlds(world_records_path)
    operating_characteristics = _OperatingCharacteristics.model_validate_json(
        operating_characteristics_path.read_text(encoding="utf-8")
    )
    reported = {
        row.qualification_spec.qualification_id: row
        for row in operating_characteristics.practical_consequences
    }
    if len(reported) != len(operating_characteristics.practical_consequences):
        raise ValueError(
            "Reported analysis comparisons contain duplicate qualification IDs."
        )

    grouped: dict[str, list[_WorldConsequence]] = defaultdict(list)
    for world in worlds:
        for row in world.practical_consequence_records:
            grouped[row.qualification_spec.qualification_id].append(row)
    if set(grouped) != set(reported):
        raise ValueError("World and summary analysis-comparison inventories differ.")

    results = tuple(
        _recompute_comparison(
            rows=tuple(grouped[qualification_id]),
            reported=reported[qualification_id],
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + index,
        )
        for index, qualification_id in enumerate(sorted(reported))
    )
    if len({row.regime_cell_id for row in results}) != len(results):
        raise ValueError(
            "Analysis reliability requires one comparison per A3 regime cell."
        )
    return results


def write_analysis_reliability_csv(
    *,
    path: Path,
    results: tuple[AnalysisReliabilityResultV1, ...],
) -> None:
    """Write verified analysis-reliability results as a tidy CSV table."""

    if not results:
        raise ValueError("Analysis reliability output cannot be empty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(AnalysisReliabilityResultV1.model_fields)
        )
        writer.writeheader()
        writer.writerows(row.model_dump(mode="json") for row in results)


def _read_worlds(path: Path) -> tuple[_WorldRecord, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[_WorldRecord] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            raise ValueError(f"Blank qualification world at line {line_number}.")
        rows.append(_WorldRecord.model_validate_json(line))
    if not rows:
        raise ValueError("Qualification world record file is empty.")
    if len({row.world_id for row in rows}) != len(rows):
        raise ValueError("Qualification world IDs must be unique.")
    return tuple(rows)


def _recompute_comparison(
    *,
    rows: tuple[_WorldConsequence, ...],
    reported: _ReportedConsequence,
    bootstrap_replicates: int,
    seed: int,
) -> AnalysisReliabilityResultV1:
    if not rows:
        raise ValueError("Analysis comparison has no repeated-trial records.")
    specifications = {row.qualification_spec for row in rows}
    if specifications != {reported.qualification_spec}:
        raise ValueError(
            "Repeated-trial analysis specifications differ from the summary."
        )
    successful = tuple(row for row in rows if row.status == "success")
    if not successful:
        raise ValueError("Analysis comparison has no successful repeated trials.")
    complete = tuple(_complete_consequence(row) for row in successful)
    default_cover_count = sum(
        bool(row.default_low <= row.reference <= row.default_high) for row in complete
    )
    qualified_cover_count = sum(
        bool(row.qualified_low <= row.reference <= row.qualified_high)
        for row in complete
    )
    consequence_count = sum(row.consequence_observed for row in complete)
    loss_count = sum(
        bool(
            row.default_low <= row.reference <= row.default_high
            and not row.qualified_low <= row.reference <= row.qualified_high
        )
        for row in complete
    )
    default_rmse = math.sqrt(
        sum((row.default_estimate - row.reference) ** 2 for row in complete)
        / len(complete)
    )
    qualified_rmse = math.sqrt(
        sum((row.qualified_estimate - row.reference) ** 2 for row in complete)
        / len(complete)
    )
    default_errors = np.asarray(
        [row.default_estimate - row.reference for row in complete],
        dtype=np.float64,
    )
    qualified_errors = np.asarray(
        [row.qualified_estimate - row.reference for row in complete],
        dtype=np.float64,
    )
    default_bias, default_bias_low, default_bias_high = _bias_interval(default_errors)
    qualified_bias, qualified_bias_low, qualified_bias_high = _bias_interval(
        qualified_errors
    )
    relative_reduction = (
        (default_rmse - qualified_rmse) / default_rmse if default_rmse else 0.0
    )
    rmse_ratio = qualified_rmse / default_rmse if default_rmse else 0.0
    rmse_ratio_low, rmse_ratio_high = _paired_rmse_ratio_interval(
        rows=complete,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    default_coverage = default_cover_count / len(rows)
    qualified_coverage = qualified_cover_count / len(rows)
    paired_recovery_rate = consequence_count / len(rows)
    recomputed = {
        "n_worlds": len(rows),
        "n_success": len(successful),
        "n_consequences": consequence_count,
        "failure_rate": (len(rows) - len(successful)) / len(rows),
        "paired_recovery_rate": paired_recovery_rate,
        "default_coverage": default_coverage,
        "corrected_coverage": qualified_coverage,
        "default_rmse": default_rmse,
        "corrected_rmse": qualified_rmse,
        "relative_rmse_reduction": relative_reduction,
    }
    for field, value in recomputed.items():
        reported_value = getattr(reported, field)
        if not math.isclose(
            float(value), float(reported_value), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                f"Analysis reliability summary disagrees with world records: "
                f"qualification_id={reported.qualification_spec.qualification_id!r} field={field!r}."
            )
    default_low, default_high = proportion_interval(default_cover_count, len(rows))
    qualified_low, qualified_high = proportion_interval(
        qualified_cover_count, len(rows)
    )
    recovery_low, recovery_high = proportion_interval(consequence_count, len(rows))
    loss_low, loss_high = proportion_interval(loss_count, len(rows))
    fit_failures = len(rows) - len(successful)
    failure_low, failure_high = proportion_interval(fit_failures, len(rows))
    specification = reported.qualification_spec
    return AnalysisReliabilityResultV1(
        series_id=specification.regime_cell_id.rsplit("-", 1)[0],
        regime_cell_id=specification.regime_cell_id,
        assumption=specification.assumption_id.replace("_", " "),
        default_estimator_id=specification.default_estimator_id,
        qualified_estimator_id=specification.corrected_estimator_id,
        independent_trials=len(rows),
        successful_pairs=len(successful),
        fit_failures=fit_failures,
        default_coverage=default_coverage,
        default_coverage_low=default_low,
        default_coverage_high=default_high,
        qualified_coverage=qualified_coverage,
        qualified_coverage_low=qualified_low,
        qualified_coverage_high=qualified_high,
        default_bias=default_bias,
        default_bias_low=default_bias_low,
        default_bias_high=default_bias_high,
        qualified_bias=qualified_bias,
        qualified_bias_low=qualified_bias_low,
        qualified_bias_high=qualified_bias_high,
        default_rmse=default_rmse,
        qualified_rmse=qualified_rmse,
        qualified_to_default_rmse_ratio=rmse_ratio,
        rmse_ratio_low=rmse_ratio_low,
        rmse_ratio_high=rmse_ratio_high,
        relative_rmse_reduction=relative_reduction,
        paired_recovery_rate=paired_recovery_rate,
        paired_recovery_rate_low=recovery_low,
        paired_recovery_rate_high=recovery_high,
        paired_loss_rate=loss_count / len(rows),
        paired_loss_rate_low=loss_low,
        paired_loss_rate_high=loss_high,
        fit_failure_rate=fit_failures / len(rows),
        fit_failure_rate_low=failure_low,
        fit_failure_rate_high=failure_high,
    )


def _paired_rmse_ratio_interval(
    *,
    rows: tuple[_CompleteConsequence, ...],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[float, float]:
    default_squared_error = np.asarray(
        [(row.default_estimate - row.reference) ** 2 for row in rows],
        dtype=np.float64,
    )
    qualified_squared_error = np.asarray(
        [(row.qualified_estimate - row.reference) ** 2 for row in rows],
        dtype=np.float64,
    )
    if not np.any(default_squared_error > 0):
        raise ValueError(
            "Default RMSE is zero; its ratio to qualified RMSE is undefined."
        )
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        low=0,
        high=len(rows),
        size=(bootstrap_replicates, len(rows)),
    )
    default_rmse = np.sqrt(default_squared_error[indices].mean(axis=1))
    qualified_rmse = np.sqrt(qualified_squared_error[indices].mean(axis=1))
    defined = default_rmse > 0
    if float(np.mean(defined)) < 0.95:
        raise ValueError(
            "RMSE-ratio bootstrap is undefined in more than 5% of replicates."
        )
    ratios = qualified_rmse[defined] / default_rmse[defined]
    point = math.sqrt(float(qualified_squared_error.mean())) / math.sqrt(
        float(default_squared_error.mean())
    )
    low, high = (float(value) for value in np.quantile(ratios, (0.025, 0.975)))
    return min(low, point), max(high, point)


def _bias_interval(errors: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """Return mean signed error and its trial-level Student t interval."""

    if errors.ndim != 1 or errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("Bias requires a non-empty finite error vector.")
    mean = float(np.mean(errors))
    if errors.size == 1:
        return mean, mean, mean
    standard_error = float(stats.sem(errors))
    if standard_error == 0.0:
        return mean, mean, mean
    low, high = stats.t.interval(
        confidence=0.95,
        df=int(errors.size - 1),
        loc=mean,
        scale=standard_error,
    )
    return mean, min(float(low), mean), max(float(high), mean)


def _complete_consequence(row: _WorldConsequence) -> _CompleteConsequence:
    if (
        row.reference_value is None
        or row.default_estimate is None
        or row.corrected_estimate is None
        or row.default_interval_low is None
        or row.default_interval_high is None
        or row.corrected_interval_low is None
        or row.corrected_interval_high is None
        or row.consequence_observed is None
    ):
        raise ValueError("Successful analysis comparison is incomplete.")
    return _CompleteConsequence(
        reference=float(row.reference_value),
        default_estimate=float(row.default_estimate),
        qualified_estimate=float(row.corrected_estimate),
        default_low=float(row.default_interval_low),
        default_high=float(row.default_interval_high),
        qualified_low=float(row.corrected_interval_low),
        qualified_high=float(row.corrected_interval_high),
        consequence_observed=bool(row.consequence_observed),
    )


__all__ = [
    "AnalysisReliabilityResultV1",
    "verify_analysis_reliability",
    "write_analysis_reliability_csv",
]
