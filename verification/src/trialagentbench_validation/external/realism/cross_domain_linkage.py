"""Exact-marginal cross-domain linkage analysis."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import t

from trialagentbench_validation.statistics import proportion_interval

_COUNT_COLUMNS = (
    "assessment_count",
    "biosample_count",
    "adverse_event_count",
    "intervention_count",
)
_RETENTION_LEVELS = (1.0, 0.75, 0.5, 0.25, 0.0)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CrossDomainWorldEstimateV1(_FrozenModel):
    """One exact-marginal linkage intervention result."""

    anchor_id: str = Field(pattern=r"^anchor_[0-9a-f]{16}$")
    world_index: int = Field(ge=0)
    linkage_retention: float = Field(ge=0, le=1)
    association_divergence: float = Field(ge=0, allow_inf_nan=False)
    safety_analysis_perturbation: float = Field(ge=0, allow_inf_nan=False)


class CrossDomainResponseV1(_FrozenModel):
    """Within-world response to progressive linkage breakage."""

    anchor_id: str = Field(pattern=r"^anchor_[0-9a-f]{16}$")
    response: Literal["association_divergence", "safety_analysis_perturbation"]
    worlds: int = Field(ge=2)
    mean_slope: float = Field(allow_inf_nan=False)
    slope_ci_low: float = Field(allow_inf_nan=False)
    slope_ci_high: float = Field(allow_inf_nan=False)
    positive_slope_fraction: float = Field(ge=0, le=1)
    positive_slope_fraction_ci_low: float = Field(ge=0, le=1)
    positive_slope_fraction_ci_high: float = Field(ge=0, le=1)


class CrossDomainPortfolioResponseV1(_FrozenModel):
    """Equal-study summary of cross-domain linkage response."""

    response: Literal["association_divergence", "safety_analysis_perturbation"]
    studies: int = Field(ge=2)
    mean_slope: float = Field(allow_inf_nan=False)
    slope_ci_low: float = Field(allow_inf_nan=False)
    slope_ci_high: float = Field(allow_inf_nan=False)
    positive_studies: int = Field(ge=0)


class CrossDomainStudyV1(_FrozenModel):
    """Aggregate identity and source associations for one study."""

    anchor_id: str = Field(pattern=r"^anchor_[0-9a-f]{16}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participants: int = Field(ge=20)
    log_count_correlations: tuple[float, ...] = Field(min_length=6, max_length=6)


class CrossDomainLinkageReportV1(_FrozenModel):
    """Complete non-disclosing cross-domain linkage report."""

    schema_id: str = "trialagentbench.cross_domain_linkage/v1"
    studies: tuple[CrossDomainStudyV1, ...]
    estimates: tuple[CrossDomainWorldEstimateV1, ...]
    responses: tuple[CrossDomainResponseV1, ...]
    portfolio_responses: tuple[CrossDomainPortfolioResponseV1, ...]
    maximum_marginal_difference: float = Field(ge=0, allow_inf_nan=False)


def analyze_cross_domain_linkage(
    frame: pd.DataFrame,
    *,
    source_object_id: str,
    source_sha256: str,
    worlds: int = 100,
    seed: int = 451_2061,
) -> CrossDomainLinkageReportV1:
    """Measure dependence and analysis loss under exact-marginal shuffling."""

    if worlds < 2:
        raise ValueError("Cross-domain linkage analysis requires at least two worlds.")
    if missing := sorted(set(_COUNT_COLUMNS) - set(frame.columns)):
        raise ValueError(f"Cross-domain frame is missing count columns: {missing!r}.")
    counts = frame.loc[:, _COUNT_COLUMNS].apply(pd.to_numeric, errors="raise")
    if len(counts) < 20 or counts.isna().any().any():
        raise ValueError(
            "Cross-domain analysis requires at least 20 complete participants."
        )
    if (counts < 0).any().any() or counts.nunique().lt(2).any():
        raise ValueError("Cross-domain counts must be non-negative and nonconstant.")
    anchor_id = "anchor_" + hashlib.sha256(source_object_id.encode()).hexdigest()[:16]
    source_matrix = _transformed(counts)
    source_correlation = np.asarray(
        np.corrcoef(source_matrix, rowvar=False),
        dtype=np.float64,
    )
    study = CrossDomainStudyV1(
        anchor_id=anchor_id,
        source_sha256=source_sha256,
        participants=len(counts),
        log_count_correlations=tuple(_lower_triangle(source_correlation)),
    )
    estimates = []
    for world_index in range(worlds):
        rng = np.random.default_rng(_world_seed(seed, source_object_id, world_index))
        sampled = counts.reset_index(drop=True)
        intact = _transformed(sampled)
        intact_correlation = np.asarray(
            np.corrcoef(intact, rowvar=False),
            dtype=np.float64,
        )
        intact_coefficients = _safety_coefficients(intact)
        for retention in _RETENTION_LEVELS:
            disrupted = _disrupt(sampled, retention=retention, rng=rng)
            if _maximum_marginal_difference(disrupted, sampled) != 0:
                raise RuntimeError(
                    "Cross-domain intervention changed a marginal multiset."
                )
            transformed = _transformed(disrupted)
            divergence = _correlation_divergence(
                np.asarray(
                    np.corrcoef(transformed, rowvar=False),
                    dtype=np.float64,
                ),
                intact_correlation,
            )
            perturbation = float(
                np.linalg.norm(
                    _safety_coefficients(transformed) - intact_coefficients,
                    ord=2,
                )
            )
            estimates.append(
                CrossDomainWorldEstimateV1(
                    anchor_id=anchor_id,
                    world_index=world_index,
                    linkage_retention=retention,
                    association_divergence=divergence,
                    safety_analysis_perturbation=perturbation,
                )
            )
    return CrossDomainLinkageReportV1(
        studies=(study,),
        estimates=tuple(estimates),
        responses=tuple(_responses(estimates, anchor_id=anchor_id)),
        portfolio_responses=(),
        maximum_marginal_difference=0.0,
    )


def combine_cross_domain_reports(
    reports: tuple[CrossDomainLinkageReportV1, ...],
) -> CrossDomainLinkageReportV1:
    """Combine distinct study reports without changing study weighting."""

    if len(reports) < 2:
        raise ValueError("Cross-domain portfolio requires at least two studies.")
    studies = tuple(row for report in reports for row in report.studies)
    if len({row.anchor_id for row in studies}) != len(studies):
        raise ValueError("Cross-domain portfolio contains duplicate study identities.")
    responses = tuple(row for report in reports for row in report.responses)
    return CrossDomainLinkageReportV1(
        studies=studies,
        estimates=tuple(row for report in reports for row in report.estimates),
        responses=responses,
        portfolio_responses=tuple(_portfolio_responses(responses)),
        maximum_marginal_difference=max(
            report.maximum_marginal_difference for report in reports
        ),
    )


def _disrupt(
    frame: pd.DataFrame,
    *,
    retention: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if not 0 <= retention <= 1:
        raise ValueError("Linkage retention must lie in [0, 1].")
    output = frame.copy()
    count = int(round((1.0 - retention) * len(output)))
    if count == 0:
        return output
    indexes = np.sort(rng.choice(len(output), size=count, replace=False))
    for column in _COUNT_COLUMNS:
        values = output.loc[indexes, column].to_numpy(copy=True)
        output.loc[indexes, column] = values[rng.permutation(len(values))]
    return output


def _transformed(frame: pd.DataFrame) -> npt.NDArray[np.float64]:
    values = np.log1p(frame.loc[:, _COUNT_COLUMNS].to_numpy(dtype=float))
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("A resampled cross-domain world has a constant domain.")
    return np.asarray((values - means) / scales, dtype=np.float64)


def _safety_coefficients(
    standardized: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    outcome = standardized[:, 2]
    predictors = np.column_stack(
        (
            np.ones(len(standardized)),
            standardized[:, 0],
            standardized[:, 1],
            standardized[:, 3],
        )
    )
    coefficients, _, rank, _ = np.linalg.lstsq(predictors, outcome, rcond=None)
    if rank < 2 or not np.isfinite(coefficients).all():
        raise ValueError("Cross-domain safety regression is not estimable.")
    return np.asarray(coefficients[1:], dtype=np.float64)


def _correlation_divergence(
    observed: npt.NDArray[np.float64],
    reference: npt.NDArray[np.float64],
) -> float:
    difference = observed - reference
    return float(np.sqrt(np.sum(np.square(difference)) / 12.0))


def _responses(
    estimates: list[CrossDomainWorldEstimateV1],
    *,
    anchor_id: str,
) -> list[CrossDomainResponseV1]:
    by_world: defaultdict[int, list[CrossDomainWorldEstimateV1]] = defaultdict(list)
    for row in estimates:
        by_world[row.world_index].append(row)
    output = []
    for response in ("association_divergence", "safety_analysis_perturbation"):
        slopes = []
        for rows in by_world.values():
            ordered = sorted(rows, key=lambda row: 1.0 - row.linkage_retention)
            broken = [1.0 - row.linkage_retention for row in ordered]
            slopes.append(
                float(
                    np.polyfit(
                        broken,
                        [float(getattr(row, response)) for row in ordered],
                        1,
                    )[0]
                )
            )
        values = np.asarray(slopes, dtype=np.float64)
        low, high = _mean_interval(values)
        positive_interval = proportion_interval(
            int(np.sum(values > 0.0)),
            len(values),
        )
        output.append(
            CrossDomainResponseV1(
                anchor_id=anchor_id,
                response=response,
                worlds=len(values),
                mean_slope=float(values.mean()),
                slope_ci_low=low,
                slope_ci_high=high,
                positive_slope_fraction=float(np.mean(values > 0)),
                positive_slope_fraction_ci_low=positive_interval[0],
                positive_slope_fraction_ci_high=positive_interval[1],
            )
        )
    return output


def _portfolio_responses(
    responses: tuple[CrossDomainResponseV1, ...],
) -> list[CrossDomainPortfolioResponseV1]:
    output = []
    for response in ("association_divergence", "safety_analysis_perturbation"):
        values = np.asarray(
            [row.mean_slope for row in responses if row.response == response],
            dtype=np.float64,
        )
        if len(values) < 2:
            raise ValueError("Cross-domain portfolio response requires two studies.")
        low, high = _mean_interval(values)
        output.append(
            CrossDomainPortfolioResponseV1(
                response=response,
                studies=len(values),
                mean_slope=float(values.mean()),
                slope_ci_low=low,
                slope_ci_high=high,
                positive_studies=int(np.sum(values > 0)),
            )
        )
    return output


def _maximum_marginal_difference(
    observed: pd.DataFrame,
    reference: pd.DataFrame,
) -> float:
    return max(
        float(
            np.max(
                np.abs(
                    np.sort(observed[column].to_numpy(dtype=float))
                    - np.sort(reference[column].to_numpy(dtype=float))
                )
            )
        )
        for column in _COUNT_COLUMNS
    )


def _lower_triangle(matrix: npt.NDArray[np.float64]) -> list[float]:
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Cross-domain correlation matrix must be finite and 4 by 4.")
    return [float(matrix[row, column]) for row in range(1, 4) for column in range(row)]


def _mean_interval(values: npt.NDArray[np.float64]) -> tuple[float, float]:
    half_width = float(
        t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )
    mean = float(values.mean())
    return mean - half_width, mean + half_width


def _world_seed(seed: int, source_object_id: str, world_index: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{source_object_id}:{world_index}".encode()).digest()[
            :8
        ],
        "big",
    )


__all__ = [
    "CrossDomainLinkageReportV1",
    "analyze_cross_domain_linkage",
    "combine_cross_domain_reports",
]
