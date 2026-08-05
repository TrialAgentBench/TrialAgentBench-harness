"""Non-disclosing recurrent-event realism summaries."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import bootstrap


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RecurrentEventStudyFingerprintV1(_FrozenModel):
    """Aggregate recurrent-event fingerprint for one study."""

    schema_id: Literal["trialagentbench.recurrent_event_study_fingerprint/v1"] = (
        "trialagentbench.recurrent_event_study_fingerprint/v1"
    )
    anchor_id: str = Field(pattern=r"^anchor_[0-9a-f]{16}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participants: int = Field(ge=20)
    events: int = Field(ge=1)
    participants_with_event: int = Field(ge=1)
    mean_count: float = Field(gt=0, allow_inf_nan=False)
    count_variance: float = Field(ge=0, allow_inf_nan=False)
    variance_to_mean_ratio: float = Field(ge=0, allow_inf_nan=False)
    any_event_proportion: float = Field(gt=0, le=1, allow_inf_nan=False)
    gamma_frailty_variance_mom: float = Field(ge=0, allow_inf_nan=False)


class RecurrentEventPortfolioSummaryV1(_FrozenModel):
    """Across-study recurrent-event heterogeneity summary."""

    schema_id: Literal["trialagentbench.recurrent_event_portfolio_summary/v1"] = (
        "trialagentbench.recurrent_event_portfolio_summary/v1"
    )
    studies: int = Field(ge=2)
    participants: int = Field(ge=40)
    events: int = Field(ge=2)
    overdispersed_studies: int = Field(ge=0)
    variance_to_mean_ratio_median: float = Field(ge=0, allow_inf_nan=False)
    variance_to_mean_ratio_q25: float = Field(ge=0, allow_inf_nan=False)
    variance_to_mean_ratio_q75: float = Field(ge=0, allow_inf_nan=False)
    variance_to_mean_ratio_median_ci_low: float = Field(ge=0, allow_inf_nan=False)
    variance_to_mean_ratio_median_ci_high: float = Field(ge=0, allow_inf_nan=False)
    gamma_frailty_variance_median: float = Field(ge=0, allow_inf_nan=False)
    gamma_frailty_variance_q25: float = Field(ge=0, allow_inf_nan=False)
    gamma_frailty_variance_q75: float = Field(ge=0, allow_inf_nan=False)
    gamma_frailty_variance_median_ci_low: float = Field(ge=0, allow_inf_nan=False)
    gamma_frailty_variance_median_ci_high: float = Field(ge=0, allow_inf_nan=False)


def recurrent_event_study_fingerprint(
    *,
    participant_ids: Sequence[str],
    event_participant_ids: Sequence[str],
    anchor_id: str,
    source_sha256: str,
) -> RecurrentEventStudyFingerprintV1:
    """Summarize recurrent-event counts without retaining participant rows."""

    participants = tuple(str(value) for value in participant_ids)
    if len(participants) < 20 or len(set(participants)) != len(participants):
        raise ValueError(
            "Recurrent-event fingerprints require at least 20 unique participants."
        )
    participant_set = set(participants)
    event_ids = tuple(str(value) for value in event_participant_ids)
    unknown = sorted(set(event_ids) - participant_set)
    if unknown:
        raise ValueError(
            "Recurrent-event rows reference participants absent from the study population."
        )
    if not event_ids:
        raise ValueError("Recurrent-event fingerprints require at least one event.")
    event_counts = Counter(event_ids)
    counts = np.asarray(
        [event_counts.get(subject, 0) for subject in participants], dtype=np.float64
    )
    mean_count = float(counts.mean())
    count_variance = float(counts.var(ddof=1))
    variance_to_mean = float(count_variance / mean_count)
    frailty_variance = float(max(0.0, (count_variance - mean_count) / (mean_count**2)))
    return RecurrentEventStudyFingerprintV1(
        anchor_id=anchor_id,
        source_sha256=source_sha256,
        participants=len(participants),
        events=len(event_ids),
        participants_with_event=int(np.count_nonzero(counts)),
        mean_count=mean_count,
        count_variance=count_variance,
        variance_to_mean_ratio=variance_to_mean,
        any_event_proportion=float(np.mean(counts > 0)),
        gamma_frailty_variance_mom=frailty_variance,
    )


def summarize_recurrent_event_portfolio(
    fingerprints: Sequence[RecurrentEventStudyFingerprintV1],
    *,
    random_state: int = 451,
) -> RecurrentEventPortfolioSummaryV1:
    """Summarize study-level recurrent-event heterogeneity with BCa intervals."""

    rows = tuple(fingerprints)
    if len(rows) < 2 or len({row.anchor_id for row in rows}) != len(rows):
        raise ValueError(
            "Portfolio summaries require at least two unique study fingerprints."
        )
    dispersion = np.asarray(
        [row.variance_to_mean_ratio for row in rows], dtype=np.float64
    )
    frailty = np.asarray(
        [row.gamma_frailty_variance_mom for row in rows], dtype=np.float64
    )
    dispersion_ci = _median_interval(dispersion, random_state=random_state)
    frailty_ci = _median_interval(frailty, random_state=random_state + 1)
    return RecurrentEventPortfolioSummaryV1(
        studies=len(rows),
        participants=sum(row.participants for row in rows),
        events=sum(row.events for row in rows),
        overdispersed_studies=int(np.count_nonzero(dispersion > 1.0)),
        variance_to_mean_ratio_median=float(np.median(dispersion)),
        variance_to_mean_ratio_q25=float(np.quantile(dispersion, 0.25)),
        variance_to_mean_ratio_q75=float(np.quantile(dispersion, 0.75)),
        variance_to_mean_ratio_median_ci_low=dispersion_ci[0],
        variance_to_mean_ratio_median_ci_high=dispersion_ci[1],
        gamma_frailty_variance_median=float(np.median(frailty)),
        gamma_frailty_variance_q25=float(np.quantile(frailty, 0.25)),
        gamma_frailty_variance_q75=float(np.quantile(frailty, 0.75)),
        gamma_frailty_variance_median_ci_low=frailty_ci[0],
        gamma_frailty_variance_median_ci_high=frailty_ci[1],
    )


def _median_interval(
    values: npt.NDArray[np.float64],
    *,
    random_state: int,
) -> tuple[float, float]:
    result = bootstrap(
        (values,),
        np.median,
        confidence_level=0.95,
        n_resamples=9_999,
        method="BCa",
        rng=np.random.default_rng(random_state),
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


__all__ = [
    "RecurrentEventPortfolioSummaryV1",
    "RecurrentEventStudyFingerprintV1",
    "recurrent_event_study_fingerprint",
    "summarize_recurrent_event_portfolio",
]
