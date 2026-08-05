"""Operating-characteristic uncertainty and numerical agreement."""

from __future__ import annotations

import math

from statsmodels.stats.proportion import proportion_confint


def proportion_interval(
    successes: int,
    total: int,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion.

    Parameters
    ----------
    successes
        Number of events satisfying the binary criterion.
    total
        Number of scheduled observations in the denominator.
    alpha
        Two-sided error probability.

    Returns
    -------
    tuple[float, float]
        Lower and upper Wilson score limits.

    Raises
    ------
    ValueError
        If counts or ``alpha`` are outside their valid ranges.
    """

    if total <= 0:
        raise ValueError("A proportion interval requires a positive denominator.")
    if successes < 0 or successes > total:
        raise ValueError("Proportion successes must lie within the denominator.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "Proportion interval alpha must lie strictly between zero and one."
        )
    low, high = proportion_confint(
        count=successes,
        nobs=total,
        alpha=alpha,
        method="wilson",
    )
    return max(0.0, float(low)), min(1.0, float(high))


def scale_aware_tolerance(
    standard_error: float,
    *,
    absolute_tolerance: float = 1e-6,
    standard_error_fraction: float = 1e-3,
) -> float:
    """Return the larger numerical or sampling-uncertainty tolerance.

    Parameters
    ----------
    standard_error
        Standard error of the reference estimate.
    absolute_tolerance
        Minimum numerical tolerance.
    standard_error_fraction
        Maximum accepted disagreement as a fraction of sampling uncertainty.

    Returns
    -------
    float
        Scale-aware estimator-agreement tolerance.

    Raises
    ------
    ValueError
        If an input is non-finite or non-positive.
    """

    if not math.isfinite(standard_error) or standard_error <= 0:
        raise ValueError(
            "Estimator agreement requires a positive finite standard error."
        )
    if not math.isfinite(absolute_tolerance) or absolute_tolerance <= 0:
        raise ValueError("Absolute estimator tolerance must be positive and finite.")
    if not math.isfinite(standard_error_fraction) or standard_error_fraction <= 0:
        raise ValueError(
            "Standard-error tolerance fraction must be positive and finite."
        )
    return max(absolute_tolerance, standard_error_fraction * standard_error)
