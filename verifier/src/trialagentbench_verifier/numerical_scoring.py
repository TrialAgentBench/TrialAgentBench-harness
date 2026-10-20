"""Shared numerical recovery loss in runner-declared scientific units."""

import math


def interval_score(lower: float, upper: float, target: float, confidence_level: float) -> float:
    """Return interval width plus the confidence-weighted noncoverage penalty.

    Parameters
    ----------
    lower, upper, target
        Finite interval endpoints and reference value in the same units.
    confidence_level
        Assigned confidence level strictly between zero and one.

    Returns
    -------
    float
        Untransformed interval loss in the supplied units.

    Raises
    ------
    ValueError
        If inputs are nonfinite, reversed or have invalid confidence.
    """
    if not all(math.isfinite(value) for value in (lower, upper, target, confidence_level)):
        raise ValueError("numerical scoring requires finite inputs")
    if lower > upper or not 0 < confidence_level < 1:
        raise ValueError("numerical scoring requires ordered limits and valid confidence")
    alpha = 1.0 - confidence_level
    penalty = 0.0
    if target < lower:
        penalty = 2.0 * (lower - target) / alpha
    elif target > upper:
        penalty = 2.0 * (target - upper) / alpha
    return upper - lower + penalty


def numerical_loss(
    *, estimate: float | None, lower: float, upper: float, target: float, scale: float, confidence_level: float
) -> float | None:
    """Return mean normalised point and interval loss in fixed scientific units.

    Parameters
    ----------
    estimate
        Finite submitted point in the declared scoring units, or None when the point is reported but not
        scored; the loss is then the interval half alone.
    lower, upper, target
        Finite submitted interval and reference in the declared scoring units.
    scale
        Positive task-owned normalization scale.
    confidence_level
        Task-assigned interval confidence level.

    Returns
    -------
    float or None
        Untransformed loss, or None when the result exceeds floating-point range.
        Missing answers are handled separately and never receive a finite penalty.

    Raises
    ------
    ValueError
        If the estimate, scale or interval inputs are invalid.
    """
    if (estimate is not None and not math.isfinite(estimate)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("numerical loss requires a finite estimate and positive finite scale")
    uncertainty = interval_score(lower, upper, target, confidence_level)
    loss = (0.0 if estimate is None else abs(estimate - target) / scale / 2) + uncertainty / scale / 2
    if math.isfinite(loss):
        return loss
    # Resolve intermediate overflow without clipping the scientific loss.
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 80
        lo, hi, truth, denominator, confidence = map(
            Decimal.from_float, (lower, upper, target, scale, confidence_level)
        )
        point = Decimal(0) if estimate is None else abs(Decimal.from_float(estimate) - truth)
        uncertainty_decimal = hi - lo + 2 * max(lo - truth, truth - hi, Decimal(0)) / (1 - confidence)
        exact = (point + uncertainty_decimal) / (2 * denominator)
    return finite_diagnostic(float(exact))


def bounded_recovery(loss: float | None) -> float:
    """Map one numerical loss to bounded recovery under the declared missing-answer rule.

    Parameters
    ----------
    loss
        Nonnegative numerical loss, or None when the scheduled answer has no
        numerical loss for any reason: missing or invalid submission, failed
        replay, timeout, evaluator error or overflow.

    Returns
    -------
    float
        ``1 / (1 + loss)`` in ``(0, 1]``; a missing loss receives the maximum
        penalty, zero. Means of this quantity are defined for every scheduled
        panel and no single unbounded loss can dominate them.

    Raises
    ------
    ValueError
        If the loss is negative or not finite.

    Examples
    --------
    >>> bounded_recovery(1.0), bounded_recovery(None)
    (0.5, 0.0)
    """
    if loss is None:
        return 0.0
    if not math.isfinite(loss) or loss < 0:
        raise ValueError("bounded recovery requires a finite nonnegative loss")
    return 1.0 / (1.0 + loss)


def finite_diagnostic(value: float) -> float | None:
    """Represent an overflowing derived diagnostic as null, without clipping it.

    Parameters
    ----------
    value
        A derived error, width or loss from validated finite scoring inputs.

    Returns
    -------
    float or None
        The original finite value, or None when it exceeds floating-point range.
        Raw submissions and the aggregate scientific loss remain separately available.
    """
    return value if math.isfinite(value) else None
