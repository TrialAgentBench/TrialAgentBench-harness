"""Evaluate the published numerical-loss equation."""

from __future__ import annotations


def numerical_loss(estimate: float, lower: float, upper: float, reference: float, scale: float) -> float:
    """Evaluate the declared mean point/interval loss at 95% confidence."""
    interval = upper - lower + 40 * max(lower - reference, reference - upper, 0)
    return (abs(estimate - reference) + interval) / (2 * scale)
