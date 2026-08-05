"""Decision-relevant scientific equivalence for TrialDev estimates."""

from __future__ import annotations

import math

from trialagentbench_harness.contracts.trialdev.scientific_grade import (
    TrialDevScientificEnvelopeV1,
)


def _finite_interval(estimate: float, lower: float, upper: float) -> bool:
    return all(math.isfinite(float(value)) for value in (estimate, lower, upper)) and float(lower) <= float(
        estimate
    ) <= float(upper)


def intervals_overlap_v1(
    *,
    lower: float,
    upper: float,
    reference_lower: float,
    reference_upper: float,
) -> bool:
    """Return whether two ordered closed uncertainty intervals overlap."""

    if not _finite_interval(lower, lower, upper) or not _finite_interval(
        reference_lower,
        reference_lower,
        reference_upper,
    ):
        return False
    return max(float(lower), float(reference_lower)) <= min(float(upper), float(reference_upper))


def threshold_state_v1(*, estimate: float, lower: float, upper: float, threshold: float) -> str:
    """Classify an estimate and interval relative to one clinical threshold."""

    if not _finite_interval(estimate, lower, upper) or not math.isfinite(float(threshold)):
        return "invalid"
    if float(lower) > float(threshold):
        return "clearly_above"
    if float(upper) < float(threshold):
        return "clearly_below"
    return "uncertain"


def scientific_bundle_agreement_v1(
    *,
    estimate: float,
    lower: float,
    upper: float,
    reference_estimate: float,
    reference_lower: float,
    reference_upper: float,
    envelope: TrialDevScientificEnvelopeV1,
) -> bool:
    """Compare estimate bundles using a frozen clinical decision boundary."""

    if not _finite_interval(estimate, lower, upper) or not _finite_interval(
        reference_estimate,
        reference_lower,
        reference_upper,
    ):
        return False
    if not intervals_overlap_v1(
        lower=lower,
        upper=upper,
        reference_lower=reference_lower,
        reference_upper=reference_upper,
    ):
        return False
    if envelope.basis == "declared_practical_equivalence_margin":
        if envelope.absolute_margin is None:
            raise ValueError("Practical-equivalence assessment lacks its absolute margin.")
        return abs(float(estimate) - float(reference_estimate)) <= float(envelope.absolute_margin)
    if envelope.basis == "declared_decision_thresholds":
        return all(
            threshold_state_v1(
                estimate=estimate,
                lower=lower,
                upper=upper,
                threshold=threshold,
            )
            == threshold_state_v1(
                estimate=reference_estimate,
                lower=reference_lower,
                upper=reference_upper,
                threshold=threshold,
            )
            for threshold in envelope.decision_thresholds
        )
    raise ValueError("Qualified nonidentification does not define numeric bundle agreement.")


__all__ = [
    "intervals_overlap_v1",
    "scientific_bundle_agreement_v1",
    "threshold_state_v1",
]
