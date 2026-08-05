"""Canonical estimator method groups used by standalone score validation."""

from __future__ import annotations

BOUNDED_DEVIATION_METHOD_IDS_V1 = frozenset(
    {
        "observed:tau_bounds_bounded_deviation",
        "observed:validated_endpoint_bounded_deviation",
    }
)
