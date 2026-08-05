"""Explicit mathematical equivalences for TrialEval effect scales."""

from __future__ import annotations

import math

_HAZARD_RATIO_SCALES = frozenset({"hazard_ratio", "log_hr"})
_ROUTE_FAMILY_BY_EFFECT_SCALE = {
    "risk_difference_tau": "risk_difference",
    "standardized_risk_difference_tau_reference": "standardized_risk",
    "rmst_difference_tau": "rmst_contrast",
    "log_hr": "global_cox_ph",
    "hr": "global_cox_ph",
    "hazard_ratio": "global_cox_ph",
    "time_varying_log_hr": "time_varying_cox",
    "piecewise_log_hr_vector": "piecewise_cox",
    "weighted_logrank_test": "weighted_logrank",
    "log_time_ratio": "aft_parametric",
    "cif_difference_tau": "competing_risk",
    "milestone_risk_difference_tau": "milestone_risk",
    "bounds_interval": "partial_identification",
    "non_identification": "qualified_limitation",
}


def route_family_for_effect_scale_v1(effect_scale: str) -> str:
    """Return the analysis family represented by a supported effect scale."""

    try:
        return _ROUTE_FAMILY_BY_EFFECT_SCALE[effect_scale]
    except KeyError as error:
        raise ValueError(f"Unsupported TrialEval effect scale: {effect_scale!r}") from error


def effect_scales_equivalent_v1(left: str, right: str) -> bool:
    """Return whether two scales encode the same oriented estimand."""

    if not left or not right:
        return False
    return left == right or {left, right} == _HAZARD_RATIO_SCALES


def convert_effect_value_v1(*, value: float, source_scale: str, target_scale: str) -> float | None:
    """Convert a finite value between explicitly equivalent scales."""

    if not math.isfinite(value):
        return None
    if source_scale == target_scale:
        return float(value)
    if source_scale == "hazard_ratio" and target_scale == "log_hr":
        return math.log(value) if value > 0.0 else None
    if source_scale == "log_hr" and target_scale == "hazard_ratio":
        converted = math.exp(value)
        return converted if math.isfinite(converted) else None
    return None


def effect_units_compatible_v1(
    *,
    expected_unit: str,
    submitted_unit: str,
    expected_scale: str,
    submitted_scale: str,
) -> bool:
    """Return whether result units match directly or through HR/log-HR equivalence."""

    if expected_unit == submitted_unit:
        return True
    return (
        expected_scale == "log_hr"
        and submitted_scale == "hazard_ratio"
        and expected_unit == "log_hazard_ratio"
        and submitted_unit == "hazard_ratio"
    ) or (
        expected_scale == "hazard_ratio"
        and submitted_scale == "log_hr"
        and expected_unit == "hazard_ratio"
        and submitted_unit == "log_hazard_ratio"
    )


__all__ = [
    "route_family_for_effect_scale_v1",
    "convert_effect_value_v1",
    "effect_scales_equivalent_v1",
    "effect_units_compatible_v1",
]
