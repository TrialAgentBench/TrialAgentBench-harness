"""Independent randomized-design reconstruction for TrialDev."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Protocol

import pandas as pd
from scipy.stats import beta, binom


class PhaseDesignRequest(Protocol):
    """Request fields required for independent design reconstruction."""

    @property
    def phase_id(self) -> str:
        """Return the randomized phase identity."""

    @property
    def target_sample_size(self) -> int:
        """Return the requested total sample size."""

    @property
    def design_cell_id(self) -> str:
        """Return the prospective design-cell identity."""

    @property
    def interim_policy(self) -> str:
        """Return the requested interim policy."""

    @property
    def follow_up_days(self) -> int:
        """Return the requested follow-up horizon."""

    @property
    def endpoint_id(self) -> str | None:
        """Return the requested primary endpoint."""


@dataclass(frozen=True)
class IndependentPhaseDesign:
    """Reconstructed operating characteristics for one public phase request."""

    adequate: bool
    failures: tuple[str, ...]
    achieved_power: float | None
    target_power: float | None
    achieved_safety_absolute_risk_power: float
    achieved_safety_excess_risk_power: float
    target_safety_decision_power: float


def _finite(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number.")
    return float(value)


def _risk_interval(
    *,
    estimate: float,
    sample_size: int,
    event_count: int,
    confidence_level: float,
) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    if event_count == 0:
        return 0.0, float(beta.ppf(1.0 - alpha / 2.0, 1, sample_size))
    if event_count == sample_size:
        return float(beta.ppf(alpha / 2.0, sample_size, 1)), 1.0
    z = float(NormalDist().inv_cdf(0.5 + confidence_level / 2.0))
    half_width = z * math.sqrt(estimate * (1.0 - estimate) / sample_size)
    return max(0.0, estimate - half_width), min(1.0, estimate + half_width)


def _absolute_safety_power(
    *,
    sample_size: int,
    decision_limit: float,
    planning_risk: float,
    confidence_level: float,
) -> float:
    if sample_size < 2:
        return 0.0
    if not 0.0 < decision_limit < planning_risk < 1.0:
        raise ValueError(
            "Absolute safety planning requires decision_limit < planning_risk in (0, 1)."
        )
    critical_count = next(
        (
            count
            for count in range(sample_size + 1)
            if _risk_interval(
                estimate=count / sample_size,
                sample_size=sample_size,
                event_count=count,
                confidence_level=confidence_level,
            )[0]
            > decision_limit
        ),
        None,
    )
    if critical_count is None:
        return 0.0
    return float(binom.sf(critical_count - 1, sample_size, planning_risk))


def _harmful_difference_power(
    *,
    control_sample_size: int,
    treatment_sample_size: int,
    control_risk: float,
    treatment_risk: float,
    decision_limit: float,
    confidence_level: float,
) -> float:
    if control_sample_size < 2 or treatment_sample_size < 2:
        return 0.0
    if not 0.0 <= control_risk < treatment_risk <= 1.0:
        raise ValueError(
            "Safety difference planning requires control_risk < treatment_risk in [0, 1]."
        )
    excess = treatment_risk - control_risk
    if not 0.0 < decision_limit < excess:
        raise ValueError(
            "Safety difference decision limit must be below the planning excess."
        )
    standard_error = math.sqrt(
        control_risk * (1.0 - control_risk) / control_sample_size
        + treatment_risk * (1.0 - treatment_risk) / treatment_sample_size
    )
    z = float(NormalDist().inv_cdf(0.5 + confidence_level / 2.0))
    return float(NormalDist().cdf((excess - decision_limit) / standard_error - z))


def _efficacy_power(
    *,
    control_sample_size: int,
    treatment_sample_size: int,
    control_risk: float,
    treatment_risk: float,
    alternative_benefit: float,
    confidence_level: float,
) -> float:
    if control_sample_size < 2 or treatment_sample_size < 2:
        return 0.0
    if not 0.0 <= treatment_risk <= control_risk <= 1.0:
        raise ValueError(
            "Efficacy planning requires treatment_risk <= control_risk in [0, 1]."
        )
    if abs((control_risk - treatment_risk) - alternative_benefit) > 1e-12:
        raise ValueError(
            "Efficacy benefit must equal control risk minus treatment risk."
        )
    standard_error = math.sqrt(
        control_risk * (1.0 - control_risk) / control_sample_size
        + treatment_risk * (1.0 - treatment_risk) / treatment_sample_size
    )
    if standard_error <= 0.0:
        raise ValueError("Efficacy planning variance must be positive.")
    z = float(NormalDist().inv_cdf(0.5 + confidence_level / 2.0))
    noncentrality = alternative_benefit / standard_error
    normal = NormalDist()
    return float(normal.cdf(noncentrality - z) + normal.cdf(-noncentrality - z))


def _planned_counts(
    *,
    target_sample_size: int,
    arm_ids: tuple[str, ...],
    raw_weights: object,
) -> dict[str, int]:
    if not isinstance(raw_weights, dict) or set(str(key) for key in raw_weights) != set(
        arm_ids
    ):
        raise ValueError("Arm mapping requires one public weight for every arm.")
    weights = {
        arm: _finite(raw_weights[arm], label=f"arm weight {arm}") for arm in arm_ids
    }
    if any(value <= 0.0 for value in weights.values()):
        raise ValueError("Arm weights must be positive.")
    total_weight = sum(weights.values())
    raw = {arm: target_sample_size * weights[arm] / total_weight for arm in arm_ids}
    counts = {arm: max(1, math.floor(raw[arm])) for arm in arm_ids}
    remainder = target_sample_size - sum(counts.values())
    if remainder > 0:
        order = sorted(arm_ids, key=lambda arm: (-(raw[arm] - counts[arm]), arm))
        for index in range(remainder):
            counts[order[index % len(order)]] += 1
    elif remainder < 0:
        order = sorted(arm_ids, key=lambda arm: (raw[arm] - counts[arm], arm))
        index = 0
        while remainder < 0:
            arm = order[index % len(order)]
            if counts[arm] > 1:
                counts[arm] -= 1
                remainder += 1
            index += 1
            if index > 10_000:
                raise ValueError(
                    "Unable to reconstruct a non-empty planned arm allocation."
                )
    if sum(counts.values()) != target_sample_size:
        raise ValueError("Planned arm counts do not sum to target_sample_size.")
    return counts


def reconstruct_phase_design(
    *,
    request: PhaseDesignRequest,
    arm_mapping: dict[str, object],
    safety: pd.DataFrame,
    design_policy: dict[str, object],
) -> IndependentPhaseDesign:
    """Reconstruct prospective phase operating characteristics from public inputs."""

    phase_id = str(request.phase_id)
    rules = design_policy.get("phase_rules")
    if not isinstance(rules, list):
        raise ValueError("Phase design policy requires phase_rules.")
    matches = [
        row
        for row in rules
        if isinstance(row, dict) and str(row.get("phase_id")) == phase_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "Phase design policy requires exactly one matching phase rule."
        )
    rule = matches[0]
    confidence_level = _finite(
        design_policy.get("confidence_level"), label="design confidence_level"
    )
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("Design confidence_level must lie in (0.5, 1).")

    control_arm = str(arm_mapping.get("control_arm_id") or "")
    candidates = arm_mapping.get("candidate_arm_ids")
    drugs = arm_mapping.get("drug_id_by_arm")
    if (
        not control_arm
        or not isinstance(candidates, list)
        or not candidates
        or not isinstance(drugs, dict)
    ):
        raise ValueError("Arm mapping lacks control, candidate, or drug identities.")
    treatment_arms = tuple(str(value) for value in candidates)
    arm_ids = (control_arm, *treatment_arms)
    target_sample_size = int(request.target_sample_size)
    planned = _planned_counts(
        target_sample_size=target_sample_size,
        arm_ids=arm_ids,
        raw_weights=arm_mapping.get("arm_weight_by_id"),
    )
    realized = {
        str(arm): int(count)
        for arm, count in safety["ARM"].astype(str).value_counts().to_dict().items()
    }
    if realized != planned:
        raise ValueError(
            "Materialized arm counts differ from the public planned allocation."
        )
    fractions = rule.get("planning_information_fraction_by_drug_id")
    if not isinstance(fractions, dict):
        raise ValueError("Phase design rule requires planning information fractions.")

    def effective_count(arm_id: str) -> int:
        drug_id = str(drugs.get(arm_id) or "")
        if drug_id not in fractions:
            raise ValueError(
                f"Phase design policy lacks an information fraction for {drug_id!r}."
            )
        fraction = _finite(fractions[drug_id], label=f"information fraction {drug_id}")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("Planning information fractions must lie in (0, 1].")
        return math.floor(planned[arm_id] * fraction)

    control_count = effective_count(control_arm)
    treatment_counts = tuple(effective_count(arm) for arm in treatment_arms)
    failures: list[str] = []
    if str(request.design_cell_id) != str(rule.get("design_cell_id")):
        failures.append("unaccepted_design_cell")
    if str(request.interim_policy) != str(rule.get("supported_interim_policy")):
        failures.append("unsupported_interim_policy")
    if int(request.follow_up_days) < int(rule.get("evaluation_horizon_days", -1)):
        failures.append("insufficient_follow_up")
    expected_endpoint = rule.get("primary_endpoint_id")
    requested_endpoint = request.endpoint_id
    if expected_endpoint is None and requested_endpoint is not None:
        failures.append("unexpected_primary_endpoint")
    elif expected_endpoint is not None and str(requested_endpoint or "") != str(
        expected_endpoint
    ):
        failures.append("primary_endpoint_mismatch")

    safety_absolute_power = _absolute_safety_power(
        sample_size=min(treatment_counts),
        decision_limit=_finite(
            rule.get("serious_ae_unacceptable_absolute_risk"),
            label="serious AE absolute limit",
        ),
        planning_risk=_finite(
            rule.get("planning_safety_absolute_treatment_risk"),
            label="planning safety absolute risk",
        ),
        confidence_level=confidence_level,
    )
    safety_excess_power = min(
        _harmful_difference_power(
            control_sample_size=control_count,
            treatment_sample_size=count,
            control_risk=_finite(
                rule.get("planning_safety_control_risk"), label="safety control risk"
            ),
            treatment_risk=_finite(
                rule.get("planning_safety_excess_treatment_risk"),
                label="safety treatment risk",
            ),
            decision_limit=_finite(
                rule.get("serious_ae_unacceptable_excess_risk"),
                label="serious AE excess limit",
            ),
            confidence_level=confidence_level,
        )
        for count in treatment_counts
    )
    target_safety = _finite(
        rule.get("target_safety_decision_power"), label="target safety power"
    )
    if min(safety_absolute_power, safety_excess_power) + 1e-12 < target_safety:
        failures.append("insufficient_safety_decision_power")

    target_power_raw = rule.get("target_power")
    target_power = (
        None
        if target_power_raw is None
        else _finite(target_power_raw, label="target efficacy power")
    )
    achieved_power = None
    if target_power is not None:
        achieved_power = min(
            _efficacy_power(
                control_sample_size=control_count,
                treatment_sample_size=count,
                control_risk=_finite(
                    rule.get("planning_control_risk"), label="planning control risk"
                ),
                treatment_risk=_finite(
                    rule.get("planning_treatment_risk"), label="planning treatment risk"
                ),
                alternative_benefit=_finite(
                    rule.get("planning_alternative_benefit"),
                    label="planning alternative benefit",
                ),
                confidence_level=confidence_level,
            )
            for count in treatment_counts
        )
        if achieved_power + 1e-12 < target_power:
            failures.append("insufficient_efficacy_power")
    return IndependentPhaseDesign(
        adequate=not failures,
        failures=tuple(failures),
        achieved_power=achieved_power,
        target_power=target_power,
        achieved_safety_absolute_risk_power=safety_absolute_power,
        achieved_safety_excess_risk_power=safety_excess_power,
        target_safety_decision_power=target_safety,
    )


__all__ = ["IndependentPhaseDesign", "reconstruct_phase_design"]
