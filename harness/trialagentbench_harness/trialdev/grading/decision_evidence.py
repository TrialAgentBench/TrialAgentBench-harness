"""Derive TrialDev phase-action sets from participant-visible trial outputs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pandas as pd
from scipy.stats import beta, binom

from trialagentbench_harness.numeric_policy import (
    FLOAT_EQUALITY_ABS_TOLERANCE_V1,
    MINIMUM_TWO_SIDED_CONFIDENCE_LEVEL_V1,
)
from trialagentbench_harness.trialdev.grading.analysis_evidence import aalen_johansen_cif_variance_v1
from trialagentbench_harness.trialdev.grading.io import read_json
from trialagentbench_harness.trialdev.grading.statistics import planned_arm_allocation_v1
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentRequestV1
from trialagentbench_harness.trialdev.share.public_method_design import TrialDevPhaseDesignPolicyV1
from trialagentbench_harness.trialdev.share.validate import (
    candidate_ids_by_role_v1,
    validate_request_against_scenario_v1,
)

__all__ = [
    "TrialDevCandidateDecisionWitnessV1",
    "TrialDevPhaseDecisionWitnessV1",
    "TrialDevPhaseDesignWitnessV1",
    "TrialDevSeriousSafetySurfaceV1",
    "derive_phase_decision_witness_v1",
    "derive_phase_design_witness_v1",
    "declared_serious_safety_surfaces_v1",
    "risk_difference_power_v1",
    "safety_decision_power_v1",
]


@dataclass(frozen=True)
class TrialDevSeriousSafetySurfaceV1:
    """Exact released columns for one serious-event endpoint."""

    endpoint_id: str
    event_column: str
    time_column: str
    seriousness_column: str


def declared_serious_safety_surfaces_v1(*, scenario_root: Path) -> tuple[TrialDevSeriousSafetySurfaceV1, ...]:
    """Return exact serious-event variable triplets from the public safety policy."""

    payload = read_json(Path(scenario_root) / "public" / "safety_decision_policy.json")
    definitions = payload.get("serious_event_definitions") if isinstance(payload, dict) else None
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("safety_decision_policy.json requires serious_event_definitions.")
    surfaces: list[TrialDevSeriousSafetySurfaceV1] = []
    for definition in definitions:
        if not isinstance(definition, dict):
            raise ValueError("Serious-event definitions must be objects.")
        values = tuple(
            str(definition.get(field) or "")
            for field in ("endpoint_id", "event_column", "time_column", "seriousness_column")
        )
        if any(not value for value in values):
            raise ValueError("Serious-event definitions require endpoint, event, time, and seriousness identities.")
        surfaces.append(
            TrialDevSeriousSafetySurfaceV1(
                endpoint_id=values[0],
                event_column=values[1],
                time_column=values[2],
                seriousness_column=values[3],
            )
        )
    identities = tuple(surface.endpoint_id for surface in surfaces)
    columns = tuple(
        column
        for surface in surfaces
        for column in (surface.event_column, surface.time_column, surface.seriousness_column)
    )
    if len(set(identities)) != len(identities) or len(set(columns)) != len(columns):
        raise ValueError("Serious-event definitions require unique endpoint and column identities.")
    return tuple(surfaces)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrialDevCandidateDecisionWitnessV1:
    """Evidence-derived actions for one investigational candidate."""

    candidate_drug_id: str
    arm_id: str
    acceptable_action_ids: tuple[str, ...]
    safety_state: str
    safety_action_ids: tuple[str, ...]
    efficacy_action_ids: tuple[str, ...]
    evidence: dict[str, object]


@dataclass(frozen=True)
class TrialDevPhaseDecisionWitnessV1:
    """One evidence-derived acceptable action set for a realized phase."""

    phase_id: str
    acceptable_action_ids: tuple[str, ...]
    recoverability_class: str
    safety_state: str
    safety_action_ids: tuple[str, ...]
    efficacy_action_ids: tuple[str, ...]
    stop_action_ids: tuple[str, ...]
    advance_action_ids: tuple[str, ...]
    candidates: tuple[TrialDevCandidateDecisionWitnessV1, ...]
    evidence: dict[str, object]

    def candidate(self, candidate_drug_id: str) -> TrialDevCandidateDecisionWitnessV1:
        """Return the evidence record for one exact candidate."""

        matches = tuple(row for row in self.candidates if row.candidate_drug_id == candidate_drug_id)
        if len(matches) != 1:
            raise ValueError(f"Decision witness has no unique candidate {candidate_drug_id!r}.")
        return matches[0]

    def action_is_acceptable(self, *, action_id: str, candidate_drug_id: str | None) -> bool:
        """Evaluate one candidate-qualified programme decision."""

        if action_id in self.advance_action_ids:
            if candidate_drug_id is None:
                return False
            return action_id in self.candidate(candidate_drug_id).acceptable_action_ids
        if action_id in self.stop_action_ids:
            return all(action_id in row.acceptable_action_ids for row in self.candidates)
        return False

    def safety_action_is_acceptable(self, *, action_id: str, candidate_drug_id: str | None) -> bool:
        """Evaluate one candidate-qualified action against the safety gate."""

        if action_id in self.advance_action_ids:
            if candidate_drug_id is None:
                return False
            return action_id in self.candidate(candidate_drug_id).safety_action_ids
        if action_id in self.stop_action_ids:
            return all(action_id in row.safety_action_ids for row in self.candidates)
        return False

    def action_requires_uncertainty(self, *, action_id: str, candidate_drug_id: str | None) -> bool:
        """Return whether set-valued evidence is needed to justify an accepted action."""

        if action_id in self.advance_action_ids:
            if candidate_drug_id is None:
                return False
            return len(self.candidate(candidate_drug_id).acceptable_action_ids) > 1
        if action_id in self.stop_action_ids:
            return any(len(row.acceptable_action_ids) > 1 for row in self.candidates)
        return False

    def relevant_action_ids(self, *, candidate_drug_id: str | None) -> tuple[str, ...]:
        """Return action ids applicable to one submitted candidate choice."""

        common_stops = {
            action_id
            for action_id in self.stop_action_ids
            if all(action_id in row.acceptable_action_ids for row in self.candidates)
        }
        candidate_advances = (
            set()
            if candidate_drug_id is None
            else set(self.candidate(candidate_drug_id).acceptable_action_ids) & set(self.advance_action_ids)
        )
        return tuple(sorted(common_stops | candidate_advances))

    @property
    def checksum(self) -> str:
        """Return a stable checksum for the evidence-derived decision reference."""

        payload = {
            "phase_id": self.phase_id,
            "acceptable_action_ids": self.acceptable_action_ids,
            "recoverability_class": self.recoverability_class,
            "safety_state": self.safety_state,
            "safety_action_ids": self.safety_action_ids,
            "efficacy_action_ids": self.efficacy_action_ids,
            "stop_action_ids": self.stop_action_ids,
            "advance_action_ids": self.advance_action_ids,
            "candidates": [
                {
                    "candidate_drug_id": row.candidate_drug_id,
                    "arm_id": row.arm_id,
                    "acceptable_action_ids": row.acceptable_action_ids,
                    "safety_state": row.safety_state,
                    "safety_action_ids": row.safety_action_ids,
                    "efficacy_action_ids": row.efficacy_action_ids,
                    "evidence": row.evidence,
                }
                for row in self.candidates
            ],
            "evidence": self.evidence,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TrialDevPhaseDesignWitnessV1:
    """Public-policy operating characteristics for one materialized design."""

    phase_id: str
    adequate: bool
    achieved_power: float | None
    achieved_safety_absolute_risk_power: float
    achieved_safety_excess_risk_power: float
    target_power: float | None
    target_safety_decision_power: float
    failures: tuple[str, ...]
    evidence: dict[str, object]


def _risk_interval(
    *,
    estimate: float,
    variance: float,
    confidence_level: float,
    primary_event_count: int,
    informative_count: int,
) -> tuple[float, float]:
    if informative_count <= 0 or primary_event_count < 0 or primary_event_count > informative_count:
        raise ValueError("Risk interval requires a valid positive informative count and event count.")
    alpha = 1.0 - confidence_level
    if primary_event_count == 0:
        return 0.0, float(beta.ppf(1.0 - alpha / 2.0, 1, informative_count))
    if primary_event_count == informative_count:
        return float(beta.ppf(alpha / 2.0, informative_count, 1)), 1.0
    z = float(NormalDist().inv_cdf(0.5 + confidence_level / 2.0))
    half_width = z * math.sqrt(max(0.0, variance))
    return max(0.0, estimate - half_width), min(1.0, estimate + half_width)


def _risk_difference_interval(
    *,
    treated_risk: float,
    treated_interval: tuple[float, float],
    control_risk: float,
    control_interval: tuple[float, float],
) -> tuple[float, float]:
    estimate = float(treated_risk - control_risk)
    lower_distance = math.sqrt((treated_risk - treated_interval[0]) ** 2 + (control_interval[1] - control_risk) ** 2)
    upper_distance = math.sqrt((treated_interval[1] - treated_risk) ** 2 + (control_risk - control_interval[0]) ** 2)
    return max(-1.0, estimate - lower_distance), min(1.0, estimate + upper_distance)


def safety_decision_power_v1(
    *,
    sample_size: int,
    decision_limit: float,
    planning_unacceptable_risk: float,
    confidence_level: float,
) -> tuple[float, int]:
    """Return power to place the absolute-risk lower bound above its limit."""

    if sample_size < 2:
        return 0.0, sample_size + 1
    if not MINIMUM_TWO_SIDED_CONFIDENCE_LEVEL_V1 < confidence_level < 1.0:
        raise ValueError("Absolute-risk safety power requires a valid two-sided confidence level.")
    if not 0.0 < decision_limit < planning_unacceptable_risk < 1.0:
        raise ValueError("Absolute-risk safety planning risk must exceed its decision limit within (0, 1).")
    critical_event_count: int | None = None
    for event_count in range(sample_size + 1):
        estimate = float(event_count / sample_size)
        lower, _upper = _risk_interval(
            estimate=estimate,
            variance=float(estimate * (1.0 - estimate) / sample_size),
            confidence_level=confidence_level,
            primary_event_count=event_count,
            informative_count=sample_size,
        )
        if lower > decision_limit:
            critical_event_count = event_count
            break
    if critical_event_count is None:
        return 0.0, sample_size + 1
    power = float(binom.sf(critical_event_count - 1, sample_size, planning_unacceptable_risk))
    return power, critical_event_count


def risk_difference_power_v1(
    *,
    control_sample_size: int,
    treatment_sample_size: int,
    control_risk: float,
    treatment_risk: float,
    alternative_benefit: float,
    confidence_level: float,
) -> float:
    """Return normal-approximation power for a two-sided risk-difference test."""

    if control_sample_size < 2 or treatment_sample_size < 2:
        return 0.0
    if not MINIMUM_TWO_SIDED_CONFIDENCE_LEVEL_V1 < confidence_level < 1.0:
        raise ValueError("Risk-difference power requires a valid two-sided confidence level.")
    if not 0.0 <= treatment_risk <= control_risk <= 1.0:
        raise ValueError("Risk-difference planning risks require treatment <= control within [0, 1].")
    if (
        alternative_benefit < 0.0
        or abs((control_risk - treatment_risk) - alternative_benefit) > FLOAT_EQUALITY_ABS_TOLERANCE_V1
    ):
        raise ValueError("Risk-difference planning benefit must equal control risk minus treatment risk.")
    standard_error = math.sqrt(
        control_risk * (1.0 - control_risk) / control_sample_size
        + treatment_risk * (1.0 - treatment_risk) / treatment_sample_size
    )
    if standard_error <= 0.0:
        raise ValueError("Risk-difference planning variance must be positive.")
    z_alpha = float(NormalDist().inv_cdf(0.5 + confidence_level / 2.0))
    noncentrality = alternative_benefit / standard_error
    normal = NormalDist()
    return float(normal.cdf(noncentrality - z_alpha) + normal.cdf(-noncentrality - z_alpha))


def harmful_risk_difference_power_v1(
    *,
    control_sample_size: int,
    treatment_sample_size: int,
    control_risk: float,
    treatment_risk: float,
    decision_limit: float,
    confidence_level: float,
) -> float:
    """Return power for a lower confidence bound to exceed a harmful risk difference."""

    if control_sample_size < 2 or treatment_sample_size < 2:
        return 0.0
    if not MINIMUM_TWO_SIDED_CONFIDENCE_LEVEL_V1 < confidence_level < 1.0:
        raise ValueError("Safety risk-difference power requires a valid two-sided confidence level.")
    if not 0.0 <= control_risk < treatment_risk <= 1.0:
        raise ValueError("Safety planning risks require control < treatment within [0, 1].")
    alternative_excess = treatment_risk - control_risk
    if not 0.0 < decision_limit < alternative_excess:
        raise ValueError("Safety risk-difference decision limit must be below the planning excess.")
    standard_error = math.sqrt(
        control_risk * (1.0 - control_risk) / control_sample_size
        + treatment_risk * (1.0 - treatment_risk) / treatment_sample_size
    )
    if standard_error <= 0.0:
        raise ValueError("Safety risk-difference planning variance must be positive.")
    z_alpha = float(NormalDist().inv_cdf(0.5 + confidence_level / 2.0))
    return float(NormalDist().cdf((alternative_excess - decision_limit) / standard_error - z_alpha))


def _informative_count(*, frame: pd.DataFrame, horizon: float) -> int:
    complete = (frame["EVENT"] == 1) | (frame["COMPETING_EVENT"] == 1) | (frame["TIME"] >= horizon)
    return int(complete.sum())


def _safety_component_state(
    *,
    absolute_interval: tuple[float, float],
    excess_interval: tuple[float, float],
    absolute_limit: float,
    excess_limit: float,
) -> str:
    if absolute_interval[0] > absolute_limit or excess_interval[0] > excess_limit:
        return "unacceptable"
    if absolute_interval[1] <= absolute_limit and excess_interval[1] <= excess_limit:
        return "acceptable"
    return "indeterminate"


def _sensitivity_limits(threshold: dict[str, Any]) -> dict[str, tuple[float, float]]:
    absolute = threshold.get("sensitivity_max_absolute_rates")
    excess = threshold.get("sensitivity_max_excess_vs_control")
    if not isinstance(absolute, dict) or not isinstance(excess, dict):
        raise ValueError("safety threshold requires explicit absolute and excess sensitivity limits.")
    profiles = ("strict", "primary", "permissive")
    if set(absolute) != set(profiles) or set(excess) != set(profiles):
        raise ValueError("safety sensitivity limits must define strict, primary, and permissive profiles.")
    limits = {profile: (float(absolute[profile]), float(excess[profile])) for profile in profiles}
    if not all(0.0 < pair[0] < 1.0 and 0.0 < pair[1] < 1.0 for pair in limits.values()):
        raise ValueError("safety sensitivity limits must lie in (0, 1).")
    if not all(
        limits[left][index] < limits[right][index]
        for left, right in (("strict", "primary"), ("primary", "permissive"))
        for index in (0, 1)
    ):
        raise ValueError("safety sensitivity limits must increase from strict to permissive.")
    return limits


def _efficacy_interval(
    *,
    endpoints: pd.DataFrame,
    control_arm: str,
    treatment_arm: str,
    event_column: str,
    time_column: str,
    horizon: float,
    confidence_level: float,
) -> tuple[float, float, float]:
    control = endpoints.loc[endpoints["ARM"].astype(str) == control_arm]
    treatment = endpoints.loc[endpoints["ARM"].astype(str) == treatment_arm]
    if event_column != "EVENT" or time_column != "TIME":
        raise ValueError("Phase efficacy policy must reference the canonical EVENT and TIME columns.")
    control_risk, control_variance = aalen_johansen_cif_variance_v1(frame=control, horizon=horizon)
    treatment_risk, treatment_variance = aalen_johansen_cif_variance_v1(frame=treatment, horizon=horizon)
    estimate = float(control_risk - treatment_risk)
    standard_error = math.sqrt(max(0.0, control_variance + treatment_variance))
    z = float(NormalDist().inv_cdf(0.5 + confidence_level / 2.0))
    return estimate, max(-1.0, estimate - z * standard_error), min(1.0, estimate + z * standard_error)


def _action_partition(
    *, scenario_root: Path, phase_id: str
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    payload = read_json(Path(scenario_root) / "public" / "phase_action_policy.json")
    specs = payload.get("action_specs") if isinstance(payload, dict) else None
    if not isinstance(specs, list):
        raise ValueError("phase_action_policy.action_specs must be an array.")
    for spec in specs:
        if isinstance(spec, dict) and str(spec.get("phase_id")) == phase_id:
            actions = spec.get("allowed_action_ids")
            stop = spec.get("stop_action_ids")
            advance = spec.get("advance_action_ids")
            if (
                not isinstance(actions, list)
                or not actions
                or not isinstance(stop, list)
                or not isinstance(advance, list)
            ):
                raise ValueError(f"phase {phase_id!r} has no complete action partition.")
            action_ids = tuple(str(action) for action in actions)
            stop_ids = tuple(str(action) for action in stop)
            advance_ids = tuple(str(action) for action in advance)
            if set(stop_ids) & set(advance_ids) or set(stop_ids) | set(advance_ids) != set(action_ids):
                raise ValueError(f"phase {phase_id!r} action classes do not partition allowed actions.")
            return action_ids, stop_ids, advance_ids
    raise ValueError(f"phase action policy has no entry for {phase_id!r}.")


def _decision_rule(*, scenario_root: Path, phase_id: str) -> tuple[dict[str, Any], float]:
    payload = read_json(Path(scenario_root) / "public" / "phase_decision_evidence_policy.json")
    if payload.get("schema_id") != "trialdev_phase_decision_evidence_policy_v1":
        raise ValueError("Unsupported phase decision evidence policy schema.")
    confidence_value = payload.get("confidence_level")
    if isinstance(confidence_value, bool) or not isinstance(confidence_value, int | float):
        raise ValueError("phase decision policy requires a numeric confidence_level.")
    confidence_level = float(confidence_value)
    if not MINIMUM_TWO_SIDED_CONFIDENCE_LEVEL_V1 < confidence_level < 1.0:
        raise ValueError("phase decision policy requires a valid two-sided confidence level.")
    rules = payload.get("phase_rules")
    if not isinstance(rules, list):
        raise ValueError("phase decision evidence policy requires phase_rules.")
    for rule in rules:
        if isinstance(rule, dict) and str(rule.get("phase_id")) == phase_id:
            return rule, confidence_level
    raise ValueError(f"phase decision evidence policy has no rule for {phase_id!r}.")


def derive_phase_design_witness_v1(
    *,
    scenario_root: Path,
    request: TrialDevelopmentRequestV1,
    phase_id: str,
    trial_output_root: Path | None = None,
) -> TrialDevPhaseDesignWitnessV1:
    """Evaluate prospective design adequacy from the submitted request."""

    phase = str(phase_id)
    validate_request_against_scenario_v1(
        scenario_root=Path(scenario_root),
        request=request,
    )
    policy = TrialDevPhaseDesignPolicyV1.model_validate(
        read_json(Path(scenario_root) / "public" / "phase_design_policy.json")
    )
    rule = policy.rule_for_phase(phase)
    if str(request.phase_id) != phase:
        raise ValueError("Submitted request phase does not match the graded phase.")
    control_drug_id = candidate_ids_by_role_v1(scenario_root=Path(scenario_root))["control"][0]
    drug_id_by_arm, arm_counts = planned_arm_allocation_v1(
        request=request,
        control_drug_id=control_drug_id,
    )
    control_arms = tuple(arm for arm, drug in drug_id_by_arm.items() if drug == control_drug_id)
    treatment_arms = tuple(arm for arm, drug in drug_id_by_arm.items() if drug != control_drug_id)
    if len(control_arms) != 1 or not treatment_arms:
        raise ValueError("Planned allocation requires one control and at least one treatment arm.")
    control_arm = control_arms[0]
    control_count = arm_counts[control_arm]
    treatment_counts = {arm: arm_counts[arm] for arm in treatment_arms}
    information_fraction_by_drug = rule.planning_information_fraction_by_drug_id

    if trial_output_root is not None:
        output_root = Path(trial_output_root)
        metadata_path = output_root / "trial_metadata.json"
        request_path = output_root / "request.json"
        if metadata_path.exists() and not request_path.exists():
            raise ValueError("Materialized trial metadata requires the corresponding request.json.")
        evidence_mode = read_json(metadata_path).get("evidence_mode") if metadata_path.exists() else None
        if evidence_mode != "fixed_trajectory" and request_path.exists():
            materialized_request = TrialDevelopmentRequestV1.model_validate(read_json(request_path))
            materialization_fields = (
                "scenario_id",
                "phase_id",
                "candidate_drug_ids",
                "target_sample_size",
                "endpoint_id",
                "follow_up_days",
                "enrollment_window_days",
                "site_count_budget",
                "allocation_ratio",
                "allocation_weights",
                "design_cell_id",
                "treatment_discontinuation_strategy",
                "interim_policy",
                "site_strategy",
                "stratification_variables",
                "analysis_covariates",
                "subgroup_variables",
            )
            if any(
                getattr(materialized_request, field) != getattr(request, field) for field in materialization_fields
            ):
                raise ValueError("Materialized trial differs from the submitted design-defining fields.")
            safety = pd.read_parquet(output_root / "safety.parquet")
            realized_counts = {
                str(arm): int(count) for arm, count in safety["ARM"].astype(str).value_counts().to_dict().items()
            }
            if realized_counts != arm_counts:
                raise ValueError("Materialized arm counts differ from the deterministic prospective allocation.")

    def effective_count(arm_id: str, enrolled_count: int) -> int:
        drug_id = str(drug_id_by_arm.get(arm_id) or "")
        if not drug_id or drug_id not in information_fraction_by_drug:
            raise ValueError(f"Design policy lacks an information fraction for arm {arm_id!r}.")
        fraction = float(information_fraction_by_drug[drug_id])
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"Design information fraction is invalid for drug {drug_id!r}.")
        return int(math.floor(enrolled_count * fraction))

    effective_control_count = effective_count(control_arm, control_count)
    effective_treatment_counts = {arm: effective_count(arm, count) for arm, count in treatment_counts.items()}

    failures: list[str] = []
    if request.design_cell_id != rule.design_cell_id:
        failures.append("unaccepted_design_cell")
    if str(request.interim_policy) != rule.supported_interim_policy:
        failures.append("unsupported_interim_policy")
    if request.follow_up_days is None:
        raise ValueError("Materialized randomized request lacks follow_up_days.")
    if int(request.follow_up_days) < rule.evaluation_horizon_days:
        failures.append("insufficient_follow_up")
    expected_endpoint = rule.primary_endpoint_id
    if expected_endpoint is None:
        if request.endpoint_id is not None:
            failures.append("unexpected_primary_endpoint")
    elif str(request.endpoint_id or "") != str(expected_endpoint):
        failures.append("primary_endpoint_mismatch")

    min_treatment_count = min(effective_treatment_counts.values())
    confidence_level = policy.confidence_level
    safety_absolute_power, critical_safety_event_count = safety_decision_power_v1(
        sample_size=min_treatment_count,
        decision_limit=rule.serious_ae_unacceptable_absolute_risk,
        planning_unacceptable_risk=rule.planning_safety_absolute_treatment_risk,
        confidence_level=confidence_level,
    )
    safety_excess_power = min(
        harmful_risk_difference_power_v1(
            control_sample_size=effective_control_count,
            treatment_sample_size=treatment_count,
            control_risk=rule.planning_safety_control_risk,
            treatment_risk=rule.planning_safety_excess_treatment_risk,
            decision_limit=rule.serious_ae_unacceptable_excess_risk,
            confidence_level=confidence_level,
        )
        for treatment_count in effective_treatment_counts.values()
    )
    target_safety_decision_power = rule.target_safety_decision_power
    if (
        min(safety_absolute_power, safety_excess_power) + FLOAT_EQUALITY_ABS_TOLERANCE_V1
        < target_safety_decision_power
    ):
        failures.append("insufficient_safety_decision_power")

    achieved_power: float | None = None
    target_power = rule.target_power
    if target_power is not None:
        if (
            rule.planning_control_risk is None
            or rule.planning_treatment_risk is None
            or rule.planning_alternative_benefit is None
        ):
            raise ValueError("Efficacy design cell lacks required planning values.")
        control_risk = rule.planning_control_risk
        treatment_risk = rule.planning_treatment_risk
        alternative = rule.planning_alternative_benefit
        powers = []
        for treatment_count in effective_treatment_counts.values():
            powers.append(
                risk_difference_power_v1(
                    control_sample_size=effective_control_count,
                    treatment_sample_size=treatment_count,
                    control_risk=control_risk,
                    treatment_risk=treatment_risk,
                    alternative_benefit=alternative,
                    confidence_level=confidence_level,
                )
            )
        achieved_power = min(powers)
        if achieved_power + FLOAT_EQUALITY_ABS_TOLERANCE_V1 < target_power:
            failures.append("insufficient_efficacy_power")

    evidence: dict[str, object] = {
        "control_arm_count": control_count,
        "treatment_arm_counts": treatment_counts,
        "effective_control_arm_count": effective_control_count,
        "effective_treatment_arm_counts": effective_treatment_counts,
        "planning_information_fraction_by_drug_id": information_fraction_by_drug,
        "design_cell_id": rule.design_cell_id,
        "planned_arm_counts": arm_counts,
        "planning_information_estimator_id": rule.planning_information_estimator_id,
        "primary_endpoint_id": expected_endpoint,
        "evaluation_horizon_days": rule.evaluation_horizon_days,
        "supported_interim_policy": rule.supported_interim_policy,
        "planning_control_risk": rule.planning_control_risk,
        "planning_treatment_risk": rule.planning_treatment_risk,
        "planning_alternative_benefit": rule.planning_alternative_benefit,
        "serious_ae_unacceptable_absolute_risk": rule.serious_ae_unacceptable_absolute_risk,
        "serious_ae_unacceptable_excess_risk": rule.serious_ae_unacceptable_excess_risk,
        "planning_safety_control_risk": rule.planning_safety_control_risk,
        "planning_safety_absolute_treatment_risk": rule.planning_safety_absolute_treatment_risk,
        "planning_safety_excess_risk": rule.planning_safety_excess_risk,
        "planning_safety_excess_treatment_risk": rule.planning_safety_excess_treatment_risk,
        "critical_safety_event_count": critical_safety_event_count,
        "planning_safety_estimator_id": rule.planning_safety_estimator_id,
        "planning_safety_analysis_population": rule.planning_safety_analysis_population,
        "planning_safety_control_support_count": rule.planning_safety_control_support_count,
        "planning_safety_min_observed_propensity": rule.planning_safety_min_observed_propensity,
        "planning_safety_max_inverse_propensity_weight": rule.planning_safety_max_inverse_propensity_weight,
        "planning_safety_weighted_effective_sample_size": rule.planning_safety_weighted_effective_sample_size,
        "planning_estimator_id": rule.planning_estimator_id,
        "planning_analysis_population": rule.planning_analysis_population,
        "planning_control_support_count": rule.planning_control_support_count,
        "planning_min_observed_propensity": rule.planning_min_observed_propensity,
        "planning_max_inverse_propensity_weight": rule.planning_max_inverse_propensity_weight,
        "planning_weighted_effective_sample_size": rule.planning_weighted_effective_sample_size,
        "policy_checksum": policy.checksum,
    }
    return TrialDevPhaseDesignWitnessV1(
        phase_id=phase,
        adequate=not failures,
        achieved_power=achieved_power,
        achieved_safety_absolute_risk_power=safety_absolute_power,
        achieved_safety_excess_risk_power=safety_excess_power,
        target_power=target_power,
        target_safety_decision_power=target_safety_decision_power,
        failures=tuple(failures),
        evidence=evidence,
    )


def _arm_ids(
    *, trial_output_root: Path, requested_candidate_drug_ids: tuple[str, ...]
) -> tuple[str, tuple[tuple[str, str], ...]]:
    mapping = read_json(Path(trial_output_root) / "arm_mapping.json")
    control = str(mapping.get("control_arm_id") or "")
    candidates = mapping.get("candidate_arm_ids")
    drug_by_arm = mapping.get("drug_id_by_arm")
    if not control or not isinstance(candidates, list) or not candidates or not isinstance(drug_by_arm, dict):
        raise ValueError("Phase decision evidence requires one control and at least one candidate arm.")
    candidate_arms = tuple(str(candidate) for candidate in candidates)
    if any(not candidate for candidate in candidate_arms) or len(set(candidate_arms)) != len(candidate_arms):
        raise ValueError("Phase decision evidence requires unique, non-empty candidate arm identifiers.")
    if control in set(candidate_arms):
        raise ValueError("Control arm cannot also be a candidate arm.")
    candidate_pairs = tuple((arm_id, str(drug_by_arm.get(arm_id) or "")) for arm_id in candidate_arms)
    candidate_drug_ids = tuple(drug_id for _arm_id, drug_id in candidate_pairs)
    if any(not drug_id for drug_id in candidate_drug_ids) or len(set(candidate_drug_ids)) != len(candidate_drug_ids):
        raise ValueError("Candidate arms require unique, non-empty drug identities.")
    if set(candidate_drug_ids) != set(requested_candidate_drug_ids):
        raise ValueError("Candidate arm mapping must exactly cover the candidates in the materialized request.")
    return control, candidate_pairs


def _binary_series(*, frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"Safety surface is missing required column: {column!r}.")
    values = pd.to_numeric(frame[column], errors="raise").astype(float)
    if bool(values.isna().any()) or not bool(values.isin((0.0, 1.0)).all()):
        raise ValueError(f"Safety indicator must be complete and binary: {column!r}.")
    return values.astype(int)


def _time_series(*, frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"Safety surface is missing required column: {column!r}.")
    values = pd.to_numeric(frame[column], errors="raise").astype(float)
    if bool(values.isna().any()) or bool((values < 0.0).any()) or not values.map(math.isfinite).all():
        raise ValueError(f"Safety time must be complete, finite, and non-negative: {column!r}.")
    return values


def _competing_risk_frame(
    *,
    safety: pd.DataFrame,
    endpoint_event: pd.Series,
    endpoint_time: pd.Series,
    horizon: float,
) -> pd.DataFrame:
    if safety.empty or len(endpoint_event) != len(safety) or len(endpoint_time) != len(safety):
        raise ValueError("Safety competing-risk inputs must be non-empty and row aligned.")
    terminal_event = _binary_series(frame=safety, column="TERMINAL_EVENT")
    terminal_time = _time_series(frame=safety, column="TERMINAL_TIME")
    ltfu_event = _binary_series(frame=safety, column="LTFU_E")
    ltfu_time = _time_series(frame=safety, column="LTFU_T")
    if bool(((terminal_event == 1) & (ltfu_event == 1)).any()):
        raise ValueError("Observed terminal-endpoint and LTFU events must be mutually exclusive.")
    endpoint = pd.to_numeric(endpoint_event, errors="raise").astype(float)
    time = pd.to_numeric(endpoint_time, errors="raise").astype(float)
    if bool(endpoint.isna().any()) or not bool(endpoint.isin((0.0, 1.0)).all()):
        raise ValueError("Safety endpoint indicator must be complete and binary.")
    if bool(time.isna().any()) or bool((time < 0.0).any()) or not time.map(math.isfinite).all():
        raise ValueError("Safety endpoint time must be complete, finite, and non-negative.")

    infinity = pd.Series(math.inf, index=safety.index, dtype=float)
    endpoint_active = time.where(endpoint.eq(1.0), infinity)
    terminal_active = terminal_time.where(terminal_event.eq(1), infinity)
    ltfu_active = ltfu_time.where(ltfu_event.eq(1), infinity)
    eps = 1e-9
    post_censor_event = endpoint.eq(1.0) & (
        ((ltfu_event == 1) & time.gt(ltfu_time + eps)) | ((terminal_event == 1) & time.ge(terminal_time - eps))
    )
    if bool(post_censor_event.any()):
        raise ValueError("Safety surface declares an endpoint event after LTFU or a terminal event.")
    event_observed = (
        endpoint_active.le(float(horizon) + eps)
        & endpoint_active.le(ltfu_active + eps)
        & endpoint_active.lt(terminal_active - eps)
    )
    competing_observed = (
        terminal_active.le(float(horizon) + eps)
        & terminal_active.le(ltfu_active + eps)
        & terminal_active.le(endpoint_active + eps)
    )
    if bool((event_observed & competing_observed).any()):
        raise ValueError("Safety event and terminal competing event must be mutually exclusive.")
    observed_time = pd.concat(
        (endpoint_active, terminal_active, ltfu_active, pd.Series(float(horizon), index=safety.index)),
        axis=1,
    ).min(axis=1)
    return pd.DataFrame(
        {
            "TIME": observed_time.astype(float),
            "EVENT": event_observed.astype(int),
            "COMPETING_EVENT": competing_observed.astype(int),
        },
        index=safety.index,
    )


def _serious_event_surface(
    safety: pd.DataFrame,
    *,
    serious_surfaces: tuple[TrialDevSeriousSafetySurfaceV1, ...],
    evaluation_horizon_days: float,
) -> pd.DataFrame:
    event_times: list[pd.Series] = []
    for surface in serious_surfaces:
        serious = pd.to_numeric(safety[surface.seriousness_column], errors="raise").astype(float)
        missing = serious.isna()
        event_column = surface.event_column
        time_column = surface.time_column
        if event_column not in safety.columns or time_column not in safety.columns:
            raise ValueError(f"Horizon-specific safety evaluation requires {event_column} and {time_column}.")
        event = _binary_series(frame=safety, column=event_column).astype(float)
        if bool((missing & event.eq(1.0)).any()):
            raise ValueError(f"Seriousness is missing for an observed safety event: {surface.seriousness_column}")
        if bool((serious.eq(1.0) & event.eq(0.0)).any()):
            raise ValueError(
                f"Seriousness contradicts the paired safety-event indicator: {surface.seriousness_column}"
            )
        serious = serious.mask(missing & event.eq(0.0), 0.0)
        event_time = _time_series(frame=safety, column=time_column)
        serious = serious.where(event.eq(1.0) & event_time.le(evaluation_horizon_days), 0.0)
        if bool(serious.isna().any()) or not bool(serious.isin((0.0, 1.0)).all()):
            raise ValueError(
                f"Serious-event field must resolve to complete binary values: {surface.seriousness_column}"
            )
        event_times.append(event_time.where(serious.eq(1.0), math.inf))
    earliest = pd.concat(event_times, axis=1).min(axis=1)
    return _competing_risk_frame(
        safety=safety,
        endpoint_event=earliest.map(math.isfinite).astype(int),
        endpoint_time=earliest.where(earliest.map(math.isfinite), float(evaluation_horizon_days)),
        horizon=evaluation_horizon_days,
    )


def _efficacy_actions(
    *,
    phase_id: str,
    endpoints: pd.DataFrame,
    control_arm: str,
    treatment_arm: str,
    rule: dict[str, Any],
    horizon: float,
    confidence_level: float,
    stop_actions: tuple[str, ...],
    advance_actions: tuple[str, ...],
) -> tuple[tuple[str, ...], dict[str, object]]:
    minimum_benefit = float(rule["minimum_benefit"])
    estimate, lower, upper = _efficacy_interval(
        endpoints=endpoints,
        control_arm=control_arm,
        treatment_arm=treatment_arm,
        event_column=str(rule["efficacy_endpoint_column"]),
        time_column=str(rule["time_column"]),
        horizon=horizon,
        confidence_level=confidence_level,
    )
    evidence: dict[str, object] = {
        "evaluated": True,
        "risk_difference_control_minus_treatment": estimate,
        "confidence_interval": (lower, upper),
        "minimum_benefit": minimum_benefit,
        "follow_up_days": horizon,
    }
    failure_actions: tuple[str, ...]
    indeterminate_actions: tuple[str, ...]
    if phase_id == "phase3":
        failure_actions = tuple(action for action in stop_actions if action == "declare_failure")
        inconclusive_actions = tuple(action for action in stop_actions if action == "declare_inconclusive")
        if len(failure_actions) != 1 or len(inconclusive_actions) != 1:
            raise ValueError("Phase-3 policy requires one failure and one inconclusive action.")
        indeterminate_actions = inconclusive_actions
    else:
        failure_actions = stop_actions
        indeterminate_actions = tuple(sorted(set(stop_actions) | set(advance_actions)))
    sensitivity_sets: dict[str, list[str]] = {}
    sensitivity_margins = rule.get("sensitivity_minimum_benefits", [])
    if not isinstance(sensitivity_margins, list) or not sensitivity_margins:
        raise ValueError("phase efficacy rule requires sensitivity_minimum_benefits.")
    for margin_value in sensitivity_margins:
        margin = float(margin_value)
        if lower >= margin:
            margin_actions = advance_actions
        elif upper < margin:
            margin_actions = failure_actions
        else:
            margin_actions = indeterminate_actions
        sensitivity_sets[f"{margin:.6f}"] = sorted(margin_actions)
    evidence["margin_sensitivity_action_sets"] = sensitivity_sets
    if lower >= minimum_benefit:
        efficacy_actions = advance_actions
    elif upper < minimum_benefit:
        efficacy_actions = failure_actions
    else:
        efficacy_actions = indeterminate_actions
    return efficacy_actions, evidence


def _safety_state(
    *,
    scenario_root: Path,
    trial_output_root: Path,
    phase_id: str,
    control_arm: str,
    treatment_arm: str,
    confidence_level: float,
    request_follow_up_days: float,
) -> tuple[str, dict[str, object]]:
    safety = pd.read_parquet(Path(trial_output_root) / "safety.parquet")
    policy = read_json(Path(scenario_root) / "public" / "safety_decision_policy.json")
    thresholds = policy.get("thresholds") if isinstance(policy, dict) else None
    if not isinstance(thresholds, list):
        raise ValueError("safety decision policy requires thresholds.")
    phase_thresholds = [
        threshold
        for threshold in thresholds
        if isinstance(threshold, dict) and str(threshold.get("phase_id")) == phase_id
    ]
    serious_thresholds = [
        threshold for threshold in phase_thresholds if str(threshold.get("component_id")) == "serious_ae"
    ]
    if len(serious_thresholds) != 1 or str(serious_thresholds[0].get("role")) != "hard_gate":
        raise ValueError(f"phase {phase_id!r} requires exactly one serious-AE hard gate.")
    threshold = serious_thresholds[0]
    evaluation_horizon = float(threshold["evaluation_horizon_days"])
    if request_follow_up_days < evaluation_horizon:
        raise ValueError(
            f"phase {phase_id!r} follow-up is shorter than its safety evaluation horizon: "
            f"{request_follow_up_days} < {evaluation_horizon}."
        )
    if any(float(item.get("evaluation_horizon_days", -1.0)) != evaluation_horizon for item in phase_thresholds):
        raise ValueError(f"phase {phase_id!r} safety thresholds must share one evaluation horizon.")
    serious_surfaces = declared_serious_safety_surfaces_v1(scenario_root=Path(scenario_root))
    declared_columns = {
        column
        for surface in serious_surfaces
        for column in (surface.event_column, surface.time_column, surface.seriousness_column)
    }
    missing_serious = tuple(sorted(declared_columns - set(str(column) for column in safety.columns)))
    if missing_serious:
        raise ValueError(f"Safety table lacks declared serious-event columns: {missing_serious!r}.")
    serious_surface = _serious_event_surface(
        safety,
        serious_surfaces=serious_surfaces,
        evaluation_horizon_days=evaluation_horizon,
    )
    arm = safety["ARM"].astype(str)
    treated_total = int((arm == treatment_arm).sum())
    control_total = int((arm == control_arm).sum())
    if treated_total <= 0 or control_total <= 0:
        raise ValueError("Safety evidence requires non-empty treatment and control arms.")
    treated_serious = serious_surface.loc[arm == treatment_arm]
    control_serious = serious_surface.loc[arm == control_arm]
    treated_events = int(treated_serious["EVENT"].sum())
    control_events = int(control_serious["EVENT"].sum())
    treated_risk, treated_variance = aalen_johansen_cif_variance_v1(
        frame=treated_serious,
        horizon=evaluation_horizon,
    )
    control_risk, control_variance = aalen_johansen_cif_variance_v1(
        frame=control_serious,
        horizon=evaluation_horizon,
    )
    treated_interval = _risk_interval(
        estimate=treated_risk,
        variance=treated_variance,
        confidence_level=confidence_level,
        primary_event_count=treated_events,
        informative_count=_informative_count(frame=treated_serious, horizon=evaluation_horizon),
    )
    control_interval = _risk_interval(
        estimate=control_risk,
        variance=control_variance,
        confidence_level=confidence_level,
        primary_event_count=control_events,
        informative_count=_informative_count(frame=control_serious, horizon=evaluation_horizon),
    )
    excess_interval = _risk_difference_interval(
        treated_risk=treated_risk,
        treated_interval=treated_interval,
        control_risk=control_risk,
        control_interval=control_interval,
    )
    absolute_limit = float(threshold["max_absolute_rate"])
    excess_limit = float(threshold["max_excess_vs_control"])
    serious_limits = _sensitivity_limits(threshold)
    if serious_limits["primary"] != (absolute_limit, excess_limit):
        raise ValueError("primary serious-AE sensitivity limits must equal the primary limits.")
    sensitivity_states = {
        profile: _safety_component_state(
            absolute_interval=treated_interval,
            excess_interval=excess_interval,
            absolute_limit=limits[0],
            excess_limit=limits[1],
        )
        for profile, limits in serious_limits.items()
    }
    state = sensitivity_states["primary"]
    evidence: dict[str, object] = {
        "treated_serious_events": treated_events,
        "treated_total": treated_total,
        "control_serious_events": control_events,
        "control_total": control_total,
        "estimator_id": "aalen_johansen_cumulative_incidence",
        "treated_serious_rate": treated_risk,
        "control_serious_rate": control_risk,
        "control_serious_rate_interval": control_interval,
        "treated_serious_rate_interval": treated_interval,
        "serious_rate_excess": float(treated_risk - control_risk),
        "serious_rate_excess_interval": excess_interval,
        "absolute_limit": absolute_limit,
        "excess_limit": excess_limit,
        "evaluation_horizon_days": evaluation_horizon,
    }
    discontinuation_thresholds = [
        threshold for threshold in phase_thresholds if str(threshold.get("component_id")) == "discontinuation"
    ]
    if len(discontinuation_thresholds) > 1:
        raise ValueError(f"phase {phase_id!r} has duplicate discontinuation thresholds.")
    if discontinuation_thresholds:
        discontinuation = _binary_series(frame=safety, column="DISCONTINUATION_E")
        discontinuation_time = _time_series(frame=safety, column="DISCONTINUATION_T")
        discontinuation_surface = _competing_risk_frame(
            safety=safety,
            endpoint_event=discontinuation,
            endpoint_time=discontinuation_time,
            horizon=evaluation_horizon,
        )
        treated_disc = discontinuation_surface.loc[arm == treatment_arm]
        control_disc = discontinuation_surface.loc[arm == control_arm]
        treated_discontinuations = int(treated_disc["EVENT"].sum())
        control_discontinuations = int(control_disc["EVENT"].sum())
        treated_disc_risk, treated_disc_variance = aalen_johansen_cif_variance_v1(
            frame=treated_disc,
            horizon=evaluation_horizon,
        )
        control_disc_risk, control_disc_variance = aalen_johansen_cif_variance_v1(
            frame=control_disc,
            horizon=evaluation_horizon,
        )
        treated_disc_interval = _risk_interval(
            estimate=treated_disc_risk,
            variance=treated_disc_variance,
            confidence_level=confidence_level,
            primary_event_count=treated_discontinuations,
            informative_count=_informative_count(frame=treated_disc, horizon=evaluation_horizon),
        )
        control_disc_interval = _risk_interval(
            estimate=control_disc_risk,
            variance=control_disc_variance,
            confidence_level=confidence_level,
            primary_event_count=control_discontinuations,
            informative_count=_informative_count(frame=control_disc, horizon=evaluation_horizon),
        )
        disc_excess_interval = _risk_difference_interval(
            treated_risk=treated_disc_risk,
            treated_interval=treated_disc_interval,
            control_risk=control_disc_risk,
            control_interval=control_disc_interval,
        )
        disc_threshold = discontinuation_thresholds[0]
        evidence["discontinuation"] = {
            "role": str(disc_threshold.get("role")),
            "treated_events": treated_discontinuations,
            "treated_total": treated_total,
            "control_events": control_discontinuations,
            "control_total": control_total,
            "estimator_id": "aalen_johansen_cumulative_incidence",
            "treated_rate": treated_disc_risk,
            "control_rate": control_disc_risk,
            "control_rate_interval": control_disc_interval,
            "treated_rate_interval": treated_disc_interval,
            "rate_excess": float(treated_disc_risk - control_disc_risk),
            "rate_excess_interval": disc_excess_interval,
            "absolute_limit": float(disc_threshold["max_absolute_rate"]),
            "excess_limit": float(disc_threshold["max_excess_vs_control"]),
        }
        if str(disc_threshold.get("role")) == "hard_gate":
            disc_limits = _sensitivity_limits(disc_threshold)
            if disc_limits["primary"] != (
                float(disc_threshold["max_absolute_rate"]),
                float(disc_threshold["max_excess_vs_control"]),
            ):
                raise ValueError("primary discontinuation sensitivity limits must equal the primary limits.")
            for profile, limits in disc_limits.items():
                disc_state = _safety_component_state(
                    absolute_interval=treated_disc_interval,
                    excess_interval=disc_excess_interval,
                    absolute_limit=limits[0],
                    excess_limit=limits[1],
                )
                if "unacceptable" in {sensitivity_states[profile], disc_state}:
                    sensitivity_states[profile] = "unacceptable"
                elif "indeterminate" in {sensitivity_states[profile], disc_state}:
                    sensitivity_states[profile] = "indeterminate"
                else:
                    sensitivity_states[profile] = "acceptable"
            state = sensitivity_states["primary"]
    evidence["sensitivity_states"] = sensitivity_states
    return state, evidence


def derive_phase_decision_witness_v1(
    *,
    scenario_root: Path,
    trial_output_root: Path,
    phase_id: str,
) -> TrialDevPhaseDecisionWitnessV1:
    """Derive acceptable actions from one participant-visible trial output."""

    phase = str(phase_id)
    if phase not in {"phase1", "phase2", "phase3"}:
        raise ValueError(f"Unsupported materialized phase: {phase!r}.")
    actions, stop_actions, advance_actions = _action_partition(scenario_root=Path(scenario_root), phase_id=phase)
    rule, confidence_level = _decision_rule(scenario_root=Path(scenario_root), phase_id=phase)
    request_payload = read_json(Path(trial_output_root) / "request.json")
    if not isinstance(request_payload, dict):
        raise ValueError("Materialized request must be a JSON object.")
    request = TrialDevelopmentRequestV1.model_validate(request_payload)
    if str(request.phase_id) != phase:
        raise ValueError("Materialized request phase does not match the decision phase.")
    if request.follow_up_days is None:
        raise ValueError("Materialized randomized request lacks follow_up_days.")
    control_arm, candidate_arms = _arm_ids(
        trial_output_root=Path(trial_output_root),
        requested_candidate_drug_ids=tuple(str(value) for value in request.candidate_drug_ids),
    )
    endpoints = None if phase == "phase1" else pd.read_parquet(Path(trial_output_root) / "endpoints.parquet")
    request_horizon = float(request.follow_up_days)
    policy_horizon = float(rule["evaluation_horizon_days"])
    if request_horizon < policy_horizon:
        raise ValueError(
            f"phase {phase!r} follow-up is shorter than its decision horizon: {request_horizon} < {policy_horizon}."
        )
    per_candidate: dict[str, dict[str, object]] = {}
    candidate_witnesses: list[TrialDevCandidateDecisionWitnessV1] = []
    for treatment_arm, candidate_drug_id in candidate_arms:
        candidate_safety_state, safety_evidence = _safety_state(
            scenario_root=Path(scenario_root),
            trial_output_root=Path(trial_output_root),
            phase_id=phase,
            control_arm=control_arm,
            treatment_arm=treatment_arm,
            confidence_level=confidence_level,
            request_follow_up_days=request_horizon,
        )
        candidate_efficacy_actions: tuple[str, ...] = actions
        efficacy_evidence: dict[str, object] = {"evaluated": False}
        if endpoints is not None:
            candidate_efficacy_actions, efficacy_evidence = _efficacy_actions(
                phase_id=phase,
                endpoints=endpoints,
                control_arm=control_arm,
                treatment_arm=treatment_arm,
                rule=rule,
                horizon=policy_horizon,
                confidence_level=confidence_level,
                stop_actions=stop_actions,
                advance_actions=advance_actions,
            )
        candidate_safety_actions: tuple[str, ...]
        candidate_actions: tuple[str, ...]
        if phase == "phase3":
            failure_actions = tuple(action for action in stop_actions if action == "declare_failure")
            inconclusive_actions = tuple(action for action in stop_actions if action == "declare_inconclusive")
            if len(failure_actions) != 1 or len(inconclusive_actions) != 1:
                raise ValueError("Phase-3 policy requires one failure and one inconclusive action.")
            if candidate_safety_state == "unacceptable":
                candidate_safety_actions = failure_actions
                candidate_actions = failure_actions
            elif candidate_safety_state == "indeterminate":
                candidate_safety_actions = tuple(sorted((*failure_actions, *inconclusive_actions)))
                candidate_actions = (
                    failure_actions if candidate_efficacy_actions == failure_actions else inconclusive_actions
                )
            else:
                candidate_safety_actions = actions
                candidate_actions = candidate_efficacy_actions
        elif candidate_safety_state == "unacceptable":
            candidate_safety_actions = stop_actions
            candidate_actions = stop_actions
        elif candidate_safety_state == "indeterminate":
            candidate_safety_actions = actions
            candidate_actions = tuple(sorted(set(candidate_efficacy_actions) | set(stop_actions)))
        elif phase == "phase1":
            candidate_safety_actions = advance_actions
            candidate_actions = advance_actions
        else:
            candidate_safety_actions = actions
            candidate_actions = candidate_efficacy_actions
        candidate_evidence: dict[str, object] = {
            "arm_id": treatment_arm,
            "acceptable_action_ids": candidate_actions,
            "safety_state": candidate_safety_state,
            "safety_action_ids": candidate_safety_actions,
            "hard_safety_stop_action_ids": stop_actions,
            "efficacy_action_ids": candidate_efficacy_actions,
            "safety": safety_evidence,
            "efficacy": efficacy_evidence,
        }
        per_candidate[candidate_drug_id] = candidate_evidence
        candidate_witnesses.append(
            TrialDevCandidateDecisionWitnessV1(
                candidate_drug_id=candidate_drug_id,
                arm_id=treatment_arm,
                acceptable_action_ids=candidate_actions,
                safety_state=candidate_safety_state,
                safety_action_ids=candidate_safety_actions,
                efficacy_action_ids=candidate_efficacy_actions,
                evidence={"safety": safety_evidence, "efficacy": efficacy_evidence},
            )
        )
    ordered_candidates = tuple(sorted(candidate_witnesses, key=lambda row: row.candidate_drug_id))
    common_stop_actions = {
        action_id
        for action_id in stop_actions
        if all(action_id in row.acceptable_action_ids for row in ordered_candidates)
    }
    candidate_advance_actions = {
        action_id
        for row in ordered_candidates
        for action_id in row.acceptable_action_ids
        if action_id in advance_actions
    }
    acceptable = tuple(sorted(common_stop_actions | candidate_advance_actions))
    common_safety_stops = {
        action_id
        for action_id in stop_actions
        if all(action_id in row.safety_action_ids for row in ordered_candidates)
    }
    candidate_safety_advances = {
        action_id for row in ordered_candidates for action_id in row.safety_action_ids if action_id in advance_actions
    }
    safety_actions = tuple(sorted(common_safety_stops | candidate_safety_advances))
    efficacy_actions = tuple(
        sorted({action_id for row in ordered_candidates for action_id in row.efficacy_action_ids})
    )
    safety_states = [row.safety_state for row in ordered_candidates]
    if len(set(safety_states)) == 1:
        safety_state = safety_states[0]
    else:
        safety_state = "mixed"
    if all(state == "unacceptable" for state in safety_states):
        recoverability = "safety_determined"
    elif (
        len(common_stop_actions)
        + sum(action_id in row.acceptable_action_ids for row in ordered_candidates for action_id in advance_actions)
        == 1
    ):
        recoverability = "unique"
    else:
        recoverability = "set_identified"
    return TrialDevPhaseDecisionWitnessV1(
        phase_id=phase,
        acceptable_action_ids=acceptable,
        recoverability_class=recoverability,
        safety_state=safety_state,
        safety_action_ids=tuple(sorted(safety_actions)),
        efficacy_action_ids=tuple(sorted(efficacy_actions)),
        stop_action_ids=stop_actions,
        advance_action_ids=advance_actions,
        candidates=ordered_candidates,
        evidence={
            "confidence_level": confidence_level,
            "control_arm_id": control_arm,
            "treatment_arm_ids": tuple(arm_id for arm_id, _drug_id in candidate_arms),
            "treatment_arm_id": candidate_arms[0][0] if len(candidate_arms) == 1 else None,
            "source_checksums": {
                "public/phase_action_policy.json": _sha256_file(
                    Path(scenario_root) / "public" / "phase_action_policy.json"
                ),
                "public/phase_decision_evidence_policy.json": _sha256_file(
                    Path(scenario_root) / "public" / "phase_decision_evidence_policy.json"
                ),
                "public/safety_decision_policy.json": _sha256_file(
                    Path(scenario_root) / "public" / "safety_decision_policy.json"
                ),
                "trial_output/arm_mapping.json": _sha256_file(Path(trial_output_root) / "arm_mapping.json"),
                "trial_output/endpoints.parquet": _sha256_file(Path(trial_output_root) / "endpoints.parquet"),
                "trial_output/execution_summary.json": _sha256_file(
                    Path(trial_output_root) / "execution_summary.json"
                ),
                "trial_output/request.json": _sha256_file(Path(trial_output_root) / "request.json"),
                "trial_output/safety.parquet": _sha256_file(Path(trial_output_root) / "safety.parquet"),
            },
            "candidates": per_candidate,
            "safety": per_candidate[candidate_arms[0][1]]["safety"] if len(candidate_arms) == 1 else None,
            "efficacy": per_candidate[candidate_arms[0][1]]["efficacy"] if len(candidate_arms) == 1 else None,
        },
    )
