"""Deterministic planning assessment for eligible TrialEval tasks."""

from __future__ import annotations

import math
from statistics import NormalDist

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.contracts.submission import ScalarEstimateV1, TrialEvalSubmissionV1
from trialagentbench_harness.grading.models import NumericPointTargetV1, ValidatedScoringKeyV1
from trialagentbench_harness.trialeval.schema import BenchmarkItem


class _PlanningModelV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlanningReferenceVariantV1(_PlanningModelV1):
    """One compatible scoreable route-specific reference for planning consequences."""

    route_reference_id: str
    log_hazard_ratio: float
    is_matched_variant: bool


class PlanningReferenceConsequenceV1(_PlanningModelV1):
    """Non-compensatory resource and operating-characteristic consequences."""

    route_reference_id: str
    is_matched_variant: bool
    reference_required_events: int = Field(ge=1)
    reference_target_sample_size: int = Field(ge=1)
    achieved_power: float = Field(ge=0.0, le=1.0)
    target_power: float = Field(gt=0.0, lt=1.0)
    power_shortfall: float = Field(ge=0.0, le=1.0)
    event_deviation: int
    participant_deviation: int
    proportional_participant_deviation: float
    log_sample_size_ratio: float
    underpowered: bool
    event_shortage: int = Field(ge=0)
    participant_shortage: int = Field(ge=0)
    excess_events: int = Field(ge=0)
    excess_participants: int = Field(ge=0)


class PlanningAssessmentV1(_PlanningModelV1):
    """Separated planning outputs for one TrialEval submission."""

    applicable: bool
    submitted: bool
    contract_valid: bool | None = None
    effect_feasible: bool | None = None
    formula_valid: bool | None = None
    sensitivity_valid: bool | None = None
    valid: bool | None = None
    required_events_expected: int | None = Field(default=None, ge=1)
    target_sample_size_expected: int | None = Field(default=None, ge=1)
    reference_target_sample_size: int | None = Field(default=None, ge=1)
    sample_size_ratio_vs_reference: float | None = None
    matched_route_reference_id: str | None = None
    matched_achieved_power: float | None = None
    matched_power_shortfall: float | None = None
    matched_underpowered: bool | None = None
    matched_proportional_participant_deviation: float | None = None
    matched_log_sample_size_ratio: float | None = None
    matched_event_shortage: int | None = Field(default=None, ge=0)
    matched_excess_events: int | None = Field(default=None, ge=0)
    matched_excess_participants: int | None = Field(default=None, ge=0)
    matched_participant_shortage: int | None = Field(default=None, ge=0)
    reference_consequences: tuple[PlanningReferenceConsequenceV1, ...] = ()

    @model_validator(mode="after")
    def _coherent_assessment(self) -> PlanningAssessmentV1:
        assessment_fields = (
            self.contract_valid,
            self.effect_feasible,
            self.formula_valid,
            self.sensitivity_valid,
            self.valid,
            self.required_events_expected,
            self.target_sample_size_expected,
            self.reference_target_sample_size,
            self.sample_size_ratio_vs_reference,
            self.matched_route_reference_id,
            self.matched_achieved_power,
            self.matched_power_shortfall,
            self.matched_underpowered,
            self.matched_proportional_participant_deviation,
            self.matched_log_sample_size_ratio,
            self.matched_event_shortage,
            self.matched_excess_events,
            self.matched_excess_participants,
            self.matched_participant_shortage,
        )
        if not self.applicable and (
            any(value is not None for value in assessment_fields) or self.reference_consequences
        ):
            raise ValueError("Inapplicable planning cannot carry assessment results.")
        if self.applicable and self.valid is None:
            raise ValueError("Applicable planning requires an explicit validity result.")
        if self.valid and not all(
            value is True
            for value in (
                self.contract_valid,
                self.effect_feasible,
                self.formula_valid,
                self.sensitivity_valid,
            )
        ):
            raise ValueError("Valid planning requires every planning component to pass.")
        return self


def schoenfeld_required_events_v1(
    *,
    log_hazard_ratio: float,
    alpha_two_sided: float,
    power: float,
    treated_allocation_fraction: float,
) -> int:
    """Return the ceiling event count for a two-sided log-rank comparison."""

    if not math.isfinite(log_hazard_ratio) or math.isclose(log_hazard_ratio, 0.0, abs_tol=1e-15):
        raise ValueError("Schoenfeld planning requires a finite non-zero log hazard ratio.")
    if not 0.0 < alpha_two_sided < 1.0:
        raise ValueError("alpha_two_sided must be in (0, 1).")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1).")
    if not 0.0 < treated_allocation_fraction < 1.0:
        raise ValueError("treated_allocation_fraction must be in (0, 1).")
    z_alpha = NormalDist().inv_cdf(1.0 - alpha_two_sided / 2.0)
    z_power = NormalDist().inv_cdf(power)
    information = treated_allocation_fraction * (1.0 - treated_allocation_fraction)
    upper = max(1, int(math.ceil((z_alpha + z_power) ** 2 / (information * log_hazard_ratio**2))))
    while (
        schoenfeld_achieved_power_v1(
            expected_events=float(upper),
            log_hazard_ratio=log_hazard_ratio,
            alpha_two_sided=alpha_two_sided,
            treated_allocation_fraction=treated_allocation_fraction,
        )
        < power
    ):
        upper *= 2

    lower = 1
    while lower < upper:
        midpoint = (lower + upper) // 2
        achieved_power = schoenfeld_achieved_power_v1(
            expected_events=float(midpoint),
            log_hazard_ratio=log_hazard_ratio,
            alpha_two_sided=alpha_two_sided,
            treated_allocation_fraction=treated_allocation_fraction,
        )
        if achieved_power >= power:
            upper = midpoint
        else:
            lower = midpoint + 1
    return lower


def participants_for_events_v1(*, required_events: int, event_probability: float) -> int:
    """Return randomized participants required to observe the event target."""

    if required_events <= 0:
        raise ValueError("required_events must be positive.")
    if not 0.0 < event_probability <= 1.0:
        raise ValueError("event_probability must be in (0, 1].")
    return int(math.ceil(required_events / event_probability))


def schoenfeld_achieved_power_v1(
    *,
    expected_events: float,
    log_hazard_ratio: float,
    alpha_two_sided: float,
    treated_allocation_fraction: float,
) -> float:
    """Return asymptotic power for an event-driven two-arm log-rank design."""

    if not math.isfinite(expected_events) or expected_events < 0.0:
        raise ValueError("expected_events must be finite and non-negative.")
    if not math.isfinite(log_hazard_ratio):
        raise ValueError("log_hazard_ratio must be finite.")
    if not 0.0 < alpha_two_sided < 1.0:
        raise ValueError("alpha_two_sided must be in (0, 1).")
    if not 0.0 < treated_allocation_fraction < 1.0:
        raise ValueError("treated_allocation_fraction must be in (0, 1).")
    information = treated_allocation_fraction * (1.0 - treated_allocation_fraction)
    z_alpha = NormalDist().inv_cdf(1.0 - alpha_two_sided / 2.0)
    noncentrality = math.sqrt(expected_events * information) * abs(log_hazard_ratio)
    normal = NormalDist()
    return float(normal.cdf(noncentrality - z_alpha) + normal.cdf(-noncentrality - z_alpha))


def _reference_consequence(
    *,
    variant: PlanningReferenceVariantV1,
    submitted_target_sample_size: int,
    submitted_required_events: int,
    event_probability: float,
    alpha_two_sided: float,
    target_power: float,
    treated_allocation_fraction: float,
) -> PlanningReferenceConsequenceV1:
    """Compare one submitted design with one compatible route-specific reference."""

    reference_events = schoenfeld_required_events_v1(
        log_hazard_ratio=variant.log_hazard_ratio,
        alpha_two_sided=alpha_two_sided,
        power=target_power,
        treated_allocation_fraction=treated_allocation_fraction,
    )
    reference_n = participants_for_events_v1(
        required_events=reference_events,
        event_probability=event_probability,
    )
    effective_events = min(
        float(submitted_required_events),
        float(submitted_target_sample_size) * event_probability,
    )
    achieved_power = schoenfeld_achieved_power_v1(
        expected_events=effective_events,
        log_hazard_ratio=variant.log_hazard_ratio,
        alpha_two_sided=alpha_two_sided,
        treated_allocation_fraction=treated_allocation_fraction,
    )
    event_deviation = submitted_required_events - reference_events
    participant_deviation = submitted_target_sample_size - reference_n
    return PlanningReferenceConsequenceV1(
        route_reference_id=variant.route_reference_id,
        is_matched_variant=variant.is_matched_variant,
        reference_required_events=reference_events,
        reference_target_sample_size=reference_n,
        achieved_power=achieved_power,
        target_power=target_power,
        power_shortfall=max(0.0, target_power - achieved_power),
        event_deviation=event_deviation,
        participant_deviation=participant_deviation,
        proportional_participant_deviation=float(participant_deviation / reference_n),
        log_sample_size_ratio=float(math.log(submitted_target_sample_size / reference_n)),
        underpowered=achieved_power + 1e-12 < target_power,
        event_shortage=max(0, -event_deviation),
        participant_shortage=max(0, -participant_deviation),
        excess_events=max(0, event_deviation),
        excess_participants=max(0, participant_deviation),
    )


def assess_planning_v1(
    *,
    task: dict[str, object],
    primary: dict[str, object],
    planning: dict[str, object] | None,
    reference_variants: tuple[PlanningReferenceVariantV1, ...],
) -> PlanningAssessmentV1:
    """Assess planning against the exact public contract and submitted effect."""

    contract = task.get("planning")
    if not isinstance(contract, dict) or contract.get("status") not in {"eligible", "ineligible"}:
        raise ValueError("TrialEval task is missing a valid planning eligibility contract.")
    if contract["status"] == "ineligible":
        return PlanningAssessmentV1(applicable=False, submitted=planning is not None)
    eligible_scale = str(contract.get("eligible_effect_scale") or "")
    submitted_scale = str(primary.get("effect_scale") or "")
    if not eligible_scale:
        raise ValueError("Eligible planning contract requires eligible_effect_scale.")
    if submitted_scale != eligible_scale:
        return PlanningAssessmentV1(applicable=False, submitted=planning is not None)
    if planning is None:
        return PlanningAssessmentV1(
            applicable=True,
            submitted=False,
            contract_valid=False,
            effect_feasible=None,
            formula_valid=False,
            sensitivity_valid=False,
            valid=False,
        )

    contract_fields = (
        "method_id",
        "alpha_two_sided",
        "power",
        "treated_allocation_fraction",
        "event_probability",
        "followup_horizon_dy",
        "multiplicity_adjustment",
    )
    contract_valid = all(planning.get(field) == contract.get(field) for field in contract_fields)
    contract_valid = contract_valid and planning.get("estimand_id") == task.get("primary_estimand_id")

    submitted_effect = primary.get("point_estimate")
    if not isinstance(submitted_effect, int | float) or isinstance(submitted_effect, bool):
        return PlanningAssessmentV1(
            applicable=True,
            submitted=True,
            contract_valid=contract_valid,
            effect_feasible=False,
            formula_valid=False,
            sensitivity_valid=False,
            valid=False,
        )
    if not math.isfinite(float(submitted_effect)) or math.isclose(float(submitted_effect), 0.0, abs_tol=1e-15):
        return PlanningAssessmentV1(
            applicable=True,
            submitted=True,
            contract_valid=contract_valid,
            effect_feasible=False,
            formula_valid=False,
            sensitivity_valid=False,
            valid=False,
        )
    expected_events = schoenfeld_required_events_v1(
        log_hazard_ratio=float(submitted_effect),
        alpha_two_sided=float(contract["alpha_two_sided"]),
        power=float(contract["power"]),
        treated_allocation_fraction=float(contract["treated_allocation_fraction"]),
    )
    expected_n = participants_for_events_v1(
        required_events=expected_events,
        event_probability=float(contract["event_probability"]),
    )
    formula_valid = (
        planning.get("required_events") == expected_events and planning.get("target_sample_size") == expected_n
    )
    submitted_required_events = planning.get("required_events")
    submitted_target_sample_size = planning.get("target_sample_size")
    if (
        isinstance(submitted_required_events, bool)
        or not isinstance(submitted_required_events, int)
        or submitted_required_events < 1
        or isinstance(submitted_target_sample_size, bool)
        or not isinstance(submitted_target_sample_size, int)
        or submitted_target_sample_size < 1
    ):
        raise ValueError("Submitted planning events and sample size must be positive integers.")

    interval = contract.get("event_probability_interval")
    if not isinstance(interval, list | tuple) or len(interval) != 2:
        raise ValueError("Eligible planning contract requires a two-value event_probability_interval.")
    expected_sensitivity = {
        float(probability): participants_for_events_v1(
            required_events=expected_events,
            event_probability=float(probability),
        )
        for probability in interval
    }
    submitted_sensitivity = planning.get("sensitivity")
    observed_sensitivity = (
        {
            float(row["event_probability"]): int(row["target_sample_size"])
            for row in submitted_sensitivity
            if isinstance(row, dict) and "event_probability" in row and "target_sample_size" in row
        }
        if isinstance(submitted_sensitivity, list | tuple)
        else {}
    )
    sensitivity_valid = observed_sensitivity == expected_sensitivity

    consequences = tuple(
        _reference_consequence(
            variant=variant,
            submitted_target_sample_size=submitted_target_sample_size,
            submitted_required_events=submitted_required_events,
            event_probability=float(contract["event_probability"]),
            alpha_two_sided=float(contract["alpha_two_sided"]),
            target_power=float(contract["power"]),
            treated_allocation_fraction=float(contract["treated_allocation_fraction"]),
        )
        for variant in reference_variants
    )
    matched = tuple(row for row in consequences if row.is_matched_variant)
    if len(matched) > 1:
        raise ValueError("Planning consequences contain more than one matched route-specific reference.")
    matched_row = matched[0] if matched else None
    reference_n = None if matched_row is None else matched_row.reference_target_sample_size
    ratio = None if reference_n is None else float(expected_n / reference_n)
    valid = bool(contract_valid and formula_valid and sensitivity_valid)
    return PlanningAssessmentV1(
        applicable=True,
        submitted=True,
        contract_valid=contract_valid,
        effect_feasible=True,
        formula_valid=formula_valid,
        sensitivity_valid=sensitivity_valid,
        valid=valid,
        required_events_expected=expected_events,
        target_sample_size_expected=expected_n,
        reference_target_sample_size=reference_n,
        sample_size_ratio_vs_reference=ratio,
        matched_route_reference_id=None if matched_row is None else matched_row.route_reference_id,
        matched_achieved_power=None if matched_row is None else matched_row.achieved_power,
        matched_power_shortfall=None if matched_row is None else matched_row.power_shortfall,
        matched_underpowered=None if matched_row is None else matched_row.underpowered,
        matched_proportional_participant_deviation=(
            None if matched_row is None else matched_row.proportional_participant_deviation
        ),
        matched_log_sample_size_ratio=None if matched_row is None else matched_row.log_sample_size_ratio,
        matched_event_shortage=None if matched_row is None else matched_row.event_shortage,
        matched_excess_events=None if matched_row is None else matched_row.excess_events,
        matched_excess_participants=None if matched_row is None else matched_row.excess_participants,
        matched_participant_shortage=None if matched_row is None else matched_row.participant_shortage,
        reference_consequences=consequences,
    )


def assess_trialeval_route_planning_v1(
    *,
    item: BenchmarkItem,
    scoring_key: ValidatedScoringKeyV1,
    matched_route_id: str | None,
    submission: TrialEvalSubmissionV1 | None,
) -> PlanningAssessmentV1:
    """Assess planning only for the explicitly matched, qualified analysis route."""

    submitted = submission is not None and submission.planning is not None
    if matched_route_id is None:
        return PlanningAssessmentV1(applicable=False, submitted=submitted)
    route = next(
        (candidate for candidate in scoring_key.credit_eligible_routes if candidate.route_id == matched_route_id),
        None,
    )
    if route is None:
        raise ValueError(f"Matched route {matched_route_id!r} is absent from the scoring key.")
    if route.planning_calculator_id is None:
        return PlanningAssessmentV1(applicable=False, submitted=submitted)
    if route.planning_calculator_id != "schoenfeld_logrank_v1":
        raise ValueError(f"Credit-eligible route {route.route_id!r} names an unqualified planning calculator.")
    if not isinstance(route.target, NumericPointTargetV1) or route.signature.effect_scale != "log_hr":
        raise ValueError("Schoenfeld planning requires a numeric log-HR analysis route.")
    primary_payload: dict[str, object] = {"effect_scale": route.signature.effect_scale}
    if submission is not None and isinstance(submission.primary_analysis.result, ScalarEstimateV1):
        primary_payload["point_estimate"] = float(submission.primary_analysis.result.value)
    return assess_planning_v1(
        task=item.task,
        primary=primary_payload,
        planning=(
            None
            if submission is None or submission.planning is None
            else submission.planning.model_dump(mode="python")
        ),
        reference_variants=(
            PlanningReferenceVariantV1(
                route_reference_id=route.route_id,
                log_hazard_ratio=float(route.target.value),
                is_matched_variant=True,
            ),
        ),
    )


__all__ = [
    "PlanningAssessmentV1",
    "PlanningReferenceVariantV1",
    "PlanningReferenceConsequenceV1",
    "assess_planning_v1",
    "assess_trialeval_route_planning_v1",
    "participants_for_events_v1",
    "schoenfeld_achieved_power_v1",
    "schoenfeld_required_events_v1",
]
