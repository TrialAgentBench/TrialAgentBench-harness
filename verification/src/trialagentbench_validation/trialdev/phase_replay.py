"""Independently replay randomized TrialDev decisions from released tables."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Literal, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import beta

from trialagentbench_validation.trialdev.phase_design import reconstruct_phase_design

_PHASE_REPLAY_SOURCE_PATHS = {
    "public/phase_action_policy.json",
    "public/phase_decision_evidence_policy.json",
    "public/phase_design_frontiers.json",
    "public/phase_design_policy.json",
    "public/safety_decision_policy.json",
    "trial_output/arm_mapping.json",
    "trial_output/endpoints.parquet",
    "trial_output/execution_summary.json",
    "trial_output/request.json",
    "trial_output/safety.parquet",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialDevPhaseRequestV1(_StrictModel):
    """Randomized request fields needed for independent replay."""

    version: Literal["v1"] = "v1"
    scenario_id: str
    phase_id: Literal["phase1", "phase2", "phase3"]
    candidate_drug_ids: tuple[str, ...]
    target_sample_size: int
    endpoint_id: str | None = None
    follow_up_days: int
    enrollment_window_days: int
    site_count_budget: int
    allocation_ratio: str | None = None
    allocation_weights: tuple[float, ...] = ()
    design_cell_id: str
    treatment_discontinuation_strategy: (
        Literal["treatment_policy", "while_on_treatment", "composite_discontinuation"]
        | None
    ) = None
    interim_policy: str
    site_strategy: str
    selection_objective: str
    stratification_variables: tuple[str, ...] = ()
    analysis_covariates: tuple[str, ...] = ()
    subgroup_variables: tuple[str, ...] = ()

    def checksum(self) -> str:
        """Return the release request checksum."""

        payload = self.model_dump(mode="json", exclude_none=True)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @model_validator(mode="after")
    def validate_discontinuation_strategy(self) -> TrialDevPhaseRequestV1:
        """Bind the strategy only where an efficacy endpoint is defined."""

        if (
            self.phase_id == "phase1"
            and self.treatment_discontinuation_strategy is not None
        ):
            raise ValueError(
                "Phase-1 requests must not bind a treatment-discontinuation strategy."
            )
        if (
            self.phase_id in {"phase2", "phase3"}
            and self.treatment_discontinuation_strategy is None
        ):
            raise ValueError(
                f"{self.phase_id} requests require a treatment-discontinuation strategy."
            )
        if len(self.candidate_drug_ids) != 1:
            raise ValueError(
                "Randomized TrialDev requests require exactly one investigational regimen plus control."
            )
        return self


class TrialDevPhaseReplayCaseV1(_StrictModel):
    """One scenario-bound public phase replay request."""

    scenario_root: str
    world_seed: int
    program_objective_ids: tuple[str, ...]
    request: TrialDevPhaseRequestV1

    @model_validator(mode="after")
    def validate_paths(self) -> TrialDevPhaseReplayCaseV1:
        """Require a safe matching scenario path."""

        if self.scenario_root.startswith("/") or ".." in self.scenario_root.split("/"):
            raise ValueError("scenario_root must be a safe relative path.")
        if (
            self.scenario_root.rstrip("/").split("/")[-1]
            != f"scenario_{self.request.scenario_id}"
        ):
            raise ValueError("scenario_root does not match request.scenario_id.")
        return self


class TrialDevIntervalV1(_StrictModel):
    """One public point estimate and confidence interval."""

    estimate: float
    lower: float
    upper: float


class TrialDevSafetyComponentV1(_StrictModel):
    """One public safety component."""

    component_id: Literal["serious_ae", "discontinuation"]
    role: Literal["hard_gate", "diagnostic_only"]
    treated: TrialDevIntervalV1
    control: TrialDevIntervalV1
    excess: TrialDevIntervalV1
    absolute_limit: float
    excess_limit: float


class TrialDevCandidateDecisionV1(_StrictModel):
    """One candidate-specific public decision."""

    candidate_arm_id: str
    acceptable_action_ids: tuple[str, ...]
    safety_state: Literal["acceptable", "unacceptable", "indeterminate"]
    efficacy: TrialDevIntervalV1 | None = None
    minimum_efficacy_benefit: float | None = None
    safety_components: tuple[TrialDevSafetyComponentV1, ...]


class TrialDevFrontierPointV1(_StrictModel):
    """One public randomized-design frontier point."""

    target_sample_size: int
    follow_up_days: int
    allocation_ratio: str
    achieved_power: float | None = None
    achieved_safety_absolute_risk_power: float
    achieved_safety_excess_risk_power: float


class TrialDevPublicPhaseReplayRecordV1(_StrictModel):
    """Harness output independently checked against public bytes."""

    schema_id: Literal["trialagentbench.trialdev_public_phase_replay/v1"]
    scenario_id: str
    world_seed: int
    trial_seed: int
    request_checksum: str
    trial_output_path: str
    phase_id: Literal["phase1", "phase2", "phase3"]
    endpoint_id: str | None = None
    treatment_discontinuation_strategy: (
        Literal["treatment_policy", "while_on_treatment", "composite_discontinuation"]
        | None
    ) = None
    follow_up_days: int
    target_sample_size: int
    allocation_ratio: str
    objective_ids: tuple[str, ...]
    candidate_drug_ids: tuple[str, ...]
    acceptable_action_ids: tuple[str, ...]
    stop_action_ids: tuple[str, ...]
    advance_action_ids: tuple[str, ...]
    sensitivity_action_sets: dict[str, tuple[str, ...]]
    public_decision_witness_checksum: str
    public_source_checksums: dict[str, str]
    candidate_decision_evidence: tuple[TrialDevCandidateDecisionV1, ...]
    public_safety_state: Literal["acceptable", "unacceptable", "indeterminate"]
    design_adequate: bool
    design_failures: tuple[str, ...]
    design_frontier: tuple[TrialDevFrontierPointV1, ...]
    design_on_frontier: bool
    design_dominated_by_frontier: bool
    minimum_frontier_participants: int
    minimum_frontier_follow_up_days: int
    participant_excess_vs_minimum: int
    participant_shortage_vs_minimum: int
    follow_up_excess_days_vs_minimum: int
    follow_up_shortage_days_vs_minimum: int
    achieved_power: float | None = None
    target_power: float | None = None
    achieved_safety_absolute_risk_power: float
    achieved_safety_excess_risk_power: float
    target_safety_decision_power: float

    @model_validator(mode="after")
    def validate_path(self) -> TrialDevPublicPhaseReplayRecordV1:
        """Require a safe materialized-table path."""

        if self.trial_output_path.startswith(
            "/"
        ) or ".." in self.trial_output_path.split("/"):
            raise ValueError("trial_output_path must be a safe relative path.")
        if set(self.public_source_checksums) != _PHASE_REPLAY_SOURCE_PATHS:
            raise ValueError(
                "Phase replay record requires the exact public source checksum set."
            )
        if any(
            len(checksum) != 64
            or any(
                character not in "0123456789abcdef" for character in checksum.lower()
            )
            for checksum in self.public_source_checksums.values()
        ):
            raise ValueError(
                "Phase replay source checksums must be SHA-256 hex digests."
            )
        return self


class TrialDevPhaseReplayValidationV1(_StrictModel):
    """Independent result for one randomized phase replay."""

    scenario_id: str
    world_seed: int
    trial_seed: int
    request_checksum: str
    source_checksums_match: bool
    ltfu_construction_match: bool
    request_match: bool
    numeric_evidence_match: bool
    action_match: bool
    design_projection_match: bool
    maximum_absolute_error: float = Field(..., ge=0.0)
    status: Literal["pass", "fail"]


class TrialDevPhaseReplayValidationReportV1(_StrictModel):
    """Independent randomized-phase replay report."""

    schema_id: Literal["trialagentbench.validation.trialdev_phase_replay/v1"] = (
        "trialagentbench.validation.trialdev_phase_replay/v1"
    )
    records: tuple[TrialDevPhaseReplayValidationV1, ...]
    status: Literal["pass", "fail"]


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _number(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number.")
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aalen_johansen(frame: pd.DataFrame, *, horizon: float) -> tuple[float, float]:
    required = {"TIME", "EVENT", "COMPETING_EVENT"}
    if frame.empty or not required <= set(frame.columns):
        raise ValueError(
            "Independent Aalen-Johansen replay requires a non-empty canonical surface."
        )
    times = pd.to_numeric(frame["TIME"], errors="raise").to_numpy(dtype=float)
    events = pd.to_numeric(frame["EVENT"], errors="raise").to_numpy(dtype=int)
    competing = pd.to_numeric(frame["COMPETING_EVENT"], errors="raise").to_numpy(
        dtype=int
    )
    if not set(events).issubset({0, 1}) or not set(competing).issubset({0, 1}):
        raise ValueError(
            "Independent Aalen-Johansen replay requires binary event indicators."
        )
    if bool(((events == 1) & (competing == 1)).any()):
        raise ValueError("Primary and competing events must be mutually exclusive.")
    survival = 1.0
    cif = 0.0
    increments: list[tuple[float, float, int, int, int]] = []
    any_event = (events == 1) | (competing == 1)
    for event_time in sorted(set(times[any_event & (times <= horizon)])):
        at_risk = int((times >= event_time).sum())
        primary_count = int(((times == event_time) & (events == 1)).sum())
        all_count = int(((times == event_time) & any_event).sum())
        if at_risk <= 0 or all_count <= 0:
            continue
        survival_before = survival
        cif += survival_before * primary_count / at_risk
        survival *= 1.0 - all_count / at_risk
        increments.append((cif, survival_before, at_risk, primary_count, all_count))
    variance = 0.0
    for cif_after, survival_before, at_risk, primary_count, all_count in increments:
        remaining = cif - cif_after
        if at_risk > all_count:
            variance += remaining**2 * all_count / (at_risk * (at_risk - all_count))
        variance += (
            survival_before**2 * primary_count * (at_risk - primary_count) / at_risk**3
        )
        variance -= 2.0 * remaining * survival_before * primary_count / at_risk**2
    return float(cif), max(0.0, float(variance))


def _risk_interval(
    *,
    estimate: float,
    variance: float,
    confidence_level: float,
    event_count: int,
    informative_count: int,
) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    if event_count == 0:
        return 0.0, float(beta.ppf(1.0 - alpha / 2.0, 1, informative_count))
    if event_count == informative_count:
        return float(beta.ppf(alpha / 2.0, informative_count, 1)), 1.0
    z = float(NormalDist().inv_cdf(0.5 + confidence_level / 2.0))
    half_width = z * math.sqrt(max(0.0, variance))
    return max(0.0, estimate - half_width), min(1.0, estimate + half_width)


def _risk_difference_interval(
    treated: tuple[float, float, float],
    control: tuple[float, float, float],
) -> tuple[float, float, float]:
    estimate = treated[0] - control[0]
    lower_distance = math.sqrt(
        (treated[0] - treated[1]) ** 2 + (control[2] - control[0]) ** 2
    )
    upper_distance = math.sqrt(
        (treated[2] - treated[0]) ** 2 + (control[0] - control[1]) ** 2
    )
    return (
        estimate,
        max(-1.0, estimate - lower_distance),
        min(1.0, estimate + upper_distance),
    )


def _competing_surface(
    safety: pd.DataFrame,
    *,
    endpoint_event: pd.Series,
    endpoint_time: pd.Series,
    horizon: float,
) -> pd.DataFrame:
    terminal_event = pd.to_numeric(safety["TERMINAL_EVENT"], errors="raise").astype(int)
    terminal_time = pd.to_numeric(safety["TERMINAL_TIME"], errors="raise").astype(float)
    ltfu_event = pd.to_numeric(safety["LTFU_E"], errors="raise").astype(int)
    ltfu_time = pd.to_numeric(safety["LTFU_T"], errors="raise").astype(float)
    infinity = pd.Series(math.inf, index=safety.index, dtype=float)
    endpoint_active = endpoint_time.where(endpoint_event.eq(1), infinity)
    terminal_active = terminal_time.where(terminal_event.eq(1), infinity)
    ltfu_active = ltfu_time.where(ltfu_event.eq(1), infinity)
    event_observed = (
        endpoint_active.le(horizon + 1e-9)
        & endpoint_active.le(ltfu_active + 1e-9)
        & endpoint_active.lt(terminal_active - 1e-9)
    )
    competing_observed = (
        terminal_active.le(horizon + 1e-9)
        & terminal_active.le(ltfu_active + 1e-9)
        & terminal_active.le(endpoint_active + 1e-9)
    )
    observed_time = pd.concat(
        (
            endpoint_active,
            terminal_active,
            ltfu_active,
            pd.Series(horizon, index=safety.index),
        ),
        axis=1,
    ).min(axis=1)
    return cast(
        pd.DataFrame,
        pd.DataFrame(
            {
                "TIME": observed_time,
                "EVENT": event_observed.astype(int),
                "COMPETING_EVENT": competing_observed.astype(int),
            },
            index=safety.index,
        ),
    )


def _serious_surface(
    safety: pd.DataFrame,
    *,
    definitions: Sequence[dict[str, object]],
    horizon: float,
) -> pd.DataFrame:
    event_times: list[pd.Series] = []
    for definition in definitions:
        event_column = str(definition["event_column"])
        time_column = str(definition["time_column"])
        seriousness_column = str(definition["seriousness_column"])
        event = pd.to_numeric(safety[event_column], errors="raise").astype(int)
        event_time = pd.to_numeric(safety[time_column], errors="raise").astype(float)
        serious = pd.to_numeric(safety[seriousness_column], errors="coerce")
        serious = serious.mask(serious.isna() & event.eq(0), 0.0)
        if bool((serious.isna() | ~serious.isin((0.0, 1.0))).any()):
            raise ValueError(
                "Serious-event indicators cannot be resolved to binary values."
            )
        observed = serious.eq(1.0) & event.eq(1) & event_time.le(horizon)
        event_times.append(event_time.where(observed, math.inf))
    earliest = pd.concat(event_times, axis=1).min(axis=1)
    return _competing_surface(
        safety,
        endpoint_event=earliest.map(math.isfinite).astype(int),
        endpoint_time=earliest.where(earliest.map(math.isfinite), horizon),
        horizon=horizon,
    )


def _component_interval(
    surface: pd.DataFrame,
    *,
    arm: pd.Series,
    arm_id: str,
    horizon: float,
    confidence_level: float,
) -> tuple[float, float, float]:
    frame = surface.loc[arm.eq(arm_id)]
    risk, variance = _aalen_johansen(frame, horizon=horizon)
    events = int(frame["EVENT"].sum())
    informative = int(
        (
            (frame["EVENT"] == 1)
            | (frame["COMPETING_EVENT"] == 1)
            | (frame["TIME"] >= horizon)
        ).sum()
    )
    lower, upper = _risk_interval(
        estimate=risk,
        variance=variance,
        confidence_level=confidence_level,
        event_count=events,
        informative_count=informative,
    )
    return risk, lower, upper


def _state(
    *,
    treated: tuple[float, float, float],
    excess: tuple[float, float, float],
    absolute_limit: float,
    excess_limit: float,
) -> str:
    if treated[1] > absolute_limit or excess[1] > excess_limit:
        return "unacceptable"
    if treated[2] <= absolute_limit and excess[2] <= excess_limit:
        return "acceptable"
    return "indeterminate"


def _maximum_error(
    expected: TrialDevIntervalV1, observed: tuple[float, float, float]
) -> float:
    return max(
        abs(expected.estimate - observed[0]),
        abs(expected.lower - observed[1]),
        abs(expected.upper - observed[2]),
    )


def _actions_for_interval(
    *,
    phase_id: str,
    interval: tuple[float, float, float],
    margin: float,
    stop: tuple[str, ...],
    advance: tuple[str, ...],
    direct_completion_margin: float | None,
) -> tuple[str, ...]:
    ordinary_advance = tuple(
        action for action in advance if action != "complete_development_without_phase3"
    )
    supported_advance = (
        advance
        if direct_completion_margin is not None
        and interval[1] > direct_completion_margin
        else ordinary_advance
    )
    if phase_id == "phase3":
        failure = tuple(action for action in stop if action == "declare_failure")
        inconclusive = tuple(
            action for action in stop if action == "declare_inconclusive"
        )
        if len(failure) != 1 or len(inconclusive) != 1:
            raise ValueError(
                "Phase-3 policy requires one failure and one inconclusive action."
            )
    else:
        failure = stop
        inconclusive = tuple(sorted(set(stop) | set(ordinary_advance)))
    if interval[1] >= margin:
        return tuple(sorted(supported_advance))
    if interval[2] < margin:
        return tuple(sorted(failure))
    return tuple(sorted(inconclusive))


def _sensitivity_action_sets(
    *,
    record: TrialDevPublicPhaseReplayRecordV1,
    rule: dict[str, object],
    threshold_by_component: dict[str, dict[str, object]],
    candidate_intervals: dict[str, tuple[float, float, float] | None],
    candidate_efficacy_actions: dict[str, tuple[str, ...]],
    candidate_components: dict[
        str,
        dict[
            str,
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
                tuple[float, float, float],
            ],
        ],
    ],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, set[str]] = {"primary": set(record.acceptable_action_ids)}
    margins = rule.get("sensitivity_minimum_benefits")
    if record.phase_id != "phase1":
        if not isinstance(margins, list) or not margins:
            raise ValueError(
                "Phase efficacy rule requires non-empty sensitivity_minimum_benefits."
            )
        direct_raw = rule.get("direct_completion_minimum_benefit")
        direct = (
            None
            if direct_raw is None
            else _number(direct_raw, label="direct completion minimum benefit")
        )
        for margin_raw in margins:
            margin = _number(margin_raw, label="efficacy sensitivity margin")
            margin_actions: set[str] = set()
            for interval in candidate_intervals.values():
                if interval is None:
                    raise ValueError(
                        "Randomized efficacy sensitivity requires a candidate interval."
                    )
                margin_actions.update(
                    _actions_for_interval(
                        phase_id=record.phase_id,
                        interval=interval,
                        margin=margin,
                        stop=record.stop_action_ids,
                        advance=record.advance_action_ids,
                        direct_completion_margin=direct,
                    )
                )
            result[f"efficacy_margin::{margin:.6f}"] = margin_actions

    profiles = ("strict", "primary", "permissive")
    for profile in profiles:
        profile_actions: set[str] = set()
        for candidate_id, components in candidate_components.items():
            hard_states: list[str] = []
            for component_id, (treated, _control, excess) in components.items():
                threshold = threshold_by_component[component_id]
                if str(threshold.get("role")) != "hard_gate":
                    continue
                absolute = threshold.get("sensitivity_max_absolute_rates")
                excess_limits = threshold.get("sensitivity_max_excess_vs_control")
                if not isinstance(absolute, dict) or not isinstance(
                    excess_limits, dict
                ):
                    raise ValueError(
                        "Hard safety gates require strict, primary, and permissive sensitivity limits."
                    )
                if set(absolute) != set(profiles) or set(excess_limits) != set(
                    profiles
                ):
                    raise ValueError("Safety sensitivity profiles are incomplete.")
                hard_states.append(
                    _state(
                        treated=treated,
                        excess=excess,
                        absolute_limit=_number(
                            absolute[profile],
                            label=f"{component_id} {profile} absolute limit",
                        ),
                        excess_limit=_number(
                            excess_limits[profile],
                            label=f"{component_id} {profile} excess limit",
                        ),
                    )
                )
            state = (
                "unacceptable"
                if "unacceptable" in hard_states
                else "indeterminate" if "indeterminate" in hard_states else "acceptable"
            )
            efficacy_actions = candidate_efficacy_actions[candidate_id]
            selected: tuple[str, ...]
            if record.phase_id == "phase3":
                failure = tuple(
                    action
                    for action in record.stop_action_ids
                    if action == "declare_failure"
                )
                inconclusive = tuple(
                    action
                    for action in record.stop_action_ids
                    if action == "declare_inconclusive"
                )
                if len(failure) != 1 or len(inconclusive) != 1:
                    raise ValueError(
                        "Phase-3 policy requires one failure and one inconclusive action."
                    )
                if state == "unacceptable" or tuple(efficacy_actions) == failure:
                    selected = failure
                elif state == "indeterminate":
                    selected = inconclusive
                else:
                    selected = efficacy_actions
            elif state == "unacceptable":
                selected = record.stop_action_ids
            elif state == "indeterminate":
                selected = tuple(set(record.stop_action_ids) | set(efficacy_actions))
            else:
                selected = efficacy_actions
            profile_actions.update(selected)
        result[f"safety_profile::{profile}"] = profile_actions
    return {label: tuple(sorted(actions)) for label, actions in sorted(result.items())}


def _frontier_for_request(
    *,
    payload: dict[str, object],
    request: TrialDevPhaseRequestV1,
) -> tuple[tuple[TrialDevFrontierPointV1, ...], int]:
    strata = payload.get("strata")
    support = payload.get("operational_support")
    if not isinstance(strata, list) or not isinstance(support, list):
        raise ValueError(
            "Public design-frontier artifact requires strata and operational_support."
        )
    matches = [
        row
        for row in strata
        if isinstance(row, dict)
        and str(row.get("phase_id")) == request.phase_id
        and tuple(str(value) for value in row.get("candidate_drug_ids", ()))
        == tuple(sorted(request.candidate_drug_ids))
        and row.get("endpoint_id") == request.endpoint_id
        and row.get("treatment_discontinuation_strategy")
        == request.treatment_discontinuation_strategy
        and str(row.get("design_cell_id")) == request.design_cell_id
        and str(row.get("interim_policy")) == request.interim_policy
    ]
    if len(matches) != 1:
        raise ValueError(
            "Public design-frontier artifact has no unique request stratum."
        )
    raw_frontier = matches[0].get("frontier")
    if not isinstance(raw_frontier, list) or not raw_frontier:
        raise ValueError("Matched public design-frontier stratum is empty.")
    frontier = tuple(
        TrialDevFrontierPointV1.model_validate(point) for point in raw_frontier
    )
    support_matches = [
        row
        for row in support
        if isinstance(row, dict)
        and str(row.get("phase_id")) == request.phase_id
        and int(row.get("enrollment_window_days", -1)) == request.enrollment_window_days
        and int(row.get("site_count_budget", -1)) == request.site_count_budget
        and str(row.get("site_strategy")) == request.site_strategy
    ]
    if len(support_matches) != 1:
        raise ValueError(
            "Public design-frontier artifact has no unique operational support record."
        )
    eligible = int(support_matches[0].get("eligible_subject_count", -1))
    if eligible < 0:
        raise ValueError("Operational support must be non-negative.")
    return frontier, eligible


def _optional_error(expected: float | None, observed: float | None) -> float:
    if expected is None or observed is None:
        return 0.0 if expected is observed else math.inf
    return abs(expected - observed)


def _frontier_error(
    expected: tuple[TrialDevFrontierPointV1, ...],
    observed: tuple[TrialDevFrontierPointV1, ...],
) -> float:
    if len(expected) != len(observed):
        return math.inf
    maximum = 0.0
    for left, right in zip(expected, observed, strict=True):
        if (
            left.target_sample_size,
            left.follow_up_days,
            left.allocation_ratio,
        ) != (
            right.target_sample_size,
            right.follow_up_days,
            right.allocation_ratio,
        ):
            return math.inf
        maximum = max(
            maximum,
            _optional_error(left.achieved_power, right.achieved_power),
            abs(
                left.achieved_safety_absolute_risk_power
                - right.achieved_safety_absolute_risk_power
            ),
            abs(
                left.achieved_safety_excess_risk_power
                - right.achieved_safety_excess_risk_power
            ),
        )
    return maximum


def _load_jsonl(path: Path, model: type[BaseModel]) -> tuple[BaseModel, ...]:
    records = tuple(
        model.model_validate_json(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    )
    if not records:
        raise ValueError(f"JSONL input is empty: {path}")
    return records


def validate_trialdev_phase_replay(
    *,
    bundle_root: Path,
    materialized_root: Path,
    cases_path: Path,
    records_path: Path,
    absolute_tolerance: float = 1e-10,
) -> TrialDevPhaseReplayValidationReportV1:
    """Independently verify randomized phase evidence from retained public tables."""

    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be finite and non-negative.")
    cases = tuple(
        record
        for record in _load_jsonl(cases_path, TrialDevPhaseReplayCaseV1)
        if isinstance(record, TrialDevPhaseReplayCaseV1)
    )
    records = tuple(
        record
        for record in _load_jsonl(records_path, TrialDevPublicPhaseReplayRecordV1)
        if isinstance(record, TrialDevPublicPhaseReplayRecordV1)
    )
    case_by_key = {(case.world_seed, case.request.checksum()): case for case in cases}
    if len(case_by_key) != len(cases):
        raise ValueError("Phase replay cases must be unique by world seed and request.")
    bundle = Path(bundle_root).resolve()
    outputs = Path(materialized_root).resolve()
    results: list[TrialDevPhaseReplayValidationV1] = []
    for record in records:
        case = case_by_key.get((record.world_seed, record.request_checksum))
        if case is None:
            raise ValueError("Phase replay record has no matching public case.")
        scenario = (bundle / case.scenario_root).resolve()
        trial_output = (outputs / record.trial_output_path).resolve()
        if not scenario.is_relative_to(bundle) or not trial_output.is_relative_to(
            outputs
        ):
            raise ValueError("Phase replay path escapes its declared root.")
        source_match = True
        for relative, expected_checksum in record.public_source_checksums.items():
            source = None
            if relative.startswith("public/"):
                source = scenario / relative
            elif relative.startswith("trial_output/"):
                source = trial_output / relative.removeprefix("trial_output/")
            if (
                source is None
                or not source.is_file()
                or _sha256(source) != expected_checksum
            ):
                source_match = False
        execution_summary = _read_json(trial_output / "execution_summary.json")
        execution_payload = execution_summary.get("payload")
        ltfu_construction_match = (
            isinstance(execution_payload, dict)
            and execution_payload.get("loss_to_follow_up_assignment")
            == "arm_conditional_random_permutation_v1"
        )
        materialized_request = TrialDevPhaseRequestV1.model_validate(
            _read_json(trial_output / "request.json")
        )
        request_match = (
            materialized_request == case.request
            and record.scenario_id == case.request.scenario_id
            and record.phase_id == case.request.phase_id
        )
        mapping = _read_json(trial_output / "arm_mapping.json")
        control_arm = str(mapping.get("control_arm_id") or "")
        candidate_arms = mapping.get("candidate_arm_ids")
        drug_by_arm = mapping.get("drug_id_by_arm")
        if (
            not control_arm
            or not isinstance(candidate_arms, list)
            or not isinstance(drug_by_arm, dict)
        ):
            raise ValueError("arm_mapping.json lacks a complete arm identity.")
        endpoints = pd.read_parquet(trial_output / "endpoints.parquet")
        safety = pd.read_parquet(trial_output / "safety.parquet")
        decision_policy = _read_json(
            scenario / "public" / "phase_decision_evidence_policy.json"
        )
        confidence_level = _number(
            decision_policy["confidence_level"], label="confidence_level"
        )
        rules = decision_policy.get("phase_rules")
        if not isinstance(rules, list):
            raise ValueError("phase decision policy requires phase_rules.")
        rule_matches = [
            row
            for row in rules
            if isinstance(row, dict) and str(row.get("phase_id")) == record.phase_id
        ]
        if len(rule_matches) != 1:
            raise ValueError("phase decision policy requires one matching rule.")
        rule = rule_matches[0]
        horizon = _number(
            rule["evaluation_horizon_days"], label="evaluation_horizon_days"
        )
        action_policy = _read_json(scenario / "public" / "phase_action_policy.json")
        action_specs = action_policy.get("action_specs")
        if not isinstance(action_specs, list):
            raise ValueError("phase action policy requires action_specs.")
        action_matches = [
            row
            for row in action_specs
            if isinstance(row, dict) and str(row.get("phase_id")) == record.phase_id
        ]
        if len(action_matches) != 1:
            raise ValueError(
                "phase action policy requires one matching action specification."
            )
        stop = tuple(str(value) for value in action_matches[0]["stop_action_ids"])
        advance = tuple(str(value) for value in action_matches[0]["advance_action_ids"])
        safety_policy = _read_json(scenario / "public" / "safety_decision_policy.json")
        definitions = safety_policy.get("serious_event_definitions")
        thresholds = safety_policy.get("thresholds")
        if not isinstance(definitions, list) or not isinstance(thresholds, list):
            raise ValueError(
                "safety decision policy requires definitions and thresholds."
            )
        phase_thresholds = [
            row
            for row in thresholds
            if isinstance(row, dict) and str(row.get("phase_id")) == record.phase_id
        ]
        threshold_by_component = {
            str(row.get("component_id")): row
            for row in phase_thresholds
            if isinstance(row.get("component_id"), str)
        }
        if set(threshold_by_component) != {"serious_ae", "discontinuation"}:
            raise ValueError(
                "Phase safety policy requires one serious-AE and one discontinuation threshold."
            )
        serious = _serious_surface(safety, definitions=definitions, horizon=horizon)
        discontinuation = _competing_surface(
            safety,
            endpoint_event=pd.to_numeric(
                safety["DISCONTINUATION_E"], errors="raise"
            ).astype(int),
            endpoint_time=pd.to_numeric(
                safety["DISCONTINUATION_T"], errors="raise"
            ).astype(float),
            horizon=horizon,
        )
        arm = safety["ARM"].astype(str)
        expected_candidates = {
            row.candidate_arm_id: row for row in record.candidate_decision_evidence
        }
        computed_actions: dict[str, tuple[str, ...]] = {}
        computed_states: list[str] = []
        candidate_intervals: dict[str, tuple[float, float, float] | None] = {}
        candidate_efficacy_actions: dict[str, tuple[str, ...]] = {}
        candidate_components: dict[
            str,
            dict[
                str,
                tuple[
                    tuple[float, float, float],
                    tuple[float, float, float],
                    tuple[float, float, float],
                ],
            ],
        ] = {}
        maximum_error = 0.0
        numeric_match = True
        action_rows_match = True
        for candidate_arm in (str(value) for value in candidate_arms):
            candidate_id = str(drug_by_arm.get(candidate_arm) or "")
            expected_candidate = expected_candidates.get(candidate_id)
            if expected_candidate is None:
                raise ValueError(
                    f"Replay record lacks candidate evidence for {candidate_id!r}."
                )
            component_values: dict[
                str,
                tuple[
                    tuple[float, float, float],
                    tuple[float, float, float],
                    tuple[float, float, float],
                ],
            ] = {}
            for component_id, surface in (
                ("serious_ae", serious),
                ("discontinuation", discontinuation),
            ):
                treated = _component_interval(
                    surface,
                    arm=arm,
                    arm_id=candidate_arm,
                    horizon=horizon,
                    confidence_level=confidence_level,
                )
                control = _component_interval(
                    surface,
                    arm=arm,
                    arm_id=control_arm,
                    horizon=horizon,
                    confidence_level=confidence_level,
                )
                component_values[component_id] = (
                    treated,
                    control,
                    _risk_difference_interval(treated, control),
                )
            hard_states: list[str] = []
            for expected_component in expected_candidate.safety_components:
                threshold = threshold_by_component[expected_component.component_id]
                if (
                    expected_component.role != str(threshold.get("role"))
                    or expected_component.absolute_limit
                    != float(threshold["max_absolute_rate"])
                    or expected_component.excess_limit
                    != float(threshold["max_excess_vs_control"])
                ):
                    raise ValueError(
                        "Replay safety component disagrees with the public threshold policy."
                    )
                treated, control, excess = component_values[
                    expected_component.component_id
                ]
                maximum_error = max(
                    maximum_error,
                    _maximum_error(expected_component.treated, treated),
                    _maximum_error(expected_component.control, control),
                    _maximum_error(expected_component.excess, excess),
                )
                if expected_component.role == "hard_gate":
                    hard_states.append(
                        _state(
                            treated=treated,
                            excess=excess,
                            absolute_limit=expected_component.absolute_limit,
                            excess_limit=expected_component.excess_limit,
                        )
                    )
            candidate_components[candidate_id] = component_values
            if "unacceptable" in hard_states:
                safety_state = "unacceptable"
            elif "indeterminate" in hard_states:
                safety_state = "indeterminate"
            else:
                safety_state = "acceptable"
            computed_states.append(safety_state)
            if record.phase_id == "phase1":
                efficacy_actions = tuple(sorted((*stop, *advance)))
                candidate_intervals[candidate_id] = None
            else:
                control_endpoints = endpoints.loc[
                    endpoints["ARM"].astype(str).eq(control_arm)
                ]
                candidate_endpoints = endpoints.loc[
                    endpoints["ARM"].astype(str).eq(candidate_arm)
                ]
                control_risk, control_variance = _aalen_johansen(
                    control_endpoints, horizon=horizon
                )
                candidate_risk, candidate_variance = _aalen_johansen(
                    candidate_endpoints, horizon=horizon
                )
                estimate = control_risk - candidate_risk
                standard_error = math.sqrt(
                    max(0.0, control_variance + candidate_variance)
                )
                z = float(NormalDist().inv_cdf(0.5 + confidence_level / 2.0))
                efficacy = (
                    estimate,
                    max(-1.0, estimate - z * standard_error),
                    min(1.0, estimate + z * standard_error),
                )
                if expected_candidate.efficacy is None:
                    raise ValueError("Phase-2/3 replay record lacks efficacy evidence.")
                maximum_error = max(
                    maximum_error, _maximum_error(expected_candidate.efficacy, efficacy)
                )
                minimum_benefit = float(rule["minimum_benefit"])
                direct_raw = rule.get("direct_completion_minimum_benefit")
                direct = (
                    None
                    if direct_raw is None
                    else _number(direct_raw, label="direct completion minimum benefit")
                )
                efficacy_actions = _actions_for_interval(
                    phase_id=record.phase_id,
                    interval=efficacy,
                    margin=minimum_benefit,
                    stop=stop,
                    advance=advance,
                    direct_completion_margin=direct,
                )
                candidate_intervals[candidate_id] = efficacy
            candidate_efficacy_actions[candidate_id] = efficacy_actions
            candidate_actions: tuple[str, ...]
            if record.phase_id == "phase3":
                failure = tuple(
                    action for action in stop if action == "declare_failure"
                )
                inconclusive = tuple(
                    action for action in stop if action == "declare_inconclusive"
                )
                if len(failure) != 1 or len(inconclusive) != 1:
                    raise ValueError(
                        "Phase-3 policy requires one failure and one inconclusive action."
                    )
                if safety_state == "unacceptable" or tuple(efficacy_actions) == failure:
                    candidate_actions = failure
                elif safety_state == "indeterminate":
                    candidate_actions = inconclusive
                else:
                    candidate_actions = efficacy_actions
            elif safety_state == "unacceptable":
                candidate_actions = stop
            elif safety_state == "indeterminate":
                candidate_actions = tuple(sorted(set(stop) | set(efficacy_actions)))
            elif record.phase_id == "phase1":
                candidate_actions = advance
            else:
                candidate_actions = efficacy_actions
            computed_actions[candidate_id] = tuple(sorted(candidate_actions))
            action_rows_match &= (
                expected_candidate.safety_state == safety_state
                and tuple(sorted(expected_candidate.acceptable_action_ids))
                == tuple(sorted(candidate_actions))
            )
        numeric_match &= maximum_error <= absolute_tolerance
        common_stops = {
            action
            for action in stop
            if all(action in actions for actions in computed_actions.values())
        }
        candidate_advances = {
            action
            for actions in computed_actions.values()
            for action in actions
            if action in set(advance)
        }
        aggregate_actions = tuple(sorted(common_stops | candidate_advances))
        public_state = (
            computed_states[0] if len(set(computed_states)) == 1 else "indeterminate"
        )
        action_match = (
            action_rows_match
            and aggregate_actions == tuple(sorted(record.acceptable_action_ids))
            and public_state == record.public_safety_state
        )
        sensitivity_actions = _sensitivity_action_sets(
            record=record,
            rule=rule,
            threshold_by_component=threshold_by_component,
            candidate_intervals=candidate_intervals,
            candidate_efficacy_actions=candidate_efficacy_actions,
            candidate_components=candidate_components,
        )
        action_match &= sensitivity_actions == {
            label: tuple(sorted(actions))
            for label, actions in sorted(record.sensitivity_action_sets.items())
        }

        design = reconstruct_phase_design(
            request=case.request,
            arm_mapping=mapping,
            safety=safety,
            design_policy=_read_json(scenario / "public" / "phase_design_policy.json"),
        )
        frontier, operational_support = _frontier_for_request(
            payload=_read_json(scenario / "public" / "phase_design_frontiers.json"),
            request=case.request,
        )
        frontier_keys = {
            (point.target_sample_size, point.follow_up_days, point.allocation_ratio)
            for point in frontier
        }
        minimum_n = min(point.target_sample_size for point in frontier)
        minimum_follow_up = min(point.follow_up_days for point in frontier)
        operationally_feasible = case.request.target_sample_size <= operational_support
        design_valid = design.adequate and operationally_feasible
        submitted_key = (
            case.request.target_sample_size,
            case.request.follow_up_days,
            str(case.request.allocation_ratio),
        )
        dominated = design_valid and any(
            point.target_sample_size <= case.request.target_sample_size
            and point.follow_up_days <= case.request.follow_up_days
            and (
                point.target_sample_size < case.request.target_sample_size
                or point.follow_up_days < case.request.follow_up_days
            )
            for point in frontier
        )
        frontier_error = _frontier_error(record.design_frontier, frontier)
        design_error = max(
            _optional_error(record.achieved_power, design.achieved_power),
            _optional_error(record.target_power, design.target_power),
            abs(
                record.achieved_safety_absolute_risk_power
                - design.achieved_safety_absolute_risk_power
            ),
            abs(
                record.achieved_safety_excess_risk_power
                - design.achieved_safety_excess_risk_power
            ),
            abs(
                record.target_safety_decision_power
                - design.target_safety_decision_power
            ),
            frontier_error,
        )
        maximum_error = max(maximum_error, design_error)
        design_projection_match = (
            frontier_error <= absolute_tolerance
            and record.design_adequate == design.adequate
            and record.design_failures == design.failures
            and design_error <= absolute_tolerance
            and record.design_on_frontier
            == (design_valid and submitted_key in frontier_keys)
            and record.design_dominated_by_frontier == dominated
            and record.minimum_frontier_participants == minimum_n
            and record.minimum_frontier_follow_up_days == minimum_follow_up
            and record.participant_excess_vs_minimum
            == max(0, record.target_sample_size - minimum_n)
            and record.participant_shortage_vs_minimum
            == max(0, minimum_n - record.target_sample_size)
            and record.follow_up_excess_days_vs_minimum
            == max(0, record.follow_up_days - minimum_follow_up)
            and record.follow_up_shortage_days_vs_minimum
            == max(0, minimum_follow_up - record.follow_up_days)
        )
        passed = (
            source_match
            and ltfu_construction_match
            and request_match
            and numeric_match
            and action_match
            and design_projection_match
        )
        results.append(
            TrialDevPhaseReplayValidationV1(
                scenario_id=record.scenario_id,
                world_seed=record.world_seed,
                trial_seed=record.trial_seed,
                request_checksum=record.request_checksum,
                source_checksums_match=source_match,
                ltfu_construction_match=ltfu_construction_match,
                request_match=request_match,
                numeric_evidence_match=numeric_match,
                action_match=action_match,
                design_projection_match=design_projection_match,
                maximum_absolute_error=maximum_error,
                status="pass" if passed else "fail",
            )
        )
    return TrialDevPhaseReplayValidationReportV1(
        records=tuple(results),
        status="pass" if all(row.status == "pass" for row in results) else "fail",
    )


__all__ = [
    "TrialDevPhaseReplayValidationReportV1",
    "validate_trialdev_phase_replay",
]
