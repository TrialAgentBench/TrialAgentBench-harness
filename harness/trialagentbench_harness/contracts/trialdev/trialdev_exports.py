"""Typed export surfaces for TrialDevBench aggregation outputs.

These models define the canonical column set for the harness-emitted CSV
surfaces. They are intentionally explicit (no dynamic keys) so downstream
analysis pipelines cannot silently drift.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TrialDevLaneScoreExportRowV1(BaseModel):
    """One row in `lane_scores.csv` emitted from validated grade reports."""

    model_config = ConfigDict(extra="forbid")

    program_id: str
    scenario_id: str
    scenario_key: str
    scenario_semantic_id: str
    objective_id: str
    phase_id: str
    program_objective_id: str
    phase_scoring_objective_id: str
    lane_id: str
    evaluation_target_checksum: str
    scoring_policy_id: str
    recoverability_policy_id: str
    submitted_target_id: str | None = None
    reference_target_ids: str
    credit_eligible_target_ids: str = ""
    score: float
    score_derivation: str = "literal_target"
    derived_from_trajectory_metric: bool = False
    terminal_action_observed: str | None = None
    terminal_asset_observed: str | None = None
    terminal_phase_observed: str | None = None
    status: str
    artifact_status: str
    missing_reason: str | None = None
    failure_reason: str | None = None
    source: str


class TrialDevResultsRowV1(BaseModel):
    """One row in `results_full.csv` (one graded item: obs_review or phase report)."""

    model_config = ConfigDict(extra="forbid")

    program_id: str
    scenario_id: str
    objective_id: str
    phase_id: str
    endpoint_id: str | None = None

    item_id: str | None = None
    source: str

    design_score: float
    evaluation_score: float
    program_score: float
    ranking_score: float
    primary_score: float

    policy_reference_regret: float | None = None
    in_set_regret: float | None = None
    selected_winner_drug_id: str | None = None
    best_candidate_drug_id: str | None = None
    feasibility_failures: list[str] = []

    lane_raw__trial_design: float
    lane_raw__trial_evaluation: float
    lane_raw__program_decision: float
    lane_raw__drug_ranking: float

    lane_active__trial_design: float | None = None
    lane_active__trial_evaluation: float | None = None
    lane_active__program_decision: float | None = None
    lane_active__drug_ranking: float | None = None

    lane_status__trial_design: str | None = None
    lane_status__trial_evaluation: str | None = None
    lane_status__program_decision: str | None = None
    lane_status__drug_ranking: str | None = None


class TrialDevRankMetricRowV1(BaseModel):
    """One row in `rank_metrics.csv` (per phase or obs_review)."""

    model_config = ConfigDict(extra="forbid")

    program_id: str
    scenario_id: str
    objective_id: str
    phase_id: str
    pick_type: str
    agent_top_pick: str | None = None
    reference_top_pick: str | None = None
    agent_top_pick_rank_in_reference: int | None = None
    reference_ranking_size: int | None = None
    bottom_n_concordance: float | None = None
    bottom_n: int | None = None
    utility_regret: float | None = None
    acceptable_pick: bool | None = None
    acceptable_candidate_set: str = ""


class TrialDevStickTwistRowV1(BaseModel):
    """One row in `stick_twist.csv`."""

    model_config = ConfigDict(extra="forbid")

    program_id: str
    obs_pick: str | None = None
    phase1_pick: str | None = None
    pivoted: bool


class TrialDevObjectiveAlignmentRowV1(BaseModel):
    """One row in `objective_alignment.csv`."""

    model_config = ConfigDict(extra="forbid")

    program_id: str
    primary_objective: str
    n_free_phases: int
    n_aligned: int
    alignment_rate: float | None = None
    phase1_selected: str | None = None
    phase1_forced: bool | str | None = None
    phase2_selected: str | None = None
    phase2_aligned: bool | None = None
    phase3_selected: str | None = None
    phase3_aligned: bool | None = None


class TrialDevViolationRowV1(BaseModel):
    """One row in `violations.csv`."""

    model_config = ConfigDict(extra="forbid")

    program_id: str
    scenario_id: str
    objective_id: str
    phase_id: str | None = None
    kind: str | None = None
    error: str


class TrialDevPhaseResourceRowV1(BaseModel):
    """One randomized-phase design and resource consequence row."""

    model_config = ConfigDict(extra="forbid")

    program_id: str
    scenario_id: str
    objective_id: str
    phase_id: str
    request_checksum: str
    target_sample_size: int
    follow_up_days: int
    enrollment_window_days: int
    site_count_budget: int
    participant_follow_up_days: int
    statistically_adequate: bool
    operationally_feasible: bool
    design_status: str
    operational_support: int
    operational_headroom: int
    operational_shortage: int
    achieved_power: float | None = None
    target_power: float | None = None
    achieved_safety_absolute_risk_power: float
    achieved_safety_excess_risk_power: float
    target_safety_decision_power: float
    participant_excess_vs_minimum: int
    participant_shortage_vs_minimum: int
    follow_up_excess_days_vs_minimum: int
    follow_up_shortage_days_vs_minimum: int
    dominating_frontier_count: int
    avoidable_participants_min: int
    avoidable_participants_max: int
    avoidable_follow_up_days_min: int
    avoidable_follow_up_days_max: int
    avoidable_participant_follow_up_days_min: int
    avoidable_participant_follow_up_days_max: int
    entered_after_unsupported_advance: bool


class TrialDevProgrammeResourceRowV1(BaseModel):
    """One cumulative programme resource and consequence row."""

    model_config = ConfigDict(extra="forbid")

    program_id: str
    scenario_id: str
    objective_id: str
    phase_count: int
    total_participants: int
    total_protocol_follow_up_days: int
    total_enrollment_window_days: int
    total_site_phase_budget: int
    total_planned_phase_duration_days: int
    total_participant_follow_up_days: int
    participant_excess_vs_minimum: int
    participant_shortage_vs_minimum: int
    follow_up_excess_days_vs_minimum: int
    follow_up_shortage_days_vs_minimum: int
    statistically_inadequate_phases: int
    operationally_infeasible_phases: int
    dominated_phases: int
    design_avoidable_participants_min: int
    design_avoidable_participants_max: int
    design_avoidable_follow_up_days_min: int
    design_avoidable_follow_up_days_max: int
    design_avoidable_participant_follow_up_days_min: int
    design_avoidable_participant_follow_up_days_max: int
    late_continuation_participants: int
    late_continuation_protocol_follow_up_days: int
    late_continuation_enrollment_window_days: int
    late_continuation_site_phase_budget: int
    late_continuation_participant_follow_up_days: int
    cost_status: str


__all__ = [
    "TrialDevLaneScoreExportRowV1",
    "TrialDevObjectiveAlignmentRowV1",
    "TrialDevPhaseResourceRowV1",
    "TrialDevProgrammeResourceRowV1",
    "TrialDevRankMetricRowV1",
    "TrialDevResultsRowV1",
    "TrialDevStickTwistRowV1",
    "TrialDevViolationRowV1",
]
