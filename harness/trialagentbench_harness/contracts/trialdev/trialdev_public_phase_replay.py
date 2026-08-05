"""Public-evidence contracts for TrialDev randomized-phase replay."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.trialdev.grading.models import TrialDevDesignFrontierPointV1
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentRequestV1

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


class _FrozenPhaseReplayModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialDevPhaseReplayCaseV1(_FrozenPhaseReplayModel):
    """One public phase request bound to a generated scenario directory."""

    scenario_root: str = Field(..., min_length=1)
    world_seed: int = Field(..., ge=0)
    program_objective_ids: tuple[str, ...] = Field(..., min_length=1)
    request: TrialDevelopmentRequestV1

    @model_validator(mode="after")
    def validate_case(self) -> TrialDevPhaseReplayCaseV1:
        """Require a canonical objective set and matching scenario path."""

        objectives = tuple(sorted(set(self.program_objective_ids)))
        if len(objectives) != len(self.program_objective_ids):
            raise ValueError("program_objective_ids must be unique.")
        if self.scenario_root.startswith("/") or ".." in self.scenario_root.split("/"):
            raise ValueError("scenario_root must be a safe path relative to the declared bundle root.")
        expected_name = f"scenario_{self.request.scenario_id}"
        if self.scenario_root.rstrip("/").split("/")[-1] != expected_name:
            raise ValueError("Phase replay scenario_root must match request.scenario_id.")
        object.__setattr__(self, "program_objective_ids", objectives)
        return self


class TrialDevPublicIntervalV1(_FrozenPhaseReplayModel):
    """Finite estimate and ordered public confidence interval."""

    estimate: float
    lower: float
    upper: float

    @model_validator(mode="after")
    def validate_interval(self) -> TrialDevPublicIntervalV1:
        """Require finite ordered values containing the estimate."""

        values = (self.estimate, self.lower, self.upper)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Public phase-replay intervals must be finite.")
        if self.lower > self.estimate or self.estimate > self.upper:
            raise ValueError("Public phase-replay intervals require lower <= estimate <= upper.")
        return self


class TrialDevPublicSafetyComponentV1(_FrozenPhaseReplayModel):
    """Public cumulative-incidence evidence for one safety component."""

    component_id: Literal["serious_ae", "discontinuation"]
    role: Literal["hard_gate", "diagnostic_only"]
    treated: TrialDevPublicIntervalV1
    control: TrialDevPublicIntervalV1
    excess: TrialDevPublicIntervalV1
    absolute_limit: float
    excess_limit: float


class TrialDevPublicCandidateDecisionV1(_FrozenPhaseReplayModel):
    """Public evidence supporting one candidate-specific action set."""

    candidate_arm_id: str = Field(..., min_length=1)
    acceptable_action_ids: tuple[str, ...] = Field(..., min_length=1)
    safety_state: Literal["acceptable", "unacceptable", "indeterminate"]
    efficacy: TrialDevPublicIntervalV1 | None = None
    minimum_efficacy_benefit: float | None = None
    safety_components: tuple[TrialDevPublicSafetyComponentV1, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_candidate(self) -> TrialDevPublicCandidateDecisionV1:
        """Require complete, unique candidate evidence."""

        if (self.efficacy is None) != (self.minimum_efficacy_benefit is None):
            raise ValueError("Efficacy evidence and its minimum benefit must be declared together.")
        if len(set(self.acceptable_action_ids)) != len(self.acceptable_action_ids):
            raise ValueError("Candidate acceptable_action_ids must be unique.")
        component_ids = tuple(component.component_id for component in self.safety_components)
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("Candidate safety components must be unique.")
        return self


class TrialDevPublicPhaseReplayRecordV1(_FrozenPhaseReplayModel):
    """One randomized-phase replay derived exclusively from public evidence."""

    schema_id: Literal["trialagentbench.trialdev_public_phase_replay/v1"] = (
        "trialagentbench.trialdev_public_phase_replay/v1"
    )
    scenario_id: str = Field(..., min_length=1)
    world_seed: int = Field(..., ge=0)
    trial_seed: int = Field(..., ge=0)
    request_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    trial_output_path: str = Field(..., min_length=1)
    phase_id: Literal["phase1", "phase2", "phase3"]
    endpoint_id: str | None = Field(default=None, min_length=1)
    treatment_discontinuation_strategy: (
        Literal["treatment_policy", "while_on_treatment", "composite_discontinuation"] | None
    ) = None
    follow_up_days: int = Field(..., ge=1)
    target_sample_size: int = Field(..., ge=1)
    allocation_ratio: str = Field(..., min_length=1)
    objective_ids: tuple[str, ...] = Field(..., min_length=1)
    candidate_drug_ids: tuple[str, ...] = Field(..., min_length=1)
    acceptable_action_ids: tuple[str, ...] = Field(..., min_length=1)
    stop_action_ids: tuple[str, ...] = Field(..., min_length=1)
    advance_action_ids: tuple[str, ...] = Field(..., min_length=1)
    sensitivity_action_sets: dict[str, tuple[str, ...]] = Field(..., min_length=1)
    public_decision_witness_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    public_source_checksums: dict[str, str] = Field(..., min_length=1)
    candidate_decision_evidence: tuple[TrialDevPublicCandidateDecisionV1, ...] = Field(..., min_length=1)
    public_safety_state: Literal["acceptable", "unacceptable", "indeterminate"]
    design_adequate: bool
    design_failures: tuple[str, ...]
    design_frontier: tuple[TrialDevDesignFrontierPointV1, ...] = Field(..., min_length=1)
    design_on_frontier: bool
    design_dominated_by_frontier: bool
    minimum_frontier_participants: int = Field(..., ge=1)
    minimum_frontier_follow_up_days: int = Field(..., ge=1)
    participant_excess_vs_minimum: int = Field(..., ge=0)
    participant_shortage_vs_minimum: int = Field(..., ge=0)
    follow_up_excess_days_vs_minimum: int = Field(..., ge=0)
    follow_up_shortage_days_vs_minimum: int = Field(..., ge=0)
    achieved_power: float | None = Field(default=None, ge=0.0, le=1.0)
    target_power: float | None = Field(default=None, ge=0.0, le=1.0)
    achieved_safety_absolute_risk_power: float = Field(..., ge=0.0, le=1.0)
    achieved_safety_excess_risk_power: float = Field(..., ge=0.0, le=1.0)
    target_safety_decision_power: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_record(self) -> TrialDevPublicPhaseReplayRecordV1:
        """Require complete public action, candidate, design, and checksum evidence."""

        if (self.phase_id == "phase1") != (self.treatment_discontinuation_strategy is None):
            raise ValueError(
                "Treatment-discontinuation strategy must be absent in phase1 and explicit in phase2/phase3."
            )
        if self.trial_output_path.startswith("/") or ".." in self.trial_output_path.split("/"):
            raise ValueError("trial_output_path must be safe and relative to the replay output root.")
        if set(self.public_source_checksums) != _PHASE_REPLAY_SOURCE_PATHS:
            raise ValueError("Public phase replay requires the exact checksummed source set.")
        if any(
            len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum.lower())
            for checksum in self.public_source_checksums.values()
        ):
            raise ValueError("Public phase replay source checksums must be SHA-256 hex digests.")
        stop = set(self.stop_action_ids)
        advance = set(self.advance_action_ids)
        domain = stop | advance
        if stop & advance:
            raise ValueError("Public phase-replay stop and advance actions must be disjoint.")
        if not set(self.acceptable_action_ids) <= domain:
            raise ValueError("Public acceptable actions must belong to the action domain.")
        if any(not actions or not set(actions) <= domain for actions in self.sensitivity_action_sets.values()):
            raise ValueError("Public sensitivity action sets must be non-empty action-domain subsets.")
        if len(set(self.candidate_drug_ids)) != len(self.candidate_drug_ids):
            raise ValueError("Phase-replay candidate_drug_ids must be unique.")
        evidence_ids = tuple(row.candidate_arm_id for row in self.candidate_decision_evidence)
        if len(evidence_ids) != len(set(evidence_ids)) or set(evidence_ids) != set(self.candidate_drug_ids):
            raise ValueError("Public candidate evidence must cover every requested candidate exactly once.")
        candidate_actions = set().union(*(set(row.acceptable_action_ids) for row in self.candidate_decision_evidence))
        if candidate_actions != set(self.acceptable_action_ids):
            raise ValueError("Candidate evidence must reproduce the aggregate acceptable action set.")
        if self.design_adequate != (not self.design_failures):
            raise ValueError("Public design adequacy must agree with design failures.")
        if self.design_on_frontier and (not self.design_adequate or self.design_dominated_by_frontier):
            raise ValueError("A public frontier design must be adequate and nondominated.")
        minimum_n = min(point.target_sample_size for point in self.design_frontier)
        minimum_follow_up = min(point.follow_up_days for point in self.design_frontier)
        if (self.minimum_frontier_participants, self.minimum_frontier_follow_up_days) != (
            minimum_n,
            minimum_follow_up,
        ):
            raise ValueError("Public design minima must replay from the declared frontier.")
        submitted_key = (self.target_sample_size, self.follow_up_days, self.allocation_ratio)
        frontier_keys = {
            (point.target_sample_size, point.follow_up_days, point.allocation_ratio) for point in self.design_frontier
        }
        if self.design_on_frontier and submitted_key not in frontier_keys:
            raise ValueError("A design marked on-frontier must match an exact frontier coordinate.")
        n_delta = self.target_sample_size - minimum_n
        follow_up_delta = self.follow_up_days - minimum_follow_up
        if (self.participant_excess_vs_minimum, self.participant_shortage_vs_minimum) != (
            max(0, n_delta),
            max(0, -n_delta),
        ):
            raise ValueError("Public participant deviations must replay from the frontier.")
        if (self.follow_up_excess_days_vs_minimum, self.follow_up_shortage_days_vs_minimum) != (
            max(0, follow_up_delta),
            max(0, -follow_up_delta),
        ):
            raise ValueError("Public follow-up deviations must replay from the frontier.")
        if any(
            not path or len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum)
            for path, checksum in self.public_source_checksums.items()
        ):
            raise ValueError("Public source checksums must be named SHA-256 digests.")
        return self


__all__ = [
    "TrialDevPhaseReplayCaseV1",
    "TrialDevPublicCandidateDecisionV1",
    "TrialDevPublicIntervalV1",
    "TrialDevPublicPhaseReplayRecordV1",
    "TrialDevPublicSafetyComponentV1",
]
