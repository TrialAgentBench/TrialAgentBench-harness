"""Canonical contracts for TrialDev sequential decisions under uncertainty."""

from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self, cast  # type: ignore[attr-defined]

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from pydantic.json_schema import SkipJsonSchema
from pydantic.types import JsonValue

from trialagentbench_harness.io.checksums import canonical_payload_sha256

TrialDevStreamIdV1 = Literal[
    "single_asset_development",
    "bounded_portfolio_reallocation",
]
TrialDevCheckpointIdV1 = Literal[
    "observational_review",
    "early_safety_study",
    "joint_early_study_review",
    "proof_of_concept",
    "lead_proof_of_concept_review",
    "promoted_reserve_proof_of_concept_review",
    "confirmation",
]
TrialDevActionIdV1 = Literal[
    "nominate_for_early_study",
    "withhold_nomination",
    "select_lead_and_reserve",
    "withhold_selection",
    "advance_to_proof_of_concept",
    "advance_lead_to_proof_of_concept",
    "promote_reserve_to_proof_of_concept",
    "advance_to_confirmation",
    "advance_active_to_confirmation",
    "stop_development",
    "terminate_portfolio",
    "declare_success",
    "declare_failure",
    "declare_inconclusive",
]
TrialDevSingleAssetActionIdV1 = Literal[
    "nominate_for_early_study",
    "withhold_nomination",
    "advance_to_proof_of_concept",
    "advance_to_confirmation",
    "stop_development",
    "declare_success",
    "declare_failure",
    "declare_inconclusive",
]
TrialDevPortfolioActionIdV1 = Literal[
    "select_lead_and_reserve",
    "withhold_selection",
    "advance_lead_to_proof_of_concept",
    "promote_reserve_to_proof_of_concept",
    "advance_active_to_confirmation",
    "terminate_portfolio",
    "declare_success",
    "declare_failure",
    "declare_inconclusive",
]
TrialDevTerminalDispositionV1 = Literal[
    "active",
    "withheld",
    "stopped",
    "success",
    "failure",
    "inconclusive",
]
TrialDevReachStatusV1 = Literal["reached", "structural_nonreach", "terminal"]
TrialDevSubmissionStatusV1 = Literal["accepted", "invalid", "missing", "not_applicable"]
TrialDevAnalysisStatusV1 = Literal["estimable", "non_estimable", "invalid", "missing", "not_applicable"]
TrialDevExecutionStatusV1 = Literal[
    "completed",
    "model_noncompletion",
    "infrastructure_failure",
    "not_applicable",
]
TrialDevEvidenceKindV1 = Literal["protocol", "dataset", "analysis", "decision"]
TrialDevAssetEligibilityStatusV1 = Literal["eligible", "permanently_ineligible"]
TrialDevAssetEligibilityReasonV1 = Literal[
    "none",
    "safety_clear_fail",
]
TrialDevRuleDomainV1 = Literal["efficacy", "safety"]
TrialDevRuleDirectionV1 = Literal["minimum", "maximum"]
TrialDevRuleClassificationV1 = Literal["clear_pass", "clear_fail", "indeterminate"]
TrialDevIdentificationStatusV1 = Literal["identified", "not_identified"]


def _contract_checksum(model: BaseModel) -> str:
    payload = model.model_dump(mode="json", exclude_none=True, exclude={"checksum"})
    return cast(str, canonical_payload_sha256(cast(JsonValue, payload)))


class _ChecksummedContractV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def bind_checksum(self) -> Self:
        """Bind the canonical payload checksum and reject mutated records."""

        expected = _contract_checksum(self)
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("Contract checksum does not match its canonical payload.")
        object.__setattr__(self, "checksum", expected)
        return self


class TrialDevEvidenceReferenceV1(_ChecksummedContractV1):
    """One participant-visible evidence artifact used at a checkpoint."""

    schema_id: Literal["trialagentbench.trialdev_evidence_reference/v1"] = (
        "trialagentbench.trialdev_evidence_reference/v1"
    )
    evidence_id: str = Field(..., min_length=1)
    evidence_kind: TrialDevEvidenceKindV1
    checkpoint_id: TrialDevCheckpointIdV1
    asset_id: str | None = Field(
        default=None,
        min_length=1,
        description="Asset identifier, or null for evidence shared across the programme.",
    )
    evidence_protocol_id: str = Field(..., min_length=1)
    evidence_protocol_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    source_family_id: str = Field(..., min_length=1)
    world_id: str = Field(..., min_length=1)
    generation_seed: int | None = Field(
        default=None,
        ge=0,
        description="Optional disclosed generation seed for a precommitted evidence dataset.",
    )
    relative_path: str = Field(..., min_length=1)
    artifact_sha256: str = Field(
        ...,
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 of the referenced file; distinct from this evidence record's checksum.",
    )

    @model_validator(mode="after")
    def validate_relative_path(self) -> Self:
        """Require a normalized participant-visible relative path."""

        if self.generation_seed is not None and self.evidence_kind != "dataset":
            raise ValueError("Only dataset evidence may declare generation_seed.")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != self.relative_path:
            raise ValueError("Evidence paths must be normalized relative paths.")
        return self


class TrialDevResourceScheduleV1(_ChecksummedContractV1):
    """Normalized programme resource units for the bounded portfolio stream."""

    schema_id: Literal["trialagentbench.trialdev_resource_schedule/v1"] = (
        "trialagentbench.trialdev_resource_schedule/v1"
    )
    early_study_units: int = Field(1, gt=0)
    proof_of_concept_units: int = Field(2, gt=0)
    confirmation_units: int = Field(4, gt=0)
    budgets: tuple[int, ...] = Field((8, 10), min_length=1)
    maximum_switches: Literal[1] = 1

    @model_validator(mode="after")
    def validate_schedule(self) -> Self:
        """Require unique budgets capable of funding the ordinary path."""

        if len(self.budgets) != len(set(self.budgets)) or tuple(sorted(self.budgets)) != self.budgets:
            raise ValueError("Resource budgets must be unique and increasing.")
        ordinary_path = 2 * self.early_study_units + self.proof_of_concept_units + self.confirmation_units
        if any(budget < ordinary_path for budget in self.budgets):
            raise ValueError("Every resource budget must fund two early studies and the ordinary path.")
        return self


class TrialDevPolicyBindingV1(_ChecksummedContractV1):
    """Public policy and design identities that govern one programme view."""

    schema_id: Literal["trialagentbench.trialdev_policy_binding/v1"] = "trialagentbench.trialdev_policy_binding/v1"
    stream_id: TrialDevStreamIdV1
    objective_id: str = Field(..., min_length=1)
    objective_policy_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    action_policy_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    design_menu_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    resource_schedule: TrialDevResourceScheduleV1 | None = None
    resource_budget_units: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_stream_policy(self) -> Self:
        """Require resources only for the bounded portfolio stream."""

        if self.stream_id == "single_asset_development" and (
            self.resource_schedule is not None or self.resource_budget_units is not None
        ):
            raise ValueError("Single-asset development does not use a portfolio resource policy.")
        if self.stream_id == "bounded_portfolio_reallocation":
            if self.resource_schedule is None or self.resource_budget_units not in self.resource_schedule.budgets:
                raise ValueError("A portfolio budget must be declared by its public resource schedule.")
        return self


class TrialDevLegalActionSpecV1(BaseModel):
    """One publicly legal action and its submission requirements."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: TrialDevActionIdV1
    action_kind: Literal["allocate", "advance", "promote", "stop", "terminal"]
    requires_target_asset: bool = False
    requires_reserve_asset: bool = False
    consumes_switch: bool = False

    @model_validator(mode="after")
    def validate_action_shape(self) -> Self:
        """Require action metadata to agree with the action's semantics."""

        if self.requires_reserve_asset and not self.requires_target_asset:
            raise ValueError("An action requiring a reserve must also require a target asset.")
        if self.consumes_switch and self.action_id != "promote_reserve_to_proof_of_concept":
            raise ValueError("Only reserve promotion may consume the portfolio switch.")
        return self


class TrialDevSingleAssetLegalActionSpecV1(TrialDevLegalActionSpecV1):
    """Legal action metadata exposed by the single-asset stream."""

    action_id: TrialDevSingleAssetActionIdV1
    requires_reserve_asset: SkipJsonSchema[Literal[False]] = Field(default=False, exclude=True)
    consumes_switch: SkipJsonSchema[Literal[False]] = Field(default=False, exclude=True)


class TrialDevPortfolioLegalActionSpecV1(TrialDevLegalActionSpecV1):
    """Legal action metadata exposed by bounded portfolio reallocation."""


class TrialDevCheckpointActionPolicyV1(_ChecksummedContractV1):
    """Public legal-action policy for one reachable checkpoint."""

    schema_id: Literal["trialagentbench.trialdev_checkpoint_action_policy/v1"] = (
        "trialagentbench.trialdev_checkpoint_action_policy/v1"
    )
    stream_id: TrialDevStreamIdV1
    checkpoint_id: TrialDevCheckpointIdV1
    policy_binding_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    actions: tuple[TrialDevLegalActionSpecV1, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_actions(self) -> Self:
        """Reject duplicate or cross-stream actions."""

        action_ids = tuple(action.action_id for action in self.actions)
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("Checkpoint action policy contains duplicate actions.")
        portfolio_only = {
            "select_lead_and_reserve",
            "withhold_selection",
            "advance_lead_to_proof_of_concept",
            "promote_reserve_to_proof_of_concept",
            "advance_active_to_confirmation",
            "terminate_portfolio",
        }
        if self.stream_id == "single_asset_development" and portfolio_only.intersection(action_ids):
            raise ValueError("Single-asset policy cannot contain portfolio actions.")
        single_asset_only = {
            "nominate_for_early_study",
            "withhold_nomination",
            "advance_to_proof_of_concept",
            "advance_to_confirmation",
        }
        if self.stream_id == "bounded_portfolio_reallocation" and single_asset_only.intersection(action_ids):
            raise ValueError("Portfolio policy cannot contain single-asset actions.")
        return self


class TrialDevSingleAssetCheckpointActionPolicyV1(TrialDevCheckpointActionPolicyV1):
    """Public legal-action policy for one single-asset checkpoint."""

    stream_id: Literal["single_asset_development"] = "single_asset_development"
    actions: tuple[TrialDevSingleAssetLegalActionSpecV1, ...] = Field(..., min_length=1)


class TrialDevPortfolioCheckpointActionPolicyV1(TrialDevCheckpointActionPolicyV1):
    """Public legal-action policy for one portfolio checkpoint."""

    stream_id: Literal["bounded_portfolio_reallocation"] = "bounded_portfolio_reallocation"
    actions: tuple[TrialDevPortfolioLegalActionSpecV1, ...] = Field(..., min_length=1)


class TrialDevActionSelectionV1(_ChecksummedContractV1):
    """One participant action justified from contemporaneous evidence."""

    schema_id: Literal["trialagentbench.trialdev_action_selection/v1"] = "trialagentbench.trialdev_action_selection/v1"
    state_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    checkpoint_id: TrialDevCheckpointIdV1
    action_id: TrialDevActionIdV1
    target_asset_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Asset selected by an action whose public policy requires a target. "
            "Omit it when the current lead, reserve, or active asset is already implied."
        ),
    )
    reserve_asset_id: str | None = Field(
        default=None,
        min_length=1,
        description="Reserve selected with a new lead; omit it for every other action.",
    )
    analysis_method_id: str = Field(
        ...,
        min_length=1,
        description="Method route used to derive the submitted decision evidence.",
    )
    supporting_evidence_ids: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="Exact evidence_id values from the current programme state's evidence records.",
    )
    proposed_trial_plan_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    justification: str = Field(..., min_length=1, max_length=4000)

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Require asset roles only where the selected action needs them."""

        if len(self.supporting_evidence_ids) != len(set(self.supporting_evidence_ids)):
            raise ValueError("supporting_evidence_ids must be unique.")
        if self.action_id == "select_lead_and_reserve":
            if self.target_asset_id is None or self.reserve_asset_id is None:
                raise ValueError("Lead-reserve selection requires both asset identifiers.")
            if self.target_asset_id == self.reserve_asset_id:
                raise ValueError("Lead and reserve must be different assets.")
        elif self.reserve_asset_id is not None:
            raise ValueError("reserve_asset_id is valid only for lead-reserve selection.")
        target_required = {
            "nominate_for_early_study",
            "select_lead_and_reserve",
        }
        if self.action_id in target_required and self.target_asset_id is None:
            raise ValueError(f"{self.action_id} requires target_asset_id.")
        return self


class TrialDevSingleAssetActionSelectionV1(TrialDevActionSelectionV1):
    """Selected action for irreversible single-asset development."""

    action_id: TrialDevSingleAssetActionIdV1
    reserve_asset_id: SkipJsonSchema[None] = Field(default=None, exclude=True)


class TrialDevPortfolioActionSelectionV1(TrialDevActionSelectionV1):
    """Selected action for bounded portfolio reallocation."""

    action_id: TrialDevPortfolioActionIdV1


class TrialDevAssetEligibilityV1(BaseModel):
    """Evidence-bound eligibility disposition for one candidate asset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str = Field(..., min_length=1)
    status: TrialDevAssetEligibilityStatusV1
    reason: TrialDevAssetEligibilityReasonV1
    policy_rule_id: str = Field(..., min_length=1)
    evidence_reference_checksums: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description=(
            "Exact checksum values from the current programme state's evidence records; " "record order is immaterial."
        ),
    )

    @model_validator(mode="after")
    def validate_eligibility(self) -> Self:
        """Bind permanent exclusion to a named failure reason and unique evidence."""

        if len(self.evidence_reference_checksums) != len(set(self.evidence_reference_checksums)):
            raise ValueError("Asset eligibility evidence references must be unique.")
        if (self.status == "eligible") != (self.reason == "none"):
            raise ValueError("Only an eligible asset may use reason='none'.")
        return self


class TrialDevCheckpointOutcomeV1(BaseModel):
    """Orthogonal reach, submission, analysis, and execution outcomes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reach_status: TrialDevReachStatusV1
    submission_status: TrialDevSubmissionStatusV1
    analysis_status: TrialDevAnalysisStatusV1
    execution_status: TrialDevExecutionStatusV1
    asset_eligibility: tuple[TrialDevAssetEligibilityV1, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_outcome_axes(self) -> Self:
        """Prevent structural nonreach or terminal states from masquerading as attempts."""

        if self.reach_status != "reached":
            if self.submission_status != "not_applicable" or self.analysis_status != "not_applicable":
                raise ValueError("Unreached checkpoints cannot contain submission or analysis outcomes.")
            if self.execution_status != "not_applicable":
                raise ValueError("Unreached checkpoints cannot contain execution outcomes.")
        elif self.execution_status in {"model_noncompletion", "infrastructure_failure"}:
            if self.submission_status != "missing" or self.analysis_status != "missing":
                raise ValueError("Noncompleted execution requires missing submission and analysis outcomes.")
        asset_ids = tuple(item.asset_id for item in self.asset_eligibility)
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("A checkpoint outcome may classify each asset at most once.")
        if self.asset_eligibility and (
            self.reach_status != "reached"
            or self.submission_status != "accepted"
            or self.analysis_status not in {"estimable", "non_estimable"}
            or self.execution_status != "completed"
        ):
            raise ValueError("Asset eligibility requires a completed checkpoint with an accepted valid analysis.")
        return self


class TrialDevCheckpointHistoryEntryV1(_ChecksummedContractV1):
    """One append-only transition in a TrialDev programme history."""

    schema_id: Literal["trialagentbench.trialdev_checkpoint_history_entry/v1"] = (
        "trialagentbench.trialdev_checkpoint_history_entry/v1"
    )
    state_index: int = Field(..., ge=0)
    checkpoint_id: TrialDevCheckpointIdV1
    evidence_reference_checksums: tuple[str, ...] = Field(default_factory=tuple)
    selected_action: TrialDevActionSelectionV1 | None = None
    outcome: TrialDevCheckpointOutcomeV1
    active_asset_id: str | None = Field(default=None, min_length=1)
    lead_asset_id: str | None = Field(default=None, min_length=1)
    reserve_asset_id: str | None = Field(default=None, min_length=1)
    retired_asset_ids: tuple[str, ...] = Field(default_factory=tuple)
    permanently_ineligible_asset_ids: tuple[str, ...] = Field(default_factory=tuple)
    resources_spent_units: int = Field(default=0, ge=0)
    previous_entry_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_history_entry(self) -> Self:
        """Require a contiguous predecessor convention and unique references."""

        if (self.state_index == 0) != (self.previous_entry_checksum is None):
            raise ValueError("History predecessor is required exactly after the first entry.")
        if len(self.evidence_reference_checksums) != len(set(self.evidence_reference_checksums)):
            raise ValueError("evidence_reference_checksums must be unique.")
        if len(self.retired_asset_ids) != len(set(self.retired_asset_ids)):
            raise ValueError("retired_asset_ids must be unique.")
        if len(self.permanently_ineligible_asset_ids) != len(set(self.permanently_ineligible_asset_ids)):
            raise ValueError("permanently_ineligible_asset_ids must be unique.")
        if not set(self.permanently_ineligible_asset_ids) <= set(self.retired_asset_ids):
            raise ValueError("Permanently ineligible assets must also be retired.")
        return self


class TrialDevSingleAssetCheckpointHistoryEntryV1(TrialDevCheckpointHistoryEntryV1):
    """History entry for the irreversible single-asset stream."""

    lead_asset_id: SkipJsonSchema[None] = Field(default=None, exclude=True)
    reserve_asset_id: SkipJsonSchema[None] = Field(default=None, exclude=True)
    resources_spent_units: SkipJsonSchema[Literal[0]] = Field(default=0, exclude=True)
    permanently_ineligible_asset_ids: SkipJsonSchema[tuple[()]] = Field(default=(), exclude=True)
    selected_action: TrialDevSingleAssetActionSelectionV1 | None = None


class TrialDevPortfolioCheckpointHistoryEntryV1(TrialDevCheckpointHistoryEntryV1):
    """History entry for bounded portfolio reallocation."""

    selected_action: TrialDevPortfolioActionSelectionV1 | None = None


class TrialDevProgrammeStateV1(_ChecksummedContractV1):
    """Common immutable state and invariants shared by both TrialDev streams."""

    schema_id: Literal["trialagentbench.trialdev_programme_state/v1"] = "trialagentbench.trialdev_programme_state/v1"
    programme_id: str = Field(..., min_length=1)
    scenario_id: str = Field(..., min_length=1)
    stream_id: TrialDevStreamIdV1
    current_checkpoint_id: TrialDevCheckpointIdV1
    candidate_asset_ids: tuple[str, ...] = Field(..., min_length=1)
    nominated_asset_id: str | None = Field(default=None, min_length=1)
    lead_asset_id: str | None = Field(default=None, min_length=1)
    reserve_asset_id: str | None = Field(default=None, min_length=1)
    active_asset_id: str | None = Field(default=None, min_length=1)
    retired_asset_ids: tuple[str, ...] = Field(default_factory=tuple)
    permanently_ineligible_asset_ids: tuple[str, ...] = Field(default_factory=tuple)
    terminal_disposition: TrialDevTerminalDispositionV1 = "active"
    policy_binding: TrialDevPolicyBindingV1
    evidence: tuple[TrialDevEvidenceReferenceV1, ...] = Field(default_factory=tuple)
    history: tuple[TrialDevCheckpointHistoryEntryV1, ...] = Field(default_factory=tuple)
    resource_spent_units: int = Field(default=0, ge=0)
    switch_count: int = Field(default=0, ge=0, le=1)
    previous_state_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_programme_state(self) -> Self:
        """Enforce roles, resources, and the complete append-only history chain."""

        candidates = self.candidate_asset_ids
        retired = self.retired_asset_ids
        ineligible = self.permanently_ineligible_asset_ids
        if (
            len(candidates) != len(set(candidates))
            or len(retired) != len(set(retired))
            or len(ineligible) != len(set(ineligible))
        ):
            raise ValueError("Candidate, retired, and ineligible asset identifiers must be unique.")
        if not set(retired) <= set(candidates):
            raise ValueError("Retired assets must belong to the programme candidate set.")
        if not set(ineligible) <= set(retired):
            raise ValueError("Permanently ineligible assets must be retired candidates.")
        roles = tuple(
            value
            for value in (
                self.nominated_asset_id,
                self.lead_asset_id,
                self.reserve_asset_id,
                self.active_asset_id,
            )
            if value is not None
        )
        if not set(roles) <= set(candidates):
            raise ValueError("Programme roles must identify candidate assets.")
        if self.terminal_disposition == "active" and self.active_asset_id in set(retired):
            raise ValueError("The active asset cannot be retired while the programme is active.")
        if self.policy_binding.stream_id != self.stream_id:
            raise ValueError("Programme state and policy binding must use the same stream.")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        evidence_checksums = tuple(item.checksum for item in self.evidence)
        if len(evidence_ids) != len(set(evidence_ids)) or len(evidence_checksums) != len(set(evidence_checksums)):
            raise ValueError("Programme evidence references must be unique.")
        if any(item.asset_id is not None and item.asset_id not in candidates for item in self.evidence):
            raise ValueError("Programme evidence must identify a candidate asset.")
        evidence_checksum_set = set(evidence_checksums)
        if any(not set(entry.evidence_reference_checksums) <= evidence_checksum_set for entry in self.history):
            raise ValueError("Programme history references evidence absent from programme state.")
        if self.terminal_disposition == "active" and (
            not self.evidence or self.evidence[-1].checkpoint_id != self.current_checkpoint_id
        ):
            raise ValueError("The most recent evidence must belong to the current checkpoint.")
        if self.stream_id == "single_asset_development":
            if self.lead_asset_id is not None or self.reserve_asset_id is not None or self.switch_count != 0:
                raise ValueError("Single-asset state cannot contain portfolio roles or switches.")
            if self.resource_spent_units != 0:
                raise ValueError("Single-asset state does not spend portfolio resource units.")
            if self.nominated_asset_id is not None and self.active_asset_id != self.nominated_asset_id:
                raise ValueError("The nominated single asset must remain the active asset.")
        else:
            if self.nominated_asset_id is not None:
                raise ValueError("Portfolio state uses lead and reserve roles, not nominated_asset_id.")
            if (self.lead_asset_id is None) != (self.reserve_asset_id is None):
                raise ValueError("Portfolio lead and reserve roles must be assigned together.")
            if self.lead_asset_id is not None and self.lead_asset_id == self.reserve_asset_id:
                raise ValueError("Portfolio lead and reserve must be different assets.")
            if self.active_asset_id is not None and self.active_asset_id not in {
                self.lead_asset_id,
                self.reserve_asset_id,
            }:
                raise ValueError("Portfolio active asset must be the lead or reserve.")
            budget = self.policy_binding.resource_budget_units
            if budget is None or self.resource_spent_units > budget:
                raise ValueError("Portfolio resource spending must not exceed its disclosed budget.")
            if self.switch_count == 1 and self.active_asset_id != self.reserve_asset_id:
                raise ValueError("After the one promotion, the reserve must be the active asset.")
        expected_previous: str | None = None
        previous_retired: set[str] = set()
        previous_ineligible: set[str] = set()
        previous_spend = 0
        for index, entry in enumerate(self.history):
            if entry.state_index != index or entry.previous_entry_checksum != expected_previous:
                raise ValueError("Programme history must form a contiguous checksum chain.")
            if not previous_retired <= set(entry.retired_asset_ids):
                raise ValueError("Retired assets cannot return in later history entries.")
            if not previous_ineligible <= set(entry.permanently_ineligible_asset_ids):
                raise ValueError("Permanent asset ineligibility cannot be reversed.")
            if entry.resources_spent_units < previous_spend:
                raise ValueError("Resource spending cannot decrease across programme history.")
            previous_retired = set(entry.retired_asset_ids)
            previous_ineligible = set(entry.permanently_ineligible_asset_ids)
            previous_spend = entry.resources_spent_units
            expected_previous = entry.checksum
        if self.history:
            latest = self.history[-1]
            if (
                tuple(latest.retired_asset_ids) != retired
                or tuple(latest.permanently_ineligible_asset_ids) != ineligible
                or latest.resources_spent_units != self.resource_spent_units
            ):
                raise ValueError("Current retirement and resource state must agree with the latest history entry.")
            if (
                latest.active_asset_id != self.active_asset_id
                or latest.lead_asset_id != self.lead_asset_id
                or latest.reserve_asset_id != self.reserve_asset_id
            ):
                raise ValueError("Current asset roles must agree with the latest history entry.")
        if (not self.history) != (self.previous_state_checksum is None):
            raise ValueError("previous_state_checksum is required exactly when the state has history.")
        return self


class TrialDevSingleAssetProgrammeStateV1(TrialDevProgrammeStateV1):
    """Public state for irreversible development of one nominated asset."""

    stream_id: Literal["single_asset_development"] = "single_asset_development"
    lead_asset_id: SkipJsonSchema[None] = Field(default=None, exclude=True)
    reserve_asset_id: SkipJsonSchema[None] = Field(default=None, exclude=True)
    resource_spent_units: SkipJsonSchema[Literal[0]] = Field(default=0, exclude=True)
    switch_count: SkipJsonSchema[Literal[0]] = Field(default=0, exclude=True)
    permanently_ineligible_asset_ids: SkipJsonSchema[tuple[()]] = Field(default=(), exclude=True)
    history: tuple[TrialDevSingleAssetCheckpointHistoryEntryV1, ...] = Field(default_factory=tuple)


class TrialDevPortfolioProgrammeStateV1(TrialDevProgrammeStateV1):
    """Public state for bounded lead-reserve portfolio reallocation."""

    stream_id: Literal["bounded_portfolio_reallocation"] = "bounded_portfolio_reallocation"
    nominated_asset_id: SkipJsonSchema[None] = Field(default=None, exclude=True)
    history: tuple[TrialDevPortfolioCheckpointHistoryEntryV1, ...] = Field(default_factory=tuple)


class TrialDevPortfolioEvidenceIndexV1(_ChecksummedContractV1):
    """Evaluator-held index of precommitted portfolio evidence episodes."""

    schema_id: Literal["trialagentbench.trialdev_portfolio_evidence_index/v1"] = (
        "trialagentbench.trialdev_portfolio_evidence_index/v1"
    )
    scenario_id: str = Field(..., min_length=1)
    source_identity: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    world_id: str = Field(..., min_length=1)
    candidate_asset_ids: tuple[str, str, str]
    evidence: tuple[TrialDevEvidenceReferenceV1, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_index(self) -> Self:
        """Require one asset-specific episode for each indexed checkpoint key."""

        if len(set(self.candidate_asset_ids)) != 3:
            raise ValueError("Portfolio evidence index requires three unique candidate assets.")
        keys: list[tuple[str, TrialDevCheckpointIdV1]] = []
        for item in self.evidence:
            if item.asset_id is None:
                raise ValueError("Portfolio evidence episodes must identify one asset.")
            if item.source_family_id != self.source_identity or item.world_id != self.world_id:
                raise ValueError("Portfolio evidence provenance must match the evidence index.")
            if item.asset_id not in self.candidate_asset_ids:
                raise ValueError("Portfolio evidence identifies an asset outside the candidate set.")
            keys.append((item.asset_id, item.checkpoint_id))
        if len(keys) != len(set(keys)):
            raise ValueError("Portfolio evidence index contains duplicate asset-checkpoint episodes.")
        branch_checkpoints: tuple[TrialDevCheckpointIdV1, ...] = (
            "joint_early_study_review",
            "lead_proof_of_concept_review",
            "promoted_reserve_proof_of_concept_review",
            "confirmation",
        )
        expected = {
            (asset_id, checkpoint_id) for asset_id in self.candidate_asset_ids for checkpoint_id in branch_checkpoints
        }
        if set(keys) != expected:
            missing = sorted(expected - set(keys))
            extra = sorted(set(keys) - expected)
            raise ValueError(f"Portfolio evidence branch census is incomplete: missing={missing!r} extra={extra!r}.")
        return self

    def resolve(
        self,
        *,
        checkpoint_id: TrialDevCheckpointIdV1,
        asset_ids: tuple[str, ...],
    ) -> tuple[TrialDevEvidenceReferenceV1, ...]:
        """Return exactly the precommitted episodes for a reached branch."""

        if not asset_ids or len(asset_ids) != len(set(asset_ids)):
            raise ValueError("Portfolio evidence resolution requires unique asset identifiers.")
        by_key = {(item.asset_id, item.checkpoint_id): item for item in self.evidence if item.asset_id is not None}
        missing = [asset_id for asset_id in asset_ids if (asset_id, checkpoint_id) not in by_key]
        if missing:
            raise ValueError(f"Portfolio evidence index lacks checkpoint={checkpoint_id!r} for assets={missing!r}.")
        return tuple(by_key[(asset_id, checkpoint_id)] for asset_id in asset_ids)


class TrialDevDecisionRuleEvidenceV1(_ChecksummedContractV1):
    """One interval and public threshold used to classify a decision domain."""

    schema_id: Literal["trialagentbench.trialdev_decision_rule_evidence/v1"] = (
        "trialagentbench.trialdev_decision_rule_evidence/v1"
    )
    rule_id: str = Field(..., min_length=1)
    asset_id: str = Field(..., min_length=1)
    domain: TrialDevRuleDomainV1
    direction: TrialDevRuleDirectionV1
    estimate: float
    lower_bound: float
    upper_bound: float
    threshold: float
    evidence_reference_checksums: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description=(
            "Exact checksum values from the current programme state's evidence records; " "record order is immaterial."
        ),
    )

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        """Require an ordered finite interval and unique evidence references."""

        values = (self.estimate, self.lower_bound, self.upper_bound, self.threshold)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Decision-rule values must be finite.")
        if not self.lower_bound <= self.estimate <= self.upper_bound:
            raise ValueError("Decision-rule estimate must lie within its interval.")
        expected_direction: TrialDevRuleDirectionV1 = "minimum" if self.domain == "efficacy" else "maximum"
        if self.direction != expected_direction:
            raise ValueError(f"Decision-rule domain {self.domain!r} requires direction={expected_direction!r}.")
        if len(self.evidence_reference_checksums) != len(set(self.evidence_reference_checksums)):
            raise ValueError("Decision-rule evidence references must be unique.")
        return self

    @property
    def classification(self) -> TrialDevRuleClassificationV1:
        """Classify the interval against its disclosed one-sided threshold."""

        if self.direction == "minimum":
            if self.lower_bound >= self.threshold:
                return "clear_pass"
            if self.upper_bound < self.threshold:
                return "clear_fail"
        else:
            if self.upper_bound <= self.threshold:
                return "clear_pass"
            if self.lower_bound > self.threshold:
                return "clear_fail"
        return "indeterminate"


class TrialDevObservationalCandidateEvidenceV1(_ChecksummedContractV1):
    """Method-specific estimates used to derive candidate allocation."""

    schema_id: Literal["trialagentbench.trialdev_observational_candidate_evidence/v1"] = (
        "trialagentbench.trialdev_observational_candidate_evidence/v1"
    )
    asset_id: str = Field(..., min_length=1)
    utility_estimate: float
    utility_lower_bound: float
    utility_upper_bound: float
    efficacy_estimate: float
    efficacy_lower_bound: float
    efficacy_upper_bound: float
    evidence_reference_checksums: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description=(
            "Exact checksum values from the current programme state's evidence records; " "record order is immaterial."
        ),
    )

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        """Require finite utility and unique public evidence references."""

        values = (
            self.utility_estimate,
            self.utility_lower_bound,
            self.utility_upper_bound,
            self.efficacy_estimate,
            self.efficacy_lower_bound,
            self.efficacy_upper_bound,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Candidate utility and efficacy estimates must be finite.")
        if not self.utility_lower_bound <= self.utility_estimate <= self.utility_upper_bound:
            raise ValueError("Candidate utility estimate must lie within its interval.")
        if not self.efficacy_lower_bound <= self.efficacy_estimate <= self.efficacy_upper_bound:
            raise ValueError("Candidate efficacy estimate must lie within its interval.")
        if len(self.evidence_reference_checksums) != len(set(self.evidence_reference_checksums)):
            raise ValueError("Candidate evidence references must be unique.")
        return self


class TrialDevPairContrastEvidenceV1(_ChecksummedContractV1):
    """Public uncertainty for one candidate pair, independent of identifier order."""

    schema_id: Literal["trialagentbench.trialdev_pair_contrast_evidence/v1"] = (
        "trialagentbench.trialdev_pair_contrast_evidence/v1"
    )
    lead_asset_id: str = Field(
        ...,
        min_length=1,
        description="One member of the candidate pair; this field does not assign the programme lead role.",
    )
    reserve_asset_id: str = Field(
        ...,
        min_length=1,
        description="The other member of the candidate pair; this field does not assign the reserve role.",
    )
    confidence_half_width: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Require a distinct pair and finite half-width."""

        if self.lead_asset_id == self.reserve_asset_id:
            raise ValueError("A pair contrast requires two distinct assets.")
        if not math.isfinite(self.confidence_half_width):
            raise ValueError("Pair contrast confidence half-width must be finite.")
        return self


class TrialDevObservationalDecisionEvidenceV1(_ChecksummedContractV1):
    """Submitted-method evidence for one observational allocation decision."""

    schema_id: Literal["trialagentbench.trialdev_observational_decision_evidence/v1"] = (
        "trialagentbench.trialdev_observational_decision_evidence/v1"
    )
    state_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    analysis_method_id: str = Field(
        ...,
        min_length=1,
        description="Exact methods[].method_route_id from observational_method_catalog.json.",
    )
    identification_status: TrialDevIdentificationStatusV1 = Field(
        ...,
        description=(
            "Whether the requested point comparison is identified under the method's declared assumptions "
            "and the public treatment-assignment provenance. This does not assert that an observational "
            "assumption has been empirically proven."
        ),
    )
    minimum_efficacy_gain: float = Field(
        ...,
        description=(
            "Minimum candidate-versus-control efficacy gain required by the public decision policy; "
            "this is distinct from the practical-equivalence margin between candidates."
        ),
    )
    practical_equivalence_margin: float = Field(..., ge=0.0)
    candidates: tuple[TrialDevObservationalCandidateEvidenceV1, ...] = Field(
        default_factory=tuple,
        description=(
            "Candidate estimates required when the point comparison is identified under the declared assumptions; "
            "otherwise empty. Each candidate record carries its own evidence references."
        ),
    )
    pair_contrasts: tuple[TrialDevPairContrastEvidenceV1, ...] = Field(
        default_factory=tuple,
        description=(
            "One confidence half-width for each candidate pair. Identifier order has no effect and does not imply "
            "a selected lead or reserve role; otherwise empty."
        ),
    )
    identification_evidence_reference_checksums: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Public treatment-assignment or support provenance required when the point comparison is "
            "not identified under the declared assumptions; copy exact checksum values from the current "
            "programme state's evidence records, not artifact_sha256 values. Must be empty when candidate "
            "estimates are supplied; record order is immaterial."
        ),
    )

    @model_validator(mode="after")
    def validate_observational_evidence(self) -> Self:
        """Require unique candidate and canonical pair evidence."""

        if not math.isfinite(self.minimum_efficacy_gain) or not math.isfinite(self.practical_equivalence_margin):
            raise ValueError("Observational policy thresholds must be finite.")
        asset_ids = tuple(item.asset_id for item in self.candidates)
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("Observational candidate evidence must be unique by asset.")
        pair_keys = tuple(tuple(sorted((item.lead_asset_id, item.reserve_asset_id))) for item in self.pair_contrasts)
        if len(pair_keys) != len(set(pair_keys)):
            raise ValueError("Candidate pair contrasts must be unique regardless of identifier order.")
        if self.identification_status == "identified":
            if not self.candidates:
                raise ValueError("Identified observational evidence requires candidate estimates.")
            if self.identification_evidence_reference_checksums:
                raise ValueError("Identified observational evidence must cite candidate-level evidence.")
        else:
            if self.candidates or self.pair_contrasts:
                raise ValueError("Nonidentified observational evidence cannot report causal candidate contrasts.")
            if not self.identification_evidence_reference_checksums:
                raise ValueError("Nonidentified observational evidence requires public identification evidence.")
        if len(self.identification_evidence_reference_checksums) != len(
            set(self.identification_evidence_reference_checksums)
        ):
            raise ValueError("Identification evidence references must be unique.")
        return self


class TrialDevRandomizedDecisionEvidenceV1(_ChecksummedContractV1):
    """Interval evidence used for one randomized checkpoint action."""

    schema_id: Literal["trialagentbench.trialdev_randomized_decision_evidence/v1"] = (
        "trialagentbench.trialdev_randomized_decision_evidence/v1"
    )
    state_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    analysis_method_id: str = Field(
        ...,
        min_length=1,
        description="Exact method_route_id for the current phase from phase_analysis_method_catalog.json.",
    )
    rules: tuple[TrialDevDecisionRuleEvidenceV1, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_rules(self) -> Self:
        """Require unique rule identities."""

        keys = tuple((rule.asset_id, rule.domain, rule.rule_id) for rule in self.rules)
        if len(keys) != len(set(keys)):
            raise ValueError("Randomized decision rules must have unique asset-domain identities.")
        return self


TrialDevProgrammeStateRecordV1 = Annotated[
    TrialDevSingleAssetProgrammeStateV1 | TrialDevPortfolioProgrammeStateV1,
    Field(discriminator="stream_id"),
]
TRIALDEV_PROGRAMME_STATE_ADAPTER_V1: TypeAdapter[TrialDevProgrammeStateRecordV1] = TypeAdapter(
    TrialDevProgrammeStateRecordV1
)


class TrialDevSupportedActionV1(_ChecksummedContractV1):
    """One concrete legal action variant considered by the evaluator."""

    schema_id: Literal["trialagentbench.trialdev_supported_action/v1"] = "trialagentbench.trialdev_supported_action/v1"
    action_id: TrialDevActionIdV1
    target_asset_id: str | None = Field(default=None, min_length=1)
    reserve_asset_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_variant(self) -> Self:
        """Require asset identifiers exactly for allocation actions."""

        if self.action_id == "select_lead_and_reserve":
            if self.target_asset_id is None or self.reserve_asset_id is None:
                raise ValueError("Lead-reserve selection requires two asset identifiers.")
            if self.target_asset_id == self.reserve_asset_id:
                raise ValueError("Lead and reserve must be distinct.")
        elif self.action_id == "nominate_for_early_study":
            if self.target_asset_id is None or self.reserve_asset_id is not None:
                raise ValueError("Nomination requires exactly one target asset.")
        elif self.target_asset_id is not None or self.reserve_asset_id is not None:
            raise ValueError("Only allocation actions may carry asset identifiers.")
        return self


class TrialDevSupportedActionSetV1(_ChecksummedContractV1):
    """Evaluator-held actions supported by public evidence and one valid method."""

    schema_id: Literal["trialagentbench.trialdev_supported_action_set/v1"] = (
        "trialagentbench.trialdev_supported_action_set/v1"
    )
    state_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    checkpoint_id: TrialDevCheckpointIdV1
    submitted_analysis_method_id: str = Field(..., min_length=1)
    policy_binding_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    legal_actions: tuple[TrialDevSupportedActionV1, ...] = Field(..., min_length=1)
    supported_actions: tuple[TrialDevSupportedActionV1, ...] = Field(..., min_length=1)
    public_evidence_checksums: tuple[str, ...] = Field(..., min_length=1)
    sensitivity_policy_id: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_supported_set(self) -> Self:
        """Require a nonempty subset derived only from named public evidence."""

        legal = tuple(item.checksum for item in self.legal_actions)
        supported = tuple(item.checksum for item in self.supported_actions)
        evidence = self.public_evidence_checksums
        if len(legal) != len(set(legal)) or len(supported) != len(set(supported)):
            raise ValueError("Legal and supported action variants must be unique.")
        if not set(supported) <= set(legal):
            raise ValueError("Supported action variants must be a subset of legal actions.")
        if len(evidence) != len(set(evidence)):
            raise ValueError("public_evidence_checksums must be unique.")
        return self


__all__ = [
    "TrialDevActionIdV1",
    "TrialDevActionSelectionV1",
    "TrialDevAssetEligibilityReasonV1",
    "TrialDevAssetEligibilityStatusV1",
    "TrialDevAssetEligibilityV1",
    "TrialDevAnalysisStatusV1",
    "TrialDevCheckpointActionPolicyV1",
    "TrialDevCheckpointHistoryEntryV1",
    "TrialDevCheckpointIdV1",
    "TrialDevCheckpointOutcomeV1",
    "TrialDevEvidenceReferenceV1",
    "TrialDevEvidenceKindV1",
    "TrialDevExecutionStatusV1",
    "TrialDevLegalActionSpecV1",
    "TrialDevDecisionRuleEvidenceV1",
    "TrialDevIdentificationStatusV1",
    "TrialDevObservationalCandidateEvidenceV1",
    "TrialDevObservationalDecisionEvidenceV1",
    "TrialDevPairContrastEvidenceV1",
    "TrialDevPolicyBindingV1",
    "TrialDevPortfolioProgrammeStateV1",
    "TrialDevPortfolioCheckpointHistoryEntryV1",
    "TrialDevPortfolioActionSelectionV1",
    "TrialDevPortfolioActionIdV1",
    "TrialDevPortfolioCheckpointActionPolicyV1",
    "TrialDevPortfolioLegalActionSpecV1",
    "TrialDevPortfolioEvidenceIndexV1",
    "TrialDevProgrammeStateV1",
    "TrialDevProgrammeStateRecordV1",
    "TrialDevReachStatusV1",
    "TrialDevRandomizedDecisionEvidenceV1",
    "TrialDevResourceScheduleV1",
    "TrialDevRuleClassificationV1",
    "TrialDevRuleDirectionV1",
    "TrialDevRuleDomainV1",
    "TrialDevStreamIdV1",
    "TrialDevSingleAssetProgrammeStateV1",
    "TrialDevSingleAssetCheckpointHistoryEntryV1",
    "TrialDevSingleAssetActionSelectionV1",
    "TrialDevSingleAssetActionIdV1",
    "TrialDevSingleAssetCheckpointActionPolicyV1",
    "TrialDevSingleAssetLegalActionSpecV1",
    "TrialDevSubmissionStatusV1",
    "TrialDevSupportedActionSetV1",
    "TrialDevSupportedActionV1",
    "TrialDevTerminalDispositionV1",
    "TRIALDEV_PROGRAMME_STATE_ADAPTER_V1",
]
