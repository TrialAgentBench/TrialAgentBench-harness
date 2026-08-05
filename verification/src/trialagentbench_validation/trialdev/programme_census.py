"""Independent finite-census verification for TrialDev programme releases."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

StreamId = Literal["single_asset_development", "bounded_portfolio_reallocation"]
CheckpointId = Literal[
    "observational_review",
    "early_safety_study",
    "joint_early_study_review",
    "proof_of_concept",
    "lead_proof_of_concept_review",
    "promoted_reserve_proof_of_concept_review",
    "confirmation",
]
Disposition = Literal[
    "active", "withheld", "stopped", "success", "failure", "inconclusive"
]

_SINGLE_ACTIONS: dict[str, tuple[str, ...]] = {
    "observational_review": ("nominate_for_early_study", "withhold_nomination"),
    "early_safety_study": ("advance_to_proof_of_concept", "stop_development"),
    "proof_of_concept": ("advance_to_confirmation", "stop_development"),
    "confirmation": ("declare_success", "declare_failure", "declare_inconclusive"),
}
_PORTFOLIO_ACTIONS: dict[str, tuple[str, ...]] = {
    "observational_review": ("select_lead_and_reserve", "withhold_selection"),
    "joint_early_study_review": (
        "advance_lead_to_proof_of_concept",
        "promote_reserve_to_proof_of_concept",
        "terminate_portfolio",
    ),
    "lead_proof_of_concept_review": (
        "advance_active_to_confirmation",
        "promote_reserve_to_proof_of_concept",
        "terminate_portfolio",
    ),
    "promoted_reserve_proof_of_concept_review": (
        "advance_active_to_confirmation",
        "terminate_portfolio",
    ),
    "confirmation": ("declare_success", "declare_failure", "declare_inconclusive"),
}
_TERMINAL_BY_ACTION: dict[str, Disposition] = {
    "withhold_nomination": "withheld",
    "withhold_selection": "withheld",
    "stop_development": "stopped",
    "terminate_portfolio": "stopped",
    "declare_success": "success",
    "declare_failure": "failure",
    "declare_inconclusive": "inconclusive",
}


def _canonical_checksum(model: BaseModel) -> str:
    payload = model.model_dump(mode="json", exclude_none=True, exclude={"checksum"})
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _ChecksummedRecord(_Record):
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def bind_checksum(self) -> Self:
        """Bind every census record to its canonical JSON payload."""

        expected = _canonical_checksum(self)
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("Census record checksum does not match its payload.")
        object.__setattr__(self, "checksum", expected)
        return self


class TrialDevCensusEvidenceV1(_ChecksummedRecord):
    """One participant-visible evidence artifact in the independent census."""

    evidence_id: str = Field(min_length=1)
    checkpoint_id: CheckpointId
    asset_id: str | None = Field(default=None, min_length=1)
    relative_path: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """Reject absolute, escaping, or evaluator-only evidence paths."""

        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != self.relative_path
        ):
            raise ValueError(
                "Evidence paths must be normalized release-relative paths."
            )
        if any(part in {"hidden", "grader", "evaluator"} for part in path.parts):
            raise ValueError(
                "Participant evidence cannot reference evaluator-only paths."
            )
        return self


class TrialDevCensusResourceScheduleV1(_ChecksummedRecord):
    """Disclosed programme resource costs reconstructed by the verifier."""

    early_study_units: int = Field(gt=0)
    proof_of_concept_units: int = Field(gt=0)
    confirmation_units: int = Field(gt=0)
    maximum_switches: int = Field(ge=0)


class TrialDevCensusStateV1(_ChecksummedRecord):
    """Decision-relevant public state independently inspected by the verifier."""

    programme_id: str = Field(min_length=1)
    stream_id: StreamId
    checkpoint_id: CheckpointId
    candidate_asset_ids: tuple[str, ...] = Field(min_length=1)
    nominated_asset_id: str | None = Field(default=None, min_length=1)
    lead_asset_id: str | None = Field(default=None, min_length=1)
    reserve_asset_id: str | None = Field(default=None, min_length=1)
    active_asset_id: str | None = Field(default=None, min_length=1)
    retired_asset_ids: tuple[str, ...] = ()
    permanently_ineligible_asset_ids: tuple[str, ...] = ()
    resource_budget_units: int | None = Field(default=None, ge=0)
    resource_spent_units: int = Field(default=0, ge=0)
    switch_count: int = Field(default=0, ge=0, le=1)
    terminal_disposition: Disposition = "active"
    evidence_checksums: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        """Require canonical sets and elementary role/resource invariants."""

        for field_name in (
            "candidate_asset_ids",
            "retired_asset_ids",
            "permanently_ineligible_asset_ids",
            "evidence_checksums",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must be unique.")
        candidates = set(self.candidate_asset_ids)
        if not set(self.retired_asset_ids) <= candidates:
            raise ValueError("Retired assets must belong to the candidate set.")
        if not set(self.permanently_ineligible_asset_ids) <= set(
            self.retired_asset_ids
        ):
            raise ValueError("Permanently ineligible assets must be retired.")
        roles = {
            value
            for value in (
                self.nominated_asset_id,
                self.lead_asset_id,
                self.reserve_asset_id,
            )
            if value
        }
        if not roles <= candidates or (
            self.active_asset_id is not None and self.active_asset_id not in candidates
        ):
            raise ValueError("State roles must identify candidate assets.")
        if self.terminal_disposition == "active" and self.active_asset_id in set(
            self.retired_asset_ids
        ):
            raise ValueError("An active programme cannot use a retired asset.")
        if self.stream_id == "single_asset_development":
            if self.lead_asset_id is not None or self.reserve_asset_id is not None:
                raise ValueError("Single-asset state cannot declare portfolio roles.")
            if (
                self.resource_budget_units is not None
                or self.resource_spent_units
                or self.switch_count
            ):
                raise ValueError(
                    "Single-asset state cannot declare portfolio resources."
                )
        elif (
            self.resource_budget_units is None
            or self.resource_spent_units > self.resource_budget_units
        ):
            raise ValueError("Portfolio state must remain within its resource budget.")
        return self


class TrialDevCensusActionV1(_ChecksummedRecord):
    """One concrete action variant exposed from one public state."""

    state_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    variant_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    target_asset_id: str | None = Field(default=None, min_length=1)
    reserve_asset_id: str | None = Field(default=None, min_length=1)


class TrialDevCensusTransitionV1(_ChecksummedRecord):
    """One materialized transition for a concrete action variant."""

    state_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    action_variant_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    next_state_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    terminal_disposition: Disposition
    newly_exposed_evidence_checksums: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        """Require exactly active transitions to identify a next state."""

        if (self.terminal_disposition == "active") != (
            self.next_state_checksum is not None
        ):
            raise ValueError("Exactly active transitions must identify a next state.")
        if len(self.newly_exposed_evidence_checksums) != len(
            set(self.newly_exposed_evidence_checksums)
        ):
            raise ValueError("New evidence references must be unique.")
        return self


class TrialDevCensusSupportedSetV1(_ChecksummedRecord):
    """Method-conditioned action variants supported by participant evidence."""

    state_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_method_id: str = Field(min_length=1)
    supported_action_variant_checksums: tuple[str, ...] = Field(min_length=1)
    evidence_checksums: tuple[str, ...] = Field(min_length=1)


class TrialDevNumericalWitnessV1(_ChecksummedRecord):
    """One independently reconstructable statistic used by an action set."""

    evidence_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    value_column: str = Field(min_length=1)
    statistic: Literal["mean", "proportion"]
    group_column: str | None = Field(default=None, min_length=1)
    group_value: str | None = Field(default=None, min_length=1)
    reported_value: float
    absolute_tolerance: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_group(self) -> Self:
        """Require grouping column and value together and finite values."""

        if (self.group_column is None) != (self.group_value is None):
            raise ValueError(
                "Numerical witness grouping requires both column and value."
            )
        if not math.isfinite(self.reported_value) or not math.isfinite(
            self.absolute_tolerance
        ):
            raise ValueError("Numerical witness values must be finite.")
        return self


class TrialDevProgrammeCensusV1(_ChecksummedRecord):
    """Complete public/evaluator role projection consumed by the verifier."""

    schema_id: Literal["trialagentbench.trialdev_programme_census/v1"] = (
        "trialagentbench.trialdev_programme_census/v1"
    )
    resource_schedule: TrialDevCensusResourceScheduleV1
    evidence: tuple[TrialDevCensusEvidenceV1, ...] = Field(min_length=1)
    states: tuple[TrialDevCensusStateV1, ...] = Field(min_length=1)
    actions: tuple[TrialDevCensusActionV1, ...] = Field(min_length=1)
    transitions: tuple[TrialDevCensusTransitionV1, ...] = Field(min_length=1)
    supported_sets: tuple[TrialDevCensusSupportedSetV1, ...] = Field(min_length=1)
    numerical_witnesses: tuple[TrialDevNumericalWitnessV1, ...] = Field(min_length=1)


class TrialDevProgrammeCensusReportV1(_Record):
    """Independent finite-census verification result."""

    schema_id: Literal[
        "trialagentbench.validation.trialdev_programme_census_report/v1"
    ] = "trialagentbench.validation.trialdev_programme_census_report/v1"
    census_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_count: int = Field(ge=0)
    action_variant_count: int = Field(ge=0)
    transition_count: int = Field(ge=0)
    terminal_transition_count: int = Field(ge=0)
    supported_set_count: int = Field(ge=0)
    numerical_witness_count: int = Field(ge=0)
    findings: tuple[str, ...]
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        """Bind pass status to an empty finding set."""

        if self.findings != tuple(sorted(set(self.findings))):
            raise ValueError("Verifier findings must be sorted and unique.")
        if (self.status == "pass") != (not self.findings):
            raise ValueError("Verifier status disagrees with its findings.")
        return self


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_actions(
    state: TrialDevCensusStateV1,
    *,
    resource_schedule: TrialDevCensusResourceScheduleV1,
) -> tuple[tuple[str, str | None, str | None], ...]:
    if state.stream_id == "single_asset_development":
        action_ids = _SINGLE_ACTIONS.get(state.checkpoint_id)
    else:
        action_ids = _PORTFOLIO_ACTIONS.get(state.checkpoint_id)
    if action_ids is None:
        raise ValueError(
            f"Checkpoint {state.checkpoint_id!r} is not valid for {state.stream_id!r}."
        )
    output: list[tuple[str, str | None, str | None]] = []
    retired = set(state.retired_asset_ids)
    for action_id in action_ids:
        if action_id == "nominate_for_early_study":
            output.extend(
                (action_id, asset_id, None)
                for asset_id in state.candidate_asset_ids
                if asset_id not in retired
            )
        elif action_id == "select_lead_and_reserve":
            available = tuple(
                asset_id
                for asset_id in state.candidate_asset_ids
                if asset_id not in retired
            )
            output.extend(
                (action_id, lead, reserve)
                for lead in available
                for reserve in available
                if lead != reserve
            )
        elif action_id == "promote_reserve_to_proof_of_concept":
            if (
                state.reserve_asset_id in retired
                or state.switch_count >= resource_schedule.maximum_switches
            ):
                continue
            required = (
                resource_schedule.proof_of_concept_units
                + resource_schedule.confirmation_units
            )
            if (
                state.resource_budget_units is None
                or state.resource_spent_units + required > state.resource_budget_units
            ):
                continue
            output.append((action_id, None, None))
        else:
            output.append((action_id, None, None))
    return tuple(sorted(output))


def _read_numeric_witness(
    *,
    release_root: Path,
    evidence: TrialDevCensusEvidenceV1,
    witness: TrialDevNumericalWitnessV1,
) -> float:
    path = (release_root / evidence.relative_path).resolve(strict=True)
    root = release_root.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(
            "Numerical evidence must be a regular file within the release root."
        )
    if _file_sha256(path) != evidence.artifact_sha256:
        raise ValueError("Numerical evidence checksum does not match released bytes.")
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    if witness.group_column is not None:
        rows = tuple(
            row for row in rows if row.get(witness.group_column) == witness.group_value
        )
    if not rows or any(witness.value_column not in row for row in rows):
        raise ValueError(
            "Numerical witness has no analysable rows or lacks its value column."
        )
    values = tuple(float(row[witness.value_column]) for row in rows)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Numerical witness contains non-finite values.")
    if witness.statistic == "proportion" and any(
        value not in {0.0, 1.0} for value in values
    ):
        raise ValueError("A proportion witness requires binary zero-one observations.")
    return sum(values) / len(values)


def _transition_findings(
    *,
    source: TrialDevCensusStateV1,
    action: TrialDevCensusActionV1,
    target: TrialDevCensusStateV1,
    resource_schedule: TrialDevCensusResourceScheduleV1,
) -> tuple[str, ...]:
    """Independently reconstruct the decision-relevant next-state properties."""

    findings: list[str] = []
    expected_checkpoint: str
    expected_active = source.active_asset_id
    expected_lead = source.lead_asset_id
    expected_reserve = source.reserve_asset_id
    expected_switch = source.switch_count
    expected_spent = source.resource_spent_units
    expected_retired = set(source.retired_asset_ids)
    if action.action_id == "nominate_for_early_study":
        expected_checkpoint = "early_safety_study"
        expected_active = action.target_asset_id
        expected_retired = set(source.candidate_asset_ids) - {action.target_asset_id}
    elif action.action_id == "advance_to_proof_of_concept":
        expected_checkpoint = "proof_of_concept"
    elif action.action_id == "advance_to_confirmation":
        expected_checkpoint = "confirmation"
    elif action.action_id == "select_lead_and_reserve":
        expected_checkpoint = "joint_early_study_review"
        expected_lead = action.target_asset_id
        expected_reserve = action.reserve_asset_id
        expected_active = action.target_asset_id
        expected_retired = set(source.candidate_asset_ids) - {
            action.target_asset_id,
            action.reserve_asset_id,
        }
        expected_spent += 2 * resource_schedule.early_study_units
    elif action.action_id == "advance_lead_to_proof_of_concept":
        expected_checkpoint = "lead_proof_of_concept_review"
        expected_active = source.lead_asset_id
        expected_spent += resource_schedule.proof_of_concept_units
    elif action.action_id == "promote_reserve_to_proof_of_concept":
        expected_checkpoint = "promoted_reserve_proof_of_concept_review"
        expected_active = source.reserve_asset_id
        expected_switch = 1
        expected_spent += resource_schedule.proof_of_concept_units
        if source.lead_asset_id is not None:
            expected_retired.add(source.lead_asset_id)
    elif action.action_id == "advance_active_to_confirmation":
        expected_checkpoint = "confirmation"
        expected_spent += resource_schedule.confirmation_units
        inactive = (
            source.reserve_asset_id
            if source.active_asset_id == source.lead_asset_id
            else source.lead_asset_id
        )
        if inactive is not None:
            expected_retired.add(inactive)
    else:
        return ("active_transition_uses_terminal_or_unknown_action",)
    expected = (
        source.programme_id,
        source.stream_id,
        expected_checkpoint,
        source.candidate_asset_ids,
        (
            action.target_asset_id
            if source.stream_id == "single_asset_development"
            and action.action_id.startswith("nominate")
            else source.nominated_asset_id
        ),
        expected_lead,
        expected_reserve,
        expected_active,
        tuple(sorted(expected_retired)),
        source.permanently_ineligible_asset_ids,
        source.resource_budget_units,
        expected_spent,
        expected_switch,
    )
    observed = (
        target.programme_id,
        target.stream_id,
        target.checkpoint_id,
        target.candidate_asset_ids,
        target.nominated_asset_id,
        target.lead_asset_id,
        target.reserve_asset_id,
        target.active_asset_id,
        tuple(sorted(target.retired_asset_ids)),
        target.permanently_ineligible_asset_ids,
        target.resource_budget_units,
        target.resource_spent_units,
        target.switch_count,
    )
    if observed != expected:
        findings.append("next_state_semantics_disagree")
    return tuple(findings)


def audit_trialdev_programme_census(
    *, census_path: Path, release_root: Path
) -> TrialDevProgrammeCensusReportV1:
    """Independently verify reachability, custody, and numerical witnesses."""

    census = TrialDevProgrammeCensusV1.model_validate_json(
        census_path.read_text(encoding="utf-8")
    )
    findings: list[str] = []
    evidence_by_checksum = {cast(str, row.checksum): row for row in census.evidence}
    states_by_checksum = {cast(str, row.checksum): row for row in census.states}
    actions_by_checksum = {cast(str, row.checksum): row for row in census.actions}
    if len(evidence_by_checksum) != len(census.evidence):
        findings.append("duplicate_evidence_record")
    if len(states_by_checksum) != len(census.states):
        findings.append("duplicate_state_record")
    if len(actions_by_checksum) != len(census.actions):
        findings.append("duplicate_action_variant")
    actions_by_state: dict[str, list[TrialDevCensusActionV1]] = {}
    for action in census.actions:
        actions_by_state.setdefault(action.state_checksum, []).append(action)
    transitions_by_action: dict[str, list[TrialDevCensusTransitionV1]] = {}
    for transition in census.transitions:
        transitions_by_action.setdefault(transition.action_variant_checksum, []).append(
            transition
        )
    for state_checksum, state in states_by_checksum.items():
        missing_evidence = sorted(
            set(state.evidence_checksums) - set(evidence_by_checksum)
        )
        if missing_evidence:
            findings.append(
                f"state_missing_evidence:{state_checksum}:{','.join(missing_evidence)}"
            )
        actual = tuple(
            sorted(
                (action.action_id, action.target_asset_id, action.reserve_asset_id)
                for action in actions_by_state.get(state_checksum, [])
            )
        )
        try:
            expected = _expected_actions(
                state, resource_schedule=census.resource_schedule
            )
        except ValueError as error:
            findings.append(f"invalid_state_route:{state_checksum}:{error}")
            continue
        if actual != expected:
            findings.append(f"action_census_mismatch:{state_checksum}")
        if not any(
            row.state_checksum == state_checksum for row in census.supported_sets
        ):
            findings.append(f"state_missing_supported_set:{state_checksum}")
    for action_checksum, action in actions_by_checksum.items():
        if action.state_checksum not in states_by_checksum:
            findings.append(f"action_unknown_state:{action_checksum}")
            continue
        transitions = transitions_by_action.get(action_checksum, [])
        if len(transitions) != 1:
            findings.append(
                f"transition_cardinality:{action_checksum}:{len(transitions)}"
            )
            continue
        transition = transitions[0]
        if transition.state_checksum != action.state_checksum:
            findings.append(f"transition_state_mismatch:{action_checksum}")
        expected_terminal = _TERMINAL_BY_ACTION.get(action.action_id, "active")
        if transition.terminal_disposition != expected_terminal:
            findings.append(f"terminal_disposition_mismatch:{action_checksum}")
        if transition.next_state_checksum is not None:
            target = states_by_checksum.get(transition.next_state_checksum)
            if target is None:
                findings.append(f"transition_unknown_target:{action_checksum}")
            else:
                source_evidence = set(
                    states_by_checksum[action.state_checksum].evidence_checksums
                )
                new_evidence = set(transition.newly_exposed_evidence_checksums)
                if not new_evidence or not new_evidence <= set(
                    target.evidence_checksums
                ):
                    findings.append(f"transition_evidence_mismatch:{action_checksum}")
                if new_evidence & source_evidence:
                    findings.append(
                        f"transition_reexposes_prior_evidence:{action_checksum}"
                    )
                if set(target.evidence_checksums) != source_evidence | new_evidence:
                    findings.append(
                        f"transition_target_evidence_leak:{action_checksum}"
                    )
                for detail in _transition_findings(
                    source=states_by_checksum[action.state_checksum],
                    action=action,
                    target=target,
                    resource_schedule=census.resource_schedule,
                ):
                    findings.append(f"transition_semantics:{action_checksum}:{detail}")
                expected_assets = (
                    {target.lead_asset_id, target.reserve_asset_id}
                    if target.checkpoint_id == "joint_early_study_review"
                    else {target.active_asset_id}
                )
                for evidence_checksum in new_evidence:
                    evidence = evidence_by_checksum.get(evidence_checksum)
                    if evidence is None:
                        findings.append(
                            f"transition_unknown_evidence:{action_checksum}"
                        )
                    elif (
                        evidence.checkpoint_id != target.checkpoint_id
                        or evidence.asset_id not in expected_assets
                    ):
                        findings.append(
                            f"transition_counterfactual_evidence:{action_checksum}"
                        )
    orphan_transitions = sorted(set(transitions_by_action) - set(actions_by_checksum))
    findings.extend(
        f"transition_unknown_action:{checksum}" for checksum in orphan_transitions
    )
    for programme_id in sorted({state.programme_id for state in census.states}):
        programme_states = {
            checksum
            for checksum, state in states_by_checksum.items()
            if state.programme_id == programme_id
        }
        initial = {
            checksum
            for checksum in programme_states
            if states_by_checksum[checksum].checkpoint_id == "observational_review"
            and states_by_checksum[checksum].terminal_disposition == "active"
        }
        if len(initial) != 1:
            findings.append(
                f"programme_initial_state_cardinality:{programme_id}:{len(initial)}"
            )
            continue
        reached = set(initial)
        frontier = list(initial)
        while frontier:
            source_checksum = frontier.pop()
            for action in actions_by_state.get(source_checksum, []):
                for transition in transitions_by_action.get(
                    cast(str, action.checksum), []
                ):
                    target_checksum = transition.next_state_checksum
                    if (
                        target_checksum in programme_states
                        and target_checksum not in reached
                    ):
                        reached.add(target_checksum)
                        frontier.append(target_checksum)
        for unreachable in sorted(programme_states - reached):
            findings.append(f"unreachable_state:{programme_id}:{unreachable}")
    for supported in census.supported_sets:
        supported_state = states_by_checksum.get(supported.state_checksum)
        if supported_state is None:
            findings.append(f"supported_set_unknown_state:{supported.checksum}")
            continue
        state_actions = {
            cast(str, action.checksum)
            for action in actions_by_state.get(supported.state_checksum, [])
        }
        if not set(supported.supported_action_variant_checksums) <= state_actions:
            findings.append(f"supported_set_illegal_action:{supported.checksum}")
        if not set(supported.evidence_checksums) <= set(
            supported_state.evidence_checksums
        ):
            findings.append(f"supported_set_unavailable_evidence:{supported.checksum}")
    for witness in census.numerical_witnesses:
        evidence = evidence_by_checksum.get(witness.evidence_checksum)
        if evidence is None:
            findings.append(f"numerical_witness_unknown_evidence:{witness.checksum}")
            continue
        try:
            reconstructed = _read_numeric_witness(
                release_root=release_root,
                evidence=evidence,
                witness=witness,
            )
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            UnicodeError,
            ValueError,
        ) as error:
            findings.append(f"numerical_witness_invalid:{witness.checksum}:{error}")
            continue
        if abs(reconstructed - witness.reported_value) > witness.absolute_tolerance:
            findings.append(f"numerical_witness_disagreement:{witness.checksum}")
    ordered_findings = tuple(sorted(set(findings)))
    return TrialDevProgrammeCensusReportV1(
        census_checksum=cast(str, census.checksum),
        state_count=len(census.states),
        action_variant_count=len(census.actions),
        transition_count=len(census.transitions),
        terminal_transition_count=sum(
            row.terminal_disposition != "active" for row in census.transitions
        ),
        supported_set_count=len(census.supported_sets),
        numerical_witness_count=len(census.numerical_witnesses),
        findings=ordered_findings,
        status="fail" if ordered_findings else "pass",
    )


__all__ = [
    "TrialDevCensusActionV1",
    "TrialDevCensusEvidenceV1",
    "TrialDevCensusResourceScheduleV1",
    "TrialDevCensusStateV1",
    "TrialDevCensusSupportedSetV1",
    "TrialDevCensusTransitionV1",
    "TrialDevNumericalWitnessV1",
    "TrialDevProgrammeCensusReportV1",
    "TrialDevProgrammeCensusV1",
    "audit_trialdev_programme_census",
]
