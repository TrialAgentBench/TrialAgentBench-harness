"""Deterministic TrialDev programme-state transitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevActionSelectionV1,
    TrialDevCheckpointActionPolicyV1,
    TrialDevCheckpointHistoryEntryV1,
    TrialDevCheckpointIdV1,
    TrialDevCheckpointOutcomeV1,
    TrialDevEvidenceReferenceV1,
    TrialDevLegalActionSpecV1,
    TrialDevPortfolioActionSelectionV1,
    TrialDevPortfolioCheckpointActionPolicyV1,
    TrialDevPortfolioCheckpointHistoryEntryV1,
    TrialDevPortfolioEvidenceIndexV1,
    TrialDevPortfolioLegalActionSpecV1,
    TrialDevPortfolioProgrammeStateV1,
    TrialDevProgrammeStateV1,
    TrialDevSingleAssetActionSelectionV1,
    TrialDevSingleAssetCheckpointActionPolicyV1,
    TrialDevSingleAssetCheckpointHistoryEntryV1,
    TrialDevSingleAssetLegalActionSpecV1,
    TrialDevSingleAssetProgrammeStateV1,
    TrialDevStreamIdV1,
    TrialDevTerminalDispositionV1,
)

_LEGAL_ACTIONS = {
    ("single_asset_development", "observational_review"): (
        TrialDevSingleAssetLegalActionSpecV1(
            action_id="nominate_for_early_study", action_kind="allocate", requires_target_asset=True
        ),
        TrialDevSingleAssetLegalActionSpecV1(action_id="withhold_nomination", action_kind="stop"),
    ),
    ("single_asset_development", "early_safety_study"): (
        TrialDevSingleAssetLegalActionSpecV1(action_id="advance_to_proof_of_concept", action_kind="advance"),
        TrialDevSingleAssetLegalActionSpecV1(action_id="stop_development", action_kind="stop"),
    ),
    ("single_asset_development", "proof_of_concept"): (
        TrialDevSingleAssetLegalActionSpecV1(action_id="advance_to_confirmation", action_kind="advance"),
        TrialDevSingleAssetLegalActionSpecV1(action_id="stop_development", action_kind="stop"),
    ),
    ("single_asset_development", "confirmation"): (
        TrialDevSingleAssetLegalActionSpecV1(action_id="declare_success", action_kind="terminal"),
        TrialDevSingleAssetLegalActionSpecV1(action_id="declare_failure", action_kind="terminal"),
        TrialDevSingleAssetLegalActionSpecV1(action_id="declare_inconclusive", action_kind="terminal"),
    ),
    ("bounded_portfolio_reallocation", "observational_review"): (
        TrialDevPortfolioLegalActionSpecV1(
            action_id="select_lead_and_reserve",
            action_kind="allocate",
            requires_target_asset=True,
            requires_reserve_asset=True,
        ),
        TrialDevPortfolioLegalActionSpecV1(action_id="withhold_selection", action_kind="stop"),
    ),
    ("bounded_portfolio_reallocation", "joint_early_study_review"): (
        TrialDevPortfolioLegalActionSpecV1(action_id="advance_lead_to_proof_of_concept", action_kind="advance"),
        TrialDevPortfolioLegalActionSpecV1(
            action_id="promote_reserve_to_proof_of_concept", action_kind="promote", consumes_switch=True
        ),
        TrialDevPortfolioLegalActionSpecV1(action_id="terminate_portfolio", action_kind="stop"),
    ),
    ("bounded_portfolio_reallocation", "lead_proof_of_concept_review"): (
        TrialDevPortfolioLegalActionSpecV1(action_id="advance_active_to_confirmation", action_kind="advance"),
        TrialDevPortfolioLegalActionSpecV1(
            action_id="promote_reserve_to_proof_of_concept", action_kind="promote", consumes_switch=True
        ),
        TrialDevPortfolioLegalActionSpecV1(action_id="terminate_portfolio", action_kind="stop"),
    ),
    ("bounded_portfolio_reallocation", "promoted_reserve_proof_of_concept_review"): (
        TrialDevPortfolioLegalActionSpecV1(action_id="advance_active_to_confirmation", action_kind="advance"),
        TrialDevPortfolioLegalActionSpecV1(action_id="terminate_portfolio", action_kind="stop"),
    ),
    ("bounded_portfolio_reallocation", "confirmation"): (
        TrialDevPortfolioLegalActionSpecV1(action_id="declare_success", action_kind="terminal"),
        TrialDevPortfolioLegalActionSpecV1(action_id="declare_failure", action_kind="terminal"),
        TrialDevPortfolioLegalActionSpecV1(action_id="declare_inconclusive", action_kind="terminal"),
    ),
}


@dataclass(frozen=True)
class _Transition:
    checkpoint_id: TrialDevCheckpointIdV1
    nominated_asset_id: str | None
    lead_asset_id: str | None
    reserve_asset_id: str | None
    active_asset_id: str | None
    retired_asset_ids: tuple[str, ...]
    permanently_ineligible_asset_ids: tuple[str, ...]
    terminal_disposition: TrialDevTerminalDispositionV1
    resource_spent_units: int
    switch_count: int


def _transition_evidence_asset_ids(*, transition: _Transition) -> tuple[str, ...]:
    if transition.checkpoint_id == "joint_early_study_review":
        if transition.lead_asset_id is None or transition.reserve_asset_id is None:
            raise ValueError("Joint early-study evidence requires assigned lead and reserve assets.")
        return (transition.lead_asset_id, transition.reserve_asset_id)
    if transition.active_asset_id is None:
        raise ValueError("An active evidence-producing checkpoint requires one active asset.")
    return (transition.active_asset_id,)


def build_checkpoint_action_policy_v1(*, state: TrialDevProgrammeStateV1) -> TrialDevCheckpointActionPolicyV1:
    """Build the exact public legal-action policy for the current state."""

    actions = legal_actions_for_checkpoint_v1(
        stream_id=state.stream_id,
        checkpoint_id=state.current_checkpoint_id,
    )
    if state.stream_id == "bounded_portfolio_reallocation":
        actions = tuple(action for action in actions if _portfolio_action_is_feasible(state=state, action=action))
        if not actions:
            raise ValueError("An active portfolio state must expose at least one feasible action.")
    policy_model = (
        TrialDevSingleAssetCheckpointActionPolicyV1
        if state.stream_id == "single_asset_development"
        else TrialDevPortfolioCheckpointActionPolicyV1
    )
    return policy_model(
        stream_id=state.stream_id,
        checkpoint_id=state.current_checkpoint_id,
        policy_binding_checksum=cast(str, state.policy_binding.checksum),
        actions=actions,
    )


def legal_actions_for_checkpoint_v1(
    *, stream_id: TrialDevStreamIdV1, checkpoint_id: TrialDevCheckpointIdV1
) -> tuple[TrialDevLegalActionSpecV1, ...]:
    """Return the canonical legal actions for one stream checkpoint."""

    actions = _LEGAL_ACTIONS.get((stream_id, checkpoint_id))
    if actions is None:
        raise ValueError(f"Checkpoint {checkpoint_id!r} is not reachable in stream {stream_id!r}.")
    return actions


def transition_portfolio_programme_state_v1(
    *,
    state: TrialDevPortfolioProgrammeStateV1,
    evidence_index: TrialDevPortfolioEvidenceIndexV1,
    action_policy: TrialDevPortfolioCheckpointActionPolicyV1,
    selection: TrialDevPortfolioActionSelectionV1,
    outcome: TrialDevCheckpointOutcomeV1,
    checkpoint_evidence: tuple[TrialDevEvidenceReferenceV1, ...] = (),
) -> TrialDevPortfolioProgrammeStateV1:
    """Transition a portfolio state using only its precommitted branch evidence."""

    if evidence_index.scenario_id != state.scenario_id:
        raise ValueError("Portfolio state and evidence index identify different scenarios.")
    if any(
        item.source_family_id != evidence_index.source_identity or item.world_id != evidence_index.world_id
        for item in state.evidence
    ):
        raise ValueError("Portfolio state evidence provenance does not match its evidence index.")
    result = transition_programme_state_v1(
        state=state,
        action_policy=action_policy,
        selection=selection,
        outcome=outcome,
        checkpoint_evidence=checkpoint_evidence,
        next_evidence_provider=lambda checkpoint_id, asset_ids: evidence_index.resolve(
            checkpoint_id=checkpoint_id,
            asset_ids=asset_ids,
        ),
    )
    if not isinstance(result, TrialDevPortfolioProgrammeStateV1):
        raise TypeError("Portfolio transition returned a non-portfolio state.")
    return result


def transition_programme_state_v1(
    *,
    state: TrialDevProgrammeStateV1,
    action_policy: TrialDevCheckpointActionPolicyV1,
    selection: TrialDevActionSelectionV1,
    outcome: TrialDevCheckpointOutcomeV1,
    checkpoint_evidence: tuple[TrialDevEvidenceReferenceV1, ...] = (),
    next_evidence: tuple[TrialDevEvidenceReferenceV1, ...] = (),
    next_evidence_provider: (
        Callable[
            [TrialDevCheckpointIdV1, tuple[str, ...]],
            tuple[TrialDevEvidenceReferenceV1, ...],
        ]
        | None
    ) = None,
) -> TrialDevProgrammeStateV1:
    """Return the unique next state for one legal TrialDev action.

    The transition validates public legality and custody only. Whether the
    action belongs to the method-conditioned supported set is an evaluator
    responsibility and does not alter the deterministic state transition.
    """

    if state.terminal_disposition != "active":
        raise ValueError("A terminal TrialDev programme cannot transition.")
    if selection.state_checksum != state.checksum:
        raise ValueError("Selected action is not bound to the current programme state.")
    if selection.checkpoint_id != state.current_checkpoint_id:
        raise ValueError("Selected action checkpoint does not match the current programme state.")
    if action_policy.stream_id != state.stream_id or action_policy.checkpoint_id != state.current_checkpoint_id:
        raise ValueError("Action policy does not govern the current programme state.")
    if action_policy.policy_binding_checksum != state.policy_binding.checksum:
        raise ValueError("Action policy and programme state use different policy bindings.")
    action_spec = next(
        (spec for spec in action_policy.actions if spec.action_id == selection.action_id),
        None,
    )
    if action_spec is None:
        raise ValueError("Selected action is not legal at the current checkpoint.")
    if action_spec.requires_target_asset != (selection.target_asset_id is not None):
        raise ValueError("Selected action target_asset_id does not match its public action specification.")
    if action_spec.requires_reserve_asset != (selection.reserve_asset_id is not None):
        raise ValueError("Selected action reserve_asset_id does not match its public action specification.")
    if (
        outcome.reach_status != "reached"
        or outcome.submission_status != "accepted"
        or outcome.analysis_status not in {"estimable", "non_estimable"}
        or outcome.execution_status != "completed"
    ):
        raise ValueError("A programme transition requires one completed, accepted checkpoint decision.")
    if any(item.checkpoint_id != state.current_checkpoint_id for item in checkpoint_evidence):
        raise ValueError("Checkpoint evidence must identify the current checkpoint.")
    if next_evidence and next_evidence_provider is not None:
        raise ValueError("Provide next evidence directly or through a provider, not both.")
    existing_ids = {item.evidence_id for item in state.evidence}
    available_evidence_ids = existing_ids | {item.evidence_id for item in checkpoint_evidence}
    available_evidence_checksums = {cast(str, item.checksum) for item in (*state.evidence, *checkpoint_evidence)}
    if not set(selection.supporting_evidence_ids) <= available_evidence_ids:
        raise ValueError("Selected action cites evidence outside the current participant-visible state.")
    if state.stream_id == "single_asset_development" and outcome.asset_eligibility:
        raise ValueError("Asset eligibility dispositions belong only to bounded portfolio reallocation.")
    for disposition in outcome.asset_eligibility:
        if disposition.asset_id not in state.candidate_asset_ids:
            raise ValueError("Asset eligibility disposition identifies an unknown candidate.")
        if not set(disposition.evidence_reference_checksums) <= available_evidence_checksums:
            raise ValueError("Asset eligibility disposition cites evidence outside the current state.")
    permanently_ineligible = tuple(
        dict.fromkeys(
            (
                *state.permanently_ineligible_asset_ids,
                *(item.asset_id for item in outcome.asset_eligibility if item.status == "permanently_ineligible"),
            )
        )
    )
    if state.stream_id == "single_asset_development":
        transition = _single_asset_transition(state=state, selection=selection)
    else:
        transition = _portfolio_transition(
            state=state,
            selection=selection,
            permanently_ineligible_asset_ids=permanently_ineligible,
        )
    if transition.terminal_disposition == "active":
        if next_evidence_provider is not None:
            next_evidence = next_evidence_provider(
                transition.checkpoint_id,
                _transition_evidence_asset_ids(transition=transition),
            )
        if not next_evidence:
            raise ValueError("An active next checkpoint requires participant-visible evidence.")
        if any(item.checkpoint_id != transition.checkpoint_id for item in next_evidence):
            raise ValueError("Next-state evidence must identify the next checkpoint.")
    elif next_evidence:
        raise ValueError("A terminal transition cannot expose a future evidence block.")
    appended_evidence = (*checkpoint_evidence, *next_evidence)
    existing_checksums = {item.checksum for item in state.evidence}
    if any(item.evidence_id in existing_ids or item.checksum in existing_checksums for item in appended_evidence):
        raise ValueError("Appended evidence references must be new to the programme history.")
    appended_ids = tuple(item.evidence_id for item in appended_evidence)
    appended_checksums = tuple(item.checksum for item in appended_evidence)
    if len(appended_ids) != len(set(appended_ids)) or len(appended_checksums) != len(set(appended_checksums)):
        raise ValueError("Appended evidence references must be unique.")
    all_evidence = (*state.evidence, *checkpoint_evidence, *next_evidence)
    history = _append_history(
        state=state,
        selection=selection,
        outcome=outcome,
        active_asset_id=transition.active_asset_id,
        lead_asset_id=transition.lead_asset_id,
        reserve_asset_id=transition.reserve_asset_id,
        retired_asset_ids=transition.retired_asset_ids,
        permanently_ineligible_asset_ids=transition.permanently_ineligible_asset_ids,
        resource_spent_units=transition.resource_spent_units,
        evidence=all_evidence,
    )
    state_model = (
        TrialDevSingleAssetProgrammeStateV1
        if state.stream_id == "single_asset_development"
        else TrialDevPortfolioProgrammeStateV1
    )
    return state_model(
        programme_id=state.programme_id,
        scenario_id=state.scenario_id,
        stream_id=state.stream_id,
        current_checkpoint_id=transition.checkpoint_id,
        candidate_asset_ids=state.candidate_asset_ids,
        nominated_asset_id=transition.nominated_asset_id,
        lead_asset_id=transition.lead_asset_id,
        reserve_asset_id=transition.reserve_asset_id,
        active_asset_id=transition.active_asset_id,
        retired_asset_ids=transition.retired_asset_ids,
        permanently_ineligible_asset_ids=transition.permanently_ineligible_asset_ids,
        terminal_disposition=transition.terminal_disposition,
        policy_binding=state.policy_binding,
        evidence=all_evidence,
        history=history,
        resource_spent_units=transition.resource_spent_units,
        switch_count=transition.switch_count,
        previous_state_checksum=state.checksum,
    )


def _single_asset_transition(
    *,
    state: TrialDevProgrammeStateV1,
    selection: TrialDevActionSelectionV1,
) -> _Transition:
    action = selection.action_id
    checkpoint = state.current_checkpoint_id
    nominated = state.nominated_asset_id
    active = state.active_asset_id
    retired = state.retired_asset_ids
    terminal: TrialDevTerminalDispositionV1 = "active"
    if checkpoint == "observational_review":
        if action == "withhold_nomination":
            terminal = "withheld"
        elif action == "nominate_for_early_study":
            nominated = selection.target_asset_id
            active = nominated
            retired = tuple(asset for asset in state.candidate_asset_ids if asset != nominated)
            checkpoint = "early_safety_study"
        else:
            raise ValueError("Illegal single-asset observational action.")
    elif checkpoint == "early_safety_study":
        if action == "advance_to_proof_of_concept":
            checkpoint = "proof_of_concept"
        elif action == "stop_development":
            terminal = "stopped"
        else:
            raise ValueError("Illegal early-safety action.")
    elif checkpoint == "proof_of_concept":
        if action == "advance_to_confirmation":
            checkpoint = "confirmation"
        elif action == "stop_development":
            terminal = "stopped"
        else:
            raise ValueError("Illegal proof-of-concept action.")
    elif checkpoint == "confirmation":
        terminal_actions: dict[str, TrialDevTerminalDispositionV1] = {
            "declare_success": "success",
            "declare_failure": "failure",
            "declare_inconclusive": "inconclusive",
        }
        if action not in terminal_actions:
            raise ValueError("Illegal confirmatory action.")
        terminal = terminal_actions[action]
    else:
        raise ValueError(f"Checkpoint is not part of single-asset development: {checkpoint!r}.")
    return _Transition(
        checkpoint_id=checkpoint,
        nominated_asset_id=nominated,
        lead_asset_id=None,
        reserve_asset_id=None,
        active_asset_id=active,
        retired_asset_ids=retired,
        permanently_ineligible_asset_ids=(),
        terminal_disposition=terminal,
        resource_spent_units=0,
        switch_count=0,
    )


def _portfolio_transition(
    *,
    state: TrialDevProgrammeStateV1,
    selection: TrialDevActionSelectionV1,
    permanently_ineligible_asset_ids: tuple[str, ...],
) -> _Transition:
    action = selection.action_id
    checkpoint = state.current_checkpoint_id
    lead = state.lead_asset_id
    reserve = state.reserve_asset_id
    active = state.active_asset_id
    retired = tuple(dict.fromkeys((*state.retired_asset_ids, *permanently_ineligible_asset_ids)))
    spent = state.resource_spent_units
    switch_count = state.switch_count
    terminal: TrialDevTerminalDispositionV1 = "active"
    if checkpoint == "observational_review":
        if action == "withhold_selection":
            terminal = "withheld"
        elif action == "select_lead_and_reserve":
            lead = selection.target_asset_id
            reserve = selection.reserve_asset_id
            if lead not in state.candidate_asset_ids or reserve not in state.candidate_asset_ids:
                raise ValueError("Selected portfolio roles must belong to the candidate set.")
            if lead in permanently_ineligible_asset_ids or reserve in permanently_ineligible_asset_ids:
                raise ValueError("A permanently ineligible asset cannot receive a portfolio role.")
            active = lead
            retired = tuple(asset for asset in state.candidate_asset_ids if asset not in {lead, reserve})
            checkpoint = "joint_early_study_review"
            schedule = state.policy_binding.resource_schedule
            if schedule is None:
                raise ValueError("Portfolio transition requires a public resource schedule.")
            spent += 2 * schedule.early_study_units
        else:
            raise ValueError("Illegal portfolio observational action.")
    elif checkpoint == "joint_early_study_review":
        if action == "advance_lead_to_proof_of_concept":
            checkpoint = "lead_proof_of_concept_review"
            active = lead
            if active is None or active in retired:
                raise ValueError("The lead is not eligible to enter proof of concept.")
            spent += _portfolio_episode_cost(state=state, checkpoint=checkpoint)
        elif action == "promote_reserve_to_proof_of_concept":
            checkpoint = "promoted_reserve_proof_of_concept_review"
            active = reserve
            if active is None or active in retired:
                raise ValueError("The reserve is not eligible for promotion.")
            if lead is not None:
                retired = tuple(dict.fromkeys((*retired, lead)))
            switch_count = 1
            spent += _portfolio_episode_cost(state=state, checkpoint=checkpoint)
        elif action == "terminate_portfolio":
            terminal = "stopped"
        else:
            raise ValueError("Illegal joint early-study action.")
    elif checkpoint == "lead_proof_of_concept_review":
        if action == "advance_active_to_confirmation":
            if active is None or active in retired:
                raise ValueError("The active asset is not eligible for confirmation.")
            checkpoint = "confirmation"
            inactive = reserve if active == lead else lead
            if inactive is not None:
                retired = tuple(dict.fromkeys((*retired, inactive)))
            spent += _portfolio_episode_cost(state=state, checkpoint=checkpoint)
        elif action == "promote_reserve_to_proof_of_concept":
            checkpoint = "promoted_reserve_proof_of_concept_review"
            active = reserve
            if active is None or active in retired:
                raise ValueError("The reserve is not eligible for promotion.")
            if lead is not None:
                retired = tuple(dict.fromkeys((*retired, lead)))
            switch_count = 1
            spent += _portfolio_episode_cost(state=state, checkpoint=checkpoint)
        elif action == "terminate_portfolio":
            terminal = "stopped"
        else:
            raise ValueError("Illegal lead proof-of-concept action.")
    elif checkpoint == "promoted_reserve_proof_of_concept_review":
        if action == "advance_active_to_confirmation":
            if active is None or active in retired:
                raise ValueError("The active asset is not eligible for confirmation.")
            checkpoint = "confirmation"
            spent += _portfolio_episode_cost(state=state, checkpoint=checkpoint)
        elif action == "terminate_portfolio":
            terminal = "stopped"
        else:
            raise ValueError("Illegal promoted-reserve proof-of-concept action.")
    elif checkpoint == "confirmation":
        terminal_actions: dict[str, TrialDevTerminalDispositionV1] = {
            "declare_success": "success",
            "declare_failure": "failure",
            "declare_inconclusive": "inconclusive",
        }
        if action not in terminal_actions:
            raise ValueError("Illegal confirmatory portfolio action.")
        terminal = terminal_actions[action]
    else:
        raise ValueError(f"Checkpoint is not part of bounded portfolio reallocation: {checkpoint!r}.")
    budget = state.policy_binding.resource_budget_units
    if budget is None:
        raise ValueError("Portfolio transition requires a disclosed resource budget.")
    required_remaining = _required_remaining_portfolio_units(
        state=state,
        checkpoint=cast(TrialDevCheckpointIdV1, checkpoint),
        terminal=terminal,
    )
    if spent + required_remaining > budget:
        raise ValueError("Selected action cannot fund its immediate and required remaining path.")
    return _Transition(
        checkpoint_id=cast(TrialDevCheckpointIdV1, checkpoint),
        nominated_asset_id=None,
        lead_asset_id=lead,
        reserve_asset_id=reserve,
        active_asset_id=active,
        retired_asset_ids=retired,
        permanently_ineligible_asset_ids=permanently_ineligible_asset_ids,
        terminal_disposition=terminal,
        resource_spent_units=spent,
        switch_count=switch_count,
    )


def _required_remaining_portfolio_units(
    *,
    state: TrialDevProgrammeStateV1,
    checkpoint: TrialDevCheckpointIdV1,
    terminal: TrialDevTerminalDispositionV1,
) -> int:
    if terminal != "active":
        return 0
    if checkpoint in {"lead_proof_of_concept_review", "promoted_reserve_proof_of_concept_review"}:
        return _portfolio_episode_cost(state=state, checkpoint="confirmation")
    if checkpoint == "joint_early_study_review":
        return _portfolio_episode_cost(
            state=state, checkpoint="lead_proof_of_concept_review"
        ) + _portfolio_episode_cost(state=state, checkpoint="confirmation")
    return 0


def _portfolio_action_is_feasible(*, state: TrialDevProgrammeStateV1, action: TrialDevLegalActionSpecV1) -> bool:
    """Return whether one route is executable from the disclosed state."""

    if action.action_id in {"terminate_portfolio", "withhold_selection"}:
        return True
    if action.action_id == "select_lead_and_reserve":
        return True
    active = state.active_asset_id
    reserve = state.reserve_asset_id
    retired = set(state.retired_asset_ids)
    if action.action_id == "promote_reserve_to_proof_of_concept":
        if reserve is None or reserve in retired or state.switch_count >= 1:
            return False
        immediate = _portfolio_episode_cost(
            state=state,
            checkpoint="promoted_reserve_proof_of_concept_review",
        )
        required = _portfolio_episode_cost(state=state, checkpoint="confirmation")
    elif action.action_id in {
        "advance_lead_to_proof_of_concept",
        "advance_active_to_confirmation",
    }:
        if active is None or active in retired:
            return False
        next_checkpoint: TrialDevCheckpointIdV1 = (
            "lead_proof_of_concept_review"
            if action.action_id == "advance_lead_to_proof_of_concept"
            else "confirmation"
        )
        immediate = _portfolio_episode_cost(state=state, checkpoint=next_checkpoint)
        required = (
            _portfolio_episode_cost(state=state, checkpoint="confirmation")
            if next_checkpoint == "lead_proof_of_concept_review"
            else 0
        )
    else:
        return True
    budget = state.policy_binding.resource_budget_units
    return budget is not None and state.resource_spent_units + immediate + required <= budget


def _portfolio_episode_cost(*, state: TrialDevProgrammeStateV1, checkpoint: TrialDevCheckpointIdV1) -> int:
    schedule = state.policy_binding.resource_schedule
    if schedule is None:
        raise ValueError("Portfolio transition requires a public resource schedule.")
    if checkpoint in {"lead_proof_of_concept_review", "promoted_reserve_proof_of_concept_review"}:
        return int(schedule.proof_of_concept_units)
    if checkpoint == "confirmation":
        return int(schedule.confirmation_units)
    raise ValueError(f"Checkpoint has no standalone portfolio episode cost: {checkpoint!r}.")


def _append_history(
    *,
    state: TrialDevProgrammeStateV1,
    selection: TrialDevActionSelectionV1,
    outcome: TrialDevCheckpointOutcomeV1,
    active_asset_id: str | None,
    lead_asset_id: str | None,
    reserve_asset_id: str | None,
    retired_asset_ids: tuple[str, ...],
    permanently_ineligible_asset_ids: tuple[str, ...],
    resource_spent_units: int,
    evidence: tuple[TrialDevEvidenceReferenceV1, ...],
) -> tuple[TrialDevCheckpointHistoryEntryV1, ...]:
    checkpoint_evidence = tuple(item for item in evidence if item.checkpoint_id == state.current_checkpoint_id)
    entry_model = (
        TrialDevSingleAssetCheckpointHistoryEntryV1
        if state.stream_id == "single_asset_development"
        else TrialDevPortfolioCheckpointHistoryEntryV1
    )
    selection_model = (
        TrialDevSingleAssetActionSelectionV1
        if state.stream_id == "single_asset_development"
        else TrialDevPortfolioActionSelectionV1
    )
    typed_selection = selection_model.model_validate(selection.model_dump(mode="json", exclude_none=True))
    entry = entry_model(
        state_index=len(state.history),
        checkpoint_id=state.current_checkpoint_id,
        evidence_reference_checksums=tuple(cast(str, item.checksum) for item in checkpoint_evidence),
        selected_action=typed_selection,
        outcome=outcome,
        active_asset_id=active_asset_id,
        lead_asset_id=lead_asset_id,
        reserve_asset_id=reserve_asset_id,
        retired_asset_ids=retired_asset_ids,
        permanently_ineligible_asset_ids=permanently_ineligible_asset_ids,
        resources_spent_units=resource_spent_units,
        previous_entry_checksum=None if not state.history else state.history[-1].checksum,
    )
    return (*state.history, entry)


__all__ = [
    "build_checkpoint_action_policy_v1",
    "legal_actions_for_checkpoint_v1",
    "transition_portfolio_programme_state_v1",
    "transition_programme_state_v1",
]
