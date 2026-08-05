"""Derive method-conditioned TrialDev actions from public interval evidence."""

from __future__ import annotations

from collections import defaultdict
from typing import cast

from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevObservationalDecisionEvidenceV1,
    TrialDevProgrammeStateV1,
    TrialDevRandomizedDecisionEvidenceV1,
    TrialDevRuleClassificationV1,
    TrialDevRuleDomainV1,
    TrialDevSupportedActionSetV1,
    TrialDevSupportedActionV1,
)
from trialagentbench_harness.trialdev.programme import build_checkpoint_action_policy_v1


def _legal_action_variants(state: TrialDevProgrammeStateV1) -> tuple[TrialDevSupportedActionV1, ...]:
    policy = build_checkpoint_action_policy_v1(state=state)
    retired = set(state.retired_asset_ids)
    variants: list[TrialDevSupportedActionV1] = []
    for action in policy.actions:
        if action.action_id == "nominate_for_early_study":
            variants.extend(
                TrialDevSupportedActionV1(
                    action_id=action.action_id,
                    target_asset_id=asset_id,
                )
                for asset_id in state.candidate_asset_ids
                if asset_id not in retired
            )
        elif action.action_id == "select_lead_and_reserve":
            available = tuple(asset_id for asset_id in state.candidate_asset_ids if asset_id not in retired)
            variants.extend(
                TrialDevSupportedActionV1(
                    action_id=action.action_id,
                    target_asset_id=lead,
                    reserve_asset_id=reserve,
                )
                for lead in available
                for reserve in available
                if lead != reserve
            )
        else:
            variants.append(TrialDevSupportedActionV1(action_id=action.action_id))
    return tuple(variants)


def _observational_supported_actions(
    *,
    state: TrialDevProgrammeStateV1,
    evidence: TrialDevObservationalDecisionEvidenceV1,
    legal_actions: tuple[TrialDevSupportedActionV1, ...],
) -> tuple[TrialDevSupportedActionV1, ...]:
    if evidence.state_checksum != state.checksum:
        raise ValueError("Observational evidence is not bound to the current state.")
    legal_by_signature: dict[tuple[str, str | None, str | None], TrialDevSupportedActionV1] = {
        (item.action_id, item.target_asset_id, item.reserve_asset_id): item for item in legal_actions
    }
    if evidence.identification_status == "not_identified":
        action_id = "withhold_nomination" if state.stream_id == "single_asset_development" else "withhold_selection"
        return (legal_by_signature[(action_id, None, None)],)
    candidates = {item.asset_id: item for item in evidence.candidates}
    if set(candidates) != set(state.candidate_asset_ids):
        raise ValueError("Identified observational evidence must cover the complete candidate set.")
    supported: list[TrialDevSupportedActionV1] = []

    possibly_qualified = {
        item.asset_id for item in evidence.candidates if item.efficacy_upper_bound >= evidence.minimum_efficacy_gain
    }
    definitely_qualified = {
        item.asset_id for item in evidence.candidates if item.efficacy_lower_bound >= evidence.minimum_efficacy_gain
    }
    if not possibly_qualified:
        action_id = "withhold_nomination" if state.stream_id == "single_asset_development" else "withhold_selection"
        return (legal_by_signature[(action_id, None, None)],)
    contrasts = {tuple(sorted((item.lead_asset_id, item.reserve_asset_id))): item for item in evidence.pair_contrasts}
    expected_pairs = {
        (first, second)
        for index, first in enumerate(sorted(state.candidate_asset_ids))
        for second in sorted(state.candidate_asset_ids)[index + 1 :]
    }
    if set(contrasts) != expected_pairs:
        raise ValueError("Observational allocation requires one contrast for every candidate pair.")

    def contrast_half_width(first: str, second: str) -> float:
        if first == second:
            return 0.0
        return contrasts[tuple(sorted((first, second)))].confidence_half_width

    lead_eligible: set[str] = set()
    if possibly_qualified:
        best = min(
            possibly_qualified,
            key=lambda asset_id: (-candidates[asset_id].utility_estimate, asset_id),
        )
        best_utility = candidates[best].utility_estimate
        lead_eligible = {
            asset_id
            for asset_id in possibly_qualified
            if best_utility - candidates[asset_id].utility_estimate
            <= max(
                evidence.practical_equivalence_margin,
                contrast_half_width(best, asset_id),
            )
        }

    if state.stream_id == "single_asset_development":
        supported.extend(
            legal_by_signature[("nominate_for_early_study", asset_id, None)] for asset_id in sorted(lead_eligible)
        )
    else:
        supported_pairs: set[tuple[str, str]] = set()
        for lead in sorted(lead_eligible):
            reserve_candidates = possibly_qualified - {lead}
            if not reserve_candidates:
                continue
            best_reserve = min(
                reserve_candidates,
                key=lambda asset_id: (-candidates[asset_id].utility_estimate, asset_id),
            )
            best_reserve_utility = candidates[best_reserve].utility_estimate
            reserve_eligible = {
                asset_id
                for asset_id in reserve_candidates
                if best_reserve_utility - candidates[asset_id].utility_estimate
                <= max(
                    evidence.practical_equivalence_margin,
                    contrast_half_width(best_reserve, asset_id),
                )
            }
            supported_pairs.update((lead, reserve) for reserve in reserve_eligible)
        for lead, reserve in sorted(supported_pairs):
            if ("select_lead_and_reserve", lead, reserve) in legal_by_signature:
                supported.append(legal_by_signature[("select_lead_and_reserve", lead, reserve)])
    allocation_supported = any(
        item.action_id in {"nominate_for_early_study", "select_lead_and_reserve"} for item in supported
    )
    if not definitely_qualified or not allocation_supported:
        action_id = "withhold_nomination" if state.stream_id == "single_asset_development" else "withhold_selection"
        supported.append(legal_by_signature[(action_id, None, None)])
    return tuple(supported)


def _domain_classifications(
    *,
    state: TrialDevProgrammeStateV1,
    evidence: TrialDevRandomizedDecisionEvidenceV1,
    required: dict[str, tuple[TrialDevRuleDomainV1, ...]],
) -> dict[str, dict[TrialDevRuleDomainV1, TrialDevRuleClassificationV1]]:
    if evidence.state_checksum != state.checksum:
        raise ValueError("Randomized evidence is not bound to the current state.")
    evidence_checksums = {cast(str, item.checksum) for item in state.evidence}
    grouped: dict[
        tuple[str, TrialDevRuleDomainV1],
        list[TrialDevRuleClassificationV1],
    ] = defaultdict(list)
    for rule in evidence.rules:
        if rule.asset_id not in state.candidate_asset_ids:
            raise ValueError("Randomized rule evidence identifies an unknown asset.")
        if not set(rule.evidence_reference_checksums) <= evidence_checksums:
            raise ValueError("Randomized rule evidence cites unavailable participant evidence.")
        grouped[(rule.asset_id, rule.domain)].append(rule.classification)
    output: dict[str, dict[TrialDevRuleDomainV1, TrialDevRuleClassificationV1]] = {}
    for asset_id, domains in required.items():
        output[asset_id] = {}
        for domain in domains:
            values = grouped.get((asset_id, domain), [])
            if not values:
                raise ValueError(f"Randomized evidence lacks {domain} rules for asset {asset_id!r}.")
            if "clear_fail" in values:
                classification: TrialDevRuleClassificationV1 = "clear_fail"
            elif all(value == "clear_pass" for value in values):
                classification = "clear_pass"
            else:
                classification = "indeterminate"
            output[asset_id][domain] = classification
    return output


def _asset_clear_pass(
    classifications: dict[TrialDevRuleDomainV1, TrialDevRuleClassificationV1],
) -> bool:
    return all(value == "clear_pass" for value in classifications.values())


def _asset_clear_fail(
    classifications: dict[TrialDevRuleDomainV1, TrialDevRuleClassificationV1],
) -> bool:
    return any(value == "clear_fail" for value in classifications.values())


def _randomized_supported_action_ids(
    *,
    state: TrialDevProgrammeStateV1,
    evidence: TrialDevRandomizedDecisionEvidenceV1,
    legal_action_ids: set[str],
) -> set[str]:
    checkpoint = state.current_checkpoint_id
    active = state.active_asset_id
    if active is None:
        raise ValueError("A randomized checkpoint requires an active asset.")
    if checkpoint == "joint_early_study_review":
        lead = state.lead_asset_id
        reserve = state.reserve_asset_id
        if lead is None or reserve is None:
            raise ValueError("Joint early-study review requires lead and reserve roles.")
        classes = _domain_classifications(
            state=state,
            evidence=evidence,
            required={lead: ("safety",), reserve: ("safety",)},
        )
        lead_pass = _asset_clear_pass(classes[lead])
        reserve_pass = _asset_clear_pass(classes[reserve])
        output = set()
        if not _asset_clear_fail(classes[lead]):
            output.add("advance_lead_to_proof_of_concept")
        if not _asset_clear_fail(classes[reserve]) and not lead_pass:
            output.add("promote_reserve_to_proof_of_concept")
        if not lead_pass and not reserve_pass:
            output.add("terminate_portfolio")
        return output & legal_action_ids
    required_domains: tuple[TrialDevRuleDomainV1, ...] = (
        ("safety",) if checkpoint == "early_safety_study" else ("efficacy", "safety")
    )
    asset_classes = _domain_classifications(
        state=state,
        evidence=evidence,
        required={active: required_domains},
    )[active]
    clear_pass = _asset_clear_pass(asset_classes)
    clear_fail = _asset_clear_fail(asset_classes)
    if checkpoint == "confirmation":
        if clear_pass:
            return {"declare_success"} & legal_action_ids
        if clear_fail:
            return {"declare_failure"} & legal_action_ids
        return {"declare_inconclusive"} & legal_action_ids
    if checkpoint == "early_safety_study":
        output = set()
        if not clear_fail:
            output.add("advance_to_proof_of_concept")
        if not clear_pass:
            output.add("stop_development")
        return output & legal_action_ids
    if checkpoint == "proof_of_concept":
        output = set()
        if not clear_fail:
            output.add("advance_to_confirmation")
        if not clear_pass:
            output.add("stop_development")
        return output & legal_action_ids
    if checkpoint == "lead_proof_of_concept_review":
        output = set()
        if not clear_fail:
            output.add("advance_active_to_confirmation")
        if not clear_pass:
            output.update({"promote_reserve_to_proof_of_concept", "terminate_portfolio"})
        return output & legal_action_ids
    if checkpoint == "promoted_reserve_proof_of_concept_review":
        output = set()
        if not clear_fail:
            output.add("advance_active_to_confirmation")
        if not clear_pass:
            output.add("terminate_portfolio")
        return output & legal_action_ids
    raise ValueError(f"No randomized supported-action policy exists for {checkpoint!r}.")


def derive_supported_action_set_v1(
    *,
    state: TrialDevProgrammeStateV1,
    evidence: TrialDevObservationalDecisionEvidenceV1 | TrialDevRandomizedDecisionEvidenceV1,
    sensitivity_policy_id: str = "primary",
) -> TrialDevSupportedActionSetV1:
    """Derive every concrete action supported by one valid submitted method."""

    legal_actions = _legal_action_variants(state)
    if isinstance(evidence, TrialDevObservationalDecisionEvidenceV1):
        if state.current_checkpoint_id != "observational_review":
            raise ValueError("Observational evidence is valid only at observational review.")
        supported = _observational_supported_actions(
            state=state,
            evidence=evidence,
            legal_actions=legal_actions,
        )
        public_evidence = tuple(
            sorted(
                {
                    *evidence.identification_evidence_reference_checksums,
                    *(
                        checksum
                        for candidate in evidence.candidates
                        for checksum in candidate.evidence_reference_checksums
                    ),
                }
            )
        )
    else:
        if state.current_checkpoint_id == "observational_review":
            raise ValueError("Randomized evidence cannot grade observational allocation.")
        legal_action_ids: set[str] = {str(item.action_id) for item in legal_actions}
        supported_ids = _randomized_supported_action_ids(
            state=state,
            evidence=evidence,
            legal_action_ids=legal_action_ids,
        )
        supported = tuple(item for item in legal_actions if item.action_id in supported_ids)
        public_evidence = tuple(
            sorted({checksum for rule in evidence.rules for checksum in rule.evidence_reference_checksums})
        )
    if not supported:
        raise ValueError("Public evidence and policy must support at least one feasible action.")
    return TrialDevSupportedActionSetV1(
        state_checksum=cast(str, state.checksum),
        checkpoint_id=state.current_checkpoint_id,
        submitted_analysis_method_id=evidence.analysis_method_id,
        policy_binding_checksum=cast(str, state.policy_binding.checksum),
        legal_actions=legal_actions,
        supported_actions=supported,
        public_evidence_checksums=public_evidence,
        sensitivity_policy_id=sensitivity_policy_id,
    )


__all__ = ["derive_supported_action_set_v1"]
