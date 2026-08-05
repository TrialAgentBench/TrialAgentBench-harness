"""Tests for method-conditioned TrialDev supported-action policies."""

from __future__ import annotations

import pytest

from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevDecisionRuleEvidenceV1,
    TrialDevEvidenceReferenceV1,
    TrialDevObservationalCandidateEvidenceV1,
    TrialDevObservationalDecisionEvidenceV1,
    TrialDevPairContrastEvidenceV1,
    TrialDevPolicyBindingV1,
    TrialDevPortfolioProgrammeStateV1,
    TrialDevRandomizedDecisionEvidenceV1,
    TrialDevResourceScheduleV1,
    TrialDevRuleClassificationV1,
    TrialDevRuleDomainV1,
    TrialDevSingleAssetProgrammeStateV1,
    TrialDevSupportedActionSetV1,
)
from trialagentbench_harness.trialdev.policy import derive_supported_action_set_v1

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def _binding(*, portfolio: bool, budget: int = 10) -> TrialDevPolicyBindingV1:
    return TrialDevPolicyBindingV1(
        stream_id=("bounded_portfolio_reallocation" if portfolio else "single_asset_development"),
        objective_id="benefit_risk",
        objective_policy_checksum=_SHA_A,
        action_policy_checksum=_SHA_B,
        design_menu_checksum=_SHA_C,
        resource_schedule=TrialDevResourceScheduleV1() if portfolio else None,
        resource_budget_units=budget if portfolio else None,
    )


def _evidence(*, checkpoint: str, asset: str) -> TrialDevEvidenceReferenceV1:
    return TrialDevEvidenceReferenceV1(
        evidence_id=f"{asset}-{checkpoint}",
        evidence_kind="dataset",
        checkpoint_id=checkpoint,
        asset_id=asset,
        evidence_protocol_id="fixed_protocol_v1",
        evidence_protocol_checksum=_SHA_A,
        source_family_id=_SHA_C,
        world_id="world",
        generation_seed=31,
        relative_path=f"public/{asset}-{checkpoint}.csv",
        artifact_sha256=_SHA_B,
    )


def _rule(
    *,
    asset: str,
    domain: TrialDevRuleDomainV1,
    classification: TrialDevRuleClassificationV1,
    evidence: TrialDevEvidenceReferenceV1,
) -> TrialDevDecisionRuleEvidenceV1:
    direction = "maximum" if domain == "safety" else "minimum"
    if direction == "minimum":
        values = {
            "clear_pass": (0.8, 0.7, 0.9),
            "clear_fail": (0.2, 0.1, 0.3),
            "indeterminate": (0.5, 0.3, 0.7),
        }[classification]
    else:
        values = {
            "clear_pass": (0.2, 0.1, 0.3),
            "clear_fail": (0.7, 0.6, 0.8),
            "indeterminate": (0.5, 0.3, 0.7),
        }[classification]
    estimate, lower, upper = values
    return TrialDevDecisionRuleEvidenceV1(
        rule_id=f"{asset}-{domain}",
        asset_id=asset,
        domain=domain,
        direction=direction,
        estimate=estimate,
        lower_bound=lower,
        upper_bound=upper,
        threshold=0.5,
        evidence_reference_checksums=(str(evidence.checksum),),
    )


def _randomized_evidence(
    *,
    state: TrialDevSingleAssetProgrammeStateV1 | TrialDevPortfolioProgrammeStateV1,
    classes: dict[str, dict[TrialDevRuleDomainV1, TrialDevRuleClassificationV1]],
) -> TrialDevRandomizedDecisionEvidenceV1:
    evidence_by_asset = {item.asset_id: item for item in state.evidence if item.asset_id is not None}
    return TrialDevRandomizedDecisionEvidenceV1(
        state_checksum=str(state.checksum),
        analysis_method_id="prespecified_interval_method_v1",
        rules=tuple(
            _rule(
                asset=asset,
                domain=domain,
                classification=classification,
                evidence=evidence_by_asset[asset],
            )
            for asset, domains in classes.items()
            for domain, classification in domains.items()
        ),
    )


def _signatures(result: TrialDevSupportedActionSetV1) -> set[tuple[str, str | None, str | None]]:
    supported = result.supported_actions
    return {(item.action_id, item.target_asset_id, item.reserve_asset_id) for item in supported}


def test_randomized_rule_rejects_self_classified_feasibility() -> None:
    """Keep feasibility in exact design and resource checks, not statistical rules."""

    evidence = _evidence(checkpoint="proof_of_concept", asset="A")
    with pytest.raises(ValueError, match="domain"):
        TrialDevDecisionRuleEvidenceV1.model_validate(
            {
                "rule_id": "A-feasibility",
                "asset_id": "A",
                "domain": "feasibility",
                "direction": "minimum",
                "estimate": 0.8,
                "lower_bound": 0.7,
                "upper_bound": 0.9,
                "threshold": 0.5,
                "evidence_reference_checksums": [str(evidence.checksum)],
            }
        )


@pytest.mark.parametrize(
    ("domain", "direction"),
    [("efficacy", "maximum"), ("safety", "minimum")],
)
def test_randomized_rule_rejects_reversed_domain_direction(
    domain: TrialDevRuleDomainV1,
    direction: str,
) -> None:
    evidence = _evidence(checkpoint="proof_of_concept", asset="A")
    with pytest.raises(ValueError, match="requires direction"):
        TrialDevDecisionRuleEvidenceV1.model_validate(
            {
                "rule_id": f"A-{domain}",
                "asset_id": "A",
                "domain": domain,
                "direction": direction,
                "estimate": 0.5,
                "lower_bound": 0.4,
                "upper_bound": 0.6,
                "threshold": 0.5,
                "evidence_reference_checksums": [str(evidence.checksum)],
            }
        )


def test_single_asset_observational_policy_accepts_multiple_candidates_and_withholding() -> None:
    visible = tuple(_evidence(checkpoint="observational_review", asset=asset) for asset in ("A", "B", "C"))
    state = TrialDevSingleAssetProgrammeStateV1(
        programme_id="single",
        scenario_id="scenario",
        current_checkpoint_id="observational_review",
        candidate_asset_ids=("A", "B", "C"),
        policy_binding=_binding(portfolio=False),
        evidence=visible,
    )
    evidence = TrialDevObservationalDecisionEvidenceV1(
        state_checksum=str(state.checksum),
        analysis_method_id="adjusted_survival_v1",
        identification_status="identified",
        minimum_efficacy_gain=0.5,
        practical_equivalence_margin=0.1,
        candidates=tuple(
            TrialDevObservationalCandidateEvidenceV1(
                asset_id=asset,
                utility_estimate=utility,
                utility_lower_bound=utility - 0.1,
                utility_upper_bound=utility + 0.1,
                efficacy_estimate=0.5 if asset in {"A", "B"} else 0.2,
                efficacy_lower_bound=0.4 if asset in {"A", "B"} else 0.1,
                efficacy_upper_bound=0.6 if asset in {"A", "B"} else 0.3,
                evidence_reference_checksums=(str(item.checksum),),
            )
            for asset, utility, item in zip(("A", "B", "C"), (1.0, 0.95, 0.4), visible, strict=True)
        ),
        pair_contrasts=tuple(
            TrialDevPairContrastEvidenceV1(
                lead_asset_id=first,
                reserve_asset_id=second,
                confidence_half_width=0.1,
            )
            for index, first in enumerate(("A", "B", "C"))
            for second in ("A", "B", "C")[index + 1 :]
        ),
    )

    result = derive_supported_action_set_v1(state=state, evidence=evidence)

    assert _signatures(result) == {
        ("nominate_for_early_study", "A", None),
        ("nominate_for_early_study", "B", None),
        ("withhold_nomination", None, None),
    }


def test_nonidentified_portfolio_supports_only_qualified_withholding() -> None:
    visible = tuple(_evidence(checkpoint="observational_review", asset=asset) for asset in ("A", "B", "C"))
    state = TrialDevPortfolioProgrammeStateV1(
        programme_id="portfolio",
        scenario_id="scenario",
        current_checkpoint_id="observational_review",
        candidate_asset_ids=("A", "B", "C"),
        policy_binding=_binding(portfolio=True),
        evidence=visible,
    )
    evidence = TrialDevObservationalDecisionEvidenceV1(
        state_checksum=str(state.checksum),
        analysis_method_id="sensitivity_bounds_v1",
        identification_status="not_identified",
        minimum_efficacy_gain=0.5,
        practical_equivalence_margin=0.1,
        identification_evidence_reference_checksums=tuple(str(item.checksum) for item in visible),
    )

    result = derive_supported_action_set_v1(state=state, evidence=evidence)

    assert _signatures(result) == {("withhold_selection", None, None)}


def test_portfolio_allocation_policy_preserves_multiple_supported_orderings() -> None:
    visible = tuple(_evidence(checkpoint="observational_review", asset=asset) for asset in ("A", "B", "C"))
    state = TrialDevPortfolioProgrammeStateV1(
        programme_id="portfolio",
        scenario_id="scenario",
        current_checkpoint_id="observational_review",
        candidate_asset_ids=("A", "B", "C"),
        policy_binding=_binding(portfolio=True),
        evidence=visible,
    )
    utilities = {"A": 1.0, "B": 0.95, "C": 0.4}
    evidence = TrialDevObservationalDecisionEvidenceV1(
        state_checksum=str(state.checksum),
        analysis_method_id="adjusted_survival_v1",
        identification_status="identified",
        minimum_efficacy_gain=0.5,
        practical_equivalence_margin=0.05,
        candidates=tuple(
            TrialDevObservationalCandidateEvidenceV1(
                asset_id=asset,
                utility_estimate=utilities[asset],
                utility_lower_bound=utilities[asset] - 0.1,
                utility_upper_bound=utilities[asset] + 0.1,
                efficacy_estimate=0.7 if asset == "A" else 0.5,
                efficacy_lower_bound=0.6 if asset == "A" else 0.4,
                efficacy_upper_bound=0.8 if asset == "A" else 0.6,
                evidence_reference_checksums=(str(item.checksum),),
            )
            for asset, item in zip(("A", "B", "C"), visible, strict=True)
        ),
        pair_contrasts=tuple(
            TrialDevPairContrastEvidenceV1(
                lead_asset_id=first,
                reserve_asset_id=second,
                confidence_half_width=0.1,
            )
            for index, first in enumerate(("A", "B", "C"))
            for second in ("A", "B", "C")[index + 1 :]
        ),
    )

    result = derive_supported_action_set_v1(state=state, evidence=evidence)

    assert _signatures(result) == {
        ("select_lead_and_reserve", "A", "B"),
        ("select_lead_and_reserve", "B", "A"),
    }


@pytest.mark.parametrize(
    ("classification", "expected"),
    [
        ("clear_pass", {"advance_to_confirmation"}),
        ("clear_fail", {"stop_development"}),
        ("indeterminate", {"advance_to_confirmation", "stop_development"}),
    ],
)
def test_single_asset_proof_of_concept_policy_is_set_valued_under_uncertainty(
    classification: TrialDevRuleClassificationV1,
    expected: set[str],
) -> None:
    visible = _evidence(checkpoint="proof_of_concept", asset="A")
    state = TrialDevSingleAssetProgrammeStateV1(
        programme_id="single",
        scenario_id="scenario",
        current_checkpoint_id="proof_of_concept",
        candidate_asset_ids=("A",),
        nominated_asset_id="A",
        active_asset_id="A",
        policy_binding=_binding(portfolio=False),
        evidence=(visible,),
    )
    evidence = _randomized_evidence(
        state=state,
        classes={
            "A": {
                "efficacy": classification,
                "safety": classification,
            }
        },
    )

    result = derive_supported_action_set_v1(state=state, evidence=evidence)

    assert {item.action_id for item in result.supported_actions} == expected


def test_joint_early_policy_excludes_promotion_when_lead_clearly_passes() -> None:
    lead = _evidence(checkpoint="joint_early_study_review", asset="A")
    reserve = _evidence(checkpoint="joint_early_study_review", asset="B")
    state = TrialDevPortfolioProgrammeStateV1(
        programme_id="portfolio",
        scenario_id="scenario",
        current_checkpoint_id="joint_early_study_review",
        candidate_asset_ids=("A", "B", "C"),
        lead_asset_id="A",
        reserve_asset_id="B",
        active_asset_id="A",
        retired_asset_ids=("C",),
        resource_spent_units=2,
        policy_binding=_binding(portfolio=True),
        evidence=(lead, reserve),
    )
    evidence = _randomized_evidence(
        state=state,
        classes={
            "A": {"safety": "clear_pass"},
            "B": {"safety": "indeterminate"},
        },
    )

    result = derive_supported_action_set_v1(state=state, evidence=evidence)

    assert {item.action_id for item in result.supported_actions} == {"advance_lead_to_proof_of_concept"}


def test_eight_unit_budget_removes_late_promotion_before_supported_set_derivation() -> None:
    visible = _evidence(checkpoint="lead_proof_of_concept_review", asset="A")
    state = TrialDevPortfolioProgrammeStateV1(
        programme_id="portfolio",
        scenario_id="scenario",
        current_checkpoint_id="lead_proof_of_concept_review",
        candidate_asset_ids=("A", "B", "C"),
        lead_asset_id="A",
        reserve_asset_id="B",
        active_asset_id="A",
        retired_asset_ids=("C",),
        resource_spent_units=4,
        policy_binding=_binding(portfolio=True, budget=8),
        evidence=(visible,),
    )
    evidence = _randomized_evidence(
        state=state,
        classes={
            "A": {
                "efficacy": "indeterminate",
                "safety": "clear_pass",
            }
        },
    )

    result = derive_supported_action_set_v1(state=state, evidence=evidence)

    assert {item.action_id for item in result.legal_actions} == {
        "advance_active_to_confirmation",
        "terminate_portfolio",
    }
    assert {item.action_id for item in result.supported_actions} == {
        "advance_active_to_confirmation",
        "terminate_portfolio",
    }


@pytest.mark.parametrize(
    ("classification", "expected"),
    [
        ("clear_pass", "declare_success"),
        ("clear_fail", "declare_failure"),
        ("indeterminate", "declare_inconclusive"),
    ],
)
def test_confirmation_policy_has_an_exhaustive_exclusive_partition(
    classification: TrialDevRuleClassificationV1,
    expected: str,
) -> None:
    visible = _evidence(checkpoint="confirmation", asset="A")
    state = TrialDevSingleAssetProgrammeStateV1(
        programme_id="single",
        scenario_id="scenario",
        current_checkpoint_id="confirmation",
        candidate_asset_ids=("A",),
        nominated_asset_id="A",
        active_asset_id="A",
        policy_binding=_binding(portfolio=False),
        evidence=(visible,),
    )
    evidence = _randomized_evidence(
        state=state,
        classes={"A": {"efficacy": classification, "safety": classification}},
    )

    result = derive_supported_action_set_v1(state=state, evidence=evidence)

    assert tuple(item.action_id for item in result.supported_actions) == (expected,)
