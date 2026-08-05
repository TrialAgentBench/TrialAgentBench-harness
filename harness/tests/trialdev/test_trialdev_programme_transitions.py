"""End-to-end tests for deterministic TrialDev programme transitions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevActionSelectionV1,
    TrialDevAssetEligibilityV1,
    TrialDevCheckpointActionPolicyV1,
    TrialDevCheckpointOutcomeV1,
    TrialDevEvidenceReferenceV1,
    TrialDevLegalActionSpecV1,
    TrialDevPolicyBindingV1,
    TrialDevPortfolioActionSelectionV1,
    TrialDevPortfolioEvidenceIndexV1,
    TrialDevPortfolioProgrammeStateV1,
    TrialDevProgrammeStateV1,
    TrialDevResourceScheduleV1,
    TrialDevSingleAssetActionSelectionV1,
    TrialDevSingleAssetProgrammeStateV1,
)
from trialagentbench_harness.trialdev.grading.hashing import sha256_file_hex
from trialagentbench_harness.trialdev.grading.sequential import build_initial_program_state_v1
from trialagentbench_harness.trialdev.programme import (
    build_checkpoint_action_policy_v1,
    transition_portfolio_programme_state_v1,
    transition_programme_state_v1,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def test_initial_single_asset_state_uses_only_participant_verifiable_provenance(
    tmp_path: Path,
) -> None:
    scenario = tmp_path / "scenario_s01"
    public = scenario / "public"
    public.mkdir(parents=True)
    manifest = tmp_path / "benchmark_suite_manifest.json"
    manifest.write_text('{"suite_id":"release"}\n', encoding="utf-8")
    for name in (
        "phase_action_policy.json",
        "objective_charter.json",
        "phase_module_catalog.json",
        "observational_method_catalog.json",
    ):
        (public / name).write_text("{}\n", encoding="utf-8")
    (public / "observational_extract.parquet").write_bytes(b"fixed public dataset")
    (public / "eval_contract.json").write_text(json.dumps({"scenario_id": "s01"}), encoding="utf-8")
    (public / "candidate_drug_catalog.json").write_text(
        json.dumps(
            {
                "candidate_drugs": [
                    {"candidate_drug_id": "control", "role": "control"},
                    {"candidate_drug_id": "candidate", "role": "investigational"},
                ]
            }
        ),
        encoding="utf-8",
    )

    state = build_initial_program_state_v1(
        scenario_root=scenario,
        programme_id="s01__benefit_risk",
        objective_id="benefit_risk",
    )

    assert {item.source_family_id for item in state.evidence} == {sha256_file_hex(manifest)}
    assert {item.world_id for item in state.evidence} == {"s01"}
    assert state.evidence[1].generation_seed is None
    assert not (scenario / "hidden").exists()


def test_only_dataset_evidence_can_disclose_a_generation_seed() -> None:
    with pytest.raises(ValueError, match="Only dataset evidence"):
        TrialDevEvidenceReferenceV1(
            evidence_id="protocol",
            evidence_kind="protocol",
            checkpoint_id="observational_review",
            asset_id=None,
            evidence_protocol_id="protocol",
            evidence_protocol_checksum=_SHA_A,
            source_family_id=_SHA_C,
            world_id="world",
            generation_seed=17,
            relative_path="public/protocol.json",
            artifact_sha256=_SHA_B,
        )


def _policy_binding(*, stream_id: str, budget: int | None = None) -> TrialDevPolicyBindingV1:
    return TrialDevPolicyBindingV1(
        stream_id=stream_id,
        objective_id="maximise_benefit",
        objective_policy_checksum=_SHA_A,
        action_policy_checksum=_SHA_B,
        design_menu_checksum=_SHA_C,
        resource_schedule=TrialDevResourceScheduleV1() if budget is not None else None,
        resource_budget_units=budget,
    )


def _evidence(*, evidence_id: str, checkpoint_id: str, asset_id: str) -> TrialDevEvidenceReferenceV1:
    return TrialDevEvidenceReferenceV1(
        evidence_id=evidence_id,
        evidence_kind="dataset",
        checkpoint_id=checkpoint_id,
        asset_id=asset_id,
        evidence_protocol_id="fixed_protocol",
        evidence_protocol_checksum=_SHA_A,
        source_family_id=_SHA_C,
        world_id="world",
        generation_seed=17,
        relative_path=f"public/evidence/{evidence_id}.json",
        artifact_sha256=_SHA_B,
    )


def _outcome() -> TrialDevCheckpointOutcomeV1:
    return TrialDevCheckpointOutcomeV1(
        reach_status="reached",
        submission_status="accepted",
        analysis_status="estimable",
        execution_status="completed",
    )


def _transition(
    state: TrialDevProgrammeStateV1,
    *,
    action_id: str,
    action_kind: str,
    target_asset_id: str | None = None,
    reserve_asset_id: str | None = None,
    next_evidence: tuple[TrialDevEvidenceReferenceV1, ...] = (),
) -> TrialDevProgrammeStateV1:
    action = TrialDevLegalActionSpecV1(
        action_id=action_id,
        action_kind=action_kind,
        requires_target_asset=target_asset_id is not None,
        requires_reserve_asset=reserve_asset_id is not None,
        consumes_switch=action_id == "promote_reserve_to_proof_of_concept",
    )
    policy = TrialDevCheckpointActionPolicyV1(
        stream_id=state.stream_id,
        checkpoint_id=state.current_checkpoint_id,
        policy_binding_checksum=state.policy_binding.checksum,
        actions=(action,),
    )
    selection = TrialDevActionSelectionV1(
        state_checksum=state.checksum,
        checkpoint_id=state.current_checkpoint_id,
        action_id=action_id,
        target_asset_id=target_asset_id,
        reserve_asset_id=reserve_asset_id,
        analysis_method_id="published_method",
        supporting_evidence_ids=tuple(item.evidence_id for item in state.evidence),
        justification="The selected action is supported by the cited participant-visible evidence.",
    )
    return transition_programme_state_v1(
        state=state,
        action_policy=policy,
        selection=selection,
        outcome=_outcome(),
        next_evidence=next_evidence,
    )


def test_single_asset_chain_is_irreversible_and_can_end_inconclusive() -> None:
    state = TrialDevSingleAssetProgrammeStateV1(
        programme_id="single",
        scenario_id="scenario",
        stream_id="single_asset_development",
        current_checkpoint_id="observational_review",
        candidate_asset_ids=("A", "B"),
        policy_binding=_policy_binding(stream_id="single_asset_development"),
        evidence=(_evidence(evidence_id="obs", checkpoint_id="observational_review", asset_id="A"),),
    )
    state = _transition(
        state,
        action_id="nominate_for_early_study",
        action_kind="allocate",
        target_asset_id="A",
        next_evidence=(_evidence(evidence_id="early", checkpoint_id="early_safety_study", asset_id="A"),),
    )
    state = _transition(
        state,
        action_id="advance_to_proof_of_concept",
        action_kind="advance",
        next_evidence=(_evidence(evidence_id="poc", checkpoint_id="proof_of_concept", asset_id="A"),),
    )
    state = _transition(
        state,
        action_id="advance_to_confirmation",
        action_kind="advance",
        next_evidence=(_evidence(evidence_id="confirm", checkpoint_id="confirmation", asset_id="A"),),
    )
    state = _transition(state, action_id="declare_inconclusive", action_kind="terminal")

    assert state.terminal_disposition == "inconclusive"
    assert state.nominated_asset_id == state.active_asset_id == "A"
    assert state.retired_asset_ids == ("B",)
    assert len(state.history) == 4


@pytest.mark.parametrize(
    ("checkpoint_id", "action_id", "next_checkpoint", "terminal_disposition"),
    [
        ("observational_review", "nominate_for_early_study", "early_safety_study", "active"),
        ("observational_review", "withhold_nomination", "observational_review", "withheld"),
        ("early_safety_study", "advance_to_proof_of_concept", "proof_of_concept", "active"),
        ("early_safety_study", "stop_development", "early_safety_study", "stopped"),
        ("proof_of_concept", "advance_to_confirmation", "confirmation", "active"),
        ("proof_of_concept", "stop_development", "proof_of_concept", "stopped"),
        ("confirmation", "declare_success", "confirmation", "success"),
        ("confirmation", "declare_failure", "confirmation", "failure"),
        ("confirmation", "declare_inconclusive", "confirmation", "inconclusive"),
    ],
)
def test_every_single_asset_action_has_one_materializable_transition(
    checkpoint_id: str,
    action_id: str,
    next_checkpoint: str,
    terminal_disposition: str,
) -> None:
    policy_binding = _policy_binding(stream_id="single_asset_development")
    committed = checkpoint_id != "observational_review"
    current_evidence = _evidence(
        evidence_id=f"evidence-{checkpoint_id}",
        checkpoint_id=checkpoint_id,
        asset_id="A",
    )
    state = TrialDevSingleAssetProgrammeStateV1(
        programme_id="single",
        scenario_id="scenario",
        current_checkpoint_id=checkpoint_id,
        candidate_asset_ids=("A", "B"),
        nominated_asset_id="A" if committed else None,
        active_asset_id="A" if committed else None,
        retired_asset_ids=("B",) if committed else (),
        policy_binding=policy_binding,
        evidence=(current_evidence,),
    )
    selection = TrialDevSingleAssetActionSelectionV1(
        state_checksum=str(state.checksum),
        checkpoint_id=checkpoint_id,
        action_id=action_id,
        target_asset_id="A" if action_id == "nominate_for_early_study" else None,
        analysis_method_id="published_method",
        supporting_evidence_ids=(current_evidence.evidence_id,),
        justification="Action is supported by the current checkpoint evidence.",
    )
    next_evidence = (
        (_evidence(evidence_id=f"evidence-{next_checkpoint}", checkpoint_id=next_checkpoint, asset_id="A"),)
        if terminal_disposition == "active"
        else ()
    )

    result = transition_programme_state_v1(
        state=state,
        action_policy=build_checkpoint_action_policy_v1(
            state=state,
        ),
        selection=selection,
        outcome=_outcome(),
        next_evidence=next_evidence,
    )

    assert result.current_checkpoint_id == next_checkpoint
    assert result.terminal_disposition == terminal_disposition


def test_eight_unit_portfolio_supports_early_but_not_late_promotion() -> None:
    initial = TrialDevPortfolioProgrammeStateV1(
        programme_id="portfolio",
        scenario_id="scenario",
        stream_id="bounded_portfolio_reallocation",
        current_checkpoint_id="observational_review",
        candidate_asset_ids=("A", "B", "C"),
        policy_binding=_policy_binding(stream_id="bounded_portfolio_reallocation", budget=8),
        evidence=(_evidence(evidence_id="obs", checkpoint_id="observational_review", asset_id="A"),),
    )
    joint = _transition(
        initial,
        action_id="select_lead_and_reserve",
        action_kind="allocate",
        target_asset_id="A",
        reserve_asset_id="B",
        next_evidence=(
            _evidence(evidence_id="lead-early", checkpoint_id="joint_early_study_review", asset_id="A"),
            _evidence(evidence_id="reserve-early", checkpoint_id="joint_early_study_review", asset_id="B"),
        ),
    )
    promoted = _transition(
        joint,
        action_id="promote_reserve_to_proof_of_concept",
        action_kind="promote",
        next_evidence=(
            _evidence(
                evidence_id="reserve-poc",
                checkpoint_id="promoted_reserve_proof_of_concept_review",
                asset_id="B",
            ),
        ),
    )
    confirmation = _transition(
        promoted,
        action_id="advance_active_to_confirmation",
        action_kind="advance",
        next_evidence=(_evidence(evidence_id="confirm", checkpoint_id="confirmation", asset_id="B"),),
    )

    assert confirmation.resource_spent_units == 8
    assert confirmation.switch_count == 1
    assert confirmation.active_asset_id == "B"

    lead_poc = _transition(
        joint,
        action_id="advance_lead_to_proof_of_concept",
        action_kind="advance",
        next_evidence=(_evidence(evidence_id="lead-poc", checkpoint_id="lead_proof_of_concept_review", asset_id="A"),),
    )
    with pytest.raises(ValueError, match="required remaining path"):
        _transition(
            lead_poc,
            action_id="promote_reserve_to_proof_of_concept",
            action_kind="promote",
        )


def test_transition_rejects_evidence_that_was_not_visible_at_the_checkpoint() -> None:
    state = TrialDevSingleAssetProgrammeStateV1(
        programme_id="single",
        scenario_id="scenario",
        stream_id="single_asset_development",
        current_checkpoint_id="observational_review",
        candidate_asset_ids=("A",),
        policy_binding=_policy_binding(stream_id="single_asset_development"),
        evidence=(_evidence(evidence_id="obs", checkpoint_id="observational_review", asset_id="A"),),
    )
    action = TrialDevLegalActionSpecV1(
        action_id="nominate_for_early_study",
        action_kind="allocate",
        requires_target_asset=True,
    )
    policy = TrialDevCheckpointActionPolicyV1(
        stream_id=state.stream_id,
        checkpoint_id=state.current_checkpoint_id,
        policy_binding_checksum=state.policy_binding.checksum,
        actions=(action,),
    )
    selection = TrialDevActionSelectionV1(
        state_checksum=state.checksum,
        checkpoint_id=state.current_checkpoint_id,
        action_id="nominate_for_early_study",
        target_asset_id="A",
        analysis_method_id="published_method",
        supporting_evidence_ids=("future",),
        justification="This citation was not available.",
    )

    with pytest.raises(ValueError, match="outside the current participant-visible state"):
        transition_programme_state_v1(
            state=state,
            action_policy=policy,
            selection=selection,
            outcome=_outcome(),
        )


def test_portfolio_index_exposes_only_action_conditioned_evidence_deterministically() -> None:
    index = TrialDevPortfolioEvidenceIndexV1(
        scenario_id="scenario",
        source_identity=_SHA_C,
        world_id="world",
        candidate_asset_ids=("A", "B", "C"),
        evidence=tuple(
            _evidence(
                evidence_id=f"{asset}-{checkpoint}",
                checkpoint_id=checkpoint,
                asset_id=asset,
            )
            for asset in ("A", "B", "C")
            for checkpoint in (
                "joint_early_study_review",
                "lead_proof_of_concept_review",
                "promoted_reserve_proof_of_concept_review",
                "confirmation",
            )
        ),
    )
    binding = _policy_binding(stream_id="bounded_portfolio_reallocation", budget=10)
    initial_evidence = _evidence(
        evidence_id="observational",
        checkpoint_id="observational_review",
        asset_id="A",
    )
    initial = TrialDevPortfolioProgrammeStateV1(
        programme_id="portfolio",
        scenario_id="scenario",
        current_checkpoint_id="observational_review",
        candidate_asset_ids=("A", "B", "C"),
        policy_binding=binding,
        evidence=(initial_evidence,),
    )
    selection = TrialDevPortfolioActionSelectionV1(
        state_checksum=str(initial.checksum),
        checkpoint_id="observational_review",
        action_id="select_lead_and_reserve",
        target_asset_id="A",
        reserve_asset_id="B",
        analysis_method_id="published_method",
        supporting_evidence_ids=(initial_evidence.evidence_id,),
        justification="A is selected as lead and B is retained as reserve.",
    )
    policy = build_checkpoint_action_policy_v1(
        state=initial,
    )

    first = transition_portfolio_programme_state_v1(
        state=initial,
        evidence_index=index,
        action_policy=policy,
        selection=selection,
        outcome=_outcome(),
    )
    second = transition_portfolio_programme_state_v1(
        state=initial,
        evidence_index=index,
        action_policy=policy,
        selection=selection,
        outcome=_outcome(),
    )

    assert first.checksum == second.checksum
    exposed = {item.evidence_id for item in first.evidence}
    assert {
        "A-joint_early_study_review",
        "B-joint_early_study_review",
    } <= exposed
    assert "C-joint_early_study_review" not in exposed
    assert "A-lead_proof_of_concept_review" not in exposed
    assert "B-promoted_reserve_proof_of_concept_review" not in exposed


def test_clear_safety_failure_permanently_removes_reserve_from_feasible_actions() -> None:
    binding = _policy_binding(stream_id="bounded_portfolio_reallocation", budget=10)
    lead_evidence = _evidence(
        evidence_id="A-early",
        checkpoint_id="joint_early_study_review",
        asset_id="A",
    )
    reserve_evidence = _evidence(
        evidence_id="B-early",
        checkpoint_id="joint_early_study_review",
        asset_id="B",
    )
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
        policy_binding=binding,
        evidence=(lead_evidence, reserve_evidence),
    )
    outcome = TrialDevCheckpointOutcomeV1(
        reach_status="reached",
        submission_status="accepted",
        analysis_status="estimable",
        execution_status="completed",
        asset_eligibility=(
            TrialDevAssetEligibilityV1(
                asset_id="B",
                status="permanently_ineligible",
                reason="safety_clear_fail",
                policy_rule_id="early_safety_bound_v1",
                evidence_reference_checksums=(str(reserve_evidence.checksum),),
            ),
        ),
    )
    selection = TrialDevPortfolioActionSelectionV1(
        state_checksum=str(state.checksum),
        checkpoint_id=state.current_checkpoint_id,
        action_id="advance_lead_to_proof_of_concept",
        analysis_method_id="published_method",
        supporting_evidence_ids=(lead_evidence.evidence_id, reserve_evidence.evidence_id),
        justification="The lead remains eligible; the reserve crossed the prespecified safety bound.",
    )
    result = transition_programme_state_v1(
        state=state,
        action_policy=build_checkpoint_action_policy_v1(state=state),
        selection=selection,
        outcome=outcome,
        next_evidence=(
            _evidence(
                evidence_id="A-poc",
                checkpoint_id="lead_proof_of_concept_review",
                asset_id="A",
            ),
        ),
    )

    assert result.permanently_ineligible_asset_ids == ("B",)
    assert set(result.retired_asset_ids) == {"B", "C"}
    assert {action.action_id for action in build_checkpoint_action_policy_v1(state=result).actions} == {
        "advance_active_to_confirmation",
        "terminate_portfolio",
    }


@pytest.mark.parametrize(
    ("checkpoint_id", "active_asset", "retired", "spent", "switch_count", "expected_actions"),
    [
        (
            "observational_review",
            None,
            (),
            0,
            0,
            {"select_lead_and_reserve", "withhold_selection"},
        ),
        (
            "joint_early_study_review",
            "A",
            ("C",),
            2,
            0,
            {
                "advance_lead_to_proof_of_concept",
                "promote_reserve_to_proof_of_concept",
                "terminate_portfolio",
            },
        ),
        (
            "lead_proof_of_concept_review",
            "A",
            ("C",),
            4,
            0,
            {
                "advance_active_to_confirmation",
                "promote_reserve_to_proof_of_concept",
                "terminate_portfolio",
            },
        ),
        (
            "promoted_reserve_proof_of_concept_review",
            "B",
            ("A", "C"),
            4,
            1,
            {"advance_active_to_confirmation", "terminate_portfolio"},
        ),
        (
            "confirmation",
            "A",
            ("B", "C"),
            8,
            0,
            {"declare_success", "declare_failure", "declare_inconclusive"},
        ),
    ],
)
def test_every_portfolio_state_exposes_exactly_its_feasible_actions(
    checkpoint_id: str,
    active_asset: str | None,
    retired: tuple[str, ...],
    spent: int,
    switch_count: int,
    expected_actions: set[str],
) -> None:
    observational = checkpoint_id == "observational_review"
    evidence_assets = ("A",) if observational or checkpoint_id != "joint_early_study_review" else ("A", "B")
    state = TrialDevPortfolioProgrammeStateV1(
        programme_id="portfolio",
        scenario_id="scenario",
        current_checkpoint_id=checkpoint_id,
        candidate_asset_ids=("A", "B", "C"),
        lead_asset_id=None if observational else "A",
        reserve_asset_id=None if observational else "B",
        active_asset_id=active_asset,
        retired_asset_ids=retired,
        resource_spent_units=spent,
        switch_count=switch_count,
        policy_binding=_policy_binding(stream_id="bounded_portfolio_reallocation", budget=10),
        evidence=tuple(
            _evidence(
                evidence_id=f"{asset}-{checkpoint_id}",
                checkpoint_id=checkpoint_id,
                asset_id=asset,
            )
            for asset in evidence_assets
        ),
    )

    policy = build_checkpoint_action_policy_v1(state=state)

    assert {action.action_id for action in policy.actions} == expected_actions
    next_checkpoint_by_action = {
        "select_lead_and_reserve": "joint_early_study_review",
        "advance_lead_to_proof_of_concept": "lead_proof_of_concept_review",
        "promote_reserve_to_proof_of_concept": "promoted_reserve_proof_of_concept_review",
        "advance_active_to_confirmation": "confirmation",
    }
    terminal_actions = {
        "withhold_selection",
        "terminate_portfolio",
        "declare_success",
        "declare_failure",
        "declare_inconclusive",
    }
    for action in policy.actions:
        target = "A" if action.action_id == "select_lead_and_reserve" else None
        reserve = "B" if action.action_id == "select_lead_and_reserve" else None
        selection = TrialDevPortfolioActionSelectionV1(
            state_checksum=str(state.checksum),
            checkpoint_id=state.current_checkpoint_id,
            action_id=action.action_id,
            target_asset_id=target,
            reserve_asset_id=reserve,
            analysis_method_id="published_method",
            supporting_evidence_ids=tuple(item.evidence_id for item in state.evidence),
            justification="The action is supported by the current evidence and resource state.",
        )
        next_checkpoint = next_checkpoint_by_action.get(action.action_id)
        if next_checkpoint == "joint_early_study_review":
            next_evidence = tuple(
                _evidence(
                    evidence_id=f"next-{asset}",
                    checkpoint_id=next_checkpoint,
                    asset_id=asset,
                )
                for asset in ("A", "B")
            )
        elif next_checkpoint is not None:
            next_asset = "B" if "reserve" in action.action_id else active_asset
            if next_asset is None:
                raise AssertionError("An evidence-producing action requires an active asset.")
            next_evidence = (
                _evidence(
                    evidence_id=f"next-{action.action_id}",
                    checkpoint_id=next_checkpoint,
                    asset_id=next_asset,
                ),
            )
        else:
            next_evidence = ()
        first = transition_programme_state_v1(
            state=state,
            action_policy=policy,
            selection=selection,
            outcome=_outcome(),
            next_evidence=next_evidence,
        )
        second = transition_programme_state_v1(
            state=state,
            action_policy=policy,
            selection=selection,
            outcome=_outcome(),
            next_evidence=next_evidence,
        )
        assert first.checksum == second.checksum
        assert len(first.history) == 1
        assert first.resource_spent_units >= state.resource_spent_units
        assert (first.terminal_disposition != "active") == (action.action_id in terminal_actions)

    illegal = TrialDevPortfolioActionSelectionV1(
        state_checksum=str(state.checksum),
        checkpoint_id=state.current_checkpoint_id,
        action_id="withhold_selection",
        analysis_method_id="published_method",
        supporting_evidence_ids=tuple(item.evidence_id for item in state.evidence),
        justification="This action belongs to a different checkpoint.",
    )
    if "withhold_selection" not in expected_actions:
        with pytest.raises(ValueError, match="not legal"):
            transition_programme_state_v1(
                state=state,
                action_policy=policy,
                selection=illegal,
                outcome=_outcome(),
            )
