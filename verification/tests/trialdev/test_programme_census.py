"""Adversarial tests for the independent TrialDev programme census."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

from trialagentbench_validation import cli
from trialagentbench_validation.trialdev.programme_census import (
    TrialDevCensusActionV1,
    TrialDevCensusEvidenceV1,
    TrialDevCensusResourceScheduleV1,
    TrialDevCensusStateV1,
    TrialDevCensusSupportedSetV1,
    TrialDevCensusTransitionV1,
    TrialDevNumericalWitnessV1,
    TrialDevProgrammeCensusV1,
    audit_trialdev_programme_census,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_single_asset_census(
    tmp_path: Path,
) -> tuple[Path, TrialDevProgrammeCensusV1]:
    evidence: list[TrialDevCensusEvidenceV1] = []
    evidence_by_checkpoint: dict[str, TrialDevCensusEvidenceV1] = {}
    cumulative: list[str] = []
    for index, checkpoint in enumerate(
        (
            "observational_review",
            "early_safety_study",
            "proof_of_concept",
            "confirmation",
        ),
        start=1,
    ):
        path = tmp_path / "public" / f"{checkpoint}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"event\n1\n0\n{index % 2}\n", encoding="utf-8")
        row = TrialDevCensusEvidenceV1(
            evidence_id=checkpoint,
            checkpoint_id=checkpoint,
            asset_id="A",
            relative_path=f"public/{checkpoint}.csv",
            artifact_sha256=_sha256(path),
        )
        evidence.append(row)
        evidence_by_checkpoint[checkpoint] = row

    states: list[TrialDevCensusStateV1] = []
    state_by_checkpoint: dict[str, TrialDevCensusStateV1] = {}
    for checkpoint in (
        "observational_review",
        "early_safety_study",
        "proof_of_concept",
        "confirmation",
    ):
        cumulative.append(cast(str, evidence_by_checkpoint[checkpoint].checksum))
        committed = checkpoint != "observational_review"
        state = TrialDevCensusStateV1(
            programme_id="programme",
            stream_id="single_asset_development",
            checkpoint_id=checkpoint,
            candidate_asset_ids=("A",),
            nominated_asset_id="A" if committed else None,
            active_asset_id="A" if committed else None,
            evidence_checksums=tuple(cumulative),
        )
        states.append(state)
        state_by_checkpoint[checkpoint] = state

    action_specs = {
        "observational_review": (
            ("nominate-A", "nominate_for_early_study", "A"),
            ("withhold", "withhold_nomination", None),
        ),
        "early_safety_study": (
            ("advance-poc", "advance_to_proof_of_concept", None),
            ("stop-early", "stop_development", None),
        ),
        "proof_of_concept": (
            ("advance-confirm", "advance_to_confirmation", None),
            ("stop-poc", "stop_development", None),
        ),
        "confirmation": (
            ("success", "declare_success", None),
            ("failure", "declare_failure", None),
            ("inconclusive", "declare_inconclusive", None),
        ),
    }
    next_checkpoint = {
        "nominate_for_early_study": "early_safety_study",
        "advance_to_proof_of_concept": "proof_of_concept",
        "advance_to_confirmation": "confirmation",
    }
    terminal = {
        "withhold_nomination": "withheld",
        "stop_development": "stopped",
        "declare_success": "success",
        "declare_failure": "failure",
        "declare_inconclusive": "inconclusive",
    }
    actions: list[TrialDevCensusActionV1] = []
    transitions: list[TrialDevCensusTransitionV1] = []
    supported: list[TrialDevCensusSupportedSetV1] = []
    for checkpoint, rows in action_specs.items():
        state = state_by_checkpoint[checkpoint]
        checkpoint_actions: list[TrialDevCensusActionV1] = []
        for variant_id, action_id, target in rows:
            action = TrialDevCensusActionV1(
                state_checksum=cast(str, state.checksum),
                variant_id=variant_id,
                action_id=action_id,
                target_asset_id=target,
            )
            actions.append(action)
            checkpoint_actions.append(action)
            following = next_checkpoint.get(action_id)
            next_state = (
                state_by_checkpoint.get(following) if following is not None else None
            )
            transitions.append(
                TrialDevCensusTransitionV1(
                    state_checksum=cast(str, state.checksum),
                    action_variant_checksum=cast(str, action.checksum),
                    next_state_checksum=(
                        None if next_state is None else next_state.checksum
                    ),
                    terminal_disposition=terminal.get(action_id, "active"),
                    newly_exposed_evidence_checksums=(
                        ()
                        if following is None
                        else (cast(str, evidence_by_checkpoint[following].checksum),)
                    ),
                )
            )
        supported.append(
            TrialDevCensusSupportedSetV1(
                state_checksum=cast(str, state.checksum),
                analysis_method_id="risk_difference_v1",
                supported_action_variant_checksums=tuple(
                    cast(str, action.checksum) for action in checkpoint_actions
                ),
                evidence_checksums=state.evidence_checksums,
            )
        )
    observational_evidence = evidence_by_checkpoint["observational_review"]
    census = TrialDevProgrammeCensusV1(
        resource_schedule=TrialDevCensusResourceScheduleV1(
            early_study_units=1,
            proof_of_concept_units=2,
            confirmation_units=4,
            maximum_switches=1,
        ),
        evidence=tuple(evidence),
        states=tuple(states),
        actions=tuple(actions),
        transitions=tuple(transitions),
        supported_sets=tuple(supported),
        numerical_witnesses=(
            TrialDevNumericalWitnessV1(
                evidence_checksum=cast(str, observational_evidence.checksum),
                value_column="event",
                statistic="proportion",
                reported_value=2 / 3,
                absolute_tolerance=1e-12,
            ),
        ),
    )
    census_path = tmp_path / "programme_census.json"
    census_path.write_text(census.model_dump_json(indent=2), encoding="utf-8")
    return census_path, census


def _write_mutation(
    tmp_path: Path, census: TrialDevProgrammeCensusV1, **updates: object
) -> Path:
    mutated = TrialDevProgrammeCensusV1.model_validate(
        census.model_copy(update={"checksum": None, **updates}).model_dump(mode="json")
    )
    path = tmp_path / f"mutation-{len(tuple(tmp_path.glob('mutation-*.json')))}.json"
    path.write_text(mutated.model_dump_json(indent=2), encoding="utf-8")
    return path


def _build_portfolio_census(tmp_path: Path) -> tuple[Path, TrialDevProgrammeCensusV1]:
    assets = ("A", "B", "C")
    evidence: list[TrialDevCensusEvidenceV1] = []
    evidence_by_key: dict[tuple[str, str], TrialDevCensusEvidenceV1] = {}
    for asset_id in assets:
        for checkpoint in (
            "observational_review",
            "joint_early_study_review",
            "lead_proof_of_concept_review",
            "promoted_reserve_proof_of_concept_review",
            "confirmation",
        ):
            path = tmp_path / "public" / "portfolio" / f"{asset_id}-{checkpoint}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("event\n1\n0\n1\n", encoding="utf-8")
            row = TrialDevCensusEvidenceV1(
                evidence_id=f"{asset_id}-{checkpoint}",
                checkpoint_id=checkpoint,
                asset_id=asset_id,
                relative_path=f"public/portfolio/{asset_id}-{checkpoint}.csv",
                artifact_sha256=_sha256(path),
            )
            evidence.append(row)
            evidence_by_key[(asset_id, checkpoint)] = row

    def evidence_checksums(keys: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
        return tuple(cast(str, evidence_by_key[key].checksum) for key in keys)

    observational_keys = tuple(
        (asset_id, "observational_review") for asset_id in assets
    )
    states: list[TrialDevCensusStateV1] = []
    actions: list[TrialDevCensusActionV1] = []
    transitions: list[TrialDevCensusTransitionV1] = []
    supported_sets: list[TrialDevCensusSupportedSetV1] = []

    initial = TrialDevCensusStateV1(
        programme_id="portfolio",
        stream_id="bounded_portfolio_reallocation",
        checkpoint_id="observational_review",
        candidate_asset_ids=assets,
        resource_budget_units=10,
        evidence_checksums=evidence_checksums(observational_keys),
    )
    states.append(initial)

    def evolve_state(
        source: TrialDevCensusStateV1,
        **updates: object,
    ) -> TrialDevCensusStateV1:
        return TrialDevCensusStateV1.model_validate(
            source.model_copy(update={"checksum": None, **updates}).model_dump(
                mode="json"
            )
        )

    def add_action(
        *,
        source: TrialDevCensusStateV1,
        variant_id: str,
        action_id: str,
        target: TrialDevCensusStateV1 | None = None,
        target_asset: str | None = None,
        reserve_asset: str | None = None,
        terminal: str = "active",
    ) -> TrialDevCensusActionV1:
        action = TrialDevCensusActionV1(
            state_checksum=cast(str, source.checksum),
            variant_id=variant_id,
            action_id=action_id,
            target_asset_id=target_asset,
            reserve_asset_id=reserve_asset,
        )
        source_evidence = set(source.evidence_checksums)
        target_evidence = set() if target is None else set(target.evidence_checksums)
        actions.append(action)
        transitions.append(
            TrialDevCensusTransitionV1(
                state_checksum=cast(str, source.checksum),
                action_variant_checksum=cast(str, action.checksum),
                next_state_checksum=None if target is None else target.checksum,
                terminal_disposition=terminal,
                newly_exposed_evidence_checksums=tuple(
                    sorted(target_evidence - source_evidence)
                ),
            )
        )
        return action

    def add_terminal_actions(state: TrialDevCensusStateV1) -> None:
        for action_id, disposition in (
            ("declare_success", "success"),
            ("declare_failure", "failure"),
            ("declare_inconclusive", "inconclusive"),
        ):
            add_action(
                source=state,
                variant_id=f"{state.checksum}-{action_id}",
                action_id=action_id,
                terminal=disposition,
            )

    initial_actions: list[TrialDevCensusActionV1] = []
    for lead in assets:
        for reserve in assets:
            if lead == reserve:
                continue
            retired = tuple(asset for asset in assets if asset not in {lead, reserve})
            early_keys = (
                *observational_keys,
                (lead, "joint_early_study_review"),
                (reserve, "joint_early_study_review"),
            )
            joint = TrialDevCensusStateV1(
                programme_id="portfolio",
                stream_id="bounded_portfolio_reallocation",
                checkpoint_id="joint_early_study_review",
                candidate_asset_ids=assets,
                lead_asset_id=lead,
                reserve_asset_id=reserve,
                active_asset_id=lead,
                retired_asset_ids=retired,
                resource_budget_units=10,
                resource_spent_units=2,
                evidence_checksums=evidence_checksums(early_keys),
            )
            lead_keys = (*early_keys, (lead, "lead_proof_of_concept_review"))
            lead_poc = evolve_state(
                joint,
                checkpoint_id="lead_proof_of_concept_review",
                resource_spent_units=4,
                evidence_checksums=evidence_checksums(lead_keys),
            )
            early_promoted_keys = (
                *early_keys,
                (reserve, "promoted_reserve_proof_of_concept_review"),
            )
            early_promoted = evolve_state(
                joint,
                checkpoint_id="promoted_reserve_proof_of_concept_review",
                active_asset_id=reserve,
                retired_asset_ids=tuple(sorted((*retired, lead))),
                resource_spent_units=4,
                switch_count=1,
                evidence_checksums=evidence_checksums(early_promoted_keys),
            )
            late_promoted_keys = (
                *lead_keys,
                (reserve, "promoted_reserve_proof_of_concept_review"),
            )
            late_promoted = evolve_state(
                early_promoted,
                resource_spent_units=6,
                evidence_checksums=evidence_checksums(late_promoted_keys),
            )
            lead_confirmation = evolve_state(
                lead_poc,
                checkpoint_id="confirmation",
                retired_asset_ids=tuple(sorted((*retired, reserve))),
                resource_spent_units=8,
                evidence_checksums=evidence_checksums(
                    (*lead_keys, (lead, "confirmation"))
                ),
            )
            early_reserve_confirmation = evolve_state(
                early_promoted,
                checkpoint_id="confirmation",
                resource_spent_units=8,
                evidence_checksums=evidence_checksums(
                    (*early_promoted_keys, (reserve, "confirmation"))
                ),
            )
            late_reserve_confirmation = evolve_state(
                late_promoted,
                checkpoint_id="confirmation",
                resource_spent_units=10,
                evidence_checksums=evidence_checksums(
                    (*late_promoted_keys, (reserve, "confirmation"))
                ),
            )
            branch_states = (
                joint,
                lead_poc,
                early_promoted,
                late_promoted,
                lead_confirmation,
                early_reserve_confirmation,
                late_reserve_confirmation,
            )
            states.extend(branch_states)
            initial_actions.append(
                add_action(
                    source=initial,
                    variant_id=f"select-{lead}-{reserve}",
                    action_id="select_lead_and_reserve",
                    target=joint,
                    target_asset=lead,
                    reserve_asset=reserve,
                )
            )
            add_action(
                source=joint,
                variant_id=f"{lead}-{reserve}-lead-poc",
                action_id="advance_lead_to_proof_of_concept",
                target=lead_poc,
            )
            add_action(
                source=joint,
                variant_id=f"{lead}-{reserve}-early-promote",
                action_id="promote_reserve_to_proof_of_concept",
                target=early_promoted,
            )
            add_action(
                source=joint,
                variant_id=f"{lead}-{reserve}-stop-early",
                action_id="terminate_portfolio",
                terminal="stopped",
            )
            add_action(
                source=lead_poc,
                variant_id=f"{lead}-{reserve}-lead-confirm",
                action_id="advance_active_to_confirmation",
                target=lead_confirmation,
            )
            add_action(
                source=lead_poc,
                variant_id=f"{lead}-{reserve}-late-promote",
                action_id="promote_reserve_to_proof_of_concept",
                target=late_promoted,
            )
            add_action(
                source=lead_poc,
                variant_id=f"{lead}-{reserve}-stop-lead",
                action_id="terminate_portfolio",
                terminal="stopped",
            )
            for label, promoted, confirmation in (
                ("early", early_promoted, early_reserve_confirmation),
                ("late", late_promoted, late_reserve_confirmation),
            ):
                add_action(
                    source=promoted,
                    variant_id=f"{lead}-{reserve}-{label}-reserve-confirm",
                    action_id="advance_active_to_confirmation",
                    target=confirmation,
                )
                add_action(
                    source=promoted,
                    variant_id=f"{lead}-{reserve}-{label}-reserve-stop",
                    action_id="terminate_portfolio",
                    terminal="stopped",
                )
            for confirmation in (
                lead_confirmation,
                early_reserve_confirmation,
                late_reserve_confirmation,
            ):
                add_terminal_actions(confirmation)
            for state in branch_states:
                state_actions = tuple(
                    action
                    for action in actions
                    if action.state_checksum == cast(str, state.checksum)
                )
                supported_sets.append(
                    TrialDevCensusSupportedSetV1(
                        state_checksum=cast(str, state.checksum),
                        analysis_method_id="prespecified_method_v1",
                        supported_action_variant_checksums=tuple(
                            cast(str, action.checksum) for action in state_actions
                        ),
                        evidence_checksums=state.evidence_checksums,
                    )
                )
    initial_actions.append(
        add_action(
            source=initial,
            variant_id="withhold",
            action_id="withhold_selection",
            terminal="withheld",
        )
    )
    supported_sets.append(
        TrialDevCensusSupportedSetV1(
            state_checksum=cast(str, initial.checksum),
            analysis_method_id="prespecified_method_v1",
            supported_action_variant_checksums=tuple(
                cast(str, action.checksum) for action in initial_actions
            ),
            evidence_checksums=initial.evidence_checksums,
        )
    )
    first_evidence = evidence_by_key[("A", "observational_review")]
    census = TrialDevProgrammeCensusV1(
        resource_schedule=TrialDevCensusResourceScheduleV1(
            early_study_units=1,
            proof_of_concept_units=2,
            confirmation_units=4,
            maximum_switches=1,
        ),
        evidence=tuple(evidence),
        states=tuple(states),
        actions=tuple(actions),
        transitions=tuple(transitions),
        supported_sets=tuple(supported_sets),
        numerical_witnesses=(
            TrialDevNumericalWitnessV1(
                evidence_checksum=cast(str, first_evidence.checksum),
                value_column="event",
                statistic="proportion",
                reported_value=2 / 3,
                absolute_tolerance=1e-12,
            ),
        ),
    )
    census_path = tmp_path / "portfolio_census.json"
    census_path.write_text(census.model_dump_json(indent=2), encoding="utf-8")
    return census_path, census


def test_independent_census_reconstructs_every_route_and_numeric_witness(
    tmp_path: Path,
) -> None:
    census_path, _ = _build_single_asset_census(tmp_path)

    report = audit_trialdev_programme_census(
        census_path=census_path, release_root=tmp_path
    )

    assert report.status == "pass"
    assert report.state_count == 4
    assert report.action_variant_count == report.transition_count == 9
    assert report.terminal_transition_count == 6
    assert report.numerical_witness_count == 1


def test_independent_census_reconstructs_complete_three_asset_portfolio(
    tmp_path: Path,
) -> None:
    census_path, census = _build_portfolio_census(tmp_path)

    report = audit_trialdev_programme_census(
        census_path=census_path, release_root=tmp_path
    )

    assert report.status == "pass", report.findings
    assert report.state_count == 43
    assert report.action_variant_count == report.transition_count == 121
    assert report.supported_set_count == len(census.states)


def test_programme_census_cli_writes_machine_readable_report(tmp_path: Path) -> None:
    census_path, _ = _build_single_asset_census(tmp_path)
    output = tmp_path / "report.json"

    status = cli.main(
        (
            "trialdev-programme-census",
            "--census",
            str(census_path),
            "--release-root",
            str(tmp_path),
            "--output",
            str(output),
        )
    )

    assert status == 0
    assert '"status": "pass"' in output.read_text(encoding="utf-8")


def test_independent_census_rejects_missing_duplicate_and_unreachable_routes(
    tmp_path: Path,
) -> None:
    _, census = _build_single_asset_census(tmp_path)

    missing_path = _write_mutation(
        tmp_path, census, transitions=census.transitions[:-1]
    )
    missing = audit_trialdev_programme_census(
        census_path=missing_path, release_root=tmp_path
    )
    assert missing.status == "fail"
    assert any(item.startswith("transition_cardinality:") for item in missing.findings)

    duplicate_path = _write_mutation(
        tmp_path, census, actions=(*census.actions, census.actions[0])
    )
    duplicate = audit_trialdev_programme_census(
        census_path=duplicate_path, release_root=tmp_path
    )
    assert duplicate.status == "fail"
    assert "duplicate_action_variant" in duplicate.findings

    unreachable_state = census.states[-1].model_copy(
        update={
            "checksum": None,
            "programme_id": "orphan",
            "checkpoint_id": "confirmation",
        }
    )
    unreachable_path = _write_mutation(
        tmp_path, census, states=(*census.states, unreachable_state)
    )
    unreachable = audit_trialdev_programme_census(
        census_path=unreachable_path,
        release_root=tmp_path,
    )
    assert unreachable.status == "fail"
    assert any(
        item.startswith("programme_initial_state_cardinality:orphan:0")
        for item in unreachable.findings
    )


def test_independent_census_rejects_counterfactual_evidence_and_numeric_drift(
    tmp_path: Path,
) -> None:
    _, census = _build_single_asset_census(tmp_path)
    transition = census.transitions[0]
    future = cast(str, census.evidence[-1].checksum)
    leaking_transition = TrialDevCensusTransitionV1.model_validate(
        transition.model_copy(
            update={"checksum": None, "newly_exposed_evidence_checksums": (future,)}
        ).model_dump(mode="json")
    )
    transitions = (leaking_transition, *census.transitions[1:])
    leakage_path = _write_mutation(tmp_path, census, transitions=transitions)

    leakage = audit_trialdev_programme_census(
        census_path=leakage_path, release_root=tmp_path
    )

    assert leakage.status == "fail"
    assert any(
        "counterfactual_evidence" in item or "evidence_mismatch" in item
        for item in leakage.findings
    )

    drifted_witness = TrialDevNumericalWitnessV1.model_validate(
        census.numerical_witnesses[0]
        .model_copy(update={"checksum": None, "reported_value": 0.1})
        .model_dump(mode="json")
    )
    drift_path = _write_mutation(
        tmp_path, census, numerical_witnesses=(drifted_witness,)
    )
    drift = audit_trialdev_programme_census(
        census_path=drift_path, release_root=tmp_path
    )
    assert drift.status == "fail"
    assert any(
        item.startswith("numerical_witness_disagreement:") for item in drift.findings
    )
