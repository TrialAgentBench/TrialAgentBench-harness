"""Tests for TrialDev decision-surface contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.trialdev.trialdev_decision_surface import (
    TrialDevDecisionSurfaceManifestV1,
    TrialDevDecisionSurfaceRecordV1,
    TrialDevDiagnosticReferenceRouteRecordV1,
    TrialDevDiagnosticReferenceRouteStepV1,
    TrialDevUtilitySensitivityProfileV1,
)
from trialagentbench_harness.trialdev.route import trialdev_route_sort_key


def test_decision_surface_rejects_overlapping_targets() -> None:
    """Official and acceptable action sets must be disjoint."""

    with pytest.raises(ValidationError, match="overlap"):
        TrialDevDecisionSurfaceRecordV1(
            scenario_id="no_progression",
            scenario_key="s01",
            program_id="s01__benefit_risk",
            objective_id="benefit_risk",
            phase_id="phase2",
            lane_id="decision_action",
            lane_kind="action",
            target_semantics="stop_advance",
            decision_context_id="phase2/benefit_risk/decision_action",
            diagnostic_reference_target_ids=("advance_to_confirmation",),
            credit_eligible_target_ids=("advance_to_confirmation",),
            diagnostic_reference_asset_id="regimen_a",
            diagnostic_reference_action_id="advance_to_confirmation",
            diagnostic_reference_route_id="oracle:s01__benefit_risk",
            margin_kind="score",
            selected_route_margin=0.2,
            utility_payload={"reference_utility": 0.8},
            public_evidence_basis=("public/eval_contract.json",),
            evaluator_evidence_basis=("grader/evaluation_target_register.jsonl",),
        )


def test_decision_surface_rejects_hidden_public_evidence() -> None:
    """Public evidence basis cannot cite hidden or grader-only files."""

    with pytest.raises(ValidationError, match="public evidence basis"):
        TrialDevDecisionSurfaceRecordV1(
            scenario_id="no_progression",
            scenario_key="s01",
            program_id="s01__benefit_risk",
            objective_id="benefit_risk",
            phase_id="phase2",
            lane_id="decision_action",
            lane_kind="action",
            target_semantics="stop_advance",
            decision_context_id="phase2/benefit_risk/decision_action",
            diagnostic_reference_target_ids=("advance_to_confirmation",),
            diagnostic_reference_action_id="advance_to_confirmation",
            diagnostic_reference_route_id="oracle:s01__benefit_risk",
            margin_kind="score",
            selected_route_margin=0.2,
            utility_payload={"reference_utility": 0.8},
            public_evidence_basis=("hidden/world_manifest.json",),
            evaluator_evidence_basis=("grader/evaluation_target_register.jsonl",),
        )


def test_decision_surface_rejects_action_on_non_action_lane() -> None:
    """Non-action lanes cannot declare diagnostic_reference_action_id."""

    with pytest.raises(ValidationError, match="non-action lane"):
        TrialDevDecisionSurfaceRecordV1(
            scenario_id="no_progression",
            scenario_key="s01",
            program_id="s01__benefit_risk",
            objective_id="benefit_risk",
            phase_id="phase2",
            lane_id="phase_analysis",
            lane_kind="diagnostic",
            target_semantics="analysis_consistency",
            decision_context_id="phase2/benefit_risk/benefit_risk/phase_analysis",
            diagnostic_reference_target_ids=("phase_analysis_consistent",),
            diagnostic_reference_action_id="phase_analysis_consistent",
            diagnostic_reference_route_id="oracle:s01__benefit_risk",
            utility_payload={},
            public_evidence_basis=("public/eval_contract.json",),
            evaluator_evidence_basis=("grader/evaluation_target_register.jsonl",),
        )


def test_runtime_decision_context_has_no_diagnostic_reference_action_or_margin() -> None:
    """Realized public-evidence targets describe a runtime context, not a static oracle action."""

    row = TrialDevDecisionSurfaceRecordV1(
        scenario_id="s01",
        scenario_key="s01",
        program_id="s01__benefit_risk",
        objective_id="benefit_risk",
        phase_id="phase2",
        lane_id="decision_action",
        lane_kind="action",
        target_semantics="stop_advance",
        decision_context_id="phase2/benefit_risk/benefit_risk/decision_action",
        diagnostic_reference_target_ids=("derived_from_realized_public_evidence",),
        diagnostic_reference_route_id="reference:s01__benefit_risk",
        target_resolution="realized_public_evidence",
        scoring_role="runtime_context",
        public_evidence_basis=("public/phase_decision_evidence_policy.json",),
        evaluator_evidence_basis=("grader/evaluation_target_register.jsonl",),
    )

    assert row.diagnostic_reference_action_id is None
    assert row.margin_kind == "not_applicable"


def test_runtime_decision_context_rejects_static_diagnostic_reference_action() -> None:
    """A runtime-resolved context cannot smuggle in a construction-time action label."""

    with pytest.raises(ValidationError, match="cannot declare a diagnostic_reference_action_id"):
        TrialDevDecisionSurfaceRecordV1(
            scenario_id="s01",
            scenario_key="s01",
            program_id="s01__benefit_risk",
            objective_id="benefit_risk",
            phase_id="phase2",
            lane_id="decision_action",
            lane_kind="action",
            target_semantics="stop_advance",
            decision_context_id="phase2/benefit_risk/benefit_risk/decision_action",
            diagnostic_reference_target_ids=("derived_from_realized_public_evidence",),
            diagnostic_reference_action_id="advance_to_confirmation",
            diagnostic_reference_route_id="reference:s01__benefit_risk",
            target_resolution="realized_public_evidence",
            scoring_role="runtime_context",
            public_evidence_basis=("public/phase_decision_evidence_policy.json",),
            evaluator_evidence_basis=("grader/evaluation_target_register.jsonl",),
        )


def test_runtime_decision_context_rejects_static_target_set() -> None:
    """A runtime-resolved context cannot retain a construction-time action target."""

    with pytest.raises(ValidationError, match="sole declared derivation target"):
        TrialDevDecisionSurfaceRecordV1(
            scenario_id="s01",
            scenario_key="s01",
            program_id="s01__benefit_risk",
            objective_id="benefit_risk",
            phase_id="phase2",
            lane_id="decision_action",
            lane_kind="action",
            target_semantics="stop_advance",
            decision_context_id="phase2/benefit_risk/benefit_risk/decision_action",
            diagnostic_reference_target_ids=("advance_to_confirmation",),
            diagnostic_reference_route_id="reference:s01__benefit_risk",
            target_resolution="realized_public_evidence",
            scoring_role="runtime_context",
            public_evidence_basis=("public/phase_decision_evidence_policy.json",),
            evaluator_evidence_basis=("public/phase_decision_evidence_policy.json",),
        )


def test_non_nomination_target_has_no_fabricated_asset_or_margin() -> None:
    """A qualified non-nomination retains its action without inventing an asset."""

    row = TrialDevDecisionSurfaceRecordV1(
        scenario_id="s01",
        scenario_key="s01",
        program_id="s01__benefit_risk",
        objective_id="benefit_risk",
        phase_id="observational_review",
        lane_id="asset_nomination",
        lane_kind="action",
        target_semantics="asset",
        decision_context_id="observational_review/benefit_risk/benefit_risk/asset_nomination",
        diagnostic_reference_target_ids=("withhold_nomination",),
        diagnostic_reference_action_id="withhold_nomination",
        diagnostic_reference_route_id="reference:s01__benefit_risk",
        margin_kind="not_applicable",
        public_evidence_basis=("public/observational_extract.parquet",),
        evaluator_evidence_basis=("grader/drug_ranking_reference_manifest.json",),
    )

    assert row.diagnostic_reference_asset_id is None
    assert row.diagnostic_reference_action_id == "withhold_nomination"
    assert row.selected_route_margin is None


def test_non_nomination_target_cannot_identify_an_asset() -> None:
    """A no-nomination action cannot simultaneously identify a candidate."""

    with pytest.raises(ValidationError, match="cannot identify an asset"):
        TrialDevDecisionSurfaceRecordV1(
            scenario_id="s01",
            scenario_key="s01",
            program_id="s01__benefit_risk",
            objective_id="benefit_risk",
            phase_id="observational_review",
            lane_id="asset_nomination",
            lane_kind="action",
            target_semantics="asset",
            decision_context_id="observational_review/benefit_risk/benefit_risk/asset_nomination",
            diagnostic_reference_target_ids=("withhold_nomination",),
            diagnostic_reference_asset_id="drug_a",
            diagnostic_reference_action_id="withhold_nomination",
            diagnostic_reference_route_id="reference:s01__benefit_risk",
            public_evidence_basis=("public/phase_decision_evidence_policy.json",),
            evaluator_evidence_basis=("grader/evaluation_target_register.jsonl",),
        )


def test_route_sort_key_places_final_decision_last() -> None:
    """Route ordering is declared, not lexicographic."""

    assert trialdev_route_sort_key("phase3", "decision_action") < trialdev_route_sort_key(
        "final_decision",
        "final_recommendation",
    )
    with pytest.raises(ValueError, match="Unknown TrialDev phase_id"):
        trialdev_route_sort_key("phase10", "decision_action")


def test_diagnostic_reference_route_and_sensitivity_profile_assign_checksums() -> None:
    """Decision-surface artifacts are stable, checksummed contracts."""

    route = TrialDevDiagnosticReferenceRouteRecordV1(
        scenario_id="no_progression",
        scenario_key="s01",
        program_id="s01__benefit_risk",
        objective_id="benefit_risk",
        route_steps=(
            TrialDevDiagnosticReferenceRouteStepV1(
                phase_id="observational_review",
                lane_id="asset_nomination",
                action_id="nominate_asset",
                asset_id="regimen_a",
                utility=0.9,
                regret=0.0,
                margin_to_next_best=0.2,
            ),
            TrialDevDiagnosticReferenceRouteStepV1(
                phase_id="final_decision",
                lane_id="final_recommendation",
                action_id="declare_success",
                utility=0.9,
                regret=0.0,
                margin_to_next_best=1.0,
            ),
        ),
        terminal_action_id="declare_success",
        terminal_recommendation_target_id="declare_success",
        terminal_asset_id="regimen_a",
        total_utility=0.9,
        regret_tolerance=0.05,
        public_evidence_basis=("public/eval_contract.json",),
        evaluator_evidence_basis=("grader/evaluation_target_register.jsonl",),
    )
    profile = TrialDevUtilitySensitivityProfileV1(
        profile_id="safety_heavy",
        description="Safety-heavy diagnostic sensitivity profile.",
        objective_id="benefit_risk",
        weights={"efficacy": 0.4, "safety": 0.6},
    )
    manifest = TrialDevDecisionSurfaceManifestV1(
        release_id="unit",
        scenario_count=1,
        decision_surface_record_count=1,
        diagnostic_reference_route_record_count=1,
        sensitivity_profile_count=1,
    )

    assert len(str(route.checksum)) == 64
    assert len(str(profile.checksum)) == 64
    assert len(str(manifest.checksum)) == 64


def test_diagnostic_reference_route_rejects_final_decision_first() -> None:
    """Oracle routes must follow the declared TrialDev order."""

    with pytest.raises(ValidationError, match="must follow declared TrialDev"):
        TrialDevDiagnosticReferenceRouteRecordV1(
            scenario_id="no_progression",
            scenario_key="s01",
            program_id="s01__benefit_risk",
            objective_id="benefit_risk",
            route_steps=(
                TrialDevDiagnosticReferenceRouteStepV1(
                    phase_id="final_decision",
                    lane_id="final_recommendation",
                    action_id="declare_success",
                ),
                TrialDevDiagnosticReferenceRouteStepV1(
                    phase_id="phase3",
                    lane_id="decision_action",
                    action_id="declare_success",
                ),
            ),
            terminal_action_id="declare_success",
            total_utility=1.0,
            regret_tolerance=0.0,
            public_evidence_basis=("public/eval_contract.json",),
            evaluator_evidence_basis=("grader/evaluation_target_register.jsonl",),
        )


def test_utility_sensitivity_rejects_zero_total_weight() -> None:
    """Sensitivity profiles must have positive total weight."""

    with pytest.raises(ValidationError, match="positive total"):
        TrialDevUtilitySensitivityProfileV1(
            profile_id="invalid",
            description="Invalid profile.",
            objective_id="benefit_risk",
            weights={"efficacy": 0.0},
        )
