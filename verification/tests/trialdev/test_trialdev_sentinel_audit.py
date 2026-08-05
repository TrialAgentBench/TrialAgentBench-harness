"""Tests for independent TrialDev sentinel closure."""

from __future__ import annotations

import hashlib
import json

from trialagentbench_validation.contracts.trialdev_scientific_inventory import (
    TrialDevScientificConstructionInventoryV1,
    TrialDevScientificConstructionRowV1,
)
from trialagentbench_validation.recovery import (
    RecoverabilityReportV1,
    RecoverabilityRouteV1,
)
from trialagentbench_validation.trialdev.sentinel_audit import audit_trialdev_sentinels


def _scientific_row(
    *,
    scenario_id: str,
    phase_id: str,
    lane_id: str,
    identification_class: str,
    intercurrent_event_bindings: tuple[str, ...] | None = None,
) -> TrialDevScientificConstructionRowV1:
    randomized = phase_id in {"phase2", "phase3"}
    return TrialDevScientificConstructionRowV1.model_validate(
        {
            "scenario_id": scenario_id,
            "generation_seed": 101,
            "phase_id": phase_id,
            "program_objective_id": "balanced_benefit_risk",
            "phase_scoring_objective_id": "balanced_benefit_risk",
            "lane_id": lane_id,
            "development_purpose": (
                "observational_candidate_prioritization_exercise"
                if phase_id == "observational_review"
                else (
                    "randomized_exploratory_evidence_exercise"
                    if phase_id == "phase2"
                    else "randomized_confirmatory_style_evidence_exercise"
                )
            ),
            "allocation_structure": (
                "nonrandomized_comparative_cohort"
                if phase_id == "observational_review"
                else "participant_randomized_with_concurrent_control"
            ),
            "identification_class": identification_class,
            "identification_assumptions": ("declared_assumption",),
            "analysis_method_route_ids": ("method.v1",),
            "design_cell_ids": ("design.v1",) if randomized else (),
            "permitted_intercurrent_event_bindings": (
                intercurrent_event_bindings
                if intercurrent_event_bindings is not None
                else (
                    (
                        "treatment_discontinuation:composite_discontinuation",
                        "treatment_discontinuation:treatment_policy",
                        "treatment_discontinuation:while_on_treatment",
                    )
                    if randomized
                    else ()
                )
            ),
            "competing_event_handling_id": "aalen_johansen_competing_event",
            "treatment_discontinuation_handling_id": (
                "strategy_specific_endpoint_materialization"
                if randomized
                else "not_applicable"
            ),
            "loss_to_follow_up_handling_id": "right_censor_at_last_observation",
            "missing_observation_handling_id": "no_endpoint_imputation",
            "scoring_policy_id": "policy.v1",
            "target_resolution": (
                "submitted_method_public_evidence"
                if phase_id == "observational_review"
                else "realized_public_evidence"
            ),
            "reference_target_ids": ("target",),
            "credit_eligible_target_ids": ("target",),
            "rejected_shortcut_ids": ("shortcut",),
            "recoverability_policy_id": "nomination_required",
            "public_evidence_basis": ("public",),
            "evaluator_evidence_basis": ("policy",),
            "normative_source_ids": ("TAB-SRC-001",),
            "method_source_ids": ("TAB-SRC-011",),
        }
    )


def _route(
    *,
    scenario_id: str,
    estimator_family: str,
    route_id: str,
) -> RecoverabilityRouteV1:
    unit_id = (
        scenario_id
        if estimator_family != "public_randomized_phase_replay"
        else f"{scenario_id}:request"
    )
    return RecoverabilityRouteV1(
        suite="trialdev",
        unit_id=unit_id,
        context_or_checkpoint_id="observational_review",
        route_id=route_id,
        estimator_family=estimator_family,
        effect_scale="phase_specific",
        result_kind="decision",
        comparison_denominator=1,
        maximum_absolute_difference=0.0,
        declared_absolute_tolerance=1e-4,
        difference_to_tolerance_ratio=0.0,
        comparison_rule="numeric_envelope",
        recovery_path="trialdev_public_replay",
        public_input_paths=("public/observational_extract.parquet",),
        expected_summary="expected",
        reproduced_summary="reproduced",
        status="pass",
    )


def _inputs() -> (
    tuple[TrialDevScientificConstructionInventoryV1, RecoverabilityReportV1]
):
    rows = (
        _scientific_row(
            scenario_id="s30",
            phase_id="observational_review",
            lane_id="asset_nomination",
            identification_class="point_identified_under_declared_measured_confounding_assumptions",
        ),
        _scientific_row(
            scenario_id="s30",
            phase_id="phase2",
            lane_id="phase_analysis",
            identification_class="randomized_comparative_risk_under_arm_conditional_independent_censoring",
        ),
        _scientific_row(
            scenario_id="s35",
            phase_id="observational_review",
            lane_id="asset_nomination",
            identification_class="qualified_nonidentification_under_residual_unmeasured_confounding",
        ),
        _scientific_row(
            scenario_id="s40",
            phase_id="phase3",
            lane_id="phase_design",
            identification_class="randomized_comparative_risk_under_arm_conditional_independent_censoring",
            intercurrent_event_bindings=("treatment_discontinuation:treatment_policy",),
        ),
    )
    sorted_rows = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.scenario_id,
                row.phase_id,
                row.program_objective_id,
                row.phase_scoring_objective_id,
                row.lane_id,
            ),
        )
    )
    inventory_payload = {
        "schema_id": "trialagentbench.trialdev.scientific_construction_inventory/v1",
        "release_id": "candidate",
        "suite_manifest_checksum": "a" * 64,
        "source_registry_checksum": "b" * 64,
        "rows": [row.model_dump(mode="json") for row in sorted_rows],
    }
    checksum = hashlib.sha256(
        json.dumps(
            inventory_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    inventory = TrialDevScientificConstructionInventoryV1.model_validate(
        {**inventory_payload, "checksum": checksum}
    )
    routes = (
        _route(
            scenario_id="s30",
            estimator_family="multinomial_propensity_weighted_stratified_aalen_johansen",
            route_id="observational.ipw",
        ),
        _route(
            scenario_id="s35",
            estimator_family="entropy_balanced_standardized_aalen_johansen",
            route_id="observational.sensitivity",
        ),
        _route(
            scenario_id="s30",
            estimator_family="public_randomized_phase_replay",
            route_id="phase2.request",
        ),
        _route(
            scenario_id="s40",
            estimator_family="public_randomized_phase_replay",
            route_id="phase3.request",
        ),
    )
    recoverability = RecoverabilityReportV1(
        suite="trialdev",
        participant_release="participant.zip",
        evaluator_release="evaluator.zip",
        verification_release="verification.zip",
        required_route_count=len(routes),
        replayed_route_count=len(routes),
        failed_route_count=0,
        maximum_absolute_difference=0.0,
        routes=routes,
        status="pass",
    )
    return inventory, recoverability


def test_trialdev_sentinels_close_high_risk_scientific_boundaries() -> None:
    inventory, recoverability = _inputs()

    report = audit_trialdev_sentinels(
        inventory=inventory,
        recoverability=recoverability,
    )

    assert report.status == "pass"
    assert len(report.records) == 4
    assert all(record.status == "pass" for record in report.records)


def test_trialdev_sentinel_fails_when_ltfu_is_bound_as_intercurrent_event() -> None:
    inventory, recoverability = _inputs()
    rows = list(inventory.rows)
    index = next(
        index
        for index, row in enumerate(rows)
        if row.scenario_id == "s30" and row.phase_id == "phase2"
    )
    rows[index] = rows[index].model_copy(
        update={
            "permitted_intercurrent_event_bindings": tuple(
                sorted(
                    (
                        *rows[index].permitted_intercurrent_event_bindings,
                        "loss_to_follow_up:censoring",
                    )
                )
            )
        }
    )
    invalid_inventory = inventory.model_copy(update={"rows": tuple(rows)})

    report = audit_trialdev_sentinels(
        inventory=invalid_inventory,
        recoverability=recoverability,
    )

    assert report.status == "fail"
    phase2 = next(
        record
        for record in report.records
        if record.sentinel_id == "phase2_discontinuation_and_ltfu"
    )
    assert "loss_to_follow_up_misclassified_as_intercurrent_event" in phase2.findings
