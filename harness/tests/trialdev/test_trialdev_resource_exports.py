"""Tests for canonical TrialDev phase and programme resource exports."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.core.manifest import AggregateManifestV1
from trialagentbench_harness.contracts.core.runs import (
    TrialDevChainSummaryV1,
    TrialDevMaterializationUsageV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_grades import (
    TrialDevTerminalSummaryV1,
    TrialDevTrajectoryGradeV1,
)
from trialagentbench_harness.io import write_json_model
from trialagentbench_harness.policies import AggregatePolicy
from trialagentbench_harness.trialdev.aggregate import _collect_resource_rows
from trialagentbench_harness.trialdev.grading.design_frontier import (
    derive_programme_resource_consequence_v1,
)
from trialagentbench_harness.trialdev.grading.models import (
    TrialDevDesignFrontierPointV1,
    TrialDevPhaseResourceConsequenceV1,
)


def _valid_inadequate_phase_payload() -> dict[str, object]:
    return {
        "phase_id": "phase2",
        "request_checksum": "r" * 64,
        "target_sample_size": 100,
        "follow_up_days": 90,
        "enrollment_window_days": 42,
        "site_count_budget": 8,
        "participant_follow_up_days": 9000,
        "statistically_adequate": False,
        "operationally_feasible": True,
        "design_status": "statistically_inadequate",
        "operational_support": 100,
        "operational_headroom": 0,
        "operational_shortage": 0,
        "achieved_power": 0.70,
        "target_power": 0.80,
        "achieved_safety_absolute_risk_power": 0.90,
        "achieved_safety_excess_risk_power": 0.90,
        "target_safety_decision_power": 0.80,
        "participant_excess_vs_minimum": 0,
        "participant_shortage_vs_minimum": 20,
        "follow_up_excess_days_vs_minimum": 0,
        "follow_up_shortage_days_vs_minimum": 0,
        "avoidable_participants_min": 0,
        "avoidable_participants_max": 0,
        "avoidable_follow_up_days_min": 0,
        "avoidable_follow_up_days_max": 0,
        "avoidable_participant_follow_up_days_min": 0,
        "avoidable_participant_follow_up_days_max": 0,
        "entered_after_unsupported_advance": False,
    }


def test_resource_exports_preserve_dominance_and_late_continuation(tmp_path: Path) -> None:
    """Export exact vectors without converting missing cost to zero."""

    run_root = tmp_path / "run"
    program_dir = run_root / "programs" / "program-1"
    program_dir.mkdir(parents=True)
    frontier = TrialDevDesignFrontierPointV1(
        target_sample_size=80,
        follow_up_days=60,
        allocation_ratio="1:1",
        achieved_power=0.8,
        achieved_safety_absolute_risk_power=0.9,
        achieved_safety_excess_risk_power=0.9,
    )
    phase = TrialDevPhaseResourceConsequenceV1(
        phase_id="phase2",
        request_checksum="r" * 64,
        target_sample_size=100,
        follow_up_days=90,
        enrollment_window_days=42,
        site_count_budget=8,
        participant_follow_up_days=9000,
        statistically_adequate=True,
        operationally_feasible=True,
        design_status="valid_dominated",
        operational_support=100,
        operational_headroom=0,
        operational_shortage=0,
        achieved_power=0.9,
        target_power=0.8,
        achieved_safety_absolute_risk_power=0.95,
        achieved_safety_excess_risk_power=0.95,
        target_safety_decision_power=0.8,
        participant_excess_vs_minimum=20,
        participant_shortage_vs_minimum=0,
        follow_up_excess_days_vs_minimum=30,
        follow_up_shortage_days_vs_minimum=0,
        dominating_frontier=(frontier,),
        avoidable_participants_min=20,
        avoidable_participants_max=20,
        avoidable_follow_up_days_min=30,
        avoidable_follow_up_days_max=30,
        avoidable_participant_follow_up_days_min=4200,
        avoidable_participant_follow_up_days_max=4200,
        entered_after_unsupported_advance=True,
    )
    resources = derive_programme_resource_consequence_v1((phase,))
    write_json_model(
        program_dir / "chain_summary.json",
        TrialDevChainSummaryV1(
            program_id="program-1",
            scenario_id="scenario-1",
            objective_id="benefit_risk",
            materialization_usage=TrialDevMaterializationUsageV1(),
            execution_status="completed",
        ),
    )
    write_json_model(
        program_dir / "trajectory_grade.json",
        TrialDevTrajectoryGradeV1(
            trajectory_primary_score=1.0,
            trajectory_decision_score=1.0,
            terminal_summary=TrialDevTerminalSummaryV1(
                scenario_id="scenario-1",
                terminal_status="completed",
                terminal_action="declare_success",
                final_program_success=True,
            ),
            resource_consequence=resources,
            payload={},
        ),
    )
    manifest = AggregateManifestV1(
        harness_version="test",
        timestamp_utc=datetime.now(UTC),
        input_run_dir=str(run_root),
        policy_strict=True,
    )

    phase_rows, programme_rows = _collect_resource_rows(
        run_root,
        policy=AggregatePolicy(),
        manifest=manifest,
    )

    assert len(phase_rows) == len(programme_rows) == 1
    assert phase_rows[0].design_status == "valid_dominated"
    assert phase_rows[0].target_power == 0.8
    assert phase_rows[0].enrollment_window_days == 42
    assert phase_rows[0].site_count_budget == 8
    assert phase_rows[0].target_safety_decision_power == 0.8
    assert phase_rows[0].participant_excess_vs_minimum == 20
    assert phase_rows[0].participant_shortage_vs_minimum == 0
    assert phase_rows[0].follow_up_excess_days_vs_minimum == 30
    assert phase_rows[0].follow_up_shortage_days_vs_minimum == 0
    assert phase_rows[0].dominating_frontier_count == 1
    assert programme_rows[0].late_continuation_participants == 100
    assert programme_rows[0].total_enrollment_window_days == 42
    assert programme_rows[0].total_site_phase_budget == 8
    assert programme_rows[0].total_planned_phase_duration_days == 132
    assert programme_rows[0].late_continuation_enrollment_window_days == 42
    assert programme_rows[0].late_continuation_site_phase_budget == 8
    assert programme_rows[0].participant_excess_vs_minimum == 20
    assert programme_rows[0].participant_shortage_vs_minimum == 0
    assert programme_rows[0].follow_up_excess_days_vs_minimum == 30
    assert programme_rows[0].follow_up_shortage_days_vs_minimum == 0
    assert programme_rows[0].cost_status == "not_available_without_public_cost_schedule"


def test_phase_resource_rejects_unpaired_efficacy_power() -> None:
    """Achieved efficacy power cannot be interpreted without its public target."""

    payload = _valid_inadequate_phase_payload()
    payload["target_power"] = None
    with pytest.raises(ValidationError, match="present together"):
        TrialDevPhaseResourceConsequenceV1.model_validate(payload)


def test_phase_resource_rejects_simultaneous_excess_and_shortage() -> None:
    """A phase cannot be both above and below the same public frontier minimum."""

    payload = _valid_inadequate_phase_payload()
    payload["participant_excess_vs_minimum"] = 10
    with pytest.raises(ValidationError, match="excess and shortage"):
        TrialDevPhaseResourceConsequenceV1.model_validate(payload)


def test_phase_resource_requires_operational_status_for_public_support_shortage() -> None:
    """Statistical adequacy cannot conceal an operational recruitment shortfall."""

    payload = _valid_inadequate_phase_payload()
    payload.update(
        {
            "design_status": "valid_nondominated",
            "statistically_adequate": True,
            "operationally_feasible": False,
            "operational_support": 80,
            "operational_shortage": 20,
            "achieved_power": 0.85,
            "participant_shortage_vs_minimum": 0,
        }
    )
    with pytest.raises(ValidationError, match="Operationally infeasible status"):
        TrialDevPhaseResourceConsequenceV1.model_validate(payload)

    payload["design_status"] = "operationally_infeasible"
    phase = TrialDevPhaseResourceConsequenceV1.model_validate(payload)
    programme = derive_programme_resource_consequence_v1((phase,))
    assert programme.statistically_inadequate_phases == 0
    assert programme.operationally_infeasible_phases == 1


def test_programme_preserves_joint_statistical_and_operational_failure() -> None:
    """A single phase may fail both independently meaningful design criteria."""

    payload = _valid_inadequate_phase_payload()
    payload.update(
        {
            "operationally_feasible": False,
            "operational_support": 80,
            "operational_shortage": 20,
        }
    )
    phase = TrialDevPhaseResourceConsequenceV1.model_validate(payload)
    programme = derive_programme_resource_consequence_v1((phase,))
    assert phase.design_status == "statistically_inadequate"
    assert programme.statistically_inadequate_phases == 1
    assert programme.operationally_infeasible_phases == 1


def test_programme_resource_rejects_duration_or_site_totals_that_do_not_replay() -> None:
    """Programme planning resources must replay exactly from entered phases."""

    phase = TrialDevPhaseResourceConsequenceV1.model_validate(_valid_inadequate_phase_payload())
    resources = derive_programme_resource_consequence_v1((phase,))
    payload = resources.model_dump(mode="python")
    payload["total_planned_phase_duration_days"] += 1
    payload["total_site_phase_budget"] += 1

    with pytest.raises(ValidationError, match="totals do not replay"):
        type(resources).model_validate(payload)
