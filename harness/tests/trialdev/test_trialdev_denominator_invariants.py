from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from trialagentbench_harness.contracts.core.coverage import (
    TrialDevCoverageCountsV1,
    TrialDevCoverageProgramV1,
    TrialDevCoverageReportV1,
)
from trialagentbench_harness.contracts.core.manifest import AggregateManifestV1
from trialagentbench_harness.contracts.core.runs import (
    TrialDevChainSummaryV1,
    TrialDevCheckpointOutcomeV1,
    TrialDevMaterializationUsageV1,
    TrialDevTrajectoryMetricsV1,
)
from trialagentbench_harness.io import write_json_model
from trialagentbench_harness.policies import AggregatePolicy
from trialagentbench_harness.trialdev.aggregate import _compute_completion_metrics


def test_trialdev_completion_metrics_use_declared_denominator(tmp_path: Path) -> None:
    output_root = tmp_path / "run"
    programs_root = output_root / "programs"
    programs_root.mkdir(parents=True, exist_ok=True)

    # Declared set: two programs, but only one directory exists.
    cov = TrialDevCoverageReportV1(
        schema_id="trialagentbench_trialdev_coverage_report_v1",
        schema_version=1,
        counts=TrialDevCoverageCountsV1(total_items_present=0),
        items=[],
        n_programs=2,
        programs=[
            TrialDevCoverageProgramV1(program_id="s01__benefit_risk", scenario_id="s01", objective_id="benefit_risk"),
            TrialDevCoverageProgramV1(program_id="s02__benefit_risk", scenario_id="s02", objective_id="benefit_risk"),
        ],
    )
    write_json_model(output_root / "coverage_report.json", cov)

    # Present program dir with schema-valid chain summary but no usable grade.
    p1 = programs_root / "s01__benefit_risk"
    p1.mkdir(parents=True, exist_ok=True)
    chain = TrialDevChainSummaryV1(
        program_id="s01__benefit_risk",
        scenario_id="s01",
        objective_id="benefit_risk",
        stopped_at_phase=None,
        started_at_utc=None,
        ended_at_utc=None,
        wall_seconds_total=None,
        phases_attempted=[],
        obs_review_path_stats={},
        materialization_usage=TrialDevMaterializationUsageV1(materialize_calls_by_phase={}),
        trajectory_metrics=TrialDevTrajectoryMetricsV1(
            trajectory_primary_score=None,
            programme_primary_score=0.0,
            checkpoint_outcomes=(
                TrialDevCheckpointOutcomeV1(
                    phase_id="observational_review",
                    status="missing_or_invalid",
                    required_lane_ids=("asset_nomination", "phase_analysis"),
                    conditional_score=0.0,
                    cumulative_score=0.0,
                ),
                TrialDevCheckpointOutcomeV1(
                    phase_id="phase1",
                    status="not_reached_after_invalid",
                    required_lane_ids=("phase_design", "phase_analysis", "safety_gate", "decision_action"),
                ),
                TrialDevCheckpointOutcomeV1(
                    phase_id="phase2",
                    status="not_reached_after_invalid",
                    required_lane_ids=("phase_design", "phase_analysis", "decision_action"),
                ),
                TrialDevCheckpointOutcomeV1(
                    phase_id="phase3",
                    status="not_reached_after_invalid",
                    required_lane_ids=("phase_design", "phase_analysis", "decision_action"),
                ),
                TrialDevCheckpointOutcomeV1(
                    phase_id="final_decision",
                    status="missing_or_invalid",
                    required_lane_ids=("route_timing", "final_recommendation"),
                    conditional_score=0.0,
                    cumulative_score=0.0,
                ),
            ),
        ),
        execution_status="model_turn_limit",
        error="AgentTurnLimitExceeded: no submission within 60 turns",
        violations_n=0,
        violations=[],
    )
    write_json_model(p1 / "chain_summary.json", chain)

    policy = AggregatePolicy(strict=True, allow_incomplete_artifacts=False)
    manifest = AggregateManifestV1(
        harness_version="test",
        timestamp_utc=datetime.now(UTC),
        input_run_dir=str(output_root),
        bundle_dir=None,
        policy_strict=True,
        allow_incomplete_artifacts=False,
    )

    cm = _compute_completion_metrics(output_root, policy=policy, manifest=manifest)
    assert cm.n_declared == 2
    assert cm.n_present == 1
    assert cm.n_completed == 0
    assert cm.failure_imputed_mean == 0.0
    assert len(cm.per_program) == 2
    statuses = {r.program_id: r.program_status for r in cm.per_program}
    assert statuses["s01__benefit_risk"] == "model_noncompletion"
    assert statuses["s02__benefit_risk"] == "missing_program_dir"


def test_trialdev_completion_metrics_distinguish_invalid_model_submissions(tmp_path: Path) -> None:
    output_root = tmp_path / "run"
    program_id = "s01__benefit_risk"
    write_json_model(
        output_root / "coverage_report.json",
        TrialDevCoverageReportV1(
            schema_id="trialagentbench_trialdev_coverage_report_v1",
            schema_version=1,
            counts=TrialDevCoverageCountsV1(total_items_present=0),
            items=[],
            n_programs=1,
            programs=[
                TrialDevCoverageProgramV1(
                    program_id=program_id,
                    scenario_id="s01",
                    objective_id="benefit_risk",
                )
            ],
        ),
    )
    write_json_model(
        output_root / "programs" / program_id / "chain_summary.json",
        TrialDevChainSummaryV1(
            program_id=program_id,
            scenario_id="s01",
            objective_id="benefit_risk",
            materialization_usage=TrialDevMaterializationUsageV1(),
            execution_status="model_invalid_submission",
            error="TrialMaterializationRejectedError: correction budget exhausted",
        ),
    )
    policy = AggregatePolicy(strict=True, allow_incomplete_artifacts=False)
    manifest = AggregateManifestV1(
        harness_version="test",
        timestamp_utc=datetime.now(UTC),
        input_run_dir=str(output_root),
        bundle_dir=None,
        policy_strict=True,
        allow_incomplete_artifacts=False,
    )

    metrics = _compute_completion_metrics(output_root, policy=policy, manifest=manifest)

    assert metrics.n_declared == 1
    assert metrics.n_completed == 0
    assert metrics.failure_imputed_mean == 0.0
    assert metrics.per_program[0].program_status == "model_invalid_submission"
