"""Tests for TrialDev suite aggregation and rendering."""

from trialagentbench_harness.contracts.core.summaries import (
    TrialDevCompletionMetricsV1,
    TrialDevGroupRollupV1,
    TrialDevResultsPayloadV1,
    TrialDevResultsRollupV1,
    TrialDevResultsSummaryV1,
)
from trialagentbench_harness.trialdev.aggregate import render_summary_md
from trialagentbench_harness.trialdev.scoring import LANE_KEYS


def test_summary_renders_each_rollup_once_after_the_lane_table() -> None:
    """Group tables are suite rollups, not repeated per-lane sections."""

    group = TrialDevGroupRollupV1(n=1, overall_mean=0.5)
    rollup = TrialDevResultsRollupV1(
        n_items=1,
        overall_mean=0.5,
        lane_active_means={lane: 0.5 for lane in LANE_KEYS},
        by_phase={"phase1": group},
        by_scenario={"scenario": group},
        by_objective={"benefit_risk": group},
    )
    summary = TrialDevResultsSummaryV1(
        schema_id="trialagentbench_trialdev_results_summary_v1",
        schema_version=1,
        completion_metrics=TrialDevCompletionMetricsV1(),
        payload=TrialDevResultsPayloadV1(results=rollup),
    )

    rendered = render_summary_md(summary)

    assert rendered.count("Per phase:") == 1
    assert rendered.count("Per scenario:") == 1
    assert rendered.count("Per objective:") == 1
    for lane in LANE_KEYS:
        assert f"| {lane} | 0.500 |" in rendered
