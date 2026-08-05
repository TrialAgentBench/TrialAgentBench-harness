"""Tests for the public cluster-response experiment."""

from __future__ import annotations

import pandas as pd
import pytest

from trialagentbench_validation.characterisation.cluster_response import (
    SETTINGS,
    SOURCE_ANCHORED_SETTING,
    simulate_cluster_response,
    write_cluster_response,
)


def test_cluster_response_is_deterministic_and_monotonic() -> None:
    """Matched worlds expose a graded response including source-anchored intensity."""

    worlds, summary = simulate_cluster_response(world_count=20)
    repeated, repeated_summary = simulate_cluster_response(world_count=20)
    pd.testing.assert_frame_equal(worlds, repeated)
    pd.testing.assert_frame_equal(summary, repeated_summary)

    assert set(worlds["setting"]) == {setting.setting for setting in SETTINGS}
    assert worlds.groupby("setting").size().eq(20).all()
    variance = summary.loc[
        summary["measure"].eq("event_variance_inflation")
    ].sort_values("hazard_ratio_90_to_10")
    assert variance["mean"].is_monotonic_increasing
    source_anchored = variance.loc[variance["setting"].eq("source_anchored")].squeeze()
    assert source_anchored["hazard_ratio_90_to_10"] == pytest.approx(
        SOURCE_ANCHORED_SETTING.hazard_ratio_90_to_10
    )
    assert source_anchored["mean"] > 1.5
    cluster_coverage = summary.loc[
        summary["measure"].eq("cluster_robust_covered")
    ].sort_values("hazard_ratio_90_to_10")
    participant_coverage = summary.loc[
        summary["measure"].eq("participant_independent_covered")
    ].sort_values("hazard_ratio_90_to_10")
    assert cluster_coverage["mean"].between(0.90, 1.0).all()
    assert participant_coverage.iloc[-1]["mean"] < cluster_coverage.iloc[-1]["mean"]


def test_cluster_response_does_not_manufacture_null_dependence() -> None:
    """The untruncated ICC estimator remains near independence at the null."""

    _, summary = simulate_cluster_response()
    variance = summary.loc[summary["measure"].eq("event_variance_inflation")]
    null = variance.loc[variance["setting"].eq("zero")].squeeze()
    source_anchored = variance.loc[variance["setting"].eq("source_anchored")].squeeze()
    assert null["mean"] == pytest.approx(1.0, abs=0.02)
    assert source_anchored["ci_low"] > 1.0
    cluster_coverage = summary.loc[
        summary["measure"].eq("cluster_robust_covered")
        & summary["setting"].eq("source_anchored")
    ].squeeze()
    participant_coverage = summary.loc[
        summary["measure"].eq("participant_independent_covered")
        & summary["setting"].eq("source_anchored")
    ].squeeze()
    strong_participant_coverage = summary.loc[
        summary["measure"].eq("participant_independent_covered")
        & summary["setting"].eq("strong")
    ].squeeze()
    assert cluster_coverage["ci_low"] < 0.95 < cluster_coverage["ci_high"]
    assert participant_coverage["ci_high"] < 0.95
    assert strong_participant_coverage["ci_high"] < 0.75


def test_cluster_response_writer_refuses_to_replace_results(tmp_path) -> None:
    """The writer creates complete tables once and fails on replacement."""

    output = tmp_path / "cluster"
    worlds_path, summary_path = write_cluster_response(output)
    assert len(pd.read_csv(worlds_path)) == 5_000
    assert len(pd.read_csv(summary_path)) == len(SETTINGS) * 8
    with pytest.raises(FileExistsError):
        write_cluster_response(output)
