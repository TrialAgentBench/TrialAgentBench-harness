"""Tests for the stepped-wedge response experiment."""

from __future__ import annotations

import pandas as pd
import pytest

from trialagentbench_validation.characterisation.cluster_response import (
    SOURCE_ANCHORED_SETTING,
)
from trialagentbench_validation.characterisation.stepped_wedge_response import (
    SETTINGS,
    simulate_stepped_wedge_response,
    write_stepped_wedge_response,
)


def test_stepped_wedge_response_is_deterministic_and_discriminating() -> None:
    """Period adjustment separates treatment from a graded secular trend."""

    worlds, summary = simulate_stepped_wedge_response(world_count=20)
    repeated, repeated_summary = simulate_stepped_wedge_response(world_count=20)
    pd.testing.assert_frame_equal(worlds, repeated)
    pd.testing.assert_frame_equal(summary, repeated_summary)

    assert set(worlds["setting"]) == {setting.setting for setting in SETTINGS}
    assert worlds.groupby("setting").size().eq(20).all()
    naive_bias = summary.loc[summary["measure"].eq("period_omitting_bias")].sort_values(
        "secular_hazard_ratio_period_4_to_1"
    )
    assert naive_bias["mean"].is_monotonic_increasing
    adjusted_bias = summary.loc[summary["measure"].eq("period_adjusted_bias")]
    assert adjusted_bias["mean"].abs().max() < 0.08


def test_stepped_wedge_response_recovers_benchmark_effect() -> None:
    """The benchmark trend retains adjusted coverage and defeats period omission."""

    _, summary = simulate_stepped_wedge_response()
    benchmark = summary.loc[summary["setting"].eq("benchmark")]
    adjusted_bias = benchmark.loc[
        benchmark["measure"].eq("period_adjusted_bias")
    ].squeeze()
    naive_bias = benchmark.loc[
        benchmark["measure"].eq("period_omitting_bias")
    ].squeeze()
    adjusted_coverage = benchmark.loc[
        benchmark["measure"].eq("period_adjusted_covered")
    ].squeeze()
    naive_coverage = benchmark.loc[
        benchmark["measure"].eq("period_omitting_covered")
    ].squeeze()
    assert adjusted_bias["ci_low"] < 0 < adjusted_bias["ci_high"]
    assert naive_bias["ci_low"] > 0
    assert adjusted_coverage["ci_low"] < 0.95 < adjusted_coverage["ci_high"]
    assert naive_coverage["ci_high"] < 0.95
    assert (
        benchmark["cluster_hazard_ratio_90_to_10"]
        .eq(SOURCE_ANCHORED_SETTING.hazard_ratio_90_to_10)
        .all()
    )


def test_stepped_wedge_response_writer_refuses_to_replace_results(tmp_path) -> None:
    """The writer creates complete tables once and fails on replacement."""

    output = tmp_path / "stepped_wedge"
    worlds_path, summary_path = write_stepped_wedge_response(output)
    assert len(pd.read_csv(worlds_path)) == 5_000
    assert len(pd.read_csv(summary_path)) == len(SETTINGS) * 10
    with pytest.raises(FileExistsError):
        write_stepped_wedge_response(output)
