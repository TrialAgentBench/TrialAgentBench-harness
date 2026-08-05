"""Tests for shared two-way cluster-bootstrap inference."""

from __future__ import annotations

import pytest

from trialagentbench_harness.analysis.experiments.crossed_bootstrap import (
    crossed_cluster_bootstrap_mean,
)


def test_crossed_bootstrap_preserves_declared_weighting_and_is_deterministic() -> None:
    values = [
        ("family-a", "seed-1", 0.0),
        ("family-a", "seed-1", 0.0),
        ("family-a", "seed-2", 0.0),
        ("family-b", "seed-1", 1.0),
        ("family-b", "seed-2", 1.0),
    ]
    kwargs = {
        "values": values,
        "min_row_clusters": 2,
        "min_column_clusters": 2,
        "resamples": 1000,
        "confidence_level": 0.9,
        "seed": 17,
        "contrast_id": "test",
    }

    observation = crossed_cluster_bootstrap_mean(**kwargs, weighting="observation")
    crossed_cell = crossed_cluster_bootstrap_mean(**kwargs, weighting="crossed_cell")

    assert observation.estimate == pytest.approx(0.4)
    assert crossed_cell.estimate == pytest.approx(0.5)
    assert crossed_cell == crossed_cluster_bootstrap_mean(**kwargs, weighting="crossed_cell")


def test_crossed_bootstrap_rejects_incomplete_crossing() -> None:
    with pytest.raises(ValueError, match="lacks crossed cells"):
        crossed_cluster_bootstrap_mean(
            values=[
                ("family-a", "seed-1", 0.0),
                ("family-a", "seed-2", 0.0),
                ("family-b", "seed-1", 1.0),
            ],
            min_row_clusters=2,
            min_column_clusters=2,
            resamples=100,
            confidence_level=0.9,
            seed=17,
            contrast_id="incomplete",
            weighting="crossed_cell",
        )


def test_crossed_bootstrap_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="invalid observation"):
        crossed_cluster_bootstrap_mean(
            values=[
                ("family-a", "seed-1", 0.0),
                ("family-a", "seed-2", float("nan")),
                ("family-b", "seed-1", 1.0),
                ("family-b", "seed-2", 1.0),
            ],
            min_row_clusters=2,
            min_column_clusters=2,
            resamples=100,
            confidence_level=0.9,
            seed=17,
            contrast_id="nonfinite",
            weighting="crossed_cell",
        )
