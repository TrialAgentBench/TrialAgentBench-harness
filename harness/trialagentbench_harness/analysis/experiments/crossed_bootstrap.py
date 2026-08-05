"""Two-way cluster bootstrap for paired benchmark contrasts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class CrossedBootstrapEstimate:
    """One estimate with two-way cluster-bootstrap uncertainty."""

    n_observations: int
    n_row_clusters: int
    n_column_clusters: int
    estimate: float
    interval_low: float
    interval_high: float


def crossed_cluster_bootstrap_mean(
    *,
    values: list[tuple[str, str, float]],
    min_row_clusters: int,
    min_column_clusters: int,
    resamples: int,
    confidence_level: float,
    seed: int,
    contrast_id: str,
    weighting: Literal["observation", "crossed_cell"],
) -> CrossedBootstrapEstimate:
    """Estimate a mean while resampling both independent cluster dimensions."""

    if not values:
        raise ValueError(f"Crossed bootstrap {contrast_id!r} requires observations.")
    if resamples < 1 or not 0.0 < confidence_level < 1.0:
        raise ValueError("Crossed bootstrap requires positive resamples and a confidence level in (0, 1).")
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row_cluster, column_cluster, value in values:
        if not row_cluster or not column_cluster or not np.isfinite(value):
            raise ValueError(f"Crossed bootstrap {contrast_id!r} received an invalid observation.")
        by_cell[(row_cluster, column_cluster)].append(float(value))
    row_clusters = sorted({row_cluster for row_cluster, _, _ in values})
    column_clusters = sorted({column_cluster for _, column_cluster, _ in values})
    if len(row_clusters) < min_row_clusters:
        raise ValueError(
            f"Crossed bootstrap {contrast_id!r} has {len(row_clusters)} row clusters; requires {min_row_clusters}."
        )
    if len(column_clusters) < min_column_clusters:
        raise ValueError(
            f"Crossed bootstrap {contrast_id!r} has {len(column_clusters)} column clusters; "
            f"requires {min_column_clusters}."
        )
    missing = [
        (row_cluster, column_cluster)
        for row_cluster in row_clusters
        for column_cluster in column_clusters
        if (row_cluster, column_cluster) not in by_cell
    ]
    if missing:
        raise ValueError(f"Crossed bootstrap {contrast_id!r} lacks crossed cells: {missing!r}.")

    cell_means = {key: float(np.mean(cell)) for key, cell in by_cell.items()}

    def sample_mean(sampled_rows: np.ndarray, sampled_columns: np.ndarray) -> float:
        if weighting == "crossed_cell":
            sample = [
                cell_means[(str(row_cluster), str(column_cluster))]
                for row_cluster in sampled_rows
                for column_cluster in sampled_columns
            ]
        else:
            sample = [
                value
                for row_cluster in sampled_rows
                for column_cluster in sampled_columns
                for value in by_cell[(str(row_cluster), str(column_cluster))]
            ]
        return float(np.mean(sample))

    if weighting == "crossed_cell":
        estimate = float(np.mean(tuple(cell_means.values())))
    else:
        estimate = float(np.mean([value for _, _, value in values]))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled_rows = rng.choice(row_clusters, size=len(row_clusters), replace=True)
        sampled_columns = rng.choice(column_clusters, size=len(column_clusters), replace=True)
        bootstrap[index] = sample_mean(sampled_rows, sampled_columns)
    tail = (1.0 - confidence_level) / 2.0
    interval_low, interval_high = np.quantile(bootstrap, [tail, 1.0 - tail]).tolist()
    return CrossedBootstrapEstimate(
        n_observations=len(values),
        n_row_clusters=len(row_clusters),
        n_column_clusters=len(column_clusters),
        estimate=estimate,
        interval_low=float(interval_low),
        interval_high=float(interval_high),
    )


__all__ = ["CrossedBootstrapEstimate", "crossed_cluster_bootstrap_mean"]
