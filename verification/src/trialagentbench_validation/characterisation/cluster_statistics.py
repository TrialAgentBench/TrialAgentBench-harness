"""Cluster-level information measures used by release validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ClusterInformation:
    """One-way event dependence and its implied information loss."""

    intraclass_correlation: float
    effective_cluster_size: float
    variance_inflation: float
    information_loss_fraction: float


def one_way_cluster_information(
    values: pd.Series,
    groups: pd.Series,
    *,
    strata: pd.Series | None = None,
) -> ClusterInformation:
    """Estimate stratified one-way random-effects dependence.

    The method-of-moments intraclass correlation remains untruncated so an
    independence experiment is centred on zero rather than biased upward.
    """

    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="raise"),
            "group": groups.astype("string"),
            "stratum": "overall" if strata is None else strata.astype("string"),
        }
    )
    if frame[["value", "group", "stratum"]].isna().any().any():
        raise ValueError(
            "intraclass correlation requires complete values, groups, and strata"
        )
    if frame.groupby("group")["stratum"].nunique().gt(1).any():
        raise ValueError("each cluster must belong to exactly one stratum")
    summaries = frame.groupby(["stratum", "group"])["value"].agg(["size", "mean"])
    group_count = len(summaries)
    stratum_count = frame["stratum"].nunique()
    observation_count = len(frame)
    if group_count - stratum_count < 2 or observation_count <= group_count:
        raise ValueError(
            "intraclass correlation requires at least two between-cluster degrees of freedom"
        )
    stratum_means = frame.groupby("stratum")["value"].mean()
    summary_strata = summaries.index.get_level_values("stratum")
    aligned_stratum_means = np.asarray(
        [float(stratum_means.loc[str(value)]) for value in summary_strata]
    )
    between = float(
        np.sum(
            summaries["size"].to_numpy(dtype=float)
            * np.square(summaries["mean"] - aligned_stratum_means)
        )
    )
    numeric = frame["value"].to_numpy(dtype=float)
    group_keys = pd.MultiIndex.from_frame(frame[["stratum", "group"]])
    group_means = summaries["mean"].reindex(group_keys).to_numpy(dtype=float)
    within = float(np.sum(np.square(numeric - group_means)))
    mean_between = between / (group_count - stratum_count)
    mean_within = within / (observation_count - group_count)
    effective_size_numerator = 0.0
    for _, rows in summaries.groupby(level="stratum"):
        stratum_size = float(rows["size"].sum())
        effective_size_numerator += (
            stratum_size - float(np.sum(np.square(rows["size"]))) / stratum_size
        )
    effective_size = effective_size_numerator / (group_count - stratum_count)
    denominator = mean_between + (effective_size - 1.0) * mean_within
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError(
            "intraclass correlation has a non-positive variance denominator"
        )
    intraclass_correlation = (mean_between - mean_within) / denominator
    variance_inflation = 1.0 + (effective_size - 1.0) * intraclass_correlation
    if not math.isfinite(variance_inflation) or variance_inflation <= 0:
        raise ValueError("intraclass correlation implies a non-positive variance ratio")
    return ClusterInformation(
        intraclass_correlation=intraclass_correlation,
        effective_cluster_size=effective_size,
        variance_inflation=variance_inflation,
        information_loss_fraction=max(0.0, 1.0 - 1.0 / variance_inflation),
    )


__all__ = ["ClusterInformation", "one_way_cluster_information"]
