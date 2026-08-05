"""Deterministic delete-a-group uncertainty for public analyses."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import ArrayLike, NDArray

DELETE_GROUP_COUNT_V1 = 20


def balanced_delete_groups_v1(
    *, unit_ids: ArrayLike, strata: ArrayLike, n_groups: int
) -> NDArray[np.int64]:
    """Assign unique units to hash-stable groups balanced within strata."""

    units = np.asarray(unit_ids, dtype=str)
    stratum_values = np.asarray(strata, dtype=str)
    if units.ndim != 1 or stratum_values.shape != units.shape:
        raise ValueError(
            "Jackknife unit IDs and strata must be aligned one-dimensional arrays."
        )
    if (
        units.size == 0
        or np.any(np.char.strip(units) == "")
        or np.any(np.char.strip(stratum_values) == "")
    ):
        raise ValueError("Jackknife unit IDs and strata must be non-empty strings.")
    if int(np.unique(units).size) != int(units.size):
        raise ValueError("Jackknife unit IDs must be unique.")
    group_count = int(n_groups)
    if group_count < 2 or group_count > int(units.size):
        raise ValueError(
            "Jackknife n_groups must be between 2 and the number of resampling units."
        )
    assignments = np.full(units.size, -1, dtype=np.int64)
    for stratum in sorted(set(str(value) for value in stratum_values)):
        indices = np.flatnonzero(stratum_values == stratum)
        if int(indices.size) < group_count:
            raise ValueError(
                f"Jackknife stratum {stratum!r} has {int(indices.size)} units and cannot populate "
                f"{group_count} delete groups."
            )
        ordered = sorted(
            indices.tolist(),
            key=lambda index: hashlib.sha256(
                str(units[index]).encode("utf-8")
            ).digest(),
        )
        for offset, index in enumerate(ordered):
            assignments[index] = int(offset % group_count)
    if np.any(assignments < 0):  # pragma: no cover - exhaustive stratum loop
        raise RuntimeError("Jackknife assignment left an unassigned resampling unit.")
    return assignments


def delete_group_standard_error_v1(estimates: ArrayLike) -> float:
    """Calculate the delete-a-group jackknife standard error."""

    values = np.asarray(estimates, dtype=np.float64)
    if values.ndim != 1 or int(values.size) < 2 or not np.isfinite(values).all():
        raise ValueError(
            "Jackknife replicate estimates must be a finite one-dimensional array of length >= 2."
        )
    centered = values - float(np.mean(values))
    variance = float(
        (float(values.size - 1) / float(values.size)) * np.sum(np.square(centered))
    )
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(
            "Jackknife replicate estimates produced non-positive variance."
        )
    return float(np.sqrt(variance))
