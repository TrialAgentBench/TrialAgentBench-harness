"""Participant-visible statistical utilities used by TrialDev grading."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from trialagentbench_harness.numeric_policy import TRIALDEV_RANDOMIZED_ARM_COUNT_V1
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentRequestV1


def complete_binary_indicator_v1(series: pd.Series) -> npt.NDArray[np.bool_]:
    """Return a complete exact binary indicator or fail on malformed data."""

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("Event indicators must be complete binary 0/1 values.")
    return values.astype(bool)


def _rank_sites(
    *,
    frame: pd.DataFrame,
    site_strategy: str,
) -> tuple[str, ...]:
    if "SITE_ID" not in frame.columns:
        raise ValueError("Site-budgeted planning requires SITE_ID in the public cohort.")
    counts = frame["SITE_ID"].astype("string").fillna("NA").value_counts().sort_index()
    if counts.empty:
        return ()
    if site_strategy != "region_balanced":
        return tuple(
            site_id
            for site_id, _ in sorted(
                ((str(site_id), int(count)) for site_id, count in counts.items()),
                key=lambda row: (-row[1], row[0]),
            )
        )
    if "REGION" not in frame.columns:
        raise ValueError("Region-balanced planning requires REGION in the public cohort.")
    site_regions = frame.loc[:, ["SITE_ID", "REGION"]].copy()
    site_regions["SITE_ID"] = site_regions["SITE_ID"].astype("string").fillna("NA")
    site_regions["REGION"] = site_regions["REGION"].astype("string").fillna("NA")
    region_by_site = (
        site_regions.groupby("SITE_ID", dropna=False)["REGION"]
        .agg(lambda values: str(values.mode(dropna=False).iloc[0]))
        .to_dict()
    )
    sites_by_region: dict[str, list[tuple[str, int]]] = {}
    for site_id, count in counts.items():
        region = str(region_by_site[str(site_id)])
        sites_by_region.setdefault(region, []).append((str(site_id), int(count)))
    for region in sites_by_region:
        sites_by_region[region].sort(key=lambda row: (-row[1], row[0]))
    region_order = sorted(
        sites_by_region,
        key=lambda region: (
            -sum(count for _, count in sites_by_region[region]),
            region,
        ),
    )
    return tuple(
        sites_by_region[region][index][0]
        for index in range(max(len(rows) for rows in sites_by_region.values()))
        for region in region_order
        if index < len(sites_by_region[region])
    )


def operational_support_count_v1(
    *,
    baseline: pd.DataFrame,
    enrollment_window_days: int,
    site_count_budget: int,
    site_strategy: str,
) -> int:
    """Return recruitment support available from the participant-visible cohort."""

    if enrollment_window_days < 0:
        raise ValueError("enrollment_window_days must be non-negative.")
    if site_count_budget < 1:
        raise ValueError("site_count_budget must be positive.")
    if "ENROLLMENT_DAY" not in baseline.columns:
        raise ValueError("Enrollment-window planning requires ENROLLMENT_DAY in the public cohort.")
    enrollment_day = pd.to_numeric(baseline["ENROLLMENT_DAY"], errors="coerce")
    eligible = baseline.loc[enrollment_day <= float(enrollment_window_days), :].copy()
    selected_sites = _rank_sites(frame=eligible, site_strategy=site_strategy)[:site_count_budget]
    return int(eligible["SITE_ID"].astype("string").isin(selected_sites).sum())


def _allocation_weights(
    request: TrialDevelopmentRequestV1,
) -> dict[str, float]:
    if request.allocation_weights:
        values = np.asarray(request.allocation_weights, dtype=float)
        if (
            values.shape != (TRIALDEV_RANDOMIZED_ARM_COUNT_V1,)
            or not np.isfinite(values).all()
            or (values <= 0.0).any()
        ):
            raise ValueError("Randomized allocation requires two finite positive weights.")
        normalized = values / float(values.sum())
        return {"CONTROL": float(normalized[0]), "TREATMENT": float(normalized[1])}
    if request.allocation_ratio is None:
        raise ValueError("Randomized request requires allocation_ratio or allocation_weights.")
    parts = str(request.allocation_ratio).strip().split(":")
    if len(parts) != TRIALDEV_RANDOMIZED_ARM_COUNT_V1:
        raise ValueError("allocation_ratio must be in the form 'a:b'.")
    treatment_weight, control_weight = (int(part) for part in parts)
    if treatment_weight < 1 or control_weight < 1:
        raise ValueError("allocation_ratio parts must be positive.")
    total = float(treatment_weight + control_weight)
    return {
        "CONTROL": float(control_weight / total),
        "TREATMENT": float(treatment_weight / total),
    }


def planned_arm_allocation_v1(
    *,
    request: TrialDevelopmentRequestV1,
    control_drug_id: str,
) -> tuple[dict[str, str], dict[str, int]]:
    """Return deterministic randomized-arm identities and planned counts."""

    if request.target_sample_size is None or request.target_sample_size < TRIALDEV_RANDOMIZED_ARM_COUNT_V1:
        raise ValueError("Randomized request requires at least two participants.")
    weights = _allocation_weights(request)
    arm_ids = ("CONTROL", "TREATMENT")
    raw = {arm_id: weights[arm_id] * request.target_sample_size for arm_id in arm_ids}
    counts = {arm_id: max(1, int(np.floor(raw[arm_id]))) for arm_id in arm_ids}
    while sum(counts.values()) < request.target_sample_size:
        selected = min(
            arm_ids,
            key=lambda arm_id: (
                -(raw[arm_id] - counts[arm_id]),
                arm_id,
            ),
        )
        counts[selected] += 1
    while sum(counts.values()) > request.target_sample_size:
        eligible = tuple(arm_id for arm_id in arm_ids if counts[arm_id] > 1)
        if not eligible:
            raise ValueError("Requested allocation cannot preserve a non-empty control and treatment arm.")
        selected = min(
            eligible,
            key=lambda arm_id: (
                raw[arm_id] - counts[arm_id],
                arm_id,
            ),
        )
        counts[selected] -= 1
    return (
        {
            "CONTROL": str(control_drug_id),
            "TREATMENT": str(request.primary_candidate_drug_id),
        },
        counts,
    )


__all__ = [
    "complete_binary_indicator_v1",
    "operational_support_count_v1",
    "planned_arm_allocation_v1",
]
