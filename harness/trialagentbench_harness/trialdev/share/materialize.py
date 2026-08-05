"""Deterministic trial materialization from frozen scenario bundles."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from trialagentbench_harness.trialdev.share.hashing import compute_sha256_hex
from trialagentbench_harness.trialdev.share.io import read_json, write_json
from trialagentbench_harness.trialdev.share.models import (
    FrozenSuperpopulationManifestV1,
    TrialDevelopmentEvalContractV1,
    TrialDevelopmentRequestV1,
    TrialMaterializationAuditV1,
    TrialMaterializationResultV1,
)
from trialagentbench_harness.trialdev.share.safety_policy import serious_event_definitions_v1
from trialagentbench_harness.trialdev.share.validate import validate_request_against_scenario_v1

__all__ = [
    "apply_endpoint_estimand_v1",
    "complete_binary_indicator_v1",
    "materialize_trial_view_v1",
    "operational_support_count_v1",
    "planned_arm_allocation_v1",
    "randomize_ltfu_within_arm_v1",
]

logger = logging.getLogger(__name__)


def _parse_allocation_ratio(value: str) -> tuple[int, int]:
    raw = str(value).strip()
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError("allocation_ratio must be in the form 'a:b'.")
    a = int(parts[0])
    b = int(parts[1])
    if a < 1 or b < 1:
        raise ValueError("allocation_ratio parts must be >= 1.")
    return a, b


def _stable_rank_scores(*, usubjid: pd.Series, seed: int, request_checksum: str) -> npt.NDArray[np.uint64]:
    ids = usubjid.astype("string").fillna("").tolist()
    scores: list[int] = []
    for subj in ids:
        payload = {"seed": int(seed), "request": str(request_checksum), "usubjid": str(subj)}
        digest = compute_sha256_hex(payload)
        scores.append(int(digest[:16], 16))
    return np.asarray(scores, dtype=np.uint64)


def randomize_ltfu_within_arm_v1(
    frame: pd.DataFrame,
    *,
    seed: int,
    request_checksum: str,
) -> pd.DataFrame:
    """Break subject-outcome dependence while preserving arm-specific LTFU margins."""

    required = {"USUBJID", "ARM", "ORACLE__LTFU_T", "ORACLE__LTFU_E"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Randomized LTFU construction lacks required columns: {missing!r}.")
    output = frame.copy()
    for arm_id, arm in output.groupby("ARM", observed=True, sort=True):
        indices = arm.index.to_numpy(dtype=np.int64)
        if len(indices) < 2:
            continue
        permutation_seed = int(
            compute_sha256_hex(
                {
                    "arm_id": str(arm_id),
                    "request_checksum": str(request_checksum),
                    "seed": int(seed),
                    "system": "arm_conditional_ltfu_permutation_v1",
                }
            )[:16],
            16,
        )
        permutation = np.random.default_rng(permutation_seed).permutation(len(indices))
        source = output.loc[indices, ["ORACLE__LTFU_T", "ORACLE__LTFU_E"]].to_numpy(copy=True)
        output.loc[indices, ["ORACLE__LTFU_T", "ORACLE__LTFU_E"]] = source[permutation]
    return output


def complete_binary_indicator_v1(series: pd.Series) -> npt.NDArray[np.bool_]:
    """Return a complete exact binary indicator or fail on malformed state."""

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("Event indicators must be complete binary 0/1 values.")
    return values.astype(bool)


def _coerce_time(series: pd.Series) -> npt.NDArray[np.float64]:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _active_time(
    *, event: npt.NDArray[np.bool_] | None, time: npt.NDArray[np.float64] | None, size: int
) -> npt.NDArray[np.float64]:
    if event is None or time is None:
        return np.full(size, np.inf, dtype=float)
    event_array = np.asarray(event, dtype=bool)
    time_array = np.asarray(time, dtype=float)
    if event_array.shape != (size,) or time_array.shape != (size,):
        raise ValueError("Intercurrent event and time arrays must match the endpoint rows.")
    if np.isnan(time_array).any() or (time_array < 0.0).any():
        raise ValueError("Event times must be non-negative and must not contain NaN values.")
    return np.where(event_array & np.isfinite(time_array), time_array, np.inf)


def apply_endpoint_estimand_v1(
    *,
    strategy: str,
    endpoint_time: npt.NDArray[np.float64],
    endpoint_event: npt.NDArray[np.bool_],
    follow_up_days: float,
    discontinuation_time: npt.NDArray[np.float64] | None,
    discontinuation_event: npt.NDArray[np.bool_] | None,
    ltfu_time: npt.NDArray[np.float64] | None,
    ltfu_event: npt.NDArray[np.bool_] | None,
    terminal_time: npt.NDArray[np.float64] | None,
    terminal_event: npt.NDArray[np.bool_] | None,
) -> tuple[
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
    npt.NDArray[np.float64],
    npt.NDArray[np.object_],
]:
    strategy_id = str(strategy)
    if strategy_id not in {"treatment_policy", "while_on_treatment", "composite_discontinuation"}:
        raise ValueError(f"Unsupported estimand strategy: {strategy_id!r}.")
    follow = float(follow_up_days)
    if not np.isfinite(follow) or follow <= 0.0:
        raise ValueError("follow_up_days must be finite and positive.")
    endpoint_t = np.asarray(endpoint_time, dtype=float)
    endpoint_e = np.asarray(endpoint_event, dtype=bool)
    if endpoint_t.ndim != 1 or endpoint_e.shape != endpoint_t.shape or endpoint_t.size == 0:
        raise ValueError("Endpoint event and time arrays must be non-empty one-dimensional arrays of equal length.")
    if np.isnan(endpoint_t).any() or (endpoint_t < 0.0).any():
        raise ValueError("Endpoint times must be non-negative and must not contain NaN values.")
    size = int(endpoint_t.size)
    endpoint_event_time = np.where(endpoint_e & np.isfinite(endpoint_t), endpoint_t, np.inf)
    endpoint_censor_time = np.where((~endpoint_e) & np.isfinite(endpoint_t), endpoint_t, np.inf)
    disc_time = _active_time(event=discontinuation_event, time=discontinuation_time, size=size)
    ltfu_time_array = _active_time(event=ltfu_event, time=ltfu_time, size=size)
    competing_time = _active_time(event=terminal_event, time=terminal_time, size=size)
    strategy_censor_time: npt.NDArray[np.float64]
    if strategy_id == "composite_discontinuation":
        event_time = np.minimum(endpoint_event_time, disc_time)
        strategy_censor_time = np.full(size, np.inf, dtype=np.float64)
    else:
        event_time = endpoint_event_time
        strategy_censor_time = (
            disc_time if strategy_id == "while_on_treatment" else np.full(size, np.inf, dtype=np.float64)
        )
    censor_time = np.minimum(np.minimum(endpoint_censor_time, ltfu_time_array), strategy_censor_time)
    eps = 1e-9
    event_observed = (
        np.isfinite(event_time)
        & (event_time <= follow + eps)
        & (event_time <= censor_time + eps)
        & (event_time < competing_time - eps)
    )
    competing_observed = (
        np.isfinite(competing_time)
        & (competing_time <= follow + eps)
        & (competing_time <= censor_time + eps)
        & (competing_time <= event_time + eps)
    )
    observed_time = np.minimum.reduce((event_time, competing_time, censor_time, np.full(size, follow))).astype(float)
    cause: npt.NDArray[np.object_] = np.full(size, "administrative", dtype=object)
    cause[competing_observed] = "terminal_event"
    cause[event_observed] = "endpoint_event"
    if strategy_id == "composite_discontinuation":
        cause[event_observed & (disc_time <= endpoint_event_time + eps)] = "composite_discontinuation"
    censored = ~(event_observed | competing_observed)
    cause[censored & (endpoint_censor_time <= observed_time + eps)] = "model_censoring"
    cause[censored & (ltfu_time_array <= observed_time + eps)] = "ltfu"
    cause[censored & (strategy_censor_time <= observed_time + eps)] = "while_on_treatment_discontinuation"
    return event_observed.astype(int), competing_observed.astype(int), observed_time, cause


def _encode_stratification_variables(*, df: pd.DataFrame, variables: tuple[str, ...]) -> pd.DataFrame:
    encoded = pd.DataFrame(index=df.index)
    for variable in variables:
        series = df[str(variable)]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == int(len(series)):
            unique = int(numeric.nunique(dropna=True))
            if unique <= 2:
                encoded[str(variable)] = numeric.round(0).astype("int64").astype("string")
                continue
            if unique > 8:
                q = min(4, unique)
                ranked = numeric.rank(method="first")
                bins = pd.qcut(
                    ranked,
                    q=q,
                    labels=[f"Q{i + 1}" for i in range(q)],
                    duplicates="drop",
                )
                encoded[str(variable)] = bins.astype("string").fillna("MISSING")
                continue
            encoded[str(variable)] = numeric.round(6).map(lambda x: f"{float(x):g}").astype("string")
            continue
        if pd.api.types.is_bool_dtype(series):
            encoded[str(variable)] = series.astype("boolean").astype("string").fillna("MISSING")
            continue
        encoded[str(variable)] = series.astype("string").fillna("MISSING")
    return encoded


def _largest_remainder_quotas(*, counts: pd.Series, total: int) -> dict[str, int]:
    if int(total) < 0:
        raise ValueError("Quota total must be >= 0.")
    series = counts.astype("int64").sort_index()
    denom = int(series.sum())
    if denom <= 0:
        return {str(k): 0 for k in series.index.astype("string")}
    raw = series.astype(float) * (float(total) / float(denom))
    base = raw.apply(np.floor).astype("int64")
    remainder = int(total) - int(base.sum())
    order = sorted(
        [str(idx) for idx in series.index.astype("string")],
        key=lambda idx: (-float(raw.loc[idx] - base.loc[idx]), str(idx)),
    )
    quotas = {str(idx): int(base.loc[idx]) for idx in series.index.astype("string")}
    for idx in order[:remainder]:
        quotas[str(idx)] = int(quotas[str(idx)]) + 1
    return quotas


def _format_stratum_key_frame(*, encoded: pd.DataFrame) -> pd.Series:
    if encoded.empty:
        return pd.Series(["ALL"] * int(encoded.shape[0]), index=encoded.index, dtype="string")
    return encoded.apply(lambda row: "|".join(f"{col}={row[col]}" for col in encoded.columns), axis=1).astype("string")


def _rank_sites_by_budget(*, df: pd.DataFrame, site_strategy: str) -> tuple[str, ...]:
    if "SITE_ID" not in df.columns:
        raise ValueError("Site-budgeted materialization requires SITE_ID in the frozen pool.")
    counts = df["SITE_ID"].astype("string").fillna("NA").value_counts().sort_index()
    if counts.empty:
        return tuple()

    def _score(site_id: str, count: int) -> tuple[float, str]:
        return (-float(count), str(site_id))

    if str(site_strategy) == "region_balanced":
        if "REGION" not in df.columns:
            raise ValueError("Region-balanced site materialization requires REGION in the frozen pool.")
        site_frame = df.loc[:, ["SITE_ID", "REGION"]].copy()
        site_frame["SITE_ID"] = site_frame["SITE_ID"].astype("string").fillna("NA")
        site_frame["REGION"] = site_frame["REGION"].astype("string").fillna("NA")
        region_by_site = (
            site_frame.groupby("SITE_ID", dropna=False)["REGION"]
            .agg(lambda values: str(values.mode(dropna=False).iloc[0]))
            .to_dict()
        )
        sites_by_region: dict[str, list[tuple[str, int]]] = {}
        for site_id, count in counts.to_dict().items():
            region = str(region_by_site.get(str(site_id), "NA"))
            sites_by_region.setdefault(region, []).append((str(site_id), int(count)))
        for region, entries in sites_by_region.items():
            sites_by_region[region] = sorted(entries, key=lambda entry: _score(*entry))
        region_order = sorted(
            sites_by_region,
            key=lambda region: (-sum(count for _, count in sites_by_region[region]), region),
        )
        ordered_sites: list[str] = []
        for index in range(max(len(entries) for entries in sites_by_region.values())):
            for region in region_order:
                entries = sites_by_region[region]
                if index < len(entries):
                    ordered_sites.append(entries[index][0])
        return tuple(ordered_sites)

    ordered = sorted(
        ((str(site_id), int(count)) for site_id, count in counts.to_dict().items()),
        key=lambda kv: _score(kv[0], kv[1]),
    )
    return tuple(str(site_id) for site_id, _ in ordered)


@dataclass(frozen=True)
class _OperationalCohort:
    """Intermediate public-cohort selection used by planning and materialization."""

    frame: pd.DataFrame
    after_enrollment_window_count: int
    selected_sites: tuple[str, ...]


def _select_operational_cohort(
    *,
    baseline: pd.DataFrame,
    enrollment_window_days: int | None,
    site_count_budget: int | None,
    site_strategy: str,
) -> _OperationalCohort:
    eligible = baseline.copy()

    if enrollment_window_days is not None:
        if "ENROLLMENT_DAY" not in eligible.columns:
            raise ValueError("Enrollment-window materialization requires ENROLLMENT_DAY in the public cohort.")
        eligible = eligible.loc[
            pd.to_numeric(eligible["ENROLLMENT_DAY"], errors="coerce").fillna(np.inf) <= float(enrollment_window_days),
            :,
        ].copy()
    after_enrollment_window_count = int(len(eligible))

    selected_sites: tuple[str, ...] = ()
    if site_count_budget is not None:
        selected_sites = _rank_sites_by_budget(
            df=eligible,
            site_strategy=site_strategy,
        )[: int(site_count_budget)]
        eligible = eligible.loc[eligible["SITE_ID"].astype("string").isin(selected_sites), :].copy()
    return _OperationalCohort(
        frame=eligible,
        after_enrollment_window_count=after_enrollment_window_count,
        selected_sites=selected_sites,
    )


def operational_support_count_v1(
    *,
    baseline: pd.DataFrame,
    enrollment_window_days: int,
    site_count_budget: int,
    site_strategy: str,
) -> int:
    """Return exact recruitment support from the participant-visible cohort."""

    return int(
        len(
            _select_operational_cohort(
                baseline=baseline,
                enrollment_window_days=enrollment_window_days,
                site_count_budget=site_count_budget,
                site_strategy=site_strategy,
            ).frame
        )
    )


def _compute_stratified_treatment_counts(*, counts_by_stratum: dict[str, int], target_treat: int) -> dict[str, int]:
    if not counts_by_stratum:
        return {}
    ordered_keys = sorted(counts_by_stratum)
    counts = pd.Series({key: int(counts_by_stratum[key]) for key in ordered_keys}, dtype="int64")
    positive = counts[counts > 0]
    if positive.empty:
        return {key: 0 for key in ordered_keys}
    if any(int(v) < 2 for v in positive.tolist()):
        raise ValueError("Stratified materialization requires at least two selected subjects per populated stratum.")
    n_strata = int(len(positive))
    total = int(positive.sum())
    target_control = int(total - int(target_treat))
    if int(target_treat) < n_strata or int(target_control) < n_strata:
        raise ValueError(
            "Requested stratified allocation is infeasible because not every populated stratum can support both arms."
        )

    ratio = float(target_treat) / float(total)
    raw = positive.astype(float) * float(ratio)
    treat = raw.apply(np.floor).astype("int64").clip(lower=1)
    for key, count in positive.items():
        key_str = str(key)
        treat.loc[key_str] = min(int(treat.loc[key_str]), int(count) - 1)

    current = int(treat.sum())
    if current < int(target_treat):
        order = sorted(
            positive.index.astype("string").tolist(),
            key=lambda key: (-float(raw.loc[key] - treat.loc[key]), str(key)),
        )
        while current < int(target_treat):
            progressed = False
            for key in order:
                max_for_key = int(positive.loc[key]) - 1
                if int(treat.loc[key]) >= max_for_key:
                    continue
                treat.loc[key] = int(treat.loc[key]) + 1
                current += 1
                progressed = True
                if current >= int(target_treat):
                    break
            if not progressed:
                raise ValueError(
                    "Unable to satisfy requested stratified treatment allocation under arm-balance bounds."
                )
    elif current > int(target_treat):
        order = sorted(
            positive.index.astype("string").tolist(),
            key=lambda key: (float(raw.loc[key] - treat.loc[key]), str(key)),
        )
        while current > int(target_treat):
            progressed = False
            for key in order:
                if int(treat.loc[key]) <= 1:
                    continue
                treat.loc[key] = int(treat.loc[key]) - 1
                current -= 1
                progressed = True
                if current <= int(target_treat):
                    break
            if not progressed:
                raise ValueError("Unable to reduce stratified treatment allocation while preserving both arms.")

    return {str(key): int(treat.loc[key]) for key in ordered_keys if int(counts.loc[key]) > 0}


def _compute_arm_weight_map(
    *,
    request: TrialDevelopmentRequestV1,
    control_drug_id: str,
) -> tuple[tuple[str, ...], dict[str, str], dict[str, float]]:
    arm_ids = ("CONTROL", "TREATMENT")
    arm_to_drug_id = {
        "CONTROL": str(control_drug_id),
        "TREATMENT": str(request.primary_candidate_drug_id),
    }

    if request.allocation_weights:
        weights = np.asarray([float(v) for v in request.allocation_weights], dtype=float)
        weights = weights / float(weights.sum())
        return (
            tuple(arm_ids),
            arm_to_drug_id,
            {str(arm): float(weights[idx]) for idx, arm in enumerate(arm_ids)},
        )

    if request.allocation_ratio is None:
        raise ValueError("Request requires allocation_ratio or allocation_weights.")
    treat_weight, control_weight = _parse_allocation_ratio(str(request.allocation_ratio))
    total = float(treat_weight + control_weight)
    return (
        tuple(arm_ids),
        arm_to_drug_id,
        {
            "CONTROL": float(control_weight / total),
            "TREATMENT": float(treat_weight / total),
        },
    )


def planned_arm_allocation_v1(
    *,
    request: TrialDevelopmentRequestV1,
    control_drug_id: str,
) -> tuple[dict[str, str], dict[str, int]]:
    """Return deterministic drug identities and planned counts by arm."""

    if request.target_sample_size is None:
        raise ValueError("Randomized request requires target_sample_size.")
    arm_ids, arm_to_drug_id, arm_weights = _compute_arm_weight_map(
        request=request,
        control_drug_id=control_drug_id,
    )
    counts = _compute_total_arm_counts(
        total=int(request.target_sample_size),
        arm_ids=arm_ids,
        arm_weights=arm_weights,
    )
    return arm_to_drug_id, counts


def _arm_mapping_payload(
    *,
    request: TrialDevelopmentRequestV1,
    control_drug_id: str,
    arm_ids: tuple[str, ...],
    arm_to_drug_id: dict[str, str],
    arm_weights: dict[str, float],
) -> dict[str, object]:
    role_by_arm: dict[str, str] = {}
    for arm_id in arm_ids:
        role_by_arm[str(arm_id)] = (
            "control" if str(arm_to_drug_id[str(arm_id)]) == str(control_drug_id) else "candidate"
        )
    control_arms = tuple(str(arm) for arm, role in role_by_arm.items() if role == "control")
    if len(control_arms) != 1:
        raise ValueError("Materialized arm mapping must contain exactly one control arm.")
    payload: dict[str, object] = {
        "version": "v1",
        "scenario_id": str(request.scenario_id),
        "phase_id": str(request.phase_id),
        "request_checksum": request.checksum(),
        "control_drug_id": str(control_drug_id),
        "control_arm_id": str(control_arms[0]),
        "candidate_arm_ids": [str(arm) for arm in arm_ids if role_by_arm[str(arm)] == "candidate"],
        "drug_id_by_arm": {str(k): str(v) for k, v in sorted(arm_to_drug_id.items())},
        "arm_role_by_id": {str(k): str(v) for k, v in sorted(role_by_arm.items())},
        "arm_weight_by_id": {str(k): float(v) for k, v in sorted(arm_weights.items())},
        "request_candidate_drug_ids": [str(v) for v in request.candidate_drug_ids],
    }
    payload["checksum"] = compute_sha256_hex(payload)
    return payload


def _compute_total_arm_counts(
    *, total: int, arm_ids: tuple[str, ...], arm_weights: dict[str, float]
) -> dict[str, int]:
    if int(total) < int(len(arm_ids)):
        raise ValueError("Requested sample size is smaller than the number of required trial arms.")
    series = pd.Series({str(arm): float(arm_weights[str(arm)]) for arm in arm_ids}, dtype="float64")
    series = series / float(series.sum())
    raw = series * float(total)
    base = raw.apply(np.floor).astype("int64")
    for arm in arm_ids:
        if int(base.loc[str(arm)]) < 1:
            base.loc[str(arm)] = 1
    remainder = int(total) - int(base.sum())
    if remainder > 0:
        order = sorted(arm_ids, key=lambda arm: (-float(raw.loc[str(arm)] - base.loc[str(arm)]), str(arm)))
        idx = 0
        while remainder > 0:
            arm = str(order[idx % len(order)])
            base.loc[arm] = int(base.loc[arm]) + 1
            remainder -= 1
            idx += 1
    elif remainder < 0:
        order = sorted(arm_ids, key=lambda arm: (float(raw.loc[str(arm)] - base.loc[str(arm)]), str(arm)))
        idx = 0
        while remainder < 0:
            arm = str(order[idx % len(order)])
            if int(base.loc[arm]) > 1:
                base.loc[arm] = int(base.loc[arm]) - 1
                remainder += 1
            idx += 1
            if idx > 10000:
                raise ValueError("Unable to satisfy requested arm allocation while preserving non-empty arms.")
    counts = {str(arm): int(base.loc[str(arm)]) for arm in arm_ids}
    if int(sum(counts.values())) != int(total):
        raise ValueError("Arm count allocation failed to match requested total.")
    return counts


def _compute_stratified_arm_counts(
    *,
    counts_by_stratum: dict[str, int],
    arm_ids: tuple[str, ...],
    total_arm_counts: dict[str, int],
) -> dict[str, dict[str, int]]:
    if not counts_by_stratum:
        return {}
    positive = {str(k): int(v) for k, v in counts_by_stratum.items() if int(v) > 0}
    n_arms = int(len(arm_ids))
    if any(int(v) < n_arms for v in positive.values()):
        raise ValueError("Each populated stratum must support at least one subject in every arm.")
    n_strata = int(len(positive))
    for arm in arm_ids:
        if int(total_arm_counts[str(arm)]) < n_strata:
            raise ValueError("Each arm must have enough total support to appear in every populated stratum.")

    result = {stratum: {str(arm): 1 for arm in arm_ids} for stratum in sorted(positive)}
    residual_by_stratum = pd.Series(
        {stratum: int(count) - n_arms for stratum, count in sorted(positive.items())},
        dtype="int64",
    )
    residual_by_arm = {str(arm): int(total_arm_counts[str(arm)]) - n_strata for arm in arm_ids}
    ordered_arms = [str(arm) for arm in arm_ids]
    for arm in ordered_arms[:-1]:
        residual_total = int(residual_by_arm[str(arm)])
        quotas = _largest_remainder_quotas(counts=residual_by_stratum, total=int(residual_total))
        for stratum, add in quotas.items():
            quota = int(add)
            if quota > int(residual_by_stratum.loc[str(stratum)]):
                raise ValueError("Stratified arm allocation exceeded remaining stratum capacity.")
            result[str(stratum)][str(arm)] = int(result[str(stratum)][str(arm)]) + quota
            residual_by_stratum.loc[str(stratum)] = int(residual_by_stratum.loc[str(stratum)]) - quota

    final_arm = str(ordered_arms[-1])
    if int(residual_by_arm[final_arm]) != int(residual_by_stratum.sum()):
        raise ValueError("Final arm residual count does not match remaining stratum capacity.")
    for stratum in residual_by_stratum.index.astype("string"):
        result[str(stratum)][final_arm] = int(result[str(stratum)][final_arm]) + int(residual_by_stratum.loc[stratum])

    for stratum, count in positive.items():
        if int(sum(result[str(stratum)].values())) != int(count):
            raise ValueError("Stratified arm counts do not sum to the selected stratum size.")
    for arm in arm_ids:
        if int(sum(int(result[stratum][str(arm)]) for stratum in result)) != int(total_arm_counts[str(arm)]):
            raise ValueError("Stratified arm counts do not match requested arm totals.")
    return result


def materialize_trial_view_v1(
    *,
    scenario_root: Path,
    request: TrialDevelopmentRequestV1,
    seed: int,
    out_dir: Path,
    overwrite: bool = False,
) -> TrialMaterializationResultV1:
    """Materialize a governed trial view deterministically from a frozen scenario bundle."""
    root = Path(scenario_root)
    public_dir = root / "public"
    hidden_dir = root / "hidden"
    if not public_dir.is_dir() or not hidden_dir.is_dir():
        raise FileNotFoundError("Scenario bundle missing public/hidden surfaces.")

    eval_contract = TrialDevelopmentEvalContractV1.model_validate(read_json(public_dir / "eval_contract.json"))
    if str(request.scenario_id) != str(eval_contract.scenario_id):
        raise ValueError("Request scenario_id does not match the scenario bundle eval contract.")

    validate_request_against_scenario_v1(scenario_root=root, request=request)
    serious_event_definitions = serious_event_definitions_v1(scenario_root=root)

    sp = FrozenSuperpopulationManifestV1.model_validate(read_json(hidden_dir / "frozen_superpopulation_manifest.json"))
    pool = pd.read_parquet(hidden_dir / "counterfactual_pool.parquet")
    world_manifest = read_json(hidden_dir / "world_manifest.json")
    endpoint_outcome_ids = dict(world_manifest.get("endpoint_outcome_ids", {}) or {})

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        if not bool(overwrite):
            raise FileExistsError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    control = str(sp.control_drug_id)
    if pool.loc[pool["CANDIDATE_DRUG_ID"].astype("string") == control, :].empty:
        raise ValueError("Counterfactual pool is missing the control drug slice.")
    baseline = pd.read_parquet(public_dir / "observational_extract.parquet")
    required = {"USUBJID"}
    missing = sorted(required - set(baseline.columns))
    if missing:
        raise ValueError(f"Public observational cohort missing required columns: {missing!r}.")

    if str(request.phase_id) == "observational_review":
        raise ValueError("Materialization is not defined for phase_id='observational_review'.")

    request_checksum = request.checksum()
    if request.target_sample_size is None or request.follow_up_days is None:
        raise ValueError("Request requires target_sample_size and follow_up_days for trial phases.")
    if request.site_strategy is None:
        raise ValueError("Request requires an explicit site strategy for trial phases.")
    target_n = int(request.target_sample_size)
    follow_up_days = int(request.follow_up_days)
    stratification_variables = tuple(str(v) for v in request.stratification_variables)
    arm_ids, arm_to_drug_id, arm_weights = _compute_arm_weight_map(
        request=request,
        control_drug_id=str(control),
    )

    def _reject(
        *, reason: str, realized_sample_size: int, payload: dict[str, object] | None = None
    ) -> TrialMaterializationResultV1:
        audit = TrialMaterializationAuditV1(
            scenario_id=str(request.scenario_id),
            phase_id=str(request.phase_id),
            request_checksum=str(request_checksum),
            seed=int(seed),
            realized_sample_size=int(realized_sample_size),
            realized_follow_up_days=int(follow_up_days),
            feasibility_status="rejected",
            rejection_reason=str(reason),
            realized_arm_ids=tuple(str(v) for v in arm_ids),
            realized_arm_counts={},
            realized_stratification_variables=stratification_variables,
            realized_analysis_covariates=tuple(request.analysis_covariates),
            realized_subgroup_variables=tuple(request.subgroup_variables),
            realized_site_mix_summary={},
            payload={} if payload is None else payload,
        )
        write_json(out / "execution_summary.json", audit.model_dump(mode="json", exclude_none=True))
        return TrialMaterializationResultV1(audit=audit, trial_tables_dir=str(out), artifacts=tuple())

    try:
        operational = _select_operational_cohort(
            baseline=baseline,
            enrollment_window_days=request.enrollment_window_days,
            site_count_budget=request.site_count_budget,
            site_strategy=str(request.site_strategy),
        )
    except ValueError as exc:
        return _reject(
            reason="missing_operational_support_surface",
            realized_sample_size=int(baseline.shape[0]),
            payload={"detail": str(exc)},
        )
    eligible = operational.frame
    selected_sites = operational.selected_sites
    if request.enrollment_window_days is not None:
        if operational.after_enrollment_window_count < target_n:
            return _reject(
                reason="insufficient_enrollment_window_support",
                realized_sample_size=operational.after_enrollment_window_count,
                payload={
                    "eligible_subject_count": operational.after_enrollment_window_count,
                    "enrollment_window_days": int(request.enrollment_window_days),
                },
            )
    if request.site_count_budget is not None:
        if int(len(eligible)) < target_n:
            return _reject(
                reason="insufficient_site_budget_support",
                realized_sample_size=int(len(eligible)),
                payload={
                    "eligible_subject_count": int(len(eligible)),
                    "site_count_budget": int(request.site_count_budget),
                    "selected_sites": list(selected_sites),
                },
            )

    if int(len(eligible)) < target_n:
        return _reject(
            reason="insufficient_eligible_subjects",
            realized_sample_size=int(len(eligible)),
            payload={"eligible_subject_count": int(len(eligible))},
        )

    scores = _stable_rank_scores(usubjid=eligible["USUBJID"], seed=int(seed), request_checksum=str(request_checksum))
    eligible = eligible.assign(__SCORE__=scores)
    encoded_strata = _encode_stratification_variables(df=eligible, variables=stratification_variables)
    eligible = eligible.assign(__STRATUM__=_format_stratum_key_frame(encoded=encoded_strata))
    if stratification_variables:
        counts = eligible["__STRATUM__"].astype("string").value_counts().sort_index()
        quotas = _largest_remainder_quotas(counts=counts, total=int(target_n))
        positive_quotas = {k: int(v) for k, v in quotas.items() if int(v) > 0}
        if not positive_quotas:
            return _reject(reason="insufficient_stratum_support", realized_sample_size=0)
        if any(int(v) < 2 for v in positive_quotas.values()):
            return _reject(
                reason="insufficient_stratum_support",
                realized_sample_size=int(sum(positive_quotas.values())),
                payload={"requested_strata": positive_quotas},
            )
        selected_frames: list[pd.DataFrame] = []
        for stratum, quota in sorted(positive_quotas.items()):
            view = eligible.loc[eligible["__STRATUM__"].astype("string") == str(stratum), :].copy()
            view = view.sort_values(by=["__SCORE__", "USUBJID"], kind="mergesort").reset_index(drop=True)
            selected_frames.append(view.iloc[: int(quota), :].copy())
        selected = pd.concat(selected_frames, axis=0, ignore_index=True)
        selected = selected.sort_values(by=["__STRATUM__", "__SCORE__", "USUBJID"], kind="mergesort").reset_index(
            drop=True
        )
    else:
        eligible = eligible.sort_values(by=["__SCORE__", "USUBJID"], kind="mergesort").reset_index(drop=True)
        selected = eligible.iloc[:target_n, :].copy()
    selected = selected.drop(columns=["__SCORE__"])

    try:
        total_arm_counts = _compute_total_arm_counts(total=target_n, arm_ids=arm_ids, arm_weights=arm_weights)
    except ValueError as exc:
        return _reject(
            reason="infeasible_arm_allocation",
            realized_sample_size=int(selected.shape[0]),
            payload={"detail": str(exc), "arm_ids": [str(v) for v in arm_ids]},
        )
    assign_scores = _stable_rank_scores(
        usubjid=selected["USUBJID"], seed=int(seed) + 13, request_checksum=str(request_checksum)
    )
    selected = selected.assign(__A__=assign_scores)
    if stratification_variables:
        selected = selected.sort_values(by=["__STRATUM__", "__A__", "USUBJID"], kind="mergesort").reset_index(
            drop=True
        )
        stratum_counts = selected["__STRATUM__"].astype("string").value_counts().sort_index().to_dict()
        try:
            arm_counts_by_stratum = _compute_stratified_arm_counts(
                counts_by_stratum={str(k): int(v) for k, v in stratum_counts.items()},
                arm_ids=arm_ids,
                total_arm_counts=total_arm_counts,
            )
        except ValueError as exc:
            return _reject(
                reason="infeasible_stratified_allocation",
                realized_sample_size=int(selected.shape[0]),
                payload={
                    "detail": str(exc),
                    "selected_strata": {str(k): int(v) for k, v in stratum_counts.items()},
                },
            )
        assigned_frames: list[pd.DataFrame] = []
        for stratum, arm_counts in sorted(arm_counts_by_stratum.items()):
            view = selected.loc[selected["__STRATUM__"].astype("string") == str(stratum), :].copy()
            view = view.sort_values(by=["__A__", "USUBJID"], kind="mergesort").reset_index(drop=True)
            stratum_labels: list[str] = []
            for arm_id in arm_ids:
                stratum_labels.extend([str(arm_id)] * int(arm_counts[str(arm_id)]))
            if int(len(stratum_labels)) != int(view.shape[0]):
                raise ValueError("Stratified arm assignment count mismatch.")
            view["ARM"] = pd.Series(stratum_labels, dtype="string")
            assigned_frames.append(view)
        selected = pd.concat(assigned_frames, axis=0, ignore_index=True)
        selected = selected.sort_values(by=["__STRATUM__", "__A__", "USUBJID"], kind="mergesort").reset_index(
            drop=True
        )
    else:
        selected = selected.sort_values(by=["__A__", "USUBJID"], kind="mergesort").reset_index(drop=True)
        global_labels: list[str] = []
        for arm_id in arm_ids:
            global_labels.extend([str(arm_id)] * int(total_arm_counts[str(arm_id)]))
        if int(len(global_labels)) != int(selected.shape[0]):
            raise ValueError("Global arm assignment count mismatch.")
        selected["ARM"] = pd.Series(global_labels, dtype="string")
        arm_counts_by_stratum = {}
    selected = selected.drop(columns=["__A__"])

    lookup_cols = [c for c in pool.columns if c not in {"CANDIDATE_DRUG_ID"}]
    slices_by_arm: dict[str, pd.DataFrame] = {}
    for arm_id in arm_ids:
        drug_id = str(arm_to_drug_id[str(arm_id)])
        arm_slice = pool.loc[pool["CANDIDATE_DRUG_ID"].astype("string") == drug_id, :].copy()
        if arm_slice.empty:
            raise ValueError(f"Counterfactual pool missing candidate drug slice: {drug_id!r}.")
        slices_by_arm[str(arm_id)] = arm_slice.loc[:, lookup_cols].set_index("USUBJID", drop=False)

    merged = selected.loc[:, ["USUBJID", "ARM"]].copy().reset_index(drop=True)
    realized_rows: list[pd.DataFrame] = []
    for i in range(int(len(merged))):
        arm = str(merged.at[i, "ARM"])
        subj = str(merged.at[i, "USUBJID"])
        source = slices_by_arm[str(arm)]
        realized = source.loc[[subj], :].reset_index(drop=True).copy()
        realized["ARM"] = arm
        realized_rows.append(realized)
    realized_df = pd.concat(realized_rows, axis=0, ignore_index=True)
    realized_df = randomize_ltfu_within_arm_v1(
        realized_df,
        seed=int(seed),
        request_checksum=str(request_checksum),
    )

    participant_cols = sorted(
        set(request.analysis_covariates)
        | set(request.subgroup_variables)
        | set(request.stratification_variables)
        | {
            "SITE_ID",
            "REGION",
            "ADHERENCE_INDEX",
            "EARLY_RESCUE_RISK",
            "PROTOCOL_DEVIATION_FLAG",
            "ASSESSMENT_QUALITY",
        }
    )
    base_cols = ["USUBJID", "ARM"] + [c for c in participant_cols if c in realized_df.columns]
    participants = realized_df.loc[:, base_cols].copy()
    terminal_outcome_id = str(endpoint_outcome_ids.get(str(sp.terminal_endpoint_id), "") or "")
    terminal_source_t = f"ORACLE__{terminal_outcome_id}_T"
    terminal_source_e = f"ORACLE__{terminal_outcome_id}_E"
    if (
        not terminal_outcome_id
        or terminal_source_t not in realized_df.columns
        or terminal_source_e not in realized_df.columns
    ):
        raise ValueError("Randomized materialization requires oracle terminal-endpoint event and time columns.")

    if request.endpoint_id is None:
        endpoints = pd.DataFrame(
            {
                "USUBJID": participants["USUBJID"].astype("string"),
                "ARM": participants["ARM"].astype("string"),
                "FOLLOW_UP_DAYS": float(follow_up_days),
                "EVENT": np.zeros(int(len(participants)), dtype=int),
                "COMPETING_EVENT": np.zeros(int(len(participants)), dtype=int),
                "TIME": np.full(int(len(participants)), float(follow_up_days), dtype=float),
                "CENSOR_CAUSE": pd.Series(["endpoint_not_requested"] * int(len(participants)), dtype="string"),
            }
        )
    else:
        if request.treatment_discontinuation_strategy is None:
            raise ValueError("An efficacy endpoint requires treatment_discontinuation_strategy.")
        outcome_id = str(endpoint_outcome_ids.get(str(request.endpoint_id), ""))
        if not outcome_id:
            raise ValueError("Requested endpoint_id is not mapped in the hidden world manifest.")
        t_col = f"ORACLE__{outcome_id}_T"
        e_col = f"ORACLE__{outcome_id}_E"
        if t_col not in realized_df.columns or e_col not in realized_df.columns:
            raise ValueError("Requested oracle endpoint columns are missing from the counterfactual pool.")
        t = _coerce_time(realized_df[t_col])
        e = complete_binary_indicator_v1(realized_df[e_col])
        follow = float(follow_up_days)

        disc_t = (
            _coerce_time(realized_df["ORACLE__DISCONTINUATION_T"])
            if "ORACLE__DISCONTINUATION_T" in realized_df
            else None
        )
        disc_e = (
            complete_binary_indicator_v1(realized_df["ORACLE__DISCONTINUATION_E"])
            if "ORACLE__DISCONTINUATION_E" in realized_df
            else None
        )
        ltfu_t = _coerce_time(realized_df["ORACLE__LTFU_T"]) if "ORACLE__LTFU_T" in realized_df else None
        ltfu_e = (
            complete_binary_indicator_v1(realized_df["ORACLE__LTFU_E"]) if "ORACLE__LTFU_E" in realized_df else None
        )

        has_distinct_terminal_endpoint = bool(terminal_outcome_id and terminal_outcome_id != outcome_id)
        terminal_t = (
            _coerce_time(realized_df[f"ORACLE__{terminal_outcome_id}_T"]) if has_distinct_terminal_endpoint else None
        )
        terminal_e = (
            complete_binary_indicator_v1(realized_df[f"ORACLE__{terminal_outcome_id}_E"])
            if has_distinct_terminal_endpoint
            else None
        )

        event, competing_event, time, censor_cause = apply_endpoint_estimand_v1(
            strategy=str(request.treatment_discontinuation_strategy),
            endpoint_time=t,
            endpoint_event=e,
            follow_up_days=follow,
            discontinuation_time=disc_t,
            discontinuation_event=disc_e,
            ltfu_time=ltfu_t,
            ltfu_event=ltfu_e,
            terminal_time=terminal_t,
            terminal_event=terminal_e,
        )

        terminal_event_out = None
        terminal_time_out = None
        if terminal_outcome_id:
            ht_col = f"ORACLE__{terminal_outcome_id}_T"
            he_col = f"ORACLE__{terminal_outcome_id}_E"
            if ht_col in realized_df.columns and he_col in realized_df.columns:
                ht = _coerce_time(realized_df[ht_col])
                he = complete_binary_indicator_v1(realized_df[he_col])
                terminal_event_out = (he & (ht <= follow)).astype(int)
                terminal_time_out = np.minimum(ht, follow).astype(float)
        endpoints = pd.DataFrame(
            {
                "USUBJID": participants["USUBJID"].astype("string"),
                "ARM": participants["ARM"].astype("string"),
                "FOLLOW_UP_DAYS": float(follow_up_days),
                "TREATMENT_DISCONTINUATION_STRATEGY": str(request.treatment_discontinuation_strategy),
                "EVENT": event.astype(int),
                "COMPETING_EVENT": competing_event.astype(int),
                "TIME": time.astype(float),
                "CENSOR_CAUSE": pd.Series(censor_cause, dtype="string"),
            }
        )
        if terminal_event_out is not None and terminal_time_out is not None:
            endpoints["TERMINAL_EVENT"] = terminal_event_out.astype(int)
            endpoints["TERMINAL_TIME"] = terminal_time_out.astype(float)

    safety_cols = ["DISCONTINUATION_E", "DISCONTINUATION_T", "LTFU_E", "LTFU_T"]
    for definition in serious_event_definitions:
        safety_cols.extend(
            (
                definition.event_column,
                definition.time_column,
                definition.seriousness_column,
                definition.severity_column,
            )
        )
    safety = realized_df.loc[:, ["USUBJID", "ARM"]].copy()
    missing_oracle_columns = sorted(
        f"ORACLE__{column}" for column in safety_cols if f"ORACLE__{column}" not in realized_df.columns
    )
    if missing_oracle_columns:
        raise ValueError(
            f"Randomized safety materialization lacks required oracle columns: {missing_oracle_columns!r}."
        )
    for column in sorted(safety_cols):
        oracle_column = f"ORACLE__{column}"
        safety[column] = realized_df[oracle_column]
    follow = float(follow_up_days)
    ltfu_t = _coerce_time(safety["LTFU_T"])
    ltfu_e = complete_binary_indicator_v1(safety["LTFU_E"])
    terminal_t = _coerce_time(realized_df[terminal_source_t])
    terminal_e = complete_binary_indicator_v1(realized_df[terminal_source_e])

    disc_t = _coerce_time(safety["DISCONTINUATION_T"])
    disc_e = complete_binary_indicator_v1(safety["DISCONTINUATION_E"])
    occurred_disc, _, disc_time, _ = apply_endpoint_estimand_v1(
        strategy="treatment_policy",
        endpoint_time=disc_t,
        endpoint_event=disc_e,
        follow_up_days=follow,
        discontinuation_time=None,
        discontinuation_event=None,
        ltfu_time=ltfu_t,
        ltfu_event=ltfu_e,
        terminal_time=terminal_t,
        terminal_event=terminal_e,
    )
    safety["DISCONTINUATION_E"] = occurred_disc.astype(int)
    safety["DISCONTINUATION_T"] = disc_time.astype(float)

    ltfu_observed, _, ltfu_time, _ = apply_endpoint_estimand_v1(
        strategy="treatment_policy",
        endpoint_time=ltfu_t,
        endpoint_event=ltfu_e,
        follow_up_days=follow,
        discontinuation_time=None,
        discontinuation_event=None,
        ltfu_time=None,
        ltfu_event=None,
        terminal_time=terminal_t,
        terminal_event=terminal_e,
    )
    safety["LTFU_E"] = ltfu_observed.astype(int)
    safety["LTFU_T"] = ltfu_time.astype(float)

    terminal_observed, _, terminal_time, _ = apply_endpoint_estimand_v1(
        strategy="treatment_policy",
        endpoint_time=terminal_t,
        endpoint_event=terminal_e,
        follow_up_days=follow,
        discontinuation_time=None,
        discontinuation_event=None,
        ltfu_time=ltfu_t,
        ltfu_event=ltfu_e,
        terminal_time=None,
        terminal_event=None,
    )
    safety["TERMINAL_EVENT"] = terminal_observed.astype(int)
    safety["TERMINAL_TIME"] = terminal_time.astype(float)

    for definition in serious_event_definitions:
        t_col = definition.time_column
        e_col = definition.event_column
        event_t = _coerce_time(safety[t_col])
        event_e = complete_binary_indicator_v1(safety[e_col])
        occurred_event, _, observed_time, _ = apply_endpoint_estimand_v1(
            strategy="treatment_policy",
            endpoint_time=event_t,
            endpoint_event=event_e,
            follow_up_days=follow,
            discontinuation_time=None,
            discontinuation_event=None,
            ltfu_time=ltfu_t,
            ltfu_event=ltfu_e,
            terminal_time=terminal_t,
            terminal_event=terminal_e,
        )
        safety[e_col] = occurred_event.astype(int)
        safety[t_col] = observed_time.astype(float)

        unobserved_event = ~occurred_event.astype(bool)
        safety.loc[unobserved_event, definition.seriousness_column] = np.nan
        safety.loc[unobserved_event, definition.severity_column] = np.nan

    participants.to_parquet(out / "participants.parquet", index=False)
    endpoints.to_parquet(out / "endpoints.parquet", index=False)
    safety.to_parquet(out / "safety.parquet", index=False)
    arm_mapping = _arm_mapping_payload(
        request=request,
        control_drug_id=str(control),
        arm_ids=arm_ids,
        arm_to_drug_id=arm_to_drug_id,
        arm_weights=arm_weights,
    )
    write_json(out / "arm_mapping.json", arm_mapping)
    write_json(out / "request.json", request.model_dump(mode="json", exclude_none=True))

    site_mix = {}
    if "SITE_ID" in participants.columns:
        counts = participants["SITE_ID"].astype("string").fillna("NA").value_counts(dropna=False)
        total = float(counts.sum())
        site_mix = {str(k): float(v) / total for k, v in counts.to_dict().items()} if total > 0 else {}

    arm_counts_payload = {
        str(stratum): {str(arm_id): int(counts.get(str(arm_id), 0)) for arm_id in arm_ids}
        for stratum, counts in sorted(arm_counts_by_stratum.items())
    }
    realized_arm_counts = {
        str(arm): int(count)
        for arm, count in participants["ARM"].astype("string").value_counts().sort_index().to_dict().items()
    }

    audit = TrialMaterializationAuditV1(
        scenario_id=str(request.scenario_id),
        phase_id=str(request.phase_id),
        request_checksum=str(request_checksum),
        seed=int(seed),
        realized_sample_size=int(target_n),
        realized_follow_up_days=int(follow_up_days),
        feasibility_status="accepted",
        realized_arm_ids=tuple(str(arm) for arm in arm_ids),
        realized_arm_counts=realized_arm_counts,
        realized_stratification_variables=stratification_variables,
        realized_analysis_covariates=tuple(request.analysis_covariates),
        realized_subgroup_variables=tuple(request.subgroup_variables),
        realized_site_mix_summary=site_mix,
        payload={
            "candidate_drug_ids": [str(v) for v in request.candidate_drug_ids],
            "control_drug_id": control,
            "selection_objective": (None if request.selection_objective is None else str(request.selection_objective)),
            "enrollment_window_days": (
                None if request.enrollment_window_days is None else int(request.enrollment_window_days)
            ),
            "site_count_budget": (None if request.site_count_budget is None else int(request.site_count_budget)),
            "selected_sites": list(selected_sites),
            "loss_to_follow_up_assignment": "arm_conditional_random_permutation_v1",
            "selection_mode": "stratified" if stratification_variables else "global_ranked",
            "arm_to_drug_id": {str(k): str(v) for k, v in sorted(arm_to_drug_id.items())},
            "arm_weights": {str(k): float(v) for k, v in sorted(arm_weights.items())},
            "realized_strata": (
                {}
                if "__STRATUM__" not in selected.columns
                else {
                    str(k): int(v)
                    for k, v in selected["__STRATUM__"].astype("string").value_counts().sort_index().to_dict().items()
                }
            ),
            "arm_counts_by_stratum": arm_counts_payload,
        },
    )
    write_json(out / "execution_summary.json", audit.model_dump(mode="json", exclude_none=True))
    result = TrialMaterializationResultV1(
        audit=audit,
        trial_tables_dir=str(out),
        artifacts=(
            "participants.parquet",
            "endpoints.parquet",
            "safety.parquet",
            "arm_mapping.json",
            "request.json",
            "execution_summary.json",
        ),
    )
    missing_artifacts = tuple(name for name in result.artifacts if not (out / name).is_file())
    if missing_artifacts:
        raise FileNotFoundError(f"Materialization omitted declared artifacts: {missing_artifacts!r}.")
    return result
