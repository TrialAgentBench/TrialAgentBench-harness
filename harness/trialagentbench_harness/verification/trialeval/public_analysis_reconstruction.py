"""Reconstruct TrialEval analysis tables deterministically from public raw data."""

from __future__ import annotations

import json
import math
from collections.abc import Collection
from io import BytesIO
from typing import cast
from zipfile import ZipFile

import pandas as pd
import pyarrow.parquet as pq

from trialagentbench_harness.contracts.release.trialeval_integrity import (
    TrialEvalPublicIntegrityPolicyV1,
)
from trialagentbench_harness.trialeval.data_integrity import repair_exact_transport_row_duplication_v1


def _table(public: ZipFile, member: str) -> pd.DataFrame:
    try:
        return cast(pd.DataFrame, pq.read_table(BytesIO(public.read(member))).to_pandas())
    except KeyError as exc:
        raise FileNotFoundError(f"Missing public reconstruction table: {member}") from exc


def _json(public: ZipFile, member: str) -> dict[str, object]:
    try:
        payload = json.loads(public.read(member))
    except KeyError as exc:
        raise FileNotFoundError(f"Missing public reconstruction metadata: {member}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Public reconstruction metadata must contain an object: {member}")
    return payload


def _repair_declared_transport_duplicate_v1(
    frame: pd.DataFrame,
    *,
    policy: TrialEvalPublicIntegrityPolicyV1,
) -> pd.DataFrame:
    """Remove the one exact payload copy in each declared duplicate-key group."""

    repaired, _, _ = repair_exact_transport_row_duplication_v1(
        frame,
        key_fields=policy.compound_key_fields,
    )
    return repaired


def load_public_item_table_v1(
    *,
    public: ZipFile,
    task_id: str,
    relative_path: str,
) -> pd.DataFrame:
    """Load one participant table after applying any declared exact repair."""

    frame = _table(public, f"items/{task_id}/{relative_path}")
    policy_member = f"items/{task_id}/data_integrity_policy.json"
    if policy_member not in set(public.namelist()):
        return frame
    policy = TrialEvalPublicIntegrityPolicyV1.model_validate(_json(public, policy_member))
    if policy.task_id != task_id:
        raise ValueError("C5 integrity policy task_id disagrees with its item path.")
    if policy.affected_domain != relative_path:
        return frame
    return _repair_declared_transport_duplicate_v1(frame, policy=policy)


def _resolve_duplicates(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if "USUBJID" not in frame.columns or frame["USUBJID"].astype("string").isna().any():
        raise ValueError(f"{name} requires complete USUBJID values.")
    if not frame["USUBJID"].duplicated().any():
        return frame.copy()
    if "DATA_QUERY_FLAG" not in frame.columns:
        raise ValueError(f"{name} contains duplicate subjects without DATA_QUERY_FLAG.")
    resolved = frame.copy()
    resolved["_QUERY"] = resolved["DATA_QUERY_FLAG"].astype("string").str.upper().eq("Y").astype(int)
    return (
        resolved.sort_values(["USUBJID", "_QUERY"], kind="mergesort")
        .drop_duplicates("USUBJID", keep="first")
        .drop(columns="_QUERY")
    )


def map_declared_randomized_arms_v1(
    frame: pd.DataFrame,
    *,
    control_arm_id: str,
    treated_arm_id: str,
    control_arm_aliases: Collection[str],
    treated_arm_aliases: Collection[str],
) -> pd.Series:
    """Map randomized arms using only task IDs and protocol-declared aliases."""

    if not control_arm_id or not treated_arm_id or control_arm_id == treated_arm_id:
        raise ValueError("Primary control and treated arm identifiers must be distinct and non-empty.")
    control_aliases = {str(value) for value in control_arm_aliases} | {control_arm_id}
    treated_aliases = {str(value) for value in treated_arm_aliases} | {treated_arm_id}
    overlap = control_aliases & treated_aliases
    if overlap:
        raise ValueError(f"Protocol arm aliases must identify one contrast arm only: {sorted(overlap)!r}.")
    alias_map = {value: "control" for value in control_aliases} | {value: "treated" for value in treated_aliases}
    observed_by_column: dict[str, list[str]] = {}
    for column in ("ARMCD", "ARM", "TRT01A"):
        if column not in frame.columns:
            continue
        values = frame[column].astype("string")
        if values.isna().any():
            observed_by_column[column] = sorted(str(value) for value in values.dropna().unique())
            continue
        observed = {str(value) for value in values.unique()}
        observed_by_column[column] = sorted(observed)
        if not observed <= set(alias_map):
            continue
        mapped = cast(pd.Series, values.map(alias_map))
        if set(str(value) for value in mapped.unique()) != {"control", "treated"}:
            continue
        return mapped
    raise ValueError(
        "Public randomization does not contain both contrast arms under task IDs or protocol-declared aliases; "
        f"control_aliases={sorted(control_aliases)!r}, treated_aliases={sorted(treated_aliases)!r}, "
        f"observed={observed_by_column!r}."
    )


def declared_contrast_arm_aliases_v1(
    protocol: dict[str, object], *, control_arm_id: str, treated_arm_id: str
) -> tuple[frozenset[str], frozenset[str]]:
    arms = protocol.get("arms")
    if not isinstance(arms, list):
        raise ValueError("protocol_summary.json requires an arms list.")
    aliases_by_id: dict[str, frozenset[str]] = {}
    alias_owner: dict[str, str] = {}
    for row in arms:
        if not isinstance(row, dict):
            raise ValueError("protocol_summary.json arms must contain objects.")
        arm_id = row.get("arm_id")
        label = row.get("label")
        if not isinstance(arm_id, str) or not arm_id or not isinstance(label, str) or not label:
            raise ValueError("Each protocol arm requires non-empty string arm_id and label.")
        if arm_id in aliases_by_id:
            raise ValueError(f"protocol_summary.json contains duplicate arm_id={arm_id!r}.")
        aliases = frozenset({arm_id, label})
        for alias in aliases:
            owner = alias_owner.get(alias)
            if owner is not None and owner != arm_id:
                raise ValueError(
                    f"Protocol arm alias {alias!r} is ambiguous between arm_id={owner!r} and arm_id={arm_id!r}."
                )
            alias_owner[alias] = arm_id
        aliases_by_id[arm_id] = aliases
    missing = sorted({control_arm_id, treated_arm_id} - set(aliases_by_id))
    if missing:
        raise ValueError(f"Protocol arms omit task-declared contrast identifiers: {missing!r}.")
    return aliases_by_id[control_arm_id], aliases_by_id[treated_arm_id]


def _dates(series: pd.Series, *, column: str) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce")
    if values.isna().any():
        raise ValueError(f"Public reconstruction cannot parse datetime column {column!r}.")
    return values


def _baseline_dates(public: ZipFile, *, task_id: str) -> pd.DataFrame:
    visits = load_public_item_table_v1(
        public=public,
        task_id=task_id,
        relative_path="data/raw/visits.parquet",
    )
    if "USUBJID" not in visits.columns or "VISITDTC" not in visits.columns:
        raise ValueError("Baseline visit reconstruction requires USUBJID and VISITDTC.")
    if "VISITNUM" in visits.columns:
        baseline = visits.loc[pd.to_numeric(visits["VISITNUM"], errors="coerce").eq(0)].copy()
    elif "VISIT" in visits.columns:
        labels = visits["VISIT"].astype("string").str.strip().str.lower()
        baseline = visits.loc[labels.isin({"baseline", "screening/baseline", "day 0"})].copy()
    else:
        raise ValueError("Baseline visit reconstruction requires VISITNUM or VISIT.")
    if baseline.empty or baseline["USUBJID"].duplicated().any():
        raise ValueError("Baseline visit reconstruction requires exactly one baseline row per subject.")
    baseline["USUBJID"] = baseline["USUBJID"].astype("string")
    baseline["_ORIGIN"] = _dates(baseline["VISITDTC"], column="VISITDTC")
    return baseline.loc[:, ["USUBJID", "_ORIGIN"]]


def _events(adjudication: pd.DataFrame, *, primary_endpoint_term: str) -> pd.DataFrame:
    required = {"USUBJID", "CLINICAL_CERTAINTY", "SOURCE_CONSISTENCY", "EXCLUSIONARY_REVIEW_FINDING"}
    missing = sorted(required - set(adjudication.columns))
    if missing:
        raise ValueError(f"Endpoint adjudication is missing columns: {missing!r}.")
    endpoint_rows = adjudication
    if "ENDPOINT_TERM" in adjudication.columns:
        if not primary_endpoint_term.strip():
            raise ValueError("Endpoint-specific adjudication requires task.primary_endpoint_term.")
        endpoint_rows = adjudication.loc[
            adjudication["ENDPOINT_TERM"].astype("string").str.strip().str.casefold()
            == primary_endpoint_term.strip().casefold()
        ]
    eligible = endpoint_rows.loc[
        endpoint_rows["CLINICAL_CERTAINTY"].astype("string").str.lower().isin({"definite", "probable"})
        & endpoint_rows["SOURCE_CONSISTENCY"].astype("string").str.lower().eq("consistent")
        & endpoint_rows["EXCLUSIONARY_REVIEW_FINDING"].astype("string").str.lower().eq("none")
    ].copy()
    if {"EVENT_WINDOW_START_DY", "EVENT_WINDOW_END_DY"} <= set(eligible.columns):
        eligible["_START_DY"] = pd.to_numeric(eligible["EVENT_WINDOW_START_DY"], errors="coerce")
        eligible["_END_DY"] = pd.to_numeric(eligible["EVENT_WINDOW_END_DY"], errors="coerce")
        if eligible[["_START_DY", "_END_DY"]].isna().any().any():
            raise ValueError("Endpoint event-day windows must be numeric.")
        order = ["USUBJID", "_START_DY", "_END_DY"]
    elif {"EVENT_WINDOW_START_DTC", "EVENT_WINDOW_END_DTC"} <= set(eligible.columns):
        eligible["_START"] = _dates(eligible["EVENT_WINDOW_START_DTC"], column="EVENT_WINDOW_START_DTC")
        eligible["_END"] = _dates(eligible["EVENT_WINDOW_END_DTC"], column="EVENT_WINDOW_END_DTC")
        order = ["USUBJID", "_START", "_END"]
    else:
        raise ValueError("Endpoint adjudication requires event-window DY or DTC columns.")
    return eligible.sort_values(order, kind="mergesort").drop_duplicates("USUBJID", keep="first")


def reconstruct_public_analysis_tables_v1(
    *, public: ZipFile, task_id: str, paramcd: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ADSL and ADTTE reconstructed solely from participant ZIP members."""

    reconstruction_task = _json(public, f"items/{task_id}/reconstruction_task.json")
    task = _json(public, f"items/{task_id}/task.json")
    protocol = _json(public, f"items/{task_id}/protocol_summary.json")
    followup_horizon = protocol.get("followup_horizon_dy")
    if isinstance(followup_horizon, bool) or not isinstance(followup_horizon, int | float):
        raise ValueError("protocol_summary.json requires numeric followup_horizon_dy.")
    followup_horizon_dy = float(followup_horizon)
    if not math.isfinite(followup_horizon_dy) or followup_horizon_dy <= 0.0:
        raise ValueError("followup_horizon_dy must be finite and positive.")
    allowed = reconstruction_task.get("allowed_sources")
    if not isinstance(allowed, list):
        raise ValueError("reconstruction_task.json requires allowed_sources.")
    allowed_sources = {str(value) for value in allowed}
    required_sources = {
        "data/raw/randomization.parquet",
        "data/raw/disposition.parquet",
        "data/raw/endpoint_adjudication.parquet",
    }
    if not required_sources <= allowed_sources:
        raise ValueError("reconstruction_task.json omits a required reconstruction source.")
    prefix = f"items/{task_id}/data/raw"
    randomization_raw = load_public_item_table_v1(
        public=public,
        task_id=task_id,
        relative_path="data/raw/randomization.parquet",
    )
    disposition = _resolve_duplicates(
        load_public_item_table_v1(
            public=public,
            task_id=task_id,
            relative_path="data/raw/disposition.parquet",
        ),
        name="disposition",
    )
    adjudication = load_public_item_table_v1(
        public=public,
        task_id=task_id,
        relative_path="data/raw/endpoint_adjudication.parquet",
    )
    subject_randomization = "USUBJID" in randomization_raw.columns
    if subject_randomization:
        randomization = _resolve_duplicates(randomization_raw, name="randomization")
    else:
        if "SITEID" not in randomization_raw.columns or "SITEID" not in disposition.columns:
            raise ValueError("Site randomization requires SITEID linkage.")
        arm_column = "ARM" if "ARM" in randomization_raw.columns else "ARMCD"
        if randomization_raw.groupby("SITEID")[arm_column].nunique(dropna=False).max() > 1:
            raise ValueError("Site randomization contains conflicting assignments.")
        site = randomization_raw.drop_duplicates("SITEID", keep="first")
        randomization = disposition[["USUBJID", "SITEID"]].merge(site, on="SITEID", validate="many_to_one")
    randomization = randomization.copy()
    randomization["USUBJID"] = randomization["USUBJID"].astype("string")
    if protocol.get("design_family") == "stepped_wedge_cluster_rollout":
        required_timing = {"ARMCD", "INTERVENTION_START_DY"}
        missing_timing = sorted(required_timing - set(randomization.columns))
        if missing_timing:
            raise ValueError(f"Stepped-wedge randomization is missing rollout fields: {missing_timing!r}.")
        randomization["TRTA"] = randomization["ARMCD"].astype("string")
    else:
        control_arm_id = task.get("primary_control_arm_id")
        treated_arm_id = task.get("primary_treated_arm_id")
        if not isinstance(control_arm_id, str) or not isinstance(treated_arm_id, str):
            raise ValueError("task.json requires string primary control and treated arm identifiers.")
        control_aliases, treated_aliases = declared_contrast_arm_aliases_v1(
            protocol,
            control_arm_id=control_arm_id,
            treated_arm_id=treated_arm_id,
        )
        randomization["TRTA"] = map_declared_randomized_arms_v1(
            randomization,
            control_arm_id=control_arm_id,
            treated_arm_id=treated_arm_id,
            control_arm_aliases=control_aliases,
            treated_arm_aliases=treated_aliases,
        )
    if subject_randomization and "RFSTDTC" in randomization.columns:
        randomization["_ORIGIN"] = _dates(randomization["RFSTDTC"], column="RFSTDTC")
    needs_subject_origin = "LAST_CONTACT_DTC" in disposition.columns or {
        "EVENT_WINDOW_START_DTC",
        "EVENT_WINDOW_END_DTC",
    } <= set(adjudication.columns)
    if needs_subject_origin and "_ORIGIN" not in randomization.columns:
        if "data/raw/visits.parquet" not in allowed_sources:
            raise ValueError("reconstruction_task.json omits visits required to establish participant time zero.")
        randomization = randomization.merge(
            _baseline_dates(public, task_id=task_id), on="USUBJID", validate="one_to_one"
        )
    disposition = disposition.copy()
    disposition["USUBJID"] = disposition["USUBJID"].astype("string")
    merged = randomization.merge(disposition, on="USUBJID", validate="one_to_one", suffixes=("", "_DISP"))
    if "LAST_CONTACT_DY" in merged.columns:
        censor = pd.to_numeric(merged["LAST_CONTACT_DY"], errors="coerce")
    elif "LAST_CONTACT_DTC" in merged.columns:
        censor = (
            _dates(merged["LAST_CONTACT_DTC"], column="LAST_CONTACT_DTC") - merged["_ORIGIN"]
        ).dt.total_seconds() / 86400.0
    else:
        raise ValueError("Disposition requires LAST_CONTACT_DY or LAST_CONTACT_DTC.")
    if censor.isna().any() or (censor < 0).any():
        raise ValueError("Public reconstruction produced invalid censoring times.")
    censor = censor.clip(upper=followup_horizon_dy)
    primary_endpoint_term = task.get("primary_endpoint_term")
    if not isinstance(primary_endpoint_term, str) or not primary_endpoint_term.strip():
        raise ValueError("task.json requires non-empty primary_endpoint_term.")
    event = _events(adjudication, primary_endpoint_term=primary_endpoint_term)
    event["USUBJID"] = event["USUBJID"].astype("string")
    merged = merged.merge(event, on="USUBJID", how="left", validate="one_to_one")
    if "_START_DY" in merged.columns:
        candidate_event = merged["_START_DY"].notna() & merged["_END_DY"].notna()
        event_time = (merged["_START_DY"] + merged["_END_DY"]) / 2.0
    else:
        candidate_event = merged["_START"].notna() & merged["_END"].notna()
        midpoint = merged["_START"] + (merged["_END"] - merged["_START"]) / 2
        event_time = (midpoint - merged["_ORIGIN"]).dt.total_seconds() / 86400.0
    if (event_time.loc[candidate_event] < 0.0).any():
        raise ValueError("Public reconstruction found an endpoint event before participant time zero.")
    observed = candidate_event & event_time.le(censor)
    aval = censor.astype(float).copy()
    aval.loc[observed] = event_time.loc[observed]
    if aval.isna().any() or (aval < 0).any():
        raise ValueError("Public reconstruction produced invalid analysis times.")
    adsl = randomization.loc[:, [column for column in randomization.columns if not str(column).startswith("_")]].copy()
    for name in ("subjects.parquet", "baseline_characteristics.parquet"):
        relative_path = f"data/raw/{name}"
        if relative_path not in allowed_sources:
            continue
        member = f"{prefix}/{name}"
        if member not in public.namelist():
            continue
        context = load_public_item_table_v1(
            public=public,
            task_id=task_id,
            relative_path=relative_path,
        )
        context["USUBJID"] = context["USUBJID"].astype("string")
        if context["USUBJID"].duplicated().any():
            raise ValueError(f"Public subject context contains duplicate USUBJID: {name}.")
        keep = [column for column in context.columns if column == "USUBJID" or column not in adsl.columns]
        adsl = adsl.merge(context.loc[:, keep], on="USUBJID", how="left", validate="one_to_one")
    adtte = pd.DataFrame(
        {
            "USUBJID": merged["USUBJID"].astype(str),
            "PARAMCD": str(paramcd),
            "AVAL": aval.astype(float),
            "CNSR": (~observed).astype(int),
        }
    )
    return adsl, adtte


def load_public_analysis_tables_v1(
    *,
    public: ZipFile,
    task_id: str,
    paramcd: str,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Load or reconstruct analysis tables and report exact participant sources."""

    adsl_member = f"items/{task_id}/data/ADSL.parquet"
    adtte_member = f"items/{task_id}/data/ADTTE.parquet"
    members = set(public.namelist())
    present = (adsl_member in members, adtte_member in members)
    if any(present) and not all(present):
        raise ValueError("Analysis-ready participant surfaces require both ADSL and ADTTE.")
    if all(present):
        return _table(public, adsl_member), _table(public, adtte_member), (adsl_member, adtte_member)

    reconstruction_task = _json(public, f"items/{task_id}/reconstruction_task.json")
    allowed = reconstruction_task.get("allowed_sources")
    if not isinstance(allowed, list) or not all(isinstance(value, str) and value for value in allowed):
        raise ValueError("reconstruction_task.json requires non-empty string allowed_sources.")
    sources = tuple(f"items/{task_id}/{value}" for value in allowed)
    missing = tuple(source for source in sources if source not in members)
    if missing:
        raise FileNotFoundError(f"Declared public reconstruction sources are missing: {missing!r}.")
    adsl, adtte = reconstruct_public_analysis_tables_v1(public=public, task_id=task_id, paramcd=paramcd)
    return adsl, adtte, sources


__all__ = [
    "declared_contrast_arm_aliases_v1",
    "load_public_analysis_tables_v1",
    "load_public_item_table_v1",
    "map_declared_randomized_arms_v1",
    "reconstruct_public_analysis_tables_v1",
]
