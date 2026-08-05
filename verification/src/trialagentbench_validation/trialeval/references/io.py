"""Public ZIP table resolution helpers for TrialEval numeric reference replay."""

from __future__ import annotations

import json
import math
from io import BytesIO
from typing import cast
from weakref import WeakKeyDictionary
from zipfile import ZipFile

import pandas as pd
import pyarrow.parquet as pq

from trialagentbench_validation.contracts.scoring.route_reference_inputs import (
    RouteReferenceInputRecordV1,
)
from trialagentbench_validation.trialeval.public_archive import (
    public_member_exists_v1,
    resolve_public_member_v1,
)
from trialagentbench_validation.trialeval.reconstruction import (
    reconstruct_public_analysis_tables_v1,
)

_RECONSTRUCTED_TABLES: WeakKeyDictionary[
    ZipFile, dict[str, tuple[pd.DataFrame, pd.DataFrame]]
] = WeakKeyDictionary()
_PUBLIC_TABLES: WeakKeyDictionary[ZipFile, dict[str, pd.DataFrame]] = (
    WeakKeyDictionary()
)


def public_surface_shape_for_scoreable_refs_v1(
    reference_input: RouteReferenceInputRecordV1,
) -> str:
    """Render the role/table shape of public scoreable evidence references."""

    parts: list[str] = []
    for ref in reference_input.required_table_refs:
        public_path = public_rel_path_for_scoreable_ref_v1(ref.rel_path)
        if "/data/" in public_path:
            table_name = public_path.split("/data/", 1)[1]
        else:
            table_name = public_path.rsplit("/", 1)[-1]
        parts.append(f"{ref.semantic_role}:{table_name}")
    return "|".join(sorted(parts)) if parts else "no_table_refs"


def public_rel_path_for_scoreable_ref_v1(scoreable_rel_path: str) -> str:
    """Validate and return a participant ZIP member path."""

    if (
        not scoreable_rel_path.startswith("items/")
        or "/data/" not in scoreable_rel_path
    ):
        raise ValueError(
            f"Scoreable reference ref is not a participant item data path: {scoreable_rel_path}"
        )
    return scoreable_rel_path


def has_public_reconstruction_tables_v1(
    reference_input: RouteReferenceInputRecordV1,
) -> bool:
    """Return whether refs point to public reconstruction/raw surfaces."""

    rel_paths = tuple(ref.rel_path for ref in reference_input.required_table_refs)
    return any("/public_reconstruction/" in rel_path for rel_path in rel_paths) or any(
        "/raw/" in rel_path for rel_path in rel_paths
    )


def public_has_table_suffixes_v1(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
    suffixes: tuple[str, ...],
) -> bool:
    """Return whether all requested table suffixes exist on the public surface."""

    return all(
        any(
            public_member_exists_v1(public, path)
            for path in _candidate_table_paths(reference_input, suffix=suffix)
        )
        for suffix in suffixes
    )


def read_required_table_by_suffix_v1(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
    suffix: str,
) -> pd.DataFrame:
    """Read a public table by scoreable, reconstruction, or item-level suffix."""

    for member in _candidate_table_paths(reference_input, suffix=suffix):
        if public_member_exists_v1(public, member):
            return read_required_parquet_v1(public, member)
    if suffix in {
        "ADSL.parquet",
        "ADTTE.parquet",
    } and has_public_reconstruction_tables_v1(reference_input):
        tables = _RECONSTRUCTED_TABLES.setdefault(public, {})
        if reference_input.task_id not in tables:
            task = read_json_from_public_v1(
                public, f"items/{reference_input.task_id}/task.json"
            )
            tables.clear()
            tables[reference_input.task_id] = reconstruct_public_analysis_tables_v1(
                public=public,
                task_id=reference_input.task_id,
                paramcd=required_str_v1(task, "primary_paramcd"),
            )
        return tables[reference_input.task_id][0 if suffix == "ADSL.parquet" else 1]
    searched = ", ".join(_candidate_table_paths(reference_input, suffix=suffix))
    raise FileNotFoundError(
        f"Missing public parquet input for suffix {suffix!r}; searched: {searched}"
    )


def read_treatment_surface_table_v1(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
) -> pd.DataFrame:
    """Read the public table carrying treatment assignment."""

    try:
        return read_required_table_by_suffix_v1(
            public=public, reference_input=reference_input, suffix="ADSL.parquet"
        )
    except FileNotFoundError:
        return read_required_table_by_suffix_v1(
            public=public,
            reference_input=reference_input,
            suffix="subject_operational_flags.parquet",
        )


def read_covariate_surface_table_v1(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
) -> pd.DataFrame:
    """Read the public table carrying visible adjustment covariates."""

    try:
        return read_required_table_by_suffix_v1(
            public=public, reference_input=reference_input, suffix="ADSL.parquet"
        )
    except FileNotFoundError:
        return read_required_table_by_suffix_v1(
            public=public,
            reference_input=reference_input,
            suffix="subject_operational_flags.parquet",
        )


def read_required_parquet_v1(public: ZipFile, member: str) -> pd.DataFrame:
    """Read a required parquet member from the public ZIP."""

    archive_member = resolve_public_member_v1(public, member)
    tables = _PUBLIC_TABLES.setdefault(public, {})
    cached = tables.get(archive_member)
    if cached is not None:
        return cached
    try:
        payload = public.read(archive_member)
    except KeyError as exc:
        raise FileNotFoundError(f"Missing public parquet input: {member}") from exc
    task_prefix = archive_member.rsplit("/data/", 1)[0] + "/data/"
    if tables and not next(iter(tables)).startswith(task_prefix):
        tables.clear()
    frame = cast(pd.DataFrame, pq.read_table(BytesIO(payload)).to_pandas())
    tables[archive_member] = frame
    return frame


def read_json_from_public_v1(public: ZipFile, member: str) -> dict[str, object]:
    """Read a required JSON object from the public ZIP."""

    try:
        payload = json.loads(public.read(resolve_public_member_v1(public, member)))
    except (FileNotFoundError, KeyError) as exc:
        raise FileNotFoundError(f"Missing public JSON input: {member}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Public JSON input must contain an object: {member}")
    return payload


def required_str_v1(payload: dict[str, object], key: str) -> str:
    """Read a required non-empty string field from public metadata."""

    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Public task metadata requires non-empty string field {key!r}."
        )
    return value


def required_positive_float_v1(payload: dict[str, object], key: str) -> float:
    """Read a required positive finite numeric field from public metadata."""

    value = payload.get(key)
    if (
        not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(
            f"Public protocol metadata requires positive numeric field {key!r}."
        )
    return float(value)


def _optional_table_path(
    reference_input: RouteReferenceInputRecordV1, *, suffix: str
) -> str | None:
    matches = tuple(
        public_rel_path_for_scoreable_ref_v1(ref.rel_path)
        for ref in reference_input.required_table_refs
        if ref.rel_path.endswith(suffix)
    )
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one public scoreable input ending with {suffix!r}; found {len(matches)}."
        )
    return matches[0]


def _candidate_table_paths(
    reference_input: RouteReferenceInputRecordV1, *, suffix: str
) -> tuple[str, ...]:
    paths: list[str] = []
    optional = _optional_table_path(reference_input, suffix=suffix)
    if optional is not None:
        paths.append(optional)
    if suffix in {"ADSL.parquet", "ADTTE.parquet"}:
        paths.append(
            f"items/{reference_input.task_id}/data/public_reconstruction/{suffix}"
        )
    paths.append(f"items/{reference_input.task_id}/data/{suffix}")
    return tuple(dict.fromkeys(paths))
