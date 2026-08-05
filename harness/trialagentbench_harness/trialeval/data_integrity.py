"""Canonical public repair for declared TrialEval data-integrity defects."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import shutil
import struct
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

CanonicalScalarV1: TypeAlias = list[str | bool | None]

EXACT_TRANSPORT_DUPLICATE_MULTIPLICITY_V1 = 2
DATA_INTEGRITY_POLICY_FILENAME_V1 = "data_integrity_policy.json"
DATA_INTEGRITY_UTILITY_RELATIVE_PATH_V1 = Path("interface") / "data_integrity.py"
CANONICAL_COMPOUND_ROW_ENCODING_V1 = "canonical_compound_row_key_v1"

__all__ = [
    "DATA_INTEGRITY_POLICY_FILENAME_V1",
    "DATA_INTEGRITY_UTILITY_RELATIVE_PATH_V1",
    "EXACT_TRANSPORT_DUPLICATE_MULTIPLICITY_V1",
    "canonical_domain_content_sha256_v1",
    "canonical_json_array_bytes_v1",
    "canonical_rows_v1",
    "repair_exact_transport_row_duplication_v1",
    "stage_data_integrity_utility_v1",
    "validate_declared_data_integrity_v1",
]


@dataclass(frozen=True)
class DataIntegrityRepairRecordV1:
    """Submission fields produced by the declared exact repair."""

    condition_id: str
    affected_domain: str
    compound_key_fields: tuple[str, ...]
    observed_duplicate_group_count: int
    observed_extra_row_count: int
    repair_action: str
    repair_status: str
    post_repair_data_checksum: str
    analysis_input_data_checksum: str


def canonical_json_array_bytes_v1(value: object) -> bytes:
    """Encode one canonical JSON array value as UTF-8 bytes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _is_missing(value: object) -> bool:
    missing = pd.isna(cast(Any, value))
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    raise TypeError("canonical scalars cannot contain array-valued cells")


def _canonical_scalar(value: object, dtype: object) -> CanonicalScalarV1:
    typed_dtype = cast(Any, dtype)
    typed_value = cast(Any, value)
    if _is_missing(value):
        return ["null", None]
    if isinstance(dtype, pd.CategoricalDtype):
        return _canonical_scalar(value, pd.Series([value]).dtype)
    if ptypes.is_bool_dtype(typed_dtype) or isinstance(value, (bool, np.bool_)):
        return ["boolean", bool(value)]
    if ptypes.is_unsigned_integer_dtype(typed_dtype):
        return [str(dtype).lower(), str(int(typed_value))]
    if ptypes.is_integer_dtype(typed_dtype) or isinstance(value, (int, np.integer)):
        tag = str(dtype).lower() if ptypes.is_integer_dtype(typed_dtype) else "int64"
        return [tag, str(int(typed_value))]
    if ptypes.is_float_dtype(typed_dtype) or isinstance(value, (float, np.floating)):
        number = float(typed_value)
        if not math.isfinite(number):
            raise ValueError("canonical scalars require finite floating-point values")
        if str(dtype).lower() == "float32" or isinstance(value, np.float32):
            return ["float32", struct.pack(">f", number).hex()]
        return ["float64", struct.pack(">d", number).hex()]
    if ptypes.is_datetime64_any_dtype(typed_dtype) or isinstance(value, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(typed_value)
        tag = "timestamp[ns]" if timestamp.tzinfo is None else f"timestamp[ns,{timestamp.tzinfo}]"
        return [tag, str(int(timestamp.value))]
    if isinstance(value, date):
        return ["date32", str((value - date(1970, 1, 1)).days)]
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("canonical scalars require finite decimal values")
        exponent = int(value.as_tuple().exponent)
        scale = max(0, -exponent)
        unscaled = int(value.scaleb(scale))
        precision = len(value.as_tuple().digits)
        return [f"decimal:{precision}:{scale}", str(unscaled)]
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = base64.urlsafe_b64encode(bytes(value)).decode("ascii").rstrip("=")
        return ["binary", encoded]
    if isinstance(value, str):
        value.encode("utf-8", errors="strict")
        return ["string", value]
    raise TypeError(f"unsupported canonical scalar type: {type(value).__name__}")


def canonical_rows_v1(frame: pd.DataFrame, fields: tuple[str, ...]) -> tuple[bytes, ...]:
    """Encode selected typed columns in one tabular pass."""

    _validate_columns(frame, fields)
    dtypes = tuple(frame[field].dtype for field in fields)
    return tuple(
        canonical_json_array_bytes_v1(
            tuple(_canonical_scalar(value, dtype) for value, dtype in zip(values, dtypes, strict=True))
        )
        for values in frame.loc[:, list(fields)].itertuples(index=False, name=None)
    )


def _canonical_records(
    frame: pd.DataFrame,
    *,
    key_fields: tuple[str, ...],
) -> tuple[tuple[bytes, bytes], ...]:
    columns = tuple(str(column) for column in frame.columns)
    positions = {column: index for index, column in enumerate(columns)}
    key_positions = tuple(positions[field] for field in key_fields)
    dtypes = tuple(frame[column].dtype for column in columns)
    records: list[tuple[bytes, bytes]] = []
    for values in frame.itertuples(index=False, name=None):
        encoded = tuple(_canonical_scalar(value, dtype) for value, dtype in zip(values, dtypes, strict=True))
        key = canonical_json_array_bytes_v1(tuple(encoded[position] for position in key_positions))
        records.append((key, canonical_json_array_bytes_v1(encoded)))
    return tuple(records)


def _validate_columns(frame: pd.DataFrame, key_fields: tuple[str, ...]) -> None:
    if frame.empty:
        raise ValueError("the declared integrity domain must contain at least one row")
    if len(set(str(column) for column in frame.columns)) != len(frame.columns):
        raise ValueError("the declared integrity domain requires unique column names")
    missing = tuple(field for field in key_fields if field not in frame.columns)
    if missing:
        raise ValueError(f"the declared integrity domain is missing compound-key fields: {missing!r}")
    for field in key_fields:
        if frame[field].map(_is_missing).any():
            raise ValueError(f"the declared compound-key field contains missing values: {field}")


def canonical_domain_content_sha256_v1(frame: pd.DataFrame, *, key_fields: tuple[str, ...]) -> str:
    """Return the canonical cross-format checksum of a typed domain."""

    _validate_columns(frame, key_fields)
    records = sorted(_canonical_records(frame, key_fields=key_fields))
    content = b"\n".join(payload for _key, payload in records)
    return hashlib.sha256(content).hexdigest()


def repair_exact_transport_row_duplication_v1(
    frame: pd.DataFrame,
    *,
    key_fields: tuple[str, ...],
) -> tuple[pd.DataFrame, int, int]:
    """Remove exact duplicate copies and fail on ambiguous same-key states."""

    _validate_columns(frame, key_fields)
    groups: dict[bytes, list[int]] = {}
    records = _canonical_records(frame, key_fields=key_fields)
    for index, (key, _payload) in enumerate(records):
        groups.setdefault(key, []).append(index)
    drop_indexes: list[int] = []
    duplicate_groups = 0
    for indexes in groups.values():
        if len(indexes) == 1:
            continue
        if len(indexes) != EXACT_TRANSPORT_DUPLICATE_MULTIPLICITY_V1:
            raise ValueError("unexpected_data_integrity_state: compound-key multiplicity must be one or two")
        payloads = {records[index][1] for index in indexes}
        if len(payloads) != 1:
            raise ValueError("unexpected_data_integrity_state: same-key payloads are not identical")
        duplicate_groups += 1
        drop_indexes.append(indexes[1])
    if duplicate_groups == 0:
        raise ValueError("unexpected_data_integrity_state: no exact transport duplicate was detected")
    repaired = frame.drop(index=drop_indexes).reset_index(drop=True)
    return repaired, duplicate_groups, len(drop_indexes)


def _load_policy_v1(path: Path) -> tuple[str, tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("data_integrity_policy.json must contain one object")
    required_values = {
        "schema_id": "trialagentbench.trialeval.c5_integrity_policy/v1",
        "condition_id": "exact_transport_row_duplication_v1",
        "repair_contract_id": "exact_transport_row_duplication_repair_v1",
        "repair_action": "remove_one_exact_duplicate_copy",
        "canonical_typed_scalar_encoding_id": "canonical_typed_scalar_v1",
        "canonical_compound_row_key_encoding_id": CANONICAL_COMPOUND_ROW_ENCODING_V1,
        "canonical_typed_row_payload_encoding_id": "canonical_typed_row_payload_v1",
        "canonical_domain_content_checksum_id": "canonical_domain_content_sha256_v1",
    }
    for field, expected in required_values.items():
        if payload.get(field) != expected:
            raise ValueError(f"data-integrity policy has unsupported {field}: {payload.get(field)!r}")
    affected_domain = payload.get("affected_domain")
    key_fields = payload.get("compound_key_fields")
    if not isinstance(affected_domain, str) or not affected_domain:
        raise ValueError("data-integrity policy requires affected_domain")
    relative_domain = PurePosixPath(affected_domain)
    if (
        relative_domain.is_absolute()
        or ".." in relative_domain.parts
        or len(relative_domain.parts) != 3
        or relative_domain.parts[:2] != ("data", "raw")
        or relative_domain.suffix != ".parquet"
    ):
        raise ValueError("data-integrity affected_domain must identify one Parquet file under data/raw/")
    if (
        not isinstance(key_fields, list)
        or not key_fields
        or not all(isinstance(field, str) and field for field in key_fields)
        or len(set(key_fields)) != len(key_fields)
    ):
        raise ValueError("data-integrity policy requires unique non-empty compound_key_fields")
    return affected_domain, tuple(key_fields)


def validate_declared_data_integrity_v1(
    *,
    analysis_input_path: str,
    root: str | Path = ".",
) -> dict[str, object]:
    """Validate a repaired analysis input and return its canonical record."""

    item_root = Path(root)
    policy_path = item_root / DATA_INTEGRITY_POLICY_FILENAME_V1
    if not policy_path.is_file():
        raise FileNotFoundError(f"declared data-integrity policy is missing: {policy_path}")
    affected_domain, key_fields = _load_policy_v1(policy_path)
    source = item_root.joinpath(*PurePosixPath(affected_domain).parts)
    if not source.is_file():
        raise FileNotFoundError(f"declared data-integrity domain is missing: {source}")
    frame = pd.read_parquet(source)
    repaired, duplicate_groups, extra_rows = repair_exact_transport_row_duplication_v1(
        frame,
        key_fields=key_fields,
    )
    relative_input = PurePosixPath(analysis_input_path)
    if (
        relative_input.is_absolute()
        or ".." in relative_input.parts
        or not relative_input.parts
        or relative_input.parts[0] != "scratch"
        or relative_input.suffix != ".parquet"
    ):
        raise ValueError("analysis_input_path must identify one Parquet file under scratch/")
    candidate_path = item_root.joinpath(*relative_input.parts)
    if not candidate_path.is_file():
        raise FileNotFoundError(f"repaired analysis input is missing: {candidate_path}")
    expected_checksum = canonical_domain_content_sha256_v1(repaired, key_fields=key_fields)
    candidate = pd.read_parquet(candidate_path)
    candidate_checksum = canonical_domain_content_sha256_v1(candidate, key_fields=key_fields)
    if candidate_checksum != expected_checksum:
        raise ValueError("unexpected_data_integrity_state: analysis input is not the exact declared repair")
    record = DataIntegrityRepairRecordV1(
        condition_id="exact_transport_row_duplication_v1",
        affected_domain=affected_domain,
        compound_key_fields=key_fields,
        observed_duplicate_group_count=duplicate_groups,
        observed_extra_row_count=extra_rows,
        repair_action="remove_one_exact_duplicate_copy",
        repair_status="repaired",
        post_repair_data_checksum=expected_checksum,
        analysis_input_data_checksum=candidate_checksum,
    )
    return {
        "analysis_input_path": relative_input.as_posix(),
        "submission_record": asdict(record),
    }


def stage_data_integrity_utility_v1(root: Path) -> Path | None:
    """Stage the public repair implementation when an item declares its use."""

    item_root = Path(root)
    if not (item_root / DATA_INTEGRITY_POLICY_FILENAME_V1).is_file():
        return None
    source = Path(__file__).resolve(strict=True)
    target = item_root / DATA_INTEGRITY_UTILITY_RELATIVE_PATH_V1
    if target.is_symlink() or target.parent.is_symlink():
        raise ValueError("TrialEval data-integrity utility path must not be a symlink")
    expected = source.read_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != expected:
            raise ValueError("existing TrialEval data-integrity utility differs from the public implementation")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as source_handle, target.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle)
    return target
