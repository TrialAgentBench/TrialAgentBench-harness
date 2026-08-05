"""Independent recovery of the TrialEval C5 data-integrity condition."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import struct
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.process_pool import (
    single_threaded_numerical_process_pool,
)

_POLICY_NAME = "data_integrity_policy.json"
_REFERENCE_NAME = "grader/domains/data_integrity_reference.jsonl"
_CONTEXT_PANELS_NAME = "grader/domains/context_panels.json"


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class C5PublicIntegrityPolicyV1(_ContractModel):
    """Participant-visible definition of the exact C5 repair."""

    schema_id: Literal["trialagentbench.trialeval.c5_integrity_policy/v1"]
    task_id: str = Field(min_length=1)
    condition_id: Literal["exact_transport_row_duplication_v1"]
    affected_domain: str = Field(min_length=1)
    compound_key_fields: tuple[str, ...] = Field(min_length=1)
    legitimate_repeat_semantics: str = Field(min_length=1)
    repair_contract_id: Literal["exact_transport_row_duplication_repair_v1"]
    repair_action: Literal["remove_one_exact_duplicate_copy"]
    canonical_typed_scalar_encoding_id: Literal["canonical_typed_scalar_v1"]
    canonical_compound_row_key_encoding_id: Literal["canonical_compound_row_key_v1"]
    canonical_typed_row_payload_encoding_id: Literal["canonical_typed_row_payload_v1"]
    canonical_domain_content_checksum_id: Literal["canonical_domain_content_sha256_v1"]
    selected_duplicate_keys_visible: Literal[False]
    expected_duplicate_count_visible: Literal[False]
    clean_parent_checksum_visible: Literal[False]


class C5IntegrityReferenceV1(_ContractModel):
    """Verification reference opened after an independent repair is projected."""

    schema_id: Literal["trialagentbench.trialeval.c5_integrity_reference/v1"]
    task_id: str = Field(min_length=1)
    clean_context_parent_task_id: str = Field(min_length=1)
    condition_id: Literal["exact_transport_row_duplication_v1"]
    affected_domain: str = Field(min_length=1)
    compound_key_fields: tuple[str, ...] = Field(min_length=1)
    defect_seed: int = Field(ge=0)
    clean_row_count: int = Field(ge=1)
    selected_duplicate_count: int = Field(ge=1)
    selected_compound_keys: tuple[str, ...] = Field(min_length=1)
    observed_duplicate_group_count: int = Field(ge=1)
    observed_extra_row_count: int = Field(ge=1)
    repair_action: Literal["remove_one_exact_duplicate_copy"]
    repair_status: Literal["repaired"]
    clean_domain_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mutated_domain_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    post_repair_data_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class C5IntegrityItemRecordV1(_ContractModel):
    """Independent repair and equality result for one C5 item."""

    schema_id: Literal["trialagentbench.validation.c5_integrity_item/v1"] = (
        "trialagentbench.validation.c5_integrity_item/v1"
    )
    task_id: str
    clean_context_parent_task_id: str
    condition_id: Literal["exact_transport_row_duplication_v1"]
    affected_domain: str
    compound_key_fields: tuple[str, ...]
    observed_duplicate_group_count: int = Field(ge=0)
    observed_extra_row_count: int = Field(ge=0)
    mutated_domain_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repaired_domain_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clean_parent_domain_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    unaffected_domain_count: int = Field(ge=0)
    independently_repaired: bool
    repaired_content_equals_c4: bool
    verification_reference_concordant: bool
    status: Literal["repaired", "mismatch", "unsupported"]
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _coherent_status(self) -> C5IntegrityItemRecordV1:
        passed = (
            self.independently_repaired
            and self.repaired_content_equals_c4
            and self.verification_reference_concordant
        )
        if (self.status == "repaired") != passed:
            raise ValueError(
                "C5 item status must agree with its repair and concordance checks"
            )
        if self.status == "repaired" and self.failure_reason is not None:
            raise ValueError("a repaired C5 item cannot carry a failure reason")
        if self.status != "repaired" and not self.failure_reason:
            raise ValueError("a failed C5 item requires a failure reason")
        return self


class C5IntegrityRecoveryReportV1(_ContractModel):
    """Full-census independent C5 recovery receipt."""

    schema_id: Literal["trialagentbench.validation.c5_integrity_recovery/v1"] = (
        "trialagentbench.validation.c5_integrity_recovery/v1"
    )
    expected_item_count: int = Field(ge=1)
    required_item_count: int = Field(ge=0)
    repaired_item_count: int = Field(ge=0)
    mismatched_item_count: int = Field(ge=0)
    unsupported_item_count: int = Field(ge=0)
    records: tuple[C5IntegrityItemRecordV1, ...]
    participant_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["pass", "fail"]
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_census(self) -> C5IntegrityRecoveryReportV1:
        records = tuple(sorted(self.records, key=lambda row: row.task_id))
        object.__setattr__(self, "records", records)
        if len({row.task_id for row in records}) != len(records):
            raise ValueError("C5 recovery records must be unique by task_id")
        counts = {
            "required_item_count": len(records),
            "repaired_item_count": sum(row.status == "repaired" for row in records),
            "mismatched_item_count": sum(row.status == "mismatch" for row in records),
            "unsupported_item_count": sum(
                row.status == "unsupported" for row in records
            ),
        }
        for name, observed in counts.items():
            if getattr(self, name) != observed:
                raise ValueError(f"{name} disagrees with the item records")
        passed = (
            self.required_item_count == self.expected_item_count
            and self.repaired_item_count == self.expected_item_count
            and self.mismatched_item_count == 0
            and self.unsupported_item_count == 0
        )
        if (self.status == "pass") != passed:
            raise ValueError("C5 report status must agree with the full-census result")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        object.__setattr__(self, "checksum", hashlib.sha256(encoded).hexdigest())
        return self


@dataclass(frozen=True)
class _C5ProjectionV1:
    task_id: str
    parent_task_id: str
    duplicate_groups: int
    extra_rows: int
    mutated_checksum: str
    repaired_checksum: str
    clean_checksum: str
    unaffected_count: int


@dataclass(frozen=True)
class _C5ProjectionFailureV1:
    task_id: str
    status: Literal["mismatch", "unsupported"]
    reason: str


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_array_bytes(value: object) -> bytes:
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
    raise TypeError("canonical scalar cells must not be array-valued")


def _canonical_scalar(value: object, dtype: object) -> list[str | bool | None]:
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
            raise ValueError("canonical floating-point values must be finite")
        if str(dtype).lower() == "float32" or isinstance(value, np.float32):
            return ["float32", struct.pack(">f", number).hex()]
        return ["float64", struct.pack(">d", number).hex()]
    if ptypes.is_datetime64_any_dtype(typed_dtype) or isinstance(
        value, (pd.Timestamp, datetime)
    ):
        timestamp = pd.Timestamp(typed_value)
        tag = (
            "timestamp[ns]"
            if timestamp.tzinfo is None
            else f"timestamp[ns,{timestamp.tzinfo}]"
        )
        return [tag, str(int(timestamp.value))]
    if isinstance(value, date):
        return ["date32", str((value - date(1970, 1, 1)).days)]
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("canonical decimal values must be finite")
        scale = max(0, -int(value.as_tuple().exponent))
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


def _canonical_records(
    frame: pd.DataFrame,
    *,
    key_fields: tuple[str, ...],
) -> tuple[tuple[bytes, bytes], ...]:
    """Encode each typed row and compound key in one tabular pass."""

    columns = tuple(str(column) for column in frame.columns)
    positions = {column: index for index, column in enumerate(columns)}
    key_positions = tuple(positions[field] for field in key_fields)
    dtypes = tuple(frame[column].dtype for column in columns)
    records: list[tuple[bytes, bytes]] = []
    for values in frame.itertuples(index=False, name=None):
        encoded = tuple(
            _canonical_scalar(value, dtype)
            for value, dtype in zip(values, dtypes, strict=True)
        )
        key = _json_array_bytes([encoded[position] for position in key_positions])
        records.append((key, _json_array_bytes(encoded)))
    return tuple(records)


def _validate_frame(frame: pd.DataFrame, key_fields: tuple[str, ...]) -> None:
    if frame.empty:
        raise ValueError("the declared integrity domain is empty")
    columns = tuple(str(column) for column in frame.columns)
    if len(set(columns)) != len(columns):
        raise ValueError("the declared integrity domain has duplicate column names")
    missing = tuple(field for field in key_fields if field not in columns)
    if missing:
        raise ValueError(f"the declared compound key is absent: {missing!r}")
    for field in key_fields:
        if frame[field].map(_is_missing).any():
            raise ValueError(
                f"the declared compound key contains missing values: {field}"
            )


def canonical_domain_content_sha256_v1(
    frame: pd.DataFrame, *, key_fields: tuple[str, ...]
) -> str:
    """Compute the public typed-content checksum without using benchmark code."""

    _validate_frame(frame, key_fields)
    records = sorted(_canonical_records(frame, key_fields=key_fields))
    return hashlib.sha256(b"\n".join(payload for _key, payload in records)).hexdigest()


def repair_exact_transport_row_duplication_v1(
    frame: pd.DataFrame,
    *,
    key_fields: tuple[str, ...],
) -> tuple[pd.DataFrame, int, int]:
    """Remove one identical copy for each duplicated compound key."""

    _validate_frame(frame, key_fields)
    groups: dict[bytes, list[int]] = defaultdict(list)
    records = _canonical_records(frame, key_fields=key_fields)
    for index, (key, _payload) in enumerate(records):
        groups[key].append(index)
    drop_indexes: list[int] = []
    for indexes in groups.values():
        if len(indexes) == 1:
            continue
        if len(indexes) != 2:
            raise ValueError("compound-key multiplicity is not one or two")
        if len({records[index][1] for index in indexes}) != 1:
            raise ValueError("same-key rows do not have identical typed payloads")
        drop_indexes.append(indexes[1])
    if not drop_indexes:
        raise ValueError("no exact transport duplicate was detected")
    return (
        frame.drop(index=drop_indexes).reset_index(drop=True),
        len(drop_indexes),
        len(drop_indexes),
    )


def _member_with_suffix(archive: ZipFile, suffix: str) -> str:
    normalized = suffix.lstrip("/")
    matches = [
        name
        for name in archive.namelist()
        if not name.endswith("/")
        and (name == normalized or name.endswith(f"/{normalized}"))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"archive must contain exactly one {normalized!r}; found {len(matches)}"
        )
    return matches[0]


def _item_prefix(policy_member: str) -> str:
    suffix = f"/{_POLICY_NAME}"
    if not policy_member.endswith(suffix):
        raise ValueError("integrity policy is not located in an item root")
    return policy_member[: -len(suffix)]


def _read_parquet_member(archive: ZipFile, member: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(archive.read(member)))


def _raw_domain_members(archive: ZipFile, item_prefix: str) -> dict[str, str]:
    prefix = f"{item_prefix}/data/raw/"
    members = {
        f"data/raw/{PurePosixPath(name).name}": name
        for name in archive.namelist()
        if name.startswith(prefix) and name.endswith(".parquet")
    }
    if not members:
        raise ValueError(f"item {item_prefix!r} contains no raw Parquet domains")
    return members


def _context_pairs(verification: ZipFile) -> dict[str, str]:
    payload = json.loads(
        verification.read(_member_with_suffix(verification, _CONTEXT_PANELS_NAME))
    )
    pairs: dict[str, str] = {}
    for panel in payload.get("panels", []):
        by_context = {row["context_tier"]: row["task_id"] for row in panel["tasks"]}
        if "C5" in by_context:
            if "C4" not in by_context:
                raise ValueError("a C5 context panel is missing its C4 sibling")
            pairs[str(by_context["C5"])] = str(by_context["C4"])
    return pairs


def _references(verification: ZipFile) -> dict[str, C5IntegrityReferenceV1]:
    member = _member_with_suffix(verification, _REFERENCE_NAME)
    references: dict[str, C5IntegrityReferenceV1] = {}
    for line in verification.read(member).decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("domain") != "data_integrity_reference":
            raise ValueError("the C5 reference file contains a row from another domain")
        reference = C5IntegrityReferenceV1.model_validate(row["payload"])
        if row.get("task_id") != reference.task_id:
            raise ValueError("a C5 reference row has inconsistent task identifiers")
        if reference.task_id in references:
            raise ValueError(f"duplicate C5 integrity reference: {reference.task_id}")
        references[reference.task_id] = reference
    return references


def _project_c5_item(
    request: tuple[str, str, str, str, str],
) -> _C5ProjectionV1 | _C5ProjectionFailureV1:
    """Recover one C5 task from participant-visible bytes."""

    participant_path, task_id, prefix, parent_task_id, policy_json = request
    policy = C5PublicIntegrityPolicyV1.model_validate_json(policy_json)
    try:
        with ZipFile(participant_path) as participant:
            parent_task_member = _member_with_suffix(
                participant,
                f"items/{parent_task_id}/task.json",
            )
            parent_prefix = parent_task_member.rsplit("/", 1)[0]
            c5_domains = _raw_domain_members(participant, prefix)
            c4_domains = _raw_domain_members(participant, parent_prefix)
            if set(c5_domains) != set(c4_domains):
                raise ValueError("C4 and C5 raw-domain inventories differ")
            if policy.affected_domain not in c5_domains:
                raise ValueError("the declared affected domain is absent")
            mutated = _read_parquet_member(
                participant,
                c5_domains[policy.affected_domain],
            )
            repaired, duplicate_groups, extra_rows = (
                repair_exact_transport_row_duplication_v1(
                    mutated,
                    key_fields=policy.compound_key_fields,
                )
            )
            clean = _read_parquet_member(
                participant,
                c4_domains[policy.affected_domain],
            )
            unaffected_count = 0
            for domain in sorted(set(c5_domains) - {policy.affected_domain}):
                c5_frame = _read_parquet_member(participant, c5_domains[domain])
                c4_frame = _read_parquet_member(participant, c4_domains[domain])
                pd.testing.assert_frame_equal(c5_frame, c4_frame, check_exact=True)
                unaffected_count += 1
        return _C5ProjectionV1(
            task_id=task_id,
            parent_task_id=parent_task_id,
            duplicate_groups=duplicate_groups,
            extra_rows=extra_rows,
            mutated_checksum=canonical_domain_content_sha256_v1(
                mutated,
                key_fields=policy.compound_key_fields,
            ),
            repaired_checksum=canonical_domain_content_sha256_v1(
                repaired,
                key_fields=policy.compound_key_fields,
            ),
            clean_checksum=canonical_domain_content_sha256_v1(
                clean,
                key_fields=policy.compound_key_fields,
            ),
            unaffected_count=unaffected_count,
        )
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        return _C5ProjectionFailureV1(
            task_id=task_id,
            status="unsupported",
            reason=str(error),
        )


def recover_c5_integrity(
    *,
    participant_zip: Path,
    verification_zip: Path,
    expected_item_count: int = 100,
    workers: int = 1,
) -> C5IntegrityRecoveryReportV1:
    """Repair every C5 item and compare the projection with verification references."""

    if expected_item_count < 1:
        raise ValueError("expected_item_count must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    participant_path = Path(participant_zip)
    verification_path = Path(verification_zip)
    with (
        ZipFile(participant_path) as participant,
        ZipFile(verification_path) as verification,
    ):
        policies: dict[str, tuple[str, C5PublicIntegrityPolicyV1]] = {}
        for name in participant.namelist():
            if PurePosixPath(name).name != _POLICY_NAME:
                continue
            policy = C5PublicIntegrityPolicyV1.model_validate_json(
                participant.read(name)
            )
            if policy.task_id in policies:
                raise ValueError(f"duplicate C5 public policy: {policy.task_id}")
            policies[policy.task_id] = (_item_prefix(name), policy)
        pairs = _context_pairs(verification)
        references = _references(verification)

    failures = {
        task_id: _C5ProjectionFailureV1(
            task_id=task_id,
            status="unsupported",
            reason="C5 item has no unique C4 context sibling",
        )
        for task_id in policies
        if task_id not in pairs
    }
    requests = tuple(
        (
            str(participant_path),
            task_id,
            prefix,
            pairs[task_id],
            policy.model_dump_json(),
        )
        for task_id, (prefix, policy) in sorted(policies.items())
        if task_id in pairs
    )
    if workers == 1 or len(requests) < 2:
        projections = tuple(_project_c5_item(request) for request in requests)
    else:
        with single_threaded_numerical_process_pool(
            workers=min(workers, len(requests))
        ) as executor:
            projections = tuple(executor.map(_project_c5_item, requests, chunksize=1))
    projected = {
        row.task_id: row for row in projections if isinstance(row, _C5ProjectionV1)
    }
    failures.update(
        {
            row.task_id: row
            for row in projections
            if isinstance(row, _C5ProjectionFailureV1)
        }
    )

    records: list[C5IntegrityItemRecordV1] = []
    for task_id, (_prefix, policy) in sorted(policies.items()):
        projected_row = projected.get(task_id)
        if projected_row is None:
            failure = failures[task_id]
            records.append(
                C5IntegrityItemRecordV1(
                    task_id=task_id,
                    clean_context_parent_task_id=pairs.get(task_id, "unresolved"),
                    condition_id=policy.condition_id,
                    affected_domain=policy.affected_domain,
                    compound_key_fields=policy.compound_key_fields,
                    observed_duplicate_group_count=0,
                    observed_extra_row_count=0,
                    mutated_domain_content_sha256="0" * 64,
                    repaired_domain_content_sha256="0" * 64,
                    clean_parent_domain_content_sha256="0" * 64,
                    unaffected_domain_count=0,
                    independently_repaired=False,
                    repaired_content_equals_c4=False,
                    verification_reference_concordant=False,
                    status=failure.status,
                    failure_reason=failure.reason,
                )
            )
            continue
        reference = references.get(task_id)
        repaired_equals_c4 = (
            projected_row.repaired_checksum == projected_row.clean_checksum
        )
        reference_concordant = bool(
            reference is not None
            and reference.clean_context_parent_task_id == projected_row.parent_task_id
            and reference.affected_domain == policy.affected_domain
            and reference.compound_key_fields == policy.compound_key_fields
            and reference.observed_duplicate_group_count
            == projected_row.duplicate_groups
            and reference.observed_extra_row_count == projected_row.extra_rows
            and reference.mutated_domain_content_sha256
            == projected_row.mutated_checksum
            and reference.clean_domain_content_sha256 == projected_row.clean_checksum
            and reference.post_repair_data_checksum == projected_row.repaired_checksum
        )
        passed = bool(repaired_equals_c4 and reference_concordant)
        records.append(
            C5IntegrityItemRecordV1(
                task_id=task_id,
                clean_context_parent_task_id=projected_row.parent_task_id,
                condition_id=policy.condition_id,
                affected_domain=policy.affected_domain,
                compound_key_fields=policy.compound_key_fields,
                observed_duplicate_group_count=projected_row.duplicate_groups,
                observed_extra_row_count=projected_row.extra_rows,
                mutated_domain_content_sha256=projected_row.mutated_checksum,
                repaired_domain_content_sha256=projected_row.repaired_checksum,
                clean_parent_domain_content_sha256=projected_row.clean_checksum,
                unaffected_domain_count=projected_row.unaffected_count,
                independently_repaired=True,
                repaired_content_equals_c4=repaired_equals_c4,
                verification_reference_concordant=reference_concordant,
                status="repaired" if passed else "mismatch",
                failure_reason=(
                    None
                    if passed
                    else "independent repair disagrees with C4 or verification reference"
                ),
            )
        )
    unexpected_references = set(references) - set(policies)
    if unexpected_references:
        raise ValueError(
            "verification references have no participant C5 items: "
            f"{sorted(unexpected_references)!r}"
        )

    repaired_count = sum(row.status == "repaired" for row in records)
    mismatched_count = sum(row.status == "mismatch" for row in records)
    unsupported_count = sum(row.status == "unsupported" for row in records)
    passed = (
        len(records) == expected_item_count
        and repaired_count == expected_item_count
        and mismatched_count == 0
        and unsupported_count == 0
    )
    return C5IntegrityRecoveryReportV1(
        expected_item_count=expected_item_count,
        required_item_count=len(records),
        repaired_item_count=repaired_count,
        mismatched_item_count=mismatched_count,
        unsupported_item_count=unsupported_count,
        records=tuple(records),
        participant_archive_sha256=_sha256_file(participant_path),
        verification_archive_sha256=_sha256_file(verification_path),
        status="pass" if passed else "fail",
    )


def write_c5_integrity_recovery(
    *,
    participant_zip: Path,
    verification_zip: Path,
    output: Path,
    expected_item_count: int = 100,
    workers: int = 1,
) -> C5IntegrityRecoveryReportV1:
    """Write the checksum-bound full-census C5 recovery receipt."""

    report = recover_c5_integrity(
        participant_zip=participant_zip,
        verification_zip=verification_zip,
        expected_item_count=expected_item_count,
        workers=workers,
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report
