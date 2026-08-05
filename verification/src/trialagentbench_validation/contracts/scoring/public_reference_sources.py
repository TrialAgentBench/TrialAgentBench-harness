"""Public reference-source contracts for standalone release validation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.io.checksums import sha256_file
from trialagentbench_validation.io.json import read_json_model


class PublicReferenceTableRefV1(BaseModel):
    """One public table used to derive a route reference."""

    model_config = ConfigDict(extra="forbid")

    rel_path: str = Field(..., min_length=1)
    semantic_role: str = Field(..., min_length=1)
    sha256: str = Field(..., min_length=64, max_length=64)
    row_count: int = Field(..., ge=0)
    column_names: tuple[str, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_path(self) -> PublicReferenceTableRefV1:
        """Validate that the table path is participant-facing."""

        if not self.rel_path.startswith("items/"):
            raise ValueError("Public reference table refs must use item-public paths.")
        return self


class PublicReferenceSourceRecordV1(BaseModel):
    """Public source record for one route reference."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval.public_reference_source/v1"]
    task_id: str = Field(..., min_length=1)
    route_reference_id: str = Field(..., min_length=1)
    estimator_method_id: str = Field(..., min_length=1)
    source_mode: Literal[
        "public_analysis_table",
        "public_scoreable_mirror",
        "public_raw_reconstruction",
        "public_warranted_bounds",
        "method_composition",
    ]
    public_evidence_refs: tuple[str, ...] = Field(..., min_length=1)
    required_table_refs: tuple[PublicReferenceTableRefV1, ...] = Field(
        default_factory=tuple
    )
    reconstruction_policy_id: str | None = None
    bounds_policy_id: str | None = None
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_record(self) -> PublicReferenceSourceRecordV1:
        """Validate checksum and public-source invariants."""

        if any(ref.startswith("grader/") for ref in self.public_evidence_refs):
            raise ValueError(
                "Public reference-source evidence cannot cite grader paths."
            )
        if any("reconstruction_reference" in ref for ref in self.public_evidence_refs):
            raise ValueError(
                "Public reference-source evidence cannot cite reconstruction_reference."
            )
        if (
            self.source_mode == "public_raw_reconstruction"
            and self.reconstruction_policy_id is None
        ):
            raise ValueError(
                "Public reconstruction sources require reconstruction_policy_id."
            )
        if (
            self.source_mode == "public_warranted_bounds"
            and self.bounds_policy_id is None
        ):
            raise ValueError(
                "Public warranted-bounds sources require bounds_policy_id."
            )
        if (
            self.source_mode in {"public_analysis_table", "public_scoreable_mirror"}
            and not self.required_table_refs
        ):
            raise ValueError(
                "Table-backed public reference sources require table refs."
            )
        if self.checksum is not None and self.checksum != _payload_checksum(
            self.model_dump(mode="json")
        ):
            raise ValueError("Public reference-source checksum mismatch.")
        return self


class PublicReferenceSourceManifestV1(BaseModel):
    """Manifest for public reference-source domains."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["v1"]
    schema_id: Literal["trialagentbench.trialeval.public_reference_source_manifest/v1"]
    release_root: str
    generated_at_utc: datetime | None = None
    row_count: int = Field(..., ge=0)
    task_count: int = Field(..., ge=0)
    public_reference_sources_jsonl_sha256: str = Field(
        ..., min_length=64, max_length=64
    )
    route_references_sha256: str = Field(..., min_length=64, max_length=64)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)


def _payload_checksum(payload: dict[str, object]) -> str:
    payload = cast(dict[str, object], _drop_none(dict(payload)))
    payload.pop("checksum", None)
    body = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _drop_none(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _drop_none(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_drop_none(item) for item in value)
    return value


def read_jsonl_public_reference_sources(
    path: Path,
) -> tuple[PublicReferenceSourceRecordV1, ...]:
    """Read public reference-source records from JSONL."""

    records: list[PublicReferenceSourceRecordV1] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid public reference-source row at {path}:{line_number}."
            )
        records.append(PublicReferenceSourceRecordV1.model_validate(payload))
    return tuple(records)


def read_public_reference_source_domains(
    *,
    release_root: Path,
) -> tuple[tuple[PublicReferenceSourceRecordV1, ...], PublicReferenceSourceManifestV1]:
    """Read and validate public reference-source domains from a release root."""

    domains = Path(release_root) / "grader" / "domains"
    path = domains / "public_reference_sources.jsonl"
    manifest_path = domains / "public_reference_sources_manifest.json"
    records = read_jsonl_public_reference_sources(path)
    manifest = read_json_model(PublicReferenceSourceManifestV1, manifest_path)
    if manifest.row_count != len(records):
        raise ValueError(
            "Public reference-source manifest row_count does not match JSONL row count."
        )
    if manifest.public_reference_sources_jsonl_sha256 != sha256_file(path):
        raise ValueError(
            "Public reference-source manifest SHA-256 does not match JSONL file."
        )
    return records, manifest


__all__ = [
    "PublicReferenceSourceManifestV1",
    "PublicReferenceSourceRecordV1",
    "PublicReferenceTableRefV1",
    "read_jsonl_public_reference_sources",
    "read_public_reference_source_domains",
]
