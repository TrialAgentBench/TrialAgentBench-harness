"""Scoreable reference-input contracts for standalone release validation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.contracts.scoring.method_ids import (
    BOUNDED_DEVIATION_METHOD_IDS_V1,
)
from trialagentbench_validation.io.checksums import sha256_file
from trialagentbench_validation.io.json import read_json_model


class RouteReferenceInputTableRefV1(BaseModel):
    """One mirrored table used to compute scoreable reference."""

    model_config = ConfigDict(extra="forbid")

    rel_path: str = Field(..., min_length=1)
    semantic_role: str = Field(..., min_length=1)
    sha256: str = Field(..., min_length=64, max_length=64)
    row_count: int = Field(..., ge=0)
    column_names: tuple[str, ...] = Field(..., min_length=1)


class RouteReferenceInputRecordV1(BaseModel):
    """Input-bundle provenance for scoreable route references."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval.route_reference_input/v1"]
    task_id: str = Field(..., min_length=1)
    input_bundle_id: str = Field(..., min_length=1)
    estimator_method_id: str = Field(..., min_length=1)
    effect_scale: str = Field(..., min_length=1)
    sensitivity_parameter: float | None = Field(default=None, ge=0.0, le=1.0)
    lane_ids: tuple[str, ...] = Field(..., min_length=1)
    route_reference_ids: tuple[str, ...] = Field(..., min_length=1)
    required_table_refs: tuple[RouteReferenceInputTableRefV1, ...] = Field(
        ..., min_length=1
    )
    source_role: Literal[
        "canonical_analysis", "reconstruction_reference", "public_surface_mirror"
    ]
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_record(self) -> RouteReferenceInputRecordV1:
        """Validate checksum and scoreable-input mirror path invariants."""

        bounded_method = self.estimator_method_id in BOUNDED_DEVIATION_METHOD_IDS_V1
        if bounded_method != (self.sensitivity_parameter is not None):
            raise ValueError(
                "Bounded-deviation scoreable inputs require a sensitivity parameter, and other methods forbid it."
            )
        for table_ref in self.required_table_refs:
            if (
                not table_ref.rel_path.startswith("items/")
                or "/data/" not in table_ref.rel_path
            ):
                raise ValueError(
                    "Scoreable reference-input table refs must be participant item data paths."
                )
        if self.checksum is not None and self.checksum != _payload_checksum(
            self.model_dump(mode="json")
        ):
            raise ValueError("Scoreable reference-input checksum mismatch.")
        return self


class RouteReferenceInputManifestV1(BaseModel):
    """Manifest for scoreable reference-input domains."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["v1"]
    schema_id: Literal["trialagentbench.trialeval.route_reference_input_manifest/v1"]
    release_root: str
    generated_at_utc: datetime | None = None
    row_count: int = Field(..., ge=0)
    table_count: int = Field(..., ge=0)
    task_count: int = Field(..., ge=0)
    route_reference_inputs_jsonl_sha256: str = Field(..., min_length=64, max_length=64)
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


def read_jsonl_route_reference_inputs(
    path: Path,
) -> tuple[RouteReferenceInputRecordV1, ...]:
    """Read scoreable reference-input records from JSONL."""

    records: list[RouteReferenceInputRecordV1] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid scoreable reference-input row at {path}:{line_number}."
            )
        records.append(RouteReferenceInputRecordV1.model_validate(payload))
    return tuple(records)


def read_route_reference_input_domains(
    *,
    release_root: Path,
) -> tuple[tuple[RouteReferenceInputRecordV1, ...], RouteReferenceInputManifestV1]:
    """Read and validate scoreable reference-input domains from a release root."""

    domains = Path(release_root) / "grader" / "domains"
    path = domains / "route_reference_inputs.jsonl"
    manifest_path = domains / "route_reference_inputs_manifest.json"
    records = read_jsonl_route_reference_inputs(path)
    manifest = read_json_model(RouteReferenceInputManifestV1, manifest_path)
    if manifest.row_count != len(records):
        raise ValueError(
            "Scoreable reference-input manifest row_count does not match JSONL row count."
        )
    if manifest.route_reference_inputs_jsonl_sha256 != sha256_file(path):
        raise ValueError(
            "Scoreable reference-input manifest SHA-256 does not match JSONL file."
        )
    return records, manifest


__all__ = [
    "RouteReferenceInputManifestV1",
    "RouteReferenceInputRecordV1",
    "RouteReferenceInputTableRefV1",
    "read_jsonl_route_reference_inputs",
    "read_route_reference_input_domains",
]
