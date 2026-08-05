"""Method-composition contracts for standalone release validation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.io.checksums import sha256_file
from trialagentbench_validation.io.json import read_json_model


class AdjustmentParametersV1(BaseModel):
    """Adjustment parameters for a composed estimator."""

    model_config = ConfigDict(extra="forbid")

    adjustment_id: str = Field(..., min_length=1)
    spending_family: Literal["obrien_fleming", "pocock"]
    information_fractions: tuple[float, ...] = Field(..., min_length=2)
    total_alpha: float = Field(..., gt=0.0, lt=1.0)
    z_critical_by_look: tuple[float, ...] = Field(..., min_length=2)
    z_value: float = Field(..., gt=0.0)
    look_count: int = Field(..., ge=2)
    analysis_look_index: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_design(self) -> AdjustmentParametersV1:
        """Validate the complete alpha-spending design."""

        fractions = tuple(float(value) for value in self.information_fractions)
        boundaries = tuple(float(value) for value in self.z_critical_by_look)
        if len(fractions) != self.look_count or len(boundaries) != self.look_count:
            raise ValueError(
                "Group-sequential look_count must match information fractions and boundaries."
            )
        if any(not 0.0 < value <= 1.0 for value in fractions):
            raise ValueError(
                "Group-sequential information fractions must lie in (0, 1]."
            )
        if any(
            right <= left for left, right in zip(fractions, fractions[1:], strict=False)
        ):
            raise ValueError(
                "Group-sequential information fractions must increase strictly."
            )
        if fractions[-1] != 1.0:
            raise ValueError("Group-sequential information fractions must end at 1.0.")
        if any(value <= 0.0 for value in boundaries):
            raise ValueError("Group-sequential critical values must be positive.")
        if self.analysis_look_index >= self.look_count:
            raise ValueError(
                "Group-sequential analysis_look_index must identify a declared look."
            )
        if abs(float(self.z_value) - boundaries[self.analysis_look_index]) > 1e-12:
            raise ValueError(
                "Group-sequential z_value must equal the realized-look critical value."
            )
        if self.adjustment_id != f"{self.spending_family}_{self.look_count}look":
            raise ValueError(
                "Group-sequential adjustment_id must identify its spending family and look count."
            )
        return self


class MethodCompositionRecordV1(BaseModel):
    """Lineage and canonical values for one composed scoreable method."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval.method_composition/v1"]
    task_id: str = Field(..., min_length=1)
    lane_id: str = Field(..., min_length=1)
    route_reference_id: str = Field(..., min_length=1)
    composed_method_id: str = Field(..., min_length=1)
    base_estimator_method_id: str = Field(..., min_length=1)
    adjustment_id: str = Field(..., min_length=1)
    adjustment_parameters: AdjustmentParametersV1
    base_value: float
    base_standard_error: float = Field(..., gt=0.0)
    adjusted_lower: float
    adjusted_upper: float
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_record(self) -> MethodCompositionRecordV1:
        """Validate checksum and interval invariants."""

        if self.adjustment_id != self.adjustment_parameters.adjustment_id:
            raise ValueError("Method-composition adjustment_id must match parameters.")
        if self.adjusted_lower > self.adjusted_upper:
            raise ValueError("Method-composition lower bound must be <= upper bound.")
        expected_half_width = (
            self.adjustment_parameters.z_value * self.base_standard_error
        )
        if (
            abs(self.adjusted_lower - (self.base_value - expected_half_width)) > 1e-12
            or abs(self.adjusted_upper - (self.base_value + expected_half_width))
            > 1e-12
        ):
            raise ValueError(
                "Method-composition interval must equal base value plus/minus realized-look critical value times SE."
            )
        if self.checksum is not None and self.checksum != _payload_checksum(
            self.model_dump(mode="json")
        ):
            raise ValueError("Method-composition checksum mismatch.")
        return self


class MethodCompositionManifestV1(BaseModel):
    """Manifest for method-composition domains."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["v1"]
    schema_id: Literal["trialagentbench.trialeval.method_composition_manifest/v1"]
    release_root: str
    generated_at_utc: datetime | None = None
    row_count: int = Field(..., ge=0)
    task_count: int = Field(..., ge=0)
    method_composition_jsonl_sha256: str = Field(..., min_length=64, max_length=64)
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


def read_jsonl_method_composition(path: Path) -> tuple[MethodCompositionRecordV1, ...]:
    """Read method-composition records from JSONL."""

    records: list[MethodCompositionRecordV1] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid method-composition row at {path}:{line_number}.")
        records.append(MethodCompositionRecordV1.model_validate(payload))
    return tuple(records)


def read_method_composition_domains(
    *,
    release_root: Path,
) -> tuple[tuple[MethodCompositionRecordV1, ...], MethodCompositionManifestV1]:
    """Read and validate method-composition domains from a release root."""

    domains = Path(release_root) / "grader" / "domains"
    path = domains / "method_composition.jsonl"
    manifest_path = domains / "method_composition_manifest.json"
    records = read_jsonl_method_composition(path)
    manifest = read_json_model(MethodCompositionManifestV1, manifest_path)
    if manifest.row_count != len(records):
        raise ValueError(
            "Method-composition manifest row_count does not match JSONL row count."
        )
    if manifest.method_composition_jsonl_sha256 != sha256_file(path):
        raise ValueError(
            "Method-composition manifest SHA-256 does not match JSONL file."
        )
    return records, manifest


__all__ = [
    "AdjustmentParametersV1",
    "MethodCompositionManifestV1",
    "MethodCompositionRecordV1",
    "read_jsonl_method_composition",
    "read_method_composition_domains",
]
