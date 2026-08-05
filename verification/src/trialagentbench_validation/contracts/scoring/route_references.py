"""Route-reference contracts for standalone TrialEvalBench grading."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.contracts.scoring.method_ids import (
    BOUNDED_DEVIATION_METHOD_IDS_V1,
)
from trialagentbench_validation.io.checksums import sha256_file
from trialagentbench_validation.io.json import read_json_model


class ReferenceVectorComponentV1(BaseModel):
    """One ordered component of a vector-valued TrialEval reference target."""

    model_config = ConfigDict(extra="forbid")

    component_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    value: float
    standard_error: float | None = Field(default=None, ge=0.0)
    ci_low: float | None = None
    ci_high: float | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_component(self) -> ReferenceVectorComponentV1:
        """Validate vector component intervals."""

        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError(
                "Vector component ci_low and ci_high must be provided together."
            )
        if (
            self.ci_low is not None
            and self.ci_high is not None
            and float(self.ci_low) > float(self.ci_high)
        ):
            raise ValueError("Vector component ci_low must be <= ci_high.")
        return self


class ReferenceTestPayloadV1(BaseModel):
    """Structured payload for test-valued TrialEval reference targets."""

    model_config = ConfigDict(extra="forbid")

    statistic: float
    variance: float | None = Field(default=None, ge=0.0)
    p_value: float = Field(..., ge=0.0, le=1.0)
    alternative: Literal["two_sided", "greater", "less"] = "two_sided"
    direction: Literal["benefit_positive", "benefit_negative"] = "benefit_negative"
    method_parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)


class NumericalEquivalencePolicyV1(BaseModel):
    """Numerical replay equivalence embedded in a canonical reference record."""

    model_config = ConfigDict(extra="forbid")

    policy_id: Literal["float64_sqrt_epsilon_v1"]
    absolute_tolerance: float = Field(..., gt=0.0)
    relative_tolerance: float = Field(..., gt=0.0)
    basis: Literal["deterministic_cross_implementation_replay"]

    @model_validator(mode="after")
    def require_canonical_threshold(self) -> NumericalEquivalencePolicyV1:
        """Reject evaluator-authored tolerance relaxation."""

        threshold = math.sqrt(sys.float_info.epsilon)
        if self.absolute_tolerance != threshold or self.relative_tolerance != threshold:
            raise ValueError(
                "float64_sqrt_epsilon_v1 requires the exact square-root-epsilon threshold."
            )
        return self

    def tolerance(self, *, expected: float, observed: float) -> float:
        """Return the scale-aware permitted floating-point difference."""

        return float(self.absolute_tolerance) + float(self.relative_tolerance) * max(
            abs(float(expected)), abs(float(observed))
        )


def float64_equivalence_policy_v1() -> NumericalEquivalencePolicyV1:
    """Return the canonical square-root-machine-epsilon replay policy."""

    tolerance = math.sqrt(sys.float_info.epsilon)
    return NumericalEquivalencePolicyV1(
        policy_id="float64_sqrt_epsilon_v1",
        absolute_tolerance=tolerance,
        relative_tolerance=tolerance,
        basis="deterministic_cross_implementation_replay",
    )


class RouteReferenceRecordV1(BaseModel):
    """One scoreable or diagnostic route reference for a TrialEval score lane."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval.route_reference/v1"]
    task_id: str = Field(..., min_length=1)
    item_id: str = Field(..., min_length=1)
    lane_id: str = Field(..., min_length=1)
    route_reference_id: str = Field(..., min_length=1)
    variant_role: Literal[
        "required_primary",
        "credit_eligible_primary_alternative",
        "sensitivity_only",
        "diagnostic_only",
    ]
    route_family: str = Field(..., min_length=1)
    estimator_method_id: str = Field(..., min_length=1)
    effect_scale: str = Field(..., min_length=1)
    sensitivity_parameter: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_shape: Literal["point", "bound", "test", "vector", "curve", "limitation"]
    credit_eligible_codes: tuple[str, ...] = Field(default_factory=tuple)
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    standard_error: float | None = Field(default=None, ge=0.0)
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    vector_components: tuple[ReferenceVectorComponentV1, ...] = Field(
        default_factory=tuple
    )
    test_payload: ReferenceTestPayloadV1 | None = None
    curve_table_ref: str | None = Field(default=None, min_length=1)
    public_evidence_basis: tuple[str, ...] = Field(..., min_length=1)
    required_modifiers: tuple[str, ...] = Field(default_factory=tuple)
    identification_class: Literal["point_identified", "partially_identified"]
    support_status: Literal["official_supported", "diagnostic_supported"]
    support_rationale: str = Field(..., min_length=1)
    numerical_equivalence: NumericalEquivalencePolicyV1
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_record(self) -> RouteReferenceRecordV1:
        """Validate scoreability and checksum invariants."""

        bounded_method = self.estimator_method_id in BOUNDED_DEVIATION_METHOD_IDS_V1
        if bounded_method and self.sensitivity_parameter is None:
            raise ValueError(
                "Bounded-deviation route references require an explicit sensitivity_parameter."
            )
        if not bounded_method and self.sensitivity_parameter is not None:
            raise ValueError(
                "Only bounded-deviation route references may declare sensitivity_parameter."
            )
        if (
            self.sensitivity_parameter is not None
            and self.effect_scale != "risk_difference_tau"
        ):
            raise ValueError(
                "Bounded-deviation route references must use risk_difference_tau."
            )
        codes = tuple(sorted(set(self.credit_eligible_codes)))
        self.credit_eligible_codes = codes
        if self.answer_shape == "limitation":
            if not codes:
                raise ValueError(
                    "Limitation route references require credit_eligible_codes."
                )
            if self.estimator_method_id != "qualified_limitation_or_abstention":
                raise ValueError(
                    "Limitation references require the canonical limitation route identifier."
                )
            if self.value is not None:
                raise ValueError(
                    "Limitation route references cannot provide a numeric value."
                )
        elif codes:
            raise ValueError(
                "Only limitation route references may provide credit_eligible_codes."
            )
        if self.answer_shape in {"point", "bound"} and self.value is None:
            raise ValueError("Numeric route references require value.")
        if (
            self.support_status == "official_supported"
            and self.answer_shape == "point"
            and (self.standard_error is None or float(self.standard_error) <= 0.0)
        ):
            raise ValueError(
                "Official point route references require a positive standard_error."
            )
        if self.answer_shape == "bound" and (self.lower is None or self.upper is None):
            raise ValueError("Bound route references require lower and upper.")
        if self.answer_shape != "bound" and (
            self.lower is not None or self.upper is not None
        ):
            raise ValueError("Only bound route references may provide lower and upper.")
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError(
                "Route-reference ci_low and ci_high must be provided together."
            )
        if (
            self.ci_low is not None
            and self.ci_high is not None
            and float(self.ci_low) > float(self.ci_high)
        ):
            raise ValueError("Route-reference ci_low must be <= ci_high.")
        if self.answer_shape != "point" and (
            self.ci_low is not None or self.ci_high is not None
        ):
            raise ValueError(
                "Only point route references may provide sampling confidence limits."
            )
        if self.answer_shape == "test":
            if self.test_payload is None:
                raise ValueError("Test route references require test_payload.")
            if self.support_status == "official_supported" and (
                self.test_payload.variance is None
                or float(self.test_payload.variance) <= 0.0
            ):
                raise ValueError(
                    "Official test route references require a positive statistic variance."
                )
            if (
                self.value is not None
                and abs(float(self.value) - float(self.test_payload.statistic)) > 1e-12
            ):
                raise ValueError(
                    "Test route reference value must mirror test_payload.statistic when provided."
                )
        elif self.test_payload is not None:
            raise ValueError("Only test route references may provide test_payload.")
        if self.answer_shape == "vector":
            if not self.vector_components:
                raise ValueError("Vector route references require vector_components.")
            component_ids = tuple(
                str(component.component_id) for component in self.vector_components
            )
            if len(component_ids) != len(set(component_ids)):
                raise ValueError(
                    "Vector route reference component_id values must be unique."
                )
            if self.value is not None:
                raise ValueError(
                    "Vector route references must not also provide scalar value."
                )
            if self.support_status == "official_supported" and any(
                component.standard_error is None
                or float(component.standard_error) <= 0.0
                for component in self.vector_components
            ):
                raise ValueError(
                    "Official vector route references require a positive standard_error per component."
                )
        elif self.vector_components:
            raise ValueError(
                "Only vector route references may provide vector_components."
            )
        if self.answer_shape == "curve":
            if self.curve_table_ref is None:
                raise ValueError("Curve route references require curve_table_ref.")
        elif self.curve_table_ref is not None:
            raise ValueError("Only curve route references may provide curve_table_ref.")
        if (self.lower is None) != (self.upper is None):
            raise ValueError(
                "Route reference lower and upper must be provided together."
            )
        if self.lower is not None and self.upper is not None:
            if float(self.lower) > float(self.upper):
                raise ValueError("Route reference lower must be <= upper.")
            if self.value is not None and not (
                float(self.lower) - 1e-12
                <= float(self.value)
                <= float(self.upper) + 1e-12
            ):
                raise ValueError(
                    "Route reference value must lie within [lower, upper]."
                )
        if (
            self.support_status == "official_supported"
            and self.variant_role == "diagnostic_only"
        ):
            raise ValueError(
                "Official-supported route references cannot be diagnostic_only."
            )
        if (
            self.support_status == "diagnostic_supported"
            and self.variant_role != "diagnostic_only"
        ):
            raise ValueError(
                "Diagnostic-supported route references must be diagnostic_only."
            )
        if self.checksum is not None and self.checksum != _payload_checksum(
            self.model_dump(mode="json")
        ):
            raise ValueError("Route reference checksum mismatch.")
        return self


class RouteReferenceTaskRangeV1(BaseModel):
    """Contiguous byte range for one task in canonical reference JSONL."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(..., min_length=1)
    byte_offset: int = Field(..., ge=0)
    byte_length: int = Field(..., gt=0)
    row_count: int = Field(..., gt=0)


class RouteReferenceManifestV1(BaseModel):
    """Manifest for route-reference JSONL domains."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["v1"]
    schema_id: Literal["trialagentbench.trialeval.route_reference_manifest/v1"]
    release_root: str
    generated_at_utc: datetime | None = None
    row_count: int = Field(..., ge=0)
    task_count: int = Field(..., ge=0)
    task_byte_ranges: tuple[RouteReferenceTaskRangeV1, ...] = Field(
        default_factory=tuple
    )
    route_references_jsonl_sha256: str = Field(..., min_length=64, max_length=64)
    evaluation_target_register_sha256: str = Field(..., min_length=64, max_length=64)
    estimator_registry_sha256: str = Field(..., min_length=64, max_length=64)
    estimator_route_family_map_sha256: str = Field(..., min_length=64, max_length=64)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_manifest(self) -> RouteReferenceManifestV1:
        """Validate task-range closure and source checksum."""

        ranges = tuple(sorted(self.task_byte_ranges, key=lambda entry: entry.task_id))
        self.task_byte_ranges = ranges
        task_ids = tuple(entry.task_id for entry in ranges)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(
                "Route-reference task byte ranges must have unique task_id values."
            )
        if len(ranges) != self.task_count:
            raise ValueError(
                "Route-reference task byte-range count must equal task_count."
            )
        if sum(entry.row_count for entry in ranges) != self.row_count:
            raise ValueError(
                "Route-reference task byte-range rows must sum to row_count."
            )
        expected_offset = 0
        for entry in sorted(ranges, key=lambda value: value.byte_offset):
            if entry.byte_offset != expected_offset:
                raise ValueError(
                    "Route-reference task byte ranges must be contiguous from byte zero."
                )
            expected_offset += entry.byte_length
        if self.checksum is not None and self.checksum != _payload_checksum(
            self.model_dump(mode="json")
        ):
            raise ValueError("Route-reference manifest checksum mismatch.")
        return self


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


def read_jsonl_route_references(path: Path) -> tuple[RouteReferenceRecordV1, ...]:
    """Read route-reference records from JSONL."""

    records: list[RouteReferenceRecordV1] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid route-reference row at {path}:{line_number}.")
        records.append(RouteReferenceRecordV1.model_validate(payload))
    return tuple(records)


def read_route_reference_domains(
    *,
    release_root: Path,
) -> tuple[tuple[RouteReferenceRecordV1, ...], RouteReferenceManifestV1]:
    """Read and validate route-reference domains from a release root."""

    domains = Path(release_root) / "grader" / "domains"
    path = domains / "route_references.jsonl"
    manifest_path = domains / "route_references_manifest.json"
    records = read_jsonl_route_references(path)
    manifest = read_json_model(RouteReferenceManifestV1, manifest_path)
    if manifest.row_count != len(records):
        raise ValueError(
            "Route-reference manifest row_count does not match JSONL row count."
        )
    if manifest.route_references_jsonl_sha256 != sha256_file(path):
        raise ValueError("Route-reference manifest SHA-256 does not match JSONL file.")
    dependency_paths = {
        "evaluation_target_register_sha256": domains
        / "evaluation_target_register.jsonl",
        "estimator_registry_sha256": domains / "estimator_registry.json",
        "estimator_route_family_map_sha256": domains
        / "estimator_route_family_map.json",
    }
    for field_name, dependency_path in dependency_paths.items():
        if getattr(manifest, field_name) != sha256_file(dependency_path):
            raise ValueError(
                f"Route-reference manifest dependency SHA-256 mismatch: {field_name}."
            )
    return records, manifest


@dataclass(frozen=True, slots=True)
class RouteReferenceStoreV1:
    """Validated task-random-access view over canonical reference JSONL."""

    path: Path
    manifest: RouteReferenceManifestV1
    _ranges: dict[str, RouteReferenceTaskRangeV1]

    @classmethod
    def from_release(cls, *, release_root: Path) -> RouteReferenceStoreV1:
        """Validate the scorer dependencies without loading reference records."""

        domains = Path(release_root) / "grader" / "domains"
        path = domains / "route_references.jsonl"
        manifest = read_json_model(
            RouteReferenceManifestV1, domains / "route_references_manifest.json"
        )
        if manifest.route_references_jsonl_sha256 != sha256_file(path):
            raise ValueError(
                "Route-reference manifest SHA-256 does not match JSONL file."
            )
        dependency_paths = {
            "evaluation_target_register_sha256": domains
            / "evaluation_target_register.jsonl",
            "estimator_registry_sha256": domains / "estimator_registry.json",
            "estimator_route_family_map_sha256": domains
            / "estimator_route_family_map.json",
        }
        for field_name, dependency_path in dependency_paths.items():
            if getattr(manifest, field_name) != sha256_file(dependency_path):
                raise ValueError(
                    f"Route-reference manifest dependency SHA-256 mismatch: {field_name}."
                )
        expected_size = sum(entry.byte_length for entry in manifest.task_byte_ranges)
        if path.stat().st_size != expected_size:
            raise ValueError(
                "Route-reference task byte ranges do not cover the exact JSONL file."
            )
        return cls(
            path=path,
            manifest=manifest,
            _ranges={entry.task_id: entry for entry in manifest.task_byte_ranges},
        )

    def for_task(self, task_id: str) -> tuple[RouteReferenceRecordV1, ...]:
        """Load and validate only one task's contiguous reference records."""

        task_range = self._ranges.get(task_id)
        if task_range is None:
            raise KeyError(
                f"Route-reference manifest does not contain task_id={task_id!r}."
            )
        with self.path.open("rb") as handle:
            handle.seek(task_range.byte_offset)
            payload = handle.read(task_range.byte_length)
        if len(payload) != task_range.byte_length or not payload.endswith(b"\n"):
            raise ValueError(
                f"Route-reference byte range is truncated for task_id={task_id!r}."
            )
        records: list[RouteReferenceRecordV1] = []
        for line_number, line in enumerate(payload.splitlines(), start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"Invalid route-reference row for task_id={task_id!r} at range line {line_number}."
                )
            record = RouteReferenceRecordV1.model_validate(row)
            if record.task_id != task_id:
                raise ValueError(
                    "Route-reference byte range contains a different task: "
                    f"expected={task_id!r} observed={record.task_id!r}."
                )
            records.append(record)
        if len(records) != task_range.row_count:
            raise ValueError(
                f"Route-reference byte-range row count mismatch for task_id={task_id!r}."
            )
        route_reference_index(tuple(records))
        return tuple(records)


def route_reference_index(
    records: tuple[RouteReferenceRecordV1, ...],
) -> dict[str, RouteReferenceRecordV1]:
    """Index route references by stable route_reference_id."""

    index = {record.route_reference_id: record for record in records}
    if len(index) != len(records):
        raise ValueError(
            "Route references contain duplicate route_reference_id values."
        )
    return index


def submission_answer_shape_for_reference_v1(answer_shape: str) -> str:
    """Return the canonical submission shape for one route-reference shape."""

    return {
        "point": "numeric_point",
        "bound": "bounds_interval",
        "test": "statistical_test",
        "vector": "numeric_vector",
        "curve": "curve",
        "limitation": "structured_notes",
    }[answer_shape]


__all__ = [
    "NumericalEquivalencePolicyV1",
    "ReferenceTestPayloadV1",
    "ReferenceVectorComponentV1",
    "RouteReferenceManifestV1",
    "RouteReferenceRecordV1",
    "RouteReferenceStoreV1",
    "RouteReferenceTaskRangeV1",
    "float64_equivalence_policy_v1",
    "read_jsonl_route_references",
    "read_route_reference_domains",
    "submission_answer_shape_for_reference_v1",
    "route_reference_index",
]
