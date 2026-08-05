"""Numeric public-evidence recomputation for TrialEval route references."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeVar, cast
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from trialagentbench_validation.contracts.scoring.method_composition import (
    MethodCompositionRecordV1,
)
from trialagentbench_validation.contracts.scoring.route_reference_inputs import (
    RouteReferenceInputRecordV1,
)
from trialagentbench_validation.contracts.scoring.route_references import (
    RouteReferenceRecordV1,
)
from trialagentbench_validation.io.json import write_json_model
from trialagentbench_validation.process_pool import (
    single_threaded_numerical_process_pool,
)
from trialagentbench_validation.trialeval.references.calculators import (
    PublicNumericBoundResultV1,
    PublicNumericVectorResultV1,
    _KmFamilyResult,
    has_required_public_inputs_for_method_v1,
    recompute_public_numeric_result_v1,
    supported_public_numeric_method_ids_v1,
)
from trialagentbench_validation.trialeval.references.io import (
    public_rel_path_for_scoreable_ref_v1,
    public_surface_shape_for_scoreable_refs_v1,
)
from trialagentbench_validation.trialeval.references.io import (
    read_json_from_public_v1 as _read_json_from_public,
)

NumericReferenceRecomputeStatusV1 = Literal["pass", "fail"]
NumericReferenceRecomputeOutcomeV1 = Literal[
    "matched",
    "mismatched",
    "unsupported_calculator",
    "not_numeric",
    "missing_public_input",
    "invalid_public_input",
]
UnsupportedReferenceDispositionV1 = Literal[
    "public_reconstruction_protocol_required",
    "ipcw_protocol_required",
    "complex_model_protocol_required",
    "standardization_protocol_required",
    "method_composition_record_required",
    "design_adjustment_protocol_required",
    "input_surface_insufficient_for_registered_calculator",
    "unregistered_public_calculator",
]
PublicReferenceDriftClassificationV1 = Literal[
    "no_drift",
    "blocked_derivation_gap",
    "supported_reference_mismatch",
    "missing_public_input",
    "invalid_public_input",
    "not_numeric_reference",
]

SUPPORTED_PUBLIC_NUMERIC_METHODS_V1 = supported_public_numeric_method_ids_v1()
_UNSUPPORTED_METHOD_DISPOSITIONS_V1: dict[
    str,
    tuple[UnsupportedReferenceDispositionV1, str],
] = {}
_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _canonical_record_checksum_v1(record: BaseModel) -> str:
    """Return the canonical checksum used by scoreable domain records."""

    payload = record.model_dump(mode="json", exclude_none=True)
    payload.pop("checksum", None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class PublicEvidenceNumericReferenceCheckV1(BaseModel):
    """One numeric reference recomputation check from public evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.public_evidence_numeric_reference_check/v1"] = (
        "trialagentbench.public_evidence_numeric_reference_check/v1"
    )
    task_id: str = Field(..., min_length=1)
    item_id: str = Field(..., min_length=1)
    lane_id: str = Field(..., min_length=1)
    route_reference_id: str = Field(..., min_length=1)
    route_reference_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    variant_role: str = Field(..., min_length=1)
    input_bundle_id: str = Field(..., min_length=1)
    input_bundle_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    method_composition_checksum: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    route_family: str = Field(..., min_length=1)
    estimator_method_id: str = Field(..., min_length=1)
    effect_scale: str = Field(..., min_length=1)
    sensitivity_parameter: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_shape: str = Field(..., min_length=1)
    identification_class: Literal["point_identified", "partially_identified"]
    support_status: str = Field(..., min_length=1)
    outcome: NumericReferenceRecomputeOutcomeV1
    old_value: float | None = None
    old_lower: float | None = None
    old_upper: float | None = None
    old_standard_error: float | None = Field(default=None, ge=0.0)
    recomputed_value: float | None = None
    recomputed_lower: float | None = None
    recomputed_upper: float | None = None
    recomputed_standard_error: float | None = Field(default=None, ge=0.0)
    abs_diff: float | None = Field(default=None, ge=0.0)
    lower_abs_diff: float | None = Field(default=None, ge=0.0)
    upper_abs_diff: float | None = Field(default=None, ge=0.0)
    standard_error_abs_diff: float | None = Field(default=None, ge=0.0)
    standard_error_abs_tolerance: float | None = Field(default=None, ge=0.0)
    vector_component_count: int | None = Field(default=None, ge=0)
    vector_max_abs_diff: float | None = Field(default=None, ge=0.0)
    recomputed_vector_values: dict[str, float] = Field(default_factory=dict)
    abs_tolerance: float | None = Field(default=None, ge=0.0)
    public_table_paths: tuple[str, ...] = Field(default_factory=tuple)
    public_table_roles: tuple[str, ...] = Field(default_factory=tuple)
    public_surface_shape: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


def _numeric_reference_check(
    base: Mapping[str, object],
    *,
    outcome: NumericReferenceRecomputeOutcomeV1,
    message: str,
    **fields: object,
) -> PublicEvidenceNumericReferenceCheckV1:
    """Build a strict numeric reference-check row from dynamic release metadata."""

    payload = dict(base)
    payload.update(fields)
    payload["outcome"] = outcome
    payload["message"] = message
    return PublicEvidenceNumericReferenceCheckV1.model_validate(payload)


class UnsupportedPublicReferenceMethodDispositionV1(BaseModel):
    """Method-level disposition for unsupported public numeric replay rows."""

    model_config = ConfigDict(extra="forbid")

    estimator_method_id: str = Field(..., min_length=1)
    unsupported_count: int = Field(..., ge=1)
    disposition: UnsupportedReferenceDispositionV1
    required_release_action: str = Field(..., min_length=1)
    surface_shape_counts: dict[str, int] = Field(default_factory=dict)


class PublicEvidenceReferenceDriftDispositionRowV1(BaseModel):
    """Per-route-reference public-evidence drift disposition."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal[
        "trialagentbench.public_evidence_reference_drift_disposition/v1"
    ] = "trialagentbench.public_evidence_reference_drift_disposition/v1"
    task_id: str = Field(..., min_length=1)
    item_id: str = Field(..., min_length=1)
    lane_id: str = Field(..., min_length=1)
    route_reference_id: str = Field(..., min_length=1)
    variant_role: str = Field(..., min_length=1)
    route_family: str = Field(..., min_length=1)
    estimator_method_id: str = Field(..., min_length=1)
    effect_scale: str = Field(..., min_length=1)
    answer_shape: str = Field(..., min_length=1)
    identification_class: Literal["point_identified", "partially_identified"]
    old_value: float | None = None
    old_lower: float | None = None
    old_upper: float | None = None
    new_value: float | None = None
    new_lower: float | None = None
    new_upper: float | None = None
    abs_diff: float | None = Field(default=None, ge=0.0)
    lower_abs_diff: float | None = Field(default=None, ge=0.0)
    upper_abs_diff: float | None = Field(default=None, ge=0.0)
    vector_component_count: int | None = Field(default=None, ge=0)
    vector_max_abs_diff: float | None = Field(default=None, ge=0.0)
    new_vector_values: dict[str, float] = Field(default_factory=dict)
    abs_tolerance: float | None = Field(default=None, ge=0.0)
    classification: PublicReferenceDriftClassificationV1
    disposition: UnsupportedReferenceDispositionV1 | None = None
    required_release_action: str | None = None
    public_surface_shape: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class PublicEvidenceNumericReferenceReportV1(BaseModel):
    """Numeric public-evidence reference recomputation report."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal[
        "trialagentbench.public_evidence_numeric_reference_report/v1"
    ] = "trialagentbench.public_evidence_numeric_reference_report/v1"
    evaluator_zip: str
    public_zip: str
    status: NumericReferenceRecomputeStatusV1
    route_reference_count: int
    route_reference_input_count: int
    supported_method_ids: tuple[str, ...]
    supported_check_count: int
    matched_count: int
    mismatched_count: int
    unsupported_calculator_count: int
    not_numeric_count: int
    missing_public_input_count: int
    invalid_public_input_count: int
    outcome_counts: dict[str, int]
    method_counts: dict[str, int]
    method_outcome_counts: dict[str, dict[str, int]]
    unsupported_surface_shape_counts: dict[str, int]
    unsupported_disposition_counts: dict[str, int]
    unsupported_method_dispositions: tuple[
        UnsupportedPublicReferenceMethodDispositionV1, ...
    ]
    drift_classification_counts: dict[str, int]
    findings: tuple[str, ...]
    checks: tuple[PublicEvidenceNumericReferenceCheckV1, ...]
    drift_dispositions: tuple[PublicEvidenceReferenceDriftDispositionRowV1, ...]


def _partition_scoreable_inputs_v1(
    *,
    scoreable_inputs: Sequence[RouteReferenceInputRecordV1],
    reference_by_id: Mapping[str, RouteReferenceRecordV1],
    workers: int,
) -> tuple[tuple[tuple[int, RouteReferenceInputRecordV1], ...], ...]:
    """Balance underlying trials while co-locating their context views."""

    grouped: dict[str, list[tuple[int, RouteReferenceInputRecordV1]]] = {}
    for index, reference_input in enumerate(scoreable_inputs):
        variants = tuple(
            reference_by_id.get(route_reference_id)
            for route_reference_id in reference_input.route_reference_ids
        )
        if any(variant is None for variant in variants):
            raise ValueError(
                f"Scoreable input {reference_input.input_bundle_id!r} references unknown reference."
            )
        item_ids = {variant.item_id for variant in variants if variant is not None}
        if len(item_ids) != 1:
            raise ValueError(
                f"Scoreable input {reference_input.input_bundle_id!r} must resolve to one underlying item."
            )
        grouped.setdefault(next(iter(item_ids)), []).append((index, reference_input))
    if int(workers) == 1:
        return (
            (tuple(row for item_id in sorted(grouped) for row in grouped[item_id]),)
            if grouped
            else ()
        )
    return tuple(tuple(grouped[item_id]) for item_id in sorted(grouped))


def _recompute_scoreable_input_batch_v1(
    *,
    public_zip: Path,
    indexed_inputs: Sequence[tuple[int, RouteReferenceInputRecordV1]],
    reference_by_id: Mapping[str, RouteReferenceRecordV1],
    method_compositions: Mapping[str, MethodCompositionRecordV1],
) -> tuple[tuple[int, tuple[PublicEvidenceNumericReferenceCheckV1, ...]], ...]:
    """Recompute one isolated batch while sharing task-local calculation caches."""

    output: list[tuple[int, tuple[PublicEvidenceNumericReferenceCheckV1, ...]]] = []
    numeric_cache: dict[
        tuple[str, str, str, str, str, float | None, tuple[str, ...]],
        float | PublicNumericBoundResultV1 | PublicNumericVectorResultV1,
    ] = {}
    uncertainty_cache: dict[
        tuple[str, str, str, str, str, float | None, tuple[str, ...]], float | None
    ] = {}
    km_family_cache: dict[tuple[str, ...], _KmFamilyResult] = {}
    with ZipFile(public_zip) as public:
        for index, reference_input in indexed_inputs:
            input_checks: list[PublicEvidenceNumericReferenceCheckV1] = []
            for route_reference_id in reference_input.route_reference_ids:
                route_reference = reference_by_id.get(route_reference_id)
                if route_reference is None:
                    raise ValueError(
                        f"Scoreable input references unknown route_reference_id={route_reference_id!r}."
                    )
                if (
                    reference_input.sensitivity_parameter
                    != route_reference.sensitivity_parameter
                ):
                    raise ValueError(
                        "Scoreable input sensitivity_parameter does not match its route reference. "
                        f"input_bundle_id={reference_input.input_bundle_id!r} route_reference_id={route_reference_id!r}."
                    )
                input_checks.append(
                    _recompute_one(
                        public=public,
                        reference_input=reference_input,
                        route_reference=route_reference,
                        method_compositions=method_compositions,
                        numeric_cache=numeric_cache,
                        uncertainty_cache=uncertainty_cache,
                        km_family_cache=km_family_cache,
                    )
                )
            output.append((index, tuple(input_checks)))
    return tuple(output)


def recompute_trialeval_public_numeric_reference_v1(
    *,
    evaluator_zip: Path,
    public_zip: Path,
    workers: int = 1,
) -> PublicEvidenceNumericReferenceReportV1:
    """Recompute supported TrialEval reference values from public ZIP evidence.

    Unsupported estimator methods are classified explicitly. They are not
    treated as a pass for public-evidence reference completion, but they also do not create a
    false failure for the verified public calculators.
    """

    if int(workers) < 1:
        raise ValueError("workers must be at least 1.")
    with ZipFile(evaluator_zip) as evaluator:
        route_references = _read_jsonl_models(
            evaluator, RouteReferenceRecordV1, "grader/domains/route_references.jsonl"
        )
        scoreable_inputs = _read_jsonl_models(
            evaluator,
            RouteReferenceInputRecordV1,
            "grader/domains/route_reference_inputs.jsonl",
        )
        method_composition_records = _read_optional_jsonl_models(
            evaluator,
            MethodCompositionRecordV1,
            "grader/domains/method_composition.jsonl",
        )
        method_compositions = {
            record.route_reference_id: record for record in method_composition_records
        }
        reference_by_id = {
            row.route_reference_id: row
            for row in route_references
            if row.answer_shape != "limitation"
        }
        if len({row.route_reference_id for row in route_references}) != len(
            route_references
        ):
            raise ValueError(
                "route_references.jsonl contains duplicate route_reference_id values."
            )
        if len(method_compositions) != len(method_composition_records):
            raise ValueError(
                "method_composition.jsonl contains duplicate route_reference_id values."
            )
        if any(
            row.composed_method_id != "observed:group_sequential_adjusted"
            for row in method_composition_records
        ):
            raise ValueError(
                "method_composition.jsonl contains non-group-sequential rows."
            )

    partitions = _partition_scoreable_inputs_v1(
        scoreable_inputs=scoreable_inputs,
        reference_by_id=reference_by_id,
        workers=int(workers),
    )
    if not partitions:
        indexed_checks: tuple[
            tuple[int, tuple[PublicEvidenceNumericReferenceCheckV1, ...]], ...
        ] = ()
    elif len(partitions) == 1:
        indexed_checks = _recompute_scoreable_input_batch_v1(
            public_zip=Path(public_zip),
            indexed_inputs=partitions[0],
            reference_by_id=reference_by_id,
            method_compositions=method_compositions,
        )
    else:
        with single_threaded_numerical_process_pool(
            workers=min(int(workers), len(partitions))
        ) as executor:
            futures = tuple(
                executor.submit(
                    _recompute_scoreable_input_batch_v1,
                    public_zip=Path(public_zip),
                    indexed_inputs=partition,
                    reference_by_id=reference_by_id,
                    method_compositions=method_compositions,
                )
                for partition in partitions
            )
            indexed_checks = tuple(row for future in futures for row in future.result())
    checks = [
        check
        for _, group in sorted(indexed_checks, key=lambda row: row[0])
        for check in group
    ]

    outcome_counts = Counter(check.outcome for check in checks)
    replayed_variant_counts = Counter(check.route_reference_id for check in checks)
    route_reference_ids = set(reference_by_id)
    missing_variant_ids = route_reference_ids - set(replayed_variant_counts)
    orphan_variant_ids = set(replayed_variant_counts) - route_reference_ids
    duplicate_variant_ids = {
        route_reference_id
        for route_reference_id, count in replayed_variant_counts.items()
        if count != 1
    }
    method_counts = Counter(check.estimator_method_id for check in checks)
    method_outcome_counts: dict[str, dict[str, int]] = {}
    for method_id in sorted(set(check.estimator_method_id for check in checks)):
        method_outcome_counts[method_id] = dict(
            sorted(
                Counter(
                    check.outcome
                    for check in checks
                    if check.estimator_method_id == method_id
                ).items()
            )
        )
    unsupported_surface_shape_counts = Counter(
        check.public_surface_shape
        for check in checks
        if check.outcome == "unsupported_calculator"
    )
    unsupported_method_dispositions = _classify_unsupported_public_reference_methods(
        checks
    )
    unsupported_disposition_counts = Counter(
        row.disposition for row in unsupported_method_dispositions
    )
    drift_dispositions = _build_public_reference_drift_dispositions(
        checks=checks,
        unsupported_method_dispositions=unsupported_method_dispositions,
    )
    drift_classification_counts = Counter(
        row.classification for row in drift_dispositions
    )
    findings: list[str] = []
    if outcome_counts.get("mismatched", 0):
        findings.append("supported_numeric_reference_mismatch")
    if outcome_counts.get("missing_public_input", 0):
        findings.append("supported_numeric_reference_missing_public_input")
    if outcome_counts.get("invalid_public_input", 0):
        findings.append("supported_numeric_reference_invalid_public_input")
    if outcome_counts.get("unsupported_calculator", 0):
        findings.append("unsupported_public_numeric_calculator")
    if outcome_counts.get("not_numeric", 0):
        findings.append("non_numeric_scoreable_reference")
    if missing_variant_ids:
        findings.append("route_reference_replay_missing")
    if orphan_variant_ids:
        findings.append("route_reference_replay_orphan")
    if duplicate_variant_ids:
        findings.append("route_reference_replay_duplicate")
    if not any(
        check.estimator_method_id in SUPPORTED_PUBLIC_NUMERIC_METHODS_V1
        for check in checks
    ):
        findings.append("no_supported_numeric_reference_checks")

    return PublicEvidenceNumericReferenceReportV1(
        evaluator_zip=evaluator_zip.as_posix(),
        public_zip=public_zip.as_posix(),
        status="fail" if findings else "pass",
        route_reference_count=len(reference_by_id),
        route_reference_input_count=len(scoreable_inputs),
        supported_method_ids=tuple(sorted(SUPPORTED_PUBLIC_NUMERIC_METHODS_V1)),
        supported_check_count=sum(
            1
            for check in checks
            if check.estimator_method_id in SUPPORTED_PUBLIC_NUMERIC_METHODS_V1
        ),
        matched_count=outcome_counts.get("matched", 0),
        mismatched_count=outcome_counts.get("mismatched", 0),
        unsupported_calculator_count=outcome_counts.get("unsupported_calculator", 0),
        not_numeric_count=outcome_counts.get("not_numeric", 0),
        missing_public_input_count=outcome_counts.get("missing_public_input", 0),
        invalid_public_input_count=outcome_counts.get("invalid_public_input", 0),
        outcome_counts=dict(sorted(outcome_counts.items())),
        method_counts=dict(sorted(method_counts.items())),
        method_outcome_counts=method_outcome_counts,
        unsupported_surface_shape_counts=dict(
            sorted(unsupported_surface_shape_counts.items())
        ),
        unsupported_disposition_counts=dict(
            sorted(unsupported_disposition_counts.items())
        ),
        unsupported_method_dispositions=tuple(unsupported_method_dispositions),
        drift_classification_counts=dict(sorted(drift_classification_counts.items())),
        findings=tuple(sorted(findings)),
        checks=tuple(checks),
        drift_dispositions=tuple(drift_dispositions),
    )


def render_public_evidence_numeric_reference_report_v1(
    report: PublicEvidenceNumericReferenceReportV1,
) -> str:
    """Render a numeric reference recomputation report as Markdown."""

    lines = [
        "# Public-Evidence Numeric Reference Recompute Report",
        "",
        f"- Evaluator zip: `{report.evaluator_zip}`",
        f"- Public zip: `{report.public_zip}`",
        f"- Status: `{report.status}`",
        f"- Route references: `{report.route_reference_count}`",
        f"- Scoreable reference inputs: `{report.route_reference_input_count}`",
        f"- Supported methods: `{', '.join(report.supported_method_ids)}`",
        f"- Supported checks: `{report.supported_check_count}`",
        f"- Matched: `{report.matched_count}`",
        f"- Mismatched: `{report.mismatched_count}`",
        f"- Unsupported calculators: `{report.unsupported_calculator_count}`",
        f"- Not numeric: `{report.not_numeric_count}`",
        f"- Missing public inputs: `{report.missing_public_input_count}`",
        f"- Invalid public inputs: `{report.invalid_public_input_count}`",
        "",
        "## Outcome Counts",
        "",
    ]
    lines.extend(
        f"- `{key}`: `{value}`" for key, value in sorted(report.outcome_counts.items())
    )
    lines.extend(["", "## Method Counts", ""])
    lines.extend(
        f"- `{key}`: `{value}`" for key, value in sorted(report.method_counts.items())
    )
    lines.extend(["", "## Method Outcome Counts", ""])
    for method_id, outcomes in sorted(report.method_outcome_counts.items()):
        rendered = ", ".join(
            f"{outcome}={count}" for outcome, count in sorted(outcomes.items())
        )
        lines.append(f"- `{method_id}`: {rendered}")
    lines.extend(["", "## Unsupported Public Surface Shapes", ""])
    if report.unsupported_surface_shape_counts:
        lines.extend(
            f"- `{shape}`: `{count}`"
            for shape, count in sorted(report.unsupported_surface_shape_counts.items())
        )
    else:
        lines.append("No unsupported public surface shapes.")
    lines.extend(["", "## Unsupported Method Dispositions", ""])
    if report.unsupported_method_dispositions:
        for row in report.unsupported_method_dispositions:
            lines.append(
                f"- `{row.estimator_method_id}`: `{row.unsupported_count}` rows, "
                f"`{row.disposition}`; required action: {row.required_release_action}"
            )
    else:
        lines.append("No unsupported method dispositions.")
    lines.extend(["", "## Drift Classification Counts", ""])
    lines.extend(
        f"- `{key}`: `{value}`"
        for key, value in sorted(report.drift_classification_counts.items())
    )
    if report.findings:
        lines.extend(["", "## Findings", ""])
        lines.extend(f"- `{finding}`" for finding in report.findings)
    return "\n".join(lines) + "\n"


def write_public_evidence_numeric_reference_artifacts_v1(
    *,
    evaluator_zip: Path,
    public_zip: Path,
    out_dir: Path,
    workers: int = 1,
) -> PublicEvidenceNumericReferenceReportV1:
    """Recompute numeric public reference and write JSON/JSONL/Markdown artifacts."""

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip,
        public_zip=public_zip,
        workers=int(workers),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_model(out_dir / "public_evidence_numeric_reference_report.json", report)
    (out_dir / "public_evidence_numeric_reference_checks.jsonl").write_text(
        "".join(
            check.model_dump_json(exclude_none=True) + "\n" for check in report.checks
        ),
        encoding="utf-8",
    )
    (out_dir / "public_evidence_reference_drift_dispositions.jsonl").write_text(
        "".join(
            row.model_dump_json(exclude_none=True) + "\n"
            for row in report.drift_dispositions
        ),
        encoding="utf-8",
    )
    (out_dir / "public_evidence_numeric_reference_report.md").write_text(
        render_public_evidence_numeric_reference_report_v1(report),
        encoding="utf-8",
    )
    return report


def _recompute_one(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
    route_reference: RouteReferenceRecordV1,
    method_compositions: Mapping[str, MethodCompositionRecordV1] | None = None,
    numeric_cache: (
        dict[
            tuple[str, str, str, str, str, float | None, tuple[str, ...]],
            float | PublicNumericBoundResultV1 | PublicNumericVectorResultV1,
        ]
        | None
    ) = None,
    uncertainty_cache: (
        dict[
            tuple[str, str, str, str, str, float | None, tuple[str, ...]], float | None
        ]
        | None
    ) = None,
    km_family_cache: dict[tuple[str, ...], _KmFamilyResult] | None = None,
) -> PublicEvidenceNumericReferenceCheckV1:
    public_paths = tuple(
        public_rel_path_for_scoreable_ref_v1(ref.rel_path)
        for ref in reference_input.required_table_refs
    )
    public_roles = tuple(
        str(ref.semantic_role) for ref in reference_input.required_table_refs
    )
    public_surface_shape = public_surface_shape_for_scoreable_refs_v1(reference_input)
    base = {
        "task_id": route_reference.task_id,
        "item_id": route_reference.item_id,
        "lane_id": route_reference.lane_id,
        "route_reference_id": route_reference.route_reference_id,
        "route_reference_checksum": (
            route_reference.checksum or _canonical_record_checksum_v1(route_reference)
        ),
        "variant_role": route_reference.variant_role,
        "input_bundle_id": reference_input.input_bundle_id,
        "input_bundle_checksum": (
            reference_input.checksum or _canonical_record_checksum_v1(reference_input)
        ),
        "route_family": route_reference.route_family,
        "estimator_method_id": route_reference.estimator_method_id,
        "effect_scale": route_reference.effect_scale,
        "sensitivity_parameter": route_reference.sensitivity_parameter,
        "answer_shape": route_reference.answer_shape,
        "identification_class": route_reference.identification_class,
        "support_status": route_reference.support_status,
        "old_value": route_reference.value,
        "old_lower": route_reference.lower,
        "old_upper": route_reference.upper,
        "old_standard_error": route_reference.standard_error,
        "public_table_paths": public_paths,
        "public_table_roles": public_roles,
        "public_surface_shape": public_surface_shape,
    }
    if route_reference.estimator_method_id not in SUPPORTED_PUBLIC_NUMERIC_METHODS_V1:
        return _numeric_reference_check(
            base,
            outcome="unsupported_calculator",
            message="No standalone public numeric calculator is registered for this estimator method.",
        )
    if (
        route_reference.answer_shape in {"point", "test"}
        and route_reference.value is None
    ):
        return _numeric_reference_check(
            base,
            outcome="not_numeric",
            message="Point/test-valued route references require a scalar value.",
        )
    if route_reference.answer_shape == "bound" and (
        route_reference.lower is None or route_reference.upper is None
    ):
        return _numeric_reference_check(
            base,
            outcome="not_numeric",
            message="Bound-valued route references require lower and upper values.",
        )
    if route_reference.estimator_method_id == "observed:group_sequential_adjusted":
        return _recompute_group_sequential_composition(
            base=base,
            public=public,
            route_reference=route_reference,
            reference_input=reference_input,
            method_compositions=(
                {} if method_compositions is None else method_compositions
            ),
        )
    if route_reference.answer_shape not in {"point", "bound", "test", "vector"}:
        return _numeric_reference_check(
            base,
            outcome="not_numeric",
            message="Supported calculator only handles point, bound, scalar test, or vector route references.",
        )
    if not has_required_public_inputs_for_method_v1(
        public=public,
        reference_input=reference_input,
        method_id=route_reference.estimator_method_id,
    ):
        return _numeric_reference_check(
            base,
            outcome="unsupported_calculator",
            message=(
                "This public-surface row does not expose the ADSL/ADTTE inputs required by the registered "
                "standalone numeric calculator."
            ),
        )
    try:
        cache_key = (
            str(route_reference.item_id),
            str(route_reference.estimator_method_id),
            str(route_reference.effect_scale),
            str(route_reference.answer_shape),
            str(reference_input.source_role),
            route_reference.sensitivity_parameter,
            tuple(
                sorted(str(ref.sha256) for ref in reference_input.required_table_refs)
            ),
        )
        recomputed = None if numeric_cache is None else numeric_cache.get(cache_key)
        uncertainty_cached = (
            uncertainty_cache is not None and cache_key in uncertainty_cache
        )
        recomputed_standard_error = (
            None if uncertainty_cache is None else uncertainty_cache.get(cache_key)
        )
        if recomputed is None or not uncertainty_cached:
            recomputed, recomputed_standard_error = recompute_public_numeric_result_v1(
                public=public,
                reference_input=reference_input,
                route_reference=route_reference,
                km_family_cache=km_family_cache,
            )
            if numeric_cache is not None:
                numeric_cache[cache_key] = recomputed
            if uncertainty_cache is not None:
                uncertainty_cache[cache_key] = recomputed_standard_error
    except FileNotFoundError as exc:
        return _numeric_reference_check(
            base,
            outcome="missing_public_input",
            message=str(exc),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        pd.errors.ParserError,
        ConvergenceWarning,
    ) as exc:
        return _numeric_reference_check(
            base,
            outcome="invalid_public_input",
            message=str(exc),
        )
    if isinstance(recomputed, PublicNumericBoundResultV1):
        if route_reference.lower is None or route_reference.upper is None:
            raise ValueError(
                "Bound recomputation requires reference lower and upper values."
            )
        lower_diff = abs(float(route_reference.lower) - float(recomputed.lower))
        upper_diff = abs(float(route_reference.upper) - float(recomputed.upper))
        lower_tolerance = route_reference.numerical_equivalence.tolerance(
            expected=float(route_reference.lower), observed=float(recomputed.lower)
        )
        upper_tolerance = route_reference.numerical_equivalence.tolerance(
            expected=float(route_reference.upper), observed=float(recomputed.upper)
        )
        tolerance = max(lower_tolerance, upper_tolerance)
        matched = bool(lower_diff <= lower_tolerance and upper_diff <= upper_tolerance)
        return _numeric_reference_check(
            base,
            outcome="matched" if matched else "mismatched",
            recomputed_lower=float(recomputed.lower),
            recomputed_upper=float(recomputed.upper),
            lower_abs_diff=float(lower_diff),
            upper_abs_diff=float(upper_diff),
            abs_tolerance=float(tolerance),
            message="ok" if matched else "bound_mismatch",
        )
    if isinstance(recomputed, PublicNumericVectorResultV1):
        old_by_component = {
            component.component_id: float(component.value)
            for component in route_reference.vector_components
        }
        new_by_component = {
            component.component_id: float(component.value)
            for component in recomputed.components
        }
        if set(old_by_component) != set(new_by_component):
            return _numeric_reference_check(
                base,
                outcome="mismatched",
                vector_component_count=len(new_by_component),
                recomputed_vector_values=new_by_component,
                abs_tolerance=float(
                    route_reference.numerical_equivalence.absolute_tolerance
                ),
                message="vector_component_id_mismatch",
            )
        diffs = {
            component_id: abs(
                float(old_by_component[component_id])
                - float(new_by_component[component_id])
            )
            for component_id in sorted(old_by_component)
        }
        tolerances = {
            component_id: route_reference.numerical_equivalence.tolerance(
                expected=float(old_by_component[component_id]),
                observed=float(new_by_component[component_id]),
            )
            for component_id in sorted(old_by_component)
        }
        max_diff = max(diffs.values(), default=0.0)
        tolerance = max(
            tolerances.values(),
            default=route_reference.numerical_equivalence.absolute_tolerance,
        )
        matched = all(
            diffs[component_id] <= tolerances[component_id] for component_id in diffs
        )
        return _numeric_reference_check(
            base,
            outcome="matched" if matched else "mismatched",
            vector_component_count=len(new_by_component),
            vector_max_abs_diff=float(max_diff),
            recomputed_vector_values=new_by_component,
            abs_tolerance=float(tolerance),
            message="ok" if matched else "vector_value_mismatch",
        )
    if route_reference.value is None:
        raise ValueError("Point recomputation requires reference scalar value.")
    diff = abs(float(route_reference.value) - float(recomputed))
    tolerance = route_reference.numerical_equivalence.tolerance(
        expected=float(route_reference.value), observed=float(recomputed)
    )
    point_matched = bool(diff <= tolerance)
    standard_error_diff: float | None = None
    standard_error_tolerance: float | None = None
    uncertainty_matched = True
    if recomputed_standard_error is not None:
        if route_reference.standard_error is None:
            uncertainty_matched = False
        else:
            standard_error_diff = abs(
                float(route_reference.standard_error) - float(recomputed_standard_error)
            )
            standard_error_tolerance = route_reference.numerical_equivalence.tolerance(
                expected=float(route_reference.standard_error),
                observed=float(recomputed_standard_error),
            )
            uncertainty_matched = bool(standard_error_diff <= standard_error_tolerance)
    matched = bool(point_matched and uncertainty_matched)
    return _numeric_reference_check(
        base,
        outcome="matched" if matched else "mismatched",
        recomputed_value=float(recomputed),
        abs_diff=float(diff),
        abs_tolerance=float(tolerance),
        recomputed_standard_error=recomputed_standard_error,
        standard_error_abs_diff=standard_error_diff,
        standard_error_abs_tolerance=standard_error_tolerance,
        message=(
            "ok"
            if matched
            else ("value_mismatch" if not point_matched else "standard_error_mismatch")
        ),
    )


def _recompute_group_sequential_composition(
    *,
    base: Mapping[str, object],
    public: ZipFile,
    route_reference: RouteReferenceRecordV1,
    reference_input: RouteReferenceInputRecordV1,
    method_compositions: Mapping[str, MethodCompositionRecordV1],
) -> PublicEvidenceNumericReferenceCheckV1:
    if route_reference.answer_shape != "point":
        return _numeric_reference_check(
            base,
            outcome="not_numeric",
            message="Group-sequential public replay requires a scalar point route reference.",
        )
    if (
        route_reference.value is None
        or route_reference.ci_low is None
        or route_reference.ci_high is None
    ):
        return _numeric_reference_check(
            base,
            outcome="not_numeric",
            message="Group-sequential route references require a value and replay confidence limits.",
        )
    record = method_compositions.get(route_reference.route_reference_id)
    if record is None:
        return _numeric_reference_check(
            base,
            outcome="unsupported_calculator",
            message="No method-composition record is exposed for this group-sequential route reference.",
        )
    base = {
        **base,
        "method_composition_checksum": (
            record.checksum or _canonical_record_checksum_v1(record)
        ),
    }
    if (
        record.task_id != route_reference.task_id
        or record.lane_id != route_reference.lane_id
    ):
        return _numeric_reference_check(
            base,
            outcome="invalid_public_input",
            message="Method-composition task/lane metadata does not match the route reference.",
        )
    if record.composed_method_id != route_reference.estimator_method_id:
        return _numeric_reference_check(
            base,
            outcome="invalid_public_input",
            message="Method-composition composed method does not match the route reference.",
        )
    try:
        protocol = _read_json_from_public(
            public, f"items/{reference_input.task_id}/protocol_summary.json"
        )
    except FileNotFoundError:
        return _numeric_reference_check(
            base,
            outcome="missing_public_input",
            message="Group-sequential public replay requires protocol_summary.json.",
        )
    if not isinstance(protocol, dict) or not isinstance(
        protocol.get("group_sequential_plan"), dict
    ):
        return _numeric_reference_check(
            base,
            outcome="invalid_public_input",
            message="Public protocol does not declare a group-sequential plan.",
        )
    plan = cast(dict[str, object], protocol["group_sequential_plan"])
    parameters = record.adjustment_parameters
    try:
        plan_family = str(plan["spending_function_id"])
        plan_fractions = tuple(
            float(value) for value in cast(list[str | int | float], plan["looks"])
        )
        plan_alpha = float(cast(str | int | float, plan["two_sided_alpha"]))
        plan_boundaries = tuple(
            float(value)
            for value in cast(list[str | int | float], plan["z_critical_by_look"])
        )
        plan_analysis_look_index = int(
            cast(str | int | float, plan["analysis_look_index"])
        )
    except (KeyError, TypeError, ValueError):
        return _numeric_reference_check(
            base,
            outcome="invalid_public_input",
            message="Public group-sequential plan is incomplete or malformed.",
        )
    if (
        plan_family != parameters.spending_family
        or plan_fractions != parameters.information_fractions
        or abs(plan_alpha - parameters.total_alpha) > 1e-12
        or plan_analysis_look_index != parameters.analysis_look_index
        or len(plan_boundaries) != len(parameters.z_critical_by_look)
        or any(
            abs(public_value - composed_value) > 1e-12
            for public_value, composed_value in zip(
                plan_boundaries, parameters.z_critical_by_look, strict=True
            )
        )
    ):
        return _numeric_reference_check(
            base,
            outcome="invalid_public_input",
            message="Method composition disagrees with the participant-visible group-sequential plan.",
        )
    if (
        record.adjustment_parameters.z_value <= 0.0
        or record.adjustment_parameters.look_count < 2
    ):
        return _numeric_reference_check(
            base,
            outcome="invalid_public_input",
            message="Method-composition record does not declare a valid multi-look alpha-spending adjustment.",
        )
    base_route_reference = route_reference.model_copy(
        update={
            "estimator_method_id": record.base_estimator_method_id,
            "answer_shape": "point",
            "lower": None,
            "upper": None,
        }
    )
    base_reference_input = reference_input.model_copy(
        update={"estimator_method_id": record.base_estimator_method_id}
    )
    try:
        replayed_base_value, replayed_base_standard_error = (
            recompute_public_numeric_result_v1(
                public=public,
                reference_input=base_reference_input,
                route_reference=base_route_reference,
            )
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        pd.errors.ParserError,
        ConvergenceWarning,
    ) as exc:
        return _numeric_reference_check(
            base,
            outcome="invalid_public_input",
            message=f"Unable to replay group-sequential base estimator: {exc}",
        )
    if (
        not isinstance(replayed_base_value, float | np.floating)
        or replayed_base_standard_error is None
    ):
        return _numeric_reference_check(
            base,
            outcome="invalid_public_input",
            message="Group-sequential base estimator did not replay to a scalar point and standard error.",
        )
    tolerance = route_reference.numerical_equivalence.tolerance(
        expected=float(route_reference.value), observed=float(replayed_base_value)
    )
    value_diff = max(
        abs(float(route_reference.value) - float(record.base_value)),
        abs(float(record.base_value) - float(replayed_base_value)),
    )
    lower_diff = abs(float(route_reference.ci_low) - float(record.adjusted_lower))
    upper_diff = abs(float(route_reference.ci_high) - float(record.adjusted_upper))
    standard_error_diff = abs(
        float(record.base_standard_error) - float(replayed_base_standard_error)
    )
    standard_error_tolerance = route_reference.numerical_equivalence.tolerance(
        expected=float(record.base_standard_error),
        observed=float(replayed_base_standard_error),
    )
    implied_lower = float(record.base_value) - float(
        record.adjustment_parameters.z_value
    ) * float(record.base_standard_error)
    implied_upper = float(record.base_value) + float(
        record.adjustment_parameters.z_value
    ) * float(record.base_standard_error)
    composition_diff = max(
        abs(float(record.adjusted_lower) - implied_lower),
        abs(float(record.adjusted_upper) - implied_upper),
    )
    matched = bool(
        value_diff <= tolerance
        and lower_diff <= tolerance
        and upper_diff <= tolerance
        and composition_diff <= tolerance
        and standard_error_diff <= standard_error_tolerance
    )
    return _numeric_reference_check(
        base,
        outcome="matched" if matched else "mismatched",
        recomputed_value=float(replayed_base_value),
        recomputed_lower=float(record.adjusted_lower),
        recomputed_upper=float(record.adjusted_upper),
        recomputed_standard_error=float(replayed_base_standard_error),
        standard_error_abs_diff=float(standard_error_diff),
        standard_error_abs_tolerance=float(standard_error_tolerance),
        abs_diff=float(value_diff),
        lower_abs_diff=float(lower_diff),
        upper_abs_diff=float(upper_diff),
        abs_tolerance=float(tolerance),
        message="ok" if matched else "method_composition_mismatch",
    )


def _classify_unsupported_public_reference_methods(
    checks: list[PublicEvidenceNumericReferenceCheckV1],
) -> list[UnsupportedPublicReferenceMethodDispositionV1]:
    rows: list[UnsupportedPublicReferenceMethodDispositionV1] = []
    unsupported_methods = sorted(
        set(
            check.estimator_method_id
            for check in checks
            if check.outcome == "unsupported_calculator"
        )
    )
    for method_id in unsupported_methods:
        method_checks = [
            check
            for check in checks
            if check.estimator_method_id == method_id
            and check.outcome == "unsupported_calculator"
        ]
        surface_shape_counts = dict(
            sorted(
                Counter(check.public_surface_shape for check in method_checks).items()
            )
        )
        if method_id in SUPPORTED_PUBLIC_NUMERIC_METHODS_V1:
            disposition: UnsupportedReferenceDispositionV1 = (
                "input_surface_insufficient_for_registered_calculator"
            )
            required_action = (
                "Expose the public ADSL/ADTTE or subject-operational-flags surface required by the registered "
                "standalone calculator, or register a separate public reconstruction protocol for this shape."
            )
        else:
            disposition, required_action = _UNSUPPORTED_METHOD_DISPOSITIONS_V1.get(
                method_id,
                (
                    "unregistered_public_calculator",
                    "Register a standalone public calculator or mark the method as non-numeric/non-replayable.",
                ),
            )
        rows.append(
            UnsupportedPublicReferenceMethodDispositionV1(
                estimator_method_id=method_id,
                unsupported_count=len(method_checks),
                disposition=disposition,
                required_release_action=required_action,
                surface_shape_counts=surface_shape_counts,
            )
        )
    return rows


def _build_public_reference_drift_dispositions(
    *,
    checks: list[PublicEvidenceNumericReferenceCheckV1],
    unsupported_method_dispositions: list[
        UnsupportedPublicReferenceMethodDispositionV1
    ],
) -> list[PublicEvidenceReferenceDriftDispositionRowV1]:
    unsupported_by_method = {
        row.estimator_method_id: row for row in unsupported_method_dispositions
    }
    rows: list[PublicEvidenceReferenceDriftDispositionRowV1] = []
    for check in checks:
        classification: PublicReferenceDriftClassificationV1
        disposition: UnsupportedReferenceDispositionV1 | None = None
        required_release_action: str | None = None
        if check.outcome == "matched":
            classification = "no_drift"
        elif check.outcome == "unsupported_calculator":
            classification = "blocked_derivation_gap"
            unsupported = unsupported_by_method.get(check.estimator_method_id)
            if unsupported is not None:
                disposition = unsupported.disposition
                required_release_action = unsupported.required_release_action
        elif check.outcome == "mismatched":
            classification = "supported_reference_mismatch"
        elif check.outcome == "missing_public_input":
            classification = "missing_public_input"
        elif check.outcome == "invalid_public_input":
            classification = "invalid_public_input"
        elif check.outcome == "not_numeric":
            classification = "not_numeric_reference"
        else:  # pragma: no cover
            raise ValueError(
                f"Unhandled public numeric reference outcome: {check.outcome!r}"
            )
        rows.append(
            PublicEvidenceReferenceDriftDispositionRowV1(
                task_id=check.task_id,
                item_id=check.item_id,
                lane_id=check.lane_id,
                route_reference_id=check.route_reference_id,
                variant_role=check.variant_role,
                route_family=check.route_family,
                estimator_method_id=check.estimator_method_id,
                effect_scale=check.effect_scale,
                answer_shape=check.answer_shape,
                identification_class=check.identification_class,
                old_value=check.old_value,
                old_lower=check.old_lower,
                old_upper=check.old_upper,
                new_value=check.recomputed_value,
                new_lower=check.recomputed_lower,
                new_upper=check.recomputed_upper,
                abs_diff=check.abs_diff,
                lower_abs_diff=check.lower_abs_diff,
                upper_abs_diff=check.upper_abs_diff,
                vector_component_count=check.vector_component_count,
                vector_max_abs_diff=check.vector_max_abs_diff,
                new_vector_values=check.recomputed_vector_values,
                abs_tolerance=check.abs_tolerance,
                classification=classification,
                disposition=disposition,
                required_release_action=required_release_action,
                public_surface_shape=check.public_surface_shape,
                message=check.message,
            )
        )
    return rows


def _read_jsonl_models(
    zf: ZipFile, model: type[_ModelT], member: str
) -> tuple[_ModelT, ...]:
    try:
        raw = zf.read(member).decode("utf-8")
    except KeyError as exc:
        raise FileNotFoundError(f"Missing evaluator bundle member: {member}") from exc
    records: list[_ModelT] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JSONL row at {member}:{line_number}.")
        records.append(model.model_validate(payload))
    return tuple(records)


def _read_optional_jsonl_models(
    zf: ZipFile, model: type[_ModelT], member: str
) -> tuple[_ModelT, ...]:
    try:
        return _read_jsonl_models(zf, model, member)
    except FileNotFoundError:
        return tuple()
