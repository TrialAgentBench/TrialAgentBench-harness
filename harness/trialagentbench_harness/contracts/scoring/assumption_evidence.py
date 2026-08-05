"""Evaluator-side contracts for generated assumption evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AssumptionStatusV1 = Literal["holds", "stressed", "broken"]
AssumptionSeverityBandV1 = Literal["holds", "mild", "fragile", "broken"]
AssumptionDiagnosabilityV1 = Literal[
    "directly_diagnosable",
    "partially_diagnosable",
    "design_declared",
    "not_identifiable",
]


def _validate_public_evidence_path(path: str) -> str:
    value = str(path)
    pure = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or pure.is_absolute()
        or value != pure.as_posix()
        or any(part in {"", ".", ".."} or part.startswith(".") for part in pure.parts)
        or ":" in value
        or {"hidden", "grader"} & {part.lower() for part in pure.parts}
    ):
        raise ValueError(f"Invalid participant-relative public evidence path: {value!r}.")
    return value


class AssumptionEvidenceRecordV1(BaseModel):
    """One evaluator-side assumption-evidence record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assumption_id: str = Field(..., min_length=1)
    expected_status: AssumptionStatusV1
    computed_status: AssumptionStatusV1
    expected_band: AssumptionSeverityBandV1
    computed_band: AssumptionSeverityBandV1
    diagnosability: AssumptionDiagnosabilityV1
    severity_metric: float | None = None
    severity_metric_name: str | None = Field(default=None, min_length=1)
    threshold_stressed: float | None = Field(default=None, ge=0.0)
    threshold_fragile: float | None = Field(default=None, ge=0.0)
    threshold_broken: float | None = Field(default=None, ge=0.0)
    decision_metric_names: dict[Literal["stressed", "fragile", "broken"], str] = Field(default_factory=dict)
    supporting_metrics: dict[str, float] = Field(default_factory=dict)
    metric_units: dict[str, str] = Field(...)
    metric_public_evidence_basis: dict[str, tuple[str, ...]] = Field(...)
    factual_public_evidence_basis: tuple[str, ...] | None = None
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_bands(self) -> AssumptionEvidenceRecordV1:
        """Reject inconsistent severity thresholds and status bands."""

        numeric_fields = (
            self.severity_metric,
            self.severity_metric_name,
            self.threshold_stressed,
            self.threshold_fragile,
            self.threshold_broken,
        )
        if all(value is None for value in numeric_fields):
            if self.diagnosability not in {"design_declared", "not_identifiable"}:
                raise ValueError(
                    "Only design-declared or nonidentified obligations may omit empirical severity fields."
                )
            if self.decision_metric_names:
                raise ValueError("Non-empirical obligations cannot contain empirical decision metrics.")
            if not self.metric_public_evidence_basis and not self.factual_public_evidence_basis:
                raise ValueError("A non-empirical obligation requires a public factual evidence basis.")
        elif any(value is None for value in numeric_fields):
            raise ValueError("Assumption evidence must provide all empirical severity fields or none of them.")
        else:
            assert self.threshold_stressed is not None
            assert self.threshold_fragile is not None
            assert self.threshold_broken is not None
            if not self.threshold_stressed <= self.threshold_fragile <= self.threshold_broken:
                raise ValueError("Assumption-evidence thresholds must be ordered.")
            if set(self.decision_metric_names) != {"stressed", "fragile", "broken"}:
                raise ValueError(
                    "Empirical assumption evidence requires decision metrics for "
                    "stressed, fragile, and broken thresholds."
                )
        status_by_band = {"holds": "holds", "mild": "stressed", "fragile": "stressed", "broken": "broken"}
        if status_by_band[self.expected_band] != self.expected_status:
            raise ValueError("expected_band does not match expected_status.")
        if status_by_band[self.computed_band] != self.computed_status:
            raise ValueError("computed_band does not match computed_status.")
        metric_names = set(self.supporting_metrics)
        if self.severity_metric_name is not None:
            metric_names.add(self.severity_metric_name)
        unknown_decision_metrics = sorted(set(self.decision_metric_names.values()) - metric_names)
        if unknown_decision_metrics:
            raise ValueError(
                f"Assumption decision metrics must reference emitted public metrics: {unknown_decision_metrics!r}."
            )
        if set(self.metric_units) != metric_names or set(self.metric_public_evidence_basis) != metric_names:
            raise ValueError(
                "metric_units and metric_public_evidence_basis must exactly cover every generated metric."
            )
        if any(not str(unit) for unit in self.metric_units.values()):
            raise ValueError("Every assumption metric requires a non-empty unit.")
        for paths in self.metric_public_evidence_basis.values():
            if not paths:
                raise ValueError("Every assumption metric requires a public evidence basis.")
            for path in paths:
                _validate_public_evidence_path(path)
        factual_paths = tuple(sorted(set(self.factual_public_evidence_basis or ())))
        for path in factual_paths:
            _validate_public_evidence_path(path)
        if self.factual_public_evidence_basis is not None:
            object.__setattr__(self, "factual_public_evidence_basis", factual_paths)
        if self.severity_metric is not None and factual_paths:
            raise ValueError("Empirical assumption evidence must not declare a separate factual evidence basis.")
        return self

    def decision_metric(self, threshold_id: Literal["stressed", "fragile", "broken"]) -> tuple[str, float, float]:
        """Return the evaluator metric and boundary for one decision threshold."""

        if not self.decision_metric_names:
            raise ValueError("Design-declared assumption evidence has no empirical decision metric.")
        metric_name = self.decision_metric_names[threshold_id]
        metric_value = (
            self.severity_metric
            if metric_name == self.severity_metric_name
            else self.supporting_metrics.get(metric_name)
        )
        threshold_value = {
            "stressed": self.threshold_stressed,
            "fragile": self.threshold_fragile,
            "broken": self.threshold_broken,
        }[threshold_id]
        if metric_value is None or threshold_value is None:  # pragma: no cover - model validation is exhaustive
            raise ValueError(f"Incomplete empirical decision metric for threshold {threshold_id!r}.")
        return metric_name, float(metric_value), float(threshold_value)


class AssumptionEvidenceManifestV1(BaseModel):
    """One item-level generated assumption-evidence manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"]
    schema_id: Literal["trial_benchmark_assumption_evidence_manifest_v1"]
    item_id: str = Field(..., min_length=1)
    base_case_id: str = Field(..., min_length=1)
    canonical_item_id: str = Field(..., min_length=1)
    variant_id: str = Field(..., min_length=1)
    context_tier: str = Field(..., min_length=1)
    replicate_index: int = Field(..., ge=0)
    records: tuple[AssumptionEvidenceRecordV1, ...] = Field(..., min_length=1)
    checksum: str = Field(..., min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_manifest(self) -> AssumptionEvidenceManifestV1:
        """Reject duplicate assumptions and a non-canonical embedded digest."""

        assumption_ids = tuple(record.assumption_id for record in self.records)
        if len(set(assumption_ids)) != len(assumption_ids):
            raise ValueError("Assumption-evidence records must be unique by assumption_id.")
        if assumption_ids != tuple(sorted(assumption_ids)):
            raise ValueError("Assumption-evidence records must use canonical assumption_id order.")
        payload = self.model_dump(mode="json", exclude={"checksum"}, exclude_none=True)
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if self.checksum != hashlib.sha256(canonical).hexdigest():
            raise ValueError("Assumption-evidence manifest checksum mismatch.")
        return self


def read_assumption_evidence_domains(*, release_root: Path) -> dict[str, AssumptionEvidenceManifestV1]:
    """Load item-level assumption evidence from an evaluator release."""

    path = Path(release_root) / "grader" / "domains" / "assumption_evidence.jsonl"
    manifests: dict[str, AssumptionEvidenceManifestV1] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("domain") != "assumption_evidence":
            raise ValueError(f"Invalid assumption-evidence domain row at {path}:{line_number}.")
        task_id = row.get("task_id")
        payload = row.get("payload")
        if not isinstance(task_id, str) or not task_id or not isinstance(payload, dict):
            raise ValueError(f"Malformed assumption-evidence domain row at {path}:{line_number}.")
        manifest_payload = payload.get("manifest")
        if not isinstance(manifest_payload, dict):
            raise ValueError(f"Assumption-evidence row lacks a manifest at {path}:{line_number}.")
        if task_id in manifests:
            raise ValueError(f"Duplicate assumption-evidence task_id: {task_id!r}.")
        manifests[task_id] = AssumptionEvidenceManifestV1.model_validate(manifest_payload)
    if not manifests:
        raise ValueError("Assumption-evidence domain cannot be empty.")
    return manifests


__all__ = [
    "AssumptionEvidenceManifestV1",
    "AssumptionEvidenceRecordV1",
    "read_assumption_evidence_domains",
]
