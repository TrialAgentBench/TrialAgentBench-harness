"""Contracts for public TrialAgentBench worked examples."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DemonstrationArtifactV1(_ContractModel):
    """One release member used by a worked example."""

    role: Literal["participant", "evaluator", "verification"]
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _safe_path(self) -> DemonstrationArtifactV1:
        parts = self.path.split("/")
        if (
            "\\" in self.path
            or self.path.startswith("/")
            or ".." in parts
            or any(not part for part in parts)
        ):
            raise ValueError("demonstration artifact paths must be safe relative paths")
        return self


class DemonstrationDiagnosticV1(_ContractModel):
    """Participant-visible diagnostic and its bounded interpretation."""

    diagnostic_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    value: float | None = None
    unit: str | None = Field(default=None, min_length=1)
    interpretation: str = Field(min_length=1)
    evidence_paths: tuple[str, ...] = Field(min_length=1)


class DemonstrationRouteV1(_ContractModel):
    """One accepted or excluded analysis route shown in the example."""

    unit_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    disposition: Literal["credit_eligible", "excluded"]
    estimator_family: str = Field(min_length=1)
    effect_scale: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    result_summary: str | None = Field(default=None, min_length=1)


class DemonstrationRecoverabilityKeyV1(_ContractModel):
    """Exact release route that an independent worked example must recover."""

    unit_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)


class DemonstrationCaseV1(_ContractModel):
    """Canonical product record for one verified worked example."""

    schema_id: Literal["trialagentbench.demonstration_case/v1"] = (
        "trialagentbench.demonstration_case/v1"
    )
    release_id: str = Field(min_length=1)
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    suite: Literal["trialeval", "trialdev"]
    disclosure_status: Literal["demonstration_only", "exposed"]
    evaluated_unit_id: str | None = Field(default=None, min_length=1)
    title: str = Field(min_length=1)
    question: str = Field(min_length=1)
    evidence_boundary: str = Field(min_length=1)
    trial_structure: tuple[str, ...] = Field(min_length=1)
    artifacts: tuple[DemonstrationArtifactV1, ...] = Field(min_length=1)
    diagnostics: tuple[DemonstrationDiagnosticV1, ...] = ()
    routes: tuple[DemonstrationRouteV1, ...] = Field(min_length=1)
    required_recoverability_routes: tuple[DemonstrationRecoverabilityKeyV1, ...] = (
        Field(min_length=1)
    )
    consequence: str = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_case(self) -> DemonstrationCaseV1:
        if self.disclosure_status == "exposed" and self.evaluated_unit_id is None:
            raise ValueError("exposed demonstrations must identify the evaluated unit")
        if (
            self.disclosure_status == "demonstration_only"
            and self.evaluated_unit_id is not None
        ):
            raise ValueError(
                "demonstration-only cases cannot identify a headline evaluation unit"
            )
        artifact_keys = tuple((row.role, row.path) for row in self.artifacts)
        if len(set(artifact_keys)) != len(artifact_keys):
            raise ValueError("demonstration artifacts must be unique by role and path")
        route_keys = tuple((route.unit_id, route.route_id) for route in self.routes)
        if len(set(route_keys)) != len(route_keys):
            raise ValueError("demonstration routes must be unique by unit and route ID")
        accepted = {
            (route.unit_id, route.route_id)
            for route in self.routes
            if route.disposition == "credit_eligible"
        }
        required_keys = tuple(
            (route.unit_id, route.route_id)
            for route in self.required_recoverability_routes
        )
        if len(set(required_keys)) != len(required_keys):
            raise ValueError("recoverability route keys must be unique")
        required = set(required_keys)
        if not required <= accepted:
            raise ValueError(
                "recoverability routes must be credit-eligible routes in the case record"
            )
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        observed = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if self.checksum is not None and self.checksum != observed:
            raise ValueError("demonstration case checksum is invalid")
        object.__setattr__(self, "checksum", observed)
        return self


class DemonstrationIndexEntryV1(_ContractModel):
    """One checksummed case in the public demonstration index."""

    case_id: str = Field(min_length=1)
    suite: Literal["trialeval", "trialdev"]
    disclosure_status: Literal["demonstration_only", "exposed"]
    case_path: str = Field(min_length=1)
    case_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class DemonstrationIndexV1(_ContractModel):
    """Immutable demonstration inventory for one benchmark release."""

    schema_id: Literal["trialagentbench.demonstration_index/v1"] = (
        "trialagentbench.demonstration_index/v1"
    )
    release_id: str = Field(min_length=1)
    cases: tuple[DemonstrationIndexEntryV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_cases(self) -> DemonstrationIndexV1:
        case_ids = tuple(row.case_id for row in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("demonstration case IDs must be unique")
        paths = tuple(row.case_path for row in self.cases)
        if len(set(paths)) != len(paths):
            raise ValueError("demonstration case paths must be unique")
        return self


class DemonstrationVerificationReportV1(_ContractModel):
    """Outcome of offline worked-example verification and record export."""

    schema_id: Literal["trialagentbench.demonstration_verification/v1"] = (
        "trialagentbench.demonstration_verification/v1"
    )
    release_id: str
    requested_case_ids: tuple[str, ...]
    verified_case_ids: tuple[str, ...]
    recoverability_status: Literal["pass", "fail"]
    status: Literal["pass", "fail"]


__all__ = [
    "DemonstrationArtifactV1",
    "DemonstrationCaseV1",
    "DemonstrationDiagnosticV1",
    "DemonstrationIndexEntryV1",
    "DemonstrationIndexV1",
    "DemonstrationRecoverabilityKeyV1",
    "DemonstrationVerificationReportV1",
    "DemonstrationRouteV1",
]
