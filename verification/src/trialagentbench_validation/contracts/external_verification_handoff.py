"""Neutral handoff contract for independently verified scientific results."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExternalVerificationResultV1(_FrozenModel):
    """One content-addressed scientific result available to downstream builds."""

    evidence_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]+$")
    scientific_family: str = Field(min_length=1)
    artifact_manifest_path: str = Field(min_length=1)
    artifact_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_status: Literal[
        "qualified", "qualified_with_estimator_limitation", "unsupported"
    ]
    reproducibility_class: Literal[
        "public_replayable", "credentialed_reacquirable", "derived_only"
    ]
    supported_scope: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = ()
    affected_components: tuple[str, ...] = Field(min_length=1)
    required_qualification_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical(self) -> ExternalVerificationResultV1:
        path = PurePosixPath(self.artifact_manifest_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.name != "artifact_manifest.json"
        ):
            raise ValueError(
                "artifact_manifest_path must be a confined relative artifact manifest"
            )
        for name in (
            "supported_scope",
            "limitations",
            "affected_components",
            "required_qualification_ids",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if self.result_status == "unsupported" and not self.limitations:
            raise ValueError("unsupported results require a scientific limitation")
        return self


class ExternalVerificationResultHandoffV1(_FrozenModel):
    """Checksum-bound inventory of reusable external verification results."""

    schema_id: Literal["trialagentbench.external_verification_result_handoff/v1"] = (
        "trialagentbench.external_verification_result_handoff/v1"
    )
    candidate_id: str = Field(pattern=r"^evh_[0-9a-f]{20}$")
    validation_package_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    results: tuple[ExternalVerificationResultV1, ...] = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _canonical_and_checksummed(self) -> ExternalVerificationResultHandoffV1:
        evidence_ids = tuple(row.evidence_id for row in self.results)
        if evidence_ids != tuple(sorted(evidence_ids)) or len(set(evidence_ids)) != len(
            evidence_ids
        ):
            raise ValueError("handoff results must be sorted and unique by evidence_id")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        if hashlib.sha256(encoded).hexdigest() != self.checksum:
            raise ValueError("external verification handoff checksum mismatch")
        return self


__all__ = [
    "ExternalVerificationResultHandoffV1",
    "ExternalVerificationResultV1",
]
