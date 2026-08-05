"""Contracts for finite-census validation of one released benchmark candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.contracts.simulation_validation_bundle import (
    ValidationArtifactV1,
    ValidationFigureV1,
)
from trialagentbench_validation.external.release.artifacts import (
    ExternalArtifactManifestV1,
)
from trialagentbench_validation.io import sha256_file


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CandidateRoleArchiveV1(_Contract):
    """Identity of one role-separated benchmark archive."""

    suite: Literal["trialeval", "trialdev"]
    role: Literal["participant", "evaluator", "verification"]
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _safe_path(self) -> CandidateRoleArchiveV1:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("candidate role archive paths must be safe and relative")
        return self


class CandidatePublicWheelV1(_Contract):
    """Identity of one public wheel bound to the candidate."""

    package: Literal[
        "trialagentbench-harness",
        "trialagentbench-validation",
    ]
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _safe_path(self) -> CandidatePublicWheelV1:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".whl":
            raise ValueError("candidate wheel paths must be safe relative wheel paths")
        return self


class CandidateIdentityV1(_Contract):
    """Immutable identity of one finite TrialAgentBench candidate."""

    schema_id: Literal["trialagentbench.candidate_identity/v1"] = (
        "trialagentbench.candidate_identity/v1"
    )
    release_id: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    environment_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    staged_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    materialization_census_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    root_seed: int = Field(ge=0)
    trialeval_item_count: int = Field(ge=1)
    trialdev_scenario_count: int = Field(ge=1)
    role_archives: tuple[CandidateRoleArchiveV1, ...] = Field(
        min_length=6, max_length=6
    )
    public_wheels: tuple[CandidatePublicWheelV1, ...] = Field(
        min_length=2, max_length=2
    )
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _canonical_and_checksummed(self) -> CandidateIdentityV1:
        roles = tuple(sorted(self.role_archives, key=lambda row: (row.suite, row.role)))
        if roles != self.role_archives:
            raise ValueError("candidate role archives must be sorted by suite and role")
        identities = tuple((row.suite, row.role) for row in roles)
        expected = tuple(
            sorted(
                (suite, role)
                for suite in ("trialeval", "trialdev")
                for role in ("participant", "evaluator", "verification")
            )
        )
        if identities != expected:
            raise ValueError(
                "candidate identity requires every suite and role exactly once"
            )
        wheels = tuple(sorted(self.public_wheels, key=lambda row: row.package))
        if wheels != self.public_wheels:
            raise ValueError("candidate public wheels must be sorted by package")
        if tuple(row.package for row in wheels) != (
            "trialagentbench-harness",
            "trialagentbench-validation",
        ):
            raise ValueError(
                "candidate identity requires the harness and validation wheels exactly once"
            )
        payload = self.model_dump(mode="json", exclude={"checksum"})
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        if hashlib.sha256(encoded).hexdigest() != self.checksum:
            raise ValueError("candidate identity checksum mismatch")
        return self


class CandidateValidationBundleV1(_Contract):
    """Checksum-bound finite-release analysis and visual bundle."""

    schema_id: Literal["trialagentbench.candidate_validation_bundle/v1"] = (
        "trialagentbench.candidate_validation_bundle/v1"
    )
    candidate: CandidateIdentityV1
    verifier_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    figures: tuple[ValidationFigureV1, ...] = Field(min_length=1)
    methods: ValidationArtifactV1
    report: ValidationArtifactV1
    results: ValidationArtifactV1
    sources: ValidationArtifactV1
    exact_membership_manifest: ValidationArtifactV1
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _canonical_and_checksummed(self) -> CandidateValidationBundleV1:
        figure_ids = tuple(figure.figure_id for figure in self.figures)
        if figure_ids != tuple(sorted(figure_ids)) or len(set(figure_ids)) != len(
            figure_ids
        ):
            raise ValueError("candidate figures must be sorted and unique")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        if hashlib.sha256(encoded).hexdigest() != self.checksum:
            raise ValueError("candidate validation bundle checksum mismatch")
        return self


def verify_candidate_validation_bundle(
    root: Path,
    bundle: CandidateValidationBundleV1,
) -> None:
    """Verify every content-addressed candidate-analysis artifact."""

    resolved_root = root.resolve()
    artifacts = [
        bundle.methods,
        bundle.report,
        bundle.results,
        bundle.sources,
        bundle.exact_membership_manifest,
    ]
    artifacts.extend(
        artifact for figure in bundle.figures for artifact in figure.artifacts
    )
    paths = [artifact.relative_path for artifact in artifacts]
    if len(paths) != len(set(paths)):
        raise ValueError(
            "candidate validation bundle references an artifact more than once"
        )
    for artifact in artifacts:
        path = (resolved_root / artifact.relative_path).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError(
                f"candidate artifact escapes its bundle root: {artifact.relative_path!r}"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"candidate artifact is missing: {artifact.relative_path!r}"
            )
        if sha256_file(path) != artifact.sha256:
            raise ValueError(
                f"candidate artifact checksum mismatch: {artifact.relative_path!r}"
            )
    manifest = ExternalArtifactManifestV1.model_validate_json(
        (resolved_root / bundle.exact_membership_manifest.relative_path).read_text(
            encoding="utf-8"
        )
    )
    expected = {row.relative_path for row in manifest.artifacts}
    observed = {
        path.relative_to(resolved_root).as_posix()
        for path in resolved_root.rglob("*")
        if path.is_file()
        and path.relative_to(resolved_root).as_posix()
        not in {
            "candidate_validation_bundle.json",
            bundle.exact_membership_manifest.relative_path,
        }
    }
    if observed != expected:
        raise ValueError(
            "candidate analysis membership differs from its manifest: "
            f"missing={sorted(expected - observed)!r}, unexpected={sorted(observed - expected)!r}"
        )
    for digest in manifest.artifacts:
        path = resolved_root / digest.relative_path
        if (
            sha256_file(path) != digest.sha256
            or path.stat().st_size != digest.byte_size
        ):
            raise ValueError(
                "candidate exact-membership identity mismatch: "
                f"{digest.relative_path!r}"
            )


__all__ = [
    "CandidateIdentityV1",
    "CandidatePublicWheelV1",
    "CandidateRoleArchiveV1",
    "CandidateValidationBundleV1",
    "verify_candidate_validation_bundle",
]
