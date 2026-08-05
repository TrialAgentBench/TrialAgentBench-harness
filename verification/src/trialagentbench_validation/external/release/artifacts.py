"""Content-addressed exports for external validation results."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_validation.io import sha256_file, write_model


class ArtifactDigestV1(BaseModel):
    """Identity of one external-validation artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(ge=1)


class ExternalArtifactManifestV1(BaseModel):
    """Deterministic checksum inventory for an evidence directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.external_artifact_manifest/v1"] = (
        "trialagentbench.external_artifact_manifest/v1"
    )
    artifacts: tuple[ArtifactDigestV1, ...] = Field(min_length=1)


def write_external_artifact_manifest(
    directory: Path,
    *,
    output_name: str = "artifact_manifest.json",
) -> ExternalArtifactManifestV1:
    """Write a checksum inventory of every other regular file below a directory."""

    root = directory.resolve()
    paths = tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() != output_name
    )
    if not paths:
        raise ValueError("Artifact manifest requires at least one file.")
    escaping = [path for path in paths if not path.resolve().is_relative_to(root)]
    if escaping:
        raise ValueError(
            "Artifact directory contains a file that resolves outside its root: "
            f"{escaping[0].relative_to(root).as_posix()!r}."
        )
    manifest = ExternalArtifactManifestV1(
        artifacts=tuple(
            ArtifactDigestV1(
                relative_path=path.relative_to(root).as_posix(),
                sha256=sha256_file(path),
                byte_size=path.stat().st_size,
            )
            for path in paths
        )
    )
    write_model(root / output_name, manifest)
    return manifest


def verify_external_artifact_manifest(
    directory: Path,
    *,
    manifest_name: str = "artifact_manifest.json",
) -> ExternalArtifactManifestV1:
    """Verify exact file membership, size, and checksum for an evidence directory."""

    root = directory.resolve()
    manifest_path = root / manifest_name
    manifest = ExternalArtifactManifestV1.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    expected_paths = {artifact.relative_path for artifact in manifest.artifacts}
    observed_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if observed_paths != expected_paths:
        raise ValueError(
            "Artifact directory membership differs from its manifest: "
            f"missing={sorted(expected_paths - observed_paths)!r}, "
            f"unexpected={sorted(observed_paths - expected_paths)!r}."
        )
    for artifact in manifest.artifacts:
        relative_path = Path(artifact.relative_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Artifact manifest path is unsafe: {artifact.relative_path!r}."
            )
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(
                f"Artifact manifest path escapes its directory: {artifact.relative_path!r}."
            )
        if path.stat().st_size != artifact.byte_size:
            raise ValueError(
                f"Artifact byte size differs for {artifact.relative_path!r}."
            )
        if sha256_file(path) != artifact.sha256:
            raise ValueError(
                f"Artifact checksum differs for {artifact.relative_path!r}."
            )
    return manifest


__all__ = [
    "ArtifactDigestV1",
    "ExternalArtifactManifestV1",
    "verify_external_artifact_manifest",
    "write_external_artifact_manifest",
]
