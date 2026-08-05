"""Typed contract for deterministic portable result exports."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import JsonValue

from trialagentbench_harness.io import canonical_payload_sha256

ResultArtifactKind = Literal["run", "grade", "verification", "analysis"]
ReleaseStage = Literal["collaborator_single_seed", "paired_release"]


class ResultExportMemberV1(BaseModel):
    """One immutable file included in a portable result archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class ResultExportSourceV1(BaseModel):
    """One caller-owned artifact tree included in a result archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ResultArtifactKind
    label: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    archive_prefix: str = Field(min_length=1)
    file_count: int = Field(ge=1)


class ResultExportBundleManifestV1(BaseModel):
    """Checksummed identity of one portable result archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.result_export_bundle/v1"] = "trialagentbench.result_export_bundle/v1"
    release_id: str = Field(min_length=1)
    release_stage: ReleaseStage
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    export_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sources: tuple[ResultExportSourceV1, ...] = Field(min_length=3)
    members: tuple[ResultExportMemberV1, ...] = Field(min_length=1)
    member_count: int = Field(ge=1)
    checksum: str = ""

    @model_validator(mode="after")
    def _validate_manifest(self) -> ResultExportBundleManifestV1:
        if self.member_count != len(self.members):
            raise ValueError("member_count must equal the result member inventory")
        source_kinds = {source.kind for source in self.sources}
        required = {"run", "grade", "verification"}
        if not required.issubset(source_kinds):
            raise ValueError("result export requires run, grade, and verification artifacts")
        source_prefixes = [source.archive_prefix for source in self.sources]
        if len(source_prefixes) != len(set(source_prefixes)):
            raise ValueError("result export source prefixes must be unique")
        member_paths = [member.path for member in self.members]
        if member_paths != sorted(member_paths) or len(member_paths) != len(set(member_paths)):
            raise ValueError("result export member paths must be sorted and unique")
        expected = canonical_payload_sha256(cast(JsonValue, self.model_dump(mode="json", exclude={"checksum"})))
        if self.checksum and self.checksum != expected:
            raise ValueError("result export manifest checksum mismatch")
        object.__setattr__(self, "checksum", expected)
        return self


__all__ = [
    "ReleaseStage",
    "ResultArtifactKind",
    "ResultExportBundleManifestV1",
    "ResultExportMemberV1",
    "ResultExportSourceV1",
]
