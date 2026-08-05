"""Portable release contracts for bounded TrialDev portfolio programmes."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevEvidenceReferenceV1,
    TrialDevPolicyBindingV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_recoverability import TrialDevObjectiveIdV1
from trialagentbench_harness.io.checksums import canonical_payload_sha256


def _checksum(model: BaseModel) -> str:
    payload = model.model_dump(mode="json", exclude_none=True, exclude={"checksum"})
    return str(canonical_payload_sha256(cast(JsonValue, payload)))


class _ChecksummedPortfolioReleaseModelV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def bind_checksum(self) -> Self:
        """Bind the canonical payload checksum and reject mutated records."""

        expected = _checksum(self)
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("Portfolio release checksum does not match its canonical payload.")
        object.__setattr__(self, "checksum", expected)
        return self


class TrialDevPortfolioEpisodeArtifactV1(_ChecksummedPortfolioReleaseModelV1):
    """Checksum inventory for one immutable randomized evidence episode."""

    schema_id: Literal["trialagentbench.trialdev_portfolio_episode_artifact/v1"] = (
        "trialagentbench.trialdev_portfolio_episode_artifact/v1"
    )
    episode_id: str = Field(..., min_length=1)
    world_id: str = Field(..., min_length=1)
    asset_id: str = Field(..., min_length=1)
    phase_id: Literal["phase1", "phase2", "phase3"]
    generation_seed: int = Field(..., ge=0)
    evidence_protocol_id: str = Field(..., min_length=1)
    protocol_relative_path: str = Field(..., min_length=1)
    protocol_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    episode_manifest_relative_path: str = Field(..., min_length=1)
    episode_manifest_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(..., ge=1)

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        """Require normalized, distinct release-relative artifact paths."""

        paths = (self.protocol_relative_path, self.episode_manifest_relative_path)
        for value in paths:
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
                raise ValueError("Portfolio episode artifacts require normalized relative paths.")
        if len(set(paths)) != len(paths):
            raise ValueError("Portfolio protocol and dataset paths must be distinct.")
        return self


class TrialDevPortfolioEpisodeFileV1(BaseModel):
    """One standard table in a randomized portfolio evidence episode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    table_id: Literal["participants", "endpoints", "safety"]
    relative_path: str = Field(..., min_length=1)
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(..., ge=1)

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """Require a normalized release-relative Parquet path."""

        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != self.relative_path:
            raise ValueError("Portfolio episode table path must be normalized and relative.")
        if path.suffix != ".parquet":
            raise ValueError("Portfolio episode tables must use Parquet serialization.")
        return self


class TrialDevPortfolioEpisodeMetadataFileV1(BaseModel):
    """One checksummed analysis input accompanying randomized trial tables."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata_id: Literal["arm_mapping", "request", "execution_summary"]
    relative_path: str = Field(..., min_length=1)
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """Require the canonical filename for each metadata role."""

        path = PurePosixPath(self.relative_path)
        expected_name = f"{self.metadata_id}.json"
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != self.relative_path
            or path.name != expected_name
        ):
            raise ValueError(f"Portfolio episode {self.metadata_id} must use {expected_name}.")
        return self


class TrialDevPortfolioEpisodeDatasetV1(_ChecksummedPortfolioReleaseModelV1):
    """Manifest for the unchanged standard tables in one evidence episode."""

    schema_id: Literal["trialagentbench.trialdev_portfolio_episode_dataset/v1"] = (
        "trialagentbench.trialdev_portfolio_episode_dataset/v1"
    )
    episode_id: str = Field(..., min_length=1)
    world_id: str = Field(..., min_length=1)
    asset_id: str = Field(..., min_length=1)
    phase_id: Literal["phase1", "phase2", "phase3"]
    generation_seed: int = Field(..., ge=0)
    files: tuple[TrialDevPortfolioEpisodeFileV1, ...]
    metadata_files: tuple[TrialDevPortfolioEpisodeMetadataFileV1, ...]

    @model_validator(mode="after")
    def validate_file_census(self) -> Self:
        """Require the complete aligned table and metadata inventory."""

        if len(self.files) != 3 or {item.table_id for item in self.files} != {
            "participants",
            "endpoints",
            "safety",
        }:
            raise ValueError("Portfolio episode dataset requires the three standard trial tables.")
        if len({item.relative_path for item in self.files}) != 3:
            raise ValueError("Portfolio episode table paths must be unique.")
        if len({item.row_count for item in self.files}) != 1:
            raise ValueError("Portfolio episode tables must have aligned row counts.")
        expected_metadata = {"arm_mapping", "request", "execution_summary"}
        if (
            len(self.metadata_files) != len(expected_metadata)
            or {item.metadata_id for item in self.metadata_files} != expected_metadata
        ):
            raise ValueError("Portfolio episode dataset requires its three standard metadata inputs.")
        relative_paths = tuple(item.relative_path for item in self.files) + tuple(
            item.relative_path for item in self.metadata_files
        )
        if len(relative_paths) != len(set(relative_paths)):
            raise ValueError("Portfolio episode artifact paths must be unique.")
        return self


class TrialDevPortfolioParticipantViewV1(_ChecksummedPortfolioReleaseModelV1):
    """Participant-visible initial state for one objective and resource policy view."""

    schema_id: Literal["trialagentbench.trialdev_portfolio_participant_view/v1"] = (
        "trialagentbench.trialdev_portfolio_participant_view/v1"
    )
    programme_id: str = Field(..., min_length=1)
    scenario_id: str = Field(..., min_length=1)
    world_id: str = Field(..., min_length=1)
    candidate_asset_ids: tuple[str, str, str]
    objective_id: TrialDevObjectiveIdV1
    resource_budget_units: Literal[8, 10]
    public_scenario_relative_path: str = Field(..., min_length=1)
    policy_binding: TrialDevPolicyBindingV1
    initial_evidence: tuple[TrialDevEvidenceReferenceV1, ...]

    @model_validator(mode="after")
    def validate_public_view(self) -> Self:
        """Require shared initial evidence and no future evidence."""

        if self.policy_binding.stream_id != "bounded_portfolio_reallocation":
            raise ValueError("Portfolio participant views require the portfolio stream policy.")
        if self.policy_binding.objective_id != self.objective_id:
            raise ValueError("Participant objective and policy binding disagree.")
        if self.policy_binding.resource_budget_units != self.resource_budget_units:
            raise ValueError("Participant budget and policy binding disagree.")
        if len(set(self.candidate_asset_ids)) != 3:
            raise ValueError("Portfolio participant view requires three unique candidates.")
        required_kinds = {"protocol", "dataset"}
        if not self.initial_evidence or not required_kinds <= {item.evidence_kind for item in self.initial_evidence}:
            raise ValueError("Portfolio participant view requires protocol and dataset evidence.")
        if any(item.checkpoint_id != "observational_review" for item in self.initial_evidence):
            raise ValueError("Participant view cannot expose future portfolio evidence.")
        if any(item.asset_id is not None for item in self.initial_evidence):
            raise ValueError("Initial observational evidence is shared across candidate assets.")
        path = PurePosixPath(self.public_scenario_relative_path)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != self.public_scenario_relative_path:
            raise ValueError("Participant public scenario path must be normalized and relative.")
        return self


class TrialDevPortfolioEvaluatorViewV1(_ChecksummedPortfolioReleaseModelV1):
    """Evaluator-only binding of one public view to its evidence inventory."""

    schema_id: Literal["trialagentbench.trialdev_portfolio_evaluator_view/v1"] = (
        "trialagentbench.trialdev_portfolio_evaluator_view/v1"
    )
    programme_id: str = Field(..., min_length=1)
    participant_view_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    family_id: str = Field(..., pattern=r"^P(?:0[1-9]|1[0-2])_[a-z0-9_]+$")
    discriminating_property: str = Field(..., min_length=1)
    world_manifest_relative_path: str = Field(..., min_length=1)
    world_manifest_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    evidence_index_relative_path: str = Field(..., min_length=1)
    evidence_index_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    observational_reference_relative_path: str = Field(..., min_length=1)
    observational_reference_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        """Require normalized evaluator artifact paths."""

        for value in (
            self.world_manifest_relative_path,
            self.evidence_index_relative_path,
            self.observational_reference_relative_path,
        ):
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
                raise ValueError("Portfolio evaluator paths must be normalized and relative.")
        if PurePosixPath(self.observational_reference_relative_path).name != "observational_reference.json":
            raise ValueError("Portfolio observational reference must use observational_reference.json.")
        return self


class TrialDevPortfolioParticipantCatalogueV1(_ChecksummedPortfolioReleaseModelV1):
    """Participant projection of all portfolio programme-policy views."""

    schema_id: Literal["trialagentbench.trialdev_portfolio_participant_catalogue/v1"] = (
        "trialagentbench.trialdev_portfolio_participant_catalogue/v1"
    )
    release_id: str = Field(..., min_length=1)
    source_identity: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    views: tuple[TrialDevPortfolioParticipantViewV1, ...]

    @model_validator(mode="after")
    def validate_view_census(self) -> Self:
        """Require 96 unique participant views over 12 opaque worlds."""

        if len(self.views) != 96 or len({item.programme_id for item in self.views}) != 96:
            raise ValueError("Participant portfolio catalogue requires 96 unique programme views.")
        combinations = {(item.world_id, item.objective_id, item.resource_budget_units) for item in self.views}
        if len(combinations) != 96 or len({item.world_id for item in self.views}) != 12:
            raise ValueError("Participant catalogue requires 12 worlds crossed by four objectives and two budgets.")
        return self


class TrialDevPortfolioReleaseManifestV1(_ChecksummedPortfolioReleaseModelV1):
    """Complete typed inventory of an immutable score-bearing portfolio release."""

    schema_id: Literal["trialagentbench.trialdev_portfolio_release_manifest/v1"] = (
        "trialagentbench.trialdev_portfolio_release_manifest/v1"
    )
    release_id: str = Field(..., min_length=1)
    source_identity: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    implementation_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    implementation_source_paths: tuple[str, ...] = Field(..., min_length=1)
    specification_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    family_count: Literal[12] = 12
    world_count: Literal[12] = 12
    objective_count: Literal[4] = 4
    resource_budget_count: Literal[2] = 2
    programme_view_count: Literal[96] = 96
    unique_randomized_episode_count: Literal[108] = 108
    evidence_reference_count: Literal[144] = 144
    episodes: tuple[TrialDevPortfolioEpisodeArtifactV1, ...]
    participant_views: tuple[TrialDevPortfolioParticipantViewV1, ...]
    evaluator_views: tuple[TrialDevPortfolioEvaluatorViewV1, ...]

    @model_validator(mode="after")
    def validate_release_census(self) -> Self:
        """Enforce exact world, episode, and view inventories without duplication."""

        if tuple(sorted(set(self.implementation_source_paths))) != self.implementation_source_paths:
            raise ValueError("Implementation source paths must be sorted and unique.")
        for value in self.implementation_source_paths:
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
                raise ValueError("Implementation source paths must be normalized and repository-relative.")

        if len(self.episodes) != self.unique_randomized_episode_count:
            raise ValueError("Portfolio release does not contain exactly 108 unique randomized episodes.")
        episode_keys = {(item.world_id, item.asset_id, item.phase_id) for item in self.episodes}
        if len(episode_keys) != self.unique_randomized_episode_count:
            raise ValueError("Portfolio release contains duplicate world-asset-phase episodes.")
        if len({item.episode_manifest_relative_path for item in self.episodes}) != len(self.episodes):
            raise ValueError("Portfolio randomized episode paths must be unique.")
        if len(self.participant_views) != self.programme_view_count:
            raise ValueError("Portfolio release does not contain exactly 96 participant views.")
        if len(self.evaluator_views) != self.programme_view_count:
            raise ValueError("Portfolio release does not contain exactly 96 evaluator views.")
        participant_by_id = {item.programme_id: item for item in self.participant_views}
        evaluator_by_id = {item.programme_id: item for item in self.evaluator_views}
        if len(participant_by_id) != self.programme_view_count or set(evaluator_by_id) != set(participant_by_id):
            raise ValueError("Participant and evaluator portfolio inventories must identify 96 common programmes.")
        for programme_id, evaluator in evaluator_by_id.items():
            if evaluator.participant_view_checksum != participant_by_id[programme_id].checksum:
                raise ValueError("Evaluator view is not bound to its participant projection.")
        worlds = {item.world_id for item in self.participant_views}
        families = {item.family_id for item in self.evaluator_views}
        if len(worlds) != self.world_count or len(families) != self.family_count:
            raise ValueError("Portfolio release requires 12 worlds and 12 coverage families.")
        views = {(item.world_id, item.objective_id, item.resource_budget_units) for item in self.participant_views}
        if len(views) != self.programme_view_count:
            raise ValueError("Portfolio objective-budget views must be unique.")
        return self


__all__ = [
    "TrialDevPortfolioEpisodeArtifactV1",
    "TrialDevPortfolioEpisodeDatasetV1",
    "TrialDevPortfolioEpisodeFileV1",
    "TrialDevPortfolioEpisodeMetadataFileV1",
    "TrialDevPortfolioEvaluatorViewV1",
    "TrialDevPortfolioParticipantCatalogueV1",
    "TrialDevPortfolioParticipantViewV1",
    "TrialDevPortfolioReleaseManifestV1",
]
