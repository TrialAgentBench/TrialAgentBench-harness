"""Independent loading of participant benchmark releases."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from trialagentbench_verifier.assets import ReleaseAssetScope
from trialagentbench_verifier.io import release_asset_scope

from trialagentbench.contracts.core.trialeval_factors import (
    TrialEvalAnalysisSpecificationV1,
    TrialEvalDataPreparationV1,
    TrialEvalEvidenceFactorsV1,
)
from trialagentbench.contracts.release.identity import BenchmarkReleaseIdentityV1
from trialagentbench.io import canonical_payload_sha256, read_json, sha256_file


class ParticipantReleaseError(ValueError):
    """Raised when participant release bytes violate their public contract."""


class _Model(BaseModel):
    """Strict immutable base for participant release contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactIdentity(_Model):
    """Expected identity of one participant artifact."""

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_length: int = Field(ge=0)


class VisibleArtifact(_Model):
    """One released participant data or document artifact."""

    artifact_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    relative_path: str = Field(
        pattern=(
            r"^(trials/trial_[0-9a-f]{20}/(data|documents)|" r"tasks/te_[0-9a-f]{20}/(data|documents))/[a-z0-9_/.-]+$"
        )
    )
    description: str = Field(min_length=1)


class AnalysisContract(_Model):
    """Exact participant-visible statistical question."""

    analysis_population_id: str = Field(min_length=1)
    estimand_id: str = Field(min_length=1)
    intercurrent_event_strategy_ids: tuple[str, ...] = ()
    assessment_horizon_days: float = Field(gt=0.0)
    treatment_id: str = Field(min_length=1)
    comparator_id: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    effect_scale: Literal[
        "mean_difference",
        "rate_ratio",
        "risk_difference",
        "restricted_mean_survival_time_difference",
    ]
    result_unit: Literal[
        "days",
        "mg/g",
        "percentage points",
        "proportion",
        "ratio",
        "score points",
    ]
    analysis_unit: Literal["participant", "cluster"]
    population_basis: Literal["finite_population", "superpopulation"]
    identification_status: Literal["point_identified", "partially_identified"]
    identification_assumption_ids: tuple[str, ...] = ()
    set_definition_id: str | None = Field(default=None, min_length=1)
    sensitivity_parameters: tuple[float, ...] = ()
    uncertainty_type: Literal[
        "central_confidence_interval",
        "repeated_confidence_interval",
        "simultaneous_outer_confidence_region",
    ]
    confidence_level: float = Field(gt=0.0, lt=1.0)
    sidedness: Literal["two_sided_central", "simultaneous_outer"]
    null_value: float
    favorable_direction: Literal["higher", "lower"]

    @model_validator(mode="after")
    def validate_question(self) -> AnalysisContract:
        """Require a coherent contrast, unit, and strategy list."""
        if self.treatment_id == self.comparator_id:
            raise ValueError("TrialEval treatment and comparator must differ")
        if tuple(sorted(set(self.intercurrent_event_strategy_ids))) != self.intercurrent_event_strategy_ids:
            raise ValueError("TrialEval intercurrent-event strategy identifiers must be sorted and unique")
        expected_units = {
            "mean_difference": {"mg/g", "percentage points", "score points"},
            "rate_ratio": {"ratio"},
            "risk_difference": {"proportion"},
            "restricted_mean_survival_time_difference": {"days"},
        }[self.effect_scale]
        if self.result_unit not in expected_units:
            raise ValueError("TrialEval result unit must match its effect scale")
        if self.identification_assumption_ids != tuple(sorted(set(self.identification_assumption_ids))):
            raise ValueError("identification assumptions must be sorted and unique")
        if self.sensitivity_parameters != tuple(sorted(set(self.sensitivity_parameters))):
            raise ValueError("sensitivity parameters must be sorted and unique")
        if (self.identification_status == "partially_identified") != (
            self.uncertainty_type == "simultaneous_outer_confidence_region"
        ):
            raise ValueError("partially identified TrialEval tasks require a simultaneous outer confidence region")
        partially_identified = self.identification_status == "partially_identified"
        expected_sidedness = "simultaneous_outer" if partially_identified else "two_sided_central"
        if self.sidedness != expected_sidedness:
            raise ValueError("TrialEval sidedness must match the requested uncertainty object")
        if partially_identified != (self.set_definition_id is not None):
            raise ValueError("partially identified questions require a set definition")
        if partially_identified != bool(self.identification_assumption_ids):
            raise ValueError("partially identified questions require named assumptions")
        if partially_identified != bool(self.sensitivity_parameters):
            raise ValueError("partially identified questions require sensitivity parameters")
        expected_null = 1.0 if self.effect_scale == "rate_ratio" else 0.0
        if self.null_value != expected_null:
            raise ValueError("TrialEval null value must match its effect scale")
        return self


class ParticipantTask(_Model):
    """Participant-visible clinical-trial analysis task."""

    schema_id: Literal["ariadne.trialeval.task/v1"]
    task_id: str = Field(pattern=r"^te_[0-9a-f]{20}$")
    title: str = Field(min_length=1)
    study_design: str = Field(min_length=1)
    question: str = Field(min_length=1)
    available_materials: tuple[VisibleArtifact, ...] = Field(min_length=1)
    requested_outputs: tuple[str, ...] = Field(min_length=1)
    target_ids: tuple[str, ...] = Field(min_length=1)


class ParticipantReleaseManifest(_Model):
    """Portable participant-only release identity."""

    schema_id: Literal["ariadne.benchmark_role_manifest/v1"]
    release_id: str = Field(pattern=r"^release_[0-9a-f]{64}$")
    role: Literal["participant"]
    trial_count: int = Field(ge=1)
    task_count: int = Field(ge=1)
    artifacts: tuple[ArtifactIdentity, ...] = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_artifacts(self) -> ParticipantReleaseManifest:
        """Require unique portable artifact paths."""
        paths = tuple(item.relative_path for item in self.artifacts)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("participant artifact paths must be sorted and unique")
        for value in paths:
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
                raise ValueError("participant artifact path is not portable")
            if not path.parts or path.parts[0] != "participant":
                raise ValueError("participant manifest contains an artifact owned by another role")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        if canonical_payload_sha256(payload) != self.checksum:
            raise ValueError("participant manifest checksum mismatch")
        return self


class EvaluationReleaseManifest(_Model):
    """Portable evaluator-only release identity."""

    schema_id: Literal["ariadne.benchmark_role_manifest/v1"]
    release_id: str = Field(pattern=r"^release_[0-9a-f]{64}$")
    role: Literal["evaluation"]
    trial_count: int = Field(ge=1)
    task_count: int = Field(ge=1)
    artifacts: tuple[ArtifactIdentity, ...] = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_artifacts(self) -> EvaluationReleaseManifest:
        """Require a complete, canonical evaluator artifact inventory."""
        paths = tuple(item.relative_path for item in self.artifacts)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("evaluation artifact paths must be sorted and unique")
        for value in paths:
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
                raise ValueError("evaluation artifact path is not portable")
            if not path.parts or path.parts[0] != "evaluation":
                raise ValueError("evaluation manifest contains an artifact owned by another role")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        if canonical_payload_sha256(payload) != self.checksum:
            raise ValueError("evaluation manifest checksum mismatch")
        return self


class LoadedParticipantRelease(_Model):
    """Verified public tasks and their extracted release root."""

    root: Path
    manifest: ParticipantReleaseManifest
    release_identity: BenchmarkReleaseIdentityV1
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tasks: tuple[ParticipantTask, ...]


class LoadedEvaluationRelease(_Model):
    """Verified evaluator release root and manifest."""

    root: Path
    manifest: EvaluationReleaseManifest


class EvaluationTaskIndexEntry(_Model):
    """Scientific classification of one opaque TrialEval task."""

    task_id: str = Field(pattern=r"^te_[0-9a-f]{20}$")
    trial_id: str = Field(pattern=r"^trial_[0-9a-f]{20}$")
    base_world_id: str = Field(pattern=r"^world_[0-9a-f]{20}$")
    design_tier: Literal["D1", "D2", "D3", "D4"]
    design_subtype: str = Field(min_length=1)
    design_family: Literal[
        "individual_randomized",
        "pragmatic",
        "covariate_structure",
        "endpoint_ascertainment",
        "cluster_parallel",
        "stepped_wedge",
        "group_sequential",
    ]
    assumption_tier: Literal["A1", "A2", "A3", "A4"]
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    evaluation_series_id: str = Field(pattern=r"^TE-S(?:0[1-9]|1[0-7])$")
    regime_cell_id: str = Field(pattern=r"^TE-S(?:0[1-9]|1[0-7])-A[1-4]$")
    effect_state: Literal["favourable", "near_null", "unfavourable"]
    data_preparation: TrialEvalDataPreparationV1
    analysis_specification: TrialEvalAnalysisSpecificationV1
    reconstruction_family: str | None = Field(pattern=r"^[a-z][a-z0-9_]+$")
    data_integrity_reference_row_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_factors(self) -> EvaluationTaskIndexEntry:
        """Require the evidence factors implied by the context configuration."""
        TrialEvalEvidenceFactorsV1(
            context_configuration=self.context_tier,
            data_preparation=self.data_preparation,
            analysis_specification=self.analysis_specification,
        )
        if (self.context_tier == "C5") != (self.reconstruction_family is not None):
            raise ValueError("Reconstruction family is required exactly for C5 tasks")
        if self.regime_cell_id != f"{self.evaluation_series_id}-{self.assumption_tier}":
            raise ValueError("evaluation regime-cell identity differs from its series and assumption tier")
        return self


class EvaluationTaskIndex(_Model):
    """Checksum-bound evaluator task classification."""

    schema_id: Literal["ariadne.trialeval.task_index/v2"]
    release_id: str = Field(pattern=r"^release_[0-9a-f]{64}$")
    entries: tuple[EvaluationTaskIndexEntry, ...] = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_index(self) -> EvaluationTaskIndex:
        """Require canonical unique tasks and a matching checksum."""
        task_ids = tuple(row.task_id for row in self.entries)
        if task_ids != tuple(sorted(task_ids)) or len(task_ids) != len(set(task_ids)):
            raise ValueError("evaluation task index must be sorted and unique")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        if canonical_payload_sha256(payload) != self.checksum:
            raise ValueError("evaluation task index checksum mismatch")
        return self


def load_participant_release(root: Path, *, scope: ReleaseAssetScope | None = None) -> LoadedParticipantRelease:
    """Verify and load a participant release without construction code."""
    scope = scope or release_asset_scope(root)
    root = Path(root)
    outer_path = root / "release_manifest.json"
    if not outer_path.is_file() and root.name == "participant":
        outer_path = root.parent / "release_manifest.json"
    if not outer_path.is_file() or outer_path.is_symlink():
        raise ParticipantReleaseError("participant release manifest must be a regular file")
    if scope is not None:
        scope.verify_release_identity(outer_path, "trialeval")
    release_identity = BenchmarkReleaseIdentityV1.model_validate(read_json(outer_path))
    if release_identity.schema_id != "ariadne.trialeval.release/v1":
        raise ParticipantReleaseError("participant release has the wrong benchmark stream")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ParticipantReleaseError("participant manifest must be a regular file")
    manifest = ParticipantReleaseManifest.model_validate(read_json(manifest_path))
    if (
        manifest.release_id != release_identity.release_id
        or manifest.checksum != release_identity.participant_manifest_sha256
        or manifest.trial_count != release_identity.trial_count
        or manifest.task_count != release_identity.task_count
    ):
        raise ParticipantReleaseError("participant role is not bound by its outer release manifest")
    for identity in manifest.artifacts:
        if scope is not None and not scope.requires(identity.relative_path, identity.sha256, identity.byte_length):
            continue
        relative = PurePosixPath(identity.relative_path)
        path = root.joinpath(*relative.parts[1:])
        if not path.is_file() or path.is_symlink():
            raise ParticipantReleaseError(f"participant artifact is missing: {identity.relative_path}")
        if path.stat().st_size != identity.byte_length or sha256_file(path) != identity.sha256:
            raise ParticipantReleaseError(f"participant artifact changed: {identity.relative_path}")
    expected = {
        "manifest.json",
        *(
            PurePosixPath(item.relative_path).relative_to("participant").as_posix()
            for item in manifest.artifacts
            if scope is None or scope.requires(item.relative_path, item.sha256, item.byte_length)
        ),
    }
    if outer_path.parent == root:
        expected.add("release_manifest.json")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != expected:
        raise ParticipantReleaseError("participant files differ from the release manifest")
    tasks = tuple(
        ParticipantTask.model_validate(read_json(path))
        for path in sorted((root / "tasks").glob("*/task.json"))
        if scope is None or path.parent.name in scope.selected_task_ids
    )
    expected_count = manifest.task_count if scope is None else len(scope.selected_task_ids)
    if scope is not None and (
        len(scope.catalog.tasks) != manifest.task_count or {task.task_id for task in tasks} != scope.selected_task_ids
    ):
        raise ParticipantReleaseError("participant task census differs from its authenticated selection")
    if len(tasks) != expected_count:
        raise ParticipantReleaseError("participant task count differs from its manifest")
    for task in tasks:
        task_root = root / "tasks" / task.task_id
        if not task_root.is_dir():
            raise ParticipantReleaseError(f"task identifier differs from its directory: {task.task_id}")
        for item in task.available_materials:
            path = root / item.relative_path
            if not path.is_file():
                raise ParticipantReleaseError(f"task data are missing: {task.task_id}/{item.relative_path}")
    return LoadedParticipantRelease(
        root=root,
        manifest=manifest,
        release_identity=release_identity,
        release_manifest_sha256=sha256_file(outer_path),
        tasks=tasks,
    )


def load_evaluation_release(root: Path, *, scope: ReleaseAssetScope | None = None) -> LoadedEvaluationRelease:
    """Verify and load an evaluator release without construction code."""
    scope = scope or release_asset_scope(root)
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ParticipantReleaseError("evaluation manifest must be a regular file")
    manifest = EvaluationReleaseManifest.model_validate(read_json(manifest_path))
    for identity in manifest.artifacts:
        if scope is not None and not scope.requires(identity.relative_path, identity.sha256, identity.byte_length):
            continue
        relative = PurePosixPath(identity.relative_path)
        path = root.joinpath(*relative.parts[1:])
        if not path.is_file() or path.is_symlink():
            raise ParticipantReleaseError(f"evaluation artifact is missing: {identity.relative_path}")
        if path.stat().st_size != identity.byte_length or sha256_file(path) != identity.sha256:
            raise ParticipantReleaseError(f"evaluation artifact changed: {identity.relative_path}")
    expected = {
        "manifest.json",
        *(
            PurePosixPath(item.relative_path).relative_to("evaluation").as_posix()
            for item in manifest.artifacts
            if scope is None or scope.requires(item.relative_path, item.sha256, item.byte_length)
        ),
    }
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != expected:
        raise ParticipantReleaseError("evaluation files differ from the release manifest")
    return LoadedEvaluationRelease(root=root, manifest=manifest)


def load_evaluation_task_index(release: LoadedEvaluationRelease) -> EvaluationTaskIndex:
    """Load the evaluator task index and bind it to its release identity."""
    index = EvaluationTaskIndex.model_validate(read_json(release.root / "grader" / "task_index.json"))
    if index.release_id != release.manifest.release_id:
        raise ParticipantReleaseError("evaluation task index belongs to a different release")
    if len(index.entries) != release.manifest.task_count:
        raise ParticipantReleaseError("evaluation task index count differs from its role manifest")
    return index


__all__ = [
    "LoadedParticipantRelease",
    "AnalysisContract",
    "EvaluationReleaseManifest",
    "EvaluationTaskIndex",
    "EvaluationTaskIndexEntry",
    "LoadedEvaluationRelease",
    "ParticipantReleaseError",
    "ParticipantReleaseManifest",
    "ParticipantTask",
    "load_participant_release",
    "load_evaluation_release",
    "load_evaluation_task_index",
]
