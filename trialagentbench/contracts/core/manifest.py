"""Artifact manifest for a canonically graded TrialEval run."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench.io.checksums import canonical_payload_sha256


class GradeManifestV1(BaseModel):
    """Content identity and denominator of one canonical TrialEval grade."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench_grade_manifest_v1"] = "trialagentbench_grade_manifest_v1"
    schema_version: Literal[1] = 1
    suite: Literal["trialeval"] = "trialeval"
    harness_version: str = Field(min_length=1)
    harness_git_sha: str | None = Field(min_length=1)
    harness_source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    timestamp_utc: datetime
    input_run_dir: str = Field(min_length=1)
    output_run_dir: str = Field(min_length=1)
    input_run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_release_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_profile_id: Literal["oracle_outcome_components_v1"] = "oracle_outcome_components_v1"
    scorer_surface_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite_task_count: int = Field(ge=1)
    graded_task_count: int = Field(ge=0)
    notes: tuple[str, ...] = ()
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """Require a bounded denominator and bind the complete manifest."""

        if self.graded_task_count > self.suite_task_count:
            raise ValueError("Graded TrialEval tasks cannot exceed the scheduled denominator")
        if self.harness_git_sha is None and self.harness_source_sha256 is None:
            raise ValueError("Grade provenance requires a Git revision or installed source identity")
        excluded = {"checksum"}
        # Historical v1 manifests predate the installed-source field.
        if self.harness_source_sha256 is None:
            excluded.add("harness_source_sha256")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude=excluded))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("TrialEval grade manifest checksum mismatch")
        object.__setattr__(self, "checksum", expected)
        return self
