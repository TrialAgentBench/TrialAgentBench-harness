"""Artifact manifests for grading and aggregation."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from trialagentbench_harness.failure_codes import TrialDevFailureCode, TrialEvalFailureCode


class ToleratedFailureV1(BaseModel):
    """A non-fatal failure explicitly tolerated by policy (must be recorded)."""

    schema_id: Literal["trialagentbench_tolerated_failure_v1"] = "trialagentbench_tolerated_failure_v1"
    schema_version: Literal[1] = 1
    code: TrialEvalFailureCode | TrialDevFailureCode
    message: str
    path: str | None = None
    item_id: str | None = None
    program_id: str | None = None


class GradeManifestV1(BaseModel):
    """Provenance snapshot for grading one canonical benchmark run."""

    schema_id: Literal["trialagentbench_grade_manifest_v1"] = "trialagentbench_grade_manifest_v1"
    schema_version: Literal[1] = 1
    suite: Literal["trialeval", "trialdev"]
    harness_version: str
    harness_git_sha: str = ""
    timestamp_utc: datetime

    input_run_dir: str
    output_run_dir: str
    input_run_sha256: str
    evaluator_release_sha256: str
    suite_dir: str | None = None
    bundle_dir: str | None = None

    data_format: str | None = None
    score_profile_id: str | None = None
    score_profile_ids_available: list[str] = Field(default_factory=list)
    evaluation_target_register_sha256: str | None = None
    estimator_route_family_map_sha256: str | None = None
    method_route_register_sha256: str | None = None
    route_references_sha256: str | None = None
    route_scoring_scope_sha256: str | None = None
    scorer_surface_sha256: str | None = None
    suite_task_count: int | None = None
    graded_task_count: int | None = None
    suite_lane_count: int | None = None
    method_route_count: int | None = None
    route_scoring_scope_count: int | None = None
    notes: list[str] = Field(default_factory=list)


class AggregateManifestV1(BaseModel):
    """Provenance manifest for an aggregation/reporting pass over a run tree."""

    schema_id: Literal["trialagentbench_aggregate_manifest_v1"] = "trialagentbench_aggregate_manifest_v1"
    schema_version: Literal[1] = 1
    harness_version: str
    harness_git_sha: str = ""
    timestamp_utc: datetime

    input_run_dir: str
    bundle_dir: str | None = None

    policy_strict: bool = True
    allow_incomplete_artifacts: bool = False
    tolerated_failures: list[ToleratedFailureV1] = Field(default_factory=list)
