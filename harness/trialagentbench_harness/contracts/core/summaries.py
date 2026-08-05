"""Contracts for suite-level summary artifacts written by the harness.

These contracts are part of the **publication surface**: every field is
schema-bearing and stable so offline analysis can be deterministic and
auditable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.types import JsonValue


class _StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrialDevProgramCompletionV1(_StrictBaseModel):
    """One per-program completion row (declared-denominator surface)."""

    program_id: str
    trajectory_primary_score: float | None = None
    completed: bool = False
    program_status: Literal[
        "completed",
        "model_noncompletion",
        "model_invalid_submission",
        "infrastructure_timeout",
        "infrastructure_error",
        "missing_program_dir",
        "missing_or_unusable_grade",
    ]


class TrialDevCompletionMetricsV1(_StrictBaseModel):
    """Completion-aware TrialDev rollups computed on declared denominators."""

    n_declared: int = 0
    n_present: int = 0
    n_completed: int = 0
    completion_rate: float | None = None
    completed_mean: float | None = None
    failure_imputed_mean: float | None = None
    per_program: list[TrialDevProgramCompletionV1] = Field(default_factory=list)


class TrialDevGroupRollupV1(_StrictBaseModel):
    """One phase, scenario, or objective rollup."""

    n: int = 0
    overall_mean: float = 0.0
    lane_raw_means: dict[str, float] = Field(default_factory=dict)


class TrialDevResultsRollupV1(_StrictBaseModel):
    """Complete-suite rollup in ``results_summary.json``."""

    n_items: int = 0
    overall_mean: float = 0.0
    lane_raw_means: dict[str, float] = Field(default_factory=dict)
    lane_active_means: dict[str, float] = Field(default_factory=dict)
    by_phase: dict[str, TrialDevGroupRollupV1] = Field(default_factory=dict)
    by_scenario: dict[str, TrialDevGroupRollupV1] = Field(default_factory=dict)
    by_objective: dict[str, TrialDevGroupRollupV1] = Field(default_factory=dict)


class TrialDevViolationsSummaryV1(_StrictBaseModel):
    """Summary of agent submission violations captured during a run."""

    n_violations: int = 0
    by_kind: dict[str, int] = Field(default_factory=dict)
    by_phase: dict[str, int] = Field(default_factory=dict)


class TrialDevRankPhaseSummaryV1(_StrictBaseModel):
    """Per-phase rank-quality rollup."""

    n: int = 0
    mean_rank: float = 0.0
    median_rank: int = 0
    rank_std: float | None = None
    mean_top_pick_quality: float | None = None


class TrialDevRankMetricsSummaryV1(_StrictBaseModel):
    """Summary of ranking-derived metrics."""

    n_phases: int = 0
    n_with_rank: int = 0
    mean_top_pick_rank: float | None = None
    median_top_pick_rank: int | None = None
    rank_std: float | None = None
    mean_top_pick_quality: float | None = None
    mean_reference_ranking_size: float | None = None
    n_with_bottom_n: int = 0
    mean_bottom_n_concordance: float | None = None
    by_phase: dict[str, TrialDevRankPhaseSummaryV1] = Field(default_factory=dict)


class TrialDevStickTwistSummaryV1(_StrictBaseModel):
    """Summary of whether the agent pivots between obs_review and phase1."""

    n_programs_with_both: int = 0
    n_pivoted: int = 0
    pivot_rate: float | None = None


class TrialDevObjectiveAlignmentByPrimaryV1(_StrictBaseModel):
    """Alignment stats for programs sharing the same primary objective."""

    n_programs: int = 0
    n_aligned: int = 0
    n_free_total: int = 0
    alignment_rate: float | None = None


class TrialDevObjectiveAlignmentSummaryV1(_StrictBaseModel):
    """Summary of whether per-phase selection objectives match the program objective."""

    n_programs_with_free_phase: int = 0
    alignment_rate_overall: float | None = None
    by_primary_objective: dict[str, TrialDevObjectiveAlignmentByPrimaryV1] = Field(default_factory=dict)


class TrialDevResultsPayloadV1(_StrictBaseModel):
    """Typed payload for `results_summary.json`.

    This payload contains derived summaries beyond `completion_metrics`. It is
    still a contract: do not add fields without bumping the schema version.
    """

    results: TrialDevResultsRollupV1 = Field(default_factory=TrialDevResultsRollupV1)
    violations: TrialDevViolationsSummaryV1 = Field(default_factory=TrialDevViolationsSummaryV1)
    rank_metrics: TrialDevRankMetricsSummaryV1 = Field(default_factory=TrialDevRankMetricsSummaryV1)
    stick_twist: TrialDevStickTwistSummaryV1 = Field(default_factory=TrialDevStickTwistSummaryV1)
    objective_alignment: TrialDevObjectiveAlignmentSummaryV1 = Field(
        default_factory=TrialDevObjectiveAlignmentSummaryV1
    )
    # Note: completion metrics are also stored at the top-level
    # `TrialDevResultsSummaryV1.completion_metrics` for fast consumers.
    completion_metrics: TrialDevCompletionMetricsV1 = Field(default_factory=TrialDevCompletionMetricsV1)


class TrialDevResultsSummaryV1(_StrictBaseModel):
    """`results_summary.json` contract for TrialDevBench."""

    schema_id: Literal["trialagentbench_trialdev_results_summary_v1"]
    schema_version: Literal[1]
    completion_metrics: TrialDevCompletionMetricsV1
    payload: TrialDevResultsPayloadV1


class TrialEvalSummaryV1(_StrictBaseModel):
    """Schema-bearing `summary.json` contract for TrialEvalBench."""

    schema_id: Literal["trialagentbench_trialeval_summary_v1"]
    schema_version: Literal[1]
    data_format: str
    n_items: int
    # Per-item rows are not part of the stable publication surface and may
    # evolve; keep as JSON values.
    items: list[JsonValue] = Field(default_factory=list)


__all__ = [
    "TrialDevProgramCompletionV1",
    "TrialDevCompletionMetricsV1",
    "TrialDevGroupRollupV1",
    "TrialDevResultsRollupV1",
    "TrialDevViolationsSummaryV1",
    "TrialDevRankPhaseSummaryV1",
    "TrialDevRankMetricsSummaryV1",
    "TrialDevStickTwistSummaryV1",
    "TrialDevObjectiveAlignmentByPrimaryV1",
    "TrialDevObjectiveAlignmentSummaryV1",
    "TrialDevResultsPayloadV1",
    "TrialDevResultsSummaryV1",
    "TrialEvalSummaryV1",
]
