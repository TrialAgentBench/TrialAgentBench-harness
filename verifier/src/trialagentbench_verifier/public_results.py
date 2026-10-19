"""Public aggregate campaign results projected from standalone analysis owners."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_verifier.campaign import (
    AnalysisStatus,
    TrialEvalBetweenModelContrastV1,
    TrialEvalCampaignAnalysisV1,
    TrialEvalDifficultySummaryV1,
    TrialEvalLossRankingV1,
    TrialEvalMetricSummaryV1,
    TrialEvalTargetSummaryV1,
)
from trialagentbench_verifier.campaign_contracts import BudgetCensoringV1, CampaignAnalysisV1, LeaderboardRowV1
from trialagentbench_verifier.development_campaign import (
    TrialDevCampaignAnalysisV1,
    TrialDevCampaignProfileV1,
    TrialDevComparisonV1,
)


class PublicTrialEvalCampaignV1(BaseModel):
    """Aggregate numerical recovery with private per-task targets excluded.

    ``plan_execution_summaries`` report, per model and plan-given view (C1, C3), the share of answers that
    executed the locked plan within tolerance; it is scored apart from the numerical loss.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["trialagentbench.public_trialeval_campaign/v3"] = "trialagentbench.public_trialeval_campaign/v3"
    source_analysis_checksum: str
    release_id: str
    campaign_plan_checksum: str
    source_archive_sha256: str
    scientific_analysis_status: AnalysisStatus
    confidence_level: float = Field(gt=0, lt=1)
    bootstrap_resamples: int = Field(ge=100)
    bootstrap_seed: int = Field(ge=0)
    loss_rankings: tuple[TrialEvalLossRankingV1, ...]
    cell_loss_summaries: tuple[TrialEvalLossRankingV1, ...]
    difficulty_summaries: tuple[TrialEvalDifficultySummaryV1, ...]
    target_summaries: tuple[TrialEvalTargetSummaryV1, ...]
    between_model_contrasts: tuple[TrialEvalBetweenModelContrastV1, ...]
    plan_execution_summaries: tuple[TrialEvalMetricSummaryV1, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )


def public_trialeval_campaign(report: TrialEvalCampaignAnalysisV1) -> PublicTrialEvalCampaignV1:
    """Preserve existing aggregate estimates without exporting oracle targets."""
    return PublicTrialEvalCampaignV1(
        source_analysis_checksum=report.checksum,
        release_id=report.release_id,
        campaign_plan_checksum=report.campaign_plan_checksum,
        source_archive_sha256=report.source_archive_sha256,
        scientific_analysis_status=report.scientific_analysis_status,
        confidence_level=report.confidence_level,
        bootstrap_resamples=report.bootstrap_resamples,
        bootstrap_seed=report.bootstrap_seed,
        loss_rankings=report.loss_rankings,
        cell_loss_summaries=report.cell_loss_summaries,
        difficulty_summaries=report.difficulty_summaries,
        target_summaries=report.target_summaries,
        between_model_contrasts=tuple(
            row for row in report.between_model_contrasts if row.metric.metric_id.endswith("_signal_loss")
        ),
        plan_execution_summaries=tuple(
            row for row in report.metric_summaries if row.metric.metric_id == "plan_execution_accuracy"
        ),
    )


class CampaignLeaderboardV1(BaseModel):
    """Shareable per-model headlines for one prospectively planned campaign.

    Rows cover the paired panel of tasks no model lost to budget censoring.
    ``budget_censoring`` reports every model's censoring, and ``warnings``
    flags materially different censoring between compared models.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["trialagentbench.campaign_leaderboard/v2"] = "trialagentbench.campaign_leaderboard/v2"
    campaign_plan_sha256: str
    release_set_manifest_sha256: str
    split: Literal["public", "withheld"]
    confidence_level: float = Field(gt=0, lt=1)
    bootstrap_resamples: int = Field(ge=100)
    bootstrap_seed: int = Field(ge=0)
    rows: tuple[LeaderboardRowV1, ...]
    budget_censoring: BudgetCensoringV1 | None = None
    warnings: tuple[str, ...] = ()


TRIALEVAL_HEADLINE_CONTEXT = "C5"
"""The main TrialEval selection: raw records with recoverable defects, no locked plan."""
_DECISION_METRICS: tuple[tuple[str, Literal["decision_regret_musd", "consequential_error_rate"]], ...] = (
    ("decision_regret_musd", "decision_regret_musd"),
    ("consequential_error", "consequential_error_rate"),
)


def trialeval_leaderboard_rows(
    report: TrialEvalCampaignAnalysisV1, *, censoring: BudgetCensoringV1 | None = None
) -> tuple[LeaderboardRowV1, ...]:
    """Project panel-level TrialEval rankings, which already resample independent worlds.

    The headline prices each answer by the development decision it drives: the cell-balanced mean
    decision regret against the closest credited analysis of the same data (US$ millions, lower is better) and
    the rate of consequential errors, both on the main C5 view. A missing answer drives the prior-only decision,
    so both are defined for every scheduled answer. Mean numerical loss is the secondary row.

    Parameters
    ----------
    report
        Analysis of the paired TrialEval panel.
    censoring
        Budget censoring whose per-model counts annotate each row.

    Returns
    -------
    tuple of LeaderboardRowV1
        Decision regret, consequential-error rate and mean loss for each model and target kind.
    """
    rows = []
    summaries = {
        (row.model_id, row.metric.metric_id): row
        for row in getattr(report, "metric_summaries", ())
        if row.context == TRIALEVAL_HEADLINE_CONTEXT
    }
    for ranking in report.loss_rankings:
        common = {
            "benchmark": "trialeval",
            "track": ranking.target_kind,
            "model_id": ranking.model_id,
            "scheduled_tasks": ranking.scheduled_attempts,
            "scored_tasks": ranking.scored_attempts,
            "independent_worlds": ranking.independent_worlds,
            "budget_censored_tasks": (
                0 if censoring is None else censoring.censored_count("trialeval", ranking.model_id)
            ),
        }
        for metric_id, label in _DECISION_METRICS:
            summary = summaries.get((ranking.model_id, metric_id))
            if summary is None:
                continue
            rows.append(
                LeaderboardRowV1(
                    **common
                    | {"scheduled_tasks": summary.scheduled_attempts, "scored_tasks": summary.observed_attempts},
                    metric=label,
                    direction="lower_is_better",
                    estimate=summary.equal_cell_effect_mean,
                    interval_low=summary.interval_low,
                    interval_high=summary.interval_high,
                )
            )
        rows.append(
            LeaderboardRowV1(
                **common,
                metric="mean_loss",
                direction="lower_is_better",
                estimate=ranking.mean_loss,
                interval_low=ranking.interval_low,
                interval_high=ranking.interval_high,
            )
        )
    return tuple(rows)


class PublicTrialDevCampaignV1(BaseModel):
    """Paired recovery and development-value comparisons without local custody paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["trialagentbench.public_trialdev_campaign/v3"] = "trialagentbench.public_trialdev_campaign/v3"
    campaign_plan_sha256: str
    release_sha256: str
    export_sha256: tuple[str, ...]
    analysis_policy: CampaignAnalysisV1
    comparisons: tuple[TrialDevComparisonV1, ...]
    profile_summaries: tuple[TrialDevCampaignProfileV1, ...] = ()
    budget_censoring: BudgetCensoringV1 | None = None


def public_trialdev_campaign(
    report: TrialDevCampaignAnalysisV1, *, policy: CampaignAnalysisV1
) -> PublicTrialDevCampaignV1:
    """Retain the prespecified inference policy and exact standalone comparisons."""
    return PublicTrialDevCampaignV1(
        campaign_plan_sha256=report.campaign_plan_sha256,
        release_sha256=report.release_sha256,
        export_sha256=tuple(row.sha256 for row in report.exports),
        analysis_policy=policy,
        comparisons=report.comparisons,
        profile_summaries=report.profile_summaries,
        budget_censoring=report.budget_censoring,
    )
