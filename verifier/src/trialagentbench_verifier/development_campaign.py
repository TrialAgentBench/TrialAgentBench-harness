"""Paired finite-world TrialDev comparisons from independently verified rankings.

Worlds, rather than views, repetitions or patients, are the resampling units.
The two metrics retain their own scales; this module creates no overall score.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from html import escape
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Literal
from zipfile import ZipFile

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_verifier.assets import read_asset_catalog
from trialagentbench_verifier.campaign import holm_adjusted_p_values_v1
from trialagentbench_verifier.campaign_contracts import (
    BudgetCensoringV1,
    CampaignAnalysisV1,
    CampaignPlanV1,
    Digest,
    LeaderboardRowV1,
)
from trialagentbench_verifier.development import _Task, _World, ranking_means, verify_trialdev_ranking
from trialagentbench_verifier.io import authenticated_archive, materialized_release
from trialagentbench_verifier.numerical_scoring import bounded_recovery
from trialagentbench_verifier.response import TrialDevRankingMeanV1, TrialDevRunRankingV1

Metric = Literal["observational_loss", "net_development_value_usd"]
Stratum = tuple[str, str, int]


class TrialDevComparisonV1(BaseModel):
    """One prespecified paired difference, with explicit unsupported inference.

    The common-world estimate is a descriptive sensitivity on worlds where both
    models completed every linked task and repetition. It weights retained strata
    equally; its counts disclose selection and it never supplies a panel ranking.
    Evaluator faults withhold both comparisons.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    metric: Metric
    stream: Literal["evidence_analysis", "portfolio_management"]
    data_view: Literal["raw", "analysis_ready"]
    model_left: str
    model_right: str
    worlds: int
    strata: int
    estimate: float | None
    confidence_low: float | None
    confidence_high: float | None
    p_value: float | None
    adjusted_p_value: float | None = None
    unavailable_reason: str | None
    common_world_estimate: float | None = None
    common_worlds: int = 0
    common_strata: int = 0


def compare_trialdev_rankings(
    rankings: tuple[TrialDevRunRankingV1, ...],
    *,
    required_strata: tuple[Stratum, ...],
    policy: CampaignAnalysisV1,
) -> tuple[TrialDevComparisonV1, ...]:
    """Compare authenticated complete common panels, keeping streams and data views separate.

    Callers must independently verify the rankings and bind their complete job
    census to the prospective campaign. This numerical owner does not authenticate
    files. Missing tasks and changed panels are errors, not zero-valued results.
    Evaluator faults withhold the affected comparison, including its estimate.
    Authored development values remain in dollars. An unfinished programme
    withholds that comparison rather than receiving an invented stop payoff.
    """
    if not rankings or not required_strata or len(set(required_strata)) != len(required_strata):
        raise ValueError("Rankings and unique required strata must be nonempty")
    by_model: dict[str, list[TrialDevRunRankingV1]] = defaultdict(list)
    panel = None
    identity = (rankings[0].release_id, rankings[0].release_manifest_sha256)
    for ranking in rankings:
        if (ranking.release_id, ranking.release_manifest_sha256) != identity:
            raise ValueError("Campaign rankings identify different releases")
        tasks = tuple(
            sorted(
                (
                    row.task_id,
                    row.world_id,
                    row.stream,
                    row.clinical_profile_id,
                    row.scenario_profile_id,
                    row.asset_count,
                    row.data_view,
                    row.budget_usd,
                    row.authored_programme,
                    tuple(sorted(row.initial_target_statuses)),
                )
                for row in ranking.items
            )
        )
        if not tasks or len({row.task_id for row in ranking.items}) != len(tasks):
            raise ValueError("Campaign ranking repeats or omits its task inventory")
        if panel is not None and panel != tasks:
            raise ValueError("Campaign rankings must retain the same scheduled task panel")
        panel = tasks
        if any(row.execution_status in {"not_started", "incomplete_run"} for row in ranking.items):
            raise ValueError("Campaign comparison requires every scheduled terminal primary")
        by_model[ranking.model_id].append(ranking)
    if len({len(rows) for rows in by_model.values()}) != 1:
        raise ValueError("Campaign models must retain the same scheduled repetitions")
    authored = {row.authored_programme for row in rankings[0].items if row.stream == "portfolio_management"}
    if False in authored:
        raise ValueError("Campaign requires authored development values")
    results: list[TrialDevComparisonV1] = []
    conditions = sorted({(row.stream, row.data_view) for row in rankings[0].items})
    for stream, data_view, metric in (
        (stream, view, metric)
        for stream, view in conditions
        for metric in ("observational_loss", "net_development_value_usd")
        if metric == "observational_loss" or stream == "portfolio_management"
    ):
        faulty_models: set[str] = set()
        incomplete_models: set[str] = set()
        values: dict[str, dict[tuple[Stratum, str], list[float | None]]] = {}
        for model, runs in sorted(by_model.items()):
            world_values: dict[tuple[Stratum, str], list[float | None]] = defaultdict(list)
            for run in runs:
                for row in run.items:
                    if row.stream != stream or row.data_view != data_view:
                        continue
                    stratum = (row.clinical_profile_id, row.scenario_profile_id, row.asset_count)
                    if stratum not in required_strata:
                        raise ValueError("Observed world falls outside the required release strata")
                    value = row.observational_loss if metric == "observational_loss" else row.net_development_value_usd
                    if row.execution_status == "evaluator_error":
                        faulty_models.add(model)
                        value = None
                    elif row.stream == "evidence_analysis" and row.execution_status == "model_noncompletion":
                        if value is not None:
                            raise ValueError("An uncompleted evidence analysis cannot carry numerical loss")
                        incomplete_models.add(model)
                    elif value is None:
                        incomplete_models.add(model)
                    elif metric == "net_development_value_usd" and row.execution_status == "model_noncompletion":
                        raise ValueError("Incomplete authored programme cannot have terminal development value")
                    world_values[stratum, row.world_id].append(value)
            values[model] = world_values
        metric_results = []
        for left, right in combinations(sorted(values), 2):
            left_values, right_values = values[left], values[right]
            if left_values.keys() != right_values.keys():
                raise ValueError("Model comparisons lost world pairing")
            groups: dict[Stratum, list[float]] = defaultdict(list)
            fault = left in faulty_models or right in faulty_models
            incomplete = left in incomplete_models or right in incomplete_models
            for key in sorted(left_values):
                a, b = left_values[key], right_values[key]
                if len(a) != len(b):
                    raise ValueError("Model comparisons lost linked resource conditions or repetitions")
                if not any(value is None for value in (*a, *b)):
                    groups[key[0]].append(
                        mean(value for value in a if value is not None)
                        - mean(value for value in b if value is not None)
                    )
            missing = set(groups) != set(required_strata)
            reason = (
                "evaluator_fault"
                if fault
                else (
                    "model_noncompletion"
                    if incomplete
                    else (
                        "missing_required_stratum"
                        if missing
                        else (
                            "insufficient_independent_worlds"
                            if any(len(group) < policy.minimum_worlds_per_stratum for group in groups.values())
                            else None
                        )
                    )
                )
            )
            estimate = None if fault or incomplete or missing else mean(mean(group) for group in groups.values())
            low = high = p_value = None
            if reason is None:
                rng = np.random.default_rng(policy.seed)
                draws = np.zeros(policy.bootstrap_repetitions)
                for stratum in sorted(groups):
                    vector = np.asarray(groups[stratum])
                    indices = rng.integers(0, len(vector), size=(policy.bootstrap_repetitions, len(vector)))
                    draws += vector[indices].mean(axis=1) / len(groups)
                tail = (1 - policy.confidence_level) / 2
                low, high = (float(value) for value in np.quantile(draws, [tail, 1 - tail]))
                assert estimate is not None
                p_value = (1 + float(np.sum(np.abs(draws - estimate) >= abs(estimate)))) / (len(draws) + 1)
            metric_results.append(
                TrialDevComparisonV1(
                    metric=metric,
                    stream=stream,
                    data_view=data_view,
                    model_left=left,
                    model_right=right,
                    worlds=len(left_values),
                    strata=len(required_strata),
                    estimate=estimate,
                    confidence_low=low,
                    confidence_high=high,
                    p_value=p_value,
                    unavailable_reason=reason,
                    common_world_estimate=(
                        mean(mean(group) for group in groups.values()) if groups and not fault else None
                    ),
                    common_worlds=sum(len(group) for group in groups.values()) if not fault else 0,
                    common_strata=len(groups) if not fault else 0,
                )
            )
        # Unavailable planned comparisons still occupy the prespecified family.
        if not metric_results:
            continue
        adjusted = holm_adjusted_p_values_v1(
            tuple(row.p_value if row.p_value is not None else 1.0 for row in metric_results)
        )
        results.extend(
            row.model_copy(update={"adjusted_p_value": value if row.p_value is not None else None})
            for row, value in zip(metric_results, adjusted, strict=True)
        )
    return tuple(results)


def trialdev_leaderboard_rows(
    rankings: tuple[TrialDevRunRankingV1, ...],
    *,
    required_strata: tuple[Stratum, ...],
    policy: CampaignAnalysisV1,
) -> tuple[LeaderboardRowV1, ...]:
    """Estimate each model's headline per stream and data view with world-clustered intervals.

    Parameters
    ----------
    rankings
        Independently verified rankings for every scheduled job.
    required_strata
        Clinical/scenario/asset strata that weight the estimate equally.
    policy
        Prospective bootstrap repetitions, seed, confidence and minimum worlds.

    Returns
    -------
    tuple of LeaderboardRowV1
        Bounded recovery for every task stream, and development value for
        authored programmes when every scheduled programme has a value.
        Linked conditions and repetitions are averaged within each world first.
    """
    rows: list[LeaderboardRowV1] = []
    for model in sorted({ranking.model_id for ranking in rankings}):
        items = [row for ranking in rankings if ranking.model_id == model for row in ranking.items]
        for stream, data_view in sorted({(row.stream, row.data_view) for row in items}):
            selected = [row for row in items if (row.stream, row.data_view) == (stream, data_view)]
            metrics: tuple[Literal["bounded_recovery", "net_development_value_usd"], ...] = (
                ("bounded_recovery", "net_development_value_usd")
                if stream == "portfolio_management"
                else ("bounded_recovery",)
            )
            for metric in metrics:
                values: dict[tuple[Stratum, str], list[float | None]] = defaultdict(list)
                for row in selected:
                    stratum = (row.clinical_profile_id, row.scenario_profile_id, row.asset_count)
                    values[stratum, row.world_id].append(
                        bounded_recovery(row.observational_loss)
                        if metric == "bounded_recovery"
                        else row.net_development_value_usd
                    )
                scored = sum(
                    (row.observational_loss if metric == "bounded_recovery" else row.net_development_value_usd)
                    is not None
                    for row in selected
                )
                estimate = low = high = None
                if all(value is not None for group in values.values() for value in group):
                    groups: dict[Stratum, list[float]] = defaultdict(list)
                    for (stratum, _), group in sorted(values.items()):
                        groups[stratum].append(mean(value for value in group if value is not None))
                    estimate = mean(mean(group) for group in groups.values())
                    if set(groups) == set(required_strata) and all(
                        len(group) >= policy.minimum_worlds_per_stratum for group in groups.values()
                    ):
                        rng = np.random.default_rng(policy.seed)
                        draws = np.zeros(policy.bootstrap_repetitions)
                        for stratum in sorted(groups):
                            vector = np.asarray(groups[stratum])
                            indices = rng.integers(0, len(vector), size=(policy.bootstrap_repetitions, len(vector)))
                            draws += vector[indices].mean(axis=1) / len(groups)
                        tail = (1 - policy.confidence_level) / 2
                        low, high = (float(value) for value in np.quantile(draws, [tail, 1 - tail]))
                rows.append(
                    LeaderboardRowV1(
                        benchmark="trialdev",
                        track=f"{stream}/{data_view}",
                        model_id=model,
                        metric=metric,
                        direction="higher_is_better",
                        estimate=estimate,
                        interval_low=low,
                        interval_high=high,
                        scheduled_tasks=len(selected),
                        scored_tasks=scored,
                        independent_worlds=len({world for _, world in values}),
                    )
                )
    return tuple(rows)


class TrialDevCampaignExportV1(BaseModel):
    """A complete native job export with an externally retained file digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    job_id: str
    path: Path
    sha256: Digest


class TrialDevCampaignProfileV1(TrialDevRankingMeanV1):
    """Model profile summary; task counts include scheduled repetitions.

    Linked resource conditions and repetitions are averaged within worlds before
    averaging worlds. Observed means retain their explicit scored denominator.
    """

    model_id: str
    observed_mean: float | None = None
    scored_worlds: int = Field(default=0, ge=0)


def summarize_trialdev_campaign_profiles(
    rankings: tuple[TrialDevRunRankingV1, ...],
) -> tuple[TrialDevCampaignProfileV1, ...]:
    """Summarize authenticated campaign rows using the existing equal-world owner."""
    return tuple(
        TrialDevCampaignProfileV1(model_id=model, **summary.model_dump())
        for model in sorted({ranking.model_id for ranking in rankings})
        for summary in ranking_means(
            [item for ranking in rankings if ranking.model_id == model for item in ranking.items]
        )
    )


class TrialDevCampaignAnalysisV1(BaseModel):
    """Release- and plan-bound comparisons, without a combined benchmark ranking.

    With budget censoring, comparisons and the leaderboard cover the paired
    panel of tasks every model completed; profile summaries cover each model's
    own completed tasks. Censored tasks are reported, never scored.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["trialagentbench.trialdev_campaign_analysis/v1"] = (
        "trialagentbench.trialdev_campaign_analysis/v1"
    )
    campaign_plan_sha256: Digest
    release_sha256: Digest
    exports: tuple[TrialDevCampaignExportV1, ...]
    comparisons: tuple[TrialDevComparisonV1, ...]
    profile_summaries: tuple[TrialDevCampaignProfileV1, ...] = ()
    leaderboard: tuple[LeaderboardRowV1, ...] = ()
    budget_censoring: BudgetCensoringV1 | None = Field(default=None, exclude_if=lambda value: value is None)


def _retain_tasks(ranking: TrialDevRunRankingV1, task_ids: frozenset[str]) -> TrialDevRunRankingV1:
    return ranking.model_copy(update={"items": tuple(row for row in ranking.items if row.task_id in task_ids)})


def analyse_trialdev_campaign(
    release: Path,
    *,
    expected_release_sha256: str,
    plan: CampaignPlanV1,
    expected_plan_checksum: str,
    exports: tuple[TrialDevCampaignExportV1, ...],
    censoring: BudgetCensoringV1 | None = None,
) -> TrialDevCampaignAnalysisV1:
    """Authenticate every planned TrialDev job before computing paired differences.

    Parameters
    ----------
    release
        Authenticated TrialDev evaluator archive or selective asset catalog.
    expected_release_sha256
        Externally retained release digest.
    plan
        Frozen campaign plan.
    expected_plan_checksum
        Externally retained campaign plan checksum.
    exports
        One full custody export per planned TrialDev job.
    censoring
        Budget censoring recorded at finish. Every verified pending task must
        be declared censored, and every censored task must still be pending.

    Returns
    -------
    TrialDevCampaignAnalysisV1
        Paired-panel comparisons and leaderboard, completed-only profiles.

    Raises
    ------
    ValueError
        If custody, the panel or the censoring record disagrees with the
        verified rankings, including pending tasks without budget censoring.
    """
    if plan.checksum != expected_plan_checksum:
        raise ValueError("Campaign plan differs from the prospectively retained identity")
    if censoring is not None and censoring.campaign_plan_sha256 != expected_plan_checksum:
        raise ValueError("Budget censoring belongs to a different campaign plan")
    jobs = {run.job_id: run for run in plan.runs if run.benchmark == "trialdev"}
    if censoring is not None and set(censoring.censored_by_job("trialdev")) != set(jobs):
        raise ValueError("Budget censoring must cover every planned TrialDev job")
    if not jobs or len(exports) != len(jobs) or {row.job_id for row in exports} != set(jobs):
        raise ValueError("TrialDev exports must cover every planned job exactly once")
    models = {row.name: row.model for row in plan.configuration.models}
    rankings: dict[str, TrialDevRunRankingV1] = {}
    with authenticated_archive(release, expected_sha256=expected_release_sha256) as archive:
        if archive.name.endswith(".assets.json"):
            catalog = read_asset_catalog(archive, expected_sha256=expected_release_sha256)
            selectors = set(plan.configuration.panel.trialdev_tasks)
            selected = tuple(
                task.task_id
                for task in catalog.tasks
                if (
                    bool(
                        selectors.intersection(
                            (task.task_id, task.case_id, task.stream, f"{task.case_id}:{task.stream}")
                        )
                    )
                    if selectors
                    else task.main
                )
            )
            if not selected:
                raise ValueError("Campaign selectors match no task in the authenticated release")
            with materialized_release(archive, task_ids=selected) as root:
                tasks = tuple(
                    _Task.model_validate_json(path.read_bytes())
                    for path in sorted((root / "participant/tasks").glob("*/task.json"))
                )
                worlds = tuple(
                    _World.model_validate_json(path.read_bytes())
                    for path in sorted((root / "evaluation/worlds").glob("*/world.json"))
                )
                if {world.world_id for world in worlds} != {task.case_id for task in catalog.tasks}:
                    raise ValueError("Selective campaign comparison requires complete shared world metadata")
        else:
            with ZipFile(archive) as files:
                tasks = tuple(
                    _Task.model_validate_json(files.read(name))
                    for name in files.namelist()
                    if name.startswith("participant/tasks/") and name.endswith("/task.json")
                )
                worlds = tuple(
                    _World.model_validate_json(files.read(name))
                    for name in files.namelist()
                    if name.startswith("evaluation/worlds/") and name.endswith("/world.json")
                )
        required = tuple(
            sorted(
                {
                    (
                        world.clinical_profile.profile_id,
                        world.difficulty_profile_id,
                        len(world.clinical_profile.candidate_labels),
                    )
                    for world in worlds
                }
            )
        )
        selectors = set(plan.configuration.panel.trialdev_tasks)
        selector_tasks: dict[str, set[str]] = defaultdict(set)
        for task in tasks:
            for selector in (task.task_id, task.case_id, task.stream, f"{task.case_id}:{task.stream}"):
                selector_tasks[selector].add(task.task_id)
        if selectors - selector_tasks.keys():
            raise ValueError("Campaign selectors match no task in the authenticated release")
        scheduled = (
            {task_id for selector in selectors for task_id in selector_tasks[selector]}
            if selectors
            else {task.task_id for task in tasks if task.data_view == "raw"}
        )
        for source in sorted(exports, key=lambda row: row.job_id):
            job = jobs[source.job_id]
            report = verify_trialdev_ranking(
                archive,
                source.path,
                expected_release_sha256=expected_release_sha256,
                expected_result_export_sha256=source.sha256,
                expected_invocation_sha256=job.invocation_sha256,
                development=plan.configuration.input_mode == "development",
            )
            if report.status != "pass":
                raise ValueError("Independent TrialDev ranking verification failed")
            ranking = TrialDevRunRankingV1.model_validate(report.checks[0].observed)
            if ranking.model_id != models[job.model_name] or {row.task_id for row in ranking.items} != scheduled:
                raise ValueError("Verified ranking differs from the frozen model or task panel")
            if censoring is not None and set(
                next(run for run in censoring.runs if run.job_id == job.job_id).scheduled_task_ids
            ) != set(scheduled):
                raise ValueError("Budget censoring differs from the frozen task panel")
            rankings[job.job_id] = ranking
    comparisons, profiles, leaderboard = paired_trialdev_results(
        rankings, censoring=censoring, required_strata=required, policy=plan.configuration.analysis
    )
    return TrialDevCampaignAnalysisV1(
        campaign_plan_sha256=expected_plan_checksum,
        release_sha256=expected_release_sha256,
        exports=exports,
        comparisons=comparisons,
        profile_summaries=profiles,
        leaderboard=leaderboard,
        budget_censoring=censoring,
    )


def paired_trialdev_results(
    rankings: dict[str, TrialDevRunRankingV1],
    *,
    censoring: BudgetCensoringV1 | None,
    required_strata: tuple[Stratum, ...],
    policy: CampaignAnalysisV1,
) -> tuple[tuple[TrialDevComparisonV1, ...], tuple[TrialDevCampaignProfileV1, ...], tuple[LeaderboardRowV1, ...]]:
    """Compare models on the paired panel and summarise each model over its completed tasks.

    Parameters
    ----------
    rankings
        Independently verified rankings keyed by campaign job identity.
    censoring
        Recorded budget censoring, or None when every task must be terminal.
    required_strata
        Release strata that weight worlds equally.
    policy
        Prospective comparison policy.

    Returns
    -------
    tuple
        Paired comparisons, completed-only profile summaries and paired
        leaderboard rows annotated with each model's censored count.

    Raises
    ------
    ValueError
        If a pending task is not declared budget-censored, or a declared
        censored task is not pending.
    """
    censored = {} if censoring is None else censoring.censored_by_job("trialdev")
    for job_id, ranking in rankings.items():
        pending = {row.task_id for row in ranking.items if row.execution_status in {"not_started", "incomplete_run"}}
        if pending != censored.get(job_id, frozenset()):
            raise ValueError(
                "Campaign comparison requires every scheduled terminal primary; "
                "only recorded budget censoring may leave a task pending"
            )
    scheduled = frozenset(row.task_id for ranking in rankings.values() for row in ranking.items)
    paired = scheduled if censoring is None else censoring.paired_task_ids("trialdev")
    compared = tuple(_retain_tasks(ranking, paired) for ranking in rankings.values())
    completed = tuple(
        _retain_tasks(ranking, scheduled - censored.get(job_id, frozenset())) for job_id, ranking in rankings.items()
    )
    comparisons = compare_trialdev_rankings(compared, required_strata=required_strata, policy=policy) if paired else ()
    leaderboard = trialdev_leaderboard_rows(compared, required_strata=required_strata, policy=policy) if paired else ()
    if censoring is not None:
        leaderboard = tuple(
            row.model_copy(update={"budget_censored_tasks": censoring.censored_count("trialdev", row.model_id)})
            for row in leaderboard
        )
    profiles = summarize_trialdev_campaign_profiles(tuple(row for row in completed if row.items))
    return comparisons, profiles, leaderboard


def render_trialdev_comparisons(report: TrialDevCampaignAnalysisV1, output: Path) -> None:
    """Write a compact comparison table and separate-scale accessible vector plot."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    with (output / "comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TrialDevComparisonV1.model_fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.model_dump() for row in report.comparisons)
    with (output / "profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TrialDevCampaignProfileV1.model_fields), lineterminator="\n")
        writer.writeheader()
        for profile in report.profile_summaries:
            values = profile.model_dump()
            values["unscored_task_counts"] = json.dumps(profile.unscored_task_counts, sort_keys=True)
            values["target_status_counts"] = (
                ""
                if profile.target_status_counts is None
                else json.dumps(profile.target_status_counts, sort_keys=True)
            )
            writer.writerow(values)
    for metric in sorted({row.metric for row in report.comparisons}):
        rows = [row for row in report.comparisons if row.metric == metric]
        height = 130 + 60 * max(1, len(rows))
        extent = (
            max(
                (
                    abs(value)
                    for row in rows
                    for value in (row.estimate, row.confidence_low, row.confidence_high)
                    if value is not None
                ),
                default=1,
            )
            or 1
        )

        def x(value: float, scale: float = extent) -> float:
            return 730 + 190 * value / scale

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">TrialDev {metric.replace("_", " ")}: paired model differences</title>',
            '<desc id="desc">Left model minus right model. Equal clinical, scenario and asset-count strata. Intervals resample independent worlds; missing support is labelled. No forced ordering.</desc>',
            '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" fill="#172b3a">',
            f'<text x="24" y="30" font-size="20">{metric.replace("_", " ").capitalize()} · paired differences</text>',
            '<text x="24" y="55" font-size="13">World-bootstrap intervals; Holm-adjusted p-values are in the table.</text>',
            f'<path d="M730 75 V{height-45}" stroke="#aab7bf"/>',
        ]
        if not rows:
            parts.append('<text x="24" y="100">No model pair scheduled.</text>')
        for index, row in enumerate(rows):
            y = 92 + index * 60
            parts.append(
                f'<text x="24" y="{y}" font-size="13">{escape(row.model_left)} − {escape(row.model_right)}</text>'
            )
            parts.append(
                f'<text x="24" y="{y+16}" font-size="11">{escape(row.stream)} · {escape(row.data_view)}</text>'
            )
            if row.estimate is not None:
                parts.append(f'<circle cx="{x(row.estimate):.2f}" cy="{y-4}" r="4" fill="#237c86"/>')
                parts.append(f'<text x="970" y="{y}" font-size="13">{row.estimate:.4g}</text>')
            if row.confidence_low is not None and row.confidence_high is not None:
                parts.append(
                    f'<path d="M{x(row.confidence_low):.2f} {y-4} H{x(row.confidence_high):.2f}" stroke="#237c86" stroke-width="2"/>'
                )
            if row.unavailable_reason:
                parts.append(
                    f'<text x="24" y="{y+30}" font-size="12">Inference unavailable: {escape(row.unavailable_reason.replace("_", " "))}</text>'
                )
        parts.extend([f'<text x="720" y="{height-20}" font-size="12">0</text>', "</g></svg>"])
        (output / f"{metric}.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")
