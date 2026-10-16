"""Project frozen native campaign custody into the independent comparison owners."""

from __future__ import annotations

import csv
import tempfile
from html import escape
from itertools import combinations
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict
from trialagentbench_verifier.campaign import (
    TrialEvalCampaignAttemptV1,
    TrialEvalCampaignPlanV1,
    TrialEvalCampaignUnavailableV1,
    UnsupportedTrialEvalPanelError,
    analyse_trial_eval_campaign_v1,
)
from trialagentbench_verifier.campaign_contracts import BudgetCensoringV1, CensoredRunV1, LeaderboardRowV1
from trialagentbench_verifier.campaign_inputs import DevelopmentCampaignInputsV1
from trialagentbench_verifier.development_campaign import (
    TrialDevCampaignExportV1,
    analyse_trialdev_campaign,
    render_trialdev_comparisons,
)
from trialagentbench_verifier.io import sha256_file
from trialagentbench_verifier.public_results import (
    CampaignLeaderboardV1,
    PublicTrialDevCampaignV1,
    PublicTrialEvalCampaignV1,
    public_trialdev_campaign,
    public_trialeval_campaign,
    trialeval_leaderboard_rows,
)
from trialagentbench_verifier.release_set import ReleaseSetManifestV1, role_filename
from trialagentbench_verifier.review import write_review

from trialagentbench.contracts.campaign import CampaignPlanV1
from trialagentbench.contracts.core.runs import RunCoverageV1
from trialagentbench.contracts.release.archive_safety import materialized_release_archive
from trialagentbench.io import read_json_model
from trialagentbench.tools.export_results import export_results_v1
from trialagentbench.tools.finish_run import _write_once
from trialagentbench.util.campaign_control import CAMPAIGN_BUDGET_SCOPE, exhausted_budgets


def campaign_censoring(root: Path, plan: CampaignPlanV1) -> BudgetCensoringV1 | None:
    """Classify every pending scheduled task, refusing any not explained by spending-limit exhaustion.

    A pending task is budget-censored only when the campaign ledger records the
    permanent exhaustion of its model's or the campaign's spending limit. Such
    work can never run, because budgets never refill. Any other pending task
    (a crash or interruption) can resume, so finishing is refused.

    Returns
    -------
    BudgetCensoringV1 or None
        Censoring of every planned job, or None when no task is censored.

    Raises
    ------
    ValueError
        If a run has pending tasks without recorded budget exhaustion.
    """
    exhausted = exhausted_budgets(root / "requests.sqlite", plan.configuration)
    models = {model.name: model.model for model in plan.configuration.models}
    runs = []
    for run in plan.runs:
        coverage = read_json_model(RunCoverageV1, root / run.relative_run_root / "coverage.json")
        pending = tuple(task for task in coverage.unit_ids if task not in coverage.completed_unit_ids)
        if pending and CAMPAIGN_BUDGET_SCOPE not in exhausted and run.model_name not in exhausted:
            raise ValueError(
                f"Finish requires all scheduled terminal checkpoints; {run.job_id} has {len(pending)} pending "
                "tasks without an exhausted spending limit. Resume pending work first."
            )
        runs.append(
            CensoredRunV1(
                job_id=run.job_id,
                model_name=run.model_name,
                model_id=models[run.model_name],
                benchmark=run.benchmark,
                repetition=run.repetition,
                scheduled_task_ids=coverage.unit_ids,
                censored_task_ids=pending,
            )
        )
    if not any(run.censored_task_ids for run in runs):
        return None
    return BudgetCensoringV1(
        campaign_plan_sha256=str(plan.checksum),
        exhausted_models=tuple(sorted(name for name in exhausted if name != CAMPAIGN_BUDGET_SCOPE)),
        campaign_budget_exhausted=CAMPAIGN_BUDGET_SCOPE in exhausted,
        runs=tuple(runs),
    )


def write_campaign_index(root: Path, plan: CampaignPlanV1) -> Path:
    """Link canonical campaign outputs without recalculating scientific results.

    The portable page is evaluator-side: it links submitted work and verification
    records which may disclose grading references. Only the explicitly marked
    public projections are intended for unrestricted sharing.
    """

    def link(path: Path, label: str) -> str:
        if not path.is_file():
            return ""
        relative = path.relative_to(root).as_posix()
        return f'<li><a href="{quote(relative)}">{escape(label)}</a></li>'

    censoring_path = root / "comparisons" / "censoring.json"
    censoring = (
        BudgetCensoringV1.model_validate_json(censoring_path.read_bytes()) if censoring_path.is_file() else None
    )
    censored = {} if censoring is None else {run.job_id: len(run.censored_task_ids) for run in censoring.runs}
    rows: list[str] = []
    for run in plan.runs:
        native = root / run.relative_run_root
        coverage = read_json_model(RunCoverageV1, native / "coverage.json")
        result = root / "results" / run.job_id
        links = [
            link(native / "coverage.json", "Scheduled and terminal task identities"),
            link(native / "run_index.json", "Submitted work and task locations"),
            link(native / "provider_telemetry_summary.json", "Model usage"),
            link(result / "grade" / "RESULTS_SUMMARY.md", "Numerical recovery, completion and failure origins"),
            link(result / "grade" / "summary.json", "Canonical task grades"),
            link(result / "grade" / "ranking.json", "Development results and completion"),
        ]
        links.extend(
            link(path, f"Verification: {path.stem}") for path in sorted((result / "verification").glob("*.json"))
        )
        rows.append(
            f"<section><h2>{escape(run.job_id)}</h2><p>{escape(run.benchmark)} · "
            f"repetition {run.repetition} · terminal checkpoints "
            f"{len(coverage.completed_unit_ids)}/{len(coverage.unit_ids)}"
            f"{f' · budget-censored {censored[run.job_id]}' if censored.get(run.job_id) else ''}</p>"
            "<p>Terminal checkpoints include unsuccessful attempts; they do not establish "
            "reproduced analyses or scientific success. Consult the native grades below.</p>"
            f'<ul>{"".join(links)}</ul></section>'
        )
    comparisons: list[str] = []
    for benchmark in plan.configuration.panel.benchmarks:
        directory = root / "comparisons" / benchmark
        links = [
            link(directory / "review" / "index.html", "Scientific comparison report"),
            link(directory / "analysis.json", "Analysis and unavailable-comparison reasons"),
            link(directory / "public-results.json", "Public summary — shareable projection"),
            link(directory / "development-results.json", "Development results — unqualified rehearsal"),
        ]
        links.extend(link(path, path.stem.replace("_", " ")) for path in sorted(directory.glob("*.csv")))
        comparisons.append(f'<h3>{escape(benchmark)}</h3><ul>{"".join(links)}</ul>')
    comparisons.append(
        "<h3>Leaderboard</h3><ul>"
        + link(root / "comparisons" / "leaderboard.json", "Leaderboard — shareable, world-clustered intervals")
        + link(root / "comparisons" / "leaderboard.csv", "Leaderboard table")
        + "</ul>"
    )
    if censoring is not None:
        comparisons.append(
            "<h3>Budget censoring</h3><p>Exhausted spending limits left these tasks unrun. They are unscored, "
            "not participant failures; comparisons and the leaderboard use only tasks every model completed.</p>"
            + "".join(f'<p role="alert"><strong>Warning:</strong> {escape(text)}</p>' for text in censoring.warnings)
            + "<ul>"
            + "".join(
                f"<li>{escape(row.benchmark)} · {escape(row.model_id)}: {row.budget_censored_tasks}/"
                f"{row.scheduled_tasks} censored ({row.censoring_rate:.0%})</li>"
                for row in censoring.models
            )
            + link(censoring_path, "Censoring record")
            + "</ul>"
        )
    diagnostics = "".join(
        link(path, f"{directory}: {path.stem.replace('_', ' ')}")
        for directory in ("behaviour-summary", "resources")
        for path in sorted((root / directory).glob("*"))
        if path.suffix in {".html", ".md", ".csv", ".svg"}
    )
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Campaign results · TrialAgentBench</title><style>"
        "body{font:17px/1.6 system-ui,sans-serif;color:#183c38;background:#faf9f5;max-width:70rem;"
        "margin:auto;padding:2rem}a{color:#165c70;overflow-wrap:anywhere}h1,h2,h3{line-height:1.2}"
        "section{border-top:1px solid #b6c9c3;margin-top:2rem;padding-top:1rem}"
        "li{margin:.4rem 0}code{overflow-wrap:anywhere}a:focus-visible{outline:3px solid #b75e24}"
        "</style><main><h1>Campaign results</h1>"
        "<p>Inspect the submitted work, its independent replay and the resulting scientific outcomes.</p>"
        f"<p>Analysis language: <strong>{escape(plan.configuration.panel.analysis_language)}</strong>. "
        f"Native runs: {len(plan.runs)}. Input mode: {escape(plan.configuration.input_mode)}.</p>"
        "<p>This evaluator report links records that may contain reference answers. For public sharing, "
        "use the designated public summary files. A diagnostic subset may have individual grades "
        "without supporting a benchmark ranking.</p><ul>"
        + link(root / "configuration.json", "Exact evaluation configuration")
        + link(root / "campaign.json", "Frozen campaign plan")
        + "</ul><section><h2>Scientific comparisons</h2>"
        + "".join(comparisons)
        + "</section><section><h2>Behavior and resources</h2><ul>"
        + diagnostics
        + "</ul></section>"
        + "".join(rows)
        + "</main></html>\n"
    )
    destination = root / "index.html"
    _write_once(destination, html)
    return destination


class _DevelopmentResults(BaseModel):
    """Keep rehearsal aggregates incompatible with qualified public result schemas."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["trialagentbench.development_campaign_results/v1"] = (
        "trialagentbench.development_campaign_results/v1"
    )
    status: Literal["unqualified"] = "unqualified"
    input_manifest_sha256: str
    result: PublicTrialEvalCampaignV1 | PublicTrialDevCampaignV1 | TrialEvalCampaignUnavailableV1


def _write_aggregate(
    destination: Path,
    result: PublicTrialEvalCampaignV1 | PublicTrialDevCampaignV1 | TrialEvalCampaignUnavailableV1,
    plan: CampaignPlanV1,
) -> None:
    if plan.configuration.input_mode == "development":
        (destination / "development-results.json").write_text(
            _DevelopmentResults(
                input_manifest_sha256=plan.configuration.expected_release_manifest_sha256, result=result
            ).model_dump_json(indent=2)
            + "\n",
            encoding="utf-8",
        )
    else:
        (destination / "public-results.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _write_leaderboard(
    destination: Path,
    plan: CampaignPlanV1,
    rows: tuple[LeaderboardRowV1, ...],
    censoring: BudgetCensoringV1 | None,
) -> None:
    """Write the shareable leaderboard as JSON and as a flat CSV table."""
    analysis = plan.configuration.analysis
    leaderboard = CampaignLeaderboardV1(
        campaign_plan_sha256=str(plan.checksum),
        release_set_manifest_sha256=plan.configuration.expected_release_manifest_sha256,
        split=plan.configuration.panel.split,
        confidence_level=analysis.confidence_level,
        bootstrap_resamples=analysis.bootstrap_repetitions,
        bootstrap_seed=analysis.seed,
        rows=rows,
        budget_censoring=censoring,
        warnings=() if censoring is None else censoring.warnings,
    )
    (destination / "leaderboard.json").write_text(leaderboard.model_dump_json(indent=2) + "\n", encoding="utf-8")
    with (destination / "leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LeaderboardRowV1.model_fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.model_dump(mode="json") for row in rows)


def combined_trialeval_plan(root: Path, plan: CampaignPlanV1) -> TrialEvalCampaignPlanV1:
    """Derive the combined plan solely from prospectively frozen per-run plans.

    Repetition one is the primary by design. Later repetitions remain linked
    stochastic duplicates in the existing variability analysis; no outcome can
    select a primary. TrialDev instead averages repetitions within each world.
    """
    runs = tuple(run for run in plan.runs if run.benchmark == "trialeval")
    sources = tuple(read_json_model(TrialEvalCampaignPlanV1, root / str(run.analysis_plan_path)) for run in runs)
    if not sources:
        raise ValueError("No TrialEval runs were scheduled")
    first = sources[0]
    identity = (
        first.release_id,
        first.release_manifest_sha256,
        first.participant_manifest_checksum,
        first.prospective_analysis_plan_sha256,
        first.cells,
    )
    attempts: list[TrialEvalCampaignAttemptV1] = []
    for run, source in zip(runs, sources, strict=True):
        if sha256_file(root / str(run.analysis_plan_path)) != run.analysis_plan_sha256:
            raise ValueError("Native TrialEval plan differs from its frozen identity")
        if (
            source.release_id,
            source.release_manifest_sha256,
            source.participant_manifest_checksum,
            source.prospective_analysis_plan_sha256,
            source.cells,
        ) != identity:
            raise ValueError("Combined TrialEval plans differ in scientific condition or task cells")
        attempts.extend(
            row.model_copy(
                update={
                    "attempt_role": "primary" if run.repetition == 1 else "stochastic_duplicate",
                    "scheduled_unit_id": row.attempt_id,
                }
            )
            for row in source.attempts
        )
    payload = first.model_dump(exclude={"checksum", "attempts", "model_comparisons"})
    return TrialEvalCampaignPlanV1.model_validate(
        {
            **payload,
            "attempts": attempts,
            "model_comparisons": tuple(combinations(sorted(row.model for row in plan.configuration.models), 2)),
        }
    )


def finish_comparisons(
    root: Path, plan: CampaignPlanV1, manifest: ReleaseSetManifestV1 | DevelopmentCampaignInputsV1
) -> None:
    """Reproduce all scheduled model comparisons without provider calls or fresh simulations.

    Budget-censored tasks are excluded from every model's comparison and
    leaderboard panel; their counts, rates and warnings are retained.
    """
    configuration = plan.configuration
    censoring = campaign_censoring(root, plan)
    anchors = {row.filename: row.sha256 for row in manifest.artifacts}
    evaluator_role = "evaluator" if configuration.panel.split == "public" else "withheld-evaluator"
    leaderboard: list[LeaderboardRowV1] = []
    with tempfile.TemporaryDirectory(prefix="trialagentbench-comparisons-") as temporary:
        working = Path(temporary)
        if "trialdev" in configuration.panel.benchmarks:
            evaluator_name = role_filename(anchors, "trialdev", evaluator_role)
            exports = []
            for run in plan.runs:
                if run.benchmark == "trialdev":
                    archives = tuple((root / "results" / run.job_id / "export").glob("*.zip"))
                    if len(archives) != 1:
                        raise ValueError("Each TrialDev job must retain exactly one full custody export")
                    exports.append(
                        TrialDevCampaignExportV1(job_id=run.job_id, path=archives[0], sha256=sha256_file(archives[0]))
                    )
            report = analyse_trialdev_campaign(
                configuration.release_directory / evaluator_name,
                expected_release_sha256=anchors[evaluator_name],
                plan=plan,
                expected_plan_checksum=str(plan.checksum),
                exports=tuple(exports),
                censoring=censoring,
            )
            # Portable relative locations remain resolvable from the copied export index.
            from pydantic import TypeAdapter

            index = tuple(
                row.model_copy(update={"path": Path("../..") / row.path.relative_to(root)}) for row in exports
            )
            report = report.model_copy(update={"exports": index})
            leaderboard.extend(report.leaderboard)
            render_trialdev_comparisons(report, working / "trialdev")
            _write_aggregate(
                working / "trialdev", public_trialdev_campaign(report, policy=configuration.analysis), plan
            )
            (working / "trialdev" / "exports.json").write_bytes(
                TypeAdapter(tuple[TrialDevCampaignExportV1, ...]).dump_json(index, indent=2)
            )
        unrun = set() if censoring is None else {row.job_id for row in censoring.runs if row.wholly_censored}
        selected = tuple(run for run in plan.runs if run.benchmark == "trialeval")
        graded = tuple(run for run in selected if run.job_id not in unrun)
        # With every TrialEval item censored there is no result to export; censoring.json reports it.
        if "trialeval" in configuration.panel.benchmarks and not (unrun and not graded):
            evaluator_name = role_filename(anchors, "trialeval", evaluator_role)
            combined = combined_trialeval_plan(root, plan)
            censored_tasks = {} if censoring is None else censoring.censored_by_job("trialeval")
            censored_attempts = frozenset(
                attempt.attempt_id
                for run in selected
                for attempt in read_json_model(TrialEvalCampaignPlanV1, root / str(run.analysis_plan_path)).attempts
                if attempt.task_id in censored_tasks.get(run.job_id, frozenset())
            )
            destination = working / "trialeval"
            destination.mkdir()
            (destination / "plan.json").write_text(combined.model_dump_json(indent=2) + "\n", encoding="utf-8")
            with materialized_release_archive(
                configuration.release_directory / evaluator_name,
                expected_sha256=anchors[evaluator_name],
            ) as release:
                export_results_v1(
                    release_root=release.root,
                    output_dir=destination / "export",
                    run_roots=tuple(root / run.relative_run_root for run in graded),
                    grade_roots=tuple(root / "results" / run.job_id / "grade" for run in graded),
                    verification_roots=tuple(root / "results" / run.job_id / "verification" for run in graded),
                    profile="analysis_reproduction",
                )
            archives = tuple((destination / "export").glob("*.zip"))
            if len(archives) != 1:
                raise ValueError("Combined TrialEval export must contain exactly one archive")
            public_result: PublicTrialEvalCampaignV1 | TrialEvalCampaignUnavailableV1
            try:
                analysis = analyse_trial_eval_campaign_v1(
                    str(archives[0]),
                    expected_release_id=combined.release_id,
                    expected_release_manifest_sha256=combined.release_manifest_sha256,
                    expected_archive_sha256=sha256_file(archives[0]),
                    plan=combined,
                    expected_campaign_plan_sha256=combined.checksum,
                    bootstrap_resamples=configuration.analysis.bootstrap_repetitions,
                    confidence_level=configuration.analysis.confidence_level,
                    bootstrap_seed=configuration.analysis.seed,
                    budget_censored_attempt_ids=censored_attempts,
                )
            except UnsupportedTrialEvalPanelError as error:
                unavailable = TrialEvalCampaignUnavailableV1(
                    reason=str(error),
                    result_export_sha256=sha256_file(archives[0]),
                    campaign_plan_sha256=combined.checksum,
                )
                (destination / "analysis.json").write_text(
                    unavailable.model_dump_json(indent=2) + "\n", encoding="utf-8"
                )
                public_result = unavailable
            else:
                write_review(analysis, destination / "review", evidence_kind="unclassified")
                (destination / "analysis.json").write_text(analysis.model_dump_json(indent=2) + "\n", encoding="utf-8")
                public_result = public_trialeval_campaign(analysis)
                leaderboard.extend(trialeval_leaderboard_rows(analysis, censoring=censoring))
            _write_aggregate(destination, public_result, plan)
        if configuration.input_mode == "qualified" and leaderboard:
            _write_leaderboard(working, plan, tuple(leaderboard), censoring)
        if censoring is not None:
            (working / "censoring.json").write_text(censoring.model_dump_json(indent=2) + "\n", encoding="utf-8")
            for warning in censoring.warnings:
                print(f"WARNING: {warning}", flush=True)
        output = root / "comparisons"
        for source in sorted(working.rglob("*")):
            if source.is_file():
                target = output / source.relative_to(working)
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.suffix == ".zip":
                    if target.exists():
                        if sha256_file(target) != sha256_file(source):
                            raise ValueError("Retained comparison export differs from rebuilt custody")
                    else:
                        target.write_bytes(source.read_bytes())
                else:
                    _write_once(target, source.read_bytes())
