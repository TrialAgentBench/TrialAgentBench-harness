"""Command-line interface for released-byte verification and score replay."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_verifier.campaign import (
    TrialEvalCampaignAnalysisV1,
    TrialEvalCampaignPlanV1,
    analyse_trial_eval_campaign_v1,
)
from trialagentbench_verifier.contracts import (
    ExecutionProofV1,
    PracticeVerificationReportV1,
    VerificationFailureReportV1,
    VerificationReportV1,
)
from trialagentbench_verifier.execution import execute_submission
from trialagentbench_verifier.io import VerificationInputError, read_json_object, sha256_file
from trialagentbench_verifier.practice import verify_practice_suite
from trialagentbench_verifier.release import score_submission, verify_release
from trialagentbench_verifier.release_set import (
    ExecutorAttestationV1,
    ReleaseSetManifestV1,
    ReproducibilityManifestV1,
    ToolkitManifestV1,
    attest_executor_from_toolkit,
    verify_release_set,
    verify_reproducibility_archive,
    verify_toolkit_archive,
)


def _parser() -> argparse.ArgumentParser:
    """Build the public verification parser."""

    parser = argparse.ArgumentParser(
        prog="trialagentbench-verify",
        description="Verify TrialAgentBench releases and reproduce submission scores.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    assets = commands.add_parser("assets", help="Package or inspect selectively downloadable benchmark assets.")
    asset_commands = assets.add_subparsers(dest="asset_command", required=True)
    pack = asset_commands.add_parser("pack", help="Losslessly package a pinned evaluator ZIP.")
    pack.add_argument("source", type=Path)
    pack.add_argument("--expected-sha256", required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--reuse-catalog", type=Path)
    pack.add_argument("--expected-reuse-sha256")
    inspect = asset_commands.add_parser("inspect", help="Inspect selection size without downloading payloads.")
    inspect.add_argument("catalog", type=Path)
    inspect.add_argument("--expected-sha256", required=True)
    inspect.add_argument("--task-id", action="append")
    inspect.add_argument(
        "--cache-directory", type=Path, help="Inspect existing verified cache reuse without downloading."
    )
    inspect.add_argument("--workers", type=int, default=1)
    inspect.add_argument("--retained-output-budget-bytes", type=int)
    fetch = asset_commands.add_parser("fetch", help="Acquire only selected verified shards into an explicit cache.")
    fetch.add_argument("catalog", type=Path)
    fetch.add_argument("--expected-sha256", required=True)
    fetch.add_argument("--task-id", action="append")
    fetch.add_argument("--cache-directory", type=Path, required=True)
    fetch.add_argument("--base-url", help="Explicit HTTPS directory containing shards; omit for local/offline reuse.")
    release = commands.add_parser("release", help="Verify a portable release directory or ZIP.")
    release.add_argument("release", type=Path)
    release.add_argument(
        "--expected-sha256",
        help=(
            "Trusted ZIP SHA-256, or release-manifest SHA-256 for a directory; "
            "required to authenticate evaluator archives."
        ),
    )
    release.add_argument("--output", type=Path, required=True)
    release.add_argument("--task-id", action="append", help="Verify only these asset-catalog tasks; repeat as needed.")
    release.add_argument(
        "--development",
        action="store_true",
        help="Check an authenticated evaluator ZIP without claiming scientific qualification.",
    )
    release.add_argument("--review-dir", type=Path, help="Also export an offline HTML/CSV review.")
    release.add_argument("--evidence-kind", choices=("local_control", "historical", "live"))
    score = commands.add_parser("score", help="Recompute a submission score from authenticated execution.")
    score.add_argument("release", type=Path)
    score.add_argument(
        "--expected-sha256",
        help=(
            "Trusted ZIP SHA-256, or release-manifest SHA-256 for a directory; "
            "required when scoring from an evaluator archive."
        ),
    )
    score.add_argument(
        "--submission",
        type=Path,
        required=True,
        help="Normalized score input, or estimates.csv for TrialEval or TrialDev initial analysis with a trusted execution proof.",
    )
    score.add_argument("--execution-proof", type=Path)
    score.add_argument("--expected-execution-proof-sha256")
    score.add_argument("--program", type=Path)
    score.add_argument("--program-result", type=Path)
    score.add_argument(
        "--graded-item",
        type=Path,
        help="Canonical graded-item JSON to bind into a campaign score-verification report.",
    )
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--review-dir", type=Path, help="Also export an offline HTML/CSV review.")
    score.add_argument("--evidence-kind", choices=("local_control", "historical", "live"))
    execute = commands.add_parser("execute", help="Replay one Python or R submission in isolated Docker.")
    execute.add_argument("release", type=Path, help="Participant release directory or ZIP.")
    execute.add_argument("--expected-sha256", required=True, help="Trusted participant release identity.")
    execute.add_argument("--task-id", required=True)
    execute.add_argument("--submission", type=Path, required=True, help="Directory containing submitted files.")
    execute.add_argument("--executor-image", help="Pinned TrialAgentBench executor image reference or ID.")
    execute.add_argument(
        "--expected-executor-image-id",
        required=True,
        help="Trusted immutable sha256: Docker image ID from the release record.",
    )
    execute.add_argument(
        "--expected-executor-environment-sha256",
        required=True,
        help="Trusted SHA-256 of the resolved executor inventory and resource limits.",
    )
    execute.add_argument("--output", type=Path, required=True)
    practice = commands.add_parser("practice", help="Run disclosed package-contained known-answer checks.")
    practice.add_argument("--output", type=Path, required=True)
    practice.add_argument("--review-dir", type=Path, help="Also export disclosed controls as HTML/CSV/SVG.")
    toolkit = commands.add_parser("toolkit", help="Verify the public software toolkit archive.")
    toolkit.add_argument("archive", type=Path)
    toolkit.add_argument("--expected-sha256", required=True)
    toolkit.add_argument("--output", type=Path, required=True)
    release_set = commands.add_parser("release-set", help="Verify the complete separated release directory.")
    release_set.add_argument("release_set", type=Path)
    release_set.add_argument("--expected-manifest-sha256", required=True)
    release_set.add_argument("--output", type=Path, required=True)
    reproducibility = commands.add_parser(
        "reproducibility", help="Verify and freshly recompute a post-result archive."
    )
    reproducibility.add_argument("archive", type=Path)
    reproducibility.add_argument("--expected-sha256", required=True)
    reproducibility.add_argument("--expected-base-release-sha256", required=True)
    reproducibility.add_argument("--output", type=Path, required=True)
    analyse = commands.add_parser("analyse-trialeval", help="Recompute the controlled TrialEval campaign report.")
    analyse.add_argument("--campaign-plan", type=Path, required=True)
    analyse.add_argument("--expected-campaign-plan-sha256", required=True)
    analyse.add_argument("--result-export", type=Path, required=True)
    analyse.add_argument("--expected-result-export-sha256", required=True)
    analyse.add_argument("--expected-release-id", required=True)
    analyse.add_argument("--expected-release-manifest-sha256", required=True)
    analyse.add_argument("--bootstrap-resamples", type=int, default=2_000)
    analyse.add_argument("--confidence-level", type=float, default=0.95)
    analyse.add_argument("--bootstrap-seed", type=int, default=65_009)
    analyse.add_argument("--output", type=Path, required=True)
    development = commands.add_parser(
        "analyse-trialdev", help="Independently verify one scheduled TrialDev run ranking."
    )
    development.add_argument("release", type=Path, help="Authorized evaluator ZIP.")
    development.add_argument("--expected-release-sha256", required=True)
    development.add_argument("--result-export", type=Path, required=True)
    development.add_argument("--expected-result-export-sha256", required=True)
    development.add_argument(
        "--expected-invocation-sha256", required=True, help="Prospectively retained invocation file SHA-256."
    )
    development.add_argument(
        "--development",
        action="store_true",
        help="Verify an explicitly unqualified development candidate; retain archive integrity and score checks.",
    )
    development.add_argument("--output", type=Path, required=True)
    development.add_argument("--review-dir", type=Path)
    development.add_argument("--evidence-kind", choices=("local_control", "historical", "live"))
    analyse.add_argument("--review-dir", type=Path, help="Also export campaign tables and recovery SVG.")
    analyse.add_argument(
        "--evidence-kind",
        choices=("local_control", "historical", "live"),
        help="Required for campaign review; descriptive label, not authenticated authority.",
    )
    attest = commands.add_parser(
        "attest-executor",
        help="Build and attest a local executor from an authenticated toolkit.",
    )
    attest.add_argument("toolkit", type=Path)
    attest.add_argument("--expected-toolkit-sha256", required=True)
    attest.add_argument("--executor-image", required=True)
    attest.add_argument("--output", type=Path, required=True)
    behaviour = commands.add_parser(
        "behaviour", help="Authenticate controlled trace tables and reproduce aggregate SVGs."
    )
    behaviour.add_argument("bundle", type=Path)
    behaviour.add_argument("--expected-manifest-sha256", required=True)
    behaviour.add_argument("--output", type=Path, required=True)
    campaign = commands.add_parser(
        "compare-trialdev", help="Verify all planned TrialDev exports and reproduce paired comparisons."
    )
    campaign.add_argument("release", type=Path)
    campaign.add_argument("--expected-release-sha256", required=True)
    campaign.add_argument("--campaign-plan", type=Path, required=True)
    campaign.add_argument(
        "--expected-plan-sha256", required=True, help="Prospectively retained campaign.json file SHA-256."
    )
    campaign.add_argument(
        "--exports",
        type=Path,
        required=True,
        help="JSON array of job_id, path and sha256 records; paths relative to this file.",
    )
    campaign.add_argument("--output", type=Path, required=True)
    return parser


def _write_report(path: Path, payload: object) -> None:
    """Write deterministic public JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one public verification operation."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "assets":
        from trialagentbench_verifier.assets import asset_footprint, pack_release_assets, read_asset_catalog

        if args.asset_command == "pack":
            if (args.reuse_catalog is None) != (args.expected_reuse_sha256 is None):
                raise ValueError("Reuse requires both a prior catalog and its trusted digest")
            paths = pack_release_assets(
                args.source,
                expected_sha256=args.expected_sha256,
                output=args.output,
                reuse=None if args.reuse_catalog is None else (args.reuse_catalog, args.expected_reuse_sha256),
            )
            print("\n".join(str(path) for path in paths))
        elif args.asset_command == "fetch":
            from trialagentbench_verifier.asset_cache import acquire_asset_selection

            path = acquire_asset_selection(
                args.catalog,
                expected_sha256=args.expected_sha256,
                cache=args.cache_directory,
                task_ids=None if args.task_id is None else tuple(args.task_id),
                base_url=args.base_url,
            )
            print(path)
        else:
            catalog = read_asset_catalog(args.catalog, expected_sha256=args.expected_sha256)
            tasks = None if args.task_id is None else tuple(args.task_id)
            from trialagentbench_verifier.asset_cache import cache_footprint

            storage = (
                None
                if args.cache_directory is None
                else cache_footprint(
                    args.catalog,
                    expected_sha256=args.expected_sha256,
                    cache=args.cache_directory,
                    task_ids=tasks,
                    workers=args.workers,
                    retained_outputs_budget_bytes=args.retained_output_budget_bytes,
                ).model_dump()
            )
            print(
                json.dumps(
                    {
                        "benchmark": catalog.benchmark,
                        "audience": catalog.audience,
                        "selected_tasks": len(catalog.tasks) if tasks is None else len(tasks),
                        "release_tasks": len(catalog.tasks),
                        "footprint": asset_footprint(catalog, tasks).model_dump(),
                        "cache": storage,
                        "verification": (
                            "catalog identity only; payloads have not been acquired or verified"
                            if storage is None
                            else "catalog identity and reported cached bytes; no downloads or scientific release qualification"
                        ),
                    },
                    indent=2,
                )
            )
        return 0
    if args.command == "compare-trialdev":
        from pydantic import TypeAdapter

        from trialagentbench_verifier.campaign_contracts import CampaignPlanV1
        from trialagentbench_verifier.development_campaign import (
            TrialDevCampaignExportV1,
            analyse_trialdev_campaign,
            render_trialdev_comparisons,
        )

        if sha256_file(args.campaign_plan) != args.expected_plan_sha256:
            raise ValueError("Campaign file differs from the prospectively retained identity")
        development_plan = CampaignPlanV1.model_validate_json(args.campaign_plan.read_bytes())
        sources = TypeAdapter(tuple[TrialDevCampaignExportV1, ...]).validate_json(args.exports.read_bytes())
        sources = tuple(
            row.model_copy(update={"path": (args.exports.parent / row.path).resolve(strict=True)}) for row in sources
        )
        development_report = analyse_trialdev_campaign(
            args.release,
            expected_release_sha256=args.expected_release_sha256,
            plan=development_plan,
            expected_plan_checksum=str(development_plan.checksum),
            exports=sources,
        )
        render_trialdev_comparisons(development_report, args.output)
        return 0
    if args.command == "behaviour":
        from trialagentbench_verifier.behaviour import reproduce_behaviour_figures

        reproduce_behaviour_figures(
            args.bundle, expected_manifest_sha256=args.expected_manifest_sha256, output=args.output
        )
        print(f"Aggregate behaviour tables and figures: {args.output}")
        return 0
    report: (
        ExecutionProofV1
        | ExecutorAttestationV1
        | PracticeVerificationReportV1
        | ReleaseSetManifestV1
        | ReproducibilityManifestV1
        | TrialEvalCampaignAnalysisV1
        | ToolkitManifestV1
        | VerificationReportV1
        | VerificationFailureReportV1
    )
    try:
        if args.command == "release":
            if args.development:
                from trialagentbench_verifier.campaign_inputs import verify_development_evaluator

                if args.task_id is not None:
                    raise ValueError("--task-id requires the released asset catalog, not --development")
                if args.expected_sha256 is None:
                    raise ValueError("Development verification requires --expected-sha256")
                report = verify_development_evaluator(args.release, expected_sha256=args.expected_sha256)
            else:
                report = verify_release(
                    args.release,
                    expected_sha256=args.expected_sha256,
                    task_ids=None if args.task_id is None else tuple(args.task_id),
                )
        elif args.command == "score":
            report = score_submission(
                args.release,
                args.submission,
                expected_sha256=args.expected_sha256,
                execution_proof=args.execution_proof,
                expected_execution_proof_sha256=args.expected_execution_proof_sha256,
                program=args.program,
                program_result=args.program_result,
                graded_item=args.graded_item,
            )
        elif args.command == "execute":
            report = execute_submission(
                args.release,
                args.submission,
                task_id=args.task_id,
                expected_sha256=args.expected_sha256,
                executor_image=args.executor_image,
                expected_executor_image_id=args.expected_executor_image_id,
                expected_executor_environment_sha256=args.expected_executor_environment_sha256,
            )
        elif args.command == "toolkit":
            report = verify_toolkit_archive(args.archive, expected_sha256=args.expected_sha256)
        elif args.command == "release-set":
            report = verify_release_set(
                args.release_set,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
        elif args.command == "reproducibility":
            report = verify_reproducibility_archive(
                args.archive,
                expected_sha256=args.expected_sha256,
                expected_base_release_sha256=args.expected_base_release_sha256,
            )
        elif args.command == "analyse-trialeval":
            if sha256_file(args.campaign_plan) != args.expected_campaign_plan_sha256:
                raise VerificationInputError("campaign plan differs from its externally trusted SHA-256")
            plan = TrialEvalCampaignPlanV1.model_validate(read_json_object(args.campaign_plan))
            report = analyse_trial_eval_campaign_v1(
                str(args.result_export),
                expected_release_id=args.expected_release_id,
                expected_release_manifest_sha256=args.expected_release_manifest_sha256,
                expected_archive_sha256=args.expected_result_export_sha256,
                plan=plan,
                expected_campaign_plan_sha256=plan.checksum,
                bootstrap_resamples=args.bootstrap_resamples,
                confidence_level=args.confidence_level,
                bootstrap_seed=args.bootstrap_seed,
            )
        elif args.command == "analyse-trialdev":
            from trialagentbench_verifier.development import verify_trialdev_ranking

            report = verify_trialdev_ranking(
                args.release,
                args.result_export,
                expected_release_sha256=args.expected_release_sha256,
                expected_result_export_sha256=args.expected_result_export_sha256,
                expected_invocation_sha256=args.expected_invocation_sha256,
                development=args.development,
            )
        elif args.command == "attest-executor":
            report = attest_executor_from_toolkit(
                args.toolkit,
                expected_toolkit_sha256=args.expected_toolkit_sha256,
                image_reference=args.executor_image,
            )
        else:
            report = verify_practice_suite()
    except (OSError, ValueError) as error:
        if args.command == "practice":
            raise
        report = VerificationFailureReportV1(
            operation=args.command,
            detail=str(error),
        )
    except RuntimeError as error:
        if args.command not in {"execute", "attest-executor"}:
            raise
        report = VerificationFailureReportV1(
            operation=args.command,
            failure_code="executor_unavailable",
            detail=str(error),
        )
    _write_report(args.output, report)
    if getattr(args, "review_dir", None) is not None:
        from trialagentbench_verifier.review import write_review

        if not isinstance(
            report,
            (
                PracticeVerificationReportV1,
                VerificationReportV1,
                TrialEvalCampaignAnalysisV1,
                VerificationFailureReportV1,
            ),
        ):
            raise TypeError("operation has no consumer review projection")
        write_review(report, args.review_dir, evidence_kind=getattr(args, "evidence_kind", None))
    if isinstance(
        report,
        (
            ExecutorAttestationV1,
            ReleaseSetManifestV1,
            ReproducibilityManifestV1,
            ToolkitManifestV1,
            TrialEvalCampaignAnalysisV1,
        ),
    ):
        return 0
    return 0 if report.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
