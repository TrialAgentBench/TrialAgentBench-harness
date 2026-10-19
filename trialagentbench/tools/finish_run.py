"""Grade, independently verify, export and report one prepared native run.

All calculations belong to the installed harness or standalone verifier. This
delivered recipe connects those existing public operations without selecting
successful cases, replacing primary attempts, or calculating another score.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from contextlib import closing
from pathlib import Path

from trialagentbench_verifier.campaign import (
    TrialEvalCampaignPlanV1,
    TrialEvalCampaignUnavailableV1,
    UnsupportedTrialEvalPanelError,
    analyse_trial_eval_campaign_v1,
    render_trial_eval_campaign_analysis_v1,
)
from trialagentbench_verifier.campaign_inputs import authenticate_development_inputs, verify_development_evaluator
from trialagentbench_verifier.contracts import VerificationReportV1
from trialagentbench_verifier.development import verify_trialdev_ranking
from trialagentbench_verifier.execution import ExecutionSubmissionV1, execute_submissions
from trialagentbench_verifier.io import sha256_file
from trialagentbench_verifier.release import ScoreSubmissionV1, score_submissions, verify_release
from trialagentbench_verifier.release_set import (
    ReleaseSetManifestV1,
    authenticate_release_set,
    role_filename,
    verify_release_set,
)
from trialagentbench_verifier.trialeval import TrialEvalScoreInputV1

from trialagentbench.cli import main as harness_main
from trialagentbench.contracts.release.archive_safety import materialized_release_archive
from trialagentbench.io.atomic import _sync_directory
from trialagentbench.tools.export_results import _artifact_files, _custody_tree_sha256, export_results_v1


def _discard_incomplete_writes(directory: Path) -> None:
    """Recover unpublished writer files before resuming an exclusively owned output.

    Only direct children with the writer's exact reserved name are disposable.
    Canonical outputs, arbitrary files and participant workspaces are untouched.
    """
    for path in directory.iterdir():
        if re.fullmatch(r"\..+\.[0-9a-f]{32}\.tmp", path.name):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Incomplete stage output is not a regular file: {path}")
            path.unlink()


def _write_once(path: Path, content: str | bytes) -> None:
    """Retain a completed stage and refuse a conflicting replay of its output."""
    encoded = content.encode("utf-8") if isinstance(content, str) else content
    if path.is_symlink() or path.exists():
        if path.is_symlink() or path.read_bytes() != encoded:
            raise ValueError(f"Existing stage output differs from verified inputs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Publish only a complete file; a concurrent first writer keeps ownership.
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != encoded:
                raise ValueError(f"Existing stage output differs from verified inputs: {path}") from None
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _retain_verified_report(path: Path, report: VerificationReportV1) -> None:
    """Preserve failed evidence and stop before presenting it as verified output."""
    _write_once(path, report.model_dump_json(indent=2) + "\n")
    if report.status != "pass":
        raise ValueError(f"Independent verification failed; inspect retained report: {path}")


def main(argv: Sequence[str] | None = None, *, qualified_manifest: ReleaseSetManifestV1 | None = None) -> int:
    """Execute offline release operations with externally retained trust anchors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("trialeval", "trialdev"))
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor-image-id")
    parser.add_argument("--campaign-plan", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--expected-invocation-sha256")
    parser.add_argument("--development", action="store_true", help="Finish explicitly unqualified rehearsal inputs.")
    parser.add_argument("--resume", action="store_true", help="Verify and reuse completed offline stages.")
    parser.add_argument(
        "--budget-censored",
        action="append",
        default=[],
        metavar="TASK_ID",
        help="TrialEval task left unrun by an exhausted campaign spending limit; reported, never scored.",
    )
    args = parser.parse_args(argv)
    if args.budget_censored and args.benchmark != "trialeval":
        parser.error("TrialDev rankings record unrun tasks natively; --budget-censored is TrialEval-only")
    if args.benchmark == "trialeval":
        if args.campaign_plan is None or args.expected_plan_sha256 is None or args.executor_image_id is None:
            parser.error("TrialEval requires its prospectively retained plan digest and local executor image ID")
        if sha256_file(args.campaign_plan) != args.expected_plan_sha256:
            raise ValueError("TrialEval plan differs from the prospectively retained identity")
        plan = TrialEvalCampaignPlanV1.model_validate_json(args.campaign_plan.read_bytes())
    elif args.expected_invocation_sha256 is None:
        parser.error("TrialDev requires its prospectively retained invocation digest")
    release_dir = args.release_dir.resolve(strict=True)
    run = args.run_root.resolve(strict=True)
    if args.output.is_symlink():
        raise ValueError("Offline output must not be a symbolic link")
    output = args.output.resolve()
    if args.development and qualified_manifest is not None:
        raise ValueError("Development finishing cannot receive a qualified release manifest")
    manifest = (
        authenticate_development_inputs(release_dir, expected_manifest_sha256=args.expected_manifest_sha256)
        if args.development
        else (
            verify_release_set(release_dir, expected_manifest_sha256=args.expected_manifest_sha256)
            if qualified_manifest is None
            else authenticate_release_set(release_dir, expected_manifest_sha256=args.expected_manifest_sha256)
        )
    )
    if qualified_manifest is not None and manifest != qualified_manifest:
        raise ValueError("Offline finishing received a different qualified release identity")
    anchors = {artifact.filename: artifact.sha256 for artifact in manifest.artifacts}
    evaluator = release_dir / role_filename(anchors, args.benchmark, "evaluator")
    evaluator_sha = anchors[evaluator.name]
    if args.development:
        development_report = verify_development_evaluator(
            evaluator, expected_sha256=evaluator_sha, benchmark=args.benchmark
        )
    if args.benchmark == "trialdev" and sha256_file(run / "invocation.json") != args.expected_invocation_sha256:
        raise ValueError("TrialDev invocation differs from the prospectively retained identity")
    output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        _discard_incomplete_writes(output)
    custody_path = output / "run-custody.sha256"
    if args.resume and not custody_path.is_file() and any(path.is_file() for path in output.rglob("*")):
        raise ValueError("Offline resume requires the original complete run-custody checkpoint")
    _write_once(custody_path, _custody_tree_sha256(run, _artifact_files(run)) + "\n")
    if args.development:
        _write_once(output / "development-inputs.json", manifest.model_dump_json(indent=2) + "\n")
    grade = output / "grade"
    verification = output / "verification"
    verification.mkdir(exist_ok=args.resume)
    if args.resume:
        _discard_incomplete_writes(verification)
    grade_arguments = (
        ["grade", "trialeval", str(run), "--evaluator-release", str(evaluator)]
        if args.benchmark == "trialeval"
        else ["grade", "trialdev", "--run-root", str(run), "--release", str(evaluator)]
    )
    if args.benchmark == "trialeval":
        grade_arguments.extend(["--replay-executor-image-id", args.executor_image_id])
        for task_id in args.budget_censored:
            grade_arguments.extend(["--budget-censored", task_id])
    if grade.exists():
        custody = "run_config.json" if args.benchmark == "trialeval" else "invocation.json"
        complete = "GRADE_MANIFEST.json" if args.benchmark == "trialeval" else "ranking.json"
        if (
            not args.resume
            or grade.is_symlink()
            or not (grade / complete).is_file()
            or sha256_file(grade / custody) != sha256_file(run / custody)
        ):
            raise ValueError("Retained native grade is incomplete or belongs to different run custody")
    elif harness_main([*grade_arguments, "--expected-sha256", evaluator_sha, "--out-dir", str(grade)]) != 0:
        raise RuntimeError("Native grading failed; retain its outputs and resolve the evaluator fault")
    release_report = (
        development_report if args.development else verify_release(evaluator, expected_sha256=evaluator_sha)
    )
    _retain_verified_report(verification / "release.json", release_report)
    if args.benchmark == "trialeval":
        participant = release_dir / role_filename(anchors, "trialeval", "participant")
        participant_sha = anchors[participant.name]
        inputs = sorted((grade / "verification_inputs").glob("*.json"))
        censored_attempts = frozenset(
            attempt.attempt_id for attempt in plan.attempts if attempt.task_id in set(args.budget_censored)
        )
        if {path.stem for path in inputs} != {
            attempt.task_id for attempt in plan.attempts if attempt.attempt_id not in censored_attempts
        }:
            raise ValueError("Graded primaries do not cover the frozen panel apart from budget censoring")
        replay_requests = tuple(
            ExecutionSubmissionV1(
                task_id=source.stem,
                submission=grade / "logs" / f"{source.stem}_workspace" / "item" / "scratch",
            )
            for source in inputs
            if TrialEvalScoreInputV1.model_validate_json(source.read_bytes()).attempt_status == "complete"
            and not (args.resume and (verification / f"{source.stem}-score.json").is_file())
            and not (args.resume and (verification / f"{source.stem}-execution.json").is_file())
        )
        with closing(
            execute_submissions(
                participant,
                replay_requests,
                expected_sha256=participant_sha,
                expected_executor_image_id=args.executor_image_id,
                expected_executor_environment_sha256=manifest.executor.environment_sha256,
                executor_image=args.executor_image_id,
            )
        ) as proofs:

            def requests() -> Iterator[ScoreSubmissionV1]:
                """Preserve each primary while sharing only authenticated archive inputs."""
                for source in inputs:
                    submitted = TrialEvalScoreInputV1.model_validate_json(source.read_bytes())
                    score_path = verification / f"{source.stem}-score.json"
                    if args.resume and score_path.is_file():
                        # The final authenticated campaign loader rechecks every retained
                        # report against the original grade, input, program and proof.
                        continue
                    proof_path = program = result = None
                    proof_sha = None
                    if submitted.attempt_status == "complete":
                        workspace = grade / "logs" / f"{source.stem}_workspace" / "item" / "scratch"
                        programs = [
                            workspace / name for name in ("analysis.py", "analysis.R") if (workspace / name).is_file()
                        ]
                        if len(programs) != 1:
                            raise ValueError("Complete primary does not expose exactly one replay program")
                        program = programs[0]
                        result = workspace / "estimates.csv"
                        proof_path = verification / f"{source.stem}-execution.json"
                        if not (args.resume and proof_path.is_file()):
                            proof = next(proofs)
                            if proof.task_id != source.stem:
                                raise ValueError("Batch replay differs from its fixed primary order")
                            _write_once(proof_path, proof.model_dump_json(indent=2) + "\n")
                        proof_sha = sha256_file(proof_path)
                    yield ScoreSubmissionV1(
                        submission=source,
                        execution_proof=proof_path,
                        expected_execution_proof_sha256=proof_sha,
                        program=program,
                        program_result=result,
                        graded_item=grade / "items" / source.name,
                    )

            with closing(
                score_submissions(evaluator, requests(), expected_sha256=evaluator_sha, development=args.development)
            ) as reports:
                for report in reports:
                    if report.score_binding is None:
                        raise ValueError("Score report omitted its primary custody binding")
                    task_id = report.score_binding.task_id
                    # Score component failures describe model outcomes. The campaign
                    # loader independently reconciles every component with its grade;
                    # rejecting these here would discard legitimate failed primaries.
                    _write_once(verification / f"{task_id}-score.json", report.model_dump_json(indent=2) + "\n")
                    print(f"Recorded primary {task_id}: score components {report.status}", flush=True)
    exported = output / "export"
    with (
        materialized_release_archive(evaluator, expected_sha256=evaluator_sha) as materialized,
        tempfile.TemporaryDirectory(prefix="trialagentbench-export-check-") as temporary,
    ):
        existing_export = args.resume and exported.exists()
        target = Path(temporary) / "export" if existing_export else exported
        export_results_v1(
            release_root=materialized.root,
            output_dir=target,
            run_roots=(run,),
            grade_roots=(grade,),
            verification_roots=(verification,),
            profile="analysis_reproduction" if args.benchmark == "trialeval" else "full_custody",
        )
        if existing_export:
            retained = {path.name: sha256_file(path) for path in exported.iterdir() if path.is_file()}
            rebuilt = {path.name: sha256_file(path) for path in target.iterdir() if path.is_file()}
            if retained != rebuilt:
                raise ValueError("Retained export differs from its complete current custody inputs")
    archives = list(exported.glob("*.zip"))
    if len(archives) != 1:
        raise ValueError("Native export did not produce exactly one result archive")
    archive = archives[0]
    if args.benchmark == "trialeval":
        try:
            analysis = analyse_trial_eval_campaign_v1(
                str(archive),
                expected_release_id=plan.release_id,
                expected_release_manifest_sha256=plan.release_manifest_sha256,
                expected_archive_sha256=sha256_file(archive),
                plan=plan,
                expected_campaign_plan_sha256=plan.checksum,
                budget_censored_attempt_ids=censored_attempts,
            )
        except UnsupportedTrialEvalPanelError as error:
            unavailable = TrialEvalCampaignUnavailableV1(
                reason=str(error), result_export_sha256=sha256_file(archive), campaign_plan_sha256=plan.checksum
            )
            _write_once(output / "analysis.json", unavailable.model_dump_json(indent=2) + "\n")
            _write_once(
                output / "analysis.md",
                f"Panel inference unavailable: {error}. Individual verified grades remain in the result export.\n",
            )
            return 0
        _write_once(output / "analysis.json", analysis.model_dump_json(indent=2) + "\n")
        _write_once(output / "analysis.md", render_trial_eval_campaign_analysis_v1(analysis))
    else:
        report = verify_trialdev_ranking(
            evaluator,
            archive,
            expected_release_sha256=evaluator_sha,
            expected_result_export_sha256=sha256_file(archive),
            expected_invocation_sha256=args.expected_invocation_sha256,
            development=args.development,
        )
        _retain_verified_report(output / "analysis.json", report)
    print(f"Retain the complete output directory: {output}")
    return 0


if __name__ == "__main__":
    main()
