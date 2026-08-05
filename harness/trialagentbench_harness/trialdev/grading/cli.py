"""CLI entrypoints for the TrialDev grader package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from trialagentbench_harness.trialdev.grading.grade import grade_bundle_v1, grade_item_v1, grade_report_payload_v1
from trialagentbench_harness.trialdev.grading.sequential import (
    advance_program_state_v1,
    build_initial_program_state_v1,
    grade_trajectory_v1,
    materialize_phase_v1,
    validate_program_state_file_v1,
)
from trialagentbench_harness.trialdev.grading.validate import validate_release_v1, validate_submission_v1

__all__ = ["main"]


def _cmd_validate_release(args: argparse.Namespace) -> int:
    validate_release_v1(scenario_root=Path(args.scenario_root))
    print("ok")
    return 0


def _cmd_validate_submission(args: argparse.Namespace) -> int:
    submission = validate_submission_v1(submission_path=Path(args.submission))
    print(json.dumps(submission.model_dump(mode="json", exclude_none=True), sort_keys=True, indent=2))
    return 0


def _cmd_grade_item(args: argparse.Namespace) -> int:
    report = grade_item_v1(
        scenario_root=Path(args.scenario_root),
        submission_path=Path(args.submission),
        write_path=None if args.out is None else Path(args.out),
        report_mode=str(args.report_mode),
        trial_output_root=None if args.trial_output_root is None else Path(args.trial_output_root),
    )
    print(
        json.dumps(
            grade_report_payload_v1(report=report, report_mode=str(args.report_mode)),
            sort_keys=True,
            indent=2,
        )
    )
    return 0


def _cmd_grade_bundle(args: argparse.Namespace) -> int:
    report = grade_bundle_v1(
        scenario_root=Path(args.scenario_root),
        submissions_root=Path(args.submissions_root),
        out_path=None if args.out is None else Path(args.out),
        report_mode=str(args.report_mode),
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


def _cmd_initial_state(args: argparse.Namespace) -> int:
    state = build_initial_program_state_v1(
        scenario_root=Path(args.scenario_root),
        programme_id=str(args.programme_id),
        objective_id=str(args.objective_id),
        out_path=None if args.out is None else Path(args.out),
    )
    print(json.dumps(state.model_dump(mode="json", exclude_none=True), sort_keys=True, indent=2))
    return 0


def _cmd_validate_state(args: argparse.Namespace) -> int:
    state = validate_program_state_file_v1(state_path=Path(args.state))
    print(json.dumps(state.model_dump(mode="json", exclude_none=True), sort_keys=True, indent=2))
    return 0


def _cmd_materialize_phase(args: argparse.Namespace) -> int:
    payload = materialize_phase_v1(
        scenario_root=Path(args.scenario_root),
        state_path=Path(args.state),
        request_path=Path(args.request),
        out_dir=Path(args.out),
        seed=int(args.seed),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


def _cmd_advance_state(args: argparse.Namespace) -> int:
    state = advance_program_state_v1(
        scenario_root=Path(args.scenario_root),
        state_path=Path(args.state),
        request_path=Path(args.request),
        trial_output_root=Path(args.trial_output_root),
        analysis_path=Path(args.analysis),
        decision_path=Path(args.decision),
        out_path=Path(args.out),
    )
    print(json.dumps(state.model_dump(mode="json", exclude_none=True), sort_keys=True, indent=2))
    return 0


def _cmd_grade_trajectory(args: argparse.Namespace) -> int:
    report = grade_trajectory_v1(
        scenario_root=Path(args.scenario_root),
        trajectory_root=Path(args.trajectory_root),
        initial_state_path=Path(args.initial_state),
        out_path=None if args.out is None else Path(args.out),
        report_mode=str(args.report_mode),
        scoring_context_path=None if args.scoring_context is None else Path(args.scoring_context),
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="trial-benchmark-grader")
    sub = parser.add_subparsers(dest="cmd", required=True)

    validate_release = sub.add_parser("validate-release", help="Validate the hidden grader surface.")
    validate_release.add_argument("--scenario-root", required=True)
    validate_release.set_defaults(_fn=_cmd_validate_release)

    validate_submission = sub.add_parser("validate-submission", help="Validate one structured submission.")
    validate_submission.add_argument("--submission", required=True)
    validate_submission.set_defaults(_fn=_cmd_validate_submission)

    grade_item = sub.add_parser("grade-item", help="Diagnostic only: grade one non-replayed submission.")
    grade_item.add_argument("--scenario-root", required=True)
    grade_item.add_argument("--submission", required=True)
    grade_item.add_argument("--out")
    grade_item.add_argument("--trial-output-root")
    grade_item.add_argument("--report-mode", choices=("score", "audit"), default="score")
    grade_item.set_defaults(_fn=_cmd_grade_item)

    grade_bundle = sub.add_parser(
        "grade-bundle", help="Diagnostic only: grade every non-replayed submission in a directory."
    )
    grade_bundle.add_argument("--scenario-root", required=True)
    grade_bundle.add_argument("--submissions-root", required=True)
    grade_bundle.add_argument("--out")
    grade_bundle.add_argument("--report-mode", choices=("score", "audit"), default="score")
    grade_bundle.set_defaults(_fn=_cmd_grade_bundle)

    initial_state = sub.add_parser("initial-state", help="Emit the initial evaluator-held program state.")
    initial_state.add_argument("--scenario-root", required=True)
    initial_state.add_argument("--programme-id", required=True)
    initial_state.add_argument("--objective-id", required=True)
    initial_state.add_argument("--out")
    initial_state.set_defaults(_fn=_cmd_initial_state)

    validate_state = sub.add_parser("validate-state", help="Validate one evaluator-held program state.")
    validate_state.add_argument("--state", required=True)
    validate_state.set_defaults(_fn=_cmd_validate_state)

    materialize_phase = sub.add_parser("materialize-phase", help="Materialize one sequential phase.")
    materialize_phase.add_argument("--scenario-root", required=True)
    materialize_phase.add_argument("--state", required=True)
    materialize_phase.add_argument("--request", required=True)
    materialize_phase.add_argument("--out", required=True)
    materialize_phase.add_argument("--seed", type=int, required=True)
    materialize_phase.add_argument("--overwrite", action="store_true")
    materialize_phase.set_defaults(_fn=_cmd_materialize_phase)

    advance_state = sub.add_parser("advance-state", help="Advance state after phase analysis and decision.")
    advance_state.add_argument("--scenario-root", required=True)
    advance_state.add_argument("--state", required=True)
    advance_state.add_argument("--request", required=True)
    advance_state.add_argument("--trial-output-root", required=True)
    advance_state.add_argument("--analysis", required=True)
    advance_state.add_argument("--decision", required=True)
    advance_state.add_argument("--out", required=True)
    advance_state.set_defaults(_fn=_cmd_advance_state)

    grade_trajectory = sub.add_parser("grade-trajectory", help="Replay and grade a TrialDev trajectory.")
    grade_trajectory.add_argument("--scenario-root", required=True)
    grade_trajectory.add_argument("--trajectory-root", required=True)
    grade_trajectory.add_argument("--initial-state", required=True)
    grade_trajectory.add_argument("--out")
    grade_trajectory.add_argument("--report-mode", choices=("score", "audit"), default="score")
    grade_trajectory.add_argument("--scoring-context")
    grade_trajectory.set_defaults(_fn=_cmd_grade_trajectory)

    args = parser.parse_args(argv)
    try:
        code = int(args._fn(args))
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        ValidationError,
    ) as exc:
        parser.error(str(exc))
        raise SystemExit(2) from exc
    raise SystemExit(code)
