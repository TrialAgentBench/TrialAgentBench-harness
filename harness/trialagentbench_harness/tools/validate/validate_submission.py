"""Score-blind linting for structured TrialEval and TrialDev submissions."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

from trialagentbench_harness.contracts.release.archive_safety import inspect_release_zip
from trialagentbench_harness.contracts.release.trialeval_runtime_surface import (
    TrialEvalSemanticSubmissionContractV1,
)
from trialagentbench_harness.contracts.submission import (
    SubmissionLintIssueV1,
    SubmissionLintReportV1,
    lint_submission_text_v1,
    render_submission_lint_v1,
)
from trialagentbench_harness.trialeval.data import (
    discover_participant_items,
    load_participant_method_dictionary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trialagentbench verify submission",
        description="Lint a structured submission without loading evaluator answers or scoring keys.",
    )
    parser.add_argument("--suite", choices=("trialeval", "trialdev"), required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument(
        "--participant",
        type=Path,
        help="Optional participant-role ZIP or extracted root for task-bound validation.",
    )
    parser.add_argument(
        "--output-format",
        choices=("human", "json"),
        default="human",
        help="Render compact human feedback or the deterministic JSON lint contract.",
    )
    return parser


def _read_submission(path: Path) -> str:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError("submission must be a regular non-symlink file")
    return target.read_text(encoding="utf-8")


@contextmanager
def _participant_root(path: Path) -> Iterator[Path]:
    target = Path(path)
    if target.is_dir() and not target.is_symlink():
        yield target
        return
    issues = inspect_release_zip(target)
    if issues:
        summary = "; ".join(
            f"{issue.code}{f' ({issue.member})' if issue.member else ''}: {issue.message}" for issue in issues
        )
        raise ValueError(f"participant archive is invalid: {summary}")
    with tempfile.TemporaryDirectory(prefix="trialagentbench-submission-lint-") as temporary:
        root = Path(temporary)
        with ZipFile(target) as archive:
            archive.extractall(root)
        yield root


def _participant_bound_trialeval_report(
    *,
    text: str,
    participant_root: Path,
    identity: str,
) -> SubmissionLintReportV1:
    try:
        item = discover_participant_items(participant_root, task_ids=(identity,))[identity]
        contract = TrialEvalSemanticSubmissionContractV1.model_validate(item.submission_contract)
        _, method_dictionary = load_participant_method_dictionary(participant_root)
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        return SubmissionLintReportV1(
            suite="trialeval",
            scope="participant_bound",
            valid=False,
            submission_identity=identity,
            issues=(
                SubmissionLintIssueV1(
                    code="participant_contract",
                    json_pointer="/task_id",
                    message=_bounded_message(str(exc)),
                ),
            ),
        )
    artifact_paths = tuple(
        sorted(path.relative_to(item.visible_dir).as_posix() for path in item.visible_dir.rglob("*") if path.is_file())
    )
    return lint_submission_text_v1(
        text,
        suite="trialeval",
        scope="participant_bound",
        expected_identity=identity,
        required_deliverables=contract.required_deliverables,
        participant_contract_checksum=contract.checksum,
        participant_artifact_paths=artifact_paths,
        participant_method_dictionary=method_dictionary,
    )


def _participant_bound_trialdev_report(
    *,
    text: str,
    participant_root: Path,
    identity: str,
) -> SubmissionLintReportV1:
    scenario_root = participant_root / f"scenario_{identity}" / "public"
    if not scenario_root.is_dir() or scenario_root.is_symlink():
        return SubmissionLintReportV1(
            suite="trialdev",
            scope="participant_bound",
            valid=False,
            submission_identity=identity,
            issues=(
                SubmissionLintIssueV1(
                    code="participant_contract",
                    json_pointer="/scenario_id",
                    message="Submitted scenario_id is absent from the participant release.",
                ),
            ),
        )
    return lint_submission_text_v1(
        text,
        suite="trialdev",
        scope="participant_bound",
        expected_identity=identity,
    )


def _bounded_message(message: str) -> str:
    normalized = " ".join(message.split())
    return normalized[:297] + "..." if len(normalized) > 300 else normalized


def _render(report: SubmissionLintReportV1, *, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True)
    return render_submission_lint_v1(report)


def main(argv: Sequence[str] | None = None) -> int:
    """Lint one submission against its public typed contract."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        text = _read_submission(args.submission)
    except (OSError, UnicodeError, ValueError) as exc:
        report = SubmissionLintReportV1(
            suite=args.suite,
            scope="schema_only",
            valid=False,
            issues=(
                SubmissionLintIssueV1(
                    code="participant_contract",
                    json_pointer="",
                    message=_bounded_message(str(exc)),
                ),
            ),
        )
        print(_render(report, output_format=args.output_format))
        return 1

    report = lint_submission_text_v1(text, suite=args.suite)
    if report.valid and args.participant is not None:
        identity = report.submission_identity
        if identity is None:
            raise RuntimeError("valid submission lint report omitted its identity")
        try:
            with _participant_root(args.participant) as participant_root:
                if args.suite == "trialeval":
                    report = _participant_bound_trialeval_report(
                        text=text,
                        participant_root=participant_root,
                        identity=identity,
                    )
                else:
                    report = _participant_bound_trialdev_report(
                        text=text,
                        participant_root=participant_root,
                        identity=identity,
                    )
        except (OSError, ValueError) as exc:
            report = SubmissionLintReportV1(
                suite=args.suite,
                scope="participant_bound",
                valid=False,
                submission_identity=identity,
                issues=(
                    SubmissionLintIssueV1(
                        code="participant_contract",
                        json_pointer="",
                        message=_bounded_message(str(exc)),
                    ),
                ),
            )
    print(_render(report, output_format=args.output_format))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
