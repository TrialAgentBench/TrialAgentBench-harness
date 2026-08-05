"""Deterministically grade a canonical TrialDevBench run tree.

The grader reads canonical submissions, evaluates every declared programme,
and refreshes aggregate outputs without invoking a model or network service.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import cast

from trialagentbench_harness.adapters import trialdev_upstream
from trialagentbench_harness.contracts.core.manifest import GradeManifestV1
from trialagentbench_harness.contracts.trialdev.metrics import TrialDevAssessmentPortfolioV1
from trialagentbench_harness.io import read_json_model, sha256_path, staged_directory, write_json, write_json_model
from trialagentbench_harness.tools.grade.release_pair import materialized_trialdev_release_root
from trialagentbench_harness.trialdev.aggregate import aggregate_run
from trialagentbench_harness.trialdev.assessment import (
    build_portfolio_programme_assessment_v1,
    build_single_asset_programme_assessment_v1,
)
from trialagentbench_harness.trialdev.data import scenario_root
from trialagentbench_harness.trialdev.grade_wrappers import (
    phase_policy_modes_from_manifest,
    summarise_programme_analysis_quality,
    trajectory_metrics_from_grade,
)
from trialagentbench_harness.trialdev.metrics import summarize_trialdev_metrics_v1
from trialagentbench_harness.util import head_sha


def _grade_portfolio_programme(*, program_dir: Path, bundle: Path) -> None:
    """Recompute every portfolio grade and verify its persisted state chain."""

    from trialagentbench_harness.contracts.trialdev.portfolio_submission import (
        TrialDevPortfolioCheckpointSubmissionV1,
        TrialDevPortfolioRunSummaryV1,
    )
    from trialagentbench_harness.contracts.trialdev.programme import (
        TrialDevPortfolioCheckpointActionPolicyV1,
        TrialDevPortfolioEvidenceIndexV1,
        TrialDevPortfolioProgrammeStateV1,
    )
    from trialagentbench_harness.trialdev.portfolio_grading import grade_portfolio_checkpoint_v1
    from trialagentbench_harness.trialdev.portfolio_release import portfolio_evaluator_view_v1
    from trialagentbench_harness.trialdev.programme import (
        build_checkpoint_action_policy_v1,
        transition_portfolio_programme_state_v1,
    )

    summary = read_json_model(TrialDevPortfolioRunSummaryV1, program_dir / "portfolio_run_summary.json")
    evaluator_view = portfolio_evaluator_view_v1(bundle, summary.programme_id)
    index = read_json_model(
        TrialDevPortfolioEvidenceIndexV1,
        bundle / evaluator_view.evidence_index_relative_path,
    )
    states = tuple(
        read_json_model(TrialDevPortfolioProgrammeStateV1, program_dir / relative_path)
        for relative_path in summary.state_relative_paths
    )
    for offset, (submission_relative_path, grade_relative_path) in enumerate(
        zip(summary.submission_relative_paths, summary.grade_relative_paths, strict=True)
    ):
        state = states[offset]
        submission = read_json_model(
            TrialDevPortfolioCheckpointSubmissionV1,
            program_dir / submission_relative_path,
        )
        grade = grade_portfolio_checkpoint_v1(
            release_root=bundle,
            state=state,
            submission=submission,
        )
        write_json_model(program_dir / grade_relative_path, grade)
        expected_next = transition_portfolio_programme_state_v1(
            state=state,
            evidence_index=index,
            action_policy=cast(
                TrialDevPortfolioCheckpointActionPolicyV1,
                build_checkpoint_action_policy_v1(state=state),
            ),
            selection=submission.selected_action,
            outcome=grade.outcome,
        )
        if expected_next != states[offset + 1]:
            raise ValueError(f"Portfolio state transition mismatch after {grade.checkpoint_id!r}.")


def _grade_portfolio_run(*, run_root: Path, bundle: Path, run_config: object) -> int:
    """Grade and aggregate all portfolio programmes from immutable submissions."""

    from trialagentbench_harness.contracts.core.runs import TrialDevRunConfigV1
    from trialagentbench_harness.trialdev.portfolio_release import (
        load_portfolio_catalogue_v1,
        validate_portfolio_release_v1,
    )

    config = TrialDevRunConfigV1.model_validate(run_config)
    validate_portfolio_release_v1(bundle)
    catalogue = load_portfolio_catalogue_v1(bundle)
    assessments = []
    programs_root = run_root / "programs"
    for program_dir in sorted(path for path in programs_root.iterdir() if path.is_dir()):
        _grade_portfolio_programme(program_dir=program_dir, bundle=bundle)
        assessment = build_portfolio_programme_assessment_v1(
            program_dir=program_dir,
            run_config=config,
            release_id=catalogue.release_id,
        )
        write_json_model(program_dir / "programme_assessment.json", assessment)
        assessments.append(assessment)
        sys.stdout.write(f"  graded portfolio {program_dir.name}\n")
    portfolio = TrialDevAssessmentPortfolioV1(programmes=tuple(assessments))
    write_json_model(run_root / "trialdev_assessments.json", portfolio)
    write_json_model(
        run_root / "trialdev_metrics.json",
        summarize_trialdev_metrics_v1(portfolio.programmes),
    )
    return len(assessments)


logger = logging.getLogger(__name__)


def grade_program(program_dir: Path, *, bundle: Path) -> tuple[bool, bool]:
    """Grade observation review and trajectory artifacts for one programme.

    Returns
    -------
    tuple[bool, bool]
        ``(observation_graded, trajectory_graded)``.
    """
    from trialagentbench_harness.contracts.core.runs import TrialDevChainSummaryV1

    chain_path = program_dir / "chain_summary.json"
    if not chain_path.is_file():
        raise FileNotFoundError(f"Missing chain_summary.json: {chain_path}")
    chain = read_json_model(TrialDevChainSummaryV1, chain_path)
    src_root = scenario_root(bundle, str(chain.scenario_id))
    program_objective_id = str(chain.objective_id)

    obs_submission = program_dir / "obs_review" / "obs_review_submission.json"
    observation_graded = False
    observational_grade = None
    if obs_submission.is_file():
        observational_grade = trialdev_upstream.grade_item(
            scenario_root=src_root,
            submission_path=obs_submission,
            write_path=program_dir / "obs_review" / "grade_report.json",
        )
        observation_graded = True

    trajectory_graded = False
    trajectory_grade = None
    workdir = program_dir / "agent_workdir"
    has_materialized_phase = workdir.is_dir() and (
        any(attempt.n_materializations > 0 for attempt in chain.phases_attempted)
        or any(count > 0 for count in chain.materialization_usage.materialize_calls_by_phase.values())
    )
    if has_materialized_phase:
        phase_scoring_objectives = {
            "phase1": "benefit_risk",
            "phase2": program_objective_id,
            "phase3": program_objective_id,
        }
        write_json(
            workdir / "scoring_context.json",
            {
                "version": "v1",
                "scenario_id": str(chain.scenario_id),
                "program_id": str(chain.program_id),
                "program_objective_id": program_objective_id,
                "phase_scoring_objectives": phase_scoring_objectives,
            },
        )
        trajectory_grade = trialdev_upstream.grade_trajectory(
            scenario_root=src_root,
            trajectory_root=workdir,
            initial_state_path=program_dir / "states" / "state_after_observational_review.json",
            out_path=program_dir / "trajectory_grade.json",
        )
        trajectory_graded = True

    attempted_phase_ids = {str(attempt.phase_id) for attempt in chain.phases_attempted}
    analysis_quality = summarise_programme_analysis_quality(
        observational_report=observational_grade,
        phase_reports=() if trajectory_grade is None else trajectory_grade.phase_reports,
        attempted_phase_ids=attempted_phase_ids,
    )
    if trajectory_grade is not None:
        chain.trajectory_grade_path = "trajectory_grade.json"
    chain.trajectory_metrics = trajectory_metrics_from_grade(
        trajectory_grade=trajectory_grade,
        observational_report=observational_grade,
        phase_policy_modes=phase_policy_modes_from_manifest(src_root / "public" / "program_loop_manifest.json"),
        analysis_quality=analysis_quality,
    )
    if observation_graded:
        chain.obs_review_grade_path = "obs_review/grade_report.json"
    write_json_model(chain_path, chain)

    return observation_graded, trajectory_graded


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", help="Canonical TrialDevBench run directory.")
    parser.add_argument(
        "--bundle",
        required=True,
        help="Exact evaluator bundle used for grading.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="New directory for graded artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    src_run_root = Path(args.run_root).resolve()
    if not src_run_root.is_dir():
        sys.stderr.write(f"Run root not found: {src_run_root}\n")
        return 2

    bundle_source = Path(args.bundle).resolve()
    if not bundle_source.exists():
        sys.stderr.write(f"Evaluator bundle not found: {bundle_source}\n")
        return 2

    out_dir = Path(args.out_dir).resolve()
    try:
        with staged_directory(out_dir) as run_root:
            shutil.copytree(src_run_root, run_root, dirs_exist_ok=True)

            from trialagentbench_harness import __version__ as harness_version
            from trialagentbench_harness.contracts.core.runs import TrialDevRunConfigV1

            run_config = read_json_model(TrialDevRunConfigV1, run_root / "run_config.json")
            manifest = GradeManifestV1(
                suite="trialdev",
                harness_version=harness_version,
                harness_git_sha=head_sha(Path(__file__).resolve().parent),
                timestamp_utc=run_config.timestamp_utc,
                input_run_dir=str(src_run_root),
                output_run_dir=str(out_dir),
                input_run_sha256=sha256_path(src_run_root),
                evaluator_release_sha256=sha256_path(bundle_source),
                bundle_dir=str(bundle_source),
                notes=["Canonical run artifacts were graded without model or network calls."],
            )
            write_json_model(run_root / "GRADE_MANIFEST.json", manifest)

            programs_root = run_root / "programs"
            if not programs_root.is_dir():
                raise FileNotFoundError(f"No programs/ subdir under {src_run_root}")

            if bundle_source.is_dir() and (bundle_source / "participant_catalogue.json").is_file():
                n_programs = _grade_portfolio_run(
                    run_root=run_root,
                    bundle=bundle_source,
                    run_config=run_config,
                )
                manifest.graded_task_count = n_programs
                write_json_model(run_root / "GRADE_MANIFEST.json", manifest)
                sys.stdout.write(f"\nGraded {n_programs} bounded portfolio programme(s) from immutable submissions.\n")
                return 0

            with materialized_trialdev_release_root(bundle_source) as bundle:
                n_obs = 0
                n_traj = 0
                n_programs = 0
                assessments = []
                for program_dir in sorted(programs_root.iterdir()):
                    if not program_dir.is_dir():
                        continue
                    obs, traj = grade_program(program_dir, bundle=bundle)
                    n_programs += 1
                    n_obs += int(obs)
                    n_traj += int(traj)
                    assessment = build_single_asset_programme_assessment_v1(
                        program_dir=program_dir,
                        run_config=run_config,
                    )
                    write_json_model(program_dir / "programme_assessment.json", assessment)
                    assessments.append(assessment)
                    sys.stdout.write(f"  graded {program_dir.name}: obs={obs} trajectory={traj}\n")

                assessment_portfolio = TrialDevAssessmentPortfolioV1(programmes=tuple(assessments))
                write_json_model(run_root / "trialdev_assessments.json", assessment_portfolio)
                write_json_model(
                    run_root / "trialdev_metrics.json",
                    summarize_trialdev_metrics_v1(assessment_portfolio.programmes),
                )

                aggregate_run(
                    run_root,
                    bundle,
                    run_config=run_config,
                )
            manifest.graded_task_count = n_programs
            write_json_model(run_root / "GRADE_MANIFEST.json", manifest)
    except FileExistsError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    sys.stdout.write(
        f"\nGraded {n_obs} obs_review and {n_traj} trajectory artifact(s). Aggregate refreshed in {out_dir}.\n"
    )
    return 0


__all__ = ["grade_program", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
