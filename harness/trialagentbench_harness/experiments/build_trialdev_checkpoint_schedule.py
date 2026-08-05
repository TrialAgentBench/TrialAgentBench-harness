"""Compile matched TrialDev checkpoint schedules from verified source custody."""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_harness.adapters import trialdev_upstream
from trialagentbench_harness.contracts.experiments import (
    TrialDevCanonicalCheckpointSourcesV1,
    TrialDevCanonicalCheckpointSourceV1,
    TrialDevCheckpointAssignmentV1,
    TrialDevCheckpointBlockPlanV1,
    TrialDevCheckpointSchedulePlanV1,
    TrialDevCheckpointScheduleV1,
    TrialDevEndogenousCheckpointSourcesV1,
    TrialDevEndogenousCheckpointSourceV1,
)
from trialagentbench_harness.contracts.trialdev.run_checkpoint import TrialDevRunCheckpointV1
from trialagentbench_harness.experiments.trialdev_checkpoint_replay import (
    _reject_symlinks,
    _safe_source_path,
    _validate_source_assignment,
)
from trialagentbench_harness.io import (
    read_json_model,
    sha256_dir_digest,
    sha256_file,
    write_json,
    write_json_model,
)
from trialagentbench_harness.trialdev.data import scenario_root


def _latest_checkpoint(program_dir: Path) -> tuple[Path, TrialDevRunCheckpointV1]:
    """Load the latest contiguous checkpoint from one partial programme."""

    if not program_dir.is_dir():
        raise NotADirectoryError(program_dir)
    _reject_symlinks(program_dir)
    if (program_dir / "chain_summary.json").exists():
        raise ValueError("Checkpoint source must be partial rather than a completed programme.")
    paths = sorted((program_dir / "checkpoints").glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"Checkpoint source has no append-only checkpoints: {program_dir}")
    checkpoint = read_json_model(TrialDevRunCheckpointV1, paths[-1])
    return paths[-1].resolve(), checkpoint


def _assignment(
    *,
    block: TrialDevCheckpointBlockPlanV1,
    condition: str,
    checkpoint_root: Path,
    program_relative_path: str,
) -> TrialDevCheckpointAssignmentV1:
    """Resolve one source programme into a checksum-bound assignment."""

    program_dir = _safe_source_path(checkpoint_root, program_relative_path)
    checkpoint_path, checkpoint = _latest_checkpoint(program_dir)
    relative_checkpoint = checkpoint_path.relative_to(checkpoint_root.resolve()).as_posix()
    assignment = TrialDevCheckpointAssignmentV1(
        assignment_id=f"{block.block_id}--{condition}",
        block_id=block.block_id,
        program_id=block.program_id,
        scenario_id=block.scenario_id,
        objective_id=block.objective_id,
        replicate_id=block.replicate_id,
        decoding_seed=block.decoding_seed,
        condition=condition,
        source_program_relative_path=program_relative_path,
        source_checkpoint_relative_path=relative_checkpoint,
        source_checkpoint_sha256=sha256_file(checkpoint_path),
        source_run_identity_sha256=checkpoint.payload.run_identity_sha256,
        checkpoint_phase_id=block.checkpoint_phase_id,
        checkpoint_step_id=block.checkpoint_step_id,
        canonical_reference_id=(block.canonical_reference_id if condition == "canonical_state" else None),
    )
    _validate_source_assignment(assignment, checkpoint_root=checkpoint_root)
    return assignment


def _reject_evaluator_artifacts(program_dir: Path) -> None:
    """Reject scorer outputs or private release surfaces in participant custody."""

    forbidden_parts = {"grader", "hidden"}
    forbidden_names = {"grade_report.json", "trajectory_grade.json"}
    for path in program_dir.rglob("*"):
        relative = path.relative_to(program_dir)
        if forbidden_parts & set(relative.parts) or path.name in forbidden_names:
            raise ValueError(f"Canonical participant custody contains evaluator-only material: {relative}")


def _validate_canonical_prefix(
    *,
    assignment: TrialDevCheckpointAssignmentV1,
    checkpoint_root: Path,
    evaluator_root: Path,
) -> None:
    """Require a violation-free canonical prefix with perfect public-evidence grades."""

    program_dir, checkpoint = _validate_source_assignment(
        assignment,
        checkpoint_root=checkpoint_root,
    )
    _reject_evaluator_artifacts(program_dir)
    continuation = checkpoint.payload.continuation.payload
    if continuation.violations:
        raise ValueError("Canonical checkpoint source contains runtime or contract violations.")
    if any(not summary.advance for summary in continuation.completed_phase_summaries):
        raise ValueError("Canonical checkpoint source contains a non-advancing completed phase.")

    source_scenario = scenario_root(evaluator_root, assignment.scenario_id)
    obs_submission = program_dir / "obs_review" / "obs_review_submission.json"
    if not obs_submission.is_file():
        raise FileNotFoundError("Canonical checkpoint source lacks its observational-review submission.")
    with tempfile.TemporaryDirectory(prefix="trialdev_canonical_prefix_") as tmp:
        temporary = Path(tmp)
        obs_grade = trialdev_upstream.grade_item(
            scenario_root=source_scenario,
            submission_path=obs_submission,
            write_path=temporary / "obs_grade.json",
        )
        if obs_grade.primary_score != 1.0 or not obs_grade.analysis_quality.phase_evaluation_valid:
            raise ValueError("Canonical observational-review prefix is not fully valid.")

        completed = {phase.phase_id for phase in checkpoint.payload.completed_phases}
        if completed:
            context_path = temporary / "scoring_context.json"
            write_json(
                context_path,
                {
                    "version": "v1",
                    "scenario_id": assignment.scenario_id,
                    "program_id": assignment.program_id,
                    "program_objective_id": assignment.objective_id,
                    "phase_scoring_objectives": {
                        "phase1": "benefit_risk",
                        "phase2": assignment.objective_id,
                        "phase3": assignment.objective_id,
                    },
                },
            )
            trajectory = trialdev_upstream.grade_trajectory(
                scenario_root=source_scenario,
                trajectory_root=program_dir / "agent_workdir",
                initial_state_path=program_dir / "states" / "state_after_observational_review.json",
                out_path=temporary / "trajectory_grade.json",
                scoring_context_path=context_path,
            )
            reports = {
                str(report.phase_id): report for report in trajectory.phase_reports if report.phase_id in completed
            }
            if set(reports) != completed:
                raise ValueError("Canonical checkpoint source lacks a grade for every completed phase.")
            for phase_id, report in reports.items():
                if report.primary_score != 1.0 or not report.analysis_quality.phase_evaluation_valid:
                    raise ValueError(f"Canonical prefix phase {phase_id!r} is not fully valid.")
                if trajectory.decision_regret_by_phase.get(phase_id) != 0.0:
                    raise ValueError(f"Canonical prefix phase {phase_id!r} has decision regret.")


def _canonical_receipt_records(
    *,
    participant_root: Path,
    checkpoint_root: Path,
) -> dict[str, TrialDevCanonicalCheckpointSourceV1]:
    """Load canonical source custody and bind it to the participant release."""

    receipt = read_json_model(
        TrialDevCanonicalCheckpointSourcesV1,
        checkpoint_root / "canonical_checkpoint_sources.json",
    )
    if receipt.participant_release_sha256 != sha256_dir_digest(participant_root):
        raise ValueError("Canonical checkpoint receipt targets a different participant release.")
    return {row.canonical_reference_id: row for row in receipt.records}


def _endogenous_receipt_records(
    *,
    participant_root: Path,
    checkpoint_root: Path,
) -> dict[str, TrialDevEndogenousCheckpointSourceV1]:
    """Load model-produced source custody and bind it to the participant release."""

    receipt = read_json_model(
        TrialDevEndogenousCheckpointSourcesV1,
        checkpoint_root / "endogenous_checkpoint_sources.json",
    )
    if receipt.participant_release_sha256 != sha256_dir_digest(participant_root):
        raise ValueError("Endogenous checkpoint receipt targets a different participant release.")
    return {row.program_relative_path: row for row in receipt.records}


def _validate_endogenous_receipt_assignment(
    *,
    block: TrialDevCheckpointBlockPlanV1,
    assignment: TrialDevCheckpointAssignmentV1,
    records: dict[str, TrialDevEndogenousCheckpointSourceV1],
) -> None:
    """Require an exact model-produced receipt for one endogenous assignment."""

    record = records.get(block.endogenous_program_relative_path)
    if record is None:
        raise ValueError("Endogenous checkpoint assignment is absent from the custody receipt.")
    expected = (
        block.program_id,
        block.scenario_id,
        block.objective_id,
        block.replicate_id,
        block.decoding_seed,
        block.checkpoint_phase_id,
        block.checkpoint_step_id,
        block.endogenous_program_relative_path,
        assignment.source_checkpoint_relative_path,
        assignment.source_checkpoint_sha256,
        assignment.source_run_identity_sha256,
    )
    observed = (
        record.program_id,
        record.scenario_id,
        record.objective_id,
        record.replicate_id,
        record.decoding_seed,
        record.phase_id,
        record.step_id,
        record.program_relative_path,
        record.checkpoint_relative_path,
        record.checkpoint_sha256,
        record.run_identity_sha256,
    )
    if observed != expected:
        raise ValueError("Endogenous checkpoint assignment does not match its custody receipt.")


def _validate_canonical_receipt_assignment(
    *,
    block: TrialDevCheckpointBlockPlanV1,
    assignment: TrialDevCheckpointAssignmentV1,
    records: dict[str, TrialDevCanonicalCheckpointSourceV1],
) -> None:
    """Require an exact receipt record for one canonical assignment."""

    record = records.get(block.canonical_reference_id)
    if record is None:
        raise ValueError("Canonical checkpoint assignment is absent from the custody receipt.")
    expected = (
        block.program_id,
        block.checkpoint_phase_id,
        block.checkpoint_step_id,
        block.canonical_program_relative_path,
        assignment.source_checkpoint_relative_path,
        assignment.source_checkpoint_sha256,
    )
    observed = (
        record.program_id,
        record.phase_id,
        record.step_id,
        record.program_relative_path,
        record.checkpoint_relative_path,
        record.checkpoint_sha256,
    )
    if observed != expected:
        raise ValueError("Canonical checkpoint assignment does not match its custody receipt.")


def compile_trialdev_checkpoint_schedule_v1(
    *,
    participant_root: Path,
    evaluator_root: Path,
    checkpoint_root: Path,
    plan: TrialDevCheckpointSchedulePlanV1,
) -> TrialDevCheckpointScheduleV1:
    """Compile one prospective plan into an immutable, verified triad schedule."""

    participant = Path(participant_root).resolve()
    evaluator = Path(evaluator_root).resolve()
    sources = Path(checkpoint_root).resolve()
    if not participant.is_dir() or not evaluator.is_dir() or not sources.is_dir():
        raise FileNotFoundError("Participant, evaluator, and checkpoint roots must be directories.")
    canonical_records = _canonical_receipt_records(
        participant_root=participant,
        checkpoint_root=sources,
    )
    endogenous_records = _endogenous_receipt_records(
        participant_root=participant,
        checkpoint_root=sources,
    )
    assignments: list[TrialDevCheckpointAssignmentV1] = []
    for block in plan.blocks:
        endogenous = _assignment(
            block=block,
            condition="endogenous",
            checkpoint_root=sources,
            program_relative_path=block.endogenous_program_relative_path,
        )
        _validate_endogenous_receipt_assignment(
            block=block,
            assignment=endogenous,
            records=endogenous_records,
        )
        context_reset = endogenous.model_copy(
            update={
                "assignment_id": f"{block.block_id}--context_reset",
                "condition": "context_reset",
            }
        )
        canonical = _assignment(
            block=block,
            condition="canonical_state",
            checkpoint_root=sources,
            program_relative_path=block.canonical_program_relative_path,
        )
        _validate_canonical_receipt_assignment(
            block=block,
            assignment=canonical,
            records=canonical_records,
        )
        _validate_canonical_prefix(
            assignment=canonical,
            checkpoint_root=sources,
            evaluator_root=evaluator,
        )
        assignments.extend((endogenous, context_reset, canonical))
    return TrialDevCheckpointScheduleV1(
        experiment_id=plan.experiment_id,
        participant_release_sha256=sha256_dir_digest(participant),
        checkpoint_source_sha256=sha256_dir_digest(sources),
        assignments=tuple(assignments),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Compile and write a checkpoint schedule."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-dir", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-source-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    plan = read_json_model(TrialDevCheckpointSchedulePlanV1, args.plan)
    schedule = compile_trialdev_checkpoint_schedule_v1(
        participant_root=args.participant_dir,
        evaluator_root=args.evaluator_dir,
        checkpoint_root=args.checkpoint_source_dir,
        plan=plan,
    )
    write_json_model(args.output, schedule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
