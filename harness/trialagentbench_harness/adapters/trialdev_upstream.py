"""Typed adapter boundary for the TrialDev simulator and grader.

This module is the *only* place in the harness that should import
`trialagentbench_harness.trialdev.share` / `trialagentbench_harness.trialdev.grading` directly.

Rationale
---------
The simulation and grading packages expose JSON artifact boundaries. The
harness validates those artifacts into schema-bearing Pydantic contracts so
downstream analysis remains strict, deterministic, and drift-resistant.
"""

from __future__ import annotations

from pathlib import Path

from trialagentbench_harness.adapters.trialdev_share import TrialDevProgrammeStateV1
from trialagentbench_harness.contracts.trialdev.programme import TRIALDEV_PROGRAMME_STATE_ADAPTER_V1
from trialagentbench_harness.contracts.trialdev.trialdev_grades import (
    TrialDevGradeRecordV1,
    TrialDevTrajectoryGradeV1,
)
from trialagentbench_harness.io.json import read_json, write_json_model
from trialagentbench_harness.trialdev.grade_wrappers import wrap_grade_record, wrap_trajectory_grade
from trialagentbench_harness.trialdev.grading.grade import grade_item_v1
from trialagentbench_harness.trialdev.grading.models import TrialDevelopmentSubmissionV1
from trialagentbench_harness.trialdev.grading.sequential import (
    TrialMaterializationRejectedError as TrialMaterializationRejectedError,
)
from trialagentbench_harness.trialdev.grading.sequential import (
    advance_observational_programme_state_v1,
    advance_program_state_v1,
    build_initial_program_state_v1,
    grade_trajectory_v1,
    materialize_phase_v1,
)


def grade_item(*, scenario_root: Path, submission_path: Path, write_path: Path) -> TrialDevGradeRecordV1:
    """Run `grade_item_v1` and return a schema-bearing grade record."""
    grade_item_v1(scenario_root=scenario_root, submission_path=submission_path, write_path=write_path)
    raw = read_json(write_path)
    if not isinstance(raw, dict):
        raise ValueError(f"grade report must be a JSON object: {write_path}")
    if raw.get("schema_id") == "trialagentbench_trialdev_grade_record_v1":
        return TrialDevGradeRecordV1.model_validate(raw)
    wrapped = wrap_grade_record(raw)
    write_json_model(write_path, wrapped)
    return wrapped


def grade_trajectory(
    *,
    scenario_root: Path,
    trajectory_root: Path,
    initial_state_path: Path,
    out_path: Path,
    scoring_context_path: Path | None = None,
) -> TrialDevTrajectoryGradeV1:
    """Run `grade_trajectory_v1` and return a schema-bearing trajectory grade."""
    grade_trajectory_v1(
        scenario_root=scenario_root,
        trajectory_root=trajectory_root,
        initial_state_path=initial_state_path,
        out_path=out_path,
        scoring_context_path=scoring_context_path,
    )
    raw = read_json(out_path)
    if not isinstance(raw, dict):
        raise ValueError(f"trajectory_grade.json must be a JSON object: {out_path}")
    if raw.get("schema_id") == "trialagentbench_trialdev_trajectory_grade_v1":
        return TrialDevTrajectoryGradeV1.model_validate(raw)
    wrapped = wrap_trajectory_grade(raw)
    write_json_model(out_path, wrapped)
    return wrapped


def build_initial_program_state(*, scenario_root: Path, programme_id: str, objective_id: str, out_path: Path) -> None:
    """Adapter for `build_initial_program_state_v1`."""
    build_initial_program_state_v1(
        scenario_root=scenario_root,
        programme_id=programme_id,
        objective_id=objective_id,
        out_path=out_path,
    )


def advance_observational_programme_state(
    *,
    state: TrialDevProgrammeStateV1,
    submission: TrialDevelopmentSubmissionV1,
    submission_path: Path,
    out_path: Path,
) -> TrialDevProgrammeStateV1:
    """Apply one accepted observational decision to programme state."""

    return advance_observational_programme_state_v1(
        state=state,
        submission=submission,
        submission_path=submission_path,
        out_path=out_path,
    )


def materialize_phase(
    *,
    scenario_root: Path,
    state_path: Path,
    request_path: Path,
    out_dir: Path,
    seed: int,
    overwrite: bool,
) -> None:
    """Adapter for `materialize_phase_v1`."""
    materialize_phase_v1(
        scenario_root=scenario_root,
        state_path=state_path,
        request_path=request_path,
        out_dir=out_dir,
        seed=seed,
        overwrite=overwrite,
    )


def advance_program_state(
    *,
    scenario_root: Path,
    state_path: Path,
    request_path: Path,
    trial_output_root: Path,
    analysis_path: Path,
    decision_path: Path,
    out_path: Path,
) -> None:
    """Adapter for `advance_program_state_v1`."""
    advance_program_state_v1(
        scenario_root=scenario_root,
        state_path=state_path,
        request_path=request_path,
        trial_output_root=trial_output_root,
        analysis_path=analysis_path,
        decision_path=decision_path,
        out_path=out_path,
    )


def load_program_state(path: Path) -> TrialDevProgrammeStateV1:
    """Load one program state through its public contract."""

    return TRIALDEV_PROGRAMME_STATE_ADAPTER_V1.validate_json(path.read_text(encoding="utf-8"))


def validate_and_write_submission(
    payload: dict[str, object],
    *,
    path: Path,
) -> TrialDevelopmentSubmissionV1:
    """Validate and persist one complete TrialDev grader submission."""

    submission = TrialDevelopmentSubmissionV1.model_validate(payload)
    write_json_model(path, submission)
    return submission
