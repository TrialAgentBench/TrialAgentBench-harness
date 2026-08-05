"""Deterministic grader for frozen TrialDev benchmark bundles."""

from trialagentbench_harness.trialdev.grading.grade import grade_bundle_v1, grade_item_v1
from trialagentbench_harness.trialdev.grading.models import (
    TrialDevelopmentAnalysisQualityV1,
    TrialDevelopmentAnalysisReportV1,
    TrialDevelopmentGradeReportV1,
    TrialDevelopmentProgramDecisionV1,
    TrialDevelopmentRequestV1,
    TrialDevelopmentSubmissionV1,
)
from trialagentbench_harness.trialdev.grading.validate import validate_release_v1, validate_submission_v1

_SEQUENTIAL_EXPORTS = {
    "TrialMaterializationRejectedError",
    "advance_program_state_v1",
    "build_initial_program_state_v1",
    "grade_trajectory_v1",
    "materialize_phase_v1",
    "validate_program_state_file_v1",
}


def __getattr__(name: str) -> object:
    """Load sequential helpers only when explicitly requested."""
    if name not in _SEQUENTIAL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from trialagentbench_harness.trialdev.grading import sequential

    return getattr(sequential, name)


__all__ = [
    "TrialDevelopmentAnalysisQualityV1",
    "TrialDevelopmentAnalysisReportV1",
    "TrialDevelopmentGradeReportV1",
    "TrialDevelopmentProgramDecisionV1",
    "TrialDevelopmentRequestV1",
    "TrialDevelopmentSubmissionV1",
    "TrialMaterializationRejectedError",
    "advance_program_state_v1",
    "build_initial_program_state_v1",
    "grade_bundle_v1",
    "grade_item_v1",
    "grade_trajectory_v1",
    "materialize_phase_v1",
    "validate_program_state_file_v1",
    "validate_release_v1",
    "validate_submission_v1",
]
