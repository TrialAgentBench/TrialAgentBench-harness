"""Independent verification and scoring for TrialAgentBench releases."""

from trialagentbench_verifier.contracts import (
    CheckResultV1,
    ExecutionProofV1,
    NormalizedSubmissionV1,
    PracticeVerificationReportV1,
    VerificationReportV1,
)
from trialagentbench_verifier.execution import ExecutionSubmissionV1, execute_submission, execute_submissions
from trialagentbench_verifier.practice import verify_practice_suite
from trialagentbench_verifier.release import (
    ScoreSubmissionV1,
    score_submission,
    score_submissions,
    verify_release,
)

__all__ = [
    "CheckResultV1",
    "ExecutionProofV1",
    "PracticeVerificationReportV1",
    "NormalizedSubmissionV1",
    "VerificationReportV1",
    "ExecutionSubmissionV1",
    "ScoreSubmissionV1",
    "execute_submissions",
    "score_submissions",
    "execute_submission",
    "verify_practice_suite",
    "score_submission",
    "verify_release",
]
