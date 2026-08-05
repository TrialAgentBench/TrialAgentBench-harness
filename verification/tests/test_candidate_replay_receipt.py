"""Tests for clean-wheel replay evidence contracts."""

import pytest
from pydantic import ValidationError

from trialagentbench_validation.candidate_clean_replay import (
    CandidateCleanWheelReplayV1,
    CandidateReplayComparisonV1,
    CleanWheelIdentityV1,
)


def _comparison(family: str) -> CandidateReplayComparisonV1:
    return CandidateReplayComparisonV1(
        evidence_family=family,
        compared_numeric_value_count=1,
        maximum_absolute_difference=0.0,
        structural_difference_count=0,
        nonnumeric_difference_count=0,
        status="pass",
    )


def _payload() -> dict[str, object]:
    return {
        "release_id": "candidate-1",
        "source_commit": "a" * 40,
        "python_implementation": "CPython",
        "python_version": "3.12.0",
        "installed_environment_sha256": "b" * 64,
        "installation_constraints_sha256": "c" * 64,
        "transcript_sha256": "d" * 64,
        "absolute_tolerance": 0.0001,
        "isolated_install": True,
        "network_blocked_during_replay": True,
        "repository_on_import_path": False,
        "prohibited_import_count": 0,
        "wheels": (
            CleanWheelIdentityV1(
                package="trial-agent-bench",
                filename="trial_agent_bench.whl",
                sha256="e" * 64,
            ),
            CleanWheelIdentityV1(
                package="trialagentbench-validation",
                filename="trialagentbench_validation.whl",
                sha256="f" * 64,
            ),
        ),
        "comparisons": tuple(
            _comparison(family)
            for family in (
                "grader_behavior",
                "grader_concordance",
                "role_boundary",
                "trialdev_recovery",
                "trialeval_c5_integrity",
                "trialeval_recovery",
            )
        ),
        "status": "pass",
    }


def test_clean_replay_binds_isolated_installation_constraints() -> None:
    receipt = CandidateCleanWheelReplayV1.model_validate(_payload())

    assert receipt.isolated_install
    assert receipt.installation_constraints_sha256 == "c" * 64


def test_clean_replay_rejects_unisolated_installation() -> None:
    payload = _payload()
    payload["isolated_install"] = False

    with pytest.raises(ValidationError, match="isolated_install"):
        CandidateCleanWheelReplayV1.model_validate(payload)


def test_clean_replay_names_out_of_tolerance_evidence_family() -> None:
    payload = _payload()
    comparisons = list(payload["comparisons"])
    comparisons[-1] = comparisons[-1].model_copy(
        update={"maximum_absolute_difference": 0.001}
    )
    payload["comparisons"] = comparisons

    with pytest.raises(ValidationError, match="trialeval_recovery"):
        CandidateCleanWheelReplayV1.model_validate(payload)
