"""Contract tests for the observational analysis-specification experiment."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from trialdev.test_trialdev_runtime_surface import _observational_specification

from trialagentbench_harness.contracts.experiments import (
    TrialDevObservationalSpecificationAssignmentV1,
    TrialDevObservationalSpecificationScheduleV1,
)


def _assignment(condition: str) -> TrialDevObservationalSpecificationAssignmentV1:
    return TrialDevObservationalSpecificationAssignmentV1(
        assignment_id=f"A-{condition}",
        pair_id="P-one",
        program_id="s01__benefit_risk",
        scenario_id="s01",
        objective_id="benefit_risk",
        replicate_id="r1",
        decoding_seed=7,
        condition=condition,
        method_catalog_checksum="a" * 64,
        method_specification=_observational_specification(),
    )


def test_schedule_requires_exact_matched_open_and_prespecified_arms() -> None:
    schedule = TrialDevObservationalSpecificationScheduleV1(
        experiment_id="obs-method-selection",
        participant_release_sha256="b" * 64,
        randomization_seed=11,
        assignments=(
            _assignment("open_selection"),
            _assignment("prespecified_execution"),
        ),
    )
    assert schedule.checksum is not None

    with pytest.raises(ValidationError, match="requires both arms"):
        TrialDevObservationalSpecificationScheduleV1(
            experiment_id="obs-method-selection",
            participant_release_sha256="b" * 64,
            randomization_seed=11,
            assignments=(
                _assignment("open_selection"),
                _assignment("open_selection").model_copy(update={"assignment_id": "A-other"}),
            ),
        )
