"""Boundary tests for the canonical TrialDev scientific assessment."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.trialdev.scientific_grade import (
    TrialDevScientificAssessmentV1,
    TrialDevScientificEnvelopeV1,
)


def _complete_assessment(**updates: object) -> TrialDevScientificAssessmentV1:
    values: dict[str, object] = {
        "execution": "passed",
        "question_estimand": "passed",
        "design": "passed",
        "assumptions": "passed",
        "analysis_classification": "uncertainty_qualified",
        "scientific_agreement": "passed",
        "exact_reproduction": "passed",
        "uncertainty": "passed",
        "action_admissibility": "passed",
        "evidential_support": "passed",
        "sequential_coherence": "passed",
        "resources": "within_budget",
        "scientific_envelope": TrialDevScientificEnvelopeV1(
            envelope_id="declared-margin",
            basis="declared_practical_equivalence_margin",
            absolute_margin=0.05,
            exact_reproduction_tolerance=0.0005,
        ),
        "decision_complete": True,
    }
    values.update(updates)
    return TrialDevScientificAssessmentV1.model_validate(values)


@pytest.mark.parametrize(
    "responsibility",
    [
        "execution",
        "question_estimand",
        "design",
        "assumptions",
        "scientific_agreement",
        "uncertainty",
        "action_admissibility",
        "evidential_support",
        "sequential_coherence",
    ],
)
def test_each_required_scientific_responsibility_is_noncompensatory(
    responsibility: str,
) -> None:
    assessment = _complete_assessment(**{responsibility: "failed", "decision_complete": False})

    assert assessment.decision_complete is False

    with pytest.raises(ValidationError, match="decision_complete"):
        _complete_assessment(**{responsibility: "failed"})


def test_exact_reproduction_is_a_diagnostic_not_a_scientific_requirement() -> None:
    assessment = _complete_assessment(exact_reproduction="failed")

    assert assessment.decision_complete is True


def test_inapplicable_design_can_complete_a_decision() -> None:
    assessment = _complete_assessment(design="not_applicable")

    assert assessment.decision_complete is True


def test_resource_use_is_reported_without_changing_scientific_completion() -> None:
    assessment = _complete_assessment(resources="exceeded")

    assert assessment.decision_complete is True
