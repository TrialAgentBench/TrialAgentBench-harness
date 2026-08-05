"""Canonical scientific assessment contract for TrialDev decisions."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

TrialDevResponsibilityStatusV1 = Literal[
    "passed",
    "failed",
    "not_applicable",
    "not_assessed",
]
TrialDevAnalysisClassificationV1 = Literal[
    "invalid",
    "descriptive",
    "adjusted",
    "uncertainty_qualified",
]
TrialDevResourceStatusV1 = Literal["within_budget", "exceeded", "not_assessed"]
TrialDevScientificResponsibilityIdV1 = Literal[
    "decision_complete",
    "question_estimand",
    "design",
    "assumptions",
    "scientific_agreement",
    "exact_reproduction",
    "uncertainty",
    "action_admissibility",
    "evidential_support",
    "sequential_coherence",
    "resource_within_budget",
]
TRIALDEV_ANALYSIS_CLASSIFICATIONS_V1: tuple[TrialDevAnalysisClassificationV1, ...] = (
    "invalid",
    "descriptive",
    "adjusted",
    "uncertainty_qualified",
)
TRIALDEV_SCIENTIFIC_RESPONSIBILITIES_V1: tuple[TrialDevScientificResponsibilityIdV1, ...] = (
    "decision_complete",
    "question_estimand",
    "design",
    "assumptions",
    "scientific_agreement",
    "exact_reproduction",
    "uncertainty",
    "action_admissibility",
    "evidential_support",
    "sequential_coherence",
    "resource_within_budget",
)


class TrialDevScientificEnvelopeV1(BaseModel):
    """Frozen numeric boundary used for scientific-equivalence assessment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    envelope_id: str = Field(..., min_length=1)
    basis: Literal[
        "declared_practical_equivalence_margin",
        "declared_decision_thresholds",
        "qualified_nonidentification",
    ]
    absolute_margin: float | None = Field(default=None, ge=0.0)
    decision_thresholds: tuple[float, ...] = ()
    exact_reproduction_tolerance: float = Field(..., gt=0.0)

    @model_validator(mode="after")
    def validate_boundary(self) -> Self:
        """Require the numeric boundary implied by the declared basis."""

        if self.basis == "declared_practical_equivalence_margin":
            if self.absolute_margin is None or self.decision_thresholds:
                raise ValueError("A practical-equivalence envelope requires only an absolute margin.")
        elif self.basis == "declared_decision_thresholds":
            if not self.decision_thresholds or self.absolute_margin is not None:
                raise ValueError("A decision-threshold envelope requires only decision thresholds.")
        elif self.absolute_margin is not None or self.decision_thresholds:
            raise ValueError("Qualified nonidentification has no numeric equivalence boundary.")
        if len(set(self.decision_thresholds)) != len(self.decision_thresholds):
            raise ValueError("Scientific decision thresholds must be unique.")
        return self


class TrialDevScientificAssessmentV1(BaseModel):
    """Orthogonal scientific responsibilities for one submitted decision.

    Exact reproduction is a reproducibility diagnostic. It is deliberately not
    part of ``decision_complete``; a scientifically equivalent estimate may be
    defensible without reproducing the reference report byte-for-byte or to its
    reporting precision.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialdev_scientific_assessment/v1"] = (
        "trialagentbench.trialdev_scientific_assessment/v1"
    )
    execution: TrialDevResponsibilityStatusV1
    question_estimand: TrialDevResponsibilityStatusV1
    design: TrialDevResponsibilityStatusV1
    assumptions: TrialDevResponsibilityStatusV1
    analysis_classification: TrialDevAnalysisClassificationV1
    scientific_agreement: TrialDevResponsibilityStatusV1
    exact_reproduction: TrialDevResponsibilityStatusV1
    uncertainty: TrialDevResponsibilityStatusV1
    action_admissibility: TrialDevResponsibilityStatusV1
    evidential_support: TrialDevResponsibilityStatusV1
    sequential_coherence: TrialDevResponsibilityStatusV1
    resources: TrialDevResourceStatusV1 = "not_assessed"
    scientific_envelope: TrialDevScientificEnvelopeV1 | None = None
    failure_reasons: tuple[str, ...] = ()
    decision_complete: bool

    @model_validator(mode="after")
    def validate_assessment(self) -> Self:
        """Bind completion to noncompensatory scientific responsibilities."""

        required = (
            self.execution,
            self.question_estimand,
            self.design,
            self.assumptions,
            self.scientific_agreement,
            self.uncertainty,
            self.action_admissibility,
            self.evidential_support,
            self.sequential_coherence,
        )
        complete = (
            self.analysis_classification == "uncertainty_qualified"
            and all(status in {"passed", "not_applicable"} for status in required)
            and all(status != "not_assessed" for status in required)
        )
        if self.decision_complete != complete:
            raise ValueError("decision_complete must equal the noncompensatory scientific responsibilities.")
        if self.analysis_classification == "invalid" and self.decision_complete:
            raise ValueError("An invalid analysis cannot complete a TrialDev decision.")
        if self.scientific_agreement != "not_assessed" and self.scientific_envelope is None:
            raise ValueError("Assessed scientific agreement requires its frozen envelope.")
        if len(set(self.failure_reasons)) != len(self.failure_reasons):
            raise ValueError("Scientific assessment failure reasons must be unique.")
        if any(not reason.strip() for reason in self.failure_reasons):
            raise ValueError("Scientific assessment failure reasons must not be empty.")
        return self


def unassessed_scientific_assessment_v1() -> TrialDevScientificAssessmentV1:
    """Return the explicit state for a report that has not been scientifically assessed."""

    return TrialDevScientificAssessmentV1(
        execution="not_assessed",
        question_estimand="not_assessed",
        design="not_assessed",
        assumptions="not_assessed",
        analysis_classification="invalid",
        scientific_agreement="not_assessed",
        exact_reproduction="not_assessed",
        uncertainty="not_assessed",
        action_admissibility="not_assessed",
        evidential_support="not_assessed",
        sequential_coherence="not_assessed",
        decision_complete=False,
    )


__all__ = [
    "TRIALDEV_ANALYSIS_CLASSIFICATIONS_V1",
    "TRIALDEV_SCIENTIFIC_RESPONSIBILITIES_V1",
    "TrialDevAnalysisClassificationV1",
    "TrialDevResourceStatusV1",
    "TrialDevResponsibilityStatusV1",
    "TrialDevScientificAssessmentV1",
    "TrialDevScientificEnvelopeV1",
    "TrialDevScientificResponsibilityIdV1",
    "unassessed_scientific_assessment_v1",
]
