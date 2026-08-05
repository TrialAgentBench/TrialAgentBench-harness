"""Orthogonal procedure-assistance conditions used by benchmark experiments."""

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from trialagentbench_harness.contracts.core.config import ProcedureAssistanceV1
from trialagentbench_harness.contracts.core.trialeval_factors import TrialEvalAnalysisSpecificationV1

TrialEvalPromptConditionV1: TypeAlias = Literal[  # noqa: UP040
    "neutral",
    "targeted_covariate_structure",
    "targeted_survival_assumptions",
    "targeted_design_structure",
    "targeted_data_integrity",
    "placebo_deliberation",
]
TrialEvalSubmissionInterfaceV1: TypeAlias = Literal["structured", "narrative"]  # noqa: UP040

_TRIALEVAL_PROCEDURE_ASSISTANCE_LEVELS: tuple[ProcedureAssistanceV1, ...] = (
    "output_contract_only",
    "unordered_checklist",
    "ordered_sop",
)
_TRIALDEV_PROCEDURE_ASSISTANCE_LEVELS: tuple[ProcedureAssistanceV1, ...] = (
    "output_contract_only",
    "unordered_checklist",
    "ordered_sop",
)


class ProcedureAssistanceExposureV1(BaseModel):
    """Participant-visible components of one procedure-assistance intervention."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    suite: Literal["trialeval", "trialdev"]
    procedure_assistance: ProcedureAssistanceV1
    unordered_completeness_instruction: bool
    ordered_analysis_procedure: bool


def procedure_assistance_exposure_v1(
    *,
    suite: Literal["trialeval", "trialdev"],
    procedure_assistance: ProcedureAssistanceV1,
) -> ProcedureAssistanceExposureV1:
    """Return the exact procedure-assistance components exposed to a participant."""

    if suite == "trialeval" and procedure_assistance not in _TRIALEVAL_PROCEDURE_ASSISTANCE_LEVELS:
        raise ValueError(f"TrialEval does not define procedure assistance {procedure_assistance!r}.")
    return ProcedureAssistanceExposureV1(
        suite=suite,
        procedure_assistance=procedure_assistance,
        unordered_completeness_instruction=procedure_assistance in {"unordered_checklist", "ordered_sop"},
        ordered_analysis_procedure=procedure_assistance == "ordered_sop",
    )


def trialeval_procedure_assistance_levels_v1() -> tuple[ProcedureAssistanceV1, ...]:
    """Return the immutable TrialEval assistance levels."""

    return _TRIALEVAL_PROCEDURE_ASSISTANCE_LEVELS


def trialdev_procedure_assistance_levels_v1() -> tuple[ProcedureAssistanceV1, ...]:
    """Return the immutable TrialDev assistance levels."""

    return _TRIALDEV_PROCEDURE_ASSISTANCE_LEVELS


__all__ = [
    "ProcedureAssistanceExposureV1",
    "ProcedureAssistanceV1",
    "TrialEvalAnalysisSpecificationV1",
    "TrialEvalPromptConditionV1",
    "TrialEvalSubmissionInterfaceV1",
    "procedure_assistance_exposure_v1",
    "trialeval_procedure_assistance_levels_v1",
    "trialdev_procedure_assistance_levels_v1",
]
