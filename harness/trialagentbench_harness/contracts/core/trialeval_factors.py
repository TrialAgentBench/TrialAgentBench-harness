"""Canonical TrialEval evidence-factor contracts."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, model_validator

TrialEvalContextConfigurationV1: TypeAlias = Literal["C1", "C2", "C3", "C4", "C5"]  # noqa: UP040
TrialEvalDataPreparationV1: TypeAlias = Literal[  # noqa: UP040
    "analysis_ready",
    "raw_domains",
    "raw_domains_declared_defect",
]
TrialEvalAnalysisSpecificationV1: TypeAlias = Literal["protocol_only", "locked_sap"]  # noqa: UP040

_CONTEXT_FACTORS: dict[
    TrialEvalContextConfigurationV1,
    tuple[TrialEvalDataPreparationV1, TrialEvalAnalysisSpecificationV1],
] = {
    "C1": ("analysis_ready", "locked_sap"),
    "C2": ("analysis_ready", "protocol_only"),
    "C3": ("raw_domains", "locked_sap"),
    "C4": ("raw_domains", "protocol_only"),
    "C5": ("raw_domains_declared_defect", "protocol_only"),
}


class TrialEvalEvidenceFactorsV1(BaseModel):
    """Participant-safe evidence factors for one immutable task surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_configuration: TrialEvalContextConfigurationV1
    data_preparation: TrialEvalDataPreparationV1
    analysis_specification: TrialEvalAnalysisSpecificationV1

    @model_validator(mode="after")
    def _validate_configuration(self) -> TrialEvalEvidenceFactorsV1:
        expected = _CONTEXT_FACTORS[self.context_configuration]
        observed = (self.data_preparation, self.analysis_specification)
        if observed != expected:
            raise ValueError(
                f"{self.context_configuration} requires data_preparation={expected[0]!r} and "
                f"analysis_specification={expected[1]!r}; observed {observed!r}."
            )
        return self


__all__ = [
    "TrialEvalAnalysisSpecificationV1",
    "TrialEvalContextConfigurationV1",
    "TrialEvalDataPreparationV1",
    "TrialEvalEvidenceFactorsV1",
]
