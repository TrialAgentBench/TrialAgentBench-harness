"""Participant-safe TrialEval diagnostic dictionary contract."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.contracts.scoring.diagnostic_registry import (
    DiagnosticKeySpecV1,
    load_diagnostic_registry_v1,
)
from trialagentbench_harness.io.checksums import canonical_payload_sha256

TRIALEVAL_LIMITATION_DIAGNOSTIC_BY_CONCLUSION_CODE_V1: Final[Mapping[str, str]] = MappingProxyType(
    {
        "point_not_identified_due_to_censoring_or_support_failure": "censoring_followup_public",
        "point_not_identified_due_to_endpoint_validation_failure": "endpoint_ascertainment_public",
    }
)


class TrialEvalParticipantDiagnosticDictionaryV1(BaseModel):
    """Task-general operations and measures for TrialEval diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialeval.participant_diagnostic_dictionary/v1"] = (
        "trialagentbench.trialeval.participant_diagnostic_dictionary/v1"
    )
    purpose: Literal["define_diagnostic_operations_without_item_eligibility_or_results"] = (
        "define_diagnostic_operations_without_item_eligibility_or_results"
    )
    diagnostics: dict[str, DiagnosticKeySpecV1] = Field(min_length=1)
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _canonical_dictionary(self) -> TrialEvalParticipantDiagnosticDictionaryV1:
        diagnostics = dict(sorted(self.diagnostics.items()))
        if any(identifier != definition.public_key for identifier, definition in diagnostics.items()):
            raise ValueError("participant diagnostic dictionary keys must match their public diagnostic IDs")
        object.__setattr__(self, "diagnostics", diagnostics)
        digest = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("participant diagnostic dictionary checksum does not match its payload")
        object.__setattr__(self, "checksum", digest)
        return self


def participant_diagnostic_dictionary_v1() -> TrialEvalParticipantDiagnosticDictionaryV1:
    """Project the canonical registry without item-specific eligibility or results."""

    registry = load_diagnostic_registry_v1()
    return TrialEvalParticipantDiagnosticDictionaryV1(
        diagnostics={definition.public_key: definition for definition in registry.diagnostic_keys.values()}
    )


__all__ = [
    "TRIALEVAL_LIMITATION_DIAGNOSTIC_BY_CONCLUSION_CODE_V1",
    "TrialEvalParticipantDiagnosticDictionaryV1",
    "participant_diagnostic_dictionary_v1",
]
