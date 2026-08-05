"""Participant-safe TrialEval analysis-method dictionary contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.io.checksums import canonical_payload_sha256


class TrialEvalParticipantMethodV1(BaseModel):
    """One complete participant-safe analysis method contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method_id: str = Field(min_length=1)
    estimator_family: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    result_kind: str = Field(min_length=1)
    effect_scale: str = Field(min_length=1)
    design_modifiers: tuple[str, ...] = ()
    uncertainty_method_id: str = Field(min_length=1)
    sensitivity_parameters: tuple[float, ...] = ()
    possible_diagnostic_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Diagnostic evidence that may be required when this method is used. "
            "The dictionary does not disclose item eligibility or expected duties."
        ),
    )
    supported_conclusion_codes: tuple[str, ...] = ()
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_capabilities(self) -> TrialEvalParticipantMethodV1:
        """Require deterministic, duplicate-free method capabilities."""

        for field_name in (
            "design_modifiers",
            "possible_diagnostic_ids",
            "supported_conclusion_codes",
        ):
            values = getattr(self, field_name)
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{field_name} must be sorted and unique")
        if tuple(sorted(set(self.sensitivity_parameters))) != self.sensitivity_parameters:
            raise ValueError("sensitivity_parameters must be sorted and unique")
        if bool(self.sensitivity_parameters) != (self.result_kind == "sensitivity_set"):
            raise ValueError("sensitivity parameters are required exactly for sensitivity-set methods")
        return self


class TrialEvalParticipantMethodDictionaryV1(BaseModel):
    """Participant-facing vocabulary for structured TrialEval submissions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialeval.participant_method_dictionary/v1"] = (
        "trialagentbench.trialeval.participant_method_dictionary/v1"
    )
    purpose: Literal["declare_supported_submission_vocabulary_without_revealing_item_answers"] = (
        "declare_supported_submission_vocabulary_without_revealing_item_answers"
    )
    methods: tuple[TrialEvalParticipantMethodV1, ...] = Field(min_length=1)
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _canonical_dictionary(self) -> TrialEvalParticipantMethodDictionaryV1:
        methods = tuple(sorted(self.methods, key=lambda row: row.method_id))
        identifiers = tuple(row.method_id for row in methods)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("participant method IDs must be unique")
        object.__setattr__(self, "methods", methods)
        digest = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("participant method dictionary checksum does not match its payload")
        object.__setattr__(self, "checksum", digest)
        return self


__all__ = [
    "TrialEvalParticipantMethodDictionaryV1",
    "TrialEvalParticipantMethodV1",
]
