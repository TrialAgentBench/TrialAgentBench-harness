"""Common configuration contracts used by harness artifacts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.io.checksums import canonical_payload_sha256

ProcedureAssistanceV1: TypeAlias = Literal[
    "output_contract_only",
    "unordered_checklist",
    "ordered_sop",
]  # noqa: UP040
ToolChoiceV1: TypeAlias = Literal["auto", "required"]  # noqa: UP040

ReasoningEffortV1: TypeAlias = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]  # noqa: UP040


class DecodingConfigV1(BaseModel):
    """Explicit decoding configuration recorded for reproducibility."""

    model_config = ConfigDict(extra="forbid")
    schema_id: Literal["trialagentbench_decoding_config_v1"] = "trialagentbench_decoding_config_v1"
    schema_version: Literal[1] = 1
    temperature: float
    max_tokens: int
    send_temperature: bool
    decoding_seed: int | None = Field(default=None, strict=True, ge=0)
    # Keep the contract small and stable. If/when we add additional decoding
    # knobs (e.g., top_p), they should be added as explicit optional fields
    # rather than a free-form dict.


class RoutingConfigV1(BaseModel):
    """Routing pins and request boundary for the selected transport."""

    model_config = ConfigDict(extra="forbid")
    schema_id: Literal["trialagentbench_routing_config_v1"] = "trialagentbench_routing_config_v1"
    schema_version: Literal[1] = 1
    provider: Literal["openai", "openai_responses", "openrouter"]
    openrouter_provider: str | None = None
    request_timeout_seconds: float = Field(gt=0.0, le=900.0)

    @model_validator(mode="after")
    def validate_exact_route(self) -> Self:
        """Require one reproducible route for the selected transport."""

        if self.provider == "openrouter" and not self.openrouter_provider:
            raise ValueError("OpenRouter runs require an explicit upstream provider pin.")
        if self.provider != "openrouter" and self.openrouter_provider is not None:
            raise ValueError("openrouter_provider is valid only for the OpenRouter transport.")
        return self


class ProviderReasoningCapabilityV1(BaseModel):
    """Source-bound reasoning controls declared for one exact model route."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.provider_reasoning_capability/v1"] = (
        "trialagentbench.provider_reasoning_capability/v1"
    )
    provider_transport: Literal["openai", "openai_responses", "openrouter"]
    model_id: str = Field(..., min_length=1)
    upstream_provider: str | None = Field(default=None, min_length=1)
    supported_efforts: tuple[ReasoningEffortV1, ...] = Field(..., min_length=1)
    source_url: str = Field(..., min_length=1)
    source_retrieved_utc: datetime
    source_payload_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_capability(self) -> Self:
        """Require a unique effort set, exact route, and canonical checksum."""

        if len(self.supported_efforts) != len(set(self.supported_efforts)):
            raise ValueError("Provider reasoning efforts must be unique.")
        if (self.provider_transport == "openrouter") != (self.upstream_provider is not None):
            raise ValueError("An upstream provider is required exactly for OpenRouter capability records.")
        digest = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("Provider reasoning capability checksum mismatch.")
        object.__setattr__(self, "checksum", digest)
        return self


class ReasoningConfigV1(BaseModel):
    """Requested reasoning configuration and its exact capability evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    effort: ReasoningEffortV1 | None = None
    exclude_from_response: Literal[True] = True
    capability: ProviderReasoningCapabilityV1 | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        """Require capability evidence exactly when an effort is requested."""

        if (self.effort is None) != (self.capability is None):
            raise ValueError("A requested reasoning effort requires one exact capability record.")
        capability = self.capability
        if self.effort is not None and capability is not None and self.effort not in capability.supported_efforts:
            raise ValueError(f"Reasoning effort {self.effort!r} is not supported by the exact model route.")
        return self


class ExperimentConditionV1(BaseModel):
    """Provider-neutral identity for one evaluated request condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.experiment_condition/v1"] = "trialagentbench.experiment_condition/v1"
    condition_id: str = Field(..., min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    request_replicate_id: str = Field(..., min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    reasoning: ReasoningConfigV1 = Field(default_factory=ReasoningConfigV1)
    procedure_assistance: ProcedureAssistanceV1 = "output_contract_only"
    maximum_turns_per_step: int = Field(..., ge=1)
    maximum_submission_attempts: int | None = Field(default=None, ge=1)
    tool_choice: ToolChoiceV1 = "auto"


class TrialDevExecutionRequestV1(BaseModel):
    """Portable input configuration for one TrialDev run or continuation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialdev_execution_request/v1"] = (
        "trialagentbench.trialdev_execution_request/v1"
    )
    bundle: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    provider: Literal["openai", "openai_responses", "openrouter"]
    dotenv: bool = False
    openrouter_provider: str | None = Field(default=None, min_length=1)
    condition_id: str | None = Field(default=None, min_length=1)
    request_replicate_id: str | None = Field(default=None, min_length=1)
    reasoning_effort: ReasoningEffortV1 | None = None
    reasoning_capability_snapshot: str | None = Field(default=None, min_length=1)
    programs: tuple[str, ...] | None = None
    master_seed: int | None = None
    seed_variants: int | None = Field(default=None, ge=1)
    max_phase_retries: int | None = Field(default=None, ge=1)
    max_submission_attempts: int | None = Field(default=None, ge=1)
    program_watchdog_seconds: int | None = Field(default=None, ge=1)
    workers: int | None = Field(default=None, ge=1)
    reported_cost_stop_usd: float | None = Field(default=None, gt=0.0)
    max_turns_per_step: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_context_characters: int | None = Field(default=None, ge=1)
    procedure_assistance: ProcedureAssistanceV1 | None = None
    tool_choice: ToolChoiceV1 | None = None
    output_root: str | None = Field(default=None, min_length=1)
    label: str | None = Field(default=None, min_length=1)
    append_run_dir: str | None = Field(default=None, min_length=1)
    request_timeout_seconds: float | None = Field(default=None, gt=0.0, le=900.0)
    omit_temperature: bool = False
    decoding_seed: int | None = Field(default=None, strict=True, ge=0)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        """Reject ambiguous routes, controls, and duplicate task selectors."""

        if (self.provider == "openrouter") != (self.openrouter_provider is not None):
            raise ValueError("openrouter_provider is required exactly for OpenRouter execution.")
        if (self.reasoning_effort is None) != (self.reasoning_capability_snapshot is None):
            raise ValueError("reasoning_effort and reasoning_capability_snapshot must be supplied together.")
        if self.provider == "openai_responses" and self.decoding_seed is not None:
            raise ValueError("decoding_seed is not supported by the OpenAI Responses transport.")
        if self.programs is not None and len(self.programs) != len(set(self.programs)):
            raise ValueError("TrialDev programme selectors must be unique.")
        return self


__all__ = [
    "DecodingConfigV1",
    "ExperimentConditionV1",
    "ProcedureAssistanceV1",
    "ProviderReasoningCapabilityV1",
    "ReasoningConfigV1",
    "ReasoningEffortV1",
    "RoutingConfigV1",
    "ToolChoiceV1",
    "TrialDevExecutionRequestV1",
]
