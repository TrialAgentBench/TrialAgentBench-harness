"""Resolve one provider-bound experimental condition before execution."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from trialagentbench_harness.contracts.core.config import (
    ExperimentConditionV1,
    ProcedureAssistanceV1,
    ProviderReasoningCapabilityV1,
    ReasoningConfigV1,
    ReasoningEffortV1,
    ToolChoiceV1,
)
from trialagentbench_harness.io import read_json_model


def resolve_experiment_condition_v1(
    *,
    condition_id: str,
    request_replicate_id: str,
    reasoning_effort: ReasoningEffortV1 | None,
    reasoning_capability_snapshot: Path | None,
    provider: Literal["openai", "openai_responses", "openrouter"],
    model: str,
    openrouter_provider: str | None,
    procedure_assistance: ProcedureAssistanceV1,
    maximum_turns_per_step: int,
    maximum_submission_attempts: int | None,
    tool_choice: ToolChoiceV1,
) -> ExperimentConditionV1:
    """Validate provider capability and bind the complete request condition."""

    if reasoning_effort is None:
        if reasoning_capability_snapshot is not None:
            raise ValueError("--reasoning-capability-snapshot requires --reasoning-effort.")
        reasoning = ReasoningConfigV1()
    else:
        if reasoning_capability_snapshot is None:
            raise ValueError("--reasoning-effort requires --reasoning-capability-snapshot.")
        capability = read_json_model(
            ProviderReasoningCapabilityV1,
            reasoning_capability_snapshot,
        )
        if capability.provider_transport != provider:
            raise ValueError("Reasoning capability transport does not match --provider.")
        if capability.model_id != model:
            raise ValueError("Reasoning capability model does not match --model.")
        if capability.upstream_provider != openrouter_provider:
            raise ValueError("Reasoning capability upstream does not match --openrouter-provider.")
        reasoning = ReasoningConfigV1(
            effort=reasoning_effort,
            capability=capability,
        )
    return ExperimentConditionV1(
        condition_id=condition_id,
        request_replicate_id=request_replicate_id,
        reasoning=reasoning,
        procedure_assistance=procedure_assistance,
        maximum_turns_per_step=maximum_turns_per_step,
        maximum_submission_attempts=maximum_submission_attempts,
        tool_choice=tool_choice,
    )


__all__ = ["resolve_experiment_condition_v1"]
