"""Typed custody contract for exact TrialDev mid-program continuation."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import JsonValue

from trialagentbench_harness.contracts.core.config import ToolChoiceV1
from trialagentbench_harness.contracts.core.runs import (
    TrialDevMaterializationUsageV1,
)
from trialagentbench_harness.io.checksums import canonical_payload_sha256
from trialagentbench_harness.ports import CodeExecutionLimitsV1


class TrialDevCheckpointFunctionCallV1(BaseModel):
    """One assistant function call retained in provider-compatible form."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., min_length=1)
    arguments: str


class TrialDevCheckpointToolCallV1(BaseModel):
    """One assistant tool call retained with exact provider identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(..., min_length=1)
    type: Literal["function"] = "function"
    function: TrialDevCheckpointFunctionCallV1


class TrialDevCheckpointMessageV1(BaseModel):
    """One role-consistent message in the full TrialDev conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[TrialDevCheckpointToolCallV1, ...] = ()
    provider_state: tuple[JsonValue, ...] = Field(default=(), exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def validate_role_shape(self) -> Self:
        """Reject provider-invalid message role combinations."""

        if self.role in {"system", "user"}:
            if self.content is None or self.tool_call_id is not None or self.tool_calls or self.provider_state:
                raise ValueError(f"{self.role} checkpoint messages require only string content.")
        elif self.role == "tool":
            if self.content is None or self.tool_call_id is None or self.tool_calls or self.provider_state:
                raise ValueError("Tool checkpoint messages require content and tool_call_id.")
        elif self.tool_call_id is not None:
            raise ValueError("Assistant checkpoint messages cannot contain tool_call_id.")
        elif self.provider_state and not all(
            isinstance(item, dict) and isinstance(item.get("type"), str) for item in self.provider_state
        ):
            raise ValueError("Assistant provider_state must contain typed provider output items.")
        return self

    def to_message(self) -> dict:
        """Return the exact provider-facing message shape."""

        message: dict = {"role": self.role}
        if self.content is not None or self.role == "assistant":
            message["content"] = self.content
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            message["tool_calls"] = [call.model_dump(mode="json") for call in self.tool_calls]
        if self.provider_state:
            message["provider_state"] = list(self.provider_state)
        return message


class TrialDevCheckpointArtifactV1(BaseModel):
    """A runner-custody artifact bound by path, kind, and SHA-256."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(..., min_length=1)
    kind: Literal["file", "directory"]
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_relative_path(self) -> Self:
        """Require a normalized path below the runner custody root."""

        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != self.relative_path:
            raise ValueError("Checkpoint artifact paths must be normalized relative paths.")
        return self


class TrialDevCheckpointViolationV1(BaseModel):
    """One typed request or materialization violation observed by the runner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase_id: str = Field(..., min_length=1)
    kind: Literal[
        "schema_validation",
        "materialize_rejection",
        "unsupported_continuation",
    ]
    error: str = Field(..., min_length=1)
    artifact_relative_path: str | None = None

    @model_validator(mode="after")
    def validate_artifact_relative_path(self) -> Self:
        """Require rejection evidence to remain below runner custody."""

        if self.artifact_relative_path is None:
            return self
        path = PurePosixPath(self.artifact_relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != self.artifact_relative_path
            or not path.parts
        ):
            raise ValueError("Violation artifact paths must be normalized relative paths.")
        if self.kind != "materialize_rejection":
            raise ValueError("Only materialization rejections may reference an archived artifact.")
        return self


class TrialDevCheckpointPhaseSummaryV1(BaseModel):
    """Exact completed-phase summary used in subsequent participant prompts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase_id: str = Field(..., min_length=1)
    decision_action: str | None = None
    advance: bool | None = None
    candidate_drug_id: str | None = None
    matched_item_id: str | None = None
    primary_effect: JsonValue = None
    safety_estimate: JsonValue = None


class TrialDevPendingStepV1(BaseModel):
    """Exact non-terminal semantic step awaiting continuation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase_id: str = Field(..., min_length=1)
    step_id: str = Field(..., min_length=1)
    turns_used: int = Field(..., ge=0)
    active_prompt_index: int = Field(..., ge=1)
    next_event_index: int = Field(..., ge=0)


class TrialDevContinuationPayloadV1(BaseModel):
    """Complete typed state needed to reconstruct one TrialDev agent loop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialdev_continuation_payload/v1"] = (
        "trialagentbench.trialdev_continuation_payload/v1"
    )
    program_id: str = Field(..., min_length=1)
    scenario_id: str = Field(..., min_length=1)
    objective_id: str = Field(..., min_length=1)
    provider_model: str = Field(..., min_length=1)
    provider_route: str = Field(..., min_length=1)
    system_prompt_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    workdir_relative_path: str = Field(..., min_length=1)
    current_state: TrialDevCheckpointArtifactV1
    scratch_workspace: TrialDevCheckpointArtifactV1
    materialization_usage: TrialDevMaterializationUsageV1
    completed_phase_summaries: tuple[TrialDevCheckpointPhaseSummaryV1, ...] = ()
    violations: tuple[TrialDevCheckpointViolationV1, ...] = ()
    conversation: tuple[TrialDevCheckpointMessageV1, ...] = Field(
        ...,
        min_length=2,
    )
    pending_step: TrialDevPendingStepV1
    temperature: float
    max_tokens: int = Field(..., ge=1)
    max_turns_per_step: int = Field(..., ge=1)
    tool_choice: ToolChoiceV1 = "auto"
    max_tool_output_chars: int = Field(..., ge=1)
    max_context_chars: int = Field(..., ge=1)
    executor_image: str | None = None
    executor_limits: CodeExecutionLimitsV1 | None = None
    remaining_deadline_seconds: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_continuation_state(self) -> Self:
        """Require internally consistent identities, phase order, and turns."""

        if self.pending_step.turns_used > self.max_turns_per_step:
            raise ValueError("Pending-step turns_used exceeds max_turns_per_step.")
        if self.pending_step.active_prompt_index >= len(self.conversation):
            raise ValueError("Pending-step active_prompt_index is outside the conversation.")
        prompt = self.conversation[self.pending_step.active_prompt_index]
        if prompt.role != "user":
            raise ValueError("Pending-step active_prompt_index must identify a user prompt.")
        observed_turns = sum(
            message.role == "assistant" for message in self.conversation[self.pending_step.active_prompt_index + 1 :]
        )
        if observed_turns != self.pending_step.turns_used:
            raise ValueError("Pending-step turns_used must equal assistant turns after the active prompt.")
        _validate_complete_tool_pairs(self.conversation)
        phases = [summary.phase_id for summary in self.completed_phase_summaries]
        if len(phases) != len(set(phases)):
            raise ValueError("Completed TrialDev phase summaries must have unique phase_id values.")
        if any(
            isinstance(count, bool) or count < 0
            for count in self.materialization_usage.materialize_calls_by_phase.values()
        ):
            raise ValueError("Materialization usage counts must be non-negative integers.")
        return self


class TrialDevContinuationCheckpointV1(BaseModel):
    """Checksum-bearing continuation envelope persisted by the runner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialdev_continuation_checkpoint/v1"] = (
        "trialagentbench.trialdev_continuation_checkpoint/v1"
    )
    payload: TrialDevContinuationPayloadV1
    payload_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        payload: TrialDevContinuationPayloadV1,
    ) -> TrialDevContinuationCheckpointV1:
        """Create a checksum-bearing envelope from a validated payload."""

        checksum = canonical_payload_sha256(cast(JsonValue, payload.model_dump(mode="json")))
        return cls(payload=payload, payload_sha256=checksum)

    @model_validator(mode="after")
    def validate_payload_checksum(self) -> Self:
        """Reject mutated or partially transcribed continuation payloads."""

        observed = canonical_payload_sha256(cast(JsonValue, self.payload.model_dump(mode="json")))
        if observed != self.payload_sha256:
            raise ValueError("TrialDev continuation payload checksum mismatch.")
        return self


def _validate_complete_tool_pairs(
    messages: tuple[TrialDevCheckpointMessageV1, ...],
) -> None:
    observed_ids: set[str] = set()
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "tool":
            raise ValueError(f"Unmatched checkpoint tool response at index {index}.")
        if message.role != "assistant" or not message.tool_calls:
            index += 1
            continue
        call_ids = [call.id for call in message.tool_calls]
        if len(call_ids) != len(set(call_ids)) or observed_ids.intersection(call_ids):
            raise ValueError("Checkpoint conversation contains duplicate tool_call_id values.")
        observed_ids.update(call_ids)
        index += 1
        response_ids: list[str] = []
        while index < len(messages) and messages[index].role == "tool":
            tool_call_id = messages[index].tool_call_id
            if tool_call_id is None:
                raise ValueError("Checkpoint tool response is missing tool_call_id.")
            response_ids.append(tool_call_id)
            index += 1
        if len(response_ids) != len(set(response_ids)) or set(response_ids) != set(call_ids):
            raise ValueError("Checkpoint assistant/tool response IDs do not match.")


__all__ = [
    "TrialDevCheckpointArtifactV1",
    "TrialDevCheckpointMessageV1",
    "TrialDevCheckpointPhaseSummaryV1",
    "TrialDevCheckpointViolationV1",
    "TrialDevContinuationCheckpointV1",
    "TrialDevContinuationPayloadV1",
    "TrialDevPendingStepV1",
]
