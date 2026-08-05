"""Reference-blind semantic assessment of narrative analysis reports."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic.types import JsonValue

from trialagentbench_harness.contracts.experiments import NarrativeParticipantContextV1
from trialagentbench_harness.io.checksums import canonical_payload_sha256
from trialagentbench_harness.ports import LLMProvider, LLMResponse

_TOOL_NAME = "record_report_assessment"
_COMPONENT_IDS = (
    "question",
    "method",
    "evidence",
    "integrity",
    "result_structure",
    "result_support",
)
_SYSTEM_PROMPT = """Assess the supplied clinical-trial analysis report against the supplied task,
method dictionary, and output contract. Use only statements present in the report. Do not infer omitted
analyses, invent evidence, or select a different analysis method. For each listed component, record whether
the report supplies adequate support, fails the requirement, or contains insufficient information. Cite
exact report line IDs for positive or negative claims; an omitted field may be marked failed without a
citation. Overall `conforms` is true exactly when every component passes. Call the supplied tool once and
emit no prose."""


class DirectJudgeRequestV1(BaseModel):
    """Participant-visible inputs for one semantic report assessment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.direct_judge_request/v1"] = "trialagentbench.direct_judge_request/v1"
    assignment_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    raw_response: str
    participant_context: NarrativeParticipantContextV1
    judge_model: str = Field(..., min_length=1)
    decoding_seed: int | None = Field(default=None, ge=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=256)
    timeout_seconds: float = Field(default=120.0, gt=0.0)

    @model_validator(mode="after")
    def validate_context(self) -> DirectJudgeRequestV1:
        """Bind the judge request to its participant-visible task context."""

        if self.participant_context.task_id != self.task_id:
            raise ValueError("Direct-judge task_id does not match participant context.")
        return self


class DirectJudgeComponentV1(BaseModel):
    """One component of a semantic report assessment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: Literal[
        "question",
        "method",
        "evidence",
        "integrity",
        "result_structure",
        "result_support",
    ]
    status: Literal["passed", "failed", "insufficient_information"]
    report_line_ids: tuple[str, ...] = Field(default_factory=tuple)
    reason: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_line_ids(self) -> DirectJudgeComponentV1:
        """Require canonical, unique report-line identifiers."""

        if any(not value.startswith("L") or not value[1:].isdigit() for value in self.report_line_ids):
            raise ValueError("Direct-judge report_line_ids must use canonical L<number> identities.")
        if len(set(self.report_line_ids)) != len(self.report_line_ids):
            raise ValueError("Direct-judge report_line_ids must be unique within a component.")
        return self


class DirectJudgeDecisionV1(BaseModel):
    """Provider-authored semantic assessment of one report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conforms: bool
    components: tuple[DirectJudgeComponentV1, ...] = Field(..., min_length=6, max_length=6)

    @model_validator(mode="after")
    def validate_conjunction(self) -> DirectJudgeDecisionV1:
        """Make the overall report assessment noncompensatory."""

        component_ids = tuple(component.component_id for component in self.components)
        if component_ids != _COMPONENT_IDS:
            raise ValueError("Direct-judge components must use the canonical order exactly once.")
        if self.conforms != all(component.status == "passed" for component in self.components):
            raise ValueError("Direct-judge conforms must equal the conjunction of component passes.")
        return self


class _DirectJudgeToolArgumentsV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    parameters: DirectJudgeDecisionV1


class DirectJudgeResultV1(BaseModel):
    """Immutable result from one semantic report assessment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.direct_judge_result/v1"] = "trialagentbench.direct_judge_result/v1"
    request_sha256: str = Field(..., min_length=64, max_length=64)
    prompt_sha256: str = Field(..., min_length=64, max_length=64)
    schema_sha256: str = Field(..., min_length=64, max_length=64)
    status: Literal["completed", "invalid_response"]
    decision: DirectJudgeDecisionV1 | None = None
    failure_reason: str | None = None
    raw_provider_response: str
    provider_response_id: str | None = None
    returned_model: str | None = None
    upstream_provider: str | None = None
    finish_reason: str | None = None
    usage: dict[str, JsonValue] | None = None
    reported_cost_usd: float | None = Field(default=None, ge=0.0)
    request_attempts: int = Field(default=1, ge=1)
    transient_failure_count: int = Field(default=0, ge=0)
    backoff_seconds: float = Field(default=0.0, ge=0.0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_result(self) -> DirectJudgeResultV1:
        """Bind disposition, telemetry, and checksum."""

        if self.status == "completed" and (self.decision is None or self.failure_reason is not None):
            raise ValueError("A completed direct judgement requires only a decision.")
        if self.status == "invalid_response" and (self.decision is not None or self.failure_reason is None):
            raise ValueError("An invalid direct judgement requires only a failure reason.")
        if self.request_attempts != self.transient_failure_count + 1:
            raise ValueError("Direct-judge attempts must equal transient failures plus one.")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("Direct-judge result checksum mismatch.")
        return self

    def with_checksum(self) -> DirectJudgeResultV1:
        """Return the result with its canonical checksum assigned."""

        return self.model_copy(
            update={"checksum": canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))}
        )


def _report_lines(report: str) -> tuple[dict[str, str], ...]:
    return tuple(
        {"line_id": f"L{line_number:06d}", "text": line}
        for line_number, line in enumerate(report.splitlines(), start=1)
        if line.strip()
    )


def _tool() -> dict[str, JsonValue]:
    return {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": "Record one semantic assessment of an analysis report.",
            "parameters": _DirectJudgeToolArgumentsV1.model_json_schema(),
        },
    }


def _raw_response(response: LLMResponse) -> str:
    return json.dumps(
        {
            "content": response.content,
            "tool_calls": [
                {"id": call.id, "name": call.name, "arguments": call.arguments} for call in response.tool_calls
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def judge_narrative_report_v1(*, request: DirectJudgeRequestV1, provider: LLMProvider) -> DirectJudgeResultV1:
    """Assess one narrative report from participant-visible inputs."""

    if provider.model != request.judge_model:
        raise ValueError("Direct-judge request model does not match the provider model.")
    schema_payload = {
        "participant_task_contract": request.participant_context.task_contract,
        "participant_output_contract": request.participant_context.participant_submission_contract,
        "participant_diagnostic_dictionary": (
            request.participant_context.participant_diagnostic_dictionary.model_dump(mode="json")
        ),
        "participant_method_dictionary": request.participant_context.participant_method_dictionary.model_dump(
            mode="json"
        ),
    }
    schema_sha256 = canonical_payload_sha256(cast(JsonValue, schema_payload))
    prompt_sha256 = hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    started = time.monotonic()
    response = provider.generate_turn(
        (
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "assignment_id": request.assignment_id,
                        "task_id": request.task_id,
                        "report_lines": _report_lines(request.raw_response),
                        "participant_context": schema_payload,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ),
        tools=(_tool(),),
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        timeout_seconds=request.timeout_seconds,
        tool_choice="required",
    )
    elapsed = time.monotonic() - started
    raw = _raw_response(response)
    decision = None
    failure_reason = None
    if len(response.tool_calls) != 1 or response.tool_calls[0].name != _TOOL_NAME:
        failure_reason = "The judge did not emit exactly one required judgement tool call."
    else:
        try:
            arguments = json.loads(response.tool_calls[0].arguments)
            envelope = _DirectJudgeToolArgumentsV1.model_validate(arguments)
            known_lines = {row["line_id"] for row in _report_lines(request.raw_response)}
            cited_lines = {
                line_id for component in envelope.parameters.components for line_id in component.report_line_ids
            }
            if not cited_lines <= known_lines:
                raise ValueError("The judge cited a report line outside the frozen report.")
            decision = envelope.parameters
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            failure_reason = f"Invalid direct-judge response: {exc}"
    metadata = response.metadata
    return DirectJudgeResultV1(
        request_sha256=canonical_payload_sha256(request.model_dump(mode="json")),
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        status="completed" if decision is not None else "invalid_response",
        decision=decision,
        failure_reason=failure_reason,
        raw_provider_response=raw,
        provider_response_id=metadata.response_id,
        returned_model=metadata.returned_model,
        upstream_provider=metadata.upstream_provider,
        finish_reason=metadata.finish_reason,
        usage=None if response.usage is None else dict(response.usage),
        reported_cost_usd=metadata.reported_cost_usd,
        request_attempts=metadata.request_attempts,
        transient_failure_count=metadata.transient_failure_count,
        backoff_seconds=metadata.backoff_seconds,
        elapsed_seconds=elapsed,
    ).with_checksum()


__all__ = [
    "DirectJudgeComponentV1",
    "DirectJudgeDecisionV1",
    "DirectJudgeRequestV1",
    "DirectJudgeResultV1",
    "judge_narrative_report_v1",
]
