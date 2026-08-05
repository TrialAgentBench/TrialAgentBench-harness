"""Agent conversation loop and tool definitions for TrialDevBench.

The orchestrator (``runner.py``) drives a per-program ``AgentLoop`` through
the obs_review and phase checkpoints. Each checkpoint is a small back-and-
forth: the runner appends a user-message context block, calls
``run_until_submit`` to drive turns until the agent emits one of the
submission tool calls, and reads the result.
"""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from pydantic.types import JsonValue

from trialagentbench_harness.adapters.docker_code_execution import DockerPythonSession
from trialagentbench_harness.adapters.trialdev_share import (
    PhaseModuleSpecV1,
    TrialDevelopmentObservationalReviewSubmissionV1,
    TrialDevelopmentPhaseActionSpecV1,
    TrialDevelopmentPhaseDecisionSubmissionV1,
)
from trialagentbench_harness.contracts.core.config import ToolChoiceV1
from trialagentbench_harness.contracts.core.runs import (
    ProviderRequestEventV1,
    TrialDevMaterializationUsageV1,
)
from trialagentbench_harness.contracts.submission.schema import (
    trialdev_phase_request_schema,
    trialdev_randomized_phase_analysis_schema,
)
from trialagentbench_harness.contracts.trace.observable import (
    BenchmarkRuntimeTraceEventV1,
    runtime_event_source_payload_v1,
)
from trialagentbench_harness.contracts.trialdev.runtime_checkpoint import (
    TrialDevCheckpointArtifactV1,
    TrialDevCheckpointMessageV1,
    TrialDevCheckpointPhaseSummaryV1,
    TrialDevCheckpointViolationV1,
    TrialDevContinuationCheckpointV1,
    TrialDevContinuationPayloadV1,
    TrialDevPendingStepV1,
)
from trialagentbench_harness.execution_policy import TRIALDEV_RELEASE_BUDGET_V1
from trialagentbench_harness.io.checksums import canonical_payload_sha256, sha256_path
from trialagentbench_harness.io.json import append_jsonl_model
from trialagentbench_harness.ports import (
    CodeExecutionLimitsV1,
    CodeExecutionResultV1,
    CodeExecutionSession,
    LLMProvider,
    LLMResponse,
    ToolCall,
)
from trialagentbench_harness.ports.tool_input import ToolInputError, parse_json_object_text, parse_tool_arguments
from trialagentbench_harness.tools.workspace import (
    WORKSPACE_TOOLS,
    handle_workspace_tool,
    read_workspace_submission_text,
)
from trialagentbench_harness.trialdev.participant_submission import participant_schema_v1
from trialagentbench_harness.util.provider_telemetry import (
    fail_provider_request_v1,
    provider_failure_type_v1,
    start_provider_request_v1,
    succeed_provider_request_v1,
)
from trialagentbench_harness.util.runtime_context import (
    FINAL_TURN_SUBMISSION_REMINDER,
    RuntimeDeadline,
    bounded_provider_context,
    persist_bulky_tool_output,
    turn_budget_tag,
)

RuntimeEventType = Literal[
    "step_started",
    "prompt",
    "assistant_message",
    "tool_call",
    "tool_result",
    "code_execution",
    "file_inspection",
    "submission",
    "step_terminal",
]

DEFAULT_MAX_TURNS_PER_STEP = TRIALDEV_RELEASE_BUDGET_V1.maximum_turns
CheckpointContextMode: TypeAlias = Literal["exact", "active_step_only"]


class AgentTurnLimitExceeded(RuntimeError):
    """Raised when an agent does not submit within a semantic-step budget."""


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI / OpenRouter function-calling format)
# ---------------------------------------------------------------------------


_TOOL_EXECUTE_CODE = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": (
            "Execute Python code in a persistent session. State (variables, "
            "imports, fitted models, dataframes) persists across calls. "
            "pandas (pd) and numpy (np) are pre-imported. Scripts saved under "
            "scratch/ can be executed here. Use relative paths such as "
            "Path('scratch/analysis.py'), never an absolute path. Returns stdout/stderr."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "purpose": {"type": "string", "description": "Brief description (for logging)"},
            },
            "required": ["code"],
        },
    },
}

_TOOL_INSPECT_PARQUET = {
    "type": "function",
    "function": {
        "name": "inspect_parquet",
        "description": (
            "Quick inspection of a parquet file: shape, columns, dtypes, "
            "head(5), describe(), null counts. Resolves paths relative to "
            "the working directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the parquet file, e.g. "
                    "'observational_extract.parquet' or "
                    "'phase_phase2/trial_output/endpoints.parquet'.",
                },
            },
            "required": ["path"],
        },
    },
}

_FILE_SUBMISSION_NAMES = frozenset(
    {
        "submit_obs_review_analysis_and_decision_file",
        "submit_phase_request_file",
        "submit_phase_analysis_file",
        "submit_phase_decision_file",
        "submit_portfolio_checkpoint_file",
    }
)


def _file_submission_tool(name: str) -> dict[str, object]:
    """Return a lossless file transport for one typed submission tool."""

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                "Submit the current typed JSON payload from a regular UTF-8 file "
                "under scratch/. This is transport only; the identical current-step "
                "schema and validation apply."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path relative to scratch/ containing one JSON object, "
                            "for example submission.json. Do not include the scratch/ prefix."
                        ),
                    }
                },
                "required": ["path"],
            },
        },
    }


_TOOL_SUBMIT_PHASE_REQUEST_BASE = {
    "type": "function",
    "function": {
        "name": "submit_phase_request",
        "description": (
            "Submit the proposed design for the current study. The available "
            "fields and values reflect the study protocol and design policy."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}


def _build_phase_request_tool(phase_module: dict) -> dict:
    """Build a phase-aware ``submit_phase_request`` tool schema.

    Reads ``phase_module`` (one entry from ``eval_contract.json:phase_modules``)
    and emits a JSON schema that exposes ONLY the request fields legal at
    this phase. Fields whose ``allowed_*`` list is empty get omitted; fields
    with an enumerated set use a JSON enum constraint.
    """
    module = PhaseModuleSpecV1.model_validate(phase_module)
    if module.phase_id == "observational_review":
        raise ValueError("submit_phase_request is unavailable for observational_review.")
    if module.max_sample_size is None:
        raise ValueError(f"Phase {module.phase_id!r} requires an explicit public max_sample_size.")

    base_request_properties = trialdev_phase_request_schema()["properties"]
    if not isinstance(base_request_properties, dict):
        raise ValueError("Phase-request base schema must define object properties.")
    rationale_schema = base_request_properties["request_rationale"]
    if not isinstance(rationale_schema, dict):
        raise ValueError("Phase-request rationale schema must be an object.")

    props: dict[str, dict] = {
        "design_cell_id": {
            "type": "string",
            "minLength": 1,
            "description": ("Prospective phase-design cell selected from public/phase_design_policy.json."),
        },
        "candidate_drug_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 1,
        },
        "request_rationale": rationale_schema,
    }
    required = [
        "design_cell_id",
        "candidate_drug_ids",
        "target_sample_size",
        "follow_up_days",
        "enrollment_window_days",
        "site_count_budget",
        "interim_policy",
        "site_strategy",
        "selection_objective",
    ]

    # target_sample_size — bounded by phase module's max
    props["target_sample_size"] = {
        "type": "integer",
        "minimum": 1,
        "maximum": int(module.max_sample_size),
    }

    # enumerated lists: only expose when the menu is non-empty
    def _enum_int(field: str, allowed: list) -> None:
        if allowed:
            props[field] = {"type": "integer", "enum": [int(v) for v in allowed]}

    def _enum_str(field: str, allowed: list) -> None:
        if allowed:
            props[field] = {"type": "string", "enum": [str(v) for v in allowed]}

    _enum_int("follow_up_days", list(module.allowed_follow_up_days))
    _enum_int("enrollment_window_days", list(module.allowed_enrollment_window_days))
    _enum_int("site_count_budget", list(module.allowed_site_count_budgets))
    _enum_str("allocation_ratio", list(module.allowed_allocation_ratios))
    _enum_str("treatment_discontinuation_strategy", list(module.allowed_treatment_discontinuation_strategies))
    if module.allowed_treatment_discontinuation_strategies:
        required.append("treatment_discontinuation_strategy")
    _enum_str("interim_policy", list(module.allowed_interim_policies))
    _enum_str("site_strategy", list(module.allowed_site_strategies))
    _enum_str("selection_objective", list(module.allowed_selection_objectives))

    # The request endpoint is the primary efficacy endpoint. Safety uses its
    # separately typed estimate and does not need a second endpoint alias.
    allowed_endpoints = list(module.allowed_endpoint_ids)
    if allowed_endpoints:
        props["endpoint_id"] = {
            "type": "string",
            "enum": [str(v) for v in allowed_endpoints],
        }
        required.append("endpoint_id")

    required.append("allocation_ratio")

    # variable list fields — exposed only if phase has any allowed variables
    allowed_vars = list(module.allowed_variable_ids)
    if allowed_vars:
        props["analysis_covariates"] = {
            "type": "array",
            "items": {"type": "string", "enum": list(allowed_vars)},
        }
        if module.max_analysis_covariates is not None:
            props["analysis_covariates"]["maxItems"] = int(module.max_analysis_covariates)
        props["stratification_variables"] = {
            "type": "array",
            "items": {"type": "string", "enum": list(allowed_vars)},
        }
        props["subgroup_variables"] = {
            "type": "array",
            "items": {"type": "string", "enum": list(allowed_vars)},
        }
        if module.max_subgroup_splits is not None:
            props["subgroup_variables"]["maxItems"] = int(module.max_subgroup_splits)

    schema = json.loads(json.dumps(_TOOL_SUBMIT_PHASE_REQUEST_BASE))
    schema["function"]["parameters"]["properties"] = props
    schema["function"]["parameters"]["required"] = required
    return schema


_TOOL_SUBMIT_PHASE_ANALYSIS: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_phase_analysis",
        "description": (
            "Submit the statistical analysis of the completed study, including "
            "the estimand, estimates, uncertainty, safety results, and diagnostics."
        ),
        "parameters": participant_schema_v1(
            trialdev_randomized_phase_analysis_schema(),
            root_fields=frozenset({"scenario_id", "phase_id", "version"}),
        ),
    },
}

_TOOL_SUBMIT_PHASE_DECISION = {
    "type": "function",
    "function": {
        "name": "submit_phase_decision",
        "description": (
            "Submit the development decision supported by the completed analysis. "
            "Choose a currently available action and cite the analysis evidence used."
        ),
        "parameters": participant_schema_v1(
            TrialDevelopmentPhaseDecisionSubmissionV1,
            root_fields=frozenset({"scenario_id", "phase_id", "version"}),
        ),
    },
}


def _build_phase_decision_tool(action_spec: TrialDevelopmentPhaseActionSpecV1) -> dict:
    """Build a decision tool from the required phase action contract."""
    tool = json.loads(json.dumps(_TOOL_SUBMIT_PHASE_DECISION))
    allowed = list(action_spec.allowed_action_ids)
    tool["function"]["parameters"]["properties"]["decision_action"]["enum"] = allowed
    notes = action_spec.notes
    if notes:
        tool["function"]["description"] = tool["function"]["description"] + "\n\nPolicy note: " + str(notes)
    return tool


_TOOL_SUBMIT_OBS_REVIEW: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_obs_review_analysis_and_decision",
        "description": (
            "Submit one observational-review response branch. Use 'estimable' with candidate estimates, "
            "uncertainty, a complete ranking, and an evidence-linked nomination decision. Use "
            "'qualified_non_nomination' with public identification or support evidence and no candidate point "
            "estimates or causal ranking. "
            "There is no trial materialization at obs_review. This payload is "
            "normally evidence-rich; prefer "
            "submit_obs_review_analysis_and_decision_file."
        ),
        "parameters": participant_schema_v1(TrialDevelopmentObservationalReviewSubmissionV1),
    },
}


def tools_for_phase_request(phase_module: dict) -> list[dict]:
    """Return tools constrained by the required phase design contract."""
    submit = _build_phase_request_tool(phase_module)
    return [
        _TOOL_EXECUTE_CODE,
        _TOOL_INSPECT_PARQUET,
        *WORKSPACE_TOOLS,
        submit,
        _file_submission_tool("submit_phase_request_file"),
    ]


def tools_for_phase_decision(action_spec: TrialDevelopmentPhaseActionSpecV1) -> list[dict]:
    """Return tools constrained by the required phase action contract."""
    submit = _build_phase_decision_tool(action_spec)
    return [
        _TOOL_EXECUTE_CODE,
        _TOOL_INSPECT_PARQUET,
        *WORKSPACE_TOOLS,
        submit,
        _file_submission_tool("submit_phase_decision_file"),
    ]


_PHASE_ANALYSIS_TOOLS: list[dict] = [
    _TOOL_EXECUTE_CODE,
    _TOOL_INSPECT_PARQUET,
    *WORKSPACE_TOOLS,
    _TOOL_SUBMIT_PHASE_ANALYSIS,
    _file_submission_tool("submit_phase_analysis_file"),
]
_OBS_REVIEW_TOOLS: list[dict] = [
    _TOOL_EXECUTE_CODE,
    _TOOL_INSPECT_PARQUET,
    *WORKSPACE_TOOLS,
    _TOOL_SUBMIT_OBS_REVIEW,
    _file_submission_tool("submit_obs_review_analysis_and_decision_file"),
]


def runtime_submission_contracts() -> dict[str, object]:
    """Return exact participant schemas for file-only checkpoints."""

    return {
        "schema_id": "trialagentbench.trialdev_runtime_submission_contracts/v1",
        "observational_review": _TOOL_SUBMIT_OBS_REVIEW["function"]["parameters"],
        "phase_analysis": _TOOL_SUBMIT_PHASE_ANALYSIS["function"]["parameters"],
    }


def write_runtime_submission_contracts(workdir: Path) -> Path:
    """Stage exact participant schemas for file-only checkpoints."""

    path = workdir / "runtime_submission_contracts.json"
    if path.exists():
        raise FileExistsError(f"Runtime submission-contract path already exists: {path}")
    path.write_text(
        json.dumps(runtime_submission_contracts(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# AgentLoop — conversation primitives
# ---------------------------------------------------------------------------


@dataclass
class SubmissionCapture:
    """Returned by ``run_until_submit`` when the agent emits a submission tool."""

    name: str
    payload: dict[str, Any]
    tool_call_id: str


@dataclass
class AgentLoop:
    """Holds one program-long conversation plus its persistent Python session."""

    provider: LLMProvider
    workdir: Path
    system_prompt: str
    temperature: float = 0.0
    max_tokens: int = TRIALDEV_RELEASE_BUDGET_V1.maximum_completion_tokens_per_turn
    max_turns_per_step: int = DEFAULT_MAX_TURNS_PER_STEP
    tool_choice: ToolChoiceV1 = "auto"
    max_tool_output_chars: int = 4000
    verbose: bool = False
    conversation_log_path: Path | None = None  # if set, dumped after every turn
    event_log_path: Path | None = None
    provider_log_path: Path | None = None
    program_id: str | None = None
    scenario_id: str | None = None
    objective_id: str | None = None
    executor_image: str | None = None
    executor_limits: CodeExecutionLimitsV1 | None = None
    deadline_monotonic: float | None = None
    max_context_chars: int = TRIALDEV_RELEASE_BUDGET_V1.maximum_context_characters

    messages: list[dict] = field(default_factory=list)
    session: CodeExecutionSession | None = None
    trace_phase_id: str | None = None
    trace_step_id: str | None = None
    event_index: int = 0
    step_turns_used: int = 0
    step_terminal_emitted: bool = True
    active_prompt_index: int | None = None
    runtime_deadline: RuntimeDeadline = field(init=False)

    def __post_init__(self) -> None:
        """Create the persistent execution session and initial message state."""

        if self.max_turns_per_step < 1:
            raise ValueError("max_turns_per_step must be at least 1.")
        if self.max_tool_output_chars < 1:
            raise ValueError("max_tool_output_chars must be at least 1.")
        if self.max_context_chars < 1:
            raise ValueError("max_context_chars must be at least 1.")
        self.runtime_deadline = RuntimeDeadline(
            monotonic=self.deadline_monotonic,
            label="TrialDev programme",
        )
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.session = DockerPythonSession(
            cwd=self.workdir,
            image=self.executor_image,
            limits=self.executor_limits,
        )
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._persist_log()

    def close(self) -> None:
        if self.trace_phase_id is not None and not self.step_terminal_emitted:
            self._finish_step(status="completed")
        if self.session is not None:
            self.session.close()
            self.session = None

    def create_checkpoint(
        self,
        *,
        custody_root: Path,
        current_state_path: Path,
        materialization_usage: TrialDevMaterializationUsageV1,
        completed_phase_summaries: Sequence[TrialDevCheckpointPhaseSummaryV1],
        violations: Sequence[TrialDevCheckpointViolationV1],
    ) -> TrialDevContinuationCheckpointV1:
        """Snapshot exact continuation state without persisting runner custody."""

        if self.program_id is None or self.scenario_id is None or self.objective_id is None:
            raise RuntimeError("TrialDev checkpoint identity requires program, scenario, and objective IDs.")
        if self.trace_phase_id is None or self.trace_step_id is None or self.step_terminal_emitted:
            raise RuntimeError("TrialDev checkpoints require one active non-terminal semantic step.")
        if self.active_prompt_index is None:
            raise RuntimeError("TrialDev checkpoints require an active-step prompt.")
        root = Path(custody_root).resolve()
        workdir = self.workdir.resolve()
        workdir_relative = _relative_custody_path(workdir, root)
        scratch = workdir / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        if self.session is None:
            raise RuntimeError("TrialDev checkpoints require an active code-execution session.")
        snapshotted_scratch = self.session.snapshot_scratch()
        if snapshotted_scratch.resolve() != scratch.resolve():
            raise RuntimeError("Code-execution scratch snapshot does not match the agent workspace.")
        remaining = self.runtime_deadline.remaining()
        payload = TrialDevContinuationPayloadV1(
            program_id=self.program_id,
            scenario_id=self.scenario_id,
            objective_id=self.objective_id,
            provider_model=self.provider.model,
            provider_route=self.provider.telemetry_route,
            system_prompt_sha256=hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest(),
            workdir_relative_path=workdir_relative,
            current_state=_checkpoint_artifact(
                Path(current_state_path),
                custody_root=root,
                kind="file",
            ),
            scratch_workspace=_checkpoint_artifact(
                scratch,
                custody_root=root,
                kind="directory",
            ),
            materialization_usage=materialization_usage,
            completed_phase_summaries=tuple(completed_phase_summaries),
            violations=tuple(violations),
            conversation=tuple(TrialDevCheckpointMessageV1.model_validate(message) for message in self.messages),
            pending_step=TrialDevPendingStepV1(
                phase_id=self.trace_phase_id,
                step_id=self.trace_step_id,
                turns_used=self.step_turns_used,
                active_prompt_index=self.active_prompt_index,
                next_event_index=self.event_index,
            ),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_turns_per_step=self.max_turns_per_step,
            tool_choice=self.tool_choice,
            max_tool_output_chars=self.max_tool_output_chars,
            max_context_chars=self.max_context_chars,
            executor_image=self.executor_image,
            executor_limits=self.executor_limits,
            remaining_deadline_seconds=remaining,
        )
        return TrialDevContinuationCheckpointV1.create(payload)

    @classmethod
    def restore_from_checkpoint(
        cls,
        checkpoint: TrialDevContinuationCheckpointV1,
        *,
        provider: LLMProvider,
        custody_root: Path,
        system_prompt: str,
        verbose: bool = False,
        conversation_log_path: Path | None = None,
        event_log_path: Path | None = None,
        provider_log_path: Path | None = None,
        context_mode: CheckpointContextMode = "exact",
        reset_conversation_log_path: Path | None = None,
        reset_event_log_path: Path | None = None,
        reset_provider_log_path: Path | None = None,
        checkpoint_deadline_monotonic: float | None = None,
    ) -> AgentLoop:
        """Restore one validated pending step without rewriting runner state.

        ``active_step_only`` verifies the full source custody, then starts a
        fresh conversation containing only the system prompt and the pending
        public prompt. It is valid only before any assistant turn in that step.
        """

        payload = checkpoint.payload
        root = Path(custody_root).resolve()
        if context_mode == "exact" and (
            provider.model != payload.provider_model or provider.telemetry_route != payload.provider_route
        ):
            raise ValueError("TrialDev checkpoint provider identity mismatch.")
        prompt_sha256 = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        if prompt_sha256 != payload.system_prompt_sha256:
            raise ValueError("TrialDev checkpoint system prompt checksum mismatch.")
        workdir = _resolve_custody_path(payload.workdir_relative_path, root)
        _validate_checkpoint_artifact(payload.current_state, custody_root=root)
        scratch = _validate_checkpoint_artifact(payload.scratch_workspace, custody_root=root)
        if scratch != workdir / "scratch":
            raise ValueError("TrialDev checkpoint scratch workspace does not belong to its workdir.")
        restored_messages = [message.to_message() for message in payload.conversation]
        _validate_conversation_log(conversation_log_path, restored_messages)
        _validate_event_log(event_log_path, payload.pending_step)
        _validate_provider_log(
            provider_log_path,
            payload=payload,
        )
        if context_mode == "active_step_only":
            if payload.pending_step.turns_used != 0:
                raise ValueError("Active-step-only continuation requires a pre-response checkpoint.")
            active_prompt = restored_messages[payload.pending_step.active_prompt_index]
            if active_prompt.get("role") != "user":
                raise ValueError("Active-step-only continuation requires one active public user prompt.")
            restored_messages = [
                {"role": "system", "content": system_prompt},
                active_prompt,
            ]
            conversation_log_path = reset_conversation_log_path
            event_log_path = reset_event_log_path
            provider_log_path = reset_provider_log_path
        elif any(
            path is not None
            for path in (
                reset_conversation_log_path,
                reset_event_log_path,
                reset_provider_log_path,
            )
        ):
            raise ValueError("Reset log paths are valid only for active-step-only continuation.")
        deadline_monotonic = checkpoint_deadline_monotonic
        if deadline_monotonic is None and payload.remaining_deadline_seconds is not None:
            deadline_monotonic = time.monotonic() + payload.remaining_deadline_seconds
        loop = cls(
            provider=provider,
            workdir=workdir,
            system_prompt=system_prompt,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            max_turns_per_step=payload.max_turns_per_step,
            tool_choice=payload.tool_choice,
            max_tool_output_chars=payload.max_tool_output_chars,
            verbose=verbose,
            conversation_log_path=conversation_log_path,
            event_log_path=event_log_path,
            provider_log_path=provider_log_path,
            program_id=payload.program_id,
            scenario_id=payload.scenario_id,
            objective_id=payload.objective_id,
            executor_image=payload.executor_image,
            executor_limits=payload.executor_limits,
            deadline_monotonic=deadline_monotonic,
            max_context_chars=payload.max_context_chars,
        )
        loop.messages = restored_messages
        loop.trace_phase_id = payload.pending_step.phase_id
        loop.trace_step_id = payload.pending_step.step_id
        loop.step_turns_used = 0 if context_mode == "active_step_only" else payload.pending_step.turns_used
        loop.step_terminal_emitted = False
        loop.active_prompt_index = (
            1 if context_mode == "active_step_only" else payload.pending_step.active_prompt_index
        )
        loop.event_index = 0 if context_mode == "active_step_only" else payload.pending_step.next_event_index
        loop._persist_log()
        return loop

    def _require_within_deadline(self) -> None:
        """Fail before new work starts once the programme budget is exhausted."""

        self.runtime_deadline.remaining()

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def begin_step(self, *, phase_id: str, step_id: str) -> None:
        """Set explicit trace context before exposing a benchmark step."""

        if self.trace_phase_id is not None and not self.step_terminal_emitted:
            self._finish_step(status="completed")
        self.trace_phase_id = str(phase_id)
        self.trace_step_id = str(step_id)
        self.step_turns_used = 0
        self.step_terminal_emitted = False
        self.active_prompt_index = None
        self._emit_event(event_type="step_started")

    def append_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        if self.active_prompt_index is None:
            self.active_prompt_index = len(self.messages) - 1
        self._persist_log()
        self._emit_event(event_type="prompt", conversation_message_index=len(self.messages) - 1)

    def append_tool_reply(
        self,
        tool_call_id: str,
        content: str,
        *,
        tool_name: str,
        execution: CodeExecutionResultV1 | None = None,
        status: Literal["observed", "invalid"] = "observed",
    ) -> None:
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})
        self._persist_log()
        self._emit_event(
            event_type="tool_result",
            conversation_message_index=len(self.messages) - 1,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status=status,
            execution_status=execution.status if execution is not None else None,
            elapsed_seconds=execution.elapsed_seconds if execution is not None else None,
            output_truncated=execution.output_truncated if execution is not None else None,
        )

    def run_until_submit(
        self,
        *,
        tools: list[dict],
        submit_tool_names: set[str],
    ) -> SubmissionCapture:
        """Drive turns until the agent calls one of ``submit_tool_names``.

        Non-submit tool calls (execute_code, inspect_parquet) are dispatched
        and the result is appended to the conversation. The submit tool's
        arguments are returned without dispatching them — the caller (runner)
        validates the payload and decides what to do next.
        """
        completed = False
        try:
            result = self._run_until_submit(
                tools=tools,
                submit_tool_names=submit_tool_names,
            )
            completed = True
            return result
        finally:
            if not completed and not self.step_terminal_emitted:
                error = sys.exception()
                failure_type = (
                    "turn_limit_no_submission" if isinstance(error, AgentTurnLimitExceeded) else type(error).__name__
                )
                self._finish_step(status="failed", failure_type=failure_type)

    def _run_until_submit(
        self,
        *,
        tools: list[dict],
        submit_tool_names: set[str],
    ) -> SubmissionCapture:
        """Execute one submission attempt within the active semantic step."""

        if self.session is None:
            raise RuntimeError("AgentLoop has been closed")
        if self.trace_phase_id is None or self.trace_step_id is None:
            raise RuntimeError("begin_step must set trace context before provider execution.")
        if self.active_prompt_index is None:
            raise RuntimeError("append_user_message must set the active-step prompt before provider execution.")
        if self.provider_log_path is not None and self.program_id is None:
            raise RuntimeError("program_id is required for provider telemetry.")
        for turn_number in range(self.step_turns_used + 1, self.max_turns_per_step + 1):
            remaining = self.runtime_deadline.remaining()
            if turn_number == self.max_turns_per_step:
                self.append_user_message(FINAL_TURN_SUBMISSION_REMINDER)
                request_tools = [
                    tool
                    for tool in tools
                    if isinstance(tool.get("function"), dict)
                    and str(tool["function"].get("name")) in submit_tool_names
                ]
            else:
                request_tools = tools
            request_handle = None
            if self.provider_log_path is not None and self.program_id is not None:
                request_handle = start_provider_request_v1(
                    path=self.provider_log_path,
                    benchmark="trialdev",
                    unit_id=self.program_id,
                    phase_id=self.trace_phase_id,
                    step_id=self.trace_step_id,
                    turn_index=turn_number,
                    requested_model=self.provider.model,
                    provider_route=self.provider.telemetry_route,
                )
            started = time.monotonic()
            response = None
            try:
                response = self.provider.generate_turn(
                    messages=bounded_provider_context(
                        self.messages,
                        session=self.session,
                        active_prompt_index=self.active_prompt_index,
                        max_chars=self.max_context_chars,
                        required_message_indices=(
                            (len(self.messages) - 1,) if turn_number == self.max_turns_per_step else ()
                        ),
                    ),
                    tools=request_tools,
                    temperature=float(self.temperature),
                    max_tokens=int(self.max_tokens),
                    timeout_seconds=remaining,
                    tool_choice=self.tool_choice,
                )
            finally:
                elapsed = time.monotonic() - started
                if request_handle is not None:
                    if response is None:
                        error = sys.exception()
                        fail_provider_request_v1(
                            request_handle,
                            elapsed_seconds=elapsed,
                            failure_type=provider_failure_type_v1(error),
                            error=error,
                        )
                    else:
                        try:
                            succeed_provider_request_v1(
                                request_handle,
                                elapsed_seconds=elapsed,
                                response=response,
                            )
                        except (TypeError, ValueError):
                            fail_provider_request_v1(
                                request_handle,
                                elapsed_seconds=elapsed,
                                failure_type="provider_error",
                            )
                            raise
            self.step_turns_used = turn_number
            assistant_msg = self._compose_assistant_message(response)
            self.messages.append(assistant_msg)
            self._persist_log()
            message_index = len(self.messages) - 1
            self._emit_event(event_type="assistant_message", conversation_message_index=message_index)
            # Surface a running turn counter to the agent on every tool reply
            # so it knows how much budget it has left without having to count
            # itself. Format chosen to be visually distinct without being noisy.
            counter_tag = turn_budget_tag(
                turn=turn_number,
                maximum=self.max_turns_per_step,
            )
            if not response.tool_calls:
                # Nudge: the agent must use tools — content alone never finishes a step.
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "(reminder) Use the available tools to make progress. "
                            "Free-text replies do not advance the program." + counter_tag
                        ),
                    }
                )
                continue

            for tool_call_index, tc in enumerate(response.tool_calls):
                self._require_within_deadline()
                event_type: RuntimeEventType = "submission" if tc.name in submit_tool_names else "tool_call"
                if tc.name == "execute_code":
                    event_type = "code_execution"
                elif tc.name in {"inspect_parquet", "read_workspace_file"}:
                    event_type = "file_inspection"
                file_accessed = None
                try:
                    payload = parse_tool_arguments(tc.arguments, tool_name=tc.name)
                    if tc.name == "inspect_parquet":
                        raw_path = payload.get("path")
                        if isinstance(raw_path, str) and raw_path:
                            resolved_path = (self.workdir / raw_path).resolve()
                            try:
                                file_accessed = resolved_path.relative_to(self.workdir.resolve()).as_posix()
                            except ValueError:
                                file_accessed = resolved_path.as_posix()
                    elif tc.name in {"read_workspace_file", "write_workspace_file"}:
                        raw_path = payload.get("path")
                        if isinstance(raw_path, str) and raw_path:
                            file_accessed = f"scratch/{raw_path}"
                    if tc.name in submit_tool_names:
                        if tc.name in _FILE_SUBMISSION_NAMES:
                            if set(payload) != {"path"}:
                                raise ToolInputError(f"{tc.name} accepts only the path field.")
                            submission_path = payload.get("path")
                            text = self.runtime_deadline.run_blocking(
                                partial(
                                    read_workspace_submission_text,
                                    path=submission_path,
                                    session=self.session,
                                ),
                                operation_name=f"TrialDev tool {tc.name}",
                                on_timeout=self.session.close,
                            )
                            payload = parse_json_object_text(
                                text,
                                label=f"{tc.name} payload",
                            )
                        self._emit_event(
                            event_type=event_type,
                            conversation_message_index=message_index,
                            tool_call_index=tool_call_index,
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            file_accessed=file_accessed,
                        )
                        return SubmissionCapture(name=tc.name, payload=payload, tool_call_id=tc.id)
                    execution = self.runtime_deadline.run_blocking(
                        partial(self._dispatch_local_tool, tc),
                        operation_name=f"TrialDev tool {tc.name}",
                        on_timeout=self.session.close,
                    )
                except ToolInputError as exc:
                    self._emit_event(
                        event_type=event_type,
                        conversation_message_index=message_index,
                        tool_call_index=tool_call_index,
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        file_accessed=file_accessed,
                        status="invalid",
                    )
                    self.append_tool_reply(
                        tc.id,
                        f"Tool input rejected: {exc}" + counter_tag,
                        tool_name=tc.name,
                        status="invalid",
                    )
                else:
                    self._emit_event(
                        event_type=event_type,
                        conversation_message_index=message_index,
                        tool_call_index=tool_call_index,
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        file_accessed=file_accessed,
                    )
                    reply = persist_bulky_tool_output(
                        self._agent_facing_execution_output(execution),
                        session=self.session,
                        artifact_id=(
                            f"{self.trace_phase_id}-{self.trace_step_id}-{turn_number}-{tool_call_index}-{tc.id}"
                        ),
                        inline_chars=self.max_tool_output_chars,
                    )
                    self.append_tool_reply(
                        tc.id,
                        reply + counter_tag,
                        tool_name=tc.name,
                        execution=execution,
                    )

        raise AgentTurnLimitExceeded(
            f"Agent did not emit one of {submit_tool_names} within {self.max_turns_per_step} turns"
        )

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _compose_assistant_message(self, response: LLMResponse) -> dict[str, Any]:
        if isinstance(response.raw, dict) and "provider_state" in response.raw:
            return response.raw
        msg: dict[str, Any] = {"role": "assistant"}
        if response.content:
            msg["content"] = response.content
        else:
            msg["content"] = None
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments or "{}"},
                }
                for tc in response.tool_calls
            ]
        return msg

    def _persist_log(self) -> None:
        """Persist the running conversation after every message."""
        if self.conversation_log_path is None:
            return
        self.conversation_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.conversation_log_path.write_text(json.dumps(self.messages, indent=2, default=str), encoding="utf-8")

    def _emit_event(
        self,
        *,
        event_type: RuntimeEventType,
        conversation_message_index: int | None = None,
        tool_call_index: int | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        file_accessed: str | None = None,
        execution_status: Literal["success", "execution_error", "timeout", "session_terminated"] | None = None,
        elapsed_seconds: float | None = None,
        output_truncated: bool | None = None,
        status: Literal["observed", "invalid"] = "observed",
        terminal_status: Literal["completed", "failed"] | None = None,
        failure_type: str | None = None,
    ) -> None:
        """Append one typed event under the active runner step."""

        if self.event_log_path is None:
            return
        if self.trace_phase_id is None or self.trace_step_id is None:
            raise RuntimeError("TrialDev trace context must be set before emitting step events.")
        source_record = {
            "benchmark": "trialdev",
            "program_id": self.program_id,
            "scenario_id": self.scenario_id,
            "objective_id": self.objective_id,
            "event_index": self.event_index,
            "phase_id": self.trace_phase_id,
            "step_id": self.trace_step_id,
            "event_type": event_type,
            "conversation_message_index": conversation_message_index,
            "tool_call_index": tool_call_index,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "file_accessed": file_accessed,
            "execution_status": execution_status,
            "elapsed_seconds": elapsed_seconds,
            "output_truncated": output_truncated,
            "status": status,
            "terminal_status": terminal_status,
            "failure_type": failure_type,
        }
        source_artifact_path = (
            self.conversation_log_path.as_posix()
            if self.conversation_log_path is not None
            else f"runner://trialdev/{self.program_id}/conversation"
        )
        conversation_message = (
            cast(JsonValue, self.messages[conversation_message_index])
            if conversation_message_index is not None
            else None
        )
        source_payload = runtime_event_source_payload_v1(
            benchmark="trialdev",
            task_id=None,
            program_id=self.program_id,
            scenario_id=self.scenario_id,
            objective_id=self.objective_id,
            phase_id=self.trace_phase_id,
            step_id=self.trace_step_id,
            event_type=event_type,
            terminal_status=terminal_status,
            failure_type=failure_type,
            conversation_message=conversation_message,
        )
        event = BenchmarkRuntimeTraceEventV1(
            **source_record,
            event_id=f"trialdev:{self.program_id}:{self.event_index:06d}",
            timestamp=datetime.now(UTC),
            source_artifact_path=source_artifact_path,
            source_payload_sha256=canonical_payload_sha256(source_payload),
        )
        append_jsonl_model(self.event_log_path, event)
        self.event_index += 1

    def _finish_step(
        self,
        *,
        status: Literal["completed", "failed"],
        failure_type: str | None = None,
    ) -> None:
        """Append exactly one terminal event for the active semantic step."""

        if self.step_terminal_emitted:
            raise RuntimeError("TrialDev step emitted more than one terminal runtime event.")
        self._emit_event(
            event_type="step_terminal",
            terminal_status=status,
            failure_type=failure_type,
        )
        self.step_terminal_emitted = True

    def _dispatch_local_tool(self, tc: ToolCall) -> CodeExecutionResultV1:
        if self.session is None:
            raise RuntimeError("AgentLoop has been closed")
        args = parse_tool_arguments(tc.arguments, tool_name=tc.name)

        if tc.name == "execute_code":
            code = args.get("code")
            if not isinstance(code, str):
                raise ToolInputError("execute_code requires string code.")
            if not code.strip():
                raise ToolInputError("execute_code requires non-empty code.")
            purpose = args.get("purpose")
            if purpose is not None and not isinstance(purpose, str):
                raise ToolInputError("execute_code purpose must be a string when provided.")
            result = self.session.execute_result(code)
            if self.verbose:
                preview = result.output[:300] + ("..." if len(result.output) > 300 else "")
                print(f"  [execute_code] {preview}")
            return result

        if tc.name in {"write_workspace_file", "read_workspace_file", "list_workspace_files"}:
            result = handle_workspace_tool(name=tc.name, arguments=args, session=self.session)
            return result

        if tc.name == "inspect_parquet":
            raw_path = args.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ToolInputError("inspect_parquet requires a non-empty string path.")
            rel_path = Path(raw_path)
            if rel_path.is_absolute() or rel_path.suffix.lower() != ".parquet":
                raise ToolInputError("inspect_parquet requires a relative .parquet path.")
            resolved_path = (self.workdir / rel_path).resolve()
            try:
                rel = resolved_path.relative_to(self.workdir.resolve()).as_posix()
            except ValueError as exc:
                raise ToolInputError("inspect_parquet path must remain inside the agent working directory.") from exc
            code = textwrap.dedent(f"""\
                import pandas as _pd
                _df = _pd.read_parquet({rel!r})
                print(f"Shape: {{_df.shape}}")
                print(f"Columns: {{list(_df.columns)}}")
                print(f"Dtypes:\\n{{_df.dtypes}}")
                print(f"Head:\\n{{_df.head()}}")
                print(f"Describe:\\n{{_df.describe(include='all')}}")
                print(f"Null counts:\\n{{_df.isnull().sum()}}")
                del _df
            """)
            result = self.session.execute_result(code)
            return result

        raise ToolInputError(f"Unknown local tool: {tc.name}")

    @staticmethod
    def _agent_facing_execution_output(result: CodeExecutionResultV1) -> str:
        """Render one execution result without suppressing its failure state."""

        if result.output:
            return result.output
        if result.status == "success":
            return "(code executed successfully; no stdout produced)"
        return f"[{result.status}]"


def _relative_custody_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"TrialDev checkpoint artifact is outside custody root: {path}") from exc


def _resolve_custody_path(relative_path: str, root: Path) -> Path:
    path = (root / relative_path).resolve()
    _relative_custody_path(path, root)
    return path


def _checkpoint_artifact(
    path: Path,
    *,
    custody_root: Path,
    kind: Literal["file", "directory"],
) -> TrialDevCheckpointArtifactV1:
    resolved = path.resolve()
    if kind == "file" and not resolved.is_file():
        raise FileNotFoundError(f"TrialDev checkpoint file does not exist: {resolved}")
    if kind == "directory" and not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return TrialDevCheckpointArtifactV1(
        relative_path=_relative_custody_path(resolved, custody_root),
        kind=kind,
        sha256=sha256_path(resolved),
    )


def _validate_checkpoint_artifact(
    artifact: TrialDevCheckpointArtifactV1,
    *,
    custody_root: Path,
) -> Path:
    path = _resolve_custody_path(artifact.relative_path, custody_root)
    if artifact.kind == "file" and not path.is_file():
        raise FileNotFoundError(f"TrialDev checkpoint file does not exist: {path}")
    if artifact.kind == "directory" and not path.is_dir():
        raise NotADirectoryError(path)
    observed = sha256_path(path)
    if observed != artifact.sha256:
        raise ValueError(f"TrialDev checkpoint artifact checksum mismatch: {artifact.relative_path}")
    return path


def _validate_conversation_log(
    path: Path | None,
    expected_messages: list[dict],
) -> None:
    if path is None:
        return
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"TrialDev continuation conversation log is missing: {target}")
    observed = json.loads(target.read_text(encoding="utf-8"))
    if observed != expected_messages:
        raise ValueError("TrialDev continuation conversation log has advanced or drifted.")


def _validate_event_log(
    path: Path | None,
    pending: TrialDevPendingStepV1,
) -> None:
    if path is None:
        return
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"TrialDev continuation event log is missing: {target}")
    events: list[BenchmarkRuntimeTraceEventV1] = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(BenchmarkRuntimeTraceEventV1.model_validate_json(line))
    if [event.event_index for event in events] != list(range(pending.next_event_index)):
        raise ValueError("TrialDev continuation event log index has advanced or drifted.")
    terminals = [
        event
        for event in events
        if event.phase_id == pending.phase_id
        and event.step_id == pending.step_id
        and event.event_type == "step_terminal"
    ]
    if terminals:
        raise ValueError("TrialDev continuation pending step already has a terminal event.")


def _validate_provider_log(
    path: Path | None,
    *,
    payload: TrialDevContinuationPayloadV1,
) -> None:
    if path is None:
        return
    target = Path(path)
    if not target.exists():
        if payload.pending_step.turns_used:
            raise FileNotFoundError(f"TrialDev continuation provider log is missing: {target}")
        return
    events: list[ProviderRequestEventV1] = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(ProviderRequestEventV1.model_validate_json(line))
    relevant = [
        event
        for event in events
        if event.benchmark == "trialdev"
        and event.unit_id == payload.program_id
        and event.phase_id == payload.pending_step.phase_id
        and event.step_id == payload.pending_step.step_id
    ]
    starts = [event for event in relevant if event.status == "started"]
    terminals = [event for event in relevant if event.status != "started"]
    start_ids = [event.request_id for event in starts]
    terminal_ids = [event.request_id for event in terminals]
    if len(start_ids) != len(set(start_ids)) or len(terminal_ids) != len(set(terminal_ids)):
        raise ValueError("TrialDev continuation provider log contains duplicate request IDs.")
    if set(start_ids) != set(terminal_ids):
        raise ValueError("TrialDev continuation provider log contains unmatched request IDs.")
    expected_turns = list(range(1, payload.pending_step.turns_used + 1))
    if sorted(event.turn_index for event in terminals) != expected_turns:
        raise ValueError("TrialDev continuation provider log turn counts have advanced or drifted.")


# ---------------------------------------------------------------------------
# Convenience tool-set accessors (the runner imports these)
# ---------------------------------------------------------------------------


def tools_for_phase_analysis() -> list[dict]:
    return list(_PHASE_ANALYSIS_TOOLS)


def tools_for_obs_review() -> list[dict]:
    return list(_OBS_REVIEW_TOOLS)


__all__ = [
    "AgentLoop",
    "AgentTurnLimitExceeded",
    "CheckpointContextMode",
    "DEFAULT_MAX_TURNS_PER_STEP",
    "SubmissionCapture",
    "tools_for_phase_request",
    "tools_for_phase_analysis",
    "tools_for_phase_decision",
    "tools_for_obs_review",
]
