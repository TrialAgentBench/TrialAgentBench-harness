"""Run the TrialEvalBench agent loop.

This module gives an LLM the visible surface of a benchmark item, a persistent
Python session, and tools for clinical trial analysis.

Structured responses are scored directly; narrative responses remain raw until
the separately qualified normalization stage.
"""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path, PurePosixPath
from typing import cast

from pydantic.types import JsonValue

from trialagentbench_harness.adapters.docker_code_execution import DockerPythonSession
from trialagentbench_harness.contracts.experiments import (
    ProcedureAssistanceV1,
    TrialEvalAnalysisSpecificationV1,
    TrialEvalPromptConditionV1,
    TrialEvalSubmissionInterfaceV1,
)
from trialagentbench_harness.contracts.release.trialeval_integrity import (
    TrialEvalPublicIntegrityPolicyV1,
)
from trialagentbench_harness.contracts.release.trialeval_manifest import (
    TrialEvalParticipantMethodDictionaryV1,
)
from trialagentbench_harness.contracts.release.trialeval_runtime_surface import (
    TrialEvalSemanticSubmissionContractV1,
)
from trialagentbench_harness.contracts.submission import (
    TrialEvalSubmissionV1,
    lint_submission_payload_v1,
    lint_submission_text_v1,
    render_submission_lint_v1,
)
from trialagentbench_harness.contracts.trace.observable import (
    BenchmarkRuntimeTraceEventV1,
    runtime_event_source_payload_v1,
)
from trialagentbench_harness.execution_policy import TRIALEVAL_RELEASE_BUDGET_V1
from trialagentbench_harness.io.checksums import canonical_payload_sha256
from trialagentbench_harness.io.json import append_jsonl_model
from trialagentbench_harness.ports import (
    CodeExecutionLimitsV1,
    CodeExecutionResultV1,
    CodeExecutionSession,
    LLMProvider,
    ToolCall,
)
from trialagentbench_harness.ports.tool_input import ToolInputError, parse_tool_arguments
from trialagentbench_harness.tools.workspace import (
    WORKSPACE_TOOLS,
    handle_workspace_tool,
    read_workspace_submission_text,
)
from trialagentbench_harness.trialeval.conditions import (
    procedure_assistance_v1,
    prompt_intervention_v1,
    prompt_set_sha256_v1,
    response_contract_sha256_v1,
    stage_response_contract_v1,
    submission_instruction_v1,
    submission_tools_v1,
)
from trialagentbench_harness.trialeval.data import (
    load_participant_diagnostic_dictionary,
    load_participant_method_dictionary,
    load_visible_context,
    participant_analysis_surface_sha256,
    participant_visible_document_names,
    stage_participant_evidence,
)
from trialagentbench_harness.trialeval.data_integrity import (
    DATA_INTEGRITY_POLICY_FILENAME_V1,
    stage_data_integrity_utility_v1,
)
from trialagentbench_harness.trialeval.schema import BenchmarkItem
from trialagentbench_harness.util.provider_telemetry import (
    fail_provider_request_v1,
    provider_failure_type_v1,
    start_provider_request_v1,
    succeed_provider_request_v1,
)
from trialagentbench_harness.util.runtime_context import (
    FINAL_TURN_SUBMISSION_REMINDER,
    SUBMISSION_WINDOW_REMINDER,
    RuntimeDeadline,
    bounded_provider_context,
    persist_bulky_tool_output,
    turn_budget_tag,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_TURNS = TRIALEVAL_RELEASE_BUDGET_V1.maximum_turns

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOL_EXECUTE_CODE: dict[str, JsonValue] = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": (
            "Execute Python code in a persistent session. State (variables, "
            "imports, fitted models, dataframes) persists across calls. "
            "Use this to load data, explore, run analyses, check diagnostics, "
            "or execute scripts saved with relative paths such as "
            "Path('scratch/analysis.py'), never an absolute path. "
            "Returns stdout/stderr."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
                "purpose": {
                    "type": "string",
                    "description": "Brief description of what this code does (for logging)",
                },
            },
            "required": ["code"],
        },
    },
}

_TOOL_INSPECT_PARQUET: dict[str, JsonValue] = {
    "type": "function",
    "function": {
        "name": "inspect_parquet",
        "description": (
            "Quick inspection of a parquet file: shows shape, columns, dtypes, "
            "head(5), describe(), and null counts. Use this before writing "
            "analysis code to understand the data structure."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": (
                        "Name of the parquet file (e.g. 'ADSL.parquet' or 'raw/subjects.parquet' for raw data)"
                    ),
                },
            },
            "required": ["filename"],
        },
    },
}


_TOOL_VALIDATE_DATA_INTEGRITY: dict[str, JsonValue] = {
    "type": "function",
    "function": {
        "name": "validate_data_integrity",
        "description": (
            "Validate a repaired Parquet analysis input against data_integrity_policy.json. The operation fails on "
            "ambiguous source records or an inexact repair and returns the canonical submission record for the "
            "validated input."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "analysis_input_path": {
                    "type": "string",
                    "description": "Relative Parquet path under scratch/, for example scratch/repaired_events.parquet",
                }
            },
            "required": ["analysis_input_path"],
        },
    },
}


def _declares_data_integrity(item: BenchmarkItem) -> bool:
    """Return whether the participant item declares a repairable integrity condition."""

    policy_path = item.visible_dir / DATA_INTEGRITY_POLICY_FILENAME_V1
    if not policy_path.is_file():
        return False
    policy = TrialEvalPublicIntegrityPolicyV1.model_validate_json(policy_path.read_text(encoding="utf-8"))
    if policy.task_id != (item.task_id or item.item_id):
        raise ValueError("TrialEval data-integrity policy does not match the selected task.")
    return True


def _get_tools(
    submission_interface: TrialEvalSubmissionInterfaceV1,
    *,
    data_integrity: bool = False,
) -> list[dict[str, JsonValue]]:
    """Return interface-invariant analysis and submission tools."""

    return [
        _TOOL_EXECUTE_CODE,
        _TOOL_INSPECT_PARQUET,
        *([_TOOL_VALIDATE_DATA_INTEGRITY] if data_integrity else []),
        *WORKSPACE_TOOLS,
        *submission_tools_v1(submission_interface),
    ]


def _get_submission_tools(
    submission_interface: TrialEvalSubmissionInterfaceV1,
) -> list[dict[str, JsonValue]]:
    """Return the only tools exposed on the terminal provider turn."""

    return list(submission_tools_v1(submission_interface))


def _tool_names(tools: list[dict[str, JsonValue]]) -> set[str]:
    """Return validated function names from provider tool declarations."""

    names: set[str] = set()
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, dict):
            raise RuntimeError("Provider tool declaration is missing a function object.")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("Provider tool declaration has an invalid function name.")
        names.add(name)
    return names


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _build_system_prompt(
    item: BenchmarkItem,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    procedure_assistance: ProcedureAssistanceV1 = "output_contract_only",
    analysis_specification: TrialEvalAnalysisSpecificationV1 = "locked_sap",
    prompt_condition: TrialEvalPromptConditionV1 = "neutral",
    submission_interface: TrialEvalSubmissionInterfaceV1 = "structured",
) -> str:
    """Build a method-neutral prompt from the participant-visible surface."""

    if analysis_specification != item.analysis_specification:
        raise ValueError(
            "Requested analysis_specification does not match the task's immutable participant evidence surface."
        )
    if item.raw_data_dir and item.raw_data_dir.exists():
        data_root = item.raw_data_dir.resolve()
        parquet_files = sorted(f"raw/{path.name}" for path in data_root.glob("*.parquet"))
        data_path_note = "data/raw/"
    else:
        data_root = item.data_dir.resolve()
        parquet_files = sorted(path.name for path in data_root.glob("*.parquet"))
        data_path_note = "data/"

    visible_documents = participant_visible_document_names(item)
    reconstruction = ""
    if item.reconstruction_task is not None:
        reconstruction = (
            "The supplied tables are source domains. Reconstruct the analysis data according to "
            "the participant-visible protocol and reconstruction specification before estimating."
        )
    data_integrity = ""
    if _declares_data_integrity(item):
        reporting = (
            "copy its submission_record into data_integrity_record in the final response"
            if submission_interface == "structured"
            else "report the returned repair action, affected domain, compound key, counts, and checksum in prose"
        )
        data_integrity = (
            "The supplied item declares a mechanically resolvable record defect in "
            "data_integrity_policy.json. Apply its declared exact-duplicate repair with executed code, write the "
            "repaired domain as a Parquet file under scratch/, then call validate_data_integrity with that path. "
            f"Use the validated analysis_input_path in place of the affected raw domain and {reporting}."
        )

    assistance = procedure_assistance_v1(procedure_assistance)
    intervention = prompt_intervention_v1(prompt_condition)
    submission = submission_instruction_v1(submission_interface)
    response_contract_files = "  - interface/response_contract.json"
    if submission_interface == "structured":
        response_contract_files += "\n  - interface/submission_shapes.json"

    files = "\\n".join(f"  - {name}" for name in parquet_files) or "  (none)"
    documents = "\\n".join(f"  - {name}" for name in visible_documents) or "  (none)"
    return textwrap.dedent(f"""\\
        You are analysing a clinical-study task from its participant-visible evidence.
        Use the persistent Python session to inspect the data and execute the analysis.
        A writable persistent scratch/ workspace is available through dedicated list/read/write tools;
        use it for reusable scripts and notes instead of repeatedly re-entering long code.
        Workspace-tool paths omit the scratch/ prefix; Python code uses relative paths such as
        Path("scratch/analysis.py"), never an absolute path.
        The isolated environment includes NumPy, pandas, PyArrow, SciPy, statsmodels, lifelines,
        scikit-learn, and matplotlib.
        Do not claim a method, diagnostic, sensitivity analysis, or result that you did not execute.
        Resolve the scientific question and required analysis inputs from the supplied task and protocol;
        if the evidence does not identify the requested result, report that limitation in the appropriate result shape.
        `method_dictionary.json` defines each analysis method and its potentially relevant diagnostics.
        After choosing a method, consider only the diagnostics named by that method. Execute each one that is
        applicable and supported by the released evidence, following the operation and reporting the metric in
        `diagnostic_dictionary.json`; cite every input file used. The diagnostic dictionary is global and does not
        identify which diagnostic or result is expected for this task.
        Follow any additional item-specific obligation declared in submission_contract.json.

        DATA FILES (./{data_path_note}):
        {files}

        PARTICIPANT DOCUMENTS:
        {documents}

        RESPONSE CONTRACT:
        {response_contract_files}

        {reconstruction}
        {data_integrity}
        {assistance}
        {intervention}
        {submission}

        You have at most {max_turns} turns. A missing submission receives no primary credit.
        """).strip()


def condition_provenance_v1(
    item: BenchmarkItem,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    procedure_assistance: ProcedureAssistanceV1 = "output_contract_only",
    analysis_specification: TrialEvalAnalysisSpecificationV1 = "locked_sap",
    prompt_condition: TrialEvalPromptConditionV1 = "neutral",
    submission_interface: TrialEvalSubmissionInterfaceV1 = "structured",
) -> dict[str, JsonValue]:
    """Return exact hashes for one rendered participant condition."""

    if item.analysis_specification != analysis_specification:
        raise ValueError("Requested analysis specification differs from the immutable participant task.")
    system_prompt = _build_system_prompt(
        item,
        max_turns=max_turns,
        procedure_assistance=procedure_assistance,
        analysis_specification=analysis_specification,
        prompt_condition=prompt_condition,
        submission_interface=submission_interface,
    )
    tools = _get_tools(
        submission_interface,
        data_integrity=_declares_data_integrity(item),
    )
    return {
        "procedure_assistance": procedure_assistance,
        "analysis_specification": analysis_specification,
        "analysis_surface_sha256": participant_analysis_surface_sha256(item),
        "prompt_condition": prompt_condition,
        "submission_interface": submission_interface,
        "max_turns": max_turns,
        "prompt_set_sha256": prompt_set_sha256_v1(),
        "rendered_system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "tool_schema_sha256": canonical_payload_sha256(cast(JsonValue, tools)),
        "response_contract_sha256": response_contract_sha256_v1(submission_interface),
    }


def _decode_response_v1(
    text: str,
    *,
    submission_interface: TrialEvalSubmissionInterfaceV1,
    required_deliverables: tuple[str, ...],
    label: str,
    expected_task_id: str | None = None,
    participant_contract_checksum: str | None = None,
    participant_artifact_paths: tuple[str, ...] | None = None,
    participant_method_dictionary: TrialEvalParticipantMethodDictionaryV1 | None = None,
) -> dict[str, object]:
    """Decode one common transport payload using its predeclared interface."""

    if submission_interface == "narrative":
        if not text.strip():
            raise ToolInputError(f"{label} requires one non-empty narrative report.")
        return {"report": text}
    report = lint_submission_text_v1(
        text,
        suite="trialeval",
        scope="participant_bound" if participant_contract_checksum is not None else "schema_only",
        expected_identity=expected_task_id,
        required_deliverables=required_deliverables,
        participant_contract_checksum=participant_contract_checksum,
        participant_artifact_paths=participant_artifact_paths,
        participant_method_dictionary=participant_method_dictionary,
    )
    if not report.valid:
        raise ToolInputError(f"{label} structured payload is invalid:\n{render_submission_lint_v1(report)}")
    payload = json.loads(text)
    submission = TrialEvalSubmissionV1.model_validate(payload)
    return submission.model_dump(mode="json")


def _validate_structured_submission_v1(
    payload: dict[str, object],
    *,
    required_deliverables: tuple[str, ...],
    label: str,
    expected_task_id: str | None = None,
    participant_contract_checksum: str | None = None,
    participant_artifact_paths: tuple[str, ...] | None = None,
    participant_method_dictionary: TrialEvalParticipantMethodDictionaryV1 | None = None,
) -> dict[str, object]:
    """Validate one already-decoded structured submission."""

    report = lint_submission_payload_v1(
        payload,
        suite="trialeval",
        scope="participant_bound" if participant_contract_checksum is not None else "schema_only",
        expected_identity=expected_task_id,
        required_deliverables=required_deliverables,
        participant_contract_checksum=participant_contract_checksum,
        participant_artifact_paths=participant_artifact_paths,
        participant_method_dictionary=participant_method_dictionary,
    )
    if not report.valid:
        raise ToolInputError(f"{label} structured payload is invalid:\n{render_submission_lint_v1(report)}")
    submission = TrialEvalSubmissionV1.model_validate(payload)
    return submission.model_dump(mode="json")


def _handle_tool_call(
    tool_call: ToolCall,
    session: CodeExecutionSession,
    data_dir: Path,
    verbose: bool,
    *,
    submission_interface: TrialEvalSubmissionInterfaceV1,
    required_deliverables: tuple[str, ...],
    expected_task_id: str | None = None,
    participant_contract_checksum: str | None = None,
    participant_artifact_paths: tuple[str, ...] | None = None,
    participant_method_dictionary: TrialEvalParticipantMethodDictionaryV1 | None = None,
) -> tuple[str, dict[str, object] | None, CodeExecutionResultV1 | None]:
    """Execute a tool call and preserve any isolated execution result."""
    name = tool_call.name
    args = parse_tool_arguments(tool_call.arguments, tool_name=name)

    if name == "execute_code":
        code = args.get("code")
        purpose = args.get("purpose", "")
        if not isinstance(code, str) or not code.strip():
            raise ToolInputError("execute_code requires non-empty string code.")
        if not isinstance(purpose, str):
            raise ToolInputError("execute_code purpose must be a string when provided.")
        if verbose:
            print(f"  [execute_code] {purpose}")
            if len(code) < 300:
                print(f"  Code: {code}")
            else:
                print(f"  Code: {code[:200]}... ({len(code)} chars)")
        execution = session.execute_result(code)
        output = _agent_facing_execution_output(execution)
        if verbose:
            preview = output[:600] + ("..." if len(output) > 600 else "")
            print(f"  Output: {preview}")
        return output, None, execution

    if name == "validate_data_integrity":
        analysis_input_path = args.get("analysis_input_path")
        if set(args) != {"analysis_input_path"} or not isinstance(analysis_input_path, str):
            raise ToolInputError("validate_data_integrity requires one string analysis_input_path.")
        code = textwrap.dedent(f"""\
            import json as _integrity_json
            from interface.data_integrity import validate_declared_data_integrity_v1

            _integrity_repair = validate_declared_data_integrity_v1(
                analysis_input_path={json.dumps(analysis_input_path)}
            )
            DATA_INTEGRITY_ANALYSIS_INPUT = _integrity_repair["analysis_input_path"]
            print(_integrity_json.dumps(_integrity_repair, indent=2, sort_keys=True))
            """)
        execution = session.execute_result(code)
        output = _agent_facing_execution_output(execution)
        if verbose:
            print(f"  [validate_data_integrity] {analysis_input_path}")
            print(f"  Output: {output[:600]}{'...' if len(output) > 600 else ''}")
        return output, None, execution

    if name in {"write_workspace_file", "read_workspace_file", "list_workspace_files"}:
        execution = handle_workspace_tool(name=name, arguments=args, session=session)
        return execution.output, None, execution

    elif name == "inspect_parquet":
        filename = args.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ToolInputError("inspect_parquet requires a non-empty filename.")
        # Strip leading data/ if agent included it (common mistake)
        if filename.startswith("data/"):
            filename = filename[5:]
        relative = PurePosixPath(filename)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".parquet":
            raise ToolInputError("inspect_parquet filename must be a relative parquet path under data/.")
        if verbose:
            print(f"  [inspect_parquet] {filename}")
        # Support both data/{filename} and data/raw/{filename}
        code = textwrap.dedent(f"""\
            _df = pd.read_parquet('data/{filename}')
            print(f"Shape: {{_df.shape}}")
            print(f"\\nColumns: {{list(_df.columns)}}")
            print(f"\\nDtypes:\\n{{_df.dtypes}}")
            print(f"\\nHead:\\n{{_df.head()}}")
            print(f"\\nDescribe:\\n{{_df.describe(include='all')}}")
            print(f"\\nNull counts:\\n{{_df.isnull().sum()}}")
            del _df
        """)
        execution = session.execute_result(code)
        output = _agent_facing_execution_output(execution)
        if verbose:
            preview = output[:600] + ("..." if len(output) > 600 else "")
            print(f"  Output: {preview}")
        return output, None, execution

    elif name == "submit_response":
        if submission_interface == "structured":
            answer = _validate_structured_submission_v1(
                args,
                required_deliverables=required_deliverables,
                label="submit_response",
                expected_task_id=expected_task_id,
                participant_contract_checksum=participant_contract_checksum,
                participant_artifact_paths=participant_artifact_paths,
                participant_method_dictionary=participant_method_dictionary,
            )
        else:
            if set(args) != {"content"} or not isinstance(args.get("content"), str):
                raise ToolInputError("Narrative submit_response accepts only one string content field.")
            answer = _decode_response_v1(
                cast(str, args["content"]),
                submission_interface=submission_interface,
                required_deliverables=required_deliverables,
                label="submit_response",
                expected_task_id=expected_task_id,
                participant_contract_checksum=participant_contract_checksum,
                participant_artifact_paths=participant_artifact_paths,
                participant_method_dictionary=participant_method_dictionary,
            )
        if verbose:
            print(f"  [submit_response] Agent submitting {submission_interface} response")
        return "Response submitted.", answer, None

    elif name == "submit_response_file":
        if set(args) != {"path"}:
            raise ToolInputError("submit_response_file accepts only the path field.")
        text = read_workspace_submission_text(path=args.get("path"), session=session)
        answer = _decode_response_v1(
            text,
            submission_interface=submission_interface,
            required_deliverables=required_deliverables,
            label="submit_response_file",
            expected_task_id=expected_task_id,
            participant_contract_checksum=participant_contract_checksum,
            participant_artifact_paths=participant_artifact_paths,
            participant_method_dictionary=participant_method_dictionary,
        )
        if verbose:
            print(f"  [submit_response_file] Agent submitting {submission_interface} response from scratch/")
        return "Response submitted.", answer, None

    raise ToolInputError(f"Unknown TrialEval tool: {name!r}.")


def _agent_facing_execution_output(result: CodeExecutionResultV1) -> str:
    """Render one execution result without suppressing its failure state."""

    if result.output:
        return result.output
    if result.status == "success":
        return "(code executed successfully; no stdout produced)"
    return f"[{result.status}]"


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------


def run_agent(
    item: BenchmarkItem,
    provider: LLMProvider,
    max_turns: int | None = None,
    temperature: float = 0.0,
    max_tokens: int = TRIALEVAL_RELEASE_BUDGET_V1.maximum_completion_tokens_per_turn,
    verbose: bool = True,
    procedure_assistance: ProcedureAssistanceV1 = "output_contract_only",
    analysis_specification: TrialEvalAnalysisSpecificationV1 | None = None,
    prompt_condition: TrialEvalPromptConditionV1 = "neutral",
    submission_interface: TrialEvalSubmissionInterfaceV1 = "structured",
    provider_log_path: Path | None = None,
    conversation_log_path: Path | None = None,
    event_log_path: Path | None = None,
    executor_image: str | None = None,
    executor_limits: CodeExecutionLimitsV1 | None = None,
    max_elapsed_seconds: float | None = None,
    item_workspace: Path | None = None,
    max_context_chars: int = TRIALEVAL_RELEASE_BUDGET_V1.maximum_context_characters,
) -> dict:
    """
    Run the analysis agent against a benchmark item.

    Returns the execution status, canonical submission, turn count, and
    observable conversation trace.
    """
    if max_turns is None:
        max_turns = DEFAULT_MAX_TURNS
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1.")
    if max_elapsed_seconds is not None and max_elapsed_seconds <= 0.0:
        raise ValueError("max_elapsed_seconds must be positive when provided.")
    if max_context_chars < 1:
        raise ValueError("max_context_chars must be at least 1.")
    if analysis_specification is None:
        analysis_specification = item.analysis_specification
    output_contract = TrialEvalSemanticSubmissionContractV1.model_validate(item.submission_contract)
    if output_contract.task_id != (item.task_id or item.item_id):
        raise ValueError("Participant semantic submission contract does not match the selected task.")
    required_deliverables = tuple(str(value) for value in output_contract.required_deliverables)
    deadline = RuntimeDeadline.after(max_elapsed_seconds, label="TrialEval item")

    system_prompt = _build_system_prompt(
        item,
        max_turns=max_turns,
        procedure_assistance=procedure_assistance,
        analysis_specification=analysis_specification,
        prompt_condition=prompt_condition,
        submission_interface=submission_interface,
    )
    visible_context = load_visible_context(item)
    tools = _get_tools(
        submission_interface,
        data_integrity=_declares_data_integrity(item),
    )
    condition_provenance = condition_provenance_v1(
        item,
        max_turns=max_turns,
        procedure_assistance=procedure_assistance,
        analysis_specification=analysis_specification,
        prompt_condition=prompt_condition,
        submission_interface=submission_interface,
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": visible_context},
    ]
    # Snapshot the rendered prompts so the saved transcript stays self-
    # contained even if agent.py or the bundle drifts later. These are turn-0
    # entries, never sent to the LLM separately (they're already in `messages`).
    conversation_log: list[dict] = [
        {"role": "system", "turn": 0, "content": system_prompt},
        {"role": "user", "turn": 0, "content": visible_context},
    ]
    task_id = item.task_id or item.item_id
    source_artifact_path = (
        conversation_log_path.as_posix()
        if conversation_log_path is not None
        else f"runner://trialeval/{task_id}/conversation"
    )
    runtime_events: list[BenchmarkRuntimeTraceEventV1] = []
    event_log_initialized = False

    def _event(**values: object) -> None:
        event_index = len(runtime_events)
        message_index = values.get("conversation_message_index")
        conversation_message = (
            cast(JsonValue, conversation_log[message_index]) if isinstance(message_index, int) else None
        )
        event_type = str(values["event_type"])
        terminal_status = values.get("terminal_status")
        failure_type = values.get("failure_type")
        source_payload = runtime_event_source_payload_v1(
            benchmark="trialeval",
            task_id=task_id,
            program_id=None,
            scenario_id=None,
            objective_id=None,
            phase_id="task",
            step_id="analysis",
            event_type=event_type,
            terminal_status=str(terminal_status) if terminal_status is not None else None,
            failure_type=str(failure_type) if failure_type is not None else None,
            conversation_message=conversation_message,
        )
        source_record = {
            "benchmark": "trialeval",
            "task_id": task_id,
            "event_index": event_index,
            "phase_id": "task",
            "step_id": "analysis",
            **values,
        }
        event = BenchmarkRuntimeTraceEventV1.model_validate(
            {
                **source_record,
                "event_id": f"trialeval:{task_id}:{event_index:06d}",
                "timestamp": datetime.now(UTC),
                "source_artifact_path": source_artifact_path,
                "source_payload_sha256": canonical_payload_sha256(source_payload),
            }
        )
        runtime_events.append(event)
        if event_log_path is not None and event_log_initialized:
            append_jsonl_model(event_log_path, event)

    _event(event_type="step_started")
    _event(event_type="prompt", conversation_message_index=1)
    if conversation_log_path is not None:
        conversation_log_path.parent.mkdir(parents=True, exist_ok=True)
        conversation_log_path.write_text(
            json.dumps(conversation_log, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if event_log_path is not None:
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
        event_log_path.write_text(
            "".join(json.dumps(event.model_dump(mode="json"), sort_keys=True) + "\n" for event in runtime_events),
            encoding="utf-8",
        )
        event_log_initialized = True

    terminal_emitted = False

    def _terminal(*, status: str, failure_type: str | None = None) -> None:
        nonlocal terminal_emitted
        if terminal_emitted:
            raise RuntimeError("TrialEval assignment emitted more than one terminal runtime event.")
        _event(
            event_type="step_terminal",
            terminal_status=status,
            failure_type=failure_type,
        )
        terminal_emitted = True

    def _persist_conversation() -> None:
        if conversation_log_path is None:
            return
        conversation_log_path.write_text(
            json.dumps(conversation_log, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if item_workspace is None:
        log_path = conversation_log_path or event_log_path or provider_log_path
        if log_path is None:
            raise ValueError("item_workspace is required when no persistent item log path is configured.")
        log_stem = Path(log_path).stem
        for suffix in ("_conversation", "_events", "_provider_responses"):
            log_stem = log_stem.removesuffix(suffix)
        item_workspace = Path(log_path).parent / f"{log_stem}_workspace"
    workspace_root = Path(item_workspace)
    evidence_root = workspace_root / "item"
    session: CodeExecutionSession | None = None
    try:
        workspace_root.mkdir(parents=True, exist_ok=True)
        if evidence_root.exists():
            if not evidence_root.is_dir():
                raise FileExistsError(f"TrialEval item workspace is not a directory: {evidence_root}")
        else:
            stage_participant_evidence(item, evidence_root)
        stage_response_contract_v1(evidence_root, submission_interface)
        stage_data_integrity_utility_v1(evidence_root)
        load_participant_diagnostic_dictionary(evidence_root)
        _, participant_method_dictionary = load_participant_method_dictionary(evidence_root)
        participant_artifact_paths = tuple(
            sorted(
                path.relative_to(evidence_root).as_posix()
                for path in evidence_root.rglob("*")
                if path.is_file() and not path.relative_to(evidence_root).parts[0] == "interface"
            )
        )
        session = DockerPythonSession(
            cwd=evidence_root,
            image=executor_image,
            limits=executor_limits,
        )
        deadline.run_blocking(
            lambda: session.execute("DATA_DIR = 'data'"),
            operation_name="TrialEval executor initialization",
            on_timeout=session.close,
        )
        if item.raw_data_dir:
            deadline.run_blocking(
                lambda: session.execute("RAW_DATA_DIR = 'data/raw'"),
                operation_name="TrialEval executor initialization",
                on_timeout=session.close,
            )

        for turn in range(max_turns):
            remaining = deadline.remaining()
            submission_only_turn = turn >= max(0, max_turns - 2)
            if submission_only_turn:
                reminder = FINAL_TURN_SUBMISSION_REMINDER if turn == max_turns - 1 else SUBMISSION_WINDOW_REMINDER
                messages.append(
                    {
                        "role": "user",
                        "content": reminder,
                    }
                )
                conversation_log.append(
                    {
                        "role": "user",
                        "turn": turn + 1,
                        "content": reminder,
                    }
                )
                _persist_conversation()
                _event(
                    event_type="prompt",
                    conversation_message_index=len(conversation_log) - 1,
                )
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Turn {turn + 1}/{max_turns}")
                print(f"{'=' * 60}")

            request_handle = None
            if provider_log_path is not None:
                request_handle = start_provider_request_v1(
                    path=provider_log_path,
                    benchmark="trialeval",
                    unit_id=task_id,
                    phase_id="task",
                    step_id="analysis",
                    turn_index=turn + 1,
                    requested_model=provider.model,
                    provider_route=provider.telemetry_route,
                )
            turn_started = time.monotonic()
            response = None
            try:
                request_tools = _get_submission_tools(submission_interface) if submission_only_turn else tools
                request_tool_names = _tool_names(request_tools)
                response = provider.generate_turn(
                    messages=bounded_provider_context(
                        messages,
                        session=session,
                        active_prompt_index=1,
                        max_chars=max_context_chars,
                        required_message_indices=(len(messages) - 1,) if submission_only_turn else (),
                    ),
                    tools=request_tools,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    timeout_seconds=remaining,
                    tool_choice="required" if submission_only_turn else "auto",
                )
            finally:
                turn_elapsed = time.monotonic() - turn_started
                if request_handle is not None:
                    if response is None:
                        error = sys.exception()
                        fail_provider_request_v1(
                            request_handle,
                            elapsed_seconds=turn_elapsed,
                            failure_type=provider_failure_type_v1(error),
                            error=error,
                        )
                    else:
                        try:
                            succeed_provider_request_v1(
                                request_handle,
                                elapsed_seconds=turn_elapsed,
                                response=response,
                            )
                        except (TypeError, ValueError):
                            fail_provider_request_v1(
                                request_handle,
                                elapsed_seconds=turn_elapsed,
                                failure_type="provider_error",
                            )
                            raise

            # Append raw assistant message for OpenAI message threading
            if response.raw is not None:
                if not isinstance(response.raw, dict):
                    raise TypeError("LLM provider raw response must be a message object")
                messages.append(response.raw)
            else:
                msg_dict: dict = {"role": "assistant"}
                if response.content:
                    msg_dict["content"] = response.content
                if response.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments or "{}",
                            },
                        }
                        for call in response.tool_calls
                    ]
                messages.append(msg_dict)

            # Log the assistant turn with usage + wall time. Pure tool-call
            # turns have content=None but we still record them so per-turn
            # cost and latency are preserved.
            conversation_log.append(
                {
                    "role": "assistant",
                    "turn": turn + 1,
                    "content": response.content,
                    "usage": response.usage,
                    "elapsed_sec": round(turn_elapsed, 3),
                }
            )
            _persist_conversation()
            _event(event_type="assistant_message", conversation_message_index=len(conversation_log) - 1)
            if response.content and verbose:
                preview = response.content[:400] + ("..." if len(response.content) > 400 else "")
                print(f"[Reasoning]: {preview}")

            if response.tool_calls:
                for tool_call_index, tc in enumerate(response.tool_calls):
                    deadline.remaining()
                    tool_started = time.monotonic()
                    parsed_arguments: dict[str, object] = {}
                    try:
                        if tc.name not in request_tool_names:
                            raise ToolInputError(f"tool {tc.name!r} was not offered on this turn")
                        parsed_arguments = parse_tool_arguments(tc.arguments, tool_name=tc.name)
                        result_str, answer, execution = deadline.run_blocking(
                            partial(
                                _handle_tool_call,
                                tc,
                                session,
                                evidence_root / "data",
                                verbose,
                                submission_interface=submission_interface,
                                required_deliverables=required_deliverables,
                                expected_task_id=task_id,
                                participant_contract_checksum=output_contract.checksum,
                                participant_artifact_paths=participant_artifact_paths,
                                participant_method_dictionary=participant_method_dictionary,
                            ),
                            operation_name=f"TrialEval tool {tc.name}",
                            on_timeout=session.close,
                        )
                        tool_status = "observed"
                    except ToolInputError as exc:
                        result_str = f"Tool input rejected: {exc}"
                        answer = None
                        execution = None
                        tool_status = "invalid"
                    tool_elapsed = time.monotonic() - tool_started
                    result_str = persist_bulky_tool_output(
                        result_str,
                        session=session,
                        artifact_id=f"turn-{turn + 1}-call-{tool_call_index}-{tc.id}",
                        inline_chars=4000,
                    )
                    result_str += turn_budget_tag(turn=turn + 1, maximum=max_turns)

                    conversation_log.append(
                        {
                            "role": "tool",
                            "turn": turn + 1,
                            "tool_call_id": tc.id,
                            "tool": tc.name,
                            "args": parsed_arguments,
                            "output": result_str,
                            "elapsed_sec": round(tool_elapsed, 3),
                        }
                    )
                    _persist_conversation()
                    message_index = len(conversation_log) - 1
                    event_type = "submission" if answer is not None else "tool_call"
                    file_accessed = None
                    if tc.name == "execute_code":
                        event_type = "code_execution"
                    elif tc.name == "inspect_parquet":
                        event_type = "file_inspection"
                        filename = str(parsed_arguments.get("filename") or "")
                        file_accessed = f"data/{filename.removeprefix('data/')}" if filename else None
                    elif tc.name == "read_workspace_file":
                        event_type = "file_inspection"
                        path = parsed_arguments.get("path")
                        file_accessed = f"scratch/{path}" if isinstance(path, str) and path else None
                    elif tc.name == "write_workspace_file":
                        path = parsed_arguments.get("path")
                        file_accessed = f"scratch/{path}" if isinstance(path, str) and path else None
                    elif tc.name == "submit_response_file":
                        path = parsed_arguments.get("path")
                        file_accessed = f"scratch/{path}" if isinstance(path, str) and path else None
                    event_execution = execution if event_type == "code_execution" else None
                    _event(
                        event_type=event_type,
                        conversation_message_index=message_index,
                        tool_call_index=tool_call_index,
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        file_accessed=file_accessed,
                        status=tool_status,
                        execution_status=event_execution.status if event_execution is not None else None,
                        elapsed_seconds=event_execution.elapsed_seconds if event_execution is not None else None,
                        output_truncated=event_execution.output_truncated if event_execution is not None else None,
                    )
                    _event(
                        event_type="tool_result",
                        conversation_message_index=message_index,
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        status=tool_status,
                        execution_status=execution.status if execution is not None else None,
                        elapsed_seconds=execution.elapsed_seconds if execution is not None else None,
                        output_truncated=execution.output_truncated if execution is not None else None,
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_str,
                        }
                    )

                    if answer is not None:
                        expected_task_id = task_id
                        if submission_interface == "structured" and answer.get("task_id") != expected_task_id:
                            raise ValueError(
                                f"Submission task_id={answer.get('task_id')!r} does not match "
                                f"the active task {expected_task_id!r}."
                            )
                        _terminal(status="completed")
                        result_dict = {
                            "status": "success",
                            "turns_used": turn + 1,
                            "conversation": conversation_log,
                            "result": None,
                            "report": None,
                            "condition_provenance": condition_provenance,
                            "events": [event.model_dump(mode="json") for event in runtime_events],
                        }
                        if submission_interface == "structured":
                            result_dict["result"] = answer
                        else:
                            result_dict["report"] = answer["report"]
                        return result_dict

            elif not response.content:
                messages.append(
                    {
                        "role": "user",
                        "content": "Please continue your analysis using the available tools.",
                    }
                )

        _terminal(status="failed", failure_type="turn_limit_no_submission")
        result_dict = {
            "status": "max_turns_reached",
            "result": None,
            "report": None,
            "turns_used": max_turns,
            "conversation": conversation_log,
            "condition_provenance": condition_provenance,
            "events": [event.model_dump(mode="json") for event in runtime_events],
        }
        return result_dict

    finally:
        if not terminal_emitted:
            error = sys.exception()
            _terminal(
                status="failed",
                failure_type=type(error).__name__ if error is not None else "runtime_failure",
            )
        if session is not None:
            session.close()
