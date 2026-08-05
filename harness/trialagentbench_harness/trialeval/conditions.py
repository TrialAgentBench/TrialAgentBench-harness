"""Immutable participant-facing conditions for TrialEval experiments."""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from pydantic.types import JsonValue

from trialagentbench_harness.contracts.experiments import (
    ProcedureAssistanceV1,
    TrialEvalInterfaceAffordanceV1,
    TrialEvalPromptConditionV1,
    TrialEvalSchemaAffordanceInventoryV1,
    TrialEvalSubmissionInterfaceV1,
    procedure_assistance_exposure_v1,
)
from trialagentbench_harness.contracts.submission import (
    trialeval_submission_schema,
    trialeval_submission_shape_catalogue,
)
from trialagentbench_harness.io.checksums import canonical_payload_sha256
from trialagentbench_harness.util.runtime_context import FINAL_TURN_SUBMISSION_REMINDER

_PROMPT_INTERVENTIONS: dict[TrialEvalPromptConditionV1, str] = {
    "neutral": "",
    "targeted_covariate_structure": (
        "Before choosing the primary analysis, explicitly assess whether the declared baseline covariate and "
        "measurement structure changes the required adjustment or standardization. Distinguish precision "
        "adjustment from identification, and state the assumptions supporting the chosen analysis."
    ),
    "targeted_survival_assumptions": (
        "For any time-indexed outcome, explicitly assess whether the analysis assumptions remain defensible over "
        "follow-up and adapt the executed analysis if the evidence warrants it."
    ),
    "targeted_design_structure": (
        "Explicitly account for the study's assignment, dependence, timing, and monitoring structure when choosing "
        "and executing the primary analysis."
    ),
    "targeted_data_integrity": (
        "Explicitly inspect whether the participant-visible records support the requested analysis, resolve any "
        "material record-level ambiguity reproducibly, and report unresolved limitations."
    ),
    "placebo_deliberation": (
        "Before choosing the primary analysis, explicitly summarize the most decision-relevant evidence, explain "
        "the sequence of work you will perform, and report the basis for the final conclusion."
    ),
}

_PROCEDURE_COMPONENTS = (
    "define the scientific question, estimand, population, contrast, endpoint, horizon, and intercurrent-event handling",
    "inspect the participant-visible data, assignment mechanism, dependence structure, timing, monitoring, and missingness",
    "assess identification and the assumptions needed by candidate analyses",
    "execute a defensible primary analysis with uncertainty and relevant diagnostics or sensitivity analyses",
    "when the public task declares planning eligible, translate the result into the requested planning implication",
    "verify every submitted claim against executed code and cited participant evidence",
)
_UNORDERED_PROCEDURE_COMPONENTS = "; ".join(_PROCEDURE_COMPONENTS)
_ORDERED_PROCEDURE_COMPONENTS = "; ".join(
    f"({index}) {component}" for index, component in enumerate(_PROCEDURE_COMPONENTS, start=1)
)
_PROCEDURE_ASSISTANCE_BOUNDARY = (
    "These requirements do not prescribe an estimator; method selection must follow the protocol and supplied data."
)
_PROCEDURE_ASSISTANCE: dict[ProcedureAssistanceV1, str] = {
    "output_contract_only": "",
    "unordered_checklist": (
        f"A complete analysis must perform these operations in any order: {_UNORDERED_PROCEDURE_COMPONENTS}. "
        f"{_PROCEDURE_ASSISTANCE_BOUNDARY}"
    ),
    "ordered_sop": (
        f"Perform the same operations in this required order: {_ORDERED_PROCEDURE_COMPONENTS}. "
        f"{_PROCEDURE_ASSISTANCE_BOUNDARY}"
    ),
}

_SUBMISSION_OBLIGATIONS = (
    "State exactly one primary analysis, including the complete estimand (population, treatment conditions and "
    "contrast, endpoint, horizon, and intercurrent-event handling), executed estimator, result shape, numerical "
    "value, unit and orientation, uncertainty, participant-evidence links, executed supporting analyses, and "
    "limitations."
)

_ROUTE_DECLARATION_GUIDANCE = (
    "Consult `method_dictionary.json` and declare one complete `analysis_method_id`. Each record fixes the "
    "estimator, design modifiers, uncertainty procedure, effect scale, and result shape, so these properties are "
    "not independently recombined in the submission. Select the method supported by the protocol, study design, "
    "and supplied data. After choosing a method, consider only the diagnostics named by that method. For "
    "each one that is applicable and supported by the released evidence, follow the task-general operation and "
    "metric in `diagnostic_dictionary.json` and supply a typed evidence record with the input files used. The "
    "diagnostic dictionary defines the calculation but does not replace diagnostics computed from the study data."
)


def _scientific_submission_vocabulary_v1() -> tuple[str, tuple[str, ...]]:
    """Render scientific output tokens shared by both response interfaces."""

    schema = trialeval_submission_schema()
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise ValueError("TrialEval submission schema must define reusable components.")

    def enum_values(definition: str, *path: str) -> tuple[str, ...]:
        value: object = definitions.get(definition)
        for component in path:
            if isinstance(value, list) and component.isdigit():
                value = value[int(component)]
            elif isinstance(value, dict):
                value = value.get(component)
            else:
                raise ValueError(f"Submission schema path is not an object: {definition}.{'.'.join(path)}")
        if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
            raise ValueError(f"Submission schema path is not a non-empty string enum: {definition}.{'.'.join(path)}")
        return tuple(str(item) for item in value)

    effect_scales = enum_values("ScalarEstimateV1", "properties", "effect_scale", "enum")
    diagnostic_ids = enum_values("EvidenceRecordV1", "properties", "diagnostic_id", "anyOf", "0", "enum")
    text = " ".join(
        (
            "Canonical output vocabulary (format tokens, not method recommendations):",
            "analysis method [one exact identifier from method_dictionary.json];",
            f"effect scale [{', '.join(effect_scales)}];",
            f"diagnostic identifier [{', '.join(diagnostic_ids)}].",
        )
    )
    return text, tuple(
        sorted(
            {
                *effect_scales,
                *diagnostic_ids,
            }
        )
    )


_SHARED_SCIENTIFIC_VOCABULARY, _SHARED_SCIENTIFIC_VOCABULARY_TOKENS = _scientific_submission_vocabulary_v1()
_STRUCTURED_SUBMISSION = (
    "Submit one JSON object conforming to `interface/response_contract.json`. "
    "`interface/submission_shapes.json` contains equally labelled, global wire-shape examples: select only the "
    "result and evidence shapes supported by your analysis and replace every example identifier, method, value, "
    "source, and interpretation. The catalogue does not identify an eligible method or answer for this task. "
    f"{_SUBMISSION_OBLIGATIONS} "
    f"{_ROUTE_DECLARATION_GUIDANCE} "
    f"{_SHARED_SCIENTIFIC_VOCABULARY} Supporting evidence must describe analyses actually executed and cite "
    "participant-visible artifacts. Call submit_response with the object as its arguments, or write the exact object to "
    "`scratch/submission.json`, review it, and call submit_response_file with path `submission.json`; both routes "
    "enforce the same canonical schema."
)
_NARRATIVE_SUBMISSION = (
    f"Submit one complete primary analysis report as Markdown. {_SUBMISSION_OBLIGATIONS} "
    f"{_ROUTE_DECLARATION_GUIDANCE} "
    f"{_SHARED_SCIENTIFIC_VOCABULARY} Do not use a "
    "structured field list or JSON in the report. Call submit_response with the report, or write the exact Markdown "
    "to `scratch/report.md`, review it, and call submit_response_file with path `report.md`; both routes preserve "
    "the same narrative without interpretation."
)

_NARRATIVE_SUBMISSION_TOOL: dict[str, JsonValue] = {
    "type": "function",
    "function": {
        "name": "submit_response",
        "description": "Submit the completed response using the predeclared interface contract.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"content": {"type": "string", "minLength": 1}},
            "required": ["content"],
        },
    },
}
_FILE_SUBMISSION_TOOL: dict[str, JsonValue] = {
    "type": "function",
    "function": {
        "name": "submit_response_file",
        "description": (
            "Submit the completed response from a UTF-8 file under scratch/. "
            "The content is interpreted using the predeclared interface contract."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path under scratch/, e.g. submission.json or report.md",
                }
            },
            "required": ["path"],
        },
    },
}
_INTERFACES: tuple[TrialEvalSubmissionInterfaceV1, ...] = ("structured", "narrative")
_NARRATIVE_RESPONSE_CONTRACT: dict[str, JsonValue] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "TrialEval narrative response",
    "description": "One complete primary analysis report represented as Markdown text.",
    "type": "string",
    "minLength": 1,
    "contentMediaType": "text/markdown",
}
_RESPONSE_CONTRACT_RELATIVE_PATH = Path("interface") / "response_contract.json"
_STRUCTURED_SHAPES_RELATIVE_PATH = Path("interface") / "submission_shapes.json"


def prompt_intervention_v1(condition: TrialEvalPromptConditionV1) -> str:
    """Return the exact global intervention text for a prompt condition."""

    return _PROMPT_INTERVENTIONS[condition]


def procedure_assistance_v1(assistance: ProcedureAssistanceV1) -> str:
    """Return the exact method-neutral procedure-assistance text."""

    procedure_assistance_exposure_v1(
        suite="trialeval",
        procedure_assistance=assistance,
    )
    return _PROCEDURE_ASSISTANCE[assistance]


def submission_instruction_v1(interface: TrialEvalSubmissionInterfaceV1) -> str:
    """Return the global submission instruction for one response interface."""

    return _STRUCTURED_SUBMISSION if interface == "structured" else _NARRATIVE_SUBMISSION


def response_contract_v1(interface: TrialEvalSubmissionInterfaceV1) -> dict[str, JsonValue]:
    """Return the canonical response contract for one predeclared interface."""

    if interface == "structured":
        return trialeval_submission_schema()
    return dict(_NARRATIVE_RESPONSE_CONTRACT)


def response_contract_sha256_v1(interface: TrialEvalSubmissionInterfaceV1) -> str:
    """Hash the canonical response contract for one interface."""

    return canonical_payload_sha256(cast(JsonValue, response_contract_v1(interface)))


def stage_response_contract_v1(root: Path, interface: TrialEvalSubmissionInterfaceV1) -> Path:
    """Stage and verify the immutable response contract in one run workspace."""

    path = Path(root) / _RESPONSE_CONTRACT_RELATIVE_PATH
    expected = json.dumps(response_contract_v1(interface), indent=2, sort_keys=True) + "\n"
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("TrialEval interface contract path must not be a symlink.")
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise ValueError("Existing TrialEval interface contract differs from the declared response interface.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")

    shapes_path = Path(root) / _STRUCTURED_SHAPES_RELATIVE_PATH
    if interface == "structured":
        expected_shapes = (
            json.dumps(
                trialeval_submission_shape_catalogue().model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if shapes_path.is_symlink():
            raise ValueError("TrialEval structured shape-catalogue path must not be a symlink.")
        if shapes_path.exists():
            if not shapes_path.is_file() or shapes_path.read_text(encoding="utf-8") != expected_shapes:
                raise ValueError("Existing TrialEval shape catalogue differs from the canonical submission models.")
        else:
            shapes_path.write_text(expected_shapes, encoding="utf-8")
    elif shapes_path.exists():
        raise ValueError("Narrative TrialEval workspaces must not contain a structured shape catalogue.")
    return path


def submission_tools_v1(
    interface: TrialEvalSubmissionInterfaceV1,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Return interface-specific direct and common file submission transports."""

    if interface == "structured":
        direct: dict[str, JsonValue] = {
            "type": "function",
            "function": {
                "name": "submit_response",
                "description": "Submit one response conforming to the canonical TrialEval submission contract.",
                "parameters": response_contract_v1("structured"),
            },
        }
    else:
        direct = _NARRATIVE_SUBMISSION_TOOL
    return direct, _FILE_SUBMISSION_TOOL


def prompt_set_sha256_v1() -> str:
    """Hash all immutable prompt and interface templates."""

    return canonical_payload_sha256(
        cast(
            JsonValue,
            {
                "interventions": _PROMPT_INTERVENTIONS,
                "procedure_assistance": _PROCEDURE_ASSISTANCE,
                "procedure_assistance_exposure": {
                    assistance: procedure_assistance_exposure_v1(
                        suite="trialeval",
                        procedure_assistance=assistance,
                    ).model_dump(mode="json")
                    for assistance in _PROCEDURE_ASSISTANCE
                },
                "submission_instructions": {
                    "structured": _STRUCTURED_SUBMISSION,
                    "narrative": _NARRATIVE_SUBMISSION,
                },
                "response_contracts": {interface: response_contract_v1(interface) for interface in _INTERFACES},
                "structured_submission_shapes": trialeval_submission_shape_catalogue().model_dump(mode="json"),
                "submission_tools": {interface: submission_tools_v1(interface) for interface in _INTERFACES},
                "final_turn_submission_reminder": FINAL_TURN_SUBMISSION_REMINDER,
            },
        )
    )


def _schema_affordance(
    response_contract: Mapping[str, JsonValue],
    *,
    interface: TrialEvalSubmissionInterfaceV1,
) -> TrialEvalInterfaceAffordanceV1:
    field_paths: set[str] = set()
    enum_vocabulary: set[str] = set()

    def visit(value: JsonValue, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                for name, child in properties.items():
                    field_path = ".".join((*path, name))
                    field_paths.add(field_path)
                    visit(child, (*path, name))
            enum = value.get("enum")
            if isinstance(enum, list):
                enum_vocabulary.update(str(item) for item in enum)
            for key, child in value.items():
                if key not in {"properties", "enum"}:
                    visit(child, path)
        elif isinstance(value, list):
            for child in value:
                visit(child, path)

    visit(cast(JsonValue, response_contract), ())
    tools = submission_tools_v1(interface)
    return TrialEvalInterfaceAffordanceV1(
        submission_interface=interface,
        tool_schema_sha256=canonical_payload_sha256(cast(JsonValue, tools)),
        response_contract_sha256=response_contract_sha256_v1(interface),
        field_paths=tuple(sorted(field_paths)),
        enum_vocabulary=tuple(sorted(enum_vocabulary)),
    )


def schema_affordance_inventory_v1() -> TrialEvalSchemaAffordanceInventoryV1:
    """Return the exact field and vocabulary exposure of both interfaces."""

    return TrialEvalSchemaAffordanceInventoryV1(
        shared_scientific_vocabulary=_SHARED_SCIENTIFIC_VOCABULARY_TOKENS,
        interfaces=tuple(
            _schema_affordance(
                cast(Mapping[str, JsonValue], response_contract_v1(interface)),
                interface=interface,
            )
            for interface in _INTERFACES
        ),
    )


def condition_contrasts_markdown_v1() -> str:
    """Render exact human-readable assistance and interface contrasts."""

    affordances = {
        row.submission_interface: json.dumps(row.model_dump(mode="json"), indent=2, sort_keys=True)
        for row in schema_affordance_inventory_v1().interfaces
    }
    conditions = {
        "P0 output-contract-only": procedure_assistance_v1("output_contract_only"),
        "P1 unordered checklist": procedure_assistance_v1("unordered_checklist"),
        "P2 ordered SOP": procedure_assistance_v1("ordered_sop"),
        "structured instruction": submission_instruction_v1("structured"),
        "narrative instruction": submission_instruction_v1("narrative"),
        "structured affordance": affordances["structured"],
        "narrative affordance": affordances["narrative"],
    }
    comparisons = (
        ("P0 versus P1", "P0 output-contract-only", "P1 unordered checklist"),
        ("P1 versus P2", "P1 unordered checklist", "P2 ordered SOP"),
        ("Structured versus narrative instruction", "structured instruction", "narrative instruction"),
        (
            "Structured versus narrative interface affordance",
            "structured affordance",
            "narrative affordance",
        ),
    )
    lines = [
        "# TrialEvalBench condition contrasts",
        "",
        f"Canonical prompt-set SHA-256: `{prompt_set_sha256_v1()}`.",
        "",
        "Each block is generated from the exact participant-facing source text. "
        "A leading `-` denotes the left condition and `+` the right condition.",
    ]
    for title, left_name, right_name in comparisons:
        diff = difflib.unified_diff(
            conditions[left_name].splitlines(),
            conditions[right_name].splitlines(),
            fromfile=left_name,
            tofile=right_name,
            lineterm="",
        )
        lines.extend(("", f"## {title}", "", "```diff", *diff, "```"))
    return "\n".join(lines) + "\n"


__all__ = [
    "condition_contrasts_markdown_v1",
    "procedure_assistance_v1",
    "prompt_intervention_v1",
    "prompt_set_sha256_v1",
    "response_contract_sha256_v1",
    "response_contract_v1",
    "schema_affordance_inventory_v1",
    "stage_response_contract_v1",
    "submission_instruction_v1",
    "submission_tools_v1",
]
