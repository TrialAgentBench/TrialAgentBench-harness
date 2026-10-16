"""Canonical participant path and JSON projection shared by both replay consumers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Final, TypeAlias, cast

JsonScalar: TypeAlias = str | int | float | bool | None  # noqa: UP040
JsonArray: TypeAlias = list["JsonValue"]  # noqa: UP040
JsonObject: TypeAlias = Mapping[str, "JsonValue"]  # noqa: UP040
JsonValue: TypeAlias = JsonScalar | JsonArray | JsonObject  # noqa: UP040

_ANALYSIS_DATA_FILENAMES: Final = frozenset({"analysis.parquet", "monitoring_results.parquet"})


def canonical_trialeval_workspace_path(relative_path: str) -> str:
    """Project one released study artifact into its task-workspace path."""

    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or "\\" in relative_path or path.as_posix() != relative_path:
        raise ValueError("TrialEval material paths must be normalized relative POSIX paths")
    if "data" in path.parts:
        if path.name in _ANALYSIS_DATA_FILENAMES:
            return f"data/{path.name}"
        return f"data/raw/{path.name}"
    if "documents" in path.parts:
        return path.name
    raise ValueError(f"TrialEval task material has no data or document role: {relative_path!r}")


TRIALEVAL_AGENT_ANALYSIS_PLAN_REDACT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "target_method_id",
        "route_reference_id",
    }
)

TRIALEVAL_AGENT_ENDPOINT_DEFINITION_REDACT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "route_reference_id",
    }
)


def sanitize_trialeval_agent_task_payload(
    payload: JsonObject,
    *,
    available_item_members: frozenset[str] | None = None,
) -> JsonObject:
    """Return a participant-facing `task.json` payload."""

    sanitized: dict[str, JsonValue] = dict(payload)
    if available_item_members is not None:
        for key, value in tuple(sanitized.items()):
            if key.endswith("_file") and isinstance(value, str) and value not in available_item_members:
                sanitized.pop(key)
    return sanitized


def sanitize_trialeval_agent_analysis_plan_payload(payload: JsonObject) -> JsonObject:
    """Return a participant-facing `analysis_plan.json` payload."""

    sanitized: dict[str, JsonValue] = dict(payload)
    for key in TRIALEVAL_AGENT_ANALYSIS_PLAN_REDACT_KEYS:
        sanitized.pop(key, None)
    lane_rules = sanitized.get("lane_rules")
    if isinstance(lane_rules, list):
        sanitized["lane_rules"] = [
            (
                {key: value for key, value in rule.items() if key not in TRIALEVAL_AGENT_ANALYSIS_PLAN_REDACT_KEYS}
                if isinstance(rule, dict)
                else rule
            )
            for rule in lane_rules
        ]
    return sanitized


def sanitize_trialeval_agent_endpoint_definition_payload(payload: JsonObject) -> JsonObject:
    """Return a participant-facing endpoint-definition payload."""

    sanitized: dict[str, JsonValue] = dict(payload)
    for key in TRIALEVAL_AGENT_ENDPOINT_DEFINITION_REDACT_KEYS:
        sanitized.pop(key, None)
    return sanitized


def sanitize_trialeval_agent_json_payload(
    *,
    archive_name: str,
    payload: JsonObject,
    available_item_members: frozenset[str] | None = None,
) -> JsonObject:
    """Sanitize a TrialEval participant JSON payload by archive member name."""

    file_name = archive_name.rsplit("/", 1)[-1]
    if file_name == "task.json":
        return sanitize_trialeval_agent_task_payload(
            payload,
            available_item_members=available_item_members,
        )
    if file_name == "analysis_plan.json":
        return sanitize_trialeval_agent_analysis_plan_payload(payload)
    if file_name == "endpoint_definition.json":
        return sanitize_trialeval_agent_endpoint_definition_payload(payload)
    return payload


def sanitize_trialeval_agent_json_text(
    *,
    archive_name: str,
    text: str,
    available_item_members: frozenset[str] | None = None,
) -> str:
    """Return sanitized JSON text for a TrialEval participant archive member."""

    payload = json.loads(text)
    if not isinstance(payload, dict):
        return text
    sanitized = sanitize_trialeval_agent_json_payload(
        archive_name=archive_name,
        payload=cast(JsonObject, payload),
        available_item_members=available_item_members,
    )
    return json.dumps(sanitized, indent=2, sort_keys=True) + "\n"
