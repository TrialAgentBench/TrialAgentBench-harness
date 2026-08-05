"""Validation boundary for untrusted model tool calls."""

from __future__ import annotations

import json


class ToolInputError(ValueError):
    """A model-supplied tool call that can be corrected within the run."""


class JsonObjectDecodeError(ValueError):
    """A stable failure while decoding an untrusted JSON object."""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def parse_tool_arguments(arguments: str, *, tool_name: str) -> dict[str, object]:
    """Parse one model tool payload as a JSON object."""

    return parse_json_object_text(arguments, label=f"Tool {tool_name!r} arguments") if arguments else {}


def parse_json_object_text(text: str, *, label: str) -> dict[str, object]:
    """Parse JSON object text while rejecting duplicate field names."""

    try:
        return decode_json_object_text(text)
    except JsonObjectDecodeError as exc:
        raise ToolInputError(f"{label} {exc}") from exc


def decode_json_object_text(text: str) -> dict[str, object]:
    """Decode one JSON object with stable fail-fast error categories."""

    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as exc:
        raise JsonObjectDecodeError(code="json_syntax", message="must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise JsonObjectDecodeError(code="json_object_required", message="must be a JSON object.")
    return payload


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise JsonObjectDecodeError(
                code="duplicate_json_field",
                message=f"Duplicate JSON field: {key!r}.",
            )
        payload[key] = value
    return payload


__all__ = [
    "JsonObjectDecodeError",
    "ToolInputError",
    "decode_json_object_text",
    "parse_json_object_text",
    "parse_tool_arguments",
]
