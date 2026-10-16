"""Validation boundary for untrusted model tool calls."""

from __future__ import annotations

import json
from typing import cast

from pydantic import BaseModel

from trialagentbench.ports.llm_provider import JsonValue


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


def provider_tool(name: str, description: str, arguments: type[BaseModel]) -> dict[str, JsonValue]:
    """Declare a function tool from its argument model in a form every provider accepts.

    References are inlined, every schema node states its ``type``, a nullable field is a type list
    such as ``["integer", "null"]`` and generated titles are dropped.

    Raises
    ------
    ValueError
        If the model needs a construct outside that form, such as a union of two non-null types.
    """
    schema = arguments.model_json_schema()
    definitions = cast(dict[str, dict[str, JsonValue]], schema.pop("$defs", {}))
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": _typed(schema, definitions)},
    }


def _typed(schema: dict[str, JsonValue], definitions: dict[str, dict[str, JsonValue]]) -> dict[str, JsonValue]:
    """Return one schema node with references inlined and an explicit type."""
    if "$ref" in schema:
        reference = str(schema["$ref"]).removeprefix("#/$defs/")
        schema = {**definitions[reference], **{key: value for key, value in schema.items() if key != "$ref"}}
    if "anyOf" in schema:
        options = cast(list[dict[str, JsonValue]], schema["anyOf"])
        present = [option for option in options if option != {"type": "null"}]
        if len(options) != 2 or len(present) != 1:
            raise ValueError("provider tool schemas allow one type, optionally nullable")
        option = _typed(present[0], definitions)
        if not isinstance(option["type"], str):
            raise ValueError("provider tool schemas allow one type, optionally nullable")
        outer = {key: value for key, value in schema.items() if key != "anyOf"}
        schema = {**option, **outer, "type": [option["type"], "null"]}
    typed: dict[str, JsonValue] = {}
    for key, value in schema.items():
        if key == "title" and isinstance(value, str):
            continue
        if key == "properties":
            properties = cast(dict[str, dict[str, JsonValue]], value)
            typed[key] = {name: _typed(child, definitions) for name, child in properties.items()}
        elif key == "items":
            typed[key] = _typed(cast(dict[str, JsonValue], value), definitions)
        else:
            typed[key] = value
    if "type" not in typed or (typed["type"] == "array" and "items" not in typed):
        raise ValueError(f"provider tool schema node lacks an explicit type: {sorted(typed)}")
    return typed


__all__ = [
    "JsonObjectDecodeError",
    "ToolInputError",
    "decode_json_object_text",
    "parse_json_object_text",
    "parse_tool_arguments",
    "provider_tool",
]
