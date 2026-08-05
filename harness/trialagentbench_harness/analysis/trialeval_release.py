"""Shared readers for the typed TrialEval release ZIP surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar
from zipfile import ZipFile

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def read_json_object_member(zf: ZipFile, member: str) -> dict[str, object]:
    """Read one JSON object from a release ZIP member."""

    payload = json.loads(zf.read(member))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {member}")
    return {str(key): value for key, value in payload.items()}


def load_evaluator_item_index(evaluator_zip: Path) -> tuple[dict[str, object], ...]:
    """Load the canonical evaluator item index."""

    with ZipFile(evaluator_zip) as zf:
        payload = read_json_object_member(zf, "grader/item_index.json")
    entries = payload.get("entries")
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("grader/item_index.json must contain an entries list of objects")
    return tuple({str(key): value for key, value in entry.items()} for entry in entries)


def load_evaluator_jsonl_models(*, evaluator_zip: Path, member: str, model: type[ModelT]) -> tuple[ModelT, ...]:
    """Load and validate one canonical evaluator JSONL domain."""

    with ZipFile(evaluator_zip) as zf:
        try:
            lines = zf.read(member).decode("utf-8").splitlines()
        except KeyError as exc:
            raise FileNotFoundError(f"Missing evaluator domain: {member}") from exc
    rows: list[ModelT] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Evaluator JSONL row must contain an object: {member}:{line_number}")
        rows.append(model.model_validate(payload))
    if not rows:
        raise ValueError(f"Evaluator JSONL domain is empty: {member}")
    return tuple(rows)


def load_evaluator_json_model(*, evaluator_zip: Path, member: str, model: type[ModelT]) -> ModelT:
    """Load and validate one canonical evaluator JSON domain."""

    with ZipFile(evaluator_zip) as zf:
        payload = read_json_object_member(zf, member)
    return model.model_validate(payload)


__all__ = [
    "load_evaluator_item_index",
    "load_evaluator_json_model",
    "load_evaluator_jsonl_models",
    "read_json_object_member",
]
