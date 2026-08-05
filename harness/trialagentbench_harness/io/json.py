"""Strict JSON IO helpers (UTF-8, fail-fast, schema-validated)."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

TModel = TypeVar("TModel", bound=BaseModel)


def read_json(path: Path) -> Any:
    """Read JSON from `path` with UTF-8 encoding.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_model(model: type[TModel], path: Path) -> TModel:  # noqa: UP047
    """Read JSON from `path` and validate against `model`.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    pydantic.ValidationError
        If the JSON payload does not validate against `model`.
    """
    payload = read_json(path)
    try:
        return model.model_validate(payload)
    except ValidationError:
        # Preserve pydantic error message; do not wrap into a generic exception.
        raise


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Atomically write stable UTF-8 JSON to `path`."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    text = json.dumps(payload, indent=indent, ensure_ascii=True, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_model(path: Path, model: BaseModel, *, indent: int = 2) -> None:
    """Write a Pydantic model to JSON with stable formatting."""
    write_json(path, model.model_dump(mode="json"), indent=indent)


def append_jsonl_model(path: Path, model: BaseModel) -> None:
    """Append one validated model as a durable JSONL record."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(model.model_dump(mode="json"), ensure_ascii=True, sort_keys=True) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
