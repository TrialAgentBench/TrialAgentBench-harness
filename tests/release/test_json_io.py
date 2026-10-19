"""Tests for strict and streaming public JSON input."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from trialagentbench.io import read_json, read_jsonl_models, write_json


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float


@pytest.mark.parametrize(
    "body",
    ('{"value": 1, "value": 2}', '{"value": NaN}'),
)
def test_read_json_rejects_nonportable_values(body: str, tmp_path: Path) -> None:
    """Reject duplicate keys and non-finite numbers at the shared input boundary."""

    path = tmp_path / "invalid.json"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError):
        read_json(path)


def test_read_jsonl_models_is_lazy_and_reports_the_invalid_line(tmp_path: Path) -> None:
    """Validate JSONL one record at a time without reading past the consumer."""

    path = tmp_path / "records.jsonl"
    path.write_text('{"value": 1}\n{"value": 2, "value": 3}\n', encoding="utf-8")
    records = read_jsonl_models(_Record, path)

    assert next(records).value == 1.0
    with pytest.raises(ValueError, match=r"records\.jsonl:2"):
        next(records)


def test_write_json_rejects_nonfinite_numbers_without_replacing_the_target(tmp_path: Path) -> None:
    """Keep the previous artifact if canonical numerical serialization fails."""

    path = tmp_path / "result.json"
    path.write_text('{"value": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON compliant"):
        write_json(path, {"value": float("nan")})

    assert path.read_text(encoding="utf-8") == '{"value": 1}\n'
