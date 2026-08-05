"""CSV IO helpers for schema-bearing harness exports."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def write_csv_rows(path: Path, rows: Sequence[dict[str, object]], field_order: Sequence[str]) -> None:
    """Write mapping rows to CSV using an explicit deterministic field order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(field_order), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in field_order})


def write_csv_models(path: Path, rows: Sequence[BaseModel], *, field_order: list[str] | None = None) -> None:
    """Write a list of Pydantic models to a CSV file deterministically.

    Parameters
    ----------
    path:
        Output CSV path.
    rows:
        Rows to write. Each row must be a Pydantic model instance.
    field_order:
        Optional explicit column order. If omitted, uses the first row's
        model field order.
    """
    if not rows and field_order is None:
        path.write_text("", encoding="utf-8")
        return
    fields = field_order or list(rows[0].__class__.model_fields.keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            payload = r.model_dump(mode="json")
            out: dict[str, Any] = {}
            for k in fields:
                v = payload.get(k)
                if isinstance(v, (list, dict)):
                    out[k] = json.dumps(v, sort_keys=True)
                else:
                    out[k] = v
            writer.writerow(out)
