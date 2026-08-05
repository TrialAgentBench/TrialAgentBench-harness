"""I/O helpers for the TrialDev grader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["read_json", "write_json"]


def read_json(path: Path) -> Any:
    """Read JSON from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> Path:
    """Write deterministic JSON to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, sort_keys=True, indent=2).rstrip() + "\n", encoding="utf-8")
    return p
