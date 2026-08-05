"""Deterministic hashing utilities for the TrialDev grader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["compute_sha256_hex", "sha256_file_hex"]


def compute_sha256_hex(payload: Any) -> str:
    """Compute a deterministic SHA256 hex digest for a JSON-compatible payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_file_hex(path: Path) -> str:
    """Compute the SHA256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
