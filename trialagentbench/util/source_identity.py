"""Stable source identities for benchmark execution and scoring behavior."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

from trialagentbench.io import canonical_payload_sha256, sha256_path


def source_file_sha256_v1(function: Callable[..., object]) -> str:
    """Hash the source file containing one callable."""

    source = inspect.getsourcefile(function)
    if source is None:
        raise RuntimeError(f"Cannot resolve source identity for {function!r}.")
    return str(sha256_path(Path(source)))


def harness_source_sha256_v1() -> str:
    """Hash every installed Python module that can affect harness behavior."""

    source_root = Path(__file__).resolve().parents[1]
    return canonical_payload_sha256(
        {
            path.relative_to(source_root).as_posix(): str(sha256_path(path))
            for path in sorted(source_root.rglob("*.py"))
        }
    )


__all__ = ["harness_source_sha256_v1", "source_file_sha256_v1"]
