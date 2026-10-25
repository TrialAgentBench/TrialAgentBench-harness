"""Stable source identities for benchmark execution and scoring behavior."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

from trialagentbench_verifier.scorer import SCORER_HARNESS_SOURCE_SHA256

from trialagentbench.io import canonical_payload_sha256, sha256_path


def source_file_sha256_v1(function: Callable[..., object]) -> str:
    """Hash the source file containing one callable."""

    source = inspect.getsourcefile(function)
    if source is None:
        raise RuntimeError(f"Cannot resolve source identity for {function!r}.")
    return str(sha256_path(Path(source)))


def _source_tree_sha256(*, include_resources: bool) -> str:
    source_root = Path(__file__).resolve().parents[1]
    return canonical_payload_sha256(
        {
            path.relative_to(source_root).as_posix(): str(sha256_path(path))
            for path in sorted(source_root.rglob("*.py"))
            if include_resources or path.relative_to(source_root).parts[0] != "resources"
        }
    )


def harness_source_sha256_v1() -> str:
    """Hash every installed Python module that can affect harness behavior."""
    return _source_tree_sha256(include_resources=True)


def harness_scoring_source_sha256_v1() -> str:
    """Hash harness modules consistently across source and wheel installations.

    Returns
    -------
    str
        Source digest excluding bundled example resources. This declaration is
        carried by the standalone verifier without a dependency on the harness.
    """
    return _source_tree_sha256(include_resources=False)


def require_harness_scorer_v1() -> None:
    """Refuse grading with harness sources outside the declared scorer.

    Raises
    ------
    RuntimeError
        If the installed harness and verifier sources do not belong together.
    """
    if harness_scoring_source_sha256_v1() != SCORER_HARNESS_SOURCE_SHA256:
        raise RuntimeError(
            "Harness sources differ from the declared scorer; install matching harness and verifier builds"
        )


__all__ = [
    "harness_source_sha256_v1",
    "harness_scoring_source_sha256_v1",
    "require_harness_scorer_v1",
    "source_file_sha256_v1",
]
