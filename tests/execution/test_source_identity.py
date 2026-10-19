"""Tests for the installed harness behavior identity."""

from pathlib import Path

from trialagentbench.io import canonical_payload_sha256, sha256_path
from trialagentbench.util.source_identity import harness_source_sha256_v1


def test_harness_source_identity_covers_every_installed_python_module() -> None:
    """Shared runtime moves must remain bound to persisted run identity."""

    source_root = Path(__file__).resolve().parents[2] / "trialagentbench"
    expected = canonical_payload_sha256(
        {
            path.relative_to(source_root).as_posix(): str(sha256_path(path))
            for path in sorted(source_root.rglob("*.py"))
        }
    )

    assert harness_source_sha256_v1() == expected
