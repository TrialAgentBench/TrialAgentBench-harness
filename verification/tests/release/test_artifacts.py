"""External evidence artifact tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from trialagentbench_validation.external.release.artifacts import (
    verify_external_artifact_manifest,
    write_external_artifact_manifest,
)


def test_external_artifact_manifest_verifies_exact_directory(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text('{"status":"pass"}\n', encoding="utf-8")
    write_external_artifact_manifest(tmp_path)

    verified = verify_external_artifact_manifest(tmp_path)

    assert len(verified.artifacts) == 1
    (tmp_path / "unexpected.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="membership differs"):
        verify_external_artifact_manifest(tmp_path)
