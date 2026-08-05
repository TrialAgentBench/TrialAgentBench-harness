"""Installed-wheel replay receipt tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trialagentbench_validation.external.recovery import (
    clean_replay as qualification_clean_replay,
)
from trialagentbench_validation.external.recovery.clean_replay import (
    record_clean_replay,
)


def test_clean_replay_records_stable_module_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        "design_sha256": "a" * 64,
        "receipt_sha256": "b" * 64,
        "value": 1.0,
    }
    reference = tmp_path / "reference.json"
    replay = tmp_path / "replay.json"
    reference.write_text(json.dumps(report), encoding="utf-8")
    replay.write_text(json.dumps(report), encoding="utf-8")
    wheel = tmp_path / "validation.whl"
    wheel.write_bytes(b"wheel")

    def clean_runtime(
        _clean_python: Path,
        *,
        repository_root: Path,
    ) -> dict[str, object]:
        assert repository_root == tmp_path / "repository"
        return {
            "python": "3.12.3",
            "validation_module_path": (
                "/isolated/lib/site-packages/" "trialagentbench_validation/__init__.py"
            ),
            "repository_modules_imported": [],
            "scientific_dependencies": {},
        }

    monkeypatch.setattr(
        qualification_clean_replay,
        "_clean_runtime",
        clean_runtime,
    )

    result = record_clean_replay(
        reference_report=reference,
        replay_report=replay,
        wheel=wheel,
        clean_python=tmp_path / "python",
        repository_root=tmp_path / "repository",
        output=tmp_path / "receipt.json",
    )

    assert (
        result.isolation.validation_module_path
        == "trialagentbench_validation/__init__.py"
    )
