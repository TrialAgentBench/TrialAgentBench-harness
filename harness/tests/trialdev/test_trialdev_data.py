"""TrialDev release item-discovery contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trialagentbench_harness.trialdev.data import SUITE_MANIFEST_NAME, discover_items


def _item(*, item_id: str) -> dict[str, object]:
    return {
        "item_id": item_id,
        "scenario_id": "s01",
        "phase_id": "observational_review",
        "objective_id": "benefit_risk",
        "task_definition_id": "observational_review__benefit_risk__none",
    }


def _write_manifest(root: Path, name: str, items: list[dict[str, object]]) -> None:
    (root / name).write_text(
        json.dumps({"version": "v1", "suite_id": "test", "release_root": ".", "items": items}),
        encoding="utf-8",
    )


def test_discover_items_requires_an_explicit_release_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=SUITE_MANIFEST_NAME):
        discover_items(tmp_path)


def test_discover_items_rejects_duplicate_ids(tmp_path: Path) -> None:
    _write_manifest(tmp_path, SUITE_MANIFEST_NAME, [_item(item_id="item_1"), _item(item_id="item_1")])
    with pytest.raises(ValueError, match="duplicate item ids"):
        discover_items(tmp_path)


def test_discover_items_validates_item_contracts(tmp_path: Path) -> None:
    malformed = _item(item_id="item_1")
    malformed["unexpected"] = True
    _write_manifest(tmp_path, SUITE_MANIFEST_NAME, [malformed])
    with pytest.raises(ValueError, match="unexpected"):
        discover_items(tmp_path)
