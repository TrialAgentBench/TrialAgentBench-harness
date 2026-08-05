"""Tests for independent TrialDev policy-value reconstruction."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from trialagentbench_validation.trialdev.policy_value_audit import (
    audit_trialdev_policy_value_v1,
)

_POLICY_VALUE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "validation_results"
    / "trialdev_v1"
    / "inputs"
    / "policy_value"
)


def test_policy_value_audit_reconstructs_every_world_and_cell() -> None:
    report = audit_trialdev_policy_value_v1(policy_value_root=_POLICY_VALUE_ROOT)

    assert report.status == "pass"
    assert report.world_count == 7_200
    assert report.candidate_record_count == 21_600
    assert report.cell_count == 18
    assert report.reconstructed_numeric_count > 72_000


def test_policy_value_audit_detects_numeric_tampering(tmp_path: Path) -> None:
    root = tmp_path / "policy_value"
    shutil.copytree(_POLICY_VALUE_ROOT, root)
    path = root / "policy_value_worlds.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
        fields = tuple(rows[0])
    rows[0]["oracle_terminal_success_probability"] = "0.0"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = audit_trialdev_policy_value_v1(policy_value_root=root)

    assert report.status == "fail"
    assert any(
        value.startswith("numeric_disagreement:oracle_terminal_success_probability:")
        for value in report.findings
    )
