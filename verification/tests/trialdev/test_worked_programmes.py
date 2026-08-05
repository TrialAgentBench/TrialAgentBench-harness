"""Adversarial tests for independent TrialDev worked-programme reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

from trialagentbench_harness.trialdev.worked_programmes import (
    build_trialdev_worked_programmes_v1,
)

from trialagentbench_validation import cli
from trialagentbench_validation.trialdev.worked_programmes import (
    audit_trialdev_worked_programmes,
)


def _build(tmp_path: Path) -> Path:
    root = tmp_path / "worked"
    build_trialdev_worked_programmes_v1(output_dir=root, source_identity="d" * 64)
    return root


def test_independent_verifier_reconstructs_both_complete_programmes(
    tmp_path: Path,
) -> None:
    report = audit_trialdev_worked_programmes(package_root=_build(tmp_path))

    assert report.status == "pass"
    assert report.programme_count == 2
    assert report.checkpoint_count == 8
    assert report.reconstructed_supported_set_count == 8
    assert report.evidence_artifact_count == 17
    assert report.terminal_dispositions == {
        "single_asset_development": "success",
        "bounded_portfolio_reallocation": "success",
    }


def test_independent_verifier_rejects_numeric_evidence_drift(tmp_path: Path) -> None:
    root = _build(tmp_path)
    path = next((root / "evidence").rglob("*-safety.csv"))
    path.write_text(
        path.read_text(encoding="utf-8").replace("0.200000", "0.210000"),
        encoding="utf-8",
    )

    report = audit_trialdev_worked_programmes(package_root=root)

    assert report.status == "fail"
    assert any(
        item.startswith("evidence_artifact_checksum_disagreement:")
        for item in report.findings
    )
    assert any(
        item.startswith("supported_set_reconstruction_failed:")
        for item in report.findings
    )


def test_independent_verifier_rejects_supported_action_drift(tmp_path: Path) -> None:
    root = _build(tmp_path)
    path = root / "worked_programmes.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    step = payload["programmes"][0]["checkpoints"][0]
    step["supported_action_set"]["supported_actions"] = step["supported_action_set"][
        "supported_actions"
    ][:1]
    step["supported_action_set"]["checksum"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_trialdev_worked_programmes(package_root=root)

    assert report.status == "fail"
    assert any(
        item.startswith("record_checksum_disagreement:") for item in report.findings
    )
    assert any(
        item.startswith("supported_action_reconstruction_disagreement:")
        for item in report.findings
    )


def test_independent_verifier_rejects_reversed_rule_direction(tmp_path: Path) -> None:
    root = _build(tmp_path)
    path = root / "worked_programmes.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    step = payload["programmes"][0]["checkpoints"][1]
    step["decision_evidence"]["rules"][0]["direction"] = "minimum"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_trialdev_worked_programmes(package_root=root)

    assert report.status == "fail"
    assert any(
        item.startswith("supported_set_reconstruction_failed:")
        for item in report.findings
    )


def test_worked_programme_cli_writes_machine_readable_report(tmp_path: Path) -> None:
    root = _build(tmp_path)
    output = tmp_path / "report.json"

    assert (
        cli.main(
            [
                "trialdev-worked-programmes",
                "--package-root",
                str(root),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
