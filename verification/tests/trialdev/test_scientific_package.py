"""Integration tests for the self-contained TrialDev scientific package."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from trialagentbench_validation import cli
from trialagentbench_validation.trialdev.scientific_package import (
    verify_trialdev_scientific_package,
)

_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2] / "validation_results" / "trialdev_v1"
)


def test_scientific_package_is_complete_and_self_contained() -> None:
    report = verify_trialdev_scientific_package(package_root=_PACKAGE_ROOT)

    assert report.status == "pass"
    assert report.artifact_count == 97
    assert len(tuple((_PACKAGE_ROOT / "figures").glob("*.svg"))) == 10
    assert len(tuple((_PACKAGE_ROOT / "figures").glob("*.pdf"))) == 10
    assert len(tuple((_PACKAGE_ROOT / "figures" / "source_data").glob("*.csv"))) == 10
    assert (_PACKAGE_ROOT / "results" / "portfolio_routes.csv").is_file()
    assert (
        _PACKAGE_ROOT / "inputs" / "worked_programmes" / "worked_programmes.json"
    ).is_file()
    assert (
        _PACKAGE_ROOT / "inputs" / "operating_characteristics" / "world_results.csv"
    ).is_file()


def test_policy_value_summary_matches_the_world_level_evidence() -> None:
    with (_PACKAGE_ROOT / "inputs" / "policy_value" / "policy_value_worlds.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 7_200
    fields = (
        "best_supported_terminal_success_probability",
        "adjusted_point_terminal_success_probability",
        "alphabetical_terminal_success_probability",
        "best_supported_expected_resource_units",
        "adjusted_point_expected_resource_units",
        "alphabetical_expected_resource_units",
    )
    means = {
        field: sum(float(row[field]) for row in rows) / len(rows) for field in fields
    }
    report = (_PACKAGE_ROOT / "REPORT.md").read_text(encoding="utf-8")

    assert "known programme probabilities" in report
    assert "same 8- or 10-unit resource budgets" in report
    for value in means.values():
        assert f"{value:.3f}" in report


def test_scientific_package_verifier_rejects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "package"
    shutil.copytree(_PACKAGE_ROOT, output)
    target = output / "results" / "capability_metrics.csv"
    target.write_text(
        target.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
    )

    report = verify_trialdev_scientific_package(package_root=output)

    assert report.status == "fail"
    assert (
        "artifact_identity_disagreement:results/capability_metrics.csv"
        in report.findings
    )


def test_scientific_package_public_cli_verifies(tmp_path: Path) -> None:
    report = tmp_path / "verification.json"
    assert (
        cli.main(
            [
                "trialdev-scientific-package-verify",
                "--package-root",
                str(_PACKAGE_ROOT),
                "--output",
                str(report),
            ]
        )
        == 0
    )
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "pass"


def test_scientific_package_cli_rejects_a_receipt_inside_the_package(
    tmp_path: Path,
) -> None:
    output = tmp_path / "package"
    shutil.copytree(_PACKAGE_ROOT, output)

    with pytest.raises(ValueError, match="outside the package root"):
        cli.main(
            [
                "trialdev-scientific-package-verify",
                "--package-root",
                str(output),
                "--output",
                str(output / "verification.json"),
            ]
        )
