"""Tests for public TrialDev worked-programme generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from trialagentbench_harness.cli import main
from trialagentbench_harness.contracts.trialdev.worked_programmes import TrialDevWorkedPackageV1
from trialagentbench_harness.io.checksums import sha256_file
from trialagentbench_harness.io.json import read_json_model
from trialagentbench_harness.trialdev.worked_programmes import build_trialdev_worked_programmes_v1


def test_worked_programmes_are_complete_reproducible_and_evidence_bound(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = build_trialdev_worked_programmes_v1(output_dir=first_root, source_identity="d" * 64)
    second = build_trialdev_worked_programmes_v1(output_dir=second_root, source_identity="d" * 64)

    assert first == second
    assert {item.stream_id for item in first.programmes} == {
        "single_asset_development",
        "bounded_portfolio_reallocation",
    }
    portfolio = next(item for item in first.programmes if item.stream_id == "bounded_portfolio_reallocation")
    assert any(
        step.selected_action.action_id == "promote_reserve_to_proof_of_concept" for step in portfolio.checkpoints
    )
    for programme in first.programmes:
        assert programme.checkpoints[-1].state_after.terminal_disposition == "success"
        for step in programme.checkpoints:
            assert step.selection_accepted
            for evidence in step.state_before.evidence:
                path = first_root / evidence.relative_path
                assert path.is_file()
                assert sha256_file(path) == evidence.artifact_sha256
    assert sha256_file(first_root / "worked_programmes.json") == sha256_file(second_root / "worked_programmes.json")
    assert sha256_file(first_root / "state_action_graph.json") == sha256_file(second_root / "state_action_graph.json")


def test_worked_programme_public_cli_and_source_validation(tmp_path: Path) -> None:
    output = tmp_path / "worked"
    assert (
        main(
            [
                "export",
                "trialdev-worked-programmes",
                "--output",
                str(output),
                "--source-identity",
                "e" * 64,
            ]
        )
        == 0
    )
    package = read_json_model(TrialDevWorkedPackageV1, output / "worked_programmes.json")
    assert len(package.programmes) == 2

    with pytest.raises(ValueError, match="SHA-256"):
        build_trialdev_worked_programmes_v1(output_dir=tmp_path / "bad", source_identity="not-a-hash")
