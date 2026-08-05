"""CLI tests for independent TrialEval replay."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from trialeval.release_fixture import write_scoreable_reference_fixture_zip

from trialagentbench_validation import cli
from trialagentbench_validation.trialeval.references import drift, numeric


def test_trialeval_replay_command_combines_independent_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = write_scoreable_reference_fixture_zip(
        tmp_path, include_reference_input=True
    )
    participant = tmp_path / "participant.zip"
    with ZipFile(participant, "w") as archive:
        archive.writestr("items/TASK0001/data/analysis_frame.parquet", b"fixture table")
        archive.writestr("items/TASK0001/task.json", "{}")
        archive.writestr("items/TASK0001/protocol_summary.json", "{}")
    monkeypatch.setattr(
        numeric,
        "write_public_evidence_numeric_reference_artifacts_v1",
        lambda **_: SimpleNamespace(status="pass"),
    )
    monkeypatch.setattr(
        drift,
        "write_public_evidence_reference_drift_validation_artifacts_v1",
        lambda **_: SimpleNamespace(status="pass"),
    )

    exit_code = cli.main(
        [
            "trialeval-replay",
            "--evaluator",
            str(evaluator),
            "--participant",
            str(participant),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "out" / "public_evidence_reference_replay_report.json").is_file()
