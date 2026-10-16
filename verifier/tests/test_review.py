"""Consumer report provenance, failure handling and numerical counterexamples."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from trialagentbench_verifier.cli import main
from trialagentbench_verifier.contracts import CheckResultV1, VerificationFailureReportV1, VerificationReportV1
from trialagentbench_verifier.io import canonical_sha256
from trialagentbench_verifier.practice import verify_practice_suite
from trialagentbench_verifier.response import TrialDevRankingMeanV1, TrialDevRunRankingV1
from trialagentbench_verifier.review import write_review


def test_disclosed_review_preserves_values_scope_and_result_identity(tmp_path: Path) -> None:
    """The figure table preserves independent known answers without model claims."""
    report = verify_practice_suite()
    directory = tmp_path / "review"
    write_review(report, directory)
    payload = json.loads((directory / "report.json").read_text())
    assert payload == report.model_dump(mode="json")
    page = (directory / "index.html").read_text()
    assert canonical_sha256(payload) in page
    assert "not benchmark items or model results" in page
    with (directory / "loss.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    values = {row["label"]: float(row["mean_loss"]) for row in rows}
    assert values == pytest.approx({"accurate": 0.1, "wide": 2, "narrow_wrong": 18.6, "point_outside_interval": 0.25})
    assert all(row["worlds"] == row["attempts"] == row["interval_low"] == "" for row in rows)
    assert (directory / "loss.svg").is_file()
    repeated = tmp_path / "repeated"
    write_review(report, repeated)
    assert (repeated / "loss.svg").read_bytes() == (directory / "loss.svg").read_bytes()
    with pytest.raises(FileExistsError):
        write_review(report, directory)


def test_invalid_input_review_remains_a_failure(tmp_path: Path) -> None:
    """A missing evaluator cannot turn into a successful or empty score display."""
    directory = tmp_path / "review"
    assert (
        main(
            [
                "score",
                str(tmp_path / "absent"),
                "--submission",
                str(tmp_path / "absent.json"),
                "--output",
                str(tmp_path / "failure.json"),
                "--review-dir",
                str(directory),
            ]
        )
        == 1
    )
    page = (directory / "index.html").read_text()
    assert "Verification failed" in page
    assert "No successful verification or official score" in page
    assert not (directory / "loss.svg").exists()


def test_report_escapes_untrusted_error_text(tmp_path: Path) -> None:
    """Participant-controlled names cannot insert executable markup into review."""
    report = VerificationFailureReportV1(operation="score", detail='<script>alert("x")</script>')
    write_review(report, tmp_path / "review")
    page = (tmp_path / "review" / "index.html").read_text()
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_development_review_distinguishes_paired_conditions(tmp_path: Path) -> None:
    """A successful rehearsal must retain its status and each aggregation stratum."""
    digest = "1" * 64
    ranking = TrialDevRunRankingV1(
        release_id="disclosed-control",
        release_manifest_sha256=digest,
        invocation_checksum=digest,
        model_id="control",
        scope="scheduled_panel",
        items=(),
        profile_means=tuple(
            TrialDevRankingMeanV1(
                stream="evidence_analysis",
                data_view=view,
                asset_count=count,
                clinical_profile_id="same_clinical",
                scenario_profile_id="same_scenario",
                metric="observational_loss",
                scored_tasks=1,
                scheduled_tasks=1,
                scheduled_worlds=1,
                evaluator_errors=0,
                unscored_task_counts={},
                target_status_counts={"reported": 1},
                mean=0.5,
            )
            for view, count in (("raw", 3), ("analysis_ready", 5))
        ),
    )
    report = VerificationReportV1(
        input_kind="archive",
        input_sha256=digest,
        release_digest=digest,
        candidate_digest=digest,
        benchmark_stream="trialdev",
        operation="analyse-trialdev",
        status="pass",
        scheduled_check_count=1,
        passed_check_count=1,
        failed_check_count=0,
        checks=(
            CheckResultV1(
                check_id="ranking",
                scope_id="control",
                status="pass",
                observed=ranking.model_dump(mode="json"),
                expected=None,
            ),
        ),
        qualification_report_digests=(),
        development=True,
        standard_component_inventory_digest=digest,
    )
    directory = tmp_path / "review"
    write_review(report, directory, evidence_kind="local_control")
    page = (directory / "index.html").read_text()
    assert "Unqualified rehearsal" in page
    assert "do not establish benchmark qualification" in page
    with (directory / "checks.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["initial_targets_scored/scheduled"] == "1/1" for row in rows)
    assert "observed_mean" in rows[0]
    assert [(row["stream"], row["data_view"], row["asset_count"]) for row in rows] == [
        ("evidence_analysis", "raw", "3"),
        ("evidence_analysis", "analysis_ready", "5"),
    ]
