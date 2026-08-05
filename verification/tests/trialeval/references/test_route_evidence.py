"""Tests for compact public-route replay evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from trialagentbench_validation.contracts.route_replay import (
    compact_public_route_replay,
)
from trialagentbench_validation.trialeval.references.numeric import (
    PublicEvidenceNumericReferenceCheckV1,
    PublicEvidenceNumericReferenceReportV1,
)


def _report(*, outcome: str = "matched") -> PublicEvidenceNumericReferenceReportV1:
    check = PublicEvidenceNumericReferenceCheckV1.model_construct(
        route_reference_id="reference-1",
        route_reference_checksum="a" * 64,
        input_bundle_id="bundle-1",
        input_bundle_checksum="b" * 64,
        outcome=outcome,
        abs_diff=2e-12,
        lower_abs_diff=None,
        upper_abs_diff=None,
        standard_error_abs_diff=1e-12,
        vector_max_abs_diff=None,
    )
    return PublicEvidenceNumericReferenceReportV1.model_construct(
        status="pass",
        findings=(),
        checks=(check,),
    )


def test_compact_replay_preserves_measured_maximum_and_archive_identity(
    tmp_path: Path,
) -> None:
    evaluator = tmp_path / "evaluator.zip"
    participant = tmp_path / "participant.zip"
    evaluator.write_bytes(b"evaluator")
    participant.write_bytes(b"participant")

    evidence = compact_public_route_replay(
        report=_report(),
        evaluator_zip=evaluator,
        participant_zip=participant,
    )

    assert evidence.records[0].max_abs_difference == pytest.approx(2e-12)
    assert evidence.checksum is not None
    assert evidence.evaluator_sha256 != evidence.participant_sha256


def test_compact_replay_rejects_nonmatching_route(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluator.zip"
    participant = tmp_path / "participant.zip"
    evaluator.write_bytes(b"evaluator")
    participant.write_bytes(b"participant")

    with pytest.raises(ValueError, match="was not reproduced"):
        compact_public_route_replay(
            report=_report(outcome="mismatched"),
            evaluator_zip=evaluator,
            participant_zip=participant,
        )
