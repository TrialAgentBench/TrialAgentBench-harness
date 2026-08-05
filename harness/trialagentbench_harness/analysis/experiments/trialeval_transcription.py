"""Source-grounded narrative transcription and representation checks."""

from __future__ import annotations

import hashlib
from typing import Literal

from trialagentbench_harness.contracts.experiments import (
    TrialEvalNarrativeTranscriptionV1,
    TrialEvalRepresentationFidelityRowV1,
    TrialEvalRepresentationFixtureV1,
    submission_from_narrative_claims_v1,
)


def validate_narrative_transcription_v1(
    *,
    transcription: TrialEvalNarrativeTranscriptionV1,
    frozen_report: str,
    expected_assignment_id: str,
    expected_task_id: str,
) -> None:
    """Validate that a transcription is bound to and supported by a frozen report."""

    if transcription.assignment_id != expected_assignment_id:
        raise ValueError("Transcription assignment_id does not match the frozen response.")
    report_hash = hashlib.sha256(frozen_report.encode("utf-8")).hexdigest()
    if transcription.report_sha256 != report_hash:
        raise ValueError("Transcription report_sha256 does not match the frozen narrative response.")
    for claim in transcription.claims:
        for span in claim.spans:
            if span.end > len(frozen_report) or frozen_report[span.start : span.end] != span.text:
                raise ValueError(f"Transcription source span for {claim.field_path} does not match the frozen report.")
    if transcription.status == "abstain":
        return
    if transcription.submission is None:
        raise ValueError("Complete transcription is missing its canonical submission.")
    if transcription.submission.task_id != expected_task_id:
        raise ValueError("Transcribed submission task_id does not match the assigned participant task.")


def evaluate_representation_fixtures_v1(
    fixtures: tuple[TrialEvalRepresentationFixtureV1, ...],
) -> tuple[TrialEvalRepresentationFidelityRowV1, ...]:
    """Verify fixed-answer parity and report optional importer fidelity."""

    if not fixtures:
        raise ValueError("Representation-fidelity analysis requires at least one fixture.")
    fixture_ids = [fixture.fixture_id for fixture in fixtures]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("Representation fixture IDs must be unique.")
    rows: list[TrialEvalRepresentationFidelityRowV1] = []
    for fixture in sorted(fixtures, key=lambda row: row.fixture_id):
        manual = fixture.manual_transcription
        validate_narrative_transcription_v1(
            transcription=manual,
            frozen_report=fixture.narrative_report,
            expected_assignment_id=manual.assignment_id,
            expected_task_id=fixture.task_id,
        )
        if manual.status != "complete" or manual.submission is None:
            raise ValueError(
                f"Representation fixture {fixture.fixture_id!r} requires a complete manual transcription."
            )
        if manual.submission != fixture.structured_submission:
            raise ValueError(
                f"Representation fixture {fixture.fixture_id!r} does not encode identical structured semantics."
            )

        automated = fixture.automated_transcription
        automated_status: Literal["not_run", "abstain", "complete"] = "not_run"
        automated_exact_match: bool | None = None
        if automated is not None:
            validate_narrative_transcription_v1(
                transcription=automated,
                frozen_report=fixture.narrative_report,
                expected_assignment_id=automated.assignment_id,
                expected_task_id=fixture.task_id,
            )
            automated_status = automated.status
            if automated.status == "complete":
                if automated.submission is None:
                    raise ValueError("Complete automated transcription is missing its submission.")
                automated_exact_match = automated.submission == fixture.structured_submission
        rows.append(
            TrialEvalRepresentationFidelityRowV1(
                fixture_id=fixture.fixture_id,
                task_id=fixture.task_id,
                manual_exact_match=True,
                automated_status=automated_status,
                automated_exact_match=automated_exact_match,
            )
        )
    return tuple(rows)


__all__ = [
    "evaluate_representation_fixtures_v1",
    "submission_from_narrative_claims_v1",
    "validate_narrative_transcription_v1",
]
