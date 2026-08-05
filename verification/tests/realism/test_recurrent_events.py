"""Tests for non-disclosing recurrent-event summaries."""

from __future__ import annotations

import pytest

from trialagentbench_validation.external.realism.recurrent_events import (
    recurrent_event_study_fingerprint,
    summarize_recurrent_event_portfolio,
)


def _fingerprint(anchor: int, counts: tuple[int, ...]):
    participants = tuple(f"P{index:03d}" for index in range(len(counts)))
    event_participants = tuple(
        participant
        for participant, count in zip(participants, counts, strict=True)
        for _ in range(count)
    )
    return recurrent_event_study_fingerprint(
        participant_ids=participants,
        event_participant_ids=event_participants,
        anchor_id=f"anchor_{anchor:016x}",
        source_sha256=f"{anchor:064x}",
    )


def test_recurrent_event_fingerprint_recovers_gamma_poisson_moment_parameter() -> None:
    fingerprint = _fingerprint(1, (0,) * 10 + (2,) * 10 + (8,) * 10)

    assert fingerprint.participants == 30
    assert fingerprint.events == 100
    assert fingerprint.participants_with_event == 20
    assert fingerprint.variance_to_mean_ratio > 1.0
    assert fingerprint.gamma_frailty_variance_mom > 0.0


def test_recurrent_event_fingerprint_rejects_unlinked_event_rows() -> None:
    with pytest.raises(ValueError, match="absent"):
        recurrent_event_study_fingerprint(
            participant_ids=tuple(f"P{index:03d}" for index in range(20)),
            event_participant_ids=("UNKNOWN",),
            anchor_id="anchor_0000000000000001",
            source_sha256="1" * 64,
        )


def test_recurrent_event_portfolio_reports_study_level_uncertainty() -> None:
    report = summarize_recurrent_event_portfolio(
        (
            _fingerprint(1, (0,) * 10 + (1,) * 10 + (4,) * 10),
            _fingerprint(2, (0,) * 10 + (2,) * 10 + (7,) * 10),
            _fingerprint(3, (0,) * 10 + (3,) * 10 + (9,) * 10),
        )
    )

    assert report.studies == 3
    assert report.overdispersed_studies == 3
    assert (
        report.variance_to_mean_ratio_median_ci_low
        <= report.variance_to_mean_ratio_median
        <= report.variance_to_mean_ratio_median_ci_high
    )
