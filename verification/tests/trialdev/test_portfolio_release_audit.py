"""Tests for independent TrialDev portfolio release auditing."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from trialagentbench_validation.trialdev.portfolio_release_audit import (
    TrialDevEpisodeRealismV1,
    TrialDevObservationalRealismV1,
    TrialDevPortfolioReleaseAuditV1,
    _check_event_time,
    _serious_event,
)


def _episode() -> TrialDevEpisodeRealismV1:
    return TrialDevEpisodeRealismV1(
        episode_id="world:regimen:phase2",
        world_id="world",
        asset_id="regimen",
        phase_id="phase2",
        row_count=4,
        follow_up_days=90,
        treated_count=2,
        control_count=2,
        efficacy_event_rate_treated=0.25,
        efficacy_event_rate_control=0.5,
        serious_ae_rate_treated=0.1,
        serious_ae_rate_control=0.1,
        discontinuation_rate_treated=0.1,
        discontinuation_rate_control=0.1,
        loss_to_follow_up_rate=0.05,
        terminal_event_rate=0.05,
    )


def _observational() -> TrialDevObservationalRealismV1:
    return TrialDevObservationalRealismV1(
        world_id="world",
        row_count=40,
        column_count=12,
        treatment_counts={
            "control": 10,
            "regimen_a": 10,
            "regimen_b": 10,
            "regimen_c": 10,
        },
        declared_adjustment_covariate_count=7,
        complete_case_rate=0.95,
        minimum_treatment_count=10,
    )


def test_event_and_seriousness_controls_accept_coherent_clinical_records() -> None:
    frame = pd.DataFrame(
        {
            "AE_CARDIAC_EVENT_E": [1, 0, 1],
            "AE_CARDIAC_EVENT_T": [10.0, 90.0, 40.0],
            "AE_CARDIAC_SERIOUS": [1.0, None, 0.0],
        }
    )

    event, time = _check_event_time(
        frame,
        event_column="AE_CARDIAC_EVENT_E",
        time_column="AE_CARDIAC_EVENT_T",
        follow_up_days=90,
    )

    assert event.tolist() == [1, 0, 1]
    assert time.tolist() == [10.0, 90.0, 40.0]
    assert _serious_event(frame).tolist() == [1, 0, 0]


@pytest.mark.parametrize(
    ("column", "values", "match"),
    [
        ("AE_CARDIAC_EVENT_E", [1, 2, 0], "must be binary"),
        ("AE_CARDIAC_EVENT_T", [10.0, 91.0, 40.0], "outside the declared follow-up"),
    ],
)
def test_event_controls_reject_impossible_records(
    column: str, values: list[float], match: str
) -> None:
    frame = pd.DataFrame(
        {
            "AE_CARDIAC_EVENT_E": [1, 0, 0],
            "AE_CARDIAC_EVENT_T": [10.0, 90.0, 40.0],
        }
    )
    frame[column] = values

    with pytest.raises(ValueError, match=match):
        _check_event_time(
            frame,
            event_column="AE_CARDIAC_EVENT_E",
            time_column="AE_CARDIAC_EVENT_T",
            follow_up_days=90,
        )


def test_seriousness_control_rejects_values_without_an_adverse_event() -> None:
    frame = pd.DataFrame(
        {
            "AE_CARDIAC_EVENT_E": [1, 0],
            "AE_CARDIAC_SERIOUS": [0.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="observed exactly when its event occurs"):
        _serious_event(frame)


def test_release_report_binds_status_and_aggregate_counts() -> None:
    episode = _episode()
    observational = _observational()
    report = TrialDevPortfolioReleaseAuditV1(
        release_manifest_sha256="a" * 64,
        release_source_identity="b" * 64,
        world_count=1,
        participant_view_count=8,
        randomized_episode_count=1,
        randomized_row_count=4,
        observational_row_count=40,
        episode_realism=(episode,),
        observational_realism=(observational,),
        findings=(),
        status="pass",
    )

    assert report.status == "pass"
    with pytest.raises(ValidationError, match="status disagrees"):
        TrialDevPortfolioReleaseAuditV1(
            **report.model_dump(exclude={"findings", "status"}),
            findings=("tampered_artifact",),
            status="pass",
        )
    with pytest.raises(ValidationError, match="row count disagrees"):
        TrialDevPortfolioReleaseAuditV1(
            **report.model_dump(exclude={"randomized_row_count"}),
            randomized_row_count=5,
        )


def test_observational_census_rejects_incomplete_treatment_counts() -> None:
    with pytest.raises(ValidationError, match="sum to the extract row count"):
        TrialDevObservationalRealismV1(
            **_observational().model_dump(exclude={"treatment_counts"}),
            treatment_counts={
                "control": 10,
                "regimen_a": 9,
                "regimen_b": 10,
                "regimen_c": 10,
            },
        )
