"""Tests for stepped-wedge public-reference replay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trialagentbench_validation.trialeval.references.stepped_wedge import (
    stepped_wedge_period_adjusted_baseline_rates_v1,
    stepped_wedge_period_adjusted_risk_difference_tau_v1,
    stepped_wedge_period_adjusted_risk_difference_tau_with_uncertainty_v1,
)


def _stepped_wedge_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    adsl_rows: list[dict[str, object]] = []
    adtte_rows: list[dict[str, object]] = []
    for site_index, switch_day in enumerate((14, 28, 42, 56, 70, 84)):
        for subject_index in range(5):
            subject_id = f"S{site_index}P{subject_index}"
            adsl_rows.append(
                {
                    "USUBJID": subject_id,
                    "RFSTDTC": f"2025-01-{site_index + 1:02d}",
                    "INTERVENTION_START_DY": switch_day,
                    "SITEID": f"S{site_index}",
                }
            )
            event = (site_index + subject_index) % 3 != 0
            event_time = (
                float(12 + 7 * subject_index + 3 * site_index) if event else 90.0
            )
            adtte_rows.append(
                {
                    "USUBJID": subject_id,
                    "PARAMCD": "death",
                    "AVAL": min(event_time, 90.0),
                    "CNSR": 0 if event else 1,
                }
            )
    return pd.DataFrame(adsl_rows), pd.DataFrame(adtte_rows)


def test_stepped_wedge_public_replay_preserves_cluster_robust_uncertainty() -> None:
    adsl, adtte = _stepped_wedge_fixture()

    risk, risk_se = (
        stepped_wedge_period_adjusted_risk_difference_tau_with_uncertainty_v1(
            adsl=adsl,
            adtte=adtte,
            paramcd="death",
            tau=90.0,
        )
    )
    assert np.isfinite([risk, risk_se]).all()
    assert risk_se > 0.0
    assert (
        stepped_wedge_period_adjusted_risk_difference_tau_v1(
            adsl=adsl,
            adtte=adtte,
            paramcd="death",
            tau=90.0,
        )
        == risk
    )


def test_stepped_wedge_period_rates_adjust_for_treatment_exposure() -> None:
    """Calendar-period rates come from the treatment-adjusted person-period model."""

    adsl, adtte = _stepped_wedge_fixture()
    rates = stepped_wedge_period_adjusted_baseline_rates_v1(
        adsl=adsl,
        adtte=adtte,
        paramcd="death",
        tau=90.0,
    )

    assert len(rates) >= 2
    assert np.isfinite(rates).all()
    assert all(rate >= 0.0 for rate in rates)
