"""Tests for standalone survival replay primitives."""

from __future__ import annotations

import numpy as np

from trialagentbench_validation.trialeval.references.survival import (
    _coxph_binary_breslow_risk_difference_tau,
)


def test_cox_fixed_horizon_risk_replay_is_finite_and_directional() -> None:
    rng = np.random.default_rng(901)
    treatment = np.repeat([0, 1], 200)
    event_time = rng.exponential(scale=np.where(treatment == 1, 18.0, 10.0))
    censor_time = rng.uniform(9.0, 25.0, size=len(treatment))

    value = _coxph_binary_breslow_risk_difference_tau(
        t=np.minimum(event_time, censor_time),
        e=(event_time <= censor_time).astype(np.int64),
        a=treatment.astype(np.int64),
        tau=12.0,
    )

    assert np.isfinite(value)
    assert -1.0 < value < 0.0
