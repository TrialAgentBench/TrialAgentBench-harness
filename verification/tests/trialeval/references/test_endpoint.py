"""Tests for independent endpoint-validation replay."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from trialagentbench_validation.trialeval.references.endpoint import (
    _objective,
    _objective_gradient,
)


def test_endpoint_validation_gradient_matches_numerical_score() -> None:
    rng = np.random.RandomState(37)
    parameters = rng.normal(scale=0.5, size=8)
    arguments = {
        "arm_index": np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64),
        "stratum_index": np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
        "observed": np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int64),
        "selected": np.asarray([True, True, True, False, False, False]),
        "validated": np.asarray([0, 1, 1, 0, 0, 0], dtype=np.int64),
        "frequency": np.asarray([7, 11, 13, 17, 19, 23], dtype=np.int64),
        "arm_count": 2,
        "stratum_count": 2,
    }

    numerical = approx_fprime(
        parameters,
        lambda value: _objective(value, **arguments),
        epsilon=1e-7,
    )
    exact = _objective_gradient(parameters, **arguments)

    assert exact == pytest.approx(numerical, rel=2e-6, abs=2e-6)
