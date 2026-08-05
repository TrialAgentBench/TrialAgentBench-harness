"""Tests for independent survival qualification contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trialagentbench_validation.external.recovery.survival import (
    SurvivalArmReferenceV1,
    SurvivalQualificationDesignV1,
)


def _arm(arm_id: str) -> SurvivalArmReferenceV1:
    return SurvivalArmReferenceV1(
        arm_id=arm_id,
        participants=100,
        events=10,
        early_censoring=2,
        survival_at_grid=(0.98, 0.95, 0.90),
        at_risk_at_grid=(100, 96, 90),
        rmst_at_horizon=350.0,
    )


def test_survival_design_requires_null_and_source_doses() -> None:
    with pytest.raises(ValidationError, match="null and source-fitted"):
        SurvivalQualificationDesignV1(
            trial_id="trial",
            source_sha256="a" * 64,
            participants=200,
            worlds=100,
            seed=1,
            horizon=365.0,
            time_grid=(90.0, 180.0, 365.0),
            control_arm_id="control",
            treatment_arm_id="treatment",
            treatment_prevalence=0.5,
            source_log_hazard_ratio=-0.2,
            dose_multipliers=(0.0, 2.0, 4.0),
            random_censoring_rate=0.001,
            arms=(_arm("control"), _arm("treatment")),
        )


def test_survival_design_requires_complete_curve_grid() -> None:
    invalid = _arm("treatment").model_copy(update={"survival_at_grid": (0.98, 0.95)})
    with pytest.raises(ValidationError, match="source curves"):
        SurvivalQualificationDesignV1(
            trial_id="trial",
            source_sha256="a" * 64,
            participants=200,
            worlds=100,
            seed=1,
            horizon=365.0,
            time_grid=(90.0, 180.0, 365.0),
            control_arm_id="control",
            treatment_arm_id="treatment",
            treatment_prevalence=0.5,
            source_log_hazard_ratio=-0.2,
            dose_multipliers=(0.0, 1.0, 2.0),
            random_censoring_rate=0.001,
            arms=(_arm("control"), invalid),
        )
