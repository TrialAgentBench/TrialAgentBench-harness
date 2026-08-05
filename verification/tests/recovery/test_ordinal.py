"""Tests for independent ordinal qualification contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trialagentbench_validation.external.recovery.ordinal import (
    OrdinalArmReferenceV1,
    OrdinalDoseDistributionV1,
    OrdinalQualificationDesignV1,
    OrdinalSafetyReferenceV1,
)


def _arm(arm_id: str) -> OrdinalArmReferenceV1:
    return OrdinalArmReferenceV1(
        arm_id=arm_id,
        participants=100,
        missing_outcomes=1,
        category_probabilities=(0.2, 0.3, 0.5),
    )


def _distributions() -> tuple[OrdinalDoseDistributionV1, ...]:
    return tuple(
        OrdinalDoseDistributionV1(
            dose_multiplier=dose,
            arm_id=arm,
            category_probabilities=(0.2, 0.3, 0.5),
        )
        for dose in (0.0, 1.0, 2.0)
        for arm in ("control", "treatment")
    )


def _safety() -> tuple[OrdinalSafetyReferenceV1, ...]:
    return tuple(
        OrdinalSafetyReferenceV1(
            endpoint="mortality",
            arm_id=arm,
            observed_participants=100,
            event_probability=0.2,
        )
        for arm in ("control", "treatment")
    )


def test_ordinal_design_requires_consecutive_category_support() -> None:
    with pytest.raises(ValidationError, match="consecutive integers"):
        OrdinalQualificationDesignV1(
            trial_id="trial",
            source_sha256="a" * 64,
            participants=200,
            worlds=100,
            seed=1,
            categories=(0, 1, 3),
            control_arm_id="control",
            treatment_arm_id="treatment",
            source_log_common_odds_ratio=-0.1,
            dose_multipliers=(0.0, 1.0, 2.0),
            arms=(_arm("control"), _arm("treatment")),
            fitted_distributions=_distributions(),
            safety_references=_safety(),
        )


def test_ordinal_design_rejects_incomplete_dose_by_arm_grid() -> None:
    distributions = list(_distributions())
    distributions[-1] = distributions[-2]
    with pytest.raises(ValidationError, match="dose-by-arm grid"):
        OrdinalQualificationDesignV1(
            trial_id="trial",
            source_sha256="a" * 64,
            participants=200,
            worlds=100,
            seed=1,
            categories=(0, 1, 2),
            control_arm_id="control",
            treatment_arm_id="treatment",
            source_log_common_odds_ratio=-0.1,
            dose_multipliers=(0.0, 1.0, 2.0),
            arms=(_arm("control"), _arm("treatment")),
            fitted_distributions=tuple(distributions),
            safety_references=_safety(),
        )


def test_ordinal_design_requires_safety_reference_for_each_arm() -> None:
    with pytest.raises(ValidationError, match="safety references"):
        OrdinalQualificationDesignV1(
            trial_id="trial",
            source_sha256="a" * 64,
            participants=200,
            worlds=100,
            seed=1,
            categories=(0, 1, 2),
            control_arm_id="control",
            treatment_arm_id="treatment",
            source_log_common_odds_ratio=-0.1,
            dose_multipliers=(0.0, 1.0, 2.0),
            arms=(_arm("control"), _arm("treatment")),
            fitted_distributions=_distributions(),
            safety_references=(_safety()[0], _safety()[0]),
        )
