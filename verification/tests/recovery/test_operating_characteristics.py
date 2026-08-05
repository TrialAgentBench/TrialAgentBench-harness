"""Shared operating-characteristic contract tests."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest
from pydantic import BaseModel

import trialagentbench_validation.external.recovery as recovery
from trialagentbench_validation.external.recovery.clustered_ordinal import (
    ClusteredOrdinalRecoveryV1,
)
from trialagentbench_validation.external.recovery.open_outcomes import (
    OpenOutcomeEvidenceSummaryV1,
)
from trialagentbench_validation.external.recovery.ordinal import (
    OrdinalDoseRecoveryV1,
)
from trialagentbench_validation.external.recovery.survival import (
    SurvivalDoseRecoveryV1,
)
from trialagentbench_validation.statistics import (
    proportion_interval,
    scale_aware_tolerance,
)


def test_wilson_interval_has_expected_limits() -> None:
    low, high = proportion_interval(94, 100)

    assert low == pytest.approx(0.875232, abs=1e-6)
    assert high == pytest.approx(0.972214, abs=1e-6)


@pytest.mark.parametrize(
    ("successes", "total"),
    [(-1, 10), (11, 10), (0, 0)],
)
def test_wilson_interval_rejects_invalid_counts(successes: int, total: int) -> None:
    with pytest.raises(ValueError):
        proportion_interval(successes, total)


@pytest.mark.parametrize("successes", [0, 100])
def test_wilson_interval_respects_probability_bounds(successes: int) -> None:
    low, high = proportion_interval(successes, 100)

    assert 0.0 <= low <= high <= 1.0


def test_scale_aware_tolerance_uses_sampling_uncertainty() -> None:
    assert scale_aware_tolerance(0.2) == pytest.approx(2e-4)
    assert scale_aware_tolerance(1e-4) == pytest.approx(1e-6)


@pytest.mark.parametrize(
    "model",
    [
        SurvivalDoseRecoveryV1,
        OrdinalDoseRecoveryV1,
        ClusteredOrdinalRecoveryV1,
    ],
)
def test_released_recovery_coverage_has_uncertainty(model: type[BaseModel]) -> None:
    fields = model.model_fields

    assert {"worlds", "coverage", "coverage_ci_low", "coverage_ci_high"} <= set(fields)


def test_open_outcome_summary_retains_coverage_uncertainty() -> None:
    fields = OpenOutcomeEvidenceSummaryV1.model_fields

    assert {
        "patency_source_dose_coverage_ci_low",
        "patency_source_dose_coverage_ci_high",
        "headsoar_source_dose_coverage_ci_low",
        "headsoar_source_dose_coverage_ci_high",
        "patency_broken_minus_intact_curve_mae_ci_low",
        "patency_broken_minus_intact_curve_mae_ci_high",
        "headsoar_broken_minus_intact_category_mae_ci_low",
        "headsoar_broken_minus_intact_category_mae_ci_high",
    } <= set(fields)


def test_released_operating_characteristics_include_intervals() -> None:
    checked = []
    binomial_suffixes = (
        "coverage",
        "rejection_rate",
        "monotone_world_fraction",
        "directionally_concordant_fraction",
        "positive_slope_fraction",
        "negative_slope_fraction",
        "improvement_fraction",
    )
    for module_info in pkgutil.iter_modules(
        recovery.__path__,
        recovery.__name__ + ".",
    ):
        module = importlib.import_module(module_info.name)
        for _, model in inspect.getmembers(module, inspect.isclass):
            if not issubclass(model, BaseModel) or model.__module__ != module.__name__:
                continue
            fields = set(model.model_fields)
            for field in fields:
                if not field.endswith(binomial_suffixes):
                    continue
                checked.append(f"{module.__name__}.{model.__name__}.{field}")
                assert {f"{field}_ci_low", f"{field}_ci_high"} <= fields

    assert checked
