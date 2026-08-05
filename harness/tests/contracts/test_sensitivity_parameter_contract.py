"""Tests for score-bearing sensitivity parameter semantics."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.submission.models import EvidenceRecordV1


def _evidence(**parameter: object) -> dict[str, object]:
    return {
        "evidence_id": "sensitivity.bounds.01",
        "evidence_type": "sensitivity",
        "principle": "sensitivity",
        "operation": "sensitivity_analysis",
        "sensitivity_parameter": parameter,
        "estimator": {
            "analysis_method_id": "bounds_bounded_deviation",
            "implementation": "bounded_deviation",
            "qualifications": ["bounded_unmeasured_deviation"],
        },
        "target": "Risk difference by day 365",
        "result": {
            "kind": "identified_interval",
            "effect_scale": "risk_difference_tau",
            "unit": "probability_difference",
            "lower": -0.2,
            "upper": 0.2,
            "interpretation": "Sensitivity-conditioned identified set.",
        },
        "interpretation": "Conclusions remain bounded around the null.",
        "source_artifacts": ["data/ADTTE.parquet"],
    }


def test_sensitivity_parameter_accepts_probability_scale_value() -> None:
    record = EvidenceRecordV1.model_validate(_evidence(value=0.1, unit="probability"))

    assert record.sensitivity_parameter is not None
    assert record.sensitivity_parameter.value == 0.1


@pytest.mark.parametrize(
    "parameter",
    (
        {"value": -0.1, "unit": "probability"},
        {"value": 1.1, "unit": "probability"},
        {"value": 0.1, "unit": "percent"},
    ),
)
def test_sensitivity_parameter_rejects_invalid_scale_or_unit(parameter: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        EvidenceRecordV1.model_validate(_evidence(**parameter))
