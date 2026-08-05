"""Lossless compilation tests for score-bearing TrialEval submissions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.submission import TrialEvalSubmissionV1
from trialagentbench_harness.trialeval.canonicalize import (
    canonicalize_trialeval_submission_v1,
)


def _submission_payload() -> dict[str, object]:
    return {
        "task_id": "TASK1",
        "primary_analysis": {
            "declared_primary": True,
            "estimand": {
                "estimand_id": "rmst_365",
                "population_id": "intention_to_treat",
                "treatment_id": "active",
                "comparator_id": "control",
                "endpoint_id": "overall_survival",
                "intercurrent_event_strategy_ids": ["treatment_policy"],
                "horizon": {"value": 365.0, "unit": "days"},
            },
            "estimator": {
                "analysis_method_id": "km_rmst_bootstrap",
                "implementation": "Kaplan-Meier integration",
                "qualifications": [
                    "independent_censoring",
                    "randomization_exchangeability",
                ],
            },
            "result_kind": "numeric_point",
            "result": {
                "kind": "scalar",
                "value": 12.0,
                "effect_scale": "rmst_difference_tau",
                "unit": "days",
                "interval": {
                    "lower": 4.0,
                    "upper": 20.0,
                    "confidence_level": 0.95,
                },
            },
            "favorable_direction": "higher",
            "evidence_ids": ["ph-check"],
        },
        "evidence": [
            {
                "evidence_id": "ph-check",
                "evidence_type": "diagnostic",
                "principle": "proportional_hazards",
                "operation": "assessment",
                "diagnostic_id": "proportional_hazards_public",
                "target": "proportional hazards",
                "result": {
                    "kind": "diagnostic_test",
                    "statistic": {
                        "metric_id": "schoenfeld_statistic",
                        "value": 3.2,
                        "unit": "chi_square",
                        "decimal_places": 1,
                    },
                    "p_value": {
                        "metric_id": "schoenfeld_p_value",
                        "value": 0.04,
                        "unit": "probability",
                        "decimal_places": 2,
                    },
                },
                "interpretation": "The proportional-hazards assumption is stressed.",
                "source_artifacts": ["data/adtte.parquet"],
            }
        ],
        "limitations": ["Administrative censoring limits later-horizon inference."],
    }


def test_submission_compiles_losslessly_without_scoring_key() -> None:
    canonical = canonicalize_trialeval_submission_v1(
        TrialEvalSubmissionV1.model_validate(_submission_payload()),
        validated_diagnostic_ids=frozenset({"proportional_hazards_public"}),
    )

    assert canonical.item_id == "TASK1"
    assert canonical.primary.model_dump(mode="json") == {
        "analysis_population_id": "intention_to_treat",
        "estimand_id": "rmst_365",
        "intercurrent_event_strategy_ids": ["treatment_policy"],
        "assessment_horizon_days": 365.0,
        "treatment_id": "active",
        "comparator_id": "control",
        "endpoint_id": "overall_survival",
        "effect_scale": "rmst_difference_tau",
        "analysis_method_id": "km_rmst_bootstrap",
    }
    assert canonical.diagnostic_ids == ("proportional_hazards_public",)
    assert canonical.result.kind == "numeric_point"


def test_unverified_diagnostic_mention_is_not_canonical_evidence() -> None:
    canonical = canonicalize_trialeval_submission_v1(
        TrialEvalSubmissionV1.model_validate(_submission_payload()),
        validated_diagnostic_ids=frozenset(),
    )

    assert canonical.diagnostic_ids == ()


def test_normalizer_rejects_diagnostic_not_linked_to_primary() -> None:
    with pytest.raises(ValueError, match="absent from the linked primary evidence"):
        canonicalize_trialeval_submission_v1(
            TrialEvalSubmissionV1.model_validate(_submission_payload()),
            validated_diagnostic_ids=frozenset({"unknown_diagnostic"}),
        )


def test_method_mention_cannot_replace_explicit_method_identity() -> None:
    payload = _submission_payload()
    primary = payload["primary_analysis"]
    assert isinstance(primary, dict)
    estimator = primary["estimator"]
    assert isinstance(estimator, dict)
    estimator.pop("analysis_method_id")
    limitations = payload["limitations"]
    assert isinstance(limitations, list)
    limitations.append("A participant bootstrap was considered.")

    with pytest.raises(ValidationError, match="analysis_method_id"):
        TrialEvalSubmissionV1.model_validate(payload)


def test_declared_result_kind_cannot_relabel_a_point_as_a_bound() -> None:
    payload = _submission_payload()
    primary = payload["primary_analysis"]
    assert isinstance(primary, dict)
    primary["result_kind"] = "identification_bound"

    with pytest.raises(ValidationError, match="incompatible"):
        TrialEvalSubmissionV1.model_validate(payload)


def test_audit_only_qualifications_do_not_change_canonical_method_identity() -> None:
    original = TrialEvalSubmissionV1.model_validate(_submission_payload())
    payload = _submission_payload()
    primary = payload["primary_analysis"]
    assert isinstance(primary, dict)
    estimator = primary["estimator"]
    assert isinstance(estimator, dict)
    estimator["qualifications"] = [
        "randomization_exchangeability",
        "independent_censoring",
        "randomization_exchangeability",
    ]
    differently_qualified = TrialEvalSubmissionV1.model_validate(payload)

    original_canonical = canonicalize_trialeval_submission_v1(
        original,
        validated_diagnostic_ids=frozenset({"proportional_hazards_public"}),
    )
    differently_qualified_canonical = canonicalize_trialeval_submission_v1(
        differently_qualified,
        validated_diagnostic_ids=frozenset({"proportional_hazards_public"}),
    )

    assert original_canonical == differently_qualified_canonical
