"""Scientific consistency checks for TrialDev release validation."""

from __future__ import annotations

import pytest

from trialagentbench_harness.trialdev.grading.validate import _validate_method_result


def _nonestimable_method_result(*, confounding_regime: str, action_id: str) -> dict[str, object]:
    method_route_id = "trialdev.observational.method.v1"
    estimator_id = "declared_estimator"
    return {
        "method_route_id": method_route_id,
        "estimator_id": estimator_id,
        "diagnostics": {
            "method_route_id": method_route_id,
            "estimator_id": estimator_id,
            "confounding_regime": confounding_regime,
        },
        "candidate_scores": [
            {
                "objective_id": "benefit_risk",
                "candidate_drug_id": "drug_a",
                "max_abs_unadjusted_smd_vs_target": 0.2,
                "max_abs_adjusted_smd_vs_target": None,
                "point_estimable": False,
                "inference_estimable": False,
            }
        ],
        "objective_policies": [
            {
                "objective_id": "benefit_risk",
                "policy": "insufficient_recoverability",
            }
        ],
        "observational_action_policies": [
            {
                "objective_id": "benefit_risk",
                "minimum_efficacy_gain": 0.005,
                "reference_target_ids": [action_id],
                "credit_eligible_target_ids": [action_id],
            }
        ],
        "estimator_comparisons": [
            {
                "objective_id": "benefit_risk",
                "estimator_id": estimator,
            }
            for estimator in (estimator_id, "raw_observed")
        ],
    }


@pytest.mark.parametrize(
    ("confounding_regime", "action_id"),
    (
        ("residual_unmeasured", "withhold_nomination"),
        ("measured_with_overlap", "withhold_nomination"),
    ),
)
def test_nonestimable_observational_action_matches_identification_regime(
    confounding_regime: str,
    action_id: str,
) -> None:
    policies, actions = _validate_method_result(
        result=_nonestimable_method_result(
            confounding_regime=confounding_regime,
            action_id=action_id,
        ),
        candidate_ids={"drug_a"},
        objective_margins={"benefit_risk": 0.01},
        observational_minimum_benefit=0.005,
    )

    assert policies["benefit_risk"]["policy"] == "insufficient_recoverability"
    assert actions["benefit_risk"]["reference_target_ids"] == [action_id]


def test_nonestimable_observational_action_rejects_identification_drift() -> None:
    with pytest.raises(ValueError, match="qualified non-nomination"):
        _validate_method_result(
            result=_nonestimable_method_result(
                confounding_regime="residual_unmeasured",
                action_id="drug_a",
            ),
            candidate_ids={"drug_a"},
            objective_margins={"benefit_risk": 0.01},
            observational_minimum_benefit=0.005,
        )
