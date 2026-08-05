"""Independent component-evidence route identity tests."""

import pytest

from trialagentbench_validation.contracts.component_evidence import (
    _canonical_route_method_id,
)


@pytest.mark.parametrize(
    ("route_id", "effect_scale", "eligible_route_ids", "expected"),
    (
        (
            "TASK1:primary_numeric.v1:max_recoverable:"
            "observed:cox_rcs_standardized_risk_tau_reference:"
            "standardized_risk_difference_tau_reference",
            "standardized_risk_difference_tau_reference",
            ("observed:cox_rcs_standardized_risk_tau_reference",),
            "observed:cox_rcs_standardized_risk_tau_reference",
        ),
        (
            "TASK1:primary_numeric.v1:max_recoverable:"
            "qualified_limitation_or_abstention:risk_difference_tau",
            "risk_difference_tau",
            ("qualified_limitation_or_abstention",),
            "qualified_limitation_or_abstention",
        ),
        (
            "TASK1:primary_numeric.v1:max_recoverable:"
            "observed:cluster_parallel_participant_weighted_km_ipcw_baseline_cox",
            "risk_difference_tau",
            ("observed:cluster_parallel_participant_weighted_km_ipcw_baseline_cox",),
            "observed:cluster_parallel_participant_weighted_km_ipcw_baseline_cox",
        ),
    ),
)
def test_task_scoped_route_recovers_canonical_method(
    route_id: str,
    effect_scale: str,
    eligible_route_ids: tuple[str, ...],
    expected: str,
) -> None:
    assert (
        _canonical_route_method_id(
            route_id=route_id,
            item_id="TASK1",
            effect_scale=effect_scale,
            eligible_route_ids=eligible_route_ids,
        )
        == expected
    )


def test_task_scoped_route_rejects_effect_scale_drift() -> None:
    with pytest.raises(ValueError, match="invalid or ambiguous canonical method"):
        _canonical_route_method_id(
            route_id="TASK1:primary_numeric.v1:max_recoverable:observed:km:risk_difference_tau",
            item_id="TASK1",
            effect_scale="rmst_difference_tau",
            eligible_route_ids=("observed:km",),
        )
