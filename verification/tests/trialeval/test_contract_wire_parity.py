"""Standalone wire-contract tests."""

from __future__ import annotations

import pytest

from trialagentbench_validation.contracts.scoring.method_ids import (
    BOUNDED_DEVIATION_METHOD_IDS_V1,
)
from trialagentbench_validation.contracts.scoring.route_reference_inputs import (
    RouteReferenceInputRecordV1,
)
from trialagentbench_validation.contracts.scoring.route_references import (
    RouteReferenceRecordV1,
)


@pytest.mark.parametrize(
    "method_id",
    (
        "observed:tau_bounds_bounded_deviation",
        "observed:validated_endpoint_bounded_deviation",
    ),
)
def test_independent_verifier_matches_bounded_deviation_method_semantics(
    method_id: str,
) -> None:
    """Require both packages to accept every bounded-deviation method."""

    reference_payload = {
        "schema_id": "trialagentbench.trialeval.route_reference/v1",
        "task_id": "TE-S06-A4-r001",
        "item_id": "TE-S06-A4",
        "lane_id": "primary_numeric.v1",
        "route_reference_id": f"TE-S06-A4-r001:primary_numeric.v1:{method_id}",
        "variant_role": "required_primary",
        "route_family": "partial_identification",
        "estimator_method_id": method_id,
        "effect_scale": "risk_difference_tau",
        "sensitivity_parameter": 0.2,
        "answer_shape": "bound",
        "value": 0.0,
        "lower": -0.2,
        "upper": 0.2,
        "public_evidence_basis": ("items/TE-S06-A4-r001/data/participants.csv",),
        "identification_class": "partially_identified",
        "support_status": "official_supported",
        "support_rationale": "Fixed bounded-deviation sensitivity analysis.",
        "numerical_equivalence": {
            "policy_id": "float64_sqrt_epsilon_v1",
            "absolute_tolerance": 1.4901161193847656e-08,
            "relative_tolerance": 1.4901161193847656e-08,
            "basis": "deterministic_cross_implementation_replay",
        },
    }

    validation_reference = RouteReferenceRecordV1.model_validate(reference_payload)

    assert validation_reference.estimator_method_id == method_id
    assert validation_reference.sensitivity_parameter == 0.2

    input_payload = {
        "schema_id": "trialagentbench.trialeval.route_reference_input/v1",
        "task_id": "TE-S06-A4-r001",
        "input_bundle_id": f"TE-S06-A4-r001:{method_id}",
        "estimator_method_id": method_id,
        "effect_scale": "risk_difference_tau",
        "sensitivity_parameter": 0.2,
        "lane_ids": ("primary_numeric.v1",),
        "route_reference_ids": (reference_payload["route_reference_id"],),
        "required_table_refs": (
            {
                "rel_path": "items/TE-S06-A4-r001/data/participants.csv",
                "semantic_role": "participant_table",
                "sha256": "0" * 64,
                "row_count": 1,
                "column_names": ("participant_id",),
            },
        ),
        "source_role": "public_surface_mirror",
    }

    validation_input = RouteReferenceInputRecordV1.model_validate(input_payload)

    assert validation_input.estimator_method_id == method_id
    assert validation_input.sensitivity_parameter == 0.2


def test_bounded_deviation_registry_is_exact() -> None:
    """Keep sensitivity-parameter method membership explicit."""

    assert BOUNDED_DEVIATION_METHOD_IDS_V1 == {
        "observed:tau_bounds_bounded_deviation",
        "observed:validated_endpoint_bounded_deviation",
    }
