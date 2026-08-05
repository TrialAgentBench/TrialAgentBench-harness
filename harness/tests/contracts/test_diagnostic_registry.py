from __future__ import annotations

import json
import math

import pytest

from trialagentbench_harness.contracts.scoring.diagnostic_registry import (
    diagnostic_key_by_assumption_id_v1,
    diagnostic_normal_critical_value_v1,
    load_diagnostic_registry_v1,
)
from trialagentbench_harness.contracts.trialeval_diagnostics import participant_diagnostic_dictionary_v1


def test_diagnostic_registry_contains_semantics_without_family_answers() -> None:
    registry = load_diagnostic_registry_v1()

    assert registry.schema_id == "trialagentbench.diagnostic_registry/v1"
    assert set(registry.assumption_tier_operating_definitions) == {"A1", "A2", "A3", "A4"}
    assert "default-defensible" in registry.assumption_tier_operating_definitions["A2"].label
    assert "cannot itself select" in registry.assumption_tier_operating_definitions["A3"].scoring_constraint
    a4 = registry.assumption_tier_operating_definitions["A4"]
    assert a4.label == "broken default; alternative response required"
    assert "Identification is classified separately" in a4.operative_definition
    assert registry.diagnostic_keys["proportional_hazards"].public_key == "proportional_hazards_public"
    assert "families" not in type(registry).model_fields
    assert "status_by_family" not in type(registry).model_fields
    assert diagnostic_key_by_assumption_id_v1()["model_form"] == "model_form_public"
    ph = registry.diagnostic_keys["proportional_hazards"]
    assert ph.severity_thresholds is not None
    assert ph.severity_thresholds.metric_name == "simultaneous_lower_abs_time_varying_log_hazard_range"
    assert set(ph.severity_thresholds.decision_metric_names.values()) == {
        "simultaneous_lower_abs_time_varying_log_hazard_range"
    }
    assert ph.primary_method_change_threshold == pytest.approx(math.log(1.25))
    assert registry.diagnostic_keys["cluster_structure"].severity_thresholds is None
    assert diagnostic_normal_critical_value_v1(assumption_id="proportional_hazards", comparisons=1) == pytest.approx(
        1.9599639845
    )


def test_participant_diagnostic_dictionary_is_an_exact_item_independent_projection() -> None:
    registry = load_diagnostic_registry_v1()
    dictionary = participant_diagnostic_dictionary_v1()

    assert set(dictionary.diagnostics) == {definition.public_key for definition in registry.diagnostic_keys.values()}
    ph = dictionary.diagnostics["proportional_hazards_public"]
    assert ph.severity_thresholds is not None
    assert ph.severity_thresholds.metric_name == "simultaneous_lower_abs_time_varying_log_hazard_range"
    assert ph.severity_thresholds.metric_unit == "log_hazard_ratio"
    assert "CoxPHFitter" in ph.participant_operation
    payload = json.dumps(dictionary.model_dump(mode="json"), sort_keys=True)
    assert '"task_id"' not in payload
    assert '"result"' not in payload
    assert diagnostic_normal_critical_value_v1(assumption_id="model_form", comparisons=4) > 1.9599639845
