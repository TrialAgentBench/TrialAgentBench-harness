"""Independent raw-witness projection tests."""

from trialagentbench_validation.raw_projection import (
    _DiagnosticMeasureV1,
    _DiagnosticObligationV1,
    _measure_matches,
    _participant_artifact_exists,
    _precision_preserves_regime,
    _route_diagnostic_assumption_id,
)


def test_required_route_diagnostic_resolves_without_prescriptive_contract() -> None:
    assert (
        _route_diagnostic_assumption_id(
            diagnostic_id="model_form_public",
            evidence_target="model_form",
            obligation=None,
            required_diagnostics=("model_form_public",),
            identification_assumptions=("model_form", "randomization_integrity"),
        )
        == "model_form"
    )


def test_unrequired_diagnostic_cannot_use_route_scoped_fallback() -> None:
    assert (
        _route_diagnostic_assumption_id(
            diagnostic_id="model_form_public",
            evidence_target="model_form",
            obligation=None,
            required_diagnostics=(),
            identification_assumptions=("model_form",),
        )
        is None
    )


def test_prescriptive_contract_controls_assumption_mapping() -> None:
    obligation = _DiagnosticObligationV1(
        assumption_id="model_form",
        diagnostic_id="model_form_public",
        evidence_requirement="empirical_diagnostic",
        primary_credit_policy="method_dependent",
        operation="Assess model form.",
        public_evidence_basis=("data.parquet",),
        interpretation="Interpret only on released evidence.",
    )

    assert (
        _route_diagnostic_assumption_id(
            diagnostic_id="model_form_public",
            evidence_target="censoring_ignorability",
            obligation=obligation,
            required_diagnostics=("model_form_public",),
            identification_assumptions=("model_form",),
        )
        == "model_form"
    )


def test_diagnostic_measure_comparison_uses_declared_precision() -> None:
    measure = _DiagnosticMeasureV1(
        metric_id="observed_duplicate_group_count",
        value=25.0004,
        unit="count",
        decimal_places=3,
    )

    assert _measure_matches(
        measure,
        expected={"observed_duplicate_group_count": 25.0},
        expected_units={"observed_duplicate_group_count": "count"},
    )
    assert not _measure_matches(
        measure.model_copy(update={"metric_id": "observed_extra_row_count"}),
        expected={"observed_duplicate_group_count": 25.0},
        expected_units={"observed_duplicate_group_count": "count"},
    )


def test_diagnostic_precision_must_preserve_severity_regime() -> None:
    imprecise = _DiagnosticMeasureV1(
        metric_id="diagnostic_value",
        value=0.1,
        unit="proportion",
        decimal_places=1,
    )
    precise = imprecise.model_copy(update={"decimal_places": 3})

    assert not _precision_preserves_regime(imprecise, thresholds=(0.05, 0.1, 0.2))
    assert _precision_preserves_regime(precise, thresholds=(0.05, 0.15, 0.2))


def test_participant_artifact_resolution_accepts_only_public_relative_paths(
    tmp_path,
) -> None:
    participant_root = tmp_path / "public"
    item_root = participant_root / "items" / "TASK1"
    item_root.mkdir(parents=True)
    (item_root / "task.json").write_text("{}", encoding="utf-8")
    (participant_root / "diagnostic_dictionary.json").write_text("{}", encoding="utf-8")

    assert _participant_artifact_exists(
        participant_item_root=item_root,
        participant_root=participant_root,
        submitted_path="task.json",
    )
    assert _participant_artifact_exists(
        participant_item_root=item_root,
        participant_root=participant_root,
        submitted_path="diagnostic_dictionary.json",
    )
    assert not _participant_artifact_exists(
        participant_item_root=item_root,
        participant_root=participant_root,
        submitted_path="../grader/answers.json",
    )
