"""Tests for prospective TrialEval narrative-normalizer qualification."""

from __future__ import annotations

from typing import cast

from trialagentbench_harness.analysis.experiments.trialeval_normalizer_qualification import (
    analyse_normalizer_qualification_v1,
    select_normalizer_qualification_sample_v1,
)
from trialagentbench_harness.contracts.experiments import (
    TrialEvalAblationEndpointRowV1,
    TrialEvalNarrativeTranscriptionV1,
    TrialEvalNormalizerFrameUnitV1,
    TrialEvalNormalizerFrameV1,
    TrialEvalNormalizerQualificationDesignV1,
    TrialEvalNormalizerQualificationObservationSetV1,
    TrialEvalNormalizerQualificationObservationV1,
    TrialEvalNormalizerSampleUnitV1,
    TrialEvalNormalizerSampleV1,
)


def _design(*, retained: int = 2, maximum_errors: int = 0) -> TrialEvalNormalizerQualificationDesignV1:
    return TrialEvalNormalizerQualificationDesignV1(
        independent_unit="one_narrative_report_per_base_trial",
        error_event="any_score_relevant_field_disagrees_with_masked_human_reference",
        acceptable_error_probability=0.03,
        unacceptable_error_probability=0.10,
        type_i_error=0.05,
        type_ii_error=0.10,
        exact_minimum_sample_size=retained,
        inclusion_strata=("evaluation_series_id", "context_configuration"),
        minimum_reports_per_stratum=1,
        retained_sample_size=retained,
        maximum_accepted_errors=maximum_errors,
        realized_type_i_error=0.04,
        realized_power=0.91,
        secondary_uncertainty_method="stratified_cluster_bootstrap_with_weighted_hoeffding_envelope",
        secondary_confidence_level=0.95,
        secondary_bootstrap_replicates=1000,
        secondary_bootstrap_seed=17,
        secondary_uncertainty_rationale="Deterministic bounded test configuration.",
    )


def _frame() -> TrialEvalNormalizerFrameV1:
    rows: list[TrialEvalNormalizerFrameUnitV1] = []
    for family, context, design, assumption in (
        ("family-1", "C1", "D1", "A1"),
        ("family-2", "C2", "D4", "A3"),
    ):
        for model_index in range(2):
            rows.append(
                TrialEvalNormalizerFrameUnitV1(
                    unit_id=f"{family}-{model_index}",
                    run_identity_sha256=str(model_index) * 64,
                    assignment_id=f"assignment-{family}-{model_index}",
                    task_id="TASK1001" if family == "family-1" else "TASK1002",
                    base_trial_id=f"base-{family}",
                    report_sha256=("a" if family == "family-1" else "b") * 64,
                    regime_cell_id=family,
                    design_tier=design,
                    assumption_tier=assumption,
                    context_configuration=context,
                    data_preparation="analysis_ready",
                    analysis_specification="locked_sap" if context == "C1" else "protocol_only",
                    result_shape="scalar",
                    model_id=f"model-{model_index}",
                )
            )
    return TrialEvalNormalizerFrameV1(
        evaluator_release_sha256="e" * 64,
        participant_release_sha256="p" * 64,
        schedule_sha256="s" * 64,
        run_identity_sha256s=("0" * 64, "1" * 64),
        units=tuple(sorted(rows, key=lambda row: row.unit_id)),
    ).with_checksum()


def _submission(task_id: str, *, value: float) -> dict[str, object]:
    return {
        "schema_id": "trialagentbench.trialeval_submission/v1",
        "task_id": task_id,
        "primary_analysis": {
            "declared_primary": True,
            "estimand": {
                "estimand_id": "primary_itt",
                "population_id": "intention_to_treat",
                "treatment_id": "active",
                "comparator_id": "control",
                "endpoint_id": "time_to_event",
                "intercurrent_event_strategy_ids": ["rescue:treatment_policy"],
                "horizon": {"value": 365.0, "unit": "days"},
            },
            "estimator": {
                "analysis_method_id": "km_rmst_greenwood",
                "implementation": "Kaplan-Meier integration",
                "qualifications": ["independent_censoring"],
            },
            "result_kind": "numeric_point",
            "result": {
                "kind": "scalar",
                "value": value,
                "effect_scale": "rmst_difference_tau",
                "unit": "days",
                "interval": {"lower": 4.0, "upper": 32.0, "confidence_level": 0.95},
            },
            "favorable_direction": "higher",
            "evidence_ids": [],
        },
        "limitations": ["Interpretation is limited to 365 days."],
    }


def _claim(
    claim_id: str,
    field_path: str,
    text: str,
    parsed_value: object,
    *,
    role: str = "primary",
    result_shape: str | None = None,
    unit: str | None = None,
    orientation: str | None = None,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "field_path": field_path,
        "claim_role": role,
        "evidence_level": "executed",
        "spans": [{"start": 0, "end": len(text), "text": text}],
        "raw_value": text,
        "parsed_value": parsed_value,
        "result_shape": result_shape,
        "unit": unit,
        "orientation": orientation,
    }


def _transcription(
    unit: TrialEvalNormalizerSampleUnitV1,
    *,
    source: str,
    value: float = 18.0,
) -> TrialEvalNarrativeTranscriptionV1:
    submission = _submission(unit.task_id, value=value)
    primary = cast(dict[str, object], submission["primary_analysis"])
    payload: dict[str, object] = {
        "assignment_id": unit.assignment_id,
        "report_sha256": unit.report_sha256,
        "source": source,
        "source_identity": "masked-panel" if source == "manual_masked" else "provider:normalizer",
        "transcriber_identities": ["transcriber-A", "transcriber-B"] if source == "manual_masked" else [],
        "transcription_disposition": "independent_exact_agreement" if source == "manual_masked" else None,
        "blinded_to_model_identity": True,
        "blinded_to_evaluator_reference": True,
        "importer_prompt_sha256": None if source == "manual_masked" else "p" * 64,
        "importer_schema_sha256": None if source == "manual_masked" else "s" * 64,
        "importer_response_sha256": None if source == "manual_masked" else "r" * 64,
        "status": "complete",
        "submission": submission,
        "claims": [
            _claim("estimand", "primary_analysis.estimand", "estimand", primary["estimand"]),
            _claim("estimator", "primary_analysis.estimator", "estimator", primary["estimator"]),
            _claim(
                "result-kind",
                "primary_analysis.result_kind",
                "numeric point",
                primary["result_kind"],
            ),
            _claim(
                "result",
                "primary_analysis.result",
                "result",
                primary["result"],
                result_shape="scalar",
                unit="days",
            ),
            _claim(
                "direction",
                "primary_analysis.favorable_direction",
                "direction",
                "higher",
                orientation="higher",
            ),
            _claim(
                "limitations",
                "limitations",
                "limitations",
                submission["limitations"],
                role="limitation",
            ),
        ],
    }
    return TrialEvalNarrativeTranscriptionV1.model_validate(payload)


def _endpoint(
    unit: TrialEvalNormalizerSampleUnitV1,
    *,
    source: str,
    score: bool = True,
) -> TrialEvalAblationEndpointRowV1:
    return TrialEvalAblationEndpointRowV1(
        assignment_id=unit.assignment_id,
        task_id=unit.task_id,
        context_tier=unit.context_configuration,
        data_preparation=unit.data_preparation,
        analysis_specification=unit.analysis_specification,
        model_id=unit.model_id,
        replicate_id="seed-1",
        procedure_assistance="output_contract_only",
        prompt_condition="neutral",
        submission_interface="narrative",
        normalization_source=source,
        normalization_status="complete",
        primary_failure_code=None if score else "numeric_result_outside_tolerance",
        usable_primary=True,
        route_match=True,
        obligations_met=True,
        credit_eligible_route_count=1,
        numeric_result_available=True,
        result_match=score,
        primary_analysis_conforms=score,
        planning_applicable=False,
    )


def _observations(
    sample: TrialEvalNormalizerSampleV1,
    *,
    alter_first: bool = False,
) -> tuple[TrialEvalNormalizerQualificationObservationV1, ...]:
    rows = []
    for index, unit in enumerate(sample.units):
        automated_value = 19.0 if alter_first and index == 0 else 18.0
        rows.append(
            TrialEvalNormalizerQualificationObservationV1(
                sample_unit=unit,
                masked_human_reference=_transcription(unit, source="manual_masked"),
                automated_repeats=(
                    _transcription(unit, source="automated_importer", value=automated_value),
                    _transcription(unit, source="automated_importer", value=automated_value),
                ),
                masked_human_endpoint=_endpoint(unit, source="manual_masked"),
                automated_endpoint=_endpoint(
                    unit,
                    source="automated_importer",
                    score=False if alter_first and index == 0 else True,
                ),
            )
        )
    return tuple(rows)


def _observation_set(
    sample: TrialEvalNormalizerSampleV1,
    *,
    alter_first: bool = False,
) -> TrialEvalNormalizerQualificationObservationSetV1:
    return TrialEvalNormalizerQualificationObservationSetV1(
        sample_checksum=str(sample.checksum),
        packet_set_checksum="p" * 64,
        normalization_batch_checksum="n" * 64,
        evaluator_release_sha256="e" * 64,
        scoring_implementation_sha256="s" * 64,
        observations=_observations(sample, alter_first=alter_first),
    ).with_checksum()


def test_sample_selection_is_stratified_deterministic_and_self_weighting() -> None:
    sample = select_normalizer_qualification_sample_v1(
        frame=_frame(),
        experiment_design_checksum="d" * 64,
        design=_design(),
        selection_seed=11,
    )
    repeated = select_normalizer_qualification_sample_v1(
        frame=_frame(),
        experiment_design_checksum="d" * 64,
        design=_design(),
        selection_seed=11,
    )

    assert sample == repeated
    assert len(sample.units) == 2
    assert {unit.regime_cell_id for unit in sample.units} == {"family-1", "family-2"}
    assert {unit.inclusion_probability for unit in sample.units} == {0.5}


def test_exact_agreement_qualifies_and_reports_nonzero_uncertainty() -> None:
    design = _design()
    sample = select_normalizer_qualification_sample_v1(
        frame=_frame(),
        experiment_design_checksum="d" * 64,
        design=design,
        selection_seed=11,
    )
    report = analyse_normalizer_qualification_v1(
        sample=sample,
        design=design,
        observation_set=_observation_set(sample),
    )

    assert report.qualified is True
    assert report.observed_error_count == 0
    error_metric = next(
        metric
        for metric in report.metrics
        if metric.metric == "score_relevant_error_rate" and metric.subgroup_dimension == "overall"
    )
    assert error_metric.estimate == 0.0
    assert error_metric.confidence_upper > 0.0
    assert report.checksum


def test_score_relevant_extraction_error_fails_exact_acceptance_rule() -> None:
    design = _design()
    sample = select_normalizer_qualification_sample_v1(
        frame=_frame(),
        experiment_design_checksum="d" * 64,
        design=design,
        selection_seed=11,
    )
    report = analyse_normalizer_qualification_v1(
        sample=sample,
        design=design,
        observation_set=_observation_set(sample, alter_first=True),
    )

    assert report.qualified is False
    assert report.observed_error_count == 1
    assert sum(row.score_relevant_error for row in report.unit_results) == 1
    assert any(row.primary_analysis_conformance_disagreement for row in report.unit_results)
