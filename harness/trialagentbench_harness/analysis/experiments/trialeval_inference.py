"""Statistical contrasts for TrialEval assistance and interface experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np

from trialagentbench_harness.analysis.experiments.crossed_bootstrap import (
    crossed_cluster_bootstrap_mean,
)
from trialagentbench_harness.contracts.experiments import (
    ProcedureAssistanceV1,
    TrialEvalAblationAnalysisConfigV1,
    TrialEvalAblationArmSummaryV1,
    TrialEvalAblationContrastV1,
    TrialEvalAblationEvaluatorLabelsV1,
    TrialEvalAblationGradeRowV1,
    TrialEvalAblationMetricV1,
    TrialEvalAblationObservableContrastV1,
    TrialEvalAblationObservableRowV1,
    TrialEvalObservableMetricV1,
    TrialEvalRouteMultiplicitySummaryV1,
    TrialEvalSubmissionInterfaceV1,
    TrialEvalToleranceSensitivitySummaryV1,
    trialeval_capability_prompt_conditions_v1,
    trialeval_metric_population_v1,
)
from trialagentbench_harness.contracts.experiments.procedure_assistance import (
    TrialEvalAnalysisSpecificationV1,
)
from trialagentbench_harness.numeric_policy import (
    TRIALEVAL_ANALYSIS_SPECIFICATION_SEED_OFFSET_V1,
    TRIALEVAL_FACTORIAL_ARM_SEED_OFFSET_V1,
    TRIALEVAL_OBSERVABLE_SEED_OFFSET_V1,
    TRIALEVAL_TARGETED_CONTROL_SEED_OFFSET_V1,
)

_OBSERVABLE_METRICS: tuple[TrialEvalObservableMetricV1, ...] = (
    "answer_submitted",
    "public_data_inspected",
    "code_executed_successfully",
    "assistant_turns",
    "tool_calls",
    "file_inspections",
    "code_executions",
    "failed_code_executions",
    "truncated_outputs",
    "events_until_first_data_inspection",
    "events_until_first_successful_code_execution",
    "events_until_submission",
    "execution_elapsed_seconds",
    "declared_diagnostics",
    "declared_sensitivity_analyses",
    "declared_uncertainty",
)

_NormalizationSource = Literal["direct_structured", "manual_masked", "automated_importer"]
_ArmKey = tuple[
    str,
    TrialEvalAnalysisSpecificationV1,
    ProcedureAssistanceV1,
    TrialEvalSubmissionInterfaceV1,
    _NormalizationSource,
]


def trialeval_route_multiplicity_summaries_v1(
    rows: tuple[TrialEvalAblationGradeRowV1, ...],
) -> tuple[TrialEvalRouteMultiplicitySummaryV1, ...]:
    """Summarize official conformance without changing the scoring policy."""

    grouped: dict[
        tuple[
            str,
            TrialEvalAnalysisSpecificationV1,
            ProcedureAssistanceV1,
            TrialEvalSubmissionInterfaceV1,
            _NormalizationSource,
            Literal["single_route", "plural_route"],
        ],
        list[TrialEvalAblationGradeRowV1],
    ] = defaultdict(list)
    for row in rows:
        multiplicity: Literal["single_route", "plural_route"] = (
            "single_route" if row.credit_eligible_route_count == 1 else "plural_route"
        )
        key = (
            row.model_id,
            row.analysis_specification,
            row.procedure_assistance,
            row.submission_interface,
            row.normalization_source,
            multiplicity,
        )
        grouped[key].append(row)
    summaries = []
    for key, arm_rows in sorted(grouped.items()):
        passed = sum(row.primary_analysis_conforms for row in arm_rows)
        summaries.append(
            TrialEvalRouteMultiplicitySummaryV1(
                model_id=key[0],
                analysis_specification=key[1],
                procedure_assistance=key[2],
                submission_interface=key[3],
                normalization_source=key[4],
                route_multiplicity=key[5],
                n_assignments=len(arm_rows),
                n_conforming=passed,
                accepted_rate=passed / len(arm_rows),
            )
        )
    return tuple(summaries)


def trialeval_tolerance_sensitivity_summaries_v1(
    rows: tuple[TrialEvalAblationGradeRowV1, ...],
) -> tuple[TrialEvalToleranceSensitivitySummaryV1, ...]:
    """Bracket numerical equivalence without reclassifying official grades."""

    grouped: dict[_ArmKey, list[float]] = defaultdict(list)
    for row in rows:
        if row.numeric_tolerance_ratio is None:
            continue
        key: _ArmKey = (
            row.model_id,
            row.analysis_specification,
            row.procedure_assistance,
            row.submission_interface,
            row.normalization_source,
        )
        grouped[key].append(row.numeric_tolerance_ratio)
    summaries = []
    for key, ratios in sorted(grouped.items()):
        for policy, multiplier in (
            ("half_qualified_envelope", 0.5),
            ("qualified_envelope", 1.0),
            ("double_qualified_envelope", 2.0),
        ):
            within = sum(ratio <= multiplier for ratio in ratios)
            summaries.append(
                TrialEvalToleranceSensitivitySummaryV1(
                    model_id=key[0],
                    analysis_specification=key[1],
                    procedure_assistance=key[2],
                    submission_interface=key[3],
                    normalization_source=key[4],
                    tolerance_policy=policy,
                    n_matched_numeric_comparisons=len(ratios),
                    n_within_envelope=within,
                    within_envelope_rate=within / len(ratios),
                )
            )
    return tuple(summaries)


def _metric_value(
    row: TrialEvalAblationGradeRowV1,
    metric: TrialEvalAblationMetricV1,
) -> float | None:
    if metric == "usable_primary":
        return float(row.usable_primary)
    if metric == "route_match":
        return float(row.route_match)
    if metric == "obligations_met":
        return float(row.obligations_met)
    if metric == "numeric_result_available":
        return float(row.numeric_result_available)
    if metric == "primary_uncertainty_valid":
        return None if row.primary_uncertainty_valid is None else float(row.primary_uncertainty_valid)
    if metric == "primary_interval_agreement":
        return row.primary_interval_agreement
    if metric == "result_match":
        return float(row.result_match)
    if metric == "numeric_absolute_error":
        return row.numeric_absolute_error
    if metric == "numeric_tolerance_ratio":
        return row.numeric_tolerance_ratio
    if metric == "primary_analysis_conforms":
        return float(row.primary_analysis_conforms)
    if metric == "planning_valid":
        return None if row.planning_valid is None else float(row.planning_valid)
    if metric == "planning_usable_with_primary":
        return None if row.planning_usable_with_primary is None else float(row.planning_usable_with_primary)
    if metric == "planning_achieved_power":
        return row.planning_achieved_power
    if metric == "planning_power_shortfall":
        return row.planning_power_shortfall
    if metric == "planning_underpowered":
        return None if row.planning_underpowered is None else float(row.planning_underpowered)
    if metric == "planning_proportional_participant_deviation":
        return row.planning_proportional_participant_deviation
    if metric == "planning_log_sample_size_ratio":
        return row.planning_log_sample_size_ratio
    if metric == "planning_event_shortage":
        return None if row.planning_event_shortage is None else float(row.planning_event_shortage)
    if metric == "planning_excess_events":
        return None if row.planning_excess_events is None else float(row.planning_excess_events)
    if metric == "planning_excess_participants":
        return None if row.planning_excess_participants is None else float(row.planning_excess_participants)
    if metric == "planning_participant_shortage":
        return None if row.planning_participant_shortage is None else float(row.planning_participant_shortage)
    raise AssertionError(f"Unsupported TrialEval ablation metric: {metric}")


def _observable_metric_value(
    row: TrialEvalAblationObservableRowV1,
    metric: TrialEvalObservableMetricV1,
) -> float | None:
    value = getattr(row, metric)
    if value is None:
        return None
    return float(value)


def _crossed_cluster_interval(
    *,
    values: list[tuple[str, str, float]],
    config: TrialEvalAblationAnalysisConfigV1,
    seed_offset: int,
    contrast_id: str,
) -> tuple[int, int, int, float, float, float]:
    estimate = crossed_cluster_bootstrap_mean(
        values=values,
        min_row_clusters=config.min_base_trial_clusters,
        min_column_clusters=config.min_decoding_replicates,
        resamples=config.bootstrap_resamples,
        confidence_level=config.confidence_level,
        seed=config.bootstrap_seed + seed_offset,
        contrast_id=contrast_id,
        weighting="observation",
    )
    return (
        estimate.n_observations,
        estimate.n_row_clusters,
        estimate.n_column_clusters,
        estimate.estimate,
        estimate.interval_low,
        estimate.interval_high,
    )


def _clustered_contrast(
    *,
    model_id: str,
    metric: TrialEvalAblationMetricV1,
    contrast_id: str,
    values: list[tuple[str, str, float]],
    config: TrialEvalAblationAnalysisConfigV1,
    seed_offset: int,
    analysis_specification: Literal["protocol_only", "locked_sap"] | None = None,
    prompt_condition: str | None = None,
    applicability: str | None = None,
) -> TrialEvalAblationContrastV1:
    n_blocks, n_clusters, n_replicates, estimate, interval_low, interval_high = _crossed_cluster_interval(
        values=values,
        config=config,
        seed_offset=seed_offset,
        contrast_id=contrast_id,
    )
    return TrialEvalAblationContrastV1(
        model_id=model_id,
        metric=metric,
        analysis_population=trialeval_metric_population_v1(metric),
        contrast_id=contrast_id,
        analysis_specification=analysis_specification,
        prompt_condition=prompt_condition,
        applicability=applicability,
        n_blocks=n_blocks,
        n_base_trial_clusters=n_clusters,
        n_decoding_replicates=n_replicates,
        estimate=estimate,
        interval_low=float(interval_low),
        interval_high=float(interval_high),
        confidence_level=config.confidence_level,
    )


def factorial_observable_contrasts_v1(
    *,
    rows: tuple[TrialEvalAblationObservableRowV1, ...],
    evaluator_labels: TrialEvalAblationEvaluatorLabelsV1,
    config: TrialEvalAblationAnalysisConfigV1,
) -> tuple[TrialEvalAblationObservableContrastV1, ...]:
    """Estimate paired assistance effects on runner-observed analysis behavior."""

    if config.design != "factorial_interface":
        raise ValueError("Observable assistance contrasts require the factorial-interface design.")
    base_trials = {row.task_id: row.base_trial_id for row in evaluator_labels.task_identities}
    missing_labels = sorted({row.task_id for row in rows}.difference(base_trials))
    if missing_labels:
        raise ValueError(f"Observable rows lack evaluator base-trial identities: {missing_labels!r}.")
    results: list[TrialEvalAblationObservableContrastV1] = []
    seed_offset = TRIALEVAL_OBSERVABLE_SEED_OFFSET_V1
    expected = {
        (assistance, interface)
        for assistance in ("output_contract_only", "unordered_checklist", "ordered_sop")
        for interface in ("structured", "narrative")
    }
    model_specifications = sorted({(row.model_id, row.analysis_specification) for row in rows})
    for model_id, specification in model_specifications:
        model_rows = [
            row
            for row in rows
            if row.model_id == model_id
            and row.analysis_specification == specification
            and row.prompt_condition == "neutral"
        ]
        blocks: dict[tuple[str, str], dict[tuple[str, str], TrialEvalAblationObservableRowV1]] = {}
        for row in model_rows:
            block = blocks.setdefault((row.task_id, row.replicate_id), {})
            cell = (row.procedure_assistance, row.submission_interface)
            if cell in block:
                raise ValueError(f"Duplicate observable ablation cell: model={model_id!r}, cell={cell!r}.")
            block[cell] = row
        for block_id, block in blocks.items():
            if set(block) != expected:
                raise ValueError(
                    f"Observable ablation block {block_id!r} is incomplete: "
                    f"missing={sorted(expected.difference(block))!r}, extra={sorted(set(block).difference(expected))!r}."
                )
        for interface in ("structured", "narrative"):
            metrics = (
                _OBSERVABLE_METRICS
                if interface == "structured"
                else tuple(metric for metric in _OBSERVABLE_METRICS if not metric.startswith("declared_"))
            )
            for contrast_id, treatment, control in (
                (
                    "unordered_checklist_minus_output_contract_only",
                    "unordered_checklist",
                    "output_contract_only",
                ),
                ("ordered_sop_minus_unordered_checklist", "ordered_sop", "unordered_checklist"),
                ("ordered_sop_minus_output_contract_only", "ordered_sop", "output_contract_only"),
            ):
                for metric in metrics:
                    values: list[tuple[str, str, float]] = []
                    for (task_id, replicate_id), block in blocks.items():
                        treated = _observable_metric_value(block[(treatment, interface)], metric)
                        controlled = _observable_metric_value(block[(control, interface)], metric)
                        if treated is None or controlled is None:
                            raise ValueError(f"Observable metric {metric!r} is missing in a declared stratum.")
                        values.append((base_trials[task_id], replicate_id, treated - controlled))
                    (
                        n_blocks,
                        n_clusters,
                        n_replicates,
                        estimate,
                        interval_low,
                        interval_high,
                    ) = _crossed_cluster_interval(
                        values=values,
                        config=config,
                        seed_offset=seed_offset,
                        contrast_id=f"{contrast_id}:{specification}:{interface}:{metric}",
                    )
                    seed_offset += 1
                    results.append(
                        TrialEvalAblationObservableContrastV1(
                            model_id=model_id,
                            metric=metric,
                            contrast_id=contrast_id,
                            analysis_specification=specification,
                            submission_interface=interface,
                            n_blocks=n_blocks,
                            n_base_trial_clusters=n_clusters,
                            n_decoding_replicates=n_replicates,
                            estimate=estimate,
                            interval_low=interval_low,
                            interval_high=interval_high,
                            confidence_level=config.confidence_level,
                        )
                    )
    return tuple(results)


def factorial_ablation_arm_summaries_v1(
    *,
    rows: tuple[TrialEvalAblationGradeRowV1, ...],
    config: TrialEvalAblationAnalysisConfigV1,
) -> tuple[TrialEvalAblationArmSummaryV1, ...]:
    """Estimate absolute TrialEval outcomes in each factorial arm."""

    if config.design != "factorial_interface":
        raise ValueError("Factorial arm summaries require the factorial-interface design.")
    eligible = tuple(
        row
        for row in rows
        if row.prompt_condition == "neutral"
        and (
            (row.submission_interface == "structured" and row.normalization_source == "direct_structured")
            or (row.submission_interface == "narrative" and row.normalization_source == "manual_masked")
        )
    )
    if not eligible:
        raise ValueError("Factorial arm summaries have no authoritative normalized rows.")
    summaries: list[TrialEvalAblationArmSummaryV1] = []
    seed_offset = TRIALEVAL_FACTORIAL_ARM_SEED_OFFSET_V1
    strata = sorted(
        {
            (
                row.model_id,
                row.analysis_specification,
                row.procedure_assistance,
                row.submission_interface,
            )
            for row in eligible
        }
    )
    for model_id, specification, assistance, interface in strata:
        arm_rows = tuple(
            row
            for row in eligible
            if row.model_id == model_id
            and row.analysis_specification == specification
            and row.procedure_assistance == assistance
            and row.submission_interface == interface
        )
        keys = [(row.task_id, row.replicate_id) for row in arm_rows]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "Factorial arm summary contains duplicate task/replicate rows for "
                f"{(model_id, specification, assistance, interface)!r}."
            )
        for metric in config.metrics:
            values = [
                (row.base_trial_id, row.replicate_id, float(value))
                for row in arm_rows
                if (value := _metric_value(row, metric)) is not None
            ]
            if not values:
                continue
            estimate = crossed_cluster_bootstrap_mean(
                values=values,
                min_row_clusters=config.min_base_trial_clusters,
                min_column_clusters=config.min_decoding_replicates,
                resamples=config.bootstrap_resamples,
                confidence_level=config.confidence_level,
                seed=config.bootstrap_seed + seed_offset,
                contrast_id=f"arm:{model_id}:{specification}:{assistance}:{interface}:{metric}",
                weighting="observation",
            )
            seed_offset += 1
            summaries.append(
                TrialEvalAblationArmSummaryV1(
                    model_id=model_id,
                    metric=metric,
                    analysis_population=trialeval_metric_population_v1(metric),
                    analysis_specification=specification,
                    procedure_assistance=assistance,
                    submission_interface=interface,
                    n_assignments=len(values),
                    n_base_trial_clusters=estimate.n_row_clusters,
                    n_decoding_replicates=estimate.n_column_clusters,
                    estimate=estimate.estimate,
                    interval_low=estimate.interval_low,
                    interval_high=estimate.interval_high,
                    confidence_level=config.confidence_level,
                )
            )
    return tuple(summaries)


def _clustered_stratum_difference(
    *,
    model_id: str,
    metric: TrialEvalAblationMetricV1,
    contrast_id: str,
    reference_values: list[tuple[str, str, float]],
    control_values: list[tuple[str, str, float]],
    config: TrialEvalAblationAnalysisConfigV1,
    seed_offset: int,
    prompt_condition: str,
    analysis_specification: Literal["protocol_only", "locked_sap"],
) -> TrialEvalAblationContrastV1:
    """Estimate a difference between two independently labelled task strata."""

    def group_by_cell(values: list[tuple[str, str, float]]) -> dict[tuple[str, str], list[float]]:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for cluster_id, replicate_id, value in values:
            grouped[(cluster_id, replicate_id)].append(float(value))
        return grouped

    reference = group_by_cell(reference_values)
    control = group_by_cell(control_values)
    reference_ids = sorted({cluster_id for cluster_id, _, _ in reference_values})
    control_ids = sorted({cluster_id for cluster_id, _, _ in control_values})
    reference_replicates = sorted({replicate_id for _, replicate_id, _ in reference_values})
    control_replicates = sorted({replicate_id for _, replicate_id, _ in control_values})
    if len(reference_ids) < config.min_base_trial_clusters or len(control_ids) < config.min_base_trial_clusters:
        raise ValueError(
            f"Ablation specificity contrast {contrast_id!r} requires at least "
            f"{config.min_base_trial_clusters} base-trial clusters in each stratum."
        )
    if reference_replicates != control_replicates:
        raise ValueError("Ablation specificity strata must use the same decoding replicates.")
    if len(reference_replicates) < config.min_decoding_replicates:
        raise ValueError(
            f"Ablation specificity contrast {contrast_id!r} requires at least "
            f"{config.min_decoding_replicates} decoding replicates."
        )
    for cluster_id in reference_ids:
        for replicate_id in reference_replicates:
            if (cluster_id, replicate_id) not in reference:
                raise ValueError("Ablation reference stratum lacks a crossed trial/replicate cell.")
    for cluster_id in control_ids:
        for replicate_id in control_replicates:
            if (cluster_id, replicate_id) not in control:
                raise ValueError("Ablation control stratum lacks a crossed trial/replicate cell.")

    estimate = float(
        np.mean([value for _, _, value in reference_values]) - np.mean([value for _, _, value in control_values])
    )
    rng = np.random.default_rng(config.bootstrap_seed + seed_offset)
    bootstrap = np.empty(config.bootstrap_resamples, dtype=np.float64)
    for index in range(config.bootstrap_resamples):
        sampled_reference = rng.choice(reference_ids, size=len(reference_ids), replace=True)
        sampled_control = rng.choice(control_ids, size=len(control_ids), replace=True)
        sampled_replicates = rng.choice(
            reference_replicates,
            size=len(reference_replicates),
            replace=True,
        )
        reference_sample = [
            value
            for cluster_id in sampled_reference
            for replicate_id in sampled_replicates
            for value in reference[(str(cluster_id), str(replicate_id))]
        ]
        control_sample = [
            value
            for cluster_id in sampled_control
            for replicate_id in sampled_replicates
            for value in control[(str(cluster_id), str(replicate_id))]
        ]
        bootstrap[index] = float(np.mean(reference_sample) - np.mean(control_sample))
    tail = (1.0 - config.confidence_level) / 2.0
    interval_low, interval_high = np.quantile(bootstrap, [tail, 1.0 - tail]).tolist()
    return TrialEvalAblationContrastV1(
        model_id=model_id,
        metric=metric,
        analysis_population=trialeval_metric_population_v1(metric),
        contrast_id=contrast_id,
        analysis_specification=analysis_specification,
        prompt_condition=prompt_condition,
        n_blocks=len(reference_values) + len(control_values),
        n_base_trial_clusters=len(set(reference_ids) | set(control_ids)),
        n_decoding_replicates=len(reference_replicates),
        estimate=estimate,
        interval_low=float(interval_low),
        interval_high=float(interval_high),
        confidence_level=config.confidence_level,
    )


def factorial_ablation_contrasts_v1(
    *,
    rows: tuple[TrialEvalAblationGradeRowV1, ...],
    config: TrialEvalAblationAnalysisConfigV1,
) -> tuple[TrialEvalAblationContrastV1, ...]:
    """Estimate paired assistance-dose, interface, and interaction effects."""

    eligible = tuple(
        row
        for row in rows
        if row.prompt_condition == "neutral"
        and (
            (row.submission_interface == "structured" and row.normalization_source == "direct_structured")
            or (row.submission_interface == "narrative" and row.normalization_source == "manual_masked")
        )
    )
    if not eligible:
        raise ValueError("Factorial ablation analysis has no direct-structured or manually transcribed rows.")
    results: list[TrialEvalAblationContrastV1] = []
    seed_offset = 0
    model_specifications = sorted({(row.model_id, row.analysis_specification) for row in eligible})
    for model_id, analysis_specification in model_specifications:
        model_rows = [
            row
            for row in eligible
            if row.model_id == model_id and row.analysis_specification == analysis_specification
        ]
        automated_rows = tuple(
            row
            for row in rows
            if row.model_id == model_id
            and row.analysis_specification == analysis_specification
            and row.prompt_condition == "neutral"
            and row.submission_interface == "narrative"
            and row.normalization_source == "automated_importer"
        )
        automated_by_block = {(row.task_id, row.replicate_id, row.procedure_assistance): row for row in automated_rows}
        if len(automated_by_block) != len(automated_rows):
            raise ValueError(f"Duplicate automated narrative grade row for model={model_id!r}.")
        blocks: dict[
            tuple[str, str],
            dict[tuple[str, str], TrialEvalAblationGradeRowV1],
        ] = {}
        for row in model_rows:
            block = blocks.setdefault((row.task_id, row.replicate_id), {})
            cell = (row.procedure_assistance, row.submission_interface)
            if cell in block:
                raise ValueError(
                    f"Duplicate factorial grade row for model={model_id!r}, block={block!r}, cell={cell!r}."
                )
            block[cell] = row
        expected = {
            ("output_contract_only", "structured"),
            ("output_contract_only", "narrative"),
            ("unordered_checklist", "structured"),
            ("unordered_checklist", "narrative"),
            ("ordered_sop", "structured"),
            ("ordered_sop", "narrative"),
        }
        if any(set(block) != expected for block in blocks.values()):
            raise ValueError(f"Factorial grade rows are incomplete for model={model_id!r}.")
        expected_automated_keys = {
            (task_id, replicate_id, assistance)
            for task_id, replicate_id in blocks
            for assistance in ("output_contract_only", "unordered_checklist", "ordered_sop")
        }
        if automated_by_block and set(automated_by_block) != expected_automated_keys:
            raise ValueError(
                f"Automated narrative normalization is incomplete for model={model_id!r}, "
                f"analysis_specification={analysis_specification!r}."
            )

        for metric in config.metrics:
            vectors: dict[str, list[tuple[str, str, float]]] = {
                "P1-P0": [],
                "P2-P1": [],
                "P2-P0": [],
                "R1-R0": [],
                "O1-O0": [],
                "I.interface_by_P1-P0": [],
                "I.interface_by_P2-P1": [],
            }
            for block in blocks.values():
                output_contract_structured = block[("output_contract_only", "structured")]
                output_contract_narrative = block[("output_contract_only", "narrative")]
                checklist_structured = block[("unordered_checklist", "structured")]
                checklist_narrative = block[("unordered_checklist", "narrative")]
                ordered_sop_structured = block[("ordered_sop", "structured")]
                ordered_sop_narrative = block[("ordered_sop", "narrative")]
                cell_rows = (
                    output_contract_structured,
                    output_contract_narrative,
                    checklist_structured,
                    checklist_narrative,
                    ordered_sop_structured,
                    ordered_sop_narrative,
                )
                cluster_ids = {row.base_trial_id for row in cell_rows}
                if len(cluster_ids) != 1:
                    raise ValueError("Factorial block rows disagree on base_trial_id.")
                cluster_id = next(iter(cluster_ids))
                cell_scores = {
                    "output_contract_structured": _metric_value(output_contract_structured, metric),
                    "output_contract_narrative": _metric_value(output_contract_narrative, metric),
                    "checklist_structured": _metric_value(checklist_structured, metric),
                    "checklist_narrative": _metric_value(checklist_narrative, metric),
                    "ordered_sop_structured": _metric_value(ordered_sop_structured, metric),
                    "ordered_sop_narrative": _metric_value(ordered_sop_narrative, metric),
                }
                if any(value is None for value in cell_scores.values()):
                    continue
                numeric_scores = {key: float(value) for key, value in cell_scores.items() if value is not None}
                replicate_id = output_contract_structured.replicate_id

                block_contrasts = {
                    "P1-P0": 0.5
                    * (
                        numeric_scores["checklist_structured"]
                        - numeric_scores["output_contract_structured"]
                        + numeric_scores["checklist_narrative"]
                        - numeric_scores["output_contract_narrative"]
                    ),
                    "P2-P1": 0.5
                    * (
                        numeric_scores["ordered_sop_structured"]
                        - numeric_scores["checklist_structured"]
                        + numeric_scores["ordered_sop_narrative"]
                        - numeric_scores["checklist_narrative"]
                    ),
                    "P2-P0": 0.5
                    * (
                        numeric_scores["ordered_sop_structured"]
                        - numeric_scores["output_contract_structured"]
                        + numeric_scores["ordered_sop_narrative"]
                        - numeric_scores["output_contract_narrative"]
                    ),
                    "R1-R0": (
                        numeric_scores["output_contract_structured"]
                        - numeric_scores["output_contract_narrative"]
                        + numeric_scores["checklist_structured"]
                        - numeric_scores["checklist_narrative"]
                        + numeric_scores["ordered_sop_structured"]
                        - numeric_scores["ordered_sop_narrative"]
                    )
                    / 3.0,
                    "I.interface_by_P1-P0": (
                        (numeric_scores["checklist_structured"] - numeric_scores["output_contract_structured"])
                        - (numeric_scores["checklist_narrative"] - numeric_scores["output_contract_narrative"])
                    ),
                    "I.interface_by_P2-P1": (
                        (numeric_scores["ordered_sop_structured"] - numeric_scores["checklist_structured"])
                        - (numeric_scores["ordered_sop_narrative"] - numeric_scores["checklist_narrative"])
                    ),
                }
                if automated_by_block:
                    automated = tuple(
                        automated_by_block[(row.task_id, row.replicate_id, row.procedure_assistance)]
                        for row in (
                            output_contract_structured,
                            checklist_structured,
                            ordered_sop_structured,
                        )
                    )
                    if {row.base_trial_id for row in automated} != {cluster_id}:
                        raise ValueError("Automated narrative rows disagree with their structured base trial.")
                    automated_scores = tuple(_metric_value(row, metric) for row in automated)
                    operational_deltas: list[float] = []
                    for structured, normalized in zip(
                        (
                            numeric_scores["output_contract_structured"],
                            numeric_scores["checklist_structured"],
                            numeric_scores["ordered_sop_structured"],
                        ),
                        automated_scores,
                        strict=True,
                    ):
                        if normalized is None:
                            break
                        operational_deltas.append(structured - normalized)
                    if len(operational_deltas) == 3:
                        block_contrasts["O1-O0"] = float(np.mean(operational_deltas))
                for contrast_id, value in block_contrasts.items():
                    vectors[contrast_id].append((cluster_id, replicate_id, value))
            for contrast_id, values in vectors.items():
                if not values:
                    continue
                results.append(
                    _clustered_contrast(
                        model_id=model_id,
                        metric=metric,
                        contrast_id=contrast_id,
                        values=values,
                        config=config,
                        seed_offset=seed_offset,
                        analysis_specification=analysis_specification,
                    )
                )
                seed_offset += 1
    return tuple(results)


def analysis_specification_contrasts_v1(
    *,
    rows: tuple[TrialEvalAblationGradeRowV1, ...],
    config: TrialEvalAblationAnalysisConfigV1,
) -> tuple[TrialEvalAblationContrastV1, ...]:
    """Estimate locked-SAP minus protocol-only effects within matched base trials."""

    eligible = tuple(
        row
        for row in rows
        if row.prompt_condition == "neutral"
        and (
            (row.submission_interface == "structured" and row.normalization_source == "direct_structured")
            or (row.submission_interface == "narrative" and row.normalization_source == "manual_masked")
        )
    )
    if {row.analysis_specification for row in eligible} != {"protocol_only", "locked_sap"}:
        return ()

    results: list[TrialEvalAblationContrastV1] = []
    seed_offset = TRIALEVAL_ANALYSIS_SPECIFICATION_SEED_OFFSET_V1
    for model_id in sorted({row.model_id for row in eligible}):
        model_rows = [row for row in eligible if row.model_id == model_id]
        blocks: dict[
            tuple[str, str, str, ProcedureAssistanceV1, TrialEvalSubmissionInterfaceV1],
            dict[str, TrialEvalAblationGradeRowV1],
        ] = {}
        for row in model_rows:
            if row.data_preparation == "raw_domains_declared_defect":
                continue
            pair_key = (
                row.base_trial_id,
                row.data_preparation,
                row.replicate_id,
                row.procedure_assistance,
                row.submission_interface,
            )
            cell = blocks.setdefault(pair_key, {})
            if row.analysis_specification in cell:
                raise ValueError(f"Duplicate analysis-specification row for paired block {pair_key!r}.")
            cell[row.analysis_specification] = row
        complete_blocks = {
            pair_key: cell for pair_key, cell in blocks.items() if set(cell) == {"protocol_only", "locked_sap"}
        }
        if not complete_blocks:
            continue

        for metric in config.metrics:
            by_treatment: dict[tuple[str, str, str], list[tuple[str, str, float]]] = defaultdict(list)
            for matched_key, cell in complete_blocks.items():
                base_trial_id, data_preparation, replicate_id, assistance, interface = matched_key
                sap_value = _metric_value(cell["locked_sap"], metric)
                protocol_value = _metric_value(cell["protocol_only"], metric)
                if sap_value is None or protocol_value is None:
                    continue
                effect = sap_value - protocol_value
                by_treatment[(data_preparation, assistance, interface)].append((base_trial_id, replicate_id, effect))
            for (treatment_data_preparation, assistance_id, interface_id), values in sorted(by_treatment.items()):
                results.append(
                    _clustered_contrast(
                        model_id=model_id,
                        metric=metric,
                        contrast_id=(f"sap_specification_{treatment_data_preparation}_{assistance_id}_{interface_id}"),
                        values=values,
                        config=config,
                        seed_offset=seed_offset,
                        analysis_specification=None,
                    )
                )
                seed_offset += 1
    return tuple(results)


def targeted_control_contrasts_v1(
    *,
    rows: tuple[TrialEvalAblationGradeRowV1, ...],
    config: TrialEvalAblationAnalysisConfigV1,
) -> tuple[TrialEvalAblationContrastV1, ...]:
    """Estimate targeted-control effects against neutral and placebo by applicability."""

    if any(row.submission_interface != "structured" for row in rows):
        raise ValueError("Targeted-control analysis accepts structured responses only.")
    capability_conditions = set(trialeval_capability_prompt_conditions_v1())
    targets = sorted({row.prompt_condition for row in rows if row.prompt_condition in capability_conditions})
    if not targets:
        raise ValueError("Targeted-control analysis has no targeted capability conditions.")
    results: list[TrialEvalAblationContrastV1] = []
    seed_offset = TRIALEVAL_TARGETED_CONTROL_SEED_OFFSET_V1
    model_specifications = sorted({(row.model_id, row.analysis_specification) for row in rows})
    for model_id, analysis_specification in model_specifications:
        model_rows = [
            row for row in rows if row.model_id == model_id and row.analysis_specification == analysis_specification
        ]
        blocks: dict[tuple[str, str], dict[str, TrialEvalAblationGradeRowV1]] = {}
        for row in model_rows:
            block = blocks.setdefault((row.task_id, row.replicate_id), {})
            if row.prompt_condition in block:
                raise ValueError("Duplicate targeted-control grade row within a task replicate.")
            block[row.prompt_condition] = row
        required = set(targets) | {"neutral", "placebo_deliberation"}
        if any(set(block) != required for block in blocks.values()):
            raise ValueError(f"Targeted-control grade rows are incomplete for model={model_id!r}.")

        for target in targets:
            effects: dict[tuple[TrialEvalAblationMetricV1, str, str], list[tuple[str, str, float]]] = {}
            for applicability in ("applicable", "mismatched", "inapplicable"):
                selected = [
                    block for block in blocks.values() if block[target].targeted_applicability == applicability
                ]
                if not selected:
                    raise ValueError(
                        f"Targeted condition {target!r} has no {applicability!r} blocks for model={model_id!r}."
                    )
                for metric in config.metrics:
                    for comparator in ("neutral", "placebo_deliberation"):
                        values: list[tuple[str, str, float]] = []
                        for block in selected:
                            target_row = block[target]
                            comparator_row = block[comparator]
                            if target_row.base_trial_id != comparator_row.base_trial_id:
                                raise ValueError("Targeted and comparator rows disagree on base_trial_id.")
                            target_value = _metric_value(target_row, metric)
                            comparator_value = _metric_value(comparator_row, metric)
                            if target_value is None or comparator_value is None:
                                continue
                            values.append(
                                (
                                    target_row.base_trial_id,
                                    target_row.replicate_id,
                                    target_value - comparator_value,
                                )
                            )
                        if not values:
                            continue
                        effects[(metric, comparator, applicability)] = values
                        results.append(
                            _clustered_contrast(
                                model_id=model_id,
                                metric=metric,
                                contrast_id=f"targeted_vs_{comparator}",
                                prompt_condition=target,
                                applicability=applicability,
                                values=values,
                                config=config,
                                seed_offset=seed_offset,
                                analysis_specification=analysis_specification,
                            )
                        )
                        seed_offset += 1
            for metric in config.metrics:
                for comparator in ("neutral", "placebo_deliberation"):
                    reference = effects.get((metric, comparator, "applicable"))
                    if reference is None:
                        continue
                    for control_applicability in ("mismatched", "inapplicable"):
                        control = effects.get((metric, comparator, control_applicability))
                        if control is None:
                            continue
                        results.append(
                            _clustered_stratum_difference(
                                model_id=model_id,
                                metric=metric,
                                contrast_id=(f"specificity_vs_{control_applicability}_adjusted_for_{comparator}"),
                                reference_values=reference,
                                control_values=control,
                                config=config,
                                seed_offset=seed_offset,
                                prompt_condition=target,
                                analysis_specification=analysis_specification,
                            )
                        )
                        seed_offset += 1
    return tuple(results)


__all__ = [
    "factorial_ablation_arm_summaries_v1",
    "factorial_ablation_contrasts_v1",
    "factorial_observable_contrasts_v1",
    "targeted_control_contrasts_v1",
]
