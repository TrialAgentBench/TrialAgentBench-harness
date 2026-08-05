"""Post-score evaluator joins for TrialEval experiment rows."""

from __future__ import annotations

from typing import Literal

from trialagentbench_harness.contracts.experiments import (
    TrialEvalAblationEndpointRowV1,
    TrialEvalAblationEvaluatorLabelsV1,
    TrialEvalAblationGradeRowV1,
    TrialEvalTargetedApplicabilityLabelV1,
    trialeval_capability_prompt_conditions_v1,
)


def join_ablation_evaluator_labels_v1(
    *,
    endpoints: tuple[TrialEvalAblationEndpointRowV1, ...],
    evaluator_labels: TrialEvalAblationEvaluatorLabelsV1,
    design: Literal["factorial_interface", "targeted_control"],
) -> tuple[TrialEvalAblationGradeRowV1, ...]:
    """Join evaluator task identity and design-required applicability after scoring."""

    if not endpoints:
        raise ValueError("Ablation label join requires at least one scored endpoint.")
    endpoint_keys = [(row.assignment_id, row.normalization_source) for row in endpoints]
    if len(endpoint_keys) != len(set(endpoint_keys)):
        raise ValueError("Ablation scored endpoints contain duplicate assignment/normalization rows.")
    label_index: dict[tuple[str, str], TrialEvalTargetedApplicabilityLabelV1] = {
        (row.task_id, row.prompt_condition): row for row in evaluator_labels.labels
    }
    task_identities = {row.task_id: row for row in evaluator_labels.task_identities}
    endpoint_task_ids = {row.task_id for row in endpoints}
    missing_task_ids = sorted(endpoint_task_ids.difference(task_identities))
    if missing_task_ids:
        raise ValueError(f"Evaluator task identities do not contain tasks: {missing_task_ids!r}.")
    if design == "targeted_control":
        capability_conditions = set(trialeval_capability_prompt_conditions_v1())
        required_keys = {(task_id, condition) for task_id in endpoint_task_ids for condition in capability_conditions}
        missing_keys = sorted(required_keys.difference(label_index))
        if missing_keys:
            raise ValueError(f"Targeted-control evaluator labels are incomplete: {missing_keys!r}.")

    joined: list[TrialEvalAblationGradeRowV1] = []
    capability_conditions = set(trialeval_capability_prompt_conditions_v1())
    for endpoint in sorted(endpoints, key=lambda row: row.assignment_id):
        task_identity = task_identities.get(endpoint.task_id)
        if task_identity is None:
            raise ValueError(f"Evaluator labels do not contain task {endpoint.task_id!r}.")
        if (
            endpoint.context_tier != task_identity.context_tier
            or endpoint.data_preparation != task_identity.data_preparation
            or endpoint.analysis_specification != task_identity.analysis_specification
        ):
            raise ValueError("Scored endpoint evidence factors disagree with evaluator task identity.")
        applicability = None
        if design == "targeted_control" and endpoint.prompt_condition in capability_conditions:
            matched_label = label_index.get((endpoint.task_id, endpoint.prompt_condition))
            if matched_label is None:
                raise AssertionError("Targeted applicability completeness check failed.")
            applicability = matched_label.applicability
        joined.append(
            TrialEvalAblationGradeRowV1(
                assignment_id=endpoint.assignment_id,
                task_id=endpoint.task_id,
                base_trial_id=task_identity.base_trial_id,
                regime_cell_id=task_identity.regime_cell_id,
                evaluation_series_id=task_identity.evaluation_series_id,
                design_tier=task_identity.design_tier,
                design_subtype=task_identity.design_subtype,
                assumption_tier=task_identity.assumption_tier,
                context_tier=endpoint.context_tier,
                data_preparation=endpoint.data_preparation,
                analysis_specification=endpoint.analysis_specification,
                model_id=endpoint.model_id,
                replicate_id=endpoint.replicate_id,
                procedure_assistance=endpoint.procedure_assistance,
                prompt_condition=endpoint.prompt_condition,
                submission_interface=endpoint.submission_interface,
                normalization_source=endpoint.normalization_source,
                normalization_status=endpoint.normalization_status,
                normalization_failure_reason=endpoint.normalization_failure_reason,
                omitted_required_deliverables=endpoint.omitted_required_deliverables,
                primary_failure_code=endpoint.primary_failure_code,
                targeted_applicability=applicability,
                usable_primary=endpoint.usable_primary,
                route_match=endpoint.route_match,
                obligations_met=endpoint.obligations_met,
                credit_eligible_route_count=endpoint.credit_eligible_route_count,
                numeric_result_available=endpoint.numeric_result_available,
                primary_uncertainty_valid=endpoint.primary_uncertainty_valid,
                primary_interval_agreement=endpoint.primary_interval_agreement,
                result_match=endpoint.result_match,
                numeric_absolute_error=endpoint.numeric_absolute_error,
                numeric_tolerance_ratio=endpoint.numeric_tolerance_ratio,
                primary_analysis_conforms=endpoint.primary_analysis_conforms,
                planning_applicable=endpoint.planning_applicable,
                planning_valid=endpoint.planning_valid,
                planning_usable_with_primary=endpoint.planning_usable_with_primary,
                planning_achieved_power=endpoint.planning_achieved_power,
                planning_power_shortfall=endpoint.planning_power_shortfall,
                planning_underpowered=endpoint.planning_underpowered,
                planning_proportional_participant_deviation=(endpoint.planning_proportional_participant_deviation),
                planning_log_sample_size_ratio=endpoint.planning_log_sample_size_ratio,
                planning_event_shortage=endpoint.planning_event_shortage,
                planning_excess_events=endpoint.planning_excess_events,
                planning_excess_participants=endpoint.planning_excess_participants,
                planning_participant_shortage=endpoint.planning_participant_shortage,
            )
        )
    return tuple(joined)


__all__ = ["join_ablation_evaluator_labels_v1"]
