"""Aggregate TrialDev decisions without denominator loss or score compensation."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np

from trialagentbench_harness.contracts.core.config import ReasoningEffortV1
from trialagentbench_harness.contracts.trialdev.metrics import (
    TRIALDEV_CAPABILITY_IDS_V1,
    TrialDevAnalysisClassificationCountV1,
    TrialDevCalibrationArmV1,
    TrialDevCalibrationSelectionV1,
    TrialDevCapabilityIdV1,
    TrialDevClusterIntervalV1,
    TrialDevConditionComparisonV1,
    TrialDevDenominatorCountsV1,
    TrialDevMetricPortfolioV1,
    TrialDevNamedRateV1,
    TrialDevPairedDifferenceV1,
    TrialDevProgrammeAssessmentV1,
    TrialDevRateMetricV1,
    TrialDevSecondarySummaryV1,
    TrialDevStreamComparisonV1,
    TrialDevStreamMetricSummaryV1,
)
from trialagentbench_harness.contracts.trialdev.programme import TrialDevStreamIdV1
from trialagentbench_harness.contracts.trialdev.scientific_grade import (
    TRIALDEV_ANALYSIS_CLASSIFICATIONS_V1,
    TRIALDEV_SCIENTIFIC_RESPONSIBILITIES_V1,
)

TRIALDEV_METRIC_CONFIDENCE_LEVEL_V1 = 0.95
TRIALDEV_METRIC_BOOTSTRAP_RESAMPLES_V1 = 10_000
TRIALDEV_METRIC_BOOTSTRAP_SEED_V1 = 20_260_802


def _seed(*, namespace: str, root_seed: int) -> int:
    digest = hashlib.sha256(f"trialdev-metric-v1:{root_seed}:{namespace}".encode()).hexdigest()
    return int(digest[:16], 16)


def _cluster_interval(
    records: Sequence[tuple[str, float]],
    *,
    namespace: str,
    confidence_level: float,
    resamples: int,
    root_seed: int,
) -> TrialDevClusterIntervalV1 | None:
    clusters: dict[str, list[float]] = defaultdict(list)
    for cluster_id, value in records:
        clusters[cluster_id].append(value)
    cluster_ids = tuple(sorted(clusters))
    if len(cluster_ids) < 2 or not records:
        return None
    rng = np.random.default_rng(_seed(namespace=namespace, root_seed=root_seed))
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        values = [value for cluster_id in sampled for value in clusters[str(cluster_id)]]
        estimates[index] = float(np.mean(values))
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(estimates, (tail, 1.0 - tail))
    return TrialDevClusterIntervalV1(
        confidence_level=confidence_level,
        resamples=resamples,
        seed=_seed(namespace=namespace, root_seed=root_seed),
        cluster_count=len(cluster_ids),
        lower=float(lower),
        upper=float(upper),
    )


def _rate(
    records: Sequence[tuple[str, bool]],
    *,
    namespace: str,
    confidence_level: float,
    resamples: int,
    root_seed: int,
) -> TrialDevRateMetricV1:
    numerator = sum(value for _, value in records)
    denominator = len(records)
    interval = _cluster_interval(
        [(cluster, float(value)) for cluster, value in records],
        namespace=namespace,
        confidence_level=confidence_level,
        resamples=resamples,
        root_seed=root_seed,
    )
    return TrialDevRateMetricV1(
        numerator=numerator,
        denominator=denominator,
        finite_estimate=None if denominator == 0 else numerator / denominator,
        cluster_interval=interval,
    )


def _checkpoint_success(programme: TrialDevProgrammeAssessmentV1, checkpoint_index: int) -> bool:
    checkpoint = programme.checkpoints[checkpoint_index]
    if checkpoint.outcome.reach_status != "reached" or checkpoint.outcome.execution_status not in {
        "completed",
        "model_noncompletion",
    }:
        raise ValueError("Checkpoint success requires a completed or model-noncompleted scheduled checkpoint.")
    return all(lane.outcome == "accepted" for lane in checkpoint.lanes) and all(
        capability.outcome in {"passed", "not_applicable"} for capability in checkpoint.capabilities
    )


def _complete_chain(programme: TrialDevProgrammeAssessmentV1) -> bool:
    if programme.execution_status != "completed":
        return False
    reached = tuple(
        (index, item)
        for index, item in enumerate(programme.checkpoints)
        if item.outcome.reach_status == "reached" and item.outcome.execution_status == "completed"
    )
    terminal = tuple(item.terminal_record_valid for _, item in reached if item.terminal_record_valid is not None)
    return bool(terminal == (True,) and all(_checkpoint_success(programme, index) for index, _ in reached))


def _capability_programme_value(
    programme: TrialDevProgrammeAssessmentV1,
    capability_id: TrialDevCapabilityIdV1,
) -> bool | None:
    outcomes = tuple(
        assessment.outcome
        for checkpoint in programme.checkpoints
        if checkpoint.outcome.reach_status == "reached" and checkpoint.outcome.execution_status == "completed"
        for assessment in checkpoint.capabilities
        if assessment.capability_id == capability_id and assessment.outcome != "not_applicable"
    )
    return None if not outcomes else all(outcome == "passed" for outcome in outcomes)


def _mean(values: Iterable[float]) -> float | None:
    rows = tuple(values)
    return None if not rows else float(np.mean(rows))


def _secondary_summary(programmes: Sequence[TrialDevProgrammeAssessmentV1]) -> TrialDevSecondarySummaryV1:
    outcomes = tuple(item.secondary_outcomes for item in programmes)
    usd = tuple(item.provider_reported_usd for item in outcomes if item.provider_reported_usd is not None)
    resources = tuple(item.programme_resource_units for item in outcomes if item.programme_resource_units is not None)
    consequences = tuple(item.downstream_consequence for item in outcomes if item.downstream_consequence is not None)
    policy_values = tuple(item.policy_value for item in outcomes if item.policy_value is not None)
    regrets = tuple(item.policy_regret for item in outcomes if item.policy_regret is not None)
    return TrialDevSecondarySummaryV1(
        programme_count=len(programmes),
        elapsed_seconds_mean=_mean(item.elapsed_seconds for item in outcomes),
        provider_calls_mean=_mean(float(item.provider_calls) for item in outcomes),
        agent_turns_mean=_mean(float(item.agent_turns) for item in outcomes),
        correction_count_mean=_mean(float(item.correction_count) for item in outcomes),
        execute_code_calls_mean=_mean(float(item.execute_code_calls) for item in outcomes),
        inspect_data_calls_mean=_mean(float(item.inspect_data_calls) for item in outcomes),
        total_tokens_mean=_mean(float(item.prompt_tokens + item.completion_tokens) for item in outcomes),
        provider_reported_usd_available=len(usd),
        provider_reported_usd_mean=_mean(float(value) for value in usd),
        programme_resource_units_available=len(resources),
        programme_resource_units_mean=_mean(float(value) for value in resources),
        switch_rate=_mean(float(item.switch_count) for item in outcomes),
        early_switch_count=sum(item.switch_timing == "early" for item in outcomes),
        late_switch_count=sum(item.switch_timing == "late" for item in outcomes),
        downstream_consequence_available=len(consequences),
        downstream_consequence_mean=_mean(float(value) for value in consequences),
        policy_value_available=len(policy_values),
        policy_value_mean=_mean(float(value) for value in policy_values),
        policy_regret_available=len(regrets),
        policy_regret_mean=_mean(float(value) for value in regrets),
    )


def _stream_summary(
    programmes: Sequence[TrialDevProgrammeAssessmentV1],
    *,
    confidence_level: float,
    resamples: int,
    root_seed: int,
) -> TrialDevStreamMetricSummaryV1:
    model_ids = {item.model_id for item in programmes}
    condition_ids = {item.condition_id for item in programmes}
    replicate_ids = {item.request_replicate_id for item in programmes}
    reasoning_efforts = {item.reasoning_effort for item in programmes}
    procedure_assistance = {item.procedure_assistance for item in programmes}
    maximum_turns = {item.maximum_turns_per_step for item in programmes}
    maximum_submission_attempts = {item.maximum_submission_attempts for item in programmes}
    tool_choices = {item.tool_choice for item in programmes}
    task_seeds = {item.task_materialization_seed for item in programmes}
    stream_ids = {item.stream_id for item in programmes}
    if any(
        len(values) != 1
        for values in (
            model_ids,
            condition_ids,
            replicate_ids,
            reasoning_efforts,
            procedure_assistance,
            maximum_turns,
            maximum_submission_attempts,
            tool_choices,
            task_seeds,
            stream_ids,
        )
    ):
        raise ValueError("A TrialDev stream summary requires one exact experiment condition and stream.")
    model_id = next(iter(model_ids))
    condition_id = next(iter(condition_ids))
    request_replicate_id = next(iter(replicate_ids))
    reasoning_effort = next(iter(reasoning_efforts))
    assistance = next(iter(procedure_assistance))
    maximum_turns_per_step = next(iter(maximum_turns))
    maximum_attempts = next(iter(maximum_submission_attempts))
    tool_choice = next(iter(tool_choices))
    task_materialization_seed = next(iter(task_seeds))
    stream_id = next(iter(stream_ids))
    checkpoints = tuple((programme, checkpoint) for programme in programmes for checkpoint in programme.checkpoints)
    scheduled = tuple(
        (programme, checkpoint)
        for programme, checkpoint in checkpoints
        if checkpoint.outcome.reach_status == "reached"
    )
    reached = tuple(
        (programme, checkpoint)
        for programme, checkpoint in scheduled
        if checkpoint.outcome.execution_status == "completed"
    )
    model_attempts = tuple(
        (programme, checkpoint)
        for programme, checkpoint in scheduled
        if checkpoint.outcome.execution_status == "model_noncompletion"
    )
    model_lanes = (*reached, *model_attempts)
    lanes = tuple((programme, lane) for programme, checkpoint in model_lanes for lane in checkpoint.lanes)
    submitted = tuple((programme, lane) for programme, lane in lanes if lane.outcome != "missing")
    denominators = TrialDevDenominatorCountsV1(
        programmes=len(programmes),
        scheduled=len(scheduled),
        reached=len(reached),
        structural_nonreach=sum(
            checkpoint.outcome.reach_status == "structural_nonreach" for _, checkpoint in checkpoints
        ),
        submitted=len(submitted),
        accepted=sum(lane.outcome == "accepted" for _, lane in lanes),
        invalid=sum(lane.outcome == "invalid" for _, lane in lanes),
        missing=sum(lane.outcome == "missing" for _, lane in lanes),
        model_noncompletion=len(model_attempts),
        infrastructure_failure=sum(
            checkpoint.outcome.execution_status == "infrastructure_failure" for _, checkpoint in checkpoints
        ),
    )
    capability_metrics = []
    for capability_id in TRIALDEV_CAPABILITY_IDS_V1:
        records = [
            (programme.scenario_family_id, assessment.outcome == "passed")
            for programme, checkpoint in reached
            for assessment in checkpoint.capabilities
            if assessment.capability_id == capability_id and assessment.outcome != "not_applicable"
        ]
        capability_metrics.append(
            TrialDevNamedRateV1(
                metric_id=capability_id,
                estimate=_rate(
                    records,
                    namespace=f"{model_id}:{stream_id}:capability:{capability_id}",
                    confidence_level=confidence_level,
                    resamples=resamples,
                    root_seed=root_seed,
                ),
            )
        )
    scientific_responsibilities = []
    for responsibility_id in TRIALDEV_SCIENTIFIC_RESPONSIBILITIES_V1:
        responsibility_records: list[tuple[str, bool]] = []
        for programme, checkpoint in reached:
            assessment = checkpoint.scientific_assessment
            if assessment is None:
                raise ValueError("A completed TrialDev checkpoint lacks its scientific assessment.")
            if responsibility_id == "decision_complete":
                passed: bool | None = assessment.decision_complete
            elif responsibility_id == "resource_within_budget":
                passed = None if assessment.resources == "not_assessed" else assessment.resources == "within_budget"
            else:
                status = getattr(assessment, responsibility_id)
                passed = None if status in {"not_applicable", "not_assessed"} else status == "passed"
            if passed is not None:
                responsibility_records.append((programme.scenario_family_id, passed))
        scientific_responsibilities.append(
            TrialDevNamedRateV1(
                metric_id=responsibility_id,
                estimate=_rate(
                    responsibility_records,
                    namespace=f"{model_id}:{stream_id}:scientific:{responsibility_id}",
                    confidence_level=confidence_level,
                    resamples=resamples,
                    root_seed=root_seed,
                ),
            )
        )
    classification_values = tuple(
        checkpoint.scientific_assessment.analysis_classification
        for _, checkpoint in reached
        if checkpoint.scientific_assessment is not None
    )
    analysis_classifications = tuple(
        TrialDevAnalysisClassificationCountV1(
            classification=classification,
            count=sum(value == classification for value in classification_values),
            denominator=len(classification_values),
        )
        for classification in TRIALDEV_ANALYSIS_CLASSIFICATIONS_V1
    )
    lane_metrics = []
    for lane_id in sorted({lane.lane_id for _, lane in lanes}):
        records = [
            (programme.scenario_family_id, lane.outcome == "accepted")
            for programme, lane in submitted
            if lane.lane_id == lane_id
        ]
        lane_metrics.append(
            TrialDevNamedRateV1(
                metric_id=lane_id,
                estimate=_rate(
                    records,
                    namespace=f"{model_id}:{stream_id}:lane:{lane_id}",
                    confidence_level=confidence_level,
                    resamples=resamples,
                    root_seed=root_seed,
                ),
            )
        )
    checkpoint_records = [
        (programme.scenario_family_id, _checkpoint_success(programme, index))
        for programme in programmes
        for index, checkpoint in enumerate(programme.checkpoints)
        if checkpoint.outcome.reach_status == "reached"
        and checkpoint.outcome.execution_status in {"completed", "model_noncompletion"}
    ]
    chain_records = [(item.scenario_family_id, _complete_chain(item)) for item in programmes]
    execution_records = [(item.scenario_family_id, item.execution_status == "completed") for item in programmes]
    return TrialDevStreamMetricSummaryV1(
        model_id=model_id,
        condition_id=condition_id,
        request_replicate_id=request_replicate_id,
        reasoning_effort=reasoning_effort,
        procedure_assistance=assistance,
        maximum_turns_per_step=maximum_turns_per_step,
        maximum_submission_attempts=maximum_attempts,
        tool_choice=tool_choice,
        task_materialization_seed=task_materialization_seed,
        stream_id=stream_id,
        denominators=denominators,
        capabilities=tuple(capability_metrics),
        scientific_responsibilities=tuple(scientific_responsibilities),
        analysis_classifications=analysis_classifications,
        lanes=tuple(lane_metrics),
        checkpoint_success=_rate(
            checkpoint_records,
            namespace=f"{model_id}:{stream_id}:checkpoint_success",
            confidence_level=confidence_level,
            resamples=resamples,
            root_seed=root_seed,
        ),
        complete_chain_success=_rate(
            chain_records,
            namespace=f"{model_id}:{stream_id}:complete_chain_success",
            confidence_level=confidence_level,
            resamples=resamples,
            root_seed=root_seed,
        ),
        execution_completion=_rate(
            execution_records,
            namespace=f"{model_id}:{stream_id}:execution_completion",
            confidence_level=confidence_level,
            resamples=resamples,
            root_seed=root_seed,
        ),
        secondary=_secondary_summary(programmes),
    )


def summarize_trialdev_metrics_v1(
    programmes: Sequence[TrialDevProgrammeAssessmentV1],
    *,
    confidence_level: float = TRIALDEV_METRIC_CONFIDENCE_LEVEL_V1,
    bootstrap_resamples: int = TRIALDEV_METRIC_BOOTSTRAP_RESAMPLES_V1,
    bootstrap_seed: int = TRIALDEV_METRIC_BOOTSTRAP_SEED_V1,
) -> TrialDevMetricPortfolioV1:
    """Summarize exact finite results with scenario-cluster uncertainty."""

    if not programmes:
        raise ValueError("TrialDev metric aggregation requires at least one programme.")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("TrialDev confidence_level must lie between 0.5 and 1.0.")
    if bootstrap_resamples < 1 or bootstrap_seed < 0:
        raise ValueError("TrialDev bootstrap resamples and seed must be positive and non-negative.")
    grouped: dict[tuple[str, str, str, int, str], list[TrialDevProgrammeAssessmentV1]] = defaultdict(list)
    for programme in programmes:
        grouped[
            (
                programme.condition_id,
                programme.request_replicate_id,
                programme.model_id,
                programme.task_materialization_seed,
                programme.stream_id,
            )
        ].append(programme)
    summaries = tuple(
        _stream_summary(
            grouped[key],
            confidence_level=confidence_level,
            resamples=bootstrap_resamples,
            root_seed=bootstrap_seed,
        )
        for key in sorted(grouped)
    )
    return TrialDevMetricPortfolioV1(streams=summaries)


def _paired_cluster_interval(
    records: Sequence[tuple[str, float]],
    *,
    namespace: str,
    confidence_level: float,
    resamples: int,
    root_seed: int,
) -> TrialDevClusterIntervalV1 | None:
    return _cluster_interval(
        records,
        namespace=namespace,
        confidence_level=confidence_level,
        resamples=resamples,
        root_seed=root_seed,
    )


def _capability_comparison_value(
    programme: TrialDevProgrammeAssessmentV1,
    metric_id: str,
) -> bool | None:
    if metric_id == "complete_chain_success":
        return _complete_chain(programme)
    if metric_id == "execution_completion":
        return bool(programme.execution_status == "completed")
    if metric_id not in TRIALDEV_CAPABILITY_IDS_V1:
        raise ValueError(f"Unknown TrialDev comparison metric: {metric_id!r}.")
    capability_id = cast(TrialDevCapabilityIdV1, metric_id)
    return _capability_programme_value(programme, capability_id)


def compare_trialdev_conditions_v1(
    programmes: Sequence[TrialDevProgrammeAssessmentV1],
    *,
    reference_condition_id: str,
    intervention_condition_id: str,
    confidence_level: float = TRIALDEV_METRIC_CONFIDENCE_LEVEL_V1,
    bootstrap_resamples: int = TRIALDEV_METRIC_BOOTSTRAP_RESAMPLES_V1,
    bootstrap_seed: int = TRIALDEV_METRIC_BOOTSTRAP_SEED_V1,
) -> TrialDevConditionComparisonV1:
    """Compare two experiment conditions on exact scenario-paired views."""

    if reference_condition_id == intervention_condition_id:
        raise ValueError("TrialDev comparison requires two distinct condition identities.")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("TrialDev confidence_level must lie between 0.5 and 1.0.")
    if bootstrap_resamples < 1 or bootstrap_seed < 0:
        raise ValueError("TrialDev bootstrap resamples and seed must be positive and non-negative.")
    selected = tuple(
        item for item in programmes if item.condition_id in {reference_condition_id, intervention_condition_id}
    )
    keyed: dict[tuple[str, str, str, str, str], TrialDevProgrammeAssessmentV1] = {}
    for item in selected:
        key = (
            item.stream_id,
            item.evaluation_unit_id,
            item.objective_variant_id,
            item.policy_variant_id,
            item.condition_id,
        )
        if key in keyed:
            raise ValueError("TrialDev model comparison contains duplicate paired views.")
        keyed[key] = item
    stream_ids = sorted({item.stream_id for item in selected})
    outputs = []
    for stream_id in stream_ids:
        reference_keys = {
            (unit, objective, policy)
            for stream, unit, objective, policy, condition in keyed
            if stream == stream_id and condition == reference_condition_id
        }
        intervention_keys = {
            (unit, objective, policy)
            for stream, unit, objective, policy, condition in keyed
            if stream == stream_id and condition == intervention_condition_id
        }
        if not reference_keys or reference_keys != intervention_keys:
            raise ValueError("TrialDev model comparison requires complete matched evaluation views per stream.")
        metrics = ("complete_chain_success", "execution_completion", *TRIALDEV_CAPABILITY_IDS_V1)
        differences = []
        for metric_id in metrics:
            pairs: list[tuple[str, float, float]] = []
            for unit, objective, policy in sorted(reference_keys):
                reference = keyed[(stream_id, unit, objective, policy, reference_condition_id)]
                intervention = keyed[(stream_id, unit, objective, policy, intervention_condition_id)]
                if reference.scenario_family_id != intervention.scenario_family_id:
                    raise ValueError("Paired TrialDev views must share scenario_family_id.")
                if (
                    reference.request_replicate_id != intervention.request_replicate_id
                    or reference.task_materialization_seed != intervention.task_materialization_seed
                ):
                    raise ValueError(
                        "Paired TrialDev views must share request replicate and task materialization seed."
                    )
                reference_value = _capability_comparison_value(reference, metric_id)
                intervention_value = _capability_comparison_value(intervention, metric_id)
                # Capability outcomes are observable only after the relevant analysis
                # completes.  Preserve that missingness instead of turning an
                # execution failure into a statistical-capability failure.  The
                # metric-specific pair count exposes the resulting denominator.
                if reference_value is None or intervention_value is None:
                    continue
                pairs.append(
                    (
                        reference.scenario_family_id,
                        float(reference_value),
                        float(intervention_value),
                    )
                )
            if not pairs:
                continue
            paired_values = [(cluster, intervention - reference) for cluster, reference, intervention in pairs]
            differences.append(
                TrialDevPairedDifferenceV1(
                    metric_id=metric_id,
                    pair_count=len(pairs),
                    scenario_family_count=len({cluster for cluster, _, _ in pairs}),
                    reference_mean=float(np.mean([reference for _, reference, _ in pairs])),
                    intervention_mean=float(np.mean([intervention for _, _, intervention in pairs])),
                    paired_difference=float(np.mean([value for _, value in paired_values])),
                    cluster_interval=_paired_cluster_interval(
                        paired_values,
                        namespace=f"comparison:{stream_id}:{metric_id}",
                        confidence_level=confidence_level,
                        resamples=bootstrap_resamples,
                        root_seed=bootstrap_seed,
                    ),
                )
            )
        outputs.append(TrialDevStreamComparisonV1(stream_id=stream_id, metrics=tuple(differences)))
    if not outputs:
        raise ValueError("TrialDev model comparison found no matched stream views.")
    return TrialDevConditionComparisonV1(
        reference_condition_id=reference_condition_id,
        intervention_condition_id=intervention_condition_id,
        streams=tuple(outputs),
    )


def _calibration_arm(
    programmes: Sequence[TrialDevProgrammeAssessmentV1],
) -> TrialDevCalibrationArmV1:
    """Summarize one complete paired calibration arm."""

    condition_ids = {item.condition_id for item in programmes}
    assistance = {item.procedure_assistance for item in programmes}
    turn_limits = {item.maximum_turns_per_step for item in programmes}
    submission_limits = {item.maximum_submission_attempts for item in programmes}
    tool_choices = {item.tool_choice for item in programmes}
    if any(len(values) != 1 for values in (condition_ids, assistance, turn_limits, submission_limits, tool_choices)):
        raise ValueError(
            "A calibration arm requires one condition, assistance level, turn ceiling, submission limit, and tool policy."
        )
    costs = tuple(item.secondary_outcomes.provider_reported_usd for item in programmes)
    if any(cost is None for cost in costs):
        raise ValueError("Calibration selection requires provider-reported cost for every scheduled programme.")
    scheduled = tuple(
        (programme, index)
        for programme in programmes
        for index, checkpoint in enumerate(programme.checkpoints)
        if checkpoint.outcome.reach_status == "reached"
    )
    if not scheduled:
        raise ValueError("A calibration arm requires at least one scheduled checkpoint.")
    outcomes = tuple(item.secondary_outcomes for item in programmes)
    return TrialDevCalibrationArmV1(
        condition_id=next(iter(condition_ids)),
        procedure_assistance=next(iter(assistance)),
        maximum_turns_per_step=next(iter(turn_limits)),
        maximum_submission_attempts=next(iter(submission_limits)),
        tool_choice=next(iter(tool_choices)),
        programme_count=len(programmes),
        completed_programmes=sum(item.execution_status == "completed" for item in programmes),
        complete_chain_success_rate=float(np.mean([_complete_chain(item) for item in programmes])),
        scheduled_checkpoint_count=len(scheduled),
        checkpoint_success_rate=float(
            np.mean(
                [
                    (
                        _checkpoint_success(programme, index)
                        if programme.checkpoints[index].outcome.execution_status
                        in {"completed", "model_noncompletion"}
                        else False
                    )
                    for programme, index in scheduled
                ]
            )
        ),
        correction_count_mean=float(np.mean([item.correction_count for item in outcomes])),
        agent_turns_mean=float(np.mean([item.agent_turns for item in outcomes])),
        elapsed_seconds_mean=float(np.mean([item.elapsed_seconds for item in outcomes])),
        total_tokens_mean=float(np.mean([item.prompt_tokens + item.completion_tokens for item in outcomes])),
        provider_reported_usd_mean=float(np.mean(cast(tuple[float, ...], costs))),
    )


def _dominates(left: TrialDevCalibrationArmV1, right: TrialDevCalibrationArmV1) -> bool:
    quality_left = (left.complete_chain_success_rate, left.checkpoint_success_rate)
    quality_right = (right.complete_chain_success_rate, right.checkpoint_success_rate)
    resources_left = (
        left.maximum_turns_per_step,
        left.correction_count_mean,
        left.agent_turns_mean,
        left.elapsed_seconds_mean,
        left.total_tokens_mean,
        left.provider_reported_usd_mean,
    )
    resources_right = (
        right.maximum_turns_per_step,
        right.correction_count_mean,
        right.agent_turns_mean,
        right.elapsed_seconds_mean,
        right.total_tokens_mean,
        right.provider_reported_usd_mean,
    )
    weakly_better = all(a >= b for a, b in zip(quality_left, quality_right, strict=True)) and all(
        a <= b for a, b in zip(resources_left, resources_right, strict=True)
    )
    strictly_better = any(a > b for a, b in zip(quality_left, quality_right, strict=True)) or any(
        a < b for a, b in zip(resources_left, resources_right, strict=True)
    )
    return weakly_better and strictly_better


def select_trialdev_calibration_v1(
    programmes: Sequence[TrialDevProgrammeAssessmentV1],
    *,
    condition_ids: Sequence[str],
) -> TrialDevCalibrationSelectionV1:
    """Select one bounded turn-and-assistance setting from paired exact results."""

    requested = tuple(condition_ids)
    if len(requested) < 2 or len(requested) != len(set(requested)):
        raise ValueError("Calibration selection requires at least two unique condition identities.")
    selected = tuple(item for item in programmes if item.condition_id in requested)
    if {item.condition_id for item in selected} != set(requested):
        raise ValueError("Calibration results do not contain every requested condition.")
    common_fields = (
        {item.model_id for item in selected},
        {item.reasoning_effort for item in selected},
        {item.request_replicate_id for item in selected},
        {item.task_materialization_seed for item in selected},
        {item.stream_id for item in selected},
    )
    if any(len(values) != 1 for values in common_fields):
        raise ValueError("Calibration arms must share model, reasoning, replicate, task seed, and stream.")
    grouped = {
        condition_id: tuple(item for item in selected if item.condition_id == condition_id)
        for condition_id in requested
    }
    paired_keys = {
        condition_id: tuple(
            sorted(
                (item.evaluation_unit_id, item.objective_variant_id, item.policy_variant_id)
                for item in condition_programmes
            )
        )
        for condition_id, condition_programmes in grouped.items()
    }
    if len(set(paired_keys.values())) != 1:
        raise ValueError("Calibration arms require exactly matched evaluation views.")
    base_arms = tuple(_calibration_arm(grouped[condition_id]) for condition_id in requested)
    eligible = tuple(
        arm for arm in base_arms if arm.procedure_assistance == "output_contract_only" and arm.tool_choice == "auto"
    )
    if not eligible:
        raise ValueError("Calibration selection requires a clean automatic-tool condition.")
    arms = tuple(
        arm.model_copy(
            update={
                "dominated_by_condition_ids": tuple(
                    candidate.condition_id
                    for candidate in base_arms
                    if candidate.procedure_assistance == arm.procedure_assistance
                    and candidate.tool_choice == arm.tool_choice
                    and _dominates(candidate, arm)
                )
            }
        )
        for arm in base_arms
    )
    pareto = tuple(
        arm
        for arm in arms
        if arm.procedure_assistance == "output_contract_only"
        and arm.tool_choice == "auto"
        and not arm.dominated_by_condition_ids
    )
    chosen = min(
        pareto,
        key=lambda arm: (
            -arm.complete_chain_success_rate,
            -arm.checkpoint_success_rate,
            arm.maximum_turns_per_step,
            arm.correction_count_mean,
            arm.agent_turns_mean,
            arm.elapsed_seconds_mean,
            arm.total_tokens_mean,
            arm.provider_reported_usd_mean,
            arm.condition_id,
        ),
    )
    values = tuple(next(iter(field)) for field in common_fields)
    return TrialDevCalibrationSelectionV1(
        model_id=cast(str, values[0]),
        reasoning_effort=cast(ReasoningEffortV1 | None, values[1]),
        request_replicate_id=cast(str, values[2]),
        task_materialization_seed=cast(int, values[3]),
        stream_id=cast(TrialDevStreamIdV1, values[4]),
        evaluation_unit_ids=tuple(key[0] for key in next(iter(paired_keys.values()))),
        arms=arms,
        pareto_condition_ids=tuple(arm.condition_id for arm in pareto),
        selected_condition_id=chosen.condition_id,
    )


__all__ = [
    "TRIALDEV_METRIC_BOOTSTRAP_RESAMPLES_V1",
    "TRIALDEV_METRIC_BOOTSTRAP_SEED_V1",
    "TRIALDEV_METRIC_CONFIDENCE_LEVEL_V1",
    "compare_trialdev_conditions_v1",
    "select_trialdev_calibration_v1",
    "summarize_trialdev_metrics_v1",
]
