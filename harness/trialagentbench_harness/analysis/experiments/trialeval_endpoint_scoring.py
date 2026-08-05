"""Project one TrialEval submission into prespecified experiment endpoints."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Literal

from trialagentbench_harness.contracts.core.runs import TrialEvalAblationItemResultV1
from trialagentbench_harness.contracts.experiments import (
    TrialEvalAblationEndpointRowV1,
    TrialEvalNormalizationStatusV1,
)
from trialagentbench_harness.contracts.scoring.assumption_evidence import (
    AssumptionEvidenceManifestV1,
)
from trialagentbench_harness.contracts.submission import (
    ScalarEstimateV1,
    TrialEvalSubmissionV1,
)
from trialagentbench_harness.grading.grader import grade
from trialagentbench_harness.grading.models import ValidatedScoringKeyV1
from trialagentbench_harness.io import canonical_payload_sha256, sha256_file
from trialagentbench_harness.trialeval.grade_submission import (
    grade_trialeval_submission_v1,
    trialeval_omitted_required_deliverables_v1,
)
from trialagentbench_harness.trialeval.planning import (
    assess_trialeval_route_planning_v1,
)
from trialagentbench_harness.trialeval.schema import BenchmarkItem


def trialeval_numeric_result_available_v1(
    submission: TrialEvalSubmissionV1 | None,
) -> bool:
    """Return whether the primary carries a numeric estimand or identified set."""

    if submission is None:
        return False
    result = submission.primary_analysis.result
    return result.kind != "non_identification" or result.identified_set is not None


def trialeval_scoring_implementation_sha256_v1() -> str:
    """Return the exact grade-and-endpoint implementation hash."""

    sources = {
        "narrow_grader": inspect.getsourcefile(grade),
        "submission_boundary": inspect.getsourcefile(grade_trialeval_submission_v1),
        "endpoint_projection": inspect.getsourcefile(score_trialeval_ablation_submission_v1),
    }
    if any(source is None for source in sources.values()):
        raise RuntimeError("Cannot resolve the TrialEval scoring implementation sources.")
    return canonical_payload_sha256(
        {name: sha256_file(Path(source)) for name, source in sources.items() if source is not None}
    )


def score_trialeval_ablation_submission_v1(
    *,
    scoring_key: ValidatedScoringKeyV1,
    assumption_evidence: AssumptionEvidenceManifestV1,
    item: BenchmarkItem,
    result: TrialEvalAblationItemResultV1,
    submission: TrialEvalSubmissionV1 | None,
    normalization_source: Literal["direct_structured", "manual_masked", "automated_importer"],
    normalization_status: TrialEvalNormalizationStatusV1,
    normalization_failure_reason: str | None = None,
) -> TrialEvalAblationEndpointRowV1:
    """Score one immutable response through the noncompensatory endpoint cascade."""

    grade_record = grade_trialeval_submission_v1(
        item=item,
        scoring_key=scoring_key,
        assumption_evidence=assumption_evidence,
        submission=submission,
    )
    omitted_required_deliverables = trialeval_omitted_required_deliverables_v1(
        item=item,
        submission=submission,
    )
    planning = assess_trialeval_route_planning_v1(
        item=item,
        scoring_key=scoring_key,
        matched_route_id=grade_record.matched_route_id,
        submission=submission,
    )
    scalar_primary = submission is not None and isinstance(submission.primary_analysis.result, ScalarEstimateV1)
    return TrialEvalAblationEndpointRowV1(
        assignment_id=result.assignment.assignment_id,
        task_id=result.assignment.task_id,
        context_tier=result.assignment.context_tier,
        data_preparation=result.assignment.data_preparation,
        analysis_specification=result.assignment.analysis_specification,
        model_id=result.run_config.model,
        replicate_id=result.assignment.replicate_id,
        procedure_assistance=result.assignment.procedure_assistance,
        prompt_condition=result.assignment.prompt_condition,
        submission_interface=result.assignment.submission_interface,
        normalization_source=normalization_source,
        normalization_status=normalization_status,
        normalization_failure_reason=normalization_failure_reason,
        omitted_required_deliverables=omitted_required_deliverables,
        primary_failure_code=(None if grade_record.passed else grade_record.failure_codes[0]),
        usable_primary=grade_record.usable_primary,
        route_match=grade_record.route_match,
        obligations_met=grade_record.obligations_met,
        credit_eligible_route_count=len(scoring_key.credit_eligible_routes),
        numeric_result_available=trialeval_numeric_result_available_v1(submission),
        primary_uncertainty_valid=(bool(grade_record.route_match) if scalar_primary else None),
        primary_interval_agreement=None,
        result_match=grade_record.result_match,
        numeric_absolute_error=grade_record.absolute_error,
        numeric_tolerance_ratio=grade_record.tolerance_ratio,
        primary_analysis_conforms=grade_record.passed,
        planning_applicable=planning.applicable,
        planning_valid=planning.valid,
        planning_usable_with_primary=(
            None if not planning.applicable else bool(planning.valid and grade_record.passed)
        ),
        planning_achieved_power=planning.matched_achieved_power,
        planning_power_shortfall=planning.matched_power_shortfall,
        planning_underpowered=planning.matched_underpowered,
        planning_proportional_participant_deviation=(planning.matched_proportional_participant_deviation),
        planning_log_sample_size_ratio=planning.matched_log_sample_size_ratio,
        planning_event_shortage=planning.matched_event_shortage,
        planning_excess_events=planning.matched_excess_events,
        planning_excess_participants=planning.matched_excess_participants,
        planning_participant_shortage=planning.matched_participant_shortage,
    )


__all__ = [
    "score_trialeval_ablation_submission_v1",
    "trialeval_numeric_result_available_v1",
    "trialeval_scoring_implementation_sha256_v1",
]
