"""Deterministic TrialEval submission grading at the evaluator boundary."""

from __future__ import annotations

from typing import cast

from trialagentbench_harness.contracts.release.trialeval_runtime_surface import (
    TrialEvalSemanticDeliverableV1,
    TrialEvalSemanticSubmissionContractV1,
)
from trialagentbench_harness.contracts.scoring.assumption_evidence import (
    AssumptionEvidenceManifestV1,
)
from trialagentbench_harness.contracts.submission import (
    TrialEvalSubmissionV1,
    missing_trialeval_required_deliverables_v1,
)
from trialagentbench_harness.grading import (
    grade,
    grade_missing_required_deliverable,
    grade_missing_submission,
)
from trialagentbench_harness.grading.models import (
    GradeRecordV1,
    ValidatedScoringKeyV1,
)
from trialagentbench_harness.trialeval.canonicalize import (
    canonicalize_trialeval_submission_v1,
)
from trialagentbench_harness.trialeval.diagnostic_evidence import (
    validated_diagnostic_ids_v1,
)
from trialagentbench_harness.trialeval.schema import BenchmarkItem


def trialeval_omitted_required_deliverables_v1(
    *,
    item: BenchmarkItem,
    submission: TrialEvalSubmissionV1 | None,
) -> tuple[TrialEvalSemanticDeliverableV1, ...]:
    """Return item-required deliverables absent from one participant response."""

    output_contract = TrialEvalSemanticSubmissionContractV1.model_validate(item.submission_contract)
    if submission is None:
        return output_contract.required_deliverables
    return cast(
        tuple[TrialEvalSemanticDeliverableV1, ...],
        missing_trialeval_required_deliverables_v1(
            submission,
            required_deliverables=output_contract.required_deliverables,
        ),
    )


def grade_trialeval_submission_v1(
    *,
    item: BenchmarkItem,
    scoring_key: ValidatedScoringKeyV1,
    assumption_evidence: AssumptionEvidenceManifestV1,
    submission: TrialEvalSubmissionV1 | None,
) -> GradeRecordV1:
    """Grade one explicit primary against independently validated public evidence."""

    if submission is None:
        return grade_missing_submission(scoring_key)
    missing_deliverables = trialeval_omitted_required_deliverables_v1(
        item=item,
        submission=submission,
    )
    if missing_deliverables:
        return grade_missing_required_deliverable(scoring_key)
    diagnostic_ids = validated_diagnostic_ids_v1(
        submission=submission,
        item=item,
        assumption_evidence=assumption_evidence,
    )
    canonical = canonicalize_trialeval_submission_v1(
        submission,
        validated_diagnostic_ids=frozenset(diagnostic_ids),
    )
    return grade(scoring_key, canonical)


__all__ = [
    "grade_trialeval_submission_v1",
    "trialeval_omitted_required_deliverables_v1",
]
