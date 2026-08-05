"""Evidence-linked feature extraction for observable action traces."""

from __future__ import annotations

from collections.abc import Sequence

from trialagentbench_harness.contracts.submission import EvidenceRecordV1
from trialagentbench_harness.contracts.trace.observable import (
    SemanticActionFeatureNameV1,
    SemanticActionFeatureRowV1,
    TraceFeatureRowV1,
)

FEATURE_ORDER: tuple[SemanticActionFeatureNameV1, ...] = (
    "confounding_mentioned",
    "confounding_adjustment_performed",
    "balance_or_overlap_reported",
    "ph_assumption_mentioned",
    "ph_diagnostic_performed",
    "censoring_mentioned",
    "censoring_adjustment_performed",
    "missingness_mentioned",
    "missingness_handling_performed",
    "uncertainty_interval_reported",
    "sensitivity_analysis_mentioned",
    "sensitivity_analysis_performed",
    "safety_tradeoff_reported",
    "cost_tradeoff_reported",
    "objective_alignment_reported",
)


def _has(
    evidence: Sequence[EvidenceRecordV1],
    *,
    principle: str,
    operations: frozenset[str] | None = None,
) -> bool:
    return any(
        record.principle == principle and (operations is None or record.operation in operations) for record in evidence
    )


def structured_feature_flags(
    evidence: Sequence[EvidenceRecordV1],
    *,
    primary_interval_reported: bool,
) -> dict[str, bool]:
    """Derive trace flags only from typed, submitted evidence records."""
    adjustment = frozenset({"adjustment"})
    handling = frozenset({"adjustment", "data_validation"})
    sensitivity = frozenset({"sensitivity_analysis"})
    return {
        "checked_confounding": _has(evidence, principle="confounding"),
        "checked_ph_assumption": _has(evidence, principle="proportional_hazards"),
        "checked_missingness": _has(evidence, principle="missingness"),
        "checked_censoring": _has(evidence, principle="censoring"),
        "quantified_uncertainty": primary_interval_reported or _has(evidence, principle="uncertainty"),
        "used_sensitivity_analysis": _has(evidence, principle="sensitivity", operations=sensitivity),
        "considered_safety": _has(evidence, principle="safety"),
        "considered_cost": _has(evidence, principle="cost"),
        "objective_aligned_rationale": _has(evidence, principle="objective_alignment"),
        "confounding_adjustment_performed": _has(evidence, principle="confounding", operations=adjustment),
        "censoring_adjustment_performed": _has(evidence, principle="censoring", operations=adjustment),
        "missingness_handling_performed": _has(evidence, principle="missingness", operations=handling),
    }


def structured_feature_rows(
    feature: TraceFeatureRowV1,
    *,
    evidence: Sequence[EvidenceRecordV1],
    primary_interval_reported: bool,
    evidence_basis: tuple[str, ...] = (),
) -> list[SemanticActionFeatureRowV1]:
    """Create conservative semantic rows from explicit structured evidence."""
    flags = structured_feature_flags(evidence, primary_interval_reported=primary_interval_reported)
    present_by_name: dict[SemanticActionFeatureNameV1, bool] = {
        "confounding_mentioned": flags["checked_confounding"],
        "confounding_adjustment_performed": flags["confounding_adjustment_performed"],
        "balance_or_overlap_reported": _has(evidence, principle="confounding", operations=frozenset({"assessment"})),
        "ph_assumption_mentioned": flags["checked_ph_assumption"],
        "ph_diagnostic_performed": _has(
            evidence,
            principle="proportional_hazards",
            operations=frozenset({"assessment"}),
        ),
        "censoring_mentioned": flags["checked_censoring"],
        "censoring_adjustment_performed": flags["censoring_adjustment_performed"],
        "missingness_mentioned": flags["checked_missingness"],
        "missingness_handling_performed": flags["missingness_handling_performed"],
        "uncertainty_interval_reported": flags["quantified_uncertainty"],
        "sensitivity_analysis_mentioned": flags["used_sensitivity_analysis"],
        "sensitivity_analysis_performed": flags["used_sensitivity_analysis"],
        "safety_tradeoff_reported": flags["considered_safety"],
        "cost_tradeoff_reported": flags["considered_cost"],
        "objective_alignment_reported": flags["objective_aligned_rationale"],
    }
    return [
        SemanticActionFeatureRowV1(
            benchmark=feature.benchmark,
            model_id=feature.model_id,
            run_id=feature.run_id,
            task_id=feature.task_id,
            assignment_id=feature.assignment_id,
            program_id=feature.program_id,
            phase_id=feature.phase_id,
            feature_name=name,
            feature_present=present_by_name[name],
            evidence_strength="structured_submission_field" if present_by_name[name] else "not_observed",
            evidence_basis=evidence_basis if present_by_name[name] else (),
            score_link_id=feature.score_link_id or "",
        )
        for name in FEATURE_ORDER
    ]


__all__ = ["FEATURE_ORDER", "structured_feature_flags", "structured_feature_rows"]
