"""Canonical failure-code vocabulary for manifests and denominator-preserving outputs.

Failure codes are part of the harness' public audit surface. They must be:
* stable (do not rename casually)
* unique
* used consistently (no ad hoc strings)
"""

from __future__ import annotations

from enum import StrEnum


class TrialEvalFailureCode(StrEnum):
    item_json_invalid = "trialeval_item_json_invalid"
    item_not_in_suite = "trialeval_item_not_in_suite"
    missing_run_artifacts = "trialeval_missing_run_artifacts"
    denominator_mismatch = "trialeval_denominator_mismatch"
    scores_contract_invalid = "trialeval_scores_contract_invalid"


class TrialDevFailureCode(StrEnum):
    coverage_report_invalid = "trialdev_coverage_report_invalid"
    chain_summary_invalid = "trialdev_chain_summary_invalid"
    trajectory_grade_invalid = "trialdev_trajectory_grade_invalid"
    obs_grade_invalid = "trialdev_obs_grade_invalid"
    phase_request_invalid = "trialdev_phase_request_invalid"
    missing_run_artifacts = "trialdev_missing_run_artifacts"
    denominator_mismatch = "trialdev_denominator_mismatch"
    grade_failed = "trialdev_grade_failed"
    obs_rank_metrics_failed = "trialdev_obs_rank_metrics_failed"
    phase_rank_metrics_failed = "trialdev_phase_rank_metrics_failed"
    stick_twist_failed = "trialdev_stick_twist_failed"
    objective_alignment_failed = "trialdev_objective_alignment_failed"


__all__ = ["TrialEvalFailureCode", "TrialDevFailureCode"]
