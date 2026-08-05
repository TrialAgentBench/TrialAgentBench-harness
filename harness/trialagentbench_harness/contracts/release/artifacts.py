"""Canonical public release artifact identities."""

from __future__ import annotations

TRIALEVAL_PARTICIPANT_ARCHIVE_NAME = "TrialEvalBench_participant.zip"
TRIALEVAL_EVALUATOR_ARCHIVE_NAME = "TrialEvalBench_evaluator.zip"
TRIALEVAL_VERIFICATION_ARCHIVE_NAME = "TrialEvalBench_verification.zip"
TRIALEVAL_PARTICIPANT_ARCHIVE = f"TrialEvalBench/{TRIALEVAL_PARTICIPANT_ARCHIVE_NAME}"
TRIALEVAL_EVALUATOR_ARCHIVE = f"TrialEvalBench/{TRIALEVAL_EVALUATOR_ARCHIVE_NAME}"
TRIALEVAL_VERIFICATION_ARCHIVE = f"TrialEvalBench/{TRIALEVAL_VERIFICATION_ARCHIVE_NAME}"

TRIALDEV_PARTICIPANT_ARCHIVE_NAME = "TrialDevBench_participant.zip"
TRIALDEV_EVALUATOR_ARCHIVE_NAME = "TrialDevBench_evaluator.zip"
TRIALDEV_VERIFICATION_ARCHIVE_NAME = "TrialDevBench_verification.zip"
TRIALDEV_PARTICIPANT_ARCHIVE = f"TrialDevBench/{TRIALDEV_PARTICIPANT_ARCHIVE_NAME}"
TRIALDEV_EVALUATOR_ARCHIVE = f"TrialDevBench/{TRIALDEV_EVALUATOR_ARCHIVE_NAME}"
TRIALDEV_VERIFICATION_ARCHIVE = f"TrialDevBench/{TRIALDEV_VERIFICATION_ARCHIVE_NAME}"

TRIALEVAL_PARTICIPANT_ROOT_MEMBERS: frozenset[str] = frozenset(
    {
        "benchmark_charter.json",
        "benchmark_map.md",
        "diagnostic_dictionary.json",
        "manifest.json",
        "method_dictionary.json",
    }
)
TRIALDEV_VERIFICATION_ROOT_MEMBERS: frozenset[str] = frozenset(
    {
        "scientific_construction_inventory.json",
        "scientific_source_registry.json",
    }
)
TRIALDEV_FIXED_TRAJECTORY_REPLICATE_MEMBERS: frozenset[str] = frozenset(
    {
        "arm_mapping.json",
        "endpoints.parquet",
        "execution_summary.json",
        "participants.parquet",
        "request.json",
        "safety.parquet",
    }
)

TRIALDEV_SCORER_RELATIVE_MEMBERS: tuple[str, ...] = (
    "grader/grading_procedure.json",
    "grader/submission_schema.json",
    "grader/drug_ranking_reference_manifest.json",
    "grader/public_recoverability_report.json",
    "grader/evaluation_target_register.jsonl",
    "grader/evaluation_target_register_manifest.json",
    "grader/evaluation_target_register_gate_report.json",
    "grader/recoverability_manifest.json",
)
