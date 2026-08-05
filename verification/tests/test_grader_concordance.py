from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from trialagentbench_validation.grader_concordance import (
    CanonicalSubmissionV1,
    CanonicalTrialEvalRouteWitnessV1,
    ScoringKeyV1,
    TrialDevEvaluationTargetV1,
    grade_trialeval_independently,
    run_grader_concordance,
)
from trialagentbench_validation.grader_stress import generate_trialeval_mutations
from trialagentbench_validation.raw_projection import RawTrialEvalRouteWitnessV1


def _write_jsonl(path: Path, records: tuple[dict[str, object], ...]) -> bytes:
    body = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return body


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def test_complete_method_identity_mutation_reaches_the_route_gate() -> None:
    signature = {
        "analysis_population_id": "intention_to_treat",
        "estimand_id": "risk_difference_365d",
        "intercurrent_event_strategy_ids": ["treatment_policy"],
        "assessment_horizon_days": 365,
        "treatment_id": "active",
        "comparator_id": "control",
        "endpoint_id": "death",
        "effect_scale": "risk_difference_tau",
        "analysis_method_id": "bounds_delta_grid",
    }
    components = [
        {"name": "delta_0.05_lower", "value": -0.10},
        {"name": "delta_0.05_upper", "value": 0.02},
        {"name": "delta_0.10_lower", "value": -0.15},
        {"name": "delta_0.10_upper", "value": 0.07},
        {"name": "delta_0.20_lower", "value": -0.25},
        {"name": "delta_0.20_upper", "value": 0.17},
    ]
    key = ScoringKeyV1.model_validate(
        {
            "schema_id": "trialagentbench.scoring_key/v1",
            "release_id": "release-1",
            "item_id": "item-1",
            "question_id": "question-1",
            "context_tier": "C1",
            "credit_eligible_routes": [
                {
                    "route_id": "bounded-route",
                    "signature": signature,
                    "method": {
                        "analysis_method_id": "bounds_delta_grid",
                        "estimator_family": "bounds",
                        "result_kind": "sensitivity_set",
                        "uncertainty_method": "identified_set",
                        "sensitivity_parameters": [0.05, 0.10, 0.20],
                        "design_modifiers": [],
                    },
                    "required_identification_assumptions": ["bounded_deviation_model"],
                    "required_diagnostics": [],
                    "target": {
                        "kind": "numeric_vector",
                        "components": components,
                        "result_unit": "probability_difference",
                        "acceptance_envelope": {
                            "schema_id": "trialagentbench.numerical_acceptance_envelope/v1",
                            "reporting_decimal_places": 3,
                            "independent_max_abs_difference": 1e-12,
                            "public_verification_id": "bounded-route",
                            "independent_verification_ids": [
                                "independent-bounded-route"
                            ],
                        },
                    },
                }
            ],
        }
    )
    submission = CanonicalSubmissionV1.model_validate(
        {
            "schema_id": "trialagentbench.canonical_submission/v1",
            "item_id": "item-1",
            "primary": signature,
            "diagnostic_ids": [],
            "result": {
                "kind": "numeric_vector",
                "components": components,
                "result_unit": "probability_difference",
            },
        }
    )
    witness = CanonicalTrialEvalRouteWitnessV1(
        witness_id="item-1::bounded-route",
        item_id="item-1",
        route_id="bounded-route",
        context_tier="C1",
        submission=submission,
    )

    mutations, _ = generate_trialeval_mutations(
        witnesses=(witness,),
        key_by_item={"item-1": key},
    )
    method_mutation = next(
        row
        for row in mutations
        if row.mutated_coordinate == "primary.analysis_method_id"
    )
    grade = grade_trialeval_independently(key, method_mutation.submission)

    assert method_mutation.expected_first_gate == "route"
    assert grade.first_failure_gate == "route"
    assert grade.failure_codes == ("unrecognized_primary_route",)


def test_separate_process_grader_concordance_matches_exact_record(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    signature = {
        "analysis_population_id": "intention_to_treat",
        "estimand_id": "primary",
        "intercurrent_event_strategy_ids": ["treatment_policy"],
        "assessment_horizon_days": None,
        "treatment_id": "active",
        "comparator_id": "control",
        "endpoint_id": "endpoint",
        "effect_scale": "risk_difference",
        "analysis_method_id": "milestone_risk_bootstrap",
    }
    key = {
        "schema_id": "trialagentbench.scoring_key/v1",
        "release_id": "release-1",
        "item_id": "item-1",
        "question_id": "question-1",
        "context_tier": "C1",
        "credit_eligible_routes": [
            {
                "route_id": "route-1",
                "signature": signature,
                "method": {
                    "analysis_method_id": "milestone_risk_bootstrap",
                    "estimator_family": "milestone_risk",
                    "result_kind": "numeric_point",
                    "uncertainty_method": "participant_bootstrap",
                    "sensitivity_parameters": [],
                    "design_modifiers": [],
                },
                "required_identification_assumptions": [
                    "randomization_exchangeability"
                ],
                "required_diagnostics": ["randomization_integrity_public"],
                "target": {
                    "kind": "numeric_point",
                    "value": 0.125,
                    "result_unit": "risk_difference",
                    "acceptance_envelope": {
                        "schema_id": "trialagentbench.numerical_acceptance_envelope/v1",
                        "reporting_decimal_places": 3,
                        "independent_max_abs_difference": 0.0001,
                        "public_verification_id": "public-replay-1",
                        "independent_verification_ids": ["independent-replay-1"],
                    },
                    "require_confidence_interval": True,
                    "confidence_interval_lower": 0.10,
                    "confidence_interval_upper": 0.15,
                },
            }
        ],
    }
    keys_body = _write_jsonl(release / "grader" / "scoring_keys.jsonl", (key,))
    (release / "grader" / "scoring_key_manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "trialagentbench.scoring_key_manifest/v1",
                "release_id": "release-1",
                "specification_sha256": "a" * 64,
                "scoring_keys_sha256": hashlib.sha256(keys_body).hexdigest(),
                "item_ids": ["item-1"],
            }
        ),
        encoding="utf-8",
    )
    submissions = tmp_path / "submissions"
    canonical_submission = {
        "schema_id": "trialagentbench.canonical_submission/v1",
        "item_id": "item-1",
        "primary": signature,
        "diagnostic_ids": ["randomization_integrity_public"],
        "result": {
            "kind": "numeric_point",
            "value": 0.1254,
            "result_unit": "risk_difference",
            "confidence_interval_lower": 0.10,
            "confidence_interval_upper": 0.15,
        },
    }
    _write_jsonl(
        submissions / "trialeval_canonical_submissions.jsonl",
        (canonical_submission,),
    )
    _write_jsonl(
        submissions / "trialeval_route_witnesses.jsonl",
        (
            {
                "schema_id": "trialagentbench.trialeval_route_witness/v1",
                "witness_id": "item-1::route-1",
                "item_id": "item-1",
                "route_id": "route-1",
                "context_tier": "C1",
                "submission": canonical_submission,
            },
        ),
    )
    item_root = release / "public" / "items" / "item-1"
    item_root.mkdir(parents=True)
    task = {"design_subtype": "individual_randomized"}
    (item_root / "task.json").write_text(
        json.dumps(task, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    protocol = {
        "design_family": "individual_randomized",
        "arms": [{"arm_id": "control"}, {"arm_id": "active"}],
    }
    (item_root / "protocol_summary.json").write_text(
        json.dumps(protocol, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    contract: dict[str, object] = {
        "schema_id": "trialagentbench.trialeval_semantic_submission_contract/v1",
        "task_id": "item-1",
        "submission_semantics_id": "trialagentbench.trialeval_submission/v1",
        "required_deliverables": ["evidence", "limitations", "primary_analysis"],
        "diagnostic_obligations": [
            {
                "assumption_id": "randomization_integrity",
                "diagnostic_id": "randomization_integrity_public",
                "evidence_requirement": "design_declaration",
                "primary_credit_policy": "design_modifier",
                "operation": "read protocol randomization declaration",
                "public_evidence_basis": ["protocol_summary.json"],
                "interpretation": "Supports the declared randomized design only.",
            }
        ],
    }
    contract["checksum"] = _canonical_sha256(contract)
    (item_root / "submission_contract.json").write_text(
        json.dumps(contract, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assumption_manifest: dict[str, object] = {
        "version": "v1",
        "schema_id": "trial_benchmark_assumption_evidence_manifest_v1",
        "item_id": "item-1",
        "base_case_id": "TE-S01-A1",
        "canonical_item_id": "item-1",
        "variant_id": "TE-S01-A1__C1",
        "context_tier": "C1",
        "replicate_index": 0,
        "records": [
            {
                "assumption_id": "randomization_integrity",
                "expected_status": "holds",
                "computed_status": "holds",
                "expected_band": "holds",
                "computed_band": "holds",
                "diagnosability": "design_declared",
                "decision_metric_names": {},
                "supporting_metrics": {},
                "metric_units": {},
                "metric_public_evidence_basis": {},
                "factual_public_evidence_basis": ["protocol_summary.json"],
                "notes": [],
            }
        ],
    }
    assumption_manifest["checksum"] = _canonical_sha256(assumption_manifest)
    _write_jsonl(
        release / "grader" / "domains" / "assumption_evidence.jsonl",
        (
            {
                "domain": "assumption_evidence",
                "task_id": "item-1",
                "payload": {"manifest": assumption_manifest},
            },
        ),
    )
    raw_submission = {
        "schema_id": "trialagentbench.trialeval_submission/v1",
        "task_id": "item-1",
        "primary_analysis": {
            "declared_primary": True,
            "estimand": {
                "estimand_id": "primary",
                "population_id": "intention_to_treat",
                "treatment_id": "active",
                "comparator_id": "control",
                "endpoint_id": "endpoint",
                "intercurrent_event_strategy_ids": ["treatment_policy"],
                "horizon_not_applicable_reason": "fixed endpoint has no time horizon",
            },
            "estimator": {
                "analysis_method_id": "milestone_risk_bootstrap",
            },
            "result_kind": "numeric_point",
            "result": {
                "kind": "scalar",
                "value": 0.1254,
                "effect_scale": "risk_difference",
                "unit": "risk_difference",
                "interval": {
                    "lower": 0.10,
                    "upper": 0.15,
                    "confidence_level": 0.95,
                },
            },
            "favorable_direction": "higher",
            "evidence_ids": ["randomization-evidence"],
        },
        "evidence": [
            {
                "evidence_id": "randomization-evidence",
                "evidence_type": "validity",
                "principle": "design_validity",
                "operation": "assessment",
                "diagnostic_id": "randomization_integrity_public",
                "target": "randomized assignment declaration",
                "result": {
                    "kind": "factual_premise",
                    "premise_id": "randomized_assignment_declared",
                    "conclusion": "supported",
                },
                "interpretation": "The protocol declares randomized assignment.",
                "source_artifacts": ["protocol_summary.json"],
            }
        ],
        "limitations": [],
    }
    normalized_witness = RawTrialEvalRouteWitnessV1.model_validate(
        {
            "schema_id": "trialagentbench.trialeval_raw_route_witness/v1",
            "release_id": "release-1",
            "witness_id": "item-1::route-1",
            "item_id": "item-1",
            "route_id": "route-1",
            "context_tier": "C1",
            "primary_evidence_class": "design_or_provenance_reasoning",
            "repair_required": False,
            "fixed_question_sha256": "0" * 64,
            "route_signature_sha256": "0" * 64,
            "participant_input_checksums": {"task.json": "0" * 64},
            "raw_response_sha256": "0" * 64,
            "submission": raw_submission,
        }
    )
    raw_submission = normalized_witness.submission.model_dump(
        mode="json", exclude_none=True
    )
    participant_checksums = {
        relative: hashlib.sha256((item_root / relative).read_bytes()).hexdigest()
        for relative in (
            "protocol_summary.json",
            "submission_contract.json",
            "task.json",
        )
    }
    _write_jsonl(
        submissions / "trialeval_raw_route_witnesses.jsonl",
        (
            {
                "schema_id": "trialagentbench.trialeval_raw_route_witness/v1",
                "release_id": "release-1",
                "witness_id": "item-1::route-1",
                "item_id": "item-1",
                "route_id": "route-1",
                "context_tier": "C1",
                "primary_evidence_class": "design_or_provenance_reasoning",
                "repair_required": False,
                "fixed_question_sha256": participant_checksums["task.json"],
                "route_signature_sha256": _canonical_sha256(signature),
                "participant_input_checksums": participant_checksums,
                "raw_response_sha256": _canonical_sha256(raw_submission),
                "submission": raw_submission,
            },
        ),
    )
    target_payload: dict[str, object] = {
        "schema_id": "trialdev_evaluation_target_register_record_v1",
        "scenario_id": "s01",
        "phase_id": "observational_review",
        "program_objective_id": "benefit_risk",
        "phase_scoring_objective_id": "benefit_risk",
        "lane_id": "asset_nomination",
        "scoring_policy_id": "categorical_target_match_v1",
        "public_evidence_basis": ["public/observational_extract.parquet"],
        "evaluator_evidence_basis": ["grader/evaluation_target_register.jsonl"],
        "reference_target_ids": ["nominate_drug_a"],
        "credit_eligible_target_ids": ["nominate_drug_b"],
        "rejected_shortcut_ids": [],
        "recoverability_policy_id": "acceptable_candidate_set",
        "target_resolution": "release_static",
        "value_payload": {},
    }
    target_payload["checksum"] = _canonical_sha256(target_payload)
    target = TrialDevEvaluationTargetV1.model_validate(target_payload)
    _write_jsonl(
        release / "scenario_s01" / "grader" / "evaluation_target_register.jsonl",
        (target.model_dump(mode="json"),),
    )
    _write_jsonl(
        submissions / "trialdev_canonical_lane_submissions.jsonl",
        (
            {
                "schema_id": "trialagentbench.trialdev_canonical_lane_submission/v1",
                "evaluation_target_checksum": target.checksum,
                "scenario_id": "s01",
                "phase_id": "observational_review",
                "program_objective_id": "benefit_risk",
                "phase_scoring_objective_id": "benefit_risk",
                "lane_id": "asset_nomination",
                "submitted_target_id": "nominate_drug_b",
                "artifact_status": "present",
            },
        ),
    )
    executable = shutil.which("trialagentbench")
    if executable is None:
        pytest.skip("The optional TrialAgentBench harness executable is not installed.")

    report = run_grader_concordance(
        release_root=release,
        canonical_submissions=submissions,
        output_dir=tmp_path / "output",
        harness_executable=executable,
    )

    assert report.passed
    assert report.required_count == 2
    assert report.trialeval_required_count == 1
    assert report.trialdev_required_count == 1
    assert report.mismatch_count == 0
    assert report.trialeval_mutation_required_count > 0
    assert report.trialeval_mutation_mismatch_count == 0
    assert report.trialeval_mutation_behavior_failure_count == 0
    stress_report = json.loads(
        (
            tmp_path
            / "output"
            / "trialeval_grader_stress"
            / "trialeval_grader_stress_report.json"
        ).read_text(encoding="utf-8")
    )
    assert stress_report["required_artifact_removal_count"] == 1
    raw_mutation_cases = tuple(
        json.loads(line)
        for line in (
            tmp_path
            / "output"
            / "trialeval_grader_stress"
            / "trialeval_raw_evidence_mutation_cases.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    artifact_case = next(
        row
        for row in raw_mutation_cases
        if row.get("removed_required_artifact") == "protocol_summary.json"
    )
    assert artifact_case["replacement_artifact"] != "protocol_summary.json"
    mutation_grades = {
        row["mutation_id"]: row
        for row in (
            json.loads(line)
            for line in (
                tmp_path
                / "output"
                / "trialeval_grader_stress"
                / "independent_mutation_grades.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    artifact_grade = mutation_grades[artifact_case["mutation_id"]]["grade"]
    assert artifact_grade["first_failure_gate"] == "evidence"
    assert artifact_grade["failure_codes"] == ["missing_required_diagnostic"]
    assert report.trialdev_mutation_required_count == 3
    assert report.trialdev_mutation_mismatch_count == 0
    assert report.trialdev_mutation_behavior_failure_count == 0

    role_release = tmp_path / "role_release"
    archives = (
        (
            role_release
            / "data_release"
            / "trialeval"
            / "TrialEvalBench_evaluator.zip",
            release / "grader",
            "grader",
            "domains/",
        ),
        (
            role_release
            / "data_release"
            / "trialeval"
            / "TrialEvalBench_participant.zip",
            release / "public",
            "",
            None,
        ),
        (
            role_release
            / "data_release"
            / "trialeval"
            / "TrialEvalBench_verification.zip",
            release / "grader" / "domains",
            "grader/domains",
            None,
        ),
        (
            role_release / "data_release" / "trialdev" / "TrialDevBench_evaluator.zip",
            release / "scenario_s01",
            "scenario_s01",
            None,
        ),
    )
    for archive_path, source, prefix, excluded_prefix in archives:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
            for path in sorted(
                candidate for candidate in source.rglob("*") if candidate.is_file()
            ):
                relative = path.relative_to(source).as_posix()
                if excluded_prefix is not None and relative.startswith(excluded_prefix):
                    continue
                archive.write(path, f"{prefix}/{relative}" if prefix else relative)

    archived_report = run_grader_concordance(
        release_root=role_release,
        canonical_submissions=submissions,
        output_dir=tmp_path / "archived_output",
        harness_executable=executable,
    )

    assert archived_report.passed
    assert archived_report.required_count == report.required_count
    assert archived_report.mismatch_count == 0
    assert archived_report.trialeval_mutation_required_count == (
        report.trialeval_mutation_required_count
    )
