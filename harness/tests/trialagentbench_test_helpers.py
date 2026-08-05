"""Shared TrialAgentBench release-test fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from trialagentbench_harness.contracts.experiments import (
    NarrativePacketIndexRowV1,
    NarrativePacketManifestV1,
    NarrativePacketSetManifestV1,
    NarrativeParticipantContextV1,
)
from trialagentbench_harness.contracts.release.benchmark_charter import (
    TrialAgentBenchCharterV1,
    render_benchmark_map_markdown,
)
from trialagentbench_harness.contracts.release.trialeval_manifest import (
    TrialEvalParticipantDiagnosticDictionaryV1,
    TrialEvalParticipantMethodDictionaryV1,
    TrialEvalParticipantMethodV1,
)
from trialagentbench_harness.contracts.submission import TrialEvalSubmissionV1
from trialagentbench_harness.contracts.trialeval_diagnostics import participant_diagnostic_dictionary_v1
from trialagentbench_harness.io.checksums import canonical_payload_sha256, sha256_file
from trialagentbench_harness.io.json import write_json, write_json_model


def write_narrative_packet_set(root: Path, *, report: str) -> Path:
    """Write one checksum-bound narrative packet set for experiment tests."""

    packet_dir = root / "masked-narrative-0001"
    packet_dir.mkdir(parents=True)
    report_path = packet_dir / "frozen_report.txt"
    report_path.write_text(report, encoding="utf-8")
    context = NarrativeParticipantContextV1(
        task_id="TASK1001",
        task_contract={"task_id": "TASK1001", "primary_estimand_id": "primary_itt"},
        participant_submission_contract={"task_id": "TASK1001"},
        participant_diagnostic_dictionary=minimal_trialeval_diagnostic_dictionary(),
        participant_method_dictionary=minimal_trialeval_method_dictionary(),
        canonical_submission_schema=TrialEvalSubmissionV1.model_json_schema(),
    ).with_checksum()
    context_path = packet_dir / "participant_context.json"
    write_json_model(context_path, context)
    packet = NarrativePacketManifestV1(
        blinded_identity="masked-narrative-0001",
        participant_task_id="TASK1001",
        assignment_id="assignment-1",
        report_state="present",
        report_sha256=sha256_file(report_path),
        participant_context_sha256=sha256_file(context_path),
    )
    packet_path = packet_dir / "packet.json"
    write_json_model(packet_path, packet)
    manifest = NarrativePacketSetManifestV1(
        schedule_sha256="s" * 64,
        run_identity_sha256="r" * 64,
        participant_release_sha256="p" * 64,
        source_files_sha256={"assignments/assignment-1.json": "a" * 64},
        packets=(
            NarrativePacketIndexRowV1(
                blinded_identity=packet.blinded_identity,
                packet_manifest_sha256=sha256_file(packet_path),
                report_sha256=packet.report_sha256,
            ),
        ),
    ).with_checksum()
    write_json_model(root / "manifest.json", manifest)
    return root


def minimal_participant_output_contract(
    task_id: str,
    *,
    data_preparation: str = "analysis_ready",
) -> dict[str, object]:
    """Return one valid representation-neutral participant output contract."""

    required = {"evidence", "limitations", "primary_analysis"}
    if data_preparation != "analysis_ready":
        required.add("reconstruction")
    if data_preparation == "raw_domains_declared_defect":
        required.add("data_integrity_record")
    payload: dict[str, object] = {
        "schema_id": "trialagentbench.trialeval_semantic_submission_contract/v1",
        "task_id": task_id,
        "submission_semantics_id": "trialagentbench.trialeval_submission/v1",
        "required_deliverables": sorted(required),
        "diagnostic_obligations": [],
    }
    payload["checksum"] = canonical_payload_sha256(payload)
    return payload


def write_minimal_trialeval_method_dictionary(root: Path) -> Path:
    """Write one valid participant-safe method dictionary for release fixtures."""

    payload = minimal_trialeval_method_dictionary()
    path = Path(root) / "method_dictionary.json"
    path.write_text(json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_minimal_trialeval_diagnostic_dictionary(root: Path) -> Path:
    """Write one valid task-general diagnostic dictionary for release fixtures."""

    payload = minimal_trialeval_diagnostic_dictionary()
    path = Path(root) / "diagnostic_dictionary.json"
    path.write_text(json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_minimal_trialeval_release_dictionaries(root: Path) -> tuple[Path, Path]:
    """Write both required participant analysis dictionaries."""

    return (
        write_minimal_trialeval_method_dictionary(root),
        write_minimal_trialeval_diagnostic_dictionary(root),
    )


def minimal_trialeval_method_dictionary() -> TrialEvalParticipantMethodDictionaryV1:
    """Return one participant-safe method dictionary for release fixtures."""

    return TrialEvalParticipantMethodDictionaryV1(
        methods=(
            TrialEvalParticipantMethodV1(
                method_id="km_rmst_greenwood",
                estimator_family="km",
                objective="estimation",
                result_kind="numeric_point",
                effect_scale="rmst_difference_tau",
                design_modifiers=("ph_robust_fixed_horizon",),
                uncertainty_method_id="greenwood",
                possible_diagnostic_ids=("censoring_followup_public",),
                description="Kaplan-Meier fixed-horizon risk contrast with Greenwood uncertainty.",
            ),
        )
    )


def minimal_trialeval_diagnostic_dictionary() -> TrialEvalParticipantDiagnosticDictionaryV1:
    """Return the task-general diagnostic dictionary projected from the registry."""

    return participant_diagnostic_dictionary_v1()


def route_reference_task_ranges(path: Path) -> list[dict[str, object]]:
    """Return contiguous task byte ranges for a test truth JSONL file."""

    ranges: list[dict[str, object]] = []
    offset = 0
    active_task: str | None = None
    active_offset = 0
    active_length = 0
    active_rows = 0
    with path.open("rb") as handle:
        for line in handle:
            task_id = str(json.loads(line)["task_id"])
            if active_task is not None and task_id != active_task:
                ranges.append(
                    {
                        "task_id": active_task,
                        "byte_offset": active_offset,
                        "byte_length": active_length,
                        "row_count": active_rows,
                    }
                )
                active_offset = offset
                active_length = 0
                active_rows = 0
            active_task = task_id
            active_length += len(line)
            active_rows += 1
            offset += len(line)
    if active_task is not None:
        ranges.append(
            {
                "task_id": active_task,
                "byte_offset": active_offset,
                "byte_length": active_length,
                "row_count": active_rows,
            }
        )
    return ranges


def minimal_estimator_family_map_payload() -> dict[str, object]:
    """Return a complete estimator-family partition for release fixtures."""

    estimands = {"km": "risk_difference", "coxph_binary": "global_cox_ph"}
    modifiers = {
        "km_ipcw": "procedure_modifier",
        "coxph_ipcw": "procedure_modifier",
        "validated_endpoint": "procedure_modifier",
        "cluster_parallel_participant_weighted": "design_modifier",
        "stepped_wedge_period_adjusted": "design_modifier",
        "group_sequential_adjustment": "procedure_modifier",
        "bounds": "procedure_modifier",
        "standardized_cox_g_computation": "procedure_modifier",
    }
    exclusions = {
        "coxph_time_interaction": "different_estimand",
        "piecewise_cox": "different_estimand",
        "weighted_logrank": "regime_invalid_diagnostic_only",
        "aft_parametric": "different_estimand",
        "milestone_risk": "failed_or_missing_qualification",
        "competing_risks": "different_estimand",
        "other": "failed_or_missing_qualification",
    }
    entries: list[dict[str, object]] = []
    for family, route_family in estimands.items():
        entries.append(
            {
                "estimator_family": family,
                "route_family": route_family,
                "role": "estimand_family",
                "requires_proportional_hazards_diagnostic": family == "coxph_binary",
                "requires_limitation": False,
                "credit_eligible_when": "Passed when the task register includes this estimand.",
                "rejected_when": "Rejected when the task register excludes this estimand.",
            }
        )
    for family, role in modifiers.items():
        entries.append(
            {
                "estimator_family": family,
                "role": role,
                "requires_proportional_hazards_diagnostic": family == "coxph_ipcw",
                "requires_limitation": family == "bounds",
                "credit_eligible_when": "Passed only when required by the method route.",
                "rejected_when": "Rejected outside a method route requiring this modifier.",
            }
        )
    for family, reason in exclusions.items():
        entries.append(
            {
                "estimator_family": family,
                "role": "unsupported",
                "requires_proportional_hazards_diagnostic": False,
                "requires_limitation": False,
                "exclusion_reason": reason,
                "credit_eligible_when": "Not passed in the current release.",
                "rejected_when": "Excluded from official primary scoring.",
            }
        )
    return {
        "schema_id": "trialagentbench.trialeval.estimator_route_family_map/v1",
        "entries": sorted(entries, key=lambda entry: str(entry["estimator_family"])),
    }


def minimal_estimator_registry_payload() -> dict[str, object]:
    """Return a checksum-bound source-method registry for release fixtures."""

    payload: dict[str, object] = {
        "methods": [
            {
                "method_id": "observed:coxph_binary_breslow",
                "family": "coxph_binary",
                "supported_effect_scales": ["log_hr"],
            },
            {
                "method_id": "observed:cox_ph",
                "family": "coxph_binary",
                "supported_effect_scales": ["log_hr"],
            },
        ]
    }
    payload["checksum"] = canonical_payload_sha256(payload)
    return payload


def minimal_benchmark_charter_payload() -> dict[str, object]:
    """Return a compact charter satisfying the standalone release boundary."""

    design_axes = (
        ("D1", ("individual_randomized",)),
        ("D2", ("pragmatic",)),
        ("D3", ("covariate_structure", "endpoint_ascertainment")),
        ("D4", ("cluster_parallel", "stepped_wedge", "group_sequential")),
    )
    contexts = (
        ("C1", "analysis_ready", "locked_sap"),
        ("C2", "analysis_ready", "protocol_only"),
        ("C3", "raw_domains", "locked_sap"),
        ("C4", "raw_domains", "protocol_only"),
        ("C5", "raw_domains_declared_defect", "protocol_only"),
    )
    contrast_pairs = (
        ("C1-C2", "C1", "C2"),
        ("C3-C4", "C3", "C4"),
        ("C3-C1", "C3", "C1"),
        ("C4-C2", "C4", "C2"),
        ("C5-C4", "C5", "C4"),
    )
    payload = {
        "schema_id": "trialagentbench.charter/v1",
        "version": "v1",
        "suites": [
            {
                "suite": suite,
                "purpose": f"{suite} purpose",
                "capability_cascade": ["completion", "validity"],
                "secondary_lanes": [],
            }
            for suite in ("trialeval", "trialdev")
        ],
        "axes": [
            {
                "axis": "design",
                "code": code,
                "label": code,
                "operative_definition": f"{code} definition",
                "ordinal_scope": "never",
                "allowed_design_subtypes": list(subtypes),
            }
            for code, subtypes in design_axes
        ]
        + [
            {
                "axis": "assumption",
                "code": code,
                "label": code,
                "operative_definition": f"{code} definition",
                "ordinal_scope": "matched_evaluation_series_only",
                "allowed_design_subtypes": [],
            }
            for code in ("A1", "A2", "A3", "A4")
        ],
        "context_configurations": [
            {
                "code": code,
                "data_preparation": preparation,
                "analysis_specification": specification,
                "canonical_procedure_assistance": "output_contract_only",
                "capability_isolated": f"{code} capability",
            }
            for code, preparation, specification in contexts
        ],
        "procedure_assistance_levels": ["output_contract_only", "unordered_checklist", "ordered_sop"],
        "response_interfaces": ["structured", "narrative"],
        "participant_artifacts": [
            {
                "artifact": artifact,
                "definition": f"{artifact} definition",
                "prohibited_content": [],
            }
            for artifact in (
                "protocol",
                "locked_sap",
                "data_specification",
                "output_contract",
                "unordered_checklist",
                "ordered_sop",
                "declared_defect",
            )
        ],
        "matched_context_contrasts": [
            {
                "contrast_id": contrast_id,
                "minuend": minuend,
                "subtrahend": subtrahend,
                "interpretation": f"{contrast_id} interpretation",
                "held_fixed": ["base_trial"],
            }
            for contrast_id, minuend, subtrahend in contrast_pairs
        ],
        "trialeval_grading_policy": {
            "credit_eligible_set_closure": "Only accepted compatible routes are passed.",
            "participant_compatibility_visibility": "Every task compatibility dimension is participant-visible.",
            "analysis_specification_policy": "Locked SAP and protocol-only configurations test different capabilities.",
            "vocabulary_notes": "Primary and sensitivity lanes are distinct.",
        },
    }
    payload["checksum"] = canonical_payload_sha256(payload)
    return payload


def minimal_benchmark_map_markdown() -> str:
    """Render the compact test charter as its required assessment map."""

    return render_benchmark_map_markdown(TrialAgentBenchCharterV1.model_validate(minimal_benchmark_charter_payload()))


def write_minimal_benchmark_charter(root: Path) -> None:
    """Write a compact charter satisfying the standalone release boundary."""

    payload = minimal_benchmark_charter_payload()
    write_json(Path(root) / "benchmark_charter.json", payload)
    (Path(root) / "benchmark_map.md").write_text(
        minimal_benchmark_map_markdown(),
        encoding="utf-8",
    )


def write_minimal_ground_truth_domains(root: Path, *, task_id: str = "TASK0001") -> None:
    """Write minimal current-release ground-truth domains for harness tests."""

    domains = Path(root) / "grader" / "domains"
    evidence_basis = [f"items/{task_id}/task.json", f"items/{task_id}/protocol_summary.json"]
    assumption_manifest = {
        "version": "v1",
        "schema_id": "trial_benchmark_assumption_evidence_manifest_v1",
        "item_id": "d1a1_rct_clean_01",
        "base_case_id": "d1a1_rct_clean",
        "canonical_item_id": "d1a1_rct_clean_01",
        "variant_id": "d1a1_rct_clean__C1",
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
                "factual_public_evidence_basis": ("protocol_summary.json",),
                "notes": [],
            }
        ],
    }
    assumption_bytes = json.dumps(
        assumption_manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assumption_manifest["checksum"] = hashlib.sha256(assumption_bytes).hexdigest()
    domains.mkdir(parents=True, exist_ok=True)
    write_json(
        domains / "grading_procedure.json",
        {
            "submission_schema_id": "trialagentbench.trialeval_submission/v1",
            "submission_schema_version": 1,
        },
    )
    write_json(domains / "context_panels.json", {"schema_id": "fixture_context_panels", "panels": []})
    score_lanes = {
        "version": "v1",
        "schema_id": "trial_benchmark_score_lane_manifest_v1",
        "item_id": "d1a1_rct_clean_01",
        "design_tier": "D1",
        "assumption_tier": "A1",
        "context_tier": "C1",
        "lanes": [
            {
                "lane_id": "primary_numeric.v1",
                "lane_class": "primary_numeric",
                "target_manifest_path": "target_manifests/primary.json",
                "scoring_reference_set": "max_recoverable",
                "evaluation_class": "identifiable_numeric",
                "estimand_id": "primary_itt",
                "expected_effect_scale": "log_hr",
                "identifiability_basis": "observed_analysis_table",
                "oracle_usage_policy": {
                    "allow_oracle_point_scoring": False,
                    "allow_oracle_interval_scoring": False,
                    "allow_oracle_calibration_only": True,
                    "identifiability_basis": "",
                },
                "allowed_answer_shapes": ["numeric_point"],
                "oracle_identifiable": False,
                "leakage_risk": "low",
                "mandatory": True,
            }
        ],
    }
    score_lanes["checksum"] = canonical_payload_sha256(score_lanes)
    grading_rubric = {
        "version": "v1",
        "schema_id": "trial_benchmark_staged_grading_rubric_v1",
        "item_id": "d1a1_rct_clean_01",
        "components": [
            {
                "component_id": "primary_analysis",
                "component_class": "numeric_analysis",
                "linked_score_lanes": ["primary_numeric.v1"],
                "grading_mode": "tolerance",
            }
        ],
    }
    grading_rubric["checksum"] = canonical_payload_sha256(grading_rubric)
    rubric_domain = {
        "version": "v1",
        "schema_id": "trialagentbench.trialeval.rubric_domain/v1",
        "items": [
            {
                "task_id": task_id,
                "item_id": "d1a1_rct_clean_01",
                "score_lanes": score_lanes,
                "grading_rubric": grading_rubric,
                "rubric_references": [],
            }
        ],
    }
    rubric_domain["checksum"] = canonical_payload_sha256(rubric_domain)
    write_json(domains / "rubric.json", rubric_domain)
    (domains / "reconstruction_scoring.jsonl").write_text("", encoding="utf-8")
    (domains / "assumption_evidence.jsonl").write_text(
        json.dumps(
            {
                "schema_id": "trialagentbench.trial_benchmark.grader_domain_row/v1",
                "domain": "assumption_evidence",
                "task_id": task_id,
                "payload": {
                    "schema_id": "trialagentbench.trial_benchmark.assumption_evidence_manifest_row/v1",
                    "manifest": assumption_manifest,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(domains / "estimator_route_family_map.json", minimal_estimator_family_map_payload())
    write_json(domains / "estimator_registry.json", minimal_estimator_registry_payload())
    register_row = {
        "schema_id": "trialagentbench.trialeval.evaluation_target_register_entry/v1",
        "task_id": task_id,
        "item_id": "d1a1_rct_clean_01",
        "lane_id": "primary_numeric.v1",
        "lane_class": "primary_numeric",
        "evaluation_class": "identifiable_numeric",
        "scoring_reference_set": "max_recoverable",
        "allowed_answer_shapes": ["numeric_point"],
        "identification_class": "point_identified",
        "contestedness": "construction_determined",
        "estimand_mode": "fixed_declared_estimand",
        "declared_primary_effect_scale": "log_hr",
        "credit_eligible_primary_effect_scales": ["log_hr"],
        "primary_route_family": "global_cox_ph",
        "credit_eligible_route_families": ["global_cox_ph"],
        "rejected_shortcut_families": [],
        "public_evidence_basis": evidence_basis,
        "score_profile_eligibility": [
            "credit_eligible_family_v1",
            "strict_method_id_v1",
            "diagnostic_recognition_v1",
        ],
    }
    (domains / "evaluation_target_register.jsonl").write_text(
        json.dumps(register_row, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_json(
        domains / "evaluation_target_register_manifest.json",
        {
            "version": "v1",
            "schema_id": "trialagentbench.trialeval.evaluation_target_register_manifest/v1",
            "release_root": ".",
            "row_count": 1,
            "task_count": 1,
            "register_jsonl_sha256": sha256_file(domains / "evaluation_target_register.jsonl"),
            "estimator_route_family_map_sha256": sha256_file(domains / "estimator_route_family_map.json"),
        },
    )
    method_route_row = {
        "schema_id": "trialagentbench.trialeval.method_route/v1",
        "cell_id": f"{task_id}:primary_numeric.v1:global_cox_ph:observed:coxph_binary_breslow:log_hr",
        "task_id": task_id,
        "item_id": "d1a1_rct_clean_01",
        "lane_id": "primary_numeric.v1",
        "route_family": "global_cox_ph",
        "estimator_families": ["coxph_binary"],
        "required_modifiers": [],
        "effect_scale": "log_hr",
        "result_unit": "log_hazard_ratio",
        "answer_shapes": ["numeric_point"],
        "route_reference_id": f"{task_id}:primary_numeric.v1:max_recoverable:observed:coxph_binary_breslow",
        "route_reference_role": "required_primary",
        "public_evidence_basis": evidence_basis,
        "uncertainty_kind": "two_sided_confidence_interval",
        "confidence_level": 0.95,
        "requires_diagnostics": [],
        "eligibility_source": "primary_numeric",
        "rationale": "Minimal PH-compatible log-HR method route for harness tests.",
    }
    (domains / "method_route_register.jsonl").write_text(
        json.dumps(method_route_row, sort_keys=True) + "\n", encoding="utf-8"
    )
    route_reference_row = {
        "schema_id": "trialagentbench.trialeval.route_reference/v1",
        "task_id": task_id,
        "item_id": "d1a1_rct_clean_01",
        "lane_id": "primary_numeric.v1",
        "route_reference_id": f"{task_id}:primary_numeric.v1:max_recoverable:observed:coxph_binary_breslow",
        "variant_role": "required_primary",
        "route_family": "global_cox_ph",
        "estimator_method_id": "observed:coxph_binary_breslow",
        "effect_scale": "log_hr",
        "answer_shape": "point",
        "value": -0.5,
        "standard_error": 0.1,
        "ci_low": -0.7,
        "ci_high": -0.3,
        "public_evidence_basis": evidence_basis,
        "required_modifiers": [],
        "identification_class": "point_identified",
        "support_status": "official_supported",
        "support_rationale": "Minimal scoreable route-specific reference for harness tests.",
        "numerical_equivalence": {
            "policy_id": "float64_sqrt_epsilon_v1",
            "absolute_tolerance": 1.4901161193847656e-08,
            "relative_tolerance": 1.4901161193847656e-08,
            "basis": "deterministic_cross_implementation_replay",
        },
    }
    (domains / "route_references.jsonl").write_text(
        json.dumps(route_reference_row, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_json(
        domains / "route_references_manifest.json",
        {
            "version": "v1",
            "schema_id": "trialagentbench.trialeval.route_reference_manifest/v1",
            "release_root": ".",
            "row_count": 1,
            "task_count": 1,
            "task_byte_ranges": route_reference_task_ranges(domains / "route_references.jsonl"),
            "route_references_jsonl_sha256": sha256_file(domains / "route_references.jsonl"),
            "evaluation_target_register_sha256": sha256_file(domains / "evaluation_target_register.jsonl"),
            "estimator_registry_sha256": sha256_file(domains / "estimator_registry.json"),
            "estimator_route_family_map_sha256": sha256_file(domains / "estimator_route_family_map.json"),
        },
    )
    write_json(
        domains / "method_route_register_manifest.json",
        {
            "version": "v1",
            "schema_id": "trialagentbench.trialeval.method_route_manifest/v1",
            "release_root": ".",
            "row_count": 1,
            "task_count": 1,
            "lane_count": 1,
            "method_route_register_jsonl_sha256": sha256_file(domains / "method_route_register.jsonl"),
            "estimator_route_family_map_sha256": sha256_file(domains / "estimator_route_family_map.json"),
            "evaluation_target_register_sha256": sha256_file(domains / "evaluation_target_register.jsonl"),
            "route_references_sha256": sha256_file(domains / "route_references.jsonl"),
        },
    )
    write_json(
        domains / "multi_primary_coverage_report.json",
        {
            "schema_id": "trialagentbench.trialeval.multi_primary_coverage_report/v1",
            "release_root": ".",
            "passed": True,
            "row_count": 1,
            "issue_count": 0,
            "rows": [
                {
                    "task_id": task_id,
                    "lane_id": "primary_numeric.v1",
                    "route_family": "global_cox_ph",
                    "credit_eligible_by_register": True,
                    "scoreable_as_primary": True,
                    "route_reference_count": 1,
                    "method_route_count": 1,
                    "eligibility_classes": ["required_primary"],
                }
            ],
            "issues": [],
        },
    )
    public_contract = {
        "schema_id": "trialagentbench.trialeval.public_estimand_contract/v1",
        "task_id": task_id,
        "item_id": "d1a1_rct_clean_01",
        "lane_id": "primary_numeric.v1",
        "estimand": {
            "estimand_id": "primary_itt",
            "treated_condition": "treated: Treated",
            "control_condition": "control: Control",
            "population": "ITT",
            "analysis_set": "ITT",
            "endpoint_id": "death",
            "endpoint_label": "All-cause death",
            "assessment_time_days": 365.0,
            "intercurrent_event_strategies": [
                {
                    "event_type": "discontinuation",
                    "strategy": "treatment_policy",
                    "description": "Retain assignment.",
                }
            ],
            "objective": "estimation",
            "direction": "two_sided",
            "alpha": 0.05,
            "multiplicity_strategy": "none",
            "multiplicity_family": "primary",
            "missing_data_strategy": "declared sensitivity analysis",
            "censoring_strategy": "right censoring",
            "design_family": "randomized_trial",
            "randomization_unit": "participant",
            "stratification_factors": [],
            "design_adjustments": [],
            "causal_identification": {
                "eligibility": "Enrolled ITT population",
                "treatment_strategies": ["Control", "Treated"],
                "assignment_mechanism": "participant randomized allocation",
                "time_zero": "randomization",
                "follow_up": "through day 365",
                "outcome": "All-cause death",
                "causal_contrast": "intention-to-treat effect",
                "adjustment_set": [],
                "positivity_support": "positive randomized allocation probability",
                "identifying_assumptions": ["consistency", "no interference"],
            },
            "checksum": "a" * 64,
        },
        "mode": "fixed_declared_estimand",
        "declared_primary_effect_scale": "log_hr",
        "primary_route_family": "global_cox_ph",
        "credit_eligible_route_families": ["global_cox_ph"],
        "public_evidence_basis": evidence_basis,
        "variants": [
            {
                "variant_id": f"{task_id}:primary_numeric.v1:global_cox_ph:log_hr",
                "route_family": "global_cox_ph",
                "effect_scale": "log_hr",
                "answer_shapes": ["numeric_point"],
                "required_modifiers": [],
                "eligibility_class": "required_primary",
                "route_reference_id": f"{task_id}:primary_numeric.v1:max_recoverable:observed:coxph_binary_breslow",
                "public_evidence_basis": evidence_basis,
                "rationale": "Minimal public estimand contract for harness tests.",
            }
        ],
    }
    (domains / "public_estimand_contract.jsonl").write_text(
        json.dumps(
            {
                "schema_id": "trialagentbench.trial_benchmark.grader_domain_row/v1",
                "domain": "public_estimand_contract",
                "task_id": task_id,
                "payload": {"contract": public_contract},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        domains / "public_estimand_contract_manifest.json",
        {
            "version": "v1",
            "schema_id": "trialagentbench.trialeval.public_estimand_contract_manifest/v1",
            "release_root": ".",
            "row_count": 1,
            "task_count": 1,
            "public_estimand_contract_jsonl_sha256": sha256_file(domains / "public_estimand_contract.jsonl"),
            "route_references_sha256": sha256_file(domains / "route_references.jsonl"),
        },
    )
    (domains / "method_composition.jsonl").write_text("", encoding="utf-8")
    write_json(
        domains / "method_composition_manifest.json",
        {
            "version": "v1",
            "schema_id": "trialagentbench.trialeval.method_composition_manifest/v1",
            "release_root": ".",
            "row_count": 0,
            "task_count": 0,
            "method_composition_jsonl_sha256": sha256_file(domains / "method_composition.jsonl"),
        },
    )
    score_scope_row = {
        "schema_id": "trialagentbench.trialeval.route_scoring_family_scope/v1",
        "task_id": task_id,
        "item_id": "d1a1_rct_clean_01",
        "lane_id": "primary_numeric.v1",
        "route_family": "global_cox_ph",
        "status": "official_scoreable",
        "credit_eligible_route_reference_ids": [
            f"{task_id}:primary_numeric.v1:max_recoverable:observed:coxph_binary_breslow"
        ],
        "credit_eligible_method_route_ids": [
            f"{task_id}:primary_numeric.v1:global_cox_ph:observed:coxph_binary_breslow:log_hr"
        ],
        "diagnostic_evidence_paths": [],
        "public_evidence_basis_paths": evidence_basis,
        "source_release_id": "unit",
    }
    (domains / "route_scoring_scope.jsonl").write_text(
        json.dumps(score_scope_row, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_json(
        domains / "route_scoring_scope_manifest.json",
        {
            "version": "v1",
            "schema_id": "trialagentbench.trialeval.route_scoring_scope_manifest/v1",
            "release_root": ".",
            "row_count": 1,
            "task_count": 1,
            "lane_count": 1,
            "route_scoring_scope_jsonl_sha256": sha256_file(domains / "route_scoring_scope.jsonl"),
            "evaluation_target_register_sha256": sha256_file(domains / "evaluation_target_register.jsonl"),
            "route_references_sha256": sha256_file(domains / "route_references.jsonl"),
            "method_route_register_sha256": sha256_file(domains / "method_route_register.jsonl"),
            "public_estimand_contract_sha256": sha256_file(domains / "public_estimand_contract.jsonl"),
        },
    )
    write_json(
        domains / "route_scoring_scope_gate_report.json",
        {
            "schema_id": "trialagentbench.trialeval.route_scoring_scope_gate_report/v1",
            "release_root": ".",
            "passed": True,
            "issues": [],
        },
    )
    public_reference_source_row = {
        "schema_id": "trialagentbench.trialeval.public_reference_source/v1",
        "task_id": task_id,
        "route_reference_id": f"{task_id}:primary_numeric.v1:max_recoverable:observed:coxph_binary_breslow",
        "estimator_method_id": "observed:coxph_binary_breslow",
        "source_mode": "public_raw_reconstruction",
        "public_evidence_refs": evidence_basis,
        "required_table_refs": [],
        "reconstruction_policy_id": "unit.public_reconstruction.v1",
    }
    (domains / "public_reference_sources.jsonl").write_text(
        json.dumps(public_reference_source_row, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_json(
        domains / "public_reference_sources_manifest.json",
        {
            "version": "v1",
            "schema_id": "trialagentbench.trialeval.public_reference_source_manifest/v1",
            "release_root": ".",
            "row_count": 1,
            "task_count": 1,
            "public_reference_sources_jsonl_sha256": sha256_file(domains / "public_reference_sources.jsonl"),
            "route_references_sha256": sha256_file(domains / "route_references.jsonl"),
        },
    )
    scoring_key = {
        "schema_id": "trialagentbench.scoring_key/v1",
        "release_id": "unit",
        "item_id": task_id,
        "question_id": "fixture.primary",
        "context_tier": "C1",
        "credit_eligible_routes": [
            {
                "route_id": "fixture.coxph",
                "signature": {
                    "analysis_population_id": "intention_to_treat",
                    "estimand_id": "primary_itt",
                    "intercurrent_event_strategy_ids": ["rescue_therapy:treatment_policy"],
                    "assessment_horizon_days": 365.0,
                    "treatment_id": "active",
                    "comparator_id": "control",
                    "endpoint_id": "primary_endpoint",
                    "effect_scale": "log_hr",
                    "analysis_method_id": "coxph_binary_wald",
                },
                "method": {
                    "analysis_method_id": "coxph_binary_wald",
                    "estimator_family": "coxph_binary",
                    "result_kind": "numeric_point",
                    "uncertainty_method": "wald_model_based",
                    "design_modifiers": [],
                },
                "required_identification_assumptions": ["randomization_exchangeability"],
                "required_diagnostics": ["randomization_integrity_public"],
                "planning_calculator_id": None,
                "target": {
                    "kind": "numeric_point",
                    "value": -0.5,
                    "result_unit": "log_hazard_ratio",
                    "acceptance_envelope": {
                        "schema_id": "trialagentbench.numerical_acceptance_envelope/v1",
                        "reporting_decimal_places": 12,
                        "independent_max_abs_difference": 5e-13,
                        "public_verification_id": "fixture-public-replay",
                        "independent_verification_ids": ["fixture-independent-replay"],
                    },
                    "require_confidence_interval": True,
                    "confidence_interval_lower": -0.7,
                    "confidence_interval_upper": -0.3,
                },
            }
        ],
    }
    scoring_keys_body = (json.dumps(scoring_key, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    grader = Path(root) / "grader"
    (grader / "scoring_keys.jsonl").write_bytes(scoring_keys_body)
    write_json(
        grader / "scoring_key_manifest.json",
        {
            "schema_id": "trialagentbench.scoring_key_manifest/v1",
            "release_id": "unit",
            "specification_sha256": "a" * 64,
            "scoring_keys_sha256": hashlib.sha256(scoring_keys_body).hexdigest(),
            "item_ids": [task_id],
        },
    )


def write_route_reference_fixture_zip(
    tmp_path: Path,
    *,
    include_truth_input: bool,
    include_table: bool = True,
) -> Path:
    """Write a compact evaluator zip with optional route-reference input materialization."""

    from zipfile import ZipFile

    root = tmp_path / "release_root"
    domains = root / "grader" / "domains"
    write_minimal_ground_truth_domains(root, task_id="TASK0001")
    write_json(root / "grader" / "item_index.json", {"entries": [{"task_id": "TASK0001"}]})
    truth_inputs_path = domains / "route_reference_inputs.jsonl"
    if include_truth_input:
        table_rel_path = "items/TASK0001/data/analysis_frame.parquet"
        table_path = root / table_rel_path
        table_path.parent.mkdir(parents=True, exist_ok=True)
        if include_table:
            table_path.write_text("fixture table", encoding="utf-8")
            table_sha256 = sha256_file(table_path)
        else:
            table_sha256 = "0" * 64
        truth_input = {
            "schema_id": "trialagentbench.trialeval.route_reference_input/v1",
            "task_id": "TASK0001",
            "input_bundle_id": "input:TASK0001:coxph",
            "estimator_method_id": "observed:coxph_binary_breslow",
            "effect_scale": "log_hr",
            "lane_ids": ["primary_numeric.v1"],
            "route_reference_ids": ["TASK0001:primary_numeric.v1:max_recoverable:observed:coxph_binary_breslow"],
            "required_table_refs": [
                {
                    "rel_path": table_rel_path,
                    "semantic_role": "canonical_analysis",
                    "sha256": table_sha256,
                    "row_count": 3,
                    "column_names": ["USUBJID", "AVAL"],
                }
            ],
            "source_role": "canonical_analysis",
        }
        truth_inputs_path.write_text(json.dumps(truth_input, sort_keys=True) + "\n", encoding="utf-8")
    else:
        truth_inputs_path.write_text("", encoding="utf-8")
    write_json(
        domains / "route_reference_inputs_manifest.json",
        {
            "version": "v1",
            "schema_id": "trialagentbench.trialeval.route_reference_input_manifest/v1",
            "release_root": ".",
            "row_count": 1 if include_truth_input else 0,
            "table_count": 1 if include_truth_input else 0,
            "task_count": 1 if include_truth_input else 0,
            "route_reference_inputs_jsonl_sha256": sha256_file(truth_inputs_path),
            "route_references_sha256": sha256_file(domains / "route_references.jsonl"),
        },
    )
    write_json(
        domains / "reference_closure_report.json",
        {
            "schema_id": "trialagentbench.trialeval.reference_closure_report/v1",
            "release_root": ".",
            "passed": True,
            "row_count": 1,
            "public_source_row_count": 1,
            "non_promotable_source_count": 0,
            "hidden_diagnostic_row_count": 0,
            "failed_parity_count": 0,
            "missing_public_reference_source_count": 0,
            "missing_route_reference_input_count": 0,
            "public_reference_sources_sha256": sha256_file(domains / "public_reference_sources.jsonl"),
            "reference_regeneration_report_sha256": "0" * 64,
            "route_reference_inputs_sha256": sha256_file(truth_inputs_path),
            "issues": [],
        },
    )
    evaluator_zip = tmp_path / "evaluator.zip"
    with ZipFile(evaluator_zip, "w") as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.relative_to(root).as_posix().startswith("items/"):
                zf.write(path, path.relative_to(root).as_posix())
    return evaluator_zip


def write_route_reference_fixture_public_zip(
    tmp_path: Path,
    *,
    include_table: bool = True,
) -> Path:
    """Write the participant evidence paired with a scoreable-truth fixture."""

    from zipfile import ZipFile

    public_zip = tmp_path / "public.zip"
    with ZipFile(public_zip, "w") as zf:
        zf.writestr("items/TASK0001/task.json", "{}\n")
        if include_table:
            zf.writestr("items/TASK0001/data/analysis_frame.parquet", "fixture table")
    return public_zip
