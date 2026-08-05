"""Compact release fixtures for independent TrialEval replay tests."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from zipfile import ZipFile


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_scoreable_reference_fixture_zip(
    tmp_path: Path,
    *,
    include_reference_input: bool,
    include_table: bool = True,
) -> Path:
    """Write the smallest evaluator archive passed by public replay."""

    task_id = "TASK0001"
    variant_id = (
        f"{task_id}:primary_numeric.v1:max_recoverable:observed:coxph_binary_breslow"
    )
    root = tmp_path / "release_root"
    domains = root / "grader" / "domains"
    domains.mkdir(parents=True, exist_ok=True)
    evidence = [f"items/{task_id}/task.json", f"items/{task_id}/protocol_summary.json"]

    _write_json(
        root / "grader" / "item_index.json", {"entries": [{"task_id": task_id}]}
    )
    register = {
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
        "public_evidence_basis": evidence,
        "score_profile_eligibility": [
            "credit_eligible_family_v1",
            "strict_method_id_v1",
            "diagnostic_recognition_v1",
        ],
    }
    register_path = domains / "evaluation_target_register.jsonl"
    register_path.write_text(
        json.dumps(register, sort_keys=True) + "\n", encoding="utf-8"
    )

    tolerance = math.sqrt(sys.float_info.epsilon)
    route_reference = {
        "schema_id": "trialagentbench.trialeval.route_reference/v1",
        "task_id": task_id,
        "item_id": "d1a1_rct_clean_01",
        "lane_id": "primary_numeric.v1",
        "route_reference_id": variant_id,
        "variant_role": "required_primary",
        "route_family": "global_cox_ph",
        "estimator_method_id": "observed:coxph_binary_breslow",
        "effect_scale": "log_hr",
        "answer_shape": "point",
        "value": -0.5,
        "standard_error": 0.1,
        "ci_low": -0.7,
        "ci_high": -0.3,
        "public_evidence_basis": evidence,
        "required_modifiers": [],
        "identification_class": "point_identified",
        "support_status": "official_supported",
        "support_rationale": "Minimal public replay fixture.",
        "numerical_equivalence": {
            "policy_id": "float64_sqrt_epsilon_v1",
            "absolute_tolerance": tolerance,
            "relative_tolerance": tolerance,
            "basis": "deterministic_cross_implementation_replay",
        },
    }
    variants_path = domains / "route_references.jsonl"
    variants_path.write_text(
        json.dumps(route_reference, sort_keys=True) + "\n", encoding="utf-8"
    )

    public_source = {
        "schema_id": "trialagentbench.trialeval.public_reference_source/v1",
        "task_id": task_id,
        "route_reference_id": variant_id,
        "estimator_method_id": "observed:coxph_binary_breslow",
        "source_mode": "public_raw_reconstruction",
        "public_evidence_refs": evidence,
        "required_table_refs": [],
        "reconstruction_policy_id": "unit.public_reconstruction.v1",
    }
    sources_path = domains / "public_reference_sources.jsonl"
    sources_path.write_text(
        json.dumps(public_source, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_json(
        domains / "public_reference_sources_manifest.json",
        {
            "version": "v1",
            "schema_id": "trialagentbench.trialeval.public_reference_source_manifest/v1",
            "release_root": ".",
            "row_count": 1,
            "task_count": 1,
            "public_reference_sources_jsonl_sha256": _sha256(sources_path),
            "route_references_sha256": _sha256(variants_path),
        },
    )

    estimand = {
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
                "event_id": "treatment_discontinuation",
                "event_label": "Treatment discontinuation",
                "event_type": "discontinuation",
                "strategy": "treatment_policy",
                "description": "Retain randomized assignment.",
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
    }
    contract = {
        "schema_id": "trialagentbench.trialeval.public_estimand_contract/v1",
        "task_id": task_id,
        "item_id": "d1a1_rct_clean_01",
        "lane_id": "primary_numeric.v1",
        "estimand": estimand,
        "mode": "fixed_declared_estimand",
        "declared_primary_effect_scale": "log_hr",
        "primary_route_family": "global_cox_ph",
        "credit_eligible_route_families": ["global_cox_ph"],
        "public_evidence_basis": evidence,
        "variants": [
            {
                "variant_id": f"{task_id}:primary_numeric.v1:global_cox_ph:log_hr",
                "estimand_id": estimand["estimand_id"],
                "route_family": "global_cox_ph",
                "effect_scale": "log_hr",
                "answer_shapes": ["numeric_point"],
                "required_modifiers": [],
                "eligibility_class": "required_primary",
                "route_reference_id": variant_id,
                "public_evidence_basis": evidence,
                "rationale": "Minimal public replay fixture.",
            }
        ],
    }
    contract_row = {
        "schema_id": "trialagentbench.trial_benchmark.grader_domain_row/v1",
        "domain": "public_estimand_contract",
        "task_id": task_id,
        "payload": {"contract": contract},
    }
    (domains / "public_estimand_contract.jsonl").write_text(
        json.dumps(contract_row, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    reference_inputs_path = domains / "route_reference_inputs.jsonl"
    if include_reference_input:
        table_rel_path = f"items/{task_id}/data/analysis_frame.parquet"
        table_path = root / table_rel_path
        table_path.parent.mkdir(parents=True, exist_ok=True)
        if include_table:
            table_path.write_bytes(b"fixture table")
            table_sha256 = _sha256(table_path)
        else:
            table_sha256 = "0" * 64
        reference_input = {
            "schema_id": "trialagentbench.trialeval.route_reference_input/v1",
            "task_id": task_id,
            "input_bundle_id": f"input:{task_id}:coxph",
            "estimator_method_id": "observed:coxph_binary_breslow",
            "effect_scale": "log_hr",
            "lane_ids": ["primary_numeric.v1"],
            "route_reference_ids": [variant_id],
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
        reference_inputs_path.write_text(
            json.dumps(reference_input, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        reference_inputs_path.write_text("", encoding="utf-8")
    _write_json(
        domains / "route_reference_inputs_manifest.json",
        {
            "version": "v1",
            "schema_id": "trialagentbench.trialeval.route_reference_input_manifest/v1",
            "release_root": ".",
            "row_count": int(include_reference_input),
            "table_count": int(include_reference_input),
            "task_count": int(include_reference_input),
            "route_reference_inputs_jsonl_sha256": _sha256(reference_inputs_path),
            "route_references_sha256": _sha256(variants_path),
        },
    )

    evaluator_zip = tmp_path / "evaluator.zip"
    with ZipFile(evaluator_zip, "w") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.relative_to(root).as_posix().startswith(
                "items/"
            ):
                archive.write(path, path.relative_to(root).as_posix())
    return evaluator_zip
