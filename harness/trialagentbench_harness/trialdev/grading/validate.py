"""Release and submission validation for the TrialDev grader."""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import ValidationError

from trialagentbench_harness.trialdev.grading.hashing import compute_sha256_hex, sha256_file_hex
from trialagentbench_harness.trialdev.grading.io import read_json
from trialagentbench_harness.trialdev.grading.models import TrialDevelopmentSubmissionV1
from trialagentbench_harness.trialdev.share.validate import candidate_ids_by_role_v1

__all__ = ["validate_release_v1", "validate_submission_v1"]


def _sorted_unique_strings(values: object) -> list[str]:
    if not isinstance(values, list | tuple):
        return []
    return sorted({str(value) for value in values})


def _finite_number(record: dict[str, object], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"public candidate score requires numeric {key!r}.")
    return float(value)


def _canonicalize_grader_manifest(payload: dict[str, object]) -> dict[str, object]:
    raw_records = payload.get("records", [])
    records = list(raw_records) if isinstance(raw_records, list | tuple) else []
    records.sort(
        key=lambda record: (
            str((record if isinstance(record, dict) else {}).get("phase_id", "")),
            str((record if isinstance(record, dict) else {}).get("lane_id", "")),
            str((record if isinstance(record, dict) else {}).get("objective_id", "")),
            str((record if isinstance(record, dict) else {}).get("metric", "")),
            str((record if isinstance(record, dict) else {}).get("endpoint_id", "")),
            str((record if isinstance(record, dict) else {}).get("method_route_id", "")),
            tuple(
                str(value)
                for value in (((record if isinstance(record, dict) else {}).get("candidate_drug_ids", [])) or [])
            ),
        )
    )
    normalized = dict(payload)
    normalized["records"] = records
    return normalized


def _recompute_checksum(payload: dict[str, object], *, label: str) -> str:
    normalized = dict(payload)
    normalized.pop("checksum", None)
    if label == "grading_procedure.json":
        normalized["supported_lanes"] = _sorted_unique_strings(normalized.get("supported_lanes", ()))
        normalized["supported_objectives"] = _sorted_unique_strings(normalized.get("supported_objectives", ()))
    elif label == "submission_schema.json":
        normalized["required_sections"] = _sorted_unique_strings(normalized.get("required_sections", ()))
    elif label.endswith("_manifest.json"):
        normalized = _canonicalize_grader_manifest(normalized)
    return str(compute_sha256_hex(normalized))


def _check_checksum(payload: object, *, label: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object.")
    expected = str(payload.get("checksum", ""))
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected.lower()):
        raise ValueError(f"{label} missing a valid checksum.")
    computed = _recompute_checksum(payload, label=label)
    if str(computed) != str(expected):
        raise ValueError(f"{label} checksum mismatch.")


def _validate_method_result(
    *,
    result: dict[str, object],
    candidate_ids: set[str],
    objective_margins: dict[str, float],
    observational_minimum_benefit: float,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Validate one method's numerical reference and action policies."""

    method_route_id = str(result.get("method_route_id") or "")
    estimator_id = str(result.get("estimator_id") or "")
    if not method_route_id or not estimator_id:
        raise ValueError("method-specific recoverability requires method_route_id and estimator_id.")
    scores = result.get("candidate_scores")
    policies = result.get("objective_policies")
    action_policies = result.get("observational_action_policies")
    diagnostics = result.get("diagnostics")
    if (
        not isinstance(scores, list)
        or not isinstance(policies, list)
        or not isinstance(action_policies, list)
        or not isinstance(diagnostics, dict)
    ):
        raise ValueError("method result requires scores, policies, actions, and diagnostics.")
    if (
        str(diagnostics.get("method_route_id") or "") != method_route_id
        or str(diagnostics.get("estimator_id") or "") != estimator_id
    ):
        raise ValueError("method result identity disagrees with its diagnostics.")
    policy_by_objective = {str(policy.get("objective_id")): policy for policy in policies if isinstance(policy, dict)}
    action_by_objective = {
        str(policy.get("objective_id")): policy for policy in action_policies if isinstance(policy, dict)
    }
    if (
        set(policy_by_objective) != set(objective_margins)
        or set(action_by_objective) != set(objective_margins)
        or len(policy_by_objective) != len(policies)
        or len(action_by_objective) != len(action_policies)
    ):
        raise ValueError("each method result must cover every objective exactly once.")
    score_keys = {
        (str(score.get("objective_id")), str(score.get("candidate_drug_id")))
        for score in scores
        if isinstance(score, dict)
    }
    if len(score_keys) != len(scores):
        raise ValueError("method-specific candidate scores must be unique objects.")
    comparisons = result.get("estimator_comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("method result requires estimator comparisons.")
    for objective_id in objective_margins:
        observed_estimators = {
            str(row.get("estimator_id"))
            for row in comparisons
            if isinstance(row, dict) and str(row.get("objective_id")) == objective_id
        }
        if {estimator_id, "raw_observed"} - observed_estimators:
            raise ValueError(f"method result lacks required comparisons for {objective_id!r}.")
        objective_scores = {
            str(score.get("candidate_drug_id")): score
            for score in scores
            if isinstance(score, dict) and str(score.get("objective_id")) == objective_id
        }
        if set(objective_scores) != candidate_ids:
            raise ValueError(f"method-specific candidate universe mismatch for {objective_id!r}.")
        policy = policy_by_objective[objective_id]
        action_policy = action_by_objective[objective_id]
        threshold = _finite_number(action_policy, "minimum_efficacy_gain")
        if not math.isclose(threshold, observational_minimum_benefit, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("observational efficacy threshold drifts from decision_charter.json.")
        for score in objective_scores.values():
            _finite_number(score, "max_abs_unadjusted_smd_vs_target")
            if score.get("inference_estimable") is True:
                _finite_number(score, "max_abs_adjusted_smd_vs_target")
            elif score.get("max_abs_adjusted_smd_vs_target") is not None:
                raise ValueError("non-estimable candidate cannot declare adjusted balance.")
        if str(policy.get("policy")) == "insufficient_recoverability":
            expected_action = "withhold_nomination"
            if set(_sorted_unique_strings(action_policy.get("reference_target_ids"))) != {expected_action} or set(
                _sorted_unique_strings(action_policy.get("credit_eligible_target_ids"))
            ) != {expected_action}:
                raise ValueError("insufficient recoverability requires qualified non-nomination.")
            continue
        if not math.isclose(
            _finite_number(policy, "near_tie_threshold"),
            objective_margins[objective_id],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("public utility indifference margin drifts from objective_charter.json.")
        if not all(
            score.get("point_estimable") is True and score.get("inference_estimable") is True
            for score in objective_scores.values()
        ):
            raise ValueError("scoreable observational objectives require complete candidate inference.")
        definite = {
            candidate_id
            for candidate_id, score in objective_scores.items()
            if _finite_number(score, "efficacy_gain_ci_low") >= threshold
        }
        possible = {
            candidate_id
            for candidate_id, score in objective_scores.items()
            if _finite_number(score, "efficacy_gain_ci_high") >= threshold
        }
        if definite != set(_sorted_unique_strings(action_policy.get("definitely_qualified_candidate_ids"))):
            raise ValueError("definitely qualified candidates drift from method-specific intervals.")
        if possible != set(_sorted_unique_strings(action_policy.get("possibly_qualified_candidate_ids"))):
            raise ValueError("possibly qualified candidates drift from method-specific intervals.")
        contrast_half_widths = action_policy.get("utility_contrast_half_widths")
        if not isinstance(contrast_half_widths, dict) or set(contrast_half_widths) != possible:
            raise ValueError("utility contrast half-widths must cover exactly the possibly qualified candidates.")
        resolved_half_widths = {
            candidate_id: _finite_number(contrast_half_widths, candidate_id) for candidate_id in possible
        }
        if any(value < 0.0 for value in resolved_half_widths.values()):
            raise ValueError("utility contrast half-widths must be non-negative.")
        stop_target = "withhold_nomination"
        if possible:
            best = max(_finite_number(objective_scores[candidate], "adjusted_utility") for candidate in possible)
            expected_acceptable = {
                candidate
                for candidate in possible
                if best - _finite_number(objective_scores[candidate], "adjusted_utility")
                <= max(objective_margins[objective_id], resolved_half_widths[candidate])
            }
        else:
            expected_acceptable = set()
        if not definite:
            expected_acceptable.add(stop_target)
            expected_reference = stop_target
        else:
            expected_reference = min(
                expected_acceptable,
                key=lambda candidate: (-_finite_number(objective_scores[candidate], "adjusted_utility"), candidate),
            )
        if set(_sorted_unique_strings(action_policy.get("credit_eligible_target_ids"))) != expected_acceptable:
            raise ValueError("acceptable actions drift from method-specific efficacy and utility references.")
        if _sorted_unique_strings(action_policy.get("reference_target_ids")) != [expected_reference]:
            raise ValueError("reference action drifts from method-specific efficacy and utility references.")
    return policy_by_objective, action_by_objective


def validate_release_v1(*, scenario_root: Path) -> None:
    """Validate the hidden grader surface for one scenario bundle."""
    from trialagentbench_harness.trialdev.grading.design_frontier import load_phase_design_frontiers_v1

    root = Path(scenario_root)
    load_phase_design_frontiers_v1(scenario_root=root)
    safety_policy_path = root / "public" / "safety_decision_policy.json"
    if not safety_policy_path.is_file():
        raise FileNotFoundError(f"Missing public safety policy artifact: {safety_policy_path}")
    safety_policy = read_json(safety_policy_path)
    if not isinstance(safety_policy, dict):
        raise ValueError("safety_decision_policy.json must be a JSON object.")
    if safety_policy.get("schema_id") != "trialdev_safety_decision_policy_v1":
        raise ValueError("safety_decision_policy.json has unsupported schema_id.")
    thresholds = safety_policy.get("thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("safety_decision_policy.thresholds must be a non-empty array.")
    hard_gate_phases: set[str] = set()
    for threshold in thresholds:
        if not isinstance(threshold, dict):
            raise ValueError("safety_decision_policy.threshold entries must be objects.")
        role = str(threshold.get("role", ""))
        component_id = str(threshold.get("component_id"))
        if role not in {"hard_gate", "diagnostic_only"}:
            raise ValueError("safety_decision_policy.threshold entries require role.")
        horizon = threshold.get("evaluation_horizon_days")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("safety_decision_policy.threshold entries require a positive integer horizon.")
        if component_id in {"discontinuation", "ltfu"} and role == "hard_gate":
            raise ValueError("diagnostic safety components cannot be hard gates.")
        if role == "hard_gate":
            hard_gate_phases.add(str(threshold.get("phase_id")))
    if hard_gate_phases != {"phase1", "phase2", "phase3"}:
        raise ValueError("safety_decision_policy must define one hard-gate component for each trial phase.")
    for phase_id in hard_gate_phases:
        phase_horizons = {
            int(threshold["evaluation_horizon_days"])
            for threshold in thresholds
            if str(threshold.get("phase_id")) == phase_id
        }
        if len(phase_horizons) != 1:
            raise ValueError(f"safety thresholds for {phase_id!r} must share one evaluation horizon.")
    grader_dir = root / "grader"
    if not grader_dir.is_dir():
        raise FileNotFoundError(f"Missing grader directory: {grader_dir}")
    required = (
        "grading_procedure.json",
        "submission_schema.json",
        "drug_ranking_reference_manifest.json",
        "public_recoverability_report.json",
    )
    for rel in required:
        path = grader_dir / rel
        if not path.is_file():
            raise FileNotFoundError(f"Missing grader artifact: {path}")
        payload = read_json(path)
        _check_checksum(payload, label=rel)
        if rel == "public_recoverability_report.json" and (
            not isinstance(payload, dict) or payload.get("schema_id") != "trialdev_public_recoverability_report_v1"
        ):
            raise ValueError("public_recoverability_report.json has unsupported schema_id.")
    register_path = grader_dir / "evaluation_target_register.jsonl"
    if not register_path.is_file() or not register_path.read_text(encoding="utf-8").strip():
        raise FileNotFoundError(f"Missing or empty grader artifact: {register_path}")
    candidate_ids = set(candidate_ids_by_role_v1(scenario_root=root)["investigational"])
    recoverability = read_json(grader_dir / "public_recoverability_report.json")
    method_results = recoverability.get("method_results") if isinstance(recoverability, dict) else None
    policies = recoverability.get("method_union_objective_sensitivity") if isinstance(recoverability, dict) else None
    action_policies = (
        recoverability.get("method_union_action_sensitivity") if isinstance(recoverability, dict) else None
    )
    if (
        not isinstance(method_results, list)
        or len(method_results) < 2
        or not isinstance(policies, list)
        or not isinstance(action_policies, list)
    ):
        raise ValueError(
            "public recoverability requires multiple method results and cross-method sensitivity summaries."
        )
    objective_spec = read_json(root / "public" / "objective_charter.json")
    raw_objectives = objective_spec.get("objectives") if isinstance(objective_spec, dict) else None
    if not isinstance(raw_objectives, list) or not raw_objectives:
        raise ValueError("objective_charter.json requires a non-empty objectives array.")
    objective_margins = {
        str(record.get("objective_id")): _finite_number(record, "indifference_margin")
        for record in raw_objectives
        if isinstance(record, dict)
    }
    if len(objective_margins) != len(raw_objectives):
        raise ValueError("objective_charter.json requires unique objective records.")
    decision_charter = read_json(root / "public" / "decision_charter.json")
    efficacy_rules = decision_charter.get("efficacy_rules") if isinstance(decision_charter, dict) else None
    observational_rules = [
        rule
        for rule in efficacy_rules or []
        if isinstance(rule, dict) and str(rule.get("phase_id")) == "observational_review"
    ]
    if len(observational_rules) != 1:
        raise ValueError("decision_charter.json requires exactly one observational efficacy rule.")
    observational_minimum_benefit = _finite_number(observational_rules[0], "minimum_benefit")
    input_checksums = recoverability.get("public_input_checksums")
    if not isinstance(input_checksums, list) or not input_checksums:
        raise ValueError("public_recoverability_report.json requires public_input_checksums.")
    observed_input_paths: set[str] = set()
    for record in input_checksums:
        if not isinstance(record, dict):
            raise ValueError("public_input_checksums entries must be objects.")
        relative_path = str(record.get("path") or "")
        expected_checksum = str(record.get("sha256") or "")
        if not relative_path.startswith("public/") or relative_path in observed_input_paths:
            raise ValueError("public_input_checksums require unique public artifact paths.")
        artifact_path = root / relative_path
        if not artifact_path.is_file() or sha256_file_hex(artifact_path) != expected_checksum:
            raise ValueError(f"public recoverability input checksum mismatch: {relative_path!r}.")
        observed_input_paths.add(relative_path)
    parsed_method_results = [result for result in method_results if isinstance(result, dict)]
    if len(parsed_method_results) != len(method_results):
        raise ValueError("method_results entries must be objects.")
    method_ids = {str(result.get("method_route_id") or "") for result in parsed_method_results}
    estimator_ids = {str(result.get("estimator_id") or "") for result in parsed_method_results}
    if (
        "" in method_ids
        or "" in estimator_ids
        or len(method_ids) != len(parsed_method_results)
        or len(estimator_ids) != len(parsed_method_results)
    ):
        raise ValueError("method results require unique method and estimator identities.")
    method_policies: list[dict[str, dict[str, object]]] = []
    method_actions: list[dict[str, dict[str, object]]] = []
    for result in parsed_method_results:
        result_policies, result_actions = _validate_method_result(
            result=result,
            candidate_ids=candidate_ids,
            objective_margins=objective_margins,
            observational_minimum_benefit=observational_minimum_benefit,
        )
        method_policies.append(result_policies)
        method_actions.append(result_actions)
    objective_ids = {str(policy.get("objective_id")) for policy in policies if isinstance(policy, dict)}
    if len(objective_ids) != len(policies):
        raise ValueError("public recoverability objective policies must be unique objects.")
    if objective_ids != set(objective_margins):
        raise ValueError("public recoverability objectives must exactly match objective_charter.json.")
    action_objective_ids = {str(policy.get("objective_id")) for policy in action_policies if isinstance(policy, dict)}
    if action_objective_ids != objective_ids or len(action_objective_ids) != len(action_policies):
        raise ValueError("observational action policies must uniquely cover every objective.")
    for objective_id in objective_ids:
        policy = next(
            policy
            for policy in policies
            if isinstance(policy, dict) and str(policy.get("objective_id")) == objective_id
        )
        targets = set(str(value) for value in policy.get("reference_target_ids", [])) | set(
            str(value) for value in policy.get("acceptable_candidate_set", [])
        )
        if not targets <= candidate_ids:
            raise ValueError(
                f"public recoverability declares non-candidate targets for objective_id={objective_id!r}."
            )
        action_policy = next(
            policy
            for policy in action_policies
            if isinstance(policy, dict) and str(policy.get("objective_id")) == objective_id
        )
        action_targets = set(str(value) for value in action_policy.get("reference_target_ids", [])) | set(
            str(value) for value in action_policy.get("credit_eligible_target_ids", [])
        )
        allowed_action_targets = candidate_ids | {
            "withhold_nomination",
        }
        if not action_targets or not action_targets <= allowed_action_targets or "control" in action_targets:
            raise ValueError(f"observational action policy has invalid targets for objective_id={objective_id!r}.")
        estimable_policy_records = [
            result_policies[objective_id]
            for result_policies in method_policies
            if str(result_policies[objective_id].get("policy")) != "insufficient_recoverability"
        ]
        expected_policy_reference = {
            target
            for method_policy in estimable_policy_records
            for target in _sorted_unique_strings(method_policy.get("reference_target_ids"))
        }
        expected_policy_acceptable = {
            target
            for method_policy in estimable_policy_records
            for target in _sorted_unique_strings(method_policy.get("acceptable_candidate_set"))
        }
        if estimable_policy_records:
            if set(_sorted_unique_strings(policy.get("reference_target_ids"))) != expected_policy_reference:
                raise ValueError("consensus utility reference targets drift from method-specific policies.")
            if set(_sorted_unique_strings(policy.get("acceptable_candidate_set"))) != expected_policy_acceptable:
                raise ValueError("consensus utility acceptable set drifts from method-specific policies.")
        elif (
            str(policy.get("policy")) != "insufficient_recoverability"
            or _sorted_unique_strings(policy.get("reference_target_ids"))
            or _sorted_unique_strings(policy.get("acceptable_candidate_set"))
        ):
            raise ValueError("consensus utility must report insufficient recoverability.")
        estimable_action_records = [
            result_actions[objective_id]
            for result_policies, result_actions in zip(method_policies, method_actions, strict=True)
            if str(result_policies[objective_id].get("policy")) != "insufficient_recoverability"
        ]
        expected_action_reference = {
            target
            for method_action in estimable_action_records
            for target in _sorted_unique_strings(method_action.get("reference_target_ids"))
        }
        expected_action_acceptable = {
            target
            for method_action in estimable_action_records
            for target in _sorted_unique_strings(method_action.get("credit_eligible_target_ids"))
        }
        if estimable_action_records:
            expected_possible = {
                target
                for method_action in estimable_action_records
                for target in _sorted_unique_strings(method_action.get("possibly_qualified_candidate_ids"))
            }
            expected_definite = set.intersection(
                *(
                    set(_sorted_unique_strings(method_action.get("definitely_qualified_candidate_ids")))
                    for method_action in estimable_action_records
                )
            )
            if set(_sorted_unique_strings(action_policy.get("reference_target_ids"))) != expected_action_reference:
                raise ValueError("consensus action reference targets drift from method-specific policies.")
            if (
                set(_sorted_unique_strings(action_policy.get("credit_eligible_target_ids")))
                != expected_action_acceptable
            ):
                raise ValueError("consensus acceptable actions drift from method-specific policies.")
            if (
                set(_sorted_unique_strings(action_policy.get("definitely_qualified_candidate_ids")))
                != expected_definite
                or set(_sorted_unique_strings(action_policy.get("possibly_qualified_candidate_ids")))
                != expected_possible
            ):
                raise ValueError("consensus qualification sets drift from method-specific policies.")
        elif set(_sorted_unique_strings(action_policy.get("reference_target_ids"))) != {"withhold_nomination"} or set(
            _sorted_unique_strings(action_policy.get("credit_eligible_target_ids"))
        ) != {"withhold_nomination"}:
            raise ValueError("consensus action must report insufficient public recoverability.")


def validate_submission_v1(*, submission_path: Path) -> TrialDevelopmentSubmissionV1:
    """Validate one structured submission."""
    payload = read_json(Path(submission_path))
    try:
        return TrialDevelopmentSubmissionV1.model_validate(payload)
    except ValidationError as exc:  # pragma: no cover
        raise ValueError(str(exc)) from exc
