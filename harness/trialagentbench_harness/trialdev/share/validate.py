"""Bundle validation utilities for TrialDev participant bundles."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from trialagentbench_harness.numeric_policy import (
    FLOAT_EQUALITY_ABS_TOLERANCE_V1,
    MINIMUM_TWO_SIDED_CONFIDENCE_LEVEL_V1,
)
from trialagentbench_harness.trialdev.share.hashing import compute_sha256_hex, sha256_file_hex
from trialagentbench_harness.trialdev.share.io import read_json
from trialagentbench_harness.trialdev.share.models import (
    FrozenSuperpopulationManifestV1,
    PhaseModuleSpecV1,
    ScenarioBundleManifestV1,
    SuperpopulationQualificationSummaryV1,
    TrialDevelopmentEvalContractV1,
    TrialDevelopmentEvaluationReferenceManifestV1,
    TrialDevelopmentGradingProcedureV1,
    TrialDevelopmentPhaseTargetsManifestV1,
    TrialDevelopmentRequestV1,
    TrialDevelopmentSafetyReferenceManifestV1,
    TrialDevelopmentSubmissionSchemaV1,
)
from trialagentbench_harness.trialdev.share.public_method_design import (
    TrialDevPhaseAnalysisMethodCatalogV1,
    TrialDevPhaseDesignPolicyV1,
    TrialDevPublicObjectiveCharterV1,
    TrialDevPublicObservationalMethodCatalogV1,
)
from trialagentbench_harness.trialdev.share.safety_policy import load_safety_policy_v1, serious_event_definitions_v1
from trialagentbench_harness.trialdev.share.sequential import (
    TrialDevelopmentPhaseActionPolicyV1,
    TrialDevelopmentProgramLoopManifestV1,
    validate_phase_action_policy_file_v1,
    validate_phase_analysis_file_v1,
    validate_phase_decision_against_policy_v1,
    validate_phase_decision_file_v1,
    validate_trial_output_bundle_v1,
)

__all__ = [
    "TrialDevelopmentRequestRejectedError",
    "candidate_ids_by_role_v1",
    "validate_public_scenario_bundle_v1",
    "validate_phase_action_policy_file_v1",
    "validate_phase_analysis_file_v1",
    "validate_phase_decision_against_policy_v1",
    "validate_phase_decision_file_v1",
    "validate_request_file_v1",
    "validate_request_against_scenario_v1",
    "validate_request_against_scenario_file_v1",
    "validate_request_shape_file_v1",
    "validate_scenario_bundle_v1",
    "validate_submission_shape_file_v1",
    "validate_trial_output_bundle_v1",
]


class TrialDevelopmentRequestRejectedError(ValueError):
    """A participant-correctable request violation against the public menu."""


def candidate_ids_by_role_v1(*, scenario_root: Path) -> dict[str, tuple[str, ...]]:
    """Return validated public candidate identifiers grouped by declared role."""

    payload = read_json(Path(scenario_root) / "public" / "candidate_drug_catalog.json")
    records = payload.get("candidate_drugs") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("candidate_drug_catalog.json must contain candidate_drugs array.")
    by_role: dict[str, list[str]] = {"control": [], "investigational": []}
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("candidate_drug_catalog.json candidate records must be objects.")
        candidate_id = str(record.get("candidate_drug_id") or "")
        role = str(record.get("role") or "")
        if not candidate_id or candidate_id in seen:
            raise ValueError("candidate_drug_catalog.json requires unique non-empty candidate identifiers.")
        if role not in by_role:
            raise ValueError("candidate_drug_catalog.json roles must be control or investigational.")
        seen.add(candidate_id)
        by_role[role].append(candidate_id)
    if len(by_role["control"]) != 1 or not by_role["investigational"]:
        raise ValueError("candidate_drug_catalog.json requires one control and investigational candidates.")
    return {role: tuple(values) for role, values in by_role.items()}


_REQUIRED_PUBLIC = (
    "study_brief.md",
    "clinical_narrative.json",
    "data_dictionary.json",
    "candidate_drug_catalog.json",
    "variable_catalog.json",
    "endpoint_catalog.json",
    "ae_taxonomy.json",
    "phase_module_catalog.json",
    "program_loop_manifest.json",
    "phase_action_policy.json",
    "objective_charter.json",
    "observational_method_catalog.json",
    "phase_design_policy.json",
    "phase_decision_evidence_policy.json",
    "phase_analysis_method_catalog.json",
    "phase_design_frontiers.json",
    "safety_decision_policy.json",
    "phase_decision_schema.json",
    "trial_request_schema.json",
    "trial_output_schema.json",
    "eval_contract.json",
    "observational_extract.parquet",
)

_PUBLIC_FORBIDDEN_JSON_KEYS = {
    "hidden_eval_count",
    "n_hidden_eval_items",
    "hidden_eval_item_ids",
    "recommended_max_sample_size",
    "recommended_max_analysis_covariates",
    "recommended_max_subgroup_splits",
    "recommended_bounds",
    "selected_winner_drug_id",
    "recommended_drug_id",
    "advance_to_next_phase",
    "primary_value",
    "best_candidate_drug_id",
    "recoverability_manifest",
    "acceptable_candidate",
    "acceptable_candidate_set",
    "policy_reference_regret",
    "released_data_reference_rank",
    "expected_action",
    "reference_action",
    "lane_activation",
    "lane_weight",
    "score_weight",
    "reference_solution",
    "shortcut_attack",
    "calibration_reference",
    "phase_coherence_reference",
    "confounding_regime",
    "supported_primary_result_kind",
    "point_analysis_status",
    "intended_outcome",
    "mechanism_environment_id",
    "debug",
    "audit_report",
}

_PUBLIC_FORBIDDEN_TEXT = (
    "hidden_eval",
    "released_data_reference",
    "reference=oracle",
    "reference=released_data_reference",
    "selected_winner_drug_id",
    "recommended_drug_id",
    "primary_value",
    "best_candidate_drug_id",
    "recoverability_manifest",
    "acceptable_candidate",
    "policy_reference_regret",
    "expected_action",
    "reference_action",
    "lane_activation",
    "lane_weight",
    "score_weight",
    "reference_solution",
    "shortcut_attack",
    "calibration_reference",
    "phase_coherence_reference",
    "qualified_nonidentification",
    "residual_unmeasured_confounding",
    "insufficient_recoverability",
)

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _load_manifest_checked(path: Path) -> ScenarioBundleManifestV1:
    payload = read_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("Manifest JSON must be an object.")
    expected_checksum = str(payload.get("checksum", ""))
    try:
        manifest = ScenarioBundleManifestV1.model_validate(payload)
    except ValidationError as exc:  # pragma: no cover
        raise ValueError(str(exc)) from exc
    if expected_checksum and expected_checksum != str(manifest.checksum):
        raise ValueError(
            "Manifest checksum mismatch. "
            f"path={str(path)!r} expected={expected_checksum!r} computed={manifest.checksum!r}."
        )
    return manifest


def _load_checksummed_model(path: Path, *, model: type[_ModelT]) -> _ModelT:
    payload = read_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("Checksummed model JSON must be an object.")
    expected_checksum = str(payload.get("checksum", ""))
    try:
        parsed = model.model_validate(payload)
    except ValidationError as exc:  # pragma: no cover
        raise ValueError(str(exc)) from exc
    computed = str(getattr(parsed, "checksum", "") or "")
    if expected_checksum and computed and expected_checksum != computed:
        raise ValueError(
            "Checksum mismatch for checksummed JSON model. "
            f"path={str(path)!r} expected={expected_checksum!r} computed={computed!r}."
        )
    return parsed


def _iter_json_keys(value: object) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_iter_json_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_iter_json_keys(child))
    return tuple(keys)


def _validate_public_surface_is_non_leading(*, root: Path) -> None:
    offenders: dict[str, tuple[str, ...]] = {}
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda item: item.as_posix()):
        rel_path = str(path.relative_to(root).as_posix())
        if "/hidden/" in f"/{rel_path}/" or "/grader/" in f"/{rel_path}/":
            raise ValueError(f"Public scenario contains evaluator-only path: {rel_path}")
        name_hits = [literal for literal in _PUBLIC_FORBIDDEN_TEXT if literal in rel_path]
        if name_hits:
            offenders[rel_path] = tuple(sorted(set(name_hits)))
            continue
        if path.suffix not in {".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8")
        hits = [literal for literal in _PUBLIC_FORBIDDEN_TEXT if literal in text]
        if path.suffix == ".json":
            payload = read_json(path)
            hits.extend(key for key in _iter_json_keys(payload) if key in _PUBLIC_FORBIDDEN_JSON_KEYS)
        if hits:
            offenders[rel_path] = tuple(sorted(set(hits)))
    if offenders:
        raise ValueError(f"Public scenario contains leading or evaluator-only metadata: {offenders!r}")


def _validate_safety_decision_policy(*, public_dir: Path, scenario_id: str) -> None:
    scenario_root = public_dir.parent
    payload = load_safety_policy_v1(scenario_root=scenario_root)
    if str(payload.get("scenario_id")) != str(scenario_id):
        raise ValueError("safety_decision_policy.scenario_id must match program_loop_manifest.scenario_id.")
    serious_event_definitions_v1(scenario_root=scenario_root)
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("safety_decision_policy.thresholds must be a non-empty array.")
    threshold_keys = {
        (str(item.get("phase_id")), str(item.get("component_id"))) for item in thresholds if isinstance(item, dict)
    }
    hard_gate_phases = set()
    for phase_id in ("phase1", "phase2", "phase3"):
        for component_id in ("serious_ae", "discontinuation"):
            if (phase_id, component_id) not in threshold_keys:
                raise ValueError(f"safety_decision_policy missing {component_id} threshold for {phase_id}.")
    for item in thresholds:
        if not isinstance(item, dict):
            raise ValueError("safety_decision_policy.thresholds entries must be objects.")
        role = str(item.get("role", ""))
        component_id = str(item.get("component_id"))
        phase_id = str(item.get("phase_id"))
        if role not in {"hard_gate", "diagnostic_only"}:
            raise ValueError("safety_decision_policy.thresholds entries require role.")
        if item.get("threshold_unit") != "absolute_probability":
            raise ValueError("safety thresholds must declare threshold_unit=absolute_probability.")
        if item.get("threshold_basis") != "scenario_declared_target_product_profile":
            raise ValueError("safety thresholds must declare their scenario target-product-profile basis.")
        if not str(item.get("rationale") or "").strip():
            raise ValueError("safety thresholds must declare a rationale.")
        if component_id in {"discontinuation", "ltfu"} and role == "hard_gate":
            raise ValueError("all-cause discontinuation and LTFU cannot be hard safety gates.")
        if role == "hard_gate":
            hard_gate_phases.add(phase_id)
    for phase_id in ("phase1", "phase2", "phase3"):
        if phase_id not in hard_gate_phases:
            raise ValueError(f"safety_decision_policy missing hard_gate threshold for {phase_id}.")


def _validate_phase_decision_evidence_policy(*, public_dir: Path, scenario_id: str) -> None:
    payload = read_json(public_dir / "phase_decision_evidence_policy.json")
    if not isinstance(payload, dict) or payload.get("schema_id") != "trialdev_phase_decision_evidence_policy_v1":
        raise ValueError("phase_decision_evidence_policy.json has an unsupported schema.")
    if str(payload.get("scenario_id")) != str(scenario_id):
        raise ValueError("phase_decision_evidence_policy.scenario_id does not match the scenario.")
    confidence_value = payload.get("confidence_level")
    if isinstance(confidence_value, bool) or not isinstance(confidence_value, int | float):
        raise ValueError("phase decision policy requires a numeric confidence_level.")
    confidence_level = float(confidence_value)
    if not MINIMUM_TWO_SIDED_CONFIDENCE_LEVEL_V1 < confidence_level < 1.0:
        raise ValueError("phase decision policy requires a valid two-sided confidence level.")
    rules = payload.get("phase_rules")
    if not isinstance(rules, list):
        raise ValueError("phase decision policy requires phase_rules.")
    by_phase = {
        str(rule.get("phase_id")): rule for rule in rules if isinstance(rule, dict) and str(rule.get("phase_id"))
    }
    if set(by_phase) != {"phase1", "phase2", "phase3"}:
        raise ValueError("phase decision policy must cover phase1, phase2, and phase3 exactly once.")
    for phase_id in ("phase2", "phase3"):
        rule = by_phase[phase_id]
        if rule.get("effect_scale_id") != "risk_difference_control_minus_treatment":
            raise ValueError("phase efficacy rules must use the policy-aligned risk-difference scale.")
        if rule.get("threshold_unit") != "absolute_probability":
            raise ValueError("phase efficacy rules must declare threshold_unit=absolute_probability.")
        if rule.get("threshold_basis") != "scenario_declared_target_product_profile":
            raise ValueError("phase efficacy rules must declare their target-product-profile basis.")
        minimum = float(rule.get("minimum_benefit", -1.0))
        sensitivity = rule.get("sensitivity_minimum_benefits")
        if not isinstance(sensitivity, list) or not any(
            abs(float(value) - minimum) <= FLOAT_EQUALITY_ABS_TOLERANCE_V1 for value in sensitivity
        ):
            raise ValueError("phase efficacy sensitivity values must include the primary minimum benefit.")
        if not str(rule.get("rationale") or "").strip():
            raise ValueError("phase efficacy rules must declare a rationale.")


def _validate_phase_design_policy(*, public_dir: Path, scenario_id: str) -> None:
    objective = TrialDevPublicObjectiveCharterV1.model_validate(read_json(public_dir / "objective_charter.json"))
    policy = TrialDevPhaseDesignPolicyV1.model_validate(read_json(public_dir / "phase_design_policy.json"))
    if objective.scenario_id != str(scenario_id) or policy.scenario_id != str(scenario_id):
        raise ValueError("phase_design_policy.scenario_id does not match the scenario.")
    if objective.decision_charter_checksum != policy.decision_charter_checksum:
        raise ValueError("objective and design contracts reference different decision charters.")
    if objective.confidence_level != policy.confidence_level:
        raise ValueError("objective and design contracts declare different confidence levels.")
    expected_source_checksums = {
        "public/observational_extract.parquet": sha256_file_hex(public_dir / "observational_extract.parquet"),
        "public/objective_charter.json": sha256_file_hex(public_dir / "objective_charter.json"),
        "public/candidate_drug_catalog.json": sha256_file_hex(public_dir / "candidate_drug_catalog.json"),
    }
    if policy.source_artifact_checksums != expected_source_checksums:
        raise ValueError("phase design policy must bind its public analysis specification and planning surface.")
    candidates_by_role = candidate_ids_by_role_v1(scenario_root=public_dir.parent)
    candidate_ids = set(candidates_by_role["control"]) | set(candidates_by_role["investigational"])
    for rule in policy.phase_rules:
        if set(rule.planning_information_fraction_by_drug_id) != candidate_ids:
            raise ValueError("phase design planning-information coverage differs from the candidate catalog.")
        if (
            min(
                rule.planning_safety_absolute_treatment_risk,
                rule.planning_safety_excess_risk,
            )
            <= 0.0
        ):
            raise ValueError("phase design safety planning alternatives must be positive.")
        for candidate_id in candidate_ids:
            fraction = rule.planning_information_fraction_by_drug_id[candidate_id]
            observed_count = rule.planning_information_support_count_by_drug_id[candidate_id]
            weighted_count = rule.planning_information_weighted_effective_sample_size_by_drug_id[candidate_id]
            if not 0.0 < fraction <= 1.0 or observed_count < 2 or not 0.0 < weighted_count <= observed_count + 1e-9:
                raise ValueError("phase design information diagnostics are outside their valid ranges.")
        if rule.phase_id == "phase1":
            continue
        if (
            rule.planning_control_risk is None
            or rule.planning_treatment_risk is None
            or rule.planning_alternative_benefit is None
        ):
            raise ValueError("phase2/phase3 design policy requires efficacy planning values.")
        control_risk = rule.planning_control_risk
        treatment_risk = rule.planning_treatment_risk
        alternative = rule.planning_alternative_benefit
        if not 0.0 < treatment_risk < control_risk < 1.0:
            raise ValueError("efficacy planning risks must be ordered within (0, 1).")
        if abs((control_risk - treatment_risk) - alternative) > FLOAT_EQUALITY_ABS_TOLERANCE_V1:
            raise ValueError("efficacy planning alternative must equal the declared risk difference.")


def _validate_decision_charter_links(*, public_dir: Path, scenario_id: str) -> None:
    charter = read_json(public_dir / "decision_charter.json")
    if not isinstance(charter, dict) or charter.get("schema_id") != "trialdev_decision_charter_v1":
        raise ValueError("decision_charter.json has an unsupported schema.")
    if str(charter.get("scenario_id")) != str(scenario_id):
        raise ValueError("decision_charter.scenario_id does not match the scenario.")
    expected_checksum = str(charter.get("checksum") or "")
    checksum_payload = dict(charter)
    checksum_payload.pop("checksum", None)
    if expected_checksum != compute_sha256_hex(checksum_payload):
        raise ValueError("decision_charter.json checksum mismatch.")
    objective_rules = charter.get("objective_rules")
    efficacy_rules = charter.get("efficacy_rules")
    safety_rules = charter.get("safety_rules")
    design_rules = charter.get("design_rules")
    if not isinstance(objective_rules, list) or {str(rule.get("objective_id")) for rule in objective_rules} != {
        "benefit_risk",
        "pure_efficacy",
        "cost_effective_best",
        "net_clinical_value_under_budget",
    }:
        raise ValueError("decision_charter must declare all four asset-selection objectives.")
    if not isinstance(efficacy_rules, list) or {str(rule.get("phase_id")) for rule in efficacy_rules} != {
        "observational_review",
        "phase2",
        "phase3",
    }:
        raise ValueError("decision_charter efficacy rules must cover observational review, phase2, and phase3.")
    if not isinstance(safety_rules, list) or {str(rule.get("phase_id")) for rule in safety_rules} != {
        "phase1",
        "phase2",
        "phase3",
    }:
        raise ValueError("decision_charter safety rules must cover every materialized phase.")
    if any(
        not isinstance(rule.get("evaluation_horizon_days"), int)
        or isinstance(rule.get("evaluation_horizon_days"), bool)
        or int(rule["evaluation_horizon_days"]) < 1
        for rule in safety_rules
    ):
        raise ValueError("decision_charter safety rules require positive integer evaluation horizons.")
    efficacy_horizons = {
        str(rule["phase_id"]): int(rule["evaluation_horizon_days"])
        for rule in efficacy_rules
        if str(rule.get("phase_id")) in {"phase2", "phase3"}
    }
    safety_horizons = {str(rule["phase_id"]): int(rule["evaluation_horizon_days"]) for rule in safety_rules}
    if any(efficacy_horizons[phase_id] != safety_horizons[phase_id] for phase_id in ("phase2", "phase3")):
        raise ValueError("randomized efficacy and safety rules must share phase-specific evaluation horizons.")
    if not isinstance(design_rules, list) or {str(rule.get("phase_id")) for rule in design_rules} != {
        "phase1",
        "phase2",
        "phase3",
    }:
        raise ValueError("decision_charter design rules must cover all randomized phases exactly once.")
    for filename in (
        "objective_charter.json",
        "observational_method_catalog.json",
        "program_loop_manifest.json",
        "phase_action_policy.json",
        "phase_design_policy.json",
        "phase_decision_evidence_policy.json",
        "safety_decision_policy.json",
    ):
        projection = read_json(public_dir / filename)
        if not isinstance(projection, dict) or projection.get("decision_charter_checksum") != expected_checksum:
            raise ValueError(f"{filename} is not linked to the active decision charter.")


def _validate_phase_analysis_method_catalog(*, public_dir: Path, scenario_id: str) -> None:
    catalog = TrialDevPhaseAnalysisMethodCatalogV1.model_validate(
        read_json(public_dir / "phase_analysis_method_catalog.json")
    )
    if catalog.scenario_id != str(scenario_id):
        raise ValueError("phase_analysis_method_catalog.scenario_id does not match the scenario.")
    objective = TrialDevPublicObjectiveCharterV1.model_validate(read_json(public_dir / "objective_charter.json"))
    observational = TrialDevPublicObservationalMethodCatalogV1.model_validate(
        read_json(public_dir / "observational_method_catalog.json")
    )
    if observational.scenario_id != str(scenario_id):
        raise ValueError("observational_method_catalog.scenario_id does not match the scenario.")
    if observational.decision_charter_checksum != objective.decision_charter_checksum:
        raise ValueError("observational method and objective contracts reference different decision charters.")
    charter = read_json(public_dir / "decision_charter.json")
    if not isinstance(charter, dict):
        raise ValueError("decision_charter.json must contain an object.")
    confidence_levels = (
        objective.confidence_level,
        observational.confidence_level,
        float(charter.get("confidence_level", 0.0)),
    )
    if any(
        not math.isclose(
            catalog.confidence_level,
            confidence,
            rel_tol=0.0,
            abs_tol=FLOAT_EQUALITY_ABS_TOLERANCE_V1,
        )
        for confidence in confidence_levels
    ):
        raise ValueError(
            "phase_analysis_method_catalog.confidence_level must match objective_charter, "
            "observational_method_catalog, and decision_charter."
        )


def _validate_data_dictionary_roles(*, public_dir: Path) -> None:
    payload = read_json(Path(public_dir) / "data_dictionary.json")
    if not isinstance(payload, dict):
        raise ValueError("data_dictionary.json must be a JSON object.")
    raw_roles = payload.get("semantic_roles")
    if not isinstance(raw_roles, list) or not raw_roles:
        raise ValueError("data_dictionary.json must contain a non-empty semantic_roles list.")
    required_roles = {"subject_id", "observed_treatment", "event_indicator", "event_time", "arm_id"}
    seen: dict[str, str] = {}
    for raw_role in raw_roles:
        if not isinstance(raw_role, dict):
            raise ValueError("data_dictionary.json semantic_roles entries must be objects.")
        role_id = str(raw_role.get("role_id", ""))
        table = str(raw_role.get("table", ""))
        column = str(raw_role.get("column", ""))
        if not role_id or not table or not column:
            raise ValueError("semantic role entries require role_id, table, and column.")
        if role_id in seen:
            raise ValueError(f"Duplicate semantic role in data_dictionary.json: {role_id!r}.")
        seen[role_id] = column
    missing = sorted(required_roles - set(seen))
    if missing:
        raise ValueError(f"data_dictionary.json missing required semantic roles: {missing!r}.")


def validate_scenario_bundle_v1(*, scenario_root: Path) -> None:
    """
    Validate a scenario bundle directory using its emitted checksummed manifest.

    Parameters
    ----------
    scenario_root
        Scenario bundle root directory, for example ``./scenario_s01``.
    """
    root = Path(scenario_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Scenario root not found: {root}")

    public_dir = root / "public"
    hidden_dir = root / "hidden"
    grader_dir = root / "grader"
    manifests_dir = root / "manifests"
    release_dir = root / "release"
    for d in (public_dir, hidden_dir, grader_dir, manifests_dir, release_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Missing required bundle surface directory: {d}")

    required_public = _REQUIRED_PUBLIC
    required_hidden = (
        "world_spec.yaml",
        "world_manifest.json",
        "frozen_superpopulation_manifest.json",
        "superpopulation_qualification_summary.json",
        "hidden_phase_targets.json",
        "evaluation_reference_manifest.json",
        "released_data_reference_manifest.json",
        "candidate_drug_reference_summary.json",
        "safety_reference_summary.json",
        "hidden_reference_key_index.json",
        "counterfactual_pool.parquet",
    )
    required_grader = (
        "grading_procedure.json",
        "submission_schema.json",
        "drug_ranking_reference_manifest.json",
        "public_recoverability_report.json",
        "evaluation_target_register.jsonl",
        "evaluation_target_register_manifest.json",
        "evaluation_target_register_gate_report.json",
    )
    required_manifests = (
        "public_manifest.json",
        "hidden_manifest.json",
        "grader_manifest.json",
        "scenario_manifest.json",
    )
    required_release = (
        "external_agent_audit_report.json",
        "release_validation_report.json",
        "release_summary.json",
    )
    for rel in required_public:
        if not (public_dir / rel).is_file():
            raise FileNotFoundError(f"Missing required public artifact: {public_dir / rel}")
    for rel in required_hidden:
        if not (hidden_dir / rel).is_file():
            raise FileNotFoundError(f"Missing required hidden artifact: {hidden_dir / rel}")
    for rel in required_grader:
        if not (grader_dir / rel).is_file():
            raise FileNotFoundError(f"Missing required grader artifact: {grader_dir / rel}")
    for rel in required_manifests:
        if not (manifests_dir / rel).is_file():
            raise FileNotFoundError(f"Missing required manifest artifact: {manifests_dir / rel}")
    for rel in required_release:
        if not (release_dir / rel).is_file():
            raise FileNotFoundError(f"Missing required release artifact: {release_dir / rel}")

    manifest_path = root / "manifests" / "scenario_manifest.json"
    manifest = _load_manifest_checked(manifest_path)

    # Verify each artifact exists and matches its recorded checksum and size.
    for artifact in manifest.artifacts:
        path = root / str(artifact.rel_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing artifact: {path}")
        size = int(path.stat().st_size)
        if int(size) != int(artifact.size_bytes):
            raise ValueError(
                "Artifact size mismatch. "
                f"rel_path={artifact.rel_path!r} expected={artifact.size_bytes!r} observed={size!r}."
            )
        digest = sha256_file_hex(path)
        if str(digest) != str(artifact.sha256):
            raise ValueError(
                "Artifact checksum mismatch. "
                f"rel_path={artifact.rel_path!r} expected={artifact.sha256!r} observed={digest!r}."
            )

        # Basic separation check: surface label must match rel_path prefix.
        parts = str(artifact.rel_path).split("/", 1)
        prefix = parts[0] if parts else ""
        if prefix and str(prefix) != str(artifact.surface):
            raise ValueError(
                "Artifact surface label does not match its rel_path prefix. "
                f"rel_path={artifact.rel_path!r} surface={artifact.surface!r}."
            )

    # Optional: ensure no hidden surface is duplicated under public.
    for hidden_path in (root / "hidden").rglob("*"):
        if hidden_path.is_file() and (root / "public" / hidden_path.relative_to(root / "hidden")).exists():
            raise ValueError("Hidden artifact leaked into public surface.")

    # Validate checksummed JSON contracts that downstream evaluation depends on.
    eval_contract = _load_checksummed_model(public_dir / "eval_contract.json", model=TrialDevelopmentEvalContractV1)
    loop_manifest = _load_checksummed_model(
        public_dir / "program_loop_manifest.json", model=TrialDevelopmentProgramLoopManifestV1
    )
    action_policy = _load_checksummed_model(
        public_dir / "phase_action_policy.json", model=TrialDevelopmentPhaseActionPolicyV1
    )
    if str(action_policy.scenario_id) != str(loop_manifest.scenario_id):
        raise ValueError("phase_action_policy.scenario_id must match program_loop_manifest.scenario_id.")
    if str(action_policy.phase_policy_checksum) != str(loop_manifest.phase_policy_checksum):
        raise ValueError("phase_action_policy.phase_policy_checksum must match program_loop_manifest.")
    _validate_decision_charter_links(
        public_dir=public_dir,
        scenario_id=str(loop_manifest.scenario_id),
    )
    _validate_safety_decision_policy(
        public_dir=public_dir,
        scenario_id=str(loop_manifest.scenario_id),
    )
    _validate_phase_decision_evidence_policy(
        public_dir=public_dir,
        scenario_id=str(loop_manifest.scenario_id),
    )
    _validate_phase_design_policy(public_dir=public_dir, scenario_id=str(loop_manifest.scenario_id))
    _validate_phase_analysis_method_catalog(
        public_dir=public_dir,
        scenario_id=str(loop_manifest.scenario_id),
    )
    _ = _load_checksummed_model(
        hidden_dir / "frozen_superpopulation_manifest.json", model=FrozenSuperpopulationManifestV1
    )
    qualification = _load_checksummed_model(
        hidden_dir / "superpopulation_qualification_summary.json",
        model=SuperpopulationQualificationSummaryV1,
    )
    if not bool(getattr(qualification, "sufficient_for_release", False)):
        raise ValueError("Scenario superpopulation qualification indicates insufficient_for_release.")
    _ = _load_checksummed_model(hidden_dir / "hidden_phase_targets.json", model=TrialDevelopmentPhaseTargetsManifestV1)
    _ = _load_checksummed_model(
        hidden_dir / "evaluation_reference_manifest.json", model=TrialDevelopmentEvaluationReferenceManifestV1
    )
    _ = _load_checksummed_model(
        hidden_dir / "released_data_reference_manifest.json",
        model=TrialDevelopmentEvaluationReferenceManifestV1,
    )
    _ = _load_checksummed_model(
        hidden_dir / "safety_reference_summary.json", model=TrialDevelopmentSafetyReferenceManifestV1
    )
    _ = _load_checksummed_model(grader_dir / "grading_procedure.json", model=TrialDevelopmentGradingProcedureV1)
    _ = _load_checksummed_model(grader_dir / "submission_schema.json", model=TrialDevelopmentSubmissionSchemaV1)
    for rel in (
        "drug_ranking_reference_manifest.json",
        "public_recoverability_report.json",
        "evaluation_target_register_manifest.json",
    ):
        payload = read_json(grader_dir / rel)
        if not isinstance(payload, dict):
            raise ValueError(f"{rel} must contain a JSON object.")
        expected = payload.get("checksum")
        normalized = dict(payload)
        normalized.pop("checksum", None)
        if not isinstance(expected, str) or compute_sha256_hex(normalized) != expected:
            raise ValueError(f"{rel} checksum mismatch.")
    gate_payload = read_json(grader_dir / "evaluation_target_register_gate_report.json")
    if (
        not isinstance(gate_payload, dict)
        or gate_payload.get("schema_id") != "trialdev_evaluation_target_register_gate_report_v1"
        or gate_payload.get("passed") is not True
        or gate_payload.get("issue_count") != 0
    ):
        raise ValueError("evaluation_target_register_gate_report.json must record a passing zero-issue validation.")
    if not (grader_dir / "evaluation_target_register.jsonl").read_text(encoding="utf-8").strip():
        raise ValueError("evaluation_target_register.jsonl must contain at least one record.")

    # Validate phase module catalog shape (non-checksummed).
    phase_payload = read_json(public_dir / "phase_module_catalog.json")
    if not isinstance(phase_payload, dict):
        raise ValueError("phase_module_catalog.json must be a JSON object.")
    modules = tuple(PhaseModuleSpecV1.model_validate(m) for m in (phase_payload.get("phase_modules", []) or []))
    phase_ids = {str(m.phase_id) for m in modules}
    required_phases = {"observational_review", "phase1", "phase2", "phase3"}
    if phase_ids != required_phases:
        raise ValueError(
            f"Phase module catalog must contain exactly {sorted(required_phases)!r}; observed={sorted(phase_ids)!r}."
        )
    # Parity check: eval contract must publish the same phase modules.
    contract_phase_ids = {str(m.phase_id) for m in getattr(eval_contract, "phase_modules", ())}
    if contract_phase_ids != required_phases:
        raise ValueError("Eval contract phase_modules missing required phase ids.")
    not_available = {
        str(phase_id)
        for phase_id, mode in getattr(loop_manifest, "phase_policy_modes", {}).items()
        if str(mode) == "not_available"
    }
    if not_available & set(str(value) for value in loop_manifest.conditionally_materializable_phase_ids):
        raise ValueError("Program loop manifest marks a not_available phase as conditionally materializable.")
    _validate_asset_development_modules(loop_manifest=loop_manifest, modules=modules)
    _validate_data_dictionary_roles(public_dir=public_dir)


def validate_public_scenario_bundle_v1(*, scenario_root: Path) -> None:
    """
    Validate the participant-visible scenario files.

    Parameters
    ----------
    scenario_root
        Scenario directory from a participant bundle, for example
        ``./scenario_s01``.
    """
    root = Path(scenario_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Scenario root not found: {root}")
    public_dir = root / "public"
    if not public_dir.is_dir():
        raise FileNotFoundError(f"Missing required public directory: {public_dir}")
    for rel in _REQUIRED_PUBLIC:
        if not (public_dir / rel).is_file():
            raise FileNotFoundError(f"Missing required public artifact: {public_dir / rel}")
    _validate_public_surface_is_non_leading(root=public_dir)

    eval_contract = _load_checksummed_model(public_dir / "eval_contract.json", model=TrialDevelopmentEvalContractV1)
    loop_manifest = _load_checksummed_model(
        public_dir / "program_loop_manifest.json", model=TrialDevelopmentProgramLoopManifestV1
    )
    action_policy = _load_checksummed_model(
        public_dir / "phase_action_policy.json", model=TrialDevelopmentPhaseActionPolicyV1
    )
    if str(action_policy.scenario_id) != str(loop_manifest.scenario_id):
        raise ValueError("phase_action_policy.scenario_id must match program_loop_manifest.scenario_id.")
    if str(action_policy.phase_policy_checksum) != str(loop_manifest.phase_policy_checksum):
        raise ValueError("phase_action_policy.phase_policy_checksum must match program_loop_manifest.")
    _validate_decision_charter_links(
        public_dir=public_dir,
        scenario_id=str(loop_manifest.scenario_id),
    )
    _validate_safety_decision_policy(
        public_dir=public_dir,
        scenario_id=str(loop_manifest.scenario_id),
    )
    _validate_phase_decision_evidence_policy(
        public_dir=public_dir,
        scenario_id=str(loop_manifest.scenario_id),
    )
    _validate_phase_design_policy(public_dir=public_dir, scenario_id=str(loop_manifest.scenario_id))
    _validate_phase_analysis_method_catalog(
        public_dir=public_dir,
        scenario_id=str(loop_manifest.scenario_id),
    )
    phase_payload = read_json(public_dir / "phase_module_catalog.json")
    if not isinstance(phase_payload, dict):
        raise ValueError("phase_module_catalog.json must be a JSON object.")
    modules = tuple(PhaseModuleSpecV1.model_validate(m) for m in (phase_payload.get("phase_modules", []) or []))
    phase_ids = {str(m.phase_id) for m in modules}
    required_phases = {"observational_review", "phase1", "phase2", "phase3"}
    if phase_ids != required_phases:
        raise ValueError(
            f"Phase module catalog must contain exactly {sorted(required_phases)!r}; observed={sorted(phase_ids)!r}."
        )
    contract_phase_ids = {str(m.phase_id) for m in getattr(eval_contract, "phase_modules", ())}
    if contract_phase_ids != required_phases:
        raise ValueError("Eval contract phase_modules missing required phase ids.")
    not_available = {
        str(phase_id)
        for phase_id, mode in getattr(loop_manifest, "phase_policy_modes", {}).items()
        if str(mode) == "not_available"
    }
    if not_available & set(str(value) for value in loop_manifest.conditionally_materializable_phase_ids):
        raise ValueError("Program loop manifest marks a not_available phase as conditionally materializable.")
    _validate_asset_development_modules(loop_manifest=loop_manifest, modules=modules)
    _validate_data_dictionary_roles(public_dir=public_dir)


def validate_request_file_v1(*, request_path: Path) -> None:
    """Validate one structured trial request JSON file."""
    payload = read_json(Path(request_path))
    if not isinstance(payload, dict):
        raise ValueError("Request JSON must be an object.")
    TrialDevelopmentRequestV1.model_validate(payload)


def _module_by_phase(*, scenario_root: Path) -> dict[str, PhaseModuleSpecV1]:
    payload = read_json(Path(scenario_root) / "public" / "phase_module_catalog.json")
    if not isinstance(payload, dict):
        raise ValueError("phase_module_catalog.json must be a JSON object.")
    modules = tuple(PhaseModuleSpecV1.model_validate(m) for m in (payload.get("phase_modules", []) or []))
    return {str(module.phase_id): module for module in modules}


def _validate_asset_development_modules(
    *, loop_manifest: TrialDevelopmentProgramLoopManifestV1, modules: tuple[PhaseModuleSpecV1, ...]
) -> None:
    if str(loop_manifest.program_archetype) != "asset_development":
        return
    by_phase = {str(module.phase_id): module for module in modules}
    for phase_id in ("phase1", "phase2", "phase3"):
        module = by_phase.get(phase_id)
        if module is None:
            raise ValueError(f"Asset-development bundle missing {phase_id} phase module.")
        if not module.includes_control_arm:
            raise ValueError(f"Asset-development {phase_id} must include the reference control.")


def validate_request_against_scenario_v1(
    *, scenario_root: Path, request: TrialDevelopmentRequestV1
) -> TrialDevelopmentRequestV1:
    """Validate one request against a scenario's public phase menus."""

    reject = TrialDevelopmentRequestRejectedError
    public_contract = _load_checksummed_model(
        Path(scenario_root) / "public" / "eval_contract.json", model=TrialDevelopmentEvalContractV1
    )
    if str(request.scenario_id) != str(public_contract.scenario_id):
        raise reject("Request scenario_id does not match the scenario public contract.")
    module = _module_by_phase(scenario_root=Path(scenario_root)).get(str(request.phase_id))
    if module is None:
        raise reject(f"Request phase_id is not in the public phase module catalog: {request.phase_id!r}.")
    loop_manifest = _load_checksummed_model(
        Path(scenario_root) / "public" / "program_loop_manifest.json",
        model=TrialDevelopmentProgramLoopManifestV1,
    )
    mode = str(getattr(loop_manifest, "phase_policy_modes", {}).get(str(request.phase_id), "optional"))
    if mode == "not_available":
        raise reject(f"Request phase_id is not available for this scenario: {request.phase_id!r}.")
    candidate_count = int(len(request.candidate_drug_ids))
    declared_candidates = set(candidate_ids_by_role_v1(scenario_root=Path(scenario_root))["investigational"])
    unknown_candidates = sorted(set(request.candidate_drug_ids) - declared_candidates)
    if unknown_candidates:
        raise reject(f"Request candidate_drug_ids are absent from the public candidate menu: {unknown_candidates!r}.")
    if str(request.phase_id) != "observational_review" and candidate_count != 1:
        raise reject("Randomized TrialDev requests require exactly one investigational regimen plus control.")
    if str(request.phase_id) == "phase1" and request.endpoint_id is not None:
        raise reject("phase1 requests must not set endpoint_id.")
    if str(request.phase_id) in {"phase2", "phase3"}:
        if request.endpoint_id is None:
            raise reject(f"{request.phase_id} requests must set endpoint_id.")
        if str(request.endpoint_id) not in set(str(value) for value in module.allowed_endpoint_ids):
            raise reject("Request endpoint_id is not in the phase menu.")
    if request.target_sample_size is None:
        raise reject("Trial phase requests must set target_sample_size.")
    if module.max_sample_size is not None and int(request.target_sample_size) > int(module.max_sample_size):
        raise reject("Request target_sample_size exceeds the phase limit.")
    if request.follow_up_days is None or int(request.follow_up_days) not in set(
        int(value) for value in module.allowed_follow_up_days
    ):
        raise reject("Request follow_up_days is not in the phase menu.")
    if request.allocation_ratio is not None and str(request.allocation_ratio) not in set(
        str(value) for value in module.allowed_allocation_ratios
    ):
        raise reject("Request allocation_ratio is not in the phase menu.")
    if request.allocation_weights:
        raise reject("Request allocation_weights are not supported by the finite public design menu.")
    if request.allocation_ratio is None:
        raise reject("Request allocation_ratio is required by the finite public design menu.")
    if str(request.phase_id) == "phase1":
        if request.treatment_discontinuation_strategy is not None:
            raise reject("Phase-1 requests must not bind a treatment-discontinuation strategy.")
    elif request.treatment_discontinuation_strategy not in set(module.allowed_treatment_discontinuation_strategies):
        raise reject("Request treatment_discontinuation_strategy is not in the phase menu.")
    if request.interim_policy not in set(module.allowed_interim_policies):
        raise reject("Request interim_policy is not in the phase menu.")
    if request.site_strategy not in set(module.allowed_site_strategies):
        raise reject("Request site_strategy is not in the phase menu.")
    if request.selection_objective not in set(module.allowed_selection_objectives):
        raise reject("Request selection_objective is not in the phase menu.")
    if request.enrollment_window_days is not None and int(request.enrollment_window_days) not in set(
        int(value) for value in module.allowed_enrollment_window_days
    ):
        raise reject("Request enrollment_window_days is not in the phase menu.")
    if request.site_count_budget is not None and int(request.site_count_budget) not in set(
        int(value) for value in module.allowed_site_count_budgets
    ):
        raise reject("Request site_count_budget is not in the phase menu.")
    allowed_vars = set(str(value) for value in module.allowed_variable_ids)
    for label, values in (
        ("stratification_variables", request.stratification_variables),
        ("analysis_covariates", request.analysis_covariates),
        ("subgroup_variables", request.subgroup_variables),
    ):
        unknown = sorted(set(str(value) for value in values) - allowed_vars)
        if unknown:
            raise reject(f"Request {label} contains variables absent from the phase menu: {unknown!r}.")
    if module.max_analysis_covariates is not None and len(request.analysis_covariates) > int(
        module.max_analysis_covariates
    ):
        raise reject("Request analysis_covariates exceeds the phase limit.")
    if module.max_subgroup_splits is not None and len(request.subgroup_variables) > int(module.max_subgroup_splits):
        raise reject("Request subgroup_variables exceeds the phase limit.")
    return request


def validate_request_against_scenario_file_v1(*, scenario_root: Path, request_path: Path) -> TrialDevelopmentRequestV1:
    """Read and validate one request against a scenario's public phase menus."""

    payload = read_json(Path(request_path))
    if not isinstance(payload, dict):
        raise ValueError("Request JSON must be an object.")
    request = TrialDevelopmentRequestV1.model_validate(payload)
    return validate_request_against_scenario_v1(scenario_root=Path(scenario_root), request=request)


def validate_request_shape_file_v1(*, request_path: Path) -> None:
    """Validate one request-shape JSON file without scenario-specific feasibility."""
    payload = read_json(Path(request_path))
    if not isinstance(payload, dict):
        raise ValueError("Request-shape JSON must be an object.")
    TrialDevelopmentRequestV1.model_validate(payload)


def validate_submission_shape_file_v1(*, submission_path: Path) -> None:
    """Validate participant submission shape without scoring."""
    payload = read_json(Path(submission_path))
    if not isinstance(payload, dict):
        raise ValueError("Submission JSON must be an object.")
    required = {"version", "scenario_id", "request", "analysis_report", "program_decision"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Submission JSON missing required sections: {missing!r}.")
    if str(payload.get("version")) != "v1":
        raise ValueError("Submission version must be 'v1'.")
    request = payload.get("request")
    if not isinstance(request, dict):
        raise ValueError("Submission request must be a JSON object.")
    TrialDevelopmentRequestV1.model_validate(request)
    for section in ("analysis_report", "program_decision"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"Submission {section} must be a JSON object.")
