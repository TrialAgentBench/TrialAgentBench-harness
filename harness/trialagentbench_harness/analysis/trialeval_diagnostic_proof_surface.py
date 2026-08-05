"""Build public diagnostic proof surfaces for TrialEvalBench items."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from zipfile import ZipFile

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from lifelines.statistics import proportional_hazard_test

from trialagentbench_harness.analysis.participant_assumption_diagnostics import (
    compute_participant_assumption_diagnostics_v1,
    derive_participant_covariate_basis_v1,
)
from trialagentbench_harness.analysis.trialeval_release import read_json_object_member
from trialagentbench_harness.contracts.analysis.diagnostic_proof import (
    AssumptionDiagnosticReplayV1,
    AssumptionSeverityBandV1,
    DiagnosticEvidenceClassV1,
    DiagnosticProofDispositionV1,
    DiagnosticProofStatusV1,
    ParticipantAssumptionDiagnosticV1,
    ParticipantDiagnosticEvidenceV1,
    ReferenceAnalysisDecisionV1,
    TrialEvalDiagnosticProofReportV1,
    TrialEvalDiagnosticProofRowV1,
    TrialEvalDiagnosticProofSummaryRowV1,
)
from trialagentbench_harness.contracts.core.trialeval_factors import TrialEvalEvidenceFactorsV1
from trialagentbench_harness.contracts.release.trialeval_sap import TrialEvalPublicSAPV1
from trialagentbench_harness.contracts.scoring.assumption_evidence import AssumptionEvidenceManifestV1
from trialagentbench_harness.contracts.scoring.diagnostic_registry import (
    diagnostic_key_by_assumption_id_v1,
    diagnostic_normal_critical_value_v1,
    load_diagnostic_registry_v1,
)
from trialagentbench_harness.execution_policy import (
    TRIALEVAL_DIAGNOSTIC_PROOF_DEFAULT_WORKERS,
    TRIALEVAL_DIAGNOSTIC_WORKER_INVALID_INPUT_EXIT_CODE,
)
from trialagentbench_harness.grading.key_store import ScoringKeyStoreV1
from trialagentbench_harness.grading.models import ValidatedScoringKeyV1
from trialagentbench_harness.trialeval.effect_scales import route_family_for_effect_scale_v1
from trialagentbench_harness.verification.trialeval.public_analysis_reconstruction import (
    load_public_analysis_tables_v1,
    load_public_item_table_v1,
)

DESIGN_FAMILIES_REQUIRING_ADJUSTMENT = frozenset(
    {
        "cluster_parallel_randomized",
        "stepped_wedge_cluster_rollout",
        "group_sequential_monitoring",
    }
)
_PH_POLICY = load_diagnostic_registry_v1().diagnostic_keys["proportional_hazards"]
if _PH_POLICY.primary_method_change_threshold is None:  # pragma: no cover - release contract
    raise ValueError("The PH diagnostic policy requires a primary-method change threshold.")
PH_METHOD_CHANGE_TIME_VARIATION_THRESHOLD = float(_PH_POLICY.primary_method_change_threshold)
PH_METHOD_CHANGE_TIME_VARIATION_HR_RATIO = math.exp(PH_METHOD_CHANGE_TIME_VARIATION_THRESHOLD)

_DIAGNOSTIC_GROUP_TASK_FIELDS = (
    "design_subtype",
    "primary_control_arm_id",
    "primary_effect_scale",
    "primary_effect_scale_options",
    "primary_endpoint_id",
    "primary_endpoint_term",
    "primary_estimand_id",
    "primary_intercurrent_event_strategy_ids",
    "primary_paramcd",
    "primary_population_id",
    "primary_result_unit",
    "primary_tau_dy",
    "primary_treated_arm_id",
)
_NUMERICAL_WORKER_THREAD_ENVIRONMENT = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class _DiagnosticComputationV1:
    assumption_diagnostics: tuple[ParticipantAssumptionDiagnosticV1, ...]
    schoenfeld_p_value: float | None
    scaled_schoenfeld_rank_slope: float | None
    scaled_schoenfeld_rank_slope_standard_error: float | None
    simultaneous_lower_abs_time_varying_log_hazard_range: float | None
    ph_method_change_threshold_crossed: bool | None
    findings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _DiagnosticWorkerProcessV1:
    process: subprocess.Popen[bytes]
    output_path: Path
    stderr_path: Path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json_member(zf: ZipFile, member: str) -> dict[str, object]:
    return cast(dict[str, object], read_json_object_member(zf, member))


def _read_jsonl_member(zf: ZipFile, member: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(zf.read(member).decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {member}:{line_number}")
        rows.append(payload)
    return rows


def _load_item_index(evaluator_zip: Path) -> list[dict[str, object]]:
    with ZipFile(evaluator_zip) as zf:
        payload = _read_json_member(zf, "grader/item_index.json")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("grader/item_index.json must contain an entries list")
    return [dict(entry) for entry in entries]


def _load_assumption_evidence(evaluator_zip: Path) -> dict[str, AssumptionEvidenceManifestV1]:
    manifests: dict[str, AssumptionEvidenceManifestV1] = {}
    with ZipFile(evaluator_zip) as zf:
        for row in _read_jsonl_member(zf, "grader/domains/assumption_evidence.jsonl"):
            task_id = str(row.get("task_id") or "")
            payload = row.get("payload")
            if not task_id or not isinstance(payload, dict):
                raise ValueError("assumption_evidence row lacks task_id or payload")
            manifest = payload.get("manifest")
            if not isinstance(manifest, dict):
                raise ValueError("assumption_evidence payload lacks manifest")
            manifests[task_id] = AssumptionEvidenceManifestV1.model_validate(manifest)
    return manifests


def _treatment_indicator(
    adsl: pd.DataFrame,
    *,
    control_arm_id: str,
    treated_arm_id: str,
) -> pd.Series:
    """Bind the primary contrast to exact participant-visible arm identifiers."""

    if not control_arm_id or not treated_arm_id or control_arm_id == treated_arm_id:
        raise ValueError("Primary control and treated arm identifiers must be distinct and non-empty")
    indicators: list[pd.Series] = []
    for column in ("ARMCD", "ARM", "TRTA", "TRT01A", "TRTAN", "TRT01AN"):
        if column not in adsl.columns:
            continue
        values = adsl[column].astype("string")
        observed = set(values.dropna().astype(str))
        if not {control_arm_id, treated_arm_id} <= observed:
            continue
        indicator = pd.Series(np.nan, index=adsl.index, dtype=float)
        indicator.loc[values == control_arm_id] = 0.0
        indicator.loc[values == treated_arm_id] = 1.0
        indicators.append(indicator)
    if not indicators:
        raise ValueError(
            "ADSL contains no arm column with both declared primary contrast identifiers: "
            f"control={control_arm_id!r}, treated={treated_arm_id!r}"
        )
    reference = indicators[0]
    for indicator in indicators[1:]:
        jointly_observed = reference.notna() & indicator.notna()
        if not reference.loc[jointly_observed].equals(indicator.loc[jointly_observed]):
            raise ValueError("ADSL arm columns disagree on the declared primary contrast")
    return reference


def _survival_frame(
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    *,
    primary_paramcd: str | None,
    control_arm_id: str,
    treated_arm_id: str,
    static_binary_treatment: bool = True,
) -> pd.DataFrame:
    if "USUBJID" not in adsl.columns or "USUBJID" not in adtte.columns:
        raise ValueError("ADSL and ADTTE must contain USUBJID")
    if "AVAL" not in adtte.columns:
        raise ValueError("ADTTE must contain AVAL")
    if "PARAMCD" in adtte.columns:
        observed_paramcd = tuple(sorted(adtte["PARAMCD"].astype("string").dropna().unique()))
        if primary_paramcd is None:
            if len(observed_paramcd) != 1:
                raise ValueError("Multi-endpoint ADTTE requires task.primary_paramcd")
            primary_paramcd = str(observed_paramcd[0])
        if len(observed_paramcd) == 1 and str(observed_paramcd[0]) == "primary":
            adtte = adtte.copy()
        else:
            adtte = adtte.loc[adtte["PARAMCD"].astype("string") == str(primary_paramcd)].copy()
            if adtte.empty:
                raise ValueError(f"ADTTE contains no rows for task.primary_paramcd={primary_paramcd!r}")
    if "CNSR" in adtte.columns:
        event = 1 - pd.to_numeric(adtte["CNSR"], errors="coerce")
    elif "EVENT" in adtte.columns:
        event = pd.to_numeric(adtte["EVENT"], errors="coerce")
    else:
        raise ValueError("ADTTE must contain CNSR or EVENT")
    merged = adtte[["USUBJID", "AVAL"]].copy()
    merged["event"] = event
    if static_binary_treatment:
        arm_frame = adsl[["USUBJID"]].copy()
        arm_frame["treatment"] = _treatment_indicator(
            adsl,
            control_arm_id=control_arm_id,
            treated_arm_id=treated_arm_id,
        )
        merged = merged.merge(arm_frame, on="USUBJID", how="inner")
    merged["duration"] = pd.to_numeric(merged["AVAL"], errors="coerce")
    columns = ["USUBJID", "duration", "event"]
    if static_binary_treatment:
        columns.append("treatment")
    merged = merged[columns].replace([np.inf, -np.inf], np.nan).dropna()
    merged = merged[merged["duration"] > 0]
    merged["event"] = (merged["event"] > 0).astype(int)
    if static_binary_treatment:
        merged["treatment"] = (merged["treatment"] > 0).astype(int)
        if merged["treatment"].nunique() < 2:
            raise ValueError("Treatment indicator lacks two groups")
    if int(merged["event"].sum()) < 5:
        raise ValueError("Too few events for PH diagnostic")
    return merged


def _optional_participant_table(
    zf: ZipFile,
    *,
    task_id: str,
    names: tuple[str, ...],
) -> pd.DataFrame | None:
    members = set(zf.namelist())
    for name in names:
        member = f"items/{task_id}/{name}"
        if member in members:
            return cast(
                pd.DataFrame,
                load_public_item_table_v1(
                    public=zf,
                    task_id=task_id,
                    relative_path=name,
                ),
            )
    return None


def _participant_subject_frame(
    adsl: pd.DataFrame,
    *,
    control_arm_id: str,
    treated_arm_id: str,
    static_binary_treatment: bool = True,
) -> pd.DataFrame:
    frame = adsl.loc[:, ["USUBJID"]].copy()
    frame["USUBJID"] = frame["USUBJID"].astype("string")
    if static_binary_treatment:
        frame["treatment"] = _treatment_indicator(
            adsl,
            control_arm_id=control_arm_id,
            treated_arm_id=treated_arm_id,
        )
    for column in ("ARMCD", "TRTA", "ARM", "TRT01A"):
        if column in adsl.columns and adsl[column].astype("string").nunique(dropna=True) >= 2:
            frame["allocation_group"] = adsl[column].astype("string").to_numpy()
            break
    if "allocation_group" not in frame.columns:
        raise ValueError("No randomized allocation group found in participant ADSL")
    if static_binary_treatment:
        frame = frame.dropna(subset=["treatment"]).copy()
        frame["treatment"] = frame["treatment"].astype(np.int64)
    for column in ("SITEID", "RFSTDTC", "INTERVENTION_START_DY"):
        if column in adsl.columns:
            frame[column] = adsl[column].to_numpy()
    return frame


def _ph_diagnostics(
    frame: pd.DataFrame,
    *,
    covariates: pd.DataFrame | None = None,
    continuous_covariates: tuple[str, ...] = (),
    effect_modifier: str | None = None,
) -> tuple[float | None, float | None, float | None, float | None, bool | None, tuple[str, ...]]:
    findings: list[str] = []
    p_value: float | None = None
    slope: float | None = None
    slope_standard_error: float | None = None
    lower_abs_range: float | None = None
    try:
        diagnostic = frame.copy()
        formula = "treatment"
        if covariates is not None:
            if "USUBJID" not in covariates.columns:
                raise ValueError("Adjusted PH evidence requires USUBJID in analysis covariates.")
            adjusted_covariates = covariates.copy()
            adjusted_covariates["USUBJID"] = adjusted_covariates["USUBJID"].astype("string")
            if adjusted_covariates["USUBJID"].duplicated().any():
                raise ValueError("Adjusted PH evidence requires one covariate row per participant.")
            model_columns = tuple(str(column) for column in adjusted_covariates.columns if column != "USUBJID")
            if not model_columns:
                raise ValueError("Adjusted PH evidence requires at least one prespecified covariate.")
            unsafe_names = tuple(column for column in model_columns if not column.replace("_", "").isalnum())
            if unsafe_names:
                raise ValueError(f"Adjusted PH evidence found formula-unsafe covariate names: {unsafe_names!r}.")
            if not set(continuous_covariates).issubset(model_columns):
                raise ValueError("Adjusted PH evidence continuous covariates must be present in the model.")
            if effect_modifier is not None and effect_modifier not in continuous_covariates:
                raise ValueError("Adjusted PH evidence requires a continuous prespecified effect modifier.")
            diagnostic = diagnostic.merge(
                adjusted_covariates,
                on="USUBJID",
                how="left",
                validate="one_to_one",
            )
            if diagnostic.loc[:, list(model_columns)].isna().any().any():
                raise ValueError("Adjusted PH evidence requires complete prespecified covariates.")
            formula_terms = [
                f"bs({column}, df=4)" if column in continuous_covariates else column for column in model_columns
            ]
            if effect_modifier is not None:
                formula_terms.append(f"treatment:bs({effect_modifier}, df=4)")
            formula = " + ".join(("treatment", *formula_terms))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"datetime\.datetime\.utcnow\(\) is deprecated.*",
                category=DeprecationWarning,
                module=r"lifelines\.fitters\.coxph_fitter",
            )
            cph = CoxPHFitter(penalizer=0.0)
            cph.fit(diagnostic, duration_col="duration", event_col="event", formula=formula)
        result = proportional_hazard_test(cph, diagnostic, time_transform="rank")
        p_value = float(result.summary.loc["treatment", "p"])
        residuals = cph.compute_residuals(diagnostic, kind="scaled_schoenfeld")
        event_times = diagnostic.loc[residuals.index, "duration"].to_numpy(dtype=np.float64)
        scaled_residual = residuals["treatment"].to_numpy(dtype=np.float64)
        if len(scaled_residual) < 3 or not np.isfinite(scaled_residual).all():
            raise ValueError("insufficient scaled Schoenfeld residuals")
        order = np.argsort(np.argsort(event_times, kind="mergesort"), kind="mergesort").astype(np.float64)
        rank_time = order / float(max(1, len(order) - 1))
        design = np.column_stack([np.ones(len(rank_time), dtype=np.float64), rank_time - 0.5])
        coefficients, _, _, _ = np.linalg.lstsq(design, scaled_residual, rcond=None)
        fitted = design @ coefficients
        residual_variance = float(np.sum(np.square(scaled_residual - fitted)) / float(len(rank_time) - 2))
        covariance = residual_variance * np.linalg.inv(design.T @ design)
        slope = float(coefficients[1])
        slope_standard_error = float(np.sqrt(max(0.0, covariance[1, 1])))
        critical = diagnostic_normal_critical_value_v1(assumption_id="proportional_hazards", comparisons=1)
        lower_abs_range = float(max(0.0, abs(slope) - critical * slope_standard_error))
    except (ConvergenceError, ValueError, KeyError, np.linalg.LinAlgError, ZeroDivisionError) as error:
        findings.append(f"ph_diagnostic_fit_failed:{type(error).__name__}")
    threshold_crossed = (
        None if lower_abs_range is None else bool(lower_abs_range >= PH_METHOD_CHANGE_TIME_VARIATION_THRESHOLD)
    )
    return p_value, slope, slope_standard_error, lower_abs_range, threshold_crossed, tuple(findings)


def _accepted_families(scoring_key: ValidatedScoringKeyV1) -> tuple[str, ...]:
    """Project credit-eligible route scales into analysis families for proof comparison."""

    return tuple(
        sorted(
            {
                route_family_for_effect_scale_v1(route.signature.effect_scale)
                for route in scoring_key.credit_eligible_routes
            }
        )
    )


def _declared_primary_family(task: dict[str, object]) -> str | None:
    effect_scale = str(task.get("primary_effect_scale") or "")
    if not effect_scale:
        return None
    return cast(str, route_family_for_effect_scale_v1(effect_scale))


def infer_reference_analysis(
    *,
    task_id: str,
    protocol: dict[str, object],
    task: dict[str, object],
    public_members: set[str],
    ph_method_change_threshold_crossed: bool | None,
) -> ReferenceAnalysisDecisionV1:
    """Infer a defensible analysis class using participant evidence without evaluator metadata."""
    design_family = str(protocol.get("design_family") or "")
    observed: list[str] = ["protocol_summary.json", "task.json"]
    ascertainment = f"items/{task_id}/ascertainment_model.json" in public_members
    declared_primary_family = _declared_primary_family(task)
    raw_scale_options = task.get("primary_effect_scale_options")
    if not isinstance(raw_scale_options, list) or not raw_scale_options:
        raise ValueError(f"Participant task lacks primary_effect_scale_options: {task_id}")
    scale_option_families = tuple(
        sorted({route_family_for_effect_scale_v1(str(scale)) for scale in raw_scale_options})
    )
    families: tuple[str, ...]
    required: tuple[str, ...]
    if scale_option_families == ("standardized_risk",):
        evidence_class: DiagnosticEvidenceClassV1 = "model_form_diagnostics"
        families = ("standardized_risk",)
        required = ("model_form_public", "randomization_integrity_public")
        observed.append("reference_population_covariates_available")
    elif ascertainment:
        evidence_class = "endpoint_defect_evidence"
        families = ("risk_difference",)
        required = ("endpoint_ascertainment_public", "randomization_integrity_public")
        observed.append("ascertainment_model.json")
    elif design_family in DESIGN_FAMILIES_REQUIRING_ADJUSTMENT:
        evidence_class = "design_adjustment_evidence"
        required_list = ["randomization_integrity_public"]
        if design_family in {"cluster_parallel_randomized", "stepped_wedge_cluster_rollout"}:
            required_list.append("cluster_structure_public")
        if design_family == "stepped_wedge_cluster_rollout":
            required_list.append("secular_trend_public")
            families = ("risk_difference",)
        elif design_family == "group_sequential_monitoring":
            required_list.append("sequential_design_adjustment_public")
            group_plan = protocol.get("group_sequential_plan")
            monitoring_scale = group_plan.get("monitoring_effect_scale") if isinstance(group_plan, dict) else None
            families = (
                ("risk_difference",)
                if monitoring_scale == "risk_difference_tau"
                else ("risk_difference", "rmst_contrast")
            )
        else:
            families = ("risk_difference", "rmst_contrast")
        required = tuple(sorted(required_list))
        observed.append(f"design_family:{design_family}")
    elif ph_method_change_threshold_crossed is True:
        evidence_class = "non_ph_diagnostics"
        # The public protocol declares the analysis horizon but no time-varying
        # Cox functional form or windows. Diagnostics can rule out an
        # unqualified constant HR; they cannot identify an analyst-chosen
        # time-varying model. Retain the two horizon-defined effect summaries.
        families = ("risk_difference", "rmst_contrast")
        required = ("proportional_hazards_public", "randomization_integrity_public")
        observed.append("ph_method_change_threshold_crossed")
    else:
        evidence_class = "ph_compatible_diagnostics"
        families = scale_option_families
        required = ("proportional_hazards_public", "randomization_integrity_public")
        observed.append("ph_method_change_threshold_not_crossed")
    if declared_primary_family is not None:
        families = (declared_primary_family,)
        observed.append(f"task_primary_effect_scale:{task['primary_effect_scale']}")
    if not str(task.get("primary_endpoint_term") or "").strip():
        raise ValueError(f"Participant task lacks primary_endpoint_term: {task_id}")
    return ReferenceAnalysisDecisionV1(
        task_id=task_id,
        design_family=design_family,
        evidence_class=evidence_class,
        candidate_route_families=tuple(sorted(families)),
        required_diagnostic_keys=tuple(sorted(required)),
        observed_evidence=tuple(observed),
        ph_diagnostic_required=evidence_class in {"ph_compatible_diagnostics", "non_ph_diagnostics"},
    )


def _assumption_status_strings(statuses: dict[str, str]) -> tuple[str, ...]:
    return tuple(f"{assumption_id}={status}" for assumption_id, status in sorted(statuses.items()))


def _input_paths_for_class(
    *,
    task_id: str,
    evidence_class: DiagnosticEvidenceClassV1,
    adsl_member: str | None,
    adtte_member: str | None,
) -> tuple[str, ...]:
    paths: list[str] = [f"items/{task_id}/protocol_summary.json", f"items/{task_id}/task.json"]
    if adsl_member:
        paths.append(adsl_member)
    if adtte_member:
        paths.append(adtte_member)
    if evidence_class == "design_adjustment_evidence":
        paths.append(f"items/{task_id}/protocol_summary.json")
    elif evidence_class == "confounding_design_adjustment_evidence":
        paths.append(f"items/{task_id}/data/raw/reference_population_covariates.parquet")
    elif evidence_class == "model_form_diagnostics":
        paths.append(f"items/{task_id}/data/reference_population_covariates.parquet")
        paths.append(f"items/{task_id}/data/analysis_frame_covariates.parquet")
        paths.append(f"items/{task_id}/data/raw/reference_population_covariates.parquet")
    elif evidence_class == "censoring_followup_diagnostics":
        paths.append(f"items/{task_id}/intercurrent_event_strategy.json")
        paths.append(f"items/{task_id}/data/subject_operational_flags.parquet")
        paths.append(f"items/{task_id}/data/raw/disposition.parquet")
        paths.append(f"items/{task_id}/data/raw/visits.parquet")
    elif evidence_class == "endpoint_defect_evidence":
        paths.append(f"items/{task_id}/ascertainment_model.json")
        paths.append(f"items/{task_id}/data/raw/endpoint_adjudication.parquet")
    elif evidence_class == "censoring_competing_risk_diagnostics":
        paths.append(f"items/{task_id}/data/raw/disposition.parquet")
    return tuple(dict.fromkeys(paths))


def _input_paths_for_diagnostic_keys(*, task_id: str, diagnostic_keys: tuple[str, ...]) -> tuple[str, ...]:
    paths: list[str] = []
    for key in diagnostic_keys:
        if key == "randomization_integrity_public":
            paths.extend(
                [
                    f"items/{task_id}/data/ADSL.parquet",
                    f"items/{task_id}/data/raw/randomization.parquet",
                ]
            )
        elif key == "proportional_hazards_public":
            paths.extend(
                [
                    f"items/{task_id}/data/ADSL.parquet",
                    f"items/{task_id}/data/ADTTE.parquet",
                ]
            )
        elif key == "censoring_followup_public":
            paths.extend(
                [
                    f"items/{task_id}/intercurrent_event_strategy.json",
                    f"items/{task_id}/data/subject_operational_flags.parquet",
                    f"items/{task_id}/data/raw/disposition.parquet",
                    f"items/{task_id}/data/raw/visits.parquet",
                ]
            )
        elif key == "model_form_public":
            paths.extend(
                [
                    f"items/{task_id}/data/analysis_frame_covariates.parquet",
                    f"items/{task_id}/data/reference_population_covariates.parquet",
                    f"items/{task_id}/data/raw/reference_population_covariates.parquet",
                ]
            )
        elif key == "endpoint_ascertainment_public":
            paths.extend(
                [
                    f"items/{task_id}/endpoint_definition.json",
                    f"items/{task_id}/ascertainment_model.json",
                    f"items/{task_id}/data/raw/endpoint_adjudication.parquet",
                    f"items/{task_id}/data/raw/endpoint_reports.parquet",
                ]
            )
        elif key == "cluster_structure_public":
            paths.extend(
                [
                    f"items/{task_id}/data/site_summary.parquet",
                    f"items/{task_id}/data/raw/sites.parquet",
                    f"items/{task_id}/data/ADSL.parquet",
                ]
            )
        elif key == "secular_trend_public":
            paths.extend(
                [
                    f"items/{task_id}/protocol_summary.json",
                    f"items/{task_id}/data/raw/randomization.parquet",
                ]
            )
        elif key == "sequential_design_adjustment_public":
            paths.extend(
                [
                    f"items/{task_id}/protocol_summary.json",
                    f"items/{task_id}/analysis_plan.json",
                ]
            )
    return tuple(dict.fromkeys(paths))


def _hash_members(zf: ZipFile, paths: tuple[str, ...]) -> tuple[str, ...]:
    hashes: list[str] = []
    members = set(zf.namelist())
    for path in paths:
        if path in members:
            hashes.append(f"{path}:sha256:{_sha256_bytes(zf.read(path))}")
    return tuple(hashes)


def _ph_method_applicability_decision(threshold_crossed: bool | None) -> str:
    if threshold_crossed is True:
        return "public_ph_diagnostics_support_non_ph_or_time_varying_effect"
    if threshold_crossed is False:
        return "public_ph_diagnostics_do_not_cross_method_change_threshold"
    return "public_ph_diagnostic_unavailable"


def _status_and_findings(
    *,
    evidence_class: DiagnosticEvidenceClassV1,
    assumption_statuses: dict[str, str],
    missing_diagnostic_keys: tuple[str, ...],
    ph_required: bool,
    ph_available: bool,
    method_change_threshold_crossed: bool | None,
    public_input_paths: tuple[str, ...],
    public_input_hashes: tuple[str, ...],
    diagnostic_findings: tuple[str, ...],
) -> tuple[DiagnosticProofStatusV1, DiagnosticProofDispositionV1, tuple[str, ...], str, tuple[str, ...]]:
    findings = list(diagnostic_findings)
    resolved_warning_keys: list[str] = []
    if not public_input_hashes:
        findings.append("missing_public_input_hashes")
    if missing_diagnostic_keys:
        findings.extend(f"missing_public_diagnostic_key:{key}" for key in missing_diagnostic_keys)
    if ph_required and not ph_available:
        findings.append("ph_diagnostic_unavailable")
    ph_status = str(assumption_statuses.get("proportional_hazards", "holds"))
    if ph_required and ph_available and ph_status == "holds" and method_change_threshold_crossed is True:
        findings.append("ph_holding_contract_crosses_public_method_change_threshold")
    if evidence_class == "non_ph_diagnostics" and method_change_threshold_crossed is not True:
        findings.append("non_ph_claim_lacks_public_method_change_signal")
    if evidence_class == "non_ph_diagnostics":
        decision = _ph_method_applicability_decision(method_change_threshold_crossed)
    elif evidence_class == "ph_compatible_diagnostics":
        decision = (
            "public_ph_diagnostics_support_ph_compatible_use"
            if method_change_threshold_crossed is not True
            else "public_ph_diagnostics_flag_time_variation_for_adjudication"
        )
        if method_change_threshold_crossed is True and ph_status == "holds":
            decision = "public_ph_diagnostics_contradict_ph_holding_contract"
        elif method_change_threshold_crossed is True and ph_status == "stressed":
            resolved_warning_keys.append("mild_ph_stress_expected_by_contract")
            decision = "public_ph_diagnostics_support_mild_ph_stress_but_retain_contract"
    else:
        decision = f"{evidence_class}_public_inputs_present"
    if findings:
        return "fail", "block_release", tuple(findings), decision, tuple(sorted(set(resolved_warning_keys)))
    return "pass", "retain_official", (), decision, tuple(sorted(set(resolved_warning_keys)))


def _safe_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    # Diagnostic decisions use the full-precision value above. Published proof
    # artifacts use a stable precision so BLAS-level noise cannot change hashes.
    return float(f"{value:.12g}")


def _canonical_diagnostic_frame_payload(
    frame: pd.DataFrame | None,
    *,
    subject_ordinals: dict[str, int],
) -> str | None:
    """Serialize one diagnostic input after replacing task-scoped subject IDs."""

    if frame is None:
        return None
    canonical = frame.copy()
    if "USUBJID" in canonical.columns:
        subject_ids = canonical["USUBJID"].astype("string")
        unknown = tuple(sorted(set(subject_ids.dropna().astype(str)) - set(subject_ordinals)))
        if unknown:
            raise ValueError(f"Diagnostic input contains subjects absent from ADSL: {unknown[:5]!r}")
        canonical["USUBJID"] = subject_ids.map(subject_ordinals).astype("Int64")
    canonical.columns = canonical.columns.map(str)
    columns = sorted(str(column) for column in canonical.columns)
    canonical = canonical.loc[:, columns]
    if columns:
        canonical = canonical.sort_values(columns, kind="mergesort", na_position="last").reset_index(drop=True)
    return canonical.to_json(
        orient="table",
        date_format="iso",
        double_precision=15,
        index=False,
    )


def _diagnostic_input_digest(
    *,
    subject_frame: pd.DataFrame,
    survival: pd.DataFrame,
    covariates: pd.DataFrame,
    reference_covariates: pd.DataFrame,
    operational: pd.DataFrame | None,
    ascertainment: dict[str, object] | None,
    endpoint_validation: pd.DataFrame | None,
    protocol: dict[str, object],
    primary_analysis: dict[str, object],
    include_treatment_dependent_diagnostics: bool,
) -> str:
    """Bind cached diagnostics to exact participant-visible numerical inputs."""

    subject_ids = tuple(subject_frame["USUBJID"].astype("string").dropna().astype(str).sort_values().unique())
    subject_ordinals = {subject_id: index for index, subject_id in enumerate(subject_ids)}
    payload = {
        "ascertainment": ascertainment,
        "covariates": _canonical_diagnostic_frame_payload(covariates, subject_ordinals=subject_ordinals),
        "endpoint_validation": _canonical_diagnostic_frame_payload(
            endpoint_validation,
            subject_ordinals=subject_ordinals,
        ),
        "include_treatment_dependent_diagnostics": include_treatment_dependent_diagnostics,
        "operational": _canonical_diagnostic_frame_payload(operational, subject_ordinals=subject_ordinals),
        "primary_analysis": primary_analysis,
        "protocol": protocol,
        "reference_covariates": _canonical_diagnostic_frame_payload(
            reference_covariates,
            subject_ordinals=subject_ordinals,
        ),
        "subject_frame": _canonical_diagnostic_frame_payload(subject_frame, subject_ordinals=subject_ordinals),
        "survival": _canonical_diagnostic_frame_payload(survival, subject_ordinals=subject_ordinals),
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))


def _severity_band_from_decisions(
    *,
    metric_values: dict[Literal["stressed", "fragile", "broken"], float],
    thresholds: dict[Literal["stressed", "fragile", "broken"], float],
) -> AssumptionSeverityBandV1:
    """Classify an assumption from its registered threshold-specific metrics."""

    stressed = metric_values["stressed"] >= thresholds["stressed"]
    fragile = metric_values["fragile"] >= thresholds["fragile"]
    broken = metric_values["broken"] >= thresholds["broken"]
    if broken and not fragile:
        raise ValueError("Broken participant evidence must also cross the fragile threshold.")
    if fragile and not stressed:
        raise ValueError("Fragile participant evidence must also cross the stressed threshold.")
    if not stressed:
        return "holds"
    if not fragile:
        return "mild"
    if not broken:
        return "fragile"
    return "broken"


def _participant_task_ids(public_members: set[str]) -> tuple[str, ...]:
    """Return the complete sorted participant task inventory."""

    task_ids = tuple(
        sorted(
            member.split("/")[1]
            for member in public_members
            if member.startswith("items/") and member.endswith("/task.json") and len(member.split("/")) == 3
        )
    )
    if not task_ids:
        raise ValueError("Participant archive contains no task.json members")
    return task_ids


def _participant_task_groups(public_zip: Path) -> tuple[tuple[str, ...], ...]:
    """Group context projections that can share an exact-input diagnostic cache."""

    groups: dict[str, list[str]] = {}
    with ZipFile(public_zip) as public_zf:
        public_members = set(public_zf.namelist())
        for task_id in _participant_task_ids(public_members):
            task = _read_json_member(public_zf, f"items/{task_id}/task.json")
            protocol = _read_json_member(public_zf, f"items/{task_id}/protocol_summary.json")
            analysis_plan = TrialEvalPublicSAPV1.model_validate(
                _read_json_member(public_zf, f"items/{task_id}/analysis_plan.json")
            )
            data_surface = (
                "analysis_ready"
                if f"items/{task_id}/data/ADTTE.parquet" in public_members
                else "reconstructed_raw_domains"
            )
            payload = {
                "data_surface": data_surface,
                "has_ascertainment_model": f"items/{task_id}/ascertainment_model.json" in public_members,
                "primary_analysis": analysis_plan.primary_analysis.model_dump(mode="json"),
                "protocol": protocol,
                "task": {field: task.get(field) for field in _DIAGNOSTIC_GROUP_TASK_FIELDS},
            }
            group_id = _sha256_bytes(
                json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            )
            groups.setdefault(group_id, []).append(task_id)
    return tuple(tuple(sorted(groups[group_id])) for group_id in sorted(groups))


def _build_participant_diagnostic_batch_v1(
    *,
    public_zip: Path,
    task_ids: tuple[str, ...],
) -> tuple[ParticipantDiagnosticEvidenceV1, ...]:
    """Compute one deterministic batch using an independent archive reader."""

    evidence_rows: list[ParticipantDiagnosticEvidenceV1] = []
    diagnostic_cache: dict[str, _DiagnosticComputationV1] = {}
    with ZipFile(public_zip) as public_zf:
        public_members = set(public_zf.namelist())
        for task_id in task_ids:
            protocol = _read_json_member(public_zf, f"items/{task_id}/protocol_summary.json")
            task = _read_json_member(public_zf, f"items/{task_id}/task.json")
            analysis_plan = TrialEvalPublicSAPV1.model_validate(
                _read_json_member(public_zf, f"items/{task_id}/analysis_plan.json")
            )
            primary_analysis = analysis_plan.primary_analysis
            design_family = str(protocol.get("design_family") or "")
            static_binary_treatment = design_family != "stepped_wedge_cluster_rollout"
            primary_paramcd = task.get("primary_paramcd")
            if not isinstance(primary_paramcd, str) or not primary_paramcd.strip():
                raise ValueError(f"Participant task lacks primary_paramcd: {task_id}")
            adsl, adtte, analysis_sources = load_public_analysis_tables_v1(
                public=public_zf,
                task_id=task_id,
                paramcd=primary_paramcd,
            )
            frame = _survival_frame(
                adsl,
                adtte,
                primary_paramcd=primary_paramcd,
                control_arm_id=str(task.get("primary_control_arm_id") or ""),
                treated_arm_id=str(task.get("primary_treated_arm_id") or ""),
                static_binary_treatment=static_binary_treatment,
            )
            covariates = _optional_participant_table(
                public_zf,
                task_id=task_id,
                names=(
                    "data/analysis_frame_covariates.parquet",
                    "data/raw/baseline_characteristics.parquet",
                ),
            )
            if covariates is None:
                raise ValueError(f"Participant archive lacks analysis covariates for {task_id}.")
            covariates = covariates.copy()
            covariates["USUBJID"] = covariates["USUBJID"].astype("string")
            reference_covariates = _optional_participant_table(
                public_zf,
                task_id=task_id,
                names=(
                    "data/reference_population_covariates.parquet",
                    "data/raw/reference_population_covariates.parquet",
                ),
            )
            if reference_covariates is None:
                raise ValueError(f"Participant archive lacks reference-population covariates for {task_id}.")
            operational = _optional_participant_table(
                public_zf,
                task_id=task_id,
                names=(
                    "data/subject_operational_flags.parquet",
                    "data/raw/disposition.parquet",
                ),
            )
            if operational is not None:
                operational = operational.copy()
                operational["USUBJID"] = operational["USUBJID"].astype("string")
            ascertainment_member = f"items/{task_id}/ascertainment_model.json"
            ascertainment = (
                _read_json_member(public_zf, ascertainment_member) if ascertainment_member in public_members else None
            )
            endpoint_validation = None
            validation_columns = {"VALIDFL", "OBSEVNT", "ADJEVNT"}
            if validation_columns.issubset(adtte.columns):
                endpoint_validation = adtte.loc[
                    adtte["PARAMCD"].astype("string").eq(primary_paramcd),
                    :,
                ].copy()
            elif ascertainment is not None:
                endpoint_validation = _optional_participant_table(
                    public_zf,
                    task_id=task_id,
                    names=("data/raw/endpoint_adjudication.parquet",),
                )
            subject_frame = _participant_subject_frame(
                adsl,
                control_arm_id=str(task.get("primary_control_arm_id") or ""),
                treated_arm_id=str(task.get("primary_treated_arm_id") or ""),
                static_binary_treatment=static_binary_treatment,
            )
            input_digest = _diagnostic_input_digest(
                subject_frame=subject_frame,
                survival=frame,
                covariates=covariates,
                reference_covariates=reference_covariates,
                operational=operational,
                ascertainment=ascertainment,
                endpoint_validation=endpoint_validation,
                protocol=protocol,
                primary_analysis=primary_analysis.model_dump(mode="json"),
                include_treatment_dependent_diagnostics=static_binary_treatment,
            )
            computation = diagnostic_cache.get(input_digest)
            if computation is None:
                baseline_covariates, continuous_covariates = derive_participant_covariate_basis_v1(
                    covariates=covariates,
                    reference_covariates=reference_covariates,
                )
                adjusted_covariates = (
                    covariates.loc[:, ["USUBJID", *baseline_covariates]]
                    if primary_analysis.baseline_covariate_strategy == "all_released"
                    else None
                )
                effect_modifier = primary_analysis.treatment_effect_modifier
                try:
                    assumption_diagnostics = list(
                        compute_participant_assumption_diagnostics_v1(
                            subject_frame=subject_frame,
                            survival=frame,
                            covariates=covariates,
                            reference_covariates=reference_covariates,
                            operational=operational,
                            ascertainment=ascertainment,
                            endpoint_validation=endpoint_validation,
                            protocol=protocol,
                            include_treatment_dependent_diagnostics=static_binary_treatment,
                            treatment_effect_modifier=effect_modifier,
                        )
                    )
                except (KeyError, ValueError) as error:
                    raise ValueError(f"Participant diagnostic replay failed for {task_id}: {error}") from error
                if design_family in DESIGN_FAMILIES_REQUIRING_ADJUSTMENT:
                    p_value = slope = slope_standard_error = lower_abs_range = None
                    method_change_threshold_crossed = None
                    diagnostic_findings: tuple[str, ...] = ()
                else:
                    (
                        p_value,
                        slope,
                        slope_standard_error,
                        lower_abs_range,
                        method_change_threshold_crossed,
                        diagnostic_findings,
                    ) = _ph_diagnostics(
                        frame,
                        covariates=adjusted_covariates,
                        continuous_covariates=(continuous_covariates if adjusted_covariates is not None else ()),
                        effect_modifier=effect_modifier,
                    )
                if lower_abs_range is not None:
                    if p_value is None or slope is None or slope_standard_error is None:
                        raise ValueError("PH diagnostic severity requires complete Schoenfeld evidence.")
                    assumption_diagnostics.append(
                        ParticipantAssumptionDiagnosticV1(
                            assumption_id="proportional_hazards",
                            severity_metric_name="simultaneous_lower_abs_time_varying_log_hazard_range",
                            severity_metric=float(lower_abs_range),
                            supporting_metrics={
                                "n_events": float(frame["event"].sum()),
                                "scaled_schoenfeld_rank_slope": float(slope),
                                "abs_scaled_schoenfeld_rank_slope": abs(float(slope)),
                                "scaled_schoenfeld_rank_slope_standard_error": float(slope_standard_error),
                                "simultaneous_lower_abs_time_varying_log_hazard_range": float(lower_abs_range),
                                "schoenfeld_rank_test_p_value": float(p_value),
                            },
                        )
                    )
                computation = _DiagnosticComputationV1(
                    assumption_diagnostics=tuple(assumption_diagnostics),
                    schoenfeld_p_value=p_value,
                    scaled_schoenfeld_rank_slope=slope,
                    scaled_schoenfeld_rank_slope_standard_error=slope_standard_error,
                    simultaneous_lower_abs_time_varying_log_hazard_range=lower_abs_range,
                    ph_method_change_threshold_crossed=method_change_threshold_crossed,
                    findings=diagnostic_findings,
                )
                diagnostic_cache[input_digest] = computation
            assumption_diagnostics = list(computation.assumption_diagnostics)
            p_value = computation.schoenfeld_p_value
            slope = computation.scaled_schoenfeld_rank_slope
            slope_standard_error = computation.scaled_schoenfeld_rank_slope_standard_error
            lower_abs_range = computation.simultaneous_lower_abs_time_varying_log_hazard_range
            method_change_threshold_crossed = computation.ph_method_change_threshold_crossed
            diagnostic_findings = computation.findings
            decision = infer_reference_analysis(
                task_id=task_id,
                protocol=protocol,
                task=task,
                public_members=public_members,
                ph_method_change_threshold_crossed=method_change_threshold_crossed,
            )
            adsl_member = analysis_sources[0] if analysis_sources else None
            adtte_member = analysis_sources[1] if len(analysis_sources) > 1 else None
            diagnostic_key_by_assumption = diagnostic_key_by_assumption_id_v1()
            satisfied_diagnostic_keys = tuple(
                sorted(
                    {
                        diagnostic_key_by_assumption[diagnostic.assumption_id]
                        for diagnostic in assumption_diagnostics
                        if diagnostic.assumption_id in diagnostic_key_by_assumption
                    }
                )
            )
            missing_diagnostic_keys = tuple(
                sorted(set(decision.required_diagnostic_keys) - set(satisfied_diagnostic_keys))
            )
            public_input_paths = _input_paths_for_class(
                task_id=task_id,
                evidence_class=decision.evidence_class,
                adsl_member=adsl_member,
                adtte_member=adtte_member,
            )
            public_input_paths = tuple(
                dict.fromkeys(
                    (
                        *public_input_paths,
                        *_input_paths_for_diagnostic_keys(
                            task_id=task_id,
                            diagnostic_keys=satisfied_diagnostic_keys,
                        ),
                    )
                )
            )
            public_input_hashes = _hash_members(public_zf, public_input_paths)
            evidence_rows.append(
                ParticipantDiagnosticEvidenceV1(
                    task_id=task_id,
                    decision=decision,
                    satisfied_diagnostic_keys=satisfied_diagnostic_keys,
                    missing_diagnostic_keys=missing_diagnostic_keys,
                    public_input_paths=public_input_paths,
                    public_input_hashes=public_input_hashes,
                    n_subjects=int(len(frame)),
                    n_events=int(frame["event"].sum()),
                    treated_events=(
                        int(frame.loc[frame["treatment"] == 1, "event"].sum()) if static_binary_treatment else None
                    ),
                    control_events=(
                        int(frame.loc[frame["treatment"] == 0, "event"].sum()) if static_binary_treatment else None
                    ),
                    schoenfeld_p_value=_safe_float(p_value),
                    scaled_schoenfeld_rank_slope=_safe_float(slope),
                    scaled_schoenfeld_rank_slope_standard_error=_safe_float(slope_standard_error),
                    simultaneous_lower_abs_time_varying_log_hazard_range=_safe_float(lower_abs_range),
                    ph_method_change_threshold_crossed=method_change_threshold_crossed,
                    assumption_diagnostics=tuple(
                        sorted(assumption_diagnostics, key=lambda diagnostic: diagnostic.assumption_id)
                    ),
                    diagnostic_findings=diagnostic_findings,
                )
            )
    return tuple(evidence_rows)


def build_participant_diagnostic_evidence_v1(
    *,
    public_zip: Path,
    workers: int = TRIALEVAL_DIAGNOSTIC_PROOF_DEFAULT_WORKERS,
) -> tuple[ParticipantDiagnosticEvidenceV1, ...]:
    """Compute participant-only diagnostics with bounded numerical parallelism."""

    if workers < 1:
        raise ValueError("Diagnostic proof workers must be at least one.")
    task_groups = _participant_task_groups(public_zip)
    worker_count = min(workers, len(task_groups))
    batch_lists: list[list[str]] = [[] for _ in range(worker_count)]
    for index, task_group in enumerate(task_groups):
        batch_lists[index % worker_count].extend(task_group)
    batches = tuple(tuple(batch) for batch in batch_lists)
    batch_results = _run_participant_diagnostic_workers_v1(
        public_zip=public_zip,
        batches=batches,
    )
    return tuple(sorted((row for batch in batch_results for row in batch), key=lambda row: row.task_id))


def _run_participant_diagnostic_workers_v1(
    *,
    public_zip: Path,
    batches: tuple[tuple[str, ...], ...],
) -> tuple[tuple[ParticipantDiagnosticEvidenceV1, ...], ...]:
    """Run independent diagnostic workers over safe JSON transport."""

    environment = os.environ.copy()
    environment.update({name: "1" for name in _NUMERICAL_WORKER_THREAD_ENVIRONMENT})
    environment["PYTHONHASHSEED"] = "0"
    workers: list[_DiagnosticWorkerProcessV1] = []
    with tempfile.TemporaryDirectory(prefix="trialeval-diagnostic-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        for index, batch in enumerate(batches):
            output_path = temporary_root / f"batch-{index:03d}.json"
            stderr_path = temporary_root / f"batch-{index:03d}.stderr"
            command = [
                sys.executable,
                "-m",
                "trialagentbench_harness.analysis.trialeval_diagnostic_worker",
                "--public-zip",
                public_zip.as_posix(),
            ]
            for task_id in batch:
                command.extend(("--task-id", task_id))
            with output_path.open("wb") as output_stream, stderr_path.open("wb") as stderr_stream:
                process = subprocess.Popen(
                    command,
                    env=environment,
                    stdout=output_stream,
                    stderr=stderr_stream,
                )
            workers.append(
                _DiagnosticWorkerProcessV1(
                    process=process,
                    output_path=output_path,
                    stderr_path=stderr_path,
                )
            )
        for worker in workers:
            return_code = worker.process.wait()
            if return_code != 0:
                for pending in workers:
                    if pending.process.poll() is None:
                        pending.process.terminate()
                for pending in workers:
                    pending.process.wait()
                message = worker.stderr_path.read_text(encoding="utf-8", errors="replace").strip()
                if return_code == TRIALEVAL_DIAGNOSTIC_WORKER_INVALID_INPUT_EXIT_CODE:
                    raise ValueError(message)
                raise RuntimeError(f"Participant diagnostic worker failed: {message}")
        results = []
        for worker in workers:
            payload = json.loads(worker.output_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("Participant diagnostic worker output must be a JSON array.")
            results.append(tuple(ParticipantDiagnosticEvidenceV1.model_validate(row) for row in payload))
    return tuple(results)


def validate_trialeval_diagnostic_proof_surface_v1(
    *,
    public_zip: Path,
    evaluator_zip: Path,
    workers: int = TRIALEVAL_DIAGNOSTIC_PROOF_DEFAULT_WORKERS,
) -> TrialEvalDiagnosticProofReportV1:
    """Compare participant-only evidence with evaluator contracts."""

    inferred = build_participant_diagnostic_evidence_v1(public_zip=public_zip, workers=workers)

    # Evaluator metadata is joined only after every participant-side decision exists.
    entries_by_task = {str(entry.get("task_id") or ""): entry for entry in _load_item_index(evaluator_zip)}
    item_ids = tuple(sorted(entries_by_task))
    scoring_keys = ScoringKeyStoreV1.from_evaluator_zip(
        evaluator_zip,
        expected_item_ids=item_ids,
    )
    assumption_evidence_by_task = _load_assumption_evidence(evaluator_zip)

    rows: list[TrialEvalDiagnosticProofRowV1] = []
    for provisional in inferred:
        decision = provisional.decision
        task_id = provisional.task_id
        entry = entries_by_task.get(task_id)
        assumption_evidence = assumption_evidence_by_task.get(task_id)
        if entry is None or assumption_evidence is None:
            raise ValueError(f"Evaluator archive lacks item metadata or assumption evidence for {task_id}")
        scoring_key = scoring_keys.for_item(task_id)
        item_id = str(entry.get("item_id") or "")
        factor_payload = entry.get("factors")
        if not isinstance(factor_payload, dict):
            raise ValueError(f"Evaluator item-index entry lacks explicit factors for {task_id}.")
        evidence_factors = TrialEvalEvidenceFactorsV1.model_validate(
            {
                "context_configuration": factor_payload.get("context_configuration"),
                "data_preparation": factor_payload.get("data_preparation"),
                "analysis_specification": factor_payload.get("analysis_specification"),
            }
        )
        assumption_statuses = {
            str(record.assumption_id): str(record.expected_status) for record in assumption_evidence.records
        }
        accepted_families = _accepted_families(scoring_key)
        p_value = provisional.schoenfeld_p_value
        slope = provisional.scaled_schoenfeld_rank_slope
        slope_standard_error = provisional.scaled_schoenfeld_rank_slope_standard_error
        lower_abs_range = provisional.simultaneous_lower_abs_time_varying_log_hazard_range
        method_change_threshold_crossed = provisional.ph_method_change_threshold_crossed
        joined_findings = list(provisional.diagnostic_findings)
        participant_diagnostics = {
            diagnostic.assumption_id: diagnostic for diagnostic in provisional.assumption_diagnostics
        }
        assumption_replays: list[AssumptionDiagnosticReplayV1] = []
        for generated in assumption_evidence.records:
            assumption_id = str(generated.assumption_id)
            if assumption_id == "proportional_hazards" and not decision.ph_diagnostic_required:
                continue
            if str(generated.diagnosability) in {"design_declared", "not_identifiable"}:
                continue
            generated_numeric = (
                generated.severity_metric,
                generated.threshold_stressed,
                generated.threshold_fragile,
                generated.threshold_broken,
            )
            if any(value is None for value in generated_numeric):
                raise ValueError(f"Empirically diagnosable assumption lacks numeric evidence: {assumption_id}")
            generated_severity = cast(float, generated.severity_metric)
            threshold_stressed = cast(float, generated.threshold_stressed)
            threshold_fragile = cast(float, generated.threshold_fragile)
            threshold_broken = cast(float, generated.threshold_broken)
            replay = participant_diagnostics.get(assumption_id)
            if replay is None:
                joined_findings.append(f"participant_assumption_replay_missing:{assumption_id}")
                continue
            if replay.severity_metric_name != generated.severity_metric_name:
                joined_findings.append(f"participant_assumption_metric_disagrees:{assumption_id}")
                continue
            numeric_agreement = math.isclose(
                generated_severity,
                float(replay.severity_metric),
                rel_tol=1e-6,
                abs_tol=1e-9,
            )
            if not numeric_agreement:
                joined_findings.append(f"participant_assumption_replay_disagrees:{assumption_id}")
            decision_threshold_ids: tuple[Literal["stressed", "fragile", "broken"], ...] = (
                "stressed",
                "fragile",
                "broken",
            )
            participant_metric_values = {
                threshold_id: float(
                    replay.severity_metric
                    if generated.decision_metric_names[threshold_id] == replay.severity_metric_name
                    else replay.supporting_metrics[generated.decision_metric_names[threshold_id]]
                )
                for threshold_id in decision_threshold_ids
            }
            evaluator_metric_values = {
                threshold_id: generated.decision_metric(threshold_id)[1] for threshold_id in decision_threshold_ids
            }
            thresholds: dict[Literal["stressed", "fragile", "broken"], float] = {
                "stressed": threshold_stressed,
                "fragile": threshold_fragile,
                "broken": threshold_broken,
            }
            decision_numeric_agreement = all(
                math.isclose(
                    participant_metric_values[threshold_id],
                    evaluator_metric_values[threshold_id],
                    rel_tol=1e-6,
                    abs_tol=1e-9,
                )
                for threshold_id in decision_threshold_ids
            )
            numeric_agreement = numeric_agreement and decision_numeric_agreement
            participant_band = _severity_band_from_decisions(
                metric_values=participant_metric_values,
                thresholds=thresholds,
            )
            classification_applicable = True
            classification_agreement = participant_band == str(generated.computed_band)
            if classification_agreement is False:
                joined_findings.append(f"participant_assumption_band_disagrees:{assumption_id}")
            assumption_replays.append(
                AssumptionDiagnosticReplayV1(
                    assumption_id=assumption_id,
                    diagnosability=str(generated.diagnosability),
                    severity_metric_name=replay.severity_metric_name,
                    participant_severity_metric=float(replay.severity_metric),
                    evaluator_severity_metric=generated_severity,
                    threshold_stressed=threshold_stressed,
                    threshold_fragile=threshold_fragile,
                    threshold_broken=threshold_broken,
                    decision_metric_names=generated.decision_metric_names,
                    participant_decision_metric_values=participant_metric_values,
                    evaluator_decision_metric_values=evaluator_metric_values,
                    participant_band=participant_band,
                    evaluator_band=generated.computed_band,
                    classification_applicable=classification_applicable,
                    numeric_agreement=numeric_agreement,
                    classification_agreement=classification_agreement,
                    nearest_threshold_margin=min(
                        abs(participant_metric_values[threshold_id] - thresholds[threshold_id])
                        for threshold_id in decision_threshold_ids
                    ),
                )
            )
        ph_available = lower_abs_range is not None
        key_by_assumption = diagnostic_key_by_assumption_id_v1()
        required_diagnostic_keys = tuple(
            sorted(
                set(decision.required_diagnostic_keys)
                | {
                    key_by_assumption[str(record.assumption_id)]
                    for record in assumption_evidence.records
                    if str(record.assumption_id) in key_by_assumption
                }
            )
        )
        missing_diagnostic_keys = tuple(
            sorted(set(required_diagnostic_keys) - set(provisional.satisfied_diagnostic_keys))
        )
        public_input_paths = provisional.public_input_paths
        public_input_hashes = provisional.public_input_hashes
        status, disposition, findings, method_applicability_decision, resolved_warning_keys = _status_and_findings(
            evidence_class=decision.evidence_class,
            assumption_statuses=assumption_statuses,
            missing_diagnostic_keys=missing_diagnostic_keys,
            ph_required=decision.ph_diagnostic_required,
            ph_available=ph_available,
            method_change_threshold_crossed=method_change_threshold_crossed,
            public_input_paths=public_input_paths,
            public_input_hashes=public_input_hashes,
            diagnostic_findings=tuple(joined_findings),
        )
        neg_log10 = -math.log10(max(p_value, 1e-300)) if p_value is not None else None
        rows.append(
            TrialEvalDiagnosticProofRowV1(
                task_id=task_id,
                item_id=item_id,
                variant_id=str(entry.get("variant_id") or ""),
                design_tier=str(factor_payload.get("design_archetype") or ""),
                assumption_tier=str(factor_payload.get("assumption_regime") or ""),
                context_tier=evidence_factors.context_configuration,
                design_family=decision.design_family,
                evidence_class=decision.evidence_class,
                intended_assumption_statuses=_assumption_status_strings(assumption_statuses),
                required_diagnostic_keys=required_diagnostic_keys,
                satisfied_diagnostic_keys=provisional.satisfied_diagnostic_keys,
                missing_diagnostic_keys=missing_diagnostic_keys,
                resolved_warning_keys=resolved_warning_keys,
                assumption_replays=tuple(sorted(assumption_replays, key=lambda replay: replay.assumption_id)),
                inferred_route_families=decision.candidate_route_families,
                credit_eligible_route_families=accepted_families,
                public_input_paths=public_input_paths,
                public_input_hashes=public_input_hashes,
                n_subjects=provisional.n_subjects,
                n_events=provisional.n_events,
                treated_events=provisional.treated_events,
                control_events=provisional.control_events,
                schoenfeld_p_value=_safe_float(p_value),
                neg_log10_schoenfeld_p=_safe_float(neg_log10),
                scaled_schoenfeld_rank_slope=_safe_float(slope),
                scaled_schoenfeld_rank_slope_standard_error=_safe_float(slope_standard_error),
                simultaneous_lower_abs_time_varying_log_hazard_range=_safe_float(lower_abs_range),
                ph_method_change_threshold_crossed=method_change_threshold_crossed,
                ph_diagnostic_required=decision.ph_diagnostic_required,
                ph_diagnostic_available=ph_available,
                method_applicability_rule_id=(
                    (
                        "scaled_schoenfeld_rank_slope_lower_95_abs_range_ge_"
                        f"{PH_METHOD_CHANGE_TIME_VARIATION_HR_RATIO:.2f}_method_change"
                    )
                    if decision.ph_diagnostic_required
                    else f"{decision.evidence_class}_public_input_presence_v1"
                ),
                method_applicability_decision=method_applicability_decision,
                proof_surface_source_rows=(
                    f"trialeval_diagnostic_proof_rows.csv:{task_id}",
                    f"trialeval_diagnostic_coverage_by_tier.csv:{decision.evidence_class}",
                ),
                status=status,
                disposition=disposition,
                findings=findings,
            )
        )
    failed = sum(1 for row in rows if row.status == "fail")
    ph_rows = [row for row in rows if row.ph_diagnostic_required]
    warning_rows = [row for row in rows if row.resolved_warning_keys or "warning" in row.method_applicability_decision]
    return TrialEvalDiagnosticProofReportV1(
        public_zip=public_zip.as_posix(),
        evaluator_zip=evaluator_zip.as_posix(),
        status="fail" if failed else "pass",
        total_items=len(rows),
        failed_items=failed,
        warning_items=len(warning_rows),
        unresolved_warning_items=sum(1 for row in rows if row.findings),
        ph_diagnostic_rows=len(ph_rows),
        ph_method_change_rows=sum(1 for row in ph_rows if row.ph_method_change_threshold_crossed is True),
        non_ph_diagnostic_rows=sum(1 for row in rows if row.evidence_class == "non_ph_diagnostics"),
        design_adjustment_rows=sum(1 for row in rows if row.evidence_class == "design_adjustment_evidence"),
        source_table_count=5,
        figure_source_count=2,
        rows=tuple(rows),
        summaries=_summary_rows(rows),
    )


def _summary_rows(rows: list[TrialEvalDiagnosticProofRowV1]) -> tuple[TrialEvalDiagnosticProofSummaryRowV1, ...]:
    summaries: list[TrialEvalDiagnosticProofSummaryRowV1] = []
    for group_kind in ("evidence_class", "design_tier", "assumption_tier", "context_tier", "design_family"):
        grouped: dict[str, list[TrialEvalDiagnosticProofRowV1]] = {}
        for row in rows:
            grouped.setdefault(str(getattr(row, group_kind)), []).append(row)
        for group_value, group_rows in sorted(grouped.items()):
            status_counts = Counter(row.status for row in group_rows)
            disposition_counts = Counter(row.disposition for row in group_rows)
            summaries.append(
                TrialEvalDiagnosticProofSummaryRowV1(
                    group_kind=group_kind,
                    group_value=group_value,
                    total=len(group_rows),
                    passed=status_counts["pass"],
                    failed=status_counts["fail"],
                    retain_official=disposition_counts["retain_official"],
                    exclude_or_downgrade=disposition_counts["exclude_or_downgrade"],
                    block_release=disposition_counts["block_release"],
                )
            )
    return tuple(summaries)


__all__ = [
    "build_participant_diagnostic_evidence_v1",
    "validate_trialeval_diagnostic_proof_surface_v1",
]
