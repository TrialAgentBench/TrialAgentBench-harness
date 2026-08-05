from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from trialagentbench_harness.analysis.participant_assumption_diagnostics import (
    ParticipantDiagnosticEstimationError,
    _censoring_diagnostic,
    _cluster_diagnostic,
    _endpoint_diagnostic,
    _interim_diagnostic,
    _model_form_diagnostic,
    _randomization_diagnostic,
    _secular_diagnostic,
    compute_participant_assumption_diagnostics_v1,
)
from trialagentbench_harness.analysis.trialeval_diagnostic_proof_surface import (
    _ph_diagnostics,
    _run_participant_diagnostic_workers_v1,
    _safe_float,
    _survival_frame,
    _treatment_indicator,
    build_participant_diagnostic_evidence_v1,
    infer_reference_analysis,
    validate_trialeval_diagnostic_proof_surface_v1,
)
from trialagentbench_harness.contracts.scoring.assumption_evidence import AssumptionEvidenceManifestV1
from trialagentbench_harness.contracts.scoring.diagnostic_registry import load_diagnostic_registry_v1
from trialagentbench_harness.tools.validate.validate_trialeval_diagnostic_proof_surface import main


def test_diagnostic_proof_canonicalizes_machine_precision_noise() -> None:
    assert _safe_float(0.3907108602967015) == _safe_float(0.3907108602967017)
    assert _safe_float(float("nan")) is None


def test_secular_diagnostic_rejects_non_common_closed_cohort_baselines() -> None:
    subject_frame = pd.DataFrame(
        {
            "USUBJID": ["S1", "S2"],
            "RFSTDTC": ["2024-01-01", "2024-01-02"],
            "INTERVENTION_START_DY": [0.0, 28.0],
        }
    )
    with pytest.raises(ValueError, match="one common RFSTDTC"):
        _secular_diagnostic(
            subject_frame=subject_frame,
            survival=pd.DataFrame({"USUBJID": ["S1", "S2"], "duration": [56.0, 56.0], "event": [1, 0]}),
            covariates=pd.DataFrame({"USUBJID": ["S1", "S2"]}),
            baseline_columns=(),
        )


def test_secular_diagnostic_fails_loudly_when_no_events_are_estimable() -> None:
    subject_frame = pd.DataFrame(
        {
            "USUBJID": ["S1", "S2", "S3", "S4"],
            "RFSTDTC": ["2024-01-01"] * 4,
            "INTERVENTION_START_DY": [0.0, 0.0, 28.0, 28.0],
        }
    )
    with pytest.raises(ParticipantDiagnosticEstimationError, match="at least one observed event"):
        _secular_diagnostic(
            subject_frame=subject_frame,
            survival=pd.DataFrame(
                {
                    "USUBJID": ["S1", "S2", "S3", "S4"],
                    "duration": [56.0] * 4,
                    "event": [0, 0, 0, 0],
                }
            ),
            covariates=pd.DataFrame({"USUBJID": ["S1", "S2", "S3", "S4"]}),
            baseline_columns=(),
        )


def test_endpoint_diagnostic_uses_registered_primary_metric_and_arm_stratum_support() -> None:
    subject_ids = [f"S{index}" for index in range(8)]
    subject_frame = pd.DataFrame(
        {
            "USUBJID": subject_ids,
            "allocation_group": ["control", "treated"] * 4,
        }
    )
    covariates = pd.DataFrame(
        {
            "USUBJID": subject_ids,
            "AGE": [40.0, 41.0, 42.0, 43.0, 60.0, 61.0, 62.0, 63.0],
        }
    )
    validation = pd.DataFrame(
        {
            "USUBJID": subject_ids,
            "VALSTRAT": ["lower_risk"] * 4 + ["higher_risk"] * 4,
            "VALIDFL": [1, 1, 1, 1, 1, 0, 1, 0],
            "OBSEVNT": [0, 0, 1, 1, 0, 0, 1, 1],
            "ADJEVNT": [0, 0, 1, 1, 0, np.nan, 1, np.nan],
        }
    )

    diagnostic = _endpoint_diagnostic(
        validation=validation,
        ascertainment={
            "validation_sampling_fraction": 0.75,
            "prognostic_stratum_variable": "AGE",
            "prognostic_stratum_cutpoint_rule": "pooled_median",
        },
        subject_frame=subject_frame,
        covariates=covariates,
    )

    assert diagnostic is not None
    assert diagnostic.severity_metric_name == "validation_discordance_fraction"
    assert diagnostic.severity_metric == 0.0
    assert diagnostic.supporting_metrics["minimum_validated_records_per_arm_stratum"] == 0.0
    assert diagnostic.supporting_metrics["unsupported_validation_stratum_fraction"] == 0.5


def test_secular_diagnostic_fails_loudly_for_rank_deficient_design() -> None:
    subject_frame = pd.DataFrame(
        {
            "USUBJID": ["S1", "S2", "S3", "S4"],
            "RFSTDTC": ["2024-01-01"] * 4,
            "INTERVENTION_START_DY": [0.0, 0.0, 28.0, 28.0],
            "SITEID": ["A", "A", "B", "B"],
        }
    )
    with pytest.raises(ParticipantDiagnosticEstimationError, match="rank deficient"):
        _secular_diagnostic(
            subject_frame=subject_frame,
            survival=pd.DataFrame(
                {
                    "USUBJID": ["S1", "S2", "S3", "S4"],
                    "duration": [28.0, 56.0, 28.0, 56.0],
                    "event": [1, 0, 1, 0],
                }
            ),
            covariates=pd.DataFrame(
                {
                    "USUBJID": ["S1", "S2", "S3", "S4"],
                    "X_SITE": [0.0, 0.0, 1.0, 1.0],
                }
            ),
            baseline_columns=("X_SITE",),
        )


def test_model_form_replay_retains_the_prespecified_linear_treatment_interaction() -> None:
    rng = np.random.RandomState(714)
    n = 2400
    subject_ids = np.asarray([f"S{index:05d}" for index in range(n)], dtype=object)
    treatment = rng.binomial(1, 0.5, size=n)
    covariate = rng.normal(size=n)
    hazard = 0.15 * np.exp(-0.15 * treatment - 0.80 * treatment * covariate)
    latent_time = rng.exponential(scale=1.0 / hazard)
    censor_time = rng.uniform(2.0, 12.0, size=n)
    observed_time = np.minimum(latent_time, censor_time)
    diagnostic = _model_form_diagnostic(
        subject_frame=pd.DataFrame(
            {
                "USUBJID": subject_ids,
                "treatment": treatment,
            }
        ),
        survival=pd.DataFrame(
            {
                "USUBJID": subject_ids,
                "duration": observed_time,
                "event": (latent_time <= censor_time).astype(np.int64),
            }
        ),
        covariates=pd.DataFrame(
            {
                "USUBJID": subject_ids,
                "X_CONT_001": covariate,
            }
        ),
        continuous_columns=("X_CONT_001",),
        effect_modifier_column="X_CONT_001",
    )

    assert diagnostic.severity_metric_name == "simultaneous_lower_abs_omitted_model_term_log_hazard"
    assert diagnostic.severity_metric == pytest.approx(0.0)


def test_model_form_replay_respects_nonlinear_interaction_hierarchy() -> None:
    rng = np.random.RandomState(714)
    n = 2400
    subject_ids = np.asarray([f"S{index:05d}" for index in range(n)], dtype=object)
    treatment = rng.binomial(1, 0.5, size=n)
    covariate = rng.normal(size=n)
    hazard = 0.15 * np.exp(-0.15 * treatment + 0.45 * covariate * covariate + 0.45 * treatment * covariate * covariate)
    latent_time = rng.exponential(scale=1.0 / hazard)
    censor_time = rng.uniform(2.0, 12.0, size=n)
    diagnostic = _model_form_diagnostic(
        subject_frame=pd.DataFrame(
            {
                "USUBJID": subject_ids,
                "treatment": treatment,
            }
        ),
        survival=pd.DataFrame(
            {
                "USUBJID": subject_ids,
                "duration": np.minimum(latent_time, censor_time),
                "event": (latent_time <= censor_time).astype(np.int64),
            }
        ),
        covariates=pd.DataFrame(
            {
                "USUBJID": subject_ids,
                "X_CONT_001": covariate,
            }
        ),
        continuous_columns=("X_CONT_001",),
        effect_modifier_column="X_CONT_001",
    )

    assert diagnostic.severity_metric > np.log(1.15)
    assert diagnostic.supporting_metrics["max_abs_omitted_model_term_log_hazard"] < 0.70
    assert diagnostic.supporting_metrics["multiplicity_adjusted_p_value"] < 0.05


def test_adjusted_ph_replay_does_not_confuse_effect_modification_with_time_variation() -> None:
    rng = np.random.default_rng(714)
    n_subjects = 5_000
    subject_ids = np.asarray([f"S{index:05d}" for index in range(n_subjects)], dtype=object)
    bmi = rng.normal(0.0, 1.0, n_subjects)
    age = rng.normal(0.0, 1.0, n_subjects)
    treatment = rng.integers(0, 2, n_subjects)
    log_hazard = 0.25 * bmi + 0.35 * np.square(bmi) + treatment * (-0.25 + 0.50 * bmi + 0.35 * np.square(bmi))
    event_time = -np.log(rng.uniform(size=n_subjects)) / (0.01 * np.exp(log_hazard))
    frame = pd.DataFrame(
        {
            "USUBJID": subject_ids,
            "duration": event_time,
            "event": np.ones(n_subjects, dtype=np.int64),
            "treatment": treatment,
        }
    )
    covariates = pd.DataFrame(
        {
            "USUBJID": subject_ids,
            "AGE": age,
            "BMI": bmi,
        }
    )

    _, _, _, marginal_severity, _, _ = _ph_diagnostics(frame)
    _, _, _, conditional_severity, _, _ = _ph_diagnostics(
        frame,
        covariates=covariates,
        continuous_covariates=("AGE", "BMI"),
        effect_modifier="BMI",
    )

    assert marginal_severity is not None and marginal_severity > 0.30
    assert conditional_severity == pytest.approx(0.0)


def test_assumption_evidence_rejects_tampered_embedded_checksum() -> None:
    payload = _assumption_evidence_manifest_payload(
        item_id="d1a1_rct_clean_00",
        context_tier="C1",
    )
    payload["checksum"] = "0" * 64

    with pytest.raises(ValueError, match="checksum mismatch"):
        AssumptionEvidenceManifestV1.model_validate(payload)


def test_design_declared_assumption_checksum_omits_absent_numeric_fields() -> None:
    payload = _assumption_evidence_manifest_payload(item_id="d4a4_adaptive", context_tier="C1")
    records = payload["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    record.update(
        {
            "assumption_id": "sequential_design_adjustment",
            "expected_status": "broken",
            "computed_status": "broken",
            "expected_band": "broken",
            "computed_band": "broken",
            "diagnosability": "design_declared",
            "factual_public_evidence_basis": ["protocol_summary.json"],
        }
    )
    for field in (
        "severity_metric",
        "severity_metric_name",
        "threshold_stressed",
        "threshold_fragile",
        "threshold_broken",
    ):
        record.pop(field, None)
    record["decision_metric_names"] = {}
    record["metric_units"] = {}
    record["metric_public_evidence_basis"] = {}
    canonical_payload = {key: value for key, value in payload.items() if key != "checksum"}
    canonical = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    payload["checksum"] = hashlib.sha256(canonical).hexdigest()

    manifest = AssumptionEvidenceManifestV1.model_validate(payload)

    assert manifest.records[0].severity_metric is None


def test_interim_design_diagnostic_validates_complete_public_plan() -> None:
    plan: dict[str, object] = {
        "looks": [0.5, 0.75, 1.0],
        "monitoring_effect_scale": "risk_difference_tau",
        "nominal_two_sided_alpha_by_look": [0.005, 0.02, 0.041],
        "spending_function_id": "obrien_fleming",
        "two_sided_alpha": 0.05,
        "z_critical_by_look": [2.8, 2.3, 2.04],
    }
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    plan["checksum"] = hashlib.sha256(canonical).hexdigest()

    diagnostic = _interim_diagnostic({"group_sequential_plan": plan})

    assert diagnostic is not None
    assert diagnostic.severity_metric_name == "declared_sequential_design"
    assert diagnostic.supporting_metrics["n_looks"] == 3.0

    malformed = dict(plan)
    malformed["looks"] = [0.5, 0.75]
    with pytest.raises(ValueError, match="checksum mismatch"):
        _interim_diagnostic({"group_sequential_plan": malformed})


def test_randomization_diagnostic_compares_every_allocation_group() -> None:
    groups = tuple(group for group in ("S0", "S1", "S2", "S3") for _ in range(40))
    age = np.array(
        [
            (-1.0 if (index % 40) % 2 == 0 else 1.0) + (0.4 if group == "S1" else 0.0)
            for index, group in enumerate(groups)
        ]
    )
    subject_ids = [f"SUBJ{index:04d}" for index in range(len(groups))]

    diagnostic = _randomization_diagnostic(
        subject_frame=pd.DataFrame({"USUBJID": subject_ids, "allocation_group": groups}),
        covariates=pd.DataFrame({"USUBJID": subject_ids, "AGE": age}),
        baseline_columns=("AGE",),
    )

    assert diagnostic is not None
    assert diagnostic.severity_metric == pytest.approx(0.3949683532)
    assert diagnostic.supporting_metrics["n_allocation_groups"] == 4.0


def test_cluster_diagnostic_residualizes_every_randomized_sequence() -> None:
    groups = tuple(group for group in ("S0", "S1", "S2", "S3") for _ in range(40))
    subject_ids = [f"SUBJ{index:04d}" for index in range(len(groups))]
    sites = [f"SITE{group[1]}_{(index % 40) // 10}" for index, group in enumerate(groups)]
    event = [1 if group in {"S0", "S1"} else 0 for group in groups]

    diagnostic = _cluster_diagnostic(
        subject_frame=pd.DataFrame(
            {
                "USUBJID": subject_ids,
                "SITEID": sites,
                "allocation_group": groups,
            }
        ),
        survival=pd.DataFrame({"USUBJID": subject_ids, "event": event}),
    )

    assert diagnostic is not None
    assert diagnostic.severity_metric == 0.0
    assert diagnostic.supporting_metrics["icc_event_by_site"] == 0.0


def test_censoring_diagnostic_adjusts_for_every_randomized_sequence() -> None:
    groups = tuple(group for group in ("S0", "S1", "S2", "S3") for _ in range(100))
    subject_ids = [f"SUBJ{index:04d}" for index in range(len(groups))]
    base_age = np.tile(np.repeat(np.linspace(-1.0, 1.0, 10), 10), 4)
    group_offset = {"S0": 0.0, "S1": 2.0, "S2": 4.0, "S3": 6.0}
    age = [float(base_age[index] + group_offset[group]) for index, group in enumerate(groups)]
    early = np.tile(np.array([True, True, True, False, False, False, False, False, False, False]), 40)
    early_time = {"S0": 20.0, "S1": 40.0, "S2": 60.0, "S3": 80.0}
    duration = [early_time[group] if early[index] else 100.0 for index, group in enumerate(groups)]

    diagnostic = _censoring_diagnostic(
        subject_frame=pd.DataFrame({"USUBJID": subject_ids, "allocation_group": groups}),
        survival=pd.DataFrame({"USUBJID": subject_ids, "duration": duration, "event": [0] * len(groups)}),
        covariates=pd.DataFrame({"USUBJID": subject_ids, "AGE": age}),
        baseline_columns=("AGE",),
    )

    assert diagnostic is not None
    assert diagnostic.severity_metric == pytest.approx(0.0, abs=1e-10)
    assert diagnostic.supporting_metrics["n_allocation_groups"] == 4.0
    assert diagnostic.supporting_metrics["n_baseline_censoring_covariates"] == 1.0
    assert diagnostic.supporting_metrics["n_candidate_censoring_models"] == 4.0
    assert diagnostic.supporting_metrics["n_estimable_censoring_models"] == 4.0


def test_participant_diagnostic_evidence_has_no_evaluator_or_tier_inputs(tmp_path: Path) -> None:
    public_zip, _ = _write_release_pair(
        tmp_path,
        task_id="TASK000",
        item_id="d1a4_rct_ph_broken_00",
        design_tier="D1",
        assumption_tier="A4",
        context_tier="C1",
        design_family="randomized_trial",
        credit_eligible_route_families=("global_cox_ph",),
        primary_effect_scale="log_hr",
    )

    evidence = build_participant_diagnostic_evidence_v1(public_zip=public_zip)

    assert len(evidence) == 1
    assert evidence[0].task_id == "TASK000"
    assert not {
        "item_id",
        "design_tier",
        "assumption_tier",
        "context_tier",
        "credit_eligible_route_families",
    }.intersection(type(evidence[0]).model_fields)


def test_diagnostic_proof_accepts_d4_design_adjustment_without_ph_requirement(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        task_id="TASK001",
        item_id="d4a2_cluster_parallel_censoring_stress_01",
        design_tier="D4",
        assumption_tier="A3",
        context_tier="C1",
        design_family="cluster_parallel_randomized",
        credit_eligible_route_families=("risk_difference",),
        primary_effect_scale="risk_difference_tau",
    )

    report = validate_trialeval_diagnostic_proof_surface_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "pass"
    assert report.rows[0].evidence_class == "design_adjustment_evidence"
    assert report.rows[0].ph_diagnostic_required is False
    assert "cluster_structure_public" in report.rows[0].satisfied_diagnostic_keys
    assert not report.rows[0].missing_diagnostic_keys


def test_diagnostic_proof_does_not_apply_unadjusted_ph_gate_to_stepped_wedge(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        task_id="TASK011",
        item_id="d4a2_stepped_wedge_stressed_01",
        design_tier="D4",
        assumption_tier="A2",
        context_tier="C3",
        design_family="stepped_wedge_cluster_rollout",
        credit_eligible_route_families=("risk_difference",),
        event_pattern="crossed",
        raw_surface=True,
    )

    report = validate_trialeval_diagnostic_proof_surface_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "pass"
    assert report.rows[0].evidence_class == "design_adjustment_evidence"
    assert report.rows[0].ph_diagnostic_required is False
    assert "ph_holding_contract_has_material_public_time_variation" not in report.rows[0].findings


def test_diagnostic_proof_accepts_rmst_as_distinct_ph_compatible_estimand(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        task_id="TASK002",
        item_id="d1a2_rct_ph_detectable_01",
        design_tier="D1",
        assumption_tier="A2",
        context_tier="C1",
        design_family="randomized_trial",
        credit_eligible_route_families=("rmst_contrast",),
        primary_effect_scale="rmst_difference_tau",
        event_pattern="parallel",
    )

    report = validate_trialeval_diagnostic_proof_surface_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "pass"
    assert report.rows[0].evidence_class == "ph_compatible_diagnostics"
    assert not report.rows[0].findings


def test_diagnostic_proof_does_not_infer_model_form_from_item_id(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        task_id="TASK004",
        item_id="d3a2_dag_full_01",
        design_tier="D3",
        assumption_tier="A2",
        context_tier="C1",
        design_family="randomized_trial",
        credit_eligible_route_families=("risk_difference",),
        primary_effect_scale="risk_difference_tau",
    )

    report = validate_trialeval_diagnostic_proof_surface_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "pass"
    assert report.rows[0].evidence_class == "ph_compatible_diagnostics"
    assert not report.rows[0].findings


def test_participant_diagnostics_reject_missing_covariate_contract(tmp_path: Path) -> None:
    public_zip, _ = _write_release_pair(
        tmp_path,
        task_id="TASK_MISSING_COVARIATES",
        item_id="missing_covariate_contract",
        design_tier="D1",
        assumption_tier="A1",
        context_tier="C1",
        design_family="randomized_trial",
        credit_eligible_route_families=("global_cox_ph",),
        include_covariate_surface=False,
    )

    with pytest.raises(ValueError, match="lacks analysis covariates"):
        build_participant_diagnostic_evidence_v1(public_zip=public_zip)


def test_participant_diagnostic_worker_uses_safe_json_transport(tmp_path: Path) -> None:
    public_zip, _ = _write_release_pair(
        tmp_path,
        task_id="TASK_WORKER",
        item_id="worker_transport",
        design_tier="D1",
        assumption_tier="A1",
        context_tier="C1",
        design_family="randomized_trial",
        credit_eligible_route_families=("global_cox_ph",),
    )

    batches = _run_participant_diagnostic_workers_v1(
        public_zip=public_zip,
        batches=(("TASK_WORKER",),),
    )

    assert len(batches) == 1
    assert tuple(row.task_id for row in batches[0]) == ("TASK_WORKER",)


def test_diagnostic_proof_cli_writes_visual_source_artifacts(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        task_id="TASK003",
        item_id="d1a2_rct_ph_detectable_01",
        design_tier="D1",
        assumption_tier="A2",
        context_tier="C1",
        design_family="randomized_trial",
        credit_eligible_route_families=("global_cox_ph",),
        primary_effect_scale="log_hr",
    )
    out_dir = tmp_path / "out"

    rc = main(
        [
            "--public-zip",
            public_zip.as_posix(),
            "--evaluator-zip",
            evaluator_zip.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
        ]
    )

    assert rc == 0
    assert (out_dir / "trialeval_diagnostic_proof_rows.csv").is_file()
    assert (out_dir / "trialeval_diagnostic_coverage_by_tier.csv").is_file()
    assert (out_dir / "trialeval_assumption_diagnostic_coverage.csv").is_file()
    replay_rows = pd.read_csv(out_dir / "trialeval_assumption_replay_rows.csv")
    assert set(replay_rows["assumption_id"]) == {"proportional_hazards"}
    assert replay_rows["numeric_agreement"].all()
    assert (out_dir / "trialeval_ph_diagnostic_panel_source.csv").is_file()
    assert "<svg" in (out_dir / "trialeval_ph_diagnostic_panel.svg").read_text(encoding="utf-8")


def test_diagnostic_proof_reconstructs_raw_participant_surface_without_evaluator_tables(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        task_id="TASK005",
        item_id="d1a2_rct_ph_detectable_05",
        design_tier="D1",
        assumption_tier="A2",
        context_tier="C3",
        design_family="randomized_trial",
        credit_eligible_route_families=("global_cox_ph",),
        raw_surface=True,
    )

    report = validate_trialeval_diagnostic_proof_surface_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "pass"
    assert report.rows[0].n_subjects == 60
    assert all("public_reconstruction" not in path for path in report.rows[0].public_input_paths)


def test_survival_diagnostic_binds_multi_endpoint_adtte_to_declared_primary() -> None:
    adsl, primary = _analysis_frames("parallel")
    primary["PARAMCD"] = "death"
    secondary = primary.copy()
    secondary["PARAMCD"] = "deterioration"
    secondary["AVAL"] = list(reversed(secondary["AVAL"].tolist()))
    adtte = pd.concat([primary, secondary], ignore_index=True)

    frame = _survival_frame(
        adsl,
        adtte,
        primary_paramcd="death",
        control_arm_id="control",
        treated_arm_id="treated",
    )

    assert len(frame) == len(adsl)


def test_survival_diagnostic_does_not_invent_static_treatment_for_stepped_wedge() -> None:
    adsl, adtte = _analysis_frames("parallel")
    adsl["ARMCD"] = np.where(np.arange(len(adsl)) % 2, "S1", "S0")
    adsl["TRTA"] = adsl["ARMCD"]

    frame = _survival_frame(
        adsl,
        adtte,
        primary_paramcd=None,
        control_arm_id="control",
        treated_arm_id="treated",
        static_binary_treatment=False,
    )

    assert len(frame) == len(adsl)
    assert "treatment" not in frame.columns


def test_treatment_indicator_uses_declared_ids_not_arm_name_heuristics() -> None:
    adsl = pd.DataFrame(
        {
            "ARMCD": ["zeta", "alpha", "third"],
            "TRTA": ["zeta", "alpha", "third"],
        }
    )

    indicator = _treatment_indicator(adsl, control_arm_id="zeta", treated_arm_id="alpha")

    assert indicator.iloc[:2].tolist() == [0.0, 1.0]
    assert pd.isna(indicator.iloc[2])


def test_treatment_indicator_rejects_absent_declared_contrast() -> None:
    adsl = pd.DataFrame({"ARMCD": ["control", "active"]})

    with pytest.raises(ValueError, match="declared primary contrast identifiers"):
        _treatment_indicator(adsl, control_arm_id="placebo", treated_arm_id="active")


def test_treatment_indicator_rejects_disagreeing_arm_columns() -> None:
    adsl = pd.DataFrame(
        {
            "ARMCD": ["control", "active"],
            "TRTA": ["active", "control"],
        }
    )

    with pytest.raises(ValueError, match="arm columns disagree"):
        _treatment_indicator(adsl, control_arm_id="control", treated_arm_id="active")


def test_survival_diagnostic_rejects_ambiguous_multi_endpoint_adtte() -> None:
    adsl, primary = _analysis_frames("parallel")
    primary["PARAMCD"] = "death"
    secondary = primary.assign(PARAMCD="deterioration")

    with pytest.raises(ValueError, match="requires task.primary_paramcd"):
        _survival_frame(
            adsl,
            pd.concat([primary, secondary], ignore_index=True),
            primary_paramcd=None,
            control_arm_id="control",
            treated_arm_id="treated",
        )


def test_diagnostic_proof_rejects_broken_ph_without_material_public_signal(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        task_id="TASK010",
        item_id="d1a4_rct_ph_broken_01",
        design_tier="D1",
        assumption_tier="A4",
        context_tier="C1",
        design_family="randomized_trial",
        credit_eligible_route_families=("rmst_contrast",),
        primary_effect_scale="rmst_difference_tau",
        event_pattern="parallel",
        expected_ph_status="broken",
    )

    report = validate_trialeval_diagnostic_proof_surface_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "fail"
    row = report.rows[0]
    assert "participant_assumption_band_disagrees:proportional_hazards" in row.findings
    replay = next(value for value in row.assumption_replays if value.assumption_id == "proportional_hazards")
    assert replay.numeric_agreement
    assert replay.evaluator_band == "broken"
    assert replay.participant_band != replay.evaluator_band
    assert replay.classification_agreement is False


def test_reference_analysis_uses_public_scale_options_without_hidden_identification_label() -> None:
    decision = infer_reference_analysis(
        task_id="TASK006",
        protocol={"design_family": "randomized_trial"},
        task={
            "primary_endpoint_term": "All-cause death",
            "primary_effect_scale_options": ["log_hr", "risk_difference_tau", "rmst_difference_tau"],
        },
        public_members=set(),
        ph_method_change_threshold_crossed=False,
    )

    assert decision.evidence_class == "ph_compatible_diagnostics"
    assert decision.candidate_route_families == ("global_cox_ph", "risk_difference", "rmst_contrast")


def test_fixed_primary_retains_prespecified_effect_family() -> None:
    decision = infer_reference_analysis(
        task_id="TASK007",
        protocol={"design_family": "randomized_trial"},
        task={
            "primary_endpoint_term": "All-cause death",
            "primary_effect_scale": "risk_difference_tau",
            "primary_effect_scale_options": ["risk_difference_tau"],
        },
        public_members=set(),
        ph_method_change_threshold_crossed=False,
    )

    assert decision.evidence_class == "ph_compatible_diagnostics"
    assert decision.candidate_route_families == ("risk_difference",)


def test_non_ph_diagnostics_do_not_invent_an_undeclared_time_varying_model() -> None:
    decision = infer_reference_analysis(
        task_id="TASK012",
        protocol={"design_family": "randomized_trial"},
        task={
            "primary_endpoint_term": "All-cause death",
            "primary_effect_scale_options": ["log_hr", "risk_difference_tau", "rmst_difference_tau"],
        },
        public_members=set(),
        ph_method_change_threshold_crossed=True,
    )

    assert decision.evidence_class == "non_ph_diagnostics"
    assert decision.candidate_route_families == ("risk_difference", "rmst_contrast")


def test_c2_ascertainment_model_is_insufficient_without_validation_endpoint_contract(
    tmp_path: Path,
) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        task_id="TASK008",
        item_id="d3a4_detection_bias_01",
        design_tier="D3",
        assumption_tier="A4",
        context_tier="C2",
        design_family="randomized_trial",
        credit_eligible_route_families=("risk_difference",),
        primary_effect_scale="risk_difference_tau",
        include_ascertainment_model=True,
        include_endpoint_definition=False,
    )

    report = validate_trialeval_diagnostic_proof_surface_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "fail"
    assert report.rows[0].missing_diagnostic_keys == ("endpoint_ascertainment_public",)


def _write_release_pair(
    tmp_path: Path,
    *,
    task_id: str,
    item_id: str,
    design_tier: str,
    assumption_tier: str,
    context_tier: str,
    design_family: str,
    credit_eligible_route_families: tuple[str, ...],
    event_pattern: str = "parallel",
    include_covariate_surface: bool = True,
    raw_surface: bool = False,
    primary_effect_scale: str | None = None,
    include_ascertainment_model: bool = False,
    include_endpoint_definition: bool = True,
    expected_ph_status: str = "holds",
) -> tuple[Path, Path]:
    public_zip = tmp_path / f"{task_id}_public.zip"
    evaluator_zip = tmp_path / f"{task_id}_evaluator.zip"
    adsl, adtte = _analysis_frames(event_pattern)
    if design_family in {"cluster_parallel_randomized", "stepped_wedge_cluster_rollout"}:
        adsl["SITEID"] = [f"SITE{index // 6:02d}" for index in range(len(adsl))]
    if design_family == "stepped_wedge_cluster_rollout":
        adsl["INTERVENTION_START_DY"] = [float((index // 6) % 3) * 30.0 for index in range(len(adsl))]
        adsl["RFSTDTC"] = "2024-01-01"
    diagnostic_frame = _survival_frame(
        adsl,
        adtte,
        primary_paramcd=None,
        control_arm_id="control",
        treated_arm_id="treated",
    )
    if design_family in {"cluster_parallel_randomized", "stepped_wedge_cluster_rollout"}:
        subject_frame = pd.DataFrame(
            {
                "USUBJID": adsl["USUBJID"].astype("string"),
                "treatment": (adsl["ARMCD"] == "treated").astype(int),
                "allocation_group": adsl["ARMCD"].astype("string"),
                "SITEID": adsl["SITEID"].astype("string"),
            }
        )
        if "INTERVENTION_START_DY" in adsl.columns:
            subject_frame["INTERVENTION_START_DY"] = adsl["INTERVENTION_START_DY"]
            subject_frame["RFSTDTC"] = adsl["RFSTDTC"].astype("string")
        diagnostics = compute_participant_assumption_diagnostics_v1(
            subject_frame=subject_frame,
            survival=diagnostic_frame,
            covariates=pd.DataFrame({"USUBJID": adsl["USUBJID"].astype("string"), "BMI": [27.0] * len(adsl)}),
            reference_covariates=pd.DataFrame(
                {"REFERENCE_ID": [f"R{index:04d}" for index in range(len(adsl))], "BMI": [27.0] * len(adsl)}
            ),
            operational=None,
            ascertainment=None,
            protocol={},
        )
        selected = next(diagnostic for diagnostic in diagnostics if diagnostic.assumption_id == "cluster_structure")
        evidence_assumption_id = selected.assumption_id
        evidence_status = "holds"
        evidence_band = "holds"
        evidence_diagnosability = "design_declared"
    else:
        _, _, _, observed_severity, _, _ = _ph_diagnostics(diagnostic_frame)
        ph_policy = load_diagnostic_registry_v1().diagnostic_keys["proportional_hazards"]
        if ph_policy.severity_thresholds is None:  # pragma: no cover - registry contract
            raise ValueError("PH diagnostic policy must define severity thresholds.")
        evidence_assumption_id = "proportional_hazards"
        evidence_metric_name = ph_policy.severity_thresholds.metric_name
        evidence_severity = float(observed_severity or 0.0)
        evidence_thresholds = (
            ph_policy.severity_thresholds.stressed,
            ph_policy.severity_thresholds.fragile,
            ph_policy.severity_thresholds.broken,
        )
        evidence_status = str(expected_ph_status)
        evidence_band = "holds"
        evidence_diagnosability = "partially_diagnosable"
        if evidence_status == "broken":
            evidence_band = "broken"
        elif evidence_status == "stressed":
            evidence_band = "mild" if assumption_tier == "A2" else "fragile"
    assumption_evidence_manifest = _assumption_evidence_manifest_payload(
        item_id=item_id,
        context_tier=context_tier,
        assumption_id=evidence_assumption_id,
        status=evidence_status,
        band=evidence_band,
        diagnosability=evidence_diagnosability,
        severity=(None if evidence_diagnosability == "design_declared" else evidence_severity),
        metric_name=(None if evidence_diagnosability == "design_declared" else evidence_metric_name),
        thresholds=(None if evidence_diagnosability == "design_declared" else evidence_thresholds),
    )
    context_factors = {
        "C1": ("analysis_ready", "locked_sap"),
        "C2": ("analysis_ready", "protocol_only"),
        "C3": ("raw_domains", "locked_sap"),
        "C4": ("raw_domains", "protocol_only"),
        "C5": ("raw_domains_declared_defect", "protocol_only"),
    }
    data_preparation, analysis_specification = context_factors[context_tier]
    design_subtype = {
        "D1": "individual_randomized",
        "D2": "pragmatic",
        "D3": "covariate_structure",
        "D4": (
            "cluster_parallel"
            if design_family == "cluster_parallel_randomized"
            else ("stepped_wedge" if design_family == "stepped_wedge_cluster_rollout" else "group_sequential")
        ),
    }[design_tier]
    effect_scale_by_family = {
        "global_cox_ph": "log_hr",
        "risk_difference": "risk_difference_tau",
        "rmst_contrast": "rmst_difference_tau",
    }
    scoring_key = {
        "schema_id": "trialagentbench.scoring_key/v1",
        "release_id": "diagnostic-proof-fixture",
        "item_id": task_id,
        "question_id": item_id,
        "context_tier": context_tier,
        "credit_eligible_routes": [
            {
                "route_id": f"route-{index}",
                "signature": {
                    "analysis_population_id": "itt",
                    "estimand_id": f"estimand-{family}",
                    "intercurrent_event_strategy_ids": ["treatment_policy"],
                    "treatment_id": "active",
                    "comparator_id": "control",
                    "endpoint_id": "time_to_event",
                    "effect_scale": (
                        primary_effect_scale
                        if index == 1 and primary_effect_scale is not None
                        else effect_scale_by_family[family]
                    ),
                    "analysis_method_id": f"fixture:{family}:wald",
                },
                "method": {
                    "analysis_method_id": f"fixture:{family}:wald",
                    "estimator_family": family,
                    "result_kind": "numeric_point",
                    "uncertainty_method": "wald",
                    "design_modifiers": [],
                },
                "required_identification_assumptions": ["randomization"],
                "target": {
                    "kind": "numeric_point",
                    "value": 0.0,
                    "result_unit": "fixture_unit",
                    "acceptance_envelope": {
                        "schema_id": "trialagentbench.numerical_acceptance_envelope/v1",
                        "reporting_decimal_places": 3,
                        "independent_max_abs_difference": 0.001,
                        "public_verification_id": "fixture-replay",
                        "independent_verification_ids": ["fixture-independent-replay"],
                    },
                    "require_confidence_interval": False,
                },
            }
            for index, family in enumerate(credit_eligible_route_families, start=1)
        ],
    }
    scoring_key_body = (json.dumps(scoring_key, sort_keys=True, separators=(",", ":")) + "\n").encode()
    scoring_key_manifest = {
        "schema_id": "trialagentbench.scoring_key_manifest/v1",
        "release_id": "diagnostic-proof-fixture",
        "specification_sha256": "a" * 64,
        "scoring_keys_sha256": hashlib.sha256(scoring_key_body).hexdigest(),
        "item_ids": [task_id],
    }
    with ZipFile(evaluator_zip, "w") as zf:
        zf.writestr(
            "grader/item_index.json",
            json.dumps(
                {
                    "schema_id": "trialagentbench.trialeval_item_index/v1",
                    "version": 1,
                    "checksum": "fixture",
                    "entries": [
                        {
                            "task_id": task_id,
                            "item_id": item_id,
                            "variant_id": f"{item_id}__{context_tier}",
                            "factors": {
                                "evaluation_series_id": item_id,
                                "design_archetype": design_tier,
                                "design_subtype": design_subtype,
                                "assumption_regime": assumption_tier,
                                "context_configuration": context_tier,
                                "data_preparation": data_preparation,
                                "analysis_specification": analysis_specification,
                                "procedure_assistance": "output_contract_only",
                                "response_interface": "structured",
                            },
                            "reconstruction_row_count": 0,
                        }
                    ],
                }
            ),
        )
        zf.writestr("grader/scoring_keys.jsonl", scoring_key_body)
        zf.writestr("grader/scoring_key_manifest.json", json.dumps(scoring_key_manifest))
        zf.writestr(
            "grader/domains/assumption_evidence.jsonl",
            json.dumps(
                {
                    "schema_id": "trialagentbench.trial_benchmark.grader_domain_row/v1",
                    "domain": "assumption_evidence",
                    "task_id": task_id,
                    "payload": {
                        "schema_id": "trialagentbench.trial_benchmark.assumption_evidence_manifest_row/v1",
                        "manifest": assumption_evidence_manifest,
                    },
                }
            )
            + "\n",
        )
    with ZipFile(public_zip, "w") as zf:
        zf.writestr("items/TASK_UNUSED/README.md", "fixture")
        scale_by_family = {
            "global_cox_ph": "log_hr",
            "risk_difference": "risk_difference_tau",
            "rmst_contrast": "rmst_difference_tau",
            "standardized_risk": "standardized_risk_difference_tau_reference",
        }
        scale_options = tuple(
            dict.fromkeys(
                [
                    *([primary_effect_scale] if primary_effect_scale is not None else []),
                    *(scale_by_family[family] for family in credit_eligible_route_families),
                ]
            )
        )
        task_payload = {
            "schema_id": "trial_analysis_task_v1",
            "primary_endpoint_term": "All-cause death",
            "primary_paramcd": "primary",
            "primary_control_arm_id": "control",
            "primary_treated_arm_id": "treated",
            "primary_effect_scale_options": list(scale_options),
        }
        if primary_effect_scale is not None:
            task_payload["primary_effect_scale"] = primary_effect_scale
        zf.writestr(f"items/{task_id}/task.json", json.dumps(task_payload))
        analysis_effect_scale = str(primary_effect_scale or scale_options[0])
        standardized_analysis = analysis_effect_scale == "standardized_risk_difference_tau_reference"
        primary_analysis: dict[str, object] = {
            "effect_scale": analysis_effect_scale,
            "method_id": (
                "observed:cox_linear_standardized_risk_tau_reference" if standardized_analysis else "fixture:primary"
            ),
            "estimator_family": ("standardized_cox_g_computation" if standardized_analysis else "km"),
            "implementation": "Apply the prespecified primary analysis.",
            "uncertainty_method": "wald",
            "required_method_modifiers": (["reference_standardization"] if standardized_analysis else []),
            "baseline_covariate_strategy": ("all_released" if standardized_analysis else "unadjusted"),
        }
        if standardized_analysis:
            primary_analysis["treatment_effect_modifier"] = "BMI"
        analysis_plan: dict[str, object] = {
            "schema_id": "trial_analysis_plan_contract_v1",
            "task_id": task_id,
            "item_id": task_id,
            "primary_estimand_id": "itt",
            "primary_endpoint_id": "time_to_event",
            "primary_analysis": primary_analysis,
            "diagnostic_requirements": [
                {
                    "assumption_id": "proportional_hazards",
                    "diagnostic_methods": ["scaled Schoenfeld residuals"],
                }
            ],
            "sensitivity_analyses": [],
            "familywise_alpha": 0.05,
            "multiplicity_strategy": "none",
            "lane_rules": [
                {
                    "lane_id": "primary_analysis.response.v1",
                    "estimand_id": "itt",
                    "endpoint_id": "time_to_event",
                    "effect_scale": analysis_effect_scale,
                    "role": "primary",
                    "multiplicity_family_id": "family_primary_confirmatory",
                    "alpha": 0.05,
                    "mandatory": True,
                }
            ],
        }
        analysis_plan["checksum"] = hashlib.sha256(
            json.dumps(
                analysis_plan,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        zf.writestr(f"items/{task_id}/analysis_plan.json", json.dumps(analysis_plan))
        zf.writestr(f"items/{task_id}/study_brief.md", "Estimate the treatment effect from public evidence.")
        if include_endpoint_definition:
            zf.writestr(
                f"items/{task_id}/endpoint_definition.json",
                json.dumps({"schema_id": "endpoint_definition_v1", "primary_endpoint": "death"}),
            )
        if include_ascertainment_model:
            zf.writestr(
                f"items/{task_id}/ascertainment_model.json",
                json.dumps(
                    {
                        "schema_id": "trialagentbench.trialeval.endpoint_validation_design/v1",
                        "endpoint_id": "primary",
                        "validation_sampling_fraction": 0.6,
                        "prognostic_stratum_variable": "AGE",
                        "prognostic_stratum_cutpoint_rule": "pooled_median",
                        "unsupported_validation_strata": ["higher_risk"],
                        "released_fields": ["VALSTRAT", "OBSEVNT", "VALIDFL", "ADJEVNT"],
                        "point_estimation_method_id": "observed:validated_endpoint_joint_likelihood",
                        "unsupported_stratum_method_id": "observed:validated_endpoint_bounded_deviation",
                    }
                ),
            )
        zf.writestr(
            f"items/{task_id}/intercurrent_event_strategy.json",
            json.dumps({"schema_id": "intercurrent_event_strategy_v1", "strategy": "treatment_policy"}),
        )
        zf.writestr(
            f"items/{task_id}/protocol_summary.json",
            json.dumps(
                {
                    "schema_id": "trial_analysis_protocol_summary_v1",
                    "design_family": design_family,
                    "primary_population": "ITT",
                    "primary_endpoint_term": "All-cause death",
                    "followup_horizon_dy": 180,
                    "arms": [
                        {"arm_id": "control", "label": "Control"},
                        {"arm_id": "treated", "label": "Treated"},
                    ],
                }
            ),
        )
        if raw_surface:
            zf.writestr(
                f"items/{task_id}/reconstruction_task.json",
                json.dumps(
                    {
                        "allowed_sources": [
                            "data/raw/randomization.parquet",
                            "data/raw/disposition.parquet",
                            "data/raw/endpoint_adjudication.parquet",
                        ]
                    }
                ),
            )
            randomization = adsl.copy()
            randomization["ARM"] = randomization["ARMCD"]
            randomization["RFSTDTC"] = "2025-01-01"
            disposition = pd.DataFrame(
                {
                    "USUBJID": adsl["USUBJID"],
                    "LAST_CONTACT_DY": adtte["AVAL"],
                }
            )
            event_rows = adtte.loc[adtte["CNSR"] == 0, ["USUBJID", "AVAL"]].copy()
            event_rows["ENDPOINT_TERM"] = "All-cause death"
            event_rows["CLINICAL_CERTAINTY"] = "definite"
            event_rows["SOURCE_CONSISTENCY"] = "consistent"
            event_rows["EXCLUSIONARY_REVIEW_FINDING"] = "none"
            event_rows["EVENT_WINDOW_START_DY"] = event_rows["AVAL"] - 0.5
            event_rows["EVENT_WINDOW_END_DY"] = event_rows["AVAL"] + 0.5
            zf.writestr(
                f"items/{task_id}/data/raw/randomization.parquet",
                _to_parquet_bytes(randomization),
            )
            zf.writestr(
                f"items/{task_id}/data/raw/disposition.parquet",
                _to_parquet_bytes(disposition),
            )
            zf.writestr(
                f"items/{task_id}/data/raw/endpoint_adjudication.parquet",
                _to_parquet_bytes(event_rows.drop(columns="AVAL")),
            )
        else:
            zf.writestr(f"items/{task_id}/data/ADSL.parquet", _to_parquet_bytes(adsl))
            zf.writestr(f"items/{task_id}/data/ADTTE.parquet", _to_parquet_bytes(adtte))
        if include_covariate_surface:
            zf.writestr(
                f"items/{task_id}/data/analysis_frame_covariates.parquet",
                _to_parquet_bytes(pd.DataFrame({"USUBJID": adsl["USUBJID"], "BMI": [27.0] * len(adsl)})),
            )
            zf.writestr(
                f"items/{task_id}/data/reference_population_covariates.parquet",
                _to_parquet_bytes(
                    pd.DataFrame(
                        {
                            "REFERENCE_ID": [f"R{index:04d}" for index in range(len(adsl))],
                            "BMI": [27.0] * len(adsl),
                        }
                    )
                ),
            )
        if design_family in {"cluster_parallel_randomized", "stepped_wedge_cluster_rollout"}:
            zf.writestr(
                f"items/{task_id}/data/site_summary.parquet",
                _to_parquet_bytes(pd.DataFrame({"SITEID": ["SITE001", "SITE002"], "n": [30, 30]})),
            )
    return public_zip, evaluator_zip


def _assumption_evidence_manifest_payload(
    *,
    item_id: str,
    context_tier: str,
    assumption_id: str = "proportional_hazards",
    status: str = "holds",
    band: str = "holds",
    diagnosability: str = "partially_diagnosable",
    severity: float | None = 0.0,
    metric_name: str | None = "simultaneous_lower_abs_time_varying_log_hazard_range",
    thresholds: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    if thresholds is None and diagnosability != "design_declared":
        policy = load_diagnostic_registry_v1().diagnostic_keys[assumption_id].severity_thresholds
        if policy is None:
            raise ValueError(f"Empirical fixture assumption lacks severity policy: {assumption_id!r}.")
        thresholds = (policy.stressed, policy.fragile, policy.broken)
    record: dict[str, object] = {
        "assumption_id": assumption_id,
        "expected_status": status,
        "computed_status": status,
        "expected_band": band,
        "computed_band": band,
        "diagnosability": diagnosability,
        "decision_metric_names": {},
        "supporting_metrics": {},
        "metric_units": {},
        "metric_public_evidence_basis": {},
        "notes": [],
    }
    if diagnosability == "design_declared":
        record["factual_public_evidence_basis"] = ("protocol_summary.json",)
    if diagnosability != "design_declared":
        if severity is None or metric_name is None or thresholds is None:
            raise ValueError("Empirical assumption evidence requires a metric and ordered thresholds.")
        record.update(
            {
                "severity_metric": severity,
                "severity_metric_name": metric_name,
                "threshold_stressed": thresholds[0],
                "threshold_fragile": thresholds[1],
                "threshold_broken": thresholds[2],
                "decision_metric_names": {
                    "stressed": metric_name,
                    "fragile": metric_name,
                    "broken": metric_name,
                },
                "metric_units": {metric_name: "log_hazard_ratio"},
                "metric_public_evidence_basis": {
                    metric_name: ("data/ADSL.parquet", "data/ADTTE.parquet"),
                },
            }
        )
    payload: dict[str, object] = {
        "version": "v1",
        "schema_id": "trial_benchmark_assumption_evidence_manifest_v1",
        "item_id": item_id,
        "base_case_id": item_id.rsplit("_", 1)[0],
        "canonical_item_id": item_id,
        "variant_id": f"{item_id}__{context_tier}",
        "context_tier": context_tier,
        "replicate_index": 0,
        "records": [record],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    payload["checksum"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _analysis_frames(event_pattern: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_ids = [f"S{i:03d}" for i in range(60)]
    arms = ["control"] * 30 + ["treated"] * 30
    adsl = pd.DataFrame({"USUBJID": subject_ids, "ARMCD": arms})
    if event_pattern == "parallel":
        durations = [30 + (i % 15) for i in range(60)]
        events = [1 if i % 3 != 0 else 0 for i in range(60)]
    else:
        durations = [10 + (i % 8) if i < 30 else 90 + (i % 8) for i in range(60)]
        events = [1 if i % 4 != 0 else 0 for i in range(60)]
    adtte = pd.DataFrame({"USUBJID": subject_ids, "AVAL": durations, "CNSR": [0 if event else 1 for event in events]})
    return adsl, adtte


def _to_parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()
