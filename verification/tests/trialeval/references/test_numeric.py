"""Tests for numeric TrialEval public-reference replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from trialagentbench_validation.contracts.scoring.route_reference_inputs import (
    RouteReferenceInputRecordV1,
)
from trialagentbench_validation.contracts.scoring.route_references import (
    RouteReferenceRecordV1,
    float64_equivalence_policy_v1,
)
from trialagentbench_validation.trialeval.references.calculators import (
    _cached_km_family_result_v1,
    _fit_censoring_model,
    _ipcw_analysis_frame,
    _KmFamilyResult,
    numeric_replay_family_id_v1,
)
from trialagentbench_validation.trialeval.references.numeric import (
    _partition_scoreable_inputs_v1,
    recompute_trialeval_public_numeric_reference_v1,
    write_public_evidence_numeric_reference_artifacts_v1,
)


def test_public_ipcw_replay_canonicalizes_machine_precision_time_noise() -> None:
    adsl = pd.DataFrame(
        {
            "USUBJID": ["C1", "T1"],
            "TRTA": ["control", "treated"],
            "AGE": [60.0, 61.0],
        }
    )
    adtte = pd.DataFrame(
        {
            "USUBJID": ["C1", "T1"],
            "PARAMCD": ["death", "death"],
            "AVAL": [14.5, 14.5 + 3e-14],
            "CNSR": [0, 1],
        }
    )

    frame = _ipcw_analysis_frame(
        adsl=adsl,
        adtte=adtte,
        paramcd="death",
        baseline_covariate_columns=("AGE",),
    )

    assert frame["AVAL"].tolist() == [14.5, 14.5]


def test_public_ipcw_replay_canonicalizes_subject_order() -> None:
    adsl = pd.DataFrame(
        {
            "USUBJID": ["T1", "C1"],
            "TRTA": ["treated", "control"],
            "AGE": [61.0, 60.0],
        }
    )
    adtte = pd.DataFrame(
        {
            "USUBJID": ["T1", "C1"],
            "PARAMCD": ["death", "death"],
            "AVAL": [14.5, 12.0],
            "CNSR": [1, 0],
        }
    )

    frame = _ipcw_analysis_frame(
        adsl=adsl,
        adtte=adtte,
        paramcd="death",
        baseline_covariate_columns=("AGE",),
    )

    assert frame["USUBJID"].tolist() == ["C1", "T1"]


def test_public_ipcw_excludes_common_administrative_close_from_censoring_fit() -> None:
    merged = pd.DataFrame(
        {
            "USUBJID": ["C1", "C2", "T1", "T2"],
            "TRTA": ["control", "control", "treated", "treated"],
            "AVAL": [280.0, 280.0, 280.0, 280.0],
            "CNSR": [1, 1, 1, 1],
            "AGE": [30.0, 70.0, 35.0, 75.0],
        }
    )

    model = _fit_censoring_model(
        merged,
        baseline_covariate_columns=("AGE",),
        administrative_horizon=280.0,
    )

    assert model.weights(np.full(4, 200.0)).tolist() == [1.0, 1.0, 1.0, 1.0]


def _validated_endpoint_fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    cells = (
        ("control", "lower_risk", 20),
        ("control", "higher_risk", 40),
        ("treated", "lower_risk", 30),
        ("treated", "higher_risk", 50),
    )
    adsl_rows: list[dict[str, object]] = []
    adtte_rows: list[dict[str, object]] = []
    category_counts_by_cell: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for arm, stratum, event_count in cells:
        validated = np.asarray(
            [1] * event_count + [0] * (80 - event_count), dtype=np.int64
        )
        sensitivity_count = int(0.8 * event_count)
        false_positive_count = int(0.1 * (80 - event_count))
        observed = np.asarray(
            [1] * sensitivity_count
            + [0] * (event_count - sensitivity_count)
            + [1] * false_positive_count
            + [0] * (80 - event_count - false_positive_count),
            dtype=np.int64,
        )
        categories, category_counts = np.unique(
            np.stack((observed, validated), axis=1),
            axis=0,
            return_counts=True,
        )
        category_counts_by_cell[(arm, stratum)] = (categories, category_counts)
        for index, (validated_event, observed_event) in enumerate(
            zip(validated, observed, strict=True)
        ):
            subject_id = f"{arm[:1].upper()}{stratum[:1].upper()}{index:03d}"
            adsl_rows.append({"USUBJID": subject_id, "TRTA": arm})
            adtte_rows.append(
                {
                    "USUBJID": subject_id,
                    "PARAMCD": "death",
                    "AVAL": 5.0 if observed_event else 20.0,
                    "CNSR": 1 - int(observed_event),
                    "VALSTRAT": stratum,
                    "OBSEVNT": int(observed_event),
                    "VALIDFL": 1,
                    "ADJEVNT": int(validated_event),
                }
            )
    rng = np.random.RandomState(2606)
    bootstrap = np.empty(1_000, dtype=np.float64)
    bootstrap_group_order = (
        category_counts_by_cell[("control", "higher_risk")],
        category_counts_by_cell[("control", "lower_risk")],
        category_counts_by_cell[("treated", "higher_risk")],
        category_counts_by_cell[("treated", "lower_risk")],
    )
    for replicate_index in range(1_000):
        means = tuple(
            float(
                np.sum(
                    rng.multinomial(
                        int(category_counts.sum()),
                        category_counts / category_counts.sum(),
                    )
                    * categories[:, 1]
                )
                / category_counts.sum()
            )
            for categories, category_counts in bootstrap_group_order
        )
        bootstrap[replicate_index] = 0.5 * (means[2] + means[3] - means[0] - means[1])
    return (
        pd.DataFrame.from_records(adsl_rows),
        pd.DataFrame.from_records(adtte_rows),
        float(bootstrap.std(ddof=1)),
    )


def _write_public_numeric_fixture(
    tmp_path: Path,
    *,
    reference_value: float = 5.0,
    method_id: str = "observed:km_rmst_tau",
    effect_scale: str = "rmst_difference_tau",
    include_adsl: bool = True,
    reference_standard_error: float | None = None,
    sensitivity_parameter: float | None = None,
    group_sequential_analysis_look_index: int = 2,
    public_root_prefix: str = "",
) -> tuple[Path, Path]:
    root = tmp_path / "fixture"
    evaluator_root = root / "evaluator"
    public_root = root / "public"
    domains = evaluator_root / "grader" / "domains"
    domains.mkdir(parents=True, exist_ok=True)
    task_id = "TASKKM001"
    item_root = public_root / "items" / task_id
    data_root = item_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    (item_root / "task.json").write_text(
        json.dumps(
            {
                "primary_paramcd": "death",
                "primary_control_arm_id": "control",
                "primary_treated_arm_id": "treated",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    group_sequential_boundaries = (
        2.771807648699362,
        2.298085834720592,
        2.0425913079225952,
    )
    group_sequential_z = group_sequential_boundaries[
        group_sequential_analysis_look_index
    ]
    protocol_payload: dict[str, object] = {"followup_horizon_dy": 20.0}
    if method_id == "observed:group_sequential_adjusted":
        protocol_payload["group_sequential_plan"] = {
            "spending_function_id": "obrien_fleming",
            "monitoring_effect_scale": "risk_difference_tau",
            "looks": [0.5, 0.75, 1.0],
            "two_sided_alpha": 0.05,
            "nominal_two_sided_alpha_by_look": [0.005592435, 0.021552, 0.041091],
            "z_critical_by_look": list(group_sequential_boundaries),
            "analysis_look_index": group_sequential_analysis_look_index,
            "analysis_information_fraction": [0.5, 0.75, 1.0][
                group_sequential_analysis_look_index
            ],
            "analysis_horizon_dy": 20.0,
            "information_basis": "planned_complete_subject_fraction",
            "stopped_early": group_sequential_analysis_look_index < 2,
        }
    (item_root / "protocol_summary.json").write_text(
        json.dumps(protocol_payload, sort_keys=True),
        encoding="utf-8",
    )
    if method_id == "observed:validated_endpoint_joint_likelihood":
        (item_root / "ascertainment_model.json").write_text(
            json.dumps(
                {
                    "schema_id": "trialagentbench.trialeval.endpoint_validation_design/v1",
                    "endpoint_id": "death",
                    "validation_sampling_fraction": 1.0,
                    "prognostic_stratum_variable": "AGE",
                    "prognostic_stratum_cutpoint_rule": "pooled_median",
                    "unsupported_validation_strata": [],
                    "released_fields": ["VALSTRAT", "OBSEVNT", "VALIDFL", "ADJEVNT"],
                    "point_estimation_method_id": "observed:validated_endpoint_joint_likelihood",
                    "unsupported_stratum_method_id": "observed:validated_endpoint_bounded_deviation",
                    "validation_basis": "Internal validation substudy.",
                    "rationale": "Endpoint error is estimated from adjudicated records.",
                    "source_registry_checksum": "0" * 64,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    uses_tau_bounds = method_id in {
        "observed:tau_bounds_bounded_deviation",
        "observed:tau_bounds_worst_case",
    }
    uses_group_sequential = method_id == "observed:group_sequential_adjusted"
    uses_ipcw = (
        method_id
        in {
            "observed:cluster_parallel_participant_weighted_km_ipcw_baseline_cox",
            "observed:coxph_binary_breslow_ipcw_baseline_cox",
            "observed:km_ipcw_baseline_cox",
        }
        or uses_group_sequential
    )
    uses_subject_flags = (
        uses_tau_bounds
        or uses_group_sequential
        or method_id
        in {
            "observed:cluster_parallel_participant_weighted_km_ipcw_baseline_cox",
            "observed:coxph_binary_breslow_ipcw_baseline_cox",
            "observed:km_ipcw_baseline_cox",
        }
    )
    if include_adsl and not uses_subject_flags:
        if method_id == "observed:validated_endpoint_joint_likelihood":
            endpoint_adsl, _, _ = _validated_endpoint_fixture_frames()
            endpoint_adsl.to_parquet(data_root / "ADSL.parquet", index=False)
        else:
            adsl_payload: dict[str, list[str]] = {
                "USUBJID": ["C1", "C2", "T1", "T2"],
                "TRTA": ["control", "control", "treated", "treated"],
            }
            if method_id == "observed:cluster_parallel_participant_weighted_km":
                adsl_payload["SITEID"] = ["S1", "S2", "S3", "S4"]
            pd.DataFrame(adsl_payload).to_parquet(
                data_root / "ADSL.parquet", index=False
            )
    if uses_subject_flags:
        n_ipcw_arm = 100
        ids = (
            [f"C{index:03d}" for index in range(n_ipcw_arm)]
            + [f"T{index:03d}" for index in range(n_ipcw_arm)]
            if uses_ipcw
            else ["C1", "C2", "T1", "T2"]
        )
        n_rows = len(ids)
        rng = np.random.default_rng(451)
        base_n = n_rows // 2
        age = rng.normal(52.0, 10.0, size=base_n)
        bmi = rng.normal(27.0, 4.0, size=base_n)
        sex = np.where(rng.random(base_n) < 0.5, "F", "M")
        race = np.where(rng.random(base_n) < 0.75, "WHITE", "OTHER")
        ethnic = np.where(rng.random(base_n) < 0.15, "HISPANIC", "NOT_HISPANIC")
        pd.DataFrame(
            {
                "USUBJID": ids,
                "TRTA": ["control"] * (n_rows // 2) + ["treated"] * (n_rows // 2),
                "AGE": (
                    np.concatenate([age, age])
                    if uses_ipcw
                    else [50.0, 51.0, 52.0, 53.0]
                ),
                "BMI": (
                    np.concatenate([bmi, bmi])
                    if uses_ipcw
                    else [27.0, 28.0, 29.0, 30.0]
                ),
                "SEX": (
                    np.concatenate([sex, sex]) if uses_ipcw else ["F", "M", "F", "M"]
                ),
                "RACE": np.concatenate([race, race]) if uses_ipcw else ["WHITE"] * 4,
                "ETHNIC": (
                    np.concatenate([ethnic, ethnic])
                    if uses_ipcw
                    else ["NOT_HISPANIC"] * 4
                ),
                **(
                    {"SITEID": [f"S{index:02d}" for index in range(n_rows)]}
                    if method_id
                    == "observed:cluster_parallel_participant_weighted_km_ipcw_baseline_cox"
                    else {}
                ),
            }
        ).to_parquet(data_root / "subject_operational_flags.parquet", index=False)
        if uses_ipcw:
            pd.DataFrame(
                {
                    "REFERENCE_ID": [f"R{index:03d}" for index in range(base_n)],
                    "AGE": age,
                    "BMI": bmi,
                    "SEX": sex,
                    "RACE": race,
                    "ETHNIC": ethnic,
                }
            ).to_parquet(
                data_root / "reference_population_covariates.parquet", index=False
            )
    if uses_ipcw:
        rng = np.random.default_rng(452)
        arm_cnsr_array = rng.binomial(1, 0.3, size=100)
        arm_times_array = rng.uniform(5.0, 20.0, size=100)
        arm_times_array[arm_cnsr_array == 1] = 20.0
        arm_times = arm_times_array.tolist()
        arm_cnsr = arm_cnsr_array.tolist()
        adtte = pd.DataFrame(
            {
                "USUBJID": [f"C{index:03d}" for index in range(100)]
                + [f"T{index:03d}" for index in range(100)],
                "PARAMCD": ["death"] * 200,
                "AVAL": arm_times + arm_times,
                "CNSR": arm_cnsr + arm_cnsr,
            }
        )
    elif uses_tau_bounds:
        adtte = pd.DataFrame(
            {
                "USUBJID": ["C1", "C2", "T1", "T2"],
                "PARAMCD": ["death", "death", "death", "death"],
                "AVAL": [10.0, 20.0, 20.0, 20.0],
                "CNSR": [0, 1, 1, 1],
            }
        )
    elif method_id in {
        "observed:coxph_binary_breslow",
        "observed:coxph_binary_breslow_ipcw_baseline_cox",
    }:
        adtte = pd.DataFrame(
            {
                "USUBJID": ["C1", "C2", "T1", "T2"],
                "PARAMCD": ["death", "death", "death", "death"],
                "AVAL": [10.0, 20.0, 10.0, 20.0],
                "CNSR": [0, 1, 0, 1],
            }
        )
    elif method_id == "observed:validated_endpoint_joint_likelihood":
        _, adtte, _ = _validated_endpoint_fixture_frames()
    else:
        adtte = pd.DataFrame(
            {
                "USUBJID": ["C1", "C2", "T1", "T2"],
                "PARAMCD": ["death", "death", "death", "death"],
                "AVAL": [10.0, 20.0, 20.0, 20.0],
                "CNSR": [0, 1, 1, 1],
            }
        )
    adtte.to_parquet(data_root / "ADTTE.parquet", index=False)
    route_reference_id = f"{task_id}:primary_numeric.v1:max_recoverable:{method_id}"
    group_sequential_base_value = 0.0
    group_sequential_base_standard_error = 0.0739843511191811
    group_sequential_half_width = (
        group_sequential_z * group_sequential_base_standard_error
    )
    if reference_standard_error is None:
        if method_id in {"observed:km", "observed:km_rmst_tau"}:
            reference_standard_error = (
                3.5355339059327378
                if effect_scale == "rmst_difference_tau"
                else 0.3535533905932738
            )
        elif method_id == "observed:coxph_binary_breslow":
            reference_standard_error = 1.4142135623730951
        elif method_id == "observed:validated_endpoint_joint_likelihood":
            _, _, reference_standard_error = _validated_endpoint_fixture_frames()
        elif uses_tau_bounds:
            reference_standard_error = None
        else:
            reference_standard_error = 0.1
    route_reference = {
        "schema_id": "trialagentbench.trialeval.route_reference/v1",
        "task_id": task_id,
        "item_id": "d1a1_unit",
        "lane_id": "primary_numeric.v1",
        "route_reference_id": route_reference_id,
        "variant_role": "required_primary",
        "route_family": (
            "risk_difference"
            if uses_tau_bounds or uses_group_sequential
            else "rmst_contrast"
        ),
        "estimator_method_id": method_id,
        "effect_scale": effect_scale,
        **(
            {"sensitivity_parameter": sensitivity_parameter}
            if sensitivity_parameter is not None
            else {}
        ),
        "answer_shape": "bound" if uses_tau_bounds else "point",
        "value": (
            group_sequential_base_value if uses_group_sequential else reference_value
        ),
        **(
            {"standard_error": reference_standard_error}
            if reference_standard_error is not None
            else {}
        ),
        **(
            {"lower": reference_value, "upper": reference_value}
            if uses_tau_bounds
            else {}
        ),
        **(
            {
                "ci_low": group_sequential_base_value - group_sequential_half_width,
                "ci_high": group_sequential_base_value + group_sequential_half_width,
                "standard_error": group_sequential_base_standard_error,
            }
            if uses_group_sequential
            else {}
        ),
        "public_evidence_basis": [
            f"items/{task_id}/task.json",
            f"items/{task_id}/protocol_summary.json",
            (
                f"items/{task_id}/data/subject_operational_flags.parquet"
                if uses_subject_flags
                else f"items/{task_id}/data/ADSL.parquet"
            ),
            f"items/{task_id}/data/ADTTE.parquet",
            *(
                [f"items/{task_id}/data/reference_population_covariates.parquet"]
                if uses_ipcw
                else []
            ),
        ],
        "identification_class": "point_identified",
        "support_status": "official_supported",
        "support_rationale": "Unit fixture.",
        "numerical_equivalence": float64_equivalence_policy_v1().model_dump(
            mode="json"
        ),
    }
    (domains / "route_references.jsonl").write_text(
        json.dumps(route_reference, sort_keys=True) + "\n", encoding="utf-8"
    )
    if uses_group_sequential:
        method_composition = {
            "schema_id": "trialagentbench.trialeval.method_composition/v1",
            "task_id": task_id,
            "lane_id": "primary_numeric.v1",
            "route_reference_id": route_reference_id,
            "composed_method_id": method_id,
            "base_estimator_method_id": "observed:km_ipcw_baseline_cox",
            "adjustment_id": "obrien_fleming_3look",
            "adjustment_parameters": {
                "adjustment_id": "obrien_fleming_3look",
                "spending_family": "obrien_fleming",
                "information_fractions": [0.5, 0.75, 1.0],
                "total_alpha": 0.05,
                "z_critical_by_look": list(group_sequential_boundaries),
                "z_value": group_sequential_z,
                "look_count": 3,
                "analysis_look_index": group_sequential_analysis_look_index,
            },
            "base_value": group_sequential_base_value,
            "base_standard_error": group_sequential_base_standard_error,
            "adjusted_lower": group_sequential_base_value - group_sequential_half_width,
            "adjusted_upper": group_sequential_base_value + group_sequential_half_width,
        }
        (domains / "method_composition.jsonl").write_text(
            json.dumps(method_composition, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    refs = []
    table_names = (
        (
            "subject_operational_flags.parquet",
            "reference_population_covariates.parquet",
            "ADTTE.parquet",
        )
        if uses_ipcw
        else (
            ("subject_operational_flags.parquet", "ADTTE.parquet")
            if uses_subject_flags
            else ("ADSL.parquet", "ADTTE.parquet")
        )
    )
    for name in table_names:
        path = data_root / name
        rel = f"items/{task_id}/data/{name}"
        if name == "subject_operational_flags.parquet":
            columns = ["USUBJID", "TRTA", "AGE", "BMI", "SEX"]
        elif name == "reference_population_covariates.parquet":
            columns = ["REFERENCE_ID", "AGE", "BMI", "SEX", "RACE", "ETHNIC"]
        elif name == "ADSL.parquet":
            columns = ["USUBJID", "TRTA"]
            if method_id == "observed:cluster_parallel_participant_weighted_km":
                columns = [*columns, "SITEID"]
        else:
            columns = [str(column) for column in pd.read_parquet(path).columns]
        refs.append(
            {
                "rel_path": rel,
                "semantic_role": "public_table",
                "sha256": (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    if path.is_file()
                    else "0" * 64
                ),
                "row_count": len(pd.read_parquet(path)) if path.is_file() else 0,
                "column_names": columns,
            }
        )
    reference_input = {
        "schema_id": "trialagentbench.trialeval.route_reference_input/v1",
        "task_id": task_id,
        "input_bundle_id": f"{task_id}:primary_numeric.v1:{method_id}:scoreable_input:v1",
        "estimator_method_id": method_id,
        "effect_scale": effect_scale,
        "sensitivity_parameter": sensitivity_parameter,
        "lane_ids": ["primary_numeric.v1"],
        "route_reference_ids": [route_reference_id],
        "required_table_refs": refs,
        "source_role": "canonical_analysis",
    }
    (domains / "route_reference_inputs.jsonl").write_text(
        json.dumps(reference_input, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evaluator_zip = root / "evaluator.zip"
    with ZipFile(evaluator_zip, "w") as zf:
        for path in sorted(evaluator_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(evaluator_root).as_posix())
    public_zip = root / "public.zip"
    with ZipFile(public_zip, "w") as zf:
        for path in sorted(public_root.rglob("*")):
            if path.is_file():
                member = path.relative_to(public_root).as_posix()
                zf.write(path, f"{public_root_prefix}{member}")
    return evaluator_zip, public_zip


def _write_adtte_only_public_numeric_fixture(tmp_path: Path) -> tuple[Path, Path]:
    evaluator_zip, public_zip = _write_public_numeric_fixture(tmp_path)
    root = tmp_path / "fixture"
    scoreable_path = (
        root / "evaluator" / "grader" / "domains" / "route_reference_inputs.jsonl"
    )
    row = json.loads(scoreable_path.read_text(encoding="utf-8"))
    row["required_table_refs"] = [
        ref
        for ref in row["required_table_refs"]
        if ref["rel_path"].endswith("ADTTE.parquet")
    ]
    scoreable_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    evaluator_zip.unlink()
    with ZipFile(evaluator_zip, "w") as zf:
        for path in sorted((root / "evaluator").rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root / "evaluator").as_posix())
    return evaluator_zip, public_zip


def _write_raw_ref_public_reconstruction_fixture(tmp_path: Path) -> tuple[Path, Path]:
    evaluator_zip, public_zip = _write_public_numeric_fixture(tmp_path)
    root = tmp_path / "fixture"
    task_id = "TASKKM001"
    data_root = root / "public" / "items" / task_id / "data"
    reconstruction_root = data_root / "public_reconstruction"
    raw_root = data_root / "raw"
    reconstruction_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    for name in ("ADSL.parquet", "ADTTE.parquet"):
        (reconstruction_root / name).write_bytes((data_root / name).read_bytes())
    raw_tables = {
        "disposition.parquet": pd.DataFrame(
            {"USUBJID": ["C1", "C2", "T1", "T2"], "LAST_CONTACT_DY": [20, 20, 20, 20]}
        ),
        "endpoint_adjudication.parquet": pd.DataFrame(
            {
                "USUBJID": ["C1"],
                "ADJUDICATION_DAY": [10],
                "ADJUDICATION_STATUS": ["confirmed"],
            }
        ),
        "visits.parquet": pd.DataFrame(
            {"USUBJID": ["C1", "C2", "T1", "T2"], "VISITDY": [10, 20, 20, 20]}
        ),
    }
    raw_refs = []
    for name, table in raw_tables.items():
        path = raw_root / name
        table.to_parquet(path, index=False)
        raw_refs.append(
            {
                "rel_path": f"items/{task_id}/data/raw/{name}",
                "semantic_role": "raw_reconstruction_input",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "row_count": int(len(table)),
                "column_names": list(table.columns),
            }
        )
    scoreable_path = (
        root / "evaluator" / "grader" / "domains" / "route_reference_inputs.jsonl"
    )
    row = json.loads(scoreable_path.read_text(encoding="utf-8"))
    row["required_table_refs"] = raw_refs
    row["source_role"] = "public_surface_mirror"
    scoreable_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    evaluator_zip.unlink()
    with ZipFile(evaluator_zip, "w") as zf:
        for path in sorted((root / "evaluator").rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root / "evaluator").as_posix())
    public_zip.unlink()
    with ZipFile(public_zip, "w") as zf:
        for path in sorted((root / "public").rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root / "public").as_posix())
    return evaluator_zip, public_zip


def test_public_evidence_numeric_reference_recomputes_km_rmst_from_public_tables(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(tmp_path)

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "pass"
    assert report.supported_check_count == 1
    assert report.matched_count == 1
    assert report.checks[0].recomputed_value == 5.0
    assert report.checks[0].recomputed_standard_error == pytest.approx(
        3.5355339059327378
    )
    assert report.checks[0].abs_diff == 0.0
    assert (
        report.checks[0].public_surface_shape
        == "public_table:ADSL.parquet|public_table:ADTTE.parquet"
    )
    assert report.method_outcome_counts == {"observed:km_rmst_tau": {"matched": 1}}
    assert report.unsupported_surface_shape_counts == {}
    assert report.drift_classification_counts == {"no_drift": 1}
    assert report.drift_dispositions[0].classification == "no_drift"
    assert report.drift_dispositions[0].new_value == 5.0


def test_public_evidence_numeric_reference_accepts_role_archive_public_root(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path, public_root_prefix="public/"
    )

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip,
        public_zip=public_zip,
    )

    assert report.status == "pass"
    assert report.matched_count == 1


def test_public_evidence_numeric_reference_parallel_option_is_deterministic(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(tmp_path)

    serial = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip,
        public_zip=public_zip,
        workers=1,
    )
    parallel = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip,
        public_zip=public_zip,
        workers=2,
    )

    assert parallel.model_dump(mode="json") == serial.model_dump(mode="json")


def test_km_family_cache_reuses_identical_public_analysis(tmp_path: Path) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(tmp_path)
    with ZipFile(evaluator_zip) as evaluator:
        reference_input = RouteReferenceInputRecordV1.model_validate_json(
            evaluator.read("grader/domains/route_reference_inputs.jsonl")
            .decode("utf-8")
            .strip()
        )
    calls = 0

    def compute() -> _KmFamilyResult:
        nonlocal calls
        calls += 1
        return _KmFamilyResult(
            risk_difference=0.1,
            rmst_difference=1.0,
            risk_difference_se=0.01,
            rmst_difference_se=0.1,
        )

    cache: dict[tuple[str, ...], _KmFamilyResult] = {}
    with ZipFile(public_zip) as public:
        first = _cached_km_family_result_v1(
            public=public,
            reference_input=reference_input,
            regime_cell_id="ipcw_km",
            cache=cache,
            compute=compute,
        )
        second = _cached_km_family_result_v1(
            public=public,
            reference_input=reference_input,
            regime_cell_id="ipcw_km",
            cache=cache,
            compute=compute,
        )

    assert first == second
    assert calls == 1


def test_numeric_replay_family_groups_shared_public_fits() -> None:
    assert numeric_replay_family_id_v1("observed:km_ipcw_rmst_tau") == "ipcw_km"
    assert numeric_replay_family_id_v1("observed:km_ipcw_baseline_cox") == "ipcw_km"
    assert numeric_replay_family_id_v1("observed:km") == "observed:km"


def test_parallel_partitions_keep_underlying_trial_views_together(
    tmp_path: Path,
) -> None:
    evaluator_zip, _ = _write_public_numeric_fixture(tmp_path)
    with ZipFile(evaluator_zip) as evaluator:
        reference_input = RouteReferenceInputRecordV1.model_validate_json(
            evaluator.read("grader/domains/route_reference_inputs.jsonl")
            .decode("utf-8")
            .strip()
        )
        reference = RouteReferenceRecordV1.model_validate_json(
            evaluator.read("grader/domains/route_references.jsonl")
            .decode("utf-8")
            .strip()
        )
    sibling_reference = reference.model_copy(update={"route_reference_id": "SIBLING"})
    other_reference = reference.model_copy(
        update={"route_reference_id": "OTHER", "item_id": "OTHER_ITEM"}
    )
    sibling_input = reference_input.model_copy(
        update={"input_bundle_id": "SIBLING_INPUT", "route_reference_ids": ("SIBLING",)}
    )
    other_input = reference_input.model_copy(
        update={"input_bundle_id": "OTHER_INPUT", "route_reference_ids": ("OTHER",)}
    )

    partitions = _partition_scoreable_inputs_v1(
        scoreable_inputs=(reference_input, sibling_input, other_input),
        reference_by_id={
            reference.route_reference_id: reference,
            sibling_reference.route_reference_id: sibling_reference,
            other_reference.route_reference_id: other_reference,
        },
        workers=2,
    )

    partition_indexes = [{index for index, _ in partition} for partition in partitions]
    assert any({0, 1}.issubset(indexes) for indexes in partition_indexes)
    assert not any({0, 2}.issubset(indexes) for indexes in partition_indexes)

    serial = _partition_scoreable_inputs_v1(
        scoreable_inputs=(reference_input, sibling_input, other_input),
        reference_by_id={
            reference.route_reference_id: reference,
            sibling_reference.route_reference_id: sibling_reference,
            other_reference.route_reference_id: other_reference,
        },
        workers=1,
    )
    assert len(serial) == 1
    assert {index for index, _ in serial[0]} == {0, 1, 2}


def test_public_evidence_numeric_reference_rejects_invalid_worker_count(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(tmp_path)

    with pytest.raises(ValueError, match="workers must be at least 1"):
        recompute_trialeval_public_numeric_reference_v1(
            evaluator_zip=evaluator_zip,
            public_zip=public_zip,
            workers=0,
        )


def test_public_evidence_numeric_reference_uses_public_reconstruction_fallback(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_raw_ref_public_reconstruction_fixture(tmp_path)

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "pass"
    assert report.matched_count == 1
    assert report.unsupported_calculator_count == 0
    assert report.checks[0].recomputed_value == 5.0
    assert report.checks[0].public_surface_shape == (
        "raw_reconstruction_input:raw/disposition.parquet|"
        "raw_reconstruction_input:raw/endpoint_adjudication.parquet|"
        "raw_reconstruction_input:raw/visits.parquet"
    )


def test_public_evidence_numeric_reference_recomputes_km_risk_difference(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path,
        reference_value=-0.5,
        method_id="observed:km",
        effect_scale="risk_difference_tau",
    )

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "pass"
    assert report.matched_count == 1
    assert report.checks[0].recomputed_value == -0.5
    assert report.checks[0].recomputed_standard_error == pytest.approx(
        0.3535533905932738
    )


def test_public_evidence_numeric_reference_recomputes_cluster_participant_weighted_km(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path,
        reference_value=-0.5,
        method_id="observed:cluster_parallel_participant_weighted_km",
        effect_scale="risk_difference_tau",
        reference_standard_error=0.5,
    )

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "pass"
    assert report.supported_check_count == 1
    assert report.matched_count == 1
    assert report.unsupported_calculator_count == 0
    assert report.checks[0].recomputed_value == -0.5


def test_public_evidence_numeric_reference_rejects_cluster_participant_weighted_without_siteid(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path,
        reference_value=-0.5,
        method_id="observed:cluster_parallel_participant_weighted_km",
        effect_scale="risk_difference_tau",
    )
    root = tmp_path / "fixture"
    adsl_path = root / "public" / "items" / "TASKKM001" / "data" / "ADSL.parquet"
    pd.read_parquet(adsl_path).drop(columns=["SITEID"]).to_parquet(
        adsl_path, index=False
    )
    public_zip.unlink()
    with ZipFile(public_zip, "w") as zf:
        for path in sorted((root / "public").rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root / "public").as_posix())

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "fail"
    assert report.invalid_public_input_count == 1
    assert report.checks[0].outcome == "invalid_public_input"
    assert "SITEID" in report.checks[0].message


def test_public_evidence_numeric_reference_recomputes_ipcw_km_from_subject_flags(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path,
        reference_value=0.0,
        method_id="observed:km_ipcw_baseline_cox",
        effect_scale="rmst_difference_tau",
        reference_standard_error=0.6262478453235499,
    )

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "pass"
    assert report.supported_check_count == 1
    assert report.matched_count == 1
    assert report.checks[0].recomputed_value == pytest.approx(0.0, abs=1e-12)
    assert report.checks[0].recomputed_standard_error == pytest.approx(
        0.6262478453235499, abs=1e-9
    )


def test_public_evidence_numeric_reference_rejects_missing_ipcw_total_uncertainty(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path,
        reference_value=0.0,
        method_id="observed:km_ipcw_baseline_cox",
        effect_scale="rmst_difference_tau",
    )

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "fail"
    assert report.checks[0].outcome == "mismatched"
    assert report.checks[0].message == "standard_error_mismatch"


def test_public_evidence_numeric_reference_recomputes_ipcw_cox_from_subject_flags(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path,
        reference_value=0.0,
        method_id="observed:coxph_binary_breslow_ipcw_baseline_cox",
        effect_scale="log_hr",
    )

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "pass"
    assert report.supported_check_count == 1
    assert report.matched_count == 1
    assert report.checks[0].recomputed_value == pytest.approx(0.0, abs=1e-12)


def test_public_evidence_numeric_reference_recomputes_cluster_ipcw_from_subject_flags(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path,
        reference_value=0.0,
        method_id="observed:cluster_parallel_participant_weighted_km_ipcw_baseline_cox",
        effect_scale="risk_difference_tau",
        reference_standard_error=0.06488856845230515,
    )

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "pass"
    assert report.supported_check_count == 1
    assert report.matched_count == 1
    assert report.checks[0].recomputed_value == 0.0
    assert report.checks[0].recomputed_standard_error == pytest.approx(
        0.06488856845230515, abs=1e-12
    )


def test_public_evidence_numeric_reference_rejects_cluster_ipcw_without_siteid(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path,
        reference_value=0.0,
        method_id="observed:cluster_parallel_participant_weighted_km_ipcw_baseline_cox",
        effect_scale="risk_difference_tau",
    )
    root = tmp_path / "fixture"
    flags_path = (
        root
        / "public"
        / "items"
        / "TASKKM001"
        / "data"
        / "subject_operational_flags.parquet"
    )
    pd.read_parquet(flags_path).drop(columns=["SITEID"]).to_parquet(
        flags_path, index=False
    )
    public_zip.unlink()
    with ZipFile(public_zip, "w") as zf:
        for path in sorted((root / "public").rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root / "public").as_posix())

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "fail"
    assert report.invalid_public_input_count == 1
    assert report.checks[0].outcome == "invalid_public_input"
    assert "SITEID" in report.checks[0].message


def test_public_evidence_numeric_reference_rejects_ipcw_without_visible_covariates(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path,
        reference_value=5.0,
        method_id="observed:km_ipcw_baseline_cox",
        effect_scale="rmst_difference_tau",
    )
    root = tmp_path / "fixture"
    flags_path = (
        root
        / "public"
        / "items"
        / "TASKKM001"
        / "data"
        / "subject_operational_flags.parquet"
    )
    flags = pd.read_parquet(flags_path)
    flags.loc[:, ["USUBJID", "TRTA"]].to_parquet(flags_path, index=False)
    public_zip.unlink()
    with ZipFile(public_zip, "w") as zf:
        for path in sorted((root / "public").rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root / "public").as_posix())

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "fail"
    assert report.invalid_public_input_count == 1
    assert report.checks[0].outcome == "invalid_public_input"
    assert "baseline covariates" in report.checks[0].message


def test_public_evidence_numeric_reference_recomputes_cox_breslow(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path,
        reference_value=0.0,
        method_id="observed:coxph_binary_breslow",
        effect_scale="log_hr",
    )

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "pass"
    assert report.matched_count == 1
    assert report.checks[0].recomputed_value == 0.0
    assert report.checks[0].recomputed_standard_error == pytest.approx(
        1.4142135623730951
    )


def test_public_evidence_numeric_reference_recomputes_tau_bounds(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path,
        reference_value=-0.5,
        method_id="observed:tau_bounds_bounded_deviation",
        effect_scale="risk_difference_tau",
        sensitivity_parameter=0.05,
    )

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "pass"
    assert report.matched_count == 1
    assert report.checks[0].recomputed_lower == -0.5
    assert report.checks[0].recomputed_upper == -0.5


def test_public_evidence_numeric_reference_recomputes_group_sequential_composition(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path,
        reference_value=-0.2,
        method_id="observed:group_sequential_adjusted",
        effect_scale="risk_difference_tau",
    )

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "pass"
    assert report.matched_count == 1
    assert report.unsupported_calculator_count == 0
    assert report.checks[0].recomputed_value == 0.0
    assert report.checks[0].recomputed_lower is not None
    assert abs(float(report.checks[0].recomputed_lower) - -0.15111979251833266) < 1e-12
    assert report.checks[0].recomputed_upper is not None
    assert abs(float(report.checks[0].recomputed_upper) - 0.15111979251833266) < 1e-12


def test_public_evidence_numeric_reference_uses_realized_early_stopping_boundary(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path,
        reference_value=-0.2,
        method_id="observed:group_sequential_adjusted",
        effect_scale="risk_difference_tau",
        group_sequential_analysis_look_index=1,
    )

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "pass"
    assert report.checks[0].recomputed_lower is not None
    assert report.checks[0].recomputed_upper is not None
    assert report.checks[0].recomputed_standard_error is not None
    half_width = 0.5 * (
        float(report.checks[0].recomputed_upper)
        - float(report.checks[0].recomputed_lower)
    )
    assert half_width == pytest.approx(
        2.298085834720592 * float(report.checks[0].recomputed_standard_error),
        abs=1e-12,
    )


def test_public_evidence_numeric_reference_recomputes_internal_validation_likelihood(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path,
        reference_value=0.125,
        method_id="observed:validated_endpoint_joint_likelihood",
        effect_scale="risk_difference_tau",
    )

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "pass"
    assert report.matched_count == 1
    assert report.unsupported_calculator_count == 0
    assert report.checks[0].recomputed_value == pytest.approx(0.125, abs=1e-8)
    assert report.checks[0].recomputed_standard_error is not None


def test_public_evidence_numeric_reference_flags_supported_value_mismatch(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path, reference_value=6.0
    )

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "fail"
    assert report.mismatched_count == 1
    assert "supported_numeric_reference_mismatch" in report.findings


def test_public_evidence_numeric_reference_classifies_unregistered_methods(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path,
        reference_value=0.0,
        method_id="observed:unregistered_method",
        effect_scale="risk_difference_tau",
    )

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "fail"
    assert report.findings == (
        "no_supported_numeric_reference_checks",
        "unsupported_public_numeric_calculator",
    )
    assert report.unsupported_calculator_count == 1
    assert report.unsupported_disposition_counts == {
        "unregistered_public_calculator": 1
    }
    assert report.drift_classification_counts == {"blocked_derivation_gap": 1}
    assert len(report.unsupported_method_dispositions) == 1
    disposition = report.unsupported_method_dispositions[0]
    assert disposition.estimator_method_id == "observed:unregistered_method"
    assert disposition.unsupported_count == 1
    assert disposition.disposition == "unregistered_public_calculator"
    assert disposition.surface_shape_counts == {
        "public_table:ADSL.parquet|public_table:ADTTE.parquet": 1
    }
    drift = report.drift_dispositions[0]
    assert drift.classification == "blocked_derivation_gap"
    assert drift.disposition == "unregistered_public_calculator"
    assert drift.required_release_action is not None


def test_public_evidence_numeric_reference_classifies_insufficient_registered_surface(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_adtte_only_public_numeric_fixture(tmp_path)

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "pass"
    assert report.matched_count == 1
    assert report.unsupported_calculator_count == 0
    assert report.drift_classification_counts == {"no_drift": 1}


def test_public_evidence_numeric_reference_classifies_missing_treatment_surface(
    tmp_path: Path,
) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(
        tmp_path, include_adsl=False
    )

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "fail"
    assert report.findings == ("unsupported_public_numeric_calculator",)
    assert report.unsupported_calculator_count == 1
    assert report.unsupported_disposition_counts == {
        "input_surface_insufficient_for_registered_calculator": 1
    }
    assert report.drift_classification_counts == {"blocked_derivation_gap": 1}
    assert report.checks[0].outcome == "unsupported_calculator"


def test_public_evidence_numeric_reference_writes_artifacts(tmp_path: Path) -> None:
    evaluator_zip, public_zip = _write_public_numeric_fixture(tmp_path)
    out_dir = tmp_path / "out"

    report = write_public_evidence_numeric_reference_artifacts_v1(
        evaluator_zip=evaluator_zip,
        public_zip=public_zip,
        out_dir=out_dir,
    )

    assert report.status == "pass"
    assert (
        json.loads(
            (out_dir / "public_evidence_numeric_reference_report.json").read_text(
                encoding="utf-8"
            )
        )["matched_count"]
        == 1
    )
    assert (out_dir / "public_evidence_numeric_reference_checks.jsonl").read_text(
        encoding="utf-8"
    ).count("\n") == 1
    assert (out_dir / "public_evidence_reference_drift_dispositions.jsonl").read_text(
        encoding="utf-8"
    ).count("\n") == 1
    assert "Matched: `1`" in (
        out_dir / "public_evidence_numeric_reference_report.md"
    ).read_text(encoding="utf-8")
