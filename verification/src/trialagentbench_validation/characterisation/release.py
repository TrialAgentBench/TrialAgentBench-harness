"""Characterise the independent trials in a public TrialEval release."""

from __future__ import annotations

import csv
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Literal, cast
from zipfile import ZipFile

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_validation.characterisation.analysis import characterise_trial
from trialagentbench_validation.characterisation.contracts import (
    CategoricalVariableSpec,
    ContinuousVariableSpec,
    DependenceSpec,
    ReleaseCharacterisation,
    SurvivalOutcomeSpec,
    TidyEstimate,
    TrialCharacterisationSpec,
    TrialData,
    TrialProfile,
    WorkedTrialLineage,
)
from trialagentbench_validation.contracts.scoring.route_references import (
    RouteReferenceRecordV1,
)
from trialagentbench_validation.io import sha256_file

_DESIGN_FAMILIES = {
    "individual_randomized": "individual_randomized",
    "pragmatic": "pragmatic_randomized",
    "covariate_structure": "covariate_subdesign",
    "endpoint_ascertainment": "ascertainment_subdesign",
    "cluster_parallel": "cluster_parallel",
    "stepped_wedge": "stepped_wedge",
    "group_sequential": "group_sequential",
}
_PUBLIC_DESIGN_SUBTYPES = {
    "individual_randomized": "individual_randomized",
    "pragmatic": "pragmatic",
    "covariate_structure": "covariate_structure",
    "endpoint_ascertainment": "endpoint_ascertainment",
    "cluster_parallel": "cluster_parallel",
    "stepped_wedge": "stepped_wedge",
    "group_sequential": "group_sequential",
}


class _CatalogueRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    release_id: str = Field(min_length=1)
    analysis_unit_id: str = Field(pattern=r"^TASK[0-9A-F]{32}$")
    independence_unit_id: str = Field(min_length=1)
    matched_set_id: str = Field(min_length=1)
    design_profile_id: Literal[
        "TE-DP01",
        "TE-DP02",
        "TE-DP03",
        "TE-DP04",
        "TE-DP05",
        "TE-DP06",
        "TE-DP07",
    ]
    design_tier: Literal["D1", "D2", "D3", "D4"]
    assumption_tier: Literal["A1", "A2", "A3", "A4"]
    evaluation_series_id: str = Field(min_length=1)
    context_id: Literal["C1"]


def characterise_trialeval_release(
    *,
    participant_archive: Path,
    catalogue: Path,
    verification_archive: Path,
    worked_trial_id: str = "TE-S03-A1_03",
    release_id: str | None = None,
) -> ReleaseCharacterisation:
    """Characterise every independent C1 trial in a public TrialEval release.

    Parameters
    ----------
    participant_archive
        Public TrialEval participant ZIP.
    catalogue
        Public simulation-properties JSONL from the paired release.
    verification_archive
        Public verification ZIP containing exact independently reproduced
        route records.
    worked_trial_id
        Independence-unit identifier used for the participant-to-analysis
        walkthrough.
    release_id
        Exact release identity. When omitted, use the catalogue identity.

    Returns
    -------
    ReleaseCharacterisation
        Complete 100-trial profile census and worked-trial lineage.

    Raises
    ------
    FileNotFoundError
        If a required input or release member is absent.
    ValueError
        If identities, context counts, tables, or selected analysis records are
        incomplete or inconsistent.
    """

    for path in (participant_archive, catalogue, verification_archive):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows, catalogue_release_id, context_view_count = _read_catalogue(catalogue)
    resolved_release_id = catalogue_release_id if release_id is None else release_id
    if not resolved_release_id:
        raise ValueError("release_id must not be empty")
    with (
        ZipFile(participant_archive) as participant,
        ZipFile(verification_archive) as verification,
    ):
        _validate_archive_members(participant)
        _validate_archive_members(verification)
        profiles = tuple(
            _characterise_profile(participant, row)
            for row in sorted(rows, key=lambda item: item.independence_unit_id)
        )
        worked_trial = _worked_trial_lineage(
            participant=participant,
            verification=verification,
            profiles=profiles,
            trial_id=worked_trial_id,
        )
    return ReleaseCharacterisation(
        release_id=resolved_release_id,
        participant_archive_sha256=sha256_file(participant_archive),
        catalogue_sha256=sha256_file(catalogue),
        verification_archive_sha256=sha256_file(verification_archive),
        context_view_count=context_view_count,
        independent_trial_count=len(profiles),
        profiles=profiles,
        worked_trial=worked_trial,
    )


def write_release_characterisation(
    output_dir: Path,
    result: ReleaseCharacterisation,
) -> None:
    """Write the TrialEval base-trial census, estimates, lineage, and canonical JSON.

    Parameters
    ----------
    output_dir
        New output directory.
    result
        Validated release characterisation.

    Raises
    ------
    FileExistsError
        If the output directory already exists.
    """

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "release_characterisation.json").write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "worked_trial_lineage.json").write_text(
        result.worked_trial.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    _write_profile_csv(output_dir / "programme_profiles.csv", result.profiles)
    _write_estimate_csv(
        output_dir / "programme_estimates.csv",
        tuple(
            estimate
            for profile in result.profiles
            for estimate in profile.characterisation.estimates
        ),
    )
    worked_profile = next(
        profile
        for profile in result.profiles
        if profile.task_id == result.worked_trial.task_id
    )
    _write_estimate_csv(
        output_dir / "worked_trial.csv", worked_profile.characterisation.estimates
    )


def read_worked_trial_participants(
    *,
    participant_archive: Path,
    result: ReleaseCharacterisation,
) -> pd.DataFrame:
    """Read the participant-visible variables used in the worked trial.

    Parameters
    ----------
    participant_archive
        Public TrialEval participant ZIP characterised by ``result``.
    result
        Validated release characterisation.

    Returns
    -------
    pandas.DataFrame
        Synthetic participant rows with analysis-neutral display names.

    Raises
    ------
    FileNotFoundError
        If the participant archive or required member is absent.
    ValueError
        If the archive identity, participant linkage, or values disagree with
        the validated release characterisation.
    """

    if not participant_archive.is_file():
        raise FileNotFoundError(participant_archive)
    if sha256_file(participant_archive) != result.participant_archive_sha256:
        raise ValueError(
            "participant archive does not match the release characterisation"
        )
    profile = next(
        row for row in result.profiles if row.task_id == result.worked_trial.task_id
    )
    prefix = f"items/{profile.task_id}"
    with ZipFile(participant_archive) as participant:
        _validate_archive_members(participant)
        adsl = _parquet_member(participant, f"{prefix}/data/ADSL.parquet")
        adtte = _parquet_member(participant, f"{prefix}/data/ADTTE.parquet")
        flags = _parquet_member(
            participant, f"{prefix}/data/subject_operational_flags.parquet"
        )
    primary = adtte.loc[
        adtte["PARAMCD"].astype("string") == profile.primary_paramcd,
        ["USUBJID", "AVAL", "CNSR"],
    ]
    required_flags = {
        "USUBJID",
        "ATTENDANCE_RATE",
        "MEAN_EXADH",
        "N_ICE_RECORDS",
        "ANY_DISCONTINUATION_ICE",
        "ANY_RESCUE_THERAPY_ICE",
        "ANY_TREATMENT_SWITCH_ICE",
    }
    if missing := sorted(required_flags - set(flags.columns)):
        raise ValueError(
            f"worked-trial operational table is missing columns: {missing!r}"
        )
    rows = (
        adsl[["USUBJID", "TRTA", "PPFL", "AGE", "BMI"]]
        .merge(primary, on="USUBJID", validate="one_to_one")
        .merge(
            flags.loc[:, sorted(required_flags)], on="USUBJID", validate="one_to_one"
        )
        .rename(
            columns={
                "USUBJID": "participant_id",
                "TRTA": "arm",
                "PPFL": "per_protocol",
                "AGE": "age_years",
                "BMI": "bmi_kg_m2",
                "AVAL": "time_days",
                "ATTENDANCE_RATE": "attendance_rate",
                "MEAN_EXADH": "exposure_adherence",
                "N_ICE_RECORDS": "intercurrent_event_count",
                "ANY_DISCONTINUATION_ICE": "discontinued",
                "ANY_RESCUE_THERAPY_ICE": "rescue_therapy",
                "ANY_TREATMENT_SWITCH_ICE": "treatment_switch",
            }
        )
    )
    rows["event"] = 1 - pd.to_numeric(rows.pop("CNSR"), errors="coerce")
    for column in (
        "per_protocol",
        "discontinued",
        "rescue_therapy",
        "treatment_switch",
    ):
        rows[column] = rows[column].astype("string").fillna("").eq("Y").astype(int)
    rows["any_intercurrent_event"] = (
        pd.to_numeric(rows["intercurrent_event_count"], errors="coerce")
        .gt(0)
        .astype(int)
    )
    numeric = (
        "age_years",
        "bmi_kg_m2",
        "time_days",
        "attendance_rate",
        "exposure_adherence",
        "intercurrent_event_count",
        "per_protocol",
        "discontinued",
        "rescue_therapy",
        "treatment_switch",
        "any_intercurrent_event",
        "event",
    )
    rows.loc[:, numeric] = rows.loc[:, numeric].apply(pd.to_numeric, errors="coerce")
    if (
        len(rows) != result.worked_trial.linked_rows
        or rows[list(numeric)].isna().any().any()
    ):
        raise ValueError("worked-trial display rows are incomplete")
    if not rows["event"].isin((0, 1)).all():
        raise ValueError("worked-trial event indicator must be binary")
    return rows.sort_values("participant_id").reset_index(drop=True)


def _read_catalogue(path: Path) -> tuple[tuple[_CatalogueRow, ...], str, int]:
    rows: list[_CatalogueRow] = []
    release_ids: set[str] = set()
    context_view_count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(
                    f"catalogue contains a blank row at line {line_number}"
                )
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"catalogue row {line_number} is not an object")
            if payload.get("suite_id") != "trialeval":
                continue
            context_view_count += 1
            release_id = _text(payload, "release_id", line_number)
            release_ids.add(release_id)
            construction = _object(payload, "construction", line_number)
            if construction.get("context_id") != "C1":
                continue
            rows.append(
                _CatalogueRow(
                    release_id=release_id,
                    analysis_unit_id=_text(payload, "analysis_unit_id", line_number),
                    independence_unit_id=_text(
                        payload, "independence_unit_id", line_number
                    ),
                    matched_set_id=_text(payload, "matched_set_id", line_number),
                    design_profile_id=_text(
                        construction, "design_profile_id", line_number
                    ),
                    design_tier=_text(construction, "design_tier", line_number),
                    assumption_tier=_text(construction, "assumption_tier", line_number),
                    evaluation_series_id=_text(
                        construction, "evaluation_series_id", line_number
                    ),
                    context_id="C1",
                )
            )
    if len(release_ids) != 1:
        raise ValueError("TrialEval catalogue must identify exactly one release")
    if context_view_count != 500 or len(rows) != 100:
        raise ValueError(
            "TrialEval release characterisation requires 500 context views and 100 independent C1 trials"
        )
    independence_ids = tuple(row.independence_unit_id for row in rows)
    matched_ids = tuple(row.matched_set_id for row in rows)
    if len(set(independence_ids)) != len(rows) or len(set(matched_ids)) != len(rows):
        raise ValueError(
            "C1 catalogue rows must be unique by independence and matched-set identity"
        )
    return tuple(rows), next(iter(release_ids)), context_view_count


def _characterise_profile(participant: ZipFile, row: _CatalogueRow) -> TrialProfile:
    prefix = f"items/{row.analysis_unit_id}"
    task = _json_member(participant, f"{prefix}/task.json")
    protocol = _json_member(participant, f"{prefix}/protocol_summary.json")
    plan = _json_member(participant, f"{prefix}/analysis_plan.json")
    adsl = _parquet_member(participant, f"{prefix}/data/ADSL.parquet")
    adtte = _parquet_member(participant, f"{prefix}/data/ADTTE.parquet")
    flags = _parquet_member(
        participant, f"{prefix}/data/subject_operational_flags.parquet"
    )
    design_subtype = _text(task, "design_subtype", 0)
    design_family = _DESIGN_FAMILIES.get(design_subtype)
    if design_family is None:
        raise ValueError(f"unsupported released design subtype: {design_subtype!r}")
    public_design_subtype = _PUBLIC_DESIGN_SUBTYPES[design_subtype]
    primary_paramcd = _text(task, "primary_paramcd", 0)
    primary_rows = adtte.loc[
        adtte["PARAMCD"].astype("string") == primary_paramcd
    ].copy()
    if primary_rows.empty:
        raise ValueError(
            f"{row.analysis_unit_id} lacks primary endpoint {primary_paramcd!r}"
        )
    endpoint = primary_rows[["USUBJID", "AVAL", "CNSR"]].copy()
    endpoint["event"] = 1 - pd.to_numeric(endpoint["CNSR"], errors="coerce")
    endpoint = endpoint.rename(columns={"AVAL": "primary_time"})
    attendance = flags[["USUBJID", "ATTENDANCE_RATE"]].copy()
    participants = adsl.merge(endpoint, on="USUBJID", validate="one_to_one").merge(
        attendance,
        on="USUBJID",
        validate="one_to_one",
    )
    tau = _positive_number(task, "primary_tau_dy")
    horizons = tuple(float(tau * fraction) for fraction in (0.25, 0.5, 0.75, 1.0))
    spec = TrialCharacterisationSpec(
        trial_id=row.independence_unit_id,
        programme_id=row.evaluation_series_id,
        design_profile_id=row.design_profile_id,
        design_family=design_family,
        participant_id_column="USUBJID",
        arm_column="TRTA",
        cluster_id_column=(
            "SITEID"
            if design_subtype in {"cluster_parallel", "stepped_wedge"}
            else None
        ),
        continuous_variables=(
            ContinuousVariableSpec(variable_id="age", column="AGE", unit="years"),
            ContinuousVariableSpec(variable_id="bmi", column="BMI", unit="kg/m2"),
            ContinuousVariableSpec(
                variable_id="attendance_rate",
                column="ATTENDANCE_RATE",
                unit="proportion",
                role="observation",
            ),
        ),
        categorical_variables=(
            CategoricalVariableSpec(
                variable_id="sex",
                column="SEX",
                unit="proportion",
                categories=("F", "M"),
            ),
        ),
        dependence=(
            DependenceSpec(
                dependence_id="age_bmi",
                left_column="AGE",
                right_column="BMI",
                stratify_by_arm=False,
            ),
        ),
        outcomes=(
            SurvivalOutcomeSpec(
                outcome_id="primary",
                table="participants",
                participant_id_column="USUBJID",
                duration_column="primary_time",
                event_column="event",
                horizons=horizons,
                unit="days",
            ),
        ),
        bootstrap_replicates=200,
        seed=_profile_seed(row.independence_unit_id),
    )
    characterisation = characterise_trial(spec, TrialData(participants=participants))
    primary_analysis = _object(plan, "primary_analysis", 0)
    if len(participants) != len(adsl):
        raise ValueError(
            f"{row.analysis_unit_id} participant-to-endpoint linkage is incomplete"
        )
    protocol_tau = _positive_number(protocol, "followup_horizon_dy")
    if not math.isclose(tau, protocol_tau, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{row.analysis_unit_id} task and protocol follow-up horizons disagree"
        )
    return TrialProfile(
        task_id=row.analysis_unit_id,
        matched_set_id=row.matched_set_id,
        independence_unit_id=row.independence_unit_id,
        context_id="C1",
        design_profile_id=row.design_profile_id,
        design_tier=row.design_tier,
        assumption_tier=row.assumption_tier,
        design_subtype=public_design_subtype,
        participant_count=len(participants),
        follow_up_horizon_days=tau,
        primary_paramcd=primary_paramcd,
        primary_estimand_id=_text(task, "primary_estimand_id", 0),
        primary_effect_scale=_text(task, "primary_effect_scale", 0),
        primary_result_unit=_text(task, "primary_result_unit", 0),
        primary_method_id=_text(primary_analysis, "method_id", 0),
        characterisation=characterisation,
    )


def _worked_trial_lineage(
    *,
    participant: ZipFile,
    verification: ZipFile,
    profiles: tuple[TrialProfile, ...],
    trial_id: str,
) -> WorkedTrialLineage:
    profile_matches = [row for row in profiles if row.independence_unit_id == trial_id]
    if len(profile_matches) != 1:
        raise ValueError(
            f"worked trial must identify one independent C1 trial: {trial_id!r}"
        )
    profile = profile_matches[0]
    task_id = profile.task_id
    references = tuple(
        RouteReferenceRecordV1.model_validate_json(line)
        for line in verification.read(
            "grader/domains/route_references.jsonl"
        ).splitlines()
        if line.strip()
    )
    reference_matches = [
        row
        for row in references
        if row.task_id == task_id
        and row.estimator_method_id == profile.primary_method_id
        and row.effect_scale == profile.primary_effect_scale
        and row.answer_shape == "point"
    ]
    if len(reference_matches) != 1:
        raise ValueError(
            "worked trial must have one point-valued primary route reference"
        )
    reference = reference_matches[0]
    if reference.value is None or reference.ci_low is None or reference.ci_high is None:
        raise ValueError(
            "worked-trial point reference requires estimate and confidence interval"
        )
    prefix = f"items/{task_id}"
    adsl = _parquet_member(participant, f"{prefix}/data/ADSL.parquet")
    adtte = _parquet_member(participant, f"{prefix}/data/ADTTE.parquet")
    primary = adtte.loc[adtte["PARAMCD"].astype("string") == profile.primary_paramcd]
    linked = adsl[["USUBJID"]].merge(
        primary[["USUBJID"]], on="USUBJID", validate="one_to_one"
    )
    plan = _json_member(participant, f"{prefix}/analysis_plan.json")
    primary_analysis = _object(plan, "primary_analysis", 0)
    return WorkedTrialLineage(
        case_id=trial_id,
        task_id=task_id,
        participant_table_path=f"{prefix}/data/ADSL.parquet",
        endpoint_table_path=f"{prefix}/data/ADTTE.parquet",
        task_path=f"{prefix}/task.json",
        protocol_path=f"{prefix}/protocol_summary.json",
        analysis_plan_path=f"{prefix}/analysis_plan.json",
        participant_rows=len(adsl),
        endpoint_rows=len(primary),
        linked_rows=len(linked),
        analysis_population=_text(
            _json_member(participant, f"{prefix}/task.json"), "primary_population_id", 0
        ),
        estimand_id=profile.primary_estimand_id,
        endpoint_id=profile.primary_paramcd,
        effect_scale=profile.primary_effect_scale,
        estimator_method=profile.primary_method_id,
        estimate=float(reference.value),
        interval_low=float(reference.ci_low),
        interval_high=float(reference.ci_high),
        unit=profile.primary_result_unit,
        uncertainty_method=_text(primary_analysis, "uncertainty_method", 0),
    )


def _write_profile_csv(path: Path, profiles: tuple[TrialProfile, ...]) -> None:
    fieldnames = (
        "task_id",
        "matched_set_id",
        "independence_unit_id",
        "context_id",
        "design_profile_id",
        "design_tier",
        "assumption_tier",
        "design_subtype",
        "participant_count",
        "follow_up_horizon_days",
        "primary_paramcd",
        "primary_estimand_id",
        "primary_effect_scale",
        "primary_result_unit",
        "primary_method_id",
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for profile in profiles:
            payload = profile.model_dump(exclude={"characterisation"})
            writer.writerow(payload)


def _write_estimate_csv(path: Path, estimates: tuple[TidyEstimate, ...]) -> None:
    fieldnames = tuple(TidyEstimate.model_fields)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for estimate in estimates:
            writer.writerow(estimate.model_dump(mode="json"))


def _json_member(archive: ZipFile, member: str) -> dict[str, object]:
    try:
        payload = json.loads(archive.read(member))
    except KeyError as exc:
        raise FileNotFoundError(f"release archive lacks {member}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"release member is not a JSON object: {member}")
    return payload


def _parquet_member(archive: ZipFile, member: str) -> pd.DataFrame:
    try:
        return cast(pd.DataFrame, pd.read_parquet(BytesIO(archive.read(member))))
    except KeyError as exc:
        raise FileNotFoundError(f"release archive lacks {member}") from exc


def _validate_archive_members(archive: ZipFile) -> None:
    names: set[str] = set()
    for member in archive.infolist():
        path = Path(member.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
            raise ValueError(f"unsafe release archive member: {member.filename!r}")
        if member.filename in names:
            raise ValueError(f"duplicate release archive member: {member.filename!r}")
        names.add(member.filename)


def _object(
    payload: dict[str, object], key: str, line_number: int
) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"row {line_number} requires object field {key!r}")
    return value


def _text(payload: dict[str, object], key: str, line_number: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"row {line_number} requires non-empty string field {key!r}")
    return value


def _positive_number(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"field {key!r} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"field {key!r} must be finite and positive")
    return number


def _profile_seed(independence_unit_id: str) -> int:
    return int.from_bytes(
        independence_unit_id.encode("utf-8")[:4].ljust(4, b"\0"), "big"
    )


__all__ = [
    "characterise_trialeval_release",
    "read_worked_trial_participants",
    "write_release_characterisation",
]
