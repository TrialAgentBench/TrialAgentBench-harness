"""Extraction of study-level observables from the pinned RCT Bench corpus."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from trialagentbench_validation.external.contracts import StudySummaryV1

_METADATA_COLUMNS = {
    "Trial_ID",
    "# of Arm",
    "Sample Size",
    "Primary Outcome Type",
}
_DICTIONARY_COLUMNS = {
    "Trial_ID",
    "variable_name",
    "variable_role",
    "variable_type",
    "n_rows",
    "n_missing",
}
_AGE_COLUMN = re.compile(r"^X_age(?:_years)?(?:_0[wdmh])?$", flags=re.IGNORECASE)
_BMI_COLUMN = re.compile(r"^X_bmi(?:_0[wdmh])?$", flags=re.IGNORECASE)
_DAYS_PER_UNIT = {
    "day": 1.0,
    "days": 1.0,
    "week": 7.0,
    "weeks": 7.0,
    "month": 365.25 / 12.0,
    "months": 365.25 / 12.0,
    "year": 365.25,
    "years": 365.25,
}


def extract_rct_bench(
    root: Path, *, source_id: str, expected_trials: int = 125
) -> tuple[StudySummaryV1, ...]:
    """Extract only source-defined, participant-observable study summaries."""

    root = root.resolve()
    metadata = pd.read_excel(root / "meta_data.xlsx", sheet_name="Sheet1")
    dictionary = pd.read_excel(
        root / "data-dictionary.xlsx", sheet_name="Data_Dictionary"
    )
    missing_metadata = _METADATA_COLUMNS - set(metadata.columns)
    missing_dictionary = _DICTIONARY_COLUMNS - set(dictionary.columns)
    if missing_metadata or missing_dictionary:
        raise ValueError(
            f"RCT Bench schema drift: metadata={sorted(missing_metadata)}, dictionary={sorted(missing_dictionary)}"
        )
    csv_paths = sorted((root / "cleaned_data").glob("trial*.csv"))
    if len(metadata) != expected_trials or len(csv_paths) != expected_trials:
        raise ValueError(
            f"RCT Bench inventory must contain {expected_trials} metadata rows and CSV trials; "
            f"observed metadata={len(metadata)}, csv={len(csv_paths)}"
        )

    summaries: list[StudySummaryV1] = []
    for row in metadata.to_dict(orient="records"):
        trial_id = int(row["Trial_ID"])
        study_id = f"rct_bench:trial{trial_id}"
        rows = dictionary.loc[dictionary["Trial_ID"] == trial_id].copy()
        if rows.empty:
            raise ValueError(f"missing data dictionary for {study_id}")
        baseline = rows.loc[rows["variable_role"] == "Baseline covariate"]
        primary = rows.loc[rows["variable_role"] == "Primary outcome"]
        if primary.empty:
            raise ValueError(f"missing primary outcome definition for {study_id}")
        data = pd.read_csv(root / "cleaned_data" / f"trial{trial_id}.csv")
        enrollment = int(row["Sample Size"])
        if len(data) != enrollment:
            if (
                "Participant_ID" not in data
                or int(data["Participant_ID"].nunique(dropna=True)) != enrollment
            ):
                raise ValueError(
                    f"analysis rows do not resolve to the registered participant count for {study_id}"
                )
        baseline_missing = _weighted_missing_fraction(baseline)
        outcome_missing = _weighted_missing_fraction(primary)
        event_fraction, follow_up_median, follow_up_unit, survival_exclusion = (
            _survival_summary(
                data=data,
                primary_dictionary=primary,
                primary_outcome_type=str(row["Primary Outcome Type"]),
                study_id=study_id,
            )
        )
        age_mean, age_sd, age_exclusion = _baseline_continuous_summary(
            data=data,
            baseline_dictionary=baseline,
            column_pattern=_AGE_COLUMN,
            valid_range=(18.0, 110.0),
            study_id=study_id,
            construct="age",
        )
        bmi_mean, bmi_sd, bmi_exclusion = _baseline_continuous_summary(
            data=data,
            baseline_dictionary=baseline,
            column_pattern=_BMI_COLUMN,
            valid_range=(10.0, 80.0),
            study_id=study_id,
            construct="BMI",
        )
        summaries.append(
            StudySummaryV1(
                study_id=study_id,
                source_id=source_id,
                enrollment=enrollment,
                observation_count=len(data),
                arm_count=int(row["# of Arm"]),
                primary_outcome_type=str(row["Primary Outcome Type"]),
                baseline_covariate_count=int(len(baseline)),
                baseline_missing_fraction=baseline_missing,
                primary_outcome_missing_fraction=outcome_missing,
                event_fraction=event_fraction,
                follow_up_time_median=follow_up_median,
                follow_up_time_unit=follow_up_unit,
                age_mean=age_mean,
                age_sd=age_sd,
                bmi_mean=bmi_mean,
                bmi_sd=bmi_sd,
                observable_exclusions=tuple(
                    exclusion
                    for exclusion in (age_exclusion, bmi_exclusion, survival_exclusion)
                    if exclusion is not None
                ),
            )
        )
    return tuple(summaries)


def _weighted_missing_fraction(rows: pd.DataFrame) -> float | None:
    if rows.empty:
        return None
    denominator = float(pd.to_numeric(rows["n_rows"], errors="raise").sum())
    if denominator <= 0:
        raise ValueError("dictionary n_rows must sum to a positive denominator")
    missing = float(pd.to_numeric(rows["n_missing"], errors="raise").sum())
    return missing / denominator


def _survival_summary(
    *,
    data: pd.DataFrame,
    primary_dictionary: pd.DataFrame,
    primary_outcome_type: str,
    study_id: str,
) -> tuple[float | None, float | None, str | None, str | None]:
    if primary_outcome_type.strip().lower() != "time-to-event":
        return None, None, None, None
    time_rows = primary_dictionary.loc[
        primary_dictionary["variable_type"]
        .astype(str)
        .str.lower()
        .eq("time-to-event/continuous time")
    ]
    event_rows = primary_dictionary.loc[
        primary_dictionary["variable_type"].astype(str).str.lower().eq("binary")
        & primary_dictionary["variable_name"]
        .astype(str)
        .str.contains("event", case=False, regex=False)
    ]
    if len(time_rows) != 1 or len(event_rows) != 1:
        return None, None, None, "follow_up_time:source_contract_not_unique"
    time_name = str(time_rows.iloc[0]["variable_name"])
    event_name = str(event_rows.iloc[0]["variable_name"])
    if time_name not in data or event_name not in data:
        raise ValueError(
            f"survival dictionary references absent columns for {study_id}"
        )
    event = pd.to_numeric(data[event_name], errors="coerce")
    observed = event.dropna()
    if observed.empty or not set(observed.unique()).issubset({0, 1}):
        return None, None, None, "follow_up_time:event_indicator_not_binary"
    time = pd.to_numeric(data[time_name], errors="coerce").dropna()
    if time.empty or bool((time < 0).any()):
        raise ValueError(f"invalid follow-up times for {study_id}")
    days_per_unit = _days_per_explicit_source_unit(time_name)
    if days_per_unit is None:
        return (
            float(observed.mean()),
            None,
            None,
            "follow_up_time:source_unit_not_explicit",
        )
    return (
        float(observed.mean()),
        float(np.median(time) * days_per_unit),
        "days",
        None,
    )


def _days_per_explicit_source_unit(variable_name: str) -> float | None:
    tokens = tuple(
        token for token in re.split(r"[^a-z]+", variable_name.lower()) if token
    )
    units = tuple(_DAYS_PER_UNIT[token] for token in tokens if token in _DAYS_PER_UNIT)
    if len(units) != 1:
        return None
    return units[0]


def _baseline_continuous_summary(
    *,
    data: pd.DataFrame,
    baseline_dictionary: pd.DataFrame,
    column_pattern: re.Pattern[str],
    valid_range: tuple[float, float],
    study_id: str,
    construct: str,
) -> tuple[float | None, float | None, str | None]:
    candidates = tuple(
        str(value)
        for value in baseline_dictionary["variable_name"]
        if column_pattern.fullmatch(str(value))
    )
    if not candidates:
        return None, None, f"{construct}:no_compatible_source_variable"
    if len(candidates) != 1:
        return None, None, f"{construct}:ambiguous_source_variables"
    column = candidates[0]
    if column not in data:
        raise ValueError(f"{construct} dictionary column is absent for {study_id}")
    source = data
    if "Participant_ID" in data:
        participant_values = data.loc[:, ["Participant_ID", column]].dropna(
            subset=[column]
        )
        distinct_per_participant = participant_values.groupby(
            "Participant_ID", dropna=False
        )[column].nunique(dropna=True)
        if bool((distinct_per_participant > 1).any()):
            return None, None, f"{construct}:varies_within_participant"
        source = participant_values.drop_duplicates(
            subset=["Participant_ID"], keep="first"
        )
    values = pd.to_numeric(source[column], errors="coerce").dropna()
    if len(values) < 2:
        return None, None, f"{construct}:insufficient_nonmissing_values"
    lower, upper = valid_range
    outside = ~values.between(lower, upper, inclusive="both")
    if bool(outside.any()):
        return None, None, f"{construct}:outside_prespecified_adult_range"
    standard_deviation = float(values.std(ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        return None, None, f"{construct}:nonpositive_or_nonfinite_variation"
    return float(values.mean()), standard_deviation, None


__all__ = ["extract_rct_bench"]
