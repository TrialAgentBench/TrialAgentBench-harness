"""Study-level observables reconstructed from a public TrialEval release."""

from __future__ import annotations

import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd

from trialagentbench_validation.external.contracts import StudySummaryV1

_PREPARED_BASELINE = "data/ADSL.parquet"
_PREPARED_TIME_TO_EVENT = "data/ADTTE.parquet"
_RAW_BASELINE = "data/raw/baseline_characteristics.parquet"
_RAW_RANDOMIZATION = "data/raw/randomization.parquet"
_ARM_COLUMNS = ("TRTA", "ARMCD", "ARM")


def extract_public_synthetic_trials(
    participant_release: Path,
) -> tuple[tuple[StudySummaryV1, ...], int, str | None, str | None]:
    """Reconstruct one demographic summary per independent public trial."""

    release_path = Path(participant_release)
    with ZipFile(release_path) as archive:
        names = _validated_members(archive)
        applied_profile_id, applied_profile_sha256 = _read_applied_profile(
            archive, names
        )
        task_members = sorted(
            name
            for name in names
            if name.startswith("items/") and name.endswith("/task.json")
        )
        if not task_members:
            raise ValueError("participant release contains no item task contracts")
        by_trial: dict[tuple[str, str], list[tuple[bool, StudySummaryV1]]] = (
            defaultdict(list)
        )
        for task_member in task_members:
            prefix = task_member.rsplit("/", 1)[0]
            task = _read_json_object(archive, task_member)
            protocol_name = task.get("protocol_summary_file")
            if not isinstance(protocol_name, str) or not protocol_name:
                raise ValueError(f"{task_member} lacks protocol_summary_file")
            protocol_member = f"{prefix}/{protocol_name}"
            if protocol_member not in names:
                raise ValueError(f"{task_member} references missing protocol summary")
            protocol = _read_json_object(archive, protocol_member)
            trial_id = _required_text(protocol, "trial_id", protocol_member)
            study_id = _required_text(protocol, "study_id", protocol_member)
            primary_paramcd = _required_text(task, "primary_paramcd", task_member)
            prepared_member = f"{prefix}/{_PREPARED_BASELINE}"
            raw_member = f"{prefix}/{_RAW_BASELINE}"
            if prepared_member in names:
                baseline_member = prepared_member
                prepared = True
            elif raw_member in names:
                baseline_member = raw_member
                prepared = False
            else:
                raise ValueError(
                    f"{task_member} exposes neither prepared nor raw baseline evidence"
                )
            baseline = _read_parquet(archive, baseline_member)
            if prepared:
                randomization = baseline
            else:
                randomization_member = f"{prefix}/{_RAW_RANDOMIZATION}"
                if randomization_member not in names:
                    raise ValueError(
                        f"{task_member} lacks public randomization evidence"
                    )
                randomization = _read_parquet(archive, randomization_member)
            time_to_event_member = f"{prefix}/{_PREPARED_TIME_TO_EVENT}"
            if time_to_event_member not in names:
                raise ValueError(f"{task_member} lacks public time-to-event evidence")
            time_to_event = _read_parquet(archive, time_to_event_member)
            by_trial[(trial_id, study_id)].append(
                (
                    prepared,
                    _summarize_trial(
                        study_id=f"synthetic:{trial_id}:{study_id}",
                        baseline=baseline,
                        randomization=randomization,
                        time_to_event=time_to_event,
                        primary_paramcd=primary_paramcd,
                    ),
                )
            )

    summaries = tuple(
        _select_prepared_summary(trial_key=trial_key, candidates=tuple(candidates))
        for trial_key, candidates in sorted(by_trial.items())
    )
    return summaries, len(task_members), applied_profile_id, applied_profile_sha256


def _read_applied_profile(
    archive: ZipFile,
    names: set[str],
) -> tuple[str | None, str | None]:
    manifest_member = "manifest.json"
    if manifest_member not in names:
        raise ValueError(
            "participant release lacks manifest.json with applied-profile provenance"
        )
    manifest = _read_json_object(archive, manifest_member)
    required_keys = {
        "applied_baseline_profile_id",
        "applied_baseline_profile_sha256",
    }
    missing_keys = sorted(required_keys - set(manifest))
    if missing_keys:
        raise ValueError(
            f"manifest.json lacks applied-profile provenance keys: {missing_keys!r}"
        )
    profile_id = manifest.get("applied_baseline_profile_id")
    profile_sha256 = manifest.get("applied_baseline_profile_sha256")
    if profile_id is None and profile_sha256 is None:
        return None, None
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("manifest.json has invalid applied_baseline_profile_id")
    if (
        not isinstance(profile_sha256, str)
        or len(profile_sha256) != 64
        or any(character not in "0123456789abcdef" for character in profile_sha256)
    ):
        raise ValueError("manifest.json has invalid applied_baseline_profile_sha256")
    return profile_id, profile_sha256


def _validated_members(archive: ZipFile) -> set[str]:
    names: set[str] = set()
    for member in archive.infolist():
        name = member.filename
        path = PurePosixPath(name)
        if "\\" in name or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe participant archive member: {name!r}")
        if name in names:
            raise ValueError(f"duplicate participant archive member: {name!r}")
        names.add(name)
    forbidden_roots = {"grader", "evaluator", "hidden", "audit"}
    leaked = sorted(
        name
        for name in names
        if PurePosixPath(name).parts[:1] and path_root(name) in forbidden_roots
    )
    if leaked:
        raise ValueError(
            f"participant release contains evaluator-only roots: {leaked[:3]}"
        )
    return names


def path_root(name: str) -> str:
    """Return the first path component of an archive member."""

    return PurePosixPath(name).parts[0]


def _read_json_object(archive: ZipFile, member: str) -> dict[str, object]:
    payload = json.loads(archive.read(member))
    if not isinstance(payload, dict):
        raise ValueError(f"{member} is not a JSON object")
    return payload


def _required_text(payload: dict[str, object], field: str, member: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{member} lacks {field}")
    return value


def _read_parquet(archive: ZipFile, member: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(BytesIO(archive.read(member)))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{member} is not a readable Parquet table") from exc


def _summarize_trial(
    *,
    study_id: str,
    baseline: pd.DataFrame,
    randomization: pd.DataFrame,
    time_to_event: pd.DataFrame,
    primary_paramcd: str,
) -> StudySummaryV1:
    required = {"USUBJID", "AGE", "BMI"}
    missing = sorted(required - set(baseline.columns))
    if missing:
        raise ValueError(f"public baseline evidence lacks required columns: {missing}")
    if baseline.empty or baseline["USUBJID"].isna().any():
        raise ValueError("public baseline evidence requires non-missing participants")
    if baseline["USUBJID"].astype("string").duplicated().any():
        raise ValueError(
            "public baseline evidence must contain one row per participant"
        )
    age = pd.to_numeric(baseline["AGE"], errors="coerce").dropna().to_numpy(dtype=float)
    bmi = pd.to_numeric(baseline["BMI"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(age) < 2 or len(bmi) < 2:
        raise ValueError(
            "public baseline evidence requires at least two observed AGE and BMI values"
        )
    if not np.isfinite(age).all() or not np.isfinite(bmi).all():
        raise ValueError("public baseline AGE and BMI values must be finite")
    arm_column = next(
        (column for column in _ARM_COLUMNS if column in randomization.columns), None
    )
    if arm_column is None:
        raise ValueError("public randomization evidence lacks a treatment-arm column")
    arms = randomization[arm_column].dropna().astype("string")
    if arms.empty:
        raise ValueError("public randomization evidence contains no treatment arms")
    event_fraction, follow_up_time_median = _summarize_time_to_event(
        time_to_event,
        primary_paramcd=primary_paramcd,
    )
    return StudySummaryV1(
        study_id=study_id,
        source_id="trialagentbench_public",
        enrollment=int(baseline["USUBJID"].astype("string").nunique()),
        observation_count=len(baseline),
        arm_count=int(arms.nunique()),
        primary_outcome_type="time-to-event",
        baseline_covariate_count=0,
        event_fraction=event_fraction,
        follow_up_time_median=follow_up_time_median,
        follow_up_time_unit="days",
        age_mean=float(np.mean(age)),
        age_sd=float(np.std(age, ddof=1)),
        bmi_mean=float(np.mean(bmi)),
        bmi_sd=float(np.std(bmi, ddof=1)),
        observable_exclusions=(
            "baseline_covariate_count:not_comparable_across_prepared_and_raw_surfaces",
            "baseline_missing_fraction:not_comparable_across_prepared_and_raw_surfaces",
            "primary_outcome_missing_fraction:not_extracted_from_baseline_evidence",
        ),
    )


def _summarize_time_to_event(
    frame: pd.DataFrame,
    *,
    primary_paramcd: str,
) -> tuple[float, float]:
    required = {"USUBJID", "PARAMCD", "AVAL", "CNSR"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"public time-to-event evidence lacks required columns: {missing}"
        )
    rows = frame.loc[frame["PARAMCD"].astype("string") == primary_paramcd].copy()
    if rows.empty:
        raise ValueError(
            "public time-to-event evidence lacks the task primary endpoint"
        )
    if (
        rows["USUBJID"].isna().any()
        or rows["USUBJID"].astype("string").duplicated().any()
    ):
        raise ValueError(
            "public time-to-event evidence must contain one primary-endpoint row per participant"
        )
    times = pd.to_numeric(rows["AVAL"], errors="coerce")
    censoring = pd.to_numeric(rows["CNSR"], errors="coerce")
    if (
        times.isna().any()
        or not np.isfinite(times.to_numpy(dtype=float)).all()
        or bool((times < 0).any())
    ):
        raise ValueError(
            "public time-to-event durations must be finite nonnegative days"
        )
    if censoring.isna().any() or not set(censoring.unique()).issubset({0, 1}):
        raise ValueError("public CNSR must use zero for events and one for censoring")
    return float((censoring == 0).mean()), float(times.median())


def _select_prepared_summary(
    *,
    trial_key: tuple[str, str],
    candidates: tuple[tuple[bool, StudySummaryV1], ...],
) -> StudySummaryV1:
    prepared = tuple(summary for is_prepared, summary in candidates if is_prepared)
    if not prepared:
        raise ValueError(f"trial {trial_key!r} has no prepared public baseline view")
    reference = prepared[0]
    reference_payload = reference.model_dump(exclude={"observable_exclusions"})
    if any(
        summary.model_dump(exclude={"observable_exclusions"}) != reference_payload
        for summary in prepared[1:]
    ):
        raise ValueError(f"prepared context views disagree for trial {trial_key!r}")
    return reference


__all__ = ["extract_public_synthetic_trials"]
