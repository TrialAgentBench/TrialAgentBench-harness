"""Observed clinical records projected through the shared SDTM/ADaM functions."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from clinical_data_tables import (
    adam_findings,
    adam_time_to_event,
    clinical_epochs,
    clinical_timing,
    completion_disposition_records,
    exposure_date_bounds,
    intercurrent_disposition_records,
    participation_end_times,
    sdtm_adverse_events,
    sdtm_death_status,
    sdtm_disposition,
    sdtm_exposure,
    sdtm_findings,
    sdtm_sex,
    withdrawal_participation_times,
)

from trialagentbench.io.directory import staged_directory


def write_observed_study_tables(
    output: Path,
    *,
    study_id: str,
    participants: pd.DataFrame,
    analysis: pd.DataFrame,
    endpoint_id: str,
    endpoint_object: str,
    endpoint_unit: str,
    endpoint_kind: Literal["continuous", "longitudinal", "binary", "count", "survival", "recurrent"],
    assessment_time: float | None,
    endpoint_event: Literal["death"] | None = None,
    intercurrent_events: pd.DataFrame | None = None,
    completion_records: pd.DataFrame | None = None,
    observational: bool = False,
    study_start: date | None = None,
    epochs: tuple[tuple[str, float, float], ...] = (),
    additional_endpoints: tuple[tuple[str, str], ...] = (),
    safety_events: pd.DataFrame | None = None,
    exposure_records: pd.DataFrame | None = None,
    endpoint_domain: Literal["QS", "LB", "VS", "FA"] = "QS",
) -> tuple[Path, ...]:
    """Write research-interchange tables from already released observations.

    Parameters
    ----------
    output
        New output directory, separate from the immutable source records.
    study_id
        Public study identity.
    participants, analysis
        Observed source tables with participant_id and assigned_arm. Additional
        participant variables remain typed in ADSL and labelled in variables.csv.
        Optional consent_time is recorded elapsed days on the participant clinical
        clock; missing values mean consent is unknown, not absent. Dates add the
        participant enrollment_day offset to the declared calendar origin.
        Closeout fields retain occurrence, observation, documentation and
        availability times in days with explicit completion_status. Jointly missing
        clocks and status mean no available form, not observed non-completion.
    endpoint_object
        Authored event or clinical condition, distinct from its assessment test.
    additional_endpoints
        Additional source field and clinical object pairs.
    endpoint_id, endpoint_unit, endpoint_kind
        Declared observed endpoint, natural unit and data family.
    endpoint_event
        Explicit death designation for the primary survival endpoint. Composite
        and other terminal endpoints are not inferred to represent death.
    assessment_time
        Follow-up time in days when a fixed-horizon assessment has no row clock.
        None retains an unknown assessment date for the initial observational data.
    observational
        Whether treatment labels are observed exposures instead of randomized arms.
    study_start
        Declared programme calendar origin, if available; absent dates are not invented.
        Recorded enrollment_day offsets define each participant reference instant;
        source observation times remain elapsed since that instant. The sponsor
        reference period ends at recorded completion, death or withdrawal, or an
        explicitly recorded administrative review when the participation end is
        unknown. Administrative review does not establish a patient-contact date.
    epochs
        Declared (name, start, end) intervals on the participant-relative day clock.
        Absent declarations retain unknown labels.
    intercurrent_events
        Already observed incident events available at readout, with participant_id,
        event_kind and occurrence_time on the participant-relative day clock.
    completion_records
        Explicit end-status forms available at readout, with participant_id,
        occurrence_time in elapsed days and status completed, not_completed or
        status_unresolved. Only completed asserts clinical completion; the other
        states document administrative review without inventing patient contact.
    exposure_records
        Optional observed participant_id, occurrence_time and received_arm rows.
        Times use the participant-relative day clock. No dosing schedule is inferred.

    Returns
    -------
    tuple of pathlib.Path
        Written tables and their source-variable dictionary.

    Raises
    ------
    ValueError
        If identities, treatment, observation status or endpoint values are invalid.
    FileExistsError
        If the output directory exists.
    """
    additional_objects = dict(additional_endpoints)
    if len(additional_objects) != len(additional_endpoints) or endpoint_id in additional_objects:
        raise ValueError("additional endpoint identities must be unique and distinct from the primary")
    if not endpoint_object.strip() or any(not value.strip() for value in additional_objects.values()):
        raise ValueError("clinical endpoints require explicit nonempty object names")
    key = "participant_id"
    required = {key, "assigned_arm"}
    if not required <= set(participants) or not required <= set(analysis):
        raise ValueError("clinical tables require source participant and treatment fields")
    if participants[key].isna().any() or participants[key].duplicated().any():
        raise ValueError("clinical subject identifiers must be unique and nonmissing")
    if participants.assigned_arm.isna().any() or analysis.assigned_arm.isna().any():
        raise ValueError("clinical treatment labels must be observed")
    subjects = pd.Series([f"{study_id}-{value}" for value in participants[key]], index=participants[key], dtype=str)
    if not subjects.is_unique or not analysis[key].isin(subjects.index).all():
        raise ValueError("clinical analysis contains an unknown or ambiguous subject")
    expected_arm = analysis[key].map(participants.set_index(key).assigned_arm)
    if not expected_arm.equals(analysis.assigned_arm):
        raise ValueError("clinical analysis treatment differs from the source participant")
    if endpoint_kind not in {"longitudinal", "recurrent"} and analysis[key].duplicated().any():
        raise ValueError("clinical endpoint requires one record per subject")
    if output.exists():
        raise FileExistsError(output)
    tables: list[tuple[str, pd.DataFrame]] = []
    references = participants.set_index(key).get("enrollment_day", pd.Series(0.0, index=subjects.index))
    reference_dates, _ = clinical_timing(
        pd.Series(0.0, index=participants.index),
        "day",
        study_start,
        reference_days=participants[key].map(references),
    )
    if endpoint_event is not None and endpoint_kind != "survival":
        raise ValueError("death designation requires an observed survival endpoint")
    deaths = (
        analysis.loc[
            analysis.event_observed.astype(bool) & analysis.event_type.eq("primary_event"), [key, "observed_time"]
        ]
        .set_index(key)
        .observed_time
        if endpoint_event == "death"
        else pd.Series(dtype=float)
    )
    death_status = sdtm_death_status(
        participants[key],
        deaths,
        time_unit="day",
        study_start=study_start,
        reference_days=participants[key].map(references),
    )
    consent = participants.get("consent_time", pd.Series(np.nan, index=participants.index))
    consent_dates, _ = clinical_timing(consent, "day", study_start, reference_days=participants[key].map(references))
    completion_clocks = {
        "completion_time": "occurrence_time",
        "completion_observation_time": "observation_time",
        "completion_documentation_time": "documentation_time",
        "completion_availability_time": "availability_time",
    }
    if (set(completion_clocks) | {"completion_status"}) & set(participants):
        if completion_records is not None or not (set(completion_clocks) | {"completion_status"}).issubset(
            participants
        ):
            raise ValueError("participant closeout requires explicit status, all clocks and exactly one source")
        clocks = participants[list(completion_clocks)]
        if (clocks.notna().any(axis=1) & clocks.isna().any(axis=1)).any():
            raise ValueError("participant completion clocks must be jointly observed or unknown")
        known = clocks.notna().all(axis=1)
        if not participants.completion_status.notna().eq(known).all():
            raise ValueError("participant closeout status and clocks must be jointly observed or unknown")
        completion_records = participants.loc[known, [key, *completion_clocks, "completion_status"]].rename(
            columns={**completion_clocks, "completion_status": "status"}
        )
    completions = completion_disposition_records(completion_records) if completion_records is not None else None
    closeout_times = pd.Series(dtype=float) if completions is None else completions.set_index(key).occurrence_time
    completion_times = (
        pd.Series(dtype=float)
        if completions is None
        else completions.loc[completions.DSDECOD.eq("COMPLETED")].set_index(key).occurrence_time
    )
    if not closeout_times.index.isin(subjects.index).all():
        raise ValueError("completion form refers to an unknown participant")
    withdrawal_times = withdrawal_participation_times(intercurrent_events) if intercurrent_events is not None else None
    if withdrawal_times is not None and not withdrawal_times.index.isin(subjects.index).all():
        raise ValueError("withdrawal refers to an unknown participant")
    participation_times = participation_end_times(deaths, completion_times, withdrawals=withdrawal_times)
    reference_end_dates, _ = clinical_timing(
        participants[key].map(participation_times.combine_first(closeout_times)),
        "day",
        study_start,
        reference_days=participants[key].map(references),
    )
    participation_dates, _ = clinical_timing(
        participants[key].map(participation_times),
        "day",
        study_start,
        reference_days=participants[key].map(references),
    )
    arm = participants.assigned_arm.astype(str)
    dm = pd.DataFrame(
        {
            "STUDYID": study_id,
            "DOMAIN": "DM",
            "USUBJID": participants[key].map(subjects),
            "SUBJID": participants[key].astype(str),
            "RFSTDTC": reference_dates,
            "RFENDTC": reference_end_dates,
            "RFICDTC": consent_dates,
            "RFPENDTC": participation_dates,
            "DTHDTC": death_status.DTHDTC,
            "DTHFL": death_status.DTHFL,
            "SEX": sdtm_sex(participants.sex) if "sex" in participants else "U",
            "ARMCD": "" if observational else arm,
            "ARM": "" if observational else arm,
            "ACTARMCD": arm,
            "ACTARM": arm,
        }
    )
    adsl = pd.DataFrame(
        {
            "STUDYID": study_id,
            "USUBJID": participants[key].map(subjects),
            "TRT01P": "" if observational else arm,
            "TRT01A": arm,
            "ITTFL": "" if observational else "Y",
            "SEX": dm.SEX,
        }
    )
    dictionary = []
    supplemental = []
    if "age_years" in participants:
        age = participants.age_years
        if not pd.api.types.is_numeric_dtype(age) or not np.isfinite(age.dropna()).all() or age.lt(0).any():
            raise ValueError("recorded age_years must contain nonnegative finite ages or missing values")
        dm.insert(dm.columns.tolist().index("SEX"), "AGE", age)
        dm.insert(dm.columns.tolist().index("SEX"), "AGEU", age.map(lambda value: "YEARS" if pd.notna(value) else ""))
        adsl["AGE"] = age
        adsl["AGEU"] = dm.AGEU
        dictionary.append(
            {
                "dataset": "ADSL",
                "variable": "AGE",
                "source_field": "age_years",
                "unit": "years",
                "role": "recorded participant characteristic",
            }
        )
    for field, name in (("site_id", "SITEID"), ("country_code", "COUNTRY")):
        if field in participants:
            if name == "SITEID":
                dm.insert(dm.columns.tolist().index("AGE" if "AGE" in dm else "SEX"), name, participants[field])
            else:
                dm[name] = participants[field]
            adsl[name] = participants[field]
            dictionary.append(
                {
                    "dataset": "ADSL",
                    "variable": name,
                    "source_field": field,
                    "unit": "",
                    "role": "recorded participant characteristic",
                }
            )
    for index, field in enumerate(str(column) for column in participants if column not in required):
        if field in {"age_years", "site_id", "country_code"}:
            continue
        name = f"BASE{index + 1:03d}"
        adsl[name] = participants[field]
        dictionary.append(
            {
                "dataset": "ADSL",
                "variable": name,
                "source_field": field,
                "unit": "",
                "role": "recorded participant characteristic",
            }
        )
        values = participants[field]
        present = values.notna()
        supplemental.append(
            pd.DataFrame(
                {
                    "STUDYID": study_id,
                    "RDOMAIN": "DM",
                    "USUBJID": participants.loc[present, key].map(subjects),
                    "IDVAR": "",
                    "IDVARVAL": "",
                    "QNAM": name,
                    "QLABEL": field[:40],
                    "QVAL": values.loc[present].astype(str),
                    "QORIG": "CRF",
                    "QEVAL": "",
                }
            )
        )
    event_dispositions = (
        intercurrent_disposition_records(intercurrent_events) if intercurrent_events is not None else None
    )
    if (
        endpoint_event == "death"
        or (completions is not None and not completions.empty)
        or consent.notna().any()
        or (event_dispositions is not None and not event_dispositions.empty)
    ):
        disposition = (
            deaths.rename("occurrence_time")
            .rename_axis(key)
            .reset_index()
            .assign(DSDECOD="DEATH", DSCAT="DISPOSITION EVENT", DSSCAT="STUDY PARTICIPATION")
        )
        if completions is not None and not completions.empty:
            disposition = pd.concat((disposition, completions), ignore_index=True)
        if consent.notna().any():
            consent_records = (
                participants.loc[consent.notna(), [key, "consent_time"]]
                .rename(columns={"consent_time": "occurrence_time"})
                .assign(DSDECOD="INFORMED CONSENT OBTAINED", DSCAT="PROTOCOL MILESTONE", DSSCAT="")
            )
            disposition = pd.concat((disposition, consent_records), ignore_index=True)
        if event_dispositions is not None and not event_dispositions.empty:
            disposition = pd.concat((disposition, event_dispositions), ignore_index=True)
        tables.append(
            (
                "DS",
                sdtm_disposition(
                    disposition,
                    study_id=study_id,
                    subject_by_participant=subjects,
                    study_start=study_start,
                    time_unit="day",
                    reference_days=references,
                    epochs=epochs,
                ),
            )
        )
    tables.append(("DM", dm))
    tables.append(("ADSL", adsl))
    if supplemental:
        tables.append(("SUPPDM", pd.concat(supplemental, ignore_index=True)))

    if exposure_records is not None:
        exposure = sdtm_exposure(
            exposure_records.rename(columns={"received_arm": "received_treatment"}),
            study_id=study_id,
            subject_by_participant=subjects,
            study_start=study_start,
            time_unit="day",
            reference_days=exposure_records[key].map(references),
        )
        treatment_dates = exposure_date_bounds(
            exposure_records,
            time_unit="day",
            study_start=study_start,
            reference_days=exposure_records[key].map(references),
        )
        for standard, analysis_name in (("RFXSTDTC", "TRTSDT"), ("RFXENDTC", "TRTEDT")):
            dates = participants[key].map(treatment_dates[standard]).fillna("")
            dm.insert(dm.columns.tolist().index("RFENDTC") + 1 + int(standard == "RFXENDTC"), standard, dates)
            adsl[analysis_name] = dates
        exposure["EPOCH"] = clinical_epochs(exposure_records.occurrence_time, epochs)
        tables.append(("EX", exposure))
        tables.append(
            (
                "SUPPEX",
                pd.DataFrame(
                    {
                        "STUDYID": study_id,
                        "RDOMAIN": "EX",
                        "USUBJID": exposure.USUBJID,
                        "IDVAR": "EXSEQ",
                        "IDVARVAL": exposure.EXSEQ.astype(str),
                        "QNAM": "ELAPDAY",
                        "QLABEL": "Recorded receipt elapsed days",
                        "QVAL": exposure_records.occurrence_time.astype(str),
                        "QORIG": "CRF",
                        "QEVAL": "",
                    }
                ),
            )
        )

    if safety_events is not None:
        events = safety_events
        ae = sdtm_adverse_events(
            events,
            study_id=study_id,
            subject_by_participant=subjects,
            study_start=study_start,
            time_unit="day",
            reference_days=events[key].map(references),
            epochs=epochs,
        )
        tables.append(("AE", ae))
        supplemental_ae = []
        for field, code in (
            ("severity", "SEVSTRAT"),
            ("medical_code", "MEDCODE"),
            ("coding_system", "CODESYS"),
            ("occurrence_time", "ELAPDAY"),
            ("availability_time", "AVAILDAY"),
            ("resolution_time", "RESOLDAY"),
        ):
            if field not in events:
                continue
            present = events[field].notna()
            supplemental_ae.append(
                pd.DataFrame(
                    {
                        "STUDYID": study_id,
                        "RDOMAIN": "AE",
                        "USUBJID": ae.loc[present, "USUBJID"],
                        "IDVAR": "AESEQ",
                        "IDVARVAL": ae.loc[present, "AESEQ"].astype(str),
                        "QNAM": code,
                        "QLABEL": field,
                        "QVAL": events.loc[present, field].astype(str),
                        "QORIG": "CRF",
                        "QEVAL": "",
                    }
                )
            )
        if supplemental_ae:
            tables.append(("SUPPAE", pd.concat(supplemental_ae, ignore_index=True)))
    findings: dict[str, list[pd.DataFrame]] = {}
    derived = []
    if endpoint_kind == "survival":
        adtte = adam_time_to_event(
            analysis,
            study_id=study_id,
            test_code="PRIMARY",
            test_name=endpoint_id,
            subject_by_participant=subjects,
        )
        adtte["SRCROW"] = np.arange(len(analysis))
        tables.append(("ADTTE", adtte))
        observed = analysis.loc[analysis.event_observed.astype(bool)].copy()
        event_dates, event_days = clinical_timing(
            observed.observed_time, "day", study_start, reference_days=observed[key].map(references)
        )
        ce = pd.DataFrame(
            {
                "STUDYID": study_id,
                "DOMAIN": "CE",
                "USUBJID": observed[key].map(subjects),
                "CESEQ": 1,
                "CETERM": endpoint_id,
                "CEPRESP": "Y",
                "CEOCCUR": "Y",
                "EPOCH": clinical_epochs(observed.observed_time, epochs),
                "CESTDTC": event_dates,
                "CESTDY": event_days,
            }
        )
        tables.append(("CE", ce))
        dictionary.append(
            {
                "dataset": "ADTTE",
                "variable": "AVAL",
                "source_field": "observed_time",
                "unit": "days",
                "role": "exact observed elapsed follow-up; CNSR=0 is event",
            }
        )
        primary_fields = []
    else:
        primary_fields = [endpoint_id]
        if endpoint_id not in analysis:
            raise ValueError("declared clinical endpoint is absent")
        endpoint_values = analysis[endpoint_id].dropna().to_numpy(dtype=float)
        if endpoint_kind in {"binary", "recurrent"} and not np.isin(endpoint_values, [0, 1]).all():
            raise ValueError("binary endpoint is not coded 0/1")
        if endpoint_kind == "count" and (
            (endpoint_values < 0).any() or not np.equal(endpoint_values, np.floor(endpoint_values)).all()
        ):
            raise ValueError("count endpoint is not a nonnegative integer")
    # These are recorded ascertainment variables, not inferred latent outcomes.
    auxiliary = [
        field
        for field in (
            *additional_objects,
            "routine_observed",
            "endpoint_recovered",
            "horizon_endpoint",
            "validation_selected",
            "adjudicated_endpoint",
            "validation_stratum",
            "prior_observed_outcome",
        )
        if field in analysis and field != endpoint_id
    ]
    for index, field in enumerate([*primary_fields, *auxiliary]):
        code = "PRIMARY" if field == endpoint_id else f"AUX{index + 1:03d}"
        domain = endpoint_domain if field == endpoint_id and endpoint_kind in {"continuous", "longitudinal"} else "FA"
        value_kind: Literal["numeric", "character"] = "character" if field == "validation_stratum" else "numeric"
        values = analysis[field] if value_kind == "character" else pd.to_numeric(analysis[field], errors="raise")
        if endpoint_kind == "longitudinal" and analysis.duplicated([key, "occurrence_time"]).any():
            raise ValueError("resolve duplicate participant assessment records before projection")
        history = pd.DataFrame(
            {
                key: analysis[key],
                "assigned_arm": analysis.assigned_arm,
                "occurrence_time": (
                    analysis["occurrence_time"]
                    if "occurrence_time" in analysis
                    else analysis["assessment_time"] if "assessment_time" in analysis else assessment_time
                ),
                "observed_outcome": values,
                "observed": values.notna(),
            }
        )
        unit = (
            endpoint_unit
            if field in {endpoint_id, "adjudicated_endpoint", "prior_observed_outcome"}
            else ("" if value_kind == "character" else "indicator")
        )
        object_name = additional_objects.get(field, endpoint_object)
        test_name = (
            ("Number of events" if endpoint_kind == "count" else "Occurrence")
            if domain == "FA" and field == endpoint_id
            else field[:40]
        )
        sdtm = sdtm_findings(
            history,
            study_id=study_id,
            domain=domain,
            test_code=code,
            test_name=test_name,
            subject_by_participant=subjects,
            study_start=study_start,
            time_unit="day",
            reference_days=history[key].map(references),
            result_unit=unit,
            value_kind=value_kind,
            object_name=object_name if domain == "FA" else None,
        )
        sdtm["EPOCH"] = clinical_epochs(history.occurrence_time, epochs)
        adam = adam_findings(
            history,
            study_id=study_id,
            test_code=code,
            test_name=field[:40],
            subject_by_participant=subjects,
            study_start=study_start,
            time_unit="day",
            reference_days=history[key].map(references),
            value_kind=value_kind,
        )
        adam["ELAPDAY"] = history.occurrence_time
        adam["SRCROW"] = np.arange(len(analysis))
        adam["SRCDOM"] = domain
        adam["SRCVAR"] = f"{domain}STRESC" if value_kind == "character" else f"{domain}STRESN"
        findings.setdefault(domain, []).append(sdtm)
        derived.append(adam)
        dictionary.append(
            {
                "dataset": "ADBDS",
                "variable": code,
                "source_field": field,
                "unit": unit,
                "role": "observed outcome" if field == endpoint_id else "observed ascertainment",
                "clinical_object": object_name if domain == "FA" else "",
            }
        )
    if findings:
        for output_domain, frames in sorted(findings.items()):
            table = pd.concat(frames, ignore_index=True)
            table[f"{output_domain}SEQ"] = table.groupby("USUBJID").cumcount() + 1
            tables.append((output_domain, table))
        tables.append(("ADBDS", pd.concat(derived, ignore_index=True)))
    if output.suffix == ".zip":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, frame in sorted(tables, key=lambda item: item[0]):
                info = zipfile.ZipInfo(f"{name}.parquet", date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, frame.to_parquet(index=False))
            archive.writestr(
                zipfile.ZipInfo("variables.csv", date_time=(2020, 1, 1, 0, 0, 0)),
                pd.DataFrame(dictionary).to_csv(index=False),
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(buffer.getvalue())
        return (output,)
    with staged_directory(output) as staging:
        for name, frame in tables:
            frame.to_parquet(staging / f"{name}.parquet", index=False)
        pd.DataFrame(dictionary).to_csv(staging / "variables.csv", index=False)
    return tuple(sorted(output.iterdir()))
