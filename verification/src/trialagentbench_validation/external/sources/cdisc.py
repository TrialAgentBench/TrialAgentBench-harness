"""Verify transport and analysis coherence in the official CDISC pilot."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_validation.io import sha256_file


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CDISCTransportPairV1(_FrozenModel):
    """Cell-level parity result for one XPT and Dataset-JSON pair."""

    layer: Literal["SDTM", "ADaM"]
    dataset: str = Field(pattern=r"^[A-Z0-9]+$")
    rows: int = Field(ge=1)
    columns: int = Field(ge=1)
    compared_cells: int = Field(ge=1)
    mismatched_cells: int = Field(ge=0)
    xpt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_json_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CDISCReferenceEvidenceV1(_FrozenModel):
    """Bounded CDISC transport, metadata, and analysis workflow evidence."""

    schema_id: Literal["trialagentbench.cdisc_reference_evidence/v1"] = (
        "trialagentbench.cdisc_reference_evidence/v1"
    )
    source_repository: Literal["cdisc-org/sdtm-adam-pilot-project"] = (
        "cdisc-org/sdtm-adam-pilot-project"
    )
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    pairs: tuple[CDISCTransportPairV1, ...] = Field(min_length=8, max_length=8)
    rows_compared: int = Field(ge=1)
    cells_compared: int = Field(ge=1)
    mismatched_cells: int = Field(ge=0)
    key_violations: int = Field(ge=0)
    subject_reference_violations: int = Field(ge=0)
    define_xml_datasets_required: int = Field(ge=8)
    define_xml_datasets_present: int = Field(ge=8)
    sdtm_active_n: int = Field(ge=1)
    sdtm_reference_n: int = Field(ge=1)
    adam_active_n: int = Field(ge=1)
    adam_reference_n: int = Field(ge=1)
    sdtm_adverse_event_discontinuation_risk_difference: float = Field(
        allow_inf_nan=False
    )
    adam_adverse_event_discontinuation_risk_difference: float = Field(
        allow_inf_nan=False
    )
    analysis_absolute_difference: float = Field(ge=0, allow_inf_nan=False)
    negative_controls_detected: int = Field(ge=0)
    negative_controls_total: int = Field(ge=5)


@dataclass(frozen=True)
class _Dataset:
    layer: Literal["SDTM", "ADaM"]
    name: str
    relative_stem: str
    keys: tuple[str, ...]


_DATASETS = (
    _Dataset("SDTM", "DM", "tabulations/sdtm/dm", ("STUDYID", "USUBJID")),
    _Dataset("SDTM", "DS", "tabulations/sdtm/ds", ("STUDYID", "USUBJID", "DSSEQ")),
    _Dataset(
        "SDTM",
        "AE",
        "tabulations/sdtm/ae",
        ("STUDYID", "USUBJID", "AESEQ"),
    ),
    _Dataset(
        "SDTM",
        "EX",
        "tabulations/sdtm/ex",
        ("STUDYID", "USUBJID", "EXSEQ"),
    ),
    _Dataset(
        "SDTM",
        "SV",
        "tabulations/sdtm/sv",
        ("STUDYID", "USUBJID", "VISITNUM", "SVSTDTC"),
    ),
    _Dataset("ADaM", "ADSL", "analysis/adam/datasets/adsl", ("STUDYID", "USUBJID")),
    _Dataset(
        "ADaM",
        "ADAE",
        "analysis/adam/datasets/adae",
        ("STUDYID", "USUBJID", "AESEQ"),
    ),
    _Dataset(
        "ADaM",
        "ADTTE",
        "analysis/adam/datasets/adtte",
        ("STUDYID", "USUBJID", "PARAMCD"),
    ),
)


def verify_cdisc_reference(
    package_root: Path,
    *,
    source_commit: str,
) -> CDISCReferenceEvidenceV1:
    """Verify the selected official CDISC pilot workflow from exact bytes."""

    if len(source_commit) != 40:
        raise ValueError("CDISC source_commit must be a full 40-character commit.")
    tables: dict[str, pd.DataFrame] = {}
    pairs = []
    key_violations = 0
    for dataset in _DATASETS:
        xpt_path = package_root / f"{dataset.relative_stem}.xpt"
        json_path = package_root / f"{dataset.relative_stem}.json"
        if not xpt_path.is_file() or not json_path.is_file():
            raise FileNotFoundError(
                f"CDISC transport pair is missing for {dataset.name}."
            )
        xpt = pd.read_sas(xpt_path, format="xport", encoding="utf-8")
        dataset_json, metadata = _read_dataset_json(json_path)
        if list(xpt.columns) != list(dataset_json.columns):
            raise ValueError(f"{dataset.name} XPT and Dataset-JSON columns differ.")
        mismatches = _transport_mismatches(xpt, dataset_json, metadata=metadata)
        key_violations += _key_violations(xpt, keys=dataset.keys)
        pairs.append(
            CDISCTransportPairV1(
                layer=dataset.layer,
                dataset=dataset.name,
                rows=len(xpt),
                columns=len(xpt.columns),
                compared_cells=xpt.shape[0] * xpt.shape[1],
                mismatched_cells=mismatches,
                xpt_sha256=sha256_file(xpt_path),
                dataset_json_sha256=sha256_file(json_path),
            )
        )
        tables[dataset.name] = xpt

    subject_violations = _subject_reference_violations(tables)
    required_datasets = {dataset.name for dataset in _DATASETS}
    defined_datasets = set()
    for relative_path in (
        "tabulations/sdtm/define.xml",
        "analysis/adam/datasets/define.xml",
    ):
        defined_datasets.update(_define_dataset_names(package_root / relative_path))
    missing_definitions = required_datasets - defined_datasets
    if missing_definitions:
        raise ValueError(
            f"Define-XML omits selected datasets: {sorted(missing_definitions)}"
        )

    sdtm_effect, sdtm_active_n, sdtm_reference_n = _sdtm_risk_difference(
        tables["DM"], tables["DS"]
    )
    adam_effect, adam_active_n, adam_reference_n = _adam_risk_difference(tables["ADSL"])
    negative_controls = _negative_controls(
        tables=tables,
        defined_datasets=defined_datasets,
    )
    return CDISCReferenceEvidenceV1(
        source_commit=source_commit,
        pairs=tuple(pairs),
        rows_compared=sum(row.rows for row in pairs),
        cells_compared=sum(row.compared_cells for row in pairs),
        mismatched_cells=sum(row.mismatched_cells for row in pairs),
        key_violations=key_violations,
        subject_reference_violations=subject_violations,
        define_xml_datasets_required=len(required_datasets),
        define_xml_datasets_present=len(required_datasets & defined_datasets),
        sdtm_active_n=sdtm_active_n,
        sdtm_reference_n=sdtm_reference_n,
        adam_active_n=adam_active_n,
        adam_reference_n=adam_reference_n,
        sdtm_adverse_event_discontinuation_risk_difference=sdtm_effect,
        adam_adverse_event_discontinuation_risk_difference=adam_effect,
        analysis_absolute_difference=abs(sdtm_effect - adam_effect),
        negative_controls_detected=sum(negative_controls),
        negative_controls_total=len(negative_controls),
    )


def _read_dataset_json(
    path: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns = payload.get("columns")
    rows = payload.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError(f"Dataset-JSON has invalid columns or rows: {path.name}")
    names = [column.get("name") for column in columns if isinstance(column, dict)]
    if len(names) != len(columns) or any(not isinstance(name, str) for name in names):
        raise ValueError(f"Dataset-JSON has invalid column metadata: {path.name}")
    frame = pd.DataFrame(rows, columns=names)
    if payload.get("records") != len(frame):
        raise ValueError(f"Dataset-JSON record count does not match rows: {path.name}")
    metadata = {str(column["name"]): column for column in columns}
    return frame, metadata


def _transport_mismatches(
    xpt: pd.DataFrame,
    dataset_json: pd.DataFrame,
    *,
    metadata: dict[str, dict[str, object]],
) -> int:
    if xpt.shape != dataset_json.shape:
        raise ValueError("XPT and Dataset-JSON shapes differ.")
    mismatches = 0
    for raw_column in xpt.columns:
        column = str(raw_column)
        data_type = metadata[column].get("dataType")
        left = xpt[column]
        right = dataset_json[column]
        if data_type in {"integer", "float", "decimal"}:
            mismatches += int(
                (
                    ~np.isclose(
                        pd.to_numeric(left),
                        pd.to_numeric(right),
                        rtol=0,
                        atol=1e-10,
                        equal_nan=True,
                    )
                ).sum()
            )
        elif data_type == "date":
            if pd.api.types.is_numeric_dtype(left):
                converted = pd.to_datetime(
                    left, unit="D", origin="1960-01-01"
                ).dt.strftime("%Y-%m-%d")
            else:
                converted = left.fillna("").astype(str)
            mismatches += int(
                converted.fillna("").ne(right.fillna("").astype(str)).sum()
            )
        else:
            mismatches += int(
                left.fillna("").astype(str).ne(right.fillna("").astype(str)).sum()
            )
    return mismatches


def _key_violations(frame: pd.DataFrame, *, keys: tuple[str, ...]) -> int:
    if missing := sorted(set(keys) - set(frame.columns)):
        raise ValueError(f"CDISC key columns are missing: {missing}")
    return int(frame.loc[:, list(keys)].isna().any(axis=1).sum()) + int(
        frame.duplicated(list(keys)).sum()
    )


def _subject_reference_violations(tables: dict[str, pd.DataFrame]) -> int:
    violations = 0
    for subject_table, domains in (
        ("DM", ("DS", "AE", "EX", "SV")),
        ("ADSL", ("ADAE", "ADTTE")),
    ):
        subjects = set(tables[subject_table]["USUBJID"].astype(str))
        for domain in domains:
            violations += len(set(tables[domain]["USUBJID"].astype(str)) - subjects)
    return violations


def _define_dataset_names(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Define-XML file is missing: {path}")
    root = ET.parse(path).getroot()
    return {
        str(element.attrib["Name"]).upper()
        for element in root.iter()
        if element.tag.endswith("ItemGroupDef") and "Name" in element.attrib
    }


def _sdtm_risk_difference(
    dm: pd.DataFrame,
    ds: pd.DataFrame,
) -> tuple[float, int, int]:
    events = set(ds.loc[ds["DSDECOD"].eq("ADVERSE EVENT"), "USUBJID"].astype(str))
    subjects = dm.loc[:, ["USUBJID", "ARM"]].copy()
    subjects["_event"] = subjects["USUBJID"].astype(str).isin(events)
    return _risk_difference(subjects, treatment="ARM")


def _adam_risk_difference(adsl: pd.DataFrame) -> tuple[float, int, int]:
    subjects = adsl.loc[adsl["SAFFL"].eq("Y"), ["TRT01A", "DSRAEFL"]].copy()
    subjects["_event"] = subjects["DSRAEFL"].eq("Y")
    return _risk_difference(subjects, treatment="TRT01A")


def _risk_difference(
    frame: pd.DataFrame,
    *,
    treatment: str,
) -> tuple[float, int, int]:
    active = frame.loc[frame[treatment].eq("Xanomeline High Dose")]
    reference = frame.loc[frame[treatment].eq("Placebo")]
    if active.empty or reference.empty:
        raise ValueError("CDISC analysis requires active and placebo subjects.")
    return (
        float(active["_event"].mean() - reference["_event"].mean()),
        len(active),
        len(reference),
    )


def _negative_controls(
    *,
    tables: dict[str, pd.DataFrame],
    defined_datasets: set[str],
) -> tuple[bool, ...]:
    duplicate = pd.concat([tables["DM"], tables["DM"].iloc[[0]]], ignore_index=True)
    orphan = tables["AE"].copy()
    orphan.loc[orphan.index[0], "USUBJID"] = "UNDECLARED-SUBJECT"
    changed = tables["DS"].copy()
    changed.loc[changed.index[0], "DSDECOD"] = "MUTATED"
    changed_adsl = tables["ADSL"].copy()
    active_event = changed_adsl[
        changed_adsl["TRT01A"].eq("Xanomeline High Dose")
        & changed_adsl["SAFFL"].eq("Y")
        & changed_adsl["DSRAEFL"].eq("Y")
    ].index
    if len(active_event) == 0:
        raise ValueError(
            "CDISC negative control requires an active discontinuation event."
        )
    changed_adsl.loc[active_event[0], "DSRAEFL"] = "N"
    changed_transport_detected = (
        _transport_mismatches(
            tables["DS"],
            changed,
            metadata={
                str(column): {"dataType": "string"} for column in tables["DS"].columns
            },
        )
        > 0
    )
    return (
        _key_violations(duplicate, keys=("STUDYID", "USUBJID")) > 0,
        _subject_reference_violations({**tables, "AE": orphan}) > 0,
        changed_transport_detected,
        bool({"DM"} - (defined_datasets - {"DM"})),
        abs(
            _sdtm_risk_difference(tables["DM"], tables["DS"])[0]
            - _adam_risk_difference(changed_adsl)[0]
        )
        > 0,
    )


__all__ = [
    "CDISCReferenceEvidenceV1",
    "CDISCTransportPairV1",
    "verify_cdisc_reference",
]
