"""Streaming extraction of aggregate trial design observables from AACT."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from zipfile import ZipFile

from trialagentbench_validation.external.contracts import (
    AACTInclusionV1,
    StudySummaryV1,
)


def extract_aact_interventional_trials(
    archive_path: Path,
    *,
    source_id: str,
    inclusion: AACTInclusionV1,
) -> tuple[StudySummaryV1, ...]:
    """Extract enrollment and arm counts without materializing the AACT archive."""

    summaries: list[StudySummaryV1] = []
    with ZipFile(archive_path) as archive:
        designs = _design_index(archive)
        raw = archive.open("studies.txt")
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="|")
        required = {
            "nct_id",
            "study_type",
            "overall_status",
            "phase",
            "enrollment",
            "number_of_arms",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"AACT studies schema drift: {sorted(missing)}")
        for row in reader:
            if row["study_type"].upper() != "INTERVENTIONAL":
                continue
            if row["overall_status"].upper() not in inclusion.overall_statuses:
                continue
            if row["phase"].upper() not in inclusion.phases:
                continue
            design = designs.get(row["nct_id"])
            if design is None:
                continue
            if (
                design[0] not in inclusion.allocations
                or design[1] not in inclusion.intervention_models
            ):
                continue
            try:
                enrollment = int(row["enrollment"])
                arm_count = int(row["number_of_arms"])
            except (TypeError, ValueError):
                continue
            if (
                not inclusion.minimum_enrollment
                <= enrollment
                <= inclusion.maximum_enrollment
            ):
                continue
            if not inclusion.minimum_arms <= arm_count <= inclusion.maximum_arms:
                continue
            summaries.append(
                StudySummaryV1(
                    study_id=f"aact:{row['nct_id']}",
                    source_id=source_id,
                    enrollment=enrollment,
                    observation_count=enrollment,
                    arm_count=arm_count,
                    primary_outcome_type="aggregate_registry_record",
                    baseline_covariate_count=0,
                )
            )
    if not summaries:
        raise ValueError("AACT extraction produced no eligible studies")
    return tuple(summaries)


def _design_index(archive: ZipFile) -> dict[str, tuple[str, str]]:
    with archive.open("designs.txt") as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="|")
        required = {"nct_id", "allocation", "intervention_model"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"AACT designs schema drift: {sorted(missing)}")
        return {
            row["nct_id"]: (
                row["allocation"].upper(),
                row["intervention_model"].upper(),
            )
            for row in reader
        }


__all__ = ["extract_aact_interventional_trials"]
