"""Tests for independent participant-data analysis reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from trialagentbench_validation.trialeval.reconstruction import (
    declared_contrast_arm_aliases_v1,
    load_public_analysis_tables_v1,
    map_declared_randomized_arms_v1,
    reconstruct_public_analysis_tables_v1,
)


def _write_reconstruction_zip(
    path: Path,
    *,
    allowed_sources: list[str] | None = None,
    site_randomization: bool = False,
    relative_days: bool = False,
    conflicting_site_assignment: bool = False,
    followup_horizon_dy: float = 20.0,
    control_arm_id: str = "control",
    treated_arm_id: str = "treated",
    event_after_contact: bool = False,
    secondary_endpoint_first: bool = False,
    mixed_date_formats: bool = False,
    public_root_prefix: str = "",
) -> Path:
    task_id = "TASK001"
    root = path / "surface"
    item = root / "items" / task_id
    raw = item / "data" / "raw"
    raw.mkdir(parents=True)
    sources = allowed_sources
    if sources is None:
        sources = [
            "data/raw/randomization.parquet",
            "data/raw/disposition.parquet",
            "data/raw/endpoint_adjudication.parquet",
            "data/raw/baseline_characteristics.parquet",
        ]
        if site_randomization and not relative_days:
            sources.append("data/raw/visits.parquet")
    (item / "reconstruction_task.json").write_text(
        json.dumps({"allowed_sources": sources}), encoding="utf-8"
    )
    (item / "task.json").write_text(
        json.dumps(
            {
                "primary_control_arm_id": control_arm_id,
                "primary_treated_arm_id": treated_arm_id,
                "primary_endpoint_term": "Primary endpoint",
            }
        ),
        encoding="utf-8",
    )
    (item / "protocol_summary.json").write_text(
        json.dumps(
            {
                "followup_horizon_dy": followup_horizon_dy,
                "arms": [
                    {"arm_id": control_arm_id, "label": "Control"},
                    {"arm_id": treated_arm_id, "label": "Treated"},
                ],
            }
        ),
        encoding="utf-8",
    )
    if site_randomization:
        site_ids = (
            ["SITE1", "SITE2", "SITE1"]
            if conflicting_site_assignment
            else ["SITE1", "SITE2"]
        )
        arms = (
            ["Control", "Treated", "Treated"]
            if conflicting_site_assignment
            else ["Control", "Treated"]
        )
        pd.DataFrame(
            {"SITEID": site_ids, "ARM": arms, "RFSTDTC": ["2025-01-01"] * len(site_ids)}
        ).to_parquet(raw / "randomization.parquet")
        disposition = {"USUBJID": ["C1", "T1"], "SITEID": ["SITE1", "SITE2"]}
        disposition["LAST_CONTACT_DY" if relative_days else "LAST_CONTACT_DTC"] = (
            [20.0, 20.0] if relative_days else ["2026-01-21", "2026-02-21"]
        )
        pd.DataFrame(disposition).to_parquet(raw / "disposition.parquet")
        if not relative_days:
            pd.DataFrame(
                {
                    "USUBJID": ["C1", "T1"],
                    "VISITNUM": [0, 0],
                    "VISITDTC": [
                        "2026-01-01",
                        "2026-02-01 00:00:00" if mixed_date_formats else "2026-02-01",
                    ],
                }
            ).to_parquet(raw / "visits.parquet")
    else:
        pd.DataFrame(
            {
                "USUBJID": ["C1", "T1"],
                "ARM": ["Control", "Treated"],
                "RFSTDTC": ["2026-01-01", "2026-01-01"],
            }
        ).to_parquet(raw / "randomization.parquet")
        pd.DataFrame(
            {
                "USUBJID": ["C1", "T1"],
                "LAST_CONTACT_DTC": ["2026-01-21", "2026-01-21"],
            }
        ).to_parquet(raw / "disposition.parquet")
    adjudication: dict[str, list[object]] = {
        "USUBJID": ["T1", "C1"],
        "ENDPOINT_TERM": ["Primary endpoint", "Primary endpoint"],
        "CLINICAL_CERTAINTY": ["definite", "possible"],
        "SOURCE_CONSISTENCY": ["consistent", "consistent"],
        "EXCLUSIONARY_REVIEW_FINDING": ["none", "none"],
    }
    if relative_days:
        adjudication["EVENT_WINDOW_START_DY"] = [
            21.0 if event_after_contact else 9.0,
            4.0,
        ]
        adjudication["EVENT_WINDOW_END_DY"] = [
            23.0 if event_after_contact else 11.0,
            6.0,
        ]
    else:
        adjudication["EVENT_WINDOW_START_DTC"] = [
            "2026-02-10" if site_randomization else "2026-01-10",
            "2026-01-05",
        ]
        adjudication["EVENT_WINDOW_END_DTC"] = [
            "2026-02-12" if site_randomization else "2026-01-12",
            "2026-01-07",
        ]
    pd.DataFrame(adjudication).to_parquet(raw / "endpoint_adjudication.parquet")
    if secondary_endpoint_first:
        frame = pd.read_parquet(raw / "endpoint_adjudication.parquet")
        secondary = frame.iloc[[0]].copy()
        secondary["ENDPOINT_TERM"] = "Secondary endpoint"
        secondary["EVENT_WINDOW_START_DTC"] = "2026-01-03"
        secondary["EVENT_WINDOW_END_DTC"] = "2026-01-05"
        pd.concat([secondary, frame], ignore_index=True).to_parquet(
            raw / "endpoint_adjudication.parquet"
        )
    pd.DataFrame({"USUBJID": ["C1", "T1"], "AGE": [60.0, 65.0]}).to_parquet(
        raw / "baseline_characteristics.parquet"
    )
    archive = path / "public.zip"
    with ZipFile(archive, "w") as output:
        for member in sorted(root.rglob("*")):
            if member.is_file():
                output.write(
                    member, f"{public_root_prefix}{member.relative_to(root).as_posix()}"
                )
    return archive


def test_public_reconstruction_recovers_event_midpoint_and_censoring(
    tmp_path: Path,
) -> None:
    archive = _write_reconstruction_zip(tmp_path)

    with ZipFile(archive) as public:
        adsl, adtte = reconstruct_public_analysis_tables_v1(
            public=public, task_id="TASK001", paramcd="death"
        )

    assert dict(zip(adsl["USUBJID"], adsl["TRTA"], strict=True)) == {
        "C1": "control",
        "T1": "treated",
    }
    assert dict(zip(adsl["USUBJID"], adsl["AGE"], strict=True)) == {
        "C1": 60.0,
        "T1": 65.0,
    }
    rows = adtte.set_index("USUBJID")
    assert rows.loc["C1", "CNSR"] == 1
    assert rows.loc["C1", "AVAL"] == pytest.approx(20.0)
    assert rows.loc["T1", "CNSR"] == 0
    assert rows.loc["T1", "AVAL"] == pytest.approx(10.0)
    assert set(rows["PARAMCD"]) == {"death"}


def test_public_reconstruction_accepts_role_archive_root(tmp_path: Path) -> None:
    archive = _write_reconstruction_zip(tmp_path, public_root_prefix="public/")

    with ZipFile(archive) as public:
        adsl, adtte = reconstruct_public_analysis_tables_v1(
            public=public, task_id="TASK001", paramcd="death"
        )

    assert set(adsl["TRTA"]) == {"control", "treated"}
    assert adtte.set_index("USUBJID").loc["T1", "AVAL"] == pytest.approx(10.0)


def test_public_reconstruction_rejects_undeclared_required_source(
    tmp_path: Path,
) -> None:
    archive = _write_reconstruction_zip(
        tmp_path,
        allowed_sources=[
            "data/raw/randomization.parquet",
            "data/raw/disposition.parquet",
        ],
    )

    with (
        ZipFile(archive) as public,
        pytest.raises(ValueError, match="omits a required reconstruction source"),
    ):
        reconstruct_public_analysis_tables_v1(
            public=public, task_id="TASK001", paramcd="death"
        )


def test_public_reconstruction_uses_subject_baseline_for_site_randomization(
    tmp_path: Path,
) -> None:
    archive = _write_reconstruction_zip(tmp_path, site_randomization=True)

    with ZipFile(archive) as public:
        _, adtte = reconstruct_public_analysis_tables_v1(
            public=public, task_id="TASK001", paramcd="death"
        )

    rows = adtte.set_index("USUBJID")
    assert rows.loc["C1", "AVAL"] == pytest.approx(20.0)
    assert rows.loc["T1", "AVAL"] == pytest.approx(10.0)


@pytest.mark.filterwarnings("error:Could not infer format")
def test_public_reconstruction_accepts_mixed_iso_date_precision(tmp_path: Path) -> None:
    archive = _write_reconstruction_zip(
        tmp_path,
        site_randomization=True,
        mixed_date_formats=True,
    )

    with ZipFile(archive) as public:
        _, adtte = reconstruct_public_analysis_tables_v1(
            public=public, task_id="TASK001", paramcd="death"
        )

    rows = adtte.set_index("USUBJID")
    assert rows.loc["C1", "AVAL"] == pytest.approx(20.0)
    assert rows.loc["T1", "AVAL"] == pytest.approx(10.0)


def test_public_reconstruction_does_not_require_dates_for_relative_day_sources(
    tmp_path: Path,
) -> None:
    archive = _write_reconstruction_zip(
        tmp_path, site_randomization=True, relative_days=True
    )

    with ZipFile(archive) as public:
        _, adtte = reconstruct_public_analysis_tables_v1(
            public=public, task_id="TASK001", paramcd="death"
        )

    rows = adtte.set_index("USUBJID")
    assert rows.loc["C1", "AVAL"] == pytest.approx(20.0)
    assert rows.loc["T1", "AVAL"] == pytest.approx(10.0)


def test_public_reconstruction_applies_protocol_followup_horizon(
    tmp_path: Path,
) -> None:
    archive = _write_reconstruction_zip(tmp_path, followup_horizon_dy=15.0)

    with ZipFile(archive) as public:
        _, adtte = reconstruct_public_analysis_tables_v1(
            public=public, task_id="TASK001", paramcd="death"
        )

    rows = adtte.set_index("USUBJID")
    assert rows.loc["C1", "AVAL"] == pytest.approx(15.0)
    assert rows.loc["T1", "AVAL"] == pytest.approx(10.0)


def test_public_reconstruction_censors_events_after_last_contact(
    tmp_path: Path,
) -> None:
    archive = _write_reconstruction_zip(
        tmp_path,
        site_randomization=True,
        relative_days=True,
        event_after_contact=True,
    )

    with ZipFile(archive) as public:
        _, adtte = reconstruct_public_analysis_tables_v1(
            public=public, task_id="TASK001", paramcd="death"
        )

    treated = adtte.set_index("USUBJID").loc["T1"]
    assert treated["CNSR"] == 1
    assert treated["AVAL"] == pytest.approx(20.0)


def test_public_reconstruction_filters_to_primary_endpoint(tmp_path: Path) -> None:
    archive = _write_reconstruction_zip(tmp_path, secondary_endpoint_first=True)

    with ZipFile(archive) as public:
        _, adtte = reconstruct_public_analysis_tables_v1(
            public=public, task_id="TASK001", paramcd="death"
        )

    treated = adtte.set_index("USUBJID").loc["T1"]
    assert treated["AVAL"] == pytest.approx(10.0)


def test_public_loader_rejects_incomplete_analysis_ready_surface(
    tmp_path: Path,
) -> None:
    archive = _write_reconstruction_zip(tmp_path)
    rewritten = tmp_path / "incomplete.zip"
    with ZipFile(archive) as source, ZipFile(rewritten, "w") as output:
        for member in source.namelist():
            output.writestr(member, source.read(member))
        output.writestr("items/TASK001/data/ADSL.parquet", b"not-read")

    with (
        ZipFile(rewritten) as public,
        pytest.raises(ValueError, match="require both ADSL and ADTTE"),
    ):
        load_public_analysis_tables_v1(
            public=public, task_id="TASK001", paramcd="death"
        )


def test_public_reconstruction_rejects_conflicting_site_assignments(
    tmp_path: Path,
) -> None:
    archive = _write_reconstruction_zip(
        tmp_path, site_randomization=True, conflicting_site_assignment=True
    )

    with (
        ZipFile(archive) as public,
        pytest.raises(ValueError, match="conflicting assignments"),
    ):
        reconstruct_public_analysis_tables_v1(
            public=public, task_id="TASK001", paramcd="death"
        )


def test_declared_arm_mapping_does_not_infer_semantics_from_labels() -> None:
    frame = pd.DataFrame({"ARM": ["experimental", "placebo"]})

    mapped = map_declared_randomized_arms_v1(
        frame,
        control_arm_id="active",
        treated_arm_id="control_named_treatment",
        control_arm_aliases={"experimental"},
        treated_arm_aliases={"placebo"},
    )

    assert mapped.tolist() == ["control", "treated"]


def test_declared_arm_mapping_rejects_aliases_not_declared_by_task() -> None:
    frame = pd.DataFrame({"ARM": ["placebo", "experimental"]})

    with pytest.raises(ValueError, match="task IDs or protocol-declared aliases"):
        map_declared_randomized_arms_v1(
            frame,
            control_arm_id="arm_a",
            treated_arm_id="arm_b",
            control_arm_aliases={"arm_a"},
            treated_arm_aliases={"arm_b"},
        )


@pytest.mark.parametrize(
    "assignments",
    [
        ["arm_a", "arm_a"],
        ["arm_a", "arm_b", "arm_c"],
        ["arm_a", None],
    ],
)
def test_declared_arm_mapping_rejects_incomplete_or_out_of_contrast_assignments(
    assignments: list[str | None],
) -> None:
    with pytest.raises(ValueError, match="task IDs or protocol-declared aliases"):
        map_declared_randomized_arms_v1(
            pd.DataFrame({"ARM": assignments}),
            control_arm_id="arm_a",
            treated_arm_id="arm_b",
            control_arm_aliases={"arm_a"},
            treated_arm_aliases={"arm_b"},
        )


def test_declared_arm_crosswalk_rejects_alias_collision_with_third_arm() -> None:
    protocol: dict[str, object] = {
        "arms": [
            {"arm_id": "arm_a", "label": "arm_c"},
            {"arm_id": "arm_b", "label": "Treatment B"},
            {"arm_id": "arm_c", "label": "Treatment C"},
        ]
    }

    with pytest.raises(ValueError, match="ambiguous"):
        declared_contrast_arm_aliases_v1(
            protocol,
            control_arm_id="arm_a",
            treated_arm_id="arm_b",
        )
