"""Tests for source extraction, fitting, and held-out validation."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest
from pydantic import ValidationError

from trialagentbench_validation.external.analysis import (
    evaluate_held_out,
    evaluate_synthetic_concordance,
    fit_observable_profile,
    split_studies,
)
from trialagentbench_validation.external.contracts import (
    AACTInclusionV1,
    ConstructConcordanceV1,
    ConstructDefinitionV1,
    ConstructKind,
    ConstructMapV1,
    ExternalSourceManifestV1,
    StudyPartitionV1,
    StudySummaryV1,
    SyntheticConcordanceReportV1,
    SyntheticConstructConcordanceV1,
    SyntheticConstructRole,
)
from trialagentbench_validation.external.sources.aact import (
    extract_aact_interventional_trials,
)
from trialagentbench_validation.external.sources.rctbench import extract_rct_bench
from trialagentbench_validation.external.synthetic import (
    extract_public_synthetic_trials,
)
from trialagentbench_validation.external.workflow import compare_synthetic_concordance


def _study(index: int, *, source: str = "fixture") -> StudySummaryV1:
    return StudySummaryV1(
        study_id=f"study-{index}",
        source_id=source,
        enrollment=100 + index,
        observation_count=100 + index,
        arm_count=2,
        primary_outcome_type="continuous",
        baseline_covariate_count=3,
        baseline_missing_fraction=index / 100,
        primary_outcome_missing_fraction=index / 200,
    )


def _construct_map() -> ConstructMapV1:
    return ConstructMapV1(
        constructs=(
            ConstructDefinitionV1(
                construct_id="enrollment",
                kind=ConstructKind.ENROLLMENT,
                source_id="fixture",
                population="fixture studies",
                unit="participants",
                transformation="log1p",
                minimum_studies=2,
                minimum_synthetic_trials=2,
                compatibility_limits="Exact fixture definition.",
            ),
        )
    )


def test_study_split_and_held_out_analysis_are_reproducible() -> None:
    studies = tuple(_study(index) for index in range(12))
    partition = split_studies(studies, seed=19)
    assert partition == split_studies(studies, seed=19)
    profile = fit_observable_profile(
        studies,
        partition=partition,
        construct_map=_construct_map(),
        profile_id="fixture-v1",
        source_manifest_sha256="a" * 64,
    )
    report = evaluate_held_out(
        studies,
        partition=partition,
        construct_map=_construct_map(),
        profile=profile,
        bootstrap_replicates=250,
        seed=11,
    )
    assert report.results[0].status == "supported"
    assert report.results[0].equivalent is None
    assert "latent_causal_truth" in profile.forbidden_uses


def test_follow_up_construct_uses_the_median_duration_field() -> None:
    studies = tuple(
        _study(index).model_copy(
            update={
                "primary_outcome_type": "time-to-event",
                "follow_up_time_median": float(100 + index),
                "follow_up_time_unit": "days",
            }
        )
        for index in range(12)
    )
    construct_map = ConstructMapV1(
        constructs=(
            ConstructDefinitionV1(
                construct_id="follow-up",
                kind=ConstructKind.FOLLOW_UP_TIME,
                source_id="fixture",
                population="fixture trials",
                unit="days",
                transformation="log1p",
                minimum_studies=2,
                minimum_synthetic_trials=2,
                compatibility_limits="Explicit fixture duration unit.",
            ),
        )
    )
    partition = split_studies(studies, seed=19)

    profile = fit_observable_profile(
        studies,
        partition=partition,
        construct_map=construct_map,
        profile_id="fixture-v1",
        source_manifest_sha256="a" * 64,
    )

    assert profile.distributions[0].construct_id == "follow-up"


def test_unsupported_concordance_reports_low_study_counts_without_inference() -> None:
    row = ConstructConcordanceV1(
        construct_id="sparse-follow-up",
        n_calibration_studies=1,
        n_held_out_studies=0,
        status="unsupported",
        interpretation="Insufficient independent-study support.",
    )

    assert row.n_calibration_studies == 1
    with pytest.raises(ValueError, match="at least two studies"):
        ConstructConcordanceV1(
            construct_id="sparse-follow-up",
            n_calibration_studies=1,
            n_held_out_studies=2,
            status="supported",
            wasserstein_distance=0.1,
            bootstrap_ci_low=0.05,
            bootstrap_ci_high=0.15,
            interpretation="Invalid supported row.",
        )


def test_held_out_uncertainty_is_invariant_to_unrelated_constructs() -> None:
    studies = tuple(_study(index) for index in range(12))
    partition = split_studies(studies, seed=19)
    base_map = _construct_map()
    augmented_map = ConstructMapV1(
        constructs=(
            ConstructDefinitionV1(
                construct_id="unsupported-follow-up",
                kind=ConstructKind.FOLLOW_UP_TIME,
                source_id="fixture",
                population="fixture studies",
                unit="days",
                transformation="log1p",
                minimum_studies=2,
                minimum_synthetic_trials=2,
                compatibility_limits="No fixture study provides follow-up.",
            ),
            *base_map.constructs,
        )
    )

    def enrollment_result(construct_map: ConstructMapV1) -> ConstructConcordanceV1:
        profile = fit_observable_profile(
            studies,
            partition=partition,
            construct_map=construct_map,
            profile_id="fixture-v1",
            source_manifest_sha256="a" * 64,
        )
        report = evaluate_held_out(
            studies,
            partition=partition,
            construct_map=construct_map,
            profile=profile,
            bootstrap_replicates=250,
            seed=11,
        )
        return next(row for row in report.results if row.construct_id == "enrollment")

    assert enrollment_result(base_map) == enrollment_result(augmented_map)


def test_partition_rejects_study_leakage() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        StudyPartitionV1(
            seed=1,
            calibration_study_ids=("same", "calibration"),
            held_out_study_ids=("same", "held-out"),
        )


def test_external_source_manifest_rejects_non_quantitative_documents() -> None:
    """The public validator accepts quantitative source records only."""

    with pytest.raises(ValidationError):
        ExternalSourceManifestV1.model_validate(
            {
                "sources": [
                    {
                        "source_id": "sap",
                        "source_type": "public_document",
                        "canonical_url": "https://example.invalid/sap.pdf",
                        "snapshot_identity": "sap.pdf",
                        "sha256": "0" * 64,
                        "retrieved_at": "2026-07-24",
                        "license_status": "acquisition_only",
                        "redistribution_rationale": "Not redistributed.",
                        "role": "design_language_only",
                    }
                ]
            }
        )


def test_rct_bench_preserves_crossover_participant_unit(tmp_path: Path) -> None:
    root = tmp_path / "rct"
    (root / "cleaned_data").mkdir(parents=True)
    metadata = pd.DataFrame(
        [
            {
                "Trial_ID": 1,
                "# of Arm": 2,
                "Sample Size": 2,
                "Primary Outcome Type": "Continuous",
            }
        ]
    )
    dictionary = pd.DataFrame(
        [
            {
                "Trial_ID": 1,
                "variable_name": "Treatment",
                "variable_role": "Treatment assignment",
                "variable_type": "factor",
                "n_rows": 4,
                "n_missing": 0,
            },
            {
                "Trial_ID": 1,
                "variable_name": "YP_value",
                "variable_role": "Primary outcome",
                "variable_type": "continuous",
                "n_rows": 4,
                "n_missing": 1,
            },
            {
                "Trial_ID": 1,
                "variable_name": "X_age",
                "variable_role": "Baseline covariate",
                "variable_type": "continuous",
                "n_rows": 4,
                "n_missing": 0,
            },
        ]
    )
    with pd.ExcelWriter(root / "meta_data.xlsx") as writer:
        metadata.to_excel(writer, sheet_name="Sheet1", index=False)
    with pd.ExcelWriter(root / "data-dictionary.xlsx") as writer:
        dictionary.to_excel(writer, sheet_name="Data_Dictionary", index=False)
    pd.DataFrame(
        {
            "Participant_ID": ["P1", "P1", "P2", "P2"],
            "Treatment": ["A", "B", "A", "B"],
            "YP_value": [1.0, 2.0, 3.0, None],
            "X_age": [40, 40, 50, 50],
        }
    ).to_csv(root / "cleaned_data" / "trial1.csv", index=False)

    summary = extract_rct_bench(root, source_id="fixture", expected_trials=1)[0]

    assert summary.enrollment == 2
    assert summary.observation_count == 4
    assert summary.primary_outcome_missing_fraction == 0.25
    assert summary.age_mean == 45.0
    assert summary.age_sd == pytest.approx(7.0710678119)
    assert summary.observable_exclusions == ("BMI:no_compatible_source_variable",)


@pytest.mark.parametrize(
    ("time_column", "expected_follow_up_days", "expected_exclusion"),
    (
        ("YP_progression_free_survival_months", 3.0 * 365.25 / 12.0, None),
        ("YP_time_to_first_infection", None, "follow_up_time:source_unit_not_explicit"),
    ),
)
def test_rct_bench_requires_an_explicit_follow_up_unit(
    tmp_path: Path,
    time_column: str,
    expected_follow_up_days: float | None,
    expected_exclusion: str | None,
) -> None:
    root = tmp_path / "rct"
    (root / "cleaned_data").mkdir(parents=True)
    metadata = pd.DataFrame(
        [
            {
                "Trial_ID": 1,
                "# of Arm": 2,
                "Sample Size": 4,
                "Primary Outcome Type": "Time-to-event",
            }
        ]
    )
    dictionary = pd.DataFrame(
        [
            {
                "Trial_ID": 1,
                "variable_name": time_column,
                "variable_role": "Primary outcome",
                "variable_type": "time-to-event/continuous time",
                "n_rows": 4,
                "n_missing": 0,
            },
            {
                "Trial_ID": 1,
                "variable_name": "YP_event",
                "variable_role": "Primary outcome",
                "variable_type": "binary",
                "n_rows": 4,
                "n_missing": 0,
            },
        ]
    )
    with pd.ExcelWriter(root / "meta_data.xlsx") as writer:
        metadata.to_excel(writer, sheet_name="Sheet1", index=False)
    with pd.ExcelWriter(root / "data-dictionary.xlsx") as writer:
        dictionary.to_excel(writer, sheet_name="Data_Dictionary", index=False)
    pd.DataFrame(
        {
            "Participant_ID": ["P1", "P2", "P3", "P4"],
            time_column: [1.0, 2.0, 4.0, 5.0],
            "YP_event": [0, 1, 1, 0],
        }
    ).to_csv(root / "cleaned_data" / "trial1.csv", index=False)

    summary = extract_rct_bench(root, source_id="fixture", expected_trials=1)[0]

    assert summary.event_fraction == 0.5
    if expected_follow_up_days is None:
        assert summary.follow_up_time_median is None
        assert summary.follow_up_time_unit is None
    else:
        assert summary.follow_up_time_median == pytest.approx(expected_follow_up_days)
        assert summary.follow_up_time_unit == "days"
    if expected_exclusion is None:
        assert expected_exclusion not in summary.observable_exclusions
    else:
        assert expected_exclusion in summary.observable_exclusions


def test_aact_requires_explicit_design_contract(tmp_path: Path) -> None:
    archive = tmp_path / "aact.zip"
    studies_header = (
        "nct_id|study_type|overall_status|phase|enrollment|number_of_arms\n"
        "NCT1|INTERVENTIONAL|COMPLETED|PHASE2|120|2\n"
        "NCT2|INTERVENTIONAL|COMPLETED|PHASE1|20|2\n"
    )
    designs_header = "id|nct_id|allocation|intervention_model\n1|NCT1|RANDOMIZED|PARALLEL\n2|NCT2|RANDOMIZED|PARALLEL\n"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as handle:
        handle.writestr("studies.txt", studies_header)
        handle.writestr("designs.txt", designs_header)
    inclusion = AACTInclusionV1(
        overall_statuses=("COMPLETED",),
        phases=("PHASE2", "PHASE3"),
        allocations=("RANDOMIZED",),
        intervention_models=("PARALLEL",),
        minimum_enrollment=20,
        maximum_enrollment=10_000,
        minimum_arms=2,
        maximum_arms=4,
    )

    rows = extract_aact_interventional_trials(
        archive, source_id="aact", inclusion=inclusion
    )

    assert [row.study_id for row in rows] == ["aact:NCT1"]


def test_profile_json_is_consumable_by_construction_contract(tmp_path: Path) -> None:
    studies = tuple(_study(index) for index in range(8))
    partition = split_studies(studies, seed=3)
    profile = fit_observable_profile(
        studies,
        partition=partition,
        construct_map=_construct_map(),
        profile_id="fixture-v1",
        source_manifest_sha256="b" * 64,
    )
    path = tmp_path / "profile.json"
    path.write_text(profile.model_dump_json(), encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_id"] == "trialagentbench.selected_observable_profile/v1"


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _write_public_task(
    archive: ZipFile,
    *,
    task_id: str,
    trial_id: str,
    study_id: str,
    age_shift: float = 0.0,
    prepared: bool = True,
    include_bmi: bool = True,
    censoring_values: tuple[int, int, int, int] = (0, 1, 0, 1),
) -> None:
    prefix = f"items/{task_id}"
    archive.writestr(
        f"{prefix}/task.json",
        json.dumps(
            {
                "protocol_summary_file": "protocol_summary.json",
                "primary_paramcd": "death",
            }
        ),
    )
    archive.writestr(
        f"{prefix}/protocol_summary.json",
        json.dumps({"trial_id": trial_id, "study_id": study_id}),
    )
    values: dict[str, list[object]] = {
        "USUBJID": ["P1", "P2", "P3", "P4"],
        "AGE": [40.0 + age_shift, 50.0 + age_shift, 60.0 + age_shift, 70.0 + age_shift],
        "TRTA": ["A", "A", "B", "B"],
    }
    if include_bmi:
        values["BMI"] = [22.0, 24.0, 26.0, 28.0]
    baseline = pd.DataFrame(values)
    time_to_event = pd.DataFrame(
        {
            "USUBJID": ["P1", "P2", "P3", "P4"],
            "PARAMCD": ["death"] * 4,
            "AVAL": [20.0, 30.0, 40.0, 50.0],
            "CNSR": censoring_values,
        }
    )
    archive.writestr(
        f"{prefix}/data/ADTTE.parquet",
        _parquet_bytes(time_to_event),
    )
    if prepared:
        archive.writestr(f"{prefix}/data/ADSL.parquet", _parquet_bytes(baseline))
    else:
        archive.writestr(
            f"{prefix}/data/raw/baseline_characteristics.parquet",
            _parquet_bytes(baseline.drop(columns="TRTA")),
        )
        archive.writestr(
            f"{prefix}/data/raw/randomization.parquet",
            _parquet_bytes(baseline.loc[:, ["USUBJID", "TRTA"]]),
        )


def _write_participant_manifest(
    archive: ZipFile,
    *,
    profile_id: str | None = "profile-v1",
    profile_sha256: str | None = "a" * 64,
) -> None:
    archive.writestr(
        "manifest.json",
        json.dumps(
            {
                "applied_baseline_profile_id": profile_id,
                "applied_baseline_profile_sha256": profile_sha256,
            }
        ),
    )


def test_public_synthetic_extraction_counts_context_siblings_once(
    tmp_path: Path,
) -> None:
    release = tmp_path / "participant.zip"
    with ZipFile(release, "w", compression=ZIP_DEFLATED) as archive:
        _write_participant_manifest(archive)
        _write_public_task(
            archive,
            task_id="C1",
            trial_id="trial-1",
            study_id="study-1",
        )
        _write_public_task(
            archive,
            task_id="C2",
            trial_id="trial-1",
            study_id="study-1",
        )
        _write_public_task(
            archive,
            task_id="C3",
            trial_id="trial-1",
            study_id="study-1",
            prepared=False,
        )

    summaries, task_count, profile_id, profile_sha256 = extract_public_synthetic_trials(
        release
    )

    assert task_count == 3
    assert profile_id == "profile-v1"
    assert profile_sha256 == "a" * 64
    assert len(summaries) == 1
    assert summaries[0].enrollment == 4
    assert summaries[0].arm_count == 2
    assert summaries[0].age_mean == 55.0
    assert summaries[0].event_fraction == 0.5
    assert summaries[0].follow_up_time_median == 35.0
    assert summaries[0].follow_up_time_unit == "days"


def test_public_synthetic_extraction_rejects_inconsistent_prepared_siblings(
    tmp_path: Path,
) -> None:
    release = tmp_path / "participant.zip"
    with ZipFile(release, "w", compression=ZIP_DEFLATED) as archive:
        _write_participant_manifest(archive)
        _write_public_task(
            archive,
            task_id="C1",
            trial_id="trial-1",
            study_id="study-1",
        )
        _write_public_task(
            archive,
            task_id="C2",
            trial_id="trial-1",
            study_id="study-1",
            age_shift=1.0,
        )

    with pytest.raises(ValueError, match="prepared context views disagree"):
        extract_public_synthetic_trials(release)


def test_public_synthetic_extraction_rejects_missing_demographic_evidence(
    tmp_path: Path,
) -> None:
    release = tmp_path / "participant.zip"
    with ZipFile(release, "w", compression=ZIP_DEFLATED) as archive:
        _write_participant_manifest(archive)
        _write_public_task(
            archive,
            task_id="C1",
            trial_id="trial-1",
            study_id="study-1",
            include_bmi=False,
        )

    with pytest.raises(ValueError, match="lacks required columns"):
        extract_public_synthetic_trials(release)


def test_public_synthetic_extraction_rejects_invalid_censoring_contract(
    tmp_path: Path,
) -> None:
    release = tmp_path / "participant.zip"
    with ZipFile(release, "w", compression=ZIP_DEFLATED) as archive:
        _write_participant_manifest(archive)
        _write_public_task(
            archive,
            task_id="C1",
            trial_id="trial-1",
            study_id="study-1",
            censoring_values=(0, 1, 0, 2),
        )

    with pytest.raises(ValueError, match="CNSR"):
        extract_public_synthetic_trials(release)


@pytest.mark.parametrize("member", ("../task.json", "grader/scoring_truth.jsonl"))
def test_public_synthetic_extraction_rejects_unsafe_or_evaluator_members(
    tmp_path: Path,
    member: str,
) -> None:
    release = tmp_path / "participant.zip"
    with ZipFile(release, "w", compression=ZIP_DEFLATED) as archive:
        _write_participant_manifest(archive)
        _write_public_task(
            archive,
            task_id="C1",
            trial_id="trial-1",
            study_id="study-1",
        )
        archive.writestr(member, "{}")

    with pytest.raises(ValueError, match="unsafe participant|evaluator-only"):
        extract_public_synthetic_trials(release)


def test_public_synthetic_extraction_requires_applied_profile_provenance(
    tmp_path: Path,
) -> None:
    release = tmp_path / "participant.zip"
    with ZipFile(release, "w", compression=ZIP_DEFLATED) as archive:
        _write_public_task(
            archive,
            task_id="C1",
            trial_id="trial-1",
            study_id="study-1",
        )

    with pytest.raises(ValueError, match="applied-profile provenance"):
        extract_public_synthetic_trials(release)


def test_public_synthetic_extraction_rejects_manifest_without_profile_keys(
    tmp_path: Path,
) -> None:
    release = tmp_path / "participant.zip"
    with ZipFile(release, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", "{}")
        _write_public_task(
            archive,
            task_id="C1",
            trial_id="trial-1",
            study_id="study-1",
        )

    with pytest.raises(ValueError, match="provenance keys"):
        extract_public_synthetic_trials(release)


@pytest.mark.parametrize(
    ("profile_id", "profile_sha256", "message"),
    (
        (None, "a" * 64, "applied_baseline_profile_id"),
        ("profile-v1", None, "applied_baseline_profile_sha256"),
    ),
)
def test_public_synthetic_extraction_rejects_partial_profile_provenance(
    tmp_path: Path,
    profile_id: str | None,
    profile_sha256: str | None,
    message: str,
) -> None:
    release = tmp_path / "participant.zip"
    with ZipFile(release, "w", compression=ZIP_DEFLATED) as archive:
        _write_participant_manifest(
            archive,
            profile_id=profile_id,
            profile_sha256=profile_sha256,
        )
        _write_public_task(
            archive,
            task_id="C1",
            trial_id="trial-1",
            study_id="study-1",
        )

    with pytest.raises(ValueError, match=message):
        extract_public_synthetic_trials(release)


def test_synthetic_concordance_requires_paired_applied_profile_provenance() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        SyntheticConcordanceReportV1(
            reference_profile_id="reference-v1",
            reference_profile_sha256="a" * 64,
            applied_profile_id="applied-v1",
            participant_release_sha256="b" * 64,
            participant_task_count=1,
            independent_synthetic_trial_count=1,
            synthetic_trial_identity_sha256="c" * 64,
            results=(
                SyntheticConstructConcordanceV1(
                    construct_id="event",
                    role=SyntheticConstructRole.DESCRIPTIVE_ONLY,
                    n_synthetic_trials=0,
                    n_held_out_studies=0,
                    status="unsupported",
                    interpretation="Insufficient support.",
                ),
            ),
            limitations=("Fixture.",),
        )


def _synthetic_report(
    *,
    applied: bool,
    identity_sha256: str = "c" * 64,
    distance: float = 0.2,
) -> SyntheticConcordanceReportV1:
    return SyntheticConcordanceReportV1(
        reference_profile_id="profile-v1",
        reference_profile_sha256="a" * 64,
        applied_profile_id="profile-v1" if applied else None,
        applied_profile_sha256="a" * 64 if applied else None,
        participant_release_sha256=("d" if applied else "b") * 64,
        participant_task_count=4,
        independent_synthetic_trial_count=2,
        synthetic_trial_identity_sha256=identity_sha256,
        results=(
            SyntheticConstructConcordanceV1(
                construct_id="age_mean",
                role=SyntheticConstructRole.EXTERNALLY_FITTED,
                n_synthetic_trials=2,
                n_held_out_studies=2,
                status="supported",
                wasserstein_distance=distance,
                bootstrap_ci_low=distance / 2,
                bootstrap_ci_high=distance * 2,
                interpretation="Fixture.",
            ),
        ),
        limitations=("Fixture.",),
    )


def test_paired_synthetic_concordance_requires_and_reports_matched_worlds(
    tmp_path: Path,
) -> None:
    prefit_path = tmp_path / "prefit.json"
    selected_path = tmp_path / "selected.json"
    output_path = tmp_path / "comparison.json"
    prefit_path.write_text(
        _synthetic_report(applied=False, distance=0.3).model_dump_json(),
        encoding="utf-8",
    )
    selected_path.write_text(
        _synthetic_report(applied=True, distance=0.1).model_dump_json(),
        encoding="utf-8",
    )

    report = compare_synthetic_concordance(
        prefit_report_path=prefit_path,
        selected_report_path=selected_path,
        output_path=output_path,
    )

    assert report.results[0].selected_minus_prefit_distance == pytest.approx(-0.2)
    assert report.results[0].prefit.bootstrap_ci_high == pytest.approx(0.6)
    assert report.results[0].selected.bootstrap_ci_high == pytest.approx(0.2)
    assert output_path.is_file()


def test_paired_synthetic_concordance_rejects_different_worlds(
    tmp_path: Path,
) -> None:
    prefit_path = tmp_path / "prefit.json"
    selected_path = tmp_path / "selected.json"
    prefit_path.write_text(
        _synthetic_report(applied=False).model_dump_json(), encoding="utf-8"
    )
    selected_path.write_text(
        _synthetic_report(applied=True, identity_sha256="e" * 64).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="matched fields"):
        compare_synthetic_concordance(
            prefit_report_path=prefit_path,
            selected_report_path=selected_path,
            output_path=tmp_path / "comparison.json",
        )


def test_synthetic_concordance_distinguishes_fitted_and_protocol_controls() -> None:
    synthetic = tuple(
        StudySummaryV1(
            study_id=f"synthetic-{index}",
            source_id="trialagentbench_public",
            enrollment=100 + index,
            observation_count=100 + index,
            arm_count=2,
            primary_outcome_type="not_assessed",
            baseline_covariate_count=0,
            age_mean=50.0 + index,
            age_sd=10.0,
            bmi_mean=26.0,
            bmi_sd=4.0,
        )
        for index in range(4)
    )
    external = tuple(
        StudySummaryV1(
            study_id=f"external-{index}",
            source_id="fixture",
            enrollment=90 + index,
            observation_count=90 + index,
            arm_count=2,
            primary_outcome_type="continuous",
            baseline_covariate_count=3,
            age_mean=49.0 + index,
            age_sd=9.0,
            bmi_mean=25.0,
            bmi_sd=4.0,
        )
        for index in range(8)
    )
    partition = StudyPartitionV1(
        seed=1,
        calibration_study_ids=tuple(row.study_id for row in external[:4]),
        held_out_study_ids=tuple(row.study_id for row in external[4:]),
    )
    construct_map = ConstructMapV1(
        constructs=(
            ConstructDefinitionV1(
                construct_id="age",
                kind=ConstructKind.AGE_MEAN,
                source_id="fixture",
                population="fixture",
                unit="years",
                transformation="identity",
                minimum_studies=2,
                minimum_synthetic_trials=2,
                compatibility_limits="fixture",
                construction_parameter="baseline.age.location_years",
            ),
            ConstructDefinitionV1(
                construct_id="arms",
                kind=ConstructKind.ARM_COUNT,
                source_id="fixture",
                population="fixture",
                unit="arms",
                transformation="identity",
                minimum_studies=2,
                minimum_synthetic_trials=2,
                compatibility_limits="fixture",
            ),
        )
    )

    results = evaluate_synthetic_concordance(
        synthetic,
        external,
        partition=partition,
        construct_map=construct_map,
        bootstrap_replicates=200,
    )

    assert results[0].role.value == "externally_fitted"
    assert results[0].calibration_reference_p95 is not None
    assert results[0].calibration_reference_tail_probability is not None
    assert results[0].within_calibration_reference is not None
    assert results[1].role.value == "protocol_control"
    assert results[1].calibration_reference_p95 is None
