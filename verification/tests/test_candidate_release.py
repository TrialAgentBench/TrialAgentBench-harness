"""Candidate-release identity and exact-membership tests."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from zipfile import ZipFile

import pandas as pd
import pytest

from trialagentbench_validation.candidate_release import (
    _CandidateReleaseStatisticsV1,
    _context_rows,
    _public_wheels,
    _route_disposition_rows,
    _route_result_kind_rows,
    _SimulationPropertyV1,
    _stable_analysis_hash,
    _stratified_characteristic_rows,
    _trialdev_characteristics,
    _trialdev_lane_rows,
    _trialeval_characteristics,
    _write_figure,
)
from trialagentbench_validation.cli import main as validation_main
from trialagentbench_validation.contracts.candidate_release import (
    CandidateIdentityV1,
    CandidatePublicWheelV1,
    CandidateRoleArchiveV1,
    CandidateValidationBundleV1,
    verify_candidate_validation_bundle,
)
from trialagentbench_validation.contracts.simulation_validation_bundle import (
    ValidationArtifactV1,
    ValidationFigureV1,
)
from trialagentbench_validation.external.release.artifacts import (
    ArtifactDigestV1,
    ExternalArtifactManifestV1,
)
from trialagentbench_validation.grader_concordance import (
    GraderConcordanceReportV1,
    TrialDevLaneGradeV1,
)
from trialagentbench_validation.io import sha256_file, write_model
from trialagentbench_validation.recovery import RecoverabilityReportV1


def _checksum(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()


def _artifact(root: Path, relative_path: str, media_type: str) -> ValidationArtifactV1:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{relative_path}\n", encoding="utf-8")
    return ValidationArtifactV1(
        relative_path=relative_path,
        sha256=sha256_file(path),
        media_type=media_type,
    )


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    stream = BytesIO()
    frame.to_parquet(stream, index=False)
    return stream.getvalue()


def _identity() -> CandidateIdentityV1:
    roles = tuple(
        sorted(
            (
                CandidateRoleArchiveV1(
                    suite=suite,
                    role=role,
                    relative_path=f"public/{suite}/{role}.zip",
                    sha256="a" * 64,
                )
                for suite in ("trialeval", "trialdev")
                for role in ("participant", "evaluator", "verification")
            ),
            key=lambda row: (row.suite, row.role),
        )
    )
    payload = {
        "schema_id": "trialagentbench.candidate_identity/v1",
        "release_id": "candidate-1",
        "source_commit": "b" * 40,
        "environment_lock_sha256": "c" * 64,
        "release_manifest_sha256": "d" * 64,
        "staged_manifest_sha256": "e" * 64,
        "seed_tree_sha256": "f" * 64,
        "materialization_census_sha256": "1" * 64,
        "root_seed": 17,
        "trialeval_item_count": 500,
        "trialdev_scenario_count": 50,
        "role_archives": [role.model_dump(mode="json") for role in roles],
        "public_wheels": [
            CandidatePublicWheelV1(
                package=package,
                relative_path=f"public/packages/{package}/{package}-0.1.0-py3-none-any.whl",
                sha256=character * 64,
            ).model_dump(mode="json")
            for package, character in (
                ("trialagentbench-harness", "4"),
                ("trialagentbench-validation", "5"),
            )
        ],
    }
    return CandidateIdentityV1(**payload, checksum=_checksum(payload))


def test_public_wheels_accept_product_only_release(tmp_path: Path) -> None:
    public = tmp_path / "public"
    artifacts = []
    for package in ("trialagentbench-harness", "trialagentbench-validation"):
        wheel = (
            public
            / "packages"
            / package
            / f"{package.replace('-', '_')}-0.1.0-py3-none-any.whl"
        )
        wheel.parent.mkdir(parents=True, exist_ok=True)
        wheel.write_bytes(f"{package}\n".encode())
        artifacts.append(
            {
                "name": wheel.relative_to(public).as_posix(),
                "sha256": sha256_file(wheel),
            }
        )
    (public / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "trialagentbench.release_package/v1",
                "trialeval_release": "TrialEvalBench",
                "trialdev_release": "TrialDevBench",
                "trace_explorer_package": None,
                "harness_source": "TrialAgentBench_harness.zip",
                "artifacts": artifacts,
                "standalone_packages": [
                    "trialagentbench-harness",
                    "trialagentbench-validation",
                ],
            }
        ),
        encoding="utf-8",
    )

    records = _public_wheels(tmp_path)

    assert tuple(row.package for row in records) == (
        "trialagentbench-harness",
        "trialagentbench-validation",
    )


def test_candidate_statistics_permit_no_static_trialdev_alternatives() -> None:
    field = _CandidateReleaseStatisticsV1.model_fields[
        "trialdev_credit_eligible_policy_target_count"
    ]

    assert field.metadata[0].ge == 0


def _bundle(root: Path) -> CandidateValidationBundleV1:
    figure_artifacts = tuple(
        sorted(
            (
                _artifact(root, "figures/figure.csv", "text/csv"),
                _artifact(root, "figures/figure.pdf", "application/pdf"),
                _artifact(root, "figures/figure.png", "image/png"),
                _artifact(root, "figures/figure.svg", "image/svg+xml"),
            ),
            key=lambda row: row.relative_path,
        )
    )
    figure = ValidationFigureV1(
        figure_id="candidate.figure",
        title="Candidate figure",
        scientific_question="Does the finite candidate contain the declared artifact?",
        independent_unit="released artifact",
        estimand="exact artifact count",
        comparator="declared candidate",
        uncertainty="none",
        interpretation=("The declared artifact is present.",),
        artifacts=figure_artifacts,
    )
    methods = _artifact(root, "METHODS.md", "text/markdown")
    report = _artifact(root, "REPORT.md", "text/markdown")
    results = _artifact(root, "RESULTS.csv", "text/csv")
    sources = _artifact(root, "SOURCES.md", "text/markdown")
    (root / "sentinels.json").write_text("{}\n", encoding="utf-8")
    paths = tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "exact_membership_manifest.json"
    )
    membership = ExternalArtifactManifestV1(
        artifacts=tuple(
            ArtifactDigestV1(
                relative_path=path.relative_to(root).as_posix(),
                sha256=sha256_file(path),
                byte_size=path.stat().st_size,
            )
            for path in paths
        )
    )
    write_model(root / "exact_membership_manifest.json", membership)
    membership_artifact = ValidationArtifactV1(
        relative_path="exact_membership_manifest.json",
        sha256=sha256_file(root / "exact_membership_manifest.json"),
        media_type="application/json",
    )
    payload = {
        "schema_id": "trialagentbench.candidate_validation_bundle/v1",
        "candidate": _identity().model_dump(mode="json"),
        "verifier_lock_sha256": "2" * 64,
        "figures": [figure.model_dump(mode="json")],
        "methods": methods.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
        "results": results.model_dump(mode="json"),
        "sources": sources.model_dump(mode="json"),
        "exact_membership_manifest": membership_artifact.model_dump(mode="json"),
    }
    return CandidateValidationBundleV1(**payload, checksum=_checksum(payload))


def test_candidate_bundle_verifies_every_declared_and_unlisted_artifact(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)

    verify_candidate_validation_bundle(tmp_path, bundle)


def test_candidate_bundle_public_verification_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _bundle(tmp_path)
    write_model(tmp_path / "candidate_validation_bundle.json", bundle)

    assert (
        validation_main(["candidate-release-verify", "--bundle-root", str(tmp_path)])
        == 0
    )
    assert "Candidate analysis verified" in capsys.readouterr().out


def test_candidate_bundle_rejects_membership_drift(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (tmp_path / "unexpected.txt").write_text("drift\n", encoding="utf-8")

    with pytest.raises(ValueError, match="membership differs"):
        verify_candidate_validation_bundle(tmp_path, bundle)


def test_candidate_bundle_rejects_unlisted_artifact_tampering(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (tmp_path / "sentinels.json").write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="identity mismatch"):
        verify_candidate_validation_bundle(tmp_path, bundle)


def _trialdev_lane_projection(root: Path) -> GraderConcordanceReportV1:
    grader = root / "recovery" / "grader_concordance"
    grader.mkdir(parents=True)
    record = TrialDevLaneGradeV1(
        scenario_id="s01",
        phase_id="phase2",
        program_objective_id="efficacy",
        phase_scoring_objective_id="efficacy",
        lane_id="decision_action",
        evaluation_target_checksum="a" * 64,
        scoring_policy_id="policy",
        recoverability_policy_id="public_evidence",
        submitted_target_id="advance",
        reference_target_ids=("advance",),
        score=1.0,
        score_derivation="public_evidence_action",
        status="scored",
        artifact_status="present",
        checksum="b" * 64,
    )
    lane_payload = record.model_dump_json() + "\n"
    for name in (
        "independent_trialeval_grade_records.jsonl",
        "public_trialeval_grade_records.jsonl",
    ):
        (grader / name).write_text("", encoding="utf-8")
    for name in (
        "independent_trialdev_lane_records.jsonl",
        "public_trialdev_lane_records.jsonl",
    ):
        (grader / name).write_text(lane_payload, encoding="utf-8")
    return GraderConcordanceReportV1(
        release_id="candidate-1",
        trialeval_item_count=0,
        trialeval_required_count=0,
        trialdev_required_count=1,
        raw_projection_required_count=0,
        independently_projected_raw_count=0,
        harness_projected_raw_count=0,
        raw_projection_mismatch_count=0,
        trialeval_mutation_required_count=0,
        trialeval_mutation_independently_graded_count=0,
        trialeval_mutation_public_graded_count=0,
        trialeval_mutation_mismatch_count=0,
        trialeval_mutation_behavior_failure_count=0,
        trialeval_mutation_crashed_count=0,
        trialdev_mutation_required_count=0,
        trialdev_mutation_independently_graded_count=0,
        trialdev_mutation_public_graded_count=0,
        trialdev_mutation_mismatch_count=0,
        trialdev_mutation_behavior_failure_count=0,
        trialdev_mutation_crashed_count=0,
        required_count=1,
        independently_graded_count=1,
        public_grader_count=1,
        mismatch_count=0,
        unsupported_count=0,
        crashed_count=0,
        independent_raw_projection_sha256=hashlib.sha256(b"").hexdigest(),
        harness_raw_projection_sha256=hashlib.sha256(b"").hexdigest(),
        independent_mutation_projection_sha256=hashlib.sha256(b"").hexdigest(),
        public_mutation_projection_sha256=hashlib.sha256(b"").hexdigest(),
        independent_trialdev_mutation_projection_sha256=hashlib.sha256(b"").hexdigest(),
        public_trialdev_mutation_projection_sha256=hashlib.sha256(b"").hexdigest(),
        independent_projection_sha256=hashlib.sha256(lane_payload.encode()).hexdigest(),
        public_projection_sha256=hashlib.sha256(lane_payload.encode()).hexdigest(),
        public_grader_command=("trialagentbench",),
        passed=True,
    )


def test_trialdev_lane_census_exports_exact_independent_public_agreement(
    tmp_path: Path,
) -> None:
    report = _trialdev_lane_projection(tmp_path)

    rows = _trialdev_lane_rows(tmp_path, report)

    assert len(rows) == 1
    assert rows[0]["lane_id"] == "decision_action"
    assert rows[0]["reference_target_ids"] == "advance"
    assert rows[0]["submitted_target_id"] == "advance"
    assert rows[0]["independent_public_match"] is True


def test_trialdev_lane_census_rejects_projection_drift(tmp_path: Path) -> None:
    report = _trialdev_lane_projection(tmp_path)
    public = (
        tmp_path
        / "recovery"
        / "grader_concordance"
        / "public_trialdev_lane_records.jsonl"
    )
    public.write_text(
        public.read_text(encoding="utf-8").replace('"score":1.0', '"score":0.0'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frozen grader projections"):
        _trialdev_lane_rows(tmp_path, report)


def test_stratified_characteristics_report_analysis_and_independence_units() -> None:
    properties = tuple(
        cast(
            _SimulationPropertyV1,
            SimpleNamespace(
                suite_id="trialeval",
                analysis_unit_id=f"item-{context}",
                independence_unit_id="base-1",
                construction=SimpleNamespace(
                    design_profile_id="parallel",
                    design_tier="D2",
                    assumption_tier="A3",
                    context_id=context,
                    regime_cell_id="TE-S04-A3",
                    mechanism_environment_id=None,
                    trajectory_template_id=None,
                ),
                study_format=SimpleNamespace(endpoint_family="time_to_event"),
                estimand=SimpleNamespace(
                    endpoint_or_variable="death",
                    estimand_id="itt_treatment_policy",
                ),
            ),
        )
        for context in ("C1", "C2")
    )
    characteristics = [
        {
            "suite": "trialeval",
            "analysis_unit_id": "item-C1",
            "metric": "event_count",
            "value": 3,
            "status": "observed",
        },
        {
            "suite": "trialeval",
            "analysis_unit_id": "item-C2",
            "metric": "event_count",
            "value": 5,
            "status": "observed",
        },
        {
            "suite": "trialeval",
            "analysis_unit_id": "item-C1",
            "metric": "cluster_count",
            "value": None,
            "status": "not_observed",
        },
        {
            "suite": "trialeval",
            "analysis_unit_id": "item-C2",
            "metric": "cluster_count",
            "value": None,
            "status": "not_observed",
        },
    ]

    rows = _stratified_characteristic_rows(properties, characteristics)

    overall_events = next(
        row
        for row in rows
        if row["stratifier"] == "all"
        and row["stratum"] == "all"
        and row["metric"] == "event_count"
    )
    assert overall_events["analysis_unit_count"] == 2
    assert overall_events["independence_unit_count"] == 1
    assert overall_events["observed_count"] == 2
    assert overall_events["not_observed_count"] == 0
    assert overall_events["minimum"] == 3.0
    assert overall_events["median"] == 4.0
    assert overall_events["maximum"] == 5.0
    c1_events = next(
        row
        for row in rows
        if row["stratifier"] == "context"
        and row["stratum"] == "C1"
        and row["metric"] == "event_count"
    )
    assert c1_events["analysis_unit_count"] == c1_events["independence_unit_count"] == 1
    overall_clusters = next(
        row
        for row in rows
        if row["stratifier"] == "all"
        and row["stratum"] == "all"
        and row["metric"] == "cluster_count"
    )
    assert overall_clusters["observed_count"] == 0
    assert overall_clusters["not_observed_count"] == 2
    assert overall_clusters["median"] is None


def test_context_invariance_compares_matched_data_representations() -> None:
    properties = tuple(
        cast(
            _SimulationPropertyV1,
            SimpleNamespace(
                suite_id="trialeval",
                analysis_unit_id=f"item-{context}",
                matched_set_id="base-1",
                provenance=SimpleNamespace(generation_seed_id="seed-1"),
                estimand=SimpleNamespace(
                    model_dump=lambda **_: {"estimand_id": "primary"}
                ),
                construction=SimpleNamespace(context_id=context),
            ),
        )
        for context in ("C1", "C2", "C3", "C4", "C5")
    )
    hashes = {
        "item-C1": "analysis-ready",
        "item-C2": "analysis-ready",
        "item-C3": "raw-domains",
        "item-C4": "raw-domains",
        "item-C5": "raw-domains-with-transport-duplicate",
    }

    rows = _context_rows(properties, hashes)

    assert rows == [
        {
            "matched_set_id": "base-1",
            "context_count": 5,
            "generation_seed_count": 1,
            "estimand_count": 1,
            "analysis_ready_hash_count_c1_c2": 1,
            "raw_domain_hash_count_c3_c4": 1,
            "status": "pass",
        }
    ]


def test_route_dispositions_reconcile_nonpoint_identification_responses() -> None:
    report = SimpleNamespace(
        suite="trialeval",
        required_route_count=3,
        status="fail",
        routes=(
            SimpleNamespace(status="pass", result_kind="point"),
            SimpleNamespace(status="pass", result_kind="limitation"),
            SimpleNamespace(status="fail", result_kind="point"),
        ),
    )

    rows = _route_disposition_rows((cast(RecoverabilityReportV1, report),))

    assert rows == [
        {
            "suite": "trialeval",
            "attempted": 3,
            "successful": 1,
            "failed": 1,
            "non_estimable": 1,
            "status": "fail",
        }
    ]


def test_route_result_kinds_keep_numeric_and_categorical_rules_separate() -> None:
    report = cast(
        RecoverabilityReportV1,
        SimpleNamespace(
            suite="trialeval",
            routes=(
                SimpleNamespace(
                    result_kind="numeric_point",
                    comparison_rule="numeric_envelope",
                    maximum_absolute_difference=0.002,
                    difference_to_tolerance_ratio=0.2,
                    status="pass",
                ),
                SimpleNamespace(
                    result_kind="limitation",
                    comparison_rule="categorical_code_membership",
                    maximum_absolute_difference=0.0,
                    difference_to_tolerance_ratio=None,
                    status="pass",
                ),
            ),
        ),
    )

    rows = _route_result_kind_rows((report,))

    assert [(row["result_kind"], row["comparison_rule"]) for row in rows] == [
        ("limitation", "categorical_code_membership"),
        ("numeric_point", "numeric_envelope"),
    ]
    assert rows[0]["maximum_tolerance_ratio"] is None
    assert rows[1]["maximum_tolerance_ratio"] == 0.2


def test_trialeval_characteristics_are_reconstructed_from_participant_bytes(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "participant.zip"
    adsl = pd.DataFrame(
        {
            "USUBJID": ["01", "02", "03"],
            "TRTA": ["control", "active", "active"],
            "SITEID": ["A", "A", "B"],
        }
    )
    adtte = pd.DataFrame(
        {
            "USUBJID": ["01", "02", "03"],
            "PARAMCD": ["PRIMARY", "PRIMARY", "PRIMARY"],
            "AVAL": [10.0, 12.0, 15.0],
            "CNSR": [0, 1, 0],
        }
    )
    operational = pd.DataFrame(
        {
            "USUBJID": ["01", "02", "03"],
            "N_ICE_RECORDS": [2, 1, 0],
            "N_ADVERSE_EVENT_ICE": [0, 1, 0],
            "N_DISCONTINUATION_ICE": [0, 0, 0],
            "N_NONADHERENCE_ICE": [1, 0, 0],
            "N_RESCUE_THERAPY_ICE": [0, 0, 0],
            "N_TREATMENT_SWITCH_ICE": [1, 0, 0],
            "ANY_STUDY_DISCONTINUATION": ["", "Y", ""],
        }
    )
    adae = pd.DataFrame(
        {
            "USUBJID": ["01", "02"],
            "AESER": ["N", "Y"],
            "TRTEMFL": ["Y", "Y"],
        }
    )
    adlb = pd.DataFrame({"USUBJID": ["01", "02"], "AVAL": [1.0, None]})
    advs = pd.DataFrame({"USUBJID": ["01", "02"], "AVAL": [None, 2.0]})
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("items/item-1/data/ADSL.parquet", _parquet_bytes(adsl))
        archive.writestr("items/item-1/data/ADTTE.parquet", _parquet_bytes(adtte))
        archive.writestr(
            "items/item-1/data/subject_operational_flags.parquet",
            _parquet_bytes(operational),
        )
        archive.writestr("items/item-1/data/ADAE.parquet", _parquet_bytes(adae))
        archive.writestr("items/item-1/data/ADLB.parquet", _parquet_bytes(adlb))
        archive.writestr("items/item-1/data/ADVS.parquet", _parquet_bytes(advs))
    property_row = cast(
        _SimulationPropertyV1,
        SimpleNamespace(
            suite_id="trialeval",
            analysis_unit_id="item-1",
            matched_set_id="base-1",
            estimand=SimpleNamespace(endpoint_or_variable="PRIMARY"),
            construction=SimpleNamespace(context_id="C1"),
        ),
    )

    rows, hashes = _trialeval_characteristics(
        archive_path,
        (property_row,),
    )

    values = {str(row["metric"]): row["value"] for row in rows}
    assert values["participant_count"] == 3
    assert values["treatment_arm_count"] == 2
    assert values["event_count"] == 2
    assert values["censoring_count"] == 1
    assert values["cluster_count"] == 2
    assert values["allocation_count::active"] == 2
    assert values["allocation_count::control"] == 1
    assert values["missing_observation_count"] == 2
    assert values["intercurrent_event_count"] == 3
    assert values["nonadherence_intercurrent_event_count"] == 1
    assert values["treatment_switch_count"] == 1
    assert values["rescue_count"] == 0
    assert values["study_discontinuation_count"] == 1
    assert values["safety_event_count"] == 2
    assert values["serious_safety_event_count"] == 1
    assert values["treatment_emergent_safety_event_count"] == 2
    assert set(hashes) == {"item-1"}
    duplicated = pd.concat((adtte, adtte.iloc[[0]]), ignore_index=True)
    assert (
        _stable_analysis_hash(adsl, duplicated, endpoint_id="PRIMARY")
        == hashes["item-1"]
    )


def test_trialeval_characteristics_mark_unreleased_auxiliary_tables_not_observed(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "participant.zip"
    adsl = pd.DataFrame(
        {
            "USUBJID": ["01", "02"],
            "TRTA": ["control", "active"],
        }
    )
    adtte = pd.DataFrame(
        {
            "USUBJID": ["01", "02"],
            "PARAMCD": ["PRIMARY", "PRIMARY"],
            "AVAL": [10.0, 12.0],
            "CNSR": [0, 1],
        }
    )
    operational = pd.DataFrame(
        {
            "USUBJID": ["01", "02"],
            "N_ICE_RECORDS": [0, 0],
            "N_ADVERSE_EVENT_ICE": [0, 0],
            "N_DISCONTINUATION_ICE": [0, 0],
            "N_NONADHERENCE_ICE": [0, 0],
            "N_RESCUE_THERAPY_ICE": [0, 0],
            "N_TREATMENT_SWITCH_ICE": [0, 0],
            "ANY_STUDY_DISCONTINUATION": ["", ""],
        }
    )
    adae = pd.DataFrame(
        {
            "USUBJID": ["01"],
            "AESER": ["N"],
            "TRTEMFL": ["Y"],
        }
    )
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("items/item-1/data/ADSL.parquet", _parquet_bytes(adsl))
        archive.writestr("items/item-1/data/ADTTE.parquet", _parquet_bytes(adtte))
        archive.writestr(
            "items/item-1/data/subject_operational_flags.parquet",
            _parquet_bytes(operational),
        )
        archive.writestr("items/item-1/data/ADAE.parquet", _parquet_bytes(adae))
    property_row = cast(
        _SimulationPropertyV1,
        SimpleNamespace(
            suite_id="trialeval",
            analysis_unit_id="item-1",
            matched_set_id="base-1",
            estimand=SimpleNamespace(endpoint_or_variable="PRIMARY"),
            construction=SimpleNamespace(context_id="C1"),
        ),
    )

    rows, _ = _trialeval_characteristics(archive_path, (property_row,))

    by_metric = {str(row["metric"]): row for row in rows}
    assert by_metric["laboratory_missing_observation_count"]["value"] is None
    assert by_metric["laboratory_missing_observation_count"]["status"] == "not_observed"
    assert by_metric["vital_sign_missing_observation_count"]["value"] is None
    assert by_metric["vital_sign_missing_observation_count"]["status"] == "not_observed"
    assert by_metric["missing_observation_count"]["value"] == 0


def test_trialdev_characteristics_keep_realized_events_and_absent_rescue_separate(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "participant.zip"
    frame = pd.DataFrame(
        {
            "USUBJID": ["01", "02", "03"],
            "TREATMENT": ["a", "b", "b"],
            "EFF_PRIMARY_E": [1, 0, 1],
            "AE_HEPATIC_EVENT_E": [0, 1, 1],
            "LTFU_E": [0, 1, 0],
            "DISCONTINUATION_E": [0, 0, 1],
            "EARLY_RESCUE_RISK": [0.1, 0.2, 0.3],
        }
    )
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "scenario_s01/public/observational_extract.parquet",
            _parquet_bytes(frame),
        )
    property_row = cast(
        _SimulationPropertyV1,
        SimpleNamespace(
            suite_id="trialdev",
            analysis_unit_id="s01",
            matched_set_id="template-1",
        ),
    )

    rows = _trialdev_characteristics(archive_path, (property_row,))

    values = {str(row["metric"]): row["value"] for row in rows}
    statuses = {str(row["metric"]): row["status"] for row in rows}
    assert values["participant_count"] == 3
    assert values["efficacy_event_count"] == 2
    assert values["safety_event_count"] == 2
    assert values["loss_to_follow_up_count"] == 1
    assert values["discontinuation_count"] == 1
    assert values["intercurrent_event_count"] == 1
    assert values["allocation_count::a"] == 1
    assert values["allocation_count::b"] == 2
    assert values["missing_observation_count"] == 0
    assert values["rescue_count"] is None
    assert statuses["rescue_count"] == "not_observed"


def test_candidate_figure_writes_deterministic_review_formats(tmp_path: Path) -> None:
    rows = [
        {
            "label": "TrialEval routes",
            "value": 12,
            "display": "12/12",
            "series": "pass",
        }
    ]
    figures = []
    for name in ("first", "second"):
        figures.append(
            _write_figure(
                tmp_path / name,
                figure_id="candidate.test",
                title="Candidate test",
                question="Are all routes present?",
                independent_unit="route",
                estimand="exact finite count",
                comparator="scheduled routes",
                uncertainty="none",
                rows=rows,
                interpretation=("All scheduled routes are present.",),
            )
        )

    first, second = figures
    assert {Path(row.relative_path).suffix for row in first.artifacts} == {
        ".csv",
        ".pdf",
        ".png",
        ".svg",
    }
    assert tuple(row.sha256 for row in first.artifacts) == tuple(
        row.sha256 for row in second.artifacts
    )
