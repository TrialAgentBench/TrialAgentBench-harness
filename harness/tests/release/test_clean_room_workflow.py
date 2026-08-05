from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trialagentbench_harness.contracts.release.artifacts import (
    TRIALDEV_FIXED_TRAJECTORY_REPLICATE_MEMBERS,
    TRIALDEV_VERIFICATION_ROOT_MEMBERS,
)
from trialagentbench_harness.contracts.release.clean_room_workflow import validate_clean_room_workflow
from trialagentbench_harness.contracts.release.trialdev_runtime_surface import TRIALDEV_PUBLIC_FILE_ROLES
from trialagentbench_harness.contracts.trialdev.trialdev_public_phase_replay import (
    TrialDevPhaseReplayCaseV1,
)
from trialagentbench_harness.io.checksums import canonical_payload_sha256
from trialagentbench_harness.tools.validate.validate_clean_room_workflow import main as validate_clean_room_main


def _write_zip(path: Path, members: dict[str, str | bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)


def _parquet_bytes(columns: dict[str, list[object]]) -> tuple[bytes, pa.Schema]:
    table = pa.table(columns)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes(), table.schema


def _parquet_bytes_with_metadata(
    columns: dict[str, list[object]],
    *,
    metadata: dict[bytes, bytes],
) -> tuple[bytes, pa.Schema]:
    table = pa.table(columns).replace_schema_metadata(metadata)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes(), table.schema


def _trialdev_dictionary(*, schema: pa.Schema, scenario_id: str = "s01") -> str:
    payload: dict[str, object] = {
        "version": "v1",
        "scenario_id": scenario_id,
        "semantic_roles": [],
        "observational_schema": [
            {"column": field.name, "arrow_type": str(field.type), "nullable": field.nullable} for field in schema
        ],
    }
    payload["checksum"] = canonical_payload_sha256(payload)
    return json.dumps(payload)


def _trialdev_columns(**extra: list[object]) -> dict[str, list[object]]:
    return {
        "USUBJID": ["01"],
        "TREATMENT": ["A"],
        "DISCONTINUATION_T": [28.0],
        "DISCONTINUATION_E": [0],
        "LTFU_T": [28.0],
        "LTFU_E": [0],
        **extra,
    }


def _trialdev_participant_members(
    *,
    observed_columns: dict[str, list[object]],
    declared_schema: pa.Schema | None = None,
    catalog_variable_ids: tuple[str, ...] = (),
) -> dict[str, str | bytes]:
    observational, observed_schema = _parquet_bytes(observed_columns)
    suite_manifest = {
        "version": "v1",
        "suite_id": "fixture_trialdev_suite",
        "release_root": ".",
        "items": [
            {
                "item_id": "fixture_observational_review",
                "scenario_id": "s01",
                "phase_id": "observational_review",
                "objective_id": "benefit_risk",
                "task_definition_id": "observational_review__benefit_risk__none",
                "allowed_endpoint_ids": [],
                "allowed_follow_up_days": [],
                "allowed_enrollment_window_days": [],
                "allowed_site_count_budgets": [],
                "metadata": {},
            }
        ],
    }
    suite_manifest["checksum"] = canonical_payload_sha256(suite_manifest)
    members: dict[str, str | bytes] = {
        "benchmark_suite_manifest.json": json.dumps(suite_manifest, sort_keys=True),
        **{f"scenario_s01/public/{name}": "{}" for name in TRIALDEV_PUBLIC_FILE_ROLES},
    }
    members["scenario_s01/public/variable_catalog.json"] = json.dumps(
        {"variables": [{"variable_id": variable_id} for variable_id in catalog_variable_ids]}
    )
    members["scenario_s01/public/endpoint_catalog.json"] = '{"endpoints":[]}'
    members["scenario_s01/public/safety_decision_policy.json"] = '{"serious_event_definitions":[]}'
    members["scenario_s01/public/data_dictionary.json"] = _trialdev_dictionary(
        schema=declared_schema or observed_schema
    )
    members["scenario_s01/public/observational_extract.parquet"] = observational
    return members


def _write_clean_package(root: Path) -> None:
    _write_zip(root / "TrialEvalBench" / "TrialEvalBench_participant.zip", {"items/TASK001/data.json": "{}"})
    _write_zip(root / "TrialEvalBench" / "TrialEvalBench_evaluator.zip", {"grader/item_index.json": "{}"})
    _write_zip(
        root / "TrialEvalBench" / "TrialEvalBench_verification.zip",
        {"verification/public_route_replay_evidence.json": "{}"},
    )
    participant_members = _trialdev_participant_members(observed_columns=_trialdev_columns())
    phase_case = TrialDevPhaseReplayCaseV1.model_validate(
        {
            "scenario_root": "scenario_s01",
            "world_seed": 101,
            "program_objective_ids": ["benefit_risk"],
            "request": {
                "scenario_id": "s01",
                "phase_id": "phase1",
                "candidate_drug_ids": ["drug_a"],
                "target_sample_size": 40,
                "follow_up_days": 28,
                "enrollment_window_days": 60,
                "site_count_budget": 4,
                "allocation_ratio": "1:1",
                "design_cell_id": "phase1_fixed_final",
                "interim_policy": "fixed_final",
                "site_strategy": "region_balanced",
                "selection_objective": "benefit_risk",
            },
        }
    )
    participant_members["fixed_trajectories/cases.jsonl"] = phase_case.model_dump_json() + "\n"
    trajectory, _ = _parquet_bytes({"USUBJID": ["01"], "ARM": ["control"]})
    trajectory_root = (
        "fixed_trajectories/materialized/world_101/" f"request_{phase_case.request.checksum()}/trial_seed_202"
    )
    for name in TRIALDEV_FIXED_TRAJECTORY_REPLICATE_MEMBERS:
        participant_members[f"{trajectory_root}/{name}"] = trajectory if name.endswith(".parquet") else "{}\n"
    _write_zip(root / "TrialDevBench" / "TrialDevBench_participant.zip", participant_members)
    scorer_members = {
        "grading_procedure.json",
        "submission_schema.json",
        "drug_ranking_reference_manifest.json",
        "public_recoverability_report.json",
        "evaluation_target_register.jsonl",
        "evaluation_target_register_manifest.json",
        "evaluation_target_register_gate_report.json",
    }
    _write_zip(
        root / "TrialDevBench" / "TrialDevBench_evaluator.zip",
        {
            **{f"scenario_s01/grader/{name}": "{}" for name in scorer_members},
            "scenario_s01/grader/recoverability_manifest.json": "{}",
        },
    )
    _write_zip(
        root / "TrialDevBench" / "TrialDevBench_verification.zip",
        {
            "phase_replay/cases.jsonl": "{}\n",
            "phase_replay/records.jsonl": "{}\n",
            "phase_replay/materialized/s01/phase1/table.parquet": b"verification",
            **{name: "{}\n" for name in TRIALDEV_VERIFICATION_ROOT_MEMBERS},
        },
    )


def _write_clean_harness(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    script_rows = ['trialagentbench = "trialagentbench_harness.cli:main"']
    (root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "trial-agent-bench"',
                "[project.scripts]",
                *script_rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "The harness accepts user-supplied traces and arbitrary model panels.\n",
        encoding="utf-8",
    )


def test_validate_clean_room_workflow_accepts_clean_minimal_release(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    audit_root = tmp_path / "audit"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    audit_root.mkdir()
    (audit_root / "trialeval_context_sufficiency_report.json").write_text("{}", encoding="utf-8")
    (audit_root / "trialeval_context_artifact_delta_report.json").write_text("{}", encoding="utf-8")

    report = validate_clean_room_workflow(
        package_root=package_root, harness_root=harness_root, audit_roots=[audit_root]
    )

    assert report.status == "pass"


def test_clean_room_workflow_accepts_declared_trialdev_public_release_docs(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    archive_path = package_root / "TrialDevBench" / "TrialDevBench_participant.zip"
    members: dict[str, str | bytes] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            members[name] = archive.read(name)
    members["distribution_mode_participant_manifest.json"] = "{}\n"
    members["docs/QUICKSTART.md"] = "Run the public harness.\n"
    _write_zip(archive_path, members)

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "pass"


def test_clean_room_workflow_rejects_undeclared_trialdev_documentation(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    archive_path = package_root / "TrialDevBench" / "TrialDevBench_participant.zip"
    members: dict[str, str | bytes] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            members[name] = archive.read(name)
    members["docs/internal_notes.md"] = "internal\n"
    _write_zip(archive_path, members)

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "fail"
    assert "participant_unknown_member" in {finding.code for finding in report.findings}


def test_clean_room_workflow_rejects_invalid_fixed_trajectory_index(tmp_path: Path) -> None:
    package = tmp_path / "package"
    harness = tmp_path / "harness"
    _write_clean_package(package)
    _write_clean_harness(harness)
    archive_path = package / "TrialDevBench" / "TrialDevBench_participant.zip"
    members: dict[str, str | bytes] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            members[name] = archive.read(name)
    members["fixed_trajectories/cases.jsonl"] = "{}\n"
    _write_zip(archive_path, members)

    report = validate_clean_room_workflow(
        package_root=package,
        harness_root=harness,
        audit_roots=[],
    )

    assert report.status == "fail"
    assert {finding.code for finding in report.findings} >= {"participant_fixed_trajectory_index_invalid"}


def test_clean_room_workflow_rejects_unindexed_fixed_trajectory_data(tmp_path: Path) -> None:
    package = tmp_path / "package"
    harness = tmp_path / "harness"
    _write_clean_package(package)
    _write_clean_harness(harness)
    archive_path = package / "TrialDevBench" / "TrialDevBench_participant.zip"
    members: dict[str, str | bytes] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            members[name] = archive.read(name)
    trajectory, _ = _parquet_bytes({"USUBJID": ["01"], "ARM": ["control"]})
    members["fixed_trajectories/materialized/world_999/" f"request_{'0' * 64}/trial_seed_1/table.parquet"] = trajectory
    _write_zip(archive_path, members)

    report = validate_clean_room_workflow(
        package_root=package,
        harness_root=harness,
        audit_roots=[],
    )

    assert report.status == "fail"
    assert {finding.code for finding in report.findings} >= {"participant_fixed_trajectory_unindexed"}


def test_clean_room_harness_permits_participant_facing_runners(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    runners = harness_root / "trialagentbench_harness" / "tools" / "run"
    runners.mkdir(parents=True)
    (runners / "trialeval.py").write_text("# participant runner\n", encoding="utf-8")
    (runners / "trialdev.py").write_text("# participant runner\n", encoding="utf-8")

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "pass"
    assert report.findings == ()
    assert set(report.participant_artifacts) == {
        "TrialEvalBench/TrialEvalBench_participant.zip",
        "TrialDevBench/TrialDevBench_participant.zip",
    }
    assert report.verification_artifacts == (
        "TrialEvalBench/TrialEvalBench_verification.zip",
        "TrialDevBench/TrialDevBench_verification.zip",
    )
    assert report.audit_artifacts == ()


def test_validate_clean_room_workflow_rejects_trial_eval_evaluator_member_in_participant_zip(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    _write_zip(
        package_root / "TrialEvalBench" / "TrialEvalBench_participant.zip",
        {"items/TASK001/grader/answer.json": '{"route_reference_id":"x"}'},
    )

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "fail"
    assert {finding.code for finding in report.findings} >= {
        "participant_evaluator_path_leak",
        "participant_target_field_leak",
    }


def test_validate_clean_room_workflow_rejects_trialdev_hidden_truth_in_participant_zip(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    _write_zip(
        package_root / "TrialDevBench" / "TrialDevBench_participant.zip",
        {"scenario_s01/hidden/oracle.json": '{"diagnostic_reference_route":"advance"}'},
    )

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "fail"
    assert {finding.code for finding in report.findings} >= {
        "participant_evaluator_path_leak",
        "participant_decision_reference_text_leak",
    }


def test_validate_clean_room_workflow_rejects_missing_trialdev_suite_manifest(tmp_path: Path) -> None:
    """The participant role must remain directly discoverable by the live harness."""

    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    participant_members = _trialdev_participant_members(observed_columns=_trialdev_columns())
    participant_members.pop("benchmark_suite_manifest.json")
    _write_zip(
        package_root / "TrialDevBench" / "TrialDevBench_participant.zip",
        participant_members,
    )

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "fail"
    assert "participant_suite_manifest_missing" in {finding.code for finding in report.findings}


def test_clean_room_workflow_rejects_undeclared_trialdev_binary_column(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    _, declared_schema = _parquet_bytes(_trialdev_columns())
    participant_members = _trialdev_participant_members(
        observed_columns=_trialdev_columns(UNDECLARED_MEASURE=[1.0]),
        declared_schema=declared_schema,
    )
    _write_zip(
        package_root / "TrialDevBench" / "TrialDevBench_participant.zip",
        participant_members,
    )

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "fail"
    assert "participant_binary_schema_mismatch" in {finding.code for finding in report.findings}


def test_clean_room_workflow_rejects_self_declared_unknown_trialdev_binary_column(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    participant_members = _trialdev_participant_members(
        observed_columns=_trialdev_columns(UNDECLARED_MEASURE=[1.0]),
    )
    _write_zip(
        package_root / "TrialDevBench" / "TrialDevBench_participant.zip",
        participant_members,
    )

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "fail"
    assert "participant_binary_unknown_column" in {finding.code for finding in report.findings}


def test_clean_room_workflow_rejects_declared_oracle_binary_column(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    participant_members = _trialdev_participant_members(
        observed_columns=_trialdev_columns(ORACLE__OUTCOME=[1.0]),
    )
    _write_zip(
        package_root / "TrialDevBench" / "TrialDevBench_participant.zip",
        participant_members,
    )

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "fail"
    assert "participant_binary_private_column" in {finding.code for finding in report.findings}


def test_clean_room_workflow_rejects_trialeval_dictionary_schema_drift(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    trial_data, _ = _parquet_bytes({"USUBJID": ["01"], "AVAL": [2.0]})
    dictionary = {
        "semantic_columns": [
            {"table": "data/ADTTE.parquet", "column": "USUBJID", "dtype": "string"},
        ]
    }
    _write_zip(
        package_root / "TrialEvalBench" / "TrialEvalBench_participant.zip",
        {
            "items/TASK001/data_dictionary.json": json.dumps(dictionary),
            "items/TASK001/data/ADTTE.parquet": trial_data,
        },
    )

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "fail"
    assert "participant_binary_schema_mismatch" in {finding.code for finding in report.findings}


def test_clean_room_workflow_rejects_undeclared_participant_parquet_metadata(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    trial_data, schema = _parquet_bytes_with_metadata(
        {"USUBJID": ["01"]},
        metadata={b"construction_note": b"private"},
    )
    dictionary = {
        "semantic_columns": [
            {"table": "data/ADTTE.parquet", "column": "USUBJID", "dtype": str(schema.field("USUBJID").type)},
        ]
    }
    _write_zip(
        package_root / "TrialEvalBench" / "TrialEvalBench_participant.zip",
        {
            "items/TASK001/data_dictionary.json": json.dumps(dictionary),
            "items/TASK001/data/ADTTE.parquet": trial_data,
        },
    )

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "fail"
    assert "participant_binary_schema_invalid" in {finding.code for finding in report.findings}


@pytest.mark.parametrize(
    ("member_name", "expected_code"),
    [
        ("../escape.json", "participant_archive_unsafe_path"),
        ("items/TASK001/payload.zip", "participant_archive_nested_archive"),
    ],
)
def test_clean_room_workflow_rejects_unsafe_participant_archive_members(
    tmp_path: Path,
    member_name: str,
    expected_code: str,
) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    _write_zip(
        package_root / "TrialEvalBench" / "TrialEvalBench_participant.zip",
        {member_name: "{}"},
    )

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "fail"
    assert expected_code in {finding.code for finding in report.findings}


def test_clean_room_workflow_rejects_archive_symlinks(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    archive_path = package_root / "TrialEvalBench" / "TrialEvalBench_participant.zip"
    link = zipfile.ZipInfo("items/TASK001/link.json")
    link.create_system = 3
    link.external_attr = (0o120777 << 16) | 0xA000
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(link, "target.json")

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "fail"
    assert "participant_archive_link" in {finding.code for finding in report.findings}


def test_clean_room_workflow_rejects_filesystem_symlinks(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    target = tmp_path / "outside.md"
    target.write_text("outside\n", encoding="utf-8")
    (harness_root / "linked.md").symlink_to(target)

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "fail"
    assert "harness_filesystem_link" in {finding.code for finding in report.findings}


def test_clean_room_workflow_rejects_construction_truth_in_trialdev_evaluator(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    evaluator = package_root / "TrialDevBench" / "TrialDevBench_evaluator.zip"
    with zipfile.ZipFile(evaluator, "a") as archive:
        archive.writestr("scenario_s01/hidden/evaluation_reference.json", "{}")

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "fail"
    assert "evaluator_unknown_member" in {finding.code for finding in report.findings}


def test_clean_room_workflow_rejects_private_state_in_fixed_trajectory(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    participant = package_root / "TrialDevBench" / "TrialDevBench_participant.zip"
    with zipfile.ZipFile(participant, "a") as archive:
        archive.writestr("fixed_trajectories/materialized/hidden/evaluation_reference.json", "{}\n")

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "fail"
    assert "participant_evaluator_path_leak" in {finding.code for finding in report.findings}


def test_clean_room_workflow_rejects_private_state_in_trialdev_verification(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    verification = package_root / "TrialDevBench" / "TrialDevBench_verification.zip"
    with zipfile.ZipFile(verification, "a") as archive:
        archive.writestr("phase_replay/hidden/evaluation_reference.json", "{}\n")

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "fail"
    assert "verification_private_state" in {finding.code for finding in report.findings}


def test_validate_clean_room_workflow_rejects_harness_local_artifact_references(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)
    (harness_root / "README.md").write_text(
        "Use " + "collaborator_" + "packages/current or release_evidence for examples.\n",
        encoding="utf-8",
    )

    report = validate_clean_room_workflow(package_root=package_root, harness_root=harness_root)

    assert report.status == "fail"
    assert "harness_local_artifact_reference" in {finding.code for finding in report.findings}


def test_clean_room_workflow_cli_writes_report(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    harness_root = tmp_path / "harness"
    report_path = tmp_path / "report.json"
    _write_clean_package(package_root)
    _write_clean_harness(harness_root)

    status = validate_clean_room_main(
        [
            "--package-root",
            str(package_root),
            "--harness-root",
            str(harness_root),
            "--write-report",
            str(report_path),
        ]
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["status"] == "pass"
