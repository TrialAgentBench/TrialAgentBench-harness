"""Tests for TrialEval public-evidence reference replay."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from trialeval.release_fixture import write_scoreable_reference_fixture_zip

from trialagentbench_validation.trialeval.references.replay import (
    replay_trialeval_public_evidence_reference_v1,
    write_public_evidence_reference_replay_artifacts_v1,
)


def _write_public_zip(
    path: Path,
    *,
    evaluator_zip: Path,
    payload: bytes | None,
    include_contract: bool = False,
    public_root_prefix: str = "",
) -> Path:
    public_zip = path / "public.zip"
    with ZipFile(evaluator_zip) as evaluator:
        contract_row = json.loads(
            evaluator.read("grader/domains/public_estimand_contract.jsonl").decode(
                "utf-8"
            )
        )
    with ZipFile(public_zip, "w") as zf:
        if payload is not None:
            zf.writestr(
                f"{public_root_prefix}items/TASK0001/data/analysis_frame.parquet",
                payload,
            )
        zf.writestr(f"{public_root_prefix}items/TASK0001/task.json", "{}")
        zf.writestr(f"{public_root_prefix}items/TASK0001/protocol_summary.json", "{}")
        if include_contract:
            zf.writestr(
                f"{public_root_prefix}public_estimand_contracts/TASK0001.json",
                json.dumps(contract_row["payload"]["contract"]),
            )
    return public_zip


def test_public_evidence_reference_replay_accepts_public_surface_mirror(
    tmp_path: Path,
) -> None:
    evaluator_zip = write_scoreable_reference_fixture_zip(
        tmp_path, include_reference_input=True
    )
    public_zip = _write_public_zip(
        tmp_path, evaluator_zip=evaluator_zip, payload=b"fixture table"
    )

    report = replay_trialeval_public_evidence_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "pass"
    assert report.task_count == 1
    assert report.replay_record_count == 1
    assert report.public_input_count == 1
    assert report.missing_public_input_count == 0
    assert report.checksum_mismatch_count == 0
    assert report.route_reference_without_replay_count == 0
    assert report.drift_status_counts == {"identical": 1}
    assert report.records[0].evaluation_class == "canonical_analysis_public_evidence"
    assert report.official_route_reference_count == 1
    assert report.public_reference_source_rows == 1
    assert report.missing_public_reference_source_count == 0
    assert report.orphan_public_reference_source_count == 0
    assert report.public_contract_bound_route_reference_count == 1
    assert report.scoreable_input_only_variant_count == 0
    assert report.public_surface_gap_variant_count == 0
    assert report.contract_defect_variant_count == 0
    assert report.route_reference_exposure_counts == {
        "public_contract_and_scoreable_input": 1
    }
    assert (
        report.route_reference_records[0].exposure_class
        == "public_contract_and_scoreable_input"
    )


def test_public_evidence_reference_replay_accepts_role_archive_public_root(
    tmp_path: Path,
) -> None:
    evaluator_zip = write_scoreable_reference_fixture_zip(
        tmp_path, include_reference_input=True
    )
    public_zip = _write_public_zip(
        tmp_path,
        evaluator_zip=evaluator_zip,
        payload=b"fixture table",
        public_root_prefix="public/",
    )

    report = replay_trialeval_public_evidence_reference_v1(
        evaluator_zip=evaluator_zip,
        public_zip=public_zip,
    )

    assert report.status == "pass"
    assert report.missing_public_input_count == 0


def test_public_evidence_reference_replay_rejects_missing_public_input(
    tmp_path: Path,
) -> None:
    evaluator_zip = write_scoreable_reference_fixture_zip(
        tmp_path, include_reference_input=True
    )
    public_zip = _write_public_zip(tmp_path, evaluator_zip=evaluator_zip, payload=None)

    report = replay_trialeval_public_evidence_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "fail"
    assert report.missing_public_input_count == 1
    assert report.records[0].evaluation_class == "public_surface_gap"
    assert "route_reference_inputs_missing_public_sources" in report.findings


def test_public_evidence_reference_replay_binds_evaluator_contracts_to_participant_evidence(
    tmp_path: Path,
) -> None:
    evaluator_zip = write_scoreable_reference_fixture_zip(
        tmp_path, include_reference_input=True
    )
    public_zip = _write_public_zip(
        tmp_path, evaluator_zip=evaluator_zip, payload=b"fixture table"
    )

    report = replay_trialeval_public_evidence_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "pass"
    assert report.public_estimand_contract_count == 0
    assert report.public_contract_missing_count == 0
    assert report.route_reference_records[0].public_contract_bound is True


def test_public_evidence_reference_replay_rejects_participant_contract_leakage(
    tmp_path: Path,
) -> None:
    evaluator_zip = write_scoreable_reference_fixture_zip(
        tmp_path, include_reference_input=True
    )
    public_zip = _write_public_zip(
        tmp_path,
        evaluator_zip=evaluator_zip,
        payload=b"fixture table",
        include_contract=True,
    )

    report = replay_trialeval_public_evidence_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "fail"
    assert report.public_contract_mismatch_count == 1
    assert "participant_public_estimand_contract_leakage" in report.findings


def test_public_evidence_reference_replay_rejects_any_participant_contract(
    tmp_path: Path,
) -> None:
    evaluator_zip = write_scoreable_reference_fixture_zip(
        tmp_path, include_reference_input=True
    )
    public_zip = _write_public_zip(
        tmp_path, evaluator_zip=evaluator_zip, payload=b"fixture table"
    )
    with ZipFile(public_zip, "a") as public:
        with ZipFile(evaluator_zip) as evaluator:
            contract_row = json.loads(
                evaluator.read("grader/domains/public_estimand_contract.jsonl").decode(
                    "utf-8"
                )
            )
        contract = contract_row["payload"]["contract"]
        contract["task_id"] = "TASK9999"
        public.writestr("public_estimand_contracts/TASK9999.json", json.dumps(contract))

    report = replay_trialeval_public_evidence_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "fail"
    assert report.public_contract_mismatch_count == 1
    assert "participant_public_estimand_contract_leakage" in report.findings


def test_public_evidence_reference_replay_rejects_public_checksum_mismatch(
    tmp_path: Path,
) -> None:
    evaluator_zip = write_scoreable_reference_fixture_zip(
        tmp_path, include_reference_input=True
    )
    public_zip = _write_public_zip(
        tmp_path, evaluator_zip=evaluator_zip, payload=b"different table"
    )

    report = replay_trialeval_public_evidence_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )

    assert report.status == "fail"
    assert report.checksum_mismatch_count == 1
    assert report.records[0].evaluation_class == "public_surface_gap"
    assert "route_reference_inputs_public_checksum_mismatch" in report.findings


def test_public_evidence_reference_replay_rejects_missing_and_orphan_public_reference_sources(
    tmp_path: Path,
) -> None:
    evaluator_zip = write_scoreable_reference_fixture_zip(
        tmp_path, include_reference_input=True
    )
    public_zip = _write_public_zip(
        tmp_path, evaluator_zip=evaluator_zip, payload=b"fixture table"
    )
    rewritten = tmp_path / "evaluator_with_bad_sources.zip"
    with ZipFile(evaluator_zip) as source, ZipFile(rewritten, "w") as target:
        for member in source.infolist():
            if member.filename == "grader/domains/public_reference_sources.jsonl":
                row = json.loads(source.read(member.filename))
                row["route_reference_id"] = "TASK0001:orphan"
                target.writestr(member, json.dumps(row, sort_keys=True) + "\n")
            else:
                target.writestr(member, source.read(member.filename))

    report = replay_trialeval_public_evidence_reference_v1(
        evaluator_zip=rewritten, public_zip=public_zip
    )

    assert report.status == "fail"
    assert report.missing_public_reference_source_count == 1
    assert report.orphan_public_reference_source_count == 1
    assert "route_references_missing_public_reference_source" in report.findings
    assert "public_reference_sources_without_route_reference" in report.findings


def test_public_evidence_reference_replay_writes_artifacts(tmp_path: Path) -> None:
    evaluator_zip = write_scoreable_reference_fixture_zip(
        tmp_path, include_reference_input=True
    )
    public_zip = _write_public_zip(
        tmp_path, evaluator_zip=evaluator_zip, payload=b"fixture table"
    )
    out_dir = tmp_path / "replay"

    report = write_public_evidence_reference_replay_artifacts_v1(
        evaluator_zip=evaluator_zip,
        public_zip=public_zip,
        out_dir=out_dir,
    )

    assert report.status == "pass"
    payload = json.loads(
        (out_dir / "public_evidence_reference_replay_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "pass"
    assert (out_dir / "public_evidence_reference_replay_records.jsonl").read_text(
        encoding="utf-8"
    ).count("\n") == 1
    assert (out_dir / "public_evidence_route_reference_replay_records.jsonl").read_text(
        encoding="utf-8"
    ).count("\n") == 1
    assert "Status: `pass`" in (
        out_dir / "public_evidence_reference_replay_report.md"
    ).read_text(encoding="utf-8")
