"""Tests for independent full-census C5 data-integrity recovery."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from trialagentbench_validation import cli
from trialagentbench_validation.trialeval.integrity import (
    canonical_domain_content_sha256_v1,
    recover_c5_integrity,
)


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _write_archives(
    tmp_path: Path, *, discordant_reference: bool = False
) -> tuple[Path, Path]:
    c4_task = "TE-S01-A1-R01-C4"
    c5_task = "TE-S01-A1-R01-C5"
    clean = pd.DataFrame(
        {
            "USUBJID": pd.Series(["P001", "P002", "P003"], dtype="string"),
            "AGE": pd.Series([60, 71, 54], dtype="int64"),
            "RESPONSE": pd.Series([0.5, 1.25, -0.75], dtype="float64"),
        }
    )
    mutated = pd.concat([clean, clean.iloc[[1]].copy()], ignore_index=True)
    clean_checksum = canonical_domain_content_sha256_v1(clean, key_fields=("USUBJID",))
    mutated_checksum = canonical_domain_content_sha256_v1(
        mutated, key_fields=("USUBJID",)
    )

    participant = tmp_path / "participant.zip"
    with ZipFile(participant, "w") as archive:
        for task_id, frame in ((c4_task, clean), (c5_task, mutated)):
            archive.writestr(f"items/{task_id}/task.json", "{}\n")
            archive.writestr(
                f"items/{task_id}/data/raw/subjects.parquet",
                _parquet_bytes(frame),
            )
            archive.writestr(
                f"items/{task_id}/data/raw/sites.parquet",
                _parquet_bytes(pd.DataFrame({"SITEID": ["S01"], "REGION": ["EU"]})),
            )
        archive.writestr(
            f"items/{c5_task}/data_integrity_policy.json",
            json.dumps(
                {
                    "schema_id": "trialagentbench.trialeval.c5_integrity_policy/v1",
                    "task_id": c5_task,
                    "condition_id": "exact_transport_row_duplication_v1",
                    "affected_domain": "data/raw/subjects.parquet",
                    "compound_key_fields": ["USUBJID"],
                    "legitimate_repeat_semantics": "Different keys are distinct records.",
                    "repair_contract_id": "exact_transport_row_duplication_repair_v1",
                    "repair_action": "remove_one_exact_duplicate_copy",
                    "canonical_typed_scalar_encoding_id": "canonical_typed_scalar_v1",
                    "canonical_compound_row_key_encoding_id": "canonical_compound_row_key_v1",
                    "canonical_typed_row_payload_encoding_id": "canonical_typed_row_payload_v1",
                    "canonical_domain_content_checksum_id": "canonical_domain_content_sha256_v1",
                    "selected_duplicate_keys_visible": False,
                    "expected_duplicate_count_visible": False,
                    "clean_parent_checksum_visible": False,
                }
            ),
        )

    reference_payload = {
        "schema_id": "trialagentbench.trialeval.c5_integrity_reference/v1",
        "task_id": c5_task,
        "clean_context_parent_task_id": c4_task,
        "condition_id": "exact_transport_row_duplication_v1",
        "affected_domain": "data/raw/subjects.parquet",
        "compound_key_fields": ["USUBJID"],
        "defect_seed": 123,
        "clean_row_count": 3,
        "selected_duplicate_count": 1,
        "selected_compound_keys": ['[["string","P002"]]'],
        "observed_duplicate_group_count": 1,
        "observed_extra_row_count": 1,
        "repair_action": "remove_one_exact_duplicate_copy",
        "repair_status": "repaired",
        "clean_domain_content_sha256": clean_checksum,
        "mutated_domain_content_sha256": (
            "f" * 64 if discordant_reference else mutated_checksum
        ),
        "post_repair_data_checksum": clean_checksum,
        "checksum": "a" * 64,
    }
    evaluator = tmp_path / "evaluator.zip"
    with ZipFile(evaluator, "w") as archive:
        archive.writestr(
            "grader/domains/context_panels.json",
            json.dumps(
                {
                    "panels": [
                        {
                            "tasks": [
                                {"task_id": c4_task, "context_tier": "C4"},
                                {"task_id": c5_task, "context_tier": "C5"},
                            ]
                        }
                    ]
                }
            ),
        )
        archive.writestr(
            "grader/domains/data_integrity_reference.jsonl",
            json.dumps(
                {
                    "domain": "data_integrity_reference",
                    "task_id": c5_task,
                    "payload": reference_payload,
                }
            )
            + "\n",
        )
    return participant, evaluator


def test_independent_c5_recovery_proves_exact_c4_equality(tmp_path: Path) -> None:
    participant, evaluator = _write_archives(tmp_path)

    report = recover_c5_integrity(
        participant_zip=participant,
        verification_zip=evaluator,
        expected_item_count=1,
    )

    assert report.status == "pass"
    assert report.required_item_count == 1
    assert report.repaired_item_count == 1
    assert report.mismatched_item_count == 0
    assert report.unsupported_item_count == 0
    assert report.records[0].repaired_content_equals_c4
    assert report.records[0].verification_reference_concordant
    assert (
        report.checksum
        == hashlib.sha256(
            json.dumps(
                report.model_dump(mode="json", exclude={"checksum"}),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
    )


def test_independent_c5_recovery_fails_on_reference_drift(tmp_path: Path) -> None:
    participant, evaluator = _write_archives(tmp_path, discordant_reference=True)

    report = recover_c5_integrity(
        participant_zip=participant,
        verification_zip=evaluator,
        expected_item_count=1,
    )

    assert report.status == "fail"
    assert report.mismatched_item_count == 1
    assert report.records[0].repaired_content_equals_c4
    assert not report.records[0].verification_reference_concordant


def test_c5_integrity_cli_writes_receipt(tmp_path: Path) -> None:
    participant, evaluator = _write_archives(tmp_path)
    output = tmp_path / "receipt.json"

    exit_code = cli.main(
        [
            "trialeval-c5-integrity",
            "--participant",
            str(participant),
            "--verification",
            str(evaluator),
            "--output",
            str(output),
            "--expected-items",
            "1",
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
