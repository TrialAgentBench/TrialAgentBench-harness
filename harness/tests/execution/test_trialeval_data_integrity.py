"""Public C5 repair and runtime-affordance tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from trialagentbench_harness.adapters.docker_code_execution import DockerPythonSession
from trialagentbench_harness.ports import CodeExecutionResultV1, ToolCall
from trialagentbench_harness.trialeval.agent import (
    _build_system_prompt,
    _get_tools,
    _handle_tool_call,
)
from trialagentbench_harness.trialeval.data_integrity import (
    canonical_domain_content_sha256_v1,
    repair_exact_transport_row_duplication_v1,
    stage_data_integrity_utility_v1,
    validate_declared_data_integrity_v1,
)
from trialagentbench_harness.trialeval.schema import BenchmarkItem


def _policy(*, affected_domain: str = "data/raw/events.parquet") -> dict[str, object]:
    return {
        "schema_id": "trialagentbench.trialeval.c5_integrity_policy/v1",
        "task_id": "TASK001",
        "condition_id": "exact_transport_row_duplication_v1",
        "affected_domain": affected_domain,
        "compound_key_fields": ["USUBJID", "EVENT_DY"],
        "legitimate_repeat_semantics": "Different declared keys are distinct records.",
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


def _write_declared_item(root: Path) -> pd.DataFrame:
    clean = pd.DataFrame(
        {
            "USUBJID": pd.Series(["S001", "S002"], dtype="string"),
            "EVENT_DY": pd.Series([10, 20], dtype="int64"),
            "EVENT": pd.Series([True, False], dtype="bool"),
            "VALUE": pd.Series([1.25, 2.5], dtype="float64"),
        }
    )
    affected = root / "data" / "raw" / "events.parquet"
    affected.parent.mkdir(parents=True)
    pd.concat([clean, clean.iloc[[0]]], ignore_index=True).to_parquet(affected, index=False)
    (root / "data_integrity_policy.json").write_text(
        json.dumps(_policy(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return clean


def _tool_names(*, data_integrity: bool) -> set[str]:
    functions = (
        cast(dict[str, object], tool["function"]) for tool in _get_tools("structured", data_integrity=data_integrity)
    )
    return {str(function["name"]) for function in functions}


def test_declared_repair_produces_exact_submission_record_and_analysis_input(tmp_path: Path) -> None:
    clean = _write_declared_item(tmp_path)
    repaired_path = tmp_path / "scratch" / "repaired_events.parquet"
    repaired_path.parent.mkdir()
    clean.to_parquet(repaired_path, index=False)
    expected_checksum = canonical_domain_content_sha256_v1(
        clean,
        key_fields=("USUBJID", "EVENT_DY"),
    )

    result = validate_declared_data_integrity_v1(
        root=tmp_path,
        analysis_input_path="scratch/repaired_events.parquet",
    )

    assert result == {
        "analysis_input_path": "scratch/repaired_events.parquet",
        "submission_record": {
            "condition_id": "exact_transport_row_duplication_v1",
            "affected_domain": "data/raw/events.parquet",
            "compound_key_fields": ("USUBJID", "EVENT_DY"),
            "observed_duplicate_group_count": 1,
            "observed_extra_row_count": 1,
            "repair_action": "remove_one_exact_duplicate_copy",
            "repair_status": "repaired",
            "post_repair_data_checksum": expected_checksum,
            "analysis_input_data_checksum": expected_checksum,
        },
    }
    repaired = pd.read_parquet(repaired_path)
    pd.testing.assert_frame_equal(repaired, clean)


def test_declared_repair_rejects_an_inexact_analysis_input(tmp_path: Path) -> None:
    clean = _write_declared_item(tmp_path)
    repaired_path = tmp_path / "scratch" / "repaired_events.parquet"
    repaired_path.parent.mkdir()
    changed = clean.copy()
    changed.loc[0, "VALUE"] = 9.0
    changed.to_parquet(repaired_path, index=False)

    with pytest.raises(ValueError, match="analysis input is not the exact declared repair"):
        validate_declared_data_integrity_v1(
            root=tmp_path,
            analysis_input_path="scratch/repaired_events.parquet",
        )


def test_declared_repair_rejects_nonidentical_same_key_rows(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "USUBJID": ["S001", "S001"],
            "EVENT_DY": [10, 10],
            "VALUE": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="same-key payloads are not identical"):
        repair_exact_transport_row_duplication_v1(
            frame,
            key_fields=("USUBJID", "EVENT_DY"),
        )


def test_runtime_stages_and_offers_repair_only_for_declared_items(tmp_path: Path) -> None:
    assert stage_data_integrity_utility_v1(tmp_path) is None
    assert "validate_data_integrity" not in _tool_names(data_integrity=False)

    _write_declared_item(tmp_path)
    staged = stage_data_integrity_utility_v1(tmp_path)

    assert staged == tmp_path / "interface" / "data_integrity.py"
    assert staged is not None and staged.read_text(encoding="utf-8").startswith('"""Canonical public repair')
    assert "validate_data_integrity" in _tool_names(data_integrity=True)


def test_declared_item_prompt_binds_the_repair_operation(tmp_path: Path) -> None:
    _write_declared_item(tmp_path)
    item = BenchmarkItem(
        item_id="TASK001",
        task_id="TASK001",
        trial_name="TASK001",
        design_tier="undisclosed",
        design_subtype="individual_randomized",
        assumption_tier="undisclosed",
        context_tier="C5",
        visible_dir=tmp_path,
        data_dir=tmp_path / "data" / "raw",
        raw_data_dir=tmp_path / "data" / "raw",
        data_preparation="raw_domains_declared_defect",
        task={},
    )

    prompt = _build_system_prompt(item)
    narrative_prompt = _build_system_prompt(item, submission_interface="narrative")

    assert "call validate_data_integrity with that path" in prompt
    assert "copy its submission_record into data_integrity_record" in prompt
    assert "report the returned repair action" in narrative_prompt
    assert "copy its submission_record into data_integrity_record" not in narrative_prompt


def test_declared_item_rejects_policy_for_another_task(tmp_path: Path) -> None:
    _write_declared_item(tmp_path)
    item = BenchmarkItem(
        item_id="TASK002",
        task_id="TASK002",
        trial_name="TASK002",
        design_tier="undisclosed",
        design_subtype="individual_randomized",
        assumption_tier="undisclosed",
        context_tier="C5",
        visible_dir=tmp_path,
        data_dir=tmp_path / "data" / "raw",
        task={},
    )

    with pytest.raises(ValueError, match="does not match the selected task"):
        _build_system_prompt(item)


def test_repair_tool_executes_the_staged_public_operation(tmp_path: Path) -> None:
    class Session:
        def execute(self, code: str) -> str:
            return code

        def execute_result(self, code: str) -> CodeExecutionResultV1:
            assert "from interface.data_integrity import validate_declared_data_integrity_v1" in code
            assert 'analysis_input_path="scratch/repaired.parquet"' in code
            return CodeExecutionResultV1(status="success", output='{"submission_record": {}}', elapsed_seconds=0.1)

        def close(self) -> None:
            return None

        def snapshot_scratch(self) -> Path:
            return tmp_path

    output, submission, execution = _handle_tool_call(
        ToolCall(
            id="repair",
            name="validate_data_integrity",
            arguments='{"analysis_input_path":"scratch/repaired.parquet"}',
        ),
        Session(),
        tmp_path,
        False,
        submission_interface="structured",
        required_deliverables=("evidence", "limitations", "primary_analysis"),
    )

    assert output == '{"submission_record": {}}'
    assert submission is None
    assert execution is not None and execution.status == "success"


@pytest.mark.executor
def test_staged_repair_runs_inside_the_release_executor(tmp_path: Path) -> None:
    clean = _write_declared_item(tmp_path)
    repaired_path = tmp_path / "scratch" / "repaired_events.parquet"
    repaired_path.parent.mkdir()
    clean.to_parquet(repaired_path, index=False)
    expected_checksum = canonical_domain_content_sha256_v1(
        clean,
        key_fields=("USUBJID", "EVENT_DY"),
    )
    stage_data_integrity_utility_v1(tmp_path)
    session = DockerPythonSession(cwd=tmp_path)
    try:
        result = session.execute_result(
            "from interface.data_integrity import validate_declared_data_integrity_v1\n"
            "import json\n"
            "print(json.dumps(validate_declared_data_integrity_v1("
            "analysis_input_path='scratch/repaired_events.parquet'), sort_keys=True))"
        )
    finally:
        session.close()

    assert result.status == "success"
    payload = json.loads(result.output)
    assert payload["submission_record"]["post_repair_data_checksum"] == expected_checksum
    assert payload["analysis_input_path"] == "scratch/repaired_events.parquet"
