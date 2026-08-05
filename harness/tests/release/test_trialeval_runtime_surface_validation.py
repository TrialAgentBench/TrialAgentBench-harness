from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from trialagentbench_test_helpers import (
    minimal_benchmark_charter_payload,
    minimal_benchmark_map_markdown,
    minimal_participant_output_contract,
)

from trialagentbench_harness.contracts.release.trialeval_manifest import (
    TrialEvalParticipantManifestV1,
)
from trialagentbench_harness.contracts.release.trialeval_runtime_surface import (
    RuntimePolicyV1,
    TrialEvalItemMemberRoleV1,
    TrialEvalParticipantTaskV1,
    classify_trialeval_item_member,
    fingerprint_runtime_surface,
    sanitize_trialeval_agent_json_payload,
    trialeval_agent_allows_item_member,
    trialeval_agent_forbidden_json_key_paths,
    visible_runtime_file,
)
from trialagentbench_harness.tools.validate.validate_trialeval_runtime_surface import (
    validate_trialeval_runtime_surface,
)
from trialagentbench_harness.trialeval.data import stage_participant_evidence
from trialagentbench_harness.trialeval.schema import BenchmarkItem


def _bytes(value: str | bytes) -> bytes:
    return value if isinstance(value, bytes) else value.encode("utf-8")


def _zip(path: Path, files: dict[str, str | bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)


def _factors(context_tier: str) -> dict[str, str]:
    return {
        "context_configuration": context_tier,
        "data_preparation": (
            "analysis_ready"
            if context_tier in {"C1", "C2"}
            else ("raw_domains_declared_defect" if context_tier == "C5" else "raw_domains")
        ),
        "analysis_specification": ("locked_sap" if context_tier in {"C1", "C3"} else "protocol_only"),
    }


def _participant_zip(path: Path, files: dict[str, str | bytes], *, context_tier: str) -> None:
    files = {
        **files,
        "benchmark_charter.json": json.dumps(minimal_benchmark_charter_payload(), sort_keys=True) + "\n",
        "benchmark_map.md": minimal_benchmark_map_markdown(),
    }
    artifacts = [
        {
            "rel_path": name,
            "sha256": hashlib.sha256(_bytes(text)).hexdigest(),
            "size_bytes": len(_bytes(text)),
        }
        for name, text in sorted(files.items())
    ]
    manifest = TrialEvalParticipantManifestV1.model_validate(
        {
            "applied_baseline_profile_id": None,
            "applied_baseline_profile_sha256": None,
            "task_ids": ["TASK0001"],
            "task_evidence_factors": {"TASK0001": _factors(context_tier)},
            "artifacts": artifacts,
        }
    )
    _zip(path, {**files, "manifest.json": manifest.model_dump_json(indent=2) + "\n"})


def _participant_task() -> dict[str, object]:
    return {
        "schema_id": "trial_analysis_task_v1",
        "task_id": "TASK0001",
        "design_subtype": "individual_randomized",
        "primary_endpoint_id": "death",
        "primary_paramcd": "death",
        "primary_estimand_id": "itt",
        "primary_effect_scale": "log_hr",
        "estimand_mode": "fixed_declared_estimand",
        "primary_effect_scale_options": ["log_hr"],
        "primary_result_unit": "log_hazard_ratio",
        "primary_population_id": "ITT",
        "primary_intercurrent_event_strategy_ids": ["rescue_therapy:treatment_policy"],
        "primary_control_arm_id": "control",
        "primary_treated_arm_id": "active",
    }


def test_participant_task_requires_complete_unambiguous_estimand_context() -> None:
    task = TrialEvalParticipantTaskV1.model_validate(_participant_task())
    assert task.primary_result_unit == "log_hazard_ratio"

    missing_unit = _participant_task()
    missing_unit.pop("primary_result_unit")
    with pytest.raises(ValueError, match="primary_result_unit"):
        TrialEvalParticipantTaskV1.model_validate(missing_unit)

    duplicate_strategies = _participant_task()
    duplicate_strategies["primary_intercurrent_event_strategy_ids"] = [
        "rescue_therapy:treatment_policy",
        "rescue_therapy:treatment_policy",
    ]
    with pytest.raises(ValueError, match="must be unique"):
        TrialEvalParticipantTaskV1.model_validate(duplicate_strategies)

    wrong_unit = _participant_task()
    wrong_unit["primary_result_unit"] = "probability_difference"
    with pytest.raises(ValueError, match="must match"):
        TrialEvalParticipantTaskV1.model_validate(wrong_unit)

    unsupported_scale = _participant_task()
    unsupported_scale["primary_effect_scale"] = "risk_ratio"
    unsupported_scale["primary_effect_scale_options"] = ["risk_ratio"]
    with pytest.raises(ValueError, match="primary_effect_scale"):
        TrialEvalParticipantTaskV1.model_validate(unsupported_scale)

    unsupported_planning = _participant_task()
    unsupported_planning["planning"] = {"status": "ineligible", "reason": "unsupported_effect_scale"}
    with pytest.raises(ValueError, match="unsupported planning"):
        TrialEvalParticipantTaskV1.model_validate(unsupported_planning)


def test_participant_task_rejects_open_estimand_mode() -> None:
    payload = _participant_task()
    payload.update(
        {
            "estimand_mode": "open_supported_estimand",
            "primary_effect_scale_options": ["log_hr", "risk_difference_tau"],
            "primary_effect_scale": None,
            "primary_result_unit": None,
            "primary_tau_dy": 365.0,
        }
    )

    with pytest.raises(ValueError, match="fixed_declared_estimand"):
        TrialEvalParticipantTaskV1.model_validate(payload)


def _evaluator_zip(path: Path, *, context_tier: str = "C1") -> None:
    _zip(
        path,
        {
            "benchmark_charter.json": json.dumps(minimal_benchmark_charter_payload(), sort_keys=True) + "\n",
            "benchmark_map.md": minimal_benchmark_map_markdown(),
            "grader/item_index.json": json.dumps(
                {
                    "entries": [
                        {
                            "task_id": "TASK0001",
                            "item_id": "fixture_01",
                            "factors": {
                                "evaluation_series_id": "randomized",
                                "design_archetype": "D1",
                                "design_subtype": "individual_randomized",
                                "assumption_regime": "A1",
                                **_factors(context_tier),
                                "procedure_assistance": "output_contract_only",
                                "response_interface": "structured",
                            },
                        }
                    ]
                },
                sort_keys=True,
            )
            + "\n",
        },
    )


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _complete_participant_item(*, context_tier: str = "C1") -> dict[str, str | bytes]:
    prefix = "items/TASK0001/"
    files: dict[str, str | bytes] = {
        f"{prefix}README.md": "item\n",
        f"{prefix}data_dictionary.json": "{}\n",
        f"{prefix}endpoint_definition.json": "{}\n",
        f"{prefix}intercurrent_event_strategy.json": "{}\n",
        f"{prefix}protocol_summary.json": "{}\n",
        f"{prefix}study_brief.md": "study\n",
        f"{prefix}submission_contract.json": json.dumps(
            minimal_participant_output_contract(
                "TASK0001",
                data_preparation=_factors(context_tier)["data_preparation"],
            ),
            sort_keys=True,
        )
        + "\n",
        f"{prefix}task.json": json.dumps(_participant_task(), sort_keys=True) + "\n",
    }
    if context_tier in {"C1", "C2"}:
        files.update(
            {
                f"{prefix}analysis_population_guidance.json": "{}\n",
                f"{prefix}data/ADSL.parquet": _parquet_bytes(
                    pd.DataFrame({"USUBJID": [f"SUBJ-{index:03d}" for index in range(10)]})
                ),
                f"{prefix}data/ADTTE.parquet": _parquet_bytes(
                    pd.DataFrame(
                        {
                            "USUBJID": [f"SUBJ-{index:03d}" for index in range(10)],
                            "PARAMCD": ["death"] * 10,
                            "CNSR": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                        }
                    )
                ),
            }
        )
    else:
        files.update(
            {
                f"{prefix}reconstruction_task.json": "{}\n",
                f"{prefix}data/raw/subjects.parquet": "fixture",
            }
        )
    if context_tier == "C5":
        files[f"{prefix}data_integrity_policy.json"] = json.dumps(
            {
                "schema_id": "trialagentbench.trialeval.c5_integrity_policy/v1",
                "task_id": "TASK0001",
                "condition_id": "exact_transport_row_duplication_v1",
                "affected_domain": "data/raw/subjects.parquet",
                "compound_key_fields": ["USUBJID"],
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
            },
            sort_keys=True,
        )
    if context_tier in {"C1", "C3"}:
        files[f"{prefix}analysis_plan.json"] = "{}\n"
        files[f"{prefix}analysis_tasks.md"] = "analysis\n"
    return files


@pytest.mark.parametrize("context_tier", ["C1", "C2", "C3", "C4", "C5"])
def test_trialeval_runtime_surface_validator_accepts_sanitized_surface(tmp_path: Path, context_tier: str) -> None:
    """A sanitized participant ZIP has no prompt/filesystem parity findings."""

    public_zip = tmp_path / "public.zip"
    evaluator_zip = tmp_path / "evaluator.zip"
    _evaluator_zip(evaluator_zip, context_tier=context_tier)
    files = _complete_participant_item(context_tier=context_tier)
    task = _participant_task()
    task["primary_question"] = "Estimate the effect."
    files["items/TASK0001/task.json"] = json.dumps(task, sort_keys=True) + "\n"
    files["items/TASK0001/endpoint_definition.json"] = '{"endpoint_id":"death"}\n'
    _participant_zip(public_zip, files, context_tier=context_tier)

    findings = validate_trialeval_runtime_surface(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert findings == []


def test_trialeval_runtime_surface_rejects_invalid_data_integrity_policy(tmp_path: Path) -> None:
    """The release boundary validates the executable C5 policy contract."""

    public_zip = tmp_path / "public.zip"
    evaluator_zip = tmp_path / "evaluator.zip"
    _evaluator_zip(evaluator_zip, context_tier="C5")
    files = _complete_participant_item(context_tier="C5")
    policy_path = "items/TASK0001/data_integrity_policy.json"
    policy = json.loads(files[policy_path])
    policy["repair_action"] = "infer_a_repair"
    files[policy_path] = json.dumps(policy, sort_keys=True)
    _participant_zip(public_zip, files, context_tier="C5")

    findings = validate_trialeval_runtime_surface(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert any("participant data-integrity policy is invalid" in finding.message for finding in findings)


def test_trialeval_runtime_surface_rejects_representation_specific_participant_contract(tmp_path: Path) -> None:
    """The participant contract cannot select JSON or narrative representation."""

    public_zip = tmp_path / "public.zip"
    evaluator_zip = tmp_path / "evaluator.zip"
    _evaluator_zip(evaluator_zip)
    files = _complete_participant_item()
    contract = minimal_participant_output_contract("TASK0001")
    contract["accepted_formats"] = ["json"]
    unsigned = {key: value for key, value in contract.items() if key != "checksum"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    contract["checksum"] = hashlib.sha256(canonical).hexdigest()
    files["items/TASK0001/submission_contract.json"] = json.dumps(contract, sort_keys=True)
    _participant_zip(public_zip, files, context_tier="C1")

    findings = validate_trialeval_runtime_surface(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert any(
        "participant output contract is invalid" in finding.message and "accepted_formats" in finding.message
        for finding in findings
    )


def test_trialeval_runtime_surface_rejects_output_contract_checksum_drift(tmp_path: Path) -> None:
    """Semantic output requirements cannot change without a new checksum."""

    public_zip = tmp_path / "public.zip"
    evaluator_zip = tmp_path / "evaluator.zip"
    _evaluator_zip(evaluator_zip)
    files = _complete_participant_item()
    contract = minimal_participant_output_contract("TASK0001")
    contract["required_deliverables"] = ["evidence", "limitations", "primary_analysis", "reconstruction"]
    files["items/TASK0001/submission_contract.json"] = json.dumps(contract, sort_keys=True)
    _participant_zip(public_zip, files, context_tier="C1")

    findings = validate_trialeval_runtime_surface(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert any("checksum does not match" in finding.message for finding in findings)


def test_trialeval_runtime_surface_validator_rejects_provenance_files(tmp_path: Path) -> None:
    """Internal provenance must not be readable from the model workdir."""

    public_zip = tmp_path / "public.zip"
    evaluator_zip = tmp_path / "evaluator.zip"
    _evaluator_zip(evaluator_zip, context_tier="C2")
    files = _complete_participant_item(context_tier="C2")
    files["items/TASK0001/source_trace.json"] = "{}\n"
    _participant_zip(public_zip, files, context_tier="C2")

    findings = validate_trialeval_runtime_surface(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert any("source_trace.json" in finding.path for finding in findings)
    assert any("evaluator-only file" in finding.message for finding in findings)


def test_trialeval_runtime_surface_validator_rejects_target_bearing_json(tmp_path: Path) -> None:
    """Target-bearing JSON fields are not allowed on the model-visible filesystem."""

    public_zip = tmp_path / "public.zip"
    evaluator_zip = tmp_path / "evaluator.zip"
    _evaluator_zip(evaluator_zip, context_tier="C3")
    _zip(
        public_zip,
        {
            "items/TASK0001/task.json": '{"task_id":"TASK0001","primary_effect_scale":"rmst_difference"}\n',
            "items/TASK0001/analysis_plan.json": '{"lane_rules":[{"target_method_id":"oracle"}]}\n',
        },
    )

    findings = validate_trialeval_runtime_surface(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert any("target_method_id" in finding.message for finding in findings)


def test_trialeval_runtime_surface_rejects_dangling_or_unsafe_task_references(tmp_path: Path) -> None:
    """Participant task file references must resolve inside the same item."""

    public_zip = tmp_path / "public.zip"
    evaluator_zip = tmp_path / "evaluator.zip"
    _evaluator_zip(evaluator_zip)
    _zip(
        public_zip,
        {
            "items/TASK0001/task.json": json.dumps(
                {
                    "task_id": "TASK0001",
                    "data_dictionary_file": "missing.json",
                    "protocol_summary_file": "../protocol.json",
                }
            ),
        },
    )

    findings = validate_trialeval_runtime_surface(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert any("references missing participant member" in finding.message for finding in findings)
    assert any("unsafe item-relative path" in finding.message for finding in findings)


def test_task_sanitizer_removes_unavailable_optional_file_references() -> None:
    """Packaged task metadata lists only files actually mounted to the participant."""

    sanitized = sanitize_trialeval_agent_json_payload(
        archive_name="items/TASK0001/task.json",
        payload={
            "task_id": "TASK0001",
            "protocol_summary_file": "protocol_summary.json",
            "data_dictionary_file": "data_dictionary.json",
        },
        available_item_members=frozenset({"task.json", "protocol_summary.json"}),
    )

    assert sanitized["protocol_summary_file"] == "protocol_summary.json"
    assert "data_dictionary_file" not in sanitized


def test_task_sanitizer_retains_declared_estimand_without_acceptance_keys() -> None:
    """Every context tier retains the requested estimand and contrast."""

    sanitized = sanitize_trialeval_agent_json_payload(
        archive_name="items/TASK0001/task.json",
        payload={
            "task_id": "TASK0001",
            "primary_endpoint_id": "death",
            "primary_paramcd": "death",
            "primary_control_arm_id": "control",
            "primary_treated_arm_id": "treated",
            "primary_effect_scale": "rmst_difference_tau",
            "estimand_mode": "fixed_declared_estimand",
            "primary_effect_scale_options": ["rmst_difference_tau"],
            "primary_estimand_id": "itt_treatment_policy",
        },
    )

    assert sanitized["primary_effect_scale"] == "rmst_difference_tau"
    assert sanitized["primary_effect_scale_options"] == ["rmst_difference_tau"]
    assert sanitized["primary_estimand_id"] == "itt_treatment_policy"


def test_task_sanitizer_preserves_declared_scale_in_c1() -> None:
    """C1 retains the same declared estimand contract as every other tier."""

    payload = {
        "task_id": "TASK0001",
        "primary_effect_scale": "rmst_difference_tau",
        "estimand_mode": "fixed_declared_estimand",
        "primary_effect_scale_options": ["rmst_difference_tau"],
        "primary_estimand_id": "itt_treatment_policy",
        "primary_question": "Estimate the RMST difference through follow-up.",
    }

    sanitized = sanitize_trialeval_agent_json_payload(
        archive_name="items/TASK0001/task.json",
        payload=payload,
    )

    assert sanitized == payload


def test_runtime_leak_policy_allows_endpoint_but_rejects_answer_keys() -> None:
    findings = trialeval_agent_forbidden_json_key_paths(
        {
            "primary_endpoint_id": "death",
            "primary_effect_scale": "rmst_difference_tau",
            "credit_eligible_route_families": ["rmst_contrast"],
        },
    )

    assert "primary_endpoint_id" not in findings
    assert findings == ["credit_eligible_route_families"]

    defect_findings = trialeval_agent_forbidden_json_key_paths(
        {
            "defect_class": "record_identity_conflict",
            "permitted_resolutions": ["drop_record", "manual_review_required"],
            "affected_records": [{"USUBJID": "SUBJ-001"}],
            "intended_resolution": "drop_record",
        }
    )
    assert defect_findings == ["affected_records", "intended_resolution"]


def test_trialeval_analysis_plan_sanitizer_removes_nested_target_fields() -> None:
    """Analysis-plan sanitization removes target-bearing fields at top level and lane level."""

    sanitized = sanitize_trialeval_agent_json_payload(
        archive_name="items/TASK0001/analysis_plan.json",
        payload={
            "task_id": "TASK0001",
            "effect_scale": "rmst_difference",
            "route_reference_id": "truth_001",
            "lane_rules": [
                {
                    "lane_id": "primary",
                    "target_method_id": "rmst_ipcw",
                    "route_reference_id": "truth_001",
                    "participant_hint": "Use the public analysis table.",
                }
            ],
        },
    )

    assert sanitized == {
        "task_id": "TASK0001",
        "effect_scale": "rmst_difference",
        "lane_rules": [
            {
                "lane_id": "primary",
                "participant_hint": "Use the public analysis table.",
            }
        ],
    }


def _runtime_policy() -> RuntimePolicyV1:
    return RuntimePolicyV1(
        suite="trialeval",
        evidence_factors=_factors("C1"),
        runner_contract_version="fixture-v1",
        network_access=False,
        writable_workdir=True,
        timeout_seconds=60,
    )


def test_runtime_surface_fingerprint_changes_with_prompt_file_or_policy(tmp_path: Path) -> None:
    """Every participant-visible task-state component affects identity."""

    root = tmp_path / "surface"
    root.mkdir()
    item = root / "task.json"
    item.write_text('{"task_id":"TASK0001"}\n', encoding="utf-8")
    record = visible_runtime_file(root=root, relative_path="task.json")
    baseline = fingerprint_runtime_surface(task_text="Analyse the trial.", files=(record,), policy=_runtime_policy())

    changed_prompt = fingerprint_runtime_surface(
        task_text="Analyse this trial.", files=(record,), policy=_runtime_policy()
    )
    item.write_text('{"task_id":"TASK0002"}\n', encoding="utf-8")
    changed_file = fingerprint_runtime_surface(
        task_text="Analyse the trial.",
        files=(visible_runtime_file(root=root, relative_path="task.json"),),
        policy=_runtime_policy(),
    )
    changed_policy = fingerprint_runtime_surface(
        task_text="Analyse the trial.",
        files=(record,),
        policy=_runtime_policy().model_copy(update={"timeout_seconds": 61}),
    )

    assert (
        len(
            {
                baseline.fingerprint_sha256,
                changed_prompt.fingerprint_sha256,
                changed_file.fingerprint_sha256,
                changed_policy.fingerprint_sha256,
            }
        )
        == 4
    )


def test_runtime_surface_fingerprint_is_order_invariant(tmp_path: Path) -> None:
    """File discovery order does not alter the task-state identity."""

    root = tmp_path / "surface"
    root.mkdir()
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "b.txt").write_text("b", encoding="utf-8")
    a = visible_runtime_file(root=root, relative_path="a.txt")
    b = visible_runtime_file(root=root, relative_path="b.txt")

    first = fingerprint_runtime_surface(task_text="task", files=(a, b), policy=_runtime_policy())
    second = fingerprint_runtime_surface(task_text="task", files=(b, a), policy=_runtime_policy())

    assert first == second


def test_visible_runtime_file_rejects_unsafe_paths_and_symlinks(tmp_path: Path) -> None:
    """Traversal, duplicate paths, missing files, and symlinks fail closed."""

    root = tmp_path / "surface"
    root.mkdir()
    (root / "item.txt").write_text("item", encoding="utf-8")
    (root / "link.txt").symlink_to(root / "item.txt")

    with pytest.raises(ValueError, match="normalized relative"):
        visible_runtime_file(root=root, relative_path="../item.txt")
    with pytest.raises(FileNotFoundError):
        visible_runtime_file(root=root, relative_path="missing.txt")
    with pytest.raises(ValueError, match="must not be symlinks"):
        visible_runtime_file(root=root, relative_path="link.txt")

    record = visible_runtime_file(root=root, relative_path="item.txt")
    with pytest.raises(ValueError, match="must be unique"):
        fingerprint_runtime_surface(task_text="task", files=(record, record), policy=_runtime_policy())


@pytest.mark.parametrize("context_tier", ["C1", "C2", "C3", "C4", "C5"])
def test_all_context_tiers_reject_global_analysis_instructions(context_tier: str) -> None:
    """Method instructions belong only in a locked SAP, never a global helper."""

    with pytest.raises(ValueError, match="Unknown TrialEval item metadata member"):
        trialeval_agent_allows_item_member(
            item_relative_path="analysis_conventions.md",
            data_preparation=_factors(context_tier)["data_preparation"],
        )


@pytest.mark.parametrize("context_tier", ["C1", "C2"])
def test_analysis_ready_tiers_allow_only_top_level_analysis_data(context_tier: str) -> None:
    """C1/C2 retain prepared tables and exclude raw/reference reconstruction trees."""

    assert trialeval_agent_allows_item_member(
        item_relative_path="data/ADSL.parquet",
        data_preparation="analysis_ready",
    )
    assert (
        classify_trialeval_item_member(
            item_relative_path="data/ADSL.parquet",
            data_preparation="analysis_ready",
        )
        == TrialEvalItemMemberRoleV1.PREPARED_ANALYSIS_DATA
    )
    with pytest.raises(ValueError, match="require data directly"):
        trialeval_agent_allows_item_member(
            item_relative_path="data/raw/subjects.parquet",
            data_preparation="analysis_ready",
        )
    with pytest.raises(ValueError, match="require data directly"):
        trialeval_agent_allows_item_member(
            item_relative_path="data/public_reconstruction/ADSL.parquet",
            data_preparation="analysis_ready",
        )


@pytest.mark.parametrize("context_tier", ["C3", "C4", "C5"])
def test_reconstruction_tiers_allow_only_raw_data(context_tier: str) -> None:
    """C3-C5 retain raw inputs and exclude prepared or completed reference tables."""

    assert trialeval_agent_allows_item_member(
        item_relative_path="data/raw/subjects.parquet",
        data_preparation="raw_domains",
    )
    assert (
        classify_trialeval_item_member(
            item_relative_path="data/raw/subjects.parquet",
            data_preparation="raw_domains",
        )
        == TrialEvalItemMemberRoleV1.SOURCE_DOMAIN_DATA
    )
    with pytest.raises(ValueError, match="require data under data/raw"):
        trialeval_agent_allows_item_member(
            item_relative_path="data/ADSL.parquet",
            data_preparation="raw_domains",
        )
    with pytest.raises(ValueError, match="Unknown TrialEval item member"):
        trialeval_agent_allows_item_member(
            item_relative_path="data/public_reconstruction/public_reconstruction_result.json",
            data_preparation="raw_domains",
        )


def test_member_classifier_rejects_unknown_metadata_and_keeps_output_contract() -> None:
    """Unknown files fail closed while the participant output contract remains visible."""

    assert (
        classify_trialeval_item_member(
            item_relative_path="submission_contract.json",
            data_preparation="analysis_ready",
        )
        == TrialEvalItemMemberRoleV1.OUTPUT_CONTRACT
    )
    assert trialeval_agent_allows_item_member(
        item_relative_path="submission_contract.json",
        data_preparation="analysis_ready",
    )
    with pytest.raises(ValueError, match="Unknown TrialEval item metadata member"):
        classify_trialeval_item_member(
            item_relative_path="notes.txt",
            data_preparation="analysis_ready",
        )


def test_staged_model_evidence_applies_participant_projection(tmp_path: Path) -> None:
    """The model mount retains output contracts and sanitizes participant JSON."""

    visible = tmp_path / "source"
    (visible / "data").mkdir(parents=True)
    (visible / "task.json").write_text(
        json.dumps({"task_id": "TASK1", "analysis_plan_file": "analysis_plan.json"}),
        encoding="utf-8",
    )
    (visible / "analysis_plan.json").write_text(
        json.dumps({"effect_scale": "risk_difference", "lane_id": "primary_analysis.response.v1"}),
        encoding="utf-8",
    )
    output_contract = minimal_participant_output_contract("TASK1")
    (visible / "submission_contract.json").write_text(json.dumps(output_contract))
    (visible / "data" / "ADSL.parquet").write_bytes(b"fixture")
    item = BenchmarkItem(
        item_id="item",
        trial_name="trial",
        design_tier="D1",
        design_subtype="individual_randomized",
        assumption_tier="A1",
        context_tier="C1",
        data_preparation="analysis_ready",
        analysis_specification="locked_sap",
        visible_dir=visible,
        data_dir=visible / "data",
        task={"task_id": "TASK1"},
    )

    staged = stage_participant_evidence(item, tmp_path / "staged")

    assert json.loads((staged / "submission_contract.json").read_text(encoding="utf-8")) == output_contract
    assert (staged / "data" / "ADSL.parquet").read_bytes() == b"fixture"
    plan = json.loads((staged / "analysis_plan.json").read_text(encoding="utf-8"))
    assert plan == {"effect_scale": "risk_difference", "lane_id": "primary_analysis.response.v1"}
