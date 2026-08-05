"""Parity and coverage tests for score-blind submission linting."""

from __future__ import annotations

import copy
import json

import pytest

from trialagentbench_harness.contracts.submission import (
    TrialEvalSubmissionV1,
    lint_submission_payload_v1,
    lint_submission_text_v1,
    trialeval_submission_shape_catalogue,
)
from trialagentbench_harness.contracts.trialeval_methods import (
    TrialEvalParticipantMethodDictionaryV1,
    TrialEvalParticipantMethodV1,
)
from trialagentbench_harness.trialeval.agent import (
    _decode_response_v1,
    _validate_structured_submission_v1,
)

_CONTRACT_CHECKSUM = "a" * 64
_CORE_DELIVERABLES = ("evidence", "limitations", "primary_analysis")


def _scalar_payload() -> dict[str, object]:
    return trialeval_submission_shape_catalogue().primary_submissions["numeric_point:scalar"].model_dump(mode="json")


def _method_dictionary() -> TrialEvalParticipantMethodDictionaryV1:
    return TrialEvalParticipantMethodDictionaryV1(
        methods=(
            TrialEvalParticipantMethodV1(
                method_id="example_analysis_method",
                estimator_family="example_family",
                objective="estimation",
                result_kind="numeric_point",
                effect_scale="log_hr",
                uncertainty_method_id="example_uncertainty",
                description="Example point-estimation method.",
            ),
            TrialEvalParticipantMethodV1(
                method_id="example_interval_method",
                estimator_family="example_bounds",
                objective="estimation",
                result_kind="identification_bound",
                effect_scale="log_hr",
                uncertainty_method_id="identified_set",
                description="Example interval method.",
            ),
        )
    )


def test_catalogue_covers_every_retained_primary_and_evidence_shape() -> None:
    catalogue = trialeval_submission_shape_catalogue()

    assert {row.primary_analysis.result_kind for row in catalogue.primary_submissions.values()} == {
        "abstention",
        "identification_bound",
        "limitation",
        "numeric_point",
        "numeric_vector",
        "sensitivity_set",
        "statistical_test",
    }
    assert {row.result.kind for row in catalogue.evidence_records.values()} == {
        "diagnostic_summary",
        "diagnostic_test",
        "factual_premise",
        "identified_interval",
        "non_identification",
        "scalar",
        "statistical_test",
        "vector",
    }
    for submission in catalogue.primary_submissions.values():
        report = lint_submission_payload_v1(
            submission.model_dump(mode="json"),
            suite="trialeval",
            required_deliverables=_CORE_DELIVERABLES,
        )
        assert report.valid
        assert report.canonical_submission_sha256 is not None


def test_optional_fragments_compose_with_their_declared_submission_shapes() -> None:
    catalogue = trialeval_submission_shape_catalogue()
    c5 = _scalar_payload()
    c5["reconstruction"] = catalogue.optional_fragments.reconstruction.model_dump(mode="json")
    c5["data_integrity_record"] = catalogue.optional_fragments.data_integrity_record.model_dump(mode="json")
    planning = _scalar_payload()
    planning["planning"] = catalogue.optional_fragments.planning.model_dump(mode="json")

    assert TrialEvalSubmissionV1.model_validate(c5).data_integrity_record is not None
    assert TrialEvalSubmissionV1.model_validate(planning).planning is not None


def test_participant_bound_lint_enforces_deliverables_and_public_source_paths() -> None:
    payload = _scalar_payload()
    payload["limitations"] = []

    report = lint_submission_payload_v1(
        payload,
        suite="trialeval",
        scope="participant_bound",
        expected_identity="TASK_EXAMPLE",
        required_deliverables=_CORE_DELIVERABLES,
        participant_contract_checksum=_CONTRACT_CHECKSUM,
        participant_artifact_paths=("task.json",),
        participant_method_dictionary=_method_dictionary(),
    )

    assert not report.valid
    assert {(issue.code, issue.json_pointer) for issue in report.issues} == {
        ("missing_source_artifact", "/evidence/0/source_artifacts/0"),
        ("required_deliverable", ""),
    }
    assert report.canonical_submission_sha256 is None


def test_participant_lint_requires_declared_diagnostic_measure_identity_and_unit() -> None:
    payload = _scalar_payload()
    diagnostic = trialeval_submission_shape_catalogue().evidence_records["diagnostic_summary"].model_dump(mode="json")
    diagnostic["evidence_id"] = "example_evidence"
    diagnostic["diagnostic_id"] = "censoring_followup_public"
    payload["evidence"] = [diagnostic]

    missing = lint_submission_payload_v1(
        payload,
        suite="trialeval",
        scope="participant_bound",
        participant_contract_checksum=_CONTRACT_CHECKSUM,
        participant_artifact_paths=("task.json",),
        participant_method_dictionary=_method_dictionary(),
    )
    assert [(issue.code, issue.json_pointer) for issue in missing.issues] == [
        ("diagnostic_metric_missing", "/evidence/0/result")
    ]

    diagnostic["result"]["measures"] = [  # type: ignore[index]
        {
            "metric_id": "lower_abs_prognostic_censoring_log_hr",
            "value": 0.0,
            "unit": "proportion",
            "decimal_places": 3,
        }
    ]
    wrong_unit = lint_submission_payload_v1(
        payload,
        suite="trialeval",
        scope="participant_bound",
        participant_contract_checksum=_CONTRACT_CHECKSUM,
        participant_artifact_paths=("task.json",),
        participant_method_dictionary=_method_dictionary(),
    )
    assert [(issue.code, issue.json_pointer) for issue in wrong_unit.issues] == [
        ("diagnostic_unit_mismatch", "/evidence/0/result")
    ]

    diagnostic["result"]["measures"][0]["unit"] = "log_hazard_ratio"  # type: ignore[index]
    valid = lint_submission_payload_v1(
        payload,
        suite="trialeval",
        scope="participant_bound",
        participant_contract_checksum=_CONTRACT_CHECKSUM,
        participant_artifact_paths=("task.json",),
        participant_method_dictionary=_method_dictionary(),
    )
    assert valid.valid


def test_invalid_result_kind_has_stable_code_and_pointer() -> None:
    payload = _scalar_payload()
    primary = payload["primary_analysis"]
    assert isinstance(primary, dict)
    primary["result_kind"] = "decision"

    report = lint_submission_payload_v1(payload, suite="trialeval")

    assert not report.valid
    assert [(issue.code, issue.json_pointer) for issue in report.issues] == [
        ("invalid_enum", "/primary_analysis/result_kind")
    ]


def test_direct_and_file_submit_adapters_share_lint_semantics() -> None:
    payload = _scalar_payload()
    artifact_paths = ("data/example.parquet",)
    direct = _validate_structured_submission_v1(
        copy.deepcopy(payload),
        required_deliverables=_CORE_DELIVERABLES,
        label="direct",
        expected_task_id="TASK_EXAMPLE",
        participant_contract_checksum=_CONTRACT_CHECKSUM,
        participant_artifact_paths=artifact_paths,
        participant_method_dictionary=_method_dictionary(),
    )
    file_submission = _decode_response_v1(
        json.dumps(payload),
        submission_interface="structured",
        required_deliverables=_CORE_DELIVERABLES,
        label="file",
        expected_task_id="TASK_EXAMPLE",
        participant_contract_checksum=_CONTRACT_CHECKSUM,
        participant_artifact_paths=artifact_paths,
        participant_method_dictionary=_method_dictionary(),
    )

    assert direct == file_submission


def test_direct_and_file_submit_adapters_reject_the_same_cross_field_defect() -> None:
    payload = _scalar_payload()
    primary = payload["primary_analysis"]
    assert isinstance(primary, dict)
    primary["evidence_ids"] = ["missing"]
    arguments = {
        "required_deliverables": _CORE_DELIVERABLES,
        "expected_task_id": "TASK_EXAMPLE",
        "participant_contract_checksum": _CONTRACT_CHECKSUM,
        "participant_artifact_paths": ("data/example.parquet",),
        "participant_method_dictionary": _method_dictionary(),
    }

    with pytest.raises(ValueError) as direct_error:
        _validate_structured_submission_v1(copy.deepcopy(payload), label="direct", **arguments)
    with pytest.raises(ValueError) as file_error:
        _decode_response_v1(
            json.dumps(payload),
            submission_interface="structured",
            label="file",
            **arguments,
        )

    assert "cross_field at <root>" in str(direct_error.value)
    assert str(direct_error.value).split("invalid:\n", 1)[1] == str(file_error.value).split("invalid:\n", 1)[1]


def test_nested_duplicate_key_is_rejected_before_schema_validation() -> None:
    report = lint_submission_text_v1(
        '{"primary_analysis":{"result_kind":"numeric_point","result_kind":"limitation"}}',
        suite="trialeval",
    )

    assert [(issue.code, issue.json_pointer) for issue in report.issues] == [("duplicate_json_field", "")]


def test_participant_lint_rejects_unknown_or_result_incompatible_method_without_item_answers() -> None:
    unknown = _scalar_payload()
    unknown["primary_analysis"]["estimator"]["analysis_method_id"] = "unknown_method"  # type: ignore[index]
    unknown_report = lint_submission_payload_v1(
        unknown,
        suite="trialeval",
        scope="participant_bound",
        participant_contract_checksum=_CONTRACT_CHECKSUM,
        participant_method_dictionary=_method_dictionary(),
    )
    assert [(row.code, row.json_pointer) for row in unknown_report.issues] == [
        ("unknown_analysis_method", "/primary_analysis/estimator/analysis_method_id")
    ]

    incompatible = _scalar_payload()
    incompatible["primary_analysis"]["estimator"]["analysis_method_id"] = "example_interval_method"  # type: ignore[index]
    incompatible["evidence"][0]["estimator"]["analysis_method_id"] = "example_interval_method"  # type: ignore[index]
    incompatible_report = lint_submission_payload_v1(
        incompatible,
        suite="trialeval",
        scope="participant_bound",
        participant_contract_checksum=_CONTRACT_CHECKSUM,
        participant_method_dictionary=_method_dictionary(),
    )
    assert {row.code for row in incompatible_report.issues} == {"method_result_incompatible"}


def test_audit_qualifications_do_not_mask_method_result_incompatibility() -> None:
    payload = _scalar_payload()
    estimator = payload["primary_analysis"]["estimator"]  # type: ignore[index]
    estimator["analysis_method_id"] = "example_interval_method"  # type: ignore[index]
    estimator["qualifications"] = ["second", "first", "second"]  # type: ignore[index]

    report = lint_submission_payload_v1(
        payload,
        suite="trialeval",
        scope="participant_bound",
        participant_contract_checksum=_CONTRACT_CHECKSUM,
        participant_method_dictionary=_method_dictionary(),
    )

    assert [(row.code, row.json_pointer) for row in report.issues] == [
        ("method_result_incompatible", "/primary_analysis/estimator")
    ]


def test_participant_lint_accepts_any_globally_known_compatible_method() -> None:
    payload = _scalar_payload()
    dictionary = _method_dictionary().model_copy(
        update={
            "methods": _method_dictionary().methods
            + (
                TrialEvalParticipantMethodV1(
                    method_id="globally_known_alternative",
                    estimator_family="alternative_family",
                    objective="estimation",
                    result_kind="numeric_point",
                    effect_scale="log_hr",
                    uncertainty_method_id="alternative_uncertainty",
                    description="A globally available compatible method.",
                ),
            )
        }
    )
    payload["primary_analysis"]["estimator"]["analysis_method_id"] = "globally_known_alternative"  # type: ignore[index]
    payload["evidence"][0]["estimator"]["analysis_method_id"] = "globally_known_alternative"  # type: ignore[index]
    report = lint_submission_payload_v1(
        payload,
        suite="trialeval",
        scope="participant_bound",
        participant_contract_checksum=_CONTRACT_CHECKSUM,
        participant_method_dictionary=dictionary,
    )
    assert report.valid


def test_participant_method_dictionary_contains_no_item_answers() -> None:
    payload = json.dumps(_method_dictionary().model_dump(mode="json"), sort_keys=True)

    for forbidden_field in (
        '"acceptance_envelope"',
        '"credit_eligible"',
        '"item_id"',
        '"required_diagnostics"',
        '"route_id"',
        '"target"',
        '"tolerance"',
    ):
        assert forbidden_field not in payload
