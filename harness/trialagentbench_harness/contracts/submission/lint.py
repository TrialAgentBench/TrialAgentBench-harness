"""Deterministic score-blind validation reports for public submissions."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from trialagentbench_harness.contracts.submission.models import (
    DiagnosticMeasureV1,
    DiagnosticSummaryResultV1,
    DiagnosticTestResultV1,
    ProbabilityMeasureV1,
    TrialEvalSubmissionV1,
    validate_trialeval_required_deliverables_v1,
)
from trialagentbench_harness.contracts.trialeval_diagnostics import participant_diagnostic_dictionary_v1
from trialagentbench_harness.contracts.trialeval_methods import (
    TrialEvalParticipantMethodDictionaryV1,
    TrialEvalParticipantMethodV1,
)
from trialagentbench_harness.io.checksums import canonical_payload_sha256
from trialagentbench_harness.ports.tool_input import JsonObjectDecodeError, decode_json_object_text

SubmissionSuiteV1 = Literal["trialeval", "trialdev"]
SubmissionValidationScopeV1 = Literal["schema_only", "participant_bound"]
SubmissionLintIssueCodeV1 = Literal[
    "json_syntax",
    "json_object_required",
    "duplicate_json_field",
    "required_field",
    "extra_field",
    "invalid_enum",
    "invalid_type",
    "constraint",
    "cross_field",
    "identity_mismatch",
    "required_deliverable",
    "invalid_source_path",
    "missing_source_artifact",
    "participant_contract",
    "unknown_analysis_method",
    "method_result_incompatible",
    "diagnostic_metric_missing",
    "diagnostic_unit_mismatch",
]


class SubmissionLintIssueV1(BaseModel):
    """One stable participant-safe submission defect."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: SubmissionLintIssueCodeV1
    json_pointer: str = Field(
        description="RFC 6901 JSON Pointer. The empty string identifies the document root.",
    )
    message: str = Field(min_length=1, max_length=300)


class SubmissionLintReportV1(BaseModel):
    """Score-blind validation result for one public submission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.submission_lint/v1"] = "trialagentbench.submission_lint/v1"
    suite: SubmissionSuiteV1
    scope: SubmissionValidationScopeV1
    valid: bool
    submission_identity: str | None = None
    submission_schema_id: str | None = None
    participant_contract_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    canonical_submission_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    issues: tuple[SubmissionLintIssueV1, ...] = ()

    @model_validator(mode="after")
    def _coherent_status(self) -> Self:
        ordered = tuple(sorted(self.issues, key=lambda issue: (issue.json_pointer, issue.code, issue.message)))
        if ordered != self.issues or len(set(ordered)) != len(ordered):
            raise ValueError("submission lint issues must be unique and deterministically ordered")
        if self.valid != (not self.issues):
            raise ValueError("submission lint validity must equal absence of issues")
        if self.valid != (self.canonical_submission_sha256 is not None):
            raise ValueError("a canonical submission digest is required exactly for valid reports")
        if self.scope == "schema_only" and self.participant_contract_checksum is not None:
            raise ValueError("schema-only lint cannot bind a participant contract")
        return self


def lint_submission_text_v1(
    text: str,
    *,
    suite: SubmissionSuiteV1,
    scope: SubmissionValidationScopeV1 = "schema_only",
    expected_identity: str | None = None,
    required_deliverables: Collection[str] = (),
    participant_contract_checksum: str | None = None,
    participant_artifact_paths: Collection[str] | None = None,
    participant_method_dictionary: TrialEvalParticipantMethodDictionaryV1 | None = None,
) -> SubmissionLintReportV1:
    """Validate one JSON submission without evaluator or scoring state."""

    try:
        payload = decode_json_object_text(text)
    except JsonObjectDecodeError as exc:
        issue = SubmissionLintIssueV1(
            code=exc.code,
            json_pointer="",
            message=str(exc),
        )
        return _report(
            suite=suite,
            scope=scope,
            issues=(issue,),
            participant_contract_checksum=participant_contract_checksum,
        )
    return lint_submission_payload_v1(
        payload,
        suite=suite,
        scope=scope,
        expected_identity=expected_identity,
        required_deliverables=required_deliverables,
        participant_contract_checksum=participant_contract_checksum,
        participant_artifact_paths=participant_artifact_paths,
        participant_method_dictionary=participant_method_dictionary,
    )


def lint_submission_payload_v1(
    payload: Mapping[str, object],
    *,
    suite: SubmissionSuiteV1,
    scope: SubmissionValidationScopeV1 = "schema_only",
    expected_identity: str | None = None,
    required_deliverables: Collection[str] = (),
    participant_contract_checksum: str | None = None,
    participant_artifact_paths: Collection[str] | None = None,
    participant_method_dictionary: TrialEvalParticipantMethodDictionaryV1 | None = None,
) -> SubmissionLintReportV1:
    """Validate one decoded public submission under an explicit scope."""

    if scope == "participant_bound" and participant_contract_checksum is None and suite == "trialeval":
        raise ValueError("participant-bound TrialEval lint requires a public contract checksum")
    if scope == "participant_bound" and participant_method_dictionary is None and suite == "trialeval":
        raise ValueError("participant-bound TrialEval lint requires the public method dictionary")
    submission_model: BaseModel
    try:
        if suite == "trialeval":
            submission_model = TrialEvalSubmissionV1.model_validate(payload)
        else:
            from trialagentbench_harness.trialdev.grading.models import TrialDevelopmentSubmissionV1

            submission_model = TrialDevelopmentSubmissionV1.model_validate(payload)
    except ValidationError as exc:
        return _report(
            suite=suite,
            scope=scope,
            submission_identity=_recover_identity(payload, suite=suite),
            submission_schema_id=_recover_schema_id(payload),
            participant_contract_checksum=participant_contract_checksum,
            issues=_pydantic_issues(exc),
        )

    issues: list[SubmissionLintIssueV1] = []
    schema_id: str
    if suite == "trialeval":
        trialeval_submission = cast(TrialEvalSubmissionV1, submission_model)
        identity = trialeval_submission.task_id
        schema_id = trialeval_submission.schema_id
    else:
        from trialagentbench_harness.trialdev.grading.models import TrialDevelopmentSubmissionV1

        trialdev_submission = cast(TrialDevelopmentSubmissionV1, submission_model)
        identity = trialdev_submission.scenario_id
        schema_id = "trialagentbench.trialdev_submission/v1"
    if expected_identity is not None and identity != expected_identity:
        identity_field = "task_id" if suite == "trialeval" else "scenario_id"
        issues.append(
            SubmissionLintIssueV1(
                code="identity_mismatch",
                json_pointer=f"/{identity_field}",
                message=f"Submission {identity_field} does not match the selected participant task.",
            )
        )
    if suite == "trialeval":
        try:
            validate_trialeval_required_deliverables_v1(
                trialeval_submission,
                required_deliverables=required_deliverables,
            )
        except ValueError as exc:
            issues.append(
                SubmissionLintIssueV1(
                    code="required_deliverable",
                    json_pointer="",
                    message=_bounded_message(str(exc)),
                )
            )
        issues.extend(
            _trialeval_source_issues(
                trialeval_submission,
                participant_artifact_paths=participant_artifact_paths,
            )
        )
        if participant_method_dictionary is not None:
            issues.extend(_trialeval_method_issues(trialeval_submission, participant_method_dictionary))
        if scope == "participant_bound":
            issues.extend(_trialeval_diagnostic_issues(trialeval_submission))
    ordered = tuple(sorted(set(issues), key=lambda issue: (issue.json_pointer, issue.code, issue.message)))
    canonical_payload = submission_model.model_dump(mode="json")
    return _report(
        suite=suite,
        scope=scope,
        submission_identity=identity,
        submission_schema_id=str(schema_id),
        participant_contract_checksum=participant_contract_checksum,
        canonical_submission_sha256=canonical_payload_sha256(canonical_payload) if not ordered else None,
        issues=ordered,
    )


def render_submission_lint_v1(report: SubmissionLintReportV1) -> str:
    """Render one compact human-readable lint report."""

    scope = report.scope.replace("_", " ")
    if report.valid:
        identity = report.submission_identity or "unknown identity"
        return f"valid {scope} {report.suite} submission: {identity}"
    lines = [f"invalid {scope} {report.suite} submission ({len(report.issues)} issue(s)):"]
    for issue in report.issues:
        pointer = issue.json_pointer or "<root>"
        lines.append(f"- {issue.code} at {pointer}: {issue.message}")
    return "\n".join(lines)


def _report(
    *,
    suite: SubmissionSuiteV1,
    scope: SubmissionValidationScopeV1,
    issues: Sequence[SubmissionLintIssueV1],
    submission_identity: str | None = None,
    submission_schema_id: str | None = None,
    participant_contract_checksum: str | None = None,
    canonical_submission_sha256: str | None = None,
) -> SubmissionLintReportV1:
    ordered = tuple(sorted(set(issues), key=lambda issue: (issue.json_pointer, issue.code, issue.message)))
    return SubmissionLintReportV1(
        suite=suite,
        scope=scope,
        valid=not ordered,
        submission_identity=submission_identity,
        submission_schema_id=submission_schema_id,
        participant_contract_checksum=participant_contract_checksum,
        canonical_submission_sha256=canonical_submission_sha256,
        issues=ordered,
    )


def _recover_identity(payload: Mapping[str, object], *, suite: SubmissionSuiteV1) -> str | None:
    field = "task_id" if suite == "trialeval" else "scenario_id"
    value = payload.get(field)
    return value if isinstance(value, str) and value else None


def _recover_schema_id(payload: Mapping[str, object]) -> str | None:
    value = payload.get("schema_id")
    return value if isinstance(value, str) and value else None


def _pydantic_issues(exc: ValidationError) -> tuple[SubmissionLintIssueV1, ...]:
    issues: list[SubmissionLintIssueV1] = []
    for error in exc.errors(include_url=False, include_context=True, include_input=False):
        error_type = str(error["type"])
        code, message = _pydantic_issue_message(error_type, error)
        issues.append(
            SubmissionLintIssueV1(
                code=code,
                json_pointer=_json_pointer(error.get("loc", ())),
                message=message,
            )
        )
    return tuple(sorted(set(issues), key=lambda issue: (issue.json_pointer, issue.code, issue.message)))


def _pydantic_issue_message(
    error_type: str,
    error: Mapping[str, object],
) -> tuple[SubmissionLintIssueCodeV1, str]:
    if error_type == "missing":
        return "required_field", "Required field is missing."
    if error_type == "extra_forbidden":
        return "extra_field", "Field is not permitted by the public submission contract."
    if error_type in {"literal_error", "enum"}:
        return "invalid_enum", "Value is not in the allowed public vocabulary."
    if error_type.startswith(
        (
            "greater_than",
            "less_than",
            "string_",
            "too_long",
            "too_short",
            "finite_number",
        )
    ):
        return "constraint", "Value violates the field constraint."
    if error_type == "value_error":
        context = error.get("ctx")
        if isinstance(context, Mapping) and context.get("error") is not None:
            return "cross_field", _bounded_message(str(context["error"]))
        return "cross_field", "Fields do not satisfy the public cross-field contract."
    return "invalid_type", "Value has the wrong public JSON type or shape."


def _json_pointer(location: object) -> str:
    if not isinstance(location, (tuple, list)):
        return ""
    encoded: list[str] = []
    for component in location:
        value = str(component).replace("~", "~0").replace("/", "~1")
        encoded.append(value)
    return "/" + "/".join(encoded) if encoded else ""


def _bounded_message(message: str) -> str:
    normalized = " ".join(message.split())
    return normalized[:297] + "..." if len(normalized) > 300 else normalized


def _trialeval_source_issues(
    submission: TrialEvalSubmissionV1,
    *,
    participant_artifact_paths: Collection[str] | None,
) -> tuple[SubmissionLintIssueV1, ...]:
    allowed = set(participant_artifact_paths) if participant_artifact_paths is not None else None
    paths: list[tuple[str, str]] = []
    for evidence_index, evidence in enumerate(submission.evidence):
        for source_index, source in enumerate(evidence.source_artifacts):
            paths.append((f"/evidence/{evidence_index}/source_artifacts/{source_index}", source))
    if submission.reconstruction is not None:
        for source_index, source in enumerate(submission.reconstruction.source_artifacts):
            paths.append((f"/reconstruction/source_artifacts/{source_index}", source))

    issues: list[SubmissionLintIssueV1] = []
    for pointer, value in paths:
        path = PurePosixPath(value)
        canonical = (
            value
            and "\\" not in value
            and not path.is_absolute()
            and value == path.as_posix()
            and all(part not in {"", ".", ".."} and not part.startswith(".") for part in path.parts)
            and ":" not in value
        )
        if not canonical:
            issues.append(
                SubmissionLintIssueV1(
                    code="invalid_source_path",
                    json_pointer=pointer,
                    message="Source artifact must be a canonical participant-relative POSIX path.",
                )
            )
        elif allowed is not None and value not in allowed:
            issues.append(
                SubmissionLintIssueV1(
                    code="missing_source_artifact",
                    json_pointer=pointer,
                    message="Source artifact is absent from the selected participant task.",
                )
            )
    return tuple(issues)


def _trialeval_method_issues(
    submission: TrialEvalSubmissionV1,
    method_dictionary: TrialEvalParticipantMethodDictionaryV1,
) -> tuple[SubmissionLintIssueV1, ...]:
    """Check complete-method compatibility without consulting item eligibility."""

    methods = {str(method.method_id): method for method in method_dictionary.methods}
    declarations = [("/primary_analysis/estimator", submission.primary_analysis.estimator)]
    declarations.extend(
        (f"/evidence/{index}/estimator", record.estimator)
        for index, record in enumerate(submission.evidence)
        if record.estimator is not None
    )
    issues: list[SubmissionLintIssueV1] = []
    for pointer, declaration in declarations:
        method = methods.get(str(declaration.analysis_method_id))
        if method is None:
            issues.append(
                SubmissionLintIssueV1(
                    code="unknown_analysis_method",
                    json_pointer=f"{pointer}/analysis_method_id",
                    message="Analysis method is absent from the public method dictionary.",
                )
            )
            continue
        result = (
            submission.primary_analysis.result
            if pointer == "/primary_analysis/estimator"
            else submission.evidence[int(pointer.split("/")[2])].result
        )
        effect_scale_value = getattr(result, "effect_scale", None)
        effect_scale = str(effect_scale_value) if effect_scale_value is not None else None
        if not _result_matches_method(method, result_kind=str(result.kind), effect_scale=effect_scale):
            issues.append(
                SubmissionLintIssueV1(
                    code="method_result_incompatible",
                    json_pointer=pointer,
                    message="Declared result shape or effect scale is incompatible with the selected method record.",
                )
            )
    return tuple(issues)


def _trialeval_diagnostic_issues(
    submission: TrialEvalSubmissionV1,
) -> tuple[SubmissionLintIssueV1, ...]:
    """Check submitted empirical measures against the task-general dictionary."""

    definitions = participant_diagnostic_dictionary_v1().diagnostics
    issues: list[SubmissionLintIssueV1] = []
    for evidence_index, evidence in enumerate(submission.evidence):
        if evidence.diagnostic_id is None or not isinstance(
            evidence.result,
            DiagnosticSummaryResultV1 | DiagnosticTestResultV1,
        ):
            continue
        thresholds = definitions[evidence.diagnostic_id].severity_thresholds
        if thresholds is None:
            continue
        measures_by_id: dict[str, DiagnosticMeasureV1 | ProbabilityMeasureV1]
        if isinstance(evidence.result, DiagnosticSummaryResultV1):
            measures_by_id = {measure.metric_id: measure for measure in evidence.result.measures}
        else:
            measures_by_id = {
                evidence.result.statistic.metric_id: evidence.result.statistic,
                evidence.result.p_value.metric_id: evidence.result.p_value,
            }
        required_metric_ids = {
            thresholds.metric_name,
            *thresholds.decision_metric_names.values(),
        }
        for metric_id in sorted(required_metric_ids):
            pointer = f"/evidence/{evidence_index}/result"
            measure = measures_by_id.get(metric_id)
            if measure is None:
                issues.append(
                    SubmissionLintIssueV1(
                        code="diagnostic_metric_missing",
                        json_pointer=pointer,
                        message=f"Diagnostic result must include the task-general measure {metric_id!r}.",
                    )
                )
            elif measure.unit != thresholds.metric_unit:
                issues.append(
                    SubmissionLintIssueV1(
                        code="diagnostic_unit_mismatch",
                        json_pointer=pointer,
                        message=(
                            f"Diagnostic measure {metric_id!r} must use the task-general unit "
                            f"{thresholds.metric_unit!r}."
                        ),
                    )
                )
    return tuple(issues)


def _result_matches_method(
    method: TrialEvalParticipantMethodV1,
    *,
    result_kind: str,
    effect_scale: str | None,
) -> bool:
    compatible_public_kinds = {
        "scalar": {"numeric_point"},
        "identified_interval": {"numeric_interval", "identification_bound", "sensitivity_set"},
        "vector": {"numeric_vector", "sensitivity_set"},
        "statistical_test": {"statistical_test"},
        "non_identification": {"limitation", "abstention"},
    }.get(result_kind, set())
    return method.result_kind in compatible_public_kinds and (
        effect_scale is None or method.effect_scale == effect_scale
    )


__all__ = [
    "SubmissionLintIssueCodeV1",
    "SubmissionLintIssueV1",
    "SubmissionLintReportV1",
    "SubmissionSuiteV1",
    "SubmissionValidationScopeV1",
    "lint_submission_payload_v1",
    "lint_submission_text_v1",
    "render_submission_lint_v1",
]
