"""Build TrialEvalBench participant-context sufficiency witnesses."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Literal
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict

from trialagentbench_harness.analysis.trialeval_release import (
    load_evaluator_item_index,
    read_json_object_member,
)
from trialagentbench_harness.contracts.core.trialeval_factors import (
    TrialEvalDataPreparationV1,
    TrialEvalEvidenceFactorsV1,
)
from trialagentbench_harness.contracts.release.trialeval_runtime_surface import (
    TrialEvalParticipantTaskV1,
    required_trialeval_participant_members,
)
from trialagentbench_harness.contracts.release.trialeval_sap import TrialEvalPublicSAPV1
from trialagentbench_harness.contracts.trialeval_diagnostics import (
    TrialEvalParticipantDiagnosticDictionaryV1,
    participant_diagnostic_dictionary_v1,
)
from trialagentbench_harness.contracts.trialeval_methods import (
    TrialEvalParticipantMethodDictionaryV1,
    TrialEvalParticipantMethodV1,
)
from trialagentbench_harness.grading.key_store import ScoringKeyStoreV1
from trialagentbench_harness.grading.models import (
    CreditEligibleScoringRouteV1,
    ValidatedScoringKeyV1,
)
from trialagentbench_harness.io.json import write_json_model
from trialagentbench_harness.verification.trialeval.analysis_class_witness import (
    AnalysisClassWitnessV1,
    PublicAnalysisClassV1,
    recover_analysis_class_v1,
)

ContextSufficiencyStatusV1 = Literal["pass", "fail"]
ItemDispositionV1 = Literal[
    "retain_official",
    "repair_required",
    "block_release",
]


class TrialEvalContextSufficiencyRowV1(BaseModel):
    """Per-item participant-context sufficiency witness row."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval_context_sufficiency_row/v1"] = (
        "trialagentbench.trialeval_context_sufficiency_row/v1"
    )
    task_id: str
    item_id: str
    variant_id: str
    design_tier: str
    assumption_tier: str
    context_tier: str
    design_family: str
    data_preparation: TrialEvalDataPreparationV1
    analysis_specification: Literal["protocol_only", "locked_sap"]
    participant_files: tuple[str, ...]
    required_public_fields: tuple[str, ...]
    design_adjustment_fields: tuple[str, ...]
    has_task_contract: bool
    has_protocol_summary: bool
    has_endpoint_definition: bool
    has_intercurrent_event_strategy: bool
    has_analysis_ready_surface: bool
    has_reconstruction_protocol: bool
    has_public_reconstruction_output: bool
    has_raw_reconstruction_inputs: bool
    has_design_adjustment_evidence: bool
    sap_integrity_passed: bool
    analysis_class_witness: AnalysisClassWitnessV1
    credit_eligible_route_ids: tuple[str, ...]
    recoverable_route_ids: tuple[str, ...]
    unrecoverable_route_ids: tuple[str, ...]
    status: ContextSufficiencyStatusV1
    disposition: ItemDispositionV1
    findings: tuple[str, ...] = ()


class TrialEvalContextSufficiencySummaryRowV1(BaseModel):
    """Grouped context-sufficiency summary row."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval_context_sufficiency_summary_row/v1"] = (
        "trialagentbench.trialeval_context_sufficiency_summary_row/v1"
    )
    group_kind: Literal["context_tier", "design_family", "design_tier"]
    group_value: str
    total: int
    passed: int
    failed: int
    retain_official: int
    repair_required: int
    block_release: int


class TrialEvalContextSufficiencyReportV1(BaseModel):
    """TrialEvalBench participant-context sufficiency report."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval_context_sufficiency_report/v1"] = (
        "trialagentbench.trialeval_context_sufficiency_report/v1"
    )
    public_zip: str
    evaluator_zip: str
    status: ContextSufficiencyStatusV1
    total_items: int
    failed_items: int
    analysis_ready_items: int
    reconstruction_items: int
    d4_items: int
    d4_design_family_counts: dict[str, int]
    rows: tuple[TrialEvalContextSufficiencyRowV1, ...]
    summaries: tuple[TrialEvalContextSufficiencySummaryRowV1, ...]


def _item_members(public_members: set[str], task_id: str) -> tuple[str, ...]:
    prefix = f"items/{task_id}/"
    return tuple(sorted(member for member in public_members if member.startswith(prefix)))


def _required_public_fields(
    *,
    factors: TrialEvalEvidenceFactorsV1,
    design_family: str,
) -> tuple[str, ...]:
    fields = set(
        required_trialeval_participant_members(
            data_preparation=factors.data_preparation,
            analysis_specification=factors.analysis_specification,
        )
    )
    if factors.data_preparation == "analysis_ready":
        fields.update(["data/ADSL.parquet", "data/ADTTE.parquet"])
    else:
        fields.update(
            [
                "data/raw/subjects.parquet",
                "data/raw/randomization.parquet",
            ]
        )
    if design_family in {"cluster_parallel_randomized", "stepped_wedge_cluster_rollout"}:
        fields.add(
            "data/site_summary.parquet" if factors.data_preparation == "analysis_ready" else "data/raw/sites.parquet"
        )
    return tuple(sorted(fields))


def _has_member(members: tuple[str, ...], task_id: str, relative_path: str) -> bool:
    return f"items/{task_id}/{relative_path}" in members


def _design_adjustment_fields(
    *,
    task_id: str,
    data_preparation: TrialEvalDataPreparationV1,
    protocol: dict[str, Any],
    members: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    design_family = str(protocol.get("design_family") or "")
    findings: list[str] = []
    fields: list[str] = ["protocol_summary.json:design_family"]
    if design_family == "randomized_trial":
        return tuple(fields), ()
    if design_family == "cluster_parallel_randomized":
        fields.extend(
            ["data/site_summary.parquet" if data_preparation == "analysis_ready" else "data/raw/sites.parquet"]
        )
        if data_preparation == "analysis_ready":
            if not _has_member(members, task_id, "data/site_summary.parquet"):
                findings.append("cluster_parallel_missing_site_summary")
        elif not _has_member(members, task_id, "data/raw/sites.parquet"):
            findings.append("cluster_parallel_missing_raw_sites")
        return tuple(fields), tuple(findings)
    if design_family == "stepped_wedge_cluster_rollout":
        plan = protocol.get("stepped_wedge_plan")
        if not isinstance(plan, dict):
            return tuple(fields), ("stepped_wedge_missing_plan",)
        for key in (
            "cluster_id_field",
            "baseline_date_field",
            "planned_intervention_start_day_field",
            "period_length_dy",
            "source_rel_path",
        ):
            fields.append(f"protocol_summary.json:stepped_wedge_plan.{key}")
            if key not in plan or plan.get(key) in ("", None):
                findings.append(f"stepped_wedge_missing_{key}")
        source_rel_path = str(plan.get("source_rel_path") or "")
        if source_rel_path and not _has_member(members, task_id, source_rel_path):
            findings.append("stepped_wedge_missing_source_rel_path")
        return tuple(fields), tuple(findings)
    if design_family == "group_sequential_monitoring":
        plan = protocol.get("group_sequential_plan")
        if not isinstance(plan, dict):
            return tuple(fields), ("group_sequential_missing_plan",)
        for key in (
            "looks",
            "spending_function_id",
            "monitoring_effect_scale",
            "two_sided_alpha",
            "nominal_two_sided_alpha_by_look",
            "z_critical_by_look",
            "analysis_look_index",
            "analysis_information_fraction",
            "analysis_horizon_dy",
            "information_basis",
            "stopped_early",
        ):
            fields.append(f"protocol_summary.json:group_sequential_plan.{key}")
            if key not in plan or plan.get(key) in ("", None, []):
                findings.append(f"group_sequential_missing_{key}")
        return tuple(fields), tuple(findings)
    return tuple(fields), (f"unknown_design_family:{design_family}",)


def _surface_findings(
    *,
    task_id: str,
    factors: TrialEvalEvidenceFactorsV1,
    members: tuple[str, ...],
    reconstruction_row_count: int,
) -> tuple[str, ...]:
    findings: list[str] = []
    required_metadata = required_trialeval_participant_members(
        data_preparation=factors.data_preparation,
        analysis_specification=factors.analysis_specification,
    )
    for relative_path in sorted(required_metadata):
        if not _has_member(members, task_id, relative_path):
            findings.append(f"missing_required_participant_member:{relative_path}")
    if factors.data_preparation == "analysis_ready":
        if not _has_member(members, task_id, "data/ADSL.parquet") or not _has_member(
            members, task_id, "data/ADTTE.parquet"
        ):
            findings.append("analysis_ready_surface_missing_analysis_tables")
        if any(member.startswith(f"items/{task_id}/data/raw/") for member in members):
            findings.append("analysis_ready_surface_contains_raw_inputs")
    else:
        if not any(member.startswith(f"items/{task_id}/data/raw/") for member in members):
            findings.append("raw_domain_surface_missing_raw_inputs")
        if any(member.startswith(f"items/{task_id}/data/public_reconstruction/") for member in members):
            findings.append("raw_domain_surface_contains_completed_reference_output")
        if reconstruction_row_count <= 0:
            findings.append("evaluator_missing_reconstruction_reference_rows")
    if factors.analysis_specification == "protocol_only" and any(
        _has_member(members, task_id, path) for path in ("analysis_plan.json", "analysis_tasks.md")
    ):
        findings.append("protocol_only_surface_contains_locked_sap")
    return tuple(findings)


def _validate_sap_integrity(
    *,
    public: ZipFile,
    task_id: str,
    factors: TrialEvalEvidenceFactorsV1,
    scoring_key: ValidatedScoringKeyV1,
) -> tuple[bool, tuple[str, ...]]:
    member = f"items/{task_id}/analysis_plan.json"
    if factors.analysis_specification == "protocol_only":
        return True, ()
    try:
        sap = TrialEvalPublicSAPV1.model_validate(read_json_object_member(public, member))
        task = TrialEvalParticipantTaskV1.model_validate(read_json_object_member(public, f"items/{task_id}/task.json"))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return False, (f"locked_sap_invalid:{exc}",)
    findings: list[str] = []
    if sap.task_id != task_id or sap.item_id != task_id:
        findings.append("locked_sap_identity_mismatch")
    if sap.primary_estimand_id != task.primary_estimand_id:
        findings.append("locked_sap_estimand_mismatch")
    if sap.primary_endpoint_id != task.primary_endpoint_id:
        findings.append("locked_sap_endpoint_mismatch")
    primary = [rule for rule in sap.lane_rules if rule.role == "primary" and rule.mandatory]
    if len(primary) != 1 or primary[0].effect_scale != task.primary_effect_scale:
        findings.append("locked_sap_primary_scale_mismatch")
    if len(scoring_key.credit_eligible_routes) != 1:
        findings.append("locked_sap_credit_eligible_route_ambiguous")
    else:
        route = scoring_key.credit_eligible_routes[0]
        required = route.signature
        method = route.method
        if sap.primary_analysis.effect_scale != required.effect_scale:
            findings.append("locked_sap_primary_analysis_scale_mismatch")
        if sap.primary_analysis.estimator_family != method.estimator_family:
            findings.append("locked_sap_primary_analysis_family_mismatch")
        if sap.primary_analysis.uncertainty_method != method.uncertainty_method:
            findings.append("locked_sap_primary_analysis_uncertainty_mismatch")
        if set(sap.primary_analysis.required_method_modifiers) != set(method.design_modifiers):
            findings.append("locked_sap_primary_analysis_modifier_mismatch")
    return not findings, tuple(findings)


def _evidence_factors(entry: dict[str, object]) -> TrialEvalEvidenceFactorsV1:
    factors = entry.get("factors")
    if not isinstance(factors, dict):
        raise ValueError("TrialEval evaluator item-index entries require a factors object.")
    return TrialEvalEvidenceFactorsV1.model_validate(
        {
            "context_configuration": factors.get("context_configuration"),
            "data_preparation": factors.get("data_preparation"),
            "analysis_specification": factors.get("analysis_specification"),
        }
    )


def _route_recoverability(
    *,
    candidates: tuple[PublicAnalysisClassV1, ...],
    routes: tuple[CreditEligibleScoringRouteV1, ...],
    method_dictionary: TrialEvalParticipantMethodDictionaryV1,
    diagnostic_dictionary: TrialEvalParticipantDiagnosticDictionaryV1,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    expected: set[str] = {route.route_id for route in routes}
    matches: set[str] = set()
    methods = {method.method_id: method for method in method_dictionary.methods}
    for route in routes:
        method = methods.get(route.method.analysis_method_id)
        if method is not None and _categorical_route_is_public(
            route=route,
            method=method,
            diagnostic_dictionary=diagnostic_dictionary,
        ):
            matches.add(route.route_id)
            continue
        for candidate in candidates:
            if (
                candidate.effect_scale == route.signature.effect_scale
                and candidate.estimator_family == route.method.estimator_family
                and set(candidate.required_modifiers) == set(route.method.design_modifiers)
            ):
                matches.add(route.route_id)
    return tuple(sorted(expected)), tuple(sorted(matches))


def _categorical_route_is_public(
    *,
    route: CreditEligibleScoringRouteV1,
    method: TrialEvalParticipantMethodV1,
    diagnostic_dictionary: TrialEvalParticipantDiagnosticDictionaryV1,
) -> bool:
    """Return whether a categorical route is completely declared to participants."""

    if route.target.kind != "categorical":
        return False
    method_matches = (
        method.estimator_family == route.method.estimator_family
        and method.result_kind == route.method.result_kind
        and method.effect_scale == route.signature.effect_scale
        and method.design_modifiers == route.method.design_modifiers
        and method.uncertainty_method_id == route.method.uncertainty_method
        and method.sensitivity_parameters == route.method.sensitivity_parameters
    )
    declared_diagnostics = set(method.possible_diagnostic_ids)
    required_diagnostics = set(route.required_diagnostics)
    diagnostics_are_public = (
        bool(required_diagnostics)
        and required_diagnostics <= declared_diagnostics
        and declared_diagnostics <= set(diagnostic_dictionary.diagnostics)
    )
    conclusions_are_public = set(route.target.credit_eligible_codes) <= set(method.supported_conclusion_codes)
    return method_matches and diagnostics_are_public and conclusions_are_public


def validate_trialeval_context_sufficiency_v1(
    *,
    public_zip: Path,
    evaluator_zip: Path,
) -> TrialEvalContextSufficiencyReportV1:
    """Validate participant-context sufficiency for a TrialEvalBench release."""

    entries = load_evaluator_item_index(evaluator_zip)
    indexed_task_ids = tuple(str(entry.get("task_id") or "") for entry in entries)
    scoring_keys = ScoringKeyStoreV1.from_evaluator_zip(
        evaluator_zip,
        expected_item_ids=indexed_task_ids,
    )
    rows: list[TrialEvalContextSufficiencyRowV1] = []
    with ZipFile(public_zip) as public_zf:
        public_members = set(public_zf.namelist())
        method_dictionary = TrialEvalParticipantMethodDictionaryV1.model_validate(
            read_json_object_member(public_zf, "method_dictionary.json")
        )
        diagnostic_dictionary = TrialEvalParticipantDiagnosticDictionaryV1.model_validate(
            read_json_object_member(public_zf, "diagnostic_dictionary.json")
        )
        if diagnostic_dictionary != participant_diagnostic_dictionary_v1():
            raise ValueError("TrialEval diagnostic dictionary differs from the installed canonical registry.")
        for entry in entries:
            task_id = str(entry.get("task_id") or "")
            members = _item_members(public_members, task_id)
            protocol_member = f"items/{task_id}/protocol_summary.json"
            protocol = read_json_object_member(public_zf, protocol_member) if protocol_member in public_members else {}
            factors = _evidence_factors(entry)
            context_tier = factors.context_configuration
            factor_payload = entry.get("factors")
            if not isinstance(factor_payload, dict):  # pragma: no cover - enforced by _evidence_factors
                raise ValueError("TrialEval evaluator item-index entries require a factors object.")
            design_family = str(protocol.get("design_family") or "")
            reconstruction_row_count = int(str(entry.get("reconstruction_row_count") or 0))
            surface_findings = _surface_findings(
                task_id=task_id,
                factors=factors,
                members=members,
                reconstruction_row_count=reconstruction_row_count,
            )
            design_adjustment_fields, design_findings = _design_adjustment_fields(
                task_id=task_id,
                data_preparation=factors.data_preparation,
                protocol=protocol,
                members=members,
            )
            witness = recover_analysis_class_v1(public=public_zf, task_id=task_id)
            scoring_key = scoring_keys.for_item(task_id)
            if scoring_key.context_tier != context_tier:
                raise ValueError(
                    "Scoring-key context tier disagrees with the item index: "
                    f"task_id={task_id!r} key={scoring_key.context_tier!r} index={context_tier!r}."
                )
            sap_integrity_passed, sap_findings = _validate_sap_integrity(
                public=public_zf,
                task_id=task_id,
                factors=factors,
                scoring_key=scoring_key,
            )
            credit_eligible_route_ids, recoverable_route_ids = _route_recoverability(
                candidates=witness.candidate_classes,
                routes=scoring_key.credit_eligible_routes,
                method_dictionary=method_dictionary,
                diagnostic_dictionary=diagnostic_dictionary,
            )
            unrecoverable_route_ids = tuple(sorted(set(credit_eligible_route_ids) - set(recoverable_route_ids)))
            witness_findings: tuple[str, ...]
            if unrecoverable_route_ids:
                witness_findings = ("credit_eligible_routes_not_recoverable_from_public_surface",)
            else:
                witness_findings = ()
            findings = tuple((*surface_findings, *design_findings, *sap_findings, *witness_findings))
            status: ContextSufficiencyStatusV1 = "fail" if findings else "pass"
            rows.append(
                TrialEvalContextSufficiencyRowV1(
                    task_id=task_id,
                    item_id=str(entry.get("item_id") or ""),
                    variant_id=str(entry.get("variant_id") or ""),
                    design_tier=str(factor_payload.get("design_archetype") or ""),
                    assumption_tier=str(factor_payload.get("assumption_regime") or ""),
                    context_tier=context_tier,
                    design_family=design_family,
                    data_preparation=factors.data_preparation,
                    analysis_specification=factors.analysis_specification,
                    participant_files=members,
                    required_public_fields=_required_public_fields(
                        factors=factors,
                        design_family=design_family,
                    ),
                    design_adjustment_fields=design_adjustment_fields,
                    has_task_contract=_has_member(members, task_id, "task.json"),
                    has_protocol_summary=bool(protocol),
                    has_endpoint_definition=_has_member(members, task_id, "endpoint_definition.json"),
                    has_intercurrent_event_strategy=_has_member(members, task_id, "intercurrent_event_strategy.json"),
                    has_analysis_ready_surface=_has_member(members, task_id, "data/ADSL.parquet")
                    and _has_member(members, task_id, "data/ADTTE.parquet"),
                    has_reconstruction_protocol=_has_member(members, task_id, "reconstruction_task.json"),
                    has_public_reconstruction_output=_has_member(
                        members, task_id, "data/public_reconstruction/public_reconstruction_result.json"
                    ),
                    has_raw_reconstruction_inputs=any(
                        member.startswith(f"items/{task_id}/data/raw/") for member in members
                    ),
                    has_design_adjustment_evidence=not design_findings,
                    sap_integrity_passed=sap_integrity_passed,
                    analysis_class_witness=witness,
                    credit_eligible_route_ids=credit_eligible_route_ids,
                    recoverable_route_ids=recoverable_route_ids,
                    unrecoverable_route_ids=unrecoverable_route_ids,
                    status=status,
                    disposition="retain_official" if status == "pass" else "block_release",
                    findings=findings,
                )
            )
    failed_items = sum(1 for row in rows if row.status == "fail")
    return TrialEvalContextSufficiencyReportV1(
        public_zip=public_zip.as_posix(),
        evaluator_zip=evaluator_zip.as_posix(),
        status="fail" if failed_items else "pass",
        total_items=len(rows),
        failed_items=failed_items,
        analysis_ready_items=sum(1 for row in rows if row.data_preparation == "analysis_ready"),
        reconstruction_items=sum(1 for row in rows if row.data_preparation != "analysis_ready"),
        d4_items=sum(1 for row in rows if row.design_tier == "D4"),
        d4_design_family_counts=dict(
            sorted(Counter(row.design_family for row in rows if row.design_tier == "D4").items())
        ),
        rows=tuple(rows),
        summaries=_summary_rows(rows),
    )


def _summary_rows(rows: list[TrialEvalContextSufficiencyRowV1]) -> tuple[TrialEvalContextSufficiencySummaryRowV1, ...]:
    summaries: list[TrialEvalContextSufficiencySummaryRowV1] = []
    for group_kind in ("context_tier", "design_family", "design_tier"):
        grouped: dict[str, list[TrialEvalContextSufficiencyRowV1]] = {}
        for row in rows:
            grouped.setdefault(str(getattr(row, group_kind)), []).append(row)
        for group_value, group_rows in sorted(grouped.items()):
            status_counts = Counter(row.status for row in group_rows)
            disposition_counts = Counter(row.disposition for row in group_rows)
            summaries.append(
                TrialEvalContextSufficiencySummaryRowV1(
                    group_kind=group_kind,
                    group_value=group_value,
                    total=len(group_rows),
                    passed=status_counts["pass"],
                    failed=status_counts["fail"],
                    retain_official=disposition_counts["retain_official"],
                    repair_required=disposition_counts["repair_required"],
                    block_release=disposition_counts["block_release"],
                )
            )
    return tuple(summaries)


def render_trialeval_context_sufficiency_report_v1(report: TrialEvalContextSufficiencyReportV1) -> str:
    """Render a Markdown TrialEvalBench context-sufficiency report."""

    lines = [
        "# TrialEvalBench Context-Sufficiency Report",
        "",
        f"- Public zip: `{report.public_zip}`",
        f"- Evaluator zip: `{report.evaluator_zip}`",
        f"- Status: `{report.status}`",
        f"- Total items: `{report.total_items}`",
        f"- Failed items: `{report.failed_items}`",
        f"- Analysis-ready items: `{report.analysis_ready_items}`",
        f"- Reconstruction items: `{report.reconstruction_items}`",
        f"- D4 items: `{report.d4_items}`",
        "",
        "## Summary",
        "",
        "| Group | Value | Total | Passed | Failed | Retain official | Repair required | Block release |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in report.summaries:
        lines.append(
            f"| `{summary.group_kind}` | `{summary.group_value}` | {summary.total} | {summary.passed} | "
            f"{summary.failed} | {summary.retain_official} | {summary.repair_required} | "
            f"{summary.block_release} |"
        )
    failed = [row for row in report.rows if row.status == "fail"]
    if failed:
        lines.extend(["", "## Failed Items", ""])
        for failed_row in failed:
            lines.append(
                f"- `{failed_row.task_id}` `{failed_row.context_tier}` `{failed_row.design_family}`: "
                f"{', '.join(failed_row.findings)}"
            )
    return "\n".join(lines) + "\n"


def write_trialeval_context_sufficiency_artifacts_v1(
    *,
    public_zip: Path,
    evaluator_zip: Path,
    out_dir: Path,
) -> TrialEvalContextSufficiencyReportV1:
    """Validate and write TrialEvalBench context-sufficiency artifacts."""

    report = validate_trialeval_context_sufficiency_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_model(out_dir / "trialeval_context_sufficiency_report.json", report)
    (out_dir / "trialeval_context_sufficiency_report.md").write_text(
        render_trialeval_context_sufficiency_report_v1(report),
        encoding="utf-8",
    )
    return report


__all__ = [
    "TrialEvalContextSufficiencyReportV1",
    "TrialEvalContextSufficiencyRowV1",
    "TrialEvalContextSufficiencySummaryRowV1",
    "render_trialeval_context_sufficiency_report_v1",
    "validate_trialeval_context_sufficiency_v1",
    "write_trialeval_context_sufficiency_artifacts_v1",
]
