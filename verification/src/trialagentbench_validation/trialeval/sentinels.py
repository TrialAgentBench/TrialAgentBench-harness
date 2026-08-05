"""Independent semantic audit of high-risk TrialEval release cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, TypeVar
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_validation.contracts.scoring_keys import (
    CreditEligibleScoringRouteV1,
    read_scoring_keys,
)
from trialagentbench_validation.contracts.trialeval_release import (
    ItemIndexEntryV1,
    ItemIndexV1,
)
from trialagentbench_validation.trialeval.references.calculators import (
    PublicIPCWSupportDiagnosticsV1,
    recompute_public_ipcw_support_v1,
)

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class _AuditModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SentinelStratumV1(_AuditModel):
    """One prospectively selected high-risk benchmark stratum."""

    sentinel_id: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    rationale: str = Field(min_length=1)
    required_observation_process_limitation: (
        Literal["unmeasured_dependent_censoring"] | None
    ) = None
    match_ordinal: int = Field(default=0, ge=0)


class SentinelAuditRecordV1(_AuditModel):
    """Semantic and public-surface findings for one selected task."""

    sentinel_id: str
    task_id: str
    variant_id: str
    context_tier: str
    design_tier: str
    assumption_tier: str
    credit_eligible_route_count: int
    estimand_ids: tuple[str, ...]
    identification_assumptions: tuple[str, ...]
    intercurrent_event_bindings: tuple[str, ...]
    effect_scales: tuple[str, ...]
    estimator_families: tuple[str, ...]
    result_kinds: tuple[str, ...]
    design_obligations: tuple[str, ...]
    required_diagnostics: tuple[str, ...]
    participant_file_count: int
    missing_task_file_references: tuple[str, ...]
    ipcw_support: PublicIPCWSupportDiagnosticsV1 | None
    findings: tuple[str, ...]


class SentinelAuditReportV1(_AuditModel):
    """Release-level sentinel audit report."""

    schema_id: Literal["trialagentbench.sentinel_audit/v1"] = (
        "trialagentbench.sentinel_audit/v1"
    )
    evaluator_sha256: str = Field(min_length=64, max_length=64)
    participant_sha256: str = Field(min_length=64, max_length=64)
    requested_sentinel_count: int = Field(ge=1)
    selected_sentinel_count: int = Field(ge=0)
    records: tuple[SentinelAuditRecordV1, ...]
    findings: tuple[str, ...]
    status: Literal["pass", "fail"]


DEFAULT_SENTINEL_STRATA_V1: tuple[SentinelStratumV1, ...] = (
    SentinelStratumV1(
        sentinel_id="nonproportional_hazards_reconstruction",
        variant_id="TE-S01-A3__C3",
        context_tier="C3",
        rationale="Consequential nonproportional-hazards mechanism with raw reconstruction and locked SAP.",
    ),
    SentinelStratumV1(
        sentinel_id="pragmatic_intercurrent_events",
        variant_id="TE-S03-A2__C4",
        context_tier="C4",
        rationale="Protocol-only pragmatic trial with intercurrent-event handling.",
    ),
    SentinelStratumV1(
        sentinel_id="prognostic_censoring_sensitivity",
        variant_id="TE-S02-A3__C1",
        context_tier="C1",
        rationale="Prognostic censoring with a participant-visible sensitivity-analysis route.",
    ),
    SentinelStratumV1(
        sentinel_id="unmeasured_dependent_censoring",
        variant_id="TE-S04-A4__C1",
        context_tier="C1",
        required_observation_process_limitation="unmeasured_dependent_censoring",
        rationale="Nonpoint A4 response under a disclosed unavailable prognostic censoring driver.",
    ),
    SentinelStratumV1(
        sentinel_id="high_dimensional_prognostic_adjustment",
        variant_id="TE-S05-A3__C2",
        context_tier="C2",
        rationale="Analysis-ready high-dimensional prognostic adjustment with protocol-only specification.",
    ),
    SentinelStratumV1(
        sentinel_id="high_dimensional_reconstruction_and_defect",
        variant_id="TE-S05-A3__C5",
        context_tier="C5",
        rationale="High-dimensional prognostic structure plus a controlled recoverable defect.",
    ),
    SentinelStratumV1(
        sentinel_id="cluster_randomized",
        variant_id="TE-S07-A3__C1",
        context_tier="C1",
        rationale="Cluster-randomized analysis with locked SAP.",
    ),
    SentinelStratumV1(
        sentinel_id="stepped_wedge",
        variant_id="TE-S08-A2__C4",
        context_tier="C4",
        rationale="Stepped-wedge calendar-time and cluster correction.",
    ),
    SentinelStratumV1(
        sentinel_id="group_sequential_monitoring",
        variant_id="TE-S09-A4__C1",
        context_tier="C1",
        rationale="Group-sequential design requiring prespecified repeated-look inference.",
    ),
)


def audit_trialeval_sentinels(
    *,
    evaluator_zip: Path,
    participant_zip: Path,
    strata: tuple[SentinelStratumV1, ...] = DEFAULT_SENTINEL_STRATA_V1,
) -> SentinelAuditReportV1:
    """Audit deterministic sentinel selections against released archive bytes."""

    evaluator_path = Path(evaluator_zip)
    participant_path = Path(participant_zip)
    with ZipFile(evaluator_path) as evaluator, ZipFile(participant_path) as participant:
        index = _read_json_model(evaluator, "grader/item_index.json", ItemIndexV1)
        scoring_keys = read_scoring_keys(
            evaluator,
            expected_item_ids=tuple(entry.task_id for entry in index.entries),
        )
        routes_by_task = {
            key.item_id: key.credit_eligible_routes for key in scoring_keys
        }
        participant_names = {
            name for name in participant.namelist() if not name.endswith("/")
        }
        records = tuple(
            _audit_stratum(
                stratum=stratum,
                index=index,
                routes_by_task=routes_by_task,
                participant=participant,
                participant_names=participant_names,
            )
            for stratum in strata
        )
    release_findings: list[str] = []
    if len(records) != len(strata):
        release_findings.append("sentinel_selection_incomplete")
    if any(record.findings for record in records):
        release_findings.append("sentinel_contract_failure")
    return SentinelAuditReportV1(
        evaluator_sha256=_sha256(evaluator_path),
        participant_sha256=_sha256(participant_path),
        requested_sentinel_count=len(strata),
        selected_sentinel_count=len(records),
        records=records,
        findings=tuple(release_findings),
        status="fail" if release_findings else "pass",
    )


def _audit_stratum(
    *,
    stratum: SentinelStratumV1,
    index: ItemIndexV1,
    routes_by_task: dict[str, tuple[CreditEligibleScoringRouteV1, ...]],
    participant: ZipFile,
    participant_names: set[str],
) -> SentinelAuditRecordV1:
    candidates = sorted(
        (
            entry
            for entry in index.entries
            if entry.variant_id == stratum.variant_id
            and entry.context_tier == stratum.context_tier
        ),
        key=lambda entry: entry.task_id,
    )
    if stratum.required_observation_process_limitation is not None:
        candidates = [
            entry
            for entry in candidates
            if _observation_process_limitation(
                participant=participant,
                task_id=entry.task_id,
            )
            == stratum.required_observation_process_limitation
        ]
    if len(candidates) <= stratum.match_ordinal:
        raise ValueError(
            f"sentinel stratum has no matching release task: {stratum.sentinel_id}"
        )
    task = candidates[stratum.match_ordinal]
    routes = routes_by_task.get(task.task_id, ())
    findings: list[str] = []
    if not routes:
        findings.append("no_accepted_scoring_route")
    scales = tuple(sorted({route.signature.effect_scale for route in routes}))
    families = tuple(sorted({route.method.estimator_family for route in routes}))
    result_kinds = tuple(sorted({route.method.result_kind for route in routes}))
    estimands = tuple(sorted({route.signature.estimand_id for route in routes}))
    identification_assumptions = tuple(
        sorted(
            {
                assumption
                for route in routes
                for assumption in route.required_identification_assumptions
            }
        )
    )
    event_bindings = tuple(
        sorted(
            {
                binding
                for route in routes
                for binding in route.signature.intercurrent_event_strategy_ids
            }
        )
    )
    design_obligations = tuple(
        sorted({value for route in routes for value in route.method.design_modifiers})
    )
    diagnostics = tuple(
        sorted({value for route in routes for value in route.required_diagnostics})
    )
    if len(estimands) != 1:
        findings.append("credit_eligible_routes_must_share_one_precise_estimand")
    if any(":" not in binding for binding in event_bindings):
        findings.append("intercurrent_event_binding_lacks_event_strategy_pair")
    locked_sap = task.factors.analysis_specification == "locked_sap"
    if locked_sap and len(routes) != 1:
        findings.append("locked_sap_requires_exactly_one_route")
    if "statistical_test" in result_kinds and any(
        result_kind != "statistical_test" for result_kind in result_kinds
    ):
        findings.append("fixed_declared_scale_mixes_estimation_and_testing")
    _check_context_surface(
        task=task, participant_names=participant_names, findings=findings
    )
    missing_task_references = _check_task_file_references(
        task=task,
        participant=participant,
        participant_names=participant_names,
        findings=findings,
    )
    _check_analysis_specification(
        task=task,
        participant=participant,
        participant_names=participant_names,
        findings=findings,
    )
    ipcw_support: PublicIPCWSupportDiagnosticsV1 | None = None
    if any("ipcw" in family for family in families):
        try:
            ipcw_support = recompute_public_ipcw_support_v1(
                public=participant,
                task_id=task.task_id,
            )
        except (FileNotFoundError, ValueError) as exc:
            findings.append(f"ipcw_support_replay_failed:{type(exc).__name__}")
        else:
            if ipcw_support.support_status != "pass":
                findings.append(
                    f"ipcw_support_status_mismatch:pass:{ipcw_support.support_status}"
                )
    return SentinelAuditRecordV1(
        sentinel_id=stratum.sentinel_id,
        task_id=task.task_id,
        variant_id=task.variant_id,
        context_tier=task.context_tier,
        design_tier=task.factors.design_tier,
        assumption_tier=task.factors.assumption_tier,
        credit_eligible_route_count=len(routes),
        estimand_ids=estimands,
        identification_assumptions=identification_assumptions,
        intercurrent_event_bindings=event_bindings,
        effect_scales=scales,
        estimator_families=families,
        result_kinds=result_kinds,
        design_obligations=design_obligations,
        required_diagnostics=diagnostics,
        participant_file_count=sum(
            name.startswith(f"items/{task.task_id}/") for name in participant_names
        ),
        missing_task_file_references=missing_task_references,
        ipcw_support=ipcw_support,
        findings=tuple(sorted(set(findings))),
    )


def _observation_process_limitation(
    *,
    participant: ZipFile,
    task_id: str,
) -> str | None:
    member = f"items/{task_id}/protocol_summary.json"
    try:
        payload = json.loads(participant.read(member))
    except KeyError as exc:
        raise FileNotFoundError(
            f"sentinel selection lacks participant protocol: {member}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"sentinel participant protocol must contain an object: {member}"
        )
    observation_process = payload.get("observation_process")
    if observation_process is None:
        return None
    if not isinstance(observation_process, dict):
        raise ValueError(
            f"sentinel observation_process must contain an object: {member}"
        )
    limitation = observation_process.get("disclosed_limitation")
    if isinstance(limitation, str):
        return limitation
    if (
        observation_process.get("factor_associated_with_primary_endpoint") is True
        and observation_process.get("factor_recorded_in_released_data") is False
    ):
        return "unmeasured_dependent_censoring"
    return None


def _check_context_surface(
    *,
    task: ItemIndexEntryV1,
    participant_names: set[str],
    findings: list[str],
) -> None:
    prefix = f"items/{task.task_id}/data/"
    item_data = {name for name in participant_names if name.startswith(prefix)}
    analysis_ready = any(
        name.endswith("/ADSL.parquet") or name.endswith("/ADTTE.parquet")
        for name in item_data
    )
    raw = any("/raw/" in name for name in item_data)
    if task.context_tier in {"C1", "C2"} and not analysis_ready:
        findings.append("analysis_ready_context_missing_prepared_tables")
    if task.context_tier in {"C3", "C4", "C5"} and not raw:
        findings.append("raw_context_missing_raw_evidence")
    if task.context_tier == "C5" and task.data_integrity_reference_row_count < 1:
        findings.append("c5_missing_declared_defect")


def _check_analysis_specification(
    *,
    task: ItemIndexEntryV1,
    participant: ZipFile,
    participant_names: set[str],
    findings: list[str],
) -> None:
    member = f"items/{task.task_id}/analysis_plan.json"
    locked = task.context_tier in {"C1", "C3"}
    if member not in participant_names:
        if locked:
            findings.append("participant_analysis_plan_missing")
            findings.append("locked_sap_missing_complete_primary_analysis")
        return
    payload = json.loads(participant.read(member))
    if not isinstance(payload, dict):
        findings.append("participant_analysis_plan_not_object")
        return
    if not locked:
        findings.append("protocol_only_context_discloses_item_specific_route")
        return

    primary_analysis = payload.get("primary_analysis")
    lane_rules = payload.get("lane_rules")
    diagnostics = payload.get("diagnostic_requirements")
    required_plan_fields = {
        "primary_estimand_id",
        "primary_endpoint_id",
        "primary_analysis",
        "diagnostic_requirements",
        "lane_rules",
    }
    required_analysis_fields = {
        "effect_scale",
        "method_id",
        "estimator_family",
        "implementation",
        "uncertainty_method",
        "required_method_modifiers",
    }
    if (
        required_plan_fields - set(payload)
        or not isinstance(primary_analysis, dict)
        or required_analysis_fields - set(primary_analysis)
        or not isinstance(lane_rules, list)
        or not isinstance(diagnostics, list)
        or not diagnostics
    ):
        findings.append("locked_sap_missing_complete_primary_analysis")
        return

    primary_rules = [
        rule
        for rule in lane_rules
        if isinstance(rule, dict)
        and rule.get("role") == "primary"
        and rule.get("mandatory") is True
    ]
    if len(primary_rules) != 1:
        findings.append("locked_sap_primary_lane_not_unique")
        return
    primary_rule = primary_rules[0]
    expected_bindings = {
        "estimand_id": payload["primary_estimand_id"],
        "endpoint_id": payload["primary_endpoint_id"],
        "effect_scale": primary_analysis["effect_scale"],
    }
    if any(
        primary_rule.get(field) != value for field, value in expected_bindings.items()
    ):
        findings.append("locked_sap_primary_lane_binding_mismatch")

    task_member = f"items/{task.task_id}/task.json"
    task_payload = json.loads(participant.read(task_member))
    if not isinstance(task_payload, dict):
        return
    task_bindings = {
        "primary_estimand_id": payload["primary_estimand_id"],
        "primary_endpoint_id": payload["primary_endpoint_id"],
        "primary_effect_scale": primary_analysis["effect_scale"],
    }
    if any(task_payload.get(field) != value for field, value in task_bindings.items()):
        findings.append("locked_sap_task_binding_mismatch")


def _check_task_file_references(
    *,
    task: ItemIndexEntryV1,
    participant: ZipFile,
    participant_names: set[str],
    findings: list[str],
) -> tuple[str, ...]:
    member = f"items/{task.task_id}/task.json"
    if member not in participant_names:
        findings.append("task_json_missing")
        return (member,)
    payload = json.loads(participant.read(member))
    if not isinstance(payload, dict):
        findings.append("task_json_not_object")
        return ()
    prefix = f"items/{task.task_id}/"
    references = {
        prefix + value
        for key, value in payload.items()
        if key.endswith("_file") and isinstance(value, str) and value
    }
    missing = tuple(
        sorted(
            reference for reference in references if reference not in participant_names
        )
    )
    if missing:
        findings.append("task_references_missing_public_file")
    return missing


def _read_json_model(archive: ZipFile, member: str, model: type[_ModelT]) -> _ModelT:
    return model.model_validate_json(archive.read(member))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DEFAULT_SENTINEL_STRATA_V1",
    "SentinelAuditRecordV1",
    "SentinelAuditReportV1",
    "SentinelStratumV1",
    "audit_trialeval_sentinels",
]
