"""Independent semantic audit of high-risk TrialDev release strata."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.contracts.trialdev_scientific_inventory import (
    TrialDevScientificConstructionInventoryV1,
)
from trialagentbench_validation.contracts.v1_scope import (
    RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1,
)
from trialagentbench_validation.recovery import RecoverabilityReportV1


class _AuditModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialDevSentinelStratumV1(_AuditModel):
    """One prospectively selected TrialDev scientific boundary."""

    sentinel_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    phase_id: Literal[
        "observational_review", "phase1", "phase2", "phase3", "final_decision"
    ]
    lane_id: Literal[
        "asset_nomination",
        "phase_design",
        "phase_analysis",
        "decision_action",
        "route_timing",
        "final_recommendation",
        "safety_gate",
    ]
    expected_identification_class: Literal[
        "point_identified_under_declared_measured_confounding_assumptions",
        "qualified_nonidentification_under_residual_unmeasured_confounding",
        "randomized_descriptive_risk_under_arm_conditional_independent_censoring",
        "randomized_comparative_risk_under_arm_conditional_independent_censoring",
        "trajectory_conditioned_decision",
    ]
    expected_intercurrent_event_bindings: tuple[
        Literal[
            "treatment_discontinuation:composite_discontinuation",
            "treatment_discontinuation:treatment_policy",
            "treatment_discontinuation:while_on_treatment",
        ],
        ...,
    ]
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_bindings(self) -> TrialDevSentinelStratumV1:
        if self.expected_intercurrent_event_bindings != tuple(
            sorted(set(self.expected_intercurrent_event_bindings))
        ):
            raise ValueError(
                "expected intercurrent-event bindings must be sorted and unique"
            )
        return self


class TrialDevSentinelRecordV1(_AuditModel):
    """Scientific-contract and replay result for one TrialDev sentinel."""

    sentinel_id: str
    scenario_id: str
    phase_id: str
    lane_id: str
    identification_class: str
    scientific_row_count: int = Field(ge=0)
    replay_row_count: int = Field(ge=0)
    intercurrent_event_bindings: tuple[str, ...]
    loss_to_follow_up_handling_ids: tuple[str, ...]
    failed_replay_route_ids: tuple[str, ...]
    findings: tuple[str, ...]
    status: Literal["pass", "fail"]


class TrialDevSentinelAuditReportV1(_AuditModel):
    """Cross-construct TrialDev sentinel qualification report."""

    schema_id: Literal["trialagentbench.trialdev_sentinel_audit/v1"] = (
        "trialagentbench.trialdev_sentinel_audit/v1"
    )
    scientific_inventory_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_sentinel_count: int = Field(ge=1)
    records: tuple[TrialDevSentinelRecordV1, ...]
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def _status_matches_records(self) -> TrialDevSentinelAuditReportV1:
        expected = all(
            record.status == "pass" and not record.findings for record in self.records
        )
        if (self.status == "pass") != expected:
            raise ValueError(
                "TrialDev sentinel status disagrees with its record findings"
            )
        if len(self.records) != self.requested_sentinel_count:
            raise ValueError(
                "TrialDev sentinel report does not cover every requested stratum"
            )
        return self


DEFAULT_TRIALDEV_SENTINEL_STRATA_V1: tuple[TrialDevSentinelStratumV1, ...] = (
    TrialDevSentinelStratumV1(
        sentinel_id="measured_confounding_nomination",
        scenario_id="s30",
        phase_id="observational_review",
        lane_id="asset_nomination",
        expected_identification_class="point_identified_under_declared_measured_confounding_assumptions",
        expected_intercurrent_event_bindings=(),
        rationale="Measured-confounding nomination with participant-only estimator replay.",
    ),
    TrialDevSentinelStratumV1(
        sentinel_id="residual_unmeasured_confounding",
        scenario_id="s35",
        phase_id="observational_review",
        lane_id="asset_nomination",
        expected_identification_class="qualified_nonidentification_under_residual_unmeasured_confounding",
        expected_intercurrent_event_bindings=(),
        rationale="Residual-confounding world requiring qualification rather than a hidden best-drug target.",
    ),
    TrialDevSentinelStratumV1(
        sentinel_id="phase2_discontinuation_and_ltfu",
        scenario_id="s30",
        phase_id="phase2",
        lane_id="phase_analysis",
        expected_identification_class="randomized_comparative_risk_under_arm_conditional_independent_censoring",
        expected_intercurrent_event_bindings=(
            "treatment_discontinuation:composite_discontinuation",
            "treatment_discontinuation:treatment_policy",
            "treatment_discontinuation:while_on_treatment",
        ),
        rationale="Phase-2 efficacy route with explicit discontinuation strategies and separate LTFU handling.",
    ),
    TrialDevSentinelStratumV1(
        sentinel_id="restricted_phase3_design_frontier",
        scenario_id="s40",
        phase_id="phase3",
        lane_id="phase_design",
        expected_identification_class="randomized_comparative_risk_under_arm_conditional_independent_censoring",
        expected_intercurrent_event_bindings=(
            "treatment_discontinuation:treatment_policy",
        ),
        rationale="Restricted design environment requiring a public adequate nondominated frontier.",
    ),
)


def audit_trialdev_sentinels(
    *,
    inventory: TrialDevScientificConstructionInventoryV1,
    recoverability: RecoverabilityReportV1,
    strata: tuple[TrialDevSentinelStratumV1, ...] = DEFAULT_TRIALDEV_SENTINEL_STRATA_V1,
) -> TrialDevSentinelAuditReportV1:
    """Audit selected TrialDev scientific boundaries against independent replay."""

    if recoverability.suite != "trialdev":
        raise ValueError(
            "TrialDev sentinel audit requires a TrialDev recoverability report"
        )
    if inventory.checksum is None:
        raise ValueError("TrialDev scientific inventory lacks its checksum")
    records = tuple(
        _audit_stratum(
            stratum=stratum,
            inventory=inventory,
            recoverability=recoverability,
        )
        for stratum in strata
    )
    return TrialDevSentinelAuditReportV1(
        scientific_inventory_checksum=inventory.checksum,
        requested_sentinel_count=len(strata),
        records=records,
        status=(
            "pass" if all(record.status == "pass" for record in records) else "fail"
        ),
    )


def audit_trialdev_release_sentinels(
    *,
    participant_release: Path,
    evaluator_release: Path,
    verification_release: Path,
    absolute_tolerance: float = RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1,
) -> TrialDevSentinelAuditReportV1:
    """Run full public replay before auditing the TrialDev sentinel strata."""

    from trialagentbench_validation.recovery import recover_release

    recoverability = recover_release(
        participant_release=participant_release,
        evaluator_release=evaluator_release,
        verification_release=verification_release,
        absolute_tolerance=absolute_tolerance,
    )
    with ZipFile(verification_release) as archive:
        inventory = TrialDevScientificConstructionInventoryV1.model_validate_json(
            archive.read("scientific_construction_inventory.json")
        )
    return audit_trialdev_sentinels(
        inventory=inventory,
        recoverability=recoverability,
    )


def _audit_stratum(
    *,
    stratum: TrialDevSentinelStratumV1,
    inventory: TrialDevScientificConstructionInventoryV1,
    recoverability: RecoverabilityReportV1,
) -> TrialDevSentinelRecordV1:
    scientific_rows = tuple(
        row
        for row in inventory.rows
        if row.scenario_id == stratum.scenario_id
        and row.phase_id == stratum.phase_id
        and row.lane_id == stratum.lane_id
    )
    replay_rows = tuple(
        row
        for row in recoverability.routes
        if _replay_matches_stratum(row.unit_id, row.estimator_family, stratum)
    )
    findings: list[str] = []
    if not scientific_rows:
        findings.append("missing_scientific_construction_row")
    if scientific_rows and any(
        row.identification_class != stratum.expected_identification_class
        for row in scientific_rows
    ):
        findings.append("identification_class_mismatch")
    bindings = tuple(
        sorted(
            {
                binding
                for row in scientific_rows
                for binding in row.permitted_intercurrent_event_bindings
            }
        )
    )
    ltfu = tuple(sorted({row.loss_to_follow_up_handling_id for row in scientific_rows}))
    if bindings != stratum.expected_intercurrent_event_bindings:
        findings.append("intercurrent_event_bindings_mismatch")
    if stratum.phase_id in {"phase2", "phase3"}:
        if any("loss_to_follow_up:" in binding for binding in bindings):
            findings.append("loss_to_follow_up_misclassified_as_intercurrent_event")
        if not ltfu or any("censor" not in value for value in ltfu):
            findings.append("loss_to_follow_up_handling_not_explicit")
    if not replay_rows:
        findings.append("missing_independent_public_replay")
    failed_routes = tuple(
        sorted(row.route_id for row in replay_rows if row.status != "pass")
    )
    if failed_routes:
        findings.append("independent_public_replay_failed")
    return TrialDevSentinelRecordV1(
        sentinel_id=stratum.sentinel_id,
        scenario_id=stratum.scenario_id,
        phase_id=stratum.phase_id,
        lane_id=stratum.lane_id,
        identification_class=(
            scientific_rows[0].identification_class
            if scientific_rows
            else stratum.expected_identification_class
        ),
        scientific_row_count=len(scientific_rows),
        replay_row_count=len(replay_rows),
        intercurrent_event_bindings=bindings,
        loss_to_follow_up_handling_ids=ltfu,
        failed_replay_route_ids=failed_routes,
        findings=tuple(findings),
        status="fail" if findings else "pass",
    )


def _replay_matches_stratum(
    unit_id: str,
    estimator_family: str,
    stratum: TrialDevSentinelStratumV1,
) -> bool:
    if stratum.phase_id == "observational_review":
        return (
            unit_id == stratum.scenario_id
            and estimator_family != "public_randomized_phase_replay"
        )
    return (
        unit_id.startswith(f"{stratum.scenario_id}:")
        and estimator_family == "public_randomized_phase_replay"
    )


__all__ = [
    "DEFAULT_TRIALDEV_SENTINEL_STRATA_V1",
    "TrialDevSentinelAuditReportV1",
    "TrialDevSentinelRecordV1",
    "TrialDevSentinelStratumV1",
    "audit_trialdev_release_sentinels",
    "audit_trialdev_sentinels",
]
