"""Build the finite-census analysis package for one benchmark candidate."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from statistics import median
from typing import Literal, cast
from xml.sax.saxutils import escape
from zipfile import ZipFile

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.candidate_clean_replay import (
    CandidateCleanWheelReplayV1,
)
from trialagentbench_validation.contracts.candidate_release import (
    CandidateIdentityV1,
    CandidatePublicWheelV1,
    CandidateRoleArchiveV1,
    CandidateValidationBundleV1,
    verify_candidate_validation_bundle,
)
from trialagentbench_validation.contracts.scientific_inventory import (
    TrialEvalScientificConstructionInventoryV1,
)
from trialagentbench_validation.contracts.simulation_validation_bundle import (
    ValidationArtifactV1,
    ValidationFigureV1,
)
from trialagentbench_validation.contracts.trialdev_scientific_inventory import (
    TrialDevScientificConstructionInventoryV1,
)
from trialagentbench_validation.contracts.v1_scope import (
    RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1,
    TRIALDEV_SCENARIO_COUNT_V1,
    TRIALEVAL_CONTEXT_COUNT_V1,
    TRIALEVAL_ITEM_COUNT_V1,
)
from trialagentbench_validation.external.release.artifacts import (
    ArtifactDigestV1,
    ExternalArtifactManifestV1,
)
from trialagentbench_validation.grader_behavior import (
    GraderBehaviorCaseResultV1,
    GraderBehaviorReportV1,
)
from trialagentbench_validation.grader_concordance import (
    GraderConcordanceReportV1,
    TrialDevLaneGradeV1,
)
from trialagentbench_validation.io import sha256_file, write_model
from trialagentbench_validation.recovery import (
    RecoverabilityReportV1,
    RecoverabilityRouteV1,
)
from trialagentbench_validation.trialdev.phase_replay import (
    TrialDevPublicPhaseReplayRecordV1,
)
from trialagentbench_validation.trialdev.reachability import (
    TrialDevReachabilityReportV1,
    audit_trialdev_reachability,
)
from trialagentbench_validation.trialdev.sentinel_audit import (
    audit_trialdev_sentinels,
)
from trialagentbench_validation.trialeval.integrity import (
    C5IntegrityRecoveryReportV1,
)
from trialagentbench_validation.trialeval.reconstruction import (
    load_public_analysis_tables_v1,
)
from trialagentbench_validation.trialeval.sentinels import (
    SentinelAuditReportV1,
    audit_trialeval_sentinels,
)

_TRIALEVAL_ANALYSIS_COLUMNS = ("USUBJID", "PARAMCD", "AVAL", "CNSR")
_TRIALDEV_REQUIRED_COLUMNS = ("USUBJID", "TREATMENT")
_TRIALDEV_EFFICACY_EVENT_SUFFIX = "_E"
_TRIALDEV_EFFICACY_EVENT_PREFIX = "EFF_"
_TRIALDEV_SAFETY_EVENT_PREFIX = "AE_"
_TRIALDEV_SAFETY_EVENT_SUFFIX = "_EVENT_E"
_TRIALEVAL_OPERATIONAL_COUNT_COLUMNS = {
    "intercurrent_event_count": "N_ICE_RECORDS",
    "adverse_event_intercurrent_event_count": "N_ADVERSE_EVENT_ICE",
    "discontinuation_count": "N_DISCONTINUATION_ICE",
    "nonadherence_intercurrent_event_count": "N_NONADHERENCE_ICE",
    "rescue_count": "N_RESCUE_THERAPY_ICE",
    "treatment_switch_count": "N_TREATMENT_SWITCH_ICE",
}


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _ReleaseArtifactV1(_Contract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _PublicPackageArtifactV1(_Contract):
    name: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _PublicDatasetManifestV1(_Contract):
    schema_id: Literal["trialagentbench.release_package/v1"]
    trialeval_release: Literal["TrialEvalBench"]
    trialdev_release: Literal["TrialDevBench"]
    trace_explorer_package: str | None
    harness_source: Literal["TrialAgentBench_harness.zip"]
    artifacts: tuple[_PublicPackageArtifactV1, ...] = Field(min_length=1)
    standalone_packages: tuple[
        Literal[
            "trialagentbench-harness",
            "trialagentbench-validation",
        ],
        ...,
    ]


class _RelativeArtifactV1(_Contract):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _ReleaseCatalogueV1(_Contract):
    path: Literal["metadata/simulation_properties.jsonl"]
    schema_path: Literal["metadata/simulation_properties.schema.json"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(ge=1)
    trialeval_record_count: int = Field(ge=1)
    trialdev_record_count: int = Field(ge=1)


class _ExperimentAssignmentsV1(_Contract):
    schema_path: str
    trialeval_path: str
    trialeval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trialeval_record_count: int = Field(ge=1)
    trialdev_path: str
    trialdev_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trialdev_record_count: int = Field(ge=1)


class _AnalysisContractV1(_Contract):
    analysis_id: str
    cluster_keys: tuple[str, str]
    weighting: str
    require_complete_crossed_cells: bool
    bootstrap_replicates: int = Field(ge=1)
    confidence_level: float = Field(gt=0, lt=1)
    seed: int = Field(ge=0)


class _ReferenceReplayFeasibilityReceiptV1(_Contract):
    schema_id: Literal["trialagentbench.reference_replay_feasibility/v1"]
    suite_id: Literal["trialeval", "trialdev"]
    execution_budget_profile_id: str
    evidence_report_path: str
    evidence_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scheduled_assignment_count: int = Field(ge=1)
    required_route_count: int = Field(ge=1)
    replayed_route_count: int = Field(ge=1)
    failed_route_count: Literal[0]
    largest_comparison_denominator: int = Field(ge=1)
    largest_route_id: str
    private_generating_state_used: Literal[False]
    network_access_used: Literal[False]
    status: Literal["pass"]

    @model_validator(mode="after")
    def _complete(self) -> _ReferenceReplayFeasibilityReceiptV1:
        if self.replayed_route_count != self.required_route_count:
            raise ValueError(
                "reference-replay feasibility has incomplete route recovery"
            )
        expected = {
            "trialeval": "trialeval_release_default_v1",
            "trialdev": "trialdev_release_default_v1",
        }[self.suite_id]
        if self.execution_budget_profile_id != expected:
            raise ValueError("reference-replay feasibility uses the wrong suite budget")
        return self


class _ReferenceReplayFeasibilityRegistryV1(_Contract):
    schema_id: Literal["trialagentbench.reference_replay_feasibility_registry/v1"]
    interpretation: Literal[
        "computational_public_reference_replay_not_agent_task_completion"
    ]
    receipts: tuple[
        _ReferenceReplayFeasibilityReceiptV1, _ReferenceReplayFeasibilityReceiptV1
    ]

    @model_validator(mode="after")
    def _both_suites(self) -> _ReferenceReplayFeasibilityRegistryV1:
        if tuple(row.suite_id for row in self.receipts) != ("trialeval", "trialdev"):
            raise ValueError("reference-replay feasibility must cover both suites")
        return self


class _CategoryCountV1(_Contract):
    category: str = Field(min_length=1)
    count: int = Field(ge=1)


class _CandidateReleaseStatisticsV1(_Contract):
    schema_id: Literal["trialagentbench.candidate_release_statistics/v1"]
    release_id: str = Field(min_length=1)
    analysis_unit_count: int = Field(ge=1)
    generation_unit_count: int = Field(ge=1)
    scoring_unit_count: int = Field(ge=1)
    trialeval_route_reference_count: int = Field(ge=1)
    trialdev_reference_policy_target_count: int = Field(ge=1)
    trialdev_credit_eligible_policy_target_count: int = Field(ge=0)
    suite_counts: tuple[_CategoryCountV1, ...] = Field(min_length=2)
    trialeval_evaluation_series_counts: tuple[_CategoryCountV1, ...] = Field(
        min_length=1
    )
    trialeval_regime_cell_counts: tuple[_CategoryCountV1, ...] = Field(min_length=1)
    trialeval_assumption_tier_counts: tuple[_CategoryCountV1, ...] = Field(min_length=1)
    trialeval_context_counts: tuple[_CategoryCountV1, ...] = Field(min_length=1)
    trialeval_estimator_family_counts: tuple[_CategoryCountV1, ...] = Field(
        min_length=1
    )
    trialeval_result_kind_counts: tuple[_CategoryCountV1, ...] = Field(min_length=1)
    trialdev_environment_counts: tuple[_CategoryCountV1, ...] = Field(min_length=1)
    trialdev_trajectory_counts: tuple[_CategoryCountV1, ...] = Field(min_length=1)
    trialdev_phase_counts: tuple[_CategoryCountV1, ...] = Field(min_length=1)
    trialdev_lane_counts: tuple[_CategoryCountV1, ...] = Field(min_length=1)
    trialdev_target_resolution_counts: tuple[_CategoryCountV1, ...] = Field(
        min_length=1
    )
    interpretation_limits: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_counts(self) -> _CandidateReleaseStatisticsV1:
        for name in (
            "suite_counts",
            "trialeval_evaluation_series_counts",
            "trialeval_regime_cell_counts",
            "trialeval_assumption_tier_counts",
            "trialeval_context_counts",
            "trialeval_estimator_family_counts",
            "trialeval_result_kind_counts",
            "trialdev_environment_counts",
            "trialdev_trajectory_counts",
            "trialdev_phase_counts",
            "trialdev_lane_counts",
            "trialdev_target_resolution_counts",
        ):
            categories = tuple(row.category for row in getattr(self, name))
            if categories != tuple(sorted(set(categories))):
                raise ValueError(f"{name} categories must be sorted and unique")
        if self.interpretation_limits != tuple(sorted(set(self.interpretation_limits))):
            raise ValueError("interpretation limits must be sorted and unique")
        return self


class _ReleaseManifestV1(_Contract):
    schema_id: Literal["trialagentbench.release_manifest/v1"]
    release_id: str = Field(min_length=1)
    release_stage: Literal["collaborator_single_seed", "paired_release"]
    parent_release_id: str | None
    creation_tool: str
    creation_tool_version: str
    source_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    environment_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    licenses: tuple[str, ...]
    citation: str
    intended_use: str
    role_archives: tuple[_ReleaseArtifactV1, ...] = Field(min_length=6, max_length=6)
    catalogue: _ReleaseCatalogueV1
    experiment_assignments: _ExperimentAssignmentsV1
    execution_budget_registry: _ReleaseArtifactV1
    reference_replay_feasibility: _ReleaseArtifactV1
    result_schema: _ReleaseArtifactV1
    result_export_manifest: _ReleaseArtifactV1
    candidate_release_statistics: _ReleaseArtifactV1
    grader_concordance: _ReleaseArtifactV1
    documentation_index: _ReleaseArtifactV1
    trialeval_design_profile_count: int
    trialeval_evaluation_series_count: int
    trialeval_regime_cell_count: int
    trialeval_base_trial_count: int
    trialeval_item_count: int
    trialdev_environment_count: int
    trialdev_trajectory_template_count: int
    trialdev_scenario_count: int
    public_commands: tuple[str, ...]
    failure_semantics: tuple[str, ...]
    comparison_semantics: tuple[str, ...]
    analysis_contracts: tuple[_AnalysisContractV1, ...]
    identity_definitions: tuple[str, ...]
    warnings: tuple[str, ...]

    @model_validator(mode="after")
    def _candidate_census(self) -> _ReleaseManifestV1:
        observed = (
            self.catalogue.record_count,
            self.catalogue.trialeval_record_count,
            self.catalogue.trialdev_record_count,
            self.trialeval_item_count,
            self.trialdev_scenario_count,
        )
        expected = (
            TRIALEVAL_ITEM_COUNT_V1 + TRIALDEV_SCENARIO_COUNT_V1,
            TRIALEVAL_ITEM_COUNT_V1,
            TRIALDEV_SCENARIO_COUNT_V1,
            TRIALEVAL_ITEM_COUNT_V1,
            TRIALDEV_SCENARIO_COUNT_V1,
        )
        if observed != expected:
            raise ValueError(
                "release manifest does not describe the complete v1 candidate census"
            )
        return self


class _CleanRoomFindingV1(_Contract):
    code: str
    path: str
    message: str


class _CleanRoomSurfaceV1(_Contract):
    role: Literal["participant", "evaluator", "verification", "audit", "harness"]
    root: str
    artifact_count: int = Field(ge=0)
    finding_count: int = Field(ge=0)


class _CleanRoomReportV1(_Contract):
    schema_id: Literal["trialagentbench.clean_room_workflow_report/v1"]
    status: Literal["pass", "fail"]
    package_root: str
    harness_root: str
    audit_roots: tuple[str, ...]
    participant_artifacts: tuple[str, ...]
    evaluator_artifacts: tuple[str, ...]
    verification_artifacts: tuple[str, ...]
    audit_artifacts: tuple[str, ...]
    surfaces: tuple[_CleanRoomSurfaceV1, ...]
    findings: tuple[_CleanRoomFindingV1, ...]


class _SeedCellV1(_Contract):
    suite: Literal["trialeval", "trialdev"]
    cell_id: str
    seed: int


class _SeedTreeV1(_Contract):
    schema_id: Literal["trialagentbench.release_seed_tree/v1"]
    root_seed: int = Field(ge=0)
    derivation_id: str
    cells: tuple[_SeedCellV1, ...]
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _valid_checksum(self) -> _SeedTreeV1:
        payload = self.model_dump(mode="json", exclude={"checksum"})
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        if hashlib.sha256(encoded).hexdigest() != self.checksum:
            raise ValueError("release seed-tree checksum does not match its content")
        return self


class _MaterializedCellV1(_Contract):
    suite: Literal["trialeval", "trialdev"]
    cell_id: str
    seed: int
    record: _RelativeArtifactV1
    status: Literal["materialized"]


class _MaterializationCensusV1(_Contract):
    schema_id: Literal["trialagentbench.release_materialization_census/v1"]
    release_id: str
    seed_tree_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    cells: tuple[_MaterializedCellV1, ...]
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _valid_checksum(self) -> _MaterializationCensusV1:
        payload = self.model_dump(mode="json", exclude={"checksum"})
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        if hashlib.sha256(encoded).hexdigest() != self.checksum:
            raise ValueError(
                "materialization census checksum does not match its content"
            )
        return self


class _ObservedValueV1(_Contract):
    value: float | int | None = None
    absence_reason: Literal["not_applicable", "not_observed", "withheld"] | None = None

    @model_validator(mode="after")
    def _one_state(self) -> _ObservedValueV1:
        if (self.value is None) == (self.absence_reason is None):
            raise ValueError(
                "observed information requires one value or absence reason"
            )
        return self


class _ObservedInformationV1(_Contract):
    participant_count: _ObservedValueV1
    treatment_arm_count: _ObservedValueV1
    event_count: _ObservedValueV1
    competing_event_count: _ObservedValueV1
    censoring_count: _ObservedValueV1
    missing_observation_count: _ObservedValueV1
    cluster_count: _ObservedValueV1
    period_count: _ObservedValueV1
    validation_substudy_count: _ObservedValueV1
    effective_sample_size: _ObservedValueV1


class _StudyFormatV1(_Contract):
    assignment: str
    allocation_unit: str
    masking: str
    comparator: str
    phase_or_purpose: str
    endpoint_family: str
    follow_up_horizon_days: float | None
    clustering: bool
    repeated_measures: str
    interim_or_adaptive: str


class _EstimandV1(_Contract):
    estimand_id: str
    analysis_population: str
    treatment_contrast: str
    endpoint_or_variable: str
    intercurrent_event_strategies: tuple[str, ...]
    population_summary_measure: str
    effect_scales: tuple[str, ...]
    identification_classes: tuple[str, ...]


class _ConstructionV1(_Contract):
    design_profile_id: str | None
    evaluation_series_id: str | None
    regime_cell_id: str | None
    design_tier: str | None
    assumption_tier: str | None
    context_id: str | None
    mechanism_environment_id: str | None
    trajectory_template_id: str | None
    within_series_stress_tier: str
    stress_class: str
    default_route_consequence: str
    context_data_surface: str
    descriptor_source: Literal["declared_construction_metadata"]
    disclosure_class: Literal[
        "public_observed", "declared_construction_metadata", "withheld"
    ]
    public_detectability_evidence_locator: str
    consequence_evidence_locator: str


class _IntegrityV1(_Contract):
    applicability: Literal["applicable", "not_applicable"]
    clean_context_parent_analysis_unit_id: str | None
    data_quality_condition_id: str | None
    defect_archetype_id: str | None
    defect_target_domain_id: str | None
    compound_key_fields: tuple[str, ...]
    defect_recoverability: str | None
    repair_contract_id: str | None
    target_domain_selector: str | None
    typed_scalar_encoding_id: str | None
    compound_key_encoding_id: str | None
    row_payload_encoding_id: str | None
    content_checksum_id: str | None
    public_disclosure_level: str | None
    evaluator_reference_locator: str | None
    verification_record_locator: str | None


class _UncertaintyV1(_Contract):
    reference_result_shapes: tuple[str, ...]
    uncertainty_methods: tuple[str, ...]
    confidence_level: float | None
    resampling_unit: str | None
    resampling_replicates: int | None
    support_requirements: tuple[str, ...]
    reporting_decimal_places: tuple[int, ...]
    maximum_comparison_tolerance: float | None
    nonpoint_output: str


class _ArtifactLinksV1(_Contract):
    participant_archive_path: str
    participant_member_path: str
    evaluator_archive_path: str
    verification_archive_path: str
    route_or_action_catalogue_path: str
    score_record_locator: str
    experiment_assignment_inventory_path: str
    canonical_result_path: str
    grade_record_path: str
    data_dictionary_path: str
    canonical_result_join_key: str


class _ProvenanceV1(_Contract):
    generation_seed_id: str
    component_contract_sha256: str
    participant_archive_sha256: str
    evaluator_archive_sha256: str
    verification_archive_sha256: str


class _GateApplicabilityV1(_Contract):
    gate_id: Literal[
        "submission",
        "question",
        "route",
        "evidence",
        "integrity",
        "result",
        "conformance",
        "decision",
    ]
    applicability: Literal["applicable", "not_applicable"]


class _ResolutionEvidenceMapEntryV1(_Contract):
    scoring_unit_id: str = Field(min_length=1)
    primary_evidence_class: Literal[
        "prescribed_execution",
        "empirical_diagnosis",
        "design_or_provenance_reasoning",
        "observed_failure_recovery",
        "evidence_insufficient",
    ]
    additional_evidence_obligations: tuple[Literal["observed_failure_recovery"], ...]
    method_policy: Literal[
        "prescribed", "participant_selects_qualified_route", "public_policy"
    ]
    qualified_method_or_policy_ids: tuple[str, ...] = Field(min_length=1)
    admissible_response_forms: tuple[str, ...] = Field(min_length=1)
    public_evidence_locators: tuple[str, ...] = Field(min_length=1)
    gates: tuple[_GateApplicabilityV1, ...] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def _canonical(self) -> _ResolutionEvidenceMapEntryV1:
        for name in (
            "additional_evidence_obligations",
            "qualified_method_or_policy_ids",
            "admissible_response_forms",
            "public_evidence_locators",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        expected_gates = (
            "submission",
            "question",
            "route",
            "evidence",
            "integrity",
            "result",
            "conformance",
            "decision",
        )
        if tuple(row.gate_id for row in self.gates) != expected_gates:
            raise ValueError(
                "resolution evidence maps require the ordered eight-gate cascade"
            )
        return self


class _TaskEnvironmentV1(_Contract):
    context_configuration: str = Field(min_length=1)
    data_preparation: str = Field(min_length=1)
    analysis_specification: str = Field(min_length=1)
    procedure_assistance: str = Field(min_length=1)
    response_interface: str = Field(min_length=1)
    tool_profile: Literal["sandboxed_python_with_released_files"]
    network_access: Literal[False]
    execution_budget_source: Literal["immutable_run_configuration"]
    execution_budget_profile_id: Literal[
        "trialeval_release_default_v1",
        "trialdev_release_default_v1",
    ]
    checkpoint_policy: Literal["write_once_assignment_checkpoint"]
    replay_policy: Literal["resume_only_from_validated_checkpoint"]


class _SimulationPropertyV1(_Contract):
    schema_id: Literal["trialagentbench.simulation_property/v1"]
    release_id: str
    suite_id: Literal["trialeval", "trialdev"]
    analysis_unit_id: str
    generation_unit_id: str
    independence_unit_id: str
    matched_set_id: str
    scoring_unit_ids: tuple[str, ...]
    study_format: _StudyFormatV1
    estimand: _EstimandV1
    construction: _ConstructionV1
    data_integrity: _IntegrityV1
    uncertainty: _UncertaintyV1
    observed_information: _ObservedInformationV1
    task_environment: _TaskEnvironmentV1
    resolution_evidence_map: tuple[_ResolutionEvidenceMapEntryV1, ...] = Field(
        min_length=1
    )
    artifact_links: _ArtifactLinksV1
    provenance: _ProvenanceV1

    @model_validator(mode="after")
    def _canonical_identity(self) -> _SimulationPropertyV1:
        if self.scoring_unit_ids != tuple(sorted(set(self.scoring_unit_ids))):
            raise ValueError("scoring-unit IDs must be sorted and unique")
        resolution_ids = tuple(
            row.scoring_unit_id for row in self.resolution_evidence_map
        )
        if resolution_ids != self.scoring_unit_ids:
            raise ValueError(
                "resolution evidence map must exactly cover the scoring-unit IDs"
            )
        expected = f"{self.release_id}:{self.suite_id}:{self.analysis_unit_id}"
        if self.artifact_links.canonical_result_join_key != expected:
            raise ValueError(
                "canonical result join key disagrees with the analysis-unit identity"
            )
        return self


class CandidateAnalysisConfigV1(_Contract):
    """Inputs for deterministic finite-candidate validation."""

    release_root: Path
    output_dir: Path
    verifier_lock: Path
    absolute_tolerance: float = Field(
        default=RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1, gt=0
    )

    @model_validator(mode="after")
    def _separate_outputs(self) -> CandidateAnalysisConfigV1:
        release = self.release_root.resolve()
        output = self.output_dir.resolve()
        if output == release or output.is_relative_to(release):
            raise ValueError(
                "candidate analysis output must be outside the immutable release root"
            )
        return self


def _read_jsonl(path: Path) -> tuple[_SimulationPropertyV1, ...]:
    rows = tuple(
        _SimulationPropertyV1.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    identities = tuple((row.suite_id, row.analysis_unit_id) for row in rows)
    if identities != tuple(sorted(identities)) or len(identities) != len(
        set(identities)
    ):
        raise ValueError("simulation properties must be sorted and unique")
    return rows


def _write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(
    headers: tuple[str, ...], rows: list[tuple[object, ...]]
) -> tuple[str, ...]:
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Markdown table rows must match their headers")
    return (
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    )


def _figure_svg(
    path: Path,
    *,
    title: str,
    axis_label: str,
    rows: list[dict[str, object]],
) -> None:
    width = 1000
    row_height = 28
    margin_left = 360
    height = max(180, 100 + row_height * len(rows))
    maximum = max((float(cast(float | int, row["value"])) for row in rows), default=1.0)
    maximum = maximum if maximum > 0 else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(axis_label)}. Values are printed beside each bar.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<defs><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" '
            'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="8" '
            'stroke="#ffffff" stroke-width="3"/></pattern><pattern id="crosshatch" '
            'width="8" height="8" patternUnits="userSpaceOnUse"><path d="M0,0 L8,8 M8,0 L0,8" '
            'stroke="#ffffff" stroke-width="2"/></pattern></defs>'
        ),
        f'<text x="24" y="38" font-family="sans-serif" font-size="22" font-weight="700">{escape(title)}</text>',
    ]
    for index, row in enumerate(rows):
        y = 72 + index * row_height
        value = float(cast(float | int, row["value"]))
        bar_width = 570 * value / maximum
        label = escape(str(row["label"]))
        display = escape(str(row.get("display", f"{value:g}")))
        series = str(row.get("series", "candidate"))
        color = {
            "candidate": "#1565c0",
            "pass": "#00796b",
            "observed": "#1565c0",
            "non_estimable": "#607d8b",
            "fail": "#b3261e",
        }.get(series, "#607d8b")
        pattern = {
            "non_estimable": '<rect fill="url(#hatch)"',
            "fail": '<rect fill="url(#crosshatch)"',
        }.get(series)
        parts.extend(
            (
                f'<text x="24" y="{y + 16}" font-family="sans-serif" font-size="13">{label}</text>',
                f'<rect x="{margin_left}" y="{y}" width="{bar_width:.3f}" height="18" fill="{color}"/>',
                (
                    f'{pattern} x="{margin_left}" y="{y}" width="{bar_width:.3f}" height="18"/>'
                    if pattern is not None
                    else ""
                ),
                f'<text x="{margin_left + bar_width + 8:.3f}" y="{y + 15}" '
                f'font-family="sans-serif" font-size="12">{display}</text>',
            )
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _figure_review_renderings(
    *,
    png_path: Path,
    pdf_path: Path,
    title: str,
    axis_label: str,
    rows: list[dict[str, object]],
) -> None:
    height = min(48.0, max(3.2, 1.2 + 0.28 * len(rows)))
    figure = Figure(figsize=(11.0, height), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    values = [float(cast(float | int, row["value"])) for row in rows]
    labels = [str(row["label"]) for row in rows]
    colors = [
        {
            "candidate": "#1565c0",
            "pass": "#00796b",
            "observed": "#1565c0",
            "non_estimable": "#607d8b",
            "fail": "#b3261e",
        }.get(str(row.get("series", "candidate")), "#607d8b")
        for row in rows
    ]
    hatches = [
        {
            "non_estimable": "///",
            "fail": "xxx",
        }.get(str(row.get("series", "candidate")), "")
        for row in rows
    ]
    positions = list(range(len(rows)))
    bars = axis.barh(
        positions, values, color=colors, edgecolor="#ffffff", linewidth=0.6
    )
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)
    axis.set_yticks(positions, labels=labels, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlabel(axis_label)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    maximum = max(values, default=0.0)
    offset = maximum * 0.01 if maximum else 0.01
    for position, value, row in zip(positions, values, rows, strict=True):
        axis.text(
            value + offset,
            position,
            str(row.get("display", f"{value:g}")),
            va="center",
            fontsize=7,
        )
    fixed_time = datetime(2026, 7, 28)
    figure.savefig(
        png_path,
        dpi=160,
        format="png",
        metadata={"Software": "TrialAgentBench"},
    )
    figure.savefig(
        pdf_path,
        format="pdf",
        metadata={
            "Creator": "TrialAgentBench",
            "Producer": "Matplotlib",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )


def _write_figure(
    root: Path,
    *,
    figure_id: str,
    title: str,
    question: str,
    independent_unit: str,
    estimand: str,
    comparator: str,
    uncertainty: str,
    rows: list[dict[str, object]],
    interpretation: tuple[str, ...],
) -> ValidationFigureV1:
    stem = figure_id.replace(".", "_")
    csv_path = root / "figures" / f"{stem}.csv"
    pdf_path = root / "figures" / f"{stem}.pdf"
    png_path = root / "figures" / f"{stem}.png"
    svg_path = root / "figures" / f"{stem}.svg"
    _write_csv(csv_path, rows, ("label", "value", "display", "series"))
    _figure_svg(svg_path, title=title, axis_label=estimand, rows=rows)
    _figure_review_renderings(
        png_path=png_path,
        pdf_path=pdf_path,
        title=title,
        axis_label=estimand,
        rows=rows,
    )
    artifacts = tuple(
        sorted(
            (
                ValidationArtifactV1(
                    relative_path=csv_path.relative_to(root).as_posix(),
                    sha256=sha256_file(csv_path),
                    media_type="text/csv",
                ),
                ValidationArtifactV1(
                    relative_path=pdf_path.relative_to(root).as_posix(),
                    sha256=sha256_file(pdf_path),
                    media_type="application/pdf",
                ),
                ValidationArtifactV1(
                    relative_path=png_path.relative_to(root).as_posix(),
                    sha256=sha256_file(png_path),
                    media_type="image/png",
                ),
                ValidationArtifactV1(
                    relative_path=svg_path.relative_to(root).as_posix(),
                    sha256=sha256_file(svg_path),
                    media_type="image/svg+xml",
                ),
            ),
            key=lambda artifact: artifact.relative_path,
        )
    )
    return ValidationFigureV1(
        figure_id=figure_id,
        title=title,
        scientific_question=question,
        independent_unit=independent_unit,
        estimand=estimand,
        comparator=comparator,
        uncertainty=uncertainty,
        interpretation=interpretation,
        artifacts=artifacts,
    )


def _role_archives(
    root: Path, manifest: _ReleaseManifestV1
) -> tuple[CandidateRoleArchiveV1, ...]:
    declared = {Path(row.path).name: row.sha256 for row in manifest.role_archives}
    records = []
    for suite, directory in (
        ("trialeval", "TrialEvalBench"),
        ("trialdev", "TrialDevBench"),
    ):
        for role in ("participant", "evaluator", "verification"):
            path = root / "public" / directory / f"{directory}_{role}.zip"
            if not path.is_file():
                raise FileNotFoundError(f"candidate role archive is missing: {path}")
            digest = sha256_file(path)
            if declared.get(path.name) != digest:
                raise ValueError(f"release manifest checksum disagrees for {path.name}")
            records.append(
                CandidateRoleArchiveV1(
                    suite=suite,
                    role=role,
                    relative_path=path.relative_to(root).as_posix(),
                    sha256=digest,
                )
            )
    return tuple(sorted(records, key=lambda row: (row.suite, row.role)))


def _public_wheels(root: Path) -> tuple[CandidatePublicWheelV1, ...]:
    public = root / "public"
    manifest = _PublicDatasetManifestV1.model_validate_json(
        (public / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    declared = {row.name: row.sha256 for row in manifest.artifacts}
    records = []
    for package in (
        "trialagentbench-harness",
        "trialagentbench-validation",
    ):
        wheels = tuple(sorted((public / "packages" / package).glob("*.whl")))
        if len(wheels) != 1:
            raise ValueError(
                f"candidate requires exactly one public wheel for {package}"
            )
        path = wheels[0]
        relative = path.relative_to(public).as_posix()
        digest = sha256_file(path)
        if declared.get(relative) != digest:
            raise ValueError(f"public package manifest does not bind {relative}")
        records.append(
            CandidatePublicWheelV1(
                package=package,
                relative_path=path.relative_to(root).as_posix(),
                sha256=digest,
            )
        )
    return tuple(records)


def _candidate_identity(
    root: Path,
    manifest: _ReleaseManifestV1,
    seed_tree: _SeedTreeV1,
    census: _MaterializationCensusV1,
) -> CandidateIdentityV1:
    if (
        census.release_id != manifest.release_id
        or census.seed_tree_checksum != seed_tree.checksum
    ):
        raise ValueError(
            "candidate materialization census is not bound to its release and seed tree"
        )
    expected_cells = tuple(
        (row.suite, row.cell_id, row.seed) for row in seed_tree.cells
    )
    observed_cells = tuple((row.suite, row.cell_id, row.seed) for row in census.cells)
    if expected_cells != observed_cells:
        raise ValueError("candidate seed tree and materialization census disagree")
    payload = {
        "schema_id": "trialagentbench.candidate_identity/v1",
        "release_id": manifest.release_id,
        "source_commit": manifest.source_commit,
        "environment_lock_sha256": manifest.environment_lock_sha256,
        "release_manifest_sha256": sha256_file(root / "RELEASE_MANIFEST.json"),
        "staged_manifest_sha256": sha256_file(root / "staged_release_manifest.json"),
        "seed_tree_sha256": sha256_file(root / "provenance" / "release_seed_tree.json"),
        "materialization_census_sha256": sha256_file(
            root / "provenance" / "materialization_census.json"
        ),
        "root_seed": seed_tree.root_seed,
        "trialeval_item_count": manifest.trialeval_item_count,
        "trialdev_scenario_count": manifest.trialdev_scenario_count,
        "role_archives": [
            row.model_dump(mode="json") for row in _role_archives(root, manifest)
        ],
        "public_wheels": [row.model_dump(mode="json") for row in _public_wheels(root)],
    }
    checksum = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
    return CandidateIdentityV1(**payload, checksum=checksum)


def _validate_properties(
    rows: tuple[_SimulationPropertyV1, ...],
    identity: CandidateIdentityV1,
) -> None:
    if any(row.release_id != identity.release_id for row in rows):
        raise ValueError("simulation properties mix release identities")
    trialeval = tuple(row for row in rows if row.suite_id == "trialeval")
    trialdev = tuple(row for row in rows if row.suite_id == "trialdev")
    if (
        len(trialeval) != TRIALEVAL_ITEM_COUNT_V1
        or len(trialdev) != TRIALDEV_SCENARIO_COUNT_V1
    ):
        raise ValueError(
            "simulation properties do not contain the exact 500/50 candidate census"
        )
    by_match: dict[str, list[_SimulationPropertyV1]] = defaultdict(list)
    for row in trialeval:
        by_match[row.matched_set_id].append(row)
    if len(by_match) * TRIALEVAL_CONTEXT_COUNT_V1 != TRIALEVAL_ITEM_COUNT_V1:
        raise ValueError(
            "TrialEval does not contain exactly 100 matched five-context base trials"
        )
    expected_contexts = {"C1", "C2", "C3", "C4", "C5"}
    for matched_set, group in by_match.items():
        contexts = {row.construction.context_id for row in group}
        if contexts != expected_contexts:
            raise ValueError(
                f"TrialEval matched set lacks the complete C1-C5 panel: {matched_set}"
            )
        invariant = {
            (
                row.provenance.generation_seed_id,
                row.estimand,
                row.study_format,
                row.construction.regime_cell_id,
            )
            for row in group
        }
        if len(invariant) != 1:
            raise ValueError(
                f"TrialEval contexts alter the generated question: {matched_set}"
            )


def _category_counts(values: tuple[str, ...]) -> tuple[_CategoryCountV1, ...]:
    return tuple(
        _CategoryCountV1(category=category, count=count)
        for category, count in sorted(Counter(values).items())
    )


def _validate_candidate_release_statistics(
    *,
    root: Path,
    manifest: _ReleaseManifestV1,
    properties: tuple[_SimulationPropertyV1, ...],
    trialeval_verification: Path,
    trialdev_verification: Path,
) -> None:
    artifact = manifest.candidate_release_statistics
    if artifact.path != "metadata/candidate_release_statistics.json":
        raise ValueError(
            "candidate-release statistics must use the canonical release path"
        )
    path = root / artifact.path
    if sha256_file(path) != artifact.sha256:
        raise ValueError(
            "candidate-release statistics checksum disagrees with the release manifest"
        )
    statistics = _CandidateReleaseStatisticsV1.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    with ZipFile(trialeval_verification) as archive:
        trialeval_inventory = (
            TrialEvalScientificConstructionInventoryV1.model_validate_json(
                archive.read("construction/scientific_construction_inventory.json")
            )
        )
    with ZipFile(trialdev_verification) as archive:
        trialdev_inventory = (
            TrialDevScientificConstructionInventoryV1.model_validate_json(
                archive.read("scientific_construction_inventory.json")
            )
        )
    if (
        statistics.release_id != manifest.release_id
        or trialeval_inventory.release_id != manifest.release_id
        or trialdev_inventory.release_id != manifest.release_id
    ):
        raise ValueError(
            "candidate-release statistics and scientific inventories must identify the release"
        )
    trialeval = tuple(row for row in properties if row.suite_id == "trialeval")
    trialdev = tuple(row for row in properties if row.suite_id == "trialdev")
    expected = {
        "analysis_unit_count": len(properties),
        "generation_unit_count": len(
            {(row.suite_id, row.generation_unit_id) for row in properties}
        ),
        "scoring_unit_count": sum(len(row.scoring_unit_ids) for row in properties),
        "trialeval_route_reference_count": len(trialeval_inventory.rows),
        "trialdev_reference_policy_target_count": len(
            {
                (row.scenario_id, row.lane_id, target_id)
                for row in trialdev_inventory.rows
                for target_id in row.reference_target_ids
            }
        ),
        "trialdev_credit_eligible_policy_target_count": len(
            {
                (row.scenario_id, row.lane_id, target_id)
                for row in trialdev_inventory.rows
                for target_id in row.credit_eligible_target_ids
            }
        ),
        "suite_counts": _category_counts(tuple(row.suite_id for row in properties)),
        "trialeval_evaluation_series_counts": _category_counts(
            tuple(
                row.construction.evaluation_series_id or "not_applicable"
                for row in trialeval
            )
        ),
        "trialeval_regime_cell_counts": _category_counts(
            tuple(
                row.construction.regime_cell_id or "not_applicable" for row in trialeval
            )
        ),
        "trialeval_assumption_tier_counts": _category_counts(
            tuple(
                row.construction.assumption_tier or "not_applicable"
                for row in trialeval
            )
        ),
        "trialeval_context_counts": _category_counts(
            tuple(row.construction.context_id or "not_applicable" for row in trialeval)
        ),
        "trialeval_estimator_family_counts": _category_counts(
            tuple(row.estimator_family for row in trialeval_inventory.rows)
        ),
        "trialeval_result_kind_counts": _category_counts(
            tuple(row.result_kind for row in trialeval_inventory.rows)
        ),
        "trialdev_environment_counts": _category_counts(
            tuple(
                row.construction.mechanism_environment_id or "not_applicable"
                for row in trialdev
            )
        ),
        "trialdev_trajectory_counts": _category_counts(
            tuple(
                row.construction.trajectory_template_id or "not_applicable"
                for row in trialdev
            )
        ),
        "trialdev_phase_counts": _category_counts(
            tuple(row.phase_id for row in trialdev_inventory.rows)
        ),
        "trialdev_lane_counts": _category_counts(
            tuple(row.lane_id for row in trialdev_inventory.rows)
        ),
        "trialdev_target_resolution_counts": _category_counts(
            tuple(row.target_resolution for row in trialdev_inventory.rows)
        ),
    }
    for field, value in expected.items():
        if getattr(statistics, field) != value:
            raise ValueError(f"candidate-release statistics disagree for {field}")


def _construction_rows(
    properties: tuple[_SimulationPropertyV1, ...],
) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in properties:
        dimensions: tuple[tuple[str, str], ...]
        shared_dimensions = (
            ("assignment", row.study_format.assignment),
            ("allocation_unit", row.study_format.allocation_unit),
            ("masking", row.study_format.masking),
            ("comparator", row.study_format.comparator),
            ("phase_or_purpose", row.study_format.phase_or_purpose),
            ("endpoint_family", row.study_format.endpoint_family),
            ("estimand", row.estimand.estimand_id),
            ("analysis_population", row.estimand.analysis_population),
            ("population_summary_measure", row.estimand.population_summary_measure),
            ("stress_class", row.construction.stress_class),
            ("within_series_stress_tier", row.construction.within_series_stress_tier),
        )
        if row.suite_id == "trialeval":
            dimensions = (
                ("design_profile", row.construction.design_profile_id or "missing"),
                ("design_tier", row.construction.design_tier or "missing"),
                ("assumption_tier", row.construction.assumption_tier or "missing"),
                ("context", row.construction.context_id or "missing"),
                *shared_dimensions,
            )
        else:
            dimensions = (
                ("environment", row.construction.mechanism_environment_id or "missing"),
                ("trajectory", row.construction.trajectory_template_id or "missing"),
                *shared_dimensions,
            )
        dimensions = (
            *dimensions,
            *(("effect_scale", label) for label in row.estimand.effect_scales),
            *(
                ("identification_class", label)
                for label in row.estimand.identification_classes
            ),
            *(
                ("intercurrent_event_strategy", label)
                for label in row.estimand.intercurrent_event_strategies
            ),
        )
        for dimension, label in dimensions:
            counts[(row.suite_id, dimension, label)] += 1
    return [
        {"suite": suite, "dimension": dimension, "label": label, "count": count}
        for (suite, dimension, label), count in sorted(counts.items())
    ]


def _analysis_unit_rows(
    properties: tuple[_SimulationPropertyV1, ...],
    characteristics: list[dict[str, object]],
) -> list[dict[str, object]]:
    realized = {
        (str(row["suite"]), str(row["analysis_unit_id"]), str(row["metric"])): row[
            "value"
        ]
        for row in characteristics
    }
    rows: list[dict[str, object]] = []
    for item in properties:
        observed = {
            metric: value.value
            for metric in _ObservedInformationV1.model_fields
            if (value := getattr(item.observed_information, metric)).value is not None
        }
        realized_item = {
            metric: value
            for (suite, unit, metric), value in realized.items()
            if suite == item.suite_id and unit == item.analysis_unit_id
        }
        values = {**observed, **realized_item}
        allocation_counts = "|".join(
            f"{metric.removeprefix('allocation_count::')}={int(cast(float | int, value))}"
            for metric, value in sorted(realized_item.items())
            if metric.startswith("allocation_count::")
            and isinstance(value, int | float)
        )

        rows.append(
            {
                "suite": item.suite_id,
                "analysis_unit_id": item.analysis_unit_id,
                "generation_unit_id": item.generation_unit_id,
                "independence_unit_id": item.independence_unit_id,
                "matched_set_id": item.matched_set_id,
                "design_profile_id": item.construction.design_profile_id,
                "design_tier": item.construction.design_tier,
                "assumption_tier": item.construction.assumption_tier,
                "context_id": item.construction.context_id,
                "mechanism_environment_id": item.construction.mechanism_environment_id,
                "trajectory_template_id": item.construction.trajectory_template_id,
                "stress_class": item.construction.stress_class,
                "assignment": item.study_format.assignment,
                "allocation_unit": item.study_format.allocation_unit,
                "masking": item.study_format.masking,
                "comparator": item.study_format.comparator,
                "phase_or_purpose": item.study_format.phase_or_purpose,
                "endpoint_family": item.study_format.endpoint_family,
                "estimand_id": item.estimand.estimand_id,
                "analysis_population": item.estimand.analysis_population,
                "treatment_contrast": item.estimand.treatment_contrast,
                "endpoint_or_variable": item.estimand.endpoint_or_variable,
                "intercurrent_event_strategies": "|".join(
                    item.estimand.intercurrent_event_strategies
                ),
                "population_summary_measure": item.estimand.population_summary_measure,
                "effect_scales": "|".join(item.estimand.effect_scales),
                "identification_classes": "|".join(
                    item.estimand.identification_classes
                ),
                "participant_count": values.get("participant_count"),
                "treatment_arm_count": values.get("treatment_arm_count"),
                "event_count": values.get("event_count"),
                "competing_event_count": values.get("competing_event_count"),
                "censoring_count": values.get("censoring_count"),
                "missing_observation_count": values.get("missing_observation_count"),
                "cluster_count": values.get("cluster_count"),
                "period_count": values.get("period_count"),
                "validation_substudy_count": values.get("validation_substudy_count"),
                "effective_sample_size": values.get("effective_sample_size"),
                "allocation_counts": allocation_counts,
                "intercurrent_event_count": values.get("intercurrent_event_count"),
                "nonadherence_intercurrent_event_count": values.get(
                    "nonadherence_intercurrent_event_count"
                ),
                "treatment_switch_count": values.get("treatment_switch_count"),
                "planned_exposure_transition_count": values.get(
                    "planned_exposure_transition_count"
                ),
                "rescue_count": values.get("rescue_count"),
                "discontinuation_count": values.get("discontinuation_count"),
                "study_discontinuation_count": values.get(
                    "study_discontinuation_count"
                ),
                "safety_event_count": values.get("safety_event_count"),
                "serious_safety_event_count": values.get("serious_safety_event_count"),
                "treatment_emergent_safety_event_count": values.get(
                    "treatment_emergent_safety_event_count"
                ),
                "loss_to_follow_up_count": values.get("loss_to_follow_up_count"),
            }
        )
    return rows


def _stratified_characteristic_rows(
    properties: tuple[_SimulationPropertyV1, ...],
    characteristics: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize finite-release characteristics over each declared design stratum."""

    property_by_unit: dict[tuple[str, str], _SimulationPropertyV1] = {
        (row.suite_id, row.analysis_unit_id): row for row in properties
    }
    values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    observed: Counter[tuple[str, str, str, str]] = Counter()
    not_observed: Counter[tuple[str, str, str, str]] = Counter()
    analysis_units: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    independence_units: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)

    for row in characteristics:
        suite = str(row["suite"])
        analysis_unit_id = str(row["analysis_unit_id"])
        try:
            item = property_by_unit[(suite, analysis_unit_id)]
        except KeyError as error:
            raise ValueError(
                "generated characteristic identifies no candidate analysis unit: "
                f"suite={suite!r} analysis_unit_id={analysis_unit_id!r}"
            ) from error
        strata = (
            ("all", "all"),
            ("design_profile", item.construction.design_profile_id),
            ("design_tier", item.construction.design_tier),
            ("assumption_tier", item.construction.assumption_tier),
            ("context", item.construction.context_id),
            ("regime_cell", item.construction.regime_cell_id),
            ("environment", item.construction.mechanism_environment_id),
            ("trajectory", item.construction.trajectory_template_id),
            ("endpoint_family", item.study_format.endpoint_family),
            ("endpoint", item.estimand.endpoint_or_variable),
            ("estimand", item.estimand.estimand_id),
        )
        metric = str(row["metric"])
        value = row["value"]
        status = str(row["status"])
        if status not in {"observed", "not_observed"}:
            raise ValueError(f"unknown generated-characteristic status: {status!r}")
        if status == "observed":
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    "observed generated characteristics require a finite numeric value"
                )
        elif value is not None:
            raise ValueError(
                "not-observed generated characteristics cannot contain a value"
            )
        for stratifier, raw_stratum in strata:
            if raw_stratum is None:
                continue
            stratum = str(raw_stratum)
            key = (suite, stratifier, stratum, metric)
            analysis_units[key].add(analysis_unit_id)
            independence_units[key].add(item.independence_unit_id)
            if status == "observed":
                observed[key] += 1
                values[key].append(float(cast(int | float, value)))
            else:
                not_observed[key] += 1

    rows: list[dict[str, object]] = []
    for key in sorted(set(analysis_units) | set(observed) | set(not_observed)):
        suite, stratifier, stratum, metric = key
        numeric = sorted(values.get(key, ()))
        rows.append(
            {
                "suite": suite,
                "stratifier": stratifier,
                "stratum": stratum,
                "metric": metric,
                "analysis_unit_count": len(analysis_units[key]),
                "independence_unit_count": len(independence_units[key]),
                "observed_count": observed[key],
                "not_observed_count": not_observed[key],
                "minimum": numeric[0] if numeric else None,
                "median": float(median(numeric)) if numeric else None,
                "maximum": numeric[-1] if numeric else None,
            }
        )
    return rows


def _observed_rows(
    properties: tuple[_SimulationPropertyV1, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in properties:
        for metric in _ObservedInformationV1.model_fields:
            value = getattr(item.observed_information, metric)
            rows.append(
                {
                    "suite": item.suite_id,
                    "analysis_unit_id": item.analysis_unit_id,
                    "matched_set_id": item.matched_set_id,
                    "metric": metric,
                    "value": value.value,
                    "absence_reason": value.absence_reason,
                }
            )
    return rows


def _stable_analysis_hash(
    adsl: pd.DataFrame, adtte: pd.DataFrame, *, endpoint_id: str
) -> str:
    """Hash the analysis-defining participant, treatment, time, and event values."""

    if "USUBJID" not in adsl or "TRTA" not in adsl:
        raise ValueError("candidate ADSL requires USUBJID and TRTA")
    missing = sorted(set(_TRIALEVAL_ANALYSIS_COLUMNS) - set(adtte.columns))
    if missing:
        raise ValueError(
            f"candidate ADTTE lacks analysis-defining columns: {missing!r}"
        )
    endpoint = adtte.loc[
        adtte["PARAMCD"]
        .astype("string")
        .isin({endpoint_id, "__RECONSTRUCTED_PRIMARY__"}),
        list(_TRIALEVAL_ANALYSIS_COLUMNS),
    ].copy()
    if endpoint.empty:
        raise ValueError(f"candidate ADTTE has no primary endpoint rows: {endpoint_id}")
    participants = adsl.loc[:, ["USUBJID", "TRTA"]].drop_duplicates().copy()
    participants["USUBJID"] = participants["USUBJID"].astype("string")
    participants["TRTA"] = participants["TRTA"].astype("string")
    endpoint["USUBJID"] = endpoint["USUBJID"].astype("string")
    endpoint["PARAMCD"] = endpoint_id
    endpoint["AVAL"] = pd.to_numeric(endpoint["AVAL"], errors="raise").round(12)
    endpoint["CNSR"] = pd.to_numeric(endpoint["CNSR"], errors="raise").astype(int)
    participants = participants.sort_values(["USUBJID", "TRTA"], kind="mergesort")
    endpoint = (
        endpoint.drop_duplicates()
        .sort_values(list(_TRIALEVAL_ANALYSIS_COLUMNS), kind="mergesort")
        .reset_index(drop=True)
    )
    payload = (
        participants.to_csv(index=False, lineterminator="\n")
        + "\n"
        + endpoint.to_csv(index=False, lineterminator="\n")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_item_parquet(
    archive: ZipFile,
    *,
    task_id: str,
    table_name: str,
) -> pd.DataFrame:
    member = f"items/{task_id}/data/{table_name}.parquet"
    if member not in archive.namelist():
        raise FileNotFoundError(
            f"candidate participant archive lacks required table: {member}"
        )
    return pd.read_parquet(BytesIO(archive.read(member)))


def _read_optional_item_parquet(
    archive: ZipFile,
    *,
    task_id: str,
    table_name: str,
) -> pd.DataFrame | None:
    member = f"items/{task_id}/data/{table_name}.parquet"
    if member not in archive.namelist():
        return None
    return pd.read_parquet(BytesIO(archive.read(member)))


def _nonnegative_integer_sum(frame: pd.DataFrame, *, column: str) -> int:
    if column not in frame:
        raise ValueError(f"candidate operational flags lack required column: {column}")
    values = pd.to_numeric(frame[column], errors="raise")
    if values.isna().any() or (values < 0).any() or not values.mod(1).eq(0).all():
        raise ValueError(
            f"candidate operational count must contain non-negative integers: {column}"
        )
    return int(values.sum())


def _trialeval_base_characteristics(
    archive: ZipFile,
    *,
    task_id: str,
) -> dict[str, tuple[int | None, str]]:
    flags = _read_item_parquet(
        archive,
        task_id=task_id,
        table_name="subject_operational_flags",
    )
    adae = _read_item_parquet(archive, task_id=task_id, table_name="ADAE")
    adlb = _read_optional_item_parquet(archive, task_id=task_id, table_name="ADLB")
    advs = _read_optional_item_parquet(archive, task_id=task_id, table_name="ADVS")
    if "USUBJID" not in flags or flags["USUBJID"].astype("string").duplicated().any():
        raise ValueError("candidate operational flags require one row per participant")
    for name, frame in (("ADLB", adlb), ("ADVS", advs)):
        if frame is not None and "AVAL" not in frame:
            raise ValueError(
                f"candidate {name} requires AVAL for missing-observation census"
            )
    if "AESER" not in adae or "TRTEMFL" not in adae:
        raise ValueError("candidate ADAE requires AESER and TRTEMFL for safety census")

    metrics: dict[str, tuple[int | None, str]] = {
        metric: (
            _nonnegative_integer_sum(flags, column=column),
            f"sum of participant-level {column} counts in the matched C1 generated trial",
        )
        for metric, column in _TRIALEVAL_OPERATIONAL_COUNT_COLUMNS.items()
    }
    metrics.update(
        {
            "laboratory_missing_observation_count": (
                (
                    int(pd.to_numeric(adlb["AVAL"], errors="coerce").isna().sum())
                    if adlb is not None
                    else None
                ),
                (
                    "ADLB rows with missing numeric AVAL in the matched C1 generated trial; "
                    "not observed when ADLB is outside the released design surface"
                ),
            ),
            "vital_sign_missing_observation_count": (
                (
                    int(pd.to_numeric(advs["AVAL"], errors="coerce").isna().sum())
                    if advs is not None
                    else None
                ),
                (
                    "ADVS rows with missing numeric AVAL in the matched C1 generated trial; "
                    "not observed when ADVS is outside the released design surface"
                ),
            ),
            "study_discontinuation_count": (
                int(flags["ANY_STUDY_DISCONTINUATION"].astype("string").eq("Y").sum()),
                "participants with ANY_STUDY_DISCONTINUATION=Y in the matched C1 generated trial",
            ),
            "safety_event_count": (
                int(len(adae)),
                "released ADAE event records in the matched C1 generated trial",
            ),
            "serious_safety_event_count": (
                int(adae["AESER"].astype("string").eq("Y").sum()),
                "released ADAE records with AESER=Y in the matched C1 generated trial",
            ),
            "treatment_emergent_safety_event_count": (
                int(adae["TRTEMFL"].astype("string").eq("Y").sum()),
                "released ADAE records with TRTEMFL=Y in the matched C1 generated trial",
            ),
        }
    )
    return metrics


def _trialeval_characteristics(
    archive_path: Path,
    properties: tuple[_SimulationPropertyV1, ...],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    rows: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    with ZipFile(archive_path) as archive:
        items = tuple(item for item in properties if item.suite_id == "trialeval")
        c1_by_match = {
            item.matched_set_id: item.analysis_unit_id
            for item in items
            if item.construction.context_id == "C1"
        }
        if set(c1_by_match) != {item.matched_set_id for item in items}:
            raise ValueError(
                "TrialEval characteristics require one C1 source for every matched context panel"
            )
        base_characteristics = {
            matched_set_id: _trialeval_base_characteristics(archive, task_id=task_id)
            for matched_set_id, task_id in sorted(c1_by_match.items())
        }
        for item in items:
            endpoint_id = item.estimand.endpoint_or_variable
            adsl, adtte, _ = load_public_analysis_tables_v1(
                public=archive,
                task_id=item.analysis_unit_id,
                paramcd=endpoint_id,
            )
            endpoint = adtte.loc[
                adtte["PARAMCD"]
                .astype("string")
                .isin({endpoint_id, "__RECONSTRUCTED_PRIMARY__"})
            ].copy()
            if endpoint.empty:
                raise ValueError(
                    f"candidate has no primary endpoint rows: {item.analysis_unit_id}"
                )
            cnsr = pd.to_numeric(endpoint["CNSR"], errors="coerce")
            aval = pd.to_numeric(endpoint["AVAL"], errors="coerce")
            if not set(cnsr.dropna().astype(int).unique()) <= {0, 1}:
                raise ValueError(
                    f"candidate CNSR is not binary: {item.analysis_unit_id}"
                )
            endpoint_missing = int((aval.isna() | cnsr.isna()).sum())
            detailed = base_characteristics[item.matched_set_id]
            metrics: list[tuple[str, int | None, str]] = [
                (
                    "participant_count",
                    int(adsl["USUBJID"].astype("string").nunique()),
                    "unique ADSL USUBJID",
                ),
                (
                    "treatment_arm_count",
                    int(adsl["TRTA"].astype("string").nunique()),
                    "unique randomized analysis treatment",
                ),
                (
                    "primary_endpoint_row_count",
                    int(len(endpoint)),
                    "released ADTTE rows for the declared primary endpoint",
                ),
                (
                    "event_count",
                    int(cnsr.eq(0).sum()),
                    "primary endpoint rows with ADaM CNSR=0",
                ),
                (
                    "censoring_count",
                    int(cnsr.eq(1).sum()),
                    "primary endpoint rows with ADaM CNSR=1",
                ),
                (
                    "primary_endpoint_missing_observation_count",
                    endpoint_missing,
                    "primary endpoint rows missing AVAL or CNSR",
                ),
                (
                    "missing_observation_count",
                    endpoint_missing
                    + sum(
                        int(value)
                        for metric in (
                            "laboratory_missing_observation_count",
                            "vital_sign_missing_observation_count",
                        )
                        if (value := detailed[metric][0]) is not None
                    ),
                    (
                        "missing primary-endpoint observations plus missing AVAL values in "
                        "released ADLB and ADVS tables"
                    ),
                ),
                (
                    "competing_event_count",
                    0,
                    "the TrialEval v1 primary endpoint contract contains no competing-event component",
                ),
                (
                    "cluster_count",
                    (
                        int(adsl["SITEID"].astype("string").nunique())
                        if "SITEID" in adsl
                        else None
                    ),
                    "unique ADSL SITEID when cluster identity is released",
                ),
                (
                    "planned_exposure_transition_count",
                    (
                        int(
                            pd.to_numeric(
                                adsl["INTERVENTION_START_DY"], errors="coerce"
                            )
                            .notna()
                            .sum()
                        )
                        if "INTERVENTION_START_DY" in adsl
                        else None
                    ),
                    "participants with a released stepped-wedge intervention start",
                ),
            ]
            allocations = adsl.groupby(
                adsl["TRTA"].astype("string"), dropna=False
            ).size()
            if any(pd.isna(arm) for arm in allocations.index):
                raise ValueError(
                    f"candidate treatment allocation is missing: {item.analysis_unit_id}"
                )
            metrics.extend(
                (
                    f"allocation_count::{arm}",
                    int(count),
                    f"participants randomized to released TRTA={arm}",
                )
                for arm, count in sorted(
                    ((str(arm), int(count)) for arm, count in allocations.items()),
                    key=lambda pair: pair[0],
                )
            )
            metrics.extend(
                (metric, value, definition)
                for metric, (value, definition) in sorted(detailed.items())
            )
            for metric, value, definition in metrics:
                rows.append(
                    {
                        "suite": "trialeval",
                        "analysis_unit_id": item.analysis_unit_id,
                        "matched_set_id": item.matched_set_id,
                        "metric": metric,
                        "value": value,
                        "status": "observed" if value is not None else "not_observed",
                        "definition": definition,
                    }
                )
            hashes[item.analysis_unit_id] = _stable_analysis_hash(
                adsl,
                adtte,
                endpoint_id=endpoint_id,
            )
    return rows, hashes


def _trialdev_characteristics(
    archive_path: Path,
    properties: tuple[_SimulationPropertyV1, ...],
) -> list[dict[str, object]]:
    def event_total(frame: pd.DataFrame, columns: tuple[object, ...]) -> int:
        return int(
            sum(
                pd.to_numeric(frame[column], errors="coerce").fillna(0).eq(1).sum()
                for column in columns
            )
        )

    rows: list[dict[str, object]] = []
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        for item in properties:
            if item.suite_id != "trialdev":
                continue
            member = (
                f"scenario_{item.analysis_unit_id}/public/"
                "observational_extract.parquet"
            )
            if member not in names:
                raise FileNotFoundError(
                    f"candidate TrialDev observational extract is missing: {member}"
                )
            frame = pd.read_parquet(BytesIO(archive.read(member)))
            missing = sorted(set(_TRIALDEV_REQUIRED_COLUMNS) - set(frame.columns))
            if missing:
                raise ValueError(
                    f"candidate TrialDev extract lacks required columns: {missing!r}"
                )
            if frame["USUBJID"].astype("string").duplicated().any():
                raise ValueError(
                    "candidate TrialDev observational extract requires one row per participant"
                )
            efficacy_columns = tuple(
                column
                for column in frame.columns
                if str(column).startswith(_TRIALDEV_EFFICACY_EVENT_PREFIX)
                and str(column).endswith(_TRIALDEV_EFFICACY_EVENT_SUFFIX)
            )
            safety_columns = tuple(
                column
                for column in frame.columns
                if str(column).startswith(_TRIALDEV_SAFETY_EVENT_PREFIX)
                and str(column).endswith(_TRIALDEV_SAFETY_EVENT_SUFFIX)
            )
            analysis_columns = tuple(
                column
                for column in frame.columns
                if column in _TRIALDEV_REQUIRED_COLUMNS
                or str(column).startswith(_TRIALDEV_EFFICACY_EVENT_PREFIX)
                or (
                    str(column).startswith(_TRIALDEV_SAFETY_EVENT_PREFIX)
                    and str(column).endswith(("_EVENT_T", "_EVENT_E", "_SERIOUS"))
                )
                or str(column) == "ANY_SERIOUS_AE_E"
                or str(column)
                in {"DISCONTINUATION_E", "DISCONTINUATION_T", "LTFU_E", "LTFU_T"}
            )
            discontinuation_count = (
                int(
                    pd.to_numeric(frame["DISCONTINUATION_E"], errors="coerce")
                    .fillna(0)
                    .eq(1)
                    .sum()
                )
                if "DISCONTINUATION_E" in frame
                else None
            )

            metrics: list[tuple[str, int | None, str]] = [
                (
                    "participant_count",
                    int(frame["USUBJID"].astype("string").nunique()),
                    "unique observational-extract USUBJID",
                ),
                (
                    "treatment_arm_count",
                    int(frame["TREATMENT"].astype("string").nunique()),
                    "unique observational treatment",
                ),
                (
                    "efficacy_event_count",
                    event_total(frame, efficacy_columns),
                    "sum of released EFF_*_E event indicators across endpoints",
                ),
                (
                    "safety_event_count",
                    event_total(frame, safety_columns),
                    "sum of released AE_*_EVENT_E indicators across event families",
                ),
                (
                    "serious_safety_event_count",
                    (
                        int(
                            pd.to_numeric(frame["ANY_SERIOUS_AE_E"], errors="coerce")
                            .fillna(0)
                            .eq(1)
                            .sum()
                        )
                        if "ANY_SERIOUS_AE_E" in frame
                        else None
                    ),
                    "participants with released ANY_SERIOUS_AE_E=1",
                ),
                (
                    "loss_to_follow_up_count",
                    (
                        int(
                            pd.to_numeric(frame["LTFU_E"], errors="coerce")
                            .fillna(0)
                            .eq(1)
                            .sum()
                        )
                        if "LTFU_E" in frame
                        else None
                    ),
                    "participants with released LTFU_E=1",
                ),
                (
                    "discontinuation_count",
                    discontinuation_count,
                    "participants with released DISCONTINUATION_E=1",
                ),
                (
                    "intercurrent_event_count",
                    discontinuation_count,
                    "realized discontinuation events; other TrialDev intercurrent-event types are not released",
                ),
                (
                    "rescue_count",
                    None,
                    "no realized rescue-event indicator is released; EARLY_RESCUE_RISK is a baseline predictor",
                ),
                (
                    "treatment_switch_count",
                    None,
                    "no realized treatment-switch event indicator is released",
                ),
                (
                    "competing_event_count",
                    None,
                    "the observational extract does not release a competing-event indicator",
                ),
                (
                    "missing_observation_count",
                    int(frame.loc[:, list(analysis_columns)].isna().sum().sum()),
                    "missing cells across released identifiers, treatment, efficacy, safety, discontinuation, and loss-to-follow-up fields",
                ),
            ]
            allocations = frame.groupby(
                frame["TREATMENT"].astype("string"), dropna=False
            ).size()
            if any(pd.isna(arm) for arm in allocations.index):
                raise ValueError(
                    f"candidate TrialDev treatment allocation is missing: {item.analysis_unit_id}"
                )
            metrics.extend(
                (
                    f"allocation_count::{arm}",
                    int(count),
                    f"observational rows assigned to released TREATMENT={arm}",
                )
                for arm, count in sorted(
                    ((str(arm), int(count)) for arm, count in allocations.items()),
                    key=lambda pair: pair[0],
                )
            )
            for metric, value, definition in metrics:
                rows.append(
                    {
                        "suite": "trialdev",
                        "analysis_unit_id": item.analysis_unit_id,
                        "matched_set_id": item.matched_set_id,
                        "metric": metric,
                        "value": value,
                        "status": "observed" if value is not None else "not_observed",
                        "definition": definition,
                    }
                )
    return rows


def _trialdev_phase_rows(
    verification_archive: Path,
    recovery: RecoverabilityReportV1,
) -> list[dict[str, object]]:
    status_by_unit = {row.unit_id: row.status for row in recovery.routes}
    with ZipFile(verification_archive) as archive:
        member = "phase_replay/records.jsonl"
        if member not in archive.namelist():
            raise FileNotFoundError(f"TrialDev verification archive lacks {member}")
        records = tuple(
            TrialDevPublicPhaseReplayRecordV1.model_validate_json(line)
            for line in archive.read(member).decode("utf-8").splitlines()
            if line.strip()
        )
    if not records:
        raise ValueError(
            "TrialDev verification archive contains no randomized-phase replay records"
        )
    rows: list[dict[str, object]] = []
    for record in records:
        unit_id = f"{record.scenario_id}:{record.request_checksum}"
        if unit_id not in status_by_unit:
            raise ValueError(
                f"TrialDev recovery omits randomized-phase record {unit_id}"
            )
        rows.append(
            {
                "scenario_id": record.scenario_id,
                "phase_id": record.phase_id,
                "checkpoint_id": record.request_checksum,
                "evaluation_lane": "randomized_phase_decision",
                "endpoint_id": record.endpoint_id,
                "policy_target_ids": "|".join(record.objective_ids),
                "treatment_discontinuation_strategy": record.treatment_discontinuation_strategy,
                "candidate_arm_ids": "|".join(record.candidate_drug_ids),
                "acceptable_action_ids": "|".join(record.acceptable_action_ids),
                "advance_action_ids": "|".join(record.advance_action_ids),
                "stop_action_ids": "|".join(record.stop_action_ids),
                "transition_action_count": len(
                    set(record.acceptable_action_ids)
                    | set(record.advance_action_ids)
                    | set(record.stop_action_ids)
                ),
                "control_comparison_count": len(record.candidate_decision_evidence),
                "public_safety_state": record.public_safety_state,
                "design_adequate": record.design_adequate,
                "recovery_status": status_by_unit[unit_id],
            }
        )
    return sorted(rows, key=lambda row: (str(row["scenario_id"]), str(row["phase_id"])))


def _trialdev_lane_rows(
    root: Path,
    report: GraderConcordanceReportV1,
) -> list[dict[str, object]]:
    """Load the complete TrialDev lane census after verifying its frozen projections."""

    grader_root = root / "recovery" / "grader_concordance"
    independent_trialeval = grader_root / "independent_trialeval_grade_records.jsonl"
    independent_trialdev = grader_root / "independent_trialdev_lane_records.jsonl"
    public_trialeval = grader_root / "public_trialeval_grade_records.jsonl"
    public_trialdev = grader_root / "public_trialdev_lane_records.jsonl"
    required = (
        independent_trialeval,
        independent_trialdev,
        public_trialeval,
        public_trialdev,
    )
    missing = tuple(path.name for path in required if not path.is_file())
    if missing:
        raise FileNotFoundError(
            f"grader concordance is missing frozen record projections: {missing!r}"
        )
    independent_digest = hashlib.sha256(
        independent_trialeval.read_bytes() + independent_trialdev.read_bytes()
    ).hexdigest()
    public_digest = hashlib.sha256(
        public_trialeval.read_bytes() + public_trialdev.read_bytes()
    ).hexdigest()
    if (
        independent_digest != report.independent_projection_sha256
        or public_digest != report.public_projection_sha256
    ):
        raise ValueError(
            "TrialDev lane records do not match the frozen grader projections"
        )
    independent = tuple(
        TrialDevLaneGradeV1.model_validate_json(line)
        for line in independent_trialdev.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    public = tuple(
        TrialDevLaneGradeV1.model_validate_json(line)
        for line in public_trialdev.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not independent or not public:
        raise ValueError("TrialDev lane census is empty")
    independent_by_target = {row.evaluation_target_checksum: row for row in independent}
    public_by_target = {row.evaluation_target_checksum: row for row in public}
    if (
        len(independent_by_target) != report.trialdev_required_count
        or len(public_by_target) != report.trialdev_required_count
        or independent_by_target != public_by_target
    ):
        raise ValueError(
            "TrialDev lane census does not reconcile to exact independent/public grader agreement"
        )
    rows: list[dict[str, object]] = [
        {
            "scenario_id": row.scenario_id,
            "phase_id": row.phase_id,
            "program_objective_id": row.program_objective_id,
            "phase_scoring_objective_id": row.phase_scoring_objective_id,
            "lane_id": row.lane_id,
            "evaluation_target_checksum": row.evaluation_target_checksum,
            "scoring_policy_id": row.scoring_policy_id,
            "recoverability_policy_id": row.recoverability_policy_id,
            "reference_target_ids": "|".join(row.reference_target_ids),
            "credit_eligible_target_ids": "|".join(row.credit_eligible_target_ids),
            "submitted_target_id": row.submitted_target_id,
            "score": row.score,
            "score_derivation": row.score_derivation,
            "derived_from_trajectory_metric": row.derived_from_trajectory_metric,
            "terminal_action_observed": row.terminal_action_observed,
            "terminal_asset_observed": row.terminal_asset_observed,
            "terminal_phase_observed": row.terminal_phase_observed,
            "status": row.status,
            "artifact_status": row.artifact_status,
            "failure_reason": row.failure_reason,
            "independent_public_match": True,
        }
        for row in independent_by_target.values()
    ]
    return sorted(
        rows,
        key=lambda row: (
            str(row["scenario_id"]),
            str(row["phase_id"]),
            str(row["program_objective_id"]),
            str(row["phase_scoring_objective_id"]),
            str(row["lane_id"]),
        ),
    )


def _context_rows(
    properties: tuple[_SimulationPropertyV1, ...],
    analysis_hashes: dict[str, str],
) -> list[dict[str, object]]:
    by_match: dict[str, list[_SimulationPropertyV1]] = defaultdict(list)
    for row in properties:
        if row.suite_id == "trialeval":
            by_match[row.matched_set_id].append(row)
    records = []
    for matched_set, rows in sorted(by_match.items()):
        seeds = {row.provenance.generation_seed_id for row in rows}
        estimands = {
            json.dumps(row.estimand.model_dump(mode="json"), sort_keys=True)
            for row in rows
        }
        contexts = {row.construction.context_id for row in rows}
        analysis_ready_hashes = {
            analysis_hashes[row.analysis_unit_id]
            for row in rows
            if row.construction.context_id in {"C1", "C2"}
        }
        raw_domain_hashes = {
            analysis_hashes[row.analysis_unit_id]
            for row in rows
            if row.construction.context_id in {"C3", "C4"}
        }
        passed = len(seeds) == len(estimands) == len(analysis_ready_hashes) == len(
            raw_domain_hashes
        ) == 1 and contexts == {"C1", "C2", "C3", "C4", "C5"}
        records.append(
            {
                "matched_set_id": matched_set,
                "context_count": len(contexts),
                "generation_seed_count": len(seeds),
                "estimand_count": len(estimands),
                "analysis_ready_hash_count_c1_c2": len(analysis_ready_hashes),
                "raw_domain_hash_count_c3_c4": len(raw_domain_hashes),
                "status": "pass" if passed else "fail",
            }
        )
    return records


def _ipcw_support_rows(
    report: SentinelAuditReportV1,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in report.records:
        if record.ipcw_support is None:
            continue
        for arm_id, support in record.ipcw_support.support_by_arm.items():
            rows.append(
                {
                    "sentinel_id": record.sentinel_id,
                    "task_id": record.task_id,
                    "support_status": record.ipcw_support.support_status,
                    "arm_id": arm_id,
                    "evaluated_event_time_count": support.evaluated_event_time_count,
                    "minimum_fitted_censoring_survival": support.minimum_fitted_censoring_survival,
                    "maximum_weight": support.maximum_weight,
                    "minimum_effective_sample_size_ratio": support.minimum_effective_sample_size_ratio,
                }
            )
    return rows


def _route_disposition_rows(
    reports: tuple[RecoverabilityReportV1, ...],
) -> list[dict[str, object]]:
    """Reconcile successful, failed, and scientifically non-estimable routes."""

    rows: list[dict[str, object]] = []
    for report in reports:
        failed = sum(route.status == "fail" for route in report.routes)
        non_estimable = sum(
            route.status == "pass" and route.result_kind in {"limitation", "abstention"}
            for route in report.routes
        )
        successful = sum(
            route.status == "pass"
            and route.result_kind not in {"limitation", "abstention"}
            for route in report.routes
        )
        attempted = len(report.routes)
        if (
            attempted != report.required_route_count
            or attempted != successful + failed + non_estimable
        ):
            raise ValueError(
                f"{report.suite} route dispositions do not reconcile to the required route census"
            )
        rows.append(
            {
                "suite": report.suite,
                "attempted": attempted,
                "successful": successful,
                "failed": failed,
                "non_estimable": non_estimable,
                "status": report.status,
            }
        )
    return rows


def _route_result_kind_rows(
    reports: tuple[RecoverabilityReportV1, ...],
) -> list[dict[str, object]]:
    """Summarize exact replay results without pooling comparison rules."""

    rows: list[dict[str, object]] = []
    for report in reports:
        groups: dict[tuple[str, str], list[RecoverabilityRouteV1]] = defaultdict(list)
        for route in report.routes:
            groups[(route.result_kind, route.comparison_rule)].append(route)
        for (result_kind, comparison_rule), routes in sorted(groups.items()):
            passed = sum(route.status == "pass" for route in routes)
            rows.append(
                {
                    "suite": report.suite,
                    "result_kind": result_kind,
                    "comparison_rule": comparison_rule,
                    "attempted": len(routes),
                    "passed": passed,
                    "failed": len(routes) - passed,
                    "maximum_absolute_difference": (
                        max(route.maximum_absolute_difference for route in routes)
                        if comparison_rule == "numeric_envelope"
                        else None
                    ),
                    "maximum_tolerance_ratio": (
                        max(
                            (route.difference_to_tolerance_ratio or 0.0)
                            for route in routes
                        )
                        if comparison_rule == "numeric_envelope"
                        else None
                    ),
                    "status": "pass" if passed == len(routes) else "fail",
                }
            )
    return rows


def _stress_rows(
    *,
    c5: C5IntegrityRecoveryReportV1,
    grader: GraderConcordanceReportV1,
    grader_behavior: GraderBehaviorReportV1,
    trialeval_sentinels: SentinelAuditReportV1,
    trialdev_sentinel_status: str,
    trialdev_sentinel_count: int,
    trialdev_reachability: TrialDevReachabilityReportV1,
    clean_room: _CleanRoomReportV1,
    clean_wheel: CandidateCleanWheelReplayV1,
) -> list[dict[str, object]]:
    """Return interpretable complete-census checks for adverse input classes."""

    rows = [
        {
            "stressor": f"grader_behavior::{case.behavior_class}",
            "attempted": case.submission_count,
            "passed": case.submission_count if case.status == "pass" else 0,
            "mismatches": case.mismatch_count,
            "crashes": 0,
            "status": case.status,
        }
        for case in grader_behavior.cases
    ]
    rows.extend(
        (
            {
                "stressor": "single_coordinate_trialeval_mutations",
                "attempted": grader.trialeval_mutation_required_count,
                "passed": grader.trialeval_mutation_independently_graded_count
                - grader.trialeval_mutation_behavior_failure_count,
                "mismatches": grader.trialeval_mutation_mismatch_count,
                "crashes": grader.trialeval_mutation_crashed_count,
                "status": (
                    "pass"
                    if grader.trialeval_mutation_mismatch_count
                    == grader.trialeval_mutation_behavior_failure_count
                    == grader.trialeval_mutation_crashed_count
                    == 0
                    else "fail"
                ),
            },
            {
                "stressor": "single_coordinate_trialdev_mutations",
                "attempted": grader.trialdev_mutation_required_count,
                "passed": grader.trialdev_mutation_independently_graded_count
                - grader.trialdev_mutation_behavior_failure_count,
                "mismatches": grader.trialdev_mutation_mismatch_count,
                "crashes": grader.trialdev_mutation_crashed_count,
                "status": (
                    "pass"
                    if grader.trialdev_mutation_mismatch_count
                    == grader.trialdev_mutation_behavior_failure_count
                    == grader.trialdev_mutation_crashed_count
                    == 0
                    else "fail"
                ),
            },
            {
                "stressor": "c5_transport_repair",
                "attempted": c5.required_item_count,
                "passed": c5.repaired_item_count,
                "mismatches": c5.mismatched_item_count + c5.unsupported_item_count,
                "crashes": 0,
                "status": c5.status,
            },
            {
                "stressor": "trialeval_high_risk_sentinels",
                "attempted": trialeval_sentinels.requested_sentinel_count,
                "passed": (
                    trialeval_sentinels.selected_sentinel_count
                    if trialeval_sentinels.status == "pass"
                    else 0
                ),
                "mismatches": len(trialeval_sentinels.findings),
                "crashes": 0,
                "status": trialeval_sentinels.status,
            },
            {
                "stressor": "trialdev_high_risk_sentinels",
                "attempted": trialdev_sentinel_count,
                "passed": (
                    trialdev_sentinel_count if trialdev_sentinel_status == "pass" else 0
                ),
                "mismatches": (
                    0 if trialdev_sentinel_status == "pass" else trialdev_sentinel_count
                ),
                "crashes": 0,
                "status": trialdev_sentinel_status,
            },
            {
                "stressor": "trialdev_programme_reachability",
                "attempted": trialdev_reachability.required_programme_count,
                "passed": (
                    trialdev_reachability.checked_programme_count
                    - trialdev_reachability.failed_programme_count
                ),
                "mismatches": trialdev_reachability.failed_programme_count
                + len(trialdev_reachability.global_findings),
                "crashes": 0,
                "status": trialdev_reachability.status,
            },
            {
                "stressor": "role_boundary_and_leakage",
                "attempted": sum(
                    surface.artifact_count for surface in clean_room.surfaces
                ),
                "passed": sum(
                    surface.artifact_count for surface in clean_room.surfaces
                ),
                "mismatches": len(clean_room.findings),
                "crashes": 0,
                "status": clean_room.status,
            },
            {
                "stressor": "isolated_public_wheel_replay",
                "attempted": len(clean_wheel.comparisons),
                "passed": sum(row.status == "pass" for row in clean_wheel.comparisons),
                "mismatches": sum(
                    row.structural_difference_count + row.nonnumeric_difference_count
                    for row in clean_wheel.comparisons
                ),
                "crashes": 0,
                "status": clean_wheel.status,
            },
        )
    )
    return rows


def _metric_row(
    metric_id: str,
    *,
    scope: str,
    numerator: int | float,
    denominator: int | float,
    unit: str,
    uncertainty: str,
    status: str,
) -> dict[str, object]:
    if denominator <= 0:
        raise ValueError(f"candidate result denominator must be positive: {metric_id}")
    return {
        "metric_id": metric_id,
        "scope": scope,
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if unit == "proportion" else numerator,
        "unit": unit,
        "uncertainty": uncertainty,
        "status": status,
    }


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    return float(value)


def _candidate_result_rows(
    *,
    property_count: int,
    route_dispositions: list[dict[str, object]],
    route_result_kinds: list[dict[str, object]],
    contexts: list[dict[str, object]],
    ipcw_support: list[dict[str, object]],
    stressors: list[dict[str, object]],
    grader: GraderConcordanceReportV1,
    grader_behavior: GraderBehaviorReportV1,
    c5: C5IntegrityRecoveryReportV1,
    clean_wheel: CandidateCleanWheelReplayV1,
) -> list[dict[str, object]]:
    """Build the public headline result table from exact candidate censuses."""

    exact = "none_complete_census"
    rows = [
        _metric_row(
            "candidate.analysis_units",
            scope="all",
            numerator=property_count,
            denominator=property_count,
            unit="analysis_units",
            uncertainty=exact,
            status="pass",
        ),
        _metric_row(
            "candidate.grader_concordance",
            scope="all",
            numerator=grader.required_count - grader.mismatch_count,
            denominator=grader.required_count,
            unit="proportion",
            uncertainty=exact,
            status="pass" if grader.passed else "fail",
        ),
        _metric_row(
            "candidate.raw_projection_concordance",
            scope="trialeval",
            numerator=grader.raw_projection_required_count
            - grader.raw_projection_mismatch_count,
            denominator=grader.raw_projection_required_count,
            unit="proportion",
            uncertainty=exact,
            status="pass" if grader.raw_projection_mismatch_count == 0 else "fail",
        ),
        _metric_row(
            "candidate.grader_behavior_classes",
            scope="trialeval",
            numerator=grader_behavior.passed_class_count,
            denominator=grader_behavior.required_class_count,
            unit="proportion",
            uncertainty=exact,
            status=grader_behavior.status,
        ),
        _metric_row(
            "candidate.c5_transport_recovery",
            scope="trialeval",
            numerator=c5.repaired_item_count,
            denominator=c5.required_item_count,
            unit="proportion",
            uncertainty=exact,
            status=c5.status,
        ),
        _metric_row(
            "candidate.context_invariance",
            scope="trialeval",
            numerator=sum(row["status"] == "pass" for row in contexts),
            denominator=len(contexts),
            unit="proportion",
            uncertainty=exact,
            status=(
                "pass" if all(row["status"] == "pass" for row in contexts) else "fail"
            ),
        ),
        _metric_row(
            "candidate.ipcw_support_records",
            scope="trialeval",
            numerator=len(ipcw_support),
            denominator=len(ipcw_support),
            unit="arm_diagnostics",
            uncertainty=exact,
            status="pass",
        ),
        _metric_row(
            "candidate.stress_checks",
            scope="all",
            numerator=sum(
                _integer(row["passed"], label="stress passed count")
                for row in stressors
            ),
            denominator=sum(
                _integer(row["attempted"], label="stress attempted count")
                for row in stressors
            ),
            unit="proportion",
            uncertainty=exact,
            status=(
                "pass" if all(row["status"] == "pass" for row in stressors) else "fail"
            ),
        ),
        _metric_row(
            "candidate.clean_wheel_families",
            scope="all",
            numerator=sum(row.status == "pass" for row in clean_wheel.comparisons),
            denominator=len(clean_wheel.comparisons),
            unit="proportion",
            uncertainty="deterministic_comparison",
            status=clean_wheel.status,
        ),
    ]
    for disposition in route_dispositions:
        suite = str(disposition["suite"])
        attempted = _integer(
            disposition["attempted"], label=f"{suite} attempted routes"
        )
        rows.extend(
            _metric_row(
                f"candidate.{suite}_routes_{name}",
                scope=suite,
                numerator=_integer(disposition[name], label=f"{suite} {name} routes"),
                denominator=attempted,
                unit="proportion",
                uncertainty=exact,
                status=str(disposition["status"]),
            )
            for name in ("successful", "non_estimable", "failed")
        )
    for result in route_result_kinds:
        suite = str(result["suite"])
        result_kind = str(result["result_kind"]).replace(" ", "_")
        rows.append(
            _metric_row(
                f"candidate.{suite}_result_kind_{result_kind}",
                scope=suite,
                numerator=_integer(
                    result["passed"], label=f"{suite} passed result-kind routes"
                ),
                denominator=_integer(
                    result["attempted"], label=f"{suite} attempted result-kind routes"
                ),
                unit="proportion",
                uncertainty=exact,
                status=str(result["status"]),
            )
        )
    return sorted(rows, key=lambda row: str(row["metric_id"]))


def _load_reports(
    root: Path,
) -> tuple[
    RecoverabilityReportV1,
    RecoverabilityReportV1,
    C5IntegrityRecoveryReportV1,
    GraderConcordanceReportV1,
    GraderBehaviorReportV1,
    _CleanRoomReportV1,
    CandidateCleanWheelReplayV1,
    TrialDevReachabilityReportV1,
]:
    trialeval = RecoverabilityReportV1.model_validate_json(
        (root / "recovery" / "trialeval" / "recoverability_report.json").read_text(
            encoding="utf-8"
        )
    )
    trialdev = RecoverabilityReportV1.model_validate_json(
        (root / "recovery" / "trialdev" / "recoverability_report.json").read_text(
            encoding="utf-8"
        )
    )
    c5 = C5IntegrityRecoveryReportV1.model_validate_json(
        (root / "recovery" / "trialeval" / "c5_integrity_recovery.json").read_text(
            encoding="utf-8"
        )
    )
    grader = GraderConcordanceReportV1.model_validate_json(
        (
            root / "recovery" / "grader_concordance" / "grader_concordance_report.json"
        ).read_text(encoding="utf-8")
    )
    grader_behavior = GraderBehaviorReportV1.model_validate_json(
        (
            root / "recovery" / "grader_behavior" / "grader_behavior_report.json"
        ).read_text(encoding="utf-8")
    )
    clean_room = _CleanRoomReportV1.model_validate_json(
        (
            root / "recovery" / "clean_room" / "clean_room_workflow_report.json"
        ).read_text(encoding="utf-8")
    )
    clean_wheel = CandidateCleanWheelReplayV1.model_validate_json(
        (root / "recovery" / "clean_wheel" / "clean_wheel_replay.json").read_text(
            encoding="utf-8"
        )
    )
    trialdev_reachability = TrialDevReachabilityReportV1.model_validate_json(
        (
            root
            / "recovery"
            / "trialdev_reachability"
            / "trialdev_reachability_report.json"
        ).read_text(encoding="utf-8")
    )
    if (
        trialeval.status != "pass"
        or trialdev.status != "pass"
        or c5.status != "pass"
        or not grader.passed
        or grader_behavior.status != "pass"
        or clean_room.status != "pass"
        or clean_room.findings
        or clean_wheel.status != "pass"
        or trialdev_reachability.status != "pass"
    ):
        raise ValueError(
            "candidate release contains a failed recovery, grader, role-boundary, or clean-wheel receipt"
        )
    return (
        trialeval,
        trialdev,
        c5,
        grader,
        grader_behavior,
        clean_room,
        clean_wheel,
        trialdev_reachability,
    )


def _artifact(root: Path, relative_path: str, media_type: str) -> ValidationArtifactV1:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(
            f"candidate validation artifact is missing: {relative_path}"
        )
    return ValidationArtifactV1(
        relative_path=relative_path,
        sha256=sha256_file(path),
        media_type=media_type,
    )


def _write_membership_manifest(root: Path) -> ExternalArtifactManifestV1:
    excluded = {"candidate_validation_bundle.json", "exact_membership_manifest.json"}
    paths = tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    )
    manifest = ExternalArtifactManifestV1(
        artifacts=tuple(
            ArtifactDigestV1(
                relative_path=path.relative_to(root).as_posix(),
                sha256=sha256_file(path),
                byte_size=path.stat().st_size,
            )
            for path in paths
        )
    )
    write_model(root / "exact_membership_manifest.json", manifest)
    return manifest


def _write_analysis(
    *,
    release_root: Path,
    output_root: Path,
    verifier_lock: Path,
    absolute_tolerance: float,
) -> CandidateValidationBundleV1:
    manifest = _ReleaseManifestV1.model_validate_json(
        (release_root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8")
    )
    seed_tree = _SeedTreeV1.model_validate_json(
        (release_root / "provenance" / "release_seed_tree.json").read_text(
            encoding="utf-8"
        )
    )
    census = _MaterializationCensusV1.model_validate_json(
        (release_root / "provenance" / "materialization_census.json").read_text(
            encoding="utf-8"
        )
    )
    identity = _candidate_identity(release_root, manifest, seed_tree, census)
    properties_path = release_root / manifest.catalogue.path
    if sha256_file(properties_path) != manifest.catalogue.sha256:
        raise ValueError(
            "simulation-properties checksum disagrees with the release manifest"
        )
    properties = _read_jsonl(properties_path)
    _validate_properties(properties, identity)
    feasibility_path = release_root / manifest.reference_replay_feasibility.path
    if sha256_file(feasibility_path) != manifest.reference_replay_feasibility.sha256:
        raise ValueError(
            "reference-replay feasibility checksum disagrees with the release manifest"
        )
    feasibility = _ReferenceReplayFeasibilityRegistryV1.model_validate_json(
        feasibility_path.read_text(encoding="utf-8")
    )
    for receipt in feasibility.receipts:
        evidence_path = release_root / receipt.evidence_report_path
        if sha256_file(evidence_path) != receipt.evidence_report_sha256:
            raise ValueError(
                f"{receipt.suite_id} feasibility evidence checksum disagrees"
            )
    (
        trialeval,
        trialdev,
        c5,
        grader,
        grader_behavior,
        clean_room,
        clean_wheel,
        trialdev_reachability,
    ) = _load_reports(release_root)
    if grader.release_id != identity.release_id:
        raise ValueError("grader concordance identifies another release")

    role = {
        (row.suite, row.role): release_root / row.relative_path
        for row in identity.role_archives
    }
    independently_replayed_reachability = audit_trialdev_reachability(
        participant_release=role[("trialdev", "participant")],
        evaluator_release=role[("trialdev", "evaluator")],
    )
    if independently_replayed_reachability != trialdev_reachability:
        raise ValueError(
            "TrialDev reachability report disagrees with the released role archives"
        )
    _validate_candidate_release_statistics(
        root=release_root,
        manifest=manifest,
        properties=properties,
        trialeval_verification=role[("trialeval", "verification")],
        trialdev_verification=role[("trialdev", "verification")],
    )
    trialeval_sentinels = audit_trialeval_sentinels(
        evaluator_zip=role[("trialeval", "evaluator")],
        participant_zip=role[("trialeval", "participant")],
    )
    with ZipFile(role[("trialdev", "verification")]) as archive:
        trialdev_inventory = (
            TrialDevScientificConstructionInventoryV1.model_validate_json(
                archive.read("scientific_construction_inventory.json")
            )
        )
    trialdev_sentinels = audit_trialdev_sentinels(
        inventory=trialdev_inventory,
        recoverability=trialdev,
    )
    if trialeval_sentinels.status != "pass" or trialdev_sentinels.status != "pass":
        raise ValueError("candidate sentinel audit failed")

    write_model(output_root / "candidate_identity.json", identity)
    write_model(output_root / "provenance" / "release_seed_tree.json", seed_tree)
    write_model(output_root / "provenance" / "materialization_census.json", census)
    reproducibility_root = output_root / "reproducibility"
    write_model(reproducibility_root / "clean_wheel_replay.json", clean_wheel)
    for name in (
        "installation_constraints.txt",
        "installed_environment.json",
        "transcript.json",
    ):
        source = release_root / "recovery" / "clean_wheel" / name
        if not source.is_file():
            raise FileNotFoundError(f"clean-wheel replay is missing {name}")
        reproducibility_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, reproducibility_root / name)
    shutil.copytree(
        release_root / "recovery" / "clean_wheel" / "import_audits",
        reproducibility_root / "import_audits",
    )
    write_model(output_root / "sentinels" / "trialeval.json", trialeval_sentinels)
    write_model(output_root / "sentinels" / "trialdev.json", trialdev_sentinels)
    write_model(output_root / "trialdev" / "reachability.json", trialdev_reachability)
    construction = _construction_rows(properties)
    observed = _observed_rows(properties)
    trialeval_characteristics, analysis_hashes = _trialeval_characteristics(
        role[("trialeval", "participant")],
        properties,
    )
    trialdev_characteristics = _trialdev_characteristics(
        role[("trialdev", "participant")],
        properties,
    )
    characteristics = [*trialeval_characteristics, *trialdev_characteristics]
    stratified_characteristics = _stratified_characteristic_rows(
        properties,
        characteristics,
    )
    ipcw_support = _ipcw_support_rows(trialeval_sentinels)
    if not ipcw_support:
        raise ValueError(
            "candidate sentinel audit emitted no independent IPCW support diagnostics"
        )
    analysis_units = _analysis_unit_rows(properties, characteristics)
    contexts = _context_rows(properties, analysis_hashes)
    trialeval_routes = [row.model_dump(mode="json") for row in trialeval.routes]
    trialdev_routes = [row.model_dump(mode="json") for row in trialdev.routes]
    route_dispositions = _route_disposition_rows((trialeval, trialdev))
    route_result_kinds = _route_result_kind_rows((trialeval, trialdev))
    trialdev_phases = _trialdev_phase_rows(
        role[("trialdev", "verification")],
        trialdev,
    )
    trialdev_lanes = _trialdev_lane_rows(release_root, grader)
    stressors = _stress_rows(
        c5=c5,
        grader=grader,
        grader_behavior=grader_behavior,
        trialeval_sentinels=trialeval_sentinels,
        trialdev_sentinel_status=trialdev_sentinels.status,
        trialdev_sentinel_count=trialdev_sentinels.requested_sentinel_count,
        trialdev_reachability=trialdev_reachability,
        clean_room=clean_room,
        clean_wheel=clean_wheel,
    )
    _write_csv(
        output_root / "tables" / "construction_census.csv",
        construction,
        ("suite", "dimension", "label", "count"),
    )
    _write_csv(
        output_root / "tables" / "role_archives.csv",
        [row.model_dump(mode="json") for row in identity.role_archives],
        ("suite", "role", "relative_path", "sha256"),
    )
    _write_csv(
        output_root / "tables" / "public_wheels.csv",
        [row.model_dump(mode="json") for row in identity.public_wheels],
        ("package", "relative_path", "sha256"),
    )
    _write_csv(
        output_root / "tables" / "observed_information.csv",
        observed,
        (
            "suite",
            "analysis_unit_id",
            "matched_set_id",
            "metric",
            "value",
            "absence_reason",
        ),
    )
    _write_csv(
        output_root / "tables" / "generated_characteristics.csv",
        characteristics,
        (
            "suite",
            "analysis_unit_id",
            "matched_set_id",
            "metric",
            "value",
            "status",
            "definition",
        ),
    )
    _write_csv(
        output_root / "tables" / "generated_characteristics_stratified.csv",
        stratified_characteristics,
        (
            "suite",
            "stratifier",
            "stratum",
            "metric",
            "analysis_unit_count",
            "independence_unit_count",
            "observed_count",
            "not_observed_count",
            "minimum",
            "median",
            "maximum",
        ),
    )
    _write_csv(
        output_root / "tables" / "trialeval_ipcw_support.csv",
        ipcw_support,
        (
            "sentinel_id",
            "task_id",
            "support_status",
            "arm_id",
            "evaluated_event_time_count",
            "minimum_fitted_censoring_survival",
            "maximum_weight",
            "minimum_effective_sample_size_ratio",
        ),
    )
    _write_csv(
        output_root / "tables" / "analysis_unit_inventory.csv",
        analysis_units,
        tuple(analysis_units[0]),
    )
    _write_csv(
        output_root / "tables" / "trialeval_recovery.csv",
        trialeval_routes,
        tuple(trialeval.routes[0].model_fields),
    )
    _write_csv(
        output_root / "tables" / "trialdev_recovery.csv",
        trialdev_routes,
        tuple(trialdev.routes[0].model_fields),
    )
    _write_csv(
        output_root / "tables" / "route_disposition_census.csv",
        route_dispositions,
        ("suite", "attempted", "successful", "failed", "non_estimable", "status"),
    )
    _write_csv(
        output_root / "tables" / "route_result_kind_census.csv",
        route_result_kinds,
        (
            "suite",
            "result_kind",
            "comparison_rule",
            "attempted",
            "passed",
            "failed",
            "maximum_absolute_difference",
            "maximum_tolerance_ratio",
            "status",
        ),
    )
    _write_csv(
        output_root / "tables" / "trialdev_phase_recovery.csv",
        trialdev_phases,
        tuple(trialdev_phases[0]),
    )
    _write_csv(
        output_root / "tables" / "trialdev_lane_census.csv",
        trialdev_lanes,
        tuple(trialdev_lanes[0]),
    )
    reachability_rows = [
        {
            "program_id": row.program_id,
            "scenario_id": row.scenario_id,
            "objective_id": row.objective_id,
            "reachability_class": row.reachability_class,
            "credit_eligible_candidate_count": len(row.credit_eligible_candidate_ids),
            "fixed_replay_candidate_count": len(row.fixed_replay_candidate_ids),
            "fixed_replay_case_count": row.fixed_replay_case_count,
            "status": row.status,
            "findings": json.dumps(row.findings),
        }
        for row in trialdev_reachability.programmes
    ]
    _write_csv(
        output_root / "tables" / "trialdev_reachability.csv",
        reachability_rows,
        (
            "program_id",
            "scenario_id",
            "objective_id",
            "reachability_class",
            "credit_eligible_candidate_count",
            "fixed_replay_candidate_count",
            "fixed_replay_case_count",
            "status",
            "findings",
        ),
    )
    _write_csv(
        output_root / "tables" / "context_invariance.csv",
        contexts,
        (
            "matched_set_id",
            "context_count",
            "generation_seed_count",
            "estimand_count",
            "analysis_ready_hash_count_c1_c2",
            "raw_domain_hash_count_c3_c4",
            "status",
        ),
    )
    grader_rows: list[dict[str, object]] = [
        {
            "required": grader.required_count,
            "independently_graded": grader.independently_graded_count,
            "public_grader": grader.public_grader_count,
            "mismatches": grader.mismatch_count,
            "unsupported": grader.unsupported_count,
            "crashed": grader.crashed_count,
            "status": "pass" if grader.passed else "fail",
        }
    ]
    _write_csv(
        output_root / "tables" / "grader_concordance.csv",
        grader_rows,
        (
            "required",
            "independently_graded",
            "public_grader",
            "mismatches",
            "unsupported",
            "crashed",
            "status",
        ),
    )
    _write_csv(
        output_root / "tables" / "grader_disagreements.csv",
        [
            {"item_id": item_id, "disposition": "unexplained_mismatch"}
            for item_id in grader.mismatched_item_ids
        ],
        ("item_id", "disposition"),
    )
    _write_csv(
        output_root / "tables" / "grader_behavior.csv",
        [row.model_dump(mode="json") for row in grader_behavior.cases],
        tuple(GraderBehaviorCaseResultV1.model_fields),
    )
    reproducibility = [
        {
            "suite": "trialeval",
            "scheduled": trialeval.required_route_count,
            "replayed": trialeval.replayed_route_count,
            "failed": trialeval.failed_route_count,
            "maximum_absolute_difference": trialeval.maximum_absolute_difference,
            "status": trialeval.status,
        },
        {
            "suite": "trialdev",
            "scheduled": trialdev.required_route_count,
            "replayed": trialdev.replayed_route_count,
            "failed": trialdev.failed_route_count,
            "maximum_absolute_difference": trialdev.maximum_absolute_difference,
            "status": trialdev.status,
        },
        {
            "suite": "clean_wheel",
            "scheduled": sum(
                row.compared_numeric_value_count for row in clean_wheel.comparisons
            ),
            "replayed": sum(
                row.compared_numeric_value_count for row in clean_wheel.comparisons
            ),
            "failed": sum(
                row.structural_difference_count + row.nonnumeric_difference_count
                for row in clean_wheel.comparisons
            ),
            "maximum_absolute_difference": max(
                row.maximum_absolute_difference for row in clean_wheel.comparisons
            ),
            "status": clean_wheel.status,
        },
    ]
    _write_csv(
        output_root / "tables" / "reproducibility.csv",
        reproducibility,
        (
            "suite",
            "scheduled",
            "replayed",
            "failed",
            "maximum_absolute_difference",
            "status",
        ),
    )
    integrity_rows = [
        {
            "condition": "C5 exact transport duplication",
            "required": c5.required_item_count,
            "repaired": c5.repaired_item_count,
            "mismatched": c5.mismatched_item_count,
            "unsupported": c5.unsupported_item_count,
            "status": c5.status,
        }
    ]
    _write_csv(
        output_root / "tables" / "data_integrity.csv",
        integrity_rows,
        ("condition", "required", "repaired", "mismatched", "unsupported", "status"),
    )
    boundary_rows: list[dict[str, object]] = [
        {
            "role": surface.role,
            "artifact_count": surface.artifact_count,
            "finding_count": surface.finding_count,
            "status": "pass" if surface.finding_count == 0 else "fail",
        }
        for surface in clean_room.surfaces
    ]
    _write_csv(
        output_root / "tables" / "release_boundary.csv",
        boundary_rows,
        ("role", "artifact_count", "finding_count", "status"),
    )
    _write_csv(
        output_root / "tables" / "stress_census.csv",
        stressors,
        ("stressor", "attempted", "passed", "mismatches", "crashes", "status"),
    )

    ipcw_status_counts = Counter(
        str(record.ipcw_support.support_status)
        for record in trialeval_sentinels.records
        if record.ipcw_support is not None
    )
    observed_characteristic_count = sum(
        row["status"] == "observed" for row in characteristics
    )
    unavailable_characteristic_count = (
        len(characteristics) - observed_characteristic_count
    )
    support_figure: list[dict[str, object]] = [
        {
            "label": "Observed generated-data quantities",
            "value": observed_characteristic_count / len(characteristics),
            "display": f"{observed_characteristic_count}/{len(characteristics)}",
            "series": "observed",
        },
        {
            "label": "Explicitly unavailable quantities",
            "value": unavailable_characteristic_count / len(characteristics),
            "display": f"{unavailable_characteristic_count}/{len(characteristics)}",
            "series": "non_estimable",
        },
        {
            "label": "TrialEval high-risk sentinels",
            "value": trialeval_sentinels.selected_sentinel_count
            / trialeval_sentinels.requested_sentinel_count,
            "display": (
                f"{trialeval_sentinels.selected_sentinel_count}/"
                f"{trialeval_sentinels.requested_sentinel_count}"
            ),
            "series": trialeval_sentinels.status,
        },
        {
            "label": "TrialDev high-risk sentinels",
            "value": len(trialdev_sentinels.records)
            / trialdev_sentinels.requested_sentinel_count,
            "display": f"{len(trialdev_sentinels.records)}/{trialdev_sentinels.requested_sentinel_count}",
            "series": trialdev_sentinels.status,
        },
    ]
    support_figure.extend(
        {
            "label": f"IPCW sentinel check · {status.replace('_', ' ')}",
            "value": count / sum(ipcw_status_counts.values()),
            "display": f"{count}/{sum(ipcw_status_counts.values())}",
            "series": "pass" if status == "point_supported" else "non_estimable",
        }
        for status, count in sorted(ipcw_status_counts.items())
    )
    route_figure: list[dict[str, object]] = [
        {
            "label": f"{row['suite']} · {disposition}",
            "value": cast(int, row[disposition]) / cast(int, row["attempted"]),
            "display": f"{row[disposition]}/{row['attempted']}",
            "series": "pass" if disposition == "successful" else disposition,
        }
        for row in route_dispositions
        for disposition in ("successful", "non_estimable", "failed")
    ]
    numeric_routes = tuple(
        row
        for row in (*trialeval.routes, *trialdev.routes)
        if row.difference_to_tolerance_ratio is not None
    )
    concordance_groups: dict[tuple[str, str], list[RecoverabilityRouteV1]] = (
        defaultdict(list)
    )
    for row in numeric_routes:
        concordance_groups[(row.suite, row.estimator_family)].append(row)
    concordance_figure = []
    for (suite, family), rows in sorted(concordance_groups.items()):
        maximum_ratio = max(row.difference_to_tolerance_ratio or 0.0 for row in rows)
        concordance_figure.append(
            {
                "label": f"{suite} · {family.replace('_', ' ')}",
                "value": maximum_ratio,
                "display": f"max {maximum_ratio:.3g} × tolerance across {len(rows)} routes",
                "series": (
                    "pass" if all(row.status == "pass" for row in rows) else "fail"
                ),
            }
        )
    grader_figure = [
        {
            "label": label,
            "value": value,
            "display": display,
            "series": "pass",
        }
        for label, value, display in (
            (
                "Canonical submissions graded",
                grader.independently_graded_count / grader.required_count,
                f"{grader.independently_graded_count}/{grader.required_count}",
            ),
            (
                "Independent/public agreement",
                (grader.required_count - grader.mismatch_count) / grader.required_count,
                f"{grader.required_count - grader.mismatch_count}/{grader.required_count}",
            ),
            (
                "Behavior classes passed",
                grader_behavior.passed_class_count
                / grader_behavior.required_class_count,
                f"{grader_behavior.passed_class_count}/{grader_behavior.required_class_count}",
            ),
        )
    ]
    context_figure = [
        {
            "label": label,
            "value": value,
            "display": str(value),
            "series": "pass",
        }
        for label, value in (
            ("matched base trials", len(contexts)),
            (
                "complete five-context panels",
                sum(row["status"] == "pass" for row in contexts),
            ),
            (
                "seed-invariant panels",
                sum(row["generation_seed_count"] == 1 for row in contexts),
            ),
            (
                "estimand-invariant panels",
                sum(row["estimand_count"] == 1 for row in contexts),
            ),
            (
                "analysis-ready invariant C1/C2 panels",
                sum(row["analysis_ready_hash_count_c1_c2"] == 1 for row in contexts),
            ),
            (
                "raw-domain invariant C3/C4 panels",
                sum(row["raw_domain_hash_count_c3_c4"] == 1 for row in contexts),
            ),
        )
    ]
    reproducibility_figure = [
        {
            "label": f"{row['suite']} replayed routes",
            "value": cast(int, row["replayed"]) / cast(int, row["scheduled"]),
            "display": f"{row['replayed']}/{row['scheduled']}",
            "series": row["status"],
        }
        for row in reproducibility
    ]
    figures = tuple(
        sorted(
            (
                _write_figure(
                    output_root,
                    figure_id="candidate.analysis_support",
                    title="Analysis support",
                    question="Do the released data expose the quantities and high-risk cases needed for analysis?",
                    independent_unit="released analysis unit",
                    estimand="Observed or passing proportion within each declared denominator",
                    comparator="complete analysis and sentinel census",
                    uncertainty="none; complete finite census",
                    rows=support_figure,
                    interpretation=(
                        "Observed quantities and explicit absence states reconcile to the full generated-data table.",
                        "High-risk sentinel cases and IPCW support states are retained in their own denominators.",
                    ),
                ),
                _write_figure(
                    output_root,
                    figure_id="candidate.route_recoverability",
                    title="Route recoverability",
                    question=(
                        "Can every C1-C5 TrialEval route and every "
                        "TrialDev recovery route be independently reconstructed?"
                    ),
                    independent_unit="route within released item or scenario",
                    estimand="Share of scheduled routes",
                    comparator="scheduled route census",
                    uncertainty="none; complete finite census",
                    rows=route_figure,
                    interpretation=(
                        f"{trialeval.replayed_route_count} TrialEval and "
                        f"{trialdev.replayed_route_count} TrialDev routes were replayed.",
                    ),
                ),
                _write_figure(
                    output_root,
                    figure_id="candidate.estimate_concordance",
                    title="Estimate concordance",
                    question="How large is independent numerical disagreement relative to each declared tolerance?",
                    independent_unit="numeric route within released analysis unit",
                    estimand="Maximum absolute difference divided by the route-specific tolerance",
                    comparator="one declared tolerance",
                    uncertainty="deterministic comparison; no sampling interval",
                    rows=concordance_figure,
                    interpretation=(
                        "Displayed values are maximum tolerance ratios by estimator family; "
                        "categorical routes use exact membership.",
                    ),
                ),
                _write_figure(
                    output_root,
                    figure_id="candidate.grader_concordance",
                    title="Grader concordance",
                    question="Does independent grading agree with the public grader over the canonical submission census?",
                    independent_unit="canonical submission",
                    estimand="Completed proportion within each declared denominator",
                    comparator="independent reconstruction versus public harness",
                    uncertainty="none; complete canonical census",
                    rows=grader_figure,
                    interpretation=(
                        f"The canonical accepted census contains {grader.required_count} submissions.",
                        "Six complete-census behavioral variants cover accepted, defensible alternative, rejected, "
                        "abstaining, malformed, and qualified non-identification responses.",
                    ),
                ),
                _write_figure(
                    output_root,
                    figure_id="candidate.context_invariance",
                    title="Context invariance",
                    question="Do the five information contexts preserve the generated question and release seed?",
                    independent_unit="matched base trial",
                    estimand="Exact matched-base-trial count",
                    comparator="all 100 matched base trials",
                    uncertainty="none; complete paired census",
                    rows=context_figure,
                    interpretation=(
                        "All context panels are paired by their prospective base-trial identity.",
                    ),
                ),
                _write_figure(
                    output_root,
                    figure_id="candidate.reproducibility",
                    title="Released-byte reproducibility",
                    question=(
                        "Do independent calculations reproduce the complete declared "
                        "analysis-ready TrialEval and TrialDev recovery censuses?"
                    ),
                    independent_unit="score-bearing route",
                    estimand="Share of declared values or routes reproduced",
                    comparator="released evaluator references under declared tolerances",
                    uncertainty="deterministic comparison; no sampling interval",
                    rows=reproducibility_figure,
                    interpretation=(
                        f"Maximum TrialEval disagreement was {trialeval.maximum_absolute_difference:.6g}; "
                        f"maximum TrialDev disagreement was {trialdev.maximum_absolute_difference:.6g}.",
                    ),
                ),
            ),
            key=lambda figure: figure.figure_id,
        )
    )

    result_rows = _candidate_result_rows(
        property_count=len(properties),
        route_dispositions=route_dispositions,
        route_result_kinds=route_result_kinds,
        contexts=contexts,
        ipcw_support=ipcw_support,
        stressors=stressors,
        grader=grader,
        grader_behavior=grader_behavior,
        c5=c5,
        clean_wheel=clean_wheel,
    )
    _write_csv(
        output_root / "RESULTS.csv",
        result_rows,
        (
            "metric_id",
            "scope",
            "numerator",
            "denominator",
            "value",
            "unit",
            "uncertainty",
            "status",
        ),
    )
    disposition_by_suite = {str(row["suite"]): row for row in route_dispositions}
    trialeval_disposition = disposition_by_suite["trialeval"]
    trialdev_disposition = disposition_by_suite["trialdev"]
    total_stress_attempted = sum(
        _integer(row["attempted"], label="stress attempted count") for row in stressors
    )
    total_stress_passed = sum(
        _integer(row["passed"], label="stress passed count") for row in stressors
    )
    context_passed = sum(row["status"] == "pass" for row in contexts)
    ipcw_item_count = len({str(row["task_id"]) for row in ipcw_support})
    ipcw_item_label = "item" if ipcw_item_count == 1 else "items"
    clean_numeric_count = sum(
        row.compared_numeric_value_count for row in clean_wheel.comparisons
    )
    clean_maximum_difference = max(
        row.maximum_absolute_difference for row in clean_wheel.comparisons
    )
    result_kind_table = _markdown_table(
        ("Suite", "Result", "Rule", "Passed", "Attempted", "Maximum tolerance ratio"),
        [
            (
                row["suite"],
                str(row["result_kind"]).replace("_", " "),
                str(row["comparison_rule"]).replace("_", " "),
                row["passed"],
                row["attempted"],
                (
                    "not applicable"
                    if row["maximum_tolerance_ratio"] is None
                    else f"{_number(row['maximum_tolerance_ratio'], label='maximum tolerance ratio'):.3g}"
                ),
            )
            for row in route_result_kinds
        ],
    )
    stress_table = _markdown_table(
        ("Check", "Passed", "Attempted", "Mismatches", "Crashes"),
        [
            (
                str(row["stressor"]).replace("_", " "),
                row["passed"],
                row["attempted"],
                row["mismatches"],
                row["crashes"],
            )
            for row in stressors
        ],
    )
    analysis_unit_table = _markdown_table(
        ("Question", "Unit", "Denominator", "Uncertainty", "Evidence"),
        [
            (
                "Which generated datasets are present?",
                "Released analysis unit",
                f"All {TRIALEVAL_ITEM_COUNT_V1 + TRIALDEV_SCENARIO_COUNT_V1} scheduled units",
                "Exact census",
                "[Construction census](tables/construction_census.csv)",
            ),
            (
                "Can TrialEval analyses be reconstructed?",
                "Analysis route within released item",
                f"All {trialeval_disposition['attempted']} scheduled routes",
                "Deterministic numerical comparison",
                "[TrialEval recovery](tables/trialeval_recovery.csv)",
            ),
            (
                "Can TrialDev analyses be reconstructed?",
                "Analysis route within released scenario",
                f"All {trialdev_disposition['attempted']} scheduled routes",
                "Deterministic numerical comparison",
                "[TrialDev recovery](tables/trialdev_recovery.csv)",
            ),
            (
                "Is randomized evidence available for every admissible TrialDev action?",
                "Programme objective",
                f"All {trialdev_reachability.required_programme_count} programmes",
                "Exact archive census",
                "[TrialDev reachability](tables/trialdev_reachability.csv)",
            ),
            (
                "Do the five contexts preserve the same trial and estimand?",
                "Matched base trial",
                f"All {len(contexts)} five-context panels",
                "Exact identity comparison",
                "[Context comparison](tables/context_invariance.csv)",
            ),
            (
                "Does an isolated installation reproduce the released results?",
                "Replayed result field",
                f"All {clean_numeric_count} numeric comparisons",
                "Deterministic numerical comparison",
                "[Reproducibility](tables/reproducibility.csv)",
            ),
        ],
    )
    (output_root / "METHODS.md").write_text(
        "\n".join(
            (
                "# Benchmark verification methods",
                "",
                "## Candidate",
                "",
                "The verification unit is the immutable paired TrialEval and TrialDev release identified in "
                "`candidate_identity.json`. The analysis reads participant, evaluator, and verification archives "
                "as separate inputs and verifies their checksums before computing any result.",
                "",
                "## Reconstruction",
                "",
                "Every declared TrialEval and TrialDev route is reconstructed from released participant data. "
                "Analysis-ready routes are evaluated directly; raw-domain routes rebuild the analysis table; C5 "
                "routes first remove the declared exact duplicate and then use the same reconstruction. Numeric "
                "points, intervals, vectors, and tests are compared with their route-specific absolute tolerance. "
                "Categorical, set-valued, limitation, abstention, and decision results use exact code membership.",
                "",
                "Passing point, interval, vector, test, set-valued, and decision routes are classified as "
                "successful. Passing limitation and abstention routes are classified as scientifically "
                "non-estimable. Failed comparisons form the third mutually exclusive disposition.",
                "",
                "## Scoring",
                "",
                "Canonical submissions are graded by an independent implementation and by the installed public "
                "harness. The comparison covers raw route projection, numerical and categorical results, "
                "TrialDev decisions, and single-coordinate mutations of each scoring obligation. Six complete "
                "submission censuses exercise accepted, defensible-alternative, rejected, abstaining, malformed, "
                "and qualified non-identification behavior.",
                "",
                "TrialDev reachability is reconstructed from the participant task inventory, evaluator action "
                "sets, and fixed randomized cases. Each programme is classified as stop-only, optionally "
                "nominating, or nomination-required. Every credit-eligible candidate must have one fixed case "
                "for each randomized phase, while stop-only programmes must have none.",
                "",
                "## Context and support",
                "",
                "The five TrialEval contexts are paired by base trial. The check compares generation seed and "
                "estimand identity across all five contexts and analysis-data identity across C1-C4. C5 is "
                "assessed through the separate repair comparison. Generated-data summaries preserve the released "
                "analysis unit and distinguish observed quantities from explicit absence states.",
                "",
                "For the prespecified censoring sentinels, censoring survival, maximum inverse-probability weight, "
                "and effective sample-size ratio are recomputed by randomized arm at every evaluated event time. "
                "The support classification is taken from these released-data diagnostics.",
                "",
                "## Reproducibility",
                "",
                "The harness and verifier wheels are installed in an isolated environment with network access "
                "blocked and the repository absent from the import path. Six evidence families are replayed and "
                "compared structurally, categorically, and numerically. Archive inspection separately checks "
                "required role membership, unsafe members, participant-target leakage, private columns, and "
                "evaluator/verification separation.",
                "",
                "Counts over the released candidate are exact finite-census quantities and therefore have no "
                "sampling interval. Bias, coverage, power, calibration, and mechanism-recovery results use "
                "simulation world or source trial as their independent unit and retain the intervals reported in "
                "the linked simulation-validity evidence.",
                "",
            )
        ),
        encoding="utf-8",
    )
    (output_root / "SOURCES.md").write_text(
        "\n".join(
            (
                "# Sources",
                "",
                f"- Release identity: `{identity.release_id}`",
                f"- Source commit: `{identity.source_commit}`",
                f"- Release manifest SHA-256: `{identity.release_manifest_sha256}`",
                f"- Candidate identity checksum: `{identity.checksum}`",
                "- Inputs: immutable release manifest, simulation-properties catalogue, role archives, "
                "materialization census, recoverability receipts, C5 integrity receipt, and grader-concordance receipt.",
                "",
            )
        ),
        encoding="utf-8",
    )
    (output_root / "REPORT.md").write_text(
        "\n".join(
            (
                "# Benchmark validity",
                "",
                f"The release contains {TRIALEVAL_ITEM_COUNT_V1} TrialEval items arranged as "
                f"{TRIALEVAL_ITEM_COUNT_V1 // TRIALEVAL_CONTEXT_COUNT_V1} matched five-context base trials and "
                f"{TRIALDEV_SCENARIO_COUNT_V1} TrialDev scenarios. Independent replay recovered "
                f"{_integer(trialeval_disposition['successful'], label='TrialEval successful routes') + _integer(trialeval_disposition['non_estimable'], label='TrialEval non-estimable routes')} of "
                f"{trialeval_disposition['attempted']} TrialEval routes and "
                f"{_integer(trialdev_disposition['successful'], label='TrialDev successful routes') + _integer(trialdev_disposition['non_estimable'], label='TrialDev non-estimable routes')} of "
                f"{trialdev_disposition['attempted']} TrialDev routes, with no failed comparison.",
                "",
                "## Results",
                "",
                *_markdown_table(
                    ("Property", "Result", "Interpretation"),
                    [
                        (
                            "TrialEval reconstruction",
                            (
                                f"{trialeval_disposition['successful']} successful, "
                                f"{trialeval_disposition['non_estimable']} non-estimable, "
                                f"{trialeval_disposition['failed']} failed"
                            ),
                            f"{trialeval_disposition['attempted']} routes",
                        ),
                        (
                            "TrialDev reconstruction",
                            (
                                f"{trialdev_disposition['successful']} successful, "
                                f"{trialdev_disposition['non_estimable']} non-estimable, "
                                f"{trialdev_disposition['failed']} failed"
                            ),
                            f"{trialdev_disposition['attempted']} routes",
                        ),
                        (
                            "TrialDev programme reachability",
                            (
                                f"{trialdev_reachability.checked_programme_count - trialdev_reachability.failed_programme_count}/"
                                f"{trialdev_reachability.required_programme_count}"
                            ),
                            (
                                f"{trialdev_reachability.fixed_replay_case_count} fixed cases; "
                                f"{trialdev_reachability.expanded_programme_case_count} programme-case assignments"
                            ),
                        ),
                        (
                            "Independent/public grading",
                            f"{grader.required_count - grader.mismatch_count}/{grader.required_count}",
                            f"{grader.mismatch_count} mismatches",
                        ),
                        (
                            "Adverse-input checks",
                            f"{total_stress_passed}/{total_stress_attempted}",
                            "complete declared stress census",
                        ),
                        (
                            "C5 repair",
                            f"{c5.repaired_item_count}/{c5.required_item_count}",
                            "repaired data equal the paired C4 analysis data",
                        ),
                        (
                            "Context panels",
                            f"{context_passed}/{len(contexts)}",
                            "seed, estimand, and paired data representations preserved",
                        ),
                        (
                            "Isolated-wheel replay",
                            f"{len(clean_wheel.comparisons)}/{len(clean_wheel.comparisons)} families",
                            (
                                f"{clean_numeric_count} numeric values; maximum absolute difference "
                                f"{clean_maximum_difference:.3g}"
                            ),
                        ),
                    ],
                ),
                "",
                "The complete numerical result set is available in [RESULTS.csv](RESULTS.csv). Detailed "
                "reconstruction, support, stress, context, and reproducibility records are in "
                "[tables](tables/).",
                "",
                "## Reconstruction",
                "",
                "TrialEval replay spans direct analysis-ready routes, raw-domain reconstruction, and "
                "repair-then-reconstruction. TrialDev replay spans observational estimates, randomized phase "
                "analyses, and decision lanes. The result classes remain separate because numerical envelopes "
                "and categorical membership answer different comparison questions.",
                "",
                *result_kind_table,
                "",
                "[Route-level TrialEval data](tables/trialeval_recovery.csv) · "
                "[Route-level TrialDev data](tables/trialdev_recovery.csv) · "
                "[Disposition census](tables/route_disposition_census.csv)",
                "",
                "## Scoring behavior",
                "",
                f"The independent implementation and public grader agreed on all {grader.required_count} "
                f"canonical submissions, all {grader.raw_projection_required_count} raw route projections, "
                f"{grader.trialeval_mutation_required_count} TrialEval mutations, and "
                f"{grader.trialdev_mutation_required_count} TrialDev mutations. "
                f"All {grader_behavior.required_class_count} complete-census behavior classes produced their "
                "prespecified outcome.",
                "",
                *stress_table,
                "",
                "[Grader comparison](tables/grader_concordance.csv) · "
                "[Behavior cases](tables/grader_behavior.csv) · "
                "[Stress census](tables/stress_census.csv)",
                "",
                "## Context and analysis support",
                "",
                f"All {len(contexts)} matched base trials retained one generation seed and one estimand across "
                "C1-C5. Each trial retained one analysis-ready data identity across C1/C2 and one raw-domain "
                "identity across C3/C4; C5 repair is assessed separately against C4. Independent event-time IPCW "
                "diagnostics "
                f"covered {len(ipcw_support)} randomized-arm records from "
                f"{ipcw_item_count} high-risk {ipcw_item_label}. The released-data "
                "characteristics table reports sample size, allocation, endpoint information, censoring, "
                "missingness, intercurrent events, discontinuation, switching, rescue, safety, and observed "
                "auxiliary domains at the analysis-unit level.",
                "",
                "[Context comparison](tables/context_invariance.csv) · "
                "[IPCW diagnostics](tables/trialeval_ipcw_support.csv) · "
                "[Generated-data characteristics](tables/generated_characteristics.csv)",
                "",
                "## Analysis units",
                "",
                "Finite-release rows answer whether this candidate is complete, reconstructable, correctly "
                "scored, and reproducible. Simulation-world and source-trial analyses answer how the mechanisms "
                "behave across repeated samples. The two evidence roles are linked without pooling their "
                "denominators.",
                "",
                *analysis_unit_table,
                "",
                "## Released-byte reproducibility",
                "",
                f"The isolated public-wheel replay compared {clean_numeric_count} numeric values across six "
                f"evidence families. The maximum absolute difference was {clean_maximum_difference:.3g}; "
                "structural and nonnumeric differences were zero. The role-boundary analysis inspected "
                f"{sum(row.artifact_count for row in clean_room.surfaces)} released artifacts and found no "
                "unsafe archive member, target leakage, private participant column, or role-boundary violation.",
                "",
                "## Figures",
                "",
                *(
                    line
                    for figure in figures
                    for line in (
                        f"### {figure.title}",
                        "",
                        figure.scientific_question,
                        "",
                        f"![{figure.title}](figures/{figure.figure_id.replace('.', '_')}.svg)",
                        "",
                        (
                            f"Independent unit: {figure.independent_unit}. "
                            f"Estimand or quantity: {figure.estimand}. "
                            f"Comparator: {figure.comparator}. "
                            f"Uncertainty: {figure.uncertainty}."
                        ),
                        "",
                        *(f"- {result}" for result in figure.interpretation),
                        "",
                        (
                            f"[Data](figures/{figure.figure_id.replace('.', '_')}.csv) · "
                            f"[PDF](figures/{figure.figure_id.replace('.', '_')}.pdf) · "
                            f"[preview](figures/{figure.figure_id.replace('.', '_')}.png)"
                        ),
                        "",
                    )
                ),
                "",
            )
        ),
        encoding="utf-8",
    )
    _write_membership_manifest(output_root)
    payload = {
        "schema_id": "trialagentbench.candidate_validation_bundle/v1",
        "candidate": identity.model_dump(mode="json"),
        "verifier_lock_sha256": sha256_file(verifier_lock),
        "figures": [figure.model_dump(mode="json") for figure in figures],
        "methods": _artifact(output_root, "METHODS.md", "text/markdown").model_dump(
            mode="json"
        ),
        "report": _artifact(output_root, "REPORT.md", "text/markdown").model_dump(
            mode="json"
        ),
        "results": _artifact(output_root, "RESULTS.csv", "text/csv").model_dump(
            mode="json"
        ),
        "sources": _artifact(output_root, "SOURCES.md", "text/markdown").model_dump(
            mode="json"
        ),
        "exact_membership_manifest": _artifact(
            output_root,
            "exact_membership_manifest.json",
            "application/json",
        ).model_dump(mode="json"),
    }
    checksum = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
    bundle = CandidateValidationBundleV1(**payload, checksum=checksum)
    write_model(output_root / "candidate_validation_bundle.json", bundle)
    verify_candidate_validation_bundle(output_root, bundle)
    return bundle


def build_candidate_validation_bundle(
    *,
    config: CandidateAnalysisConfigV1,
) -> Path:
    """Build, verify, and atomically publish one candidate-analysis bundle."""

    release = config.release_root.resolve()
    if not release.is_dir():
        raise FileNotFoundError(f"candidate release root is missing: {release}")
    if not config.verifier_lock.resolve().is_file():
        raise FileNotFoundError(f"verifier lock is missing: {config.verifier_lock}")
    output = config.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace candidate analysis: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "candidate_validation"
        staged.mkdir()
        _write_analysis(
            release_root=release,
            output_root=staged,
            verifier_lock=config.verifier_lock.resolve(),
            absolute_tolerance=config.absolute_tolerance,
        )
        staged.replace(output)
    return output


__all__ = [
    "CandidateAnalysisConfigV1",
    "build_candidate_validation_bundle",
]
