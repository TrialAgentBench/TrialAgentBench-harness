"""TrialDevBench participant evidence roles."""

from __future__ import annotations

from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from trialagentbench_harness.io.checksums import canonical_payload_sha256

TRIALDEV_FIXED_OBSERVATIONAL_COLUMNS_V1: frozenset[str] = frozenset(
    {
        "USUBJID",
        "TREATMENT",
        "DISCONTINUATION_T",
        "DISCONTINUATION_E",
        "LTFU_T",
        "LTFU_E",
    }
)


class TrialDevObservationalFieldV1(BaseModel):
    """One exact field in the participant observational Parquet surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str = Field(min_length=1)
    arrow_type: str = Field(min_length=1)
    nullable: bool


class TrialDevPublicDataDictionaryV1(BaseModel):
    """Participant dictionary bound to the exact observational schema."""

    model_config = ConfigDict(extra="allow", frozen=True)

    version: Literal["v1"]
    scenario_id: str = Field(min_length=1)
    observational_schema: tuple[TrialDevObservationalFieldV1, ...] = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _schema_and_checksum_are_exact(self) -> TrialDevPublicDataDictionaryV1:
        names = [field.column for field in self.observational_schema]
        if len(names) != len(set(names)):
            raise ValueError("observational_schema columns must be unique")
        payload = self.model_dump(mode="json")
        checksum = str(payload.pop("checksum"))
        if canonical_payload_sha256(cast(JsonValue, payload)) != checksum:
            raise ValueError("public data dictionary checksum does not match its payload")
        return self


class TrialDevPublicMemberRoleV1(str, Enum):
    """Scientific role of one participant-visible TrialDev member."""

    SCENARIO_CONTEXT = "scenario_context"
    CATALOG = "catalog"
    DATA_DICTIONARY = "data_dictionary"
    DECISION_POLICY = "decision_policy"
    INTERFACE_CONTRACT = "interface_contract"
    OBSERVATIONAL_DATA = "observational_data"
    FIXED_TRAJECTORY = "fixed_trajectory"
    SUITE_MANIFEST = "suite_manifest"
    DISTRIBUTION_MANIFEST = "distribution_manifest"
    DOCUMENTATION = "documentation"


TRIALDEV_PUBLIC_DOCUMENTATION_MEMBERS: Final[frozenset[str]] = frozenset(
    {
        "docs/DATA_DICTIONARY.md",
        "docs/HARNESS_INTEGRATION.md",
        "docs/QUICKSTART.md",
        "docs/README.md",
        "docs/SCORING.md",
        "docs/SUBMISSION_FORMAT.md",
        "docs/examples/phase1_analysis_submission_shape.json",
        "docs/examples/phase1_decision_submission_shape.json",
        "docs/examples/request_phase1_shape.json",
        "docs/examples/request_phase1_smoke_s01.json",
        "docs/examples/request_phase2_after_advance_shape.json",
        "docs/examples/submission_shape.json",
    }
)


TRIALDEV_PUBLIC_FILE_ROLES: Final[MappingProxyType[str, TrialDevPublicMemberRoleV1]] = MappingProxyType(
    {
        "ae_taxonomy.json": TrialDevPublicMemberRoleV1.CATALOG,
        "candidate_drug_catalog.json": TrialDevPublicMemberRoleV1.CATALOG,
        "checkpoint_outcome_schema.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "clinical_narrative.json": TrialDevPublicMemberRoleV1.SCENARIO_CONTEXT,
        "data_dictionary.json": TrialDevPublicMemberRoleV1.DATA_DICTIONARY,
        "decision_charter.json": TrialDevPublicMemberRoleV1.DECISION_POLICY,
        "endpoint_catalog.json": TrialDevPublicMemberRoleV1.CATALOG,
        "eval_contract.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "objective_charter.json": TrialDevPublicMemberRoleV1.DECISION_POLICY,
        "observational_method_catalog.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "observational_extract.parquet": TrialDevPublicMemberRoleV1.OBSERVATIONAL_DATA,
        "phase_action_policy.json": TrialDevPublicMemberRoleV1.DECISION_POLICY,
        "phase_analysis_method_catalog.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "phase_decision_evidence_policy.json": TrialDevPublicMemberRoleV1.DECISION_POLICY,
        "phase_decision_schema.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "phase_design_policy.json": TrialDevPublicMemberRoleV1.DECISION_POLICY,
        "phase_design_frontiers.json": TrialDevPublicMemberRoleV1.DECISION_POLICY,
        "phase_module_catalog.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "policy_binding_schema.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "portfolio_action_selection_schema.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "portfolio_checkpoint_action_policy_schema.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "portfolio_programme_state_schema.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "program_loop_manifest.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "resource_schedule_schema.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "safety_decision_policy.json": TrialDevPublicMemberRoleV1.DECISION_POLICY,
        "single_asset_action_selection_schema.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "single_asset_checkpoint_action_policy_schema.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "single_asset_programme_state_schema.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "study_brief.md": TrialDevPublicMemberRoleV1.SCENARIO_CONTEXT,
        "trial_output_schema.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "trial_request_schema.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "variable_catalog.json": TrialDevPublicMemberRoleV1.CATALOG,
    }
)


def classify_trialdev_public_member(relative_path: str) -> TrialDevPublicMemberRoleV1:
    """Classify one flat public scenario member and reject unknown paths."""

    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative_path:
        raise ValueError("TrialDev public members must be normalized relative POSIX paths")
    if len(path.parts) != 1:
        raise ValueError(f"TrialDev public scenario members must be flat: {relative_path!r}")
    try:
        return TRIALDEV_PUBLIC_FILE_ROLES[path.name]
    except KeyError as exc:
        raise ValueError(f"Unknown TrialDev public scenario member: {relative_path!r}") from exc


def classify_trialdev_participant_archive_member(relative_path: str) -> TrialDevPublicMemberRoleV1:
    """Classify one participant archive member under a scenario root."""

    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative_path:
        raise ValueError("TrialDev participant members must be normalized relative POSIX paths")
    if path.parts == ("benchmark_suite_manifest.json",):
        return TrialDevPublicMemberRoleV1.SUITE_MANIFEST
    if path.parts == ("distribution_mode_participant_manifest.json",):
        return TrialDevPublicMemberRoleV1.DISTRIBUTION_MANIFEST
    if relative_path in TRIALDEV_PUBLIC_DOCUMENTATION_MEMBERS:
        return TrialDevPublicMemberRoleV1.DOCUMENTATION
    if len(path.parts) >= 2 and path.parts[0] == "fixed_trajectories":
        if path.parts[1] == "cases.jsonl" or path.parts[1] == "materialized":
            return TrialDevPublicMemberRoleV1.FIXED_TRAJECTORY
        raise ValueError(f"Unknown TrialDev fixed-trajectory member: {relative_path!r}")
    if len(path.parts) != 3 or not path.parts[0].startswith("scenario_"):
        raise ValueError(f"Unknown TrialDev participant archive member: {relative_path!r}")
    if path.parts[1] == "public":
        return classify_trialdev_public_member(path.parts[2])
    raise ValueError(f"Unknown TrialDev participant archive member: {relative_path!r}")


def required_trialdev_public_members() -> frozenset[str]:
    """Return the complete public evidence inventory required per scenario."""

    return frozenset(TRIALDEV_PUBLIC_FILE_ROLES)
