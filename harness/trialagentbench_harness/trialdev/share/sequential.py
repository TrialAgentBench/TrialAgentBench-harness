"""Sequential public validation contracts for clinical program trajectories."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.contracts.trialdev.programme import TrialDevProgrammeStateV1
from trialagentbench_harness.trialdev.share.hashing import compute_sha256_hex
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentRequestV1

PhasePolicyMode = Literal["required", "optional", "not_available"]
ProgramArchetype = Literal["asset_development"]
PhaseActionIdV1 = Literal[
    "nominate_for_early_study",
    "advance_to_proof_of_concept",
    "stop_development",
    "withhold_nomination",
    "advance_to_confirmation",
    "declare_success",
    "declare_failure",
    "declare_inconclusive",
]


__all__ = [
    "PhaseActionIdV1",
    "TrialDevelopmentAnalysisDiagnosticV1",
    "TrialDevelopmentCandidateUtilityEstimateV1",
    "TrialDevelopmentEffectEstimateV1",
    "TrialDevelopmentIdentificationEvidenceV1",
    "TrialDevelopmentPhaseActionPolicyV1",
    "TrialDevelopmentPhaseActionSpecV1",
    "TrialDevelopmentPhaseAnalysisSubmissionV1",
    "TrialDevelopmentPhaseDecisionSubmissionV1",
    "TrialDevelopmentObservationalReviewSubmissionV1",
    "TrialDevelopmentProgramLoopManifestV1",
    "TrialDevProgrammeStateV1",
    "TrialDevelopmentSafetyEstimateV1",
    "TrialDevelopmentTrialOutputManifestV1",
    "validate_design_request_file_v1",
    "validate_phase_action_policy_file_v1",
    "validate_phase_analysis_file_v1",
    "validate_phase_decision_against_policy_v1",
    "validate_phase_decision_file_v1",
    "validate_trial_output_bundle_v1",
]


class _ChecksummedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checksum: str | None = Field(default=None, min_length=64, max_length=64)


class TrialDevelopmentProgramLoopManifestV1(_ChecksummedModel):
    """Public state-machine manifest for one sequential clinical program."""

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    program_archetype: ProgramArchetype
    phase_order: tuple[str, ...] = Field(min_length=4)
    conditionally_materializable_phase_ids: tuple[str, ...] = Field(
        min_length=1,
        description="Randomized phases that can be materialized after supported prior progression.",
    )
    phase_policy_modes: dict[str, PhasePolicyMode] = Field(
        min_length=3,
        description="Execution requirement for each phase if the program reaches that phase.",
    )
    phase1_carryover_consequential: bool
    phase_policy_checksum: str | None = Field(default=None, min_length=64, max_length=64)
    decision_charter_checksum: str = Field(..., min_length=64, max_length=64)
    terminal_statuses: tuple[str, ...] = Field(min_length=2)
    public_state_summary_fields: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_manifest(self) -> TrialDevelopmentProgramLoopManifestV1:
        modes = {str(k): v for k, v in self.phase_policy_modes.items()}
        if self.phase_order != ("observational_review", "phase1", "phase2", "phase3"):
            raise ValueError("phase_order must declare the canonical sequential programme order.")
        if set(modes) != {"phase1", "phase2", "phase3"}:
            raise ValueError("phase_policy_modes must declare exactly phase1, phase2, and phase3.")
        for phase_id in ("phase1", "phase2"):
            if modes.get(phase_id) == "not_available":
                raise ValueError(f"{phase_id} cannot be marked not_available.")
        for phase_id in self.conditionally_materializable_phase_ids:
            if modes.get(str(phase_id), "optional") == "not_available":
                raise ValueError("conditionally_materializable_phase_ids cannot include a not_available phase.")
        policy_payload = {
            "scenario_id": str(self.scenario_id),
            "program_archetype": str(self.program_archetype),
            "phase_order": list(self.phase_order),
            "conditionally_materializable_phase_ids": list(self.conditionally_materializable_phase_ids),
            "phase_policy_modes": {str(k): str(v) for k, v in sorted(modes.items())},
            "phase1_carryover_consequential": bool(self.phase1_carryover_consequential),
        }
        expected_policy_checksum = compute_sha256_hex(policy_payload)
        if self.phase_policy_checksum is not None and self.phase_policy_checksum != expected_policy_checksum:
            raise ValueError("phase_policy_checksum does not match the declared phase policy.")
        object.__setattr__(self, "phase_policy_checksum", expected_policy_checksum)
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        expected_checksum = compute_sha256_hex(payload)
        if self.checksum is not None and self.checksum != expected_checksum:
            raise ValueError("Contract checksum does not match its canonical payload.")
        object.__setattr__(self, "checksum", expected_checksum)
        return self


class TrialDevelopmentPhaseActionSpecV1(BaseModel):
    """Legal action set for one phase in a public action policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase_id: str = Field(..., min_length=1)
    allowed_action_ids: tuple[PhaseActionIdV1, ...] = Field(..., min_length=1)
    stop_action_ids: tuple[PhaseActionIdV1, ...] = Field(..., min_length=1)
    advance_action_ids: tuple[PhaseActionIdV1, ...] = Field(..., min_length=1)
    terminal_action_ids: tuple[PhaseActionIdV1, ...] = Field(default_factory=tuple)
    requires_candidate_drug_id: tuple[PhaseActionIdV1, ...] = Field(default_factory=tuple)
    notes: str = Field("", max_length=1000)

    @model_validator(mode="after")
    def _validate_action_spec(self) -> TrialDevelopmentPhaseActionSpecV1:
        allowed = tuple(str(value) for value in self.allowed_action_ids)
        stop = tuple(str(value) for value in self.stop_action_ids)
        advance = tuple(str(value) for value in self.advance_action_ids)
        terminal = tuple(str(value) for value in self.terminal_action_ids)
        requires_candidate = tuple(str(value) for value in self.requires_candidate_drug_id)
        for name, values in (
            ("allowed_action_ids", allowed),
            ("stop_action_ids", stop),
            ("advance_action_ids", advance),
            ("terminal_action_ids", terminal),
            ("requires_candidate_drug_id", requires_candidate),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates.")
        if not set(terminal) <= set(allowed):
            raise ValueError("terminal_action_ids must be a subset of allowed_action_ids.")
        if not set(requires_candidate) <= set(allowed):
            raise ValueError("requires_candidate_drug_id must be a subset of allowed_action_ids.")
        if set(stop) & set(advance) or set(stop) | set(advance) != set(allowed):
            raise ValueError("stop_action_ids and advance_action_ids must partition allowed_action_ids.")
        phase = str(self.phase_id)
        if phase in {"observational_review", "phase1", "phase2"} and {
            "declare_success",
            "declare_failure",
            "declare_inconclusive",
        } & set(allowed):
            raise ValueError("Only phase3 may allow final program actions.")
        if phase == "phase2" and set(allowed) != {
            "advance_to_confirmation",
            "stop_development",
        }:
            raise ValueError("Phase2 must allow only continuation to phase3 or stopping.")
        return self


class TrialDevelopmentPhaseActionPolicyV1(_ChecksummedModel):
    """Public phase-action policy for one scenario."""

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    program_archetype: ProgramArchetype = Field("asset_development")
    phase_policy_checksum: str = Field(..., min_length=64, max_length=64)
    decision_charter_checksum: str = Field(..., min_length=64, max_length=64)
    action_specs: tuple[TrialDevelopmentPhaseActionSpecV1, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_policy(self) -> TrialDevelopmentPhaseActionPolicyV1:
        phase_ids = [str(spec.phase_id) for spec in self.action_specs]
        if len(set(phase_ids)) != len(phase_ids):
            raise ValueError("phase_action_policy contains duplicate phase_id values.")
        if "phase1" not in set(phase_ids) or "phase2" not in set(phase_ids):
            raise ValueError("phase_action_policy must include phase1 and phase2.")
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        expected_checksum = compute_sha256_hex(payload)
        if self.checksum is not None and self.checksum != expected_checksum:
            raise ValueError("Contract checksum does not match its canonical payload.")
        object.__setattr__(self, "checksum", expected_checksum)
        return self

    def action_spec(self, phase_id: str) -> TrialDevelopmentPhaseActionSpecV1:
        """Return the action specification for one phase."""
        for spec in self.action_specs:
            if str(spec.phase_id) == str(phase_id):
                return spec
        raise ValueError(f"phase_action_policy missing phase_id={phase_id!r}.")


class TrialDevelopmentTrialOutputManifestV1(_ChecksummedModel):
    """Manifest for participant-safe trial data returned after one phase."""

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    phase_id: str = Field(..., min_length=1)
    request_checksum: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 of the participant's prospective design proposal.",
    )
    evidence_request_checksum: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 of the released design that generated the fixed trial evidence.",
    )
    state_checksum: str | None = Field(default=None, min_length=64, max_length=64)
    table_files: tuple[str, ...] = Field(default=("participants.parquet", "endpoints.parquet", "safety.parquet"))
    metadata_files: tuple[str, ...] = Field(
        default=(
            "trial_metadata.json",
            "execution_summary.json",
            "phase_summary_public.json",
            "arm_mapping.json",
            "request.json",
            "agent_request.json",
        )
    )
    table_checksums: dict[str, str] = Field(default_factory=dict)
    n_participants: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _validate_manifest(self) -> TrialDevelopmentTrialOutputManifestV1:
        missing = set(self.table_files) - set(self.table_checksums)
        if missing:
            raise ValueError(f"Trial output manifest missing table checksums: {sorted(missing)!r}.")
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        payload["table_files"] = sorted(set(payload.get("table_files", [])))
        payload["metadata_files"] = sorted(set(payload.get("metadata_files", [])))
        payload["table_checksums"] = {str(k): str(v) for k, v in sorted(payload.get("table_checksums", {}).items())}
        expected_checksum = compute_sha256_hex(payload)
        if self.checksum is not None and self.checksum != expected_checksum:
            raise ValueError("Contract checksum does not match its canonical payload.")
        object.__setattr__(self, "checksum", expected_checksum)
        return self


class TrialDevelopmentAnalysisDiagnosticV1(BaseModel):
    """One quantitative diagnostic reported for a phase analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(..., min_length=1)
    metric_family: Literal["balance", "positivity", "temporal", "sensitivity", "calibration", "robustness"]
    primary_value: float
    endpoint_id: str | None = None


class TrialDevelopmentIdentificationEvidenceV1(BaseModel):
    """One public premise supporting an observational identification decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(..., min_length=1)
    premise_id: Literal[
        "measured_conditional_exchangeability",
        "practical_positivity",
        "structural_positivity",
        "method_estimability",
    ]
    premise_state: Literal["satisfied", "failed", "unresolved"]
    evidence_kind: Literal["factual_provenance", "empirical_diagnostic", "observed_failure"]
    public_artifact_path: str = Field(..., pattern=r"^public/[A-Za-z0-9_.\-/]+$")
    public_artifact_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    source_record_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Exact factor_id from observational_method_catalog.assignment_prognostic_factors for factual "
            "provenance, or exact method_route_id from observational_method_catalog.methods for an empirical "
            "method-support diagnostic."
        ),
    )
    interpretation: str = Field(..., min_length=1, max_length=2000)


class TrialDevelopmentEffectEstimateV1(BaseModel):
    """One identified treatment-effect result used as decision evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(..., min_length=1)
    method_route_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional catalog identifier for the executed phase method. "
            "The estimator, estimand, scale, horizon, and population fields are authoritative."
        ),
    )
    candidate_drug_id: str = Field(..., min_length=1)
    endpoint_id: str = Field(..., min_length=1)
    estimand_id: str = Field(..., min_length=1)
    estimator_id: str = Field(..., min_length=1)
    effect_scale_id: str = Field(..., min_length=1)
    orientation_id: str = Field(..., min_length=1)
    estimate: float
    lower: float
    upper: float
    confidence_level: float = Field(..., gt=0.0, lt=1.0)
    horizon_days: int | None = Field(default=None, ge=1)
    analysis_population: str = Field(..., min_length=1)
    source_artifact_checksums: dict[str, str] = Field(
        ...,
        min_length=1,
        description=(
            "Exact primary-effect SHA-256 map surfaced by phase materialization; "
            "copy every key and value without substitution."
        ),
    )
    diagnostic_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _validate_interval(self) -> TrialDevelopmentEffectEstimateV1:
        if float(self.lower) > float(self.upper):
            raise ValueError("effect-estimate lower bound must not exceed upper bound")
        if not float(self.lower) <= float(self.estimate) <= float(self.upper):
            raise ValueError("effect estimate must lie within its submitted confidence interval")
        if len(set(self.diagnostic_evidence_ids)) != len(self.diagnostic_evidence_ids):
            raise ValueError("diagnostic_evidence_ids must be unique")
        if any(not path or len(checksum) != 64 for path, checksum in self.source_artifact_checksums.items()):
            raise ValueError("source_artifact_checksums require non-empty paths and SHA-256 values")
        return self


class TrialDevelopmentSafetyEstimateV1(BaseModel):
    """One candidate-specific quantitative safety result used as evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(..., min_length=1)
    method_route_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional catalog identifier for the executed safety method. "
            "The submitted semantic method fields are authoritative."
        ),
    )
    candidate_drug_id: str = Field(..., min_length=1)
    estimator_id: str = Field(..., min_length=1)
    estimand_ids: tuple[str, ...] = Field(..., min_length=2, max_length=2)
    absolute_risk_scale_id: Literal["absolute_risk"]
    excess_risk_scale_id: Literal["risk_difference_treatment_minus_control"]
    orientation_id: Literal["absolute_risk_higher_is_worse_and_risk_difference_positive_is_treatment_harm"]
    horizon_days: int = Field(..., ge=1)
    analysis_population: str = Field(..., min_length=1)
    serious_ae_treatment_rate: float = Field(..., ge=0.0, le=1.0)
    serious_ae_treatment_lower: float = Field(..., ge=0.0, le=1.0)
    serious_ae_treatment_upper: float = Field(..., ge=0.0, le=1.0)
    serious_ae_control_rate: float = Field(..., ge=0.0, le=1.0)
    serious_ae_control_lower: float = Field(..., ge=0.0, le=1.0)
    serious_ae_control_upper: float = Field(..., ge=0.0, le=1.0)
    serious_ae_excess: float = Field(..., ge=-1.0, le=1.0)
    serious_ae_excess_lower: float = Field(..., ge=-1.0, le=1.0)
    serious_ae_excess_upper: float = Field(..., ge=-1.0, le=1.0)
    discontinuation_treatment_rate: float = Field(..., ge=0.0, le=1.0)
    discontinuation_treatment_lower: float = Field(..., ge=0.0, le=1.0)
    discontinuation_treatment_upper: float = Field(..., ge=0.0, le=1.0)
    discontinuation_control_rate: float = Field(..., ge=0.0, le=1.0)
    discontinuation_control_lower: float = Field(..., ge=0.0, le=1.0)
    discontinuation_control_upper: float = Field(..., ge=0.0, le=1.0)
    discontinuation_excess: float = Field(..., ge=-1.0, le=1.0)
    discontinuation_excess_lower: float = Field(..., ge=-1.0, le=1.0)
    discontinuation_excess_upper: float = Field(..., ge=-1.0, le=1.0)
    confidence_level: float = Field(..., gt=0.0, lt=1.0)
    source_artifact_checksums: dict[str, str] = Field(
        ...,
        min_length=1,
        description=(
            "Exact safety-bundle SHA-256 map surfaced by phase materialization; "
            "copy every key and value without substitution."
        ),
    )

    @model_validator(mode="after")
    def _validate_intervals(self) -> TrialDevelopmentSafetyEstimateV1:
        for label, estimate, lower, upper in (
            (
                "serious_ae_treatment",
                self.serious_ae_treatment_rate,
                self.serious_ae_treatment_lower,
                self.serious_ae_treatment_upper,
            ),
            (
                "serious_ae_control",
                self.serious_ae_control_rate,
                self.serious_ae_control_lower,
                self.serious_ae_control_upper,
            ),
            (
                "serious_ae_excess",
                self.serious_ae_excess,
                self.serious_ae_excess_lower,
                self.serious_ae_excess_upper,
            ),
            (
                "discontinuation_treatment",
                self.discontinuation_treatment_rate,
                self.discontinuation_treatment_lower,
                self.discontinuation_treatment_upper,
            ),
            (
                "discontinuation_control",
                self.discontinuation_control_rate,
                self.discontinuation_control_lower,
                self.discontinuation_control_upper,
            ),
            (
                "discontinuation_excess",
                self.discontinuation_excess,
                self.discontinuation_excess_lower,
                self.discontinuation_excess_upper,
            ),
        ):
            if float(lower) > float(upper) or not float(lower) <= float(estimate) <= float(upper):
                raise ValueError(f"{label} estimate must lie within an ordered interval")
        for label, treated, control, excess in (
            (
                "serious_ae",
                self.serious_ae_treatment_rate,
                self.serious_ae_control_rate,
                self.serious_ae_excess,
            ),
            (
                "discontinuation",
                self.discontinuation_treatment_rate,
                self.discontinuation_control_rate,
                self.discontinuation_excess,
            ),
        ):
            if not math.isclose(
                float(excess),
                float(treated) - float(control),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{label} excess must equal treated rate minus control rate")
        if len(set(self.estimand_ids)) != len(self.estimand_ids):
            raise ValueError("safety estimand_ids must be unique")
        if any(not path or len(checksum) != 64 for path, checksum in self.source_artifact_checksums.items()):
            raise ValueError("source_artifact_checksums require non-empty paths and SHA-256 values")
        return self


class TrialDevelopmentCandidateUtilityEstimateV1(BaseModel):
    """One observational candidate utility reconstructed from public evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(..., min_length=1)
    method_route_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional catalog identifier for the executed observational method. "
            "The estimator, utility scale, adjustment set, and uncertainty fields are authoritative."
        ),
    )
    candidate_drug_id: str = Field(..., min_length=1)
    objective_id: str = Field(..., min_length=1)
    estimator_id: str = Field(..., min_length=1)
    utility_unit: Literal["dimensionless_declared_net_benefit"] = "dimensionless_declared_net_benefit"
    estimate: float
    lower: float
    upper: float
    confidence_level: float = Field(..., gt=0.5, lt=1.0)
    analysis_covariate_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Baseline covariates used by the estimator. An empty set identifies an "
            "unadjusted descriptive analysis; it does not support a causal ranking."
        ),
    )
    diagnostic_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    source_artifact_checksums: dict[str, str] = Field(
        ...,
        min_length=1,
        description=(
            "Exact observational source checksum map declared by "
            "objective_charter.json; copy every key and value without substitution."
        ),
    )

    @model_validator(mode="after")
    def _validate_observational_estimate(self) -> TrialDevelopmentCandidateUtilityEstimateV1:
        if float(self.lower) > float(self.upper) or not float(self.lower) <= float(self.estimate) <= float(self.upper):
            raise ValueError("candidate utility estimate must lie within an ordered interval")
        if len(set(self.analysis_covariate_ids)) != len(self.analysis_covariate_ids):
            raise ValueError("analysis_covariate_ids must be unique")
        if len(set(self.diagnostic_evidence_ids)) != len(self.diagnostic_evidence_ids):
            raise ValueError("candidate utility diagnostic_evidence_ids must be unique")
        if any(not path or len(checksum) != 64 for path, checksum in self.source_artifact_checksums.items()):
            raise ValueError("source_artifact_checksums require non-empty paths and SHA-256 values")
        return self


class TrialDevelopmentObservationalReviewSubmissionV1(BaseModel):
    """Participant evidence and action at the observational-review checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    response_branch: Literal["estimable", "qualified_non_nomination"]
    primary_resolution_evidence_class: Literal[
        "empirical_diagnosis",
        "design_or_provenance_reasoning",
        "evidence_insufficient",
    ]
    ranked_drug_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Complete permutation of every investigational candidate in "
            "candidate_drug_catalog.json for the estimable branch; empty for qualified non-nomination."
        ),
    )
    candidate_utility_estimates: tuple[TrialDevelopmentCandidateUtilityEstimateV1, ...] = Field(
        default_factory=tuple,
        description="Exactly one utility estimate per ranked candidate in the estimable branch.",
    )
    identification_evidence: tuple[TrialDevelopmentIdentificationEvidenceV1, ...] = Field(default_factory=tuple)
    supporting_evidence_ids: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="Submitted analysis evidence IDs that directly support the decision.",
    )
    candidate_drug_id: str | None = Field(
        default=None,
        description="Nominated candidate; required only for nominate_for_early_study.",
    )
    decision_action: Literal["nominate_for_early_study", "withhold_nomination"]
    decision_rationale: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Brief justification grounded in the observational evidence.",
    )
    claimed_subgroup_variables: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Variables used in executed subgroup-effect analyses, not adjustment or stratification variables.",
    )
    diagnostic_artifacts: tuple[TrialDevelopmentAnalysisDiagnosticV1, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _validate_observational_review(self) -> TrialDevelopmentObservationalReviewSubmissionV1:
        ranked = tuple(self.ranked_drug_ids)
        if len(set(ranked)) != len(ranked):
            raise ValueError("ranked_drug_ids must be unique")
        estimates = {row.candidate_drug_id: row for row in self.candidate_utility_estimates}
        if len(estimates) != len(self.candidate_utility_estimates):
            raise ValueError("candidate utility estimates must be unique by candidate_drug_id")
        if set(ranked) != set(estimates):
            raise ValueError("ranked_drug_ids must exactly match candidate utility estimates")
        identification = {row.evidence_id for row in self.identification_evidence}
        if len(identification) != len(self.identification_evidence):
            raise ValueError("identification evidence IDs must be unique")
        if self.response_branch == "estimable":
            if self.primary_resolution_evidence_class != "empirical_diagnosis":
                raise ValueError("estimable observational responses require empirical_diagnosis")
            if not ranked:
                raise ValueError("estimable observational responses require candidate estimates and ranking")
            if self.identification_evidence:
                raise ValueError("estimable observational responses cannot include non-nomination evidence")
        else:
            if ranked or estimates:
                raise ValueError("qualified non-nomination cannot contain candidate estimates or a causal ranking")
            if self.decision_action != "withhold_nomination":
                raise ValueError("qualified non-nomination must decline nomination")
            if not self.identification_evidence:
                raise ValueError("qualified non-nomination requires identification or support evidence")
        if self.decision_action == "nominate_for_early_study":
            if self.candidate_drug_id is None or self.candidate_drug_id not in estimates:
                raise ValueError("nomination requires an estimated candidate_drug_id")
        elif self.candidate_drug_id is not None:
            raise ValueError("withhold_nomination requires a null candidate_drug_id")
        diagnostics = {row.artifact_id for row in self.diagnostic_artifacts}
        if len(diagnostics) != len(self.diagnostic_artifacts):
            raise ValueError("diagnostic artifact IDs must be unique")
        evidence_ids = set(estimates[row].evidence_id for row in ranked) | diagnostics | identification
        if not set(self.supporting_evidence_ids) <= evidence_ids:
            raise ValueError("supporting_evidence_ids must reference submitted evidence")
        if len(set(self.supporting_evidence_ids)) != len(self.supporting_evidence_ids):
            raise ValueError("supporting_evidence_ids must be unique")
        return self


class TrialDevelopmentPhaseAnalysisSubmissionV1(BaseModel):
    """Participant analysis report after receiving one materialized trial output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    phase_id: str = Field(..., min_length=1)
    request_checksum: str = Field(..., min_length=64, max_length=64)
    trial_output_checksum: str = Field(..., min_length=64, max_length=64)
    selected_winner_drug_id: str | None = None
    ranked_drug_ids: tuple[str, ...] = Field(default_factory=tuple)
    candidate_utility_estimates: tuple[TrialDevelopmentCandidateUtilityEstimateV1, ...] = Field(default_factory=tuple)
    primary_effect: TrialDevelopmentEffectEstimateV1 | None = None
    safety_estimate: TrialDevelopmentSafetyEstimateV1 | None = None
    claimed_subgroup_variables: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Variables for subgroup effect claims executed in this analysis; each must be declared "
            "in request.subgroup_variables. Adjustment and stratification variables are not subgroup claims."
        ),
    )
    diagnostic_artifacts: tuple[TrialDevelopmentAnalysisDiagnosticV1, ...] = Field(default_factory=tuple)
    evidence_summary: str = Field("", max_length=4000)

    @model_validator(mode="after")
    def _validate_evidence_ids(self) -> TrialDevelopmentPhaseAnalysisSubmissionV1:
        evidence_ids = [artifact.artifact_id for artifact in self.diagnostic_artifacts]
        evidence_ids.extend(estimate.evidence_id for estimate in self.candidate_utility_estimates)
        available_diagnostics = {artifact.artifact_id for artifact in self.diagnostic_artifacts}
        for estimate in self.candidate_utility_estimates:
            missing = sorted(set(estimate.diagnostic_evidence_ids) - available_diagnostics)
            if missing:
                raise ValueError(f"candidate utility estimate references unknown diagnostic evidence: {missing!r}")
        estimated_candidates = {estimate.candidate_drug_id for estimate in self.candidate_utility_estimates}
        if len(estimated_candidates) != len(self.candidate_utility_estimates):
            raise ValueError("candidate utility estimates must be unique by candidate_drug_id")
        if self.phase_id == "observational_review":
            required_candidates = set(self.ranked_drug_ids)
            if self.selected_winner_drug_id is not None:
                required_candidates.add(self.selected_winner_drug_id)
            missing_candidates = sorted(required_candidates - estimated_candidates)
            if missing_candidates:
                raise ValueError(f"ranked or selected candidates lack utility estimates: {missing_candidates!r}")
        elif self.candidate_utility_estimates:
            raise ValueError("candidate utility estimates are only valid for observational_review")
        if self.primary_effect is not None:
            evidence_ids.append(self.primary_effect.evidence_id)
            missing = sorted(set(self.primary_effect.diagnostic_evidence_ids) - available_diagnostics)
            if missing:
                raise ValueError(f"primary_effect references unknown diagnostic evidence: {missing!r}")
        if self.safety_estimate is not None:
            evidence_ids.append(self.safety_estimate.evidence_id)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("analysis evidence IDs must be unique")
        return self

    def evidence_ids(self) -> frozenset[str]:
        """Return scoreable evidence IDs declared by this analysis."""

        values = {artifact.artifact_id for artifact in self.diagnostic_artifacts}
        values.update(estimate.evidence_id for estimate in self.candidate_utility_estimates)
        if self.primary_effect is not None:
            values.add(self.primary_effect.evidence_id)
        if self.safety_estimate is not None:
            values.add(self.safety_estimate.evidence_id)
        return frozenset(values)


class TrialDevelopmentPhaseDecisionSubmissionV1(BaseModel):
    """Participant advancement decision for one completed phase."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    phase_id: str = Field(..., min_length=1)
    request_checksum: str = Field(..., min_length=64, max_length=64)
    analysis_checksum: str = Field(..., min_length=64, max_length=64)
    decision_action: PhaseActionIdV1
    supporting_evidence_ids: tuple[str, ...] = Field(..., min_length=1)
    candidate_drug_id: str | None = None
    decision_rationale: str = Field("", max_length=4000)

    @model_validator(mode="after")
    def _validate_supporting_evidence_ids(self) -> TrialDevelopmentPhaseDecisionSubmissionV1:
        if len(set(self.supporting_evidence_ids)) != len(self.supporting_evidence_ids):
            raise ValueError("supporting_evidence_ids must be unique")
        return self


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}.")
    return payload


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_design_request_file_v1(*, request_path: Path) -> TrialDevelopmentRequestV1:
    """Validate one stepwise design request."""
    return TrialDevelopmentRequestV1.model_validate(_read_json(Path(request_path)))


def _validate_arm_mapping(*, root: Path, frames: tuple[pd.DataFrame, ...]) -> dict[str, object]:
    mapping = _read_json(Path(root) / "arm_mapping.json")
    expected_checksum = str(mapping.get("checksum", ""))
    if expected_checksum:
        payload = dict(mapping)
        payload.pop("checksum", None)
        if compute_sha256_hex(payload) != expected_checksum:
            raise ValueError("arm_mapping.json checksum mismatch.")
    required = {
        "control_arm_id",
        "candidate_arm_ids",
        "drug_id_by_arm",
        "arm_role_by_id",
        "request_candidate_drug_ids",
    }
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(f"arm_mapping.json missing required keys: {missing!r}.")
    drug_by_arm = mapping.get("drug_id_by_arm")
    role_by_arm = mapping.get("arm_role_by_id")
    if not isinstance(drug_by_arm, dict) or not isinstance(role_by_arm, dict):
        raise TypeError("arm_mapping.json drug_id_by_arm and arm_role_by_id must be JSON objects.")
    mapped_arms = {str(value) for value in drug_by_arm}
    table_arms: set[str] = set()
    for frame in frames:
        table_arms.update(str(value) for value in frame["ARM"].astype("string").dropna().unique().tolist())
    unknown = sorted(table_arms - mapped_arms)
    if unknown:
        raise ValueError(f"Trial tables contain ARM values absent from arm_mapping.json: {unknown!r}.")
    control_arm = str(mapping.get("control_arm_id", ""))
    if control_arm not in mapped_arms or str(role_by_arm.get(control_arm, "")) != "control":
        raise ValueError("arm_mapping.json control_arm_id must identify the control arm.")
    candidate_arm_values = mapping.get("candidate_arm_ids", ())
    if not isinstance(candidate_arm_values, list | tuple):
        raise TypeError("arm_mapping.json candidate_arm_ids must be a JSON array.")
    candidate_arms = tuple(str(value) for value in candidate_arm_values)
    if not candidate_arms:
        raise ValueError("arm_mapping.json candidate_arm_ids must be non-empty.")
    if any(str(role_by_arm.get(arm, "")) != "candidate" for arm in candidate_arms):
        raise ValueError("arm_mapping.json candidate_arm_ids must all have candidate role.")
    return mapping


def validate_phase_analysis_file_v1(*, submission_path: Path) -> TrialDevelopmentPhaseAnalysisSubmissionV1:
    """Validate one phase analysis submission."""
    return TrialDevelopmentPhaseAnalysisSubmissionV1.model_validate(_read_json(Path(submission_path)))


def validate_phase_action_policy_file_v1(*, scenario_root: Path) -> TrialDevelopmentPhaseActionPolicyV1:
    """Validate one scenario-local phase-action policy."""
    return TrialDevelopmentPhaseActionPolicyV1.model_validate(
        _read_json(Path(scenario_root) / "public" / "phase_action_policy.json")
    )


def validate_phase_decision_file_v1(*, submission_path: Path) -> TrialDevelopmentPhaseDecisionSubmissionV1:
    """Validate one phase decision submission."""
    return TrialDevelopmentPhaseDecisionSubmissionV1.model_validate(_read_json(Path(submission_path)))


def validate_phase_decision_against_policy_v1(
    *, scenario_root: Path, submission_path: Path
) -> TrialDevelopmentPhaseDecisionSubmissionV1:
    """Validate one phase decision against the scenario-local public action policy."""
    decision = validate_phase_decision_file_v1(submission_path=Path(submission_path))
    policy = validate_phase_action_policy_file_v1(scenario_root=Path(scenario_root))
    spec = policy.action_spec(str(decision.phase_id))
    action = str(decision.decision_action)
    if action not in set(str(value) for value in spec.allowed_action_ids):
        raise ValueError(f"decision_action={action!r} is not allowed for phase_id={decision.phase_id!r}.")
    if action in set(str(value) for value in spec.requires_candidate_drug_id) and not decision.candidate_drug_id:
        raise ValueError(f"decision_action={action!r} requires candidate_drug_id.")
    return decision


def validate_trial_output_bundle_v1(*, trial_output_root: Path) -> TrialDevelopmentTrialOutputManifestV1:
    """Validate a participant-safe materialized trial output bundle."""
    root = Path(trial_output_root)
    manifest = TrialDevelopmentTrialOutputManifestV1.model_validate(_read_json(root / "trial_output_manifest.json"))
    required = set(manifest.table_files) | set(manifest.metadata_files)
    missing = sorted(rel for rel in required if not (root / rel).is_file())
    if missing:
        raise FileNotFoundError(f"Trial output bundle missing required files: {missing!r}.")
    proposal = TrialDevelopmentRequestV1.model_validate(_read_json(root / "agent_request.json"))
    evidence_request = TrialDevelopmentRequestV1.model_validate(_read_json(root / "request.json"))
    if proposal.checksum() != str(manifest.request_checksum):
        raise ValueError("agent_request.json does not match the trial output manifest request checksum.")
    if evidence_request.checksum() != str(manifest.evidence_request_checksum):
        raise ValueError("request.json does not match the trial output manifest evidence request checksum.")
    for rel, expected in sorted(manifest.table_checksums.items()):
        observed = _sha256_file(root / rel)
        if observed != str(expected):
            raise ValueError(f"Checksum mismatch for trial output table: {rel}.")
    participants = pd.read_parquet(root / "participants.parquet")
    endpoints = pd.read_parquet(root / "endpoints.parquet")
    safety = pd.read_parquet(root / "safety.parquet")
    if int(len(participants)) != int(manifest.n_participants):
        raise ValueError("participants.parquet row count does not match trial output manifest.")
    for label, frame in (
        ("participants", participants),
        ("endpoints", endpoints),
        ("safety", safety),
    ):
        missing_cols = sorted({"USUBJID", "ARM"} - set(frame.columns))
        if missing_cols:
            raise ValueError(f"{label}.parquet missing required columns: {missing_cols!r}.")
    for col in ("EVENT", "TIME", "FOLLOW_UP_DAYS"):
        if col not in endpoints.columns:
            raise ValueError(f"endpoints.parquet missing required column: {col}.")
    _validate_arm_mapping(root=root, frames=(participants, endpoints, safety))
    return manifest
