"""Typed contracts for participant diagnostics and TrialEval proof reports."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DiagnosticProofStatusV1 = Literal["pass", "fail"]
DiagnosticProofDispositionV1 = Literal[
    "retain_official",
    "repair_required",
    "exclude_or_downgrade",
    "block_release",
]
AssumptionSeverityBandV1 = Literal["holds", "mild", "fragile", "broken"]
DiagnosticEvidenceClassV1 = Literal[
    "ph_compatible_diagnostics",
    "non_ph_diagnostics",
    "censoring_competing_risk_diagnostics",
    "censoring_followup_diagnostics",
    "confounding_design_adjustment_evidence",
    "model_form_diagnostics",
    "design_adjustment_evidence",
    "partial_identification_evidence",
    "endpoint_defect_evidence",
    "qualified_limitation_evidence",
    "scale_alignment_evidence",
]


class ParticipantAssumptionDiagnosticV1(BaseModel):
    """One continuous diagnostic computed before evaluator evidence is read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assumption_id: str = Field(..., min_length=1)
    severity_metric_name: str = Field(..., min_length=1)
    severity_metric: float = Field(..., ge=0.0)
    supporting_metrics: dict[str, float] = Field(default_factory=dict)


class AssumptionDiagnosticReplayV1(BaseModel):
    """Independent participant-side replay of one evaluator diagnostic."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assumption_id: str
    diagnosability: str
    severity_metric_name: str
    participant_severity_metric: float
    evaluator_severity_metric: float
    threshold_stressed: float
    threshold_fragile: float
    threshold_broken: float
    decision_metric_names: dict[Literal["stressed", "fragile", "broken"], str]
    participant_decision_metric_values: dict[Literal["stressed", "fragile", "broken"], float]
    evaluator_decision_metric_values: dict[Literal["stressed", "fragile", "broken"], float]
    participant_band: AssumptionSeverityBandV1
    evaluator_band: AssumptionSeverityBandV1
    classification_applicable: bool
    numeric_agreement: bool
    classification_agreement: bool | None
    nearest_threshold_margin: float


class TrialEvalDiagnosticProofRowV1(BaseModel):
    """Per-item public diagnostic proof witness."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval_diagnostic_proof_row/v1"] = (
        "trialagentbench.trialeval_diagnostic_proof_row/v1"
    )
    task_id: str
    item_id: str
    variant_id: str
    design_tier: str
    assumption_tier: str
    context_tier: str
    design_family: str
    evidence_class: DiagnosticEvidenceClassV1
    intended_assumption_statuses: tuple[str, ...]
    required_diagnostic_keys: tuple[str, ...]
    satisfied_diagnostic_keys: tuple[str, ...]
    missing_diagnostic_keys: tuple[str, ...]
    resolved_warning_keys: tuple[str, ...] = ()
    assumption_replays: tuple[AssumptionDiagnosticReplayV1, ...]
    inferred_route_families: tuple[str, ...]
    credit_eligible_route_families: tuple[str, ...]
    public_input_paths: tuple[str, ...]
    public_input_hashes: tuple[str, ...]
    n_subjects: int
    n_events: int
    treated_events: int | None
    control_events: int | None
    schoenfeld_p_value: float | None = None
    neg_log10_schoenfeld_p: float | None = None
    scaled_schoenfeld_rank_slope: float | None = None
    scaled_schoenfeld_rank_slope_standard_error: float | None = None
    simultaneous_lower_abs_time_varying_log_hazard_range: float | None = None
    ph_method_change_threshold_crossed: bool | None = None
    ph_diagnostic_required: bool
    ph_diagnostic_available: bool
    method_applicability_rule_id: str
    method_applicability_decision: str
    proof_surface_source_rows: tuple[str, ...]
    status: DiagnosticProofStatusV1
    disposition: DiagnosticProofDispositionV1
    findings: tuple[str, ...] = ()


class ReferenceAnalysisDecisionV1(BaseModel):
    """Analysis decision inferred before evaluator reference is available."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.reference_analysis_decision/v1"] = (
        "trialagentbench.reference_analysis_decision/v1"
    )
    task_id: str
    design_family: str
    evidence_class: DiagnosticEvidenceClassV1
    candidate_route_families: tuple[str, ...]
    required_diagnostic_keys: tuple[str, ...]
    observed_evidence: tuple[str, ...]
    ph_diagnostic_required: bool


class ParticipantDiagnosticEvidenceV1(BaseModel):
    """Diagnostic evidence computed exclusively from one participant archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.participant_diagnostic_evidence/v1"] = (
        "trialagentbench.participant_diagnostic_evidence/v1"
    )
    task_id: str
    decision: ReferenceAnalysisDecisionV1
    satisfied_diagnostic_keys: tuple[str, ...]
    missing_diagnostic_keys: tuple[str, ...]
    public_input_paths: tuple[str, ...]
    public_input_hashes: tuple[str, ...]
    n_subjects: int
    n_events: int
    treated_events: int | None
    control_events: int | None
    schoenfeld_p_value: float | None
    scaled_schoenfeld_rank_slope: float | None
    scaled_schoenfeld_rank_slope_standard_error: float | None
    simultaneous_lower_abs_time_varying_log_hazard_range: float | None
    ph_method_change_threshold_crossed: bool | None
    assumption_diagnostics: tuple[ParticipantAssumptionDiagnosticV1, ...]
    diagnostic_findings: tuple[str, ...]


class TrialEvalDiagnosticProofSummaryRowV1(BaseModel):
    """Grouped diagnostic proof summary row."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval_diagnostic_proof_summary_row/v1"] = (
        "trialagentbench.trialeval_diagnostic_proof_summary_row/v1"
    )
    group_kind: Literal["evidence_class", "design_tier", "assumption_tier", "context_tier", "design_family"]
    group_value: str
    total: int
    passed: int
    failed: int
    retain_official: int
    exclude_or_downgrade: int
    block_release: int


class TrialEvalDiagnosticProofReportV1(BaseModel):
    """Release-scale TrialEvalBench diagnostic proof-surface report."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval_diagnostic_proof_report/v1"] = (
        "trialagentbench.trialeval_diagnostic_proof_report/v1"
    )
    public_zip: str
    evaluator_zip: str
    status: DiagnosticProofStatusV1
    total_items: int
    failed_items: int
    warning_items: int
    unresolved_warning_items: int
    ph_diagnostic_rows: int
    ph_method_change_rows: int
    non_ph_diagnostic_rows: int
    design_adjustment_rows: int
    source_table_count: int
    figure_source_count: int
    rows: tuple[TrialEvalDiagnosticProofRowV1, ...]
    summaries: tuple[TrialEvalDiagnosticProofSummaryRowV1, ...]


__all__ = [
    "AssumptionDiagnosticReplayV1",
    "AssumptionSeverityBandV1",
    "DiagnosticEvidenceClassV1",
    "DiagnosticProofDispositionV1",
    "DiagnosticProofStatusV1",
    "ParticipantAssumptionDiagnosticV1",
    "ParticipantDiagnosticEvidenceV1",
    "ReferenceAnalysisDecisionV1",
    "TrialEvalDiagnosticProofReportV1",
    "TrialEvalDiagnosticProofRowV1",
    "TrialEvalDiagnosticProofSummaryRowV1",
]
