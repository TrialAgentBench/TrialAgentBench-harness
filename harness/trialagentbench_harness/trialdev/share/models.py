"""Pydantic models for the offline frozen trial-development bundle contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from trialagentbench_harness.trialdev.share.hashing import compute_sha256_hex

__all__ = [
    "FrozenSuperpopulationManifestV1",
    "TrialDevelopmentBenchmarkItemV1",
    "TrialDevelopmentBenchmarkSuiteManifestV1",
    "PhaseModuleSpecV1",
    "ScenarioBundleArtifactV1",
    "ScenarioBundleManifestV1",
    "SuperpopulationQualificationSummaryV1",
    "SuperpopulationRequestBoundV1",
    "TrialDevelopmentGraderManifestV1",
    "TrialDevelopmentGraderRecordV1",
    "TrialDevelopmentGradingProcedureV1",
    "TrialDevelopmentSubmissionSchemaV1",
    "TrialDevelopmentDiscontinuationReferenceRecordV1",
    "TrialDevelopmentEvalContractV1",
    "TrialDevelopmentPhaseTargetRecordV1",
    "TrialDevelopmentPhaseTargetsManifestV1",
    "TrialDevelopmentRequestV1",
    "TrialDevelopmentSeriousEventDefinitionV1",
    "TrialDevelopmentSafetyReferenceManifestV1",
    "TrialDevelopmentSafetyReferenceRecordV1",
    "TrialDevelopmentEvaluationReferenceManifestRecordV1",
    "TrialDevelopmentEvaluationReferenceManifestV1",
    "TrialMaterializationAuditV1",
    "TrialMaterializationResultV1",
]


PhaseIdV1 = Literal["observational_review", "phase1", "phase2", "phase3"]
SelectionObjectiveIdV1 = Literal[
    "pure_efficacy",
    "benefit_risk",
    "cost_effective_best",
    "net_clinical_value_under_budget",
]
TreatmentDiscontinuationStrategyV1 = Literal["treatment_policy", "composite_discontinuation", "while_on_treatment"]
InterimPolicyV1 = Literal["fixed_final"]
SiteStrategyV1 = Literal["high_enrolling", "region_balanced"]


class TrialDevelopmentSeriousEventDefinitionV1(BaseModel):
    """Exact public column identities for one serious adverse-event family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint_id: str = Field(..., min_length=1)
    event_column: str = Field(..., min_length=1)
    time_column: str = Field(..., min_length=1)
    seriousness_column: str = Field(..., min_length=1)
    severity_column: str = Field(..., min_length=1)


class ScenarioBundleArtifactV1(BaseModel):
    """One checksummed file included in a scenario bundle surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rel_path: str = Field(..., min_length=1)
    sha256: str = Field(..., min_length=64, max_length=64)
    size_bytes: int = Field(..., ge=0)
    surface: Literal["public", "hidden", "grader", "manifests", "release"]


class ScenarioBundleManifestV1(BaseModel):
    """Checksummed manifest for one scenario bundle directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    artifacts: tuple[ScenarioBundleArtifactV1, ...] = Field(default_factory=tuple)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _canonicalize_and_hash(self) -> ScenarioBundleManifestV1:
        artifacts = tuple(sorted(self.artifacts, key=lambda a: (str(a.surface), str(a.rel_path))))
        object.__setattr__(self, "artifacts", artifacts)
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        payload["artifacts"] = sorted(
            payload.get("artifacts", []),
            key=lambda a: (str(a.get("surface", "")), str(a.get("rel_path", ""))),
        )
        object.__setattr__(self, "checksum", compute_sha256_hex(payload))
        return self


class TrialDevelopmentBenchmarkItemV1(BaseModel):
    """One offline benchmark item projected from the frozen scenario suite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(..., min_length=1)
    scenario_id: str = Field(..., min_length=1)
    phase_id: PhaseIdV1
    objective_id: str = Field(..., min_length=1)
    task_definition_id: str = Field(..., min_length=1)
    endpoint_id: str | None = None
    allowed_endpoint_ids: tuple[str, ...] = Field(default_factory=tuple)
    allowed_follow_up_days: tuple[int, ...] = Field(default_factory=tuple)
    allowed_enrollment_window_days: tuple[int, ...] = Field(default_factory=tuple)
    allowed_site_count_budgets: tuple[int, ...] = Field(default_factory=tuple)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class TrialDevelopmentBenchmarkSuiteManifestV1(BaseModel):
    """Checksummed suite-level manifest for the offline benchmark inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    suite_id: str = Field(..., min_length=1)
    release_root: str = Field(..., min_length=1)
    items: tuple[TrialDevelopmentBenchmarkItemV1, ...] = Field(default_factory=tuple)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _validate_and_hash(self) -> TrialDevelopmentBenchmarkSuiteManifestV1:
        item_ids = [str(item.item_id) for item in self.items]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("TrialDevelopmentBenchmarkSuiteManifestV1 contains duplicate item ids.")
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        payload["items"] = sorted(
            payload.get("items", []),
            key=lambda item: (
                str(item.get("phase_id", "")),
                str(item.get("objective_id", "")),
                str(item.get("scenario_id", "")),
                str(item.get("item_id", "")),
            ),
        )
        object.__setattr__(self, "checksum", compute_sha256_hex(payload))
        return self


class TrialDevelopmentEvaluationReferenceManifestRecordV1(BaseModel):
    """One reference record for a (scenario, drug, endpoint, metric, reference_basis) tuple."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(..., min_length=1)
    candidate_drug_id: str = Field(..., min_length=1)
    reference_basis: Literal["construction_state", "released_data_reference"]
    endpoint_id: str = Field(..., min_length=1)
    metric: str = Field(..., min_length=1)
    value: float
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class TrialDevelopmentEvaluationReferenceManifestV1(BaseModel):
    """Checksummed reference manifest container for one scenario bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    reference: tuple[TrialDevelopmentEvaluationReferenceManifestRecordV1, ...] = Field(default_factory=tuple)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _validate_and_hash(self) -> TrialDevelopmentEvaluationReferenceManifestV1:
        keys: set[tuple[str, str, str, str, str, str]] = set()
        for record in self.reference:
            phase_id = str(record.payload.get("phase_id", ""))
            key = (
                str(record.candidate_drug_id),
                str(record.reference_basis),
                str(record.endpoint_id),
                str(record.metric),
                str(phase_id),
                str(record.payload.get("treatment_discontinuation_strategy", "")),
            )
            if key in keys:
                raise ValueError("TrialDevelopmentEvaluationReferenceManifestV1 contains duplicate reference keys.")
            keys.add(key)
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        payload["reference"] = sorted(
            payload.get("reference", []),
            key=lambda r: (
                str(r.get("candidate_drug_id", "")),
                str(r.get("reference_basis", "")),
                str(r.get("endpoint_id", "")),
                str(r.get("metric", "")),
                str((r.get("payload") or {}).get("phase_id", "")),
                str((r.get("payload") or {}).get("treatment_discontinuation_strategy", "")),
            ),
        )
        object.__setattr__(self, "checksum", compute_sha256_hex(payload))
        return self


class TrialDevelopmentPhaseTargetRecordV1(BaseModel):
    """Hidden phase-target record consumed by downstream scoring tooling."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(..., min_length=1)
    phase_id: PhaseIdV1
    target_id: str = Field(..., min_length=1)
    candidate_drug_id: str | None = None
    endpoint_id: str | None = None
    reference_basis: Literal["construction_state", "released_data_reference"]
    metric: str = Field(..., min_length=1)
    value: float
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class TrialDevelopmentPhaseTargetsManifestV1(BaseModel):
    """Checksummed container for hidden phase targets for one scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    phase_targets: tuple[TrialDevelopmentPhaseTargetRecordV1, ...] = Field(default_factory=tuple)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _validate_and_hash(self) -> TrialDevelopmentPhaseTargetsManifestV1:
        keys: set[tuple[str, str, str, str, str, str]] = set()
        for record in self.phase_targets:
            key = (
                str(record.phase_id),
                str(record.target_id),
                str(record.reference_basis),
                str(record.metric),
                str(record.candidate_drug_id or ""),
                str(record.endpoint_id or ""),
            )
            if key in keys:
                raise ValueError("TrialDevelopmentPhaseTargetsManifestV1 contains duplicate phase target keys.")
            keys.add(key)
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        payload["phase_targets"] = sorted(
            payload.get("phase_targets", []),
            key=lambda r: (
                str(r.get("phase_id", "")),
                str(r.get("target_id", "")),
                str(r.get("reference_basis", "")),
                str(r.get("metric", "")),
                str(r.get("candidate_drug_id", "")),
                str(r.get("endpoint_id", "")),
            ),
        )
        object.__setattr__(self, "checksum", compute_sha256_hex(payload))
        return self


class TrialDevelopmentSafetyReferenceRecordV1(BaseModel):
    """Grouped hidden safety reference record aligned to the AE taxonomy contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(..., min_length=1)
    candidate_drug_id: str = Field(..., min_length=1)
    phase_id: PhaseIdV1
    ae_family_id: str = Field(..., min_length=1)
    seriousness_status: Literal["non_serious", "serious"]
    severity_tier: str | None = None
    metric: str = Field(..., min_length=1)
    value: float
    reference_basis: Literal["construction_state", "released_data_reference"]
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class TrialDevelopmentDiscontinuationReferenceRecordV1(BaseModel):
    """Grouped discontinuation reference record (explicitly separate from AE rows)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(..., min_length=1)
    candidate_drug_id: str = Field(..., min_length=1)
    phase_id: PhaseIdV1
    metric: str = Field(..., min_length=1)
    value: float
    reference_basis: Literal["construction_state", "released_data_reference"]
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class TrialDevelopmentSafetyReferenceManifestV1(BaseModel):
    """Checksummed safety reference manifest for one scenario bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    adverse_event_reference: tuple[TrialDevelopmentSafetyReferenceRecordV1, ...] = Field(default_factory=tuple)
    discontinuation_reference: tuple[TrialDevelopmentDiscontinuationReferenceRecordV1, ...] = Field(
        default_factory=tuple
    )
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _validate_and_hash(self) -> TrialDevelopmentSafetyReferenceManifestV1:
        ae_keys: set[tuple[str, str, str, str, str, str, str]] = set()
        for ae_record in self.adverse_event_reference:
            ae_key = (
                str(ae_record.candidate_drug_id),
                str(ae_record.phase_id),
                str(ae_record.ae_family_id),
                str(ae_record.seriousness_status),
                str(ae_record.severity_tier or ""),
                str(ae_record.reference_basis),
                str(ae_record.metric),
            )
            if ae_key in ae_keys:
                raise ValueError("TrialDevelopmentSafetyReferenceManifestV1 contains duplicate AE reference keys.")
            ae_keys.add(ae_key)
        disc_keys: set[tuple[str, str, str, str]] = set()
        for disc_record in self.discontinuation_reference:
            disc_key = (
                str(disc_record.candidate_drug_id),
                str(disc_record.phase_id),
                str(disc_record.reference_basis),
                str(disc_record.metric),
            )
            if disc_key in disc_keys:
                raise ValueError(
                    "TrialDevelopmentSafetyReferenceManifestV1 contains duplicate discontinuation reference keys."
                )
            disc_keys.add(disc_key)
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        payload["adverse_event_reference"] = sorted(
            payload.get("adverse_event_reference", []),
            key=lambda r: (
                str(r.get("candidate_drug_id", "")),
                str(r.get("phase_id", "")),
                str(r.get("ae_family_id", "")),
                str(r.get("seriousness_status", "")),
                str(r.get("severity_tier", "")),
                str(r.get("reference_basis", "")),
                str(r.get("metric", "")),
            ),
        )
        payload["discontinuation_reference"] = sorted(
            payload.get("discontinuation_reference", []),
            key=lambda r: (
                str(r.get("candidate_drug_id", "")),
                str(r.get("phase_id", "")),
                str(r.get("reference_basis", "")),
                str(r.get("metric", "")),
            ),
        )
        object.__setattr__(self, "checksum", compute_sha256_hex(payload))
        return self


class FrozenSuperpopulationManifestV1(BaseModel):
    """Frozen superpopulation metadata for one scenario bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    n_subjects: int = Field(..., ge=1)
    horizon_days: int = Field(..., ge=1)
    candidate_drug_ids: tuple[str, ...] = Field(..., min_length=2)
    control_drug_id: str = Field(..., min_length=1)
    terminal_endpoint_id: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> FrozenSuperpopulationManifestV1:
        ids = [str(x) for x in self.candidate_drug_ids]
        if len(set(ids)) != len(ids):
            raise ValueError("FrozenSuperpopulationManifestV1.candidate_drug_ids must be unique.")
        if str(self.control_drug_id) not in set(ids):
            raise ValueError("FrozenSuperpopulationManifestV1.control_drug_id must be in candidate_drug_ids.")
        return self


class SuperpopulationRequestBoundV1(BaseModel):
    """Recommended enforceable request bound for the external evaluation harness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase_id: PhaseIdV1
    max_sample_size: int = Field(..., ge=1)
    max_follow_up_days: int = Field(..., ge=1)
    max_analysis_covariates: int = Field(..., ge=0)
    max_subgroup_splits: int = Field(..., ge=0)
    justification: str = Field(..., min_length=1)


class SuperpopulationQualificationSummaryV1(BaseModel):
    """Release-gated qualification evidence for superpopulation sufficiency."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    candidate_request_families_tested: tuple[str, ...] = Field(default_factory=tuple)
    sufficient_for_release: bool
    limiting_factors: tuple[str, ...] = Field(default_factory=tuple)
    diagnostic_observations: tuple[str, ...] = Field(default_factory=tuple)
    recommended_bounds: tuple[SuperpopulationRequestBoundV1, ...] = Field(default_factory=tuple)
    realism_summary: dict[str, JsonValue] = Field(default_factory=dict)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _hash(self) -> SuperpopulationQualificationSummaryV1:
        if self.sufficient_for_release == bool(self.limiting_factors):
            raise ValueError("sufficient_for_release must be equivalent to having no limiting_factors.")
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        bounds = list(payload.get("recommended_bounds", []) or [])
        bounds.sort(key=lambda item: (str(item.get("phase_id", "")), int(item.get("max_sample_size", 0))))
        payload["recommended_bounds"] = bounds
        payload["candidate_request_families_tested"] = sorted(
            set(payload.get("candidate_request_families_tested", []))
        )
        payload["limiting_factors"] = sorted(set(payload.get("limiting_factors", [])))
        payload["diagnostic_observations"] = sorted(set(payload.get("diagnostic_observations", [])))
        object.__setattr__(self, "checksum", compute_sha256_hex(payload))
        return self


class PhaseModuleSpecV1(BaseModel):
    """Phase-specific request menus and visibility rules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase_id: PhaseIdV1
    primary_population_policy: Literal["fixed_public_cohort"] = "fixed_public_cohort"
    allowed_endpoint_ids: tuple[str, ...] = Field(default_factory=tuple)
    allowed_follow_up_days: tuple[int, ...] = Field(default_factory=tuple)
    allowed_enrollment_window_days: tuple[int, ...] = Field(default_factory=tuple)
    allowed_site_count_budgets: tuple[int, ...] = Field(default_factory=tuple)
    allowed_allocation_ratios: tuple[str, ...] = Field(default_factory=tuple)
    allowed_variable_ids: tuple[str, ...] = Field(default_factory=tuple)
    max_sample_size: int | None = Field(default=None, ge=1)
    max_analysis_covariates: int | None = Field(default=None, ge=0)
    max_subgroup_splits: int | None = Field(default=None, ge=0)
    includes_control_arm: bool = True
    allowed_treatment_discontinuation_strategies: tuple[TreatmentDiscontinuationStrategyV1, ...] = Field(
        default_factory=tuple
    )
    allowed_interim_policies: tuple[InterimPolicyV1, ...] = Field(default_factory=tuple)
    allowed_site_strategies: tuple[SiteStrategyV1, ...] = Field(default_factory=tuple)
    allowed_selection_objectives: tuple[SelectionObjectiveIdV1, ...] = Field(default_factory=tuple)
    visible_outputs: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _validate_uniques(self) -> PhaseModuleSpecV1:
        for label, values in (
            ("allowed_endpoint_ids", self.allowed_endpoint_ids),
            ("allowed_follow_up_days", self.allowed_follow_up_days),
            ("allowed_enrollment_window_days", self.allowed_enrollment_window_days),
            ("allowed_site_count_budgets", self.allowed_site_count_budgets),
            ("allowed_allocation_ratios", self.allowed_allocation_ratios),
            ("allowed_variable_ids", self.allowed_variable_ids),
            (
                "allowed_treatment_discontinuation_strategies",
                self.allowed_treatment_discontinuation_strategies,
            ),
            ("allowed_interim_policies", self.allowed_interim_policies),
            ("allowed_site_strategies", self.allowed_site_strategies),
            ("allowed_selection_objectives", self.allowed_selection_objectives),
            ("visible_outputs", self.visible_outputs),
        ):
            items = [str(v) for v in values]
            if len(set(items)) != len(items):
                raise ValueError(f"PhaseModuleSpecV1.{label} must be unique.")
        if any(str(v).startswith("_") for v in self.allowed_variable_ids):
            raise ValueError("PhaseModuleSpecV1.allowed_variable_ids must not include underscore-prefixed ids.")
        randomized_menus = (
            ("allowed_follow_up_days", self.allowed_follow_up_days),
            ("allowed_enrollment_window_days", self.allowed_enrollment_window_days),
            ("allowed_site_count_budgets", self.allowed_site_count_budgets),
            ("allowed_allocation_ratios", self.allowed_allocation_ratios),
            ("allowed_interim_policies", self.allowed_interim_policies),
            ("allowed_site_strategies", self.allowed_site_strategies),
            ("allowed_selection_objectives", self.allowed_selection_objectives),
        )
        if self.phase_id == "observational_review":
            populated = [label for label, values in randomized_menus[:-1] if values]
            if populated:
                raise ValueError(f"Observational phase module must not expose randomized-trial menus: {populated}.")
            if self.includes_control_arm:
                raise ValueError("Observational phase module must not include a control arm.")
        else:
            missing = [label for label, values in randomized_menus if not values]
            if missing:
                raise ValueError(f"Randomized phase module requires complete request menus: {missing}.")
            if self.phase_id == "phase1" and self.allowed_endpoint_ids:
                raise ValueError("Phase-1 module must not expose efficacy endpoint choices.")
            if self.phase_id == "phase1" and self.allowed_treatment_discontinuation_strategies:
                raise ValueError("Phase-1 module must not expose a treatment-discontinuation strategy.")
            if self.phase_id in {"phase2", "phase3"} and not self.allowed_endpoint_ids:
                raise ValueError(f"{self.phase_id} module requires at least one endpoint choice.")
            if self.phase_id in {"phase2", "phase3"} and not self.allowed_treatment_discontinuation_strategies:
                raise ValueError(f"{self.phase_id} module requires a treatment-discontinuation strategy menu.")
        return self


class TrialDevelopmentRequestV1(BaseModel):
    """One governed trial-development request for deterministic materialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    phase_id: PhaseIdV1
    candidate_drug_ids: tuple[str, ...] = Field(..., min_length=1)
    target_sample_size: int | None = Field(default=None, ge=1)
    endpoint_id: str | None = Field(default=None, min_length=1)
    follow_up_days: int | None = Field(default=None, ge=1)
    enrollment_window_days: int | None = Field(default=None, ge=1)
    site_count_budget: int | None = Field(default=None, ge=1)
    allocation_ratio: str | None = Field(default=None, min_length=1)
    allocation_weights: tuple[float, ...] = Field(default_factory=tuple)
    design_cell_id: str | None = Field(default=None, min_length=1)
    treatment_discontinuation_strategy: TreatmentDiscontinuationStrategyV1 | None = None
    interim_policy: InterimPolicyV1 | None = None
    site_strategy: SiteStrategyV1 | None = None
    selection_objective: SelectionObjectiveIdV1 | None = None
    stratification_variables: tuple[str, ...] = Field(default_factory=tuple)
    analysis_covariates: tuple[str, ...] = Field(default_factory=tuple)
    subgroup_variables: tuple[str, ...] = Field(default_factory=tuple)

    def checksum(self) -> str:
        """Return a deterministic checksum of the request payload."""
        payload = self.model_dump(mode="json", exclude_none=True)
        return compute_sha256_hex(payload)

    @property
    def primary_candidate_drug_id(self) -> str:
        """Return the first candidate drug id."""
        return str(self.candidate_drug_ids[0])

    @model_validator(mode="after")
    def _validate_request(self) -> TrialDevelopmentRequestV1:
        if str(self.scenario_id).startswith("_"):
            raise ValueError("scenario_id must not start with underscore.")
        candidate_drug_ids = [str(v) for v in self.candidate_drug_ids]
        if any(not v for v in candidate_drug_ids):
            raise ValueError("candidate_drug_ids must not contain empty values.")
        if len(set(candidate_drug_ids)) != len(candidate_drug_ids):
            raise ValueError("candidate_drug_ids must be unique.")
        if any(str(v).startswith("_") for v in candidate_drug_ids):
            raise ValueError("candidate_drug_ids must not start with underscore.")
        for label, values in (
            ("stratification_variables", self.stratification_variables),
            ("analysis_covariates", self.analysis_covariates),
            ("subgroup_variables", self.subgroup_variables),
        ):
            items = [str(v) for v in values]
            if any(not v for v in items):
                raise ValueError(f"{label} must not contain empty values.")
            if len(set(items)) != len(items):
                raise ValueError(f"{label} must be unique.")
            if any(v.startswith("_") for v in items):
                raise ValueError(f"{label} must not contain underscore-prefixed values.")
        if self.allocation_weights:
            weights = [float(v) for v in self.allocation_weights]
            if any((not float(v) > 0.0) for v in weights):
                raise ValueError("allocation_weights must be > 0.")
            expected = len(self.candidate_drug_ids) + 1
            if int(len(weights)) != int(expected):
                raise ValueError(
                    "allocation_weights length must equal len(candidate_drug_ids) plus the randomized control arm."
                )
        if self.phase_id == "observational_review":
            forbidden = {
                "target_sample_size": self.target_sample_size,
                "endpoint_id": self.endpoint_id,
                "follow_up_days": self.follow_up_days,
                "enrollment_window_days": self.enrollment_window_days,
                "site_count_budget": self.site_count_budget,
                "allocation_ratio": self.allocation_ratio,
                "treatment_discontinuation_strategy": self.treatment_discontinuation_strategy,
                "interim_policy": self.interim_policy,
                "site_strategy": self.site_strategy,
                "design_cell_id": self.design_cell_id,
            }
            populated = [name for name, value in forbidden.items() if value is not None]
            if self.allocation_weights:
                populated.append("allocation_weights")
            if populated:
                raise ValueError(f"Observational-review request contains randomized-trial fields: {populated}.")
        else:
            if len(self.candidate_drug_ids) != 1:
                raise ValueError(
                    "Randomized TrialDev requests require exactly one investigational regimen plus control."
                )
            required = {
                "target_sample_size": self.target_sample_size,
                "follow_up_days": self.follow_up_days,
                "enrollment_window_days": self.enrollment_window_days,
                "site_count_budget": self.site_count_budget,
                "interim_policy": self.interim_policy,
                "site_strategy": self.site_strategy,
                "selection_objective": self.selection_objective,
                "design_cell_id": self.design_cell_id,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"Randomized-phase request requires explicit design fields: {missing}.")
            if (self.allocation_ratio is None) == (not self.allocation_weights):
                raise ValueError(
                    "Randomized-phase request requires exactly one allocation_ratio or allocation_weights."
                )
            if self.phase_id == "phase1" and self.endpoint_id is not None:
                raise ValueError("Phase-1 request must not set endpoint_id.")
            if self.phase_id == "phase1" and self.treatment_discontinuation_strategy is not None:
                raise ValueError("Phase-1 request must not set treatment_discontinuation_strategy.")
            if self.phase_id in {"phase2", "phase3"} and self.endpoint_id is None:
                raise ValueError(f"{self.phase_id} request requires endpoint_id.")
            if self.phase_id in {"phase2", "phase3"} and self.treatment_discontinuation_strategy is None:
                raise ValueError(f"{self.phase_id} request requires treatment_discontinuation_strategy.")
        return self


class TrialDevelopmentGraderRecordV1(BaseModel):
    """One typed hidden grader record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(..., min_length=1)
    phase_id: str = Field(..., min_length=1)
    lane_id: str = Field(..., min_length=1)
    objective_id: str = Field(..., min_length=1)
    candidate_drug_ids: tuple[str, ...] = Field(default_factory=tuple)
    endpoint_id: str | None = None
    method_route_id: str | None = Field(default=None, min_length=1)
    metric: str = Field(..., min_length=1)
    value: float
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class TrialDevelopmentGraderManifestV1(BaseModel):
    """Checksummed hidden grader manifest for one domain file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    domain_id: str = Field(..., min_length=1)
    source: str | None = Field(default=None, min_length=1)
    public_recoverability_report_checksum: str | None = Field(default=None, min_length=64, max_length=64)
    records: tuple[TrialDevelopmentGraderRecordV1, ...] = Field(default_factory=tuple)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _validate_and_hash(self) -> TrialDevelopmentGraderManifestV1:
        keys: set[tuple[str, str, str, str, str, str, tuple[str, ...], str]] = set()
        for record in self.records:
            key = (
                str(record.phase_id),
                str(record.lane_id),
                str(record.objective_id),
                str(record.metric),
                str(record.endpoint_id or ""),
                str(record.method_route_id or ""),
                tuple(str(value) for value in record.candidate_drug_ids),
                str(record.payload.get("task_id", "")),
            )
            if key in keys:
                raise ValueError("TrialDevelopmentGraderManifestV1 contains duplicate grader keys.")
            keys.add(key)
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        payload["records"] = sorted(
            payload.get("records", []),
            key=lambda r: (
                str(r.get("phase_id", "")),
                str(r.get("lane_id", "")),
                str(r.get("objective_id", "")),
                str(r.get("metric", "")),
                str(r.get("endpoint_id", "")),
                str(r.get("method_route_id", "")),
                tuple(str(x) for x in (r.get("candidate_drug_ids", []) or [])),
            ),
        )
        object.__setattr__(self, "checksum", compute_sha256_hex(payload))
        return self


class TrialDevelopmentSubmissionSchemaV1(BaseModel):
    """Checksummed submission contract for the standalone grader package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    schema_id: Literal["clinical_program_submission_v1"] = Field("clinical_program_submission_v1")
    required_sections: tuple[str, ...] = Field(default=("request", "analysis_report", "program_decision"))
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _hash(self) -> TrialDevelopmentSubmissionSchemaV1:
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        payload["required_sections"] = sorted(set(payload.get("required_sections", [])))
        object.__setattr__(self, "checksum", compute_sha256_hex(payload))
        return self


class TrialDevelopmentGradingProcedureV1(BaseModel):
    """Checksummed grading procedure contract for the standalone grader package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    submission_schema_id: Literal["clinical_program_submission_v1"] = Field("clinical_program_submission_v1")
    supported_lanes: tuple[str, ...] = Field(default_factory=tuple)
    supported_objectives: tuple[str, ...] = Field(default_factory=tuple)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _hash(self) -> TrialDevelopmentGradingProcedureV1:
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        payload["supported_lanes"] = sorted(set(payload.get("supported_lanes", [])))
        payload["supported_objectives"] = sorted(set(payload.get("supported_objectives", [])))
        object.__setattr__(self, "checksum", compute_sha256_hex(payload))
        return self


class TrialDevelopmentEvalContractV1(BaseModel):
    """Eval-facing contract surface shipped in the public scenario bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    phase_modules: tuple[PhaseModuleSpecV1, ...] = Field(..., min_length=1)
    request_schema_version: str = Field("v1", min_length=1)
    feasibility_bounds: tuple[SuperpopulationRequestBoundV1, ...] = Field(default_factory=tuple)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _hash(self) -> TrialDevelopmentEvalContractV1:
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        payload["phase_modules"] = sorted(payload.get("phase_modules", []), key=lambda m: str(m.get("phase_id", "")))
        feasibility_bounds = list(payload.get("feasibility_bounds", []) or [])
        feasibility_bounds.sort(key=lambda b: (str(b.get("phase_id", "")), int(b.get("max_sample_size", 0))))
        if feasibility_bounds:
            payload["feasibility_bounds"] = feasibility_bounds
        else:
            payload.pop("feasibility_bounds", None)
        object.__setattr__(self, "checksum", compute_sha256_hex(payload))
        return self


class TrialMaterializationAuditV1(BaseModel):
    """Audit record for one materialization request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    phase_id: str = Field(..., min_length=1)
    request_checksum: str = Field(..., min_length=64, max_length=64)
    seed: int
    realized_sample_size: int = Field(..., ge=0)
    realized_follow_up_days: int = Field(..., ge=1)
    feasibility_status: Literal["accepted", "rejected"]
    rejection_reason: str | None = None
    realized_arm_ids: tuple[str, ...] = Field(default_factory=tuple)
    realized_arm_counts: dict[str, int] = Field(default_factory=dict)
    realized_stratification_variables: tuple[str, ...] = Field(default_factory=tuple)
    realized_analysis_covariates: tuple[str, ...] = Field(default_factory=tuple)
    realized_subgroup_variables: tuple[str, ...] = Field(default_factory=tuple)
    realized_site_mix_summary: dict[str, float] = Field(default_factory=dict)
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _hash(self) -> TrialMaterializationAuditV1:
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        payload["realized_arm_ids"] = sorted(set(payload.get("realized_arm_ids", [])))
        payload["realized_arm_counts"] = {
            str(k): int(v) for k, v in sorted(payload.get("realized_arm_counts", {}).items())
        }
        payload["realized_stratification_variables"] = sorted(
            set(payload.get("realized_stratification_variables", []))
        )
        payload["realized_analysis_covariates"] = sorted(set(payload.get("realized_analysis_covariates", [])))
        payload["realized_subgroup_variables"] = sorted(set(payload.get("realized_subgroup_variables", [])))
        payload["realized_site_mix_summary"] = {
            str(k): float(v) for k, v in sorted(payload["realized_site_mix_summary"].items())
        }
        object.__setattr__(self, "checksum", compute_sha256_hex(payload))
        return self


class TrialMaterializationResultV1(BaseModel):
    """Materialization result referencing emitted artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    audit: TrialMaterializationAuditV1
    trial_tables_dir: str | None = None
    artifacts: tuple[str, ...] = Field(default_factory=tuple)
