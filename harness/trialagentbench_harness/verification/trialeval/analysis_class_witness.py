"""Derive executable TrialEval analysis classes from participant evidence only."""

from __future__ import annotations

import hashlib
import json
from typing import Literal
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.contracts.release.trialeval_runtime_surface import TrialEvalParticipantTaskV1
from trialagentbench_harness.contracts.scoring.modifier_evidence import MethodModifierV1
from trialagentbench_harness.contracts.submission.models import EffectScaleV1, EstimatorFamilyV1

AnalysisClassApplicabilityV1 = Literal[
    "fixed_prespecified",
    "public_candidate",
    "diagnostic_contingent",
    "sensitivity_option",
]


class PublicAnalysisClassV1(BaseModel):
    """One analysis class executable from the participant surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    estimator_family: EstimatorFamilyV1
    effect_scale: EffectScaleV1
    required_modifiers: tuple[MethodModifierV1, ...] = ()
    applicability: AnalysisClassApplicabilityV1
    evidence_paths: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _canonicalize(self) -> PublicAnalysisClassV1:
        """Canonicalize set-valued fields for stable comparison."""

        object.__setattr__(self, "required_modifiers", tuple(sorted(set(self.required_modifiers))))
        object.__setattr__(self, "evidence_paths", tuple(sorted(set(self.evidence_paths))))
        return self


class AnalysisClassWitnessV1(BaseModel):
    """Participant-only evidence that bounds executable, not necessarily scoreable, analysis classes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialeval.analysis_class_witness/v1"] = (
        "trialagentbench.trialeval.analysis_class_witness/v1"
    )
    task_id: str = Field(min_length=1)
    design_subtype: str = Field(min_length=1)
    primary_effect_scale: EffectScaleV1 | None
    primary_effect_scale_options: tuple[EffectScaleV1, ...] = Field(min_length=1)
    bounded_deviation_delta_options: tuple[float, ...] = ()
    source_sha256: dict[str, str]
    candidate_classes: tuple[PublicAnalysisClassV1, ...] = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_sources_and_checksum(self) -> AnalysisClassWitnessV1:
        """Require every cited public source and validate the witness checksum."""

        cited = {path for candidate in self.candidate_classes for path in candidate.evidence_paths}
        if not cited <= set(self.source_sha256):
            raise ValueError("Analysis-class candidates cite unhashed participant evidence.")
        if self.primary_effect_scale is not None and self.primary_effect_scale not in set(
            self.primary_effect_scale_options
        ):
            raise ValueError("The primary effect scale must occur in the public result-scale options.")
        if len(self.primary_effect_scale_options) != len(set(self.primary_effect_scale_options)):
            raise ValueError("Public primary effect-scale options must be unique.")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        if self.checksum != expected:
            raise ValueError("Analysis-class witness checksum mismatch.")
        return self


def _read_json_object(public: ZipFile, member: str) -> dict[str, object]:
    try:
        payload = json.loads(public.read(member))
    except KeyError as exc:
        raise FileNotFoundError(f"Missing participant evidence: {member}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Participant JSON member must contain an object: {member}")
    return {str(key): value for key, value in payload.items()}


def _candidate(
    *,
    estimator_family: EstimatorFamilyV1,
    effect_scale: EffectScaleV1,
    required_modifiers: tuple[MethodModifierV1, ...],
    applicability: AnalysisClassApplicabilityV1,
    evidence_paths: tuple[str, ...],
    rationale: str,
) -> PublicAnalysisClassV1:
    return PublicAnalysisClassV1(
        estimator_family=estimator_family,
        effect_scale=effect_scale,
        required_modifiers=required_modifiers,
        applicability=applicability,
        evidence_paths=evidence_paths,
        rationale=rationale,
    )


def _derive_public_analysis_classes_for_scale_v1(
    *,
    design_subtype: str,
    primary_effect_scale: EffectScaleV1,
    fixed_prespecified: bool,
    ipcw_executable: bool,
) -> tuple[PublicAnalysisClassV1, ...]:
    """Derive executable method classes for one participant-supported result scale."""

    supported_designs = {
        "individual_randomized",
        "pragmatic",
        "covariate_structure",
        "endpoint_ascertainment",
        "cluster_parallel",
        "stepped_wedge",
        "group_sequential",
    }
    if design_subtype not in supported_designs:
        raise ValueError(f"Unsupported participant design_subtype: {design_subtype!r}.")
    scale = primary_effect_scale
    common = ("task.json", "protocol_summary.json", "data_dictionary.json")
    primary_applicability: AnalysisClassApplicabilityV1 = (
        "fixed_prespecified" if fixed_prespecified else "public_candidate"
    )
    if design_subtype == "endpoint_ascertainment":
        return (
            _candidate(
                estimator_family="km",
                effect_scale=scale,
                required_modifiers=("ph_robust_fixed_horizon",),
                applicability=primary_applicability,
                evidence_paths=common,
                rationale=(
                    "Arm-specific Kaplan-Meier estimation targets the declared horizon risk when the released "
                    "endpoint definition does not require misclassification correction."
                ),
            ),
            _candidate(
                estimator_family="validated_endpoint",
                effect_scale=scale,
                required_modifiers=("misclassification_corrected",),
                applicability=primary_applicability,
                evidence_paths=(*common, "ascertainment_model.json", "endpoint_definition.json"),
                rationale=(
                    "The released internal-validation records identify endpoint error only where treatment-arm "
                    "and prognostic-stratum support is present."
                ),
            ),
        )
    if design_subtype == "cluster_parallel":
        modifiers: tuple[MethodModifierV1, ...] = (
            "cluster_robust_inference",
            "participant_population_target",
        )
        classes = [
            _candidate(
                estimator_family="cluster_parallel_participant_weighted",
                effect_scale=scale,
                required_modifiers=modifiers,
                applicability=primary_applicability,
                evidence_paths=common,
                rationale="The public protocol randomizes clusters and declares a participant-level target.",
            )
        ]
        if ipcw_executable:
            classes.append(
                _candidate(
                    estimator_family="cluster_parallel_participant_weighted",
                    effect_scale=scale,
                    required_modifiers=(*modifiers, "ipcw_adjusted"),
                    applicability="diagnostic_contingent",
                    evidence_paths=common,
                    rationale="The participant data support the declared baseline-covariate IPCW sensitivity.",
                )
            )
        return tuple(classes)
    if design_subtype == "stepped_wedge":
        return (
            _candidate(
                estimator_family="stepped_wedge_period_adjusted",
                effect_scale=scale,
                required_modifiers=(
                    "cluster_robust_inference",
                    "participant_population_target",
                    "stepped_wedge_period_adjusted",
                ),
                applicability=primary_applicability,
                evidence_paths=common,
                rationale="The rollout schedule requires period adjustment and cluster-aware participant inference.",
            ),
        )
    if design_subtype == "group_sequential":
        return (
            _candidate(
                estimator_family="km",
                effect_scale=scale,
                required_modifiers=(
                    "group_sequential_adjustment",
                    "ph_robust_fixed_horizon",
                ),
                applicability=primary_applicability,
                evidence_paths=common,
                rationale=(
                    "The public group-sequential plan applies its declared alpha-spending adjustment to a "
                    "Kaplan-Meier fixed-horizon risk contrast."
                ),
            ),
        )
    if scale == "standardized_risk_difference_tau_reference":
        return (
            _candidate(
                estimator_family="standardized_cox_g_computation",
                effect_scale=scale,
                required_modifiers=("reference_standardization",),
                applicability=primary_applicability,
                evidence_paths=common,
                rationale="The declared target is a risk contrast standardized to the released reference population.",
            ),
            _candidate(
                estimator_family="standardized_cox_g_computation",
                effect_scale=scale,
                required_modifiers=("flexible_model_form", "reference_standardization"),
                applicability="diagnostic_contingent",
                evidence_paths=common,
                rationale=(
                    "A participant-recomputable model-form diagnostic can require flexible standardization while "
                    "preserving the declared reference-population risk contrast."
                ),
            ),
        )
    if scale == "risk_difference_tau":
        classes = [
            _candidate(
                estimator_family="coxph_binary",
                effect_scale=scale,
                required_modifiers=(),
                applicability="diagnostic_contingent",
                evidence_paths=common,
                rationale=(
                    "A treatment-only Cox fit can imply the declared horizon risk contrast when proportional "
                    "hazards is compatible with the participant evidence."
                ),
            ),
            _candidate(
                estimator_family="km",
                effect_scale=scale,
                required_modifiers=("ph_robust_fixed_horizon",),
                applicability=primary_applicability,
                evidence_paths=common,
                rationale=(
                    "Arm-specific Kaplan-Meier estimation directly targets the declared fixed-horizon risk "
                    "contrast without requiring proportional hazards."
                ),
            ),
        ]
        if ipcw_executable:
            classes.append(
                _candidate(
                    estimator_family="km_ipcw",
                    effect_scale=scale,
                    required_modifiers=("ipcw_adjusted",),
                    applicability="diagnostic_contingent",
                    evidence_paths=common,
                    rationale="The participant data support a PH-robust IPCW horizon-risk sensitivity.",
                )
            )
        return tuple(classes)
    base_family: EstimatorFamilyV1 = "coxph_binary" if scale == "log_hr" else "km"
    base_modifiers: tuple[MethodModifierV1, ...] = (
        ("ph_robust_fixed_horizon",) if scale == "rmst_difference_tau" else ()
    )
    classes = [
        _candidate(
            estimator_family=base_family,
            effect_scale=scale,
            required_modifiers=base_modifiers,
            applicability=primary_applicability,
            evidence_paths=common,
            rationale="The estimator family directly implements the public task's declared effect scale.",
        )
    ]
    if ipcw_executable:
        ipcw_family: EstimatorFamilyV1 = "coxph_ipcw" if scale == "log_hr" else "km_ipcw"
        classes.append(
            _candidate(
                estimator_family=ipcw_family,
                effect_scale=scale,
                required_modifiers=("ipcw_adjusted",),
                applicability="diagnostic_contingent",
                evidence_paths=common,
                rationale="The participant data support the declared baseline-covariate IPCW sensitivity.",
            )
        )
    return tuple(classes)


def derive_public_analysis_classes_v1(
    *,
    design_subtype: str,
    primary_effect_scale: EffectScaleV1 | None,
    primary_effect_scale_options: tuple[EffectScaleV1, ...],
    bounded_deviation_delta_options: tuple[float, ...],
    ipcw_executable: bool,
) -> tuple[PublicAnalysisClassV1, ...]:
    """Derive executable candidates from explicit participant facts without inferring evaluator credit."""

    if primary_effect_scale is not None and primary_effect_scale not in set(primary_effect_scale_options):
        raise ValueError("The primary effect scale must be included in primary_effect_scale_options.")
    if len(primary_effect_scale_options) != len(set(primary_effect_scale_options)):
        raise ValueError("primary_effect_scale_options must be unique.")
    classes: list[PublicAnalysisClassV1] = []
    for effect_scale in primary_effect_scale_options:
        derived = _derive_public_analysis_classes_for_scale_v1(
            design_subtype=design_subtype,
            primary_effect_scale=effect_scale,
            fixed_prespecified=primary_effect_scale is not None,
            ipcw_executable=ipcw_executable,
        )
        classes.extend(derived)
    if bounded_deviation_delta_options and "risk_difference_tau" in set(primary_effect_scale_options):
        classes.append(
            _candidate(
                estimator_family="bounds",
                effect_scale="risk_difference_tau",
                required_modifiers=(
                    ("misclassification_corrected",) if design_subtype == "endpoint_ascertainment" else ()
                ),
                applicability="sensitivity_option",
                evidence_paths=("task.json", "study_brief.md"),
                rationale=(
                    "The participant contract supplies bounded-deviation parameters for a sensitivity analysis; "
                    "its use remains contingent on the released evidence."
                ),
            )
        )
    return tuple(classes)


def recover_analysis_class_v1(*, public: ZipFile, task_id: str) -> AnalysisClassWitnessV1:
    """Derive bounded executable analysis classes without evaluator or scorer inputs."""

    prefix = f"items/{task_id}/"
    task = TrialEvalParticipantTaskV1.model_validate(_read_json_object(public, f"{prefix}task.json"))
    if task.task_id != task_id:
        raise ValueError("Participant task_id does not match its archive path.")
    member_names = set(public.namelist())
    ipcw_input_surfaces = (
        {
            f"{prefix}data/ADTTE.parquet",
            f"{prefix}data/analysis_frame_covariates.parquet",
            f"{prefix}data/subject_operational_flags.parquet",
        },
        {
            f"{prefix}data/raw/baseline_characteristics.parquet",
            f"{prefix}data/raw/disposition.parquet",
            f"{prefix}data/raw/endpoint_adjudication.parquet",
            f"{prefix}data/raw/visits.parquet",
        },
    )
    ipcw_executable = any(required <= member_names for required in ipcw_input_surfaces)
    candidates = derive_public_analysis_classes_v1(
        design_subtype=task.design_subtype,
        primary_effect_scale=task.primary_effect_scale,
        primary_effect_scale_options=task.primary_effect_scale_options,
        bounded_deviation_delta_options=task.bounded_deviation_delta_options,
        ipcw_executable=ipcw_executable,
    )
    source_paths = sorted({path for candidate in candidates for path in candidate.evidence_paths})
    source_sha256: dict[str, str] = {}
    for relative_path in source_paths:
        member = f"{prefix}{relative_path}"
        try:
            source_sha256[relative_path] = hashlib.sha256(public.read(member)).hexdigest()
        except KeyError as exc:
            raise FileNotFoundError(f"Analysis-class evidence is missing: {member}") from exc
    payload = {
        "schema_id": "trialagentbench.trialeval.analysis_class_witness/v1",
        "task_id": task_id,
        "design_subtype": task.design_subtype,
        "primary_effect_scale": task.primary_effect_scale,
        "primary_effect_scale_options": task.primary_effect_scale_options,
        "bounded_deviation_delta_options": task.bounded_deviation_delta_options,
        "source_sha256": source_sha256,
        "candidate_classes": [candidate.model_dump(mode="json") for candidate in candidates],
    }
    checksum = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return AnalysisClassWitnessV1.model_validate({**payload, "checksum": checksum})


__all__ = [
    "AnalysisClassWitnessV1",
    "PublicAnalysisClassV1",
    "derive_public_analysis_classes_v1",
    "recover_analysis_class_v1",
]
