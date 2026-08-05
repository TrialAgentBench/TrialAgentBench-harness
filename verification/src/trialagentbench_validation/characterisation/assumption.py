"""Characterise analysis-relevant assumptions from a public TrialEval release."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal, TypeVar, cast
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pydantic.types import JsonValue
from scipy import stats

from trialagentbench_validation.characterisation.contracts import (
    AssumptionAnalysisBridge,
    AssumptionIdentificationResult,
    AssumptionPairContrast,
    AssumptionReleaseCharacterisation,
    AssumptionResponseFigureRow,
    AssumptionSeriesId,
    AssumptionSeriesIdentity,
    AssumptionTier,
    AssumptionTierSummary,
    MatchedAssumptionDesign,
    ReleaseCharacterisation,
    TrialProfile,
)
from trialagentbench_validation.contracts.scoring.public_estimand import (
    PublicEstimandContractV1,
)
from trialagentbench_validation.contracts.scoring.route_reference_inputs import (
    RouteReferenceInputRecordV1,
)
from trialagentbench_validation.contracts.scoring.route_references import (
    RouteReferenceRecordV1,
)
from trialagentbench_validation.io import canonical_payload_sha256, sha256_file
from trialagentbench_validation.trialeval.references.calculators import (
    PublicNumericBoundResultV1,
    recompute_public_numeric_result_v1,
)
from trialagentbench_validation.trialeval.references.stepped_wedge import (
    stepped_wedge_unadjusted_risk_difference_tau_with_uncertainty_v1,
)

_Z_95 = float(stats.norm.ppf(0.975))
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class _AssumptionEvidenceRecord(BaseModel):
    """Fields used from one public assumption-evidence record."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    assumption_id: str = Field(min_length=1)
    computed_band: str = Field(min_length=1)
    computed_status: str = Field(min_length=1)
    diagnosability: str = Field(min_length=1)
    severity_metric: float | None = None
    severity_metric_name: str | None = None
    supporting_metrics: dict[str, float] = Field(default_factory=dict)
    metric_units: dict[str, str] = Field(default_factory=dict)


class _AssumptionEvidenceManifest(BaseModel):
    """Fields used from one public assumption-evidence manifest."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    base_case_id: str = Field(pattern=r"^TE-S0[1-9]-A[1-4]$")
    item_id: str = Field(min_length=1)
    replicate_index: int = Field(ge=0)
    records: tuple[_AssumptionEvidenceRecord, ...] = Field(min_length=1)


class _AssumptionEvidencePayload(BaseModel):
    """Wrapped public assumption-evidence payload."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    manifest: _AssumptionEvidenceManifest


class _AssumptionEvidenceRow(BaseModel):
    """One public assumption-evidence domain row."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    task_id: str = Field(pattern=r"^TASK[0-9A-F]{32}$")
    payload: _AssumptionEvidencePayload


class _PublicEstimandPayload(BaseModel):
    """Wrapped public estimand contract."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    contract: PublicEstimandContractV1


class _PublicEstimandRow(BaseModel):
    """One public estimand domain row."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    task_id: str = Field(pattern=r"^TASK[0-9A-F]{32}$")
    payload: _PublicEstimandPayload


class _ReplayRecord(BaseModel):
    """Independent public replay result for one route reference."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    route_reference_id: str = Field(min_length=1)
    max_abs_difference: float = Field(ge=0)


class _ReplayEvidence(BaseModel):
    """Independent replay evidence distributed with the verification archive."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    records: tuple[_ReplayRecord, ...] = Field(min_length=1)


class _SeriesDefinition(BaseModel):
    """Scientific comparison definition for one Assumption-axis series."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    series_id: AssumptionSeriesId
    mechanism_id: str
    mechanism_label: str
    assumption_id: str | None
    default_method: str
    qualified_method: str
    result_unit: str


_SERIES = {
    row.series_id: row
    for row in (
        _SeriesDefinition(
            series_id="TE-S01",
            mechanism_id="nonproportional_hazards",
            mechanism_label="time variation in the treatment effect",
            assumption_id="proportional_hazards",
            default_method="observed:coxph_binary_breslow_risk_tau",
            qualified_method="observed:km",
            result_unit="risk difference",
        ),
        _SeriesDefinition(
            series_id="TE-S02",
            mechanism_id="prognostic_censoring",
            mechanism_label="prognostic censoring signal",
            assumption_id="censoring_ignorability",
            default_method="observed:km_rmst_tau",
            qualified_method="observed:km_ipcw_rmst_tau",
            result_unit="days",
        ),
        _SeriesDefinition(
            series_id="TE-S03",
            mechanism_id="treatment_nonadherence",
            mechanism_label="treated-arm recorded-dose nonadherence",
            assumption_id=None,
            default_method="observed:km",
            qualified_method="observed:km",
            result_unit="risk difference",
        ),
        _SeriesDefinition(
            series_id="TE-S04",
            mechanism_id="dependent_censoring",
            mechanism_label="prognostic censoring signal",
            assumption_id="censoring_ignorability",
            default_method="observed:km",
            qualified_method="observed:km_ipcw_baseline_cox",
            result_unit="risk difference",
        ),
        _SeriesDefinition(
            series_id="TE-S05",
            mechanism_id="nonlinear_prognostic_effect",
            mechanism_label="omitted nonlinear prognostic signal",
            assumption_id="model_form",
            default_method="observed:cox_linear_standardized_risk_tau_reference",
            qualified_method="observed:cox_rcs_standardized_risk_tau_reference",
            result_unit="standardized risk difference",
        ),
        _SeriesDefinition(
            series_id="TE-S06",
            mechanism_id="endpoint_misclassification",
            mechanism_label="endpoint classification disagreement",
            assumption_id="endpoint_ascertainment",
            default_method="observed:km",
            qualified_method="observed:validated_endpoint_joint_likelihood",
            result_unit="risk difference",
        ),
        _SeriesDefinition(
            series_id="TE-S07",
            mechanism_id="clustered_dependent_censoring",
            mechanism_label="prognostic censoring signal",
            assumption_id="censoring_ignorability",
            default_method="observed:cluster_parallel_participant_weighted_km",
            qualified_method="observed:cluster_parallel_participant_weighted_km_ipcw_baseline_cox",
            result_unit="risk difference",
        ),
        _SeriesDefinition(
            series_id="TE-S08",
            mechanism_id="secular_trend",
            mechanism_label="treatment-adjusted secular trend",
            assumption_id="secular_trend",
            default_method="observed:km",
            qualified_method="observed:stepped_wedge_period_cluster_adjusted_risk_tau",
            result_unit="risk difference",
        ),
        _SeriesDefinition(
            series_id="TE-S09",
            mechanism_id="sequential_monitoring",
            mechanism_label="group-sequential monitoring",
            assumption_id="sequential_design_adjustment",
            default_method="observed:km",
            qualified_method="observed:group_sequential_adjusted",
            result_unit="risk difference",
        ),
    )
}


@dataclass(frozen=True)
class _AssumptionProfile:
    """Fields required to analyse one matched Assumption-axis trial."""

    task_id: str
    independence_unit_id: str
    design_profile_id: str
    assumption_tier: AssumptionTier
    participant_count: int
    follow_up_horizon_days: float
    primary_paramcd: str


def characterise_assumption_release(
    *,
    participant_archive: Path,
    verification_archive: Path,
    release: ReleaseCharacterisation,
) -> AssumptionReleaseCharacterisation:
    """Build the finite-release Assumption-axis census from public evidence."""

    for path in (participant_archive, verification_archive):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(participant_archive) != release.participant_archive_sha256:
        raise ValueError(
            "participant archive does not match the release characterisation"
        )
    if sha256_file(verification_archive) != release.verification_archive_sha256:
        raise ValueError(
            "verification archive does not match the release characterisation"
        )
    with (
        ZipFile(participant_archive) as participant,
        ZipFile(verification_archive) as verification,
    ):
        evidence = {
            row.task_id: row
            for row in _read_wrapped_models(
                verification,
                "grader/domains/assumption_evidence.jsonl",
                _AssumptionEvidenceRow,
            )
        }
        estimands = {
            row.task_id: row.payload.contract
            for row in _read_wrapped_models(
                verification,
                "grader/domains/public_estimand_contract.jsonl",
                _PublicEstimandRow,
            )
        }
        references = _read_models(
            verification,
            "grader/domains/route_references.jsonl",
            RouteReferenceRecordV1,
        )
        inputs = _read_models(
            verification,
            "grader/domains/route_reference_inputs.jsonl",
            RouteReferenceInputRecordV1,
        )
        replay = _ReplayEvidence.model_validate(
            json.loads(
                verification.read("verification/public_route_replay_evidence.json")
            )
        )
        replay_errors = {
            row.route_reference_id: row.max_abs_difference for row in replay.records
        }
        bridges = tuple(
            _bridge(
                participant=participant,
                profile=profile,
                evidence=evidence[profile.task_id],
                estimand=estimands[profile.task_id],
                references=references,
                inputs=inputs,
                replay_errors=replay_errors,
            )
            for profile in release.profiles
        )
        identification_results = tuple(
            result
            for bridge in bridges
            for result in _identification_results(
                bridge=bridge,
                references=references,
                replay_errors=replay_errors,
            )
        )
    summaries = _summaries(bridges)
    return AssumptionReleaseCharacterisation(
        release_id=release.release_id,
        participant_archive_sha256=release.participant_archive_sha256,
        verification_archive_sha256=release.verification_archive_sha256,
        analysis_count=release.independent_trial_count,
        bridges=bridges,
        identification_results=identification_results,
        summaries=summaries,
    )


def characterise_matched_assumption_release(
    *,
    participant_archive: Path,
    verification_archive: Path,
    design: MatchedAssumptionDesign,
) -> AssumptionReleaseCharacterisation:
    """Analyse a pair-matched Assumption-axis response experiment."""

    for path in (participant_archive, verification_archive):
        if not path.is_file():
            raise FileNotFoundError(path)
    with (
        ZipFile(participant_archive) as participant,
        ZipFile(verification_archive) as verification,
    ):
        evidence = {
            row.task_id: row
            for row in _read_wrapped_models(
                verification,
                "grader/domains/assumption_evidence.jsonl",
                _AssumptionEvidenceRow,
            )
        }
        estimands = {
            row.task_id: row.payload.contract
            for row in _read_wrapped_models(
                verification,
                "grader/domains/public_estimand_contract.jsonl",
                _PublicEstimandRow,
            )
        }
        references = _read_models(
            verification,
            "grader/domains/route_references.jsonl",
            RouteReferenceRecordV1,
        )
        inputs = _read_models(
            verification,
            "grader/domains/route_reference_inputs.jsonl",
            RouteReferenceInputRecordV1,
        )
        replay = _ReplayEvidence.model_validate(
            json.loads(
                verification.read("verification/public_route_replay_evidence.json")
            )
        )
        replay_errors = {
            row.route_reference_id: row.max_abs_difference for row in replay.records
        }
        bridges: list[AssumptionAnalysisBridge] = []
        for identity in design.identities:
            for tier, task_id in sorted(identity.task_ids.items()):
                try:
                    task_estimand = estimands[task_id]
                    task_evidence = evidence[task_id]
                except KeyError as error:
                    raise ValueError(
                        f"matched design task {task_id!r} is absent from verification evidence"
                    ) from error
                task = _json_member(participant, f"items/{task_id}/task.json")
                participants = _parquet(
                    participant, f"items/{task_id}/data/ADSL.parquet"
                )
                profile = _AssumptionProfile(
                    task_id=task_id,
                    independence_unit_id=identity.random_stream_id,
                    design_profile_id=identity.design_profile_id,
                    assumption_tier=tier,
                    participant_count=len(participants),
                    follow_up_horizon_days=_positive_float(task, "primary_tau_dy"),
                    primary_paramcd=_required_text(task, "primary_paramcd"),
                )
                bridges.append(
                    _bridge(
                        participant=participant,
                        profile=profile,
                        evidence=task_evidence,
                        estimand=task_estimand,
                        references=references,
                        inputs=inputs,
                        replay_errors=replay_errors,
                    )
                )
        bridge_rows = tuple(bridges)
        _validate_identities(
            identities=design.identities,
            bridges=bridge_rows,
            estimands=estimands,
        )
    return AssumptionReleaseCharacterisation(
        release_id=design.release_id,
        analysis_scope="matched_response",
        participant_archive_sha256=sha256_file(participant_archive),
        verification_archive_sha256=sha256_file(verification_archive),
        analysis_count=design.analysis_count,
        identities=design.identities,
        bridges=bridge_rows,
        summaries=_summaries(bridge_rows),
        paired_contrasts=_paired_contrasts(bridge_rows),
    )


def write_assumption_release(
    output_dir: Path, result: AssumptionReleaseCharacterisation
) -> None:
    """Write canonical and tidy Assumption-axis results."""

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "assumption_characterisation.json").write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    if result.identities:
        _write_models(
            output_dir / "assumption_series_identities.csv",
            AssumptionSeriesIdentity,
            result.identities,
        )
        _write_models(
            output_dir / "assumption_paired_contrasts.csv",
            AssumptionPairContrast,
            result.paired_contrasts,
        )
        _write_models(
            output_dir / "assumption_response.csv",
            AssumptionResponseFigureRow,
            _assumption_response_rows(result.bridges),
        )
    _write_models(
        output_dir / "assumption_bridges.csv", AssumptionAnalysisBridge, result.bridges
    )
    if result.identification_results:
        _write_models(
            output_dir / "assumption_identification_results.csv",
            AssumptionIdentificationResult,
            result.identification_results,
        )
    _write_models(
        output_dir / "assumption_summaries.csv", AssumptionTierSummary, result.summaries
    )


_CONSEQUENCE_LABELS: dict[AssumptionSeriesId, str] = {
    "TE-S01": "Direct minus Cox-projected risk difference",
    "TE-S02": "Ordinary minus weighted RMST difference",
    "TE-S03": "Treatment-effect attenuation from reference",
    "TE-S04": "Weighted minus ordinary risk difference",
    "TE-S05": "Spline minus linear standardized risk",
    "TE-S06": "Routine minus validation-corrected risk",
    "TE-S07": "Weighted minus unweighted risk difference",
    "TE-S08": "Period-omitting minus period-adjusted risk",
}

_CONSEQUENCE_ORIENTATION: dict[AssumptionSeriesId, float] = {
    "TE-S01": -1.0,
    "TE-S02": 1.0,
    "TE-S03": 1.0,
    "TE-S04": -1.0,
    "TE-S05": -1.0,
    "TE-S06": 1.0,
    "TE-S07": -1.0,
    "TE-S08": 1.0,
}


def _assumption_response_rows(
    bridges: tuple[AssumptionAnalysisBridge, ...],
) -> tuple[AssumptionResponseFigureRow, ...]:
    grouped: dict[
        tuple[AssumptionSeriesId, AssumptionTier], list[AssumptionAnalysisBridge]
    ] = defaultdict(list)
    reference_by_replicate = {
        (row.series_id, row.replicate_index): row
        for row in bridges
        if row.assumption_tier == "A1"
    }
    for row in bridges:
        if row.assumption_tier != "A4":
            grouped[(row.series_id, row.assumption_tier)].append(row)
    rows: list[AssumptionResponseFigureRow] = []
    for (series_id, tier), group in sorted(grouped.items()):
        mechanism = [
            float(row.mechanism_value)
            for row in group
            if row.mechanism_value is not None
        ]
        if len(mechanism) != len(group):
            raise ValueError(
                f"{series_id}-{tier} requires a numeric observed mechanism"
            )
        mechanism_interval = _mean_interval(mechanism, lower_bound=0.0)
        consequence = [
            _analysis_consequence(
                row=row,
                reference=reference_by_replicate[(series_id, row.replicate_index)],
            )
            for row in group
        ]
        consequence_interval = _mean_interval(consequence)
        mechanism_units = {
            row.mechanism_unit for row in group if row.mechanism_unit is not None
        }
        consequence_units = {row.result_unit for row in group}
        if len(mechanism_units) != 1 or len(consequence_units) != 1:
            raise ValueError(f"{series_id}-{tier} response units are inconsistent")
        assert all(value is not None for value in mechanism_interval)
        assert all(value is not None for value in consequence_interval)
        rows.append(
            AssumptionResponseFigureRow(
                series_id=series_id,
                assumption_tier=cast(Literal["A1", "A2", "A3"], tier),
                trial_count=len(group),
                mechanism_value_mean=cast(float, mechanism_interval[0]),
                mechanism_value_interval_low=cast(float, mechanism_interval[1]),
                mechanism_value_interval_high=cast(float, mechanism_interval[2]),
                mechanism_label=_SERIES[series_id].mechanism_label,
                mechanism_unit=next(iter(mechanism_units)),
                consequence_value_mean=cast(float, consequence_interval[0]),
                consequence_interval_low=cast(float, consequence_interval[1]),
                consequence_interval_high=cast(float, consequence_interval[2]),
                consequence_unit=next(iter(consequence_units)),
                consequence_label=_CONSEQUENCE_LABELS[series_id],
            )
        )
    return tuple(rows)


def _analysis_consequence(
    *,
    row: AssumptionAnalysisBridge,
    reference: AssumptionAnalysisBridge,
) -> float:
    """Return the analysis consequence defined for one matched series."""

    if row.series_id == "TE-S03":
        if reference.default_value is None or row.default_value is None:
            raise ValueError("TE-S03 requires point-valued treatment-policy estimates")
        return abs(reference.default_value) - abs(row.default_value)
    if row.default_value is None:
        raise ValueError(
            f"{row.series_id}-{row.assumption_tier} requires two same-estimand point analyses"
        )
    return float(
        _CONSEQUENCE_ORIENTATION[row.series_id]
        * (row.default_value - row.qualified_value)
    )


def _bridge(
    *,
    participant: ZipFile,
    profile: TrialProfile | _AssumptionProfile,
    evidence: _AssumptionEvidenceRow,
    estimand: PublicEstimandContractV1,
    references: tuple[RouteReferenceRecordV1, ...],
    inputs: tuple[RouteReferenceInputRecordV1, ...],
    replay_errors: dict[str, float],
) -> AssumptionAnalysisBridge:
    base_case_id = evidence.payload.manifest.base_case_id
    series_id = cast(AssumptionSeriesId, base_case_id.rsplit("-", maxsplit=1)[0])
    definition = _SERIES[series_id]
    if evidence.payload.manifest.replicate_index + 1 < 1:
        raise ValueError("assumption replicate index is invalid")
    mechanism_value, mechanism_unit, diagnostic_status = _mechanism(
        participant=participant,
        task_id=profile.task_id,
        definition=definition,
        records=evidence.payload.manifest.records,
    )
    task_references = tuple(
        row
        for row in references
        if row.task_id == profile.task_id
        and row.lane_id == "primary_numeric.v1"
        and row.support_status == "official_supported"
    )
    task_inputs = tuple(row for row in inputs if row.task_id == profile.task_id)
    if not task_references or not task_inputs:
        raise ValueError(f"{profile.task_id} lacks public route-reference evidence")

    qualified_reference = _qualified_reference(
        task_references=task_references,
        definition=definition,
        assumption_tier=profile.assumption_tier,
    )
    qualified_value, qualified_se, qualified_low, qualified_high, qualified_shape = (
        _official_result(qualified_reference)
    )
    try:
        replay_error = replay_errors[qualified_reference.route_reference_id]
    except KeyError as error:
        raise ValueError(
            f"{profile.task_id} qualified route lacks independent replay evidence"
        ) from error
    default_incompatible = profile.assumption_tier == "A4"
    if default_incompatible:
        default_value = default_se = default_low = default_high = None
    else:
        default_reference = next(
            (
                row
                for row in task_references
                if row.estimator_method_id == definition.default_method
            ),
            None,
        )
        if default_reference is None:
            if definition.series_id == "TE-S08":
                default_value, default_se = _stepped_wedge_default(
                    participant=participant,
                    task_inputs=task_inputs,
                    paramcd=profile.primary_paramcd,
                    tau=profile.follow_up_horizon_days,
                )
                default_low = float(default_value - _Z_95 * default_se)
                default_high = float(default_value + _Z_95 * default_se)
                default_shape: Literal["point", "bound"] = "point"
            else:
                default_value, default_se, default_low, default_high, default_shape = (
                    _recomputed_result(
                        participant=participant,
                        task_inputs=task_inputs,
                        reference=qualified_reference,
                        requested_method=definition.default_method,
                    )
                )
        else:
            default_value, default_se, default_low, default_high, default_shape = (
                _official_result(default_reference)
            )
        if default_shape != "point":
            raise ValueError(f"{profile.task_id} default analysis must be point-valued")
    return AssumptionAnalysisBridge(
        task_id=profile.task_id,
        independence_unit_id=profile.independence_unit_id,
        series_id=series_id,
        replicate_index=evidence.payload.manifest.replicate_index + 1,
        assumption_tier=profile.assumption_tier,
        design_profile_id=profile.design_profile_id,
        participant_count=profile.participant_count,
        follow_up_horizon_days=profile.follow_up_horizon_days,
        endpoint_id=estimand.estimand.endpoint_id,
        estimand_id=estimand.estimand.estimand_id,
        effect_scale=estimand.declared_primary_effect_scale,
        mechanism_id=definition.mechanism_id,
        mechanism_label=definition.mechanism_label,
        mechanism_value=mechanism_value,
        mechanism_unit=mechanism_unit,
        mechanism_band=diagnostic_status,
        diagnostic_status=diagnostic_status,
        default_method=definition.default_method,
        default_status="incompatible" if default_incompatible else "estimated",
        default_value=default_value,
        default_standard_error=default_se,
        default_interval_low=default_low,
        default_interval_high=default_high,
        qualified_method=qualified_reference.estimator_method_id,
        qualified_shape=qualified_shape,
        qualified_value=qualified_value,
        qualified_standard_error=qualified_se,
        qualified_interval_low=qualified_low,
        qualified_interval_high=qualified_high,
        result_unit=definition.result_unit,
        absolute_analysis_difference=(
            None
            if default_value is None
            else abs(float(default_value) - float(qualified_value))
        ),
        default_rejects_null=_rejects_null(default_low, default_high),
        qualified_rejects_null=(
            _rejects_null(qualified_low, qualified_high)
            if qualified_shape == "point"
            else None
        ),
        qualified_replay_abs_error=replay_error,
        analysis_failure=False,
    )


def _identification_results(
    *,
    bridge: AssumptionAnalysisBridge,
    references: tuple[RouteReferenceRecordV1, ...],
    replay_errors: dict[str, float],
) -> tuple[AssumptionIdentificationResult, ...]:
    if bridge.assumption_tier != "A4" or bridge.series_id not in {"TE-S04", "TE-S06"}:
        return ()
    method_ids = (
        {"observed:tau_bounds_bounded_deviation", "observed:tau_bounds_worst_case"}
        if bridge.series_id == "TE-S04"
        else {
            "observed:validated_endpoint_bounded_deviation",
            "observed:validated_endpoint_worst_case",
        }
    )
    task_references = tuple(
        row
        for row in references
        if row.task_id == bridge.task_id
        and row.support_status == "official_supported"
        and row.answer_shape == "bound"
        and row.estimator_method_id in method_ids
    )
    if len(task_references) != 4:
        raise ValueError(
            f"{bridge.task_id} requires three bounded deviations and one worst-case range"
        )
    rows: list[AssumptionIdentificationResult] = []
    for reference in task_references:
        if reference.lower is None or reference.upper is None:
            raise ValueError(
                f"{reference.route_reference_id} lacks identified-set endpoints"
            )
        try:
            replay_error = replay_errors[reference.route_reference_id]
        except KeyError as error:
            raise ValueError(
                f"{reference.route_reference_id} lacks independent replay evidence"
            ) from error
        assumption: Literal["bounded_deviation", "worst_case"] = (
            "bounded_deviation"
            if reference.sensitivity_parameter is not None
            else "worst_case"
        )
        lower = float(reference.lower)
        upper = float(reference.upper)
        rows.append(
            AssumptionIdentificationResult(
                task_id=bridge.task_id,
                series_id=cast(Literal["TE-S04", "TE-S06"], bridge.series_id),
                replicate_index=bridge.replicate_index,
                model=(
                    "dependent_censoring"
                    if bridge.series_id == "TE-S04"
                    else "endpoint_validation_transport"
                ),
                assumption=assumption,
                sensitivity_parameter=reference.sensitivity_parameter,
                lower=lower,
                upper=upper,
                midpoint=(lower + upper) / 2.0,
                width=upper - lower,
                reference_role=cast(
                    Literal[
                        "required_primary",
                        "sensitivity_only",
                        "credit_eligible_primary_alternative",
                    ],
                    reference.variant_role,
                ),
                replay_absolute_error=replay_error,
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.sensitivity_parameter is None,
                float(row.sensitivity_parameter or 1.0),
            ),
        )
    )


def _mechanism(
    *,
    participant: ZipFile,
    task_id: str,
    definition: _SeriesDefinition,
    records: tuple[_AssumptionEvidenceRecord, ...],
) -> tuple[float | None, str | None, str]:
    if definition.series_id == "TE-S03":
        frame = _parquet(
            participant, f"items/{task_id}/data/subject_operational_flags.parquet"
        )
        required = {"TRTA", "MEAN_EXADH"}
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"{task_id} lacks treatment-use columns: {missing!r}")
        treatment = frame["TRTA"].astype("string").str.strip().str.casefold()
        if treatment.isna().any() or set(treatment.unique()) != {"control", "treated"}:
            raise ValueError(
                f"{task_id} treatment use requires control and treated randomized arms"
            )
        adherence = pd.to_numeric(
            frame.loc[treatment.eq("treated"), "MEAN_EXADH"],
            errors="raise",
        )
        if adherence.empty:
            raise ValueError(f"{task_id} lacks treated-arm adherence observations")
        if adherence.isna().any() or not adherence.between(0.0, 1.0).all():
            raise ValueError(f"{task_id} treated-arm dose adherence must lie in [0, 1]")
        value = float(1.0 - adherence.mean())
        return value, "proportion", "observed"
    if definition.assumption_id is None:
        raise ValueError(
            f"{definition.series_id} lacks an assumption evidence definition"
        )
    matches = tuple(
        row for row in records if row.assumption_id == definition.assumption_id
    )
    if len(matches) != 1:
        raise ValueError(
            f"{definition.series_id} requires one {definition.assumption_id} evidence record"
        )
    record = matches[0]
    if record.severity_metric is None:
        return None, None, record.computed_band
    if record.severity_metric_name is None:
        raise ValueError("numeric assumption severity requires its metric name")
    unit = record.metric_units.get(record.severity_metric_name)
    if unit is None:
        raise ValueError("numeric assumption severity requires an explicit unit")
    return float(record.severity_metric), unit.replace("_", " "), record.computed_band


def _qualified_reference(
    *,
    task_references: tuple[RouteReferenceRecordV1, ...],
    definition: _SeriesDefinition,
    assumption_tier: str,
) -> RouteReferenceRecordV1:
    if assumption_tier == "A4":
        matches = tuple(
            row
            for row in task_references
            if row.variant_role == "required_primary"
            and row.answer_shape in {"point", "bound"}
        )
    else:
        matches = tuple(
            row
            for row in task_references
            if row.estimator_method_id == definition.qualified_method
            and row.answer_shape == "point"
        )
    if len(matches) != 1:
        raise ValueError(
            f"qualified public route does not resolve uniquely for {task_references[0].task_id}"
        )
    return matches[0]


def _official_result(
    reference: RouteReferenceRecordV1,
) -> tuple[float, float | None, float, float, Literal["point", "bound"]]:
    if reference.answer_shape == "bound":
        if reference.lower is None or reference.upper is None:
            raise ValueError(f"{reference.task_id} bound route lacks endpoints")
        value = float((reference.lower + reference.upper) / 2.0)
        return value, None, float(reference.lower), float(reference.upper), "bound"
    if (
        reference.answer_shape != "point"
        or reference.value is None
        or reference.standard_error is None
        or reference.ci_low is None
        or reference.ci_high is None
    ):
        raise ValueError(
            f"{reference.task_id} point route lacks its estimate or uncertainty"
        )
    return (
        float(reference.value),
        float(reference.standard_error),
        float(reference.ci_low),
        float(reference.ci_high),
        "point",
    )


def _recomputed_result(
    *,
    participant: ZipFile,
    task_inputs: tuple[RouteReferenceInputRecordV1, ...],
    reference: RouteReferenceRecordV1,
    requested_method: str,
) -> tuple[float, float | None, float, float, Literal["point", "bound"]]:
    matching_inputs = tuple(
        row for row in task_inputs if row.estimator_method_id == requested_method
    )
    source_input = matching_inputs[0] if matching_inputs else task_inputs[0]
    replay_reference = reference.model_copy(
        update={
            "estimator_method_id": requested_method,
            "answer_shape": "point",
            "sensitivity_parameter": None,
            "lower": None,
            "upper": None,
        }
    )
    replay_input = source_input.model_copy(
        update={"estimator_method_id": requested_method, "sensitivity_parameter": None}
    )
    if (
        reference.answer_shape == "bound"
        and requested_method == reference.estimator_method_id
    ):
        replay_reference = reference
        replay_input = source_input
    try:
        result, standard_error = recompute_public_numeric_result_v1(
            public=participant,
            reference_input=replay_input,
            route_reference=replay_reference,
        )
    except ValueError:
        if requested_method != "observed:group_sequential_adjusted":
            raise
        replay_reference = replay_reference.model_copy(
            update={"estimator_method_id": "observed:km"}
        )
        replay_input = replay_input.model_copy(
            update={"estimator_method_id": "observed:km"}
        )
        result, _ = recompute_public_numeric_result_v1(
            public=participant,
            reference_input=replay_input,
            route_reference=replay_reference,
        )
        standard_error = reference.standard_error
    if isinstance(result, PublicNumericBoundResultV1):
        value = float((result.lower + result.upper) / 2.0)
        return value, None, float(result.lower), float(result.upper), "bound"
    if not isinstance(result, float):
        raise ValueError(
            f"{reference.task_id} assumption route must be scalar or bound"
        )
    if standard_error is None:
        if reference.standard_error is None:
            raise ValueError(f"{reference.task_id} point route lacks uncertainty")
        standard_error = float(reference.standard_error)
    low = float(result - _Z_95 * standard_error)
    high = float(result + _Z_95 * standard_error)
    return float(result), float(standard_error), low, high, "point"


def _stepped_wedge_default(
    *,
    participant: ZipFile,
    task_inputs: tuple[RouteReferenceInputRecordV1, ...],
    paramcd: str,
    tau: float,
) -> tuple[float, float]:
    paths = {
        table.rel_path
        for reference_input in task_inputs
        for table in reference_input.required_table_refs
    }
    adsl_paths = tuple(path for path in paths if path.endswith("/ADSL.parquet"))
    adtte_paths = tuple(path for path in paths if path.endswith("/ADTTE.parquet"))
    if len(adsl_paths) != 1 or len(adtte_paths) != 1:
        raise ValueError(
            "stepped-wedge default analysis requires one ADSL and one ADTTE table"
        )
    value, standard_error = (
        stepped_wedge_unadjusted_risk_difference_tau_with_uncertainty_v1(
            adsl=_parquet(participant, adsl_paths[0]),
            adtte=_parquet(participant, adtte_paths[0]),
            paramcd=paramcd,
            tau=tau,
        )
    )
    return float(value), float(standard_error)


def _summaries(
    bridges: tuple[AssumptionAnalysisBridge, ...],
) -> tuple[AssumptionTierSummary, ...]:
    groups: dict[tuple[AssumptionSeriesId, str], list[AssumptionAnalysisBridge]] = (
        defaultdict(list)
    )
    for row in bridges:
        groups[(row.series_id, row.assumption_tier)].append(row)
    rows: list[AssumptionTierSummary] = []
    for (series_id, tier), group in sorted(groups.items()):
        mechanism = [
            float(row.mechanism_value)
            for row in group
            if row.mechanism_value is not None
        ]
        differences = [
            float(row.absolute_analysis_difference)
            for row in group
            if row.absolute_analysis_difference is not None
        ]
        mechanism_interval = _mean_interval(mechanism, lower_bound=0.0)
        difference_interval = _mean_interval(differences, lower_bound=0.0)
        mechanism_units = {
            row.mechanism_unit for row in group if row.mechanism_unit is not None
        }
        result_units = {row.result_unit for row in group}
        if len(mechanism_units) > 1 or len(result_units) != 1:
            raise ValueError(f"{series_id}-{tier} summary units are inconsistent")
        rows.append(
            AssumptionTierSummary(
                series_id=series_id,
                assumption_tier=cast(Literal["A1", "A2", "A3", "A4"], tier),
                trial_count=len(group),
                mechanism_value_mean=mechanism_interval[0],
                mechanism_value_interval_low=mechanism_interval[1],
                mechanism_value_interval_high=mechanism_interval[2],
                mechanism_unit=next(iter(mechanism_units), None),
                mean_absolute_analysis_difference=difference_interval[0],
                difference_interval_low=difference_interval[1],
                difference_interval_high=difference_interval[2],
                result_unit=next(iter(result_units)),
                default_rejection_fraction=_fraction(
                    row.default_rejects_null
                    for row in group
                    if row.default_rejects_null is not None
                ),
                qualified_rejection_fraction=_fraction(
                    row.qualified_rejects_null
                    for row in group
                    if row.qualified_rejects_null is not None
                ),
                analysis_failure_count=sum(row.analysis_failure for row in group),
                uncertainty_method=(
                    "t_interval_across_independent_trials"
                    if mechanism or differences
                    else "not_applicable"
                ),
            )
        )
    return tuple(rows)


def _paired_contrasts(
    bridges: tuple[AssumptionAnalysisBridge, ...],
) -> tuple[AssumptionPairContrast, ...]:
    cells = {
        (row.series_id, row.replicate_index, row.assumption_tier): row
        for row in bridges
    }
    reference_by_replicate = {
        (row.series_id, row.replicate_index): row
        for row in bridges
        if row.assumption_tier == "A1"
    }
    tiers_by_series: dict[AssumptionSeriesId, set[AssumptionTier]] = defaultdict(set)
    replicates_by_series: dict[AssumptionSeriesId, set[int]] = defaultdict(set)
    for row in bridges:
        tiers_by_series[row.series_id].add(row.assumption_tier)
        replicates_by_series[row.series_id].add(row.replicate_index)
    rows: list[AssumptionPairContrast] = []
    for series_id, tier_set in sorted(tiers_by_series.items()):
        tiers = tuple(sorted(tier_set))
        for lower_tier, upper_tier in zip(tiers, tiers[1:], strict=False):
            mechanism_changes: list[float] = []
            consequence_changes: list[float] = []
            default_value_changes: list[float] = []
            default_magnitude_changes: list[float] = []
            mechanism_units: set[str] = set()
            result_units: set[str] = set()
            for replicate_index in sorted(replicates_by_series[series_id]):
                lower = cells[(series_id, replicate_index, lower_tier)]
                upper = cells[(series_id, replicate_index, upper_tier)]
                if (
                    lower.mechanism_value is None
                    or upper.mechanism_value is None
                    or lower.default_value is None
                    or upper.default_value is None
                    or lower.mechanism_unit is None
                    or upper.mechanism_unit is None
                ):
                    raise ValueError(
                        f"{series_id} matched response requires numeric adjacent tiers"
                    )
                mechanism_changes.append(upper.mechanism_value - lower.mechanism_value)
                reference = reference_by_replicate[(series_id, replicate_index)]
                consequence_changes.append(
                    _analysis_consequence(row=upper, reference=reference)
                    - _analysis_consequence(row=lower, reference=reference)
                )
                default_value_changes.append(upper.default_value - lower.default_value)
                default_magnitude_changes.append(
                    abs(upper.default_value) - abs(lower.default_value)
                )
                mechanism_units.update((lower.mechanism_unit, upper.mechanism_unit))
                result_units.update((lower.result_unit, upper.result_unit))
            if len(mechanism_units) != 1 or len(result_units) != 1:
                raise ValueError(f"{series_id} matched response units are inconsistent")
            mechanism = _mean_interval(mechanism_changes)
            consequence = _mean_interval(consequence_changes)
            default_value = _mean_interval(default_value_changes)
            default_magnitude = _mean_interval(default_magnitude_changes)
            assert all(value is not None for value in mechanism)
            assert all(value is not None for value in consequence)
            assert all(value is not None for value in default_value)
            assert all(value is not None for value in default_magnitude)
            rows.append(
                AssumptionPairContrast(
                    series_id=series_id,
                    lower_tier=cast(Literal["A1", "A2"], lower_tier),
                    upper_tier=cast(Literal["A2", "A3"], upper_tier),
                    trial_pair_count=len(mechanism_changes),
                    mechanism_change_mean=cast(float, mechanism[0]),
                    mechanism_change_interval_low=cast(float, mechanism[1]),
                    mechanism_change_interval_high=cast(float, mechanism[2]),
                    mechanism_unit=next(iter(mechanism_units)),
                    consequence_change_mean=cast(float, consequence[0]),
                    consequence_change_interval_low=cast(float, consequence[1]),
                    consequence_change_interval_high=cast(float, consequence[2]),
                    default_value_change_mean=cast(float, default_value[0]),
                    default_value_change_interval_low=cast(float, default_value[1]),
                    default_value_change_interval_high=cast(float, default_value[2]),
                    default_magnitude_change_mean=cast(float, default_magnitude[0]),
                    default_magnitude_change_interval_low=cast(
                        float, default_magnitude[1]
                    ),
                    default_magnitude_change_interval_high=cast(
                        float, default_magnitude[2]
                    ),
                    result_unit=next(iter(result_units)),
                )
            )
    return tuple(rows)


def _validate_identities(
    *,
    identities: tuple[AssumptionSeriesIdentity, ...],
    bridges: tuple[AssumptionAnalysisBridge, ...],
    estimands: dict[str, PublicEstimandContractV1],
) -> None:
    groups: dict[tuple[AssumptionSeriesId, int], list[AssumptionAnalysisBridge]] = (
        defaultdict(list)
    )
    for bridge in bridges:
        groups[(bridge.series_id, bridge.replicate_index)].append(bridge)
    declared = {(row.series_id, row.replicate_index): row for row in identities}
    for (series_id, replicate_index), group in sorted(groups.items()):
        invariant_rows = {
            (
                row.design_profile_id,
                estimands[row.task_id].estimand.population,
                row.endpoint_id,
                row.estimand_id,
                row.effect_scale,
                row.default_method,
                row.participant_count,
                row.follow_up_horizon_days,
            )
            for row in group
        }
        if len(invariant_rows) != 1:
            raise ValueError(
                f"{series_id} replicate {replicate_index} changes a matched scientific identity"
            )
        (
            design_profile_id,
            population,
            endpoint_id,
            estimand_id,
            effect_scale,
            default_method,
            participant_count,
            follow_up_horizon_days,
        ) = next(iter(invariant_rows))
        identity = declared[(series_id, replicate_index)]
        identity_payload = {
            "series_id": series_id,
            "replicate_index": replicate_index,
            "design_profile_id": design_profile_id,
            "population": population,
            "endpoint_id": endpoint_id,
            "estimand_id": estimand_id,
            "effect_scale": effect_scale,
            "default_method": default_method,
            "participant_count": participant_count,
            "follow_up_horizon_days": follow_up_horizon_days,
        }
        observed_tasks = {row.assumption_tier: row.task_id for row in group}
        expected = identity.model_dump(
            mode="json",
            include=set(identity_payload),
        )
        if identity_payload != expected:
            raise ValueError(
                f"{series_id} replicate {replicate_index} disagrees with its declared identity"
            )
        if observed_tasks != identity.task_ids:
            raise ValueError(
                f"{series_id} replicate {replicate_index} task mapping is inconsistent"
            )
        if identity.identity_sha256 != canonical_payload_sha256(
            cast(JsonValue, identity_payload)
        ):
            raise ValueError(
                f"{series_id} replicate {replicate_index} identity checksum is invalid"
            )


def _mean_interval(
    values: list[float],
    *,
    lower_bound: float | None = None,
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, len(array) - 1) * stats.sem(array))
    low = mean - half
    if lower_bound is not None:
        low = max(lower_bound, low)
    return mean, low, mean + half


def _fraction(values: Iterable[bool]) -> float | None:
    rows = tuple(bool(value) for value in values)
    return None if not rows else float(np.mean(rows))


def _rejects_null(low: float | None, high: float | None) -> bool | None:
    if low is None or high is None:
        return None
    return bool(low > 0.0 or high < 0.0)


def _read_models(
    archive: ZipFile,
    member: str,
    model: type[_ModelT],
) -> tuple[_ModelT, ...]:
    try:
        payload = archive.read(member).decode("utf-8")
    except KeyError as error:
        raise FileNotFoundError(member) from error
    return tuple(
        model.model_validate(json.loads(line))
        for line in payload.splitlines()
        if line.strip()
    )


def _read_wrapped_models(
    archive: ZipFile,
    member: str,
    model: type[_ModelT],
) -> tuple[_ModelT, ...]:
    return _read_models(archive, member, model)


def _parquet(archive: ZipFile, member: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(BytesIO(archive.read(member)))
    except KeyError as error:
        raise FileNotFoundError(member) from error


def _json_member(archive: ZipFile, member: str) -> dict[str, object]:
    try:
        payload = json.loads(archive.read(member))
    except KeyError as error:
        raise FileNotFoundError(member) from error
    if not isinstance(payload, dict):
        raise ValueError(f"{member} must contain a JSON object")
    return payload


def _required_text(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_float(payload: dict[str, object], field: str) -> float:
    value = payload.get(field)
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive number")
    return float(value)


def _write_models(
    path: Path,
    model: type[_ModelT],
    rows: tuple[_ModelT, ...],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(model.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


__all__ = [
    "characterise_matched_assumption_release",
    "characterise_assumption_release",
    "write_assumption_release",
]
