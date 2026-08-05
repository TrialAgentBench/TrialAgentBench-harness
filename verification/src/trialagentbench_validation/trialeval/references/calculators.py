"""Public-evidence numeric calculators for TrialEval reference replay."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast
from zipfile import ZipFile

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from trialagentbench_validation.analysis.delete_group_jackknife import (
    DELETE_GROUP_COUNT_V1,
    balanced_delete_groups_v1,
    delete_group_standard_error_v1,
)
from trialagentbench_validation.contracts.scoring.route_reference_inputs import (
    RouteReferenceInputRecordV1,
)
from trialagentbench_validation.contracts.scoring.route_references import (
    RouteReferenceRecordV1,
)
from trialagentbench_validation.contracts.v1_scope import (
    TRIALEVAL_STANDARDIZED_RISK_BASELINE_COVARIATES_V1,
)
from trialagentbench_validation.trialeval.references.endpoint import (
    validated_endpoint_bounds_v1,
    validated_endpoint_point_and_standard_error_v1,
)
from trialagentbench_validation.trialeval.references.io import (
    has_public_reconstruction_tables_v1 as _has_public_reconstruction_tables,
)
from trialagentbench_validation.trialeval.references.io import (
    public_has_table_suffixes_v1 as _public_has_table_suffixes,
)
from trialagentbench_validation.trialeval.references.io import (
    read_covariate_surface_table_v1 as _read_covariate_surface_table,
)
from trialagentbench_validation.trialeval.references.io import (
    read_json_from_public_v1 as _read_json_from_public,
)
from trialagentbench_validation.trialeval.references.io import (
    read_required_parquet_v1 as _read_required_parquet,
)
from trialagentbench_validation.trialeval.references.io import (
    read_required_table_by_suffix_v1 as _read_required_table_by_suffix,
)
from trialagentbench_validation.trialeval.references.io import (
    read_treatment_surface_table_v1 as _read_treatment_surface_table,
)
from trialagentbench_validation.trialeval.references.io import (
    required_positive_float_v1 as _required_positive_float,
)
from trialagentbench_validation.trialeval.references.io import (
    required_str_v1 as _required_str,
)
from trialagentbench_validation.trialeval.references.standardized import (
    cox_linear_standardized_risk_difference_tau_reference_v1,
    cox_linear_standardized_risk_difference_tau_reference_with_uncertainty_v1,
    cox_rcs_standardized_risk_difference_tau_reference_v1,
    cox_rcs_standardized_risk_difference_tau_reference_with_uncertainty_v1,
)
from trialagentbench_validation.trialeval.references.stepped_wedge import (
    stepped_wedge_period_adjusted_risk_difference_tau_v1,
    stepped_wedge_period_adjusted_risk_difference_tau_with_uncertainty_v1,
)
from trialagentbench_validation.trialeval.references.survival import (
    _coxph_binary_breslow_newton,
    _coxph_binary_breslow_risk_difference_tau,
    _km_risk_rmst_and_se,
    _weighted_km_risk_and_rmst,
)

SUPPORTED_PUBLIC_NUMERIC_METHODS_V1 = frozenset(
    {
        "observed:cluster_parallel_participant_weighted_km",
        "observed:cluster_parallel_participant_weighted_km_ipcw_baseline_cox",
        "observed:cox_linear_standardized_risk_tau_reference",
        "observed:cox_rcs_standardized_risk_tau_reference",
        "observed:coxph_binary_breslow",
        "observed:coxph_binary_breslow_ipcw_baseline_cox",
        "observed:coxph_binary_breslow_risk_tau",
        "observed:group_sequential_adjusted",
        "observed:km",
        "observed:km_ipcw_baseline_cox",
        "observed:km_ipcw_rmst_tau",
        "observed:km_rmst_tau",
        "observed:stepped_wedge_period_cluster_adjusted_risk_tau",
        "observed:tau_bounds_bounded_deviation",
        "observed:tau_bounds_worst_case",
        "observed:validated_endpoint_bounded_deviation",
        "observed:validated_endpoint_joint_likelihood",
        "observed:validated_endpoint_worst_case",
    }
)

IPCW_SUPPORT_MIN_SURVIVAL_PASS_V1 = 0.05
IPCW_SUPPORT_MAX_WEIGHT_PASS_V1 = 20.0
IPCW_SUPPORT_MIN_ESS_RATIO_PASS_V1 = 0.50
IPCW_SUPPORT_MIN_SURVIVAL_FAILURE_V1 = 0.01
IPCW_SUPPORT_MAX_WEIGHT_FAILURE_V1 = 100.0
IPCW_SUPPORT_MIN_ESS_RATIO_FAILURE_V1 = 0.20


class PublicIPCWArmSupportV1(BaseModel):
    """Event-time censoring support over one randomized arm's target population."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluated_event_time_count: int = Field(ge=1)
    minimum_fitted_censoring_survival: float = Field(gt=0.0, le=1.0)
    maximum_weight: float = Field(ge=1.0)
    minimum_effective_sample_size_ratio: float = Field(gt=0.0, le=1.0)


class PublicIPCWSupportDiagnosticsV1(BaseModel):
    """Independent implementation of the frozen public IPCW support contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    support_by_arm: dict[str, PublicIPCWArmSupportV1] = Field(min_length=2)
    support_status: (
        Literal["pass", "intermediate", "practical_positivity_failure"] | None
    ) = None

    @model_validator(mode="after")
    def _classify_support(self) -> PublicIPCWSupportDiagnosticsV1:
        support_by_arm = {
            str(arm): row for arm, row in sorted(self.support_by_arm.items())
        }
        passes = all(
            row.minimum_fitted_censoring_survival >= IPCW_SUPPORT_MIN_SURVIVAL_PASS_V1
            and row.maximum_weight <= IPCW_SUPPORT_MAX_WEIGHT_PASS_V1
            and row.minimum_effective_sample_size_ratio
            >= IPCW_SUPPORT_MIN_ESS_RATIO_PASS_V1
            for row in support_by_arm.values()
        )
        fails = any(
            row.minimum_fitted_censoring_survival < IPCW_SUPPORT_MIN_SURVIVAL_FAILURE_V1
            or row.maximum_weight > IPCW_SUPPORT_MAX_WEIGHT_FAILURE_V1
            or row.minimum_effective_sample_size_ratio
            < IPCW_SUPPORT_MIN_ESS_RATIO_FAILURE_V1
            for row in support_by_arm.values()
        )
        resolved = (
            "pass"
            if passes
            else ("practical_positivity_failure" if fails else "intermediate")
        )
        if self.support_status is not None and self.support_status != resolved:
            raise ValueError(
                "Declared IPCW support status disagrees with independent diagnostics."
            )
        object.__setattr__(self, "support_by_arm", support_by_arm)
        object.__setattr__(self, "support_status", resolved)
        return self


def supported_public_numeric_method_ids_v1() -> frozenset[str]:
    """Return estimator methods implemented by the independent calculators."""

    return SUPPORTED_PUBLIC_NUMERIC_METHODS_V1


class PublicNumericBoundResultV1(BaseModel):
    """Bound-valued public numeric replay result."""

    model_config = ConfigDict(extra="forbid")

    lower: float
    upper: float


class PublicNumericVectorComponentResultV1(BaseModel):
    """One component of a vector-valued public replay result."""

    model_config = ConfigDict(extra="forbid")

    component_id: str
    value: float


class PublicNumericVectorResultV1(BaseModel):
    """Vector-valued public replay result reserved for qualified vector calculators."""

    model_config = ConfigDict(extra="forbid")

    components: tuple[PublicNumericVectorComponentResultV1, ...]


class _KmFamilyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_difference: float
    rmst_difference: float
    risk_difference_se: float
    rmst_difference_se: float


def numeric_replay_family_id_v1(method_id: str) -> str:
    """Return the shared computation family for one public estimator method."""

    if method_id in {"observed:km_ipcw_baseline_cox", "observed:km_ipcw_rmst_tau"}:
        return "ipcw_km"
    if method_id == "observed:cluster_parallel_participant_weighted_km":
        return "cluster_participant_weighted_km"
    if (
        method_id
        == "observed:cluster_parallel_participant_weighted_km_ipcw_baseline_cox"
    ):
        return "cluster_participant_weighted_ipcw_km"
    return str(method_id)


def _public_ipcw_covariate_columns(
    *, public: ZipFile, reference_input: RouteReferenceInputRecordV1, adsl: pd.DataFrame
) -> tuple[str, ...]:
    reference = _read_required_table_by_suffix(
        public=public,
        reference_input=reference_input,
        suffix="reference_population_covariates.parquet",
    )
    columns = tuple(
        sorted(
            column
            for column in {str(value) for value in adsl.columns}
            & {str(value) for value in reference.columns}
            if column not in {"USUBJID", "REFERENCE_ID"}
        )
    )
    if not columns:
        raise ValueError(
            "IPCW public replay found no shared trial/reference baseline covariates."
        )
    return columns


def ipcw_km_contrasts_with_uncertainty_v1(
    *,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    paramcd: str,
    tau: float,
    control_arm_id: str,
    treated_arm_id: str,
    baseline_covariate_columns: tuple[str, ...],
) -> tuple[float, float, float, float]:
    """Independently replay IPCW risk/RMST contrasts and total uncertainty."""

    result = _ipcw_km_family_from_frame(
        merged=_ipcw_analysis_frame(
            adsl=adsl,
            adtte=adtte,
            paramcd=paramcd,
            baseline_covariate_columns=baseline_covariate_columns,
        ),
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
        baseline_covariate_columns=baseline_covariate_columns,
        _include_nuisance_uncertainty=True,
    )
    return (
        float(result.risk_difference),
        float(result.rmst_difference),
        float(result.risk_difference_se),
        float(result.rmst_difference_se),
    )


def recompute_public_numeric_value_v1(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
    route_reference: RouteReferenceRecordV1,
) -> float | PublicNumericBoundResultV1 | PublicNumericVectorResultV1:
    """Recompute one supported numeric reference value from public evidence."""

    if route_reference.estimator_method_id in {"observed:km", "observed:km_rmst_tau"}:
        recomputed = _recompute_public_km_family(
            public=public, reference_input=reference_input
        )
        return (
            recomputed.rmst_difference
            if route_reference.effect_scale == "rmst_difference_tau"
            else recomputed.risk_difference
        )
    if (
        route_reference.estimator_method_id
        == "observed:cluster_parallel_participant_weighted_km"
    ):
        if route_reference.effect_scale not in {
            "risk_difference_tau",
            "rmst_difference_tau",
        }:
            raise ValueError(
                "Cluster participant-weighted KM requires a risk-difference or RMST effect scale."
            )
        recomputed = _recompute_public_cluster_participant_weighted_km(
            public=public, reference_input=reference_input
        )
        return (
            recomputed.rmst_difference
            if route_reference.effect_scale == "rmst_difference_tau"
            else recomputed.risk_difference
        )
    if route_reference.estimator_method_id in {
        "observed:km_ipcw_baseline_cox",
        "observed:km_ipcw_rmst_tau",
    }:
        recomputed = _recompute_public_ipcw_km_family(
            public=public, reference_input=reference_input
        )
        return (
            recomputed.rmst_difference
            if route_reference.effect_scale == "rmst_difference_tau"
            else recomputed.risk_difference
        )
    if (
        route_reference.estimator_method_id
        == "observed:coxph_binary_breslow_ipcw_baseline_cox"
    ):
        if route_reference.effect_scale != "log_hr":
            raise ValueError(
                "IPCW Cox public recomputation requires effect_scale='log_hr'."
            )
        return _recompute_public_ipcw_coxph_binary_breslow(
            public=public, reference_input=reference_input
        )
    if (
        route_reference.estimator_method_id
        == "observed:cluster_parallel_participant_weighted_km_ipcw_baseline_cox"
    ):
        if route_reference.effect_scale not in {
            "risk_difference_tau",
            "rmst_difference_tau",
        }:
            raise ValueError(
                "Cluster-IPCW participant-weighted KM requires a risk-difference or RMST effect scale."
            )
        recomputed = _recompute_public_cluster_participant_weighted_ipcw_km(
            public=public, reference_input=reference_input
        )
        return (
            recomputed.rmst_difference
            if route_reference.effect_scale == "rmst_difference_tau"
            else recomputed.risk_difference
        )
    if (
        route_reference.estimator_method_id
        == "observed:validated_endpoint_joint_likelihood"
    ):
        if route_reference.effect_scale != "risk_difference_tau":
            raise ValueError(
                "Validated-endpoint public recomputation requires risk_difference_tau."
            )
        return validated_endpoint_point_and_standard_error_v1(
            public=public,
            reference_input=reference_input,
        )[0]
    if route_reference.estimator_method_id == "observed:coxph_binary_breslow":
        if route_reference.effect_scale != "log_hr":
            raise ValueError(
                "Cox Breslow public recomputation requires effect_scale='log_hr'."
            )
        return _recompute_public_coxph_binary_breslow_with_uncertainty(
            public=public,
            reference_input=reference_input,
        )[0]
    if route_reference.estimator_method_id == "observed:coxph_binary_breslow_risk_tau":
        if route_reference.effect_scale != "risk_difference_tau":
            raise ValueError(
                "Cox fixed-horizon public recomputation requires risk_difference_tau."
            )
        return _recompute_public_coxph_binary_breslow_risk_tau_with_uncertainty(
            public=public,
            reference_input=reference_input,
        )[0]
    if route_reference.estimator_method_id in {
        "observed:cox_linear_standardized_risk_tau_reference",
        "observed:cox_rcs_standardized_risk_tau_reference",
    }:
        if route_reference.effect_scale != "standardized_risk_difference_tau_reference":
            raise ValueError(
                "Standardized-risk public recomputation requires standardized_risk_difference_tau_reference."
            )
        return _recompute_public_standardized_risk(
            public=public,
            reference_input=reference_input,
            method_id=route_reference.estimator_method_id,
        )
    if (
        route_reference.estimator_method_id
        == "observed:stepped_wedge_period_cluster_adjusted_risk_tau"
    ):
        if route_reference.effect_scale != "risk_difference_tau":
            raise ValueError(
                "Stepped-wedge period-adjusted public recomputation requires risk_difference_tau."
            )
        return _recompute_public_stepped_wedge_risk(
            public=public, reference_input=reference_input
        )
    if route_reference.estimator_method_id in {
        "observed:tau_bounds_bounded_deviation",
        "observed:tau_bounds_worst_case",
    }:
        if route_reference.answer_shape != "bound":
            raise ValueError(
                "Tau-bound public recomputation requires bound answer shape."
            )
        if route_reference.effect_scale != "risk_difference_tau":
            raise ValueError(
                "Tau-bound public recomputation requires risk_difference_tau."
            )
        if (
            route_reference.estimator_method_id
            == "observed:tau_bounds_bounded_deviation"
        ):
            if route_reference.sensitivity_parameter is None:
                raise ValueError(
                    "Bounded-deviation replay requires sensitivity_parameter."
                )
            delta = float(route_reference.sensitivity_parameter)
        else:
            if route_reference.sensitivity_parameter is not None:
                raise ValueError(
                    "Worst-case replay must not receive sensitivity_parameter."
                )
            delta = 1.0
        return _recompute_public_tau_bounds(
            public=public, reference_input=reference_input, delta=delta
        )
    if route_reference.estimator_method_id in {
        "observed:validated_endpoint_bounded_deviation",
        "observed:validated_endpoint_worst_case",
    }:
        if (
            route_reference.answer_shape != "bound"
            or route_reference.effect_scale != "risk_difference_tau"
        ):
            raise ValueError(
                "Validated-endpoint bounds require a risk-difference bound answer."
            )
        validated_delta = (
            None
            if route_reference.estimator_method_id
            == "observed:validated_endpoint_worst_case"
            else route_reference.sensitivity_parameter
        )
        if (
            route_reference.estimator_method_id
            == "observed:validated_endpoint_bounded_deviation"
            and validated_delta is None
        ):
            raise ValueError(
                "Validated-endpoint bounded-deviation replay requires sensitivity_parameter."
            )
        lower, upper = validated_endpoint_bounds_v1(
            public=public,
            reference_input=reference_input,
            delta=validated_delta,
        )
        return PublicNumericBoundResultV1(lower=lower, upper=upper)
    raise ValueError(
        f"Unsupported numeric recomputation method: {route_reference.estimator_method_id}"
    )


def recompute_public_standard_error_v1(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
    route_reference: RouteReferenceRecordV1,
) -> float | None:
    """Recompute total uncertainty for methods with nuisance-model fitting."""

    if route_reference.estimator_method_id in {"observed:km", "observed:km_rmst_tau"}:
        result = _recompute_public_km_family(
            public=public, reference_input=reference_input
        )
        return float(
            result.rmst_difference_se
            if route_reference.effect_scale == "rmst_difference_tau"
            else result.risk_difference_se
        )
    if route_reference.estimator_method_id in {
        "observed:km_ipcw_baseline_cox",
        "observed:km_ipcw_rmst_tau",
    }:
        result = _recompute_public_ipcw_km_family(
            public=public,
            reference_input=reference_input,
            include_nuisance_uncertainty=True,
        )
        return float(
            result.rmst_difference_se
            if route_reference.effect_scale == "rmst_difference_tau"
            else result.risk_difference_se
        )
    if route_reference.estimator_method_id in {
        "observed:cox_linear_standardized_risk_tau_reference",
        "observed:cox_rcs_standardized_risk_tau_reference",
    }:
        return float(
            _recompute_public_standardized_risk_with_uncertainty(
                public=public,
                reference_input=reference_input,
                method_id=route_reference.estimator_method_id,
            )[1]
        )
    if (
        route_reference.estimator_method_id
        == "observed:cluster_parallel_participant_weighted_km"
    ):
        result = _recompute_public_cluster_participant_weighted_km(
            public=public, reference_input=reference_input
        )
        return float(
            result.rmst_difference_se
            if route_reference.effect_scale == "rmst_difference_tau"
            else result.risk_difference_se
        )
    if (
        route_reference.estimator_method_id
        == "observed:cluster_parallel_participant_weighted_km_ipcw_baseline_cox"
    ):
        result = (
            _recompute_public_cluster_participant_weighted_ipcw_km_with_uncertainty(
                public=public,
                reference_input=reference_input,
            )
        )
        return float(
            result.rmst_difference_se
            if route_reference.effect_scale == "rmst_difference_tau"
            else result.risk_difference_se
        )
    if route_reference.estimator_method_id == "observed:coxph_binary_breslow":
        return _recompute_public_coxph_binary_breslow_with_uncertainty(
            public=public,
            reference_input=reference_input,
        )[1]
    if route_reference.estimator_method_id == "observed:coxph_binary_breslow_risk_tau":
        return _recompute_public_coxph_binary_breslow_risk_tau_with_uncertainty(
            public=public,
            reference_input=reference_input,
        )[1]
    if (
        route_reference.estimator_method_id
        == "observed:validated_endpoint_joint_likelihood"
    ):
        return validated_endpoint_point_and_standard_error_v1(
            public=public,
            reference_input=reference_input,
        )[1]
    return None


def recompute_public_numeric_result_v1(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
    route_reference: RouteReferenceRecordV1,
    km_family_cache: dict[tuple[str, ...], _KmFamilyResult] | None = None,
) -> tuple[
    float | PublicNumericBoundResultV1 | PublicNumericVectorResultV1, float | None
]:
    """Recompute a numeric value and uncertainty without duplicate model fits."""

    method_id = route_reference.estimator_method_id
    scale = route_reference.effect_scale
    if method_id in {"observed:km_ipcw_baseline_cox", "observed:km_ipcw_rmst_tau"}:
        result = _cached_km_family_result_v1(
            public=public,
            reference_input=reference_input,
            regime_cell_id=numeric_replay_family_id_v1(method_id),
            cache=km_family_cache,
            compute=lambda: _recompute_public_ipcw_km_family(
                public=public,
                reference_input=reference_input,
                include_nuisance_uncertainty=True,
            ),
        )
        if scale == "rmst_difference_tau":
            return float(result.rmst_difference), float(result.rmst_difference_se)
        return float(result.risk_difference), float(result.risk_difference_se)
    if method_id in {
        "observed:cox_linear_standardized_risk_tau_reference",
        "observed:cox_rcs_standardized_risk_tau_reference",
    }:
        value, standard_error = _recompute_public_standardized_risk_with_uncertainty(
            public=public,
            reference_input=reference_input,
            method_id=method_id,
        )
        return float(value), float(standard_error)
    if method_id == "observed:cluster_parallel_participant_weighted_km":
        result = _cached_km_family_result_v1(
            public=public,
            reference_input=reference_input,
            regime_cell_id=numeric_replay_family_id_v1(method_id),
            cache=km_family_cache,
            compute=lambda: _recompute_public_cluster_participant_weighted_km(
                public=public,
                reference_input=reference_input,
            ),
        )
        if scale == "rmst_difference_tau":
            return float(result.rmst_difference), float(result.rmst_difference_se)
        return float(result.risk_difference), float(result.risk_difference_se)
    if (
        method_id
        == "observed:cluster_parallel_participant_weighted_km_ipcw_baseline_cox"
    ):
        result = _cached_km_family_result_v1(
            public=public,
            reference_input=reference_input,
            regime_cell_id=numeric_replay_family_id_v1(method_id),
            cache=km_family_cache,
            compute=lambda: _recompute_public_cluster_participant_weighted_ipcw_km_with_uncertainty(
                public=public,
                reference_input=reference_input,
            ),
        )
        if scale == "rmst_difference_tau":
            return float(result.rmst_difference), float(result.rmst_difference_se)
        return float(result.risk_difference), float(result.risk_difference_se)
    if method_id == "observed:stepped_wedge_period_cluster_adjusted_risk_tau":
        return _recompute_public_stepped_wedge_risk_with_uncertainty(
            public=public,
            reference_input=reference_input,
        )
    if method_id == "observed:validated_endpoint_joint_likelihood":
        value, standard_error = validated_endpoint_point_and_standard_error_v1(
            public=public,
            reference_input=reference_input,
        )
        return float(value), float(standard_error)
    return (
        recompute_public_numeric_value_v1(
            public=public,
            reference_input=reference_input,
            route_reference=route_reference,
        ),
        recompute_public_standard_error_v1(
            public=public,
            reference_input=reference_input,
            route_reference=route_reference,
        ),
    )


def _cached_km_family_result_v1(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
    regime_cell_id: str,
    cache: dict[tuple[str, ...], _KmFamilyResult] | None,
    compute: Callable[[], _KmFamilyResult],
) -> _KmFamilyResult:
    """Reuse one fitted KM-family result across compatible public estimands."""

    if cache is None:
        return compute()
    task = _read_json_from_public(public, f"items/{reference_input.task_id}/task.json")
    protocol = _read_json_from_public(
        public, f"items/{reference_input.task_id}/protocol_summary.json"
    )
    key = (
        str(regime_cell_id),
        _required_str(task, "primary_paramcd"),
        _required_str(task, "primary_control_arm_id"),
        _required_str(task, "primary_treated_arm_id"),
        f"{_required_positive_float(protocol, 'followup_horizon_dy'):.17g}",
        str(reference_input.source_role),
        *(
            f"{ref.semantic_role}:{ref.sha256}"
            for ref in sorted(
                reference_input.required_table_refs,
                key=lambda row: (row.semantic_role, row.rel_path),
            )
        ),
    )
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = compute()
    cache[key] = result
    return result


def has_required_public_inputs_for_method_v1(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
    method_id: str,
) -> bool:
    """Return whether a scoreable reference input exposes required public tables."""

    if _has_public_reconstruction_tables(reference_input):
        return True
    if method_id in {
        "observed:tau_bounds_bounded_deviation",
        "observed:tau_bounds_worst_case",
    }:
        return _public_has_table_suffixes(
            public=public,
            reference_input=reference_input,
            suffixes=("ADTTE.parquet", "subject_operational_flags.parquet"),
        ) or _public_has_table_suffixes(
            public=public,
            reference_input=reference_input,
            suffixes=("ADTTE.parquet", "ADSL.parquet"),
        )
    return _public_has_table_suffixes(
        public=public, reference_input=reference_input, suffixes=("ADTTE.parquet",)
    ) and (
        _public_has_table_suffixes(
            public=public, reference_input=reference_input, suffixes=("ADSL.parquet",)
        )
        or _public_has_table_suffixes(
            public=public,
            reference_input=reference_input,
            suffixes=("subject_operational_flags.parquet",),
        )
    )


def _recompute_public_km_family(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
) -> _KmFamilyResult:
    task = _read_json_from_public(public, f"items/{reference_input.task_id}/task.json")
    protocol = _read_json_from_public(
        public, f"items/{reference_input.task_id}/protocol_summary.json"
    )
    paramcd = _required_str(task, "primary_paramcd")
    control_arm_id = _required_str(task, "primary_control_arm_id")
    treated_arm_id = _required_str(task, "primary_treated_arm_id")
    tau = _required_positive_float(protocol, "followup_horizon_dy")
    adsl = _read_treatment_surface_table(public=public, reference_input=reference_input)
    adtte = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADTTE.parquet"
    )
    merged = _analysis_frame(adsl=adsl, adtte=adtte, paramcd=paramcd)
    arm_results: dict[str, tuple[float, float, float, float]] = {}
    for arm_id in (control_arm_id, treated_arm_id):
        rows = merged.loc[merged["TRTA"].astype("string") == arm_id]
        if rows.empty:
            raise ValueError(
                f"Public evidence contains no analysis rows for arm_id={arm_id!r}."
            )
        arm_results[arm_id] = _km_risk_rmst_and_se(
            t=rows["AVAL"].to_numpy(dtype=np.float64, copy=False),
            e=(rows["CNSR"].to_numpy(dtype=np.int64, copy=False) == 0).astype(
                np.int64, copy=False
            ),
            tau=tau,
        )
    control = arm_results[control_arm_id]
    treated = arm_results[treated_arm_id]
    return _KmFamilyResult(
        risk_difference=float(treated[0] - control[0]),
        rmst_difference=float(treated[1] - control[1]),
        risk_difference_se=float(
            math.sqrt(float(control[2]) ** 2 + float(treated[2]) ** 2)
        ),
        rmst_difference_se=float(
            math.sqrt(float(control[3]) ** 2 + float(treated[3]) ** 2)
        ),
    )


def _recompute_public_coxph_binary_breslow_with_uncertainty(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
) -> tuple[float, float]:
    task = _read_json_from_public(public, f"items/{reference_input.task_id}/task.json")
    paramcd = _required_str(task, "primary_paramcd")
    treated_arm_id = _required_str(task, "primary_treated_arm_id")
    adsl = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADSL.parquet"
    )
    adtte = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADTTE.parquet"
    )
    merged = _analysis_frame(adsl=adsl, adtte=adtte, paramcd=paramcd)
    t = merged["AVAL"].to_numpy(dtype=np.float64, copy=False)
    e = (merged["CNSR"].to_numpy(dtype=np.int64, copy=False) == 0).astype(
        np.int64, copy=False
    )
    a = np.asarray(
        (merged["TRTA"].astype("string") == treated_arm_id)
        .astype("int64")
        .to_numpy(dtype=np.int64, copy=False),
        dtype=np.int64,
    )
    beta, standard_error = _coxph_binary_breslow_newton(t=t, e=e, a=a)
    return float(beta), float(standard_error)


def _recompute_public_coxph_binary_breslow_risk_tau_with_uncertainty(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
) -> tuple[float, float]:
    task = _read_json_from_public(public, f"items/{reference_input.task_id}/task.json")
    protocol = _read_json_from_public(
        public, f"items/{reference_input.task_id}/protocol_summary.json"
    )
    paramcd = _required_str(task, "primary_paramcd")
    treated_arm_id = _required_str(task, "primary_treated_arm_id")
    tau = _required_positive_float(protocol, "followup_horizon_dy")
    adsl = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADSL.parquet"
    )
    adtte = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADTTE.parquet"
    )
    merged = _analysis_frame(adsl=adsl, adtte=adtte, paramcd=paramcd).sort_values(
        "USUBJID",
        kind="mergesort",
    )
    t = merged["AVAL"].to_numpy(dtype=np.float64, copy=False)
    e = (merged["CNSR"].to_numpy(dtype=np.int64, copy=False) == 0).astype(
        np.int64, copy=False
    )
    a = np.asarray(
        (merged["TRTA"].astype("string") == treated_arm_id)
        .astype("int64")
        .to_numpy(dtype=np.int64, copy=False),
        dtype=np.int64,
    )
    value = _coxph_binary_breslow_risk_difference_tau(t=t, e=e, a=a, tau=float(tau))
    groups = balanced_delete_groups_v1(
        unit_ids=merged["USUBJID"].to_numpy(dtype=str),
        strata=merged["TRTA"].to_numpy(dtype=str),
        n_groups=DELETE_GROUP_COUNT_V1,
    )
    replicates = [
        _coxph_binary_breslow_risk_difference_tau(
            t=t[groups != group],
            e=e[groups != group],
            a=a[groups != group],
            tau=float(tau),
        )
        for group in range(DELETE_GROUP_COUNT_V1)
    ]
    return float(value), float(delete_group_standard_error_v1(replicates))


def _recompute_public_cluster_participant_weighted_km(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
) -> _KmFamilyResult:
    task = _read_json_from_public(public, f"items/{reference_input.task_id}/task.json")
    protocol = _read_json_from_public(
        public, f"items/{reference_input.task_id}/protocol_summary.json"
    )
    paramcd = _required_str(task, "primary_paramcd")
    control_arm_id = _required_str(task, "primary_control_arm_id")
    treated_arm_id = _required_str(task, "primary_treated_arm_id")
    tau = _required_positive_float(protocol, "followup_horizon_dy")
    adsl = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADSL.parquet"
    )
    adtte = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADTTE.parquet"
    )
    merged = _analysis_frame(adsl=adsl, adtte=adtte, paramcd=paramcd)
    if "SITEID" not in adsl.columns:
        raise ValueError(
            "Cluster participant-weighted KM requires SITEID in the public ADSL surface."
        )
    site_assignments = adsl.loc[:, ["USUBJID", "SITEID"]].copy()
    site_assignments["SITEID"] = site_assignments["SITEID"].astype("string")
    if site_assignments["SITEID"].isna().any():
        raise ValueError(
            "Cluster participant-weighted KM requires complete SITEID assignments."
        )
    merged = merged.merge(
        site_assignments, on="USUBJID", how="left", validate="one_to_one"
    )
    if merged["SITEID"].isna().any():
        raise ValueError(
            "Cluster participant-weighted KM contains analysis rows without SITEID assignments."
        )
    return _cluster_participant_weighted_km_from_frame(
        merged=merged,
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
        weights=None,
    )


def _cluster_participant_weighted_km_from_frame(
    *,
    merged: pd.DataFrame,
    tau: float,
    control_arm_id: str,
    treated_arm_id: str,
    weights: NDArray[np.float64] | None,
) -> _KmFamilyResult:
    arm_results: dict[str, tuple[float, float, float, float]] = {}
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if int(w.size) != int(len(merged)):
            raise ValueError(
                "Cluster participant-weighted KM requires weights aligned to merged rows."
            )
        if not np.isfinite(w).all() or np.any(w < 0.0):
            raise ValueError(
                "Cluster participant-weighted KM requires finite non-negative weights."
            )
    else:
        w = np.ones(int(len(merged)), dtype=np.float64)
    frame = merged.copy()
    frame["_W"] = w
    frame["SITEID"] = frame["SITEID"].astype("string")
    if frame["SITEID"].isna().any():
        raise ValueError(
            "Cluster participant-weighted KM requires complete SITEID assignments."
        )
    for arm_id in (control_arm_id, treated_arm_id):
        arm_rows = frame.loc[frame["TRTA"].astype("string") == arm_id]
        if arm_rows.empty:
            raise ValueError(
                f"Cluster participant-weighted KM contains no analysis rows for arm_id={arm_id!r}."
            )
        site_risks: list[float] = []
        site_rmsts: list[float] = []
        site_sizes: list[float] = []
        for _site_id, site_rows in arm_rows.groupby("SITEID", sort=True):
            if site_rows.empty:
                continue
            site_arms = tuple(
                sorted(
                    set(
                        str(value)
                        for value in site_rows["TRTA"].astype("string").tolist()
                    )
                )
            )
            if len(site_arms) != 1:
                raise ValueError(
                    f"Cluster participant-weighted KM requires cluster-pure SITEID assignments: {site_arms!r}."
                )
            risk, rmst = _weighted_km_risk_and_rmst(
                t=site_rows["AVAL"].to_numpy(dtype=np.float64, copy=False),
                e=(site_rows["CNSR"].to_numpy(dtype=np.int64, copy=False) == 0).astype(
                    np.int64, copy=False
                ),
                weights=site_rows["_W"].to_numpy(dtype=np.float64, copy=False),
                tau=tau,
            )
            site_risks.append(float(risk))
            site_rmsts.append(float(rmst))
            site_sizes.append(float(len(site_rows)))
        if not site_risks:
            raise ValueError(
                f"Cluster participant-weighted KM contains no clusters for arm_id={arm_id!r}."
            )
        sizes = np.asarray(site_sizes, dtype=np.float64)

        def _participant_weighted_mean_and_se(
            values: list[float], cluster_sizes: NDArray[np.float64]
        ) -> tuple[float, float]:
            array = np.asarray(values, dtype=np.float64)
            mean = float(np.average(array, weights=cluster_sizes))
            n_clusters = int(array.size)
            if n_clusters <= 1:
                return mean, 0.0
            normalized = cluster_sizes / float(np.sum(cluster_sizes))
            variance = float(n_clusters / (n_clusters - 1)) * float(
                np.sum(np.square(normalized) * np.square(array - mean))
            )
            return mean, float(np.sqrt(max(variance, 0.0)))

        risk, risk_se = _participant_weighted_mean_and_se(site_risks, sizes)
        rmst, rmst_se = _participant_weighted_mean_and_se(site_rmsts, sizes)
        arm_results[arm_id] = (risk, rmst, risk_se, rmst_se)
    control = arm_results[control_arm_id]
    treated = arm_results[treated_arm_id]
    return _KmFamilyResult(
        risk_difference=float(treated[0] - control[0]),
        rmst_difference=float(treated[1] - control[1]),
        risk_difference_se=float(np.hypot(treated[2], control[2])),
        rmst_difference_se=float(np.hypot(treated[3], control[3])),
    )


def _recompute_public_ipcw_km_family(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
    include_nuisance_uncertainty: bool = False,
) -> _KmFamilyResult:
    task = _read_json_from_public(public, f"items/{reference_input.task_id}/task.json")
    protocol = _read_json_from_public(
        public, f"items/{reference_input.task_id}/protocol_summary.json"
    )
    paramcd = _required_str(task, "primary_paramcd")
    control_arm_id = _required_str(task, "primary_control_arm_id")
    treated_arm_id = _required_str(task, "primary_treated_arm_id")
    tau = _required_positive_float(protocol, "followup_horizon_dy")
    adsl = _read_covariate_surface_table(public=public, reference_input=reference_input)
    adtte = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADTTE.parquet"
    )
    baseline_columns = _public_ipcw_covariate_columns(
        public=public, reference_input=reference_input, adsl=adsl
    )
    merged = _ipcw_analysis_frame(
        adsl=adsl,
        adtte=adtte,
        paramcd=paramcd,
        baseline_covariate_columns=baseline_columns,
    )
    return _ipcw_km_family_from_frame(
        merged=merged,
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
        baseline_covariate_columns=baseline_columns,
        _include_nuisance_uncertainty=bool(include_nuisance_uncertainty),
        _require_support_pass=True,
    )


def _recompute_public_ipcw_coxph_binary_breslow(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
) -> float:
    task = _read_json_from_public(public, f"items/{reference_input.task_id}/task.json")
    protocol = _read_json_from_public(
        public, f"items/{reference_input.task_id}/protocol_summary.json"
    )
    paramcd = _required_str(task, "primary_paramcd")
    control_arm_id = _required_str(task, "primary_control_arm_id")
    treated_arm_id = _required_str(task, "primary_treated_arm_id")
    tau = _required_positive_float(protocol, "followup_horizon_dy")
    adsl = _read_covariate_surface_table(public=public, reference_input=reference_input)
    adtte = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADTTE.parquet"
    )
    baseline_columns = _public_ipcw_covariate_columns(
        public=public, reference_input=reference_input, adsl=adsl
    )
    merged = _ipcw_analysis_frame(
        adsl=adsl,
        adtte=adtte,
        paramcd=paramcd,
        baseline_covariate_columns=baseline_columns,
    )
    support = _ipcw_support_from_model(
        merged=merged,
        model=_fit_censoring_model(
            merged,
            baseline_covariate_columns=baseline_columns,
            administrative_horizon=tau,
        ),
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
    )
    if support.support_status != "pass":
        raise ValueError(
            f"IPCW Cox route lacks point-support qualification: {support.support_status}."
        )
    weights = _ipcw_subject_weights(
        merged,
        baseline_covariate_columns=baseline_columns,
        administrative_horizon=tau,
    )
    t = merged["AVAL"].to_numpy(dtype=np.float64, copy=False)
    e = (merged["CNSR"].to_numpy(dtype=np.int64, copy=False) == 0).astype(
        np.int64, copy=False
    )
    a = np.asarray(
        (merged["TRTA"].astype("string") == treated_arm_id)
        .astype("int64")
        .to_numpy(dtype=np.int64, copy=False),
        dtype=np.int64,
    )
    beta, _se = _coxph_binary_breslow_newton(t=t, e=e, a=a, weights=weights)
    return float(beta)


def _recompute_public_cluster_participant_weighted_ipcw_km(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
) -> _KmFamilyResult:
    task = _read_json_from_public(public, f"items/{reference_input.task_id}/task.json")
    protocol = _read_json_from_public(
        public, f"items/{reference_input.task_id}/protocol_summary.json"
    )
    paramcd = _required_str(task, "primary_paramcd")
    control_arm_id = _required_str(task, "primary_control_arm_id")
    treated_arm_id = _required_str(task, "primary_treated_arm_id")
    tau = _required_positive_float(protocol, "followup_horizon_dy")
    adsl = _read_covariate_surface_table(public=public, reference_input=reference_input)
    adtte = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADTTE.parquet"
    )
    baseline_columns = _public_ipcw_covariate_columns(
        public=public, reference_input=reference_input, adsl=adsl
    )
    merged = _ipcw_analysis_frame(
        adsl=adsl,
        adtte=adtte,
        paramcd=paramcd,
        baseline_covariate_columns=baseline_columns,
    )
    if "SITEID" not in merged.columns:
        raise ValueError(
            "Cluster-IPCW participant-weighted KM requires SITEID in the public covariate surface."
        )
    support = _ipcw_support_from_model(
        merged=merged,
        model=_fit_censoring_model(
            merged,
            baseline_covariate_columns=baseline_columns,
            administrative_horizon=tau,
        ),
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
    )
    if support.support_status != "pass":
        raise ValueError(
            f"Cluster-IPCW route lacks point-support qualification: {support.support_status}."
        )
    weights = _ipcw_subject_weights(
        merged,
        baseline_covariate_columns=baseline_columns,
        administrative_horizon=tau,
    )
    return _cluster_participant_weighted_km_from_frame(
        merged=merged,
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
        weights=weights,
    )


def _recompute_public_cluster_participant_weighted_ipcw_km_with_uncertainty(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
) -> _KmFamilyResult:
    task = _read_json_from_public(public, f"items/{reference_input.task_id}/task.json")
    protocol = _read_json_from_public(
        public, f"items/{reference_input.task_id}/protocol_summary.json"
    )
    paramcd = _required_str(task, "primary_paramcd")
    control_arm_id = _required_str(task, "primary_control_arm_id")
    treated_arm_id = _required_str(task, "primary_treated_arm_id")
    tau = _required_positive_float(protocol, "followup_horizon_dy")
    adsl = _read_covariate_surface_table(public=public, reference_input=reference_input)
    adtte = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADTTE.parquet"
    )
    baseline_columns = _public_ipcw_covariate_columns(
        public=public, reference_input=reference_input, adsl=adsl
    )
    return cluster_ipcw_contrasts_with_uncertainty_v1(
        adsl=adsl,
        adtte=adtte,
        paramcd=paramcd,
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
        baseline_covariate_columns=baseline_columns,
        _require_support_pass=True,
    )


def cluster_ipcw_contrasts_with_uncertainty_v1(
    *,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    paramcd: str,
    tau: float,
    control_arm_id: str,
    treated_arm_id: str,
    baseline_covariate_columns: tuple[str, ...],
    _require_support_pass: bool = False,
) -> _KmFamilyResult:
    """Replay cluster-IPCW risk/RMST contrasts with complete nuisance refits."""

    merged = _ipcw_analysis_frame(
        adsl=adsl,
        adtte=adtte,
        paramcd=paramcd,
        baseline_covariate_columns=baseline_covariate_columns,
    )
    if "SITEID" not in merged.columns:
        raise ValueError(
            "Cluster-IPCW uncertainty requires SITEID in the public covariate surface."
        )
    if _require_support_pass:
        support = _ipcw_support_from_model(
            merged=merged,
            model=_fit_censoring_model(
                merged,
                baseline_covariate_columns=baseline_covariate_columns,
                administrative_horizon=tau,
            ),
            tau=tau,
            control_arm_id=control_arm_id,
            treated_arm_id=treated_arm_id,
        )
        if support.support_status != "pass":
            raise ValueError(
                f"Cluster-IPCW route lacks point-support qualification: {support.support_status}."
            )
    point = _cluster_participant_weighted_km_from_frame(
        merged=merged,
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
        weights=_ipcw_subject_weights(
            merged,
            baseline_covariate_columns=baseline_covariate_columns,
            administrative_horizon=tau,
        ),
    )
    cluster_assignments = merged.loc[:, ["SITEID", "TRTA"]].drop_duplicates().copy()
    if cluster_assignments.duplicated(subset=["SITEID"]).any():
        raise ValueError(
            "Cluster-IPCW jackknife requires one randomized arm per SITEID."
        )
    groups = balanced_delete_groups_v1(
        unit_ids=cluster_assignments["SITEID"].astype("string").to_numpy(dtype=str),
        strata=cluster_assignments["TRTA"].astype("string").to_numpy(dtype=str),
        n_groups=DELETE_GROUP_COUNT_V1,
    )
    risk_replicates: list[float] = []
    rmst_replicates: list[float] = []
    for group in range(DELETE_GROUP_COUNT_V1):
        retained_sites = set(
            cluster_assignments.loc[groups != group, "SITEID"].astype("string").tolist()
        )
        replicate = merged.loc[
            merged["SITEID"].astype("string").isin(retained_sites), :
        ]
        result = _cluster_participant_weighted_km_from_frame(
            merged=replicate,
            tau=tau,
            control_arm_id=control_arm_id,
            treated_arm_id=treated_arm_id,
            weights=_ipcw_subject_weights(
                replicate,
                baseline_covariate_columns=baseline_covariate_columns,
                administrative_horizon=tau,
            ),
        )
        risk_replicates.append(float(result.risk_difference))
        rmst_replicates.append(float(result.rmst_difference))
    return point.model_copy(
        update={
            "risk_difference_se": delete_group_standard_error_v1(risk_replicates),
            "rmst_difference_se": delete_group_standard_error_v1(rmst_replicates),
        }
    )


def _recompute_public_tau_bounds(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
    delta: float,
) -> PublicNumericBoundResultV1:
    task = _read_json_from_public(public, f"items/{reference_input.task_id}/task.json")
    protocol = _read_json_from_public(
        public, f"items/{reference_input.task_id}/protocol_summary.json"
    )
    paramcd = _required_str(task, "primary_paramcd")
    control_arm_id = _required_str(task, "primary_control_arm_id")
    treated_arm_id = _required_str(task, "primary_treated_arm_id")
    tau = _required_positive_float(protocol, "followup_horizon_dy")
    adtte = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADTTE.parquet"
    )
    covariates = _read_covariate_surface_table(
        public=public, reference_input=reference_input
    )
    lower, upper = _bounds_rd_tau(
        covariates=covariates,
        adtte=adtte,
        paramcd=paramcd,
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
        delta=delta,
    )
    return PublicNumericBoundResultV1(lower=lower, upper=upper)


def _public_standardized_risk_covariates(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
) -> pd.DataFrame:
    """Select the independently frozen five-covariate TE-S05 model surface."""

    try:
        available = _read_required_table_by_suffix(
            public=public,
            reference_input=reference_input,
            suffix="baseline_characteristics.parquet",
        )
    except FileNotFoundError:
        available = _read_required_table_by_suffix(
            public=public,
            reference_input=reference_input,
            suffix="analysis_frame_covariates.parquet",
        )
    required = (
        "USUBJID",
        *TRIALEVAL_STANDARDIZED_RISK_BASELINE_COVARIATES_V1,
    )
    missing = tuple(column for column in required if column not in available.columns)
    if missing:
        raise ValueError(f"TE-S05 public replay lacks frozen covariates: {missing!r}.")
    return available.loc[:, list(required)].copy()


def _public_standardized_risk_reference_covariates(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
) -> pd.DataFrame:
    """Load the protocol-defined standardization population."""

    available = _read_required_table_by_suffix(
        public=public,
        reference_input=reference_input,
        suffix="reference_population_covariates.parquet",
    )
    required = (
        "REFERENCE_ID",
        *TRIALEVAL_STANDARDIZED_RISK_BASELINE_COVARIATES_V1,
    )
    missing = tuple(column for column in required if column not in available.columns)
    if missing:
        raise ValueError(
            f"TE-S05 public replay lacks reference-population covariates: {missing!r}."
        )
    return available.loc[:, list(required)].copy()


def _recompute_public_standardized_risk(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
    method_id: str,
) -> float:
    task = _read_json_from_public(public, f"items/{reference_input.task_id}/task.json")
    protocol = _read_json_from_public(
        public, f"items/{reference_input.task_id}/protocol_summary.json"
    )
    paramcd = _required_str(task, "primary_paramcd")
    control_arm_id = _required_str(task, "primary_control_arm_id")
    treated_arm_id = _required_str(task, "primary_treated_arm_id")
    tau = _required_positive_float(protocol, "followup_horizon_dy")
    adsl = _read_treatment_surface_table(public=public, reference_input=reference_input)
    adtte = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADTTE.parquet"
    )
    analysis_covariates = _public_standardized_risk_covariates(
        public=public,
        reference_input=reference_input,
    )
    reference_covariates = _public_standardized_risk_reference_covariates(
        public=public,
        reference_input=reference_input,
    )
    calculator = {
        "observed:cox_linear_standardized_risk_tau_reference": (
            cox_linear_standardized_risk_difference_tau_reference_v1
        ),
        "observed:cox_rcs_standardized_risk_tau_reference": (
            cox_rcs_standardized_risk_difference_tau_reference_v1
        ),
    }.get(method_id)
    if calculator is None:
        raise ValueError(f"Unsupported standardized-risk method: {method_id!r}.")
    return calculator(
        adsl=adsl,
        adtte=adtte,
        analysis_covariates=analysis_covariates,
        reference_covariates=reference_covariates,
        paramcd=paramcd,
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
    )


def _recompute_public_standardized_risk_with_uncertainty(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
    method_id: str,
) -> tuple[float, float]:
    task = _read_json_from_public(public, f"items/{reference_input.task_id}/task.json")
    protocol = _read_json_from_public(
        public, f"items/{reference_input.task_id}/protocol_summary.json"
    )
    paramcd = _required_str(task, "primary_paramcd")
    control_arm_id = _required_str(task, "primary_control_arm_id")
    treated_arm_id = _required_str(task, "primary_treated_arm_id")
    tau = _required_positive_float(protocol, "followup_horizon_dy")
    adsl = _read_treatment_surface_table(public=public, reference_input=reference_input)
    adtte = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADTTE.parquet"
    )
    analysis_covariates = _public_standardized_risk_covariates(
        public=public,
        reference_input=reference_input,
    )
    reference_covariates = _public_standardized_risk_reference_covariates(
        public=public,
        reference_input=reference_input,
    )
    calculator = {
        "observed:cox_linear_standardized_risk_tau_reference": (
            cox_linear_standardized_risk_difference_tau_reference_with_uncertainty_v1
        ),
        "observed:cox_rcs_standardized_risk_tau_reference": (
            cox_rcs_standardized_risk_difference_tau_reference_with_uncertainty_v1
        ),
    }.get(method_id)
    if calculator is None:
        raise ValueError(f"Unsupported standardized-risk method: {method_id!r}.")
    return calculator(
        adsl=adsl,
        adtte=adtte,
        analysis_covariates=analysis_covariates,
        reference_covariates=reference_covariates,
        paramcd=paramcd,
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
    )


def _recompute_public_stepped_wedge_risk(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
) -> float:
    task = _read_json_from_public(public, f"items/{reference_input.task_id}/task.json")
    protocol = _read_json_from_public(
        public, f"items/{reference_input.task_id}/protocol_summary.json"
    )
    paramcd = _required_str(task, "primary_paramcd")
    tau = _required_positive_float(protocol, "followup_horizon_dy")
    adsl = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADSL.parquet"
    )
    adtte = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADTTE.parquet"
    )
    return stepped_wedge_period_adjusted_risk_difference_tau_v1(
        adsl=adsl,
        adtte=adtte,
        paramcd=paramcd,
        tau=tau,
    )


def _recompute_public_stepped_wedge_risk_with_uncertainty(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
) -> tuple[float, float]:
    task = _read_json_from_public(public, f"items/{reference_input.task_id}/task.json")
    protocol = _read_json_from_public(
        public, f"items/{reference_input.task_id}/protocol_summary.json"
    )
    adsl = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADSL.parquet"
    )
    adtte = _read_required_table_by_suffix(
        public=public, reference_input=reference_input, suffix="ADTTE.parquet"
    )
    return stepped_wedge_period_adjusted_risk_difference_tau_with_uncertainty_v1(
        adsl=adsl,
        adtte=adtte,
        paramcd=_required_str(task, "primary_paramcd"),
        tau=_required_positive_float(protocol, "followup_horizon_dy"),
    )


def _analysis_frame(
    *, adsl: pd.DataFrame, adtte: pd.DataFrame, paramcd: str
) -> pd.DataFrame:
    missing_adsl = sorted(
        {"USUBJID", "TRTA"} - {str(column) for column in adsl.columns}
    )
    if missing_adsl:
        raise ValueError(f"ADSL is missing required columns: {missing_adsl!r}.")
    missing_adtte = sorted(
        {"USUBJID", "PARAMCD", "AVAL", "CNSR"}
        - {str(column) for column in adtte.columns}
    )
    if missing_adtte:
        raise ValueError(f"ADTTE is missing required columns: {missing_adtte!r}.")
    primary = _primary_adtte_rows(adtte=adtte, paramcd=paramcd)
    if primary.empty:
        raise ValueError(f"ADTTE contains no rows for primary_paramcd={paramcd!r}.")
    merged = primary.merge(
        adsl.loc[:, ["USUBJID", "TRTA"]],
        on="USUBJID",
        how="left",
        validate="many_to_one",
    )
    if merged["TRTA"].isna().any():
        raise ValueError(
            "ADTTE contains subjects absent from ADSL treatment assignments."
        )
    merged["AVAL"] = pd.to_numeric(merged["AVAL"], errors="raise").astype("float64")
    merged["CNSR"] = pd.to_numeric(merged["CNSR"], errors="raise").astype("int64")
    if not np.isfinite(merged["AVAL"].to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("ADTTE AVAL contains non-finite values.")
    if not set(int(value) for value in merged["CNSR"].unique()) <= {0, 1}:
        raise ValueError("ADTTE CNSR must use ADaM event/censoring codes 0 or 1.")
    return cast(pd.DataFrame, merged)


def _ipcw_analysis_frame(
    *,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    paramcd: str,
    baseline_covariate_columns: tuple[str, ...],
) -> pd.DataFrame:
    baseline_columns = tuple(
        sorted(set(str(column) for column in baseline_covariate_columns))
    )
    missing_adsl = sorted(
        {"USUBJID", "TRTA"} - {str(column) for column in adsl.columns}
    )
    if missing_adsl:
        raise ValueError(
            f"IPCW public covariate surface is missing required columns: {missing_adsl!r}."
        )
    if not baseline_columns:
        raise ValueError(
            "IPCW public covariate surface has no pretreatment covariates."
        )
    missing_adtte = sorted(
        {"USUBJID", "PARAMCD", "AVAL", "CNSR"}
        - {str(column) for column in adtte.columns}
    )
    if missing_adtte:
        raise ValueError(f"IPCW ADTTE is missing required columns: {missing_adtte!r}.")
    primary = _primary_adtte_rows(adtte=adtte, paramcd=paramcd)
    columns = ["USUBJID", "TRTA", *baseline_columns]
    if "SITEID" in adsl.columns:
        columns.append("SITEID")
    merged = primary.merge(
        adsl.loc[:, columns],
        on="USUBJID",
        how="left",
        validate="many_to_one",
    )
    if merged[["TRTA", *baseline_columns]].isna().any().any():
        raise ValueError(
            "IPCW public replay requires complete treatment and baseline-covariate rows."
        )
    merged["USUBJID"] = merged["USUBJID"].astype("string")
    merged = merged.sort_values("USUBJID", kind="mergesort").reset_index(drop=True)
    merged["AVAL"] = np.round(
        pd.to_numeric(merged["AVAL"], errors="raise").to_numpy(dtype=np.float64),
        decimals=10,
    )
    merged["CNSR"] = pd.to_numeric(merged["CNSR"], errors="raise").astype("int64")
    if not np.isfinite(merged["AVAL"].to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("IPCW ADTTE AVAL contains non-finite values.")
    if not set(int(value) for value in merged["CNSR"].unique()) <= {0, 1}:
        raise ValueError("IPCW ADTTE CNSR must use ADaM event/censoring codes 0 or 1.")
    return cast(pd.DataFrame, merged)


@dataclass(frozen=True, slots=True)
class _BaselineCumulativeHazard:
    event_times: NDArray[np.float64]
    cumulative_hazard: NDArray[np.float64]

    def __call__(self, query: object) -> object:
        values = np.asarray(query, dtype=np.float64)
        indices = np.searchsorted(self.event_times, values, side="left") - 1
        result = np.zeros(values.shape, dtype=np.float64)
        valid = indices >= 0
        result[valid] = self.cumulative_hazard[indices[valid]]
        return float(result) if result.ndim == 0 else result


@dataclass(frozen=True, slots=True)
class _CensoringModel:
    linear_predictor: NDArray[np.float64]
    arms: NDArray[np.str_]
    baseline_hazard: dict[str, _BaselineCumulativeHazard]

    def weights(self, query_times: NDArray[np.float64]) -> NDArray[np.float64]:
        query = np.asarray(query_times, dtype=np.float64)
        if query.shape != self.linear_predictor.shape:
            raise ValueError(
                "Censoring-model query times do not align with fitted rows."
            )
        cumulative_hazard = np.empty(query.size, dtype=np.float64)
        for arm in sorted(set(str(value) for value in self.arms)):
            mask = self.arms == arm
            cumulative_hazard[mask] = np.asarray(
                self.baseline_hazard[arm](query[mask]), dtype=np.float64
            )
        survival = np.exp(-cumulative_hazard * np.exp(self.linear_predictor))
        if not np.isfinite(survival).all() or np.any(
            (survival <= 0.0) | (survival > 1.0)
        ):
            raise ValueError("Censoring model produced invalid survival probabilities.")
        return np.asarray(1.0 / survival, dtype=np.float64)

    def weights_at(
        self, *, time: float, selected_rows: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """Evaluate inverse censoring survival for selected rows at one event time."""

        if not np.isfinite(float(time)) or float(time) < 0.0:
            raise ValueError(
                "Censoring-model event time must be finite and non-negative."
            )
        mask = np.asarray(selected_rows, dtype=bool)
        if mask.shape != self.linear_predictor.shape:
            raise ValueError(
                "Censoring-model row-selection mask does not align with fitted rows."
            )
        selected_lp = self.linear_predictor[mask]
        selected_arms = self.arms[mask]
        cumulative_hazard = np.empty(int(mask.sum()), dtype=np.float64)
        for arm in sorted(set(str(value) for value in selected_arms)):
            arm_mask = selected_arms == arm
            value = np.asarray(
                self.baseline_hazard[arm](float(time)), dtype=np.float64
            ).reshape(-1)
            if value.size != 1:
                raise ValueError(
                    "Censoring baseline hazard did not return one event-time value."
                )
            cumulative_hazard[arm_mask] = float(value[0])
        survival = np.exp(-cumulative_hazard * np.exp(selected_lp))
        if not np.isfinite(survival).all() or np.any(
            (survival <= 0.0) | (survival > 1.0)
        ):
            raise ValueError(
                "Censoring model produced invalid event-time survival probabilities."
            )
        return np.asarray(1.0 / survival, dtype=np.float64)


def _breslow_baselines(
    *,
    times: NDArray[np.float64],
    censor_events: NDArray[np.int64],
    linear_predictor: NDArray[np.float64],
    arms: NDArray[np.str_],
) -> dict[str, _BaselineCumulativeHazard]:
    baselines: dict[str, _BaselineCumulativeHazard] = {}
    for arm in sorted(set(str(value) for value in arms)):
        mask = arms == arm
        arm_times = times[mask]
        arm_events = censor_events[mask]
        arm_weights = np.exp(linear_predictor[mask])
        order = np.argsort(-arm_times, kind="mergesort")
        sorted_times = arm_times[order]
        sorted_events = arm_events[order]
        risk_weight = np.cumsum(arm_weights[order], dtype=np.float64)
        group_ends = np.flatnonzero(np.r_[sorted_times[1:] != sorted_times[:-1], True])
        group_starts = np.r_[0, group_ends[:-1] + 1]
        event_counts = np.add.reduceat(sorted_events, group_starts).astype(np.float64)
        event_groups = event_counts > 0.0
        event_times = sorted_times[group_ends[event_groups]][::-1].astype(
            np.float64, copy=True
        )
        increments = event_counts[event_groups] / risk_weight[group_ends[event_groups]]
        baselines[arm] = _BaselineCumulativeHazard(
            event_times=event_times,
            cumulative_hazard=np.cumsum(increments[::-1], dtype=np.float64),
        )
    return baselines


def _censoring_design(
    merged: pd.DataFrame, *, baseline_covariate_columns: tuple[str, ...]
) -> NDArray[np.float64]:
    columns: list[NDArray[np.float64]] = []
    baseline_names = tuple(
        sorted(set(str(column) for column in baseline_covariate_columns))
    )
    for name in baseline_names:
        if not pd.api.types.is_numeric_dtype(merged[name]):
            categorical_values = merged[name].astype("string").str.strip()
            if categorical_values.isna().any() or categorical_values.eq("").any():
                raise ValueError(
                    f"Censoring-model covariate {name!r} must be complete."
                )
            levels = tuple(sorted(str(value) for value in categorical_values.unique()))
            columns.extend(
                np.asarray(
                    categorical_values.eq(level).to_numpy(dtype=bool), dtype=np.float64
                )
                for level in levels[1:]
            )
            continue
        numeric_values = pd.to_numeric(merged[name], errors="raise").to_numpy(
            dtype=np.float64
        )
        unique = np.unique(numeric_values)
        if int(unique.size) <= 1:
            continue
        if int(unique.size) == 2:
            columns.append(numeric_values)
            continue
        scale = float(np.std(numeric_values, ddof=0))
        if not np.isfinite(numeric_values).all() or scale <= 0.0:
            raise ValueError(
                f"Censoring-model covariate {name!r} must be finite with positive variance."
            )
        columns.append((numeric_values - float(np.mean(numeric_values))) / scale)
    selected: list[NDArray[np.float64]] = []
    rank = 0
    for column in columns:
        candidate = np.column_stack((*selected, column))
        candidate_rank = int(np.linalg.matrix_rank(candidate))
        if candidate_rank > rank:
            selected.append(column)
            rank = candidate_rank
    if not selected:
        raise ValueError("Censoring model has no estimable pretreatment covariates.")
    return np.column_stack(selected).astype(np.float64, copy=False)


def _fit_censoring_model(
    merged: pd.DataFrame,
    *,
    baseline_covariate_columns: tuple[str, ...],
    administrative_horizon: float,
) -> _CensoringModel:
    if (
        not math.isfinite(float(administrative_horizon))
        or float(administrative_horizon) <= 0.0
    ):
        raise ValueError(
            "Censoring-model administrative horizon must be finite and positive."
        )
    observed_times = merged["AVAL"].to_numpy(dtype=np.float64, copy=False)
    horizon = float(administrative_horizon)
    tolerance = max(1e-10, abs(horizon) * 1e-10)
    times = np.minimum(observed_times, horizon)
    censor_events = np.asarray(
        (merged["CNSR"].to_numpy(dtype=np.int64, copy=False) == 1)
        & (observed_times < horizon - tolerance),
        dtype=np.int64,
    )
    arms = merged["TRTA"].astype("string").str.strip().to_numpy(dtype=str)
    if int(censor_events.sum()) == 0:
        empty_baseline = _BaselineCumulativeHazard(
            event_times=np.asarray([], dtype=np.float64),
            cumulative_hazard=np.asarray([], dtype=np.float64),
        )
        return _CensoringModel(
            linear_predictor=np.zeros(times.size, dtype=np.float64),
            arms=np.asarray(arms, dtype=str),
            baseline_hazard={str(arm): empty_baseline for arm in sorted(set(arms))},
        )
    design = _censoring_design(
        merged, baseline_covariate_columns=baseline_covariate_columns
    )
    retained: list[NDArray[np.float64]] = []
    for column in design.T:
        levels = np.unique(column)
        if int(levels.size) <= 2:
            supported = all(
                int(np.unique(censor_events[(arms == arm) & (column == level)]).size)
                == 2
                for arm in sorted(set(str(value) for value in arms))
                for level in levels
            )
            if not supported:
                continue
        retained.append(np.asarray(column, dtype=np.float64))
    if not retained:
        raise ValueError(
            "Censoring model has no covariates with within-arm censoring support."
        )
    design = np.column_stack(retained).astype(np.float64, copy=False)
    linear_predictor = np.empty(times.size, dtype=np.float64)
    for arm in sorted(set(str(value) for value in arms)):
        arm_mask = arms == arm
        if int(censor_events[arm_mask].sum()) == 0:
            raise ValueError(f"Censoring model is unidentified in arm {arm!r}.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fit = PHReg(
                times[arm_mask],
                design[arm_mask, :],
                status=censor_events[arm_mask],
                ties="breslow",
            ).fit(disp=0)
        coefficients = np.asarray(fit.params, dtype=np.float64)
        if not np.isfinite(coefficients).all():
            raise ValueError("Censoring model produced non-finite coefficients.")
        linear_predictor[arm_mask] = design[arm_mask, :] @ coefficients
    return _CensoringModel(
        linear_predictor=linear_predictor,
        arms=np.asarray(arms, dtype=str),
        baseline_hazard=_breslow_baselines(
            times=times,
            censor_events=censor_events,
            linear_predictor=linear_predictor,
            arms=np.asarray(arms, dtype=str),
        ),
    )


def _ipcw_support_from_model(
    *,
    merged: pd.DataFrame,
    model: _CensoringModel,
    tau: float,
    control_arm_id: str,
    treated_arm_id: str,
) -> PublicIPCWSupportDiagnosticsV1:
    if not math.isfinite(float(tau)) or float(tau) <= 0.0:
        raise ValueError("IPCW support diagnostics require finite tau > 0.")
    times = merged["AVAL"].to_numpy(dtype=np.float64, copy=False)
    censoring = merged["CNSR"].to_numpy(dtype=np.int64, copy=False)
    arms = merged["TRTA"].astype("string").to_numpy(dtype=str)
    support: dict[str, PublicIPCWArmSupportV1] = {}
    for arm_id in (str(control_arm_id), str(treated_arm_id)):
        arm_mask = arms == arm_id
        event_times = np.unique(
            times[arm_mask & (censoring == 0) & (times <= float(tau))]
        )
        event_times = np.asarray(
            event_times[np.isfinite(event_times)], dtype=np.float64
        )
        if int(event_times.size) == 0:
            raise ValueError(
                f"IPCW support diagnostics found no evaluated event time in arm {arm_id!r}."
            )
        minimum_survival = 1.0
        maximum_weight = 1.0
        minimum_ess_ratio = 1.0
        for event_time in event_times:
            weights = model.weights_at(time=float(event_time), selected_rows=arm_mask)
            if int(weights.size) == 0:
                raise ValueError(
                    f"IPCW support diagnostics found an empty randomized population in arm {arm_id!r}."
                )
            weight_sum = float(np.sum(weights))
            weight_square_sum = float(np.sum(np.square(weights)))
            ess_ratio = float(
                (weight_sum * weight_sum) / (weight_square_sum * float(weights.size))
            )
            minimum_survival = min(minimum_survival, float(np.min(1.0 / weights)))
            maximum_weight = max(maximum_weight, float(np.max(weights)))
            minimum_ess_ratio = min(minimum_ess_ratio, ess_ratio)
        support[arm_id] = PublicIPCWArmSupportV1(
            evaluated_event_time_count=int(event_times.size),
            minimum_fitted_censoring_survival=minimum_survival,
            maximum_weight=maximum_weight,
            minimum_effective_sample_size_ratio=minimum_ess_ratio,
        )
    return PublicIPCWSupportDiagnosticsV1(support_by_arm=support)


def ipcw_support_diagnostics_v1(
    *,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    paramcd: str,
    tau: float,
    control_arm_id: str,
    treated_arm_id: str,
    baseline_covariate_columns: tuple[str, ...],
) -> PublicIPCWSupportDiagnosticsV1:
    """Independently fit and evaluate the public IPCW support contract."""

    merged = _ipcw_analysis_frame(
        adsl=adsl,
        adtte=adtte,
        paramcd=paramcd,
        baseline_covariate_columns=baseline_covariate_columns,
    )
    model = _fit_censoring_model(
        merged,
        baseline_covariate_columns=baseline_covariate_columns,
        administrative_horizon=tau,
    )
    return _ipcw_support_from_model(
        merged=merged,
        model=model,
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
    )


def recompute_public_ipcw_support_v1(
    *,
    public: ZipFile,
    task_id: str,
) -> PublicIPCWSupportDiagnosticsV1:
    """Recompute IPCW support from one analysis-ready participant item."""

    prefix = f"items/{task_id}"
    task = _read_json_from_public(public, f"{prefix}/task.json")
    protocol = _read_json_from_public(public, f"{prefix}/protocol_summary.json")
    adsl = _read_required_parquet(public, f"{prefix}/data/ADSL.parquet")
    adtte = _read_required_parquet(public, f"{prefix}/data/ADTTE.parquet")
    reference = _read_required_parquet(
        public,
        f"{prefix}/data/reference_population_covariates.parquet",
    )
    baseline_columns = tuple(
        sorted(
            column
            for column in {str(value) for value in adsl.columns}
            & {str(value) for value in reference.columns}
            if column not in {"USUBJID", "REFERENCE_ID"}
        )
    )
    if not baseline_columns:
        raise ValueError(
            "IPCW support replay found no shared public baseline covariates."
        )
    return ipcw_support_diagnostics_v1(
        adsl=adsl,
        adtte=adtte,
        paramcd=_required_str(task, "primary_paramcd"),
        tau=_required_positive_float(protocol, "followup_horizon_dy"),
        control_arm_id=_required_str(task, "primary_control_arm_id"),
        treated_arm_id=_required_str(task, "primary_treated_arm_id"),
        baseline_covariate_columns=baseline_columns,
    )


def _ipcw_subject_weights(
    merged: pd.DataFrame,
    *,
    baseline_covariate_columns: tuple[str, ...],
    administrative_horizon: float,
) -> NDArray[np.float64]:
    model = _fit_censoring_model(
        merged,
        baseline_covariate_columns=baseline_covariate_columns,
        administrative_horizon=administrative_horizon,
    )
    query_times = np.minimum(
        merged["AVAL"].to_numpy(dtype=np.float64, copy=False),
        float(administrative_horizon),
    )
    return model.weights(query_times)


def _ipcw_km_family_from_frame(
    *,
    merged: pd.DataFrame,
    tau: float,
    control_arm_id: str,
    treated_arm_id: str,
    baseline_covariate_columns: tuple[str, ...],
    _include_nuisance_uncertainty: bool = True,
    _require_support_pass: bool = False,
) -> _KmFamilyResult:
    if not math.isfinite(float(tau)) or float(tau) <= 0.0:
        raise ValueError("IPCW KM public replay requires finite tau > 0.")
    trta = merged["TRTA"].astype("string")
    aval = merged["AVAL"].to_numpy(dtype=np.float64, copy=False)
    cnsr = merged["CNSR"].to_numpy(dtype=np.int64, copy=False)
    aval_tau = np.minimum(aval, float(tau)).astype(np.float64, copy=False)
    event_tau = ((cnsr == 0) & (aval <= float(tau) + 1e-12)).astype(
        np.int64, copy=False
    )
    censoring_model = _fit_censoring_model(
        merged,
        baseline_covariate_columns=baseline_covariate_columns,
        administrative_horizon=tau,
    )
    if _require_support_pass:
        support = _ipcw_support_from_model(
            merged=merged,
            model=censoring_model,
            tau=tau,
            control_arm_id=control_arm_id,
            treated_arm_id=treated_arm_id,
        )
        if support.support_status != "pass":
            raise ValueError(
                f"IPCW KM route lacks point-support qualification: {support.support_status}."
            )
    for arm_id in (str(control_arm_id), str(treated_arm_id)):
        arm_mask = np.asarray((trta == arm_id).to_numpy(dtype=bool), dtype=bool)
        if not bool(arm_mask.any()):
            raise ValueError(
                "IPCW KM public replay requires non-empty rows in both arms."
            )

    control = _ipcw_km_arm(
        arm_id=str(control_arm_id),
        trta=trta,
        aval_tau=aval_tau,
        event_tau=event_tau,
        tau=tau,
        censoring_model=censoring_model,
    )
    treated = _ipcw_km_arm(
        arm_id=str(treated_arm_id),
        trta=trta,
        aval_tau=aval_tau,
        event_tau=event_tau,
        tau=tau,
        censoring_model=censoring_model,
    )
    result = _KmFamilyResult(
        risk_difference=float(treated[0] - control[0]),
        rmst_difference=float(treated[1] - control[1]),
        risk_difference_se=float(
            math.sqrt(float(control[2]) ** 2 + float(treated[2]) ** 2)
        ),
        rmst_difference_se=float(
            math.sqrt(float(control[3]) ** 2 + float(treated[3]) ** 2)
        ),
    )
    if not bool(_include_nuisance_uncertainty):
        return result
    units = merged.loc[:, ["USUBJID", "TRTA"]].copy()
    units["USUBJID"] = units["USUBJID"].astype("string")
    units["TRTA"] = units["TRTA"].astype("string")
    if units.duplicated(subset=["USUBJID"]).any():
        raise ValueError("IPCW jackknife requires one endpoint row per participant.")
    groups = balanced_delete_groups_v1(
        unit_ids=units["USUBJID"].to_numpy(dtype=str),
        strata=units["TRTA"].to_numpy(dtype=str),
        n_groups=DELETE_GROUP_COUNT_V1,
    )
    risk_replicates: list[float] = []
    rmst_replicates: list[float] = []
    for group in range(DELETE_GROUP_COUNT_V1):
        retained_ids = set(
            units.loc[groups != group, "USUBJID"].astype("string").tolist()
        )
        replicate = _ipcw_km_family_from_frame(
            merged=merged.loc[merged["USUBJID"].astype("string").isin(retained_ids), :],
            tau=tau,
            control_arm_id=control_arm_id,
            treated_arm_id=treated_arm_id,
            baseline_covariate_columns=baseline_covariate_columns,
            _include_nuisance_uncertainty=False,
            _require_support_pass=False,
        )
        risk_replicates.append(float(replicate.risk_difference))
        rmst_replicates.append(float(replicate.rmst_difference))
    return _KmFamilyResult(
        risk_difference=float(result.risk_difference),
        rmst_difference=float(result.rmst_difference),
        risk_difference_se=delete_group_standard_error_v1(risk_replicates),
        rmst_difference_se=delete_group_standard_error_v1(rmst_replicates),
    )


def _ipcw_km_arm(
    *,
    arm_id: str,
    trta: pd.Series,
    aval_tau: NDArray[np.float64],
    event_tau: NDArray[np.int64],
    tau: float,
    censoring_model: _CensoringModel,
) -> tuple[float, float, float, float]:
    arm_mask = np.asarray((trta == str(arm_id)).to_numpy(dtype=bool), dtype=bool)
    t_arm = aval_tau[arm_mask]
    e_arm = event_tau[arm_mask]
    if int(t_arm.size) <= 0:
        raise ValueError("IPCW KM public replay requires non-empty arm rows.")
    event_times = np.asarray(np.unique(t_arm[e_arm == 1]), dtype=np.float64)
    event_times = event_times[np.isfinite(event_times)]
    event_times.sort(kind="mergesort")
    survival = 1.0
    rmst = 0.0
    previous = 0.0
    survival_after: list[float] = []
    green_terms: list[float] = []
    used_event_times: list[float] = []
    for event_time in event_times.tolist():
        event_time_f = float(event_time)
        if event_time_f <= previous + 1e-12:
            continue
        rmst += float((event_time_f - previous) * survival)
        previous = event_time_f
        at_risk_mask = arm_mask & (aval_tau >= event_time_f - 1e-12)
        event_mask = (
            arm_mask
            & (event_tau == 1)
            & np.isclose(aval_tau, event_time_f, rtol=0.0, atol=1e-12)
        )
        query = np.full(aval_tau.shape, event_time_f, dtype=np.float64)
        weights = censoring_model.weights(query)[at_risk_mask]
        y_total = float(np.sum(weights))
        d_total = float(np.sum(weights[event_mask[at_risk_mask]]))
        if not y_total > 0.0:
            raise ValueError(
                "IPCW KM public replay encountered non-positive weighted risk set."
            )
        if d_total <= 0.0:
            continue
        denom = float(y_total - d_total)
        term = 0.0 if denom <= 1e-12 else float(d_total / (y_total * denom))
        hazard = float(d_total / y_total)
        survival = float(survival * (1.0 - hazard))
        used_event_times.append(float(event_time_f))
        survival_after.append(float(survival))
        green_terms.append(float(term))
        if survival <= 1e-15:
            break
    if float(tau) > previous + 1e-12:
        rmst += float((float(tau) - previous) * survival)
    risk = float(1.0 - survival)
    if not (math.isfinite(risk) and math.isfinite(rmst) and math.isfinite(survival)):
        raise ValueError("IPCW KM public replay produced non-finite point estimates.")
    if not used_event_times:
        return float(risk), float(rmst), 0.0, 0.0
    greenwood = float(
        np.sum(np.asarray(green_terms, dtype=np.float64), dtype=np.float64)
    )
    se_risk = float(math.sqrt(max(0.0, float(survival * survival * greenwood))))
    boundaries = np.asarray([0.0, *used_event_times, float(tau)], dtype=np.float64)
    interval_survival = np.asarray([1.0, *survival_after], dtype=np.float64)
    lengths = np.diff(boundaries)
    if lengths.shape != interval_survival.shape:
        raise ValueError("IPCW KM public replay encountered invalid interval shapes.")
    areas = lengths * interval_survival
    suffix_area = np.cumsum(areas[::-1], dtype=np.float64)[::-1]
    a = suffix_area[1 : 1 + len(used_event_times)]
    terms = np.asarray(green_terms, dtype=np.float64)
    if a.shape != terms.shape:
        raise ValueError("IPCW KM public replay encountered invalid Greenwood arrays.")
    se_rmst = float(
        math.sqrt(max(0.0, float(np.sum((a * a) * terms, dtype=np.float64))))
    )
    return float(risk), float(rmst), float(se_risk), float(se_rmst)


def _bounds_rd_tau(
    *,
    covariates: pd.DataFrame,
    adtte: pd.DataFrame,
    paramcd: str,
    tau: float,
    control_arm_id: str,
    treated_arm_id: str,
    delta: float,
) -> tuple[float, float]:
    if float(delta) <= 0.0 or float(delta) > 1.0:
        raise ValueError("Tau-bound delta must be in (0, 1].")
    primary = _primary_adtte_rows(adtte=adtte, paramcd=paramcd)
    required_covariates = {"USUBJID", "TRTA"}
    missing_covariates = sorted(
        required_covariates - {str(column) for column in covariates.columns}
    )
    if missing_covariates:
        raise ValueError(
            f"Tau-bound covariate table is missing required columns: {missing_covariates!r}."
        )
    merged = primary.merge(
        covariates.loc[:, sorted(required_covariates)],
        on="USUBJID",
        how="left",
        validate="many_to_one",
    )
    if merged[["TRTA"]].isna().any().any():
        raise ValueError(
            "Tau-bound public evidence requires complete treatment assignments."
        )
    merged["AVAL"] = pd.to_numeric(merged["AVAL"], errors="raise").astype("float64")
    merged["CNSR"] = pd.to_numeric(merged["CNSR"], errors="raise").astype("int64")
    if not np.isfinite(merged["AVAL"].to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("Tau-bound ADTTE AVAL contains non-finite values.")
    if not set(int(value) for value in merged["CNSR"].unique()) <= {0, 1}:
        raise ValueError(
            "Tau-bound ADTTE CNSR must use ADaM event/censoring codes 0 or 1."
        )
    t = merged["AVAL"].to_numpy(dtype=np.float64, copy=False)
    cnsr = merged["CNSR"].to_numpy(dtype=np.int64, copy=False)
    trta = merged["TRTA"].astype("string").to_numpy()

    def _arm_bounds(arm_id: str) -> tuple[float, float]:
        idx_arm = trta == str(arm_id)
        n_arm = int(np.sum(idx_arm))
        if n_arm <= 0:
            raise ValueError(
                "Tau-bound recomputation requires non-empty rows in both arms."
            )
        arm_t = t[idx_arm]
        arm_cnsr = cnsr[idx_arm]
        event_by_tau = int(((arm_cnsr == 0) & (arm_t <= float(tau))).sum())
        censored_before_tau = int(((arm_cnsr != 0) & (arm_t < float(tau))).sum())
        if float(delta) >= 1.0:
            return (
                float(event_by_tau) / float(n_arm),
                float(event_by_tau + censored_before_tau) / float(n_arm),
            )
        known_n = int(n_arm - censored_before_tau)
        if known_n <= 0:
            raise ValueError(
                "Tau-bound sensitivity is unidentified when an arm has no observed tau status."
            )
        observed_probability = float(event_by_tau) / float(known_n)
        lower_events = float(event_by_tau) + max(
            0.0, observed_probability - float(delta)
        ) * float(censored_before_tau)
        upper_events = float(event_by_tau) + min(
            1.0, observed_probability + float(delta)
        ) * float(censored_before_tau)
        return float(lower_events) / float(n_arm), float(upper_events) / float(n_arm)

    control_low, control_high = _arm_bounds(control_arm_id)
    treated_low, treated_high = _arm_bounds(treated_arm_id)
    return float(treated_low - control_high), float(treated_high - control_low)


def _primary_adtte_rows(*, adtte: pd.DataFrame, paramcd: str) -> pd.DataFrame:
    want = str(paramcd)
    view = adtte.loc[adtte["PARAMCD"].astype("string") == want].copy()
    if not view.empty:
        return cast(pd.DataFrame, view)
    unique_paramcds = tuple(
        sorted(set(str(value) for value in adtte["PARAMCD"].astype("string").tolist()))
    )
    if unique_paramcds == ("__RECONSTRUCTED_PRIMARY__",):
        return cast(pd.DataFrame, adtte.copy())
    raise ValueError(f"ADTTE contains no rows for primary_paramcd={want!r}.")
