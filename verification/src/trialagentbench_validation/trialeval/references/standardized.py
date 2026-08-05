"""Standardized-risk calculators for TrialEval public reference replay."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from trialagentbench_validation.analysis.delete_group_jackknife import (
    DELETE_GROUP_COUNT_V1,
    balanced_delete_groups_v1,
    delete_group_standard_error_v1,
)

_COX_RCS_QUANTILES = (0.05, 0.35, 0.65, 0.95)
_STANDARDIZATION_EFFECT_MODIFIER_V1 = "BMI"
_STANDARDIZATION_NONCOVARIATE_COLUMNS_V1 = frozenset(
    {
        "STUDYID",
        "USUBJID",
        "SUBJID",
        "SITE",
        "SITEID",
        "REFERENCE_ID",
        "TRTA",
        "ARM",
        "ARMCD",
    }
)


def cox_linear_standardized_risk_difference_tau_reference_with_uncertainty_v1(
    *,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    analysis_covariates: pd.DataFrame,
    reference_covariates: pd.DataFrame,
    paramcd: str,
    tau: float,
    control_arm_id: str,
    treated_arm_id: str,
) -> tuple[float, float]:
    """Replay linear Cox standardization with complete-refit uncertainty."""

    return _cox_standardized_risk_difference_tau_reference_with_uncertainty_v1(
        adsl=adsl,
        adtte=adtte,
        analysis_covariates=analysis_covariates,
        reference_covariates=reference_covariates,
        paramcd=paramcd,
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
        continuous_basis="linear",
    )


def cox_linear_standardized_risk_difference_tau_reference_v1(
    *,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    analysis_covariates: pd.DataFrame,
    reference_covariates: pd.DataFrame,
    paramcd: str,
    tau: float,
    control_arm_id: str,
    treated_arm_id: str,
) -> float:
    """Independently replay linear Cox reference-population g-computation."""

    return _cox_standardized_risk_difference_tau_reference_v1(
        adsl=adsl,
        adtte=adtte,
        analysis_covariates=analysis_covariates,
        reference_covariates=reference_covariates,
        paramcd=paramcd,
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
        continuous_basis="linear",
    )


def cox_rcs_standardized_risk_difference_tau_reference_with_uncertainty_v1(
    *,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    analysis_covariates: pd.DataFrame,
    reference_covariates: pd.DataFrame,
    paramcd: str,
    tau: float,
    control_arm_id: str,
    treated_arm_id: str,
) -> tuple[float, float]:
    """Replay the standardized Cox contrast and complete-refit uncertainty."""

    return _cox_standardized_risk_difference_tau_reference_with_uncertainty_v1(
        adsl=adsl,
        adtte=adtte,
        analysis_covariates=analysis_covariates,
        reference_covariates=reference_covariates,
        paramcd=paramcd,
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
        continuous_basis="restricted_cubic_spline",
    )


def _cox_standardized_risk_difference_tau_reference_with_uncertainty_v1(
    *,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    analysis_covariates: pd.DataFrame,
    reference_covariates: pd.DataFrame,
    paramcd: str,
    tau: float,
    control_arm_id: str,
    treated_arm_id: str,
    continuous_basis: str,
) -> tuple[float, float]:
    """Replay one declared standardized-risk basis and jackknife uncertainty."""

    value = _cox_standardized_risk_difference_tau_reference_v1(
        adsl=adsl,
        adtte=adtte,
        analysis_covariates=analysis_covariates,
        reference_covariates=reference_covariates,
        paramcd=paramcd,
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
        continuous_basis=continuous_basis,
    )
    endpoint_ids = set(
        adtte.loc[adtte["PARAMCD"].astype("string") == str(paramcd), "USUBJID"]
        .astype("string")
        .tolist()
    )
    covariate_ids = set(analysis_covariates["USUBJID"].astype("string").tolist())
    units = adsl.loc[
        adsl["USUBJID"].astype("string").isin(endpoint_ids & covariate_ids),
        ["USUBJID", "TRTA"],
    ].copy()
    units["USUBJID"] = units["USUBJID"].astype("string")
    units["TRTA"] = units["TRTA"].astype("string")
    if (
        units.duplicated(subset=["USUBJID"]).any()
        or units[["USUBJID", "TRTA"]].isna().any().any()
    ):
        raise ValueError(
            "Standardized-risk jackknife requires unique, complete participant assignments."
        )
    groups = balanced_delete_groups_v1(
        unit_ids=units["USUBJID"].to_numpy(dtype=str),
        strata=units["TRTA"].to_numpy(dtype=str),
        n_groups=DELETE_GROUP_COUNT_V1,
    )
    replicates: list[float] = []
    for group in range(DELETE_GROUP_COUNT_V1):
        retained_ids = set(
            units.loc[groups != group, "USUBJID"].astype("string").tolist()
        )
        replicates.append(
            _cox_standardized_risk_difference_tau_reference_v1(
                adsl=adsl.loc[adsl["USUBJID"].astype("string").isin(retained_ids), :],
                adtte=adtte.loc[
                    adtte["USUBJID"].astype("string").isin(retained_ids), :
                ],
                analysis_covariates=analysis_covariates.loc[
                    analysis_covariates["USUBJID"].astype("string").isin(retained_ids),
                    :,
                ],
                reference_covariates=reference_covariates,
                paramcd=paramcd,
                tau=tau,
                control_arm_id=control_arm_id,
                treated_arm_id=treated_arm_id,
                continuous_basis=continuous_basis,
            )
        )
    return float(value), delete_group_standard_error_v1(replicates)


def cox_rcs_standardized_risk_difference_tau_reference_v1(
    *,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    analysis_covariates: pd.DataFrame,
    reference_covariates: pd.DataFrame,
    paramcd: str,
    tau: float,
    control_arm_id: str,
    treated_arm_id: str,
) -> float:
    """Independently replay Cox-RCS reference-population g-computation."""

    return _cox_standardized_risk_difference_tau_reference_v1(
        adsl=adsl,
        adtte=adtte,
        analysis_covariates=analysis_covariates,
        reference_covariates=reference_covariates,
        paramcd=paramcd,
        tau=tau,
        control_arm_id=control_arm_id,
        treated_arm_id=treated_arm_id,
        continuous_basis="restricted_cubic_spline",
    )


def _cox_standardized_risk_difference_tau_reference_v1(
    *,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    analysis_covariates: pd.DataFrame,
    reference_covariates: pd.DataFrame,
    paramcd: str,
    tau: float,
    control_arm_id: str,
    treated_arm_id: str,
    continuous_basis: str,
) -> float:
    """Replay a standardized Cox contrast with the declared continuous basis."""

    if not np.isfinite(float(tau)) or float(tau) <= 0.0:
        raise ValueError("Standardized risk estimator requires finite tau > 0.")
    primary = adtte.loc[adtte["PARAMCD"].astype("string") == str(paramcd), :].copy()
    if primary.empty:
        raise ValueError(
            f"Standardized risk estimator found no ADTTE rows for PARAMCD={paramcd!r}."
        )
    missing = sorted(
        {"USUBJID", "AVAL", "CNSR"} - {str(column) for column in primary.columns}
    )
    if missing:
        raise ValueError(
            f"Standardized risk estimator requires ADTTE columns: {missing!r}."
        )
    if not {"USUBJID", "TRTA"}.issubset(adsl.columns):
        raise ValueError("Standardized risk estimator requires ADSL USUBJID and TRTA.")
    if "USUBJID" not in analysis_covariates.columns:
        raise ValueError(
            "Standardized risk estimator requires analysis_covariates USUBJID."
        )

    assignments = adsl.loc[:, ["USUBJID", "TRTA"]].copy()
    assignments["USUBJID"] = assignments["USUBJID"].astype("string")
    covariates = analysis_covariates.copy()
    covariates["USUBJID"] = covariates["USUBJID"].astype("string")
    merged = primary.merge(
        assignments, on="USUBJID", how="left", validate="one_to_one"
    ).merge(covariates, on="USUBJID", how="left", validate="one_to_one")
    if merged["TRTA"].isna().any():
        raise ValueError(
            "Standardized risk estimator requires treatment for every analysis row."
        )
    reference = reference_covariates.copy()
    if reference.empty:
        raise ValueError(
            "Standardized risk estimator requires a non-empty reference population."
        )
    baseline_columns = tuple(
        sorted(
            str(column)
            for column in covariates.columns
            if str(column) not in _STANDARDIZATION_NONCOVARIATE_COLUMNS_V1
        )
    )
    if not baseline_columns:
        raise ValueError(
            "TE-S05 standardized risk requires at least one released pretreatment covariate."
        )
    missing_reference = tuple(
        column for column in baseline_columns if column not in reference.columns
    )
    if missing_reference:
        raise ValueError(
            "Standardized risk reference population lacks prespecified covariates: "
            f"{missing_reference!r}."
        )
    trial_basis, reference_basis = _cox_baseline_basis(
        analysis=merged,
        reference=reference,
        columns=baseline_columns,
        continuous_basis=continuous_basis,
    )
    if np.linalg.matrix_rank(trial_basis) != int(trial_basis.shape[1]):
        raise ValueError("TE-S05 baseline-covariate basis is rank deficient.")
    modifier_trial_basis, modifier_reference_basis = _cox_baseline_basis(
        analysis=merged,
        reference=reference,
        columns=(_STANDARDIZATION_EFFECT_MODIFIER_V1,),
        continuous_basis=continuous_basis,
    )
    arm = merged["TRTA"].astype("string").to_numpy()
    if not ((arm == str(control_arm_id)).any() and (arm == str(treated_arm_id)).any()):
        raise ValueError(
            "Standardized risk estimator requires non-empty control and treated arms."
        )
    treatment = (arm == str(treated_arm_id)).astype(np.float64)
    treatment_interactions = treatment[:, None] * modifier_trial_basis
    design = np.column_stack([treatment[:, None], trial_basis, treatment_interactions])
    if not np.isfinite(design).all() or np.linalg.matrix_rank(design) != int(
        design.shape[1]
    ):
        raise ValueError(
            "Standardized risk Cox design is non-finite or rank deficient."
        )
    times = pd.to_numeric(merged["AVAL"], errors="raise").to_numpy(dtype=np.float64)
    events = (
        pd.to_numeric(merged["CNSR"], errors="raise").to_numpy(dtype=np.int64) == 0
    ).astype(np.int64)
    if int(events.sum()) <= int(design.shape[1]):
        raise ValueError(
            "Standardized risk Cox model has no positive event-to-parameter degrees of freedom."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        fit = PHReg(times, design, status=events, ties="breslow").fit(disp=0)
    parameters = np.asarray(fit.params, dtype=np.float64)
    if not np.isfinite(parameters).all():
        raise ValueError("Standardized risk Cox fit produced non-finite parameters.")
    baseline_functions = tuple(fit.baseline_cumulative_hazard_function.values())
    if len(baseline_functions) != 1:
        raise ValueError(
            "Standardized risk Cox fit must produce one baseline cumulative hazard."
        )
    baseline_hazard = float(
        np.asarray(baseline_functions[0](float(tau)), dtype=np.float64).reshape(-1)[0]
    )
    if not np.isfinite(baseline_hazard) or baseline_hazard < 0.0:
        raise ValueError(
            "Standardized risk Cox fit produced an invalid baseline cumulative hazard."
        )

    def mean_risk(treatment_value: float) -> float:
        treatment_column = np.full(
            (int(len(reference_basis)), 1), treatment_value, dtype=np.float64
        )
        treatment_interactions = treatment_value * modifier_reference_basis
        prediction = np.column_stack(
            [treatment_column, reference_basis, treatment_interactions]
        )
        relative_hazard = np.exp(prediction @ parameters)
        if not np.isfinite(relative_hazard).all():
            raise ValueError(
                "Standardized risk prediction produced non-finite relative hazards."
            )
        cumulative_hazard = baseline_hazard * relative_hazard
        return float(np.mean(1.0 - np.exp(-cumulative_hazard)))

    return float(mean_risk(1.0) - mean_risk(0.0))


def _cox_baseline_basis(
    *,
    analysis: pd.DataFrame,
    reference: pd.DataFrame,
    columns: tuple[str, ...],
    continuous_basis: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    analysis_parts: list[NDArray[np.float64]] = []
    reference_parts: list[NDArray[np.float64]] = []
    for column in columns:
        observed = analysis[column]
        target = reference[column]
        observed_numeric = pd.to_numeric(observed, errors="coerce")
        target_numeric = pd.to_numeric(target, errors="coerce")
        if observed_numeric.notna().all() and target_numeric.notna().all():
            x = observed_numeric.to_numpy(dtype=np.float64)
            x_target = target_numeric.to_numpy(dtype=np.float64)
            if not pd.api.types.is_float_dtype(observed):
                levels = np.unique(x)
                if int(levels.size) != 2:
                    raise ValueError(
                        f"Non-continuous baseline covariate {column!r} must have exactly two levels."
                    )
                if not set(np.unique(x_target)).issubset(set(np.unique(x))):
                    raise ValueError(
                        f"Reference population has unsupported values for binary covariate {column!r}."
                    )
                analysis_parts.append(x[:, None])
                reference_parts.append(x_target[:, None])
                continue
            standardized = _cox_standardize(x, x)
            target_standardized = _cox_standardize(x_target, x)
            if continuous_basis == "linear":
                analysis_parts.append(standardized[:, None])
                reference_parts.append(target_standardized[:, None])
                continue
            if continuous_basis != "restricted_cubic_spline":
                raise ValueError(
                    f"Unsupported standardized-risk continuous basis: {continuous_basis!r}."
                )
            knots = np.quantile(x, _COX_RCS_QUANTILES)
            if int(np.unique(knots).size) < 4:
                raise ValueError(
                    f"Covariate {column!r} does not support the prespecified four distinct RCS knots."
                )
            analysis_parts.append(_cox_rcs(x=x, knots=knots, training=x))
            reference_parts.append(_cox_rcs(x=x_target, knots=knots, training=x))
            continue
        categories = tuple(
            sorted(str(value) for value in observed.astype("string").dropna().unique())
        )
        if len(categories) < 2:
            raise ValueError(
                f"Categorical baseline covariate {column!r} has fewer than two observed levels."
            )
        if not set(
            str(value) for value in target.astype("string").dropna().unique()
        ).issubset(set(categories)):
            raise ValueError(
                f"Reference population has unsupported levels for covariate {column!r}."
            )
        for category in categories[1:]:
            analysis_parts.append(
                np.asarray(
                    (observed.astype("string") == category).to_numpy(), dtype=np.float64
                )[:, None]
            )
            reference_parts.append(
                np.asarray(
                    (target.astype("string") == category).to_numpy(), dtype=np.float64
                )[:, None]
            )
    if not analysis_parts:
        raise ValueError(
            "Standardized risk estimator produced an empty baseline-covariate basis."
        )
    return np.column_stack(analysis_parts), np.column_stack(reference_parts)


def _cox_standardize(
    values: NDArray[np.float64], training: NDArray[np.float64]
) -> NDArray[np.float64]:
    scale = float(np.std(training, ddof=1))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "Continuous baseline covariates must have positive finite variance."
        )
    return (np.asarray(values, dtype=np.float64) - float(np.mean(training))) / scale


def _cox_rcs(
    *, x: NDArray[np.float64], knots: NDArray[np.float64], training: NDArray[np.float64]
) -> NDArray[np.float64]:
    values = _cox_standardize(x, training)
    mean = float(np.mean(training))
    scale = float(np.std(training, ddof=1))
    knot_values = (np.asarray(knots, dtype=np.float64) - mean) / scale
    last_inner = float(knot_values[-2])
    last = float(knot_values[-1])
    denominator = last - last_inner

    def positive_cube(knot: float) -> NDArray[np.float64]:
        return np.maximum(values - knot, 0.0) ** 3

    d_inner = positive_cube(last_inner)
    d_last = positive_cube(last)
    basis: list[NDArray[np.float64]] = [values]
    for knot in knot_values[:-2]:
        basis.append(
            positive_cube(float(knot))
            - d_inner * ((last - float(knot)) / denominator)
            + d_last * ((last_inner - float(knot)) / denominator)
        )
    return np.column_stack(basis)
