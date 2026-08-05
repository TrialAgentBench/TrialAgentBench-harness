"""Participant-only diagnostics for TrialEvalBench assumption evidence."""

from __future__ import annotations

import hashlib
import json
import math
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import chi2
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from trialagentbench_harness.contracts.analysis.diagnostic_proof import (
    ParticipantAssumptionDiagnosticV1,
)
from trialagentbench_harness.contracts.scoring.diagnostic_registry import diagnostic_normal_critical_value_v1


class ParticipantDiagnosticEstimationError(ValueError):
    """Raised when a required participant-visible diagnostic is not estimable."""


def derive_participant_covariate_basis_v1(
    *, covariates: pd.DataFrame, reference_covariates: pd.DataFrame
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return numeric and continuous covariates shared by trial and target tables."""

    if "USUBJID" not in covariates.columns:
        raise ValueError("Participant covariates require USUBJID.")
    if "REFERENCE_ID" not in reference_covariates.columns:
        raise ValueError("Participant reference-population covariates require REFERENCE_ID.")
    reference_columns = {str(column) for column in reference_covariates.columns}
    numeric = tuple(
        sorted(
            str(column)
            for column in covariates.columns
            if str(column) != "USUBJID"
            and str(column) in reference_columns
            and pd.api.types.is_numeric_dtype(covariates[column])
            and pd.api.types.is_numeric_dtype(reference_covariates[str(column)])
        )
    )
    if not numeric:
        raise ValueError("Participant trial and reference tables share no numeric analysis covariates.")
    continuous = tuple(
        column
        for column in numeric
        if pd.api.types.is_float_dtype(covariates[column])
        and pd.api.types.is_float_dtype(reference_covariates[column])
    )
    return numeric, continuous


def _standardized_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, npt.NDArray[np.float64]]:
    standardized: dict[str, npt.NDArray[np.float64]] = {}
    for name in columns:
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Participant diagnostic requires finite covariate {name!r}.")
        scale = float(np.std(values, ddof=0))
        if scale > 0.0:
            standardized[name] = np.asarray((values - float(np.mean(values))) / scale, dtype=np.float64)
    return standardized


def _randomization_diagnostic(
    *, subject_frame: pd.DataFrame, covariates: pd.DataFrame, baseline_columns: tuple[str, ...]
) -> ParticipantAssumptionDiagnosticV1 | None:
    frame = subject_frame.loc[:, ["USUBJID", "allocation_group"]].merge(
        covariates,
        on="USUBJID",
        how="left",
        validate="one_to_one",
    )
    groups = tuple(sorted(frame["allocation_group"].astype("string").unique()))
    if len(groups) < 2:
        raise ValueError("Participant randomization diagnostic requires at least two allocation groups.")
    columns = baseline_columns
    effects: list[float] = []
    for name in columns:
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float64)
        allocation = frame["allocation_group"].astype("string").to_numpy()
        for left_index, left_group in enumerate(groups[:-1]):
            left = values[allocation == left_group]
            for right_group in groups[left_index + 1 :]:
                right = values[allocation == right_group]
                if left.size < 2 or right.size < 2:
                    continue
                pooled = float(np.sqrt(0.5 * (np.var(left, ddof=1) + np.var(right, ddof=1))))
                effects.append(0.0 if pooled <= 0.0 else abs(float(np.mean(right) - np.mean(left))) / pooled)
    if not effects:
        return None
    severity = float(max(effects))
    return ParticipantAssumptionDiagnosticV1(
        assumption_id="randomization_integrity",
        severity_metric_name="max_abs_smd",
        severity_metric=severity,
        supporting_metrics={
            "max_abs_smd": severity,
            "n_covariates_checked": float(len(columns)),
            "n_allocation_groups": float(len(groups)),
        },
    )


def _censoring_diagnostic(
    *,
    subject_frame: pd.DataFrame,
    survival: pd.DataFrame,
    covariates: pd.DataFrame,
    baseline_columns: tuple[str, ...],
) -> ParticipantAssumptionDiagnosticV1 | None:
    columns = baseline_columns
    frame = survival.loc[:, ["USUBJID", "duration", "event"]].merge(
        subject_frame.loc[:, ["USUBJID", "allocation_group"]],
        on="USUBJID",
        validate="one_to_one",
    )
    frame = frame.merge(covariates.loc[:, ["USUBJID", *columns]], on="USUBJID", validate="one_to_one")
    standardized = _standardized_columns(frame, columns)
    if not standardized:
        return None
    allocation = frame["allocation_group"].astype("string")
    allocation_levels = tuple(sorted(allocation.unique()))
    if len(allocation_levels) < 2:
        raise ValueError("Censoring diagnostics require at least two randomized groups.")
    duration = frame["duration"].to_numpy(dtype=np.float64)
    censored = frame["event"].to_numpy(dtype=np.int64) == 0
    horizon = float(np.max(duration[censored])) if np.any(censored) else float(np.max(duration))
    censor_event = np.asarray(censored & (duration < horizon - max(1e-8, abs(horizon) * 1e-8)), dtype=np.int64)
    metrics: dict[str, float] = {
        "administrative_horizon_dy": horizon,
        "n_early_censoring_events": float(np.sum(censor_event)),
        "early_censoring_fraction": float(np.mean(censor_event)),
        "n_allocation_groups": float(len(allocation_levels)),
        "n_baseline_censoring_covariates": float(len(standardized)),
        "n_candidate_censoring_models": float(len(standardized) * len(allocation_levels)),
    }
    n_censor_events = int(np.sum(censor_event))
    if n_censor_events == 0:
        metrics.update(
            {
                "n_estimable_censoring_models": 0.0,
                "diagnostic_estimable": 0.0,
                "abs_prognostic_censoring_log_hr": 0.0,
                "lower_abs_prognostic_censoring_log_hr": 0.0,
                "prognostic_censoring_z": 0.0,
            }
        )
        return ParticipantAssumptionDiagnosticV1(
            assumption_id="censoring_ignorability",
            severity_metric_name="lower_abs_prognostic_censoring_log_hr",
            severity_metric=0.0,
            supporting_metrics=metrics,
        )
    critical = diagnostic_normal_critical_value_v1(
        assumption_id="censoring_ignorability",
        comparisons=len(standardized) * len(allocation_levels),
    )
    candidates: list[tuple[float, str, float, float]] = []
    for name, values in standardized.items():
        covariate_candidates: list[tuple[float, str, float, float]] = []
        for level in allocation_levels:
            arm_mask = allocation.eq(level).to_numpy()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=ConvergenceWarning)
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    fit = PHReg(
                        duration[arm_mask],
                        values[arm_mask, None],
                        status=censor_event[arm_mask],
                        ties="breslow",
                    ).fit(disp=False)
                    coefficient = float(fit.params[0])
                    standard_error = float(fit.bse[0])
            except (np.linalg.LinAlgError, ConvergenceWarning, RuntimeWarning):
                continue
            if np.isfinite(coefficient) and np.isfinite(standard_error) and standard_error > 0.0:
                lower = float(max(0.0, abs(coefficient) - critical * standard_error))
                covariate_candidates.append((lower, name, coefficient, standard_error))
        if covariate_candidates:
            metrics[f"lower_abs_censoring_log_hr__{name}"] = max(candidate[0] for candidate in covariate_candidates)
            candidates.extend(covariate_candidates)
    if not candidates:
        metrics.update(
            {
                "n_estimable_censoring_models": 0.0,
                "diagnostic_estimable": 0.0,
                "multiplicity_adjusted_critical_z": float(critical),
                "abs_prognostic_censoring_log_hr": 0.0,
                "lower_abs_prognostic_censoring_log_hr": 0.0,
                "prognostic_censoring_z": 0.0,
            }
        )
        return ParticipantAssumptionDiagnosticV1(
            assumption_id="censoring_ignorability",
            severity_metric_name="lower_abs_prognostic_censoring_log_hr",
            severity_metric=0.0,
            supporting_metrics=metrics,
        )
    severity, _, coefficient, standard_error = max(candidates, key=lambda value: value[0])
    metrics.update(
        {
            "n_estimable_censoring_models": float(len(candidates)),
            "diagnostic_estimable": 1.0,
            "multiplicity_adjusted_critical_z": float(critical),
            "abs_prognostic_censoring_log_hr": abs(coefficient),
            "lower_abs_prognostic_censoring_log_hr": severity,
            "prognostic_censoring_standard_error": standard_error,
            "prognostic_censoring_z": abs(coefficient) / standard_error,
        }
    )
    return ParticipantAssumptionDiagnosticV1(
        assumption_id="censoring_ignorability",
        severity_metric_name="lower_abs_prognostic_censoring_log_hr",
        severity_metric=severity,
        supporting_metrics=metrics,
    )


def _model_form_diagnostic(
    *,
    subject_frame: pd.DataFrame,
    survival: pd.DataFrame,
    covariates: pd.DataFrame,
    continuous_columns: tuple[str, ...],
    effect_modifier_column: str,
) -> ParticipantAssumptionDiagnosticV1:
    if effect_modifier_column not in continuous_columns:
        raise ValueError(
            "Participant model-form diagnostic requires its prespecified continuous "
            f"effect modifier: {effect_modifier_column!r}."
        )
    frame = survival.loc[:, ["USUBJID", "duration", "event"]].merge(
        subject_frame.loc[:, ["USUBJID", "treatment"]], on="USUBJID", validate="one_to_one"
    )
    frame = frame.merge(
        covariates.loc[:, ["USUBJID", *continuous_columns]],
        on="USUBJID",
        validate="one_to_one",
    )
    standardized = _standardized_columns(frame, continuous_columns)
    n_events = int(frame["event"].sum())
    metrics: dict[str, float] = {
        "n_continuous_covariates": float(len(standardized)),
        "n_events": float(n_events),
    }
    if effect_modifier_column not in standardized:
        raise ValueError(
            "Participant model-form diagnostic requires variation in its prespecified "
            f"effect modifier: {effect_modifier_column!r}."
        )
    duration = frame["duration"].to_numpy(dtype=np.float64)
    event = frame["event"].to_numpy(dtype=np.int64)
    treatment = frame["treatment"].to_numpy(dtype=np.float64)
    modifier = standardized[effect_modifier_column]
    linear = np.column_stack([treatment, *standardized.values(), treatment * modifier])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        warnings.filterwarnings("error", category=RuntimeWarning)
        linear_fit = PHReg(duration, linear, status=event, ties="breslow").fit(disp=False)
    linear_likelihood = float(linear_fit.llf)
    if not np.isfinite(linear_likelihood):
        raise ValueError("Participant model-form baseline model produced a non-finite likelihood.")
    critical = diagnostic_normal_critical_value_v1(
        assumption_id="model_form",
        comparisons=2,
    )
    quadratic = modifier * modifier
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        warnings.filterwarnings("error", category=RuntimeWarning)
        quadratic_fit = PHReg(
            duration,
            np.column_stack([linear, quadratic]),
            status=event,
            ties="breslow",
        ).fit(disp=False)
        hierarchical_fit = PHReg(
            duration,
            np.column_stack([linear, quadratic, treatment * quadratic]),
            status=event,
            ties="breslow",
        ).fit(disp=False)
    quadratic_likelihood = float(quadratic_fit.llf)
    hierarchical_likelihood = float(hierarchical_fit.llf)
    candidates: list[tuple[float, float, float, float, float, str]] = []
    for coefficient, standard_error, likelihood_ratio, term_kind in (
        (
            float(quadratic_fit.params[-1]),
            float(quadratic_fit.bse[-1]),
            max(0.0, 2.0 * (quadratic_likelihood - linear_likelihood)),
            "quadratic",
        ),
        (
            float(hierarchical_fit.params[-1]),
            float(hierarchical_fit.bse[-1]),
            max(0.0, 2.0 * (hierarchical_likelihood - quadratic_likelihood)),
            "nonlinear_treatment_interaction",
        ),
    ):
        if not all(np.isfinite(value) for value in (coefficient, standard_error, likelihood_ratio)):
            raise ValueError("Participant model-form comparison produced non-finite results.")
        candidates.append(
            (
                max(0.0, abs(coefficient) - critical * standard_error),
                abs(coefficient),
                standard_error,
                likelihood_ratio,
                float(chi2.sf(likelihood_ratio, 1)),
                term_kind,
            )
        )
    severity, coefficient, standard_error, likelihood_ratio, raw_p_value, term_kind = max(
        candidates, key=lambda value: value[0]
    )
    metrics.update(
        {
            "max_abs_omitted_model_term_log_hazard": coefficient,
            "simultaneous_lower_abs_omitted_model_term_log_hazard": severity,
            "omitted_model_term_standard_error": standard_error,
            "selected_term_is_treatment_interaction": float(term_kind == "nonlinear_treatment_interaction"),
            "selected_term_is_nonlinear_treatment_interaction": float(term_kind == "nonlinear_treatment_interaction"),
            "likelihood_ratio": likelihood_ratio,
            "multiplicity_adjusted_p_value": min(1.0, raw_p_value * len(candidates)),
        }
    )
    return ParticipantAssumptionDiagnosticV1(
        assumption_id="model_form",
        severity_metric_name="simultaneous_lower_abs_omitted_model_term_log_hazard",
        severity_metric=float(severity),
        supporting_metrics=metrics,
    )


def _cluster_diagnostic(
    *, subject_frame: pd.DataFrame, survival: pd.DataFrame
) -> ParticipantAssumptionDiagnosticV1 | None:
    if "SITEID" not in subject_frame.columns:
        return None
    frame = subject_frame.loc[:, ["USUBJID", "SITEID", "allocation_group"]].merge(
        survival.loc[:, ["USUBJID", "event"]], on="USUBJID", validate="one_to_one"
    )
    frame["residual"] = frame["event"] - frame.groupby("allocation_group")["event"].transform("mean")
    grouped = frame.groupby("SITEID", sort=False)["residual"]
    sizes = grouped.size().astype(float)
    means = grouped.mean().astype(float)
    n_subjects = float(sizes.sum())
    n_clusters = len(sizes)
    if n_clusters < 2 or n_subjects <= n_clusters:
        return None
    grand = float(np.average(means, weights=sizes))
    between = float(np.sum(sizes * np.square(means - grand))) / float(n_clusters - 1)
    mapped = frame["SITEID"].map(means)
    within = float(np.sum(np.square(frame["residual"] - mapped))) / float(n_subjects - n_clusters)
    effective_size = float((n_subjects - float(np.sum(np.square(sizes))) / n_subjects) / float(n_clusters - 1))
    denominator = between + (effective_size - 1.0) * within
    icc = float(np.clip(0.0 if denominator <= 0.0 else (between - within) / denominator, 0.0, 1.0))
    design_effect = 1.0 + max(0.0, effective_size - 1.0) * icc
    severity = float(1.0 - 1.0 / design_effect)
    return ParticipantAssumptionDiagnosticV1(
        assumption_id="cluster_structure",
        severity_metric_name="cluster_information_loss_fraction",
        severity_metric=severity,
        supporting_metrics={
            "icc_event_by_site": icc,
            "effective_cluster_size": effective_size,
            "cluster_design_effect": design_effect,
            "cluster_information_loss_fraction": severity,
        },
    )


def _secular_diagnostic(
    *,
    subject_frame: pd.DataFrame,
    survival: pd.DataFrame,
    covariates: pd.DataFrame,
    baseline_columns: tuple[str, ...],
) -> ParticipantAssumptionDiagnosticV1 | None:
    timing = subject_frame if "INTERVENTION_START_DY" in subject_frame.columns else covariates
    if "INTERVENTION_START_DY" not in timing.columns:
        return None
    if "RFSTDTC" not in subject_frame.columns:
        raise ValueError("Closed-cohort stepped-wedge diagnostics require participant-visible RFSTDTC.")
    baseline_dates = pd.to_datetime(subject_frame["RFSTDTC"], errors="raise", utc=False)
    if int(baseline_dates.nunique(dropna=False)) != 1:
        raise ValueError("Closed-cohort stepped-wedge diagnostics require one common RFSTDTC study baseline.")
    timing_columns = ["USUBJID", "INTERVENTION_START_DY"]
    if "SITEID" in timing.columns:
        timing_columns.append("SITEID")
    frame = survival.loc[:, ["USUBJID", "duration", "event"]].merge(
        timing.loc[:, timing_columns], on="USUBJID", validate="one_to_one"
    )
    if baseline_columns:
        frame = frame.merge(
            covariates.loc[:, ["USUBJID", *baseline_columns]],
            on="USUBJID",
            validate="one_to_one",
        )
    standardized = _standardized_columns(frame, baseline_columns)
    intervention_start = pd.to_numeric(frame["INTERVENTION_START_DY"], errors="raise").to_numpy(dtype=np.float64)
    period_starts = np.unique(intervention_start)
    horizon = float(frame["duration"].max())
    period_starts = np.sort(period_starts[np.isfinite(period_starts) & (period_starts < horizon)])
    if len(period_starts) < 2 or float(period_starts[0]) != 0.0:
        return None
    period_ends = np.concatenate([period_starts[1:], np.asarray([horizon])])
    cluster_indicators = np.empty((len(frame), 0), dtype=np.float64)
    n_clusters = 0
    if "SITEID" in frame.columns:
        levels = tuple(sorted(frame["SITEID"].astype("string").unique()))
        n_clusters = len(levels)
        if len(levels) > 1:
            cluster_indicators = np.column_stack(
                [(frame["SITEID"].astype("string") == level).to_numpy(dtype=np.float64) for level in levels[1:]]
            )
    outcomes: list[float] = []
    offsets: list[float] = []
    design_rows: list[list[float]] = []
    denominator = float(len(period_starts) - 1)
    duration = frame["duration"].to_numpy(dtype=np.float64)
    event = frame["event"].to_numpy(dtype=np.int64)
    for subject_index in range(len(frame)):
        for period_index, (start, end) in enumerate(zip(period_starts, period_ends, strict=True)):
            person_time = max(0.0, min(float(duration[subject_index]), float(end)) - float(start))
            if person_time <= 0.0:
                continue
            outcomes.append(
                float(event[subject_index] > 0 and duration[subject_index] > start and duration[subject_index] <= end)
            )
            offsets.append(float(np.log(person_time)))
            design_rows.append(
                [
                    1.0,
                    float(period_index) / denominator,
                    float(start >= intervention_start[subject_index]),
                    *[float(values[subject_index]) for values in standardized.values()],
                    *cluster_indicators[subject_index].tolist(),
                ]
            )
    outcome_array = np.asarray(outcomes, dtype=np.float64)
    design_array = np.asarray(design_rows, dtype=np.float64)
    if outcome_array.size == 0 or float(np.sum(outcome_array)) <= 0.0:
        raise ParticipantDiagnosticEstimationError(
            "Stepped-wedge secular diagnostic requires at least one observed event."
        )
    if np.linalg.matrix_rank(design_array) < design_array.shape[1]:
        raise ParticipantDiagnosticEstimationError("Stepped-wedge secular diagnostic design matrix is rank deficient.")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            warnings.filterwarnings("error", category=RuntimeWarning)
            result = GLM(
                outcome_array,
                design_array,
                family=Poisson(),
                offset=np.asarray(offsets),
            ).fit()
    except (np.linalg.LinAlgError, ConvergenceWarning, RuntimeWarning, ZeroDivisionError) as error:
        raise ParticipantDiagnosticEstimationError(
            "Stepped-wedge secular diagnostic model could not be estimated."
        ) from error
    coefficient = float(result.params[1])
    standard_error = float(result.bse[1])
    if not np.isfinite(coefficient) or not np.isfinite(standard_error) or standard_error <= 0.0:
        raise ParticipantDiagnosticEstimationError(
            "Stepped-wedge secular diagnostic returned invalid uncertainty estimates."
        )
    z_value = coefficient / standard_error
    critical = diagnostic_normal_critical_value_v1(assumption_id="secular_trend", comparisons=1)
    severity = float(max(0.0, abs(coefficient) - critical * standard_error))
    return ParticipantAssumptionDiagnosticV1(
        assumption_id="secular_trend",
        severity_metric_name="lower_abs_treatment_adjusted_secular_log_hazard_range",
        severity_metric=float(severity),
        supporting_metrics={
            "secular_log_hazard_range": abs(coefficient),
            "lower_abs_treatment_adjusted_secular_log_hazard_range": severity,
            "secular_period_coefficient": coefficient,
            "secular_period_standard_error": standard_error,
            "secular_period_z": z_value,
            "secular_n_events": float(np.sum(event)),
            "secular_n_periods": float(len(period_starts)),
            "secular_n_baseline_covariates": float(len(standardized)),
            "secular_n_clusters": float(n_clusters),
        },
    )


def _endpoint_diagnostic(
    *,
    validation: pd.DataFrame | None,
    ascertainment: dict[str, object] | None,
    subject_frame: pd.DataFrame,
    covariates: pd.DataFrame,
) -> ParticipantAssumptionDiagnosticV1 | None:
    if ascertainment is None and validation is None:
        return None
    metrics: dict[str, float] = {}
    validation_discordance: float | None = None
    if ascertainment is not None:
        sampling_fraction = ascertainment.get("validation_sampling_fraction")
        if isinstance(sampling_fraction, bool) or not isinstance(sampling_fraction, int | float):
            raise ValueError("Endpoint-validation design requires a numeric sampling fraction.")
        metrics["ascertainment_model_present"] = 1.0
        metrics["declared_validation_sampling_fraction"] = float(sampling_fraction)
    else:
        metrics["ascertainment_model_present"] = 0.0
    if validation is not None:
        required = {"USUBJID", "VALSTRAT", "VALIDFL", "OBSEVNT", "ADJEVNT"}
        missing = sorted(required - set(validation.columns))
        if missing:
            raise ValueError(f"Endpoint diagnostic lacks validation columns: {missing!r}.")
        allocation = subject_frame.loc[:, ["USUBJID", "allocation_group"]].drop_duplicates()
        if allocation["USUBJID"].duplicated().any():
            raise ValueError("Endpoint diagnostic requires one randomized allocation per participant.")
        joined = validation.drop(columns=["TRTA"], errors="ignore").merge(
            allocation,
            on="USUBJID",
            how="left",
            validate="many_to_one",
        )
        if joined["allocation_group"].isna().any():
            raise ValueError("Endpoint diagnostic cannot resolve randomized allocation for every validation row.")
        if ascertainment is None:
            raise ValueError("Endpoint diagnostic requires the released ascertainment design.")
        prognostic_variable = ascertainment.get("prognostic_stratum_variable")
        cutpoint_rule = ascertainment.get("prognostic_stratum_cutpoint_rule")
        if not isinstance(prognostic_variable, str) or prognostic_variable not in covariates.columns:
            raise ValueError("Endpoint diagnostic cannot resolve its prognostic-stratum variable.")
        if cutpoint_rule != "pooled_median":
            raise ValueError("Endpoint diagnostic requires the supported pooled-median stratum rule.")
        prognostic_frame = covariates.loc[:, ["USUBJID", prognostic_variable]].drop_duplicates()
        if prognostic_frame["USUBJID"].duplicated().any():
            raise ValueError("Endpoint diagnostic requires one prognostic value per participant.")
        prognostic_value = pd.to_numeric(prognostic_frame[prognostic_variable], errors="raise")
        if prognostic_value.isna().any():
            raise ValueError("Endpoint diagnostic requires complete prognostic-stratum values.")
        cutpoint = float(prognostic_value.median())
        prognostic_frame["derived_stratum"] = prognostic_value.ge(cutpoint).map(
            {False: "lower_risk", True: "higher_risk"}
        )
        joined = joined.merge(
            prognostic_frame.loc[:, ["USUBJID", "derived_stratum"]],
            on="USUBJID",
            how="left",
            validate="many_to_one",
        )
        if joined["derived_stratum"].isna().any():
            raise ValueError("Endpoint diagnostic cannot derive every participant's validation stratum.")
        released_stratum = joined["VALSTRAT"].astype("string")
        mismatch = released_stratum.notna() & released_stratum.ne(joined["derived_stratum"].astype("string"))
        if mismatch.any():
            raise ValueError("Released endpoint-validation strata disagree with the declared derivation rule.")
        joined["VALSTRAT"] = joined["derived_stratum"].astype("string")
        selected = joined.loc[
            pd.to_numeric(joined["VALIDFL"], errors="coerce").eq(1),
            :,
        ]
        if selected.empty or selected["ADJEVNT"].isna().any():
            raise ValueError("Endpoint diagnostic requires complete selected adjudication records.")
        observed = pd.to_numeric(selected["OBSEVNT"], errors="raise").astype("int64")
        adjudicated = pd.to_numeric(selected["ADJEVNT"], errors="raise").astype("int64")
        discordance = float((observed != adjudicated).mean())
        strata = tuple(sorted(str(value) for value in joined["VALSTRAT"].astype("string").dropna().unique()))
        arms = tuple(sorted(str(value) for value in joined["allocation_group"].astype("string").dropna().unique()))
        if len(strata) < 2 or len(arms) < 2:
            raise ValueError("Endpoint diagnostic requires at least two validation strata and treatment arms.")
        supported_cells = {
            (str(arm), str(stratum))
            for arm, stratum in selected[["allocation_group", "VALSTRAT"]]
            .astype("string")
            .itertuples(index=False, name=None)
        }
        unsupported_strata = tuple(
            stratum for stratum in strata if any((arm, stratum) not in supported_cells for arm in arms)
        )
        cell_counts = selected.groupby(["allocation_group", "VALSTRAT"], observed=True).size()
        unsupported_fraction = float(len(unsupported_strata) / len(strata))
        metrics["validation_sampling_fraction"] = float(len(selected) / len(joined))
        metrics["n_validated_endpoint_records"] = float(len(selected))
        metrics["minimum_validated_records_per_arm_stratum"] = float(
            min(int(cell_counts.get((arm, stratum), 0)) for arm in arms for stratum in strata)
        )
        metrics["validation_discordance_fraction"] = discordance
        metrics["unsupported_validation_stratum_fraction"] = unsupported_fraction
        metrics["n_validation_strata"] = float(len(strata))
        metrics["n_unsupported_validation_strata"] = float(len(unsupported_strata))
        validation_discordance = discordance
    if validation_discordance is None:
        return None
    return ParticipantAssumptionDiagnosticV1(
        assumption_id="endpoint_ascertainment",
        severity_metric_name="validation_discordance_fraction",
        severity_metric=validation_discordance,
        supporting_metrics=metrics,
    )


def _interim_diagnostic(protocol: dict[str, object]) -> ParticipantAssumptionDiagnosticV1 | None:
    raw_plan = protocol.get("group_sequential_plan") or protocol.get("interim_monitoring_plan")
    if raw_plan is None:
        return None
    if not isinstance(raw_plan, dict):
        raise ValueError("Sequential-design plan must be a JSON object.")
    plan = dict(raw_plan)
    checksum = plan.pop("checksum", None)
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("Sequential-design plan requires a SHA-256 checksum.")
    canonical = json.dumps(
        plan,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != checksum:
        raise ValueError("Sequential-design plan checksum mismatch.")
    looks_raw = plan.get("looks")
    alpha_by_look_raw = plan.get("nominal_two_sided_alpha_by_look")
    critical_by_look_raw = plan.get("z_critical_by_look")
    if (
        not isinstance(looks_raw, list)
        or not isinstance(alpha_by_look_raw, list)
        or not isinstance(critical_by_look_raw, list)
    ):
        raise ValueError("Sequential-design plan requires looks, nominal alpha, and critical-value arrays.")
    looks = np.asarray(looks_raw, dtype=np.float64)
    alpha_by_look = np.asarray(alpha_by_look_raw, dtype=np.float64)
    critical_by_look = np.asarray(critical_by_look_raw, dtype=np.float64)
    if looks.size < 2 or alpha_by_look.size != looks.size or critical_by_look.size != looks.size:
        raise ValueError("Sequential-design arrays must have the same length and at least two looks.")
    if not np.isfinite(looks).all() or not np.isfinite(alpha_by_look).all() or not np.isfinite(critical_by_look).all():
        raise ValueError("Sequential-design arrays must contain finite values.")
    if np.any(np.diff(looks) <= 0.0) or not math.isclose(float(looks[-1]), 1.0, abs_tol=1e-12):
        raise ValueError("Sequential information fractions must increase strictly and end at one.")
    overall_alpha = plan.get("two_sided_alpha")
    if isinstance(overall_alpha, bool) or not isinstance(overall_alpha, int | float):
        raise ValueError("Sequential-design plan requires numeric two_sided_alpha.")
    alpha = float(overall_alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("Sequential-design two_sided_alpha must lie in (0, 1).")
    if np.any(alpha_by_look <= 0.0) or np.any(alpha_by_look > alpha) or np.any(np.diff(alpha_by_look) < 0.0):
        raise ValueError("Sequential nominal alpha must be positive, nondecreasing, and no greater than total alpha.")
    if np.any(critical_by_look <= 0.0) or np.any(np.diff(critical_by_look) > 0.0):
        raise ValueError("Sequential critical values must be positive and nonincreasing.")
    for field in ("monitoring_effect_scale", "spending_function_id"):
        if not isinstance(plan.get(field), str) or not str(plan[field]).strip():
            raise ValueError(f"Sequential-design plan requires non-empty {field}.")
    return ParticipantAssumptionDiagnosticV1(
        assumption_id="sequential_design_adjustment",
        severity_metric_name="declared_sequential_design",
        severity_metric=1.0,
        supporting_metrics={
            "declared_sequential_design": 1.0,
            "n_looks": float(looks.size),
            "two_sided_alpha": alpha,
            "final_nominal_two_sided_alpha": float(alpha_by_look[-1]),
            "final_critical_z": float(critical_by_look[-1]),
        },
    )


def compute_participant_assumption_diagnostics_v1(
    *,
    subject_frame: pd.DataFrame,
    survival: pd.DataFrame,
    covariates: pd.DataFrame,
    reference_covariates: pd.DataFrame,
    operational: pd.DataFrame | None,
    ascertainment: dict[str, object] | None,
    endpoint_validation: pd.DataFrame | None = None,
    protocol: dict[str, object],
    include_treatment_dependent_diagnostics: bool = True,
    treatment_effect_modifier: str | None = None,
) -> tuple[ParticipantAssumptionDiagnosticV1, ...]:
    """Compute all estimable diagnostics without evaluator labels or thresholds."""

    baseline_columns, continuous_columns = derive_participant_covariate_basis_v1(
        covariates=covariates,
        reference_covariates=reference_covariates,
    )
    candidates: list[ParticipantAssumptionDiagnosticV1 | None] = [
        _randomization_diagnostic(
            subject_frame=subject_frame,
            covariates=covariates,
            baseline_columns=baseline_columns,
        ),
        _censoring_diagnostic(
            subject_frame=subject_frame,
            survival=survival,
            covariates=covariates,
            baseline_columns=baseline_columns,
        ),
    ]
    if include_treatment_dependent_diagnostics and treatment_effect_modifier is not None:
        candidates.append(
            _model_form_diagnostic(
                subject_frame=subject_frame,
                survival=survival,
                covariates=covariates,
                continuous_columns=continuous_columns,
                effect_modifier_column=treatment_effect_modifier,
            )
        )
    diagnostics = [diagnostic for diagnostic in candidates if diagnostic is not None]
    optional = (
        _cluster_diagnostic(subject_frame=subject_frame, survival=survival),
        _secular_diagnostic(
            subject_frame=subject_frame,
            survival=survival,
            covariates=covariates,
            baseline_columns=baseline_columns,
        ),
        _endpoint_diagnostic(
            validation=endpoint_validation,
            ascertainment=ascertainment,
            subject_frame=subject_frame,
            covariates=covariates,
        ),
        _interim_diagnostic(protocol),
    )
    diagnostics.extend(diagnostic for diagnostic in optional if diagnostic is not None)
    return tuple(sorted(diagnostics, key=lambda diagnostic: diagnostic.assumption_id))


__all__ = ["ParticipantAssumptionDiagnosticV1", "compute_participant_assumption_diagnostics_v1"]
