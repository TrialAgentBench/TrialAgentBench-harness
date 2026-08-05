"""Stepped-wedge calculators for TrialEval public reference replay."""

from __future__ import annotations

from math import inf
from typing import Any, cast

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray


def stepped_wedge_period_adjusted_risk_difference_tau_v1(
    *,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    paramcd: str,
    tau: float,
    period_length_dy: float = 28.0,
) -> float:
    """Compute period-adjusted stepped-wedge tau risk difference."""

    value, _standard_error = (
        stepped_wedge_period_adjusted_risk_difference_tau_with_uncertainty_v1(
            adsl=adsl,
            adtte=adtte,
            paramcd=paramcd,
            tau=tau,
            period_length_dy=period_length_dy,
        )
    )
    return value


def stepped_wedge_period_adjusted_risk_difference_tau_with_uncertainty_v1(
    *,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    paramcd: str,
    tau: float,
    period_length_dy: float = 28.0,
) -> tuple[float, float]:
    """Compute a period-adjusted risk difference and cluster-robust delta-method SE."""

    df = _stepped_wedge_frame(adsl=adsl, adtte=adtte, paramcd=paramcd, tau=tau)
    table = _build_person_period_table(
        df=df, tau=tau, period_length_dy=period_length_dy, keep_subject_maps=True
    )
    beta_treat, baseline_hazard = _fit_period_poisson(table=table)

    def standardized_risk_difference(
        *,
        coefficient: float,
        hazards: NDArray[np.float64],
        maps: tuple[dict[int, float], ...],
    ) -> float:
        if not maps:
            raise ValueError(
                "Stepped-wedge standardization requires participant exposure histories."
            )
        relative_rate = float(np.exp(coefficient))
        cumulative = np.asarray(
            [
                sum(
                    float(hazards[period]) * float(dt) for period, dt in mapping.items()
                )
                for mapping in maps
            ],
            dtype=np.float64,
        )
        return float(np.mean(np.exp(-cumulative) - np.exp(-relative_rate * cumulative)))

    value = standardized_risk_difference(
        coefficient=beta_treat,
        hazards=baseline_hazard,
        maps=table.subject_time_by_period,
    )
    cluster_ids = sorted({str(value) for value in table.cluster})
    if len(cluster_ids) < 3:
        raise ValueError(
            "Stepped-wedge uncertainty requires at least three randomized clusters."
        )
    subject_sites = df["SITEID"].astype("string").to_numpy()
    jackknife: list[float] = []
    for cluster_id in cluster_ids:
        row_mask = np.asarray(
            [str(value) != cluster_id for value in table.cluster], dtype=bool
        )
        reduced = _SteppedPersonPeriodTable(
            period=table.period[row_mask],
            treated=table.treated[row_mask],
            dt=table.dt[row_mask],
            y=table.y[row_mask],
            cluster=table.cluster[row_mask],
            n_periods=table.n_periods,
            subject_time_by_period=tuple(
                mapping
                for mapping, site in zip(
                    table.subject_time_by_period, subject_sites, strict=True
                )
                if str(site) != cluster_id
            ),
        )
        beta_leave_one_out, hazards_leave_one_out = _fit_period_poisson(table=reduced)
        jackknife.append(
            standardized_risk_difference(
                coefficient=beta_leave_one_out,
                hazards=hazards_leave_one_out,
                maps=reduced.subject_time_by_period,
            )
        )
    jackknife_array = np.asarray(jackknife, dtype=np.float64)
    jackknife_mean = float(np.mean(jackknife_array))
    standard_error = float(
        np.sqrt(
            ((len(cluster_ids) - 1.0) / len(cluster_ids))
            * np.sum(np.square(jackknife_array - jackknife_mean))
        )
    )
    if not np.isfinite(standard_error) or standard_error <= 0.0:
        raise ValueError(
            f"Stepped-wedge cluster jackknife produced invalid standard error={standard_error!r}."
        )
    return value, standard_error


def stepped_wedge_unadjusted_risk_difference_tau_with_uncertainty_v1(
    *,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    paramcd: str,
    tau: float,
    period_length_dy: float = 28.0,
) -> tuple[float, float]:
    """Compute a period-naive stepped-wedge risk difference and robust SE."""

    df = _stepped_wedge_frame(adsl=adsl, adtte=adtte, paramcd=paramcd, tau=tau)
    table = _build_person_period_table(
        df=df, tau=tau, period_length_dy=period_length_dy, keep_subject_maps=False
    )
    design = np.column_stack((np.ones(table.y.size, dtype=np.float64), table.treated))
    fit = sm.GLM(
        table.y,
        design,
        family=sm.families.Poisson(),
        offset=np.log(table.dt),
    ).fit(cov_type="cluster", cov_kwds={"groups": table.cluster})
    coefficients = np.asarray(fit.params, dtype=np.float64)
    covariance = np.asarray(fit.cov_params(), dtype=np.float64)
    if coefficients.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError(
            "Period-naive stepped-wedge fit returned an invalid coefficient shape."
        )
    baseline_hazard = float(np.exp(coefficients[0]))
    relative_rate = float(np.exp(coefficients[1]))
    control_survival = float(np.exp(-baseline_hazard * float(tau)))
    treated_survival = float(np.exp(-baseline_hazard * relative_rate * float(tau)))
    value = float(control_survival - treated_survival)
    gradient = np.asarray(
        [
            float(tau)
            * baseline_hazard
            * (relative_rate * treated_survival - control_survival),
            float(tau) * baseline_hazard * relative_rate * treated_survival,
        ],
        dtype=np.float64,
    )
    variance = float(gradient @ covariance @ gradient)
    standard_error = float(np.sqrt(max(0.0, variance)))
    if not np.isfinite(value) or not np.isfinite(standard_error) or standard_error <= 0:
        raise ValueError(
            "Period-naive stepped-wedge analysis produced an invalid result."
        )
    return value, standard_error


def stepped_wedge_period_adjusted_baseline_rates_v1(
    *,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    paramcd: str,
    tau: float,
    period_length_dy: float = 28.0,
) -> tuple[float, ...]:
    """Estimate calendar-period baseline event rates after treatment adjustment."""

    df = _stepped_wedge_frame(adsl=adsl, adtte=adtte, paramcd=paramcd, tau=tau)
    table = _build_person_period_table(
        df=df,
        tau=tau,
        period_length_dy=period_length_dy,
        keep_subject_maps=False,
    )
    _beta_treat, baseline_hazard = _fit_period_poisson(table=table)
    observed_periods = np.unique(table.period)
    if observed_periods.size < 2:
        raise ValueError(
            "Stepped-wedge period rates require at least two observed calendar periods."
        )
    if not np.array_equal(observed_periods, np.arange(int(observed_periods[-1]) + 1)):
        raise ValueError(
            "Stepped-wedge period rates require consecutive observed calendar periods."
        )
    rates = baseline_hazard[observed_periods] * 1_000.0
    if not np.isfinite(rates).all() or bool((rates < 0.0).any()):
        raise ValueError("Stepped-wedge period rates must be finite and non-negative.")
    return tuple(float(rate) for rate in rates)


class _SteppedPersonPeriodTable:
    def __init__(
        self,
        *,
        period: NDArray[np.int64],
        treated: NDArray[np.float64],
        dt: NDArray[np.float64],
        y: NDArray[np.float64],
        cluster: NDArray[np.object_],
        n_periods: int,
        subject_time_by_period: tuple[dict[int, float], ...],
    ) -> None:
        self.period = period
        self.treated = treated
        self.dt = dt
        self.y = y
        self.cluster = cluster
        self.n_periods = n_periods
        self.subject_time_by_period = subject_time_by_period


def _stepped_wedge_frame(
    *, adsl: pd.DataFrame, adtte: pd.DataFrame, paramcd: str, tau: float
) -> pd.DataFrame:
    if not np.isfinite(float(tau)) or float(tau) <= 0.0:
        raise ValueError("Stepped-wedge estimator requires finite tau > 0.")
    required_adsl = {"USUBJID", "RFSTDTC", "INTERVENTION_START_DY", "SITEID"}
    missing_adsl = sorted(required_adsl - {str(column) for column in adsl.columns})
    if missing_adsl:
        raise ValueError(
            f"Stepped-wedge estimator requires ADSL columns: {missing_adsl!r}."
        )
    primary = adtte.loc[adtte["PARAMCD"].astype("string") == str(paramcd), :].copy()
    missing_adtte = sorted(
        {"USUBJID", "AVAL", "CNSR"} - {str(column) for column in primary.columns}
    )
    if missing_adtte:
        raise ValueError(
            f"Stepped-wedge estimator requires ADTTE columns: {missing_adtte!r}."
        )
    resolved = _resolve_stepped_wedge_switch_days_v1(
        adsl=adsl, tau_dy=float(tau), strict_missing_switch=False
    )
    df = primary.merge(
        adsl.loc[:, ["USUBJID", "RFSTDTC", "SITEID"]],
        on="USUBJID",
        how="inner",
        validate="one_to_one",
    ).merge(
        resolved.loc[
            :, ["USUBJID", "INTERVENTION_START_DY_RESOLVED", "EXPOSURE_SWITCH_STATE"]
        ],
        on="USUBJID",
        how="inner",
        validate="one_to_one",
    )
    if df.empty:
        raise ValueError("Stepped-wedge estimator requires non-empty merged rows.")
    df["AVAL"] = pd.to_numeric(df["AVAL"], errors="raise").astype("float64")
    df["CNSR"] = pd.to_numeric(df["CNSR"], errors="raise").astype("int64")
    df["INTERVENTION_START_DY_RESOLVED"] = pd.to_numeric(
        df["INTERVENTION_START_DY_RESOLVED"], errors="raise"
    ).astype("float64")
    rfstdt = pd.to_datetime(df["RFSTDTC"], errors="raise", utc=False)
    global0 = rfstdt.min()
    df["_BASE_DAY"] = ((rfstdt - global0).dt.total_seconds() / (24.0 * 3600.0)).astype(
        "float64"
    )
    df["_END_DY"] = np.minimum(df["AVAL"].to_numpy(dtype=float), float(tau))
    df["_EVENT"] = (df["CNSR"].to_numpy(dtype=int) == 0).astype(int)
    if int(df["_EVENT"].sum()) <= 0:
        raise ValueError(
            "Stepped-wedge estimator requires at least one observed event."
        )
    return cast(pd.DataFrame, df)


def _build_person_period_table(
    *,
    df: pd.DataFrame,
    tau: float,
    period_length_dy: float,
    keep_subject_maps: bool,
) -> _SteppedPersonPeriodTable:
    if not np.isfinite(float(period_length_dy)) or float(period_length_dy) <= 0.0:
        raise ValueError("period_length_dy must be finite and > 0.")
    max_global_end = float((df["_BASE_DAY"] + df["_END_DY"]).max())
    n_periods = int(np.ceil(max_global_end / float(period_length_dy))) + 2
    boundaries = np.asarray(
        [float(i) * float(period_length_dy) for i in range(n_periods)], dtype=np.float64
    )
    rows_period: list[int] = []
    rows_treated: list[int] = []
    rows_dt: list[float] = []
    rows_y: list[int] = []
    rows_cluster: list[str] = []
    subject_time_by_period: list[dict[int, float]] = []

    base_days = df["_BASE_DAY"].to_numpy(dtype=np.float64, copy=False)
    ends = df["_END_DY"].to_numpy(dtype=np.float64, copy=False)
    events = df["_EVENT"].to_numpy(dtype=np.int64, copy=False)
    switches = df["INTERVENTION_START_DY_RESOLVED"].to_numpy(
        dtype=np.float64, copy=False
    )
    avals = df["AVAL"].to_numpy(dtype=np.float64, copy=False)
    sites = df["SITEID"].astype("string").to_numpy()
    for i in range(int(len(df))):
        base = float(base_days[i])
        end = float(ends[i])
        switch_global = float(base + float(switches[i]))
        start_global = float(base)
        end_global = float(base + float(end))
        cutpoints = [start_global, end_global, switch_global]
        left_idx = max(
            0, int(np.searchsorted(boundaries, start_global, side="right") - 1)
        )
        right_idx = min(
            int(boundaries.size - 1),
            int(np.searchsorted(boundaries, end_global, side="left") + 1),
        )
        cutpoints.extend(float(boundaries[j]) for j in range(left_idx, right_idx + 1))
        cut = sorted(
            {
                float(value)
                for value in cutpoints
                if np.isfinite(float(value))
                and start_global <= float(value) <= end_global
            }
        )
        cut = [cut[0]] + [
            cut[j] for j in range(1, len(cut)) if cut[j] > cut[j - 1] + 1e-12
        ]
        seg_rows: list[int] = []
        seg_starts: list[float] = []
        seg_ends: list[float] = []
        subj_map: dict[int, float] = {}
        for a, b in zip(cut[:-1], cut[1:], strict=False):
            if b <= a + 1e-12:
                continue
            mid = 0.5 * (a + b)
            period = max(
                0,
                min(
                    int(np.searchsorted(boundaries, mid, side="right") - 1),
                    int(n_periods - 1),
                ),
            )
            treated = int(mid >= switch_global - 1e-12)
            dt = float(b - a)
            rows_period.append(period)
            rows_treated.append(treated)
            rows_dt.append(dt)
            rows_y.append(0)
            rows_cluster.append(str(sites[i]))
            seg_rows.append(len(rows_y) - 1)
            seg_starts.append(float(a))
            seg_ends.append(float(b))
            if keep_subject_maps:
                subj_map[period] = float(subj_map.get(period, 0.0) + dt)
        if keep_subject_maps:
            subject_time_by_period.append(subj_map)
        if int(events[i]) and float(avals[i]) <= float(tau) + 1e-12:
            g_event = float(base + float(avals[i]))
            assigned = False
            for seg_a, seg_b, idx in zip(seg_starts, seg_ends, seg_rows, strict=False):
                if (g_event > float(seg_a) + 1e-12) and (
                    g_event <= float(seg_b) + 1e-12
                ):
                    rows_y[idx] = 1
                    assigned = True
                    break
            if not assigned and seg_rows:
                rows_y[seg_rows[-1]] = 1
    table = _SteppedPersonPeriodTable(
        period=np.asarray(rows_period, dtype=np.int64),
        treated=np.asarray(rows_treated, dtype=np.float64),
        dt=np.asarray(rows_dt, dtype=np.float64),
        y=np.asarray(rows_y, dtype=np.float64),
        cluster=np.asarray(rows_cluster, dtype=object),
        n_periods=n_periods,
        subject_time_by_period=tuple(subject_time_by_period),
    )
    if table.dt.size == 0:
        raise ValueError(
            "Stepped-wedge estimator cannot fit an empty person-period table."
        )
    if (table.dt <= 0.0).any() or (not np.isfinite(table.dt).all()):
        raise ValueError("Stepped-wedge estimator requires positive finite durations.")
    return table


def _fit_period_poisson(
    *,
    table: _SteppedPersonPeriodTable,
) -> tuple[float, NDArray[np.float64]]:
    if float(table.treated.min()) == float(table.treated.max()):
        raise ValueError(
            "Stepped-wedge estimator requires both treated and untreated time."
        )
    if (
        table.y.shape != table.dt.shape
        or table.treated.shape != table.dt.shape
        or table.period.shape != table.dt.shape
    ):
        raise ValueError("Stepped-wedge profile-likelihood inputs must be row aligned.")
    if (table.y < 0.0).any() or not np.isfinite(table.y).all():
        raise ValueError("Stepped-wedge event counts must be finite and non-negative.")
    exposure_control = np.bincount(
        table.period,
        weights=table.dt * (1.0 - table.treated),
        minlength=int(table.n_periods),
    ).astype(np.float64)
    exposure_treated = np.bincount(
        table.period, weights=table.dt * table.treated, minlength=int(table.n_periods)
    ).astype(np.float64)
    events_control = np.bincount(
        table.period,
        weights=table.y * (1.0 - table.treated),
        minlength=int(table.n_periods),
    ).astype(np.float64)
    events_treated = np.bincount(
        table.period, weights=table.y * table.treated, minlength=int(table.n_periods)
    ).astype(np.float64)
    events_total = events_control + events_treated

    def score_and_information(coefficient: float) -> tuple[float, float]:
        relative_rate = float(np.exp(coefficient))
        denominator = exposure_control + relative_rate * exposure_treated
        if bool(((events_total > 0.0) & (denominator <= 0.0)).any()):
            raise ValueError(
                "Stepped-wedge events require positive period-specific person-time."
            )
        treated_fraction = np.divide(
            relative_rate * exposure_treated,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0.0,
        )
        return (
            float(np.sum(events_treated - events_total * treated_fraction)),
            float(np.sum(events_total * treated_fraction * (1.0 - treated_fraction))),
        )

    lower_score, _ = score_and_information(-30.0)
    upper_score, _ = score_and_information(30.0)
    if lower_score <= 0.0 or upper_score >= 0.0:
        raise ValueError(
            "Stepped-wedge treatment effect is separated and has no finite maximum-likelihood estimate."
        )
    beta = 0.0
    converged = False
    for _ in range(100):
        score, information = score_and_information(beta)
        if information <= 0.0 or not np.isfinite(information):
            raise ValueError(
                "Stepped-wedge treatment effect has zero profile information."
            )
        updated = float(np.clip(beta + score / information, -30.0, 30.0))
        if abs(updated - beta) <= 1e-10:
            beta = updated
            converged = True
            break
        beta = updated
    if not converged:
        raise ValueError(
            "Stepped-wedge profile likelihood did not converge within 100 iterations."
        )
    relative_rate = float(np.exp(beta))
    denominator = exposure_control + relative_rate * exposure_treated
    baseline_hazard = np.divide(
        events_total,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    return float(beta), baseline_hazard.astype(np.float64, copy=False)


def _resolve_stepped_wedge_switch_days_v1(
    *,
    adsl: pd.DataFrame,
    tau_dy: float,
    strict_missing_switch: bool,
) -> pd.DataFrame:
    if not np.isfinite(float(tau_dy)) or float(tau_dy) <= 0.0:
        raise ValueError(
            "tau_dy must be finite and > 0 for stepped-wedge exposure resolution."
        )
    missing = sorted(
        {"USUBJID", "INTERVENTION_START_DY"} - {str(column) for column in adsl.columns}
    )
    if missing:
        raise ValueError(
            f"Stepped-wedge switch resolution requires ADSL columns: {missing!r}."
        )
    state_col = _switch_state_column(adsl)
    rows: list[dict[str, object]] = []
    for _, source in adsl.iterrows():
        raw_state = None if state_col is None else source[state_col]
        switch_day, state, intervention_start = _resolve_switch_day(
            raw_switch_day=source["INTERVENTION_START_DY"],
            raw_state=raw_state,
            tau_dy=float(tau_dy),
            strict_missing_switch=bool(strict_missing_switch),
        )
        rows.append(
            {
                "USUBJID": str(source["USUBJID"]),
                "INTERVENTION_START_DY_RESOLVED": float(switch_day),
                "EXPOSURE_SWITCH_STATE": state,
                "INTERVENTION_START_DY": intervention_start,
            }
        )
    resolved = pd.DataFrame(rows)
    if resolved["USUBJID"].duplicated().any():
        duplicated = sorted(
            set(resolved.loc[resolved["USUBJID"].duplicated(), "USUBJID"].astype(str))
        )
        raise ValueError(
            f"Stepped-wedge switch resolution requires unique USUBJID rows: {duplicated[:5]!r}."
        )
    return cast(pd.DataFrame, resolved)


def _switch_state_column(adsl: pd.DataFrame) -> str | None:
    columns = {str(column) for column in adsl.columns}
    for candidate in ("EXPOSURE_SWITCH_STATE", "TRT_SWITCH_STATE", "SWITCH_STATE"):
        if candidate in columns:
            return candidate
    return None


def _resolve_switch_day(
    *,
    raw_switch_day: Any,
    raw_state: Any,
    tau_dy: float,
    strict_missing_switch: bool,
) -> tuple[float, str, float | None]:
    switch_day = pd.to_numeric(pd.Series([raw_switch_day]), errors="coerce").iloc[0]
    state = _coerce_switch_state(raw_state)
    if pd.notna(switch_day):
        switch = float(switch_day)
        if not np.isfinite(switch) or switch < 0.0:
            raise ValueError(
                "Stepped-wedge treatment switch days must be finite and non-negative."
            )
        if state in {"never_exposed", "not_yet_exposed_by_tau", "unknown_query_open"}:
            raise ValueError(
                "Observed stepped-wedge switch day conflicts with exposure switch state."
            )
        return switch, "observed_switch", switch
    if state is None:
        if strict_missing_switch:
            raise ValueError(
                "Missing stepped-wedge treatment switch day requires explicit exposure switch state."
            )
        state = "not_yet_exposed_by_tau"
    if state == "observed_switch":
        raise ValueError(
            "observed_switch state requires a finite stepped-wedge treatment switch day."
        )
    if state == "unknown_query_open":
        raise ValueError(
            "Open stepped-wedge exposure queries must be resolved before target replay."
        )
    if state in {"never_exposed", "not_yet_exposed_by_tau"}:
        return inf, state, None
    raise ValueError(f"Unsupported stepped-wedge exposure switch state: {state!r}.")


def _coerce_switch_state(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return token or None
