"""Survival-analysis primitives for TrialEval public reference replay."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def _coxph_binary_breslow_newton(
    *,
    t: NDArray[np.float64],
    e: NDArray[np.int64],
    a: NDArray[np.int64],
    weights: NDArray[np.float64] | None = None,
) -> tuple[float, float]:
    tt = np.asarray(t, dtype=np.float64).reshape(-1)
    ee = np.asarray(e, dtype=np.int64).reshape(-1)
    aa = np.asarray(a, dtype=np.int64).reshape(-1)
    if not (tt.size == ee.size == aa.size):
        raise ValueError("Cox inputs t/e/a must have the same length.")
    ww = (
        np.ones(int(tt.size), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    if ww.size != tt.size:
        raise ValueError("Cox weights must align to t/e/a inputs.")
    if not np.isfinite(ww).all() or np.any(ww < 0.0):
        raise ValueError("Cox weights must be finite and non-negative.")
    if int(np.sum(ee)) <= 0:
        raise ValueError("Cox fit requires at least one event.")
    if int(np.sum(aa)) == 0 or int(np.sum(1 - aa)) == 0:
        raise ValueError("Cox fit requires both treatment arms.")
    if not np.isfinite(tt).all():
        raise ValueError("Cox event times must be finite.")
    order = np.argsort(tt, kind="mergesort")
    tt_sorted = tt[order]
    ee_sorted = ee[order]
    aa_sorted = aa[order]
    ww_sorted = ww[order]
    event_times = np.unique(tt_sorted[ee_sorted.astype(bool)])
    beta = 0.0
    info = 0.0
    for _ in range(50):
        u = 0.0
        info = 0.0
        exp_beta = math.exp(beta)
        for event_time in event_times:
            event_time_f = float(event_time)
            risk = tt_sorted >= event_time_f - 1e-12
            a_risk = aa_sorted[risk].astype(np.float64)
            w_risk = ww_sorted[risk].astype(np.float64)
            exp_xb = w_risk * np.where(a_risk > 0.0, exp_beta, 1.0)
            s0 = float(np.sum(exp_xb))
            if s0 <= 0.0:
                continue
            s1 = float(np.sum(a_risk * exp_xb))
            p = s1 / s0
            events = (tt_sorted == event_time_f) & ee_sorted.astype(bool)
            a_events = aa_sorted[events].astype(np.float64)
            event_weights = ww_sorted[events].astype(np.float64)
            event_weight_sum = float(np.sum(event_weights))
            u += float(np.sum(event_weights * (a_events - p)))
            info += float(event_weight_sum) * float(p - p * p)
        if info <= 1e-12:
            raise ValueError("Cox fit failed with non-positive information.")
        beta_new = beta + (u / info)
        if abs(beta_new - beta) <= 1e-10:
            beta = beta_new
            break
        beta = beta_new
    se = math.sqrt(1.0 / max(info, 1e-12))
    return float(beta), float(se)


def _coxph_binary_breslow_risk_difference_tau(
    *,
    t: NDArray[np.float64],
    e: NDArray[np.int64],
    a: NDArray[np.int64],
    tau: float,
) -> float:
    """Return the treatment-only Cox-implied risk difference at ``tau``."""

    if not math.isfinite(float(tau)) or float(tau) <= 0.0:
        raise ValueError("Cox fixed-horizon risk requires finite tau > 0.")
    tt = np.asarray(t, dtype=np.float64).reshape(-1)
    ee = np.asarray(e, dtype=np.int64).reshape(-1)
    aa = np.asarray(a, dtype=np.int64).reshape(-1)
    beta, _ = _coxph_binary_breslow_newton(t=tt, e=ee, a=aa)
    event_times = np.unique(tt[(ee == 1) & (tt <= float(tau) + 1e-12)])
    baseline_hazard = 0.0
    exp_beta = math.exp(float(np.clip(beta, -30.0, 30.0)))
    for event_time in event_times:
        at_risk = tt >= float(event_time) - 1e-12
        denominator = float(np.sum(np.where(aa[at_risk] == 1, exp_beta, 1.0)))
        if denominator <= 0.0:
            raise ValueError("Cox fixed-horizon risk encountered an empty risk set.")
        event_count = int(np.sum((tt == float(event_time)) & (ee == 1)))
        baseline_hazard += float(event_count) / denominator
    control_risk = 1.0 - math.exp(-baseline_hazard)
    treated_risk = 1.0 - math.exp(-baseline_hazard * exp_beta)
    return float(treated_risk - control_risk)


def _km_survival_at(
    *,
    t: NDArray[np.float64],
    e: NDArray[np.int64],
    query_t: NDArray[np.float64],
) -> NDArray[np.float64]:
    return _km_step_survival_at(_km_step_survival(t=t, e=e), query_t)


def _km_step_survival(
    *,
    t: NDArray[np.float64],
    e: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    tt = np.asarray(t, dtype=np.float64).reshape(-1)
    ee = np.asarray(e, dtype=np.int64).reshape(-1)
    if tt.size != ee.size:
        raise ValueError(
            "KM step survival time/event inputs must have the same length."
        )
    if tt.size == 0:
        raise ValueError("KM step survival requires at least one row.")
    if not np.isfinite(tt).all():
        raise ValueError("KM step survival times must be finite.")
    order = np.argsort(tt, kind="mergesort")
    tt_sorted = tt[order]
    ee_sorted = ee[order]
    risk_suffix = np.cumsum(
        np.ones_like(tt_sorted, dtype=np.float64)[::-1], dtype=np.float64
    )[::-1]
    step_times: list[float] = []
    survival_values: list[float] = []
    survival = 1.0
    unique_times, first_index = np.unique(tt_sorted, return_index=True)
    for i, time_value in enumerate(unique_times.tolist()):
        start = int(first_index[i])
        end = (
            int(first_index[i + 1])
            if i + 1 < int(first_index.size)
            else int(tt_sorted.size)
        )
        event_count = float(np.sum(ee_sorted[start:end] == 1, dtype=np.float64))
        if event_count <= 0.0:
            continue
        n_at_risk = float(risk_suffix[start])
        if n_at_risk <= 0.0:
            raise ValueError("KM step survival encountered non-positive risk set.")
        survival *= 1.0 - min(1.0, event_count / n_at_risk)
        step_times.append(float(time_value))
        survival_values.append(float(survival))
        if survival <= 1e-15:
            break
    return np.asarray(step_times, dtype=np.float64), np.asarray(
        survival_values, dtype=np.float64
    )


def _km_step_survival_at(
    fit: tuple[NDArray[np.float64], NDArray[np.float64]],
    query_t: NDArray[np.float64],
) -> NDArray[np.float64]:
    step_times, survival_values = fit
    queries = np.asarray(query_t, dtype=np.float64).reshape(-1)
    if step_times.size == 0:
        return np.ones_like(queries, dtype=np.float64)
    indices = np.searchsorted(step_times, queries, side="right") - 1
    out = np.ones_like(queries, dtype=np.float64)
    valid = indices >= 0
    out[valid] = survival_values[indices[valid]]
    return out


def _km_risk_rmst_and_se(
    *,
    t: NDArray[np.float64],
    e: NDArray[np.int64],
    tau: float,
) -> tuple[float, float, float, float]:
    if not math.isfinite(float(tau)) or float(tau) <= 0.0:
        raise ValueError("tau must be finite and positive.")
    tt = np.asarray(t, dtype=np.float64).reshape(-1)
    ee = np.asarray(e, dtype=np.int64).reshape(-1)
    if tt.size != ee.size:
        raise ValueError("KM time and event arrays must have the same length.")
    if tt.size == 0:
        raise ValueError("KM recomputation requires at least one row.")
    if not np.isfinite(tt).all():
        raise ValueError("KM times must be finite.")
    durations_tau = np.minimum(tt, float(tau)).astype(np.float64, copy=False)
    events_tau = np.where(tt <= float(tau) + 1e-12, ee, 0).astype(np.int64, copy=False)
    order = np.argsort(durations_tau, kind="mergesort")
    tt_sorted = durations_tau[order]
    ee_sorted = events_tau[order]
    risk_suffix = np.cumsum(
        np.ones_like(tt_sorted, dtype=np.float64)[::-1], dtype=np.float64
    )[::-1]
    survival = 1.0
    previous = 0.0
    rmst = 0.0
    event_times: list[float] = []
    survival_after: list[float] = []
    green_terms: list[float] = []
    unique_times, first_index = np.unique(tt_sorted, return_index=True)
    for i, time_value in enumerate(unique_times.tolist()):
        event_time_f = float(time_value)
        stop = min(event_time_f, float(tau))
        if stop > previous:
            rmst += survival * (stop - previous)
        previous = event_time_f
        start = int(first_index[i])
        end = (
            int(first_index[i + 1])
            if i + 1 < int(first_index.size)
            else int(tt_sorted.size)
        )
        if not int(np.any(ee_sorted[start:end] == 1)):
            continue
        n_at_risk = float(risk_suffix[start])
        if n_at_risk <= 0.0:
            raise ValueError("KM variance encountered non-positive risk set.")
        event_count = float(np.sum(ee_sorted[start:end] == 1, dtype=np.float64))
        denom = float(n_at_risk - event_count)
        if denom < -1e-12:
            raise ValueError(
                "KM variance encountered event count greater than risk set."
            )
        green_term = 0.0 if denom <= 1e-12 else float(event_count / (n_at_risk * denom))
        event_fraction = min(1.0, float(event_count) / float(n_at_risk))
        survival *= 1.0 - event_fraction
        event_times.append(float(event_time_f))
        survival_after.append(float(survival))
        green_terms.append(float(green_term))
        if survival <= 1e-15:
            break
    if previous < float(tau):
        rmst += survival * (float(tau) - previous)
    risk = float(1.0 - survival)
    greenwood = float(
        np.sum(np.asarray(green_terms, dtype=np.float64), dtype=np.float64)
    )
    se_risk = float(math.sqrt(max(0.0, survival * survival * greenwood)))
    if not event_times:
        se_rmst = 0.0
    else:
        boundaries = np.asarray([0.0, *event_times, float(tau)], dtype=np.float64)
        interval_survival = np.asarray([1.0, *survival_after], dtype=np.float64)
        lengths = np.diff(boundaries)
        if lengths.shape != interval_survival.shape:
            raise ValueError("KM variance encountered invalid interval shapes.")
        areas = lengths * interval_survival
        suffix_area = np.cumsum(areas[::-1], dtype=np.float64)[::-1]
        a = suffix_area[1 : 1 + len(event_times)]
        terms = np.asarray(green_terms, dtype=np.float64)
        if a.shape != terms.shape:
            raise ValueError("KM variance encountered invalid Greenwood arrays.")
        se_rmst = float(
            math.sqrt(max(0.0, float(np.sum((a * a) * terms, dtype=np.float64))))
        )
    if not all(math.isfinite(value) for value in (risk, rmst, se_risk, se_rmst)):
        raise ValueError("KM variance produced non-finite outputs.")
    return float(risk), float(rmst), float(se_risk), float(se_rmst)


def _weighted_km_risk_and_rmst(
    *,
    t: NDArray[np.float64],
    e: NDArray[np.int64],
    weights: NDArray[np.float64],
    tau: float,
) -> tuple[float, float]:
    if not math.isfinite(float(tau)) or float(tau) <= 0.0:
        raise ValueError("Weighted KM requires finite positive tau.")
    durations: NDArray[np.float64] = np.asarray(t, dtype=np.float64).reshape(-1)
    events: NDArray[np.int64] = np.asarray(e, dtype=np.int64).reshape(-1)
    w: NDArray[np.float64] = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not (durations.size == events.size == w.size):
        raise ValueError("Weighted KM t/e/weights must have the same length.")
    if durations.size == 0:
        raise ValueError("Weighted KM requires at least one row.")
    if not np.isfinite(durations).all() or np.any(durations < -1e-12):
        raise ValueError("Weighted KM requires finite non-negative times.")
    if not set(int(value) for value in np.unique(events).tolist()) <= {0, 1}:
        raise ValueError("Weighted KM events must be binary.")
    if not np.isfinite(w).all() or np.any(w < 0.0):
        raise ValueError("Weighted KM requires finite non-negative weights.")
    keep = w > 0.0
    durations = np.asarray(durations[keep], dtype=np.float64).reshape(-1)
    events = np.asarray(events[keep], dtype=np.int64).reshape(-1)
    w = np.asarray(w[keep], dtype=np.float64).reshape(-1)
    if durations.size == 0:
        raise ValueError("Weighted KM requires at least one row with positive weight.")

    durations_tau: NDArray[np.float64] = np.minimum(durations, float(tau)).astype(
        np.float64, copy=False
    )
    events_tau: NDArray[np.int64] = np.where(
        durations <= float(tau) + 1e-12, events, 0
    ).astype(
        np.int64,
        copy=False,
    )
    order = np.argsort(durations_tau, kind="mergesort")
    durations_tau = durations_tau[order]
    events_tau = events_tau[order]
    w = w[order]
    unique_times, first_index = np.unique(durations_tau, return_index=True)
    risk_suffix = np.cumsum(w[::-1], dtype=np.float64)[::-1]
    survival = 1.0
    rmst = 0.0
    previous = 0.0
    for i, time_value in enumerate(unique_times.tolist()):
        time_f = float(time_value)
        if time_f > float(tau) + 1e-12:
            break
        start = int(first_index[i])
        end = (
            int(first_index[i + 1])
            if i + 1 < int(first_index.size)
            else int(durations_tau.size)
        )
        rmst += float(max(0.0, time_f - previous) * survival)
        previous = time_f
        if not int(np.any(events_tau[start:end] == 1)):
            continue
        y = float(risk_suffix[start])
        if not (math.isfinite(y) and y > 0.0):
            raise ValueError("Weighted KM encountered non-positive risk set.")
        d = float(np.sum(w[start:end][events_tau[start:end] == 1], dtype=np.float64))
        if not (math.isfinite(d) and d >= 0.0 and d <= y + 1e-12):
            raise ValueError("Weighted KM encountered invalid event weight sums.")
        survival = float(survival * (1.0 - float(d / y)))
        if survival <= 1e-15:
            break
    if float(tau) > previous + 1e-12:
        rmst += float((float(tau) - previous) * survival)
    risk = float(1.0 - survival)
    if not (math.isfinite(risk) and math.isfinite(rmst)):
        raise ValueError("Weighted KM produced non-finite outputs.")
    return float(risk), float(rmst)
