"""Independent replay of TrialDevBench observational point reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import NormalDist
from typing import Literal, cast

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray
from scipy.optimize import least_squares

from trialagentbench_validation.contracts.v1_scope import (
    RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1,
)
from trialagentbench_validation.trialdev.contracts import (
    TrialDevCandidateReplayV1,
    TrialDevMethodReplayV1,
    TrialDevObservationalReplayReportV1,
)

FloatArray = NDArray[np.float64]
DO_NOT_NOMINATE = "withhold_nomination"
COMPETING_EVENT_TIME_TOLERANCE_DAYS_V1 = 1e-9
NonEstimabilityReason = Literal[
    "empirical_positivity_violation",
    "empty_standardization_cell",
    "residual_unmeasured_confounding",
]


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _records(payload: dict[str, object], key: str) -> tuple[dict[str, object], ...]:
    raw = payload.get(key)
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(row, dict) for row in raw)
    ):
        raise ValueError(f"{key!r} must be a non-empty array of objects.")
    return tuple(cast(dict[str, object], row) for row in raw)


def _strings(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label} must be a non-empty string array.")
    return tuple(value)


def _optional_strings(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label} must be a string array.")
    return tuple(value)


def _number(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not np.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite numeric evidence.")
    return float(value)


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return value


def _design_matrix(frame: pd.DataFrame, covariates: tuple[str, ...]) -> FloatArray:
    """Build reference-level one-hot covariates from the public encoding rule."""

    parts: list[FloatArray] = [np.ones((len(frame), 1), dtype=np.float64)]
    for column in covariates:
        if column not in frame:
            raise ValueError(
                f"observational extract lacks adjustment covariate {column!r}."
            )
        series = frame[column]
        if bool(series.isna().any()):
            raise ValueError(
                f"analysis frame contains missing adjustment covariate {column!r}."
            )
        if isinstance(
            series.dtype, pd.CategoricalDtype
        ) or pd.api.types.is_object_dtype(series.dtype):
            values = series.astype("string")
            categories = (
                tuple(str(value) for value in series.cat.categories)
                if isinstance(series.dtype, pd.CategoricalDtype)
                else tuple(sorted(set(values.astype(str))))
            )
            for category in categories[1:]:
                parts.append(
                    np.asarray(
                        values.eq(category).to_numpy(), dtype=np.float64
                    ).reshape(-1, 1)
                )
        else:
            numeric = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
            if not np.isfinite(numeric).all():
                raise ValueError(
                    f"adjustment covariate {column!r} contains non-finite values."
                )
            parts.append(numeric.reshape(-1, 1))
    return np.concatenate(parts, axis=1)


def _propensity_weights(
    frame: pd.DataFrame,
    *,
    covariates: tuple[str, ...],
    treatment_ids: tuple[str, ...],
    max_iterations: int,
    tolerance: float,
) -> dict[str, pd.Series]:
    treatment = frame["TREATMENT"].astype(str)
    class_index = {
        candidate_id: index for index, candidate_id in enumerate(treatment_ids)
    }
    if set(treatment) - set(class_index):
        raise ValueError("observed treatments exceed the public candidate catalog.")
    design = _design_matrix(frame, covariates)
    if design.shape[1] > 1:
        values = design[:, 1:]
        means = values.mean(axis=0)
        scales = values.std(axis=0)
        scales[scales <= 0.0] = 1.0
        design = np.column_stack((design[:, 0], (values - means) / scales))
    endog = treatment.map(class_index).to_numpy(dtype=np.int64)
    result = sm.MNLogit(endog, design).fit(
        method="lbfgs",
        maxiter=int(max_iterations),
        pgtol=float(tolerance),
        disp=False,
        full_output=True,
        skip_hessian=True,
    )
    if not bool(result.mle_retvals.get("converged", False)):
        raise ValueError("independent multinomial propensity fit did not converge.")
    probabilities = np.asarray(result.predict(design), dtype=np.float64)
    if (
        probabilities.shape != (len(frame), len(treatment_ids))
        or not np.isfinite(probabilities).all()
    ):
        raise ValueError("independent multinomial propensity predictions are invalid.")
    output: dict[str, pd.Series] = {}
    for candidate_id, index in class_index.items():
        mask = treatment.eq(candidate_id).to_numpy(dtype=bool)
        observed = probabilities[mask, index]
        if observed.size == 0 or bool((observed <= 0.0).any()):
            raise ValueError(
                f"candidate {candidate_id!r} lacks positive observed propensity."
            )
        values = np.zeros(len(frame), dtype=np.float64)
        values[mask] = 1.0 / observed
        output[candidate_id] = pd.Series(values, index=frame.index, dtype=float)
    return output


def _entropy_weights(
    frame: pd.DataFrame,
    *,
    covariates: tuple[str, ...],
    treatment_ids: tuple[str, ...],
    max_iterations: int,
    tolerance: float,
    maximum_mean_balance_error: float,
) -> dict[str, pd.Series]:
    treatment = frame["TREATMENT"].astype(str)
    raw = _design_matrix(frame, covariates)[:, 1:]
    means = raw.mean(axis=0)
    scales = raw.std(axis=0)
    variable = scales > 0.0
    if not bool(variable.any()):
        raise ValueError("entropy balancing requires variable adjustment covariates.")
    design = (raw[:, variable] - means[variable]) / scales[variable]
    target = design.mean(axis=0)
    output: dict[str, pd.Series] = {}
    for candidate_id in treatment_ids:
        mask = treatment.eq(candidate_id).to_numpy(dtype=bool)
        arm = design[mask]
        if arm.shape[0] < arm.shape[1] + 2:
            raise ValueError(
                f"candidate {candidate_id!r} lacks entropy-balancing support."
            )

        def residual(
            coefficients: FloatArray, arm_design: FloatArray = arm
        ) -> FloatArray:
            linear = arm_design @ coefficients
            raw_weights = np.exp(linear - float(np.max(linear)))
            normalized = raw_weights / raw_weights.sum()
            return np.asarray(normalized @ arm_design - target, dtype=np.float64)

        fit = least_squares(
            residual,
            np.zeros(arm.shape[1], dtype=np.float64),
            xtol=float(tolerance),
            ftol=float(tolerance),
            gtol=float(tolerance),
            max_nfev=int(max_iterations),
        )
        if not bool(fit.success) or not np.isfinite(fit.x).all():
            raise ValueError(
                f"independent entropy-balancing fit failed for {candidate_id!r}."
            )
        linear = arm @ np.asarray(fit.x, dtype=np.float64)
        raw_weights = np.exp(linear - float(np.max(linear)))
        arm_weights = raw_weights * (float(mask.sum()) / float(raw_weights.sum()))
        error = float(
            np.max(np.abs(np.average(arm, axis=0, weights=arm_weights) - target))
        )
        if error > float(maximum_mean_balance_error):
            raise ValueError(
                f"independent entropy-balancing error {error:.6g} exceeds "
                f"{maximum_mean_balance_error:.6g} for {candidate_id!r}."
            )
        values = np.zeros(len(frame), dtype=np.float64)
        values[mask] = arm_weights
        output[candidate_id] = pd.Series(values, index=frame.index, dtype=float)
    return output


def _analysis_weights(
    frame: pd.DataFrame,
    *,
    method: dict[str, object],
    treatment_ids: tuple[str, ...],
) -> dict[str, pd.Series]:
    """Refit the public treatment-adjustment method on one analysis sample."""

    estimator_id = str(method["primary_estimator_id"])
    covariates = _strings(
        method.get("adjustment_covariates"), label="adjustment_covariates"
    )
    if estimator_id == "multinomial_propensity_weighted_stratified_aalen_johansen":
        return _propensity_weights(
            frame,
            covariates=covariates,
            treatment_ids=treatment_ids,
            max_iterations=_integer(
                method["propensity_max_iterations"],
                label="propensity maximum iterations",
            ),
            tolerance=_number(
                method["propensity_tolerance"], label="propensity tolerance"
            ),
        )
    if estimator_id == "entropy_balanced_standardized_aalen_johansen":
        return _entropy_weights(
            frame,
            covariates=covariates,
            treatment_ids=treatment_ids,
            max_iterations=_integer(
                method["calibration_max_iterations"],
                label="calibration maximum iterations",
            ),
            tolerance=_number(
                method["calibration_tolerance"], label="calibration tolerance"
            ),
            maximum_mean_balance_error=_number(
                method["maximum_mean_balance_error"],
                label="maximum mean balance error",
            ),
        )
    raise ValueError(f"unsupported observational estimator {estimator_id!r}.")


def _strata(frame: pd.DataFrame, method: dict[str, object]) -> pd.Series:
    exact = _optional_strings(
        method.get("exact_stratification_covariates", []),
        label="exact_stratification_covariates",
    )
    quantile = method.get("quantile_stratification_bins", {})
    if not isinstance(quantile, dict):
        raise ValueError("quantile_stratification_bins must be an object.")
    parts: list[pd.Series] = []
    for column, bins in quantile.items():
        if (
            not isinstance(column, str)
            or isinstance(bins, bool)
            or not isinstance(bins, int)
        ):
            raise ValueError(
                "quantile strata require string columns and integer bin counts."
            )
        parts.append(
            pd.qcut(
                pd.to_numeric(frame[column], errors="raise"), q=bins, duplicates="drop"
            ).astype(str)
        )
    parts.extend(
        frame[column].astype("string").fillna("missing").astype(str) for column in exact
    )
    if not parts:
        return pd.Series("all", index=frame.index, dtype="string")
    return cast(
        pd.Series,
        pd.concat(parts, axis=1).astype(str).agg("|".join, axis=1),
    )


def _weighted_cumulative_incidence(
    time: FloatArray,
    event_code: NDArray[np.int64],
    weights: FloatArray,
    *,
    horizon: float,
) -> float:
    event_times = np.unique(time[(event_code > 0) & (time <= horizon)])
    survival = 1.0
    cumulative_incidence = 0.0
    for event_time in event_times:
        at_risk = float(weights[time >= event_time].sum())
        all_events = float(weights[(time == event_time) & (event_code > 0)].sum())
        target_events = float(weights[(time == event_time) & (event_code == 1)].sum())
        if at_risk <= 0.0 or all_events > at_risk:
            raise ValueError(
                "independent Aalen-Johansen replay encountered an invalid risk set."
            )
        cumulative_incidence += survival * target_events / at_risk
        survival *= 1.0 - all_events / at_risk
    return float(cumulative_incidence)


def _standardized_cumulative_incidence(
    *,
    candidate_id: str,
    time: FloatArray,
    event: NDArray[np.int64],
    competing_time: FloatArray,
    competing_event: NDArray[np.int64],
    horizon: float,
    stratum_weights: tuple[tuple[str, float], ...],
    group_masks: dict[tuple[str, str], NDArray[np.bool_]],
    analysis_weights: FloatArray,
) -> float:
    result = 0.0
    for stratum_id, target_weight in stratum_weights:
        mask = group_masks[(candidate_id, stratum_id)]
        group_event = event[mask]
        group_competing_event = competing_event[mask]
        if not set(np.unique(group_event)).issubset({0, 1}) or not set(
            np.unique(group_competing_event)
        ).issubset({0, 1}):
            raise ValueError("efficacy and competing-event columns must be binary.")
        group_time = time[mask]
        group_competing_time = competing_time[mask]
        competing = (group_competing_event == 1) & (
            (group_event == 0)
            | (
                group_competing_time
                < group_time - COMPETING_EVENT_TIME_TOLERANCE_DAYS_V1
            )
        )
        observed_time = np.where(competing, group_competing_time, group_time)
        event_code = np.where(competing, 2, group_event).astype(np.int64)
        weights = analysis_weights[mask]
        result += float(target_weight) * _weighted_cumulative_incidence(
            observed_time,
            event_code,
            weights,
            horizon=horizon,
        )
    return float(result)


def _component_values(
    frame: pd.DataFrame,
    *,
    objectives: tuple[dict[str, object], ...],
    candidate_ids: tuple[str, ...],
    control_id: str,
    strata: pd.Series,
    analysis_weights: dict[str, pd.Series],
) -> dict[tuple[str, str], float]:
    stratum_weights = tuple(
        (str(stratum_id), float(target_weight))
        for stratum_id, target_weight in strata.value_counts(normalize=True)
        .sort_index()
        .items()
    )
    treatment_values = frame["TREATMENT"].astype(str).to_numpy()
    stratum_values = strata.astype(str).to_numpy()
    group_masks = {
        (candidate_id, stratum_id): (treatment_values == candidate_id)
        & (stratum_values == stratum_id)
        for candidate_id in candidate_ids
        for stratum_id, _target_weight in stratum_weights
    }
    missing_groups = tuple(
        group for group, mask in group_masks.items() if not bool(mask.any())
    )
    if missing_groups:
        raise ValueError(f"candidate-stratum evidence is absent: {missing_groups!r}.")
    weight_arrays = {
        candidate_id: analysis_weights[candidate_id].to_numpy(dtype=np.float64)
        for candidate_id in candidate_ids
    }
    endpoints = {
        str(endpoint["endpoint_id"]): endpoint
        for objective in objectives
        for endpoint in _records(objective, "efficacy_endpoints")
    }
    utility_events: dict[str, dict[str, object]] = {}
    for objective in objectives:
        definitions = objective.get("utility_event_definitions")
        if not isinstance(definitions, list) or not all(
            isinstance(row, dict) for row in definitions
        ):
            raise ValueError("utility_event_definitions must be an array of objects.")
        for raw_definition in definitions:
            definition = cast(dict[str, object], raw_definition)
            source = str(definition.get("component_source") or "")
            if not source:
                raise ValueError("utility event definition lacks component_source.")
            previous = utility_events.get(source)
            if previous is not None and previous != definition:
                raise ValueError(
                    f"public objectives disagree on utility event {source!r}."
                )
            utility_events[source] = definition
    endpoint_values = tuple(
        (
            pd.to_numeric(frame[str(endpoint["time_column"])], errors="raise").to_numpy(
                dtype=np.float64
            ),
            pd.to_numeric(
                frame[str(endpoint["event_column"])], errors="raise"
            ).to_numpy(dtype=np.int64),
            pd.to_numeric(
                frame[str(endpoint["competing_time_column"])], errors="raise"
            ).to_numpy(dtype=np.float64),
            pd.to_numeric(
                frame[str(endpoint["competing_event_column"])], errors="raise"
            ).to_numpy(dtype=np.int64),
            _number(endpoint["horizon_days"], label="endpoint horizon"),
        )
        for endpoint in endpoints.values()
    )
    utility_event_values = {
        source: (
            pd.to_numeric(
                frame[str(definition["time_column"])], errors="raise"
            ).to_numpy(dtype=np.float64),
            pd.to_numeric(
                frame[str(definition["event_column"])], errors="raise"
            ).to_numpy(dtype=np.int64),
            pd.to_numeric(
                frame[str(definition["competing_time_column"])], errors="raise"
            ).to_numpy(dtype=np.float64),
            pd.to_numeric(
                frame[str(definition["competing_event_column"])], errors="raise"
            ).to_numpy(dtype=np.int64),
            _number(definition["horizon_days"], label="utility event horizon"),
        )
        for source, definition in utility_events.items()
    }
    output: dict[tuple[str, str], float] = {}
    for candidate_id in candidate_ids:
        efficacy = [
            _standardized_cumulative_incidence(
                candidate_id=candidate_id,
                time=time,
                event=event,
                competing_time=competing_time,
                competing_event=competing_event,
                horizon=horizon,
                stratum_weights=stratum_weights,
                group_masks=group_masks,
                analysis_weights=weight_arrays[candidate_id],
            )
            for time, event, competing_time, competing_event, horizon in endpoint_values
        ]
        output[(candidate_id, "efficacy_risk")] = float(np.mean(efficacy))
        for source, (
            time,
            event,
            competing_time,
            competing_event,
            horizon,
        ) in utility_event_values.items():
            output[(candidate_id, source)] = _standardized_cumulative_incidence(
                candidate_id=candidate_id,
                time=time,
                event=event,
                competing_time=competing_time,
                competing_event=competing_event,
                horizon=horizon,
                stratum_weights=stratum_weights,
                group_masks=group_masks,
                analysis_weights=weight_arrays[candidate_id],
            )
    control_risk = output[(control_id, "efficacy_risk")]
    for candidate_id in candidate_ids:
        output[(candidate_id, "efficacy_gain")] = (
            control_risk - output[(candidate_id, "efficacy_risk")]
        )
        for source in utility_events:
            output[(candidate_id, source)] -= output[(control_id, source)]
    return output


def _utility(
    candidate_id: str,
    objective: dict[str, object],
    components: dict[tuple[str, str], float],
) -> float:
    candidate_costs = objective.get("candidate_costs", {})
    if not isinstance(candidate_costs, dict):
        raise ValueError("candidate_costs must be an object.")
    total = 0.0
    for component in _records(objective, "utility_components"):
        source = str(component["source"])
        if source == "candidate_cost":
            value = _number(
                candidate_costs.get(candidate_id), label=f"cost for {candidate_id}"
            )
        else:
            value = components[(candidate_id, source)]
        direction = str(component["direction"])
        if direction not in {"benefit", "penalty"}:
            raise ValueError(f"unsupported utility direction {direction!r}.")
        weight = _number(component["weight"], label="utility weight")
        total += weight * value * (1.0 if direction == "benefit" else -1.0)
    return float(total)


def _minimum_efficacy_gain(decision_charter: dict[str, object]) -> float:
    matches = [
        row
        for row in _records(decision_charter, "efficacy_rules")
        if str(row.get("phase_id")) == "observational_review"
    ]
    if len(matches) != 1:
        raise ValueError(
            "decision charter must define one observational efficacy rule."
        )
    return _number(
        matches[0].get("minimum_benefit"), label="observational minimum efficacy gain"
    )


def _unrecorded_assignment_prognostic_factor(method_catalog: dict[str, object]) -> bool:
    """Return whether public provenance disproves measured conditional exchangeability."""

    raw = method_catalog.get("assignment_prognostic_factors", [])
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("assignment_prognostic_factors must be an array of objects.")
    return any(
        bool(row.get("used_in_treatment_assignment"))
        and bool(row.get("prognostic_for_primary_endpoint"))
        and not bool(row.get("recorded_in_observational_extract"))
        for row in raw
    )


def _expected_non_estimability_reason(
    expected_method: dict[str, object],
) -> NonEstimabilityReason | None:
    """Read one coherent method-level non-estimability reason from evaluator records."""

    comparisons = _records(expected_method, "estimator_comparisons")
    primary = tuple(
        row
        for row in comparisons
        if str(row.get("estimator_id")) == str(expected_method.get("estimator_id"))
    )
    if not primary:
        raise ValueError(
            "public recoverability report lacks its primary estimator comparison."
        )
    states = {(str(row.get("status")), row.get("failure_reason")) for row in primary}
    if states == {("estimated", None)}:
        return None
    if len(states) != 1:
        raise ValueError("primary estimator comparisons disagree on estimability.")
    status, reason = next(iter(states))
    allowed = {
        "empirical_positivity_violation",
        "empty_standardization_cell",
        "residual_unmeasured_confounding",
    }
    if status != "not_estimable" or reason not in allowed:
        raise ValueError(
            "non-estimable primary comparison lacks one supported failure reason."
        )
    policies = _records(expected_method, "objective_policies")
    if any(str(row.get("policy")) != "insufficient_recoverability" for row in policies):
        raise ValueError(
            "non-estimable method must declare insufficient recoverability for every objective."
        )
    return cast(NonEstimabilityReason, reason)


def _replay_non_estimability_reason(
    *,
    method_catalog: dict[str, object],
    frame: pd.DataFrame,
    method: dict[str, object],
    treatment_ids: tuple[str, ...],
) -> NonEstimabilityReason | None:
    """Independently determine why a declared point method cannot be evaluated."""

    if _unrecorded_assignment_prognostic_factor(method_catalog):
        return "residual_unmeasured_confounding"
    strata = _strata(frame, method)
    support = pd.crosstab(strata, frame["TREATMENT"].astype(str)).reindex(
        columns=list(treatment_ids),
        fill_value=0,
    )
    if bool((support == 0).any(axis=None)):
        return "empty_standardization_cell"
    try:
        _analysis_weights(
            frame,
            method=method,
            treatment_ids=treatment_ids,
        )
    except (ValueError, np.linalg.LinAlgError):
        return "empirical_positivity_violation"
    return None


def _target_ids(row: dict[str, object]) -> tuple[str, ...]:
    """Read the current public action-set field."""

    return _strings(row.get("reference_target_ids"), label="reference target ids")


def _bootstrap_observational_draws(
    frame: pd.DataFrame,
    *,
    method: dict[str, object],
    objectives: tuple[dict[str, object], ...],
    candidate_ids: tuple[str, ...],
    control_id: str,
    replicates: int,
    seed: int,
) -> tuple[dict[tuple[str, str], FloatArray], dict[str, FloatArray]]:
    """Independently replay participant bootstrap with nuisance refitting."""

    if replicates < 2:
        raise ValueError(
            "observational uncertainty replay requires at least two replicates."
        )
    if frame.empty:
        raise ValueError(
            "observational uncertainty replay requires a non-empty analysis population."
        )
    utility_draws: dict[tuple[str, str], FloatArray] = {
        (str(objective["objective_id"]), candidate_id): np.empty(
            replicates, dtype=np.float64
        )
        for objective in objectives
        for candidate_id in candidate_ids
    }
    efficacy_draws: dict[str, FloatArray] = {
        candidate_id: np.empty(replicates, dtype=np.float64)
        for candidate_id in candidate_ids
    }
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        sampled = frame.iloc[rng.integers(0, len(frame), size=len(frame))].reset_index(
            drop=True
        )
        missing_candidates = sorted(
            set(candidate_ids) - set(sampled["TREATMENT"].astype(str))
        )
        if missing_candidates:
            raise ValueError(
                f"bootstrap replicate {replicate} has no support for candidates {missing_candidates!r}."
            )
        sampled_strata = _strata(sampled, method)
        support = pd.crosstab(sampled_strata, sampled["TREATMENT"].astype(str)).reindex(
            columns=list(candidate_ids),
            fill_value=0,
        )
        if bool((support == 0).any(axis=None)):
            raise ValueError(
                f"bootstrap replicate {replicate} lacks treatment-stratum support."
            )
        sampled_weights = _analysis_weights(
            sampled,
            method=method,
            treatment_ids=candidate_ids,
        )
        components = _component_values(
            sampled,
            objectives=objectives,
            candidate_ids=candidate_ids,
            control_id=control_id,
            strata=sampled_strata,
            analysis_weights=sampled_weights,
        )
        for candidate_id in candidate_ids:
            efficacy_draws[candidate_id][replicate] = components[
                (candidate_id, "efficacy_gain")
            ]
            for objective in objectives:
                objective_id = str(objective["objective_id"])
                utility_draws[(objective_id, candidate_id)][replicate] = _utility(
                    candidate_id,
                    objective,
                    components,
                )
    return utility_draws, efficacy_draws


def _checksum_match(root: Path, expected_report: dict[str, object]) -> bool:
    raw = expected_report.get("public_input_checksums")
    if not isinstance(raw, list) or not raw:
        raise ValueError("public recoverability report lacks input checksums.")
    for record in raw:
        if not isinstance(record, dict):
            raise ValueError("public input checksum entries must be objects.")
        relative = record.get("path")
        expected = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("public input checksum entries require path and sha256.")
        path = root / relative
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            return False
    return True


def replay_trialdev_observational_reference(
    scenario_root: Path,
    *,
    absolute_tolerance: float = RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1,
    selected_method_route_id: str | None = None,
) -> TrialDevObservationalReplayReportV1:
    """Replay one or all TrialDev observational method references from public files."""

    root = Path(scenario_root)
    if absolute_tolerance <= 0.0:
        raise ValueError("absolute_tolerance must be positive.")
    public = root / "public"
    expected_report = _read_object(
        root / "grader" / "public_recoverability_report.json"
    )
    objective_charter = _read_object(public / "objective_charter.json")
    method_catalog = _read_object(public / "observational_method_catalog.json")
    candidate_catalog = _read_object(public / "candidate_drug_catalog.json")
    decision_charter = _read_object(public / "decision_charter.json")
    scenario_id = str(expected_report.get("scenario_id") or "")
    if not scenario_id:
        raise ValueError("public recoverability report lacks scenario_id.")

    candidates = _records(candidate_catalog, "candidate_drugs")
    treatment_ids = tuple(str(row["candidate_drug_id"]) for row in candidates)
    controls = tuple(
        str(row["candidate_drug_id"])
        for row in candidates
        if row.get("role") == "control"
    )
    if len(controls) != 1:
        raise ValueError("candidate catalog must declare exactly one control.")
    control_id = controls[0]
    investigational_ids = tuple(value for value in treatment_ids if value != control_id)
    objectives = _records(objective_charter, "objectives")
    objective_checksum = objective_charter.get("checksum")
    confidence_level = _number(
        objective_charter.get("confidence_level"),
        label="objective-charter confidence level",
    )
    if not isinstance(objective_checksum, str) or len(objective_checksum) != 64:
        raise ValueError(
            "objective charter requires a SHA-256 checksum for uncertainty replay."
        )
    critical_value = float(NormalDist().inv_cdf(0.5 + confidence_level / 2.0))
    source = pd.read_parquet(public / "observational_extract.parquet").reset_index(
        drop=True
    )
    expected_methods = {
        str(row["method_route_id"]): row
        for row in _records(expected_report, "method_results")
    }
    replayed_methods: list[TrialDevMethodReplayV1] = []
    minimum_efficacy = _minimum_efficacy_gain(decision_charter)
    methods = _records(method_catalog, "methods")
    if selected_method_route_id is not None:
        methods = tuple(
            row
            for row in methods
            if row.get("method_route_id") == selected_method_route_id
        )
        if len(methods) != 1:
            raise ValueError(
                f"unknown or duplicate TrialDev method route: {selected_method_route_id!r}."
            )
    for method in methods:
        method_route_id = str(method["method_route_id"])
        estimator_id = str(method["primary_estimator_id"])
        expected = expected_methods.get(method_route_id)
        if expected is None:
            raise ValueError(
                f"evaluator report lacks method route {method_route_id!r}."
            )
        if method.get("uncertainty_estimator_id") != (
            "refitted_nuisance_participant_nonparametric_bootstrap"
        ):
            raise ValueError(
                f"method route {method_route_id!r} declares an unsupported uncertainty estimator."
            )
        method_confidence = _number(
            method.get("confidence_level"), label="method confidence level"
        )
        if method_confidence != confidence_level:
            raise ValueError("method and objective-charter confidence levels differ.")
        bootstrap_replicates = _integer(
            method.get("bootstrap_replicates"),
            label="bootstrap replicates",
        )
        bootstrap_seed = _integer(
            method.get("bootstrap_seed"),
            label="bootstrap seed",
        )
        if bootstrap_seed < 0:
            raise ValueError("bootstrap seed must be non-negative.")
        if method.get("bootstrap_rng_id") != "numpy_default_rng_pcg64":
            raise ValueError(
                f"method route {method_route_id!r} declares an unsupported bootstrap RNG."
            )
        if method.get("bootstrap_standard_error_ddof") != 1:
            raise ValueError(
                f"method route {method_route_id!r} declares an unsupported bootstrap standard error."
            )
        if (
            method.get("confidence_interval_id")
            != "normal_critical_value_times_bootstrap_standard_error"
        ):
            raise ValueError(
                f"method route {method_route_id!r} declares an unsupported confidence interval."
            )
        covariates = _strings(
            method.get("adjustment_covariates"), label="adjustment_covariates"
        )
        frame = source.loc[
            source.loc[:, list(covariates)].notna().all(axis=1)
        ].reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                f"method route {method_route_id!r} has an empty complete-case population."
            )
        expected_reason = _expected_non_estimability_reason(expected)
        replayed_reason = _replay_non_estimability_reason(
            method_catalog=method_catalog,
            frame=frame,
            method=method,
            treatment_ids=treatment_ids,
        )
        objective_by_id = {str(row["objective_id"]): row for row in objectives}
        if expected_reason is not None or replayed_reason is not None:
            expected_action_policies = {
                str(row["objective_id"]): row
                for row in _records(expected, "observational_action_policies")
            }
            expected_actions = {
                objective_id: _target_ids(expected_action_policies[objective_id])
                for objective_id in objective_by_id
            }
            replayed_nonestimable_actions = {
                objective_id: (
                    (DO_NOT_NOMINATE,) if replayed_reason is not None else ()
                )
                for objective_id in objective_by_id
            }
            empty_sets = {objective_id: () for objective_id in objective_by_id}
            non_estimability_match = expected_reason == replayed_reason
            action_match = expected_actions == replayed_nonestimable_actions
            passed = non_estimability_match and action_match
            replayed_methods.append(
                TrialDevMethodReplayV1(
                    method_route_id=method_route_id,
                    estimator_id=estimator_id,
                    uncertainty_estimator_id=str(method["uncertainty_estimator_id"]),
                    bootstrap_replicates=bootstrap_replicates,
                    confidence_level=method_confidence,
                    result_form="qualified_non_nomination",
                    expected_non_estimability_reason=expected_reason,
                    replayed_non_estimability_reason=replayed_reason,
                    non_estimability_match=non_estimability_match,
                    candidate_results=(),
                    expected_rankings=empty_sets,
                    replayed_rankings=empty_sets,
                    expected_actions=expected_actions,
                    replayed_actions=replayed_nonestimable_actions,
                    expected_acceptable_utility_sets=empty_sets,
                    replayed_acceptable_utility_sets=empty_sets,
                    expected_definitely_qualified_sets=empty_sets,
                    replayed_definitely_qualified_sets=empty_sets,
                    expected_possibly_qualified_sets=empty_sets,
                    replayed_possibly_qualified_sets=empty_sets,
                    expected_pairwise_contrast_half_widths={
                        objective_id: {} for objective_id in objective_by_id
                    },
                    replayed_pairwise_contrast_half_widths={
                        objective_id: {} for objective_id in objective_by_id
                    },
                    maximum_utility_absolute_error=0.0,
                    maximum_efficacy_gain_absolute_error=0.0,
                    maximum_standard_error_absolute_error=0.0,
                    maximum_interval_endpoint_absolute_error=0.0,
                    maximum_pairwise_contrast_absolute_error=0.0,
                    ranking_match=True,
                    action_match=action_match,
                    uncertainty_policy_match=True,
                    status="pass" if passed else "fail",
                )
            )
            continue
        weights = _analysis_weights(
            frame,
            method=method,
            treatment_ids=treatment_ids,
        )
        strata = _strata(frame, method)
        components = _component_values(
            frame,
            objectives=objectives,
            candidate_ids=treatment_ids,
            control_id=control_id,
            strata=strata,
            analysis_weights=weights,
        )
        utility_draws, efficacy_draws = _bootstrap_observational_draws(
            frame,
            method=method,
            objectives=objectives,
            candidate_ids=treatment_ids,
            control_id=control_id,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        utilities = {
            (objective_id, candidate_id): _utility(candidate_id, objective, components)
            for objective_id, objective in objective_by_id.items()
            for candidate_id in investigational_ids
        }
        expected_scores = _records(expected, "candidate_scores")
        candidate_rows: list[TrialDevCandidateReplayV1] = []
        for score in expected_scores:
            if not bool(score.get("point_estimable")):
                raise ValueError(
                    "independent point replay does not accept a non-estimable official method."
                )
            objective_id = str(score["objective_id"])
            candidate_id = str(score["candidate_drug_id"])
            expected_utility = _number(
                score.get("adjusted_utility"), label="expected adjusted utility"
            )
            expected_efficacy = _number(
                score.get("efficacy_gain"), label="expected efficacy gain"
            )
            expected_utility_se = _number(
                score.get("utility_se"),
                label="expected utility standard error",
            )
            expected_efficacy_se = _number(
                score.get("efficacy_gain_se"),
                label="expected efficacy-gain standard error",
            )
            replayed_utility = utilities[(objective_id, candidate_id)]
            replayed_efficacy = components[(candidate_id, "efficacy_gain")]
            replayed_utility_se = float(
                np.std(utility_draws[(objective_id, candidate_id)], ddof=1)
            )
            replayed_efficacy_se = float(np.std(efficacy_draws[candidate_id], ddof=1))
            utility_error = abs(replayed_utility - expected_utility)
            efficacy_error = abs(replayed_efficacy - expected_efficacy)
            utility_se_error = abs(replayed_utility_se - expected_utility_se)
            efficacy_se_error = abs(replayed_efficacy_se - expected_efficacy_se)
            expected_utility_interval = (
                _number(score.get("ci_low"), label="expected utility interval lower"),
                _number(score.get("ci_high"), label="expected utility interval upper"),
            )
            replayed_utility_interval = (
                replayed_utility - critical_value * replayed_utility_se,
                replayed_utility + critical_value * replayed_utility_se,
            )
            expected_efficacy_interval = (
                _number(
                    score.get("efficacy_gain_ci_low"),
                    label="expected efficacy interval lower",
                ),
                _number(
                    score.get("efficacy_gain_ci_high"),
                    label="expected efficacy interval upper",
                ),
            )
            replayed_efficacy_interval = (
                replayed_efficacy - critical_value * replayed_efficacy_se,
                replayed_efficacy + critical_value * replayed_efficacy_se,
            )
            interval_error = max(
                abs(replayed_utility_interval[index] - expected_utility_interval[index])
                for index in (0, 1)
            )
            interval_error = max(
                interval_error,
                *(
                    abs(
                        replayed_efficacy_interval[index]
                        - expected_efficacy_interval[index]
                    )
                    for index in (0, 1)
                ),
            )
            candidate_rows.append(
                TrialDevCandidateReplayV1(
                    objective_id=objective_id,
                    candidate_drug_id=candidate_id,
                    expected_utility=expected_utility,
                    replayed_utility=replayed_utility,
                    utility_absolute_error=utility_error,
                    expected_efficacy_gain=expected_efficacy,
                    replayed_efficacy_gain=replayed_efficacy,
                    efficacy_gain_absolute_error=efficacy_error,
                    expected_utility_standard_error=expected_utility_se,
                    replayed_utility_standard_error=replayed_utility_se,
                    utility_standard_error_absolute_error=utility_se_error,
                    expected_efficacy_gain_standard_error=expected_efficacy_se,
                    replayed_efficacy_gain_standard_error=replayed_efficacy_se,
                    efficacy_gain_standard_error_absolute_error=efficacy_se_error,
                    expected_utility_interval=expected_utility_interval,
                    replayed_utility_interval=replayed_utility_interval,
                    expected_efficacy_gain_interval=expected_efficacy_interval,
                    replayed_efficacy_gain_interval=replayed_efficacy_interval,
                    maximum_interval_endpoint_absolute_error=interval_error,
                    within_tolerance=max(
                        utility_error,
                        efficacy_error,
                        utility_se_error,
                        efficacy_se_error,
                        interval_error,
                    )
                    <= absolute_tolerance,
                )
            )
        expected_rankings = {
            objective_id: tuple(
                str(row["candidate_drug_id"])
                for row in sorted(
                    (
                        score
                        for score in expected_scores
                        if str(score["objective_id"]) == objective_id
                    ),
                    key=lambda score: _integer(score["rank"], label="expected rank"),
                )
            )
            for objective_id in objective_by_id
        }
        replayed_rankings = {
            objective_id: tuple(
                candidate_id
                for candidate_id in sorted(
                    investigational_ids,
                    key=lambda candidate_id: (
                        -utilities[(objective_id, candidate_id)],
                        candidate_id,
                    ),
                )
            )
            for objective_id in objective_by_id
        }
        expected_actions = {
            str(row["objective_id"]): _target_ids(row)
            for row in _records(expected, "observational_action_policies")
        }
        expected_action_policies = {
            str(row["objective_id"]): row
            for row in _records(expected, "observational_action_policies")
        }
        expected_acceptable_utility_sets = {
            str(row["objective_id"]): tuple(
                sorted(
                    _strings(
                        row["acceptable_candidate_set"],
                        label="acceptable candidate set",
                    )
                )
            )
            for row in _records(expected, "objective_policies")
        }
        expected_definitely_qualified_sets = {
            objective_id: tuple(
                sorted(
                    _optional_strings(
                        row.get("definitely_qualified_candidate_ids", []),
                        label="definitely qualified candidate ids",
                    )
                )
            )
            for objective_id, row in expected_action_policies.items()
        }
        expected_possibly_qualified_sets = {
            objective_id: tuple(
                sorted(
                    _optional_strings(
                        row.get("possibly_qualified_candidate_ids", []),
                        label="possibly qualified candidate ids",
                    )
                )
            )
            for objective_id, row in expected_action_policies.items()
        }
        expected_pairwise_contrast_half_widths = {
            objective_id: {
                str(key): _number(value, label=f"pairwise contrast half-width {key}")
                for key, value in cast(
                    dict[str, object],
                    row.get("pairwise_utility_contrast_half_widths"),
                ).items()
            }
            for objective_id, row in expected_action_policies.items()
        }
        ordered_investigational_ids = tuple(sorted(investigational_ids))
        expected_pair_keys = {
            f"{first}|{second}"
            for index, first in enumerate(ordered_investigational_ids)
            for second in ordered_investigational_ids[index + 1 :]
        }
        for objective_id, values in expected_pairwise_contrast_half_widths.items():
            expected_keys = (
                expected_pair_keys
                if expected_possibly_qualified_sets[objective_id]
                else set()
            )
            if set(values) != expected_keys:
                raise ValueError(
                    "observational action policy pairwise uncertainty disagrees with candidate qualification."
                )
        replayed_actions: dict[str, tuple[str, ...]] = {}
        replayed_acceptable_utility_sets: dict[str, tuple[str, ...]] = {}
        replayed_definitely_qualified_sets: dict[str, tuple[str, ...]] = {}
        replayed_possibly_qualified_sets: dict[str, tuple[str, ...]] = {}
        replayed_pairwise_contrast_half_widths: dict[str, dict[str, float]] = {}
        for objective_id in objective_by_id:
            objective = objective_by_id[objective_id]
            top_candidate = replayed_rankings[objective_id][0]
            top_draws = utility_draws[(objective_id, top_candidate)]
            indifference_margin = _number(
                objective["indifference_margin"],
                label="objective indifference margin",
            )
            replayed_acceptable_utility_sets[objective_id] = tuple(
                sorted(
                    candidate_id
                    for candidate_id in investigational_ids
                    if utilities[(objective_id, top_candidate)]
                    - utilities[(objective_id, candidate_id)]
                    <= max(
                        indifference_margin,
                        critical_value
                        * float(
                            np.std(
                                top_draws - utility_draws[(objective_id, candidate_id)],
                                ddof=1,
                            )
                        ),
                    )
                )
            )
            definitely_qualified = tuple(
                sorted(
                    candidate_id
                    for candidate_id in investigational_ids
                    if components[(candidate_id, "efficacy_gain")]
                    - critical_value
                    * float(np.std(efficacy_draws[candidate_id], ddof=1))
                    >= minimum_efficacy
                )
            )
            possibly_qualified = tuple(
                sorted(
                    candidate_id
                    for candidate_id in investigational_ids
                    if components[(candidate_id, "efficacy_gain")]
                    + critical_value
                    * float(np.std(efficacy_draws[candidate_id], ddof=1))
                    >= minimum_efficacy
                )
            )
            ranked_qualified = tuple(
                sorted(
                    possibly_qualified,
                    key=lambda candidate_id: (
                        -utilities[(objective_id, candidate_id)],
                        candidate_id,
                    ),
                )
            )
            acceptable_targets: set[str] = set()
            if ranked_qualified:
                action_best = ranked_qualified[0]
                action_best_draws = utility_draws[(objective_id, action_best)]
                acceptable_targets = {
                    candidate_id
                    for candidate_id in ranked_qualified
                    if utilities[(objective_id, action_best)]
                    - utilities[(objective_id, candidate_id)]
                    <= max(
                        indifference_margin,
                        critical_value
                        * float(
                            np.std(
                                action_best_draws
                                - utility_draws[(objective_id, candidate_id)],
                                ddof=1,
                            )
                        ),
                    )
                }
            if not definitely_qualified:
                selected = DO_NOT_NOMINATE
            elif acceptable_targets:
                selected = min(
                    acceptable_targets,
                    key=lambda candidate_id: (
                        -utilities[(objective_id, candidate_id)],
                        candidate_id,
                    ),
                )
            else:
                raise ValueError(
                    "definitely qualified candidates produced no supportable action."
                )
            replayed_actions[objective_id] = (selected,)
            replayed_definitely_qualified_sets[objective_id] = definitely_qualified
            replayed_possibly_qualified_sets[objective_id] = possibly_qualified
            replayed_pairwise_contrast_half_widths[objective_id] = (
                {
                    f"{first}|{second}": critical_value
                    * float(
                        np.std(
                            utility_draws[(objective_id, first)]
                            - utility_draws[(objective_id, second)],
                            ddof=1,
                        )
                    )
                    for index, first in enumerate(ordered_investigational_ids)
                    for second in ordered_investigational_ids[index + 1 :]
                }
                if possibly_qualified
                else {}
            )
        max_utility_error = max(row.utility_absolute_error for row in candidate_rows)
        max_efficacy_error = max(
            row.efficacy_gain_absolute_error for row in candidate_rows
        )
        max_standard_error = max(
            max(
                row.utility_standard_error_absolute_error,
                row.efficacy_gain_standard_error_absolute_error,
            )
            for row in candidate_rows
        )
        max_interval_error = max(
            row.maximum_interval_endpoint_absolute_error for row in candidate_rows
        )
        pairwise_errors = tuple(
            abs(
                expected_pairwise_contrast_half_widths[objective_id][pair_id]
                - replayed_pairwise_contrast_half_widths[objective_id][pair_id]
            )
            for objective_id in objective_by_id
            for pair_id in expected_pairwise_contrast_half_widths[objective_id]
        )
        max_pairwise_error = max(pairwise_errors, default=0.0)
        ranking_match = expected_rankings == replayed_rankings
        action_match = expected_actions == replayed_actions
        uncertainty_policy_match = (
            expected_acceptable_utility_sets == replayed_acceptable_utility_sets
            and expected_definitely_qualified_sets == replayed_definitely_qualified_sets
            and expected_possibly_qualified_sets == replayed_possibly_qualified_sets
            and max_pairwise_error <= absolute_tolerance
        )
        passed = (
            max(
                max_utility_error,
                max_efficacy_error,
                max_standard_error,
                max_interval_error,
                max_pairwise_error,
            )
            <= absolute_tolerance
            and ranking_match
            and action_match
            and uncertainty_policy_match
        )
        replayed_methods.append(
            TrialDevMethodReplayV1(
                method_route_id=method_route_id,
                estimator_id=estimator_id,
                uncertainty_estimator_id=str(method["uncertainty_estimator_id"]),
                bootstrap_replicates=bootstrap_replicates,
                confidence_level=method_confidence,
                candidate_results=tuple(candidate_rows),
                expected_rankings=expected_rankings,
                replayed_rankings=replayed_rankings,
                expected_actions=expected_actions,
                replayed_actions=replayed_actions,
                expected_acceptable_utility_sets=expected_acceptable_utility_sets,
                replayed_acceptable_utility_sets=replayed_acceptable_utility_sets,
                expected_definitely_qualified_sets=expected_definitely_qualified_sets,
                replayed_definitely_qualified_sets=replayed_definitely_qualified_sets,
                expected_possibly_qualified_sets=expected_possibly_qualified_sets,
                replayed_possibly_qualified_sets=replayed_possibly_qualified_sets,
                expected_pairwise_contrast_half_widths=expected_pairwise_contrast_half_widths,
                replayed_pairwise_contrast_half_widths=replayed_pairwise_contrast_half_widths,
                maximum_utility_absolute_error=max_utility_error,
                maximum_efficacy_gain_absolute_error=max_efficacy_error,
                maximum_standard_error_absolute_error=max_standard_error,
                maximum_interval_endpoint_absolute_error=max_interval_error,
                maximum_pairwise_contrast_absolute_error=max_pairwise_error,
                ranking_match=ranking_match,
                action_match=action_match,
                uncertainty_policy_match=uncertainty_policy_match,
                status="pass" if passed else "fail",
            )
        )
    checksums_match = _checksum_match(root, expected_report)
    passed = checksums_match and all(row.status == "pass" for row in replayed_methods)
    return TrialDevObservationalReplayReportV1(
        scenario_id=scenario_id,
        absolute_tolerance=absolute_tolerance,
        public_input_checksums_match=checksums_match,
        methods=tuple(replayed_methods),
        status="pass" if passed else "fail",
    )


__all__ = ["replay_trialdev_observational_reference"]
