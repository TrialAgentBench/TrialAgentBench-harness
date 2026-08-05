"""Replay method-specific TrialDev analysis reference from participant evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from trialagentbench_harness.numeric_policy import MINIMUM_TWO_SIDED_CONFIDENCE_LEVEL_V1
from trialagentbench_harness.trialdev.grading.io import read_json
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentRequestV1
from trialagentbench_harness.trialdev.share.public_method_design import TrialDevPhaseAnalysisMethodCatalogV1

__all__ = [
    "TrialDevEffectReferenceV1",
    "aalen_johansen_cif_variance_v1",
    "derive_effect_references_v1",
    "interval_equivalence_score_v1",
    "point_equivalence_score_v1",
    "point_interval_equivalence_score_v1",
    "reporting_tolerance_v1",
]


@dataclass(frozen=True, slots=True)
class TrialDevEffectReferenceV1:
    """One method-specific effect result replayed from public trial tables."""

    candidate_drug_id: str
    method_route_id: str
    endpoint_id: str
    estimand_id: str
    estimator_id: str
    effect_scale_id: str
    horizon_days: int
    estimate: float
    standard_error: float
    lower: float
    upper: float
    confidence_level: float
    analysis_population: str
    source_checksums: tuple[tuple[str, str], ...]
    absolute_tolerance: float

    @property
    def checksum(self) -> str:
        """Return a deterministic reference checksum."""

        payload = {
            "candidate_drug_id": self.candidate_drug_id,
            "method_route_id": self.method_route_id,
            "endpoint_id": self.endpoint_id,
            "estimand_id": self.estimand_id,
            "estimator_id": self.estimator_id,
            "effect_scale_id": self.effect_scale_id,
            "horizon_days": self.horizon_days,
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "analysis_population": self.analysis_population,
            "source_checksums": self.source_checksums,
            "absolute_tolerance": self.absolute_tolerance,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aalen_johansen_cif_variance_v1(*, frame: pd.DataFrame, horizon: float) -> tuple[float, float]:
    """Estimate event-of-interest cumulative incidence and asymptotic variance."""

    if frame.empty:
        raise ValueError("Aalen-Johansen replay requires a non-empty analysis surface.")
    missing_columns = sorted({"TIME", "EVENT", "COMPETING_EVENT"} - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Aalen-Johansen replay surface is missing required columns: {missing_columns!r}.")
    times = pd.to_numeric(frame["TIME"], errors="raise").astype(float)
    raw_events = pd.to_numeric(frame["EVENT"], errors="raise").astype(float)
    raw_competing = pd.to_numeric(frame["COMPETING_EVENT"], errors="raise").astype(float)
    if not raw_events.isin((0.0, 1.0)).all() or not raw_competing.isin((0.0, 1.0)).all():
        raise ValueError("Aalen-Johansen replay requires binary EVENT and COMPETING_EVENT surfaces.")
    if bool(((raw_events == 1.0) & (raw_competing == 1.0)).any()):
        raise ValueError("EVENT and COMPETING_EVENT must be mutually exclusive.")
    if not times.map(math.isfinite).all() or (times < 0.0).any():
        raise ValueError("Aalen-Johansen replay requires finite, non-negative TIME values.")
    if not math.isfinite(float(horizon)) or float(horizon) <= 0.0:
        raise ValueError("Aalen-Johansen replay horizon must be finite and positive.")

    time_values = times.to_numpy(dtype=float, copy=False)
    event_values = raw_events.to_numpy(dtype=int, copy=False)
    competing_values = raw_competing.to_numpy(dtype=int, copy=False)
    any_event = (event_values == 1) | (competing_values == 1)
    observed = any_event & (time_values <= float(horizon))
    event_times, event_time_index = np.unique(time_values[observed], return_inverse=True)
    primary_counts = (
        np.bincount(
            event_time_index,
            weights=(event_values[observed] == 1).astype(int),
            minlength=len(event_times),
        )
        .astype(int)
        .tolist()
    )
    all_counts = np.bincount(event_time_index, minlength=len(event_times)).tolist()
    sorted_times = np.sort(time_values)
    at_risk_counts = [
        len(time_values) - int(np.searchsorted(sorted_times, event_time, side="left")) for event_time in event_times
    ]

    survival = 1.0
    cif = 0.0
    increments: list[tuple[float, float, int, int, int]] = []
    for at_risk_value, primary_value, all_value in zip(
        at_risk_counts,
        primary_counts,
        all_counts,
        strict=True,
    ):
        at_risk = int(at_risk_value)
        primary_count = int(primary_value)
        all_count = int(all_value)
        if at_risk <= 0 or all_count <= 0:
            continue
        survival_before = survival
        cif += survival_before * float(primary_count) / float(at_risk)
        survival *= 1.0 - float(all_count) / float(at_risk)
        increments.append((cif, survival_before, at_risk, primary_count, all_count))

    variance = 0.0
    for cif_after, survival_before, at_risk, primary_count, all_count in increments:
        remaining_cif = cif - cif_after
        if at_risk > all_count:
            variance += remaining_cif * remaining_cif * float(all_count) / float(at_risk * (at_risk - all_count))
        variance += (
            survival_before * survival_before * float(primary_count * (at_risk - primary_count)) / float(at_risk**3)
        )
        variance -= 2.0 * remaining_cif * survival_before * float(primary_count) / float(at_risk**2)
    return float(cif), max(0.0, float(variance))


def derive_effect_references_v1(
    *, scenario_root: Path, trial_output_root: Path
) -> tuple[TrialDevEffectReferenceV1, ...]:
    """Derive accepted cumulative-incidence effects from participant evidence."""

    policy = read_json(Path(scenario_root) / "public" / "phase_decision_evidence_policy.json")
    if not isinstance(policy, dict) or policy.get("schema_id") != "trialdev_phase_decision_evidence_policy_v1":
        raise ValueError("Unsupported phase decision evidence policy schema.")
    confidence_value = policy.get("confidence_level")
    if isinstance(confidence_value, bool) or not isinstance(confidence_value, int | float):
        raise ValueError("phase decision policy requires a numeric confidence_level.")
    confidence_level = float(confidence_value)
    if not MINIMUM_TWO_SIDED_CONFIDENCE_LEVEL_V1 < float(confidence_level) < 1.0:
        raise ValueError(
            "confidence_level must exceed " f"{MINIMUM_TWO_SIDED_CONFIDENCE_LEVEL_V1} and be less than 1."
        )
    objective_charter = read_json(Path(scenario_root) / "public" / "objective_charter.json")
    decimal_places = objective_charter.get("numeric_reporting_decimal_places")
    if isinstance(decimal_places, bool) or not isinstance(decimal_places, int):
        raise ValueError("objective charter requires integer numeric_reporting_decimal_places.")
    absolute_tolerance = reporting_tolerance_v1(decimal_places=decimal_places)
    root = Path(trial_output_root)
    method_catalog = TrialDevPhaseAnalysisMethodCatalogV1.model_validate(
        read_json(Path(scenario_root) / "public" / "phase_analysis_method_catalog.json")
    )
    if method_catalog.confidence_level != confidence_level:
        raise ValueError("Phase method catalog and decision policy confidence levels differ.")
    request_payload = read_json(root / "request.json")
    mapping = read_json(root / "arm_mapping.json")
    if not isinstance(request_payload, dict) or not isinstance(mapping, dict):
        raise ValueError("TrialDev public replay requires object request and arm-mapping payloads.")
    request = TrialDevelopmentRequestV1.model_validate(request_payload)
    if request.endpoint_id is None:
        return tuple()
    if request.follow_up_days is None or request.treatment_discontinuation_strategy is None:
        raise ValueError("Randomized efficacy replay requires an explicit horizon and estimand strategy.")
    endpoint_id = str(request.endpoint_id)
    horizon = int(request.follow_up_days)
    treatment_discontinuation_strategy = str(request.treatment_discontinuation_strategy)
    method = method_catalog.method_for_phase(str(request.phase_id))
    if method.result_shape != "efficacy_safety_bundle":
        return tuple()
    if method.efficacy_estimand_id_template is None or method.efficacy_effect_scale_id is None:
        raise ValueError(f"Phase {request.phase_id!r} lacks an efficacy method specification.")
    control_arm = str(mapping.get("control_arm_id") or "")
    candidate_arms_raw = mapping.get("candidate_arm_ids")
    drug_by_arm = mapping.get("drug_id_by_arm")
    if not control_arm or not isinstance(candidate_arms_raw, list) or not isinstance(drug_by_arm, dict):
        raise ValueError("arm_mapping.json lacks the control/candidate identity contract.")
    endpoints = pd.read_parquet(root / "endpoints.parquet")
    if "ARM" not in endpoints.columns:
        raise ValueError("TrialDev endpoint surface is missing ARM.")
    expected_arms = {control_arm, *(str(value) for value in candidate_arms_raw)}
    observed_arms = set(endpoints["ARM"].astype(str))
    if observed_arms != expected_arms:
        raise ValueError(
            f"Endpoint ARM values do not match arm_mapping.json: observed={sorted(observed_arms)!r}, "
            f"expected={sorted(expected_arms)!r}."
        )
    strategies = set(endpoints.get("TREATMENT_DISCONTINUATION_STRATEGY", pd.Series(dtype="string")).astype(str))
    if strategies != {treatment_discontinuation_strategy}:
        raise ValueError(
            "Endpoint TREATMENT_DISCONTINUATION_STRATEGY does not match request.json: "
            f"observed={sorted(strategies)!r}, expected={treatment_discontinuation_strategy!r}."
        )
    control = endpoints.loc[endpoints["ARM"].astype(str) == control_arm]
    control_risk, control_risk_var = aalen_johansen_cif_variance_v1(
        frame=control,
        horizon=float(horizon),
    )
    z = float(NormalDist().inv_cdf(0.5 + float(confidence_level) / 2.0))
    checksums = tuple(
        sorted(
            {
                "trial_output/arm_mapping.json": _sha256_file(root / "arm_mapping.json"),
                "trial_output/endpoints.parquet": _sha256_file(root / "endpoints.parquet"),
                "trial_output/request.json": _sha256_file(root / "request.json"),
            }.items()
        )
    )
    references: list[TrialDevEffectReferenceV1] = []
    for treatment_arm in (str(value) for value in candidate_arms_raw):
        treatment = endpoints.loc[endpoints["ARM"].astype(str) == treatment_arm]
        treatment_risk, treatment_risk_var = aalen_johansen_cif_variance_v1(
            frame=treatment,
            horizon=float(horizon),
        )
        candidate_id = str(drug_by_arm.get(treatment_arm) or "")
        if not candidate_id:
            raise ValueError(f"arm_mapping.json has no drug identity for candidate arm {treatment_arm!r}.")
        if method.estimator_id != "observed:aalen_johansen_cif_tau":
            raise ValueError(f"No public replay calculator exists for estimator_id={method.estimator_id!r}.")
        estimate = float(control_risk - treatment_risk)
        variance = float(control_risk_var + treatment_risk_var)
        standard_error = math.sqrt(max(variance, 0.0))
        references.append(
            TrialDevEffectReferenceV1(
                candidate_drug_id=candidate_id,
                method_route_id=method.method_route_id,
                endpoint_id=endpoint_id,
                estimand_id=method.efficacy_estimand_id_template.format(
                    treatment_discontinuation_strategy=treatment_discontinuation_strategy
                ),
                estimator_id=method.estimator_id,
                effect_scale_id=method.efficacy_effect_scale_id,
                horizon_days=horizon,
                estimate=estimate,
                standard_error=standard_error,
                lower=estimate - z * standard_error,
                upper=estimate + z * standard_error,
                confidence_level=float(confidence_level),
                analysis_population=method.analysis_population,
                source_checksums=checksums,
                absolute_tolerance=absolute_tolerance,
            )
        )
    return tuple(references)


def reporting_tolerance_v1(*, decimal_places: int) -> float:
    """Return half one unit in the participant-declared reporting precision."""

    if isinstance(decimal_places, bool) or not 2 <= decimal_places <= 6:
        raise ValueError("numeric reporting precision must be an integer from 2 through 6.")
    return 0.5 * 10.0 ** (-decimal_places)


def point_equivalence_score_v1(
    *,
    observed: float,
    expected: float,
    absolute_tolerance: float,
) -> float:
    """Return binary agreement under a prospective native-unit envelope."""

    values = (observed, expected, absolute_tolerance)
    if not all(math.isfinite(float(value)) for value in values) or absolute_tolerance <= 0.0:
        raise ValueError("point equivalence requires finite values and a positive tolerance.")
    return float(abs(float(observed) - float(expected)) <= float(absolute_tolerance))


def interval_equivalence_score_v1(
    *,
    lower: float,
    upper: float,
    reference_lower: float,
    reference_upper: float,
    endpoint_tolerance: float,
) -> float:
    """Return binary endpoint agreement under a prospective native-unit envelope."""

    values = (lower, upper, reference_lower, reference_upper, endpoint_tolerance)
    if not all(math.isfinite(float(value)) for value in values) or endpoint_tolerance <= 0.0:
        raise ValueError("interval equivalence requires finite values and a positive tolerance.")
    if lower > upper or reference_lower > reference_upper:
        raise ValueError("interval lower bound must not exceed upper bound.")
    endpoint_error = max(
        abs(float(lower) - float(reference_lower)),
        abs(float(upper) - float(reference_upper)),
    )
    return float(endpoint_error <= float(endpoint_tolerance))


def point_interval_equivalence_score_v1(
    *,
    estimate: float,
    lower: float,
    upper: float,
    reference_estimate: float,
    reference_lower: float,
    reference_upper: float,
    absolute_tolerance: float,
) -> float:
    """Require both point and interval endpoints to satisfy one declared envelope."""

    point = point_equivalence_score_v1(
        observed=estimate,
        expected=reference_estimate,
        absolute_tolerance=absolute_tolerance,
    )
    interval = interval_equivalence_score_v1(
        lower=lower,
        upper=upper,
        reference_lower=reference_lower,
        reference_upper=reference_upper,
        endpoint_tolerance=absolute_tolerance,
    )
    return min(point, interval)
