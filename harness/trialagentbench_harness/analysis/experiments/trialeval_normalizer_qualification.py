"""Select and analyse a prospective TrialEval narrative-normalizer qualification sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np

from trialagentbench_harness.contracts.experiments import (
    TrialEvalExperimentProtocolV1,
    TrialEvalNarrativeClaimV1,
    TrialEvalNarrativeTranscriptionV1,
    TrialEvalNormalizerFrameUnitV1,
    TrialEvalNormalizerFrameV1,
    TrialEvalNormalizerQualificationDesignV1,
    TrialEvalNormalizerQualificationMetricV1,
    TrialEvalNormalizerQualificationObservationSetV1,
    TrialEvalNormalizerQualificationObservationV1,
    TrialEvalNormalizerQualificationReportV1,
    TrialEvalNormalizerQualificationUnitResultV1,
    TrialEvalNormalizerSampleUnitV1,
    TrialEvalNormalizerSampleV1,
)
from trialagentbench_harness.io import read_json_model, write_json_model


def _stratum_id(unit: TrialEvalNormalizerFrameUnitV1, fields: tuple[str, ...]) -> str:
    values = {
        "evaluation_series_id": unit.regime_cell_id.rsplit("-", maxsplit=1)[0],
        "assumption_tier": unit.assumption_tier,
        "context_configuration": unit.context_configuration,
    }
    return "|".join(f"{field}={values[field]}" for field in fields)


def _rank(seed: int, *parts: str) -> str:
    payload = ":".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_normalizer_qualification_sample_v1(
    *,
    frame: TrialEvalNormalizerFrameV1,
    experiment_design_checksum: str,
    design: TrialEvalNormalizerQualificationDesignV1,
    selection_seed: int,
) -> TrialEvalNormalizerSampleV1:
    """Select the frozen sample using outcome-blind stratified hash ranking."""

    if frame.checksum is None:
        raise ValueError("Normalizer qualification frame must be frozen with a checksum before sampling.")
    ordered_frame = frame.units
    unit_ids = tuple(row.unit_id for row in ordered_frame)
    assignment_keys = tuple((row.model_id, row.assignment_id) for row in ordered_frame)
    if len(unit_ids) != len(set(unit_ids)) or len(assignment_keys) != len(set(assignment_keys)):
        raise ValueError("Normalizer qualification frame units must be unique.")
    fields = cast(tuple[str, ...], design.inclusion_strata)
    candidates_by_base: dict[str, list[TrialEvalNormalizerFrameUnitV1]] = defaultdict(list)
    for unit in ordered_frame:
        candidates_by_base[unit.base_trial_id].append(unit)
    base_units: dict[str, TrialEvalNormalizerFrameUnitV1] = {}
    for base_trial_id, candidates in candidates_by_base.items():
        reference = candidates[0]
        scientific_identity = (
            reference.task_id,
            reference.regime_cell_id,
            reference.design_tier,
            reference.assumption_tier,
            reference.context_configuration,
            reference.data_preparation,
            reference.analysis_specification,
            reference.result_shape,
        )
        for candidate in candidates[1:]:
            observed = (
                candidate.task_id,
                candidate.regime_cell_id,
                candidate.design_tier,
                candidate.assumption_tier,
                candidate.context_configuration,
                candidate.data_preparation,
                candidate.analysis_specification,
                candidate.result_shape,
            )
            if observed != scientific_identity:
                raise ValueError(f"Normalizer frame scientific identity drifts within base trial {base_trial_id!r}.")
        base_units[base_trial_id] = reference

    strata: dict[str, list[str]] = defaultdict(list)
    for base_trial_id, unit in base_units.items():
        strata[_stratum_id(unit, fields)].append(base_trial_id)
    minimum_total = len(strata) * design.minimum_reports_per_stratum
    if design.retained_sample_size < minimum_total:
        raise ValueError("Normalizer qualification sample cannot meet its minimum stratum allocation.")
    if design.retained_sample_size > len(base_units):
        raise ValueError("Normalizer qualification sample exceeds its eligible frame.")
    if any(len(units) < design.minimum_reports_per_stratum for units in strata.values()):
        raise ValueError("Normalizer qualification frame has an undersized required stratum.")

    allocation = {stratum: design.minimum_reports_per_stratum for stratum in strata}
    remaining = design.retained_sample_size - minimum_total
    for step in range(remaining):
        eligible = [stratum for stratum, units in strata.items() if allocation[stratum] < len(units)]
        if not eligible:
            raise ValueError("Normalizer qualification frame lacks capacity for the retained sample size.")
        chosen = min(
            eligible,
            key=lambda stratum: (
                allocation[stratum] / len(strata[stratum]),
                _rank(selection_seed, str(step), stratum),
            ),
        )
        allocation[chosen] += 1

    selected: list[TrialEvalNormalizerSampleUnitV1] = []
    for stratum, base_trial_ids in sorted(strata.items()):
        n_selected = allocation[stratum]
        ranked_bases = sorted(
            base_trial_ids,
            key=lambda base_trial_id: (
                _rank(selection_seed, "base_trial", stratum, base_trial_id),
                base_trial_id,
            ),
        )
        for base_trial_id in ranked_bases[:n_selected]:
            candidates = candidates_by_base[base_trial_id]
            unit = min(
                candidates,
                key=lambda candidate: (
                    _rank(selection_seed, "report", base_trial_id, candidate.unit_id),
                    candidate.unit_id,
                ),
            )
            base_probability = n_selected / len(base_trial_ids)
            report_probability = 1.0 / len(candidates)
            selected.append(
                TrialEvalNormalizerSampleUnitV1(
                    **unit.model_dump(mode="python"),
                    stratum_id=stratum,
                    frame_base_trial_count=len(base_trial_ids),
                    sampled_base_trial_count=n_selected,
                    base_trial_candidate_report_count=len(candidates),
                    base_trial_inclusion_probability=base_probability,
                    within_base_report_inclusion_probability=report_probability,
                    inclusion_probability=base_probability * report_probability,
                    selected_without_normalizer_or_score_outcomes=True,
                )
            )
    return TrialEvalNormalizerSampleV1(
        experiment_design_checksum=experiment_design_checksum,
        frame_checksum=frame.checksum,
        selection_method="stratified_base_trial_then_within_base_hash_rank_v1",
        selection_seed=selection_seed,
        units=tuple(sorted(selected, key=lambda row: row.unit_id)),
    ).with_checksum()


def _claim_key(claim: TrialEvalNarrativeClaimV1) -> str:
    payload = claim.model_dump(
        mode="json",
        exclude={"claim_id", "spans", "extraction_confidence"},
    )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _claim_keys(transcription: TrialEvalNarrativeTranscriptionV1) -> set[str]:
    return {_claim_key(claim) for claim in transcription.claims}


def _eligible_primary_claims(transcription: TrialEvalNarrativeTranscriptionV1) -> set[str]:
    return {
        _claim_key(claim)
        for claim in transcription.claims
        if claim.claim_role == "primary" and claim.evidence_level in {"executed", "substantiated"}
    }


def _primary_result_claim(transcription: TrialEvalNarrativeTranscriptionV1) -> TrialEvalNarrativeClaimV1 | None:
    claims = tuple(
        claim
        for claim in transcription.claims
        if claim.field_path == "primary_analysis.result"
        and claim.claim_role == "primary"
        and claim.evidence_level in {"executed", "substantiated"}
        and not claim.conflict
    )
    return claims[0] if len(claims) == 1 else None


def _numeric_leaves(value: object, prefix: str = "") -> dict[str, float]:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return {}
    if isinstance(value, (int, float)):
        return {prefix: float(value)}
    if isinstance(value, list):
        output: dict[str, float] = {}
        for index, child in enumerate(value):
            output.update(_numeric_leaves(child, f"{prefix}[{index}]"))
        return output
    if isinstance(value, dict):
        output = {}
        for key, child in sorted(value.items()):
            output.update(_numeric_leaves(child, f"{prefix}.{key}" if prefix else str(key)))
        return output
    raise TypeError(f"Unsupported parsed narrative value: {type(value).__name__}.")


def _semantic_transcription(transcription: TrialEvalNarrativeTranscriptionV1) -> str:
    submission = None if transcription.submission is None else transcription.submission.model_dump(mode="json")
    return json.dumps(
        {
            "status": transcription.status,
            "submission": submission,
            "claims": sorted(_claim_key(claim) for claim in transcription.claims),
            "abstention_reason": transcription.abstention_reason,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _unit_result(
    observation: TrialEvalNormalizerQualificationObservationV1,
) -> TrialEvalNormalizerQualificationUnitResultV1:
    unit = observation.sample_unit
    human = observation.masked_human_reference
    automated = observation.automated_repeats[0]
    human_keys = _claim_keys(human)
    automated_keys = _claim_keys(automated)
    human_primary = _eligible_primary_claims(human)
    automated_primary = _eligible_primary_claims(automated)
    human_result = _primary_result_claim(human)
    automated_result = _primary_result_claim(automated)
    shape_agreement: bool | None = None
    unit_agreement: bool | None = None
    numeric_agreement: bool | None = None
    if human_result is not None and automated_result is not None:
        shape_agreement = human_result.result_shape == automated_result.result_shape
        unit_agreement = human_result.unit == automated_result.unit
        human_numeric = _numeric_leaves(human_result.parsed_value)
        automated_numeric = _numeric_leaves(automated_result.parsed_value)
        numeric_agreement = human_numeric == automated_numeric

    human_endpoint = observation.masked_human_endpoint
    automated_endpoint = observation.automated_endpoint
    return TrialEvalNormalizerQualificationUnitResultV1(
        unit_id=unit.unit_id,
        model_id=unit.model_id,
        regime_cell_id=unit.regime_cell_id,
        base_trial_id=unit.base_trial_id,
        design_tier=unit.design_tier,
        assumption_tier=unit.assumption_tier,
        context_configuration=unit.context_configuration,
        result_shape=unit.result_shape,
        inclusion_probability=unit.inclusion_probability,
        score_relevant_error=(human.status != automated.status or human.submission != automated.submission),
        claim_true_positive=len(human_keys & automated_keys),
        claim_false_positive=len(automated_keys - human_keys),
        claim_false_negative=len(human_keys - automated_keys),
        primary_role_false_positive=bool(automated_primary - human_primary),
        result_shape_agreement=shape_agreement,
        result_unit_agreement=unit_agreement,
        numeric_value_agreement=numeric_agreement,
        abstention_agreement=(human.status == "abstain") == (automated.status == "abstain"),
        stable_across_repeats=len({_semantic_transcription(row) for row in observation.automated_repeats}) == 1,
        usable_primary_disagreement=human_endpoint.usable_primary != automated_endpoint.usable_primary,
        route_match_disagreement=human_endpoint.route_match != automated_endpoint.route_match,
        numeric_availability_disagreement=(
            human_endpoint.numeric_result_available != automated_endpoint.numeric_result_available
        ),
        primary_analysis_conformance_disagreement=human_endpoint.primary_analysis_conforms
        != automated_endpoint.primary_analysis_conforms,
        planning_valid_disagreement=(
            None
            if human_endpoint.planning_valid is None and automated_endpoint.planning_valid is None
            else human_endpoint.planning_valid != automated_endpoint.planning_valid
        ),
    )


_Contribution = Callable[[TrialEvalNormalizerQualificationUnitResultV1], tuple[float, float]]


def _binary(field: str, *, invert: bool = False, nullable: bool = False) -> _Contribution:
    def contribution(row: TrialEvalNormalizerQualificationUnitResultV1) -> tuple[float, float]:
        value = getattr(row, field)
        if value is None and nullable:
            return 0.0, 0.0
        if not isinstance(value, bool):
            raise TypeError(f"Normalizer metric field {field!r} is not binary.")
        return float(not value if invert else value), 1.0

    return contribution


_METRICS: tuple[tuple[str, _Contribution], ...] = (
    ("score_relevant_error_rate", _binary("score_relevant_error")),
    ("claim_precision", lambda row: (row.claim_true_positive, row.claim_true_positive + row.claim_false_positive)),
    ("claim_recall", lambda row: (row.claim_true_positive, row.claim_true_positive + row.claim_false_negative)),
    ("primary_role_false_positive_rate", _binary("primary_role_false_positive")),
    ("result_shape_agreement_rate", _binary("result_shape_agreement", nullable=True)),
    ("result_unit_agreement_rate", _binary("result_unit_agreement", nullable=True)),
    ("numeric_value_agreement_rate", _binary("numeric_value_agreement", nullable=True)),
    ("abstention_agreement_rate", _binary("abstention_agreement")),
    ("instability_rate", _binary("stable_across_repeats", invert=True)),
    ("usable_primary_disagreement_rate", _binary("usable_primary_disagreement")),
    ("route_match_disagreement_rate", _binary("route_match_disagreement")),
    ("numeric_availability_disagreement_rate", _binary("numeric_availability_disagreement")),
    ("primary_analysis_conformance_disagreement_rate", _binary("primary_analysis_conformance_disagreement")),
    ("planning_valid_disagreement_rate", _binary("planning_valid_disagreement", nullable=True)),
)


def _estimate(rows: Sequence[TrialEvalNormalizerQualificationUnitResultV1], contribution: _Contribution) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        row_numerator, row_denominator = contribution(row)
        weight = 1.0 / row.inclusion_probability
        numerator += weight * row_numerator
        denominator += weight * row_denominator
    if denominator <= 0.0:
        raise ValueError("Normalizer metric has no eligible denominator.")
    return numerator / denominator


def _metric(
    *,
    metric: str,
    rows: tuple[TrialEvalNormalizerQualificationUnitResultV1, ...],
    contribution: _Contribution,
    subgroup_dimension: str,
    subgroup_value: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> TrialEvalNormalizerQualificationMetricV1 | None:
    numerator = 0.0
    denominator = 0.0
    denominator_masses: list[float] = []
    for row in rows:
        row_numerator, row_denominator = contribution(row)
        weight = 1.0 / row.inclusion_probability
        numerator += weight * row_numerator
        denominator += weight * row_denominator
        if row_denominator > 0.0:
            denominator_masses.append(weight * row_denominator)
    if denominator <= 0.0:
        return None
    estimate = numerator / denominator
    by_stratum: dict[tuple[float, str], list[TrialEvalNormalizerQualificationUnitResultV1]] = defaultdict(list)
    for row in rows:
        by_stratum[(row.inclusion_probability, row.regime_cell_id + "|" + row.context_configuration)].append(row)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.empty(bootstrap_replicates, dtype=np.float64)
    strata = tuple(by_stratum.values())
    for index in range(bootstrap_replicates):
        sampled: list[TrialEvalNormalizerQualificationUnitResultV1] = []
        for stratum in strata:
            positions = rng.integers(0, len(stratum), size=len(stratum))
            sampled.extend(stratum[int(position)] for position in positions)
        bootstrap[index] = _estimate(sampled, contribution)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(bootstrap, (alpha / 2.0, 1.0 - alpha / 2.0))
    effective_n = denominator * denominator / sum(mass * mass for mass in denominator_masses)
    concentration_radius = math.sqrt(math.log(2.0 / alpha) / (2.0 * effective_n))
    return TrialEvalNormalizerQualificationMetricV1(
        metric=cast(
            Literal[
                "score_relevant_error_rate",
                "claim_precision",
                "claim_recall",
                "primary_role_false_positive_rate",
                "result_shape_agreement_rate",
                "result_unit_agreement_rate",
                "numeric_value_agreement_rate",
                "abstention_agreement_rate",
                "instability_rate",
                "usable_primary_disagreement_rate",
                "route_match_disagreement_rate",
                "numeric_availability_disagreement_rate",
                "primary_analysis_conformance_disagreement_rate",
                "planning_valid_disagreement_rate",
            ],
            metric,
        ),
        subgroup_dimension=cast(
            Literal["overall", "model_id", "design_tier", "assumption_tier", "context", "result_shape"],
            subgroup_dimension,
        ),
        subgroup_value=subgroup_value,
        sampled_units=len(rows),
        weighted_numerator=numerator,
        weighted_denominator=denominator,
        estimate=estimate,
        uncertainty_method="stratified_cluster_bootstrap_with_weighted_hoeffding_envelope",
        confidence_level=confidence_level,
        confidence_lower=max(0.0, min(float(lower), estimate - concentration_radius)),
        confidence_upper=min(1.0, max(float(upper), estimate + concentration_radius)),
    )


def analyse_normalizer_qualification_v1(
    *,
    sample: TrialEvalNormalizerSampleV1,
    design: TrialEvalNormalizerQualificationDesignV1,
    observation_set: TrialEvalNormalizerQualificationObservationSetV1,
) -> TrialEvalNormalizerQualificationReportV1:
    """Estimate normalizer error against masked-human references and apply the frozen rule."""

    bootstrap_replicates = design.secondary_bootstrap_replicates
    bootstrap_seed = design.secondary_bootstrap_seed
    confidence_level = design.secondary_confidence_level
    if sample.checksum is None or observation_set.sample_checksum != sample.checksum:
        raise ValueError("Normalizer qualification observations do not match the frozen sample.")
    observations = observation_set.observations
    sample_by_id = {unit.unit_id: unit for unit in sample.units}
    observed_by_id = {row.sample_unit.unit_id: row for row in observations}
    if len(observed_by_id) != len(observations) or set(observed_by_id) != set(sample_by_id):
        raise ValueError("Normalizer qualification observations do not cover the exact sampled denominator.")
    if any(observed_by_id[unit_id].sample_unit != unit for unit_id, unit in sample_by_id.items()):
        raise ValueError("Normalizer qualification observation sample metadata drift.")
    rows = tuple(_unit_result(observed_by_id[unit.unit_id]) for unit in sample.units)
    groups: list[tuple[str, str, tuple[TrialEvalNormalizerQualificationUnitResultV1, ...]]] = [
        ("overall", "all", rows)
    ]
    dimensions = {
        "model_id": lambda row: row.model_id,
        "design_tier": lambda row: row.design_tier,
        "assumption_tier": lambda row: row.assumption_tier,
        "context": lambda row: row.context_configuration,
        "result_shape": lambda row: row.result_shape,
    }
    for dimension, key in dimensions.items():
        values = sorted({key(row) for row in rows})
        groups.extend((dimension, value, tuple(row for row in rows if key(row) == value)) for value in values)
    metrics: list[TrialEvalNormalizerQualificationMetricV1] = []
    for group_index, (dimension, value, group_rows) in enumerate(groups):
        for metric_index, (metric_name, contribution) in enumerate(_METRICS):
            result = _metric(
                metric=metric_name,
                rows=group_rows,
                contribution=contribution,
                subgroup_dimension=dimension,
                subgroup_value=value,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + group_index * len(_METRICS) + metric_index,
                confidence_level=confidence_level,
            )
            if result is not None:
                metrics.append(result)
    exact_binomial_eligible = len({row.base_trial_id for row in rows}) == len(rows)
    error_count = sum(row.score_relevant_error for row in rows)
    return TrialEvalNormalizerQualificationReportV1(
        sample_checksum=cast(str, sample.checksum),
        qualification_design=design,
        confidence_level=confidence_level,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        exact_binomial_eligible=exact_binomial_eligible,
        observed_error_count=error_count,
        qualified=(
            exact_binomial_eligible
            and len(rows) == design.retained_sample_size
            and error_count <= design.maximum_accepted_errors
        ),
        unit_results=rows,
        metrics=tuple(metrics),
    ).with_checksum()


def select_main(argv: Sequence[str] | None = None) -> int:
    """Select an outcome-blind normalizer qualification sample from a typed frame."""

    parser = argparse.ArgumentParser(description=select_main.__doc__)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--experiment-design", required=True)
    parser.add_argument("--selection-seed", required=True, type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    frame = read_json_model(TrialEvalNormalizerFrameV1, Path(args.frame))
    experiment_design = read_json_model(TrialEvalExperimentProtocolV1, Path(args.experiment_design))
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite normalizer qualification sample: {output}")
    sample = select_normalizer_qualification_sample_v1(
        frame=frame,
        experiment_design_checksum=experiment_design.checksum,
        design=experiment_design.normalizer_qualification,
        selection_seed=args.selection_seed,
    )
    write_json_model(output, sample)
    print(f"Selected {len(sample.units)} narrative reports: {output}")
    return 0


def analyse_main(argv: Sequence[str] | None = None) -> int:
    """Analyse paired human and automated normalization evidence."""

    parser = argparse.ArgumentParser(description=analyse_main.__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--experiment-design", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite normalizer qualification report: {output}")
    sample = read_json_model(TrialEvalNormalizerSampleV1, Path(args.sample))
    experiment_design = read_json_model(TrialEvalExperimentProtocolV1, Path(args.experiment_design))
    if sample.experiment_design_checksum != experiment_design.checksum:
        raise ValueError("Normalizer qualification sample does not match the experiment design.")
    observation_set = read_json_model(TrialEvalNormalizerQualificationObservationSetV1, Path(args.observations))
    report = analyse_normalizer_qualification_v1(
        sample=sample,
        design=experiment_design.normalizer_qualification,
        observation_set=observation_set,
    )
    write_json_model(output, report)
    print(f"Analysed {len(report.unit_results)} narrative reports: {output}")
    return 0


__all__ = [
    "analyse_main",
    "analyse_normalizer_qualification_v1",
    "select_main",
    "select_normalizer_qualification_sample_v1",
]
