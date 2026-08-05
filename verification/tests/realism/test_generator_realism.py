from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from trialagentbench_validation.cli import main
from trialagentbench_validation.external.realism.generator_realism import (
    GeneratorRealismSummaryV1,
    compare_generator_realism,
    extract_generated_trial_fingerprints,
    fingerprint_standard_trial,
    fingerprint_survival_trial,
    fingerprint_trial_baseline,
)


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    stream = BytesIO()
    frame.to_parquet(stream, index=False)
    return stream.getvalue()


def _participant_release(path: Path, *, include_prepared: bool = True) -> None:
    with ZipFile(path, "w") as archive:
        for trial_index in range(3):
            trial_id = f"TRIAL-{trial_index}"
            study_id = f"STUDY-{trial_index}"
            common = {
                "task.json": {
                    "primary_endpoint_term": "Death",
                    "protocol_summary_file": "protocol_summary.json",
                },
                "protocol_summary.json": {
                    "trial_id": trial_id,
                    "study_id": study_id,
                    "design_family": "randomized_trial",
                },
                "endpoint_definition.json": {
                    "endpoints": [{"term": "Death", "endpoint_id": "death"}],
                },
            }
            raw_prefix = f"items/RAW-{trial_index}"
            generated = _survival_trial(100 + trial_index)
            subject_ids = [f"S{index}" for index in range(len(generated))]
            for name, payload in common.items():
                archive.writestr(f"{raw_prefix}/{name}", json.dumps(payload))
            archive.writestr(f"{raw_prefix}/reconstruction_task.json", "{}")
            archive.writestr(
                f"{raw_prefix}/data/raw/subjects.parquet",
                _parquet_bytes(pd.DataFrame({"USUBJID": subject_ids})),
            )
            if not include_prepared:
                continue
            prefix = f"items/PREPARED-{trial_index}"
            for name, payload in common.items():
                archive.writestr(f"{prefix}/{name}", json.dumps(payload))
            baseline = pd.DataFrame(
                {
                    "USUBJID": subject_ids,
                    "TRTA": generated["treatment"],
                    "AGE": generated["age"],
                    "BMI": generated["bmi"],
                }
            )
            endpoint = pd.DataFrame(
                {
                    "USUBJID": baseline["USUBJID"],
                    "PARAMCD": "death",
                    "AVAL": generated["time"],
                    "CNSR": 1 - generated["event"],
                }
            )
            archive.writestr(f"{prefix}/data/ADSL.parquet", _parquet_bytes(baseline))
            archive.writestr(f"{prefix}/data/ADTTE.parquet", _parquet_bytes(endpoint))


def _standard_trial(seed: int, *, treatment_effect: float = 0.3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    participants = 160
    treatment = np.repeat(("control", "treated"), participants // 2)
    age = rng.normal(58.0, 11.0, participants)
    bmi = 27.0 + 0.08 * (age - 58.0) + rng.normal(0.0, 3.0, participants)
    outcome = (
        treatment_effect * (treatment == "treated")
        + 0.03 * age
        + 0.08 * bmi
        + rng.normal(0.0, 1.0, participants)
    )
    return pd.DataFrame(
        {
            "treatment": treatment,
            "age": age,
            "bmi": bmi,
            "outcome": outcome,
        }
    )


def _survival_trial(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    participants = 240
    treatment = np.repeat(("control", "treated"), participants // 2)
    age = rng.normal(59.0, 10.0, participants)
    bmi = 27.5 + 0.06 * (age - 59.0) + rng.normal(0.0, 3.2, participants)
    linear_predictor = (
        -0.25 * (treatment == "treated") + 0.015 * (age - 59.0) + 0.025 * (bmi - 27.5)
    )
    event_time = rng.exponential(120.0 / np.exp(linear_predictor))
    censoring_time = rng.uniform(80.0, 240.0, participants)
    return pd.DataFrame(
        {
            "treatment": treatment,
            "age": age,
            "bmi": bmi,
            "time": np.minimum(event_time, censoring_time),
            "event": (event_time <= censoring_time).astype(float),
        }
    )


def test_endpoint_aware_fingerprints_are_scale_free() -> None:
    standard = fingerprint_standard_trial(
        _standard_trial(4),
        trial_id="RCT-1",
        source="external_rct",
        outcome_kind="continuous",
    )
    survival = fingerprint_survival_trial(_survival_trial(8), trial_id="GENERATED-1")

    assert standard.participants == 160
    assert survival.participants == 240
    assert standard.adjustment_shift_in_unadjusted_se >= 0
    assert survival.adjusted_to_unadjusted_se_ratio > 0
    assert 0 < standard.maximum_scaled_baseline_smd < 5
    assert 0 < survival.maximum_scaled_baseline_smd < 5


def test_release_fingerprints_use_matched_prepared_context_for_raw_views(
    tmp_path: Path,
) -> None:
    participant_release = tmp_path / "participant.zip"
    _participant_release(participant_release)

    fingerprints = extract_generated_trial_fingerprints(participant_release)

    assert len(fingerprints) == 3
    assert all(row.participants == 240 for row in fingerprints)


def test_release_fingerprints_reject_trials_without_an_analysis_view(
    tmp_path: Path,
) -> None:
    participant_release = tmp_path / "participant.zip"
    _participant_release(participant_release, include_prepared=False)

    with pytest.raises(ValueError, match="no matched analysis-ready view"):
        extract_generated_trial_fingerprints(participant_release)


def test_generator_comparison_reports_three_validation_domains() -> None:
    external = tuple(
        fingerprint_standard_trial(
            _standard_trial(seed),
            trial_id=f"RCT-{seed}",
            source="external_rct",
            outcome_kind="continuous",
        )
        for seed in range(5)
    )
    synthetic = tuple(
        fingerprint_survival_trial(_survival_trial(seed), trial_id=f"GENERATED-{seed}")
        for seed in range(8)
    )

    summary = compare_generator_realism(
        external,
        external,
        synthetic,
        bootstrap_replicates=500,
        seed=12,
    )

    assert summary.external_baseline_trials == 5
    assert summary.external_analysis_trials == 5
    assert summary.synthetic_trials == 8
    assert {row.validation_domain for row in summary.constructs} == {
        "marginal",
        "joint_structure",
        "analysis_impact",
    }
    assert all(
        row.standardized_wasserstein_ci_high >= row.standardized_wasserstein_ci_low
        for row in summary.constructs
    )


def test_public_cli_replays_generator_comparison(tmp_path: Path) -> None:
    external_baseline = tuple(
        fingerprint_trial_baseline(
            _standard_trial(seed),
            trial_id=f"RCT-{seed}",
            source="external_rct",
        )
        for seed in range(5)
    )
    external = tuple(
        fingerprint_standard_trial(
            _standard_trial(seed),
            trial_id=f"RCT-{seed}",
            source="external_rct",
            outcome_kind="continuous",
        )
        for seed in range(5)
    )
    synthetic = tuple(
        fingerprint_survival_trial(_survival_trial(seed), trial_id=f"GENERATED-{seed}")
        for seed in range(8)
    )
    baseline_path = tmp_path / "external_baseline.jsonl"
    analysis_path = tmp_path / "external_analysis.jsonl"
    synthetic_path = tmp_path / "synthetic.jsonl"
    output_path = tmp_path / "summary.json"
    baseline_path.write_text(
        "".join(row.model_dump_json() + "\n" for row in external_baseline),
        encoding="utf-8",
    )
    analysis_path.write_text(
        "".join(row.model_dump_json() + "\n" for row in external), encoding="utf-8"
    )
    synthetic_path.write_text(
        "".join(row.model_dump_json() + "\n" for row in synthetic), encoding="utf-8"
    )

    result = main(
        [
            "generator-realism-summary",
            "--external-baseline",
            str(baseline_path),
            "--external-analysis",
            str(analysis_path),
            "--synthetic",
            str(synthetic_path),
            "--output",
            str(output_path),
            "--bootstrap-replicates",
            "500",
            "--seed",
            "12",
        ]
    )

    expected = compare_generator_realism(
        external_baseline,
        external,
        synthetic,
        bootstrap_replicates=500,
        seed=12,
    )
    replayed = GeneratorRealismSummaryV1.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert result == 0
    assert replayed == expected
