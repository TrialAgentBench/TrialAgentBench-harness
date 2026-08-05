"""Tests for the canonical clinical-trial simulation validation builder."""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

import pytest

from trialagentbench_validation.contracts.simulation_validation_bundle import (
    read_validation_results,
)
from trialagentbench_validation.external.release.bundle import (
    build_simulation_validation_bundle,
)
from trialagentbench_validation.validation_figures.report import (
    render_validation_report_figures,
)


def test_repository_validation_bundle_builds_from_public_files(tmp_path: Path) -> None:
    """The checked-in validation bundle has complete methods, data, and figures."""

    package_root = Path(__file__).resolve().parents[2]
    validation_root = tmp_path / "validation_results"
    shutil.copytree(package_root / "validation_results", validation_root)
    bundle = build_simulation_validation_bundle(
        validation_root=validation_root,
        verifier_lock=package_root / "uv.lock",
    )
    assert len(bundle.figures) == 15
    assert {figure.figure_id for figure in bundle.figures} == {
        "assumption.limits",
        "assumption.response",
        "characterisation.programme",
        "characterisation.worked_trial",
        "context.workflow",
        "design.analysis",
        "design.properties",
        "external.realism",
        "joint.structure",
        "mechanism.response",
        "parameter.recovery",
        "outcome.longitudinal",
        "outcome.ordinal",
        "outcome.survival",
        "negative.control",
    }
    assert len(bundle.supporting_data) >= 30
    assert all(
        artifact.relative_path.startswith("data/")
        for artifact in bundle.supporting_data
    )
    assert any(
        artifact.relative_path.endswith("frailty_operating_characteristics.csv")
        for artifact in bundle.supporting_data
    )
    assert bundle.methods.relative_path == "METHODS.md"
    assert bundle.report.relative_path == "REPORT.md"
    assert {chapter.relative_path for chapter in bundle.chapters} == {
        "reports/mechanism-and-effect-recovery.md",
        "reports/participant-linkage-preservation.md",
        "reports/source-trial-anchoring.md",
        "reports/trial-design-and-assumption-response.md",
        "reports/trialeval-release-contents.md",
    }
    assert bundle.results.relative_path == "RESULTS.csv"
    assert bundle.sources.relative_path == "SOURCES.md"
    report = (validation_root / bundle.report.relative_path).read_text(encoding="utf-8")
    chapters = {
        chapter.relative_path: (validation_root / chapter.relative_path).read_text(
            encoding="utf-8"
        )
        for chapter in bundle.chapters
    }
    report_set = "\n".join((report, *chapters.values()))
    assert re.search(r"(?<!\])\(\.\./", report_set) is None
    for figure in bundle.figures:
        svg = next(
            artifact.relative_path
            for artifact in figure.artifacts
            if artifact.media_type == "image/svg+xml"
        )
        assert report_set.count(f"](../{svg})") == 1
        assert all(
            artifact.media_type != "text/markdown" for artifact in figure.artifacts
        )
        assert figure.scientific_question in report_set
    for relative_path in chapters:
        assert f"]({relative_path})" in report
    assert "](METHODS.md)" in report
    assert "](SOURCES.md)" in report


def test_numerical_inventory_reconciles_with_figure_data() -> None:
    """Headline outcome and control results equal the plotted source data."""

    package_root = Path(__file__).resolve().parents[2]
    result_root = package_root / "validation_results"
    results = {
        (row.evidence_layer, row.metric): row
        for row in read_validation_results(result_root / "RESULTS.csv")
    }

    survival = _csv_rows(result_root / "figures/outcome_survival.csv")
    survival_error = sum(
        abs(float(row["source_survival"]) - float(row["mean_survival"]))
        for row in survival
    ) / len(survival)
    assert survival_error == pytest.approx(
        results[("source-scale survival", "Kaplan-Meier mean absolute error")].estimate
    )

    ordinal = _csv_rows(result_root / "figures/outcome_ordinal.csv")
    ordinal_error = sum(
        abs(
            float(row["category_probability_observed"])
            - float(row["category_probability_simulated_mean"])
        )
        for row in ordinal
    ) / len(ordinal)
    assert ordinal_error == pytest.approx(
        results[
            ("source-scale ordinal", "arm-by-category mean absolute error")
        ].estimate
    )

    longitudinal = _csv_rows(result_root / "figures/outcome_longitudinal.csv")
    longitudinal_error = sum(
        abs(float(row["source_mean"]) - float(row["mean_median"]))
        for row in longitudinal
    ) / len(longitudinal)
    assert longitudinal_error == pytest.approx(
        results[
            (
                "source-scale longitudinal",
                "six-minute-walk source-to-repeated-trial-median mean absolute difference",
            )
        ].estimate
    )
    means_in_range = sum(
        float(row["mean_interval_95_low"])
        <= float(row["source_mean"])
        <= float(row["mean_interval_95_high"])
        for row in longitudinal
    )
    assert means_in_range == 6

    for row in _csv_rows(result_root / "figures/negative_control.csv"):
        metrics = {
            "PATENCY survival curve": {
                "intact": "PATENCY intact survival-curve error",
                "broken": "PATENCY broken-linkage survival-curve error",
                "difference": "PATENCY paired survival-curve error increase",
            },
            "HeadSOAR category probabilities": {
                "intact": "HeadSOAR intact category-probability error",
                "broken": "HeadSOAR broken-linkage category-probability error",
                "difference": "HeadSOAR paired category-probability error increase",
            },
        }[row["outcome"]]
        for prefix, metric in metrics.items():
            result = results[("structural control", metric)]
            estimate_field = (
                "difference" if prefix == "difference" else f"{prefix}_error"
            )
            assert float(row[estimate_field]) == result.estimate
            assert float(row[f"{prefix}_ci_low"]) == result.ci_low
            assert float(row[f"{prefix}_ci_high"]) == result.ci_high

    profiles = _csv_rows(result_root / "data/programme_profiles.csv")
    assert len(profiles) == int(
        results[("release characterisation", "independent trial count")].estimate
    )
    assert sum(int(row["participant_count"]) for row in profiles) == int(
        results[("release characterisation", "synthetic participant count")].estimate
    )


def test_report_figures_regenerate_exactly_from_packaged_data(tmp_path: Path) -> None:
    """Released CSV tables reproduce every checked-in vector figure."""

    package_root = Path(__file__).resolve().parents[2]
    result_root = package_root / "validation_results"
    rendered = render_validation_report_figures(
        validation_root=result_root,
        output_dir=tmp_path,
    )
    expected_names = {
        f"{stem}.{suffix}"
        for stem in (
            "trial_programme",
            "assumption_response",
            "assumption_limits",
            "context_workflow",
            "design_consequences",
            "trial_designs",
            "generator_realism",
            "joint_structure",
            "mechanism_response",
            "negative_control",
            "outcome_longitudinal",
            "outcome_ordinal",
            "outcome_survival",
            "parameter_recovery",
            "worked_trial",
        )
        for suffix in ("pdf", "svg")
    }
    assert {path.name for path in rendered} == expected_names
    for path in rendered:
        assert path.read_bytes() == (result_root / "figures" / path.name).read_bytes()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))
