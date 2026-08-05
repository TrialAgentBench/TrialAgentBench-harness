"""Tests for the portable simulation validation bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trialagentbench_validation.contracts.simulation_validation_bundle import (
    SimulationValidationBundleV1,
    ValidationArtifactV1,
    ValidationFigureV1,
    verify_simulation_validation_bundle,
)
from trialagentbench_validation.io import sha256_file


def _artifact(root: Path, name: str, media_type: str) -> ValidationArtifactV1:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name, encoding="utf-8")
    return ValidationArtifactV1(
        relative_path=name, sha256=sha256_file(path), media_type=media_type
    )


def _bundle(root: Path) -> SimulationValidationBundleV1:
    figure = ValidationFigureV1(
        figure_id="outcome.realism",
        title="Native outcome realism",
        scientific_question="Are clinical outcome processes reproduced?",
        independent_unit="trial",
        estimand="source-scale outcome process",
        comparator="source, fitted generation, and broken process",
        uncertainty="95% simulation interval across replicates",
        interpretation=("The fitted process separates from the broken control.",),
        artifacts=tuple(
            sorted(
                (
                    _artifact(root, "figures/outcome.csv", "text/csv"),
                    _artifact(root, "figures/outcome.pdf", "application/pdf"),
                    _artifact(root, "figures/outcome.svg", "image/svg+xml"),
                ),
                key=lambda artifact: artifact.relative_path,
            )
        ),
    )
    methods = _artifact(root, "METHODS.md", "text/markdown")
    report = _artifact(root, "REPORT.md", "text/markdown")
    chapters = (_artifact(root, "reports/source-trial-anchoring.md", "text/markdown"),)
    results = _artifact(root, "RESULTS.csv", "text/csv")
    sources = _artifact(root, "SOURCES.md", "text/markdown")
    supporting_data = (
        _artifact(root, "data/operating_characteristics.csv", "text/csv"),
    )
    payload = {
        "schema_id": "trialagentbench.simulation_validation_bundle/v1",
        "verifier_lock_sha256": "1" * 64,
        "figures": [figure.model_dump(mode="json")],
        "supporting_data": [
            artifact.model_dump(mode="json") for artifact in supporting_data
        ],
        "methods": methods.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
        "chapters": [artifact.model_dump(mode="json") for artifact in chapters],
        "results": results.model_dump(mode="json"),
        "sources": sources.model_dump(mode="json"),
    }
    checksum = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
    return SimulationValidationBundleV1(**payload, checksum=checksum)


def test_validation_bundle_verifies_all_files(tmp_path: Path) -> None:
    """A complete bundle verifies every content-addressed artifact."""

    bundle = _bundle(tmp_path)
    verify_simulation_validation_bundle(tmp_path, bundle)


def test_validation_bundle_rejects_tampered_figure_data(tmp_path: Path) -> None:
    """Changed figure data fail verification."""

    bundle = _bundle(tmp_path)
    (tmp_path / "figures/outcome.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_simulation_validation_bundle(tmp_path, bundle)


def test_validation_bundle_rejects_tampered_report(tmp_path: Path) -> None:
    """Changed public interpretation fails verification."""

    bundle = _bundle(tmp_path)
    (tmp_path / "REPORT.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_simulation_validation_bundle(tmp_path, bundle)


def test_validation_bundle_rejects_tampered_methods(tmp_path: Path) -> None:
    """Changed public methods fail verification."""

    bundle = _bundle(tmp_path)
    (tmp_path / "METHODS.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_simulation_validation_bundle(tmp_path, bundle)


def test_validation_bundle_rejects_tampered_chapter(tmp_path: Path) -> None:
    """Changed statistical detail fails verification."""

    bundle = _bundle(tmp_path)
    (tmp_path / "reports/source-trial-anchoring.md").write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_simulation_validation_bundle(tmp_path, bundle)


def test_validation_bundle_rejects_tampered_sources(tmp_path: Path) -> None:
    """Changed public sources fail verification."""

    bundle = _bundle(tmp_path)
    (tmp_path / "SOURCES.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_simulation_validation_bundle(tmp_path, bundle)


def test_validation_bundle_rejects_tampered_results(tmp_path: Path) -> None:
    """Changed numerical results fail verification."""

    bundle = _bundle(tmp_path)
    (tmp_path / "RESULTS.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_simulation_validation_bundle(tmp_path, bundle)


def test_validation_bundle_rejects_tampered_supporting_data(tmp_path: Path) -> None:
    """Changed cell-level evidence fails verification."""

    bundle = _bundle(tmp_path)
    (tmp_path / "data/operating_characteristics.csv").write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_simulation_validation_bundle(tmp_path, bundle)


def test_figure_rejects_a_separate_explanation_document(tmp_path: Path) -> None:
    """Figure interpretation remains part of the statistical report set."""

    with pytest.raises(ValueError, match="statistical report set"):
        ValidationFigureV1(
            figure_id="outcome.realism",
            title="Native outcome realism",
            scientific_question="Are clinical outcome processes reproduced?",
            independent_unit="trial",
            estimand="source-scale outcome process",
            comparator="source and fitted generation",
            uncertainty="95% simulation interval",
            interpretation=("The process is reproduced.",),
            artifacts=tuple(
                sorted(
                    (
                        _artifact(tmp_path, "figure.csv", "text/csv"),
                        _artifact(tmp_path, "figure.md", "text/markdown"),
                        _artifact(tmp_path, "figure.pdf", "application/pdf"),
                        _artifact(tmp_path, "figure.svg", "image/svg+xml"),
                    ),
                    key=lambda artifact: artifact.relative_path,
                )
            ),
        )
