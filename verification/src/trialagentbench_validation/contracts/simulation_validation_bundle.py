"""Portable contract for clinical-trial simulation validation results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.io import sha256_file


class ValidationArtifactV1(BaseModel):
    """One content-addressed artifact in a simulation validation bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: Literal[
        "application/json",
        "application/pdf",
        "image/png",
        "image/svg+xml",
        "text/csv",
        "text/markdown",
    ]

    @model_validator(mode="after")
    def _confined_path(self) -> ValidationArtifactV1:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "validation artifact paths must be confined relative paths"
            )
        return self


class ValidationFigureV1(BaseModel):
    """Methods and files for one simulation validation figure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    figure_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]+$")
    title: str = Field(min_length=1)
    scientific_question: str = Field(min_length=1)
    independent_unit: str = Field(min_length=1)
    estimand: str = Field(min_length=1)
    comparator: str = Field(min_length=1)
    uncertainty: str = Field(min_length=1)
    interpretation: tuple[str, ...] = Field(min_length=1)
    artifacts: tuple[ValidationArtifactV1, ...] = Field(min_length=3)

    @model_validator(mode="after")
    def _complete_and_canonical(self) -> ValidationFigureV1:
        paths = tuple(artifact.relative_path for artifact in self.artifacts)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError(
                "figure artifacts must be sorted and unique by relative path"
            )
        suffixes = {PurePosixPath(path).suffix for path in paths}
        if not {".csv", ".pdf", ".svg"}.issubset(suffixes):
            raise ValueError("each public figure requires CSV data, a PDF, and an SVG")
        if ".md" in suffixes:
            raise ValueError("figure explanations belong in the statistical report set")
        if self.interpretation != tuple(dict.fromkeys(self.interpretation)):
            raise ValueError("figure interpretations must be unique")
        return self


class ValidationResultV1(BaseModel):
    """One result in the complete public numerical inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_layer: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    estimate: float
    ci_low: float | None = None
    ci_high: float | None = None
    unit: str = Field(min_length=1)
    independent_units: int = Field(ge=1)

    @model_validator(mode="after")
    def _valid_interval(self) -> ValidationResultV1:
        values = (self.estimate, self.ci_low, self.ci_high)
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("validation result values must be finite")
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError("validation result intervals require both endpoints")
        if (
            self.ci_low is not None
            and self.ci_high is not None
            and not self.ci_low <= self.estimate <= self.ci_high
        ):
            raise ValueError("validation result estimate must lie inside its interval")
        return self


class SimulationValidationBundleV1(BaseModel):
    """Checksum-bound clinical-trial simulation validation bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.simulation_validation_bundle/v1"] = (
        "trialagentbench.simulation_validation_bundle/v1"
    )
    verifier_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    figures: tuple[ValidationFigureV1, ...] = Field(min_length=1)
    supporting_data: tuple[ValidationArtifactV1, ...] = Field(min_length=1)
    methods: ValidationArtifactV1
    report: ValidationArtifactV1
    chapters: tuple[ValidationArtifactV1, ...] = Field(min_length=1)
    results: ValidationArtifactV1
    sources: ValidationArtifactV1
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _canonical_and_checksummed(self) -> SimulationValidationBundleV1:
        figure_ids = tuple(figure.figure_id for figure in self.figures)
        if figure_ids != tuple(sorted(figure_ids)) or len(set(figure_ids)) != len(
            figure_ids
        ):
            raise ValueError(
                "validation figures must be sorted and unique by figure_id"
            )
        data_paths = tuple(artifact.relative_path for artifact in self.supporting_data)
        if data_paths != tuple(sorted(data_paths)) or len(set(data_paths)) != len(
            data_paths
        ):
            raise ValueError(
                "supporting data must be sorted and unique by relative path"
            )
        if any(not path.startswith("data/") for path in data_paths):
            raise ValueError("supporting data must be stored below data/")
        chapter_paths = tuple(artifact.relative_path for artifact in self.chapters)
        if chapter_paths != tuple(sorted(chapter_paths)) or len(
            set(chapter_paths)
        ) != len(chapter_paths):
            raise ValueError("report chapters must be sorted and unique")
        if any(
            not path.startswith("reports/") or not path.endswith(".md")
            for path in chapter_paths
        ):
            raise ValueError("report chapters must be Markdown files below reports/")
        if any(artifact.media_type != "text/markdown" for artifact in self.chapters):
            raise ValueError("report chapters must use the text/markdown media type")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        if hashlib.sha256(encoded).hexdigest() != self.checksum:
            raise ValueError("simulation validation bundle checksum mismatch")
        return self


def verify_simulation_validation_bundle(
    root: Path,
    bundle: SimulationValidationBundleV1,
) -> None:
    """Verify every file referenced by a simulation validation bundle."""

    resolved_root = root.resolve()
    artifacts = [bundle.methods, bundle.report, bundle.results, bundle.sources]
    artifacts.extend(bundle.chapters)
    artifacts.extend(bundle.supporting_data)
    artifacts.extend(
        artifact for figure in bundle.figures for artifact in figure.artifacts
    )
    paths = [artifact.relative_path for artifact in artifacts]
    if len(paths) != len(set(paths)):
        raise ValueError("validation bundle references an artifact more than once")
    for artifact in artifacts:
        path = (resolved_root / artifact.relative_path).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError(
                f"validation artifact escapes the bundle root: {artifact.relative_path!r}"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"validation artifact is missing: {artifact.relative_path!r}"
            )
        if sha256_file(path) != artifact.sha256:
            raise ValueError(
                f"validation artifact checksum mismatch: {artifact.relative_path!r}"
            )


def read_validation_results(path: Path) -> tuple[ValidationResultV1, ...]:
    """Read and validate the complete public numerical inventory."""

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_fields = tuple(ValidationResultV1.model_fields)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"validation result columns must be {expected_fields!r}")
        rows = tuple(
            ValidationResultV1.model_validate(
                {key: None if value == "" else value for key, value in row.items()}
            )
            for row in reader
        )
    identities = tuple((row.evidence_layer, row.metric) for row in rows)
    if not rows:
        raise ValueError("validation result inventory must not be empty")
    if len(identities) != len(set(identities)):
        raise ValueError(
            "validation result metrics must be unique within each evidence layer"
        )
    return rows


__all__ = [
    "SimulationValidationBundleV1",
    "ValidationArtifactV1",
    "ValidationFigureV1",
    "ValidationResultV1",
    "read_validation_results",
    "verify_simulation_validation_bundle",
]
