"""Load one complete immutable TrialEval ablation run without grading it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trialagentbench_harness.contracts.core.runs import (
    ProviderTelemetrySummaryV1,
    RunCoverageV1,
    TrialEvalAblationItemResultV1,
    TrialEvalAblationRunConfigV1,
)
from trialagentbench_harness.contracts.experiments import TrialEvalAblationScheduleV1
from trialagentbench_harness.io import read_json_model, sha256_file


@dataclass(frozen=True)
class CompletedTrialEvalAblationRun:
    """Validated run metadata, complete assignment results, and source hashes."""

    source: Path
    schedule: TrialEvalAblationScheduleV1
    run_config: TrialEvalAblationRunConfigV1
    coverage: RunCoverageV1
    telemetry: ProviderTelemetrySummaryV1
    results: tuple[TrialEvalAblationItemResultV1, ...]
    source_hashes: tuple[tuple[Path, str], ...]

    def result_by_assignment(self) -> dict[str, TrialEvalAblationItemResultV1]:
        """Return the already-validated unique assignment index."""

        return {row.assignment.assignment_id: row for row in self.results}

    def assert_unchanged(self) -> None:
        """Fail if any immutable source file changed after loading."""

        for path, expected in self.source_hashes:
            _regular_file(path)
            if sha256_file(path) != expected:
                raise ValueError(f"TrialEval ablation source drifted after loading: {path}")


def _regular_file(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"Source path must not be a symbolic link: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_completed_trialeval_ablation_run(run_dir: Path) -> CompletedTrialEvalAblationRun:
    """Load and validate one complete source-frozen TrialEval ablation run."""

    candidate = Path(run_dir)
    if candidate.is_symlink():
        raise ValueError(f"Ablation run path must not be a symbolic link: {candidate}")
    source = candidate.resolve(strict=True)
    if not source.is_dir():
        raise ValueError(f"Ablation run path must be a regular directory: {source}")

    schedule_path = _regular_file(source / "schedule.json")
    config_path = _regular_file(source / "run_config.json")
    coverage_path = _regular_file(source / "coverage.json")
    telemetry_path = _regular_file(source / "provider_telemetry_summary.json")
    schedule = read_json_model(TrialEvalAblationScheduleV1, schedule_path)
    run_config = read_json_model(TrialEvalAblationRunConfigV1, config_path)
    coverage = read_json_model(RunCoverageV1, coverage_path)
    telemetry = read_json_model(ProviderTelemetrySummaryV1, telemetry_path)
    assignment_ids = tuple(row.assignment_id for row in schedule.assignments)
    if run_config.schedule_checksum != schedule.checksum or run_config.n_assignments != len(assignment_ids):
        raise ValueError("Ablation run configuration does not match its complete schedule.")
    if (
        coverage.run_identity_sha256 != run_config.run_identity_sha256
        or coverage.schedule_sha256 != schedule.checksum
        or coverage.unit_ids != assignment_ids
        or coverage.completed_unit_ids != assignment_ids
    ):
        raise ValueError("Ablation run has an incomplete or inconsistent denominator.")
    if (
        telemetry.run_identity_sha256 != coverage.run_identity_sha256
        or telemetry.schedule_sha256 != coverage.schedule_sha256
        or telemetry.unit_ids != coverage.unit_ids
        or telemetry.completed_unit_ids != coverage.completed_unit_ids
    ):
        raise ValueError("Ablation telemetry summary does not match the completed denominator.")

    assignment_root = source / "assignments"
    if assignment_root.is_symlink() or not assignment_root.is_dir():
        raise ValueError(f"Ablation assignments path must be a regular directory: {assignment_root}")
    loaded: list[tuple[Path, TrialEvalAblationItemResultV1]] = []
    for path in sorted(assignment_root.glob("*.json")):
        _regular_file(path)
        loaded.append((path, read_json_model(TrialEvalAblationItemResultV1, path)))
    embedded_ids = tuple(result.assignment.assignment_id for _, result in loaded)
    if len(embedded_ids) != len(set(embedded_ids)):
        raise ValueError("Ablation run contains duplicate assignment IDs.")
    expected = {row.assignment_id: row for row in schedule.assignments}
    observed = {result.assignment.assignment_id: result for _, result in loaded}
    missing = sorted(set(expected).difference(observed))
    extra = sorted(set(observed).difference(expected))
    if missing or extra:
        raise ValueError(f"Ablation run denominator mismatch: missing={missing}, extra={extra}")
    for path, result in loaded:
        assignment_id = result.assignment.assignment_id
        if path.stem != assignment_id:
            raise ValueError(f"Ablation assignment path does not match its embedded ID: {path}")
        if result.assignment != expected[assignment_id]:
            raise ValueError(f"Ablation result assignment drift: {assignment_id!r}.")
        if result.run_config != run_config:
            raise ValueError(f"Ablation result run configuration drift: {assignment_id!r}.")

    authority = (schedule_path, config_path, coverage_path, telemetry_path)
    source_hashes = tuple((path, sha256_file(path)) for path in (*authority, *(path for path, _ in loaded)))
    run = CompletedTrialEvalAblationRun(
        source=source,
        schedule=schedule,
        run_config=run_config,
        coverage=coverage,
        telemetry=telemetry,
        results=tuple(result for _, result in loaded),
        source_hashes=source_hashes,
    )
    run.assert_unchanged()
    return run


__all__ = ["CompletedTrialEvalAblationRun", "load_completed_trialeval_ablation_run"]
