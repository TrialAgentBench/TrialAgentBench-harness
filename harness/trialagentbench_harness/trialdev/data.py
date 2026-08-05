"""Item discovery, program enumeration, and working-directory staging.

Reads the suite manifests shipped in the bundle root, groups items into
programs (one program per ``(scenario_id, objective_id)``), and stages
per-program working directories that contain only the agent-visible
``public/`` surface, never evaluator-reserved generation or grading records.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from trialagentbench_harness.contracts.experiments import ProcedureAssistanceV1
from trialagentbench_harness.contracts.release.trialdev_runtime_surface import (
    classify_trialdev_public_member,
    required_trialdev_public_members,
)
from trialagentbench_harness.contracts.trialdev.trialdev_run_grid import (
    TrialDevProgramGridEntryV1,
    TrialDevProgramGridV1,
)
from trialagentbench_harness.trialdev.schema import (
    BenchmarkItem,
    PhaseId,
    Program,
)

SCENARIO_DIR_PREFIX = "scenario_"
PUBLIC_DIRNAME = "public"
HIDDEN_DIRNAME = "hidden"
GRADER_DIRNAME = "grader"
SUITE_MANIFEST_NAME = "benchmark_suite_manifest.json"


def _item_from_record(record: dict) -> BenchmarkItem:
    from trialagentbench_harness.trialdev.share.models import TrialDevelopmentBenchmarkItemV1

    item = TrialDevelopmentBenchmarkItemV1.model_validate(record)
    return BenchmarkItem(
        item_id=item.item_id,
        scenario_id=item.scenario_id,
        phase_id=item.phase_id,
        objective_id=item.objective_id,
        endpoint_id=item.endpoint_id,
        task_definition_id=item.task_definition_id,
        allowed_endpoint_ids=item.allowed_endpoint_ids,
        allowed_follow_up_days=item.allowed_follow_up_days,
        allowed_enrollment_window_days=item.allowed_enrollment_window_days,
        allowed_site_count_budgets=item.allowed_site_count_budgets,
    )


def discover_items(bundle_root: Path) -> list[BenchmarkItem]:
    """Load the complete checksummed release inventory."""
    from trialagentbench_harness.trialdev.share.models import TrialDevelopmentBenchmarkSuiteManifestV1

    manifest_path = bundle_root / SUITE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"TrialDev release is missing {SUITE_MANIFEST_NAME}.")
    manifest = TrialDevelopmentBenchmarkSuiteManifestV1.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    return [_item_from_record(item.model_dump(mode="python")) for item in manifest.items]


_PROGRAM_OBJECTIVES = (
    "benefit_risk",
    "pure_efficacy",
    "cost_effective_best",
    "net_clinical_value_under_budget",
)


def discover_programs(
    bundle_root: Path,
) -> list[Program]:
    """Enumerate one program per ``(scenario_id, objective_id)``.

    Each program's ``items_by_phase`` is keyed by phase_id and contains the
    item (if any) corresponding to that scenario+objective+phase, derived
    from the per-scenario ``phase_module_catalog.json``.
    """
    items = discover_items(bundle_root)

    grouped: dict[tuple[str, str], dict[str, list[BenchmarkItem]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        grouped[(item.scenario_id, item.objective_id)][item.phase_id].append(item)

    # Ensure every (scenario × objective) pair is enumerated even if some
    # phase_module_catalogs omit the objective entirely.
    scenario_ids = sorted(
        d.name.removeprefix(SCENARIO_DIR_PREFIX) for d in bundle_root.glob(f"{SCENARIO_DIR_PREFIX}*") if d.is_dir()
    )
    for scenario_id in scenario_ids:
        for objective_id in _PROGRAM_OBJECTIVES:
            grouped.setdefault((scenario_id, objective_id), defaultdict(list))

    programs: list[Program] = []
    for (scenario_id, objective_id), by_phase in sorted(grouped.items()):
        items_by_phase: dict[PhaseId, tuple[BenchmarkItem, ...]] = {}
        for phase_id in ("observational_review", "phase1", "phase2", "phase3"):
            items_by_phase[phase_id] = tuple(by_phase.get(phase_id, ()))  # type: ignore[index]
        programs.append(
            Program(
                program_id=f"{scenario_id}__{objective_id}",
                scenario_id=scenario_id,
                objective_id=objective_id,
                items_by_phase=items_by_phase,
            )
        )
    return programs


def _semantic_scenario_id(scenario_dir: Path) -> str:
    """Return the semantic scenario id declared by the evaluation-target register."""

    register_path = scenario_dir / GRADER_DIRNAME / "evaluation_target_register.jsonl"
    if not register_path.is_file():
        return scenario_dir.name.removeprefix(SCENARIO_DIR_PREFIX)
    for line in register_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"evaluation_target_register.jsonl row must be an object: {register_path}")
        raw = payload.get("scenario_id")
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"evaluation_target_register.jsonl row is missing scenario_id: {register_path}")
        return raw
    raise ValueError(f"evaluation_target_register.jsonl contains no rows: {register_path}")


def discover_release_program_grid(bundle_root: Path) -> TrialDevProgramGridV1:
    """Return the canonical release program grid for TrialDevBench."""

    entries: list[TrialDevProgramGridEntryV1] = []
    for scenario_dir in sorted(bundle_root.glob(f"{SCENARIO_DIR_PREFIX}*")):
        if not scenario_dir.is_dir():
            continue
        scenario_key = scenario_dir.name.removeprefix(SCENARIO_DIR_PREFIX)
        semantic_id = _semantic_scenario_id(scenario_dir)
        for objective_id in _PROGRAM_OBJECTIVES:
            entries.append(
                TrialDevProgramGridEntryV1(
                    program_id=f"{scenario_key}__{objective_id}",
                    scenario_key=scenario_key,
                    scenario_semantic_id=semantic_id,
                    objective_id=objective_id,
                )
            )
    return TrialDevProgramGridV1(
        release_id=Path(bundle_root).name or "TrialDevBench",
        programs=tuple(entries),
    )


def require_complete_trialdev_grid(grid: TrialDevProgramGridV1) -> None:
    """Require the manifest-declared scenario-by-objective denominator."""

    scenario_keys = {program.scenario_key for program in grid.programs}
    objective_ids = {program.objective_id for program in grid.programs}
    if not scenario_keys:
        raise ValueError("TrialDev release grid requires at least one scenario.")
    if objective_ids != set(_PROGRAM_OBJECTIVES):
        raise ValueError("TrialDev release grid must contain every declared objective.")
    expected: set[tuple[str, str]] = {
        (scenario_key, objective_id) for scenario_key in scenario_keys for objective_id in _PROGRAM_OBJECTIVES
    }
    observed: set[tuple[str, str]] = {(program.scenario_key, program.objective_id) for program in grid.programs}
    if observed != expected:
        raise ValueError(
            "TrialDev release grid must be the complete scenario-by-objective product. "
            f"missing={sorted(expected - observed)!r}, extra={sorted(observed - expected)!r}."
        )


def scenario_root(bundle_root: Path, scenario_id: str) -> Path:
    """Path to one scenario directory inside the bundle (read-only source)."""
    root = bundle_root / f"{SCENARIO_DIR_PREFIX}{scenario_id}"
    if not root.is_dir():
        raise FileNotFoundError(
            f"Scenario directory not found: {root}. Check that --bundle points at an extracted TrialDevBench release."
        )
    return root


def stage_working_dir(
    bundle_root: Path,
    scenario_id: str,
    dest_root: Path,
    *,
    procedure_assistance: ProcedureAssistanceV1,
    overwrite: bool = False,
) -> Path:
    """Stage a per-program agent working directory with the public surface flat.

    Copies every role-classified file under ``<bundle>/scenario_<id>/public/`` directly into
    ``dest_root/`` so the agent can read files with simple paths like
    ``observational_extract.parquet`` or ``eval_contract.json``. The
    ``hidden/`` and ``grader/`` surfaces are never copied.

    Returns ``dest_root`` (i.e. the agent's working directory itself).
    """
    src = scenario_root(bundle_root, scenario_id)
    src_public = src / PUBLIC_DIRNAME
    if not src_public.is_dir():
        raise FileNotFoundError(f"Public surface missing for scenario: {src_public}")

    if dest_root.exists():
        if not overwrite:
            raise FileExistsError(f"Working directory already exists: {dest_root}. Pass overwrite=True to replace.")
        for entry in list(dest_root.iterdir()):
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    dest_root.mkdir(parents=True, exist_ok=True)
    source_files = tuple(sorted(src_public.iterdir()))
    source_names = {path.name for path in source_files if path.is_file()}
    missing = sorted(required_trialdev_public_members() - source_names)
    if missing:
        raise FileNotFoundError(f"TrialDev public scenario is missing required members: {missing}")
    for src_file in source_files:
        if not src_file.is_file() or src_file.is_symlink():
            raise ValueError(f"TrialDev public scenario members must be regular files: {src_file}")
        classify_trialdev_public_member(src_file.name)
        shutil.copy2(src_file, dest_root / src_file.name)

    # Defensive: hidden / grader names should never end up in the staged dir.
    forbidden_dirs = {HIDDEN_DIRNAME, GRADER_DIRNAME}
    for entry in dest_root.iterdir():
        if entry.name in forbidden_dirs:
            raise RuntimeError(f"Sandbox violation: {entry} should never appear in a staged working dir.")

    return dest_root


def coverage_grid(programs: Iterable[Program]) -> dict:
    """Build a coverage record for scenario/objective/phase/endpoint cells.

    The returned JSON-serializable dictionary is saved next to the run output
    for traceability.
    """
    program_rows = tuple(programs)
    rows: list[dict] = []
    for program in program_rows:
        for phase_id, items in program.items_by_phase.items():
            for item in items:
                rows.append(
                    {
                        "program_id": program.program_id,
                        "scenario_id": program.scenario_id,
                        "objective_id": program.objective_id,
                        "phase_id": phase_id,
                        "endpoint_id": item.endpoint_id,
                        "item_id": item.item_id,
                        "task_definition_id": item.task_definition_id,
                    }
                )
    return {
        "counts": {"total_items_present": len(rows)},
        "items": rows,
        "n_programs": len(program_rows),
    }


def write_coverage_report(programs: list[Program], out_path: Path) -> None:
    """Write coverage_report.json next to a run's outputs."""
    from trialagentbench_harness.contracts.core.coverage import (
        TrialDevCoverageCountsV1,
        TrialDevCoverageItemV1,
        TrialDevCoverageProgramV1,
        TrialDevCoverageReportV1,
    )
    from trialagentbench_harness.io.json import write_json_model

    grid = coverage_grid(programs)
    counts = grid.get("counts") or {}
    items = grid.get("items") or []

    report = TrialDevCoverageReportV1(
        schema_id="trialagentbench_trialdev_coverage_report_v1",
        schema_version=1,
        counts=TrialDevCoverageCountsV1.model_validate(counts),
        items=[TrialDevCoverageItemV1.model_validate(r) for r in items],
        n_programs=len(programs),
        programs=[
            TrialDevCoverageProgramV1(program_id=p.program_id, scenario_id=p.scenario_id, objective_id=p.objective_id)
            for p in programs
        ],
    )
    write_json_model(out_path, report)


def write_coverage_report_from_program_grid(grid: TrialDevProgramGridV1, out_path: Path) -> None:
    """Write a coverage report using a release program grid as denominator."""

    from trialagentbench_harness.contracts.core.coverage import (
        TrialDevCoverageCountsV1,
        TrialDevCoverageProgramV1,
        TrialDevCoverageReportV1,
    )
    from trialagentbench_harness.io.json import write_json_model

    report = TrialDevCoverageReportV1(
        schema_id="trialagentbench_trialdev_coverage_report_v1",
        schema_version=1,
        counts=TrialDevCoverageCountsV1(total_items_present=0),
        items=[],
        n_programs=len(grid.programs),
        programs=[
            TrialDevCoverageProgramV1(
                program_id=program.program_id,
                scenario_id=program.scenario_key,
                objective_id=program.objective_id,
            )
            for program in grid.programs
        ],
    )
    write_json_model(out_path, report)


__all__ = [
    "discover_items",
    "discover_programs",
    "discover_release_program_grid",
    "require_complete_trialdev_grid",
    "scenario_root",
    "stage_working_dir",
    "coverage_grid",
    "write_coverage_report",
    "write_coverage_report_from_program_grid",
]
