"""Smoke tests shipped with the public TrialAgentBench harness."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path


def test_public_runtime_modules_import() -> None:
    """Offline modules and live runner adapters import without credentials."""
    for module_name in (
        "trialagentbench_harness.cli",
        "trialagentbench_harness.contracts.trace.observable",
        "trialagentbench_harness.tools.build.build_trace_analysis_bundle",
        "trialagentbench_harness.tools.run.trialeval",
        "trialagentbench_harness.tools.run.trialdev",
        "trialagentbench_harness.tools.grade.grade_run",
        "trialagentbench_harness.tools.grade.grade_trialeval",
        "trialagentbench_harness.tools.grade.grade_trialdev",
        "trialagentbench_harness.tools.validate.validate_clean_room_workflow",
        "trialagentbench_harness.grading.grader",
        "trialagentbench_harness.grading.reporting",
        "trialagentbench_harness.trialdev.action_trace",
        "trialagentbench_harness.trialdev.scoring",
        "trialagentbench_harness.trialeval.action_trace",
        "trialagentbench_harness.trialeval.grade_submission",
    ):
        importlib.import_module(module_name)


def test_public_entrypoint_modules_exist() -> None:
    """Every advertised command resolves to source included in the package."""
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    missing: list[str] = []
    for entrypoint in pyproject["project"]["scripts"].values():
        module_name = str(entrypoint).split(":", maxsplit=1)[0]
        module_path = root / Path(*module_name.split("."))
        if not module_path.with_suffix(".py").is_file() and not (module_path / "__init__.py").is_file():
            missing.append(module_name)
    assert missing == []
