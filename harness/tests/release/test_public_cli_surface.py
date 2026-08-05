from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from trialagentbench_harness import cli
from trialagentbench_harness.tools.build import build_trace_analysis_bundle

PUBLIC_CONSOLE_COMMANDS = {"trialagentbench"}


def _harness_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "trialagentbench_harness").is_dir():
            return parent
    raise RuntimeError("Could not locate TrialAgentBench harness root")


def test_source_pyproject_exposes_only_public_console_commands() -> None:
    """Installed source-package commands must stay aligned to the public harness surface."""

    pyproject = tomllib.loads((_harness_root() / "pyproject.toml").read_text(encoding="utf-8"))

    assert set(pyproject["project"]["scripts"]) == PUBLIC_CONSOLE_COMMANDS


def test_public_cli_exposes_five_workflows() -> None:
    """The installed CLI groups behavior by user workflow."""

    assert set(cli._COMMANDS) == {"run", "grade", "analyse", "verify", "export"}
    assert "trialdev-worlds" not in cli._COMMANDS["analyse"]
    assert "trialdev-ablation" not in cli._COMMANDS["analyse"]
    assert "analysis-bundle" not in cli._COMMANDS["verify"]
    assert "analysis-claims" not in cli._COMMANDS["verify"]
    assert "analysis-source-data" not in cli._COMMANDS["export"]
    assert cli._COMMANDS["export"]["results"].module == "trialagentbench_harness.tools.export_results"
    assert cli._COMMANDS["run"]["trialeval-direct-assessment"].module == (
        "trialagentbench_harness.experiments.assess_trialeval_narrative_packets"
    )


def test_public_cli_forwards_leaf_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dispatcher must preserve leaf arguments without interpreting them."""

    captured: dict[str, object] = {}

    def fake_import(module: str) -> type:
        captured["module"] = module

        class Imported:
            @staticmethod
            def main(argv: list[str]) -> int:
                captured["argv"] = argv
                return 7

        return Imported

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    assert cli.main(["run", "trialdev", "--release", "participant.zip", "--max-turns-per-step", "90"]) == 7
    assert captured == {
        "module": "trialagentbench_harness.tools.run.trialdev",
        "argv": ["--release", "participant.zip", "--max-turns-per-step", "90"],
    }


def test_public_cli_reports_missing_optional_analysis_dependency(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Analysis-only commands must fail with an actionable core-install error."""

    def missing_lifelines(_: str) -> type:
        raise ModuleNotFoundError("No module named 'lifelines'", name="lifelines")

    monkeypatch.setattr(cli.importlib, "import_module", missing_lifelines)

    with pytest.raises(SystemExit) as error:
        cli.main(["verify", "trialeval-diagnostics", "--help"])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "trial-agent-bench[analysis]" in stderr
    assert "Traceback" not in stderr


def test_public_cli_does_not_mask_missing_required_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing core or internal modules remain packaging failures."""

    def missing_internal(_: str) -> type:
        raise ModuleNotFoundError("No module named 'broken_internal'", name="broken_internal")

    monkeypatch.setattr(cli.importlib, "import_module", missing_internal)

    with pytest.raises(ModuleNotFoundError, match="broken_internal"):
        cli.main(["verify", "trialeval-diagnostics", "--help"])


def test_trace_analysis_cli_maps_to_bundle_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public trace-analysis CLI must execute the shared bundle builder."""

    captured: dict[str, object] = {}

    def fake_builder(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(build_trace_analysis_bundle, "build_trace_analysis_bundle", fake_builder)

    result = build_trace_analysis_bundle.main(
        [
            "--out-dir",
            "trace_bundle",
            "--trialeval-root",
            "runs/trialeval",
            "--trialdev-root",
            "runs/trialdev",
            "--trialdev-release-root",
            "release/trialdev",
        ]
    )

    assert result == 0
    assert captured == {
        "out_dir": Path("trace_bundle"),
        "trialeval_root": Path("runs/trialeval"),
        "trialdev_root": Path("runs/trialdev"),
        "trialdev_release_root": Path("release/trialdev"),
    }


def test_trace_analysis_console_entrypoint_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """The advertised console entrypoint delegates to the trace-analysis module."""

    captured: dict[str, object] = {}

    def fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(build_trace_analysis_bundle, "main", fake_main)

    assert cli.build_trace_analysis(["--out-dir", "trace_bundle"]) == 0
    assert captured == {"argv": ["--out-dir", "trace_bundle"]}


def test_public_cli_routes_trace_bundle_to_full_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public trace verification must use the trace-bundle validator."""
    captured: dict[str, object] = {}

    def fake_import(module: str) -> type:
        captured["module"] = module

        class Imported:
            @staticmethod
            def main(argv: list[str]) -> int:
                captured["argv"] = argv
                return 0

        return Imported

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)

    assert cli.main(["verify", "trace-bundle", "--bundle", "trace_bundle"]) == 0
    assert captured == {
        "module": "trialagentbench_harness.tools.validate.validate_trace_analysis_bundle",
        "argv": ["--bundle", "trace_bundle"],
    }
