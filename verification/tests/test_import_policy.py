"""Package-boundary tests for independent benchmark validation."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def test_validation_imports_only_declared_standalone_dependencies() -> None:
    root = Path(__file__).parents[1] / "src" / "trialagentbench_validation"
    harness_control_modules = {
        Path("trialdev/portfolio_difficulty.py"),
        Path("trialdev/portfolio_grader_controls.py"),
        Path("trialdev/portfolio_observational_replay.py"),
        Path("trialdev/portfolio_routes.py"),
    }
    allowed = set(sys.stdlib_module_names) | {
        "lifelines",
        "matplotlib",
        "numpy",
        "openpyxl",
        "pandas",
        "plotnine",
        "pyarrow",
        "pydantic",
        "scipy",
        "statsmodels",
        "trialagentbench_validation",
        "typing_extensions",
    }
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = (node.module,)
            for name in names:
                top_level = name.split(".", maxsplit=1)[0]
                relative_path = path.relative_to(root)
                if (
                    top_level == "trialagentbench_harness"
                    and relative_path in harness_control_modules
                ):
                    continue
                if top_level not in allowed:
                    violations.append(
                        f"{path.relative_to(root)}:{getattr(node, 'lineno', 0)}:{name}"
                    )
    assert not violations, "\n".join(violations)


def test_harness_controls_are_an_explicit_optional_surface() -> None:
    """Keep public-grader controls separate from independent replay dependencies."""

    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert "grader-controls = [" in pyproject
    assert '"trial-agent-bench>=0.1.0,<0.2"' in pyproject


def test_simulation_validation_has_one_index_and_bounded_chapters() -> None:
    """The public result set exposes one index and the declared chapters."""

    package_root = Path(__file__).resolve().parents[1]
    result_root = package_root / "validation_results"
    assert not tuple((package_root / "docs").glob("*.md"))
    assert not tuple((package_root / "reports").glob("*.md"))
    assert {path.name for path in result_root.glob("*.md")} == {
        "METHODS.md",
        "REPORT.md",
        "SOURCES.md",
    }
    assert {path.name for path in (result_root / "reports").glob("*.md")} == {
        "mechanism-and-effect-recovery.md",
        "participant-linkage-preservation.md",
        "source-trial-anchoring.md",
        "trial-design-and-assumption-response.md",
        "trialeval-release-contents.md",
    }

    report = (result_root / "REPORT.md").read_text(encoding="utf-8")
    methods = (result_root / "METHODS.md").read_text(encoding="utf-8")
    sources = (result_root / "SOURCES.md").read_text(encoding="utf-8")
    assert (result_root / "RESULTS.csv").is_file()
    assert "](METHODS.md)" in report
    assert "](RESULTS.csv)" in report
    assert "](SOURCES.md)" in report
    assert "](REPORT.md)" in methods
    assert report.count("](reports/") == 5
    assert not tuple((result_root / "figures").glob("*.md"))
    assert sources.startswith("# Sources")
