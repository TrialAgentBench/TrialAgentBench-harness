"""Protect the standalone harness dependency boundary."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def test_harness_imports_only_declared_standalone_dependencies() -> None:
    root = Path(__file__).parents[2] / "trialagentbench_harness"
    allowed = set(sys.stdlib_module_names) | {
        "dotenv",
        "lifelines",
        "matplotlib",
        "numpy",
        "openai",
        "pandas",
        "plotnine",
        "pyarrow",
        "pydantic",
        "scipy",
        "statsmodels",
        "trialagentbench_harness",
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
                if name.split(".", maxsplit=1)[0] not in allowed:
                    violations.append(f"{path.relative_to(root)}:{getattr(node, 'lineno', 0)}:{name}")
    assert not violations, "\n".join(violations)
