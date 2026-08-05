"""Verify that core CLI modules do not import optional dependencies."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

from trialagentbench_harness import cli

_OPTIONAL_ROOTS = frozenset({"dotenv", "lifelines", "matplotlib", "openai", "plotnine"})


class _BlockOptionalImports(MetaPathFinder):
    """Reject imports from every optional dependency root."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        root = fullname.partition(".")[0]
        if root in _OPTIONAL_ROOTS:
            raise ModuleNotFoundError(f"No module named {root!r}", name=root)
        return None


def test_core_cli_modules_import_without_optional_dependencies() -> None:
    """Every core command except the declared analysis command must import cleanly."""

    modules_before = set(sys.modules)
    removed_modules: dict[str, ModuleType] = {}
    for name in tuple(sys.modules):
        if name.partition(".")[0] in _OPTIONAL_ROOTS or (
            name.startswith("trialagentbench_harness.") and name != cli.__name__
        ):
            removed_modules[name] = sys.modules.pop(name)

    blocker = _BlockOptionalImports()
    sys.meta_path.insert(0, blocker)
    try:
        commands = {command.module: command for actions in cli._COMMANDS.values() for command in actions.values()}
        for command in commands.values():
            if command.optional_dependency is not None:
                with pytest.raises(cli.MissingOptionalDependencyError, match=r"trial-agent-bench\[analysis\]"):
                    command.invoke(("--help",))
                continue
            importlib.import_module(command.module)
    finally:
        sys.meta_path.remove(blocker)
        for name in tuple(sys.modules):
            if name.startswith("trialagentbench_harness.") and name not in modules_before:
                del sys.modules[name]
        sys.modules.update(removed_modules)
        for name, module in sorted(removed_modules.items(), key=lambda item: item[0].count(".")):
            parent_name, separator, attribute = name.rpartition(".")
            if separator and parent_name in sys.modules:
                setattr(sys.modules[parent_name], attribute, module)
