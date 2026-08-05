"""Guardrail: offline harness paths must not import provider implementations.

This test enforces the "hexagonal isolation" rule from the hardening plan:
offline grading and analysis should be importable without pulling in `openai`
or other provider libraries.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType


class _BlockNetworkProviderImports(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        blocked = ("openai", "httpx", "requests")
        if fullname in blocked or any(fullname.startswith(f"{b}.") for b in blocked):
            raise AssertionError(f"Offline import path attempted to import provider/network module: {fullname}")
        return None


def test_offline_modules_do_not_import_openai() -> None:
    blocker = _BlockNetworkProviderImports()

    # Ensure the test is meaningful even if another test imported openai first.
    removed_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "openai"
        or name.startswith("openai.")
        or name == "httpx"
        or name.startswith("httpx.")
        or name == "requests"
        or name.startswith("requests.")
    }
    for name in removed_modules:
        del sys.modules[name]

    sys.meta_path.insert(0, blocker)
    try:
        # Package-level offline core
        importlib.import_module("trialagentbench_harness")
        importlib.import_module("trialagentbench_harness.io")
        importlib.import_module("trialagentbench_harness.contracts")
        importlib.import_module("trialagentbench_harness.analysis")

        # Offline entrypoints (should not import adapters/providers at import time)
        importlib.import_module("trialagentbench_harness.tools.grade.grade_trialeval")
        importlib.import_module("trialagentbench_harness.tools.grade.grade_trialdev")
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(removed_modules)
