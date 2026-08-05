from __future__ import annotations

import builtins
import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType


@contextmanager
def _block_import(module_name: str) -> Iterator[None]:
    """Fail the test if `module_name` is imported within the context."""

    real_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == module_name or name.startswith(module_name + "."):
            raise AssertionError(f"Unexpected import during offline path: {name}")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = _guarded_import
    try:
        yield
    finally:
        builtins.__import__ = real_import


def test_offline_grade_imports_do_not_import_openai() -> None:
    """Offline modules must not import provider dependencies transitively."""
    with _block_import("openai"):
        m: ModuleType = importlib.import_module("trialagentbench_harness.tools.grade.grade_trialeval")
        assert hasattr(m, "grade_trialeval_run")
        m2: ModuleType = importlib.import_module("trialagentbench_harness.grading.reporting")
        assert hasattr(m2, "write_trialeval_grade_summary")
