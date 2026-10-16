from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """Expose shared test fixtures as importable test-only modules."""

    tests_root = Path(__file__).resolve().parent
    if str(tests_root) not in sys.path:
        sys.path.insert(0, str(tests_root))
