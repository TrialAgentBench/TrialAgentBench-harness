"""Tests for bounded independent numerical workers."""

from __future__ import annotations

import os

import pytest

from trialagentbench_validation.process_pool import (
    single_threaded_numerical_process_pool,
)

_THREAD_VARIABLES = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _worker_thread_settings(_: int) -> tuple[str | None, ...]:
    return tuple(os.environ.get(name) for name in _THREAD_VARIABLES)


def test_process_pool_bounds_worker_threads_and_restores_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "7")

    with single_threaded_numerical_process_pool(workers=2) as executor:
        observed = tuple(executor.map(_worker_thread_settings, range(2)))

    assert observed == (("1",) * len(_THREAD_VARIABLES),) * 2
    assert os.environ["OMP_NUM_THREADS"] == "7"


def test_process_pool_rejects_nonpositive_worker_count() -> None:
    with pytest.raises(ValueError, match="workers must be positive"):
        with single_threaded_numerical_process_pool(workers=0):
            pass
