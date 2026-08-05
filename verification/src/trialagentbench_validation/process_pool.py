"""Bounded process execution for independent numerical verification."""

from __future__ import annotations

import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from multiprocessing import get_context

_NUMERICAL_THREAD_VARIABLES = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@contextmanager
def single_threaded_numerical_process_pool(
    *,
    workers: int,
) -> Iterator[ProcessPoolExecutor]:
    """Yield spawned workers with one numerical-kernel thread per process.

    Parameters
    ----------
    workers
        Exact positive process count.

    Yields
    ------
    concurrent.futures.ProcessPoolExecutor
        A process pool whose workers import numerical libraries under bounded
        native-thread settings.

    Raises
    ------
    ValueError
        If ``workers`` is not positive.
    """

    if workers < 1:
        raise ValueError("workers must be positive")
    previous = {name: os.environ.get(name) for name in _NUMERICAL_THREAD_VARIABLES}
    try:
        for name in _NUMERICAL_THREAD_VARIABLES:
            os.environ[name] = "1"
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
        ) as executor:
            yield executor
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
