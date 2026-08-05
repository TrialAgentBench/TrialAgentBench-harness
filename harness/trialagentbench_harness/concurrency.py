"""Concurrency utilities for the harness."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _WorkItem:
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    fut: Future


class DaemonThreadPool:
    """A minimal daemon-thread pool.

    This is used to ensure a parent process can exit cleanly even if a worker
    thread is stuck in uninterruptible I/O. Futures that haven't completed can
    be treated as failures by the caller.
    """

    def __init__(self, *, max_workers: int, thread_name_prefix: str = "daemon-pool"):
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._q: queue.Queue[_WorkItem | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._closed = False
        self._fatal: BaseException | None = None
        self._fatal_lock = threading.Lock()
        for i in range(max_workers):
            t = threading.Thread(
                target=self._worker,
                name=f"{thread_name_prefix}-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        if self._closed:
            raise RuntimeError("pool is closed")
        with self._fatal_lock:
            if self._fatal is not None:
                raise RuntimeError("pool is in fatal state; prior work item raised") from self._fatal
        fut: Future = Future()
        self._q.put(_WorkItem(fn=fn, args=args, kwargs=kwargs, fut=fut))
        return fut

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._threads:
            self._q.put(None)

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            if item.fut.cancelled():
                continue
            # Fail-closed worker: if a work item raises, record the exception,
            # mark the pool as fatal, and terminate the worker thread. This
            # prevents subsequent tasks from running in a corrupted state and
            # ensures failures cannot be "missed" by callers.
            try:
                res = item.fn(*item.args, **item.kwargs)
            except BaseException as exc:  # noqa: BLE001
                item.fut.set_exception(exc)
                with self._fatal_lock:
                    if self._fatal is None:
                        self._fatal = exc
                # Fail-closed termination: do not accept further submissions
                # once any work item raises. We log loudly but avoid unhandled
                # thread exceptions (pytest treats these as warnings).
                logger.exception("DaemonThreadPool worker exiting due to fatal exception")
                return
            item.fut.set_result(res)
