from __future__ import annotations

import pytest

from trialagentbench_harness.concurrency import DaemonThreadPool


def test_daemon_thread_pool_records_fatal_and_rejects_future_submissions() -> None:
    pool = DaemonThreadPool(max_workers=1, thread_name_prefix="test-pool")

    def boom() -> None:
        raise ValueError("boom")

    fut = pool.submit(boom)
    with pytest.raises(ValueError, match="boom"):
        fut.result(timeout=5.0)

    with pytest.raises(RuntimeError, match="fatal state"):
        pool.submit(lambda: 1)

    pool.close()
