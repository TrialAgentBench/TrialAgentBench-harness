"""Bound independent tasks and drain interrupted native runs without replacing primaries."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

from trialagentbench_verifier.execution_contracts import EXECUTOR_SLOTS_PER_TASK
from trialagentbench_verifier.executor import ExecutorUnavailableError, executor_session_capacity

from trialagentbench.util.campaign_control import CampaignCapacityStop, CampaignRequestControl
from trialagentbench.util.reported_cost import RunStopSignal

Task = TypeVar("Task")
Result = TypeVar("Result")


def run_bounded_tasks(
    tasks: Iterable[Task],
    execute: Callable[[Task], Result],
    *,
    workers: int,
    consume: Callable[[Result], None],
    stop: RunStopSignal,
    request_control: CampaignRequestControl | None = None,
) -> None:
    """Run tasks at one worker bound; any interruption cancels queued work and stops future requests.

    Executor containers are capped at EXECUTOR_SLOTS_PER_TASK per worker unless an enclosing campaign
    scope already bounds them. A task that finds no free executor is left pending (its own trace
    records the failure) while the other tasks finish; the run then pauses, without stopping other
    campaign routes, so a resume reruns it.
    """
    if workers < 1:
        raise ValueError("Independent task workers must be positive")

    def process(task: Task) -> Result:
        stop.before_request()
        return execute(task)

    unavailable: list[ExecutorUnavailableError] = []
    with executor_session_capacity(EXECUTOR_SLOTS_PER_TASK * workers):
        pool = ThreadPoolExecutor(max_workers=workers)
        completed = False
        campaign_wide = True
        try:
            futures = [pool.submit(process, task) for task in tasks]
            for future in as_completed(futures):
                try:
                    result = future.result()
                except ExecutorUnavailableError as error:
                    unavailable.append(error)
                    print(f"Task left pending: {error}", flush=True)
                    continue
                consume(result)
            completed = True
        except CampaignCapacityStop as error:
            campaign_wide = error.campaign_wide
            raise
        finally:
            if not completed:
                stop.request()
                if request_control is not None and campaign_wide:
                    request_control.set_cancelled(True)
            pool.shutdown(wait=True, cancel_futures=True)
    if unavailable:
        raise CampaignCapacityStop(
            f"{len(unavailable)} task(s) found no free executor and remain pending; resume the run to rerun them",
            campaign_wide=False,
        )
