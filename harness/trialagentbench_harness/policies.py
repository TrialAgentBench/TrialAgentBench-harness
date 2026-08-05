"""Strict policy objects controlling aggregation error tolerance."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AggregatePolicy:
    """Policy for aggregation/reporting operations over saved run trees.

    Aggregation is strict by default: any corrupt/invalid artifact fails the
    aggregation. Use explicit permissive mode only for exploratory inspection
    of partially written run trees, and ensure tolerated failures are recorded
    into an aggregate manifest.
    """

    strict: bool = True
    allow_incomplete_artifacts: bool = False
