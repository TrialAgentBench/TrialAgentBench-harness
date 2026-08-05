"""Public finite state-action graph contract for TrialDev."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevActionIdV1,
    TrialDevCheckpointIdV1,
    TrialDevStreamIdV1,
    TrialDevTerminalDispositionV1,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialDevStateNodeV1(_StrictModel):
    """One decision-relevant reachable state, excluding path-specific history."""

    node_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    stream_id: TrialDevStreamIdV1
    checkpoint_id: TrialDevCheckpointIdV1
    candidate_asset_ids: tuple[str, ...] = Field(min_length=1)
    nominated_asset_id: str | None = None
    lead_asset_id: str | None = None
    reserve_asset_id: str | None = None
    active_asset_id: str | None = None
    retired_asset_ids: tuple[str, ...] = ()
    resource_budget_units: int | None = Field(default=None, ge=0)
    resource_spent_units: int = Field(default=0, ge=0)
    switch_count: int = Field(default=0, ge=0, le=1)


class TrialDevStateActionEdgeV1(_StrictModel):
    """One concrete legal action and its active or terminal destination."""

    edge_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    stream_id: TrialDevStreamIdV1
    source_node_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    action_id: TrialDevActionIdV1
    target_asset_id: str | None = None
    reserve_asset_id: str | None = None
    target_node_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    terminal_disposition: TrialDevTerminalDispositionV1

    @model_validator(mode="after")
    def validate_destination(self) -> Self:
        """Require exactly active edges to identify another graph node."""

        if (self.terminal_disposition == "active") != (self.target_node_id is not None):
            raise ValueError("Exactly active graph edges must identify a target node.")
        return self


class TrialDevStateActionGraphV1(_StrictModel):
    """Complete finite legal-action graph for both TrialDev streams."""

    schema_id: Literal["trialagentbench.trialdev_state_action_graph/v1"] = (
        "trialagentbench.trialdev_state_action_graph/v1"
    )
    source_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    nodes: tuple[TrialDevStateNodeV1, ...] = Field(min_length=1)
    edges: tuple[TrialDevStateActionEdgeV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        """Require unique, closed, two-stream graph records."""

        node_ids = tuple(item.node_id for item in self.nodes)
        edge_ids = tuple(item.edge_id for item in self.edges)
        if len(node_ids) != len(set(node_ids)) or len(edge_ids) != len(set(edge_ids)):
            raise ValueError("State-action graph nodes and edges must be unique.")
        known = set(node_ids)
        if any(
            item.source_node_id not in known or (item.target_node_id and item.target_node_id not in known)
            for item in self.edges
        ):
            raise ValueError("State-action graph edges must reference graph nodes.")
        if {item.stream_id for item in self.nodes} != {
            "single_asset_development",
            "bounded_portfolio_reallocation",
        }:
            raise ValueError("State-action graph requires both TrialDev streams.")
        return self


__all__ = ["TrialDevStateActionEdgeV1", "TrialDevStateActionGraphV1", "TrialDevStateNodeV1"]
