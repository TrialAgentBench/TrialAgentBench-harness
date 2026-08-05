"""Enumerate the complete finite legal-action graph for TrialDev."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import cast

from pydantic.types import JsonValue

from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevActionSelectionV1,
    TrialDevCheckpointOutcomeV1,
    TrialDevEvidenceReferenceV1,
    TrialDevPolicyBindingV1,
    TrialDevPortfolioProgrammeStateV1,
    TrialDevProgrammeStateV1,
    TrialDevResourceScheduleV1,
    TrialDevSingleAssetProgrammeStateV1,
    TrialDevStreamIdV1,
)
from trialagentbench_harness.contracts.trialdev.state_action_graph import (
    TrialDevStateActionEdgeV1,
    TrialDevStateActionGraphV1,
    TrialDevStateNodeV1,
)
from trialagentbench_harness.io.checksums import canonical_payload_sha256
from trialagentbench_harness.io.json import write_json_model
from trialagentbench_harness.trialdev.programme import (
    build_checkpoint_action_policy_v1,
    transition_programme_state_v1,
)

_GRAPH_SEED = 20_260_802


def _sha(payload: dict[str, JsonValue]) -> str:
    return cast(
        str,
        canonical_payload_sha256(cast(JsonValue, {key: value for key, value in payload.items() if value is not None})),
    )


def _binding(stream_id: TrialDevStreamIdV1, source_identity: str) -> TrialDevPolicyBindingV1:
    return TrialDevPolicyBindingV1(
        stream_id=stream_id,
        objective_id="benefit_risk",
        objective_policy_checksum=source_identity,
        action_policy_checksum=source_identity,
        design_menu_checksum=source_identity,
        resource_schedule=(TrialDevResourceScheduleV1() if stream_id == "bounded_portfolio_reallocation" else None),
        resource_budget_units=10 if stream_id == "bounded_portfolio_reallocation" else None,
    )


def _evidence(
    *, checkpoint_id: str, asset_ids: tuple[str, ...], branch_id: str, source_identity: str
) -> tuple[TrialDevEvidenceReferenceV1, ...]:
    return tuple(
        TrialDevEvidenceReferenceV1(
            evidence_id=f"graph-{branch_id}-{checkpoint_id}-{asset_id}",
            evidence_kind="dataset",
            checkpoint_id=checkpoint_id,
            asset_id=asset_id,
            evidence_protocol_id="state_action_graph_enumeration_v1",
            evidence_protocol_checksum=source_identity,
            source_family_id=source_identity,
            world_id="finite-state-action-graph",
            generation_seed=_GRAPH_SEED,
            relative_path=f"graph_evidence/{branch_id}/{checkpoint_id}-{asset_id}.csv",
            artifact_sha256=source_identity,
        )
        for asset_id in asset_ids
    )


def _node(state: TrialDevProgrammeStateV1) -> TrialDevStateNodeV1:
    payload: dict[str, JsonValue] = {
        "stream_id": state.stream_id,
        "checkpoint_id": state.current_checkpoint_id,
        "candidate_asset_ids": list(state.candidate_asset_ids),
        "nominated_asset_id": state.nominated_asset_id,
        "lead_asset_id": state.lead_asset_id,
        "reserve_asset_id": state.reserve_asset_id,
        "active_asset_id": state.active_asset_id,
        "retired_asset_ids": list(state.retired_asset_ids),
        "resource_budget_units": state.policy_binding.resource_budget_units,
        "resource_spent_units": state.resource_spent_units,
        "switch_count": state.switch_count,
    }
    return TrialDevStateNodeV1(node_id=_sha(payload), **payload)


def _action_variants(state: TrialDevProgrammeStateV1) -> tuple[tuple[str, str | None, str | None], ...]:
    policy = build_checkpoint_action_policy_v1(state=state)
    variants: list[tuple[str, str | None, str | None]] = []
    retired = set(state.retired_asset_ids)
    for action in policy.actions:
        if action.action_id == "nominate_for_early_study":
            variants.extend(
                (action.action_id, asset, None) for asset in state.candidate_asset_ids if asset not in retired
            )
        elif action.action_id == "select_lead_and_reserve":
            available = tuple(asset for asset in state.candidate_asset_ids if asset not in retired)
            variants.extend(
                (action.action_id, lead, reserve) for lead in available for reserve in available if lead != reserve
            )
        else:
            variants.append((action.action_id, None, None))
    return tuple(variants)


def _next_evidence(
    *, state: TrialDevProgrammeStateV1, action_id: str, target: str | None, reserve: str | None, source_identity: str
) -> tuple[TrialDevEvidenceReferenceV1, ...]:
    mapping = {
        "nominate_for_early_study": ("early_safety_study", (cast(str, target),)),
        "select_lead_and_reserve": ("joint_early_study_review", (cast(str, target), cast(str, reserve))),
        "advance_to_proof_of_concept": ("proof_of_concept", (cast(str, state.active_asset_id),)),
        "advance_lead_to_proof_of_concept": ("lead_proof_of_concept_review", (cast(str, state.lead_asset_id),)),
        "promote_reserve_to_proof_of_concept": (
            "promoted_reserve_proof_of_concept_review",
            (cast(str, state.reserve_asset_id),),
        ),
        "advance_to_confirmation": ("confirmation", (cast(str, state.active_asset_id),)),
        "advance_active_to_confirmation": ("confirmation", (cast(str, state.active_asset_id),)),
    }
    next_spec = mapping.get(action_id)
    if next_spec is None:
        return ()
    branch_id = cast(str, state.checksum)[:12] + "-" + action_id
    return _evidence(
        checkpoint_id=next_spec[0],
        asset_ids=next_spec[1],
        branch_id=branch_id,
        source_identity=source_identity,
    )


def _initial_states(source_identity: str) -> tuple[TrialDevProgrammeStateV1, ...]:
    single_assets = ("A",)
    portfolio_assets = ("A", "B", "C")
    return (
        TrialDevSingleAssetProgrammeStateV1(
            programme_id="single-asset-finite-graph",
            scenario_id="single-asset-finite-graph",
            current_checkpoint_id="observational_review",
            candidate_asset_ids=single_assets,
            policy_binding=_binding("single_asset_development", source_identity),
            evidence=_evidence(
                checkpoint_id="observational_review",
                asset_ids=single_assets,
                branch_id="single-initial",
                source_identity=source_identity,
            ),
        ),
        TrialDevPortfolioProgrammeStateV1(
            programme_id="portfolio-finite-graph",
            scenario_id="portfolio-finite-graph",
            current_checkpoint_id="observational_review",
            candidate_asset_ids=portfolio_assets,
            policy_binding=_binding("bounded_portfolio_reallocation", source_identity),
            evidence=_evidence(
                checkpoint_id="observational_review",
                asset_ids=portfolio_assets,
                branch_id="portfolio-initial",
                source_identity=source_identity,
            ),
        ),
    )


def build_trialdev_state_action_graph_v1(*, source_identity: str) -> TrialDevStateActionGraphV1:
    """Enumerate every reachable decision state and concrete legal action."""

    if len(source_identity) != 64 or any(character not in "0123456789abcdef" for character in source_identity):
        raise ValueError("State-action graph source identity must be a SHA-256 hex digest.")
    representatives: dict[str, TrialDevProgrammeStateV1] = {}
    queue: deque[str] = deque()
    for state in _initial_states(source_identity):
        node = _node(state)
        representatives[node.node_id] = state
        queue.append(node.node_id)
    edges: dict[str, TrialDevStateActionEdgeV1] = {}
    while queue:
        source_node_id = queue.popleft()
        state = representatives[source_node_id]
        policy = build_checkpoint_action_policy_v1(state=state)
        for action_id, target, reserve in _action_variants(state):
            selection = TrialDevActionSelectionV1(
                state_checksum=cast(str, state.checksum),
                checkpoint_id=state.current_checkpoint_id,
                action_id=action_id,
                target_asset_id=target,
                reserve_asset_id=reserve,
                analysis_method_id="finite_graph_enumeration_v1",
                supporting_evidence_ids=tuple(item.evidence_id for item in state.evidence),
                justification="Enumerated legal transition.",
            )
            after = transition_programme_state_v1(
                state=state,
                action_policy=policy,
                selection=selection,
                outcome=TrialDevCheckpointOutcomeV1(
                    reach_status="reached",
                    submission_status="accepted",
                    analysis_status="estimable",
                    execution_status="completed",
                ),
                next_evidence=_next_evidence(
                    state=state,
                    action_id=action_id,
                    target=target,
                    reserve=reserve,
                    source_identity=source_identity,
                ),
            )
            target_node_id = None
            if after.terminal_disposition == "active":
                target_node = _node(after)
                target_node_id = target_node.node_id
                if target_node_id not in representatives:
                    representatives[target_node_id] = after
                    queue.append(target_node_id)
            edge_payload: dict[str, JsonValue] = {
                "stream_id": state.stream_id,
                "source_node_id": source_node_id,
                "action_id": action_id,
                "target_asset_id": target,
                "reserve_asset_id": reserve,
                "target_node_id": target_node_id,
                "terminal_disposition": after.terminal_disposition,
            }
            edge = TrialDevStateActionEdgeV1(edge_id=_sha(edge_payload), **edge_payload)
            edges[edge.edge_id] = edge
    nodes = tuple(sorted((_node(state) for state in representatives.values()), key=lambda item: item.node_id))
    return TrialDevStateActionGraphV1(
        source_identity=source_identity,
        nodes=nodes,
        edges=tuple(sorted(edges.values(), key=lambda item: item.edge_id)),
    )


def write_trialdev_state_action_graph_v1(*, output_path: Path, source_identity: str) -> TrialDevStateActionGraphV1:
    """Write the complete TrialDev state-action graph as canonical JSON."""

    graph = build_trialdev_state_action_graph_v1(source_identity=source_identity)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_model(output_path, graph)
    return graph


__all__ = ["build_trialdev_state_action_graph_v1", "write_trialdev_state_action_graph_v1"]
