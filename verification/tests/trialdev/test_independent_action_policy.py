"""Independent action-policy witnesses for every TrialDev action class."""

from __future__ import annotations

from pydantic.types import JsonValue

from trialagentbench_validation.trialdev.worked_programmes import (
    derive_supported_action_signatures_v1,
)

_STATE_SHA = "a" * 64
_ALL_ACTIONS = {
    "nominate_for_early_study",
    "withhold_nomination",
    "advance_to_proof_of_concept",
    "advance_to_confirmation",
    "stop_development",
    "select_lead_and_reserve",
    "withhold_selection",
    "advance_lead_to_proof_of_concept",
    "promote_reserve_to_proof_of_concept",
    "advance_active_to_confirmation",
    "terminate_portfolio",
    "declare_success",
    "declare_failure",
    "declare_inconclusive",
}


def _state(
    *,
    stream: str,
    checkpoint: str,
    active: str | None = "A",
) -> dict[str, JsonValue]:
    return {
        "stream_id": stream,
        "current_checkpoint_id": checkpoint,
        "checksum": _STATE_SHA,
        "candidate_asset_ids": ["A", "B", "C"],
        "retired_asset_ids": [],
        "lead_asset_id": "A",
        "reserve_asset_id": "B",
        "active_asset_id": active,
        "resource_budget_units": 10,
        "resource_spent_units": 0,
        "switch_count": 0,
    }


def _rule(*, asset: str, domain: str, state: str) -> dict[str, JsonValue]:
    direction = "minimum" if domain == "efficacy" else "maximum"
    threshold = 0.5
    interval = {
        "clear_pass": (0.7, 0.6, 0.8) if direction == "minimum" else (0.3, 0.2, 0.4),
        "clear_fail": (0.3, 0.2, 0.4) if direction == "minimum" else (0.7, 0.6, 0.8),
        "indeterminate": (0.5, 0.4, 0.6),
    }[state]
    return {
        "asset_id": asset,
        "domain": domain,
        "direction": direction,
        "estimate": interval[0],
        "lower_bound": interval[1],
        "upper_bound": interval[2],
        "threshold": threshold,
        "evidence_reference_checksums": [],
    }


def _randomized(*rules: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"state_checksum": _STATE_SHA, "rules": list(rules)}


def _observational(
    *, identified: bool, stream: str
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    state = _state(stream=stream, checkpoint="observational_review", active=None)
    if not identified:
        return state, {
            "state_checksum": _STATE_SHA,
            "identification_status": "not_identified",
            "candidates": [],
            "pair_contrasts": [],
        }
    candidates: list[JsonValue] = [
        {
            "asset_id": asset,
            "utility_estimate": utility,
            "efficacy_lower_bound": efficacy_lower,
            "efficacy_upper_bound": efficacy_upper,
        }
        for asset, utility, efficacy_lower, efficacy_upper in (
            ("A", 0.80, 0.60, 0.80),
            ("B", 0.76, 0.40, 0.70),
            ("C", 0.20, 0.10, 0.30),
        )
    ]
    contrasts: list[JsonValue] = [
        {
            "lead_asset_id": lead,
            "reserve_asset_id": reserve,
            "confidence_half_width": 0.10,
        }
        for lead in ("A", "B", "C")
        for reserve in ("A", "B", "C")
        if lead != reserve
    ]
    return state, {
        "state_checksum": _STATE_SHA,
        "identification_status": "identified",
        "candidates": candidates,
        "pair_contrasts": contrasts,
        "minimum_efficacy_gain": 0.50,
        "practical_equivalence_margin": 0.05,
    }


def _actions(state: dict[str, JsonValue], evidence: dict[str, JsonValue]) -> set[str]:
    return {
        action_id
        for action_id, _target, _reserve in derive_supported_action_signatures_v1(
            state=state,
            evidence=evidence,
        )
    }


def test_independent_policy_covers_every_action_class() -> None:
    observed: set[str] = set()
    for stream in ("single_asset_development", "bounded_portfolio_reallocation"):
        for identified in (True, False):
            observed |= _actions(*_observational(identified=identified, stream=stream))

    observed |= _actions(
        _state(stream="single_asset_development", checkpoint="early_safety_study"),
        _randomized(_rule(asset="A", domain="safety", state="indeterminate")),
    )
    observed |= _actions(
        _state(stream="single_asset_development", checkpoint="proof_of_concept"),
        _randomized(
            _rule(asset="A", domain="efficacy", state="indeterminate"),
            _rule(asset="A", domain="safety", state="clear_pass"),
        ),
    )
    observed |= _actions(
        _state(
            stream="bounded_portfolio_reallocation",
            checkpoint="joint_early_study_review",
        ),
        _randomized(
            _rule(asset="A", domain="safety", state="indeterminate"),
            _rule(asset="B", domain="safety", state="indeterminate"),
        ),
    )
    observed |= _actions(
        _state(
            stream="bounded_portfolio_reallocation",
            checkpoint="lead_proof_of_concept_review",
        ),
        _randomized(
            _rule(asset="A", domain="efficacy", state="indeterminate"),
            _rule(asset="A", domain="safety", state="clear_pass"),
        ),
    )
    observed |= _actions(
        _state(
            stream="bounded_portfolio_reallocation",
            checkpoint="promoted_reserve_proof_of_concept_review",
        ),
        _randomized(
            _rule(asset="A", domain="efficacy", state="indeterminate"),
            _rule(asset="A", domain="safety", state="clear_pass"),
        ),
    )
    for decision_state in ("clear_pass", "clear_fail", "indeterminate"):
        observed |= _actions(
            _state(stream="single_asset_development", checkpoint="confirmation"),
            _randomized(
                _rule(asset="A", domain="efficacy", state=decision_state),
                _rule(asset="A", domain="safety", state="clear_pass"),
            ),
        )

    assert observed == _ALL_ACTIONS
