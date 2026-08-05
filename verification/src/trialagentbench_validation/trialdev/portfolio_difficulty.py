"""Quantify TrialDev portfolio difficulty and resistance to action-only shortcuts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from trialagentbench_harness.trialdev.policy import derive_supported_action_set_v1
from trialagentbench_harness.trialdev.portfolio_grading import (
    _evaluator_inputs,
    _expected_observational_evidence,
)
from trialagentbench_harness.trialdev.portfolio_release import (
    initial_portfolio_state_v1,
    load_portfolio_catalogue_v1,
)

from trialagentbench_validation.contracts.v1_scope import (
    TRIALDEV_MAX_ACTION_ONLY_SHORTCUT_SUPPORT_RATE_V1,
    TRIALDEV_MAX_POINT_ESTIMATE_SHORTCUT_SUPPORT_RATE_V1,
)
from trialagentbench_validation.trialdev.portfolio_grader_controls import _evidence_stub

StrategyId = Literal[
    "evidence_and_policy",
    "always_withhold",
    "alphabetical_pair",
    "adjusted_point_pair",
    "raw_observed_pair",
]
StrategyClass = Literal["complete_analysis", "action_only", "point_estimate_only"]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialDevShortcutPerformanceV1(_Record):
    """Action-support performance of one prespecified strategy."""

    strategy_id: StrategyId
    strategy_class: StrategyClass
    evaluated_view_count: int = Field(gt=0)
    supported_view_count: int = Field(ge=0)
    supported_view_rate: float = Field(ge=0.0, le=1.0)
    complete_statistical_submission: bool
    primary_grade_eligible: bool

    @model_validator(mode="after")
    def validate_rate(self) -> Self:
        """Bind the reported rate and grade eligibility to finite counts."""

        if self.supported_view_count > self.evaluated_view_count:
            raise ValueError("Supported shortcut views cannot exceed evaluated views.")
        expected = self.supported_view_count / self.evaluated_view_count
        if abs(self.supported_view_rate - expected) > 1e-12:
            raise ValueError("Shortcut support rate disagrees with its counts.")
        if self.primary_grade_eligible and not self.complete_statistical_submission:
            raise ValueError(
                "An incomplete statistical submission cannot be primary-grade eligible."
            )
        return self


class TrialDevDifficultyViewV1(_Record):
    """Action-support and shortcut outcomes for one released programme view."""

    world_id: str = Field(min_length=1)
    objective_id: str = Field(min_length=1)
    resource_budget_units: int = Field(gt=0)
    identification_status: Literal["identified", "not_identified"]
    supported_action_count: int = Field(gt=0)
    evidence_and_policy_supported: bool
    always_withhold_supported: bool
    alphabetical_pair_supported: bool
    adjusted_point_pair_supported: bool
    raw_observed_pair_supported: bool


class TrialDevPortfolioDifficultyReportV1(_Record):
    """Exact-view difficulty and action-only shortcut analysis."""

    schema_id: Literal[
        "trialagentbench.validation.trialdev_portfolio_difficulty/v1"
    ] = "trialagentbench.validation.trialdev_portfolio_difficulty/v1"
    release_source_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    participant_view_count: int = Field(gt=0)
    identified_view_count: int = Field(ge=0)
    nonidentified_view_count: int = Field(ge=0)
    supported_action_count_distribution: dict[int, int]
    views: tuple[TrialDevDifficultyViewV1, ...] = Field(min_length=1)
    strategies: tuple[TrialDevShortcutPerformanceV1, ...] = Field(min_length=1)
    maximum_action_only_shortcut_support_rate: float = Field(ge=0.0, le=1.0)
    maximum_point_estimate_shortcut_support_rate: float = Field(ge=0.0, le=1.0)
    findings: tuple[str, ...]
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        """Require a complete view census and one canonical strategy inventory."""

        if (
            self.identified_view_count + self.nonidentified_view_count
            != self.participant_view_count
        ):
            raise ValueError(
                "Difficulty identification counts do not cover all participant views."
            )
        if (
            sum(self.supported_action_count_distribution.values())
            != self.participant_view_count
        ):
            raise ValueError(
                "Supported-action distribution does not cover all participant views."
            )
        if len(self.views) != self.participant_view_count:
            raise ValueError("Difficulty view rows do not cover all participant views.")
        expected = {
            "evidence_and_policy",
            "always_withhold",
            "alphabetical_pair",
            "adjusted_point_pair",
            "raw_observed_pair",
        }
        if {row.strategy_id for row in self.strategies} != expected:
            raise ValueError(
                "Difficulty report lacks a prespecified shortcut strategy."
            )
        if self.findings != tuple(sorted(set(self.findings))):
            raise ValueError("Difficulty findings must be sorted and unique.")
        if (self.status == "pass") != (not self.findings):
            raise ValueError("Difficulty status disagrees with its findings.")
        return self


def _pair_signature(rank_order: tuple[str, ...]) -> tuple[str, str, str] | None:
    if len(rank_order) < 2:
        return None
    return ("select_lead_and_reserve", rank_order[0], rank_order[1])


def _difficulty_findings(
    performance: tuple[TrialDevShortcutPerformanceV1, ...],
) -> tuple[str, ...]:
    findings: list[str] = []
    reference = next(
        row for row in performance if row.strategy_id == "evidence_and_policy"
    )
    if reference.supported_view_rate != 1.0:
        findings.append("evidence_policy_not_universally_supported")
    if any(
        row.supported_view_rate > TRIALDEV_MAX_ACTION_ONLY_SHORTCUT_SUPPORT_RATE_V1
        for row in performance
        if row.strategy_class == "action_only"
    ):
        findings.append("action_only_shortcut_exceeds_majority_support_ceiling")
    if any(
        row.supported_view_rate > TRIALDEV_MAX_POINT_ESTIMATE_SHORTCUT_SUPPORT_RATE_V1
        for row in performance
        if row.strategy_class == "point_estimate_only"
    ):
        findings.append("point_estimate_shortcut_exceeds_three_quarter_support_ceiling")
    return tuple(sorted(set(findings)))


def audit_trialdev_portfolio_difficulty_v1(
    *, release_root: Path
) -> TrialDevPortfolioDifficultyReportV1:
    """Compare evidence-aware allocation with prespecified action-only shortcuts."""

    root = Path(release_root).resolve(strict=True)
    catalogue = load_portfolio_catalogue_v1(root)
    supported_counts: Counter[int] = Counter()
    identification_counts: Counter[str] = Counter()
    strategy_hits: Counter[str] = Counter()
    strategy_denominators: Counter[str] = Counter()
    view_rows: list[TrialDevDifficultyViewV1] = []
    for view in catalogue.views:
        state = initial_portfolio_state_v1(view)
        _, report = _evaluator_inputs(root, state)
        method = report.method_results[0]
        evidence = _expected_observational_evidence(
            world_root=root / "worlds" / state.scenario_id,
            state=state,
            submitted=_evidence_stub(state, method.method_route_id),
            report=report,
        )
        supported = derive_supported_action_set_v1(state=state, evidence=evidence)
        signatures = {
            (row.action_id, row.target_asset_id, row.reserve_asset_id)
            for row in supported.supported_actions
        }
        supported_counts[len(signatures)] += 1
        identification_counts[evidence.identification_status] += 1
        objective_id = state.policy_binding.objective_id
        adjusted_rank = tuple(
            str(row.candidate_drug_id)
            for row in sorted(
                (
                    row
                    for row in method.candidate_scores
                    if row.objective_id == objective_id
                ),
                key=lambda row: (row.rank, row.candidate_drug_id),
            )
        )
        raw = next(
            row
            for row in method.estimator_comparisons
            if row.objective_id == objective_id and row.estimator_id == "raw_observed"
        )
        strategy_signatures: dict[str, tuple[str, str | None, str | None] | None] = {
            "evidence_and_policy": min(signatures),
            "always_withhold": ("withhold_selection", None, None),
            "alphabetical_pair": _pair_signature(
                tuple(sorted(state.candidate_asset_ids))
            ),
            "adjusted_point_pair": _pair_signature(adjusted_rank),
            "raw_observed_pair": _pair_signature(
                tuple(str(value) for value in raw.rank_order)
            ),
        }
        for strategy_id, signature in strategy_signatures.items():
            strategy_denominators[strategy_id] += 1
            strategy_hits[strategy_id] += int(signature in signatures)
        budget = state.policy_binding.resource_budget_units
        if budget is None:
            raise ValueError("Portfolio difficulty views require a resource budget.")
        view_rows.append(
            TrialDevDifficultyViewV1(
                world_id=state.scenario_id,
                objective_id=objective_id,
                resource_budget_units=budget,
                identification_status=evidence.identification_status,
                supported_action_count=len(signatures),
                evidence_and_policy_supported=strategy_signatures["evidence_and_policy"]
                in signatures,
                always_withhold_supported=strategy_signatures["always_withhold"]
                in signatures,
                alphabetical_pair_supported=strategy_signatures["alphabetical_pair"]
                in signatures,
                adjusted_point_pair_supported=strategy_signatures["adjusted_point_pair"]
                in signatures,
                raw_observed_pair_supported=strategy_signatures["raw_observed_pair"]
                in signatures,
            )
        )
    strategy_order: tuple[StrategyId, ...] = (
        "evidence_and_policy",
        "always_withhold",
        "alphabetical_pair",
        "adjusted_point_pair",
        "raw_observed_pair",
    )
    performance = tuple(
        TrialDevShortcutPerformanceV1(
            strategy_id=strategy_id,
            strategy_class=(
                "complete_analysis"
                if strategy_id == "evidence_and_policy"
                else (
                    "action_only"
                    if strategy_id in {"always_withhold", "alphabetical_pair"}
                    else "point_estimate_only"
                )
            ),
            evaluated_view_count=strategy_denominators[strategy_id],
            supported_view_count=strategy_hits[strategy_id],
            supported_view_rate=strategy_hits[strategy_id]
            / strategy_denominators[strategy_id],
            complete_statistical_submission=strategy_id == "evidence_and_policy",
            primary_grade_eligible=strategy_id == "evidence_and_policy",
        )
        for strategy_id in strategy_order
    )
    ordered_findings = _difficulty_findings(performance)
    return TrialDevPortfolioDifficultyReportV1(
        release_source_identity=catalogue.source_identity,
        participant_view_count=len(catalogue.views),
        identified_view_count=identification_counts["identified"],
        nonidentified_view_count=identification_counts["not_identified"],
        supported_action_count_distribution=dict(sorted(supported_counts.items())),
        views=tuple(view_rows),
        strategies=performance,
        maximum_action_only_shortcut_support_rate=TRIALDEV_MAX_ACTION_ONLY_SHORTCUT_SUPPORT_RATE_V1,
        maximum_point_estimate_shortcut_support_rate=TRIALDEV_MAX_POINT_ESTIMATE_SHORTCUT_SUPPORT_RATE_V1,
        findings=ordered_findings,
        status="fail" if ordered_findings else "pass",
    )


__all__ = [
    "TrialDevPortfolioDifficultyReportV1",
    "TrialDevDifficultyViewV1",
    "TrialDevShortcutPerformanceV1",
    "audit_trialdev_portfolio_difficulty_v1",
]
