"""Audit exact TrialDev portfolio routes supported by released evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import JsonValue
from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevCheckpointOutcomeV1,
    TrialDevDecisionRuleEvidenceV1,
    TrialDevObservationalDecisionEvidenceV1,
    TrialDevPortfolioCheckpointActionPolicyV1,
    TrialDevPortfolioEvidenceIndexV1,
    TrialDevPortfolioProgrammeStateV1,
    TrialDevRandomizedDecisionEvidenceV1,
    TrialDevSupportedActionV1,
)
from trialagentbench_harness.io import read_json
from trialagentbench_harness.trialdev.policy import derive_supported_action_set_v1
from trialagentbench_harness.trialdev.portfolio_grading import (
    _asset_eligibility_v1,
    _evaluator_inputs,
    _expected_observational_evidence,
    _expected_randomized_evidence,
)
from trialagentbench_harness.trialdev.portfolio_release import (
    initial_portfolio_state_v1,
    load_portfolio_catalogue_v1,
    load_portfolio_manifest_v1,
)
from trialagentbench_harness.trialdev.programme import (
    build_checkpoint_action_policy_v1,
    transition_portfolio_programme_state_v1,
)
from trialagentbench_harness.trialdev.share.public_method_design import (
    TrialDevPhaseAnalysisMethodCatalogV1,
)
from typing_extensions import Self  # noqa: UP035

from trialagentbench_validation.trialdev.portfolio_grader_controls import (
    _evidence_stub,
    _selection,
)
from trialagentbench_validation.trialdev.worked_programmes import (
    derive_supported_action_signatures_v1,
)

_PHASE_BY_CHECKPOINT = {
    "joint_early_study_review": "phase1",
    "lead_proof_of_concept_review": "phase2",
    "promoted_reserve_proof_of_concept_review": "phase2",
    "confirmation": "phase3",
}
_REQUIRED_ACTIONS = {
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
_REQUIRED_CHECKPOINTS = {
    "observational_review",
    "joint_early_study_review",
    "lead_proof_of_concept_review",
    "promoted_reserve_proof_of_concept_review",
    "confirmation",
}
_REQUIRED_TERMINAL_DISPOSITIONS = {
    "withheld",
    "stopped",
    "success",
    "failure",
    "inconclusive",
}

_DecisionEvidence: TypeAlias = (
    TrialDevObservationalDecisionEvidenceV1 | TrialDevRandomizedDecisionEvidenceV1
)
_ReferenceVariant: TypeAlias = tuple[
    TrialDevPortfolioEvidenceIndexV1, _DecisionEvidence
]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialDevPortfolioFamilyRouteSummaryV1(_Record):
    """Reachable evidence-supported behavior for one construction family."""

    family_id: str = Field(min_length=1)
    world_id: str = Field(min_length=1)
    participant_view_count: int = Field(gt=0)
    state_count_min: int = Field(gt=0)
    state_count_max: int = Field(gt=0)
    terminal_route_count_min: int = Field(gt=0)
    terminal_route_count_max: int = Field(gt=0)
    supported_action_ids: tuple[str, ...] = Field(min_length=1)
    reached_checkpoint_ids: tuple[str, ...] = Field(min_length=1)
    terminal_dispositions: tuple[str, ...] = Field(min_length=1)
    initial_supported_pair_count_min: int = Field(ge=0)
    initial_supported_pair_count_max: int = Field(ge=0)
    selection_supported_view_count: int = Field(ge=0)
    withholding_only_view_count: int = Field(ge=0)
    nonidentified_view_count: int = Field(ge=0)
    joint_safety_stop_state_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Require ordered ranges and complete initial-decision counts."""

        if self.state_count_min > self.state_count_max:
            raise ValueError("Family state-count range is reversed.")
        if self.terminal_route_count_min > self.terminal_route_count_max:
            raise ValueError("Family terminal-route range is reversed.")
        if (
            self.initial_supported_pair_count_min
            > self.initial_supported_pair_count_max
        ):
            raise ValueError("Family initial-pair range is reversed.")
        if (
            self.selection_supported_view_count + self.withholding_only_view_count
            > self.participant_view_count
        ):
            raise ValueError(
                "Family initial-decision counts exceed its view denominator."
            )
        return self


class TrialDevPortfolioRouteAuditV1(_Record):
    """Complete exact-release portfolio route and decision-contrast audit."""

    schema_id: Literal["trialagentbench.validation.trialdev_portfolio_routes/v1"] = (
        "trialagentbench.validation.trialdev_portfolio_routes/v1"
    )
    release_source_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    participant_view_count: int = Field(gt=0)
    evaluated_state_count: int = Field(gt=0)
    terminal_route_count: int = Field(gt=0)
    supported_action_ids: tuple[str, ...] = Field(min_length=1)
    reached_checkpoint_ids: tuple[str, ...] = Field(min_length=1)
    terminal_dispositions: tuple[str, ...] = Field(min_length=1)
    independent_action_ids: tuple[str, ...] = ()
    independent_comparison_count: int = Field(default=0, ge=0)
    independent_discrepancy_count: int = Field(default=0, ge=0)
    nonidentified_view_count: int = Field(ge=0)
    identified_withholding_only_view_count: int = Field(ge=0)
    multiple_initial_action_view_count: int = Field(ge=0)
    multiple_randomized_action_state_count: int = Field(ge=0)
    safety_exclusion_state_count: int = Field(ge=0)
    joint_safety_stop_state_count: int = Field(ge=0)
    early_reserve_promotion_route_count: int = Field(ge=0)
    late_reserve_promotion_route_count: int = Field(ge=0)
    promoted_reserve_confirmation_route_count: int = Field(ge=0)
    promoted_reserve_stop_route_count: int = Field(ge=0)
    budget_contrast_world_objectives: tuple[str, ...]
    families: tuple[TrialDevPortfolioFamilyRouteSummaryV1, ...] = Field(min_length=1)
    findings: tuple[str, ...]
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        """Bind pass status to canonical findings and complete family coverage."""

        if self.findings != tuple(sorted(set(self.findings))):
            raise ValueError("Portfolio route findings must be sorted and unique.")
        if (self.status == "pass") != (not self.findings):
            raise ValueError("Portfolio route status disagrees with its findings.")
        if (
            sum(row.participant_view_count for row in self.families)
            != self.participant_view_count
        ):
            raise ValueError(
                "Portfolio family summaries do not cover every participant view."
            )
        if self.independent_comparison_count:
            if self.independent_action_ids != self.supported_action_ids:
                raise ValueError(
                    "Independent and public supported-action inventories differ."
                )
            if self.status == "pass" and self.independent_discrepancy_count:
                raise ValueError(
                    "A passing route audit cannot contain an independent discrepancy."
                )
        elif self.independent_action_ids or self.independent_discrepancy_count:
            raise ValueError(
                "Independent action results require a positive comparison count."
            )
        return self


def _reference_variants(
    *,
    release_root: Path,
    state: TrialDevPortfolioProgrammeStateV1,
) -> tuple[_ReferenceVariant, ...]:
    index, report = _evaluator_inputs(release_root, state)
    world_root = release_root / "worlds" / state.scenario_id
    if state.current_checkpoint_id == "observational_review":
        return tuple(
            (
                index,
                _expected_observational_evidence(
                    world_root=world_root,
                    state=state,
                    submitted=_evidence_stub(state, method.method_route_id),
                    report=report,
                ),
            )
            for method in report.method_results
        )
    phase_id = _PHASE_BY_CHECKPOINT[state.current_checkpoint_id]
    method = TrialDevPhaseAnalysisMethodCatalogV1.model_validate(
        read_json(world_root / "public" / "phase_analysis_method_catalog.json")
    ).method_for_phase(phase_id)
    reference = next(
        item
        for item in state.evidence
        if item.checkpoint_id == state.current_checkpoint_id
    )
    if reference.asset_id is None:
        raise ValueError("Portfolio randomized evidence must identify an asset.")
    stub = TrialDevRandomizedDecisionEvidenceV1(
        state_checksum=cast(str, state.checksum),
        analysis_method_id=method.method_route_id,
        rules=(
            TrialDevDecisionRuleEvidenceV1(
                rule_id="portfolio-route-audit-placeholder",
                asset_id=reference.asset_id,
                domain="safety",
                direction="maximum",
                estimate=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                threshold=1.0,
                evidence_reference_checksums=(cast(str, reference.checksum),),
            ),
        ),
    )
    return (
        (
            index,
            _expected_randomized_evidence(
                release_root=release_root,
                world_root=world_root,
                state=state,
                submitted=stub,
            ),
        ),
    )


def _reference_cache_key(
    state: TrialDevPortfolioProgrammeStateV1,
) -> tuple[object, ...]:
    """Identify evidence calculations shared across objective and budget views."""

    current_references = tuple(
        (item.checkpoint_id, item.asset_id, item.checksum)
        for item in state.evidence
        if item.checkpoint_id == state.current_checkpoint_id
    )
    if state.current_checkpoint_id == "observational_review":
        return (
            state.scenario_id,
            state.current_checkpoint_id,
            state.policy_binding.objective_id,
            current_references,
        )
    return (state.scenario_id, state.current_checkpoint_id, current_references)


def _rebind_reference_variants(
    variants: tuple[_ReferenceVariant, ...],
    *,
    state_checksum: str,
) -> tuple[_ReferenceVariant, ...]:
    """Bind cached numeric evidence to the current immutable programme state."""

    rebound: list[_ReferenceVariant] = []
    for index, evidence in variants:
        payload = evidence.model_dump(mode="python", exclude={"checksum"})
        payload["state_checksum"] = state_checksum
        if isinstance(evidence, TrialDevObservationalDecisionEvidenceV1):
            bound: _DecisionEvidence = (
                TrialDevObservationalDecisionEvidenceV1.model_validate(payload)
            )
        else:
            bound = TrialDevRandomizedDecisionEvidenceV1.model_validate(payload)
        rebound.append((index, bound))
    return tuple(rebound)


def _findings(
    *,
    action_ids: set[str],
    checkpoint_ids: set[str],
    terminal_dispositions: set[str],
    nonidentified_view_count: int,
    identified_withholding_only_view_count: int,
    multiple_initial_action_view_count: int,
    multiple_randomized_action_state_count: int,
    safety_exclusion_state_count: int,
    joint_safety_stop_state_count: int,
    early_reserve_promotion_route_count: int,
    late_reserve_promotion_route_count: int,
    promoted_reserve_confirmation_route_count: int,
    promoted_reserve_stop_route_count: int,
    budget_contrasts: set[str],
) -> tuple[str, ...]:
    findings: list[str] = []
    if action_ids != _REQUIRED_ACTIONS:
        findings.append("incomplete_supported_action_coverage")
    if checkpoint_ids != _REQUIRED_CHECKPOINTS:
        findings.append("incomplete_checkpoint_coverage")
    if terminal_dispositions != _REQUIRED_TERMINAL_DISPOSITIONS:
        findings.append("incomplete_terminal_decision_coverage")
    for count, finding in (
        (nonidentified_view_count, "missing_nonidentification_control"),
        (
            identified_withholding_only_view_count,
            "missing_identified_withholding_control",
        ),
        (multiple_initial_action_view_count, "missing_set_valued_initial_decision"),
        (
            multiple_randomized_action_state_count,
            "missing_set_valued_randomized_decision",
        ),
        (safety_exclusion_state_count, "missing_clear_safety_exclusion"),
        (joint_safety_stop_state_count, "missing_joint_safety_stop"),
        (early_reserve_promotion_route_count, "missing_early_reserve_promotion"),
        (late_reserve_promotion_route_count, "missing_late_reserve_promotion"),
        (
            promoted_reserve_confirmation_route_count,
            "missing_promoted_reserve_confirmation",
        ),
        (promoted_reserve_stop_route_count, "missing_promoted_reserve_stop"),
        (len(budget_contrasts), "missing_resource_budget_contrast"),
    ):
        if count == 0:
            findings.append(finding)
    return tuple(sorted(findings))


def _family_contrast_findings(
    *,
    grouped: dict[str, list[dict[str, object]]],
    budget_contrasts: set[str],
) -> tuple[str, ...]:
    """Verify that each named construction family realizes its declared contrast."""

    expected_families = {f"P{index:02d}" for index in range(1, 13)}
    by_prefix = {
        family_id.split("_", maxsplit=1)[0]: rows for family_id, rows in grouped.items()
    }
    if set(by_prefix) != expected_families:
        return ("incomplete_construction_family_inventory",)

    def checkpoint_actions(row: dict[str, object], checkpoint_id: str) -> set[str]:
        actions = cast(dict[str, set[str]], row["checkpoint_actions"])
        return actions.get(checkpoint_id, set())

    findings: list[str] = []
    if not all(
        bool(row["selection_supported"])
        and "joint_early_study_review" in cast(set[str], row["checkpoints"])
        for row in by_prefix["P01"]
    ):
        findings.append("P01_reference_allocation_not_recovered")
    if not all(
        cast(int, row["initial_supported_pair_count"]) >= 2 for row in by_prefix["P02"]
    ):
        findings.append("P02_multiple_pair_contrast_not_recovered")
    if not any(
        len(checkpoint_actions(row, "joint_early_study_review")) >= 2
        for row in by_prefix["P03"]
    ):
        findings.append("P03_early_safety_uncertainty_not_recovered")
    if not any(
        cast(int, row["joint_safety_stop_state_count"]) > 0 for row in by_prefix["P04"]
    ):
        findings.append("P04_shared_safety_stop_not_recovered")
    for family_id in ("P05", "P06"):
        if not all(
            bool(row["withholding_only"])
            and cast(set[str], row["identification_statuses"]) == {"identified"}
            for row in by_prefix[family_id]
        ):
            findings.append(
                f"{family_id}_identified_withholding_contrast_not_recovered"
            )
    if not any(
        len(checkpoint_actions(row, "lead_proof_of_concept_review")) >= 2
        for row in by_prefix["P07"]
    ):
        findings.append("P07_lead_uncertainty_contrast_not_recovered")
    if not any(value.startswith("portfolio-world-08:") for value in budget_contrasts):
        findings.append("P08_budget_contrast_not_recovered")
    if not any(
        "terminate_portfolio"
        in checkpoint_actions(row, "promoted_reserve_proof_of_concept_review")
        for row in by_prefix["P09"]
    ):
        findings.append("P09_promoted_reserve_futility_not_recovered")
    if not any(
        "success" in cast(set[str], row["terminals"]) for row in by_prefix["P10"]
    ):
        findings.append("P10_confirmatory_success_not_recovered")
    if not all(
        bool(row["withholding_only"])
        and cast(set[str], row["identification_statuses"]) == {"not_identified"}
        for row in by_prefix["P11"]
    ):
        findings.append("P11_nonidentification_contrast_not_recovered")
    return tuple(sorted(findings))


def audit_trialdev_portfolio_routes_v1(
    *, release_root: Path
) -> TrialDevPortfolioRouteAuditV1:
    """Traverse every method-conditioned route supported by exact release bytes."""

    root = Path(release_root).resolve(strict=True)
    catalogue = load_portfolio_catalogue_v1(root)
    manifest = load_portfolio_manifest_v1(root)
    family_by_programme = {
        view.programme_id: view.family_id for view in manifest.evaluator_views
    }
    state_total = 0
    terminal_route_total = 0
    action_ids: set[str] = set()
    independent_action_ids: set[str] = set()
    independent_comparisons = 0
    independent_discrepancies = 0
    checkpoint_ids: set[str] = set()
    terminal_dispositions: set[str] = set()
    nonidentified_views = 0
    identified_withholding_only_views = 0
    multiple_initial_views = 0
    multiple_randomized_states = 0
    safety_exclusion_states = 0
    joint_safety_stop_states = 0
    early_promotions = 0
    late_promotions = 0
    promoted_confirmations = 0
    promoted_stops = 0
    per_view: dict[str, dict[str, object]] = {}
    late_promotion_by_budget: dict[tuple[str, str, int], bool] = {}
    reference_cache: dict[tuple[object, ...], tuple[_ReferenceVariant, ...]] = {}

    for view in catalogue.views:
        frontier: list[tuple[TrialDevPortfolioProgrammeStateV1, tuple[str, ...]]] = [
            (initial_portfolio_state_v1(view), ())
        ]
        seen: set[str] = set()
        routes: set[tuple[str, tuple[str, ...]]] = set()
        view_actions: set[str] = set()
        view_checkpoints: set[str] = set()
        initial_signatures: set[tuple[str, str | None, str | None]] = set()
        identification_statuses: set[str] = set()
        checkpoint_actions: dict[str, set[str]] = {}
        view_joint_safety_stops = 0
        view_late_promotion = False
        while frontier:
            state, path = frontier.pop()
            state_checksum = cast(str, state.checksum)
            if state_checksum in seen:
                continue
            seen.add(state_checksum)
            view_checkpoints.add(str(state.current_checkpoint_id))
            supported_by_signature: dict[
                tuple[str, str | None, str | None],
                tuple[
                    TrialDevPortfolioEvidenceIndexV1,
                    TrialDevObservationalDecisionEvidenceV1
                    | TrialDevRandomizedDecisionEvidenceV1,
                    TrialDevSupportedActionV1,
                ],
            ] = {}
            state_safety_exclusions: set[str] = set()
            cache_key = _reference_cache_key(state)
            cached = reference_cache.get(cache_key)
            if cached is None:
                variants = _reference_variants(release_root=root, state=state)
                reference_cache[cache_key] = variants
            else:
                variants = _rebind_reference_variants(
                    cached, state_checksum=state_checksum
                )
            for index, evidence in variants:
                if isinstance(evidence, TrialDevObservationalDecisionEvidenceV1):
                    identification_statuses.add(str(evidence.identification_status))
                eligibility = _asset_eligibility_v1(evidence)
                state_safety_exclusions.update(item.asset_id for item in eligibility)
                supported = derive_supported_action_set_v1(
                    state=state, evidence=evidence
                )
                public_signatures = {
                    (str(row.action_id), row.target_asset_id, row.reserve_asset_id)
                    for row in supported.supported_actions
                }
                independent_signatures = derive_supported_action_signatures_v1(
                    state=cast(dict[str, JsonValue], state.model_dump(mode="json")),
                    evidence=cast(
                        dict[str, JsonValue], evidence.model_dump(mode="json")
                    ),
                )
                independent_comparisons += 1
                independent_action_ids.update(
                    signature[0] for signature in independent_signatures
                )
                if independent_signatures != public_signatures:
                    independent_discrepancies += 1
                for action in supported.supported_actions:
                    signature = (
                        str(action.action_id),
                        action.target_asset_id,
                        action.reserve_asset_id,
                    )
                    supported_by_signature.setdefault(
                        signature, (index, evidence, action)
                    )
            if (
                state.current_checkpoint_id != "observational_review"
                and len(supported_by_signature) > 1
            ):
                multiple_randomized_states += 1
            if state_safety_exclusions:
                safety_exclusion_states += 1
            if (
                state.current_checkpoint_id == "joint_early_study_review"
                and state.lead_asset_id in state_safety_exclusions
                and state.reserve_asset_id in state_safety_exclusions
            ):
                joint_safety_stop_states += 1
                view_joint_safety_stops += 1
            for signature, (index, evidence, action) in supported_by_signature.items():
                action_id = signature[0]
                view_actions.add(action_id)
                checkpoint_actions.setdefault(
                    str(state.current_checkpoint_id), set()
                ).add(action_id)
                if state.current_checkpoint_id == "observational_review":
                    initial_signatures.add(signature)
                selection = _selection(
                    state=state,
                    action=action,
                    method_id=evidence.analysis_method_id,
                )
                outcome = TrialDevCheckpointOutcomeV1(
                    reach_status="reached",
                    submission_status="accepted",
                    analysis_status=(
                        "non_estimable"
                        if isinstance(evidence, TrialDevObservationalDecisionEvidenceV1)
                        and evidence.identification_status == "not_identified"
                        else "estimable"
                    ),
                    execution_status="completed",
                    asset_eligibility=_asset_eligibility_v1(evidence),
                )
                next_state = transition_portfolio_programme_state_v1(
                    state=state,
                    evidence_index=index,
                    action_policy=cast(
                        TrialDevPortfolioCheckpointActionPolicyV1,
                        build_checkpoint_action_policy_v1(state=state),
                    ),
                    selection=selection,
                    outcome=outcome,
                )
                next_path = (*path, action_id)
                if action_id == "promote_reserve_to_proof_of_concept":
                    if state.current_checkpoint_id == "joint_early_study_review":
                        early_promotions += 1
                    elif state.current_checkpoint_id == "lead_proof_of_concept_review":
                        late_promotions += 1
                        view_late_promotion = True
                if (
                    state.current_checkpoint_id
                    == "promoted_reserve_proof_of_concept_review"
                ):
                    if action_id == "advance_active_to_confirmation":
                        promoted_confirmations += 1
                    elif action_id == "terminate_portfolio":
                        promoted_stops += 1
                if next_state.terminal_disposition == "active":
                    frontier.append((next_state, next_path))
                else:
                    disposition = str(next_state.terminal_disposition)
                    routes.add((disposition, next_path))
                    terminal_dispositions.add(disposition)
        if "not_identified" in identification_statuses:
            nonidentified_views += 1
        if initial_signatures == {
            ("withhold_selection", None, None)
        } and identification_statuses == {"identified"}:
            identified_withholding_only_views += 1
        if len(initial_signatures) > 1:
            multiple_initial_views += 1
        state_total += len(seen)
        terminal_route_total += len(routes)
        action_ids.update(view_actions)
        checkpoint_ids.update(view_checkpoints)
        late_promotion_by_budget[
            (view.world_id, view.objective_id, view.resource_budget_units)
        ] = view_late_promotion
        family_id = family_by_programme[view.programme_id]
        per_view[view.programme_id] = {
            "family_id": family_id,
            "world_id": view.world_id,
            "state_count": len(seen),
            "terminal_route_count": len(routes),
            "actions": view_actions,
            "checkpoint_actions": checkpoint_actions,
            "checkpoints": view_checkpoints,
            "terminals": {disposition for disposition, _path in routes},
            "initial_signatures": initial_signatures,
            "initial_supported_pair_count": sum(
                signature[0] == "select_lead_and_reserve"
                for signature in initial_signatures
            ),
            "identification_statuses": identification_statuses,
            "joint_safety_stop_state_count": view_joint_safety_stops,
            "selection_supported": any(
                signature[0] == "select_lead_and_reserve"
                for signature in initial_signatures
            ),
            "withholding_only": initial_signatures
            == {("withhold_selection", None, None)},
        }

    budget_contrasts = {
        f"{world_id}:{objective_id}"
        for world_id, objective_id, budget in late_promotion_by_budget
        if budget == 10
        and late_promotion_by_budget[(world_id, objective_id, 10)]
        and not late_promotion_by_budget.get((world_id, objective_id, 8), False)
    }
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in per_view.values():
        grouped.setdefault(cast(str, row["family_id"]), []).append(row)
    families = tuple(
        TrialDevPortfolioFamilyRouteSummaryV1(
            family_id=family_id,
            world_id=cast(str, rows[0]["world_id"]),
            participant_view_count=len(rows),
            state_count_min=min(cast(int, row["state_count"]) for row in rows),
            state_count_max=max(cast(int, row["state_count"]) for row in rows),
            terminal_route_count_min=min(
                cast(int, row["terminal_route_count"]) for row in rows
            ),
            terminal_route_count_max=max(
                cast(int, row["terminal_route_count"]) for row in rows
            ),
            supported_action_ids=tuple(
                sorted(
                    {item for row in rows for item in cast(set[str], row["actions"])}
                )
            ),
            reached_checkpoint_ids=tuple(
                sorted(
                    {
                        item
                        for row in rows
                        for item in cast(set[str], row["checkpoints"])
                    }
                )
            ),
            terminal_dispositions=tuple(
                sorted(
                    {item for row in rows for item in cast(set[str], row["terminals"])}
                )
            ),
            initial_supported_pair_count_min=min(
                cast(int, row["initial_supported_pair_count"]) for row in rows
            ),
            initial_supported_pair_count_max=max(
                cast(int, row["initial_supported_pair_count"]) for row in rows
            ),
            selection_supported_view_count=sum(
                bool(row["selection_supported"]) for row in rows
            ),
            withholding_only_view_count=sum(
                bool(row["withholding_only"]) for row in rows
            ),
            nonidentified_view_count=sum(
                "not_identified" in cast(set[str], row["identification_statuses"])
                for row in rows
            ),
            joint_safety_stop_state_count=sum(
                cast(int, row["joint_safety_stop_state_count"]) for row in rows
            ),
        )
        for family_id, rows in sorted(grouped.items())
    )
    findings = _findings(
        action_ids=action_ids,
        checkpoint_ids=checkpoint_ids,
        terminal_dispositions=terminal_dispositions,
        nonidentified_view_count=nonidentified_views,
        identified_withholding_only_view_count=identified_withholding_only_views,
        multiple_initial_action_view_count=multiple_initial_views,
        multiple_randomized_action_state_count=multiple_randomized_states,
        safety_exclusion_state_count=safety_exclusion_states,
        joint_safety_stop_state_count=joint_safety_stop_states,
        early_reserve_promotion_route_count=early_promotions,
        late_reserve_promotion_route_count=late_promotions,
        promoted_reserve_confirmation_route_count=promoted_confirmations,
        promoted_reserve_stop_route_count=promoted_stops,
        budget_contrasts=budget_contrasts,
    )
    findings = tuple(
        sorted(
            {
                *findings,
                *_family_contrast_findings(
                    grouped=grouped, budget_contrasts=budget_contrasts
                ),
            }
        )
    )
    if independent_discrepancies:
        findings = tuple(
            sorted({*findings, "independent_supported_action_disagreement"})
        )
    return TrialDevPortfolioRouteAuditV1(
        release_source_identity=catalogue.source_identity,
        participant_view_count=len(catalogue.views),
        evaluated_state_count=state_total,
        terminal_route_count=terminal_route_total,
        supported_action_ids=tuple(sorted(action_ids)),
        reached_checkpoint_ids=tuple(sorted(checkpoint_ids)),
        terminal_dispositions=tuple(sorted(terminal_dispositions)),
        independent_action_ids=tuple(sorted(independent_action_ids)),
        independent_comparison_count=independent_comparisons,
        independent_discrepancy_count=independent_discrepancies,
        nonidentified_view_count=nonidentified_views,
        identified_withholding_only_view_count=identified_withholding_only_views,
        multiple_initial_action_view_count=multiple_initial_views,
        multiple_randomized_action_state_count=multiple_randomized_states,
        safety_exclusion_state_count=safety_exclusion_states,
        joint_safety_stop_state_count=joint_safety_stop_states,
        early_reserve_promotion_route_count=early_promotions,
        late_reserve_promotion_route_count=late_promotions,
        promoted_reserve_confirmation_route_count=promoted_confirmations,
        promoted_reserve_stop_route_count=promoted_stops,
        budget_contrast_world_objectives=tuple(sorted(budget_contrasts)),
        families=families,
        findings=findings,
        status="fail" if findings else "pass",
    )


__all__ = [
    "TrialDevPortfolioFamilyRouteSummaryV1",
    "TrialDevPortfolioRouteAuditV1",
    "audit_trialdev_portfolio_routes_v1",
]
