"""Exercise TrialDev portfolio grading with accepted and single-fault submissions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from trialagentbench_harness.contracts.trialdev.portfolio_release import (
    TrialDevPortfolioEpisodeDatasetV1,
)
from trialagentbench_harness.contracts.trialdev.portfolio_submission import (
    TrialDevPortfolioCheckpointSubmissionV1,
    TrialDevScheduledStudyV1,
)
from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevObservationalDecisionEvidenceV1,
    TrialDevPortfolioActionSelectionV1,
    TrialDevPortfolioProgrammeStateV1,
    TrialDevSupportedActionV1,
)
from trialagentbench_harness.io import read_json_model
from trialagentbench_harness.trialdev.portfolio_grading import (
    PortfolioSubmissionError,
    _evaluator_inputs,
    _expected_next_references,
    _expected_observational_evidence,
    grade_portfolio_checkpoint_v1,
)
from trialagentbench_harness.trialdev.portfolio_release import (
    initial_portfolio_state_v1,
    load_portfolio_catalogue_v1,
)
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentRequestV1
from typing_extensions import Self  # noqa: UP035

ControlId = Literal[
    "accepted_reference",
    "numeric_evidence_error",
    "efficacy_evidence_error",
    "pairwise_uncertainty_error",
    "evidence_reference_error",
    "provenance_error",
    "unsupported_action",
    "scheduled_design_error",
    "stale_state",
]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialDevGraderControlV1(_Record):
    """Outcome of one accepted or intentionally faulty checkpoint submission."""

    programme_id: str = Field(min_length=1)
    world_id: str = Field(min_length=1)
    objective_id: str = Field(min_length=1)
    resource_budget_units: int = Field(gt=0)
    identification_status: Literal["identified", "not_identified"]
    control_id: ControlId
    control_kind: Literal["positive", "negative"]
    expected_submission_status: Literal["accepted", "invalid", "rejected_before_grade"]
    observed_submission_status: Literal["accepted", "invalid", "rejected_before_grade"]
    evidence_numeric_agreement: bool | None = None
    provenance_valid: bool | None = None
    selected_action_supported: bool | None = None
    scheduled_designs_valid: bool | None = None
    detected: bool


class TrialDevPortfolioGraderControlReportV1(_Record):
    """Complete exact-release portfolio grader behavior report."""

    schema_id: Literal[
        "trialagentbench.validation.trialdev_portfolio_grader_controls/v1"
    ] = "trialagentbench.validation.trialdev_portfolio_grader_controls/v1"
    release_source_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    participant_view_count: int = Field(gt=0)
    controls: tuple[TrialDevGraderControlV1, ...] = Field(min_length=1)
    findings: tuple[str, ...]
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        """Bind pass status to complete detection and canonical findings."""

        if self.findings != tuple(sorted(set(self.findings))):
            raise ValueError("Grader-control findings must be sorted and unique.")
        if (self.status == "pass") != (
            not self.findings and all(row.detected for row in self.controls)
        ):
            raise ValueError(
                "Grader-control status disagrees with its control outcomes."
            )
        return self


def _evidence_stub(
    state: TrialDevPortfolioProgrammeStateV1, method_id: str
) -> TrialDevObservationalDecisionEvidenceV1:
    return TrialDevObservationalDecisionEvidenceV1(
        state_checksum=cast(str, state.checksum),
        analysis_method_id=method_id,
        identification_status="not_identified",
        minimum_efficacy_gain=0.0,
        practical_equivalence_margin=0.0,
        identification_evidence_reference_checksums=(
            cast(str, state.evidence[0].checksum),
        ),
    )


def _selection(
    *,
    state: TrialDevPortfolioProgrammeStateV1,
    action: TrialDevSupportedActionV1,
    method_id: str,
    supporting_evidence_ids: tuple[str, ...] | None = None,
) -> TrialDevPortfolioActionSelectionV1:
    return TrialDevPortfolioActionSelectionV1(
        state_checksum=cast(str, state.checksum),
        checkpoint_id=state.current_checkpoint_id,
        action_id=action.action_id,
        target_asset_id=action.target_asset_id,
        reserve_asset_id=action.reserve_asset_id,
        analysis_method_id=method_id,
        supporting_evidence_ids=(
            supporting_evidence_ids
            if supporting_evidence_ids is not None
            else tuple(
                item.evidence_id
                for item in state.evidence
                if item.checkpoint_id == state.current_checkpoint_id
            )
        ),
        justification="The action is supported by the reported interval evidence and disclosed decision policy.",
    )


def _scheduled_studies(
    *,
    release_root: Path,
    state: TrialDevPortfolioProgrammeStateV1,
    index: object,
    selection: TrialDevPortfolioActionSelectionV1,
    evidence: TrialDevObservationalDecisionEvidenceV1,
) -> tuple[TrialDevScheduledStudyV1, ...]:
    from trialagentbench_harness.contracts.trialdev.programme import (
        TrialDevPortfolioEvidenceIndexV1,
    )

    typed_index = TrialDevPortfolioEvidenceIndexV1.model_validate(index)
    provisional = TrialDevPortfolioCheckpointSubmissionV1(
        state_checksum=cast(str, state.checksum),
        decision_evidence=evidence,
        selected_action=selection,
        scheduled_studies=(
            ()
            if selection.action_id in {"withhold_selection", "terminate_portfolio"}
            else tuple(
                TrialDevScheduledStudyV1(
                    asset_id=asset_id,
                    phase_id="phase1",
                    design_cell_id="placeholder",
                )
                for asset_id in (
                    tuple(
                        value
                        for value in (
                            selection.target_asset_id,
                            selection.reserve_asset_id,
                        )
                        if value is not None
                    )
                )
            )
        ),
    )
    references = _expected_next_references(
        state=state, index=typed_index, submission=provisional
    )
    output: list[TrialDevScheduledStudyV1] = []
    for reference in references:
        dataset = read_json_model(
            TrialDevPortfolioEpisodeDatasetV1,
            Path(release_root) / reference.relative_path,
        )
        request_record = next(
            item for item in dataset.metadata_files if item.metadata_id == "request"
        )
        request = read_json_model(
            TrialDevelopmentRequestV1,
            Path(release_root) / request_record.relative_path,
        )
        output.append(
            TrialDevScheduledStudyV1(
                asset_id=cast(str, reference.asset_id),
                phase_id=request.phase_id,
                design_cell_id=request.design_cell_id,
            )
        )
    return tuple(output)


def _submission(
    *,
    release_root: Path,
    state: TrialDevPortfolioProgrammeStateV1,
    index: object,
    evidence: TrialDevObservationalDecisionEvidenceV1,
    action: TrialDevSupportedActionV1,
    supporting_evidence_ids: tuple[str, ...] | None = None,
) -> TrialDevPortfolioCheckpointSubmissionV1:
    selection = _selection(
        state=state,
        action=action,
        method_id=evidence.analysis_method_id,
        supporting_evidence_ids=supporting_evidence_ids,
    )
    scheduled = _scheduled_studies(
        release_root=release_root,
        state=state,
        index=index,
        selection=selection,
        evidence=evidence,
    )
    return TrialDevPortfolioCheckpointSubmissionV1(
        state_checksum=cast(str, state.checksum),
        decision_evidence=evidence,
        selected_action=selection,
        scheduled_studies=scheduled,
    )


def _control_row(
    *,
    state: TrialDevPortfolioProgrammeStateV1,
    identification_status: Literal["identified", "not_identified"],
    control_id: ControlId,
    expected: Literal["accepted", "invalid", "rejected_before_grade"],
    grade: object | None,
    rejected: bool = False,
) -> TrialDevGraderControlV1:
    from trialagentbench_harness.contracts.trialdev.portfolio_submission import (
        TrialDevPortfolioCheckpointGradeV1,
    )

    typed_grade = (
        None
        if grade is None
        else TrialDevPortfolioCheckpointGradeV1.model_validate(grade)
    )
    if rejected:
        if typed_grade is not None:
            raise ValueError("A pre-grade rejection cannot include a grade.")
        observed: Literal["accepted", "invalid", "rejected_before_grade"] = (
            "rejected_before_grade"
        )
    else:
        if typed_grade is None:
            raise ValueError("A completed grader control requires a grade.")
        observed = (
            "accepted"
            if typed_grade.scientific_assessment.decision_complete
            else "invalid"
        )
    orthogonal_detection = {
        "accepted_reference": typed_grade is not None
        and all(
            (
                typed_grade.evidence_numeric_agreement,
                typed_grade.provenance_valid,
                typed_grade.selected_action_supported,
                typed_grade.scheduled_designs_valid,
            )
        ),
        "numeric_evidence_error": typed_grade is not None
        and not typed_grade.evidence_numeric_agreement
        and typed_grade.scientific_assessment.scientific_agreement == "failed"
        and typed_grade.provenance_valid
        and typed_grade.selected_action_supported
        and typed_grade.scheduled_designs_valid,
        "efficacy_evidence_error": typed_grade is not None
        and not typed_grade.evidence_numeric_agreement
        and typed_grade.scientific_assessment.scientific_agreement == "failed"
        and typed_grade.provenance_valid
        and typed_grade.selected_action_supported
        and typed_grade.scheduled_designs_valid,
        "pairwise_uncertainty_error": typed_grade is not None
        and not typed_grade.evidence_numeric_agreement
        and typed_grade.scientific_assessment.scientific_agreement == "failed"
        and typed_grade.provenance_valid
        and typed_grade.selected_action_supported
        and typed_grade.scheduled_designs_valid,
        "evidence_reference_error": typed_grade is not None
        and typed_grade.evidence_numeric_agreement
        and not typed_grade.provenance_valid
        and typed_grade.scientific_assessment.analysis_classification
        == "uncertainty_qualified"
        and typed_grade.selected_action_supported
        and typed_grade.scheduled_designs_valid,
        "provenance_error": typed_grade is not None
        and typed_grade.evidence_numeric_agreement
        and not typed_grade.provenance_valid
        and typed_grade.scientific_assessment.analysis_classification
        == "uncertainty_qualified"
        and typed_grade.selected_action_supported
        and typed_grade.scheduled_designs_valid,
        "unsupported_action": typed_grade is not None
        and typed_grade.evidence_numeric_agreement
        and typed_grade.provenance_valid
        and not typed_grade.selected_action_supported
        and typed_grade.scheduled_designs_valid,
        "scheduled_design_error": typed_grade is not None
        and typed_grade.evidence_numeric_agreement
        and typed_grade.provenance_valid
        and typed_grade.selected_action_supported
        and not typed_grade.scheduled_designs_valid,
        "stale_state": rejected,
    }[control_id]
    return TrialDevGraderControlV1(
        programme_id=state.programme_id,
        world_id=state.scenario_id,
        objective_id=state.policy_binding.objective_id,
        resource_budget_units=cast(int, state.policy_binding.resource_budget_units),
        identification_status=identification_status,
        control_id=control_id,
        control_kind="positive" if control_id == "accepted_reference" else "negative",
        expected_submission_status=expected,
        observed_submission_status=observed,
        evidence_numeric_agreement=(
            None if typed_grade is None else typed_grade.evidence_numeric_agreement
        ),
        provenance_valid=None if typed_grade is None else typed_grade.provenance_valid,
        selected_action_supported=(
            None if typed_grade is None else typed_grade.selected_action_supported
        ),
        scheduled_designs_valid=(
            None if typed_grade is None else typed_grade.scheduled_designs_valid
        ),
        detected=observed == expected and orthogonal_detection,
    )


def _choose_supported_action(
    actions: tuple[TrialDevSupportedActionV1, ...],
) -> TrialDevSupportedActionV1:
    priorities = {"select_lead_and_reserve": 0, "withhold_selection": 1}
    return min(
        actions,
        key=lambda row: (
            priorities.get(str(row.action_id), 9),
            str(row.action_id),
            row.target_asset_id or "",
            row.reserve_asset_id or "",
        ),
    )


def _mutate_evidence_reference(
    evidence: TrialDevObservationalDecisionEvidenceV1,
) -> TrialDevObservationalDecisionEvidenceV1:
    """Replace exactly one evidence-record checksum while preserving estimates."""

    payload = evidence.model_dump(mode="python", exclude={"checksum"})
    if evidence.identification_status == "not_identified":
        payload["identification_evidence_reference_checksums"] = ("0" * 64,)
    else:
        first = evidence.candidates[0]
        candidate_payload = first.model_dump(mode="python", exclude={"checksum"})
        candidate_payload["evidence_reference_checksums"] = ("0" * 64,)
        candidates = list(evidence.candidates)
        candidates[0] = type(first).model_validate(candidate_payload)
        payload["candidates"] = tuple(candidates)
    return TrialDevObservationalDecisionEvidenceV1.model_validate(payload)


def run_trialdev_portfolio_grader_controls_v1(
    *, release_root: Path, controls_csv: Path | None = None
) -> TrialDevPortfolioGraderControlReportV1:
    """Run complete and orthogonal single-fault controls across all portfolio views."""

    root = Path(release_root).resolve(strict=True)
    catalogue = load_portfolio_catalogue_v1(root)
    controls: list[TrialDevGraderControlV1] = []
    findings: list[str] = []
    for view in catalogue.views:
        state = initial_portfolio_state_v1(view)
        index, report = _evaluator_inputs(root, state)
        method_id = report.method_results[0].method_route_id
        expected_evidence = _expected_observational_evidence(
            world_root=root / "worlds" / state.scenario_id,
            state=state,
            submitted=_evidence_stub(state, method_id),
            report=report,
        )
        from trialagentbench_harness.trialdev.policy import (
            derive_supported_action_set_v1,
        )

        supported = derive_supported_action_set_v1(
            state=state, evidence=expected_evidence
        )
        action = _choose_supported_action(supported.supported_actions)
        reference = _submission(
            release_root=root,
            state=state,
            index=index,
            evidence=expected_evidence,
            action=action,
        )
        reference_grade = grade_portfolio_checkpoint_v1(
            release_root=root, state=state, submission=reference
        )
        controls.append(
            _control_row(
                state=state,
                identification_status=expected_evidence.identification_status,
                control_id="accepted_reference",
                expected="accepted",
                grade=reference_grade,
            )
        )
        numeric_payload = expected_evidence.model_dump(
            mode="python", exclude={"checksum"}
        )
        numeric_payload["minimum_efficacy_gain"] = (
            expected_evidence.minimum_efficacy_gain + 0.1
        )
        numeric_error = _submission(
            release_root=root,
            state=state,
            index=index,
            evidence=TrialDevObservationalDecisionEvidenceV1.model_validate(
                numeric_payload
            ),
            action=action,
        )
        controls.append(
            _control_row(
                state=state,
                identification_status=expected_evidence.identification_status,
                control_id="numeric_evidence_error",
                expected="invalid",
                grade=grade_portfolio_checkpoint_v1(
                    release_root=root, state=state, submission=numeric_error
                ),
            )
        )
        if expected_evidence.identification_status == "identified":
            efficacy_payload = expected_evidence.model_dump(
                mode="python", exclude={"checksum"}
            )
            efficacy_candidates = list(expected_evidence.candidates)
            first_candidate = efficacy_candidates[0]
            candidate_payload = first_candidate.model_dump(
                mode="python", exclude={"checksum"}
            )
            candidate_payload.update(
                {
                    "efficacy_estimate": first_candidate.efficacy_estimate + 0.5,
                    "efficacy_lower_bound": first_candidate.efficacy_lower_bound + 0.5,
                    "efficacy_upper_bound": first_candidate.efficacy_upper_bound + 0.5,
                }
            )
            efficacy_candidates[0] = type(first_candidate).model_validate(
                candidate_payload
            )
            efficacy_payload["candidates"] = tuple(efficacy_candidates)
            efficacy_error = _submission(
                release_root=root,
                state=state,
                index=index,
                evidence=TrialDevObservationalDecisionEvidenceV1.model_validate(
                    efficacy_payload
                ),
                action=action,
            )
            controls.append(
                _control_row(
                    state=state,
                    identification_status=expected_evidence.identification_status,
                    control_id="efficacy_evidence_error",
                    expected="invalid",
                    grade=grade_portfolio_checkpoint_v1(
                        release_root=root,
                        state=state,
                        submission=efficacy_error,
                    ),
                )
            )
        if expected_evidence.pair_contrasts:
            uncertainty_payload = expected_evidence.model_dump(
                mode="python", exclude={"checksum"}
            )
            pair_contrasts = list(expected_evidence.pair_contrasts)
            first_pair = pair_contrasts[0]
            pair_payload = first_pair.model_dump(mode="python", exclude={"checksum"})
            pair_payload.update(
                {
                    "confidence_half_width": (
                        first_pair.confidence_half_width
                        + expected_evidence.practical_equivalence_margin
                        + 0.01
                    )
                }
            )
            pair_contrasts[0] = type(first_pair).model_validate(pair_payload)
            uncertainty_payload["pair_contrasts"] = tuple(pair_contrasts)
            uncertainty_error = _submission(
                release_root=root,
                state=state,
                index=index,
                evidence=TrialDevObservationalDecisionEvidenceV1.model_validate(
                    uncertainty_payload
                ),
                action=action,
            )
            controls.append(
                _control_row(
                    state=state,
                    identification_status=expected_evidence.identification_status,
                    control_id="pairwise_uncertainty_error",
                    expected="invalid",
                    grade=grade_portfolio_checkpoint_v1(
                        release_root=root,
                        state=state,
                        submission=uncertainty_error,
                    ),
                )
            )
        evidence_reference_error = _submission(
            release_root=root,
            state=state,
            index=index,
            evidence=_mutate_evidence_reference(expected_evidence),
            action=action,
        )
        controls.append(
            _control_row(
                state=state,
                identification_status=expected_evidence.identification_status,
                control_id="evidence_reference_error",
                expected="invalid",
                grade=grade_portfolio_checkpoint_v1(
                    release_root=root,
                    state=state,
                    submission=evidence_reference_error,
                ),
            )
        )
        provenance_error = _submission(
            release_root=root,
            state=state,
            index=index,
            evidence=expected_evidence,
            action=action,
            supporting_evidence_ids=("unknown-evidence",),
        )
        controls.append(
            _control_row(
                state=state,
                identification_status=expected_evidence.identification_status,
                control_id="provenance_error",
                expected="invalid",
                grade=grade_portfolio_checkpoint_v1(
                    release_root=root, state=state, submission=provenance_error
                ),
            )
        )
        supported_signatures = {
            (row.action_id, row.target_asset_id, row.reserve_asset_id)
            for row in supported.supported_actions
        }
        unsupported = next(
            (
                row
                for row in supported.legal_actions
                if (row.action_id, row.target_asset_id, row.reserve_asset_id)
                not in supported_signatures
            ),
            None,
        )
        if unsupported is not None:
            unsupported_submission = _submission(
                release_root=root,
                state=state,
                index=index,
                evidence=expected_evidence,
                action=unsupported,
            )
            controls.append(
                _control_row(
                    state=state,
                    identification_status=expected_evidence.identification_status,
                    control_id="unsupported_action",
                    expected="invalid",
                    grade=grade_portfolio_checkpoint_v1(
                        release_root=root,
                        state=state,
                        submission=unsupported_submission,
                    ),
                )
            )
        if reference.scheduled_studies:
            drifted = reference.scheduled_studies[0].model_copy(
                update={
                    "design_cell_id": f"{reference.scheduled_studies[0].design_cell_id}.invalid"
                }
            )
            design_error = reference.model_copy(
                update={
                    "scheduled_studies": (drifted, *reference.scheduled_studies[1:])
                }
            )
            controls.append(
                _control_row(
                    state=state,
                    identification_status=expected_evidence.identification_status,
                    control_id="scheduled_design_error",
                    expected="invalid",
                    grade=grade_portfolio_checkpoint_v1(
                        release_root=root, state=state, submission=design_error
                    ),
                )
            )
        stale_evidence = expected_evidence.model_copy(
            update={"state_checksum": "0" * 64, "checksum": None}
        )
        stale_selection = reference.selected_action.model_copy(
            update={"state_checksum": "0" * 64, "checksum": None}
        )
        stale = reference.model_copy(
            update={
                "state_checksum": "0" * 64,
                "decision_evidence": stale_evidence,
                "selected_action": stale_selection,
            }
        )
        try:
            grade_portfolio_checkpoint_v1(
                release_root=root, state=state, submission=stale
            )
        except PortfolioSubmissionError:
            controls.append(
                _control_row(
                    state=state,
                    identification_status=expected_evidence.identification_status,
                    control_id="stale_state",
                    expected="rejected_before_grade",
                    grade=None,
                    rejected=True,
                )
            )
        else:
            findings.append(f"stale_state_not_rejected:{state.programme_id}")
    ordered_findings = tuple(sorted(set(findings)))
    control_report = TrialDevPortfolioGraderControlReportV1(
        release_source_identity=catalogue.source_identity,
        participant_view_count=len(catalogue.views),
        controls=tuple(controls),
        findings=ordered_findings,
        status=(
            "pass"
            if not ordered_findings and all(row.detected for row in controls)
            else "fail"
        ),
    )
    if controls_csv is not None:
        path = Path(controls_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            fields = tuple(TrialDevGraderControlV1.model_fields)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                row.model_dump(mode="json") for row in control_report.controls
            )
    return control_report


__all__ = [
    "TrialDevGraderControlV1",
    "TrialDevPortfolioGraderControlReportV1",
    "run_trialdev_portfolio_grader_controls_v1",
]
