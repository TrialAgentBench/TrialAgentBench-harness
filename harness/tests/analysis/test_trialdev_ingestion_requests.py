"""Canonical TrialDev request ingestion tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from trialagentbench_harness.analysis.trialdev_ingestion import (
    TrialDevAnalysisSourceError,
    _canonical_phase_request,
)
from trialagentbench_harness.contracts.core.runs import (
    TrialDevPhaseRequestSummaryV1,
    TrialDevPhaseStepSummaryV1,
)
from trialagentbench_harness.io.json import write_json_model
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentRequestV1


def _request() -> TrialDevelopmentRequestV1:
    return TrialDevelopmentRequestV1(
        scenario_id="scenario-1",
        phase_id="phase1",
        candidate_drug_ids=("drug-a",),
        endpoint_id=None,
        target_sample_size=120,
        follow_up_days=90,
        enrollment_window_days=60,
        site_count_budget=12,
        allocation_ratio="1:1",
        design_cell_id="trialdev.phase1.fixed_final_operating_characteristics.v1",
        interim_policy="fixed_final",
        site_strategy="high_enrolling",
        selection_objective="benefit_risk",
    )


def _summary(request: TrialDevelopmentRequestV1) -> TrialDevPhaseStepSummaryV1:
    return TrialDevPhaseStepSummaryV1(
        program_id="program-1",
        scenario_id=request.scenario_id,
        objective_id="benefit_risk",
        phase_id=request.phase_id,
        request=TrialDevPhaseRequestSummaryV1(
            phase_id=request.phase_id,
            endpoint_id=request.endpoint_id,
            selection_objective=request.selection_objective,
            target_sample_size=request.target_sample_size,
            follow_up_days=request.follow_up_days,
            allocation_ratio=request.allocation_ratio,
            site_count_budget=request.site_count_budget,
            enrollment_window_days=request.enrollment_window_days,
        ),
    )


def _write_request(program_dir: Path, request: TrialDevelopmentRequestV1) -> None:
    phase_dir = program_dir / "agent_workdir" / "phase_phase1"
    phase_dir.mkdir(parents=True)
    write_json_model(phase_dir / "request.json", request)


def test_canonical_phase_request_preserves_full_design_contract(tmp_path: Path) -> None:
    request = _request()
    _write_request(tmp_path, request)

    loaded = _canonical_phase_request(tmp_path, "phase1", _summary(request))

    assert loaded == request
    assert loaded is not None
    assert loaded.allocation_weights == request.allocation_weights
    assert loaded.checksum() == request.checksum()


@pytest.mark.parametrize("canonical_present", [False, True])
def test_canonical_phase_request_rejects_one_sided_records(
    tmp_path: Path,
    canonical_present: bool,
) -> None:
    request = _request()
    if canonical_present:
        _write_request(tmp_path, request)
    summary = None if canonical_present else _summary(request)

    with pytest.raises(TrialDevAnalysisSourceError, match="canonical request and phase summary disagree"):
        _canonical_phase_request(tmp_path, "phase1", summary)


def test_canonical_phase_request_rejects_summary_drift(tmp_path: Path) -> None:
    request = _request()
    _write_request(tmp_path, request)
    summary = _summary(request)
    assert summary.request is not None
    summary.request.target_sample_size += 1

    with pytest.raises(TrialDevAnalysisSourceError, match="canonical request disagrees"):
        _canonical_phase_request(tmp_path, "phase1", summary)
