"""Terminal payoff is reconstructed from fixed truth and unavoidable patient costs."""

import json
import math
from pathlib import Path

import pytest

from trialagentbench_verifier.io import sha256_file
from trialagentbench_verifier.resource_accounting import (
    reconstruct_trialdev_terminal_value,
    verify_trialdev_terminal_value,
)
from trialagentbench_verifier.response import TrialDevFrozenProgrammeValue


def test_terminal_comparison_accepts_independent_summation_roundoff() -> None:
    expected = TrialDevFrozenProgrammeValue(
        decision="stop",
        approved=False,
        clinical_value_usd=0,
        safety_cost_usd=0,
        study_cost_usd=7_548_000,
        time_cost_usd=1_500_000,
        net_value_usd=-9_048_000,
    )
    observed = expected.model_copy(update={"study_cost_usd": 7_547_999.999999999})
    verify_trialdev_terminal_value(expected, observed)


@pytest.mark.parametrize(
    "field",
    ["clinical_value_usd", "safety_cost_usd", "study_cost_usd", "time_cost_usd", "net_value_usd"],
)
def test_terminal_comparison_rejects_monetary_changes_without_relative_tolerance(field: str) -> None:
    expected = TrialDevFrozenProgrammeValue(
        decision="advance",
        approved=True,
        clinical_value_usd=120_000_000,
        safety_cost_usd=10_000_000,
        study_cost_usd=8_000_000,
        time_cost_usd=1_500_000,
        net_value_usd=100_500_000,
    )
    observed = expected.model_copy(update={field: getattr(expected, field) + 0.0001})
    with pytest.raises(ValueError, match="Terminal value differs"):
        verify_trialdev_terminal_value(expected, observed)


def test_terminal_comparison_rejects_absent_or_different_decisions() -> None:
    expected = TrialDevFrozenProgrammeValue(
        decision="stop",
        approved=False,
        clinical_value_usd=0,
        safety_cost_usd=0,
        study_cost_usd=0,
        time_cost_usd=0,
        net_value_usd=0,
    )
    for observed in (None, expected.model_copy(update={"decision": "advance"})):
        with pytest.raises(ValueError, match="Terminal value differs"):
            verify_trialdev_terminal_value(expected, observed)


def _case(root: Path, decision: str = "stop", approved: bool = True) -> tuple[Path, Path, Path]:
    pool = root / "pool.json"
    pool.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "asset_id": "candidate",
                        "dose": "standard",
                        "outcome_unit": "DAS28",
                        "outcome_mean_difference": -1,
                        "safety_risk_difference": -0.01,
                    }
                ]
            }
        )
    )
    configuration = root / "configuration.json"
    configuration.write_text(
        json.dumps(
            {
                "pool_manifest_sha256": sha256_file(pool),
                "budget_usd": 2_000_000,
                "preferences": {
                    "outcome_unit": "DAS28",
                    "benefit_unit": 1,
                    "annual_profit_per_benefit_unit_usd": 250_000_000,
                    "annual_cost_per_safety_percentage_point_usd": 20_000_000,
                    "discount_rate_per_year": 0.1,
                    "exclusivity_end_day": 6575,
                    "regulatory_review_days": 365,
                    "programme_cost_per_day_usd": 15_000,
                },
            }
        )
    )
    ledger = root / "ledger.json"
    terminal = {"decision": decision, "programme_day": 100}
    if decision == "advance":
        terminal.update(
            asset_id="candidate",
            dose="standard",
            regulatory_review={
                "completed_treated": 300,
                "completed_control": 300,
                "benefit_z_statistic": 2.5 if approved else 1.0,
                "approved": approved,
            },
        )
    ledger.write_text(
        json.dumps(
            {
                "budget_usd": 2_000_000,
                "resource_day": 100,
                "pool_manifest_sha256": sha256_file(pool),
                "terminal": terminal,
                "enrollments": [
                    {
                        "study_id": "safety",
                        "screened": 20,
                        "patients": [{"participant_id": i, "enrollment_day": 75} for i in range(20)],
                    }
                ],
                "recruitment_commitments": [
                    {
                        "study_id": "safety",
                        "phase": "early_safety",
                        "sites": 2,
                        "follow_up_days": 28,
                        "assessment_days": [0, 28],
                        "surveillance_days": [0, 28],
                    }
                ],
                # These are deliberately untrusted: terminal obligations come from quantities.
                "settlement_usd": 0,
                "receipts": [{"cost": {"accrued_usd": 0, "remaining_liability_usd": 0}}],
            }
        )
    )
    return ledger, configuration, pool


@pytest.mark.parametrize(("decision", "approved"), [("stop", False), ("advance", True), ("advance", False)])
def test_terminal_cost_includes_future_obligations_and_signed_safety(
    tmp_path: Path, decision: str, approved: bool
) -> None:
    ledger, configuration, pool = _case(tmp_path, decision, approved)
    result = reconstruct_trialdev_terminal_value(ledger_path=ledger, configuration_path=configuration, pool_path=pool)
    rate = math.log1p(0.1)
    years = (math.exp(-rate * 465 / 365.25) - math.exp(-rate * 6575 / 365.25)) / rate
    assert result.approved == approved
    assert result.study_cost_usd == 1_453_600
    assert result.time_cost_usd == 1_500_000
    assert result.clinical_value_usd == pytest.approx(250_000_000 * years if approved else 0)
    assert result.safety_cost_usd == pytest.approx(-20_000_000 * years if approved else 0)
    assert result.net_value_usd == pytest.approx(270_000_000 * years * approved - 2_953_600)


@pytest.mark.parametrize(
    "defect", ["patient_reuse", "future_enrollment", "clock", "pool", "budget", "study_census", "stop_selection"]
)
def test_terminal_accounting_rejects_changed_boundary_inputs(tmp_path: Path, defect: str) -> None:
    ledger, configuration, pool = _case(tmp_path)
    payload = json.loads(ledger.read_text())
    if defect == "patient_reuse":
        payload["enrollments"][0]["patients"][1]["participant_id"] = 0
    elif defect == "future_enrollment":
        payload["enrollments"][0]["patients"][0]["enrollment_day"] = 101
    elif defect == "clock":
        payload["resource_day"] = 99
    elif defect == "pool":
        payload["pool_manifest_sha256"] = "0" * 64
    elif defect == "budget":
        payload["budget_usd"] = 3_000_000
    elif defect == "study_census":
        payload["enrollments"] = []
    else:
        payload["terminal"]["asset_id"] = "candidate"
    ledger.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        reconstruct_trialdev_terminal_value(ledger_path=ledger, configuration_path=configuration, pool_path=pool)
