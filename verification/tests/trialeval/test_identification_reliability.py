"""Independent A4 reliability tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trialagentbench_validation.cli import main
from trialagentbench_validation.trialeval.identification_reliability import (
    verify_identification_reliability,
)


def _set_world(
    world_id: str, cell: str, limits: tuple[float, float]
) -> dict[str, object]:
    primary_estimator = {
        "TE-S04-A4": "observed:tau_bounds_bounded_deviation",
        "TE-S06-A4": "observed:validated_endpoint_bounded_deviation",
    }[cell]
    alternative_estimator = {
        "TE-S04-A4": "observed:tau_bounds_worst_case",
        "TE-S06-A4": "observed:validated_endpoint_worst_case",
    }[cell]
    return {
        "world_id": world_id,
        "regime_cell_id": cell,
        "identified_set_records": [
            {
                "regime_cell_id": cell,
                "method_signature_id": f"{cell}:bounded",
                "analysis_role": "required_primary",
                "estimator_id": primary_estimator,
                "sensitivity_parameter": 0.2,
                "reference_value": 0.1,
                "status": "success",
                "set_low": limits[0],
                "set_high": limits[1],
            },
            {
                "regime_cell_id": cell,
                "method_signature_id": f"{cell}:worst-case",
                "analysis_role": "credit_eligible_primary_alternative",
                "estimator_id": alternative_estimator,
                "sensitivity_parameter": None,
                "reference_value": 0.1,
                "status": "success",
                "set_low": -1.0,
                "set_high": 1.0,
            },
        ],
    }


def _sequential_world(
    world_id: str,
    *,
    estimate: float,
    interval: tuple[float, float],
    stopped_early: bool,
) -> dict[str, object]:
    return {
        "world_id": world_id,
        "regime_cell_id": "TE-S09-A4",
        "point_records": [
            {
                "regime_cell_id": "TE-S09-A4",
                "method_signature_id": "TE-S09-A4:repeated",
                "estimator_id": "observed:group_sequential_adjusted",
                "reference_value": 0.1,
                "status": "success",
                "estimate": estimate,
                "interval_low": interval[0],
                "interval_high": interval[1],
            }
        ],
        "group_sequential_monitoring": {
            "regime_cell_id": "TE-S09-A4",
            "analysis_look_index": 0 if stopped_early else 1,
            "stopped_early": stopped_early,
            "stopping_regime": "efficacy_stop" if stopped_early else "final_analysis",
        },
    }


def _write_inputs(
    root: Path,
    *,
    s04_coverage: float = 1.0,
    s04_bounded_width: float = 0.3,
) -> tuple[Path, Path]:
    worlds = (
        _set_world("s04-1", "TE-S04-A4", (-0.1, 0.2)),
        _set_world("s04-2", "TE-S04-A4", (0.0, 0.3)),
        _set_world("s06-1", "TE-S06-A4", (-0.2, 0.2)),
        _set_world("s06-2", "TE-S06-A4", (-0.1, 0.3)),
        _sequential_world(
            "s09-1", estimate=0.08, interval=(-0.02, 0.18), stopped_early=True
        ),
        _sequential_world(
            "s09-2", estimate=0.12, interval=(0.02, 0.22), stopped_early=False
        ),
    )
    world_path = root / "world_records.jsonl"
    world_path.write_text(
        "".join(json.dumps(world, sort_keys=True) + "\n" for world in worlds),
        encoding="utf-8",
    )
    operating = {
        "identified_sets": [
            result
            for cell, primary_estimator, alternative_estimator in (
                (
                    "TE-S04-A4",
                    "observed:tau_bounds_bounded_deviation",
                    "observed:tau_bounds_worst_case",
                ),
                (
                    "TE-S06-A4",
                    "observed:validated_endpoint_bounded_deviation",
                    "observed:validated_endpoint_worst_case",
                ),
            )
            for result in (
                {
                    "regime_cell_id": cell,
                    "method_signature_id": f"{cell}:bounded",
                    "analysis_role": "required_primary",
                    "estimator_id": primary_estimator,
                    "sensitivity_parameter": 0.2,
                    "n_worlds": 2,
                    "n_success": 2,
                    "failure_rate": 0.0,
                    "conditional_set_coverage": (
                        s04_coverage if cell == "TE-S04-A4" else 1.0
                    ),
                    "unconditional_set_coverage": (
                        s04_coverage if cell == "TE-S04-A4" else 1.0
                    ),
                    "mean_set_width": s04_bounded_width if cell == "TE-S04-A4" else 0.4,
                },
                {
                    "regime_cell_id": cell,
                    "method_signature_id": f"{cell}:worst-case",
                    "analysis_role": "credit_eligible_primary_alternative",
                    "estimator_id": alternative_estimator,
                    "sensitivity_parameter": None,
                    "n_worlds": 2,
                    "n_success": 2,
                    "failure_rate": 0.0,
                    "conditional_set_coverage": 1.0,
                    "unconditional_set_coverage": 1.0,
                    "mean_set_width": 2.0,
                },
            )
        ],
        "point_methods": [
            {
                "regime_cell_id": "TE-S09-A4",
                "method_signature_id": "TE-S09-A4:repeated",
                "estimator_id": "observed:group_sequential_adjusted",
                "component_id": None,
                "n_worlds": 2,
                "n_success": 2,
                "failure_rate": 0.0,
                "bias": 0.0,
                "rmse": 0.02,
                "conditional_coverage": 1.0,
                "unconditional_coverage": 1.0,
                "mean_interval_width": 0.2,
            }
        ],
        "group_sequential_monitoring": [
            {
                "regime_cell_id": "TE-S09-A4",
                "n_worlds": 2,
                "efficacy_stop_worlds": 1,
                "final_analysis_worlds": 1,
            }
        ],
    }
    operating_path = root / "operating_characteristics.json"
    operating_path.write_text(json.dumps(operating), encoding="utf-8")
    return world_path, operating_path


def test_identification_reliability_recomputes_all_a4_conclusions(
    tmp_path: Path,
) -> None:
    worlds, operating = _write_inputs(tmp_path)

    results = verify_identification_reliability(
        world_records_path=worlds,
        operating_characteristics_path=operating,
    )

    assert [row.regime_cell_id for row in results] == [
        "TE-S04-A4",
        "TE-S04-A4",
        "TE-S06-A4",
        "TE-S06-A4",
        "TE-S09-A4",
    ]
    assert results[0].conclusion_type == "identified_range"
    assert results[0].analysis_role == "prespecified_bounded_departure"
    assert results[0].sensitivity_parameter == 0.2
    assert results[0].coverage == 1.0
    assert results[0].mean_width == pytest.approx(0.3)
    assert results[1].analysis_role == "unrestricted_worst_case"
    assert results[1].mean_width == pytest.approx(2.0)
    assert results[4].conclusion_type == "repeated_interval"
    assert results[4].bias == pytest.approx(0.0)
    assert results[4].rmse == pytest.approx(0.02)
    assert results[4].rmse_low is not None
    assert results[4].rmse_high is not None
    assert results[4].rmse_low <= results[4].rmse <= results[4].rmse_high
    assert results[4].early_stop_rate == 0.5


def test_identification_reliability_rejects_summary_drift(tmp_path: Path) -> None:
    worlds, operating = _write_inputs(tmp_path, s04_coverage=0.5)

    with pytest.raises(ValueError, match="summary disagrees"):
        verify_identification_reliability(
            world_records_path=worlds,
            operating_characteristics_path=operating,
        )


def test_identification_reliability_rejects_blank_world_records(tmp_path: Path) -> None:
    worlds, operating = _write_inputs(tmp_path)
    worlds.write_text(worlds.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Blank qualification world"):
        verify_identification_reliability(
            world_records_path=worlds,
            operating_characteristics_path=operating,
        )


def test_identification_reliability_rejects_uninformative_bounded_range(
    tmp_path: Path,
) -> None:
    worlds, operating = _write_inputs(tmp_path, s04_bounded_width=2.0)
    payload = json.loads(operating.read_text(encoding="utf-8"))
    for row in payload["identified_sets"]:
        if (
            row["regime_cell_id"] == "TE-S04-A4"
            and row["analysis_role"] == "required_primary"
        ):
            row["mean_set_width"] = 2.0
    operating.write_text(json.dumps(payload), encoding="utf-8")
    world_payloads = [
        json.loads(line)
        for line in worlds.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for world in world_payloads:
        if world["regime_cell_id"] != "TE-S04-A4":
            continue
        bounded = world["identified_set_records"][0]
        bounded["set_low"] = -1.0
        bounded["set_high"] = 1.0
    worlds.write_text(
        "".join(json.dumps(world, sort_keys=True) + "\n" for world in world_payloads),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not narrower"):
        verify_identification_reliability(
            world_records_path=worlds,
            operating_characteristics_path=operating,
        )


def test_identification_reliability_rejects_one_uninformative_trial(
    tmp_path: Path,
) -> None:
    worlds, operating = _write_inputs(tmp_path)
    world_payloads = [
        json.loads(line) for line in worlds.read_text(encoding="utf-8").splitlines()
    ]
    s04_worlds = [
        world for world in world_payloads if world["regime_cell_id"] == "TE-S04-A4"
    ]
    s04_worlds[0]["identified_set_records"][0]["set_low"] = -1.1
    s04_worlds[0]["identified_set_records"][0]["set_high"] = 1.1
    s04_worlds[1]["identified_set_records"][0]["set_low"] = 0.05
    s04_worlds[1]["identified_set_records"][0]["set_high"] = 0.15
    worlds.write_text(
        "".join(json.dumps(world, sort_keys=True) + "\n" for world in world_payloads),
        encoding="utf-8",
    )
    payload = json.loads(operating.read_text(encoding="utf-8"))
    for row in payload["identified_sets"]:
        if (
            row["regime_cell_id"] == "TE-S04-A4"
            and row["analysis_role"] == "required_primary"
        ):
            row["mean_set_width"] = 1.15
    operating.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not narrow"):
        verify_identification_reliability(
            world_records_path=worlds,
            operating_characteristics_path=operating,
        )


def test_identification_reliability_cli_writes_tidy_results(tmp_path: Path) -> None:
    worlds, operating = _write_inputs(tmp_path)
    output = tmp_path / "identification_reliability.csv"

    exit_code = main(
        [
            "trialeval-identification-reliability",
            "--world-records",
            str(worlds),
            "--operating-characteristics",
            str(operating),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("series_id,regime_cell_id,conclusion_type")
    assert len(lines) == 6
