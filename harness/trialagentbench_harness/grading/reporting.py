"""Compact reporting for deterministic scoring-key grades."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from trialagentbench_harness.io import write_json


def _failure_codes(row: dict[str, object]) -> tuple[str, ...]:
    value = row["failure_codes"]
    if not isinstance(value, list):
        raise TypeError("failure_codes must be a list")
    if not all(isinstance(code, str) for code in value):
        raise TypeError("failure_codes entries must be strings")
    return tuple(value)


def _credit_eligible_route_count(row: dict[str, object]) -> int:
    value = row["credit_eligible_route_count"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TypeError("credit_eligible_route_count must be a positive integer")
    return value


def write_trialeval_grade_summary(
    *,
    rows: list[dict[str, object]],
    output_dir: Path,
) -> None:
    """Write stable machine-readable and human-readable run summaries."""

    if not rows:
        raise ValueError("TrialEval grade summary cannot be empty")
    output = Path(output_dir)
    fieldnames = (
        "item_id",
        "design_tier",
        "design_subtype",
        "assumption_tier",
        "context_tier",
        "credit_eligible_route_count",
        "model",
        "agent_status",
        "turns_used",
        "usable_primary",
        "route_match",
        "obligations_met",
        "result_match",
        "passed",
        "matched_route_id",
        "failure_codes",
        "absolute_error",
        "tolerance_ratio",
        "planning_applicable",
        "planning_valid",
        "planning_achieved_power",
        "planning_power_shortfall",
        "planning_underpowered",
        "planning_proportional_participant_deviation",
        "planning_log_sample_size_ratio",
        "planning_event_shortage",
        "planning_excess_events",
        "planning_excess_participants",
        "planning_participant_shortage",
    )
    write_json(
        output / "summary.json",
        {
            "schema_id": "trialagentbench.trialeval_grade_summary/v1",
            "n_items": len(rows),
            "n_conforming": sum(bool(row["passed"]) for row in rows),
            "n_single_route_items": sum(_credit_eligible_route_count(row) == 1 for row in rows),
            "n_plural_route_items": sum(_credit_eligible_route_count(row) > 1 for row in rows),
            "n_planning_eligible_items": sum(bool(row["planning_applicable"]) for row in rows),
            "n_valid_planning_items": sum(row["planning_valid"] is True for row in rows),
            "items": rows,
        },
    )
    with (output / "results_full.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["failure_codes"] = "|".join(_failure_codes(row))
            writer.writerow(csv_row)

    failures = Counter(code for row in rows for code in _failure_codes(row))
    passed = sum(bool(row["passed"]) for row in rows)
    lines = [
        "# TrialEvalBench grade summary",
        "",
        f"- Declared denominator: **{len(rows)}**",
        f"- Credit-eligible primary analyses: **{passed}/{len(rows)}**",
        f"- Usable primary submissions: **{sum(bool(row['usable_primary']) for row in rows)}/{len(rows)}**",
        f"- Exact qualified-route matches: **{sum(bool(row['route_match']) for row in rows)}/{len(rows)}**",
        f"- Single-route items: **{sum(_credit_eligible_route_count(row) == 1 for row in rows)}/{len(rows)}**",
        f"- Plural-route items: **{sum(_credit_eligible_route_count(row) > 1 for row in rows)}/{len(rows)}**",
        f"- Valid planning submissions: **{sum(row['planning_valid'] is True for row in rows)}/"
        f"{sum(bool(row['planning_applicable']) for row in rows)} eligible**",
        "",
        "## Failure codes",
        "",
        "| code | count |",
        "|---|---:|",
    ]
    lines.extend(f"| `{code}` | {count} |" for code, count in sorted(failures.items()))
    (output / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["write_trialeval_grade_summary"]
