"""Independently reconstruct TrialDev mechanism-relative policy-value evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrialDevPolicyValueAuditV1(BaseModel):
    """Independent result for one complete policy-value qualification study."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.validation.trialdev_policy_value_audit/v1"] = (
        "trialagentbench.validation.trialdev_policy_value_audit/v1"
    )
    source_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    world_count: int = Field(ge=1)
    candidate_record_count: int = Field(ge=1)
    cell_count: int = Field(ge=1)
    reconstructed_numeric_count: int = Field(ge=1)
    findings: tuple[str, ...]
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        """Bind pass status to a sorted unique empty finding set."""

        if self.findings != tuple(sorted(set(self.findings))):
            raise ValueError("Policy-value audit findings must be sorted and unique.")
        if (self.status == "pass") != (not self.findings):
            raise ValueError("Policy-value audit status disagrees with its findings.")
        return self


def _rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Policy-value audit input is empty: {path}")
    return rows


def _key(row: dict[str, str]) -> tuple[str, int, int, int, int]:
    return (
        row["scenario_id"],
        int(row["information_size"]),
        int(row["resource_budget_units"]),
        int(row["world_index"]),
        int(row["world_seed"]),
    )


def _action_value(
    *,
    action_id: str,
    candidates: dict[str, dict[str, str]],
    budget: int,
    early_study_units: int,
    proof_of_concept_units: int,
    confirmation_units: int,
    late_switch_budget_units: int,
) -> tuple[float, float]:
    if action_id == "withhold_selection":
        return 0.0, 0.0
    parts = action_id.split(":")
    if len(parts) != 3 or parts[0] != "select_lead_and_reserve":
        raise ValueError(f"Unknown policy-value action {action_id!r}.")
    lead = candidates[parts[1]]
    reserve = candidates[parts[2]]
    lead_efficacy = float(lead["efficacy_probability"])
    lead_safety = float(lead["early_safety_pass_probability"])
    reserve_efficacy = float(reserve["efficacy_probability"])
    reserve_safety = float(reserve["early_safety_pass_probability"])
    early_switch = 1.0 - lead_safety
    late_switch = (
        lead_safety * (1.0 - lead_efficacy)
        if budget >= late_switch_budget_units
        else 0.0
    )
    reserve_path = early_switch + late_switch
    success = (
        lead_safety * lead_efficacy + reserve_path * reserve_safety * reserve_efficacy
    )
    resources = (
        2.0 * early_study_units
        + proof_of_concept_units * lead_safety
        + confirmation_units * lead_safety * lead_efficacy
        + proof_of_concept_units * reserve_path * reserve_safety
        + confirmation_units * reserve_path * reserve_safety * reserve_efficacy
    )
    return success, resources


def _best(
    *,
    action_ids: tuple[str, ...],
    candidates: dict[str, dict[str, str]],
    budget: int,
    schedule: tuple[int, int, int, int],
) -> tuple[str, float, float]:
    values = tuple(
        (
            action,
            *_action_value(
                action_id=action,
                candidates=candidates,
                budget=budget,
                early_study_units=schedule[0],
                proof_of_concept_units=schedule[1],
                confirmation_units=schedule[2],
                late_switch_budget_units=schedule[3],
            ),
        )
        for action in action_ids
    )
    if not values:
        raise ValueError("Policy-value audit cannot select from an empty action set.")
    return min(values, key=lambda row: (-row[1], row[2], row[0]))


def _wilson_interval(
    *, events: int, total: int, confidence_level: float
) -> tuple[float, float, float]:
    if total < 1 or not 0 <= events <= total:
        raise ValueError("Wilson interval requires 0 <= events <= total.")
    estimate = events / total
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total**2))
        / denominator
    )
    return estimate, max(0.0, center - half_width), min(1.0, center + half_width)


def _bootstrap_mean_interval(
    values: npt.NDArray[np.float64],
    *,
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    batch_size = 1_000
    for start in range(0, resamples, batch_size):
        stop = min(resamples, start + batch_size)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def audit_trialdev_policy_value_v1(
    *, policy_value_root: Path
) -> TrialDevPolicyValueAuditV1:
    """Recompute every action value, oracle, comparator, and cell mean."""

    root = Path(policy_value_root).resolve(strict=True)
    report = json.loads((root / "policy_value_report.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") != "pass":
        raise ValueError("Policy-value audit requires a passing construction report.")
    worlds = _rows(root / "policy_value_worlds.csv")
    candidate_rows = _rows(root / "policy_value_candidates.csv")
    cells = _rows(root / "policy_value_cells.csv")
    by_world: dict[tuple[str, int, int, int, int], dict[str, dict[str, str]]] = (
        defaultdict(dict)
    )
    for row in candidate_rows:
        key = _key(row)
        asset_id = row["asset_id"]
        if asset_id in by_world[key]:
            raise ValueError(
                f"Duplicate policy-value candidate {asset_id!r} for {key!r}."
            )
        by_world[key][asset_id] = row
    findings: list[str] = []
    numeric_count = 0
    tolerance = 1e-12
    schedule = (
        int(report["early_study_units"]),
        int(report["proof_of_concept_units"]),
        int(report["confirmation_units"]),
        int(report["late_switch_budget_units"]),
    )
    efficacy_minimum = float(report["efficacy_minimum"])
    confidence_level = float(report["confidence_level"])
    bootstrap_resamples = int(report["regret_bootstrap_resamples"])
    grouped_worlds: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    mechanism_by_public_asset: dict[tuple[str, int, str], set[tuple[str, str]]] = (
        defaultdict(set)
    )
    candidate_by_matched_budget: dict[
        tuple[str, int, int, int, str], list[dict[str, str]]
    ] = defaultdict(list)
    for row in candidate_rows:
        mechanism_key = (
            row["scenario_id"],
            int(row["information_size"]),
            row["asset_id"],
        )
        mechanism_by_public_asset[mechanism_key].add(
            (row["efficacy_probability"], row["early_safety_pass_probability"])
        )
        candidate_key = (
            row["scenario_id"],
            int(row["information_size"]),
            int(row["world_index"]),
            int(row["world_seed"]),
            row["asset_id"],
        )
        candidate_by_matched_budget[candidate_key].append(row)
    for mechanism_key, mechanisms in mechanism_by_public_asset.items():
        if len(mechanisms) != 3:
            findings.append(f"asset_labels_not_exchangeable:{mechanism_key!r}")
    matched_fields = (
        "efficacy_probability",
        "early_safety_pass_probability",
        "efficacy_estimate",
        "efficacy_lower_bound",
        "efficacy_upper_bound",
        "utility_estimate",
        "utility_lower_bound",
        "utility_upper_bound",
    )
    for candidate_key, rows in candidate_by_matched_budget.items():
        if len(rows) != 2 or any(
            rows[0][field] != rows[1][field] for field in matched_fields
        ):
            findings.append(f"budget_arms_not_evidence_matched:{candidate_key!r}")
    for row in worlds:
        key = _key(row)
        candidates = by_world.get(key, {})
        if len(candidates) != 3:
            findings.append(f"candidate_census_disagreement:{key!r}")
            continue
        budget = key[2]
        supported = tuple(
            value for value in row["supported_action_ids"].split("|") if value
        )
        all_pairs = tuple(
            f"select_lead_and_reserve:{lead}:{reserve}"
            for lead in candidates
            for reserve in candidates
            if lead != reserve
        )
        oracle = _best(
            action_ids=all_pairs,
            candidates=candidates,
            budget=budget,
            schedule=schedule,
        )
        supported_values = tuple(
            (
                action,
                *_action_value(
                    action_id=action,
                    candidates=candidates,
                    budget=budget,
                    early_study_units=schedule[0],
                    proof_of_concept_units=schedule[1],
                    confirmation_units=schedule[2],
                    late_switch_budget_units=schedule[3],
                ),
            )
            for action in supported
        )
        best_supported = min(
            supported_values, key=lambda value: (-value[1], value[2], value[0])
        )
        worst_supported = min(
            supported_values, key=lambda value: (value[1], value[2], value[0])
        )
        point_order = tuple(
            item["asset_id"]
            for item in sorted(
                (
                    item
                    for item in candidates.values()
                    if float(item["efficacy_estimate"]) >= efficacy_minimum
                ),
                key=lambda item: (-float(item["utility_estimate"]), item["asset_id"]),
            )
        )
        point_action = (
            f"select_lead_and_reserve:{point_order[0]}:{point_order[1]}"
            if len(point_order) >= 2
            else "withhold_selection"
        )
        alphabetical = tuple(sorted(candidates))
        alphabetical_action = (
            f"select_lead_and_reserve:{alphabetical[0]}:{alphabetical[1]}"
        )
        point = _action_value(
            action_id=point_action,
            candidates=candidates,
            budget=budget,
            early_study_units=schedule[0],
            proof_of_concept_units=schedule[1],
            confirmation_units=schedule[2],
            late_switch_budget_units=schedule[3],
        )
        alphabetical_value = _action_value(
            action_id=alphabetical_action,
            candidates=candidates,
            budget=budget,
            early_study_units=schedule[0],
            proof_of_concept_units=schedule[1],
            confirmation_units=schedule[2],
            late_switch_budget_units=schedule[3],
        )
        expected = {
            "oracle_terminal_success_probability": oracle[1],
            "best_supported_terminal_success_probability": best_supported[1],
            "worst_supported_terminal_success_probability": worst_supported[1],
            "adjusted_point_terminal_success_probability": point[0],
            "alphabetical_terminal_success_probability": alphabetical_value[0],
            "oracle_expected_resource_units": oracle[2],
            "best_supported_expected_resource_units": best_supported[2],
            "worst_supported_expected_resource_units": worst_supported[2],
            "adjusted_point_expected_resource_units": point[1],
            "alphabetical_expected_resource_units": alphabetical_value[1],
        }
        if row["oracle_action_id"] != oracle[0]:
            findings.append(f"oracle_action_disagreement:{key!r}")
        expected_support = oracle[0] in supported
        if (row["oracle_action_supported"] == "True") != expected_support:
            findings.append(f"oracle_support_disagreement:{key!r}")
        for field, value in expected.items():
            numeric_count += 1
            if not math.isclose(
                float(row[field]), value, rel_tol=0.0, abs_tol=tolerance
            ):
                findings.append(f"numeric_disagreement:{field}:{key!r}")
        grouped_worlds[(key[0], key[1], key[2])].append(row)
    report_cells = report.get("cells")
    if not isinstance(report_cells, list):
        raise ValueError("Policy-value report lacks cells.")
    report_by_cell = {
        (
            str(row["scenario_id"]),
            int(row["information_size"]),
            int(row["resource_budget_units"]),
        ): row
        for row in report_cells
        if isinstance(row, dict)
    }
    for row in cells:
        cell_key = (
            row["scenario_id"],
            int(row["information_size"]),
            int(row["resource_budget_units"]),
        )
        world_rows = grouped_worlds.get(cell_key, [])
        report_row = report_by_cell.get(cell_key)
        if not world_rows or report_row is None:
            findings.append(f"cell_census_disagreement:{cell_key!r}")
            continue
        mean_fields = (
            "oracle_terminal_success_probability",
            "best_supported_terminal_success_probability",
            "worst_supported_terminal_success_probability",
            "adjusted_point_terminal_success_probability",
            "alphabetical_terminal_success_probability",
            "oracle_expected_resource_units",
            "best_supported_expected_resource_units",
        )
        for field in mean_fields:
            mean = sum(float(world[field]) for world in world_rows) / len(world_rows)
            numeric_count += 2
            if not math.isclose(
                float(row[field]), mean, rel_tol=0.0, abs_tol=tolerance
            ):
                findings.append(f"cell_csv_mean_disagreement:{field}:{cell_key!r}")
            if not math.isclose(
                float(report_row[field]), mean, rel_tol=0.0, abs_tol=tolerance
            ):
                findings.append(f"cell_report_mean_disagreement:{field}:{cell_key!r}")
        for strategy, value_field in (
            ("best_supported", "best_supported_terminal_success_probability"),
            ("worst_supported", "worst_supported_terminal_success_probability"),
            ("adjusted_point", "adjusted_point_terminal_success_probability"),
            ("alphabetical", "alphabetical_terminal_success_probability"),
        ):
            expected_regret = sum(
                float(world["oracle_terminal_success_probability"])
                - float(world[value_field])
                for world in world_rows
            ) / len(world_rows)
            field = f"{strategy}_regret"
            numeric_count += 2
            if not math.isclose(
                float(row[field]), expected_regret, rel_tol=0.0, abs_tol=tolerance
            ):
                findings.append(f"cell_csv_regret_disagreement:{field}:{cell_key!r}")
            if not math.isclose(
                float(report_row[field]),
                expected_regret,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                findings.append(f"cell_report_regret_disagreement:{field}:{cell_key!r}")
        support_count = sum(
            world["oracle_action_supported"] == "True" for world in world_rows
        )
        support = _wilson_interval(
            events=support_count,
            total=len(world_rows),
            confidence_level=confidence_level,
        )
        for field, expected_value in zip(
            (
                "oracle_action_supported_rate",
                "oracle_action_supported_lower",
                "oracle_action_supported_upper",
            ),
            support,
            strict=True,
        ):
            numeric_count += 2
            if not math.isclose(
                float(row[field]), expected_value, rel_tol=0.0, abs_tol=tolerance
            ):
                findings.append(f"cell_csv_interval_disagreement:{field}:{cell_key!r}")
            if not math.isclose(
                float(report_row[field]), expected_value, rel_tol=0.0, abs_tol=tolerance
            ):
                findings.append(
                    f"cell_report_interval_disagreement:{field}:{cell_key!r}"
                )
        cell_seed = int(
            hashlib.sha256(
                f"trialdev-policy-value-ci-v1:{cell_key[0]}:{cell_key[1]}:{cell_key[2]}".encode()
            ).hexdigest()[:8],
            16,
        )
        for index, (strategy, value_field) in enumerate(
            (
                ("best_supported", "best_supported_terminal_success_probability"),
                ("worst_supported", "worst_supported_terminal_success_probability"),
                ("adjusted_point", "adjusted_point_terminal_success_probability"),
                ("alphabetical", "alphabetical_terminal_success_probability"),
            )
        ):
            values = np.asarray(
                [
                    float(world["oracle_terminal_success_probability"])
                    - float(world[value_field])
                    for world in world_rows
                ],
                dtype=np.float64,
            )
            lower, upper = _bootstrap_mean_interval(
                values,
                seed=cell_seed + index,
                resamples=bootstrap_resamples,
            )
            for suffix, expected_value in (("lower", lower), ("upper", upper)):
                field = f"{strategy}_regret_{suffix}"
                numeric_count += 2
                if not math.isclose(
                    float(row[field]), expected_value, rel_tol=0.0, abs_tol=tolerance
                ):
                    findings.append(
                        f"cell_csv_interval_disagreement:{field}:{cell_key!r}"
                    )
                if not math.isclose(
                    float(report_row[field]),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=tolerance,
                ):
                    findings.append(
                        f"cell_report_interval_disagreement:{field}:{cell_key!r}"
                    )
    ordered = tuple(sorted(set(findings)))
    source_identity = report.get("source_identity")
    if not isinstance(source_identity, str):
        raise ValueError("Policy-value report lacks a source identity.")
    return TrialDevPolicyValueAuditV1(
        source_identity=source_identity,
        world_count=len(worlds),
        candidate_record_count=len(candidate_rows),
        cell_count=len(cells),
        reconstructed_numeric_count=numeric_count,
        findings=ordered,
        status="fail" if ordered else "pass",
    )


__all__ = ["TrialDevPolicyValueAuditV1", "audit_trialdev_policy_value_v1"]
