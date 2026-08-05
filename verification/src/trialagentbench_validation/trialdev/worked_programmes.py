"""Independent reconstruction of public TrialDev worked programmes."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from pydantic.types import JsonValue
from typing_extensions import Self  # noqa: UP035

StreamId = Literal["single_asset_development", "bounded_portfolio_reallocation"]
ActionSignature = tuple[str, str | None, str | None]

_JSON_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_CHECKPOINTS: dict[StreamId, tuple[str, ...]] = {
    "single_asset_development": (
        "observational_review",
        "early_safety_study",
        "proof_of_concept",
        "confirmation",
    ),
    "bounded_portfolio_reallocation": (
        "observational_review",
        "joint_early_study_review",
        "lead_proof_of_concept_review",
        "promoted_reserve_proof_of_concept_review",
        "confirmation",
    ),
}
_REQUIRED_LANES: dict[tuple[StreamId, str], tuple[str, ...]] = {
    ("single_asset_development", "observational_review"): (
        "asset_nomination",
        "phase_analysis",
    ),
    ("single_asset_development", "early_safety_study"): (
        "phase_design",
        "phase_analysis",
        "safety_gate",
        "decision_action",
    ),
    ("single_asset_development", "proof_of_concept"): (
        "phase_design",
        "phase_analysis",
        "decision_action",
    ),
    ("single_asset_development", "confirmation"): (
        "phase_design",
        "phase_analysis",
        "decision_action",
    ),
    ("bounded_portfolio_reallocation", "observational_review"): (
        "asset_nomination",
        "phase_analysis",
        "portfolio_allocation",
        "resource_feasibility",
    ),
    ("bounded_portfolio_reallocation", "joint_early_study_review"): (
        "phase_design",
        "phase_analysis",
        "safety_gate",
        "decision_action",
        "portfolio_allocation",
        "resource_feasibility",
    ),
    ("bounded_portfolio_reallocation", "lead_proof_of_concept_review"): (
        "phase_design",
        "phase_analysis",
        "decision_action",
        "portfolio_allocation",
        "resource_feasibility",
    ),
    ("bounded_portfolio_reallocation", "promoted_reserve_proof_of_concept_review"): (
        "phase_design",
        "phase_analysis",
        "decision_action",
        "portfolio_allocation",
        "resource_feasibility",
    ),
    ("bounded_portfolio_reallocation", "confirmation"): (
        "phase_design",
        "phase_analysis",
        "decision_action",
        "portfolio_allocation",
        "resource_feasibility",
    ),
}
_CAPABILITY_CHECKS: dict[str, tuple[str, ...]] = {
    "evidence_validity": ("evidence_integrity", "method_eligibility"),
    "identification_and_uncertainty": (
        "identification_status",
        "uncertainty_qualification",
    ),
    "policy_compatibility": ("policy_conclusion_compatibility",),
    "safety": ("safety_evidence",),
    "temporal_coherence": ("transition_legality", "history_immutability"),
    "workflow_execution": ("required_output_presence", "workflow_completion"),
    "action_admissibility": ("selected_action_membership",),
}


class TrialDevWorkedProgrammeVerificationV1(BaseModel):
    """Independent verification result for two public worked trajectories."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.validation.trialdev_worked_programmes/v1"] = (
        "trialagentbench.validation.trialdev_worked_programmes/v1"
    )
    source_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_action_graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    programme_count: int = Field(ge=0)
    checkpoint_count: int = Field(ge=0)
    evidence_artifact_count: int = Field(ge=0)
    reconstructed_supported_set_count: int = Field(ge=0)
    terminal_dispositions: dict[StreamId, str]
    graph_node_counts: dict[StreamId, int]
    graph_edge_counts: dict[StreamId, int]
    findings: tuple[str, ...]
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        """Bind pass status to a complete finding-free reconstruction."""

        if self.findings != tuple(sorted(set(self.findings))):
            raise ValueError("Worked-programme findings must be sorted and unique.")
        if (self.status == "pass") != (not self.findings):
            raise ValueError("Worked-programme status disagrees with its findings.")
        return self


def _object(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    return value


def _array(value: JsonValue, label: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array.")
    return value


def _text(value: JsonValue, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string.")
    return value


def _optional_text(value: JsonValue, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _number(value: JsonValue, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be finite numeric evidence.")
    return float(value)


def _boolean(value: JsonValue, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean.")
    return value


def _sha(value: JsonValue, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return text


def _prune_none(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: _prune_none(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, list):
        return [_prune_none(item) for item in value]
    return value


def _record_checksum(record: dict[str, JsonValue]) -> str:
    payload = {key: value for key, value in record.items() if key != "checksum"}
    canonical = json.dumps(
        _prune_none(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _verify_embedded_checksums(
    value: JsonValue, label: str, findings: list[str]
) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _verify_embedded_checksums(item, f"{label}[{index}]", findings)
        return
    if not isinstance(value, dict):
        return
    if "checksum" in value:
        try:
            observed = _sha(value["checksum"], f"{label}.checksum")
        except ValueError:
            findings.append(f"invalid_record_checksum:{label}")
        else:
            if observed != _record_checksum(value):
                findings.append(f"record_checksum_disagreement:{label}")
    for key, item in value.items():
        _verify_embedded_checksums(item, f"{label}.{key}", findings)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence_rows(
    *, package_root: Path, states: list[dict[str, JsonValue]], findings: list[str]
) -> dict[str, dict[str, str]]:
    root = package_root.resolve(strict=True)
    records: dict[str, dict[str, JsonValue]] = {}
    for state in states:
        for raw in _array(state.get("evidence"), "state.evidence"):
            record = _object(raw, "evidence reference")
            checksum = _sha(record.get("checksum"), "evidence.checksum")
            if checksum in records and records[checksum] != record:
                findings.append(f"evidence_checksum_collision:{checksum}")
            records[checksum] = record
    rows: dict[str, dict[str, str]] = {}
    for checksum, record in records.items():
        relative = PurePosixPath(
            _text(record.get("relative_path"), "evidence.relative_path")
        )
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != str(relative)
        ):
            findings.append(f"invalid_evidence_path:{checksum}")
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            findings.append(f"missing_evidence_artifact:{checksum}")
            continue
        if _file_sha256(path) != _sha(
            record.get("artifact_sha256"), "evidence.artifact_sha256"
        ):
            findings.append(f"evidence_artifact_checksum_disagreement:{checksum}")
            continue
        with path.open("r", encoding="utf-8", newline="") as stream:
            source_rows = tuple(csv.DictReader(stream))
        if len(source_rows) != 1 or set(source_rows[0]) != {
            "asset_id",
            "measure_id",
            "estimate",
            "lower",
            "upper",
        }:
            findings.append(f"invalid_evidence_table:{checksum}")
            continue
        rows[checksum] = source_rows[0]
    return rows


def _signature(record: dict[str, JsonValue]) -> ActionSignature:
    return (
        _text(record.get("action_id"), "action.action_id"),
        _optional_text(record.get("target_asset_id"), "action.target_asset_id"),
        _optional_text(record.get("reserve_asset_id"), "action.reserve_asset_id"),
    )


def _legal_signatures(state: dict[str, JsonValue]) -> set[ActionSignature]:
    stream = cast(StreamId, _text(state.get("stream_id"), "state.stream_id"))
    checkpoint = _text(
        state.get("current_checkpoint_id", state.get("checkpoint_id")),
        "state checkpoint_id",
    )
    candidates = tuple(
        _text(item, "candidate asset")
        for item in _array(state.get("candidate_asset_ids"), "candidates")
    )
    retired = {
        _text(item, "retired asset")
        for item in _array(state.get("retired_asset_ids"), "retired")
    }
    if checkpoint == "observational_review":
        if stream == "single_asset_development":
            return {
                ("nominate_for_early_study", asset, None)
                for asset in candidates
                if asset not in retired
            } | {("withhold_nomination", None, None)}
        available = tuple(asset for asset in candidates if asset not in retired)
        return {
            ("select_lead_and_reserve", lead, reserve)
            for lead in available
            for reserve in available
            if lead != reserve
        } | {("withhold_selection", None, None)}
    actions: dict[tuple[StreamId, str], tuple[str, ...]] = {
        ("single_asset_development", "early_safety_study"): (
            "advance_to_proof_of_concept",
            "stop_development",
        ),
        ("single_asset_development", "proof_of_concept"): (
            "advance_to_confirmation",
            "stop_development",
        ),
        ("single_asset_development", "confirmation"): (
            "declare_success",
            "declare_failure",
            "declare_inconclusive",
        ),
        ("bounded_portfolio_reallocation", "joint_early_study_review"): (
            "advance_lead_to_proof_of_concept",
            "promote_reserve_to_proof_of_concept",
            "terminate_portfolio",
        ),
        ("bounded_portfolio_reallocation", "lead_proof_of_concept_review"): (
            "advance_active_to_confirmation",
            "promote_reserve_to_proof_of_concept",
            "terminate_portfolio",
        ),
        (
            "bounded_portfolio_reallocation",
            "promoted_reserve_proof_of_concept_review",
        ): (
            "advance_active_to_confirmation",
            "terminate_portfolio",
        ),
        ("bounded_portfolio_reallocation", "confirmation"): (
            "declare_success",
            "declare_failure",
            "declare_inconclusive",
        ),
    }
    action_ids = list(actions.get((stream, checkpoint), ()))
    if "promote_reserve_to_proof_of_concept" in action_ids:
        binding_value = state.get("policy_binding")
        if binding_value is None:
            budget = _number(
                state.get("resource_budget_units"), "state.resource_budget_units"
            )
            proof_of_concept_units = 2.0
            confirmation_units = 4.0
            maximum_switches = 1.0
        else:
            binding = _object(binding_value, "state.policy_binding")
            schedule = _object(
                binding.get("resource_schedule"), "policy.resource_schedule"
            )
            budget = _number(
                binding.get("resource_budget_units"), "policy.resource_budget_units"
            )
            proof_of_concept_units = _number(
                schedule.get("proof_of_concept_units"),
                "schedule.proof_of_concept_units",
            )
            confirmation_units = _number(
                schedule.get("confirmation_units"), "schedule.confirmation_units"
            )
            maximum_switches = _number(
                schedule.get("maximum_switches"), "schedule.maximum_switches"
            )
        spent = _number(state.get("resource_spent_units"), "state.resource_spent_units")
        switches = _number(state.get("switch_count"), "state.switch_count")
        required = proof_of_concept_units + confirmation_units
        if switches >= maximum_switches or spent + required > budget:
            action_ids.remove("promote_reserve_to_proof_of_concept")
    return {(action_id, None, None) for action_id in action_ids}


def _classification(
    rule: dict[str, JsonValue],
    evidence_rows: dict[str, dict[str, str]] | None,
) -> str:
    estimate = _number(rule.get("estimate"), "rule.estimate")
    lower = _number(rule.get("lower_bound"), "rule.lower_bound")
    upper = _number(rule.get("upper_bound"), "rule.upper_bound")
    threshold = _number(rule.get("threshold"), "rule.threshold")
    if not lower <= estimate <= upper:
        raise ValueError("Rule interval is not ordered.")
    domain = _text(rule.get("domain"), "rule.domain")
    if domain not in {"efficacy", "safety"}:
        raise ValueError("Randomized rule domain must be efficacy or safety.")
    if evidence_rows is not None:
        refs = tuple(
            _sha(item, "rule evidence checksum")
            for item in _array(rule.get("evidence_reference_checksums"), "rule refs")
        )
        if len(refs) != 1 or refs[0] not in evidence_rows:
            raise ValueError("Rule must bind one available numeric evidence row.")
        row = evidence_rows[refs[0]]
        if (
            row["asset_id"] != _text(rule.get("asset_id"), "rule.asset_id")
            or row["measure_id"] != domain
        ):
            raise ValueError("Rule identity disagrees with its numeric source row.")
        for key, value in (("estimate", estimate), ("lower", lower), ("upper", upper)):
            if float(row[key]) != value:
                raise ValueError("Rule interval disagrees with its numeric source row.")
    direction = _text(rule.get("direction"), "rule.direction")
    expected_direction = "minimum" if domain == "efficacy" else "maximum"
    if direction != expected_direction:
        raise ValueError(
            f"Randomized {domain} rule requires direction={expected_direction!r}."
        )
    if direction == "minimum":
        if lower >= threshold:
            return "clear_pass"
        if upper < threshold:
            return "clear_fail"
    elif direction == "maximum":
        if upper <= threshold:
            return "clear_pass"
        if lower > threshold:
            return "clear_fail"
    else:
        raise ValueError("Rule direction is unknown.")
    return "indeterminate"


def derive_supported_action_signatures_v1(
    *,
    state: dict[str, JsonValue],
    evidence: dict[str, JsonValue],
    evidence_rows: dict[str, dict[str, str]] | None = None,
) -> set[ActionSignature]:
    """Derive evidence-supported actions without importing the public grader."""

    legal = _legal_signatures(state)
    stream = cast(StreamId, _text(state.get("stream_id"), "state.stream_id"))
    checkpoint = _text(
        state.get("current_checkpoint_id"), "state.current_checkpoint_id"
    )
    if _text(evidence.get("state_checksum"), "decision.state_checksum") != _sha(
        state.get("checksum"), "state.checksum"
    ):
        raise ValueError("Decision evidence is not bound to its state.")
    if checkpoint == "observational_review":
        identified = (
            _text(evidence.get("identification_status"), "identification_status")
            == "identified"
        )
        if not identified:
            stop_action = (
                "withhold_nomination"
                if stream == "single_asset_development"
                else "withhold_selection"
            )
            supported_stop: set[ActionSignature] = {(stop_action, None, None)}
            return supported_stop & legal
        candidate_rows = [
            _object(item, "candidate")
            for item in _array(evidence.get("candidates"), "decision.candidates")
        ]
        candidates = {
            _text(row.get("asset_id"), "candidate.asset_id"): row
            for row in candidate_rows
        }
        state_candidates = {
            _text(item, "candidate asset")
            for item in _array(state.get("candidate_asset_ids"), "state candidates")
        }
        if set(candidates) != state_candidates:
            raise ValueError("Observational evidence does not cover all candidates.")
        if evidence_rows is not None:
            for asset, candidate in candidates.items():
                refs = tuple(
                    _sha(item, "candidate evidence checksum")
                    for item in _array(
                        candidate.get("evidence_reference_checksums"), "candidate refs"
                    )
                )
                if len(refs) != 1 or refs[0] not in evidence_rows:
                    raise ValueError(
                        "Candidate must bind one available numeric evidence row."
                    )
                row = evidence_rows[refs[0]]
                if row["asset_id"] != asset or row["measure_id"] != "utility":
                    raise ValueError(
                        "Candidate identity disagrees with its utility source row."
                    )
                if float(row["estimate"]) != _number(
                    candidate.get("utility_estimate"), "candidate utility"
                ):
                    raise ValueError(
                        "Candidate utility disagrees with its numeric source row."
                    )
        supported: set[ActionSignature] = set()
        minimum_gain = _number(
            evidence.get("minimum_efficacy_gain"), "minimum efficacy gain"
        )
        margin = _number(
            evidence.get("practical_equivalence_margin"), "equivalence margin"
        )
        possibly_qualified = {
            asset
            for asset, row in candidates.items()
            if _number(
                row.get("efficacy_upper_bound"), "candidate efficacy upper bound"
            )
            >= minimum_gain
        }
        definitely_qualified = {
            asset
            for asset, row in candidates.items()
            if _number(
                row.get("efficacy_lower_bound"), "candidate efficacy lower bound"
            )
            >= minimum_gain
        }
        if not possibly_qualified:
            stop_action = (
                "withhold_nomination"
                if stream == "single_asset_development"
                else "withhold_selection"
            )
            supported_stop = {(stop_action, None, None)}
            return supported_stop & legal
        contrast_rows = [
            _object(item, "contrast")
            for item in _array(evidence.get("pair_contrasts"), "pair_contrasts")
        ]
        contrasts = {
            tuple(
                sorted(
                    (
                        _text(row.get("lead_asset_id"), "contrast.lead"),
                        _text(row.get("reserve_asset_id"), "contrast.reserve"),
                    )
                )
            ): row
            for row in contrast_rows
        }
        candidate_ids = tuple(sorted(candidates))
        expected_pairs = {
            (first, second)
            for index, first in enumerate(candidate_ids)
            for second in candidate_ids[index + 1 :]
        }
        if set(contrasts) != expected_pairs:
            raise ValueError(
                "Observational evidence lacks one contrast for every candidate pair."
            )

        def contrast_width(first: str, second: str) -> float:
            if first == second:
                return 0.0
            return _number(
                contrasts[tuple(sorted((first, second)))].get("confidence_half_width"),
                "contrast width",
            )

        lead_eligible: set[str] = set()
        best = min(
            possibly_qualified,
            key=lambda asset: (
                -_number(candidates[asset].get("utility_estimate"), "utility"),
                asset,
            ),
        )
        best_utility = _number(candidates[best].get("utility_estimate"), "best utility")
        lead_eligible = {
            asset
            for asset in possibly_qualified
            if best_utility
            - _number(candidates[asset].get("utility_estimate"), "candidate utility")
            <= max(
                margin,
                (contrast_width(best, asset)),
            )
        }
        if stream == "single_asset_development":
            supported |= {
                ("nominate_for_early_study", asset, None)
                for asset in sorted(lead_eligible)
            }
        else:
            for lead in sorted(lead_eligible):
                reserve_candidates = possibly_qualified - {lead}
                if not reserve_candidates:
                    continue
                best_reserve = min(
                    reserve_candidates,
                    key=lambda asset: (
                        -_number(candidates[asset].get("utility_estimate"), "utility"),
                        asset,
                    ),
                )
                best_reserve_utility = _number(
                    candidates[best_reserve].get("utility_estimate"),
                    "best reserve utility",
                )
                for reserve in sorted(reserve_candidates):
                    permitted = max(
                        margin,
                        contrast_width(best_reserve, reserve),
                    )
                    if (
                        best_reserve_utility
                        - _number(
                            candidates[reserve].get("utility_estimate"),
                            "reserve utility",
                        )
                        <= permitted
                    ):
                        supported.add(("select_lead_and_reserve", lead, reserve))
        allocation_supported = any(
            action in {"nominate_for_early_study", "select_lead_and_reserve"}
            for action, _, _ in supported
        )
        if not definitely_qualified or not allocation_supported:
            supported.add(
                ("withhold_nomination", None, None)
                if stream == "single_asset_development"
                else ("withhold_selection", None, None)
            )
        return supported & legal

    grouped: dict[tuple[str, str], list[str]] = {}
    for raw_rule in _array(evidence.get("rules"), "decision.rules"):
        rule = _object(raw_rule, "rule")
        key = (
            _text(rule.get("asset_id"), "rule.asset_id"),
            _text(rule.get("domain"), "rule.domain"),
        )
        grouped.setdefault(key, []).append(_classification(rule, evidence_rows))

    def classes(asset: str, domains: tuple[str, ...]) -> dict[str, str]:
        output: dict[str, str] = {}
        for domain in domains:
            values = grouped.get((asset, domain), [])
            if not values:
                raise ValueError(f"Missing {domain} evidence for {asset}.")
            output[domain] = (
                "clear_fail"
                if "clear_fail" in values
                else (
                    "clear_pass"
                    if all(value == "clear_pass" for value in values)
                    else "indeterminate"
                )
            )
        return output

    def clear_pass(values: dict[str, str]) -> bool:
        return all(value == "clear_pass" for value in values.values())

    def clear_fail(values: dict[str, str]) -> bool:
        return any(value == "clear_fail" for value in values.values())

    active = _text(state.get("active_asset_id"), "state.active_asset_id")
    action_ids: set[str]
    if checkpoint == "joint_early_study_review":
        lead = _text(state.get("lead_asset_id"), "state.lead_asset_id")
        reserve = _text(state.get("reserve_asset_id"), "state.reserve_asset_id")
        lead_values = classes(lead, ("safety",))
        reserve_values = classes(reserve, ("safety",))
        action_ids = set()
        if not clear_fail(lead_values):
            action_ids.add("advance_lead_to_proof_of_concept")
        if not clear_fail(reserve_values) and not clear_pass(lead_values):
            action_ids.add("promote_reserve_to_proof_of_concept")
        if not clear_pass(lead_values) and not clear_pass(reserve_values):
            action_ids.add("terminate_portfolio")
    else:
        domains = (
            ("efficacy", "safety")
            if checkpoint == "confirmation"
            else (
                ("safety",)
                if checkpoint == "early_safety_study"
                else ("efficacy", "safety")
            )
        )
        values = classes(active, domains)
        if checkpoint == "confirmation":
            action_ids = {
                (
                    "declare_success"
                    if clear_pass(values)
                    else (
                        "declare_failure"
                        if clear_fail(values)
                        else "declare_inconclusive"
                    )
                )
            }
        else:
            advance = {
                "early_safety_study": "advance_to_proof_of_concept",
                "proof_of_concept": "advance_to_confirmation",
                "lead_proof_of_concept_review": "advance_active_to_confirmation",
                "promoted_reserve_proof_of_concept_review": "advance_active_to_confirmation",
            }[checkpoint]
            stop = (
                "stop_development"
                if stream == "single_asset_development"
                else "terminate_portfolio"
            )
            action_ids = set()
            if not clear_fail(values):
                action_ids.add(advance)
            if not clear_pass(values):
                action_ids.add(stop)
                if checkpoint == "lead_proof_of_concept_review":
                    action_ids.add("promote_reserve_to_proof_of_concept")
    return {signature for signature in legal if signature[0] in action_ids}


def _assessment_findings(
    *, programme: dict[str, JsonValue], steps: list[dict[str, JsonValue]]
) -> list[str]:
    findings: list[str] = []
    stream = cast(StreamId, _text(programme.get("stream_id"), "programme.stream_id"))
    assessment = _object(programme.get("assessment"), "programme.assessment")
    if _text(assessment.get("programme_id"), "assessment.programme_id") != _text(
        programme.get("programme_id"), "programme.programme_id"
    ):
        findings.append("assessment_programme_identity_disagreement")
    rows = [
        _object(item, "assessment checkpoint")
        for item in _array(assessment.get("checkpoints"), "assessment checkpoints")
    ]
    if (
        tuple(_text(row.get("checkpoint_id"), "checkpoint id") for row in rows)
        != _CHECKPOINTS[stream]
    ):
        findings.append("assessment_checkpoint_denominator_disagreement")
        return findings
    step_by_checkpoint = {
        _text(
            _object(step.get("state_before"), "state before").get(
                "current_checkpoint_id"
            ),
            "checkpoint",
        ): step
        for step in steps
    }
    for row in rows:
        checkpoint = _text(row.get("checkpoint_id"), "checkpoint")
        step = step_by_checkpoint.get(checkpoint)
        outcome = _object(row.get("outcome"), "checkpoint outcome")
        reach_status = _text(outcome.get("reach_status"), "checkpoint reach status")
        submission_status = _text(
            outcome.get("submission_status"), "checkpoint submission status"
        )
        analysis_status = _text(
            outcome.get("analysis_status"), "checkpoint analysis status"
        )
        execution_status = _text(
            outcome.get("execution_status"), "checkpoint execution status"
        )
        if step is None:
            if (
                reach_status != "structural_nonreach"
                or submission_status != "not_applicable"
                or analysis_status != "not_applicable"
                or execution_status != "not_applicable"
                or _array(row.get("lanes"), "lanes")
                or _array(row.get("capabilities"), "capabilities")
            ):
                findings.append(f"invalid_structural_nonreach:{checkpoint}")
            continue
        if (
            reach_status != "reached"
            or submission_status != "accepted"
            or analysis_status not in {"estimable", "non_estimable"}
            or execution_status != "completed"
        ):
            findings.append(f"reached_checkpoint_not_scored:{checkpoint}")
            continue
        supported_sha = _sha(
            _object(step.get("supported_action_set"), "supported set").get("checksum"),
            "supported checksum",
        )
        evidence_sha = _sha(
            _object(step.get("decision_evidence"), "decision evidence").get("checksum"),
            "decision checksum",
        )
        transition_sha = _sha(
            _object(step.get("state_after"), "state after").get("checksum"),
            "transition checksum",
        )
        terminal = (
            _text(
                _object(step.get("state_after"), "state after").get(
                    "terminal_disposition"
                ),
                "terminal",
            )
            != "active"
        )
        required_lanes = set(_REQUIRED_LANES[(stream, checkpoint)])
        if terminal:
            required_lanes |= {"route_timing", "final_recommendation"}
        lanes = [_object(item, "lane") for item in _array(row.get("lanes"), "lanes")]
        if {_text(lane.get("lane_id"), "lane id") for lane in lanes} != required_lanes:
            findings.append(f"lane_denominator_disagreement:{checkpoint}")
        for lane in lanes:
            lane_id = _text(lane.get("lane_id"), "lane id")
            expected_source = (
                transition_sha
                if lane_id in {"route_timing", "final_recommendation"}
                else supported_sha
            )
            if (
                _text(lane.get("outcome"), "lane outcome") != "accepted"
                or _sha(lane.get("source_record_sha256"), "lane source")
                != expected_source
            ):
                findings.append(f"lane_source_disagreement:{checkpoint}:{lane_id}")
        capabilities = [
            _object(item, "capability")
            for item in _array(row.get("capabilities"), "capabilities")
        ]
        if {
            _text(item.get("capability_id"), "capability id") for item in capabilities
        } != set(_CAPABILITY_CHECKS):
            findings.append(f"capability_denominator_disagreement:{checkpoint}")
        source_by_check = {
            "evidence_integrity": evidence_sha,
            "method_eligibility": evidence_sha,
            "identification_status": evidence_sha,
            "uncertainty_qualification": evidence_sha,
            "policy_conclusion_compatibility": supported_sha,
            "safety_evidence": evidence_sha,
            "transition_legality": transition_sha,
            "history_immutability": transition_sha,
            "required_output_presence": transition_sha,
            "workflow_completion": transition_sha,
            "selected_action_membership": supported_sha,
        }
        for capability in capabilities:
            capability_id = _text(capability.get("capability_id"), "capability id")
            checks = [
                _object(item, "capability check")
                for item in _array(capability.get("checks"), "checks")
            ]
            if {(_text(item.get("check_id"), "check id")) for item in checks} != set(
                _CAPABILITY_CHECKS.get(capability_id, ())
            ):
                findings.append(
                    f"capability_check_denominator_disagreement:{checkpoint}:{capability_id}"
                )
            for check in checks:
                check_id = _text(check.get("check_id"), "check id")
                if (
                    not _boolean(check.get("passed"), "check passed")
                    or _sha(check.get("source_record_sha256"), "check source")
                    != source_by_check[check_id]
                ):
                    findings.append(
                        f"capability_source_disagreement:{checkpoint}:{check_id}"
                    )
    return findings


def _audit_state_action_graph(
    *, graph_path: Path, findings: list[str]
) -> tuple[str, dict[StreamId, int], dict[StreamId, int]]:
    graph = _object(
        _JSON_ADAPTER.validate_json(graph_path.read_bytes()), "state-action graph"
    )
    if graph.get("schema_id") != "trialagentbench.trialdev_state_action_graph/v1":
        findings.append("unexpected_state_action_graph_schema")
    source_identity = _sha(graph.get("source_identity"), "graph source identity")
    nodes = [
        _object(item, "graph node")
        for item in _array(graph.get("nodes"), "graph nodes")
    ]
    edges = [
        _object(item, "graph edge")
        for item in _array(graph.get("edges"), "graph edges")
    ]
    node_by_id: dict[str, dict[str, JsonValue]] = {}
    for node in nodes:
        node_id = _sha(node.get("node_id"), "graph node id")
        payload = {key: value for key, value in node.items() if key != "node_id"}
        if node_id != _record_checksum(payload):
            findings.append(f"graph_node_identity_disagreement:{node_id}")
        if node_id in node_by_id:
            findings.append(f"duplicate_graph_node:{node_id}")
        node_by_id[node_id] = node
    edge_by_id: dict[str, dict[str, JsonValue]] = {}
    outgoing: dict[str, list[dict[str, JsonValue]]] = {}
    for edge in edges:
        edge_id = _sha(edge.get("edge_id"), "graph edge id")
        payload = {key: value for key, value in edge.items() if key != "edge_id"}
        if edge_id != _record_checksum(payload):
            findings.append(f"graph_edge_identity_disagreement:{edge_id}")
        if edge_id in edge_by_id:
            findings.append(f"duplicate_graph_edge:{edge_id}")
        edge_by_id[edge_id] = edge
        source = _sha(edge.get("source_node_id"), "graph source node")
        outgoing.setdefault(source, []).append(edge)
        target = _optional_text(edge.get("target_node_id"), "graph target node")
        disposition = _text(edge.get("terminal_disposition"), "graph disposition")
        if source not in node_by_id or (
            target is not None and target not in node_by_id
        ):
            findings.append(f"graph_edge_not_closed:{edge_id}")
        if (disposition == "active") != (target is not None):
            findings.append(f"graph_edge_destination_disagreement:{edge_id}")
    terminal_by_action = {
        "withhold_nomination": "withheld",
        "withhold_selection": "withheld",
        "stop_development": "stopped",
        "terminate_portfolio": "stopped",
        "declare_success": "success",
        "declare_failure": "failure",
        "declare_inconclusive": "inconclusive",
    }
    for node_id, node in node_by_id.items():
        observed = {_signature(edge) for edge in outgoing.get(node_id, [])}
        if observed != _legal_signatures(node):
            findings.append(f"graph_legal_action_disagreement:{node_id}")
        for edge in outgoing.get(node_id, []):
            action_id = _text(edge.get("action_id"), "graph action")
            disposition = _text(edge.get("terminal_disposition"), "graph disposition")
            expected_disposition = terminal_by_action.get(action_id, "active")
            if disposition != expected_disposition:
                findings.append(
                    f"graph_terminal_disposition_disagreement:{_sha(edge.get('edge_id'), 'edge id')}"
                )
    for stream in _CHECKPOINTS:
        stream_nodes = {
            node_id
            for node_id, node in node_by_id.items()
            if _text(node.get("stream_id"), "node stream") == stream
        }
        roots = {
            node_id
            for node_id in stream_nodes
            if _text(node_by_id[node_id].get("checkpoint_id"), "node checkpoint")
            == "observational_review"
        }
        if len(roots) != 1:
            findings.append(f"graph_root_cardinality:{stream}:{len(roots)}")
            continue
        reached = set(roots)
        frontier = list(roots)
        while frontier:
            source = frontier.pop()
            for edge in outgoing.get(source, []):
                target = _optional_text(edge.get("target_node_id"), "target node")
                if target in stream_nodes and target not in reached:
                    reached.add(target)
                    frontier.append(target)
        for node_id in sorted(stream_nodes - reached):
            findings.append(f"unreachable_graph_node:{stream}:{node_id}")
    node_counts = {
        stream: sum(
            _text(node.get("stream_id"), "node stream") == stream for node in nodes
        )
        for stream in _CHECKPOINTS
    }
    edge_counts = {
        stream: sum(
            _text(edge.get("stream_id"), "edge stream") == stream for edge in edges
        )
        for stream in _CHECKPOINTS
    }
    if node_counts != {
        "single_asset_development": 4,
        "bounded_portfolio_reallocation": 43,
    }:
        findings.append("graph_node_census_disagreement")
    if edge_counts != {
        "single_asset_development": 9,
        "bounded_portfolio_reallocation": 121,
    }:
        findings.append("graph_edge_census_disagreement")
    return source_identity, node_counts, edge_counts


def audit_trialdev_worked_programmes(
    *, package_root: Path
) -> TrialDevWorkedProgrammeVerificationV1:
    """Reconstruct worked actions, numeric evidence, state chains, and assessments."""

    root = Path(package_root)
    package_path = root / "worked_programmes.json"
    graph_path = root / "state_action_graph.json"
    parsed = _JSON_ADAPTER.validate_json(package_path.read_bytes())
    package = _object(parsed, "worked package")
    findings: list[str] = []
    _verify_embedded_checksums(package, "package", findings)
    if package.get("schema_id") != "trialagentbench.trialdev_worked_programmes/v1":
        findings.append("unexpected_package_schema")
    if package.get("purpose") != "non_score_bearing_scientific_demonstration":
        findings.append("unexpected_package_purpose")
    programmes = [
        _object(item, "programme")
        for item in _array(package.get("programmes"), "programmes")
    ]
    streams = tuple(
        cast(StreamId, _text(item.get("stream_id"), "programme.stream_id"))
        for item in programmes
    )
    if len(programmes) != 2 or set(streams) != set(_CHECKPOINTS):
        findings.append("worked_stream_census_disagreement")
    all_states = [
        _object(step.get(state_key), state_key)
        for programme in programmes
        for step in [
            _object(item, "checkpoint")
            for item in _array(programme.get("checkpoints"), "checkpoints")
        ]
        for state_key in ("state_before", "state_after")
    ]
    evidence_rows = _evidence_rows(
        package_root=root, states=all_states, findings=findings
    )
    graph_source_identity, graph_node_counts, graph_edge_counts = (
        _audit_state_action_graph(graph_path=graph_path, findings=findings)
    )
    worked_source_identities = {
        _sha(evidence.get("source_family_id"), "evidence source family")
        for state in all_states
        for evidence in [
            _object(item, "evidence")
            for item in _array(state.get("evidence"), "state evidence")
        ]
    }
    if worked_source_identities != {graph_source_identity}:
        findings.append("worked_source_identity_disagreement")
    checkpoint_count = 0
    terminal_dispositions: dict[StreamId, str] = {}
    reconstructed = 0
    for programme in programmes:
        stream = cast(
            StreamId, _text(programme.get("stream_id"), "programme.stream_id")
        )
        steps = [
            _object(item, "checkpoint")
            for item in _array(programme.get("checkpoints"), "checkpoints")
        ]
        checkpoint_count += len(steps)
        previous_after: dict[str, JsonValue] | None = None
        for index, step in enumerate(steps):
            before = _object(step.get("state_before"), "state before")
            after = _object(step.get("state_after"), "state after")
            decision = _object(step.get("decision_evidence"), "decision evidence")
            supported_set = _object(
                step.get("supported_action_set"), "supported action set"
            )
            selection = _object(step.get("selected_action"), "selected action")
            before_sha = _sha(before.get("checksum"), "state before checksum")
            if previous_after is not None and before != previous_after:
                findings.append(f"noncontiguous_state_chain:{stream}:{index}")
            previous_after = after
            if (
                _sha(after.get("previous_state_checksum"), "previous state checksum")
                != before_sha
            ):
                findings.append(
                    f"previous_state_checksum_disagreement:{stream}:{index}"
                )
            before_history = _array(before.get("history"), "history before")
            after_history = _array(after.get("history"), "history after")
            if (
                after_history[:-1] != before_history
                or len(after_history) != len(before_history) + 1
            ):
                findings.append(f"history_not_append_only:{stream}:{index}")
            else:
                history_selection = _object(
                    _object(after_history[-1], "history entry").get("selected_action"),
                    "history selection",
                )
                if _sha(
                    history_selection.get("checksum"), "history selection checksum"
                ) != _sha(selection.get("checksum"), "selection checksum"):
                    findings.append(f"history_selection_disagreement:{stream}:{index}")
            try:
                expected = derive_supported_action_signatures_v1(
                    state=before,
                    evidence=decision,
                    evidence_rows=evidence_rows,
                )
            except (KeyError, TypeError, ValueError) as error:
                findings.append(
                    f"supported_set_reconstruction_failed:{stream}:{index}:{error}"
                )
                expected = set()
            observed_legal = {
                _signature(_object(item, "legal action"))
                for item in _array(supported_set.get("legal_actions"), "legal actions")
            }
            observed_supported = {
                _signature(_object(item, "supported action"))
                for item in _array(
                    supported_set.get("supported_actions"), "supported actions"
                )
            }
            if observed_legal != _legal_signatures(before):
                findings.append(
                    f"legal_action_reconstruction_disagreement:{stream}:{index}"
                )
            if observed_supported != expected:
                findings.append(
                    f"supported_action_reconstruction_disagreement:{stream}:{index}"
                )
            if _signature(selection) not in observed_supported:
                findings.append(f"selected_action_not_supported:{stream}:{index}")
            reconstructed += 1
        if steps:
            terminal_dispositions[stream] = _text(
                _object(steps[-1].get("state_after"), "final state").get(
                    "terminal_disposition"
                ),
                "terminal disposition",
            )
        findings.extend(_assessment_findings(programme=programme, steps=steps))
    ordered_findings = tuple(sorted(set(findings)))
    return TrialDevWorkedProgrammeVerificationV1(
        source_identity=graph_source_identity,
        package_sha256=_file_sha256(package_path),
        state_action_graph_sha256=_file_sha256(graph_path),
        programme_count=len(programmes),
        checkpoint_count=checkpoint_count,
        evidence_artifact_count=len(evidence_rows),
        reconstructed_supported_set_count=reconstructed,
        terminal_dispositions=terminal_dispositions,
        graph_node_counts=graph_node_counts,
        graph_edge_counts=graph_edge_counts,
        findings=ordered_findings,
        status="fail" if ordered_findings else "pass",
    )


__all__ = [
    "TrialDevWorkedProgrammeVerificationV1",
    "audit_trialdev_worked_programmes",
    "derive_supported_action_signatures_v1",
]
