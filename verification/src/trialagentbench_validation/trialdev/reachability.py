"""Independently verify TrialDev programme reachability from released roles."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal, cast
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field, model_validator

STOP_ACTION_ID = "withhold_nomination"
RANDOMIZED_PHASE_IDS = ("phase1", "phase2", "phase3")
ReachabilityClass = Literal[
    "structural_nonreach",
    "optional_nomination",
    "nomination_required",
]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialDevProgrammeReachabilityV1(_Contract):
    """Released action support and replay availability for one programme."""

    program_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    objective_id: str = Field(min_length=1)
    reachability_class: ReachabilityClass
    reference_action_ids: tuple[str, ...] = Field(min_length=1)
    credit_eligible_action_ids: tuple[str, ...] = Field(min_length=1)
    credit_eligible_candidate_ids: tuple[str, ...]
    fixed_replay_candidate_ids: tuple[str, ...]
    fixed_replay_case_count: int = Field(ge=0)
    findings: tuple[str, ...] = ()
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def _canonical_and_consistent(self) -> TrialDevProgrammeReachabilityV1:
        """Require canonical action sets and status."""

        if self.program_id != f"{self.scenario_id}__{self.objective_id}":
            raise ValueError("program_id must identify its scenario and objective")
        for field_name in (
            "reference_action_ids",
            "credit_eligible_action_ids",
            "credit_eligible_candidate_ids",
            "fixed_replay_candidate_ids",
            "findings",
        ):
            values = getattr(self, field_name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field_name} must be sorted and unique")
        expected_class = _reachability_class(self.credit_eligible_action_ids)
        if self.reachability_class != expected_class:
            raise ValueError(
                "reachability_class disagrees with credit-eligible actions"
            )
        if (self.status == "pass") != (not self.findings):
            raise ValueError("programme reachability status disagrees with findings")
        return self


class TrialDevReachabilityReportV1(_Contract):
    """Complete release-level TrialDev reachability and replay census."""

    schema_id: Literal["trialagentbench.validation.trialdev_reachability/v1"] = (
        "trialagentbench.validation.trialdev_reachability/v1"
    )
    participant_release_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_release_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_programme_count: int = Field(ge=1)
    checked_programme_count: int = Field(ge=0)
    fixed_replay_case_count: int = Field(ge=0)
    expanded_programme_case_count: int = Field(ge=0)
    reachability_counts: dict[ReachabilityClass, int]
    failed_programme_count: int = Field(ge=0)
    global_findings: tuple[str, ...] = ()
    programmes: tuple[TrialDevProgrammeReachabilityV1, ...]
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def _census_is_complete(self) -> TrialDevReachabilityReportV1:
        """Require report totals to equal the programme records."""

        identities = tuple(row.program_id for row in self.programmes)
        if identities != tuple(sorted(identities)) or len(identities) != len(
            set(identities)
        ):
            raise ValueError("programme reachability records must be sorted and unique")
        if self.checked_programme_count != len(self.programmes):
            raise ValueError("checked_programme_count does not match programme records")
        if self.required_programme_count != self.checked_programme_count:
            raise ValueError(
                "reachability report must cover the complete programme denominator"
            )
        observed_counts = Counter(row.reachability_class for row in self.programmes)
        if dict(observed_counts) != self.reachability_counts:
            raise ValueError("reachability_counts do not match programme records")
        failed = sum(row.status == "fail" for row in self.programmes)
        if self.failed_programme_count != failed:
            raise ValueError("failed_programme_count does not match programme records")
        expected_status = failed == 0 and not self.global_findings
        if (self.status == "pass") != expected_status:
            raise ValueError("reachability status disagrees with findings")
        if self.global_findings != tuple(sorted(set(self.global_findings))):
            raise ValueError("global_findings must be sorted and unique")
        return self


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(payload: object, *, label: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], payload)


def _object_rows(value: object, *, label: str) -> tuple[dict[str, object], ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(row, dict) for row in value)
    ):
        raise ValueError(f"{label} must be a non-empty array of objects")
    return tuple(cast(dict[str, object], row) for row in value)


def _strings(value: object, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"{label} must be a non-empty string array")
    return tuple(value)


def _safe_identifier(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in "/\\")
        or value in {".", ".."}
    ):
        raise ValueError(f"{label} must be a safe non-empty identifier")
    return value


def _reachability_class(actions: tuple[str, ...]) -> ReachabilityClass:
    candidates = tuple(action for action in actions if action != STOP_ACTION_ID)
    stop_supported = STOP_ACTION_ID in actions
    if not candidates:
        if not stop_supported:
            raise ValueError(
                "a programme action set supports neither nomination nor non-nomination"
            )
        return "structural_nonreach"
    return "optional_nomination" if stop_supported else "nomination_required"


def _programme_contexts(participant: ZipFile) -> tuple[tuple[str, str], ...]:
    suite = _object(
        json.loads(participant.read("benchmark_suite_manifest.json")),
        label="suite manifest",
    )
    contexts = {
        (
            _safe_identifier(item.get("scenario_id"), label="scenario_id"),
            _safe_identifier(item.get("objective_id"), label="objective_id"),
        )
        for item in _object_rows(suite.get("items"), label="suite items")
        if item.get("phase_id") == "observational_review"
    }
    if not contexts:
        raise ValueError("suite manifest contains no observational programme contexts")
    return tuple(sorted(contexts))


def _action_support(
    evaluator: ZipFile,
    contexts: tuple[tuple[str, str], ...],
) -> dict[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]]:
    output: dict[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for scenario_id in sorted({scenario_id for scenario_id, _objective_id in contexts}):
        member = f"scenario_{scenario_id}/grader/public_recoverability_report.json"
        report = _object(json.loads(evaluator.read(member)), label=member)
        if report.get("scenario_id") != scenario_id:
            raise ValueError(f"{member} identifies another scenario")
        for row in _object_rows(
            report.get("method_union_action_sensitivity"),
            label=f"{member} method_union_action_sensitivity",
        ):
            objective_id = _safe_identifier(
                row.get("objective_id"), label="objective_id"
            )
            key = (scenario_id, objective_id)
            if key in output:
                raise ValueError(f"duplicate TrialDev action policy: {key!r}")
            reference = tuple(
                sorted(
                    set(
                        _strings(
                            row.get("reference_target_ids"), label="reference actions"
                        )
                    )
                )
            )
            eligible = tuple(
                sorted(
                    set(
                        _strings(
                            row.get("credit_eligible_target_ids"),
                            label="credit-eligible actions",
                        )
                    )
                )
            )
            if not set(reference) <= set(eligible):
                raise ValueError(
                    f"reference actions exceed credit-eligible actions: {key!r}"
                )
            output[key] = (reference, eligible)
    expected = set(contexts)
    if set(output) != expected:
        missing = sorted(expected - set(output))
        extra = sorted(set(output) - expected)
        raise ValueError(
            f"TrialDev action policies disagree with programme contexts: missing={missing!r}, extra={extra!r}"
        )
    return output


def _fixed_replay_index(
    participant: ZipFile,
    contexts: set[tuple[str, str]],
) -> tuple[
    dict[tuple[str, str, str], set[str]],
    int,
    int,
    tuple[str, ...],
]:
    expanded: dict[tuple[str, str, str], set[str]] = {}
    physical_count = 0
    expanded_count = 0
    findings: list[str] = []
    for line_number, line in enumerate(
        participant.read("fixed_trajectories/cases.jsonl").decode("utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        physical_count += 1
        case = _object(json.loads(line), label=f"fixed replay line {line_number}")
        request = _object(
            case.get("request"), label=f"fixed replay request {line_number}"
        )
        scenario_id = _safe_identifier(request.get("scenario_id"), label="scenario_id")
        scenario_root = _safe_identifier(
            case.get("scenario_root"), label="scenario_root"
        )
        if scenario_root != f"scenario_{scenario_id}":
            findings.append(f"line_{line_number}:scenario_root_mismatch")
        phase_id = _safe_identifier(request.get("phase_id"), label="phase_id")
        if phase_id not in RANDOMIZED_PHASE_IDS:
            findings.append(f"line_{line_number}:unsupported_phase:{phase_id}")
            continue
        candidates = _strings(
            request.get("candidate_drug_ids"), label="candidate_drug_ids"
        )
        if len(candidates) != 1:
            findings.append(f"line_{line_number}:request_must_identify_one_candidate")
            continue
        candidate_id = candidates[0]
        objectives = _strings(
            case.get("program_objective_ids"), label="program_objective_ids"
        )
        for objective_id in objectives:
            expanded_count += 1
            context = (scenario_id, objective_id)
            if context not in contexts:
                findings.append(
                    f"line_{line_number}:unknown_programme:{scenario_id}__{objective_id}"
                )
                continue
            key = (scenario_id, objective_id, candidate_id)
            phases = expanded.setdefault(key, set())
            if phase_id in phases:
                findings.append(
                    f"line_{line_number}:duplicate_case:{scenario_id}:{objective_id}:{candidate_id}:{phase_id}"
                )
            phases.add(phase_id)
    return expanded, physical_count, expanded_count, tuple(sorted(set(findings)))


def audit_trialdev_reachability(
    *,
    participant_release: Path,
    evaluator_release: Path,
) -> TrialDevReachabilityReportV1:
    """Verify programme actions and randomized replay coverage from release archives."""

    participant_path = Path(participant_release)
    evaluator_path = Path(evaluator_release)
    if not participant_path.is_file() or not evaluator_path.is_file():
        raise FileNotFoundError(
            "TrialDev participant and evaluator releases must be files"
        )
    with ZipFile(participant_path) as participant, ZipFile(evaluator_path) as evaluator:
        contexts = _programme_contexts(participant)
        action_support = _action_support(evaluator, contexts)
        fixed_replay, physical_count, expanded_count, global_findings = (
            _fixed_replay_index(
                participant,
                set(contexts),
            )
        )

    programmes: list[TrialDevProgrammeReachabilityV1] = []
    used_replay_keys: set[tuple[str, str, str]] = set()
    for scenario_id, objective_id in contexts:
        reference, eligible = action_support[(scenario_id, objective_id)]
        eligible_candidates = tuple(
            action for action in eligible if action != STOP_ACTION_ID
        )
        replay_candidates = tuple(
            sorted(
                candidate_id
                for replay_scenario, replay_objective, candidate_id in fixed_replay
                if replay_scenario == scenario_id and replay_objective == objective_id
            )
        )
        findings: list[str] = []
        missing_candidates = sorted(set(eligible_candidates) - set(replay_candidates))
        extra_candidates = sorted(set(replay_candidates) - set(eligible_candidates))
        if missing_candidates:
            findings.append(
                f"accepted_candidates_missing_replay:{','.join(missing_candidates)}"
            )
        if extra_candidates:
            findings.append(
                f"replay_candidates_not_credit_eligible:{','.join(extra_candidates)}"
            )
        programme_case_count = 0
        for candidate_id in replay_candidates:
            key = (scenario_id, objective_id, candidate_id)
            used_replay_keys.add(key)
            phases = fixed_replay[key]
            programme_case_count += len(phases)
            missing_phases = sorted(set(RANDOMIZED_PHASE_IDS) - phases)
            extra_phases = sorted(phases - set(RANDOMIZED_PHASE_IDS))
            if missing_phases:
                findings.append(
                    f"candidate_missing_phases:{candidate_id}:{','.join(missing_phases)}"
                )
            if extra_phases:
                findings.append(
                    f"candidate_has_unknown_phases:{candidate_id}:{','.join(extra_phases)}"
                )
        programmes.append(
            TrialDevProgrammeReachabilityV1(
                program_id=f"{scenario_id}__{objective_id}",
                scenario_id=scenario_id,
                objective_id=objective_id,
                reachability_class=_reachability_class(eligible),
                reference_action_ids=reference,
                credit_eligible_action_ids=eligible,
                credit_eligible_candidate_ids=tuple(sorted(eligible_candidates)),
                fixed_replay_candidate_ids=replay_candidates,
                fixed_replay_case_count=programme_case_count,
                findings=tuple(sorted(set(findings))),
                status="fail" if findings else "pass",
            )
        )
    orphaned = sorted(set(fixed_replay) - used_replay_keys)
    if orphaned:
        global_findings = tuple(
            sorted(
                {
                    *global_findings,
                    *(
                        f"orphaned_fixed_replay:{scenario}:{objective}:{candidate}"
                        for scenario, objective, candidate in orphaned
                    ),
                }
            )
        )
    ordered = tuple(sorted(programmes, key=lambda row: row.program_id))
    counts = Counter(row.reachability_class for row in ordered)
    failed = sum(row.status == "fail" for row in ordered)
    return TrialDevReachabilityReportV1(
        participant_release_sha256=_sha256_file(participant_path),
        evaluator_release_sha256=_sha256_file(evaluator_path),
        required_programme_count=len(contexts),
        checked_programme_count=len(ordered),
        fixed_replay_case_count=physical_count,
        expanded_programme_case_count=expanded_count,
        reachability_counts=dict(counts),
        failed_programme_count=failed,
        global_findings=global_findings,
        programmes=ordered,
        status="pass" if failed == 0 and not global_findings else "fail",
    )


__all__ = [
    "TrialDevProgrammeReachabilityV1",
    "TrialDevReachabilityReportV1",
    "audit_trialdev_reachability",
]
