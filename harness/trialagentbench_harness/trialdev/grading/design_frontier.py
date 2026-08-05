"""Public statistical design frontier for TrialDev randomized phases."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.trialdev.grading.decision_evidence import (
    TrialDevPhaseDesignWitnessV1,
    derive_phase_design_witness_v1,
)
from trialagentbench_harness.trialdev.grading.hashing import compute_sha256_hex, sha256_file_hex
from trialagentbench_harness.trialdev.grading.io import read_json, write_json
from trialagentbench_harness.trialdev.grading.models import (
    TrialDevDesignEfficiencyV1,
    TrialDevDesignFrontierPointV1,
    TrialDevPhaseResourceConsequenceV1,
    TrialDevProgrammeResourceConsequenceV1,
)
from trialagentbench_harness.trialdev.grading.statistics import operational_support_count_v1
from trialagentbench_harness.trialdev.share.models import (
    PhaseModuleSpecV1,
    TrialDevelopmentRequestV1,
)
from trialagentbench_harness.trialdev.share.public_method_design import TrialDevPhaseDesignPolicyV1
from trialagentbench_harness.trialdev.share.validate import (
    candidate_ids_by_role_v1,
    validate_request_against_scenario_v1,
)

__all__ = [
    "TrialDevDesignEfficiencyV1",
    "TrialDevDesignFrontierArtifactV1",
    "TrialDevDesignFrontierPointV1",
    "TrialDevDesignFrontierStratumV1",
    "TrialDevOperationalSupportV1",
    "build_phase_design_frontiers_v1",
    "derive_phase_design_efficiency_v1",
    "derive_phase_resource_consequence_v1",
    "derive_programme_resource_consequence_v1",
    "load_phase_design_frontiers_v1",
    "select_operational_support_v1",
]

_FRONTIER_PATH = Path("public") / "phase_design_frontiers.json"
_FRONTIER_INPUTS = (
    "public/candidate_drug_catalog.json",
    "public/eval_contract.json",
    "public/observational_extract.parquet",
    "public/phase_design_policy.json",
    "public/phase_module_catalog.json",
)


class TrialDevOperationalSupportV1(BaseModel):
    """Exact recruitment capacity for one public operational design."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase_id: Literal["phase1", "phase2", "phase3"]
    enrollment_window_days: int = Field(..., ge=1)
    site_count_budget: int = Field(..., ge=1)
    site_strategy: str = Field(..., min_length=1)
    eligible_subject_count: int = Field(..., ge=0)

    def key(self) -> tuple[object, ...]:
        """Return the exact operational-design identity."""

        return (
            self.phase_id,
            self.enrollment_window_days,
            self.site_count_budget,
            self.site_strategy,
        )


class TrialDevDesignFrontierStratumV1(BaseModel):
    """One finite public design stratum and its exact Pareto frontier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase_id: Literal["phase1", "phase2", "phase3"]
    candidate_drug_ids: tuple[str, ...] = Field(..., min_length=1)
    endpoint_id: str | None = None
    treatment_discontinuation_strategy: str | None = Field(default=None, min_length=1)
    planning_analysis_population: Literal["complete_on_declared_adjustment_covariates"]
    design_cell_id: str = Field(..., min_length=1)
    interim_policy: str = Field(..., min_length=1)
    frontier: tuple[TrialDevDesignFrontierPointV1, ...]

    @model_validator(mode="after")
    def validate_frontier(self) -> TrialDevDesignFrontierStratumV1:
        """Require a canonical exact nondominated set."""

        if not self.frontier:
            raise ValueError("Design-frontier stratum requires at least one feasible statistical design.")
        if self.phase_id == "phase1" and self.treatment_discontinuation_strategy is not None:
            raise ValueError("Phase-1 design frontiers must not bind a treatment-discontinuation strategy.")
        if self.phase_id in {"phase2", "phase3"} and self.treatment_discontinuation_strategy is None:
            raise ValueError(f"{self.phase_id} design frontiers require a treatment-discontinuation strategy.")
        if tuple(sorted(self.candidate_drug_ids)) != self.candidate_drug_ids:
            raise ValueError("Design-frontier candidate ids must be canonically ordered.")
        keys = tuple(
            (point.target_sample_size, point.follow_up_days, point.allocation_ratio) for point in self.frontier
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("Design-frontier points must be unique and canonically ordered.")
        if any(_dominates(left, right) for left in self.frontier for right in self.frontier if left is not right):
            raise ValueError("Frozen design frontier contains a dominated point.")
        return self

    def key(self) -> tuple[object, ...]:
        """Return the exact score-bearing stratum identity."""

        return (
            self.phase_id,
            self.candidate_drug_ids,
            self.endpoint_id,
            self.treatment_discontinuation_strategy,
            self.planning_analysis_population,
            self.design_cell_id,
            self.interim_policy,
        )


class TrialDevDesignFrontierArtifactV1(BaseModel):
    """Checksum-bound public design frontiers frozen before model execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_phase_design_frontiers_v1"] = "trialdev_phase_design_frontiers_v1"
    version: Literal["v1"] = "v1"
    scenario_id: str = Field(..., min_length=1)
    source_artifact_checksums: dict[str, str] = Field(..., min_length=len(_FRONTIER_INPUTS))
    operational_support: tuple[TrialDevOperationalSupportV1, ...] = Field(..., min_length=1)
    strata: tuple[TrialDevDesignFrontierStratumV1, ...] = Field(..., min_length=1)
    checksum: str = Field(..., min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_artifact(self) -> TrialDevDesignFrontierArtifactV1:
        """Require exact input identity, unique strata, and artifact checksum."""

        if set(self.source_artifact_checksums) != set(_FRONTIER_INPUTS):
            raise ValueError("Design-frontier artifact has an incomplete source checksum set.")
        if any(len(value) != 64 for value in self.source_artifact_checksums.values()):
            raise ValueError("Design-frontier source checksums must be SHA-256 values.")
        support_keys = tuple(record.key() for record in self.operational_support)
        if support_keys != tuple(sorted(support_keys)) or len(support_keys) != len(set(support_keys)):
            raise ValueError("Operational-support records must be unique and canonically ordered.")
        keys = tuple(stratum.key() for stratum in self.strata)
        if keys != tuple(sorted(keys, key=repr)) or len(keys) != len(set(keys)):
            raise ValueError("Design-frontier strata must be unique and canonically ordered.")
        supported_phases = {record.phase_id for record in self.operational_support}
        if {stratum.phase_id for stratum in self.strata} - supported_phases:
            raise ValueError("Every design-frontier phase requires public operational support.")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        if self.checksum != compute_sha256_hex(payload):
            raise ValueError("Design-frontier artifact checksum mismatch.")
        return self


def _phase_module(*, scenario_root: Path, phase_id: str) -> PhaseModuleSpecV1:
    payload = read_json(Path(scenario_root) / "public" / "phase_module_catalog.json")
    modules = payload.get("phase_modules")
    if not isinstance(modules, list):
        raise ValueError("Design frontier requires the public phase-module catalog.")
    matches = [
        PhaseModuleSpecV1.model_validate(module)
        for module in modules
        if isinstance(module, dict) and str(module.get("phase_id")) == phase_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Design frontier requires exactly one module for phase={phase_id!r}.")
    return matches[0]


def _request_variant(
    *,
    request: TrialDevelopmentRequestV1,
    target_sample_size: int,
    follow_up_days: int,
    allocation_ratio: str,
) -> TrialDevelopmentRequestV1:
    payload = request.model_dump(mode="json")
    payload.update(
        {
            "target_sample_size": int(target_sample_size),
            "follow_up_days": int(follow_up_days),
            "allocation_ratio": str(allocation_ratio),
            "allocation_weights": [],
        }
    )
    return TrialDevelopmentRequestV1.model_validate(payload)


def _minimum_adequate_request(
    *,
    scenario_root: Path,
    request: TrialDevelopmentRequestV1,
    phase_id: str,
    follow_up_days: int,
    allocation_ratio: str,
    maximum_sample_size: int,
) -> tuple[TrialDevelopmentRequestV1, TrialDevPhaseDesignWitnessV1] | None:
    high_request = _request_variant(
        request=request,
        target_sample_size=maximum_sample_size,
        follow_up_days=follow_up_days,
        allocation_ratio=allocation_ratio,
    )
    high_witness = derive_phase_design_witness_v1(
        scenario_root=scenario_root,
        request=high_request,
        phase_id=phase_id,
    )
    if not high_witness.adequate:
        return None

    lower = max(2, len(request.candidate_drug_ids) + 1)
    upper = maximum_sample_size
    while lower < upper:
        midpoint = (lower + upper) // 2
        candidate = _request_variant(
            request=request,
            target_sample_size=midpoint,
            follow_up_days=follow_up_days,
            allocation_ratio=allocation_ratio,
        )
        witness = derive_phase_design_witness_v1(
            scenario_root=scenario_root,
            request=candidate,
            phase_id=phase_id,
        )
        if witness.adequate:
            upper = midpoint
        else:
            lower = midpoint + 1
    minimum = _request_variant(
        request=request,
        target_sample_size=lower,
        follow_up_days=follow_up_days,
        allocation_ratio=allocation_ratio,
    )
    witness = derive_phase_design_witness_v1(
        scenario_root=scenario_root,
        request=minimum,
        phase_id=phase_id,
    )
    if not witness.adequate:
        raise ValueError("Design adequacy is non-monotone over sample size for a fixed public design.")
    return minimum, witness


def _dominates(
    left: TrialDevDesignFrontierPointV1,
    right: TrialDevDesignFrontierPointV1,
) -> bool:
    no_worse = left.target_sample_size <= right.target_sample_size and left.follow_up_days <= right.follow_up_days
    strictly_better = left.target_sample_size < right.target_sample_size or left.follow_up_days < right.follow_up_days
    return bool(no_worse and strictly_better)


def _source_checksums(*, scenario_root: Path) -> dict[str, str]:
    root = Path(scenario_root)
    checksums: dict[str, str] = {}
    for relative_path in _FRONTIER_INPUTS:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Design-frontier source artifact is missing: {path}")
        checksums[relative_path] = sha256_file_hex(path)
    return checksums


def _operational_support(
    *,
    baseline: pd.DataFrame,
    modules: tuple[PhaseModuleSpecV1, ...],
) -> tuple[TrialDevOperationalSupportV1, ...]:
    records = [
        TrialDevOperationalSupportV1(
            phase_id=phase_id,
            enrollment_window_days=int(enrollment_window_days),
            site_count_budget=int(site_count_budget),
            site_strategy=str(site_strategy),
            eligible_subject_count=operational_support_count_v1(
                baseline=baseline,
                enrollment_window_days=int(enrollment_window_days),
                site_count_budget=int(site_count_budget),
                site_strategy=str(site_strategy),
            ),
        )
        for module in modules
        if (phase_id := str(module.phase_id)) in {"phase1", "phase2", "phase3"}
        for enrollment_window_days, site_count_budget, site_strategy in product(
            module.allowed_enrollment_window_days,
            module.allowed_site_count_budgets,
            module.allowed_site_strategies,
        )
    ]
    return tuple(sorted(records, key=lambda record: record.key()))


def _candidate_sets(*, scenario_root: Path) -> tuple[tuple[str, ...], ...]:
    candidates = tuple(sorted(candidate_ids_by_role_v1(scenario_root=scenario_root)["investigational"]))
    return tuple((candidate,) for candidate in candidates)


def _reference_request(
    *,
    scenario_id: str,
    phase_id: str,
    candidate_drug_ids: tuple[str, ...],
    endpoint_id: str | None,
    treatment_discontinuation_strategy: str | None,
    design_cell_id: str,
    interim_policy: str,
    enrollment_window_days: int,
    site_count_budget: int,
    site_strategy: str,
    selection_objective: str,
    follow_up_days: int,
    allocation_ratio: str,
    target_sample_size: int,
) -> TrialDevelopmentRequestV1:
    return TrialDevelopmentRequestV1.model_validate(
        {
            "scenario_id": scenario_id,
            "phase_id": phase_id,
            "candidate_drug_ids": candidate_drug_ids,
            "target_sample_size": target_sample_size,
            "endpoint_id": endpoint_id,
            "follow_up_days": follow_up_days,
            "enrollment_window_days": enrollment_window_days,
            "site_count_budget": site_count_budget,
            "allocation_ratio": allocation_ratio,
            "design_cell_id": design_cell_id,
            "treatment_discontinuation_strategy": treatment_discontinuation_strategy,
            "interim_policy": interim_policy,
            "site_strategy": site_strategy,
            "selection_objective": selection_objective,
        }
    )


def _frontier_for_reference(
    *,
    scenario_root: Path,
    reference_request: TrialDevelopmentRequestV1,
    module: PhaseModuleSpecV1,
) -> tuple[TrialDevDesignFrontierPointV1, ...]:
    if module.max_sample_size is None:
        raise ValueError("Design frontier requires a public maximum sample size.")
    candidates: list[TrialDevDesignFrontierPointV1] = []
    for follow_up_days in sorted(module.allowed_follow_up_days):
        for allocation_ratio in sorted(module.allowed_allocation_ratios):
            result = _minimum_adequate_request(
                scenario_root=scenario_root,
                request=reference_request,
                phase_id=str(reference_request.phase_id),
                follow_up_days=int(follow_up_days),
                allocation_ratio=str(allocation_ratio),
                maximum_sample_size=int(module.max_sample_size),
            )
            if result is None:
                continue
            candidate, witness = result
            assert candidate.target_sample_size is not None
            assert candidate.follow_up_days is not None
            candidates.append(
                TrialDevDesignFrontierPointV1(
                    target_sample_size=int(candidate.target_sample_size),
                    follow_up_days=int(candidate.follow_up_days),
                    allocation_ratio=str(candidate.allocation_ratio),
                    achieved_power=witness.achieved_power,
                    achieved_safety_absolute_risk_power=witness.achieved_safety_absolute_risk_power,
                    achieved_safety_excess_risk_power=witness.achieved_safety_excess_risk_power,
                )
            )
    return tuple(
        sorted(
            (
                candidate
                for candidate in candidates
                if not any(_dominates(other, candidate) for other in candidates if other is not candidate)
            ),
            key=lambda point: (
                point.target_sample_size,
                point.follow_up_days,
                point.allocation_ratio,
            ),
        )
    )


def build_phase_design_frontiers_v1(*, scenario_root: Path) -> TrialDevDesignFrontierArtifactV1:
    """Build each statistically distinct public randomized design frontier."""

    root = Path(scenario_root)
    policy = TrialDevPhaseDesignPolicyV1.model_validate(read_json(root / "public" / "phase_design_policy.json"))
    module_payload = read_json(root / "public" / "phase_module_catalog.json")
    raw_modules = module_payload.get("phase_modules") if isinstance(module_payload, dict) else None
    if not isinstance(raw_modules, list):
        raise ValueError("Design-frontier construction requires the public phase-module catalog.")
    randomized_phase_ids = tuple(
        sorted(
            str(module.get("phase_id"))
            for module in raw_modules
            if isinstance(module, dict) and str(module.get("phase_id")) in {"phase1", "phase2", "phase3"}
        )
    )
    if not randomized_phase_ids:
        raise ValueError("Design-frontier construction requires at least one randomized phase.")
    modules = tuple(_phase_module(scenario_root=root, phase_id=phase_id) for phase_id in randomized_phase_ids)
    baseline = pd.read_parquet(root / "public" / "observational_extract.parquet")
    operational_support = _operational_support(baseline=baseline, modules=modules)
    strata: list[TrialDevDesignFrontierStratumV1] = []
    for module in modules:
        phase_id = str(module.phase_id)
        rule = policy.rule_for_phase(phase_id)
        endpoints: tuple[str | None, ...] = (
            (None,) if phase_id == "phase1" else tuple(sorted(module.allowed_endpoint_ids))
        )
        for (
            candidate_drug_ids,
            endpoint_id,
            treatment_discontinuation_strategy,
            interim_policy,
        ) in product(
            _candidate_sets(scenario_root=root),
            endpoints,
            ((None,) if phase_id == "phase1" else tuple(sorted(module.allowed_treatment_discontinuation_strategies))),
            sorted(module.allowed_interim_policies),
        ):
            reference = _reference_request(
                scenario_id=policy.scenario_id,
                phase_id=phase_id,
                candidate_drug_ids=candidate_drug_ids,
                endpoint_id=endpoint_id,
                treatment_discontinuation_strategy=treatment_discontinuation_strategy,
                design_cell_id=rule.design_cell_id,
                interim_policy=str(interim_policy),
                enrollment_window_days=min(module.allowed_enrollment_window_days),
                site_count_budget=min(module.allowed_site_count_budgets),
                site_strategy=min(module.allowed_site_strategies),
                selection_objective=min(module.allowed_selection_objectives),
                follow_up_days=min(module.allowed_follow_up_days),
                allocation_ratio=min(module.allowed_allocation_ratios),
                target_sample_size=int(module.max_sample_size or 0),
            )
            validate_request_against_scenario_v1(scenario_root=root, request=reference)
            planning_population = (
                rule.planning_analysis_population
                if rule.planning_analysis_population is not None
                else rule.planning_safety_analysis_population
            )
            frontier = _frontier_for_reference(
                scenario_root=root,
                reference_request=reference,
                module=module,
            )
            strata.append(
                TrialDevDesignFrontierStratumV1(
                    phase_id=phase_id,
                    candidate_drug_ids=candidate_drug_ids,
                    endpoint_id=endpoint_id,
                    treatment_discontinuation_strategy=treatment_discontinuation_strategy,
                    planning_analysis_population=planning_population,
                    design_cell_id=rule.design_cell_id,
                    interim_policy=str(interim_policy),
                    frontier=frontier,
                )
            )
    ordered = tuple(sorted(strata, key=lambda stratum: repr(stratum.key())))
    payload = {
        "schema_id": "trialdev_phase_design_frontiers_v1",
        "version": "v1",
        "scenario_id": policy.scenario_id,
        "source_artifact_checksums": _source_checksums(scenario_root=root),
        "operational_support": [record.model_dump(mode="json") for record in operational_support],
        "strata": [stratum.model_dump(mode="json") for stratum in ordered],
    }
    payload["checksum"] = compute_sha256_hex(payload)
    artifact = TrialDevDesignFrontierArtifactV1.model_validate(payload)
    write_json(root / _FRONTIER_PATH, artifact.model_dump(mode="json"))
    return artifact


def load_phase_design_frontiers_v1(*, scenario_root: Path) -> TrialDevDesignFrontierArtifactV1:
    """Load a frozen frontier and reject missing or stale source identity."""

    root = Path(scenario_root)
    path = root / _FRONTIER_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Frozen public design-frontier artifact is missing: {path}")
    artifact = TrialDevDesignFrontierArtifactV1.model_validate(read_json(path))
    if artifact.source_artifact_checksums != _source_checksums(scenario_root=root):
        raise ValueError("Frozen public design-frontier source checksums are stale.")
    return artifact


def _stratum_for_request(
    *,
    artifact: TrialDevDesignFrontierArtifactV1,
    request: TrialDevelopmentRequestV1,
    design_witness: TrialDevPhaseDesignWitnessV1,
) -> TrialDevDesignFrontierStratumV1:
    planning_population = design_witness.evidence.get("planning_analysis_population")
    if planning_population is None:
        planning_population = design_witness.evidence.get("planning_safety_analysis_population")
    expected_endpoint = design_witness.evidence.get("primary_endpoint_id")
    expected_design_cell = design_witness.evidence.get("design_cell_id")
    expected_interim_policy = design_witness.evidence.get("supported_interim_policy")
    key = (
        str(request.phase_id),
        tuple(sorted(request.candidate_drug_ids)),
        expected_endpoint,
        request.treatment_discontinuation_strategy,
        str(planning_population),
        str(expected_design_cell),
        str(expected_interim_policy),
    )
    matches = tuple(stratum for stratum in artifact.strata if stratum.key() == key)
    if len(matches) != 1:
        raise ValueError(
            "Frozen public design frontier requires exactly one matching design stratum; "
            f"observed={len(matches)} key={key!r}."
        )
    return matches[0]


def _operational_support_for_request(
    *,
    artifact: TrialDevDesignFrontierArtifactV1,
    request: TrialDevelopmentRequestV1,
) -> int:
    if request.enrollment_window_days is None or request.site_count_budget is None or request.site_strategy is None:
        raise ValueError("Randomized design efficiency requires a complete operational design.")
    key = (
        str(request.phase_id),
        int(request.enrollment_window_days),
        int(request.site_count_budget),
        str(request.site_strategy),
    )
    matches = tuple(record for record in artifact.operational_support if record.key() == key)
    if len(matches) != 1:
        raise ValueError(
            "Frozen public design frontier requires exactly one matching operational-support record; "
            f"observed={len(matches)} key={key!r}."
        )
    return int(matches[0].eligible_subject_count)


def select_operational_support_v1(
    *,
    artifact: TrialDevDesignFrontierArtifactV1,
    phase_id: str,
    target_sample_size: int,
) -> TrialDevOperationalSupportV1:
    """Select one deterministic nondominated recruitment design with adequate support."""

    feasible = tuple(
        record
        for record in artifact.operational_support
        if record.phase_id == phase_id and record.eligible_subject_count >= target_sample_size
    )
    if not feasible:
        raise ValueError(
            "Public operational support cannot recruit any released statistical frontier point; "
            f"phase={phase_id!r} target_sample_size={target_sample_size}."
        )

    def dominates(
        left: TrialDevOperationalSupportV1,
        right: TrialDevOperationalSupportV1,
    ) -> bool:
        no_worse = (
            left.enrollment_window_days <= right.enrollment_window_days
            and left.site_count_budget <= right.site_count_budget
        )
        strictly_better = (
            left.enrollment_window_days < right.enrollment_window_days
            or left.site_count_budget < right.site_count_budget
        )
        return bool(no_worse and strictly_better)

    nondominated = tuple(
        record for record in feasible if not any(dominates(other, record) for other in feasible if other is not record)
    )
    return min(nondominated, key=lambda record: record.key())


def derive_phase_design_efficiency_v1(
    *,
    scenario_root: Path,
    request: TrialDevelopmentRequestV1,
    design_witness: TrialDevPhaseDesignWitnessV1,
) -> TrialDevDesignEfficiencyV1:
    """Compare a request with its independently frozen public Pareto frontier."""

    phase_id = str(request.phase_id)
    if phase_id not in {"phase1", "phase2", "phase3"}:
        raise ValueError("Design efficiency applies only to randomized TrialDev phases.")
    validate_request_against_scenario_v1(
        scenario_root=Path(scenario_root),
        request=request,
    )
    if request.allocation_weights:
        raise ValueError("Design frontier does not support custom allocation weights.")
    artifact = load_phase_design_frontiers_v1(scenario_root=Path(scenario_root))
    stratum = _stratum_for_request(
        artifact=artifact,
        request=request,
        design_witness=design_witness,
    )
    frontier = stratum.frontier
    assert request.target_sample_size is not None
    assert request.follow_up_days is not None
    operational_support = _operational_support_for_request(
        artifact=artifact,
        request=request,
    )
    operational_headroom = max(0, operational_support - int(request.target_sample_size))
    operational_shortage = max(0, int(request.target_sample_size) - operational_support)
    statistically_adequate = bool(design_witness.adequate)
    operationally_feasible = operational_shortage == 0
    design_valid = statistically_adequate and operationally_feasible
    submitted_key = (
        int(request.target_sample_size),
        int(request.follow_up_days),
        str(request.allocation_ratio),
    )
    frontier_keys = {(point.target_sample_size, point.follow_up_days, point.allocation_ratio) for point in frontier}
    dominated = design_valid and any(
        point.target_sample_size <= int(request.target_sample_size)
        and point.follow_up_days <= int(request.follow_up_days)
        and (
            point.target_sample_size < int(request.target_sample_size)
            or point.follow_up_days < int(request.follow_up_days)
        )
        for point in frontier
    )
    minimum_n = min(point.target_sample_size for point in frontier)
    minimum_follow_up = min(point.follow_up_days for point in frontier)
    n_delta = int(request.target_sample_size) - minimum_n
    follow_up_delta = int(request.follow_up_days) - minimum_follow_up
    return TrialDevDesignEfficiencyV1(
        statistically_adequate=statistically_adequate,
        operationally_feasible=operationally_feasible,
        design_valid=design_valid,
        on_frontier=design_valid and submitted_key in frontier_keys,
        dominated_by_frontier=dominated,
        operational_support=operational_support,
        operational_headroom=operational_headroom,
        operational_shortage=operational_shortage,
        minimum_frontier_participants=minimum_n,
        minimum_frontier_follow_up_days=minimum_follow_up,
        participant_excess_vs_minimum=max(0, n_delta),
        participant_shortage_vs_minimum=max(0, -n_delta),
        follow_up_excess_days_vs_minimum=max(0, follow_up_delta),
        follow_up_shortage_days_vs_minimum=max(0, -follow_up_delta),
        achieved_power=design_witness.achieved_power,
        target_power=design_witness.target_power,
        achieved_safety_absolute_risk_power=design_witness.achieved_safety_absolute_risk_power,
        achieved_safety_excess_risk_power=design_witness.achieved_safety_excess_risk_power,
        target_safety_decision_power=design_witness.target_safety_decision_power,
        frontier=frontier,
    )


def derive_phase_resource_consequence_v1(
    *,
    request: TrialDevelopmentRequestV1,
    design_efficiency: TrialDevDesignEfficiencyV1,
    entered_after_unsupported_advance: bool,
) -> TrialDevPhaseResourceConsequenceV1:
    """Derive resource consequences without scalarizing the Pareto frontier."""

    if request.phase_id not in {"phase1", "phase2", "phase3"}:
        raise ValueError("Phase resource consequences apply only to randomized TrialDev phases.")
    if (
        request.target_sample_size is None
        or request.follow_up_days is None
        or request.enrollment_window_days is None
        or request.site_count_budget is None
    ):
        raise ValueError(
            "Randomized phase resource consequences require sample size, follow-up, "
            "enrollment window, and site budget."
        )
    target_n = int(request.target_sample_size)
    follow_up = int(request.follow_up_days)
    dominators = tuple(
        point
        for point in design_efficiency.frontier
        if point.target_sample_size <= target_n
        and point.follow_up_days <= follow_up
        and (point.target_sample_size < target_n or point.follow_up_days < follow_up)
    )
    if not design_efficiency.statistically_adequate:
        status = "statistically_inadequate"
        dominators = ()
    elif not design_efficiency.operationally_feasible:
        status = "operationally_infeasible"
        dominators = ()
    elif design_efficiency.on_frontier:
        status = "valid_frontier"
    elif dominators:
        status = "valid_dominated"
    else:
        status = "valid_nondominated"

    participant_reductions = tuple(target_n - point.target_sample_size for point in dominators)
    follow_up_reductions = tuple(follow_up - point.follow_up_days for point in dominators)
    burden = target_n * follow_up
    burden_reductions = tuple(burden - point.target_sample_size * point.follow_up_days for point in dominators)

    def _bounds(values: tuple[int, ...]) -> tuple[int, int]:
        return (min(values), max(values)) if values else (0, 0)

    participant_min, participant_max = _bounds(participant_reductions)
    follow_up_min, follow_up_max = _bounds(follow_up_reductions)
    burden_min, burden_max = _bounds(burden_reductions)
    return TrialDevPhaseResourceConsequenceV1(
        phase_id=request.phase_id,
        request_checksum=request.checksum(),
        target_sample_size=target_n,
        follow_up_days=follow_up,
        enrollment_window_days=int(request.enrollment_window_days),
        site_count_budget=int(request.site_count_budget),
        participant_follow_up_days=burden,
        statistically_adequate=design_efficiency.statistically_adequate,
        operationally_feasible=design_efficiency.operationally_feasible,
        design_status=status,
        operational_support=design_efficiency.operational_support,
        operational_headroom=design_efficiency.operational_headroom,
        operational_shortage=design_efficiency.operational_shortage,
        achieved_power=design_efficiency.achieved_power,
        target_power=design_efficiency.target_power,
        achieved_safety_absolute_risk_power=(design_efficiency.achieved_safety_absolute_risk_power),
        achieved_safety_excess_risk_power=(design_efficiency.achieved_safety_excess_risk_power),
        target_safety_decision_power=design_efficiency.target_safety_decision_power,
        participant_excess_vs_minimum=design_efficiency.participant_excess_vs_minimum,
        participant_shortage_vs_minimum=design_efficiency.participant_shortage_vs_minimum,
        follow_up_excess_days_vs_minimum=design_efficiency.follow_up_excess_days_vs_minimum,
        follow_up_shortage_days_vs_minimum=design_efficiency.follow_up_shortage_days_vs_minimum,
        dominating_frontier=dominators,
        avoidable_participants_min=participant_min,
        avoidable_participants_max=participant_max,
        avoidable_follow_up_days_min=follow_up_min,
        avoidable_follow_up_days_max=follow_up_max,
        avoidable_participant_follow_up_days_min=burden_min,
        avoidable_participant_follow_up_days_max=burden_max,
        entered_after_unsupported_advance=entered_after_unsupported_advance,
    )


def derive_programme_resource_consequence_v1(
    phases: tuple[TrialDevPhaseResourceConsequenceV1, ...],
) -> TrialDevProgrammeResourceConsequenceV1:
    """Aggregate exact phase vectors without introducing a resource weighting."""

    return TrialDevProgrammeResourceConsequenceV1(
        phases=phases,
        total_participants=sum(row.target_sample_size for row in phases),
        total_protocol_follow_up_days=sum(row.follow_up_days for row in phases),
        total_enrollment_window_days=sum(row.enrollment_window_days for row in phases),
        total_site_phase_budget=sum(row.site_count_budget for row in phases),
        total_planned_phase_duration_days=sum(row.enrollment_window_days + row.follow_up_days for row in phases),
        total_participant_follow_up_days=sum(row.participant_follow_up_days for row in phases),
        participant_excess_vs_minimum=sum(row.participant_excess_vs_minimum for row in phases),
        participant_shortage_vs_minimum=sum(row.participant_shortage_vs_minimum for row in phases),
        follow_up_excess_days_vs_minimum=sum(row.follow_up_excess_days_vs_minimum for row in phases),
        follow_up_shortage_days_vs_minimum=sum(row.follow_up_shortage_days_vs_minimum for row in phases),
        statistically_inadequate_phases=sum(not row.statistically_adequate for row in phases),
        operationally_infeasible_phases=sum(not row.operationally_feasible for row in phases),
        dominated_phases=sum(row.design_status == "valid_dominated" for row in phases),
        design_avoidable_participants_min=sum(row.avoidable_participants_min for row in phases),
        design_avoidable_participants_max=sum(row.avoidable_participants_max for row in phases),
        design_avoidable_follow_up_days_min=sum(row.avoidable_follow_up_days_min for row in phases),
        design_avoidable_follow_up_days_max=sum(row.avoidable_follow_up_days_max for row in phases),
        design_avoidable_participant_follow_up_days_min=sum(
            row.avoidable_participant_follow_up_days_min for row in phases
        ),
        design_avoidable_participant_follow_up_days_max=sum(
            row.avoidable_participant_follow_up_days_max for row in phases
        ),
        late_continuation_participants=sum(
            row.target_sample_size for row in phases if row.entered_after_unsupported_advance
        ),
        late_continuation_protocol_follow_up_days=sum(
            row.follow_up_days for row in phases if row.entered_after_unsupported_advance
        ),
        late_continuation_enrollment_window_days=sum(
            row.enrollment_window_days for row in phases if row.entered_after_unsupported_advance
        ),
        late_continuation_site_phase_budget=sum(
            row.site_count_budget for row in phases if row.entered_after_unsupported_advance
        ),
        late_continuation_participant_follow_up_days=sum(
            row.participant_follow_up_days for row in phases if row.entered_after_unsupported_advance
        ),
    )
