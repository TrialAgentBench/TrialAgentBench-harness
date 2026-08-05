"""Sequential evaluator controls for clinical program trajectories."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from trialagentbench_harness.contracts.release.artifacts import (
    TRIALDEV_FIXED_TRAJECTORY_REPLICATE_MEMBERS,
)
from trialagentbench_harness.contracts.trialdev.programme import (
    TRIALDEV_PROGRAMME_STATE_ADAPTER_V1,
    TrialDevActionIdV1,
    TrialDevActionSelectionV1,
    TrialDevCheckpointIdV1,
    TrialDevCheckpointOutcomeV1,
    TrialDevEvidenceReferenceV1,
    TrialDevPolicyBindingV1,
    TrialDevProgrammeStateV1,
    TrialDevSingleAssetProgrammeStateV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_evaluation_target_register import (
    required_trialdev_lanes_v1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_public_phase_replay import (
    TrialDevPhaseReplayCaseV1,
)
from trialagentbench_harness.trialdev.grading.decision_evidence import (
    declared_serious_safety_surfaces_v1,
    derive_phase_design_witness_v1,
)
from trialagentbench_harness.trialdev.grading.design_frontier import (
    derive_phase_design_efficiency_v1,
    derive_phase_resource_consequence_v1,
    derive_programme_resource_consequence_v1,
)
from trialagentbench_harness.trialdev.grading.evaluation_target_register import (
    load_evaluation_target_index,
    score_evaluation_target,
)
from trialagentbench_harness.trialdev.grading.grade import grade_item_v1, grade_report_payload_v1
from trialagentbench_harness.trialdev.grading.hashing import sha256_file_hex
from trialagentbench_harness.trialdev.grading.io import read_json, write_json
from trialagentbench_harness.trialdev.grading.models import (
    TrialDevelopmentGradeReportV1,
    TrialDevelopmentInvalidAttemptReportV1,
    TrialDevelopmentLaneScoreRecordV1,
    TrialDevelopmentScoringContextV1,
    TrialDevelopmentSubmissionV1,
    TrialDevelopmentTerminalSummaryV1,
    TrialDevelopmentTrajectoryReplayReportV1,
    TrialDevelopmentValidityReportV1,
    TrialDevPhaseResourceConsequenceV1,
)
from trialagentbench_harness.trialdev.grading.statistics import complete_binary_indicator_v1
from trialagentbench_harness.trialdev.programme import (
    build_checkpoint_action_policy_v1,
    transition_programme_state_v1,
)
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentRequestV1
from trialagentbench_harness.trialdev.share.sequential import (
    TrialDevelopmentPhaseActionPolicyV1,
    TrialDevelopmentPhaseAnalysisSubmissionV1,
    TrialDevelopmentPhaseDecisionSubmissionV1,
    TrialDevelopmentProgramLoopManifestV1,
    TrialDevelopmentTrialOutputManifestV1,
    validate_design_request_file_v1,
    validate_phase_decision_against_policy_v1,
    validate_trial_output_bundle_v1,
)
from trialagentbench_harness.trialdev.share.validate import (
    candidate_ids_by_role_v1,
)

__all__ = [
    "TrialMaterializationRejectedError",
    "advance_observational_programme_state_v1",
    "advance_program_state_v1",
    "build_initial_program_state_v1",
    "final_decision_lane_scores_from_trajectory",
    "grade_trajectory_v1",
    "materialize_phase_v1",
    "phase_summary_v1",
    "validate_program_state_file_v1",
]


class TrialMaterializationRejectedError(ValueError):
    """A participant-correctable phase-request feasibility rejection."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason).rstrip()
        message = self.reason if self.reason.endswith((".", "!", "?")) else f"{self.reason}."
        super().__init__(f"Phase materialization rejected the request: {message}")


_PHASE_ORDER = ("observational_review", "phase1", "phase2", "phase3")
_MATERIALIZABLE_PHASES = ("phase1", "phase2", "phase3")
_PHASE_TO_CHECKPOINT = {
    "phase1": "early_safety_study",
    "phase2": "proof_of_concept",
    "phase3": "confirmation",
}
_CHECKPOINT_TO_PHASE = {checkpoint: phase for phase, checkpoint in _PHASE_TO_CHECKPOINT.items()}
_CONTINUE_ACTIONS = {"advance_to_proof_of_concept", "advance_to_confirmation"}
_TERMINAL_ACTIONS = {
    "withhold_nomination",
    "stop_development",
    "declare_success",
    "declare_failure",
    "declare_inconclusive",
}


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = read_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def _candidate_ids(*, scenario_root: Path) -> tuple[str, ...]:
    return tuple(sorted(candidate_ids_by_role_v1(scenario_root=Path(scenario_root))["investigational"]))


def _scenario_id(*, scenario_root: Path) -> str:
    payload = _read_json_object(Path(scenario_root) / "public" / "eval_contract.json")
    scenario_id = str(payload.get("scenario_id", ""))
    if not scenario_id:
        raise ValueError("eval_contract.json is missing scenario_id.")
    return scenario_id


def _public_release_provenance(*, scenario_root: Path) -> tuple[str, str]:
    """Return participant-verifiable release and scenario identities."""

    scenario = Path(scenario_root)
    manifest_path = scenario.parent / "benchmark_suite_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("TrialDev participant release is missing benchmark_suite_manifest.json.")
    return sha256_file_hex(manifest_path), _scenario_id(scenario_root=scenario)


def _phase_index(phase_id: str) -> int:
    if str(phase_id) not in _PHASE_ORDER:
        raise ValueError(f"Unsupported phase_id: {phase_id!r}.")
    return _PHASE_ORDER.index(str(phase_id))


def _next_phase(*, phase_id: str) -> tuple[str, str]:
    idx = _phase_index(str(phase_id))
    if idx >= len(_PHASE_ORDER) - 1:
        return str(phase_id), "completed"
    return _PHASE_ORDER[idx + 1], "active"


def _program_loop_manifest(*, scenario_root: Path) -> TrialDevelopmentProgramLoopManifestV1:
    path = Path(scenario_root) / "public" / "program_loop_manifest.json"
    return TrialDevelopmentProgramLoopManifestV1.model_validate(_read_json_object(path))


def _phase_action_policy(*, scenario_root: Path) -> TrialDevelopmentPhaseActionPolicyV1:
    return TrialDevelopmentPhaseActionPolicyV1.model_validate(
        _read_json_object(Path(scenario_root) / "public" / "phase_action_policy.json")
    )


def _action_advances_program(action: str) -> bool:
    return str(action) in {
        "advance_to_proof_of_concept",
        "advance_to_confirmation",
        "declare_success",
    }


def _phase_policy_modes(*, scenario_root: Path) -> dict[str, str]:
    manifest = _program_loop_manifest(scenario_root=Path(scenario_root))
    return {str(key): str(value) for key, value in manifest.phase_policy_modes.items()}


def _read_scoring_context(
    *,
    scenario_root: Path,
    trajectory_root: Path,
    scoring_context_path: Path | None,
) -> TrialDevelopmentScoringContextV1:
    path = (
        Path(scoring_context_path)
        if scoring_context_path is not None
        else Path(trajectory_root) / "scoring_context.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing TrialDev scoring context: {path}")
    context = TrialDevelopmentScoringContextV1.model_validate(_read_json_object(path))
    scenario_id = _scenario_id(scenario_root=Path(scenario_root))
    if str(context.scenario_id) != str(scenario_id):
        raise ValueError("scoring_context scenario_id does not match scenario bundle.")
    return context


def _phase_scoring_objective(
    *,
    context: TrialDevelopmentScoringContextV1,
    phase_id: str,
) -> str:
    objective_id = context.phase_scoring_objectives.get(str(phase_id))
    if objective_id is None:
        raise ValueError(f"scoring_context missing phase_scoring_objective for phase_id={phase_id!r}.")
    return str(objective_id)


def _phase_module_payload(*, scenario_root: Path, phase_id: str) -> dict[str, Any]:
    payload = _read_json_object(Path(scenario_root) / "public" / "phase_module_catalog.json")
    raw = payload.get("phase_modules")
    if not isinstance(raw, list):
        raise ValueError("phase_module_catalog.json must contain phase_modules list.")
    for record in raw:
        if not isinstance(record, dict):
            raise ValueError("phase_module_catalog.json phase_modules entries must be objects.")
        if str(record.get("phase_id", "")) == str(phase_id):
            return dict(record)
    raise ValueError(f"phase_module_catalog.json missing phase_id={phase_id!r}.")


def _file_checksums(*, root: Path, rel_paths: tuple[str, ...]) -> dict[str, str]:
    return {str(rel): sha256_file_hex(Path(root) / str(rel)) for rel in rel_paths}


def _fixed_evidence_participant_count(out_dir: Path) -> int:
    """Validate one copied fixed trajectory and return its participant count."""

    root = Path(out_dir)
    missing = sorted(name for name in TRIALDEV_FIXED_TRAJECTORY_REPLICATE_MEMBERS if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(f"Fixed TrialDev trajectory is incomplete: {missing!r}.")
    return int(len(pd.read_parquet(root / "participants.parquet")))


def phase_summary_v1(*, scenario_root: Path, trial_output_root: Path) -> dict[str, Any]:
    """Summarize participant-visible randomized evidence without reference data."""

    out_dir = Path(trial_output_root)
    participants = pd.read_parquet(Path(out_dir) / "participants.parquet")
    endpoints = pd.read_parquet(Path(out_dir) / "endpoints.parquet")
    safety = pd.read_parquet(Path(out_dir) / "safety.parquet")
    arm_mapping = _read_json_object(Path(out_dir) / "arm_mapping.json")
    if "ARM" not in participants.columns or "ARM" not in endpoints.columns or "ARM" not in safety.columns:
        raise ValueError("Materialized trial tables must expose ARM in every table.")
    endpoint_events = complete_binary_indicator_v1(endpoints["EVENT"])
    event_rate_by_arm = (
        endpoints.assign(EVENT=endpoint_events.astype(float))
        .groupby("ARM", dropna=False)["EVENT"]
        .mean()
        .sort_index()
        .to_dict()
    )
    serious_surfaces = declared_serious_safety_surfaces_v1(scenario_root=Path(scenario_root))
    ae_columns = tuple(surface.event_column for surface in serious_surfaces)
    missing_ae = tuple(sorted(set(ae_columns) - set(str(column) for column in safety.columns)))
    if missing_ae:
        raise ValueError(f"Materialized safety table lacks declared event columns: {missing_ae!r}.")
    ae_event_count = 0
    for col in ae_columns:
        ae_event_count += int(complete_binary_indicator_v1(safety[col]).sum())
    nuisance_summary: dict[str, float] = {}
    for col in ("ADHERENCE_INDEX", "EARLY_RESCUE_RISK", "ASSESSMENT_QUALITY"):
        if col in participants.columns:
            values = pd.to_numeric(participants[col], errors="coerce").dropna()
            if not values.empty:
                nuisance_summary[f"{col.lower()}_mean"] = float(values.mean())
    if "PROTOCOL_DEVIATION_FLAG" in participants.columns:
        nuisance_summary["protocol_deviation_rate"] = float(
            complete_binary_indicator_v1(participants["PROTOCOL_DEVIATION_FLAG"]).mean()
        )
    return {
        "n_participants": int(len(participants)),
        "arm_counts": {
            str(k): int(v) for k, v in participants["ARM"].astype("string").value_counts().sort_index().items()
        },
        "endpoint_event_rate_by_arm": {str(k): float(v) for k, v in event_rate_by_arm.items()},
        "safety_event_count": int(ae_event_count),
        "execution_nuisance_summary": nuisance_summary,
        "arm_mapping": {
            "control_arm_id": str(arm_mapping.get("control_arm_id", "")),
            "candidate_arm_ids": list(arm_mapping.get("candidate_arm_ids", []) or []),
            "drug_id_by_arm": dict(arm_mapping.get("drug_id_by_arm", {}) or {}),
            "arm_role_by_id": dict(arm_mapping.get("arm_role_by_id", {}) or {}),
        },
    }


def build_initial_program_state_v1(
    *,
    scenario_root: Path,
    programme_id: str,
    objective_id: str,
    out_path: Path | None = None,
) -> TrialDevProgrammeStateV1:
    """Build one objective-bound state at the observational checkpoint."""
    scenario = Path(scenario_root)
    public = scenario / "public"
    action_policy_path = public / "phase_action_policy.json"
    objective_policy_path = public / "objective_charter.json"
    design_menu_path = public / "phase_module_catalog.json"
    for path in (action_policy_path, objective_policy_path, design_menu_path):
        if not path.is_file():
            raise FileNotFoundError(f"TrialDev programme policy artifact not found: {path}.")
    protocol_path = public / "observational_method_catalog.json"
    extract_path = public / "observational_extract.parquet"
    source_identity, world_id = _public_release_provenance(scenario_root=scenario)
    initial_evidence = (
        TrialDevEvidenceReferenceV1(
            evidence_id=f"public:{protocol_path.name}",
            evidence_kind="protocol",
            checkpoint_id="observational_review",
            asset_id=None,
            evidence_protocol_id="observational_review_v1",
            evidence_protocol_checksum=sha256_file_hex(protocol_path),
            source_family_id=source_identity,
            world_id=world_id,
            relative_path=f"public/{protocol_path.name}",
            artifact_sha256=sha256_file_hex(protocol_path),
        ),
        TrialDevEvidenceReferenceV1(
            evidence_id=f"public:{extract_path.name}",
            evidence_kind="dataset",
            checkpoint_id="observational_review",
            asset_id=None,
            evidence_protocol_id="observational_review_v1",
            evidence_protocol_checksum=sha256_file_hex(protocol_path),
            source_family_id=source_identity,
            world_id=world_id,
            relative_path=f"public/{extract_path.name}",
            artifact_sha256=sha256_file_hex(extract_path),
        ),
        TrialDevEvidenceReferenceV1(
            evidence_id=f"public:{objective_policy_path.name}",
            evidence_kind="protocol",
            checkpoint_id="observational_review",
            asset_id=None,
            evidence_protocol_id="observational_review_v1",
            evidence_protocol_checksum=sha256_file_hex(protocol_path),
            source_family_id=source_identity,
            world_id=world_id,
            relative_path=f"public/{objective_policy_path.name}",
            artifact_sha256=sha256_file_hex(objective_policy_path),
        ),
    )
    state = TrialDevSingleAssetProgrammeStateV1(
        programme_id=str(programme_id),
        scenario_id=_scenario_id(scenario_root=scenario),
        stream_id="single_asset_development",
        current_checkpoint_id="observational_review",
        candidate_asset_ids=_candidate_ids(scenario_root=scenario),
        policy_binding=TrialDevPolicyBindingV1(
            stream_id="single_asset_development",
            objective_id=str(objective_id),
            objective_policy_checksum=sha256_file_hex(objective_policy_path),
            action_policy_checksum=sha256_file_hex(action_policy_path),
            design_menu_checksum=sha256_file_hex(design_menu_path),
        ),
        evidence=initial_evidence,
    )
    if out_path is not None:
        write_json(Path(out_path), state.model_dump(mode="json", exclude_none=True))
    return state


def validate_program_state_file_v1(*, state_path: Path) -> TrialDevProgrammeStateV1:
    """Validate one evaluator-held program state file."""
    return TRIALDEV_PROGRAMME_STATE_ADAPTER_V1.validate_python(_read_json_object(Path(state_path)))


def advance_observational_programme_state_v1(
    *,
    state: TrialDevProgrammeStateV1,
    submission: TrialDevelopmentSubmissionV1,
    submission_path: Path,
    out_path: Path | None = None,
) -> TrialDevProgrammeStateV1:
    """Apply an accepted observational nomination or withholding decision."""

    if state.current_checkpoint_id != "observational_review":
        raise ValueError("Observational decisions require the observational-review state.")
    if submission.request.phase_id != "observational_review":
        raise ValueError("Observational state requires an observational-review submission.")
    if submission.scenario_id != state.scenario_id:
        raise ValueError("Observational submission scenario does not match programme state.")
    decision = submission.program_decision
    if decision.objective_id != state.policy_binding.objective_id or decision.decision_action is None:
        raise ValueError("Observational decision does not match the objective-bound programme policy.")
    selected_asset = decision.recommended_drug_id
    method_ids = {
        str(estimate.method_route_id)
        for estimate in submission.analysis_report.candidate_utility_estimates
        if estimate.method_route_id is not None
    }
    if len(method_ids) > 1:
        raise ValueError("One observational decision cannot combine analysis methods.")
    analysis_method_id = next(iter(method_ids), str(submission.analysis_report.primary_resolution_evidence_class))
    artifact_sha256 = sha256_file_hex(Path(submission_path))
    provenance = state.evidence[0]
    checkpoint_evidence = tuple(
        TrialDevEvidenceReferenceV1(
            evidence_id=evidence_id,
            evidence_kind="analysis",
            checkpoint_id="observational_review",
            asset_id=selected_asset,
            evidence_protocol_id="observational_analysis_submission_v1",
            evidence_protocol_checksum=state.policy_binding.objective_policy_checksum,
            source_family_id=provenance.source_family_id,
            world_id=provenance.world_id,
            relative_path=f"run/observational_review/submission.json#{evidence_id}",
            artifact_sha256=artifact_sha256,
        )
        for evidence_id in decision.supporting_evidence_ids
    )
    selection = TrialDevActionSelectionV1(
        state_checksum=cast(str, state.checksum),
        checkpoint_id="observational_review",
        action_id=cast(TrialDevActionIdV1, str(decision.decision_action)),
        target_asset_id=selected_asset,
        analysis_method_id=analysis_method_id,
        supporting_evidence_ids=decision.supporting_evidence_ids,
        justification="The nomination decision is bound to the submitted observational analysis.",
    )
    next_evidence: tuple[TrialDevEvidenceReferenceV1, ...] = tuple()
    if decision.decision_action == "nominate_for_early_study":
        if selected_asset is None:
            raise ValueError("Nomination requires one selected asset.")
        next_evidence = (
            TrialDevEvidenceReferenceV1(
                evidence_id="public:phase1:design_protocol",
                evidence_kind="protocol",
                checkpoint_id="early_safety_study",
                asset_id=selected_asset,
                evidence_protocol_id="phase1_design_v1",
                evidence_protocol_checksum=state.policy_binding.design_menu_checksum,
                source_family_id=provenance.source_family_id,
                world_id=provenance.world_id,
                relative_path="public/phase_module_catalog.json",
                artifact_sha256=state.policy_binding.design_menu_checksum,
            ),
        )
    next_state = transition_programme_state_v1(
        state=state,
        action_policy=build_checkpoint_action_policy_v1(
            state=state,
        ),
        selection=selection,
        outcome=TrialDevCheckpointOutcomeV1(
            reach_status="reached",
            submission_status="accepted",
            analysis_status=(
                "non_estimable"
                if submission.analysis_report.response_branch == "qualified_non_nomination"
                else "estimable"
            ),
            execution_status="completed",
        ),
        checkpoint_evidence=checkpoint_evidence,
        next_evidence=next_evidence,
    )
    if out_path is not None:
        write_json(Path(out_path), next_state.model_dump(mode="json", exclude_none=True))
    return cast(TrialDevProgrammeStateV1, next_state)


def _validate_phase_request(
    *,
    scenario_root: Path,
    state: TrialDevProgrammeStateV1,
    request: TrialDevelopmentRequestV1,
) -> None:
    scenario_id = _scenario_id(scenario_root=Path(scenario_root))
    if str(state.scenario_id) != str(scenario_id):
        raise ValueError("Program state scenario_id does not match the scenario bundle.")
    if str(request.scenario_id) != str(scenario_id):
        raise TrialMaterializationRejectedError("Request scenario_id does not match the scenario bundle.")
    if str(request.phase_id) not in _MATERIALIZABLE_PHASES:
        raise TrialMaterializationRejectedError("Only phase1, phase2, and phase3 can be materialized.")
    if state.policy_binding.action_policy_checksum != sha256_file_hex(
        Path(scenario_root) / "public" / "phase_action_policy.json"
    ):
        raise ValueError("Programme state action policy does not match the scenario policy artifact.")
    mode = str(_phase_policy_modes(scenario_root=Path(scenario_root)).get(str(request.phase_id), "optional"))
    if mode == "not_available":
        raise TrialMaterializationRejectedError(
            f"Request phase_id is not available for this scenario: {request.phase_id!r}."
        )
    expected_phase = _CHECKPOINT_TO_PHASE.get(str(state.current_checkpoint_id))
    if str(request.phase_id) != expected_phase:
        raise TrialMaterializationRejectedError("Request phase_id must match the current program state phase.")
    if state.terminal_disposition != "active":
        raise TrialMaterializationRejectedError("Cannot materialize a trial from a terminal program state.")
    eligible = {str(state.active_asset_id)} if state.active_asset_id is not None else set()
    requested = set(str(v) for v in request.candidate_drug_ids)
    if not requested <= eligible:
        raise TrialMaterializationRejectedError(
            "Request candidate_drug_ids must be a subset of eligible candidates in program state."
        )
    candidate_count = int(len(request.candidate_drug_ids))
    if candidate_count != 1:
        raise TrialMaterializationRejectedError(
            "Randomized TrialDev requests require exactly one investigational regimen plus control."
        )


def _next_state(
    *,
    previous: TrialDevProgrammeStateV1,
    request: TrialDevelopmentRequestV1,
    analysis_artifact_sha256: str,
    supporting_evidence_ids: tuple[str, ...],
    analysis_method_id: str,
    decision_action: str = "advance_to_proof_of_concept",
) -> TrialDevProgrammeStateV1:
    if _PHASE_TO_CHECKPOINT.get(str(request.phase_id)) != previous.current_checkpoint_id:
        raise ValueError("Phase request does not match the canonical programme checkpoint.")
    if not supporting_evidence_ids:
        raise ValueError("A phase decision requires cited analysis evidence.")
    provenance = previous.evidence[0]
    checkpoint_evidence = tuple(
        TrialDevEvidenceReferenceV1(
            evidence_id=evidence_id,
            evidence_kind="analysis",
            checkpoint_id=previous.current_checkpoint_id,
            asset_id=previous.active_asset_id,
            evidence_protocol_id=f"{request.phase_id}_analysis_v1",
            evidence_protocol_checksum=previous.policy_binding.design_menu_checksum,
            source_family_id=provenance.source_family_id,
            world_id=provenance.world_id,
            relative_path=f"run/{request.phase_id}/analysis_submission.json#{evidence_id}",
            artifact_sha256=analysis_artifact_sha256,
        )
        for evidence_id in supporting_evidence_ids
    )
    selection = TrialDevActionSelectionV1(
        state_checksum=cast(str, previous.checksum),
        checkpoint_id=previous.current_checkpoint_id,
        action_id=cast(TrialDevActionIdV1, str(decision_action)),
        analysis_method_id=str(analysis_method_id),
        supporting_evidence_ids=supporting_evidence_ids,
        justification="The action is bound to the submitted analysis and participant-visible evidence.",
    )
    next_evidence: tuple[TrialDevEvidenceReferenceV1, ...] = tuple()
    if str(decision_action) in _CONTINUE_ACTIONS:
        next_phase, _ = _next_phase(phase_id=str(request.phase_id))
        next_checkpoint = _PHASE_TO_CHECKPOINT[next_phase]
        next_evidence = (
            TrialDevEvidenceReferenceV1(
                evidence_id=f"public:{next_phase}:design_protocol",
                evidence_kind="protocol",
                checkpoint_id=cast(TrialDevCheckpointIdV1, next_checkpoint),
                asset_id=previous.active_asset_id,
                evidence_protocol_id=f"{next_phase}_design_v1",
                evidence_protocol_checksum=previous.policy_binding.design_menu_checksum,
                source_family_id=provenance.source_family_id,
                world_id=provenance.world_id,
                relative_path="public/phase_module_catalog.json",
                artifact_sha256=previous.policy_binding.design_menu_checksum,
            ),
        )
    return cast(
        TrialDevProgrammeStateV1,
        transition_programme_state_v1(
            state=previous,
            action_policy=build_checkpoint_action_policy_v1(
                state=previous,
            ),
            selection=selection,
            outcome=TrialDevCheckpointOutcomeV1(
                reach_status="reached",
                submission_status="accepted",
                analysis_status="estimable",
                execution_status="completed",
            ),
            checkpoint_evidence=checkpoint_evidence,
            next_evidence=next_evidence,
        ),
    )


def _copy_fixed_phase_evidence_v1(
    *,
    scenario_root: Path,
    request: TrialDevelopmentRequestV1,
    out_dir: Path,
) -> tuple[str, int, int]:
    """Copy the unique released evidence trajectory for one asset and phase."""

    trajectory_root = scenario_root.parent / "fixed_trajectories"
    cases_path = trajectory_root / "cases.jsonl"
    if not cases_path.is_file():
        raise FileNotFoundError(
            "TrialDev participant release has neither generating state nor a fixed trajectory index."
        )
    cases = tuple(
        TrialDevPhaseReplayCaseV1.model_validate_json(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    matches = tuple(
        case
        for case in cases
        if case.request.scenario_id == request.scenario_id
        and case.request.phase_id == request.phase_id
        and case.request.candidate_drug_ids == request.candidate_drug_ids
    )
    if len(matches) != 1:
        raise TrialMaterializationRejectedError(
            "No unique fixed phase trajectory is available for the nominated asset and phase."
        )
    case = matches[0]
    request_root = trajectory_root / "materialized" / f"world_{case.world_seed}" / f"request_{case.request.checksum()}"
    sources = tuple(sorted(path for path in request_root.glob("trial_seed_*") if path.is_dir()))
    if len(sources) != 1:
        raise ValueError("Fixed TrialDev trajectory must contain exactly one evidence replicate per request.")
    trial_seed_text = sources[0].name.removeprefix("trial_seed_")
    if not trial_seed_text.isdigit():
        raise ValueError("Fixed TrialDev evidence directory must encode a non-negative integer trial seed.")
    shutil.copytree(sources[0], out_dir)
    return case.request.checksum(), int(case.world_seed), int(trial_seed_text)


def materialize_phase_v1(
    *,
    scenario_root: Path,
    state_path: Path,
    request_path: Path,
    out_dir: Path,
    seed: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize one sequential trial phase and emit participant-safe outputs plus next state."""
    scenario = Path(scenario_root)
    request = validate_design_request_file_v1(request_path=Path(request_path))
    state = validate_program_state_file_v1(state_path=Path(state_path))
    _validate_phase_request(scenario_root=scenario, state=state, request=request)
    out = Path(out_dir)
    fixed_trajectory_index = scenario.parent / "fixed_trajectories" / "cases.jsonl"
    if not fixed_trajectory_index.is_file():
        raise FileNotFoundError(
            "Released TrialDev programmes require fixed_trajectories/cases.jsonl; "
            "the public harness does not regenerate construction-time trial worlds."
        )
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"Trial output already exists: {out}")
        shutil.rmtree(out)
    fixed_evidence_checksum, evidence_world_seed, evidence_trial_seed = _copy_fixed_phase_evidence_v1(
        scenario_root=scenario,
        request=request,
        out_dir=out,
    )
    feasibility_status = "accepted"
    rejection_reason = None
    realized_sample_size = _fixed_evidence_participant_count(out)
    summary_path = out / "execution_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Materialization did not produce execution_summary.json.")
    summary_payload = _read_json_object(summary_path)
    write_json(out / "execution_summary.json", summary_payload)
    evidence_request = validate_design_request_file_v1(request_path=out / "request.json")
    if evidence_request.checksum() != fixed_evidence_checksum:
        raise ValueError("Fixed trajectory request.json does not match its replay-index request checksum.")
    write_json(out / "agent_request.json", request.model_dump(mode="json", exclude_none=True))
    metadata = {
        "version": "v1",
        "scenario_id": str(request.scenario_id),
        "phase_id": str(request.phase_id),
        "request_checksum": request.checksum(),
        "evidence_request_checksum": evidence_request.checksum(),
        "state_checksum": str(state.checksum),
        "execution_seed": int(seed),
        "feasibility_status": feasibility_status,
        "evidence_mode": "fixed_trajectory",
        "fixed_evidence_request_checksum": fixed_evidence_checksum,
        "evidence_world_seed": evidence_world_seed,
        "evidence_trial_seed": evidence_trial_seed,
    }
    write_json(out / "trial_metadata.json", metadata)
    if feasibility_status != "accepted":
        reason = str(rejection_reason or "rejected")
        raise TrialMaterializationRejectedError(reason)
    phase_summary = phase_summary_v1(scenario_root=scenario, trial_output_root=out)
    write_json(out / "phase_summary_public.json", phase_summary)
    table_files = ("participants.parquet", "endpoints.parquet", "safety.parquet")
    missing_tables = sorted(rel for rel in table_files if not (out / rel).is_file())
    if missing_tables:
        raise FileNotFoundError(f"Accepted materialization missing tables: {missing_tables!r}.")
    manifest = TrialDevelopmentTrialOutputManifestV1(
        scenario_id=str(request.scenario_id),
        phase_id=str(request.phase_id),
        request_checksum=request.checksum(),
        evidence_request_checksum=evidence_request.checksum(),
        state_checksum=str(state.checksum),
        table_files=table_files,
        metadata_files=(
            "trial_metadata.json",
            "execution_summary.json",
            "phase_summary_public.json",
            "arm_mapping.json",
            "request.json",
            "agent_request.json",
        ),
        table_checksums=_file_checksums(root=out, rel_paths=table_files),
        n_participants=realized_sample_size,
    )
    write_json(out / "trial_output_manifest.json", manifest.model_dump(mode="json", exclude_none=True))
    validate_trial_output_bundle_v1(trial_output_root=out)
    write_json(
        out / "program_state_pending_decision.json",
        state.model_dump(mode="json", exclude_none=True),
    )
    public_summary = {
        "programme_id": state.programme_id,
        "scenario_id": str(state.scenario_id),
        "stream_id": state.stream_id,
        "current_checkpoint_id": state.current_checkpoint_id,
        "active_asset_id": state.active_asset_id,
        "retired_asset_ids": list(state.retired_asset_ids),
        "terminal_disposition": state.terminal_disposition,
    }
    write_json(out / "program_state_public_summary.json", public_summary)
    return {
        "version": "v1",
        "scenario_id": str(request.scenario_id),
        "phase_id": str(request.phase_id),
        "trial_output_root": str(out),
        "trial_output_checksum": str(manifest.checksum),
        "pending_decision_state_path": str(out / "program_state_pending_decision.json"),
        "pending_decision_state_checksum": str(state.checksum),
    }


def _validate_decision_evidence_links(
    *,
    analysis: TrialDevelopmentPhaseAnalysisSubmissionV1,
    decision: TrialDevelopmentPhaseDecisionSubmissionV1,
) -> None:
    supporting_ids = set(decision.supporting_evidence_ids)
    unknown_evidence_ids = sorted(supporting_ids - set(analysis.evidence_ids()))
    if unknown_evidence_ids:
        raise ValueError(
            f"Phase decision references evidence absent from the bound analysis: {unknown_evidence_ids!r}."
        )
    primary_effect = analysis.primary_effect
    safety_estimate = analysis.safety_estimate
    effect_linked = primary_effect is not None and primary_effect.evidence_id in supporting_ids
    safety_linked = safety_estimate is not None and safety_estimate.evidence_id in supporting_ids
    selected_candidate = None if decision.candidate_drug_id is None else str(decision.candidate_drug_id)
    if selected_candidate is not None:
        if (
            analysis.selected_winner_drug_id is not None
            and str(analysis.selected_winner_drug_id) != selected_candidate
        ):
            raise ValueError("Phase decision candidate does not match the analysis-selected candidate.")
        if (
            effect_linked
            and primary_effect is not None
            and str(primary_effect.candidate_drug_id) != selected_candidate
        ):
            raise ValueError("Linked primary-effect evidence does not belong to the decision candidate.")
        if (
            safety_linked
            and safety_estimate is not None
            and str(safety_estimate.candidate_drug_id) != selected_candidate
        ):
            raise ValueError("Linked safety evidence does not belong to the decision candidate.")
    action = str(decision.decision_action)
    if action == "advance_to_proof_of_concept" and not safety_linked:
        raise ValueError("advance_to_proof_of_concept requires linked safety evidence.")
    if action in {"advance_to_confirmation", "declare_success"}:
        if not effect_linked or not safety_linked:
            raise ValueError(f"{action} requires linked effect and safety evidence.")
    if action == "stop_development" and str(analysis.phase_id) == "phase1" and not safety_linked:
        raise ValueError("phase1 stop_development requires linked safety evidence.")
    if action in {"stop_development", "declare_failure", "declare_inconclusive"} and not (
        effect_linked or safety_linked
    ):
        raise ValueError(f"{action} requires linked effect or safety evidence.")


def _phase_analysis_method_id(analysis: TrialDevelopmentPhaseAnalysisSubmissionV1) -> str:
    method_ids = {
        value
        for value in (
            None if analysis.primary_effect is None else analysis.primary_effect.method_route_id,
            None if analysis.safety_estimate is None else analysis.safety_estimate.method_route_id,
        )
        if value is not None
    }
    if len(method_ids) > 1:
        raise ValueError("One phase decision cannot combine analysis methods.")
    return next(iter(method_ids), "declared_phase_analysis")


def advance_program_state_v1(
    *,
    scenario_root: Path,
    state_path: Path,
    request_path: Path,
    trial_output_root: Path,
    analysis_path: Path,
    decision_path: Path,
    out_path: Path,
) -> TrialDevProgrammeStateV1:
    """Advance program state after a bound phase analysis and decision."""
    scenario = Path(scenario_root)
    state = validate_program_state_file_v1(state_path=Path(state_path))
    request = validate_design_request_file_v1(request_path=Path(request_path))
    _validate_phase_request(scenario_root=scenario, state=state, request=request)
    trial_manifest = validate_trial_output_bundle_v1(trial_output_root=Path(trial_output_root))
    analysis = TrialDevelopmentPhaseAnalysisSubmissionV1.model_validate(_read_json_object(Path(analysis_path)))
    decision = validate_phase_decision_against_policy_v1(scenario_root=scenario, submission_path=Path(decision_path))
    request_checksum = request.checksum()
    if str(trial_manifest.request_checksum) != str(request_checksum):
        raise ValueError("Trial output manifest request_checksum does not match request.")
    if str(analysis.request_checksum) != str(request_checksum):
        raise ValueError("Phase analysis request_checksum does not match request.")
    if str(analysis.trial_output_checksum) != str(trial_manifest.checksum):
        raise ValueError("Phase analysis trial_output_checksum does not match trial output manifest.")
    if str(decision.request_checksum) != str(request_checksum):
        raise ValueError("Phase decision request_checksum does not match request.")
    if str(decision.analysis_checksum) != sha256_file_hex(Path(analysis_path)):
        raise ValueError("Phase decision analysis_checksum does not match analysis submission file.")
    _validate_decision_evidence_links(analysis=analysis, decision=decision)
    action = str(decision.decision_action)
    if str(analysis.phase_id) != str(request.phase_id) or str(decision.phase_id) != str(request.phase_id):
        raise ValueError("Analysis and decision phase_id values must match request phase_id.")
    if decision.candidate_drug_id is not None and decision.candidate_drug_id != state.active_asset_id:
        raise ValueError("Phase decision candidate must be the committed active asset.")
    next_state = _next_state(
        previous=state,
        request=request,
        analysis_artifact_sha256=sha256_file_hex(Path(analysis_path)),
        supporting_evidence_ids=decision.supporting_evidence_ids,
        analysis_method_id=_phase_analysis_method_id(analysis),
        decision_action=action,
    )
    write_json(Path(out_path), next_state.model_dump(mode="json", exclude_none=True))
    return next_state


def _full_submission_from_stepwise(
    *,
    request: dict[str, Any],
    analysis: TrialDevelopmentPhaseAnalysisSubmissionV1,
    decision: TrialDevelopmentPhaseDecisionSubmissionV1,
    phase_scoring_objective_id: str,
) -> dict[str, Any]:
    validated_request = TrialDevelopmentRequestV1.model_validate(request)
    if str(analysis.scenario_id) != str(decision.scenario_id):
        raise ValueError("Phase analysis and decision scenario_id values must match.")
    if str(analysis.phase_id) != str(decision.phase_id):
        raise ValueError("Phase analysis and decision phase_id values must match.")
    if str(validated_request.phase_id) != str(analysis.phase_id):
        raise ValueError("Stepwise request phase_id must match phase analysis phase_id.")
    return {
        "version": "v1",
        "scenario_id": str(analysis.scenario_id),
        "request": validated_request.model_dump(mode="json", exclude_none=True),
        "analysis_report": {
            "selected_winner_drug_id": analysis.selected_winner_drug_id,
            "ranked_drug_ids": list(analysis.ranked_drug_ids),
            "candidate_utility_estimates": [
                estimate.model_dump(mode="json", exclude_none=True)
                for estimate in analysis.candidate_utility_estimates
            ],
            "primary_effect": (
                None if analysis.primary_effect is None else analysis.primary_effect.model_dump(mode="json")
            ),
            "safety_estimate": (
                None if analysis.safety_estimate is None else analysis.safety_estimate.model_dump(mode="json")
            ),
            "claimed_subgroup_variables": list(analysis.claimed_subgroup_variables),
            "diagnostic_artifacts": [
                record.model_dump(mode="json", exclude_none=True) for record in analysis.diagnostic_artifacts
            ],
            "evidence_summary": analysis.evidence_summary,
        },
        "program_decision": {
            "objective_id": str(phase_scoring_objective_id),
            "decision_action": str(decision.decision_action),
            "recommended_drug_id": decision.candidate_drug_id,
            "supporting_evidence_ids": list(decision.supporting_evidence_ids),
        },
    }


def _grade_stepwise_dir(
    *,
    scenario_root: Path,
    phase_dir: Path,
    report_mode: str,
    phase_scoring_objective_id: str,
    program_objective_id: str,
) -> dict[str, Any]:
    report = _grade_stepwise_report(
        scenario_root=Path(scenario_root),
        phase_dir=Path(phase_dir),
        phase_scoring_objective_id=phase_scoring_objective_id,
        program_objective_id=program_objective_id,
    )
    return cast(dict[str, Any], grade_report_payload_v1(report=report, report_mode=report_mode))


def _grade_stepwise_report(
    *,
    scenario_root: Path,
    phase_dir: Path,
    phase_scoring_objective_id: str,
    program_objective_id: str,
    eligible_candidate_drug_ids: tuple[str, ...] | None = None,
    terminal_status: str | None = None,
) -> TrialDevelopmentGradeReportV1:
    trial_output_root = _resolve_trial_output_root(phase_dir=Path(phase_dir))
    if trial_output_root is None:
        raise FileNotFoundError("Submitted phase is missing a trial_output_manifest or trial_output_ref.json.")
    request = _read_json_object(Path(phase_dir) / "request.json")
    analysis = TrialDevelopmentPhaseAnalysisSubmissionV1.model_validate(
        _read_json_object(Path(phase_dir) / "analysis_submission.json")
    )
    decision = TrialDevelopmentPhaseDecisionSubmissionV1.model_validate(
        _read_json_object(Path(phase_dir) / "decision_submission.json")
    )
    payload = _full_submission_from_stepwise(
        request=request,
        analysis=analysis,
        decision=decision,
        phase_scoring_objective_id=phase_scoring_objective_id,
    )
    with tempfile.TemporaryDirectory(prefix="trial_benchmark_stepwise_grade_") as tmp_dir:
        submission_path = Path(tmp_dir) / "submission.json"
        write_json(submission_path, payload)
        report = grade_item_v1(
            scenario_root=Path(scenario_root),
            submission_path=submission_path,
            report_mode="audit",
            assigned_objective_id=str(phase_scoring_objective_id),
            program_objective_id=str(program_objective_id),
            eligible_candidate_drug_ids=eligible_candidate_drug_ids,
            terminal_status=terminal_status,
            trial_output_root=trial_output_root,
        )
    return TrialDevelopmentGradeReportV1.model_validate(report)


def _invalid_attempt_report(
    *,
    scenario_id: str,
    attempted_phase_id: str | None,
    reason_code: str,
    message: str,
) -> TrialDevelopmentInvalidAttemptReportV1:
    return TrialDevelopmentInvalidAttemptReportV1(
        scenario_id=str(scenario_id),
        attempted_phase_id=None if attempted_phase_id is None else str(attempted_phase_id),
        reason_code=cast(Any, reason_code),
        message=str(message),
        validity=TrialDevelopmentValidityReportV1(
            valid=False,
            invalid_reasons=(f"{reason_code}:{attempted_phase_id}" if attempted_phase_id else str(reason_code),),
            warnings=tuple(),
        ),
    )


def _invalid_reason_code(exc: Exception) -> str:
    message = str(exc)
    if "trial_output" in message or "materialized" in message:
        return "missing_materialized_output"
    if "not available" in message:
        return "phase_not_available"
    if "current program state phase" in message:
        return "phase_not_current"
    if "candidate" in message and ("subset" in message or "eligible" in message):
        return "ineligible_candidate"
    if "decision_action" in message or "Unsupported decision_action" in message:
        return "invalid_action"
    if "checksum" in message:
        return "checksum_mismatch"
    if "analysis" in message or "analysis_submission" in message:
        return "invalid_analysis"
    if "decision_submission" in message:
        return "invalid_action"
    return "invalid_request"


def _zero_phase_report(
    *,
    scenario_root: Path,
    scenario_id: str,
    phase_id: str,
    reason: str,
    program_objective_id: str,
    phase_scoring_objective_id: str,
) -> dict[str, Any]:
    lane_scores = _zero_lane_scores(
        scenario_root=Path(scenario_root),
        scenario_id=scenario_id,
        phase_id=phase_id,
        reason=reason,
        program_objective_id=program_objective_id,
        phase_scoring_objective_id=phase_scoring_objective_id,
    )
    return {
        "schema_id": "trialdev_grade_report_v1",
        "version": "v1",
        "scenario_id": str(scenario_id),
        "phase_id": str(phase_id),
        "objective_id": str(phase_scoring_objective_id),
        "program_objective_id": str(program_objective_id),
        "phase_scoring_objective_id": str(phase_scoring_objective_id),
        "primary_score": 0.0,
        "design_score": 0.0,
        "evaluation_score": 0.0,
        "program_score": 0.0,
        "ranking_score": 1.0,
        "analysis_quality": {
            "schema_id": "trialdev_analysis_quality_v1",
            "observational_analysis_eligible": False,
            "observational_analysis_valid": None,
            "observational_analysis_score": None,
            "randomized_primary_effect_eligible": phase_id in {"phase2", "phase3"},
            "randomized_primary_effect_valid": (False if phase_id in {"phase2", "phase3"} else None),
            "randomized_primary_effect_point_agreement": (0.0 if phase_id in {"phase2", "phase3"} else None),
            "randomized_primary_effect_interval_agreement": (0.0 if phase_id in {"phase2", "phase3"} else None),
            "safety_evidence_eligible": phase_id in {"phase1", "phase2", "phase3"},
            "safety_evidence_valid": (False if phase_id in {"phase1", "phase2", "phase3"} else None),
            "safety_evidence_agreement": (0.0 if phase_id in {"phase1", "phase2", "phase3"} else None),
            "phase_evaluation_valid": False,
        },
        "lane_status": {
            "trial_design": "invalid",
            "trial_evaluation": "invalid",
            "program_decision": "invalid",
            "drug_ranking": "not_applicable",
        },
        "active_lane_scores": {},
        "validity": {
            "valid": False,
            "invalid_reasons": [str(reason)],
            "warnings": [],
        },
        "audit_gates": {
            "gates_triggered": [str(reason).split(":", 1)[0]],
            "diagnostic_alignment_score": 0.0,
        },
        "feasibility_failures": [str(reason)],
        "lane_breakdown": {
            "trial_design": 0.0,
            "trial_evaluation": 0.0,
            "program_decision": 0.0,
            "drug_ranking": 1.0,
        },
        "payload": {
            "decision_action_score": 0.0,
            "required_evidence_flags": [str(reason)],
            "lane_scores": lane_scores,
        },
        "lane_scores": lane_scores,
    }


def _zero_lane_scores(
    *,
    scenario_root: Path,
    scenario_id: str,
    phase_id: str,
    reason: str,
    program_objective_id: str,
    phase_scoring_objective_id: str,
) -> list[dict[str, Any]]:
    register_index = load_evaluation_target_index(Path(scenario_root))
    rows: list[dict[str, Any]] = []
    for lane_id in required_trialdev_lanes_v1(phase_id):
        evaluation_target = register_index.require(
            phase_id=phase_id,
            program_objective_id=program_objective_id,
            phase_scoring_objective_id=phase_scoring_objective_id,
            lane_id=lane_id,
        )
        row = score_evaluation_target(
            scenario_id=scenario_id,
            phase_id=phase_id,
            program_objective_id=program_objective_id,
            phase_scoring_objective_id=phase_scoring_objective_id,
            lane_id=lane_id,
            submitted_target_id=None,
            evaluation_target=evaluation_target,
            artifact_status="missing",
            failure_reason=reason,
            score_override=0.0 if evaluation_target.target_resolution == "realized_public_evidence" else None,
            score_derivation=(
                "public_evidence_action" if evaluation_target.target_resolution == "realized_public_evidence" else None
            ),
        )
        rows.append(row.model_dump(mode="json", exclude_none=True))
    return rows


def final_decision_lane_scores_from_trajectory(
    *,
    scenario_root: Path,
    scenario_id: str,
    program_objective_id: str,
    terminal_action: str | None,
    terminal_recommendation_score: float | None,
    trajectory_decision_score: float | None,
    artifact_status: str,
    failure_reason: str | None,
) -> tuple[TrialDevelopmentLaneScoreRecordV1, ...]:
    """Resolve terminal final-decision lane scores from trajectory state."""

    if artifact_status not in {"present", "missing", "invalid"}:
        raise ValueError(f"Unsupported artifact_status for final decision lanes: {artifact_status!r}")
    if terminal_action == "":
        raise ValueError("terminal_action cannot be an empty string.")
    register_index = load_evaluation_target_index(Path(scenario_root))
    score_by_lane = {
        "route_timing": trajectory_decision_score,
        "final_recommendation": terminal_recommendation_score,
    }
    if artifact_status == "present":
        missing_scores = [lane for lane, score in score_by_lane.items() if score is None]
        if missing_scores:
            raise ValueError(f"Present final-decision lanes require scores for: {missing_scores!r}")
    rows: list[TrialDevelopmentLaneScoreRecordV1] = []
    for lane_id in ("route_timing", "final_recommendation"):
        evaluation_target = register_index.require(
            phase_id="final_decision",
            program_objective_id=program_objective_id,
            phase_scoring_objective_id=program_objective_id,
            lane_id=lane_id,
        )
        accepted_targets = {
            *evaluation_target.reference_target_ids,
            *evaluation_target.credit_eligible_target_ids,
        }
        submitted_target_id = terminal_action
        score_override = None
        runtime_resolved = evaluation_target.target_resolution == "realized_trajectory"
        if (
            artifact_status == "present"
            and submitted_target_id is not None
            and (runtime_resolved or submitted_target_id in accepted_targets)
        ):
            score_override = score_by_lane[lane_id]
        rows.append(
            score_evaluation_target(
                scenario_id=scenario_id,
                phase_id="final_decision",
                program_objective_id=program_objective_id,
                phase_scoring_objective_id=program_objective_id,
                lane_id=lane_id,
                submitted_target_id=submitted_target_id,
                evaluation_target=evaluation_target,
                artifact_status=cast(Any, artifact_status),
                failure_reason=failure_reason,
                score_override=score_override,
                score_derivation="numeric_diagnostic" if score_override is not None else None,
                derived_from_trajectory_metric=score_override is not None,
                terminal_action_observed=None if terminal_action is None else str(terminal_action),
                terminal_phase_observed="final_decision",
            )
        )
    return tuple(rows)


def _resolve_trial_output_root(*, phase_dir: Path) -> Path | None:
    direct = Path(phase_dir)
    if (direct / "trial_output_manifest.json").is_file():
        return direct
    nested = direct / "trial_output"
    if (nested / "trial_output_manifest.json").is_file():
        return nested
    for rel in ("trial_output_ref.json", "trial_output_root.json"):
        path = direct / rel
        if not path.is_file():
            continue
        payload = _read_json_object(path)
        raw = payload.get("trial_output_root")
        if raw is None:
            raw = payload.get("path")
        if raw is None:
            continue
        root = Path(str(raw))
        if not root.is_absolute():
            root = path.parent / root
        return root
    return None


def _submitted_phase_dirs(
    *, trajectory_root: Path, scenario_id: str
) -> tuple[dict[str, Path], list[TrialDevelopmentInvalidAttemptReportV1]]:
    by_phase: dict[str, Path] = {}
    invalid: list[TrialDevelopmentInvalidAttemptReportV1] = []
    stepwise_dirs = sorted(
        {
            path.parent
            for path in Path(trajectory_root).rglob("request.json")
            if _resolve_trial_output_root(phase_dir=path.parent)
            and (
                (path.parent / "analysis_submission.json").is_file()
                or (path.parent / "decision_submission.json").is_file()
                or not (path.parent / "trial_output_manifest.json").is_file()
            )
        },
        key=lambda path: path.as_posix(),
    )
    for phase_dir in stepwise_dirs:
        phase_id: str | None = None
        try:
            request = validate_design_request_file_v1(request_path=phase_dir / "request.json")
            phase_id = str(request.phase_id)
        except (TypeError, ValueError) as exc:
            invalid.append(
                _invalid_attempt_report(
                    scenario_id=scenario_id,
                    attempted_phase_id=None,
                    reason_code="invalid_request",
                    message=f"{phase_dir}: {exc}",
                )
            )
            continue
        if phase_id in by_phase:
            invalid.append(
                _invalid_attempt_report(
                    scenario_id=scenario_id,
                    attempted_phase_id=phase_id,
                    reason_code="duplicate_phase_submission",
                    message=f"Duplicate submission for phase_id={phase_id!r}.",
                )
            )
            continue
        by_phase[phase_id] = phase_dir
    for submission_path in sorted(Path(trajectory_root).rglob("submission.json"), key=lambda path: path.as_posix()):
        invalid.append(
            _invalid_attempt_report(
                scenario_id=scenario_id,
                attempted_phase_id=None,
                reason_code="unsupported_submission_format",
                message=(
                    "Canonical trajectory grading requires stepwise request/analysis/decision files: "
                    f"{submission_path}"
                ),
            )
        )
    return by_phase, invalid


def _decision_action_score(*, report: TrialDevelopmentGradeReportV1) -> float:
    raw = report.payload.get("decision_action_score")
    if isinstance(raw, bool):
        return float(int(raw))
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, float(raw)))
    raise ValueError(f"Phase report is missing decision_action_score: phase_id={report.phase_id!r}.")


def grade_trajectory_v1(
    *,
    scenario_root: Path,
    trajectory_root: Path,
    initial_state_path: Path,
    out_path: Path | None = None,
    report_mode: str = "score",
    scoring_context_path: Path | None = None,
) -> dict[str, Any]:
    """Grade a sequential trajectory by replaying evaluator-held program state."""
    root = Path(trajectory_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Trajectory root not found: {root}.")
    scenario = Path(scenario_root)
    scenario_id = _scenario_id(scenario_root=scenario)
    scoring_context = _read_scoring_context(
        scenario_root=scenario,
        trajectory_root=root,
        scoring_context_path=scoring_context_path,
    )
    state = validate_program_state_file_v1(state_path=Path(initial_state_path))
    if state.current_checkpoint_id != "early_safety_study":
        raise ValueError("Trajectory grading must start from the accepted observational-review state.")
    reports: list[dict[str, Any]] = []
    resource_consequences: list[TrialDevPhaseResourceConsequenceV1] = []
    decision_scores_by_phase: dict[str, float] = {}
    invalid_attempts: list[TrialDevelopmentInvalidAttemptReportV1] = []
    invalid_attempts_without_phase_report = 0
    terminal_action: str | None = None
    unsupported_advance_active = False
    phase_dirs, discovery_invalid = _submitted_phase_dirs(trajectory_root=root, scenario_id=scenario_id)
    invalid_attempts.extend(discovery_invalid)
    invalid_attempts_without_phase_report += len(discovery_invalid)

    while state.terminal_disposition == "active":
        phase_id = _CHECKPOINT_TO_PHASE.get(str(state.current_checkpoint_id))
        if phase_id is None:
            raise ValueError(f"Unsupported single-asset checkpoint: {state.current_checkpoint_id!r}.")
        phase_scoring_objective_id = _phase_scoring_objective(context=scoring_context, phase_id=phase_id)
        phase_dir = phase_dirs.pop(phase_id, None)
        if phase_dir is None:
            mode = str(_phase_policy_modes(scenario_root=scenario).get(phase_id, "optional"))
            if mode == "required":
                reason = f"missing_required_phase:{phase_id}"
                invalid_attempts.append(
                    _invalid_attempt_report(
                        scenario_id=scenario_id,
                        attempted_phase_id=phase_id,
                        reason_code="missing_required_phase",
                        message=f"Required current phase was not submitted: {phase_id}.",
                    )
                )
                reports.append(
                    _zero_phase_report(
                        scenario_root=scenario,
                        scenario_id=scenario_id,
                        phase_id=phase_id,
                        reason=reason,
                        program_objective_id=str(scoring_context.program_objective_id),
                        phase_scoring_objective_id=phase_scoring_objective_id,
                    )
                )
                decision_scores_by_phase[phase_id] = 0.0
            break
        try:
            request = validate_design_request_file_v1(request_path=phase_dir / "request.json")
            _validate_phase_request(scenario_root=scenario, state=state, request=request)
            trial_output_root = _resolve_trial_output_root(phase_dir=phase_dir)
            if trial_output_root is None:
                raise FileNotFoundError("Submitted phase is missing a trial_output_manifest or trial_output_ref.json.")
            trial_manifest = validate_trial_output_bundle_v1(trial_output_root=trial_output_root)
            request_checksum = request.checksum()
            if str(trial_manifest.request_checksum) != str(request_checksum):
                raise ValueError("Trial output manifest request_checksum does not match request.")
            design_efficiency = None
            if phase_id in _MATERIALIZABLE_PHASES:
                design_witness = derive_phase_design_witness_v1(
                    scenario_root=scenario,
                    request=request,
                    trial_output_root=trial_output_root,
                    phase_id=phase_id,
                )
                design_efficiency = derive_phase_design_efficiency_v1(
                    scenario_root=scenario,
                    request=request,
                    design_witness=design_witness,
                )
                resource_consequences.append(
                    derive_phase_resource_consequence_v1(
                        request=request,
                        design_efficiency=design_efficiency,
                        entered_after_unsupported_advance=unsupported_advance_active,
                    )
                )
            analysis = TrialDevelopmentPhaseAnalysisSubmissionV1.model_validate(
                _read_json_object(phase_dir / "analysis_submission.json")
            )
            decision = validate_phase_decision_against_policy_v1(
                scenario_root=scenario, submission_path=phase_dir / "decision_submission.json"
            )
            if str(analysis.request_checksum) != str(request_checksum):
                raise ValueError("Phase analysis request_checksum does not match request.")
            if str(analysis.trial_output_checksum) != str(trial_manifest.checksum):
                raise ValueError("Phase analysis trial_output_checksum does not match trial output manifest.")
            if str(decision.request_checksum) != str(request_checksum):
                raise ValueError("Phase decision request_checksum does not match request.")
            if str(decision.analysis_checksum) != sha256_file_hex(phase_dir / "analysis_submission.json"):
                raise ValueError("Phase decision analysis_checksum does not match analysis submission file.")
            _validate_decision_evidence_links(analysis=analysis, decision=decision)
            if str(analysis.phase_id) != str(request.phase_id) or str(decision.phase_id) != str(request.phase_id):
                raise ValueError("Analysis and decision phase_id values must match request phase_id.")
            if decision.candidate_drug_id is not None and decision.candidate_drug_id != state.active_asset_id:
                raise ValueError("Phase decision candidate must be the committed active asset.")
            report = _grade_stepwise_report(
                scenario_root=scenario,
                phase_dir=phase_dir,
                phase_scoring_objective_id=phase_scoring_objective_id,
                program_objective_id=str(scoring_context.program_objective_id),
                eligible_candidate_drug_ids=(str(state.active_asset_id),),
                terminal_status=state.terminal_disposition,
            )
            reports.append(grade_report_payload_v1(report=report, report_mode=report_mode))
            decision_score = _decision_action_score(report=report)
            decision_scores_by_phase[phase_id] = decision_score
            if phase_id in _MATERIALIZABLE_PHASES and (
                design_efficiency is None or report.design_efficiency != design_efficiency
            ):
                raise ValueError(f"Randomized phase design-efficiency replay drift: phase_id={phase_id!r}.")
            terminal_action = str(decision.decision_action)
            if terminal_action in _CONTINUE_ACTIONS and decision_score != 1.0:
                unsupported_advance_active = True
            state = _next_state(
                previous=state,
                request=request,
                analysis_artifact_sha256=sha256_file_hex(phase_dir / "analysis_submission.json"),
                supporting_evidence_ids=decision.supporting_evidence_ids,
                analysis_method_id=_phase_analysis_method_id(analysis),
                decision_action=str(decision.decision_action),
            )
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            reason_code = _invalid_reason_code(exc)
            invalid_attempts.append(
                _invalid_attempt_report(
                    scenario_id=scenario_id,
                    attempted_phase_id=phase_id,
                    reason_code=reason_code,
                    message=str(exc),
                )
            )
            reports.append(
                _zero_phase_report(
                    scenario_root=scenario,
                    scenario_id=scenario_id,
                    phase_id=phase_id,
                    reason=f"{reason_code}:{phase_id}",
                    program_objective_id=str(scoring_context.program_objective_id),
                    phase_scoring_objective_id=phase_scoring_objective_id,
                )
            )
            decision_scores_by_phase[phase_id] = 0.0
            break

    for phase_id, phase_dir in sorted(
        phase_dirs.items(),
        key=lambda item: _phase_index(item[0]) if item[0] in _PHASE_ORDER else 99,
    ):
        mode = str(_phase_policy_modes(scenario_root=scenario).get(str(phase_id), "optional"))
        reason_code = (
            "post_terminal_submission"
            if state.terminal_disposition != "active"
            else "phase_not_available" if mode == "not_available" else "phase_not_current"
        )
        invalid_attempts.append(
            _invalid_attempt_report(
                scenario_id=scenario_id,
                attempted_phase_id=str(phase_id),
                reason_code=reason_code,
                message=f"Unreachable phase submission at {phase_dir}.",
            )
        )
        invalid_attempts_without_phase_report += 1

    if not reports and not invalid_attempts:
        raise FileNotFoundError("Trajectory root contains no gradeable phase submissions.")
    score_keys = (
        "primary_score",
        "design_score",
        "evaluation_score",
        "program_score",
        "ranking_score",
    )
    denominator = max(1, len(reports) + int(invalid_attempts_without_phase_report))
    mean_scores = {
        key: float(sum(float(report.get(key, 0.0)) for report in reports) / float(denominator)) for key in score_keys
    }
    decision_regret_by_phase: dict[str, float] = {}
    for idx, phase_report in enumerate(reports):
        phase_id = str(phase_report.get("phase_id", f"phase_{idx}"))
        decision_score = float(decision_scores_by_phase.get(phase_id, 0.0))
        decision_regret_by_phase[phase_id] = max(0.0, min(1.0, 1.0 - float(decision_score)))
    for idx in range(invalid_attempts_without_phase_report):
        decision_regret_by_phase[f"invalid_attempt_{idx + 1}"] = 1.0
    trajectory_decision_score = float(
        min((1.0 - float(value) for value in decision_regret_by_phase.values()), default=0.0)
    )
    phase_primary_scores = [float(report.get("primary_score", 0.0)) for report in reports]
    phase_primary_scores.extend(0.0 for _ in range(invalid_attempts_without_phase_report))
    trajectory_primary_score = float(min(phase_primary_scores, default=0.0))
    terminal_status: Literal["active", "stopped", "completed", "invalid"]
    if invalid_attempts:
        terminal_status = "invalid"
    elif state.terminal_disposition in {"withheld", "stopped"}:
        terminal_status = "stopped"
    elif state.terminal_disposition in {"success", "failure", "inconclusive"}:
        terminal_status = "completed"
    else:
        terminal_status = "active"
    terminal_summary = TrialDevelopmentTerminalSummaryV1(
        scenario_id=scenario_id,
        terminal_status=terminal_status,
        terminal_action=None if terminal_action is None else str(terminal_action),
        final_program_success=str(terminal_action) == "declare_success",
        recommended_drug_id=reports[-1].get("selected_winner_drug_id") if reports else None,
    )
    terminal_recommendation_score = float(reports[-1].get("program_score", 0.0)) if reports else 0.0
    final_lane_scores = final_decision_lane_scores_from_trajectory(
        scenario_root=scenario,
        scenario_id=scenario_id,
        program_objective_id=str(scoring_context.program_objective_id),
        terminal_action=None if terminal_action is None else str(terminal_action),
        terminal_recommendation_score=terminal_recommendation_score,
        trajectory_decision_score=float(trajectory_decision_score),
        artifact_status="invalid" if invalid_attempts else "present",
        failure_reason=";".join(str(item.reason_code) for item in invalid_attempts) if invalid_attempts else None,
    )
    replay_report = TrialDevelopmentTrajectoryReplayReportV1(
        scenario_id=str(terminal_summary.scenario_id),
        program_objective_id=str(scoring_context.program_objective_id),
        phase_scoring_objectives=dict(scoring_context.phase_scoring_objectives),
        terminal_status=terminal_status,
        n_phase_submissions=int(len(reports)),
        n_invalid_attempts=int(len(invalid_attempts)),
        trajectory_primary_score=float(trajectory_primary_score),
        trajectory_decision_score=float(trajectory_decision_score),
        decision_regret_by_phase=decision_regret_by_phase,
        mean_scores=mean_scores,
        terminal_summary=terminal_summary,
        resource_consequence=derive_programme_resource_consequence_v1(tuple(resource_consequences)),
        phase_reports=tuple(reports),
        final_lane_scores=final_lane_scores,
        invalid_attempts=tuple(invalid_attempts),
    )
    payload = cast(dict[str, Any], replay_report.model_dump(mode="json", exclude_none=True))
    if out_path is not None:
        write_json(Path(out_path), payload)
    return payload
