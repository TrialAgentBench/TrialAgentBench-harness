"""Replay verified public submissions into canonical TrialDev checkpoint custody."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel

from trialagentbench_harness.contracts.experiments import (
    TrialDevCanonicalCheckpointSourcesV1,
    TrialDevCanonicalCheckpointSourceV1,
    TrialDevCheckpointBlockPlanV1,
    TrialDevCheckpointSchedulePlanV1,
)
from trialagentbench_harness.contracts.trialdev.run_checkpoint import TrialDevRunCheckpointV1
from trialagentbench_harness.execution_policy import TRIALDEV_RELEASE_BUDGET_V1
from trialagentbench_harness.io import (
    canonical_payload_sha256,
    read_json,
    read_json_model,
    sha256_dir_digest,
    sha256_path,
    write_json_model,
)
from trialagentbench_harness.ports import LLMResponse, ToolCall
from trialagentbench_harness.trialdev.data import discover_programs
from trialagentbench_harness.trialdev.grading.reference_submissions import (
    build_phase_reference_analysis_v1,
    build_phase_reference_decision_v1,
)
from trialagentbench_harness.trialdev.runner import (
    CheckpointCaptureComplete,
    RunOptions,
    run_program,
)
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentRequestV1
from trialagentbench_harness.trialdev.share.sequential import (
    TrialDevelopmentObservationalReviewSubmissionV1,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


def _json_object(path: Path) -> dict[str, object]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Recorded canonical submission must be a JSON object: {path}")
    return payload


def _validated_payload(model: type[ModelT], path: Path) -> dict[str, object]:
    payload = _json_object(path)
    return model.model_validate(payload).model_dump(mode="json")


def _tool_names(tools: Sequence[Mapping[str, object]] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or ():
        function = tool.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            if isinstance(name, str):
                names.add(name)
    return names


class RecordedTrialDevProvider:
    """Replay a qualified design path and recompute realization-specific evidence."""

    model = "trialagentbench-canonical-reference"
    telemetry_route = "offline-recorded-public-trajectory"

    def __init__(
        self,
        *,
        reference_program: Path,
        scenario_root: Path,
        program_workdir: Path,
    ) -> None:
        self.reference_program = Path(reference_program)
        self.reference_workdir = self.reference_program / "agent_workdir"
        self.scenario_root = Path(scenario_root)
        self.program_workdir = Path(program_workdir)
        self.phase_ids = tuple(
            phase for phase in ("phase1", "phase2", "phase3") if (self.reference_workdir / f"phase_{phase}").is_dir()
        )
        self.phase_index = 0
        self.current_phase: str | None = None
        self.call_index = 0
        self.staged_files: set[str] = set()
        self.observational_payload = _validated_payload(
            TrialDevelopmentObservationalReviewSubmissionV1,
            self.reference_program / "obs_review" / "agent_obs_review_payload.json",
        )
        selected_candidate = self.observational_payload.get("candidate_drug_id")
        if not isinstance(selected_candidate, str) or not selected_candidate:
            raise ValueError("Recorded reference lacks a selected candidate drug.")
        self.selected_candidate = selected_candidate
        self.requests = {
            phase: _validated_payload(
                TrialDevelopmentRequestV1,
                self.reference_workdir / f"phase_{phase}" / "request.json",
            )
            for phase in self.phase_ids
        }

    def _file_response(
        self,
        *,
        name: str,
        payload: dict[str, object],
        filename: str,
    ) -> LLMResponse:
        relative = f"canonical_reference/{filename}"
        if relative not in self.staged_files:
            content = json.dumps(payload, indent=2, sort_keys=True)
            self.staged_files.add(relative)
            code = (
                "from pathlib import Path\n"
                f"_path = Path('scratch/{relative}')\n"
                "_path.parent.mkdir(parents=True, exist_ok=True)\n"
                f"_path.write_text({content!r}, encoding='utf-8')\n"
            )
            return self._response(
                name="execute_code",
                payload={
                    "code": code,
                    "purpose": "Stage a frozen canonical public-reference submission.",
                },
            )
        return self._response(name=name, payload={"path": relative})

    def _response(self, *, name: str, payload: dict[str, object]) -> LLMResponse:
        self.call_index += 1
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id=f"canonical-reference-{self.call_index}",
                    name=name,
                    arguments=json.dumps(payload, sort_keys=True),
                )
            ]
        )

    def generate_turn(
        self,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] | None = None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
    ) -> LLMResponse:
        """Return the recorded submission matching the current public tool contract."""

        del messages, temperature, max_tokens, timeout_seconds, tool_choice
        offered = _tool_names(tools)
        if "submit_obs_review_analysis_and_decision_file" in offered:
            return self._file_response(
                name="submit_obs_review_analysis_and_decision_file",
                payload=self.observational_payload,
                filename="observational_review.json",
            )
        if "submit_phase_request" in offered:
            if self.phase_index >= len(self.phase_ids):
                raise ValueError("Recorded trajectory has no remaining phase request.")
            self.current_phase = self.phase_ids[self.phase_index]
            return self._response(
                name="submit_phase_request",
                payload=self.requests[self.current_phase],
            )
        if "submit_phase_analysis_file" in offered:
            if self.current_phase is None:
                raise ValueError("Recorded phase analysis was requested before a phase design.")
            trial_output_root = self.program_workdir / f"phase_{self.current_phase}" / "trial_output"
            analysis = build_phase_reference_analysis_v1(
                scenario_root=self.scenario_root,
                trial_output_root=trial_output_root,
                phase_id=self.current_phase,
                candidate_drug_id=self.selected_candidate,
            )
            return self._file_response(
                name="submit_phase_analysis_file",
                payload=analysis.model_dump(mode="json", exclude_none=True),
                filename=f"{self.current_phase}_analysis.json",
            )
        if "submit_phase_decision" in offered:
            if self.current_phase is None:
                raise ValueError("Recorded phase decision was requested before a phase design.")
            phase_dir = self.program_workdir / f"phase_{self.current_phase}"
            decision = build_phase_reference_decision_v1(
                scenario_root=self.scenario_root,
                trial_output_root=phase_dir / "trial_output",
                analysis_path=phase_dir / "analysis_submission.json",
                phase_id=self.current_phase,
                candidate_drug_id=self.selected_candidate,
            )
            response = self._response(
                name="submit_phase_decision",
                payload=decision.model_dump(mode="json", exclude_none=True),
            )
            self.phase_index += 1
            self.current_phase = None
            return response
        raise ValueError(f"Recorded canonical provider received an unsupported tool surface: {sorted(offered)!r}")


def _canonical_output_root(*, checkpoint_root: Path, block: TrialDevCheckpointBlockPlanV1) -> Path:
    program_path = Path(block.canonical_program_relative_path)
    if program_path.name != block.program_id or program_path.parent.name != "programs":
        raise ValueError("canonical_program_relative_path must end with programs/{program_id}.")
    return (Path(checkpoint_root) / program_path.parent.parent).resolve()


def _capture_block(
    *,
    participant_root: Path,
    reference_root: Path,
    checkpoint_root: Path,
    block: TrialDevCheckpointBlockPlanV1,
    max_tokens: int,
    max_turns_per_step: int,
    program_watchdog_seconds: int,
) -> Path:
    programs = {program.program_id: program for program in discover_programs(participant_root)}
    program = programs.get(block.program_id)
    if program is None:
        raise ValueError(f"Checkpoint plan references unknown programme {block.program_id!r}.")
    if (program.scenario_id, program.objective_id) != (block.scenario_id, block.objective_id):
        raise ValueError("Checkpoint plan programme, scenario, and objective identities disagree.")
    reference_program = Path(reference_root) / "programs" / block.program_id
    if not reference_program.is_dir():
        raise FileNotFoundError(reference_program)
    output_root = _canonical_output_root(checkpoint_root=checkpoint_root, block=block)
    provider = RecordedTrialDevProvider(
        reference_program=reference_program,
        scenario_root=Path(participant_root) / f"scenario_{program.scenario_id}",
        program_workdir=output_root / "programs" / block.program_id / "agent_workdir",
    )
    run_identity = canonical_payload_sha256(
        {
            "schema_id": "trialagentbench.canonical_checkpoint_source/v1",
            "participant_release_sha256": sha256_dir_digest(participant_root),
            "reference_program_sha256": sha256_path(reference_program),
            "canonical_reference_id": block.canonical_reference_id,
            "program_id": block.program_id,
            "phase_id": block.checkpoint_phase_id,
            "step_id": block.checkpoint_step_id,
        }
    )

    captured: Path | None = None

    def observer(checkpoint: TrialDevRunCheckpointV1) -> None:
        nonlocal captured
        pending = checkpoint.payload.continuation.payload.pending_step
        if (pending.phase_id, pending.step_id) != (
            block.checkpoint_phase_id,
            block.checkpoint_step_id,
        ):
            return
        captured = (
            output_root / "programs" / block.program_id / "checkpoints" / f"{checkpoint.payload.sequence:08d}.json"
        )
        raise CheckpointCaptureComplete

    options = RunOptions(
        bundle_root=participant_root,
        output_root=output_root,
        model=provider.model,
        procedure_assistance="output_contract_only",
        temperature=0.0,
        max_tokens=max_tokens,
        max_turns_per_step=max_turns_per_step,
        program_watchdog_seconds=program_watchdog_seconds,
        run_identity_sha256=run_identity,
        checkpoint_observer=observer,
    )
    try:
        run_program(program, options=options, provider=provider)
    except CheckpointCaptureComplete:
        pass
    if captured is None or not captured.is_file():
        raise RuntimeError("Canonical trajectory ended before the requested checkpoint was captured.")
    return captured


def build_trialdev_canonical_checkpoints_v1(
    *,
    participant_root: Path,
    reference_root: Path,
    checkpoint_root: Path,
    plan: TrialDevCheckpointSchedulePlanV1,
    max_tokens: int = 4096,
    max_turns_per_step: int = 50,
    program_watchdog_seconds: int = 1800,
) -> TrialDevCanonicalCheckpointSourcesV1:
    """Replay one canonical source per unique reference and write a receipt."""

    participant = Path(participant_root).resolve()
    references = Path(reference_root).resolve()
    output = Path(checkpoint_root).resolve()
    if not participant.is_dir() or not references.is_dir():
        raise FileNotFoundError("Participant and recorded-reference roots must be directories.")
    identities: dict[str, tuple[str, str, str, str]] = {}
    for block in plan.blocks:
        identity = (
            block.program_id,
            block.checkpoint_phase_id,
            block.checkpoint_step_id,
            block.canonical_program_relative_path,
        )
        prior = identities.setdefault(block.canonical_reference_id, identity)
        if prior != identity:
            raise ValueError("One canonical reference ID cannot describe multiple source checkpoints.")
    records: list[TrialDevCanonicalCheckpointSourceV1] = []
    for block in plan.blocks:
        if any(row.canonical_reference_id == block.canonical_reference_id for row in records):
            continue
        checkpoint = _capture_block(
            participant_root=participant,
            reference_root=references,
            checkpoint_root=output,
            block=block,
            max_tokens=max_tokens,
            max_turns_per_step=max_turns_per_step,
            program_watchdog_seconds=program_watchdog_seconds,
        )
        records.append(
            TrialDevCanonicalCheckpointSourceV1(
                canonical_reference_id=block.canonical_reference_id,
                program_id=block.program_id,
                phase_id=block.checkpoint_phase_id,
                step_id=block.checkpoint_step_id,
                program_relative_path=block.canonical_program_relative_path,
                checkpoint_relative_path=checkpoint.relative_to(output).as_posix(),
                checkpoint_sha256=sha256_path(checkpoint),
            )
        )
    receipt = TrialDevCanonicalCheckpointSourcesV1(
        participant_release_sha256=sha256_dir_digest(participant),
        recorded_reference_sha256=sha256_dir_digest(references),
        records=tuple(records),
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json_model(output / "canonical_checkpoint_sources.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    """Replay recorded public trajectories into canonical checkpoint sources."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-dir", type=Path, required=True)
    parser.add_argument("--recorded-reference-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-source-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=TRIALDEV_RELEASE_BUDGET_V1.maximum_completion_tokens_per_turn,
    )
    parser.add_argument(
        "--max-turns-per-step",
        type=int,
        default=TRIALDEV_RELEASE_BUDGET_V1.maximum_turns,
    )
    parser.add_argument(
        "--program-watchdog-seconds",
        type=int,
        default=TRIALDEV_RELEASE_BUDGET_V1.wall_time_limit_seconds,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    plan = read_json_model(TrialDevCheckpointSchedulePlanV1, args.plan)
    build_trialdev_canonical_checkpoints_v1(
        participant_root=args.participant_dir,
        reference_root=args.recorded_reference_dir,
        checkpoint_root=args.checkpoint_source_dir,
        plan=plan,
        max_tokens=args.max_tokens,
        max_turns_per_step=args.max_turns_per_step,
        program_watchdog_seconds=args.program_watchdog_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
