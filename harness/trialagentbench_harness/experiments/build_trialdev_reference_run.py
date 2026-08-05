"""Execute a deterministic TrialDev reference trajectory from released evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from trialagentbench_harness.io import canonical_payload_sha256, sha256_dir_digest
from trialagentbench_harness.ports import LLMResponse, ToolCall
from trialagentbench_harness.trialdev.data import discover_programs
from trialagentbench_harness.trialdev.grading.reference_submissions import (
    build_observational_reference_submission_v1,
    build_phase_reference_analysis_v1,
    build_phase_reference_decision_v1,
    build_phase_reference_request_v1,
)
from trialagentbench_harness.trialdev.participant_submission import participant_payload_v1
from trialagentbench_harness.trialdev.runner import RunOptions, run_program

_HARNESS_BOUND_PHASE_FIELDS = frozenset({"scenario_id", "phase_id", "version"})
_REFERENCE_MAX_TURNS_PER_STEP = 2


def _tool_names(tools: Sequence[Mapping[str, object]] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or ():
        function = tool.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            if isinstance(name, str):
                names.add(name)
    return names


class PublicEvidenceReferenceProvider:
    """Compose one reference trajectory from released contracts and trial tables."""

    model = "trialagentbench-public-evidence-reference"
    telemetry_route = "offline-public-evidence-replay"

    def __init__(
        self,
        *,
        scenario_root: Path,
        program_workdir: Path,
        objective_id: str,
    ) -> None:
        self.scenario_root = Path(scenario_root)
        self.program_workdir = Path(program_workdir)
        self.objective_id = objective_id
        self.phase_ids = ("phase1", "phase2", "phase3")
        self.phase_index = 0
        self.current_phase: str | None = None
        self.selected_candidate: str | None = None
        self.staged_files: set[str] = set()
        self.call_index = 0

    def _response(self, *, name: str, payload: dict[str, object]) -> LLMResponse:
        self.call_index += 1
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id=f"public-reference-{self.call_index}",
                    name=name,
                    arguments=json.dumps(payload, sort_keys=True),
                )
            ]
        )

    def _file_response(
        self,
        *,
        tool_name: str,
        payload: dict[str, object],
        filename: str,
    ) -> LLMResponse:
        relative = f"public_reference/{filename}"
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
                    "purpose": "Stage a deterministic public-evidence reference submission.",
                },
            )
        return self._response(name=tool_name, payload={"path": relative})

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
        """Return the current evidence-derived submission under the offered tool contract."""

        del messages, temperature, max_tokens, timeout_seconds, tool_choice
        offered = _tool_names(tools)
        if "submit_obs_review_analysis_and_decision_file" in offered:
            submission = build_observational_reference_submission_v1(
                scenario_root=self.scenario_root,
                objective_id=self.objective_id,
            )
            self.selected_candidate = submission.candidate_drug_id
            return self._file_response(
                tool_name="submit_obs_review_analysis_and_decision_file",
                payload=participant_payload_v1(submission),
                filename="observational_review.json",
            )
        if "submit_phase_request" in offered:
            if self.selected_candidate is None:
                raise ValueError("Reference trajectory stopped at observational review.")
            if self.phase_index >= len(self.phase_ids):
                raise ValueError("Reference trajectory has no remaining phase request.")
            self.current_phase = self.phase_ids[self.phase_index]
            request = build_phase_reference_request_v1(
                scenario_root=self.scenario_root,
                phase_id=self.current_phase,
                candidate_drug_id=self.selected_candidate,
                objective_id=self.objective_id,
            )
            return self._response(
                name="submit_phase_request",
                payload=participant_payload_v1(request, root_fields=_HARNESS_BOUND_PHASE_FIELDS),
            )
        if "submit_phase_analysis_file" in offered:
            if self.current_phase is None or self.selected_candidate is None:
                raise ValueError("Reference phase analysis was requested before a phase design.")
            trial_output_root = self.program_workdir / f"phase_{self.current_phase}" / "trial_output"
            phase_submission = build_phase_reference_analysis_v1(
                scenario_root=self.scenario_root,
                trial_output_root=trial_output_root,
                phase_id=self.current_phase,
                candidate_drug_id=self.selected_candidate,
            )
            return self._file_response(
                tool_name="submit_phase_analysis_file",
                payload=participant_payload_v1(
                    phase_submission,
                    root_fields=_HARNESS_BOUND_PHASE_FIELDS,
                ),
                filename=f"{self.current_phase}_analysis.json",
            )
        if "submit_phase_decision" in offered:
            if self.current_phase is None or self.selected_candidate is None:
                raise ValueError("Reference phase decision was requested before a phase analysis.")
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
                payload=participant_payload_v1(
                    decision,
                    root_fields=_HARNESS_BOUND_PHASE_FIELDS,
                ),
            )
            self.phase_index += 1
            self.current_phase = None
            return response
        raise ValueError(f"Public reference provider received an unsupported tool surface: {sorted(offered)!r}")


def build_trialdev_reference_run(
    *,
    participant_root: Path,
    evaluator_root: Path,
    output_root: Path,
    program_id: str,
    master_seed: int,
) -> Path:
    """Run one participant-interface reference program and return its output path."""

    programs = {program.program_id: program for program in discover_programs(Path(participant_root))}
    program = programs.get(program_id)
    if program is None:
        raise ValueError(f"Unknown TrialDev program_id={program_id!r}.")
    scenario = Path(evaluator_root) / f"scenario_{program.scenario_id}"
    if not scenario.is_dir():
        raise FileNotFoundError(scenario)
    program_dir = Path(output_root) / "programs" / str(program.program_id)
    provider = PublicEvidenceReferenceProvider(
        scenario_root=scenario,
        program_workdir=program_dir / "agent_workdir",
        objective_id=program.objective_id,
    )
    run_identity = canonical_payload_sha256(
        {
            "schema_id": "trialagentbench.public_evidence_reference_run/v1",
            "participant_release_sha256": sha256_dir_digest(Path(participant_root)),
            "evaluator_release_sha256": sha256_dir_digest(Path(evaluator_root)),
            "program_id": program.program_id,
            "master_seed": int(master_seed),
        }
    )
    run_program(
        program,
        options=RunOptions(
            bundle_root=Path(participant_root),
            output_root=Path(output_root),
            model=provider.model,
            master_seed=int(master_seed),
            temperature=0.0,
            max_turns_per_step=_REFERENCE_MAX_TURNS_PER_STEP,
            run_identity_sha256=run_identity,
        ),
        provider=provider,
    )
    return program_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Run one deterministic reference trajectory from a role-separated release."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--participant-root", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--program-id", required=True)
    parser.add_argument("--master-seed", type=int, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    path = build_trialdev_reference_run(
        participant_root=args.participant_root,
        evaluator_root=args.evaluator_root,
        output_root=args.output_root,
        program_id=args.program_id,
        master_seed=args.master_seed,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
