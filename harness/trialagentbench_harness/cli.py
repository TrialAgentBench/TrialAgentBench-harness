"""Public TrialAgentBench command-line interface."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from dataclasses import dataclass


class MissingOptionalDependencyError(RuntimeError):
    """A selected command requires an extra that is not installed."""


@dataclass(frozen=True)
class _Command:
    module: str
    description: str
    function: str = "main"
    optional_dependency: tuple[str, str] | None = None

    def invoke(self, argv: Sequence[str]) -> int:
        try:
            command = getattr(importlib.import_module(self.module), self.function)
        except ModuleNotFoundError as error:
            if self.optional_dependency is None or error.name != self.optional_dependency[0]:
                raise
            module, extra = self.optional_dependency
            raise MissingOptionalDependencyError(
                f"This command requires optional dependency {module!r}; install trial-agent-bench[{extra}]."
            ) from error
        result = command(list(argv))
        return int(result or 0)


_COMMANDS: dict[str, dict[str, _Command]] = {
    "run": {
        "trialeval": _Command("trialagentbench_harness.tools.run.trialeval", "Run TrialEvalBench."),
        "trialdev": _Command("trialagentbench_harness.tools.run.trialdev", "Run TrialDevBench."),
        "trialeval-ablation": _Command(
            "trialagentbench_harness.experiments.trialeval_ablation",
            "Run a frozen TrialEvalBench interface ablation schedule.",
        ),
        "trialeval-normalizer-qualification": _Command(
            "trialagentbench_harness.analysis.experiments.trialeval_normalizer_qualification",
            "Analyse TrialEvalBench narrative-normalizer measurement error.",
            "analyse_main",
        ),
        "trialeval-narrative-normalizer": _Command(
            "trialagentbench_harness.experiments.normalize_trialeval_narrative_packets",
            "Normalize frozen TrialEvalBench narrative reports through a reference-blind provider.",
        ),
        "trialeval-direct-assessment": _Command(
            "trialagentbench_harness.experiments.assess_trialeval_narrative_packets",
            "Assess frozen TrialEvalBench narrative reports through a reference-blind provider.",
        ),
        "trialdev-observational-specification": _Command(
            "trialagentbench_harness.experiments.trialdev_observational_specification",
            "Run a frozen TrialDev observational selection-versus-execution experiment.",
        ),
        "trialdev-checkpoint-replay": _Command(
            "trialagentbench_harness.experiments.trialdev_checkpoint_replay",
            "Run a frozen matched TrialDev checkpoint-replay experiment.",
        ),
        "trialdev-checkpoint-source": _Command(
            "trialagentbench_harness.experiments.capture_trialdev_endogenous_checkpoints",
            "Capture model-produced TrialDev checkpoint sources from a frozen plan.",
        ),
    },
    "grade": {
        "trialeval": _Command(
            "trialagentbench_harness.tools.grade.grade_trialeval",
            "Grade a canonical TrialEvalBench run.",
            "grade_trialeval_run",
        ),
        "trialdev": _Command(
            "trialagentbench_harness.tools.grade.grade_trialdev",
            "Grade a canonical TrialDevBench run.",
        ),
        "trialeval-ablation": _Command(
            "trialagentbench_harness.tools.grade.grade_trialeval_ablation",
            "Grade a TrialEvalBench interface ablation.",
        ),
        "trialdev-observational-specification": _Command(
            "trialagentbench_harness.tools.grade.grade_trialdev_observational_specification",
            "Grade a TrialDev observational selection-versus-execution experiment.",
        ),
        "trialdev-checkpoint-replay": _Command(
            "trialagentbench_harness.tools.grade.grade_trialdev_checkpoint_replay",
            "Grade a TrialDev checkpoint-replay experiment.",
        ),
        "canonical-trialeval": _Command(
            "trialagentbench_harness.tools.grade.grade_canonical_trialeval",
            "Grade a complete canonical TrialEval submission census.",
        ),
        "canonical-trialeval-witnesses": _Command(
            "trialagentbench_harness.tools.grade.grade_canonical_trialeval_witnesses",
            "Grade every canonical TrialEval item-route witness.",
        ),
        "project-trialeval-witnesses": _Command(
            "trialagentbench_harness.tools.grade.project_trialeval_witnesses",
            "Project raw TrialEval route witnesses into the canonical grader contract.",
        ),
        "canonical-trialeval-mutations": _Command(
            "trialagentbench_harness.tools.grade.grade_canonical_trialeval_mutations",
            "Grade generated TrialEval single-fault mutations.",
        ),
        "canonical-trialdev": _Command(
            "trialagentbench_harness.tools.grade.grade_canonical_trialdev",
            "Grade a complete canonical TrialDev evaluation-lane census.",
        ),
        "canonical-trialdev-mutations": _Command(
            "trialagentbench_harness.tools.grade.grade_canonical_trialdev_mutations",
            "Grade generated TrialDev target mutations.",
        ),
    },
    "analyse": {
        "trace": _Command(
            "trialagentbench_harness.tools.build.build_trace_analysis_bundle",
            "Build trace analysis from user-owned runs.",
        ),
        "trialeval-ablation": _Command(
            "trialagentbench_harness.analysis.experiments.trialeval_ablation_cli",
            "Analyse a graded TrialEvalBench interface ablation.",
        ),
        "trialdev-observational-specification": _Command(
            "trialagentbench_harness.analysis.experiments.trialdev_observational_specification",
            "Analyse TrialDev observational selection versus prespecified execution.",
        ),
        "trialdev-checkpoint-replay": _Command(
            "trialagentbench_harness.analysis.experiments.trialdev_checkpoint_replay",
            "Analyse matched TrialDev checkpoint continuations.",
        ),
        "trialdev-results": _Command(
            "trialagentbench_harness.tools.analyse_trialdev_metrics",
            "Summarize typed TrialDev programme assessments by stream.",
        ),
    },
    "verify": {
        "clean-room": _Command(
            "trialagentbench_harness.tools.validate.validate_clean_room_workflow",
            "Verify participant/evaluator release separation.",
        ),
        "trialeval-context": _Command(
            "trialagentbench_harness.tools.validate.validate_trialeval_context_sufficiency",
            "Verify TrialEvalBench context sufficiency.",
        ),
        "trialeval-context-deltas": _Command(
            "trialagentbench_harness.tools.validate.validate_trialeval_context_artifact_deltas",
            "Verify matched TrialEvalBench context artifact semantics.",
        ),
        "trialeval-diagnostics": _Command(
            "trialagentbench_harness.tools.validate.validate_trialeval_diagnostic_proof_surface",
            "Verify TrialEvalBench diagnostic proof surfaces.",
            optional_dependency=("lifelines", "analysis"),
        ),
        "trace-bundle": _Command(
            "trialagentbench_harness.tools.validate.validate_trace_analysis_bundle",
            "Verify a public trace-analysis bundle.",
        ),
        "submission": _Command(
            "trialagentbench_harness.tools.validate.validate_submission",
            "Validate a TrialEval or TrialDev structured submission without evaluator answers.",
        ),
    },
    "export": {
        "results": _Command(
            "trialagentbench_harness.tools.export_results",
            "Export a deterministic, checksummed result bundle.",
        ),
        "trialeval-ablation-schedule": _Command(
            "trialagentbench_harness.experiments.build_trialeval_ablation_schedule",
            "Export a frozen TrialEvalBench interface ablation schedule.",
        ),
        "trialeval-factorial-task-sample": _Command(
            "trialagentbench_harness.experiments.select_trialeval_factorial_tasks",
            "Export the design-bound TrialEvalBench factorial task sample.",
        ),
        "trialdev-observational-specification-schedule": _Command(
            "trialagentbench_harness.experiments.build_trialdev_observational_specification_schedule",
            "Export a frozen TrialDev observational specification schedule.",
        ),
        "trialdev-checkpoint-schedule": _Command(
            "trialagentbench_harness.experiments.build_trialdev_checkpoint_schedule",
            "Compile a verified matched TrialDev checkpoint schedule.",
        ),
        "trialdev-canonical-checkpoints": _Command(
            "trialagentbench_harness.experiments.build_trialdev_canonical_checkpoints",
            "Capture canonical TrialDev checkpoints from verified public trajectories.",
        ),
        "trialdev-public-reference": _Command(
            "trialagentbench_harness.experiments.build_trialdev_reference_run",
            "Execute a deterministic TrialDev trajectory from released evidence.",
        ),
        "trialdev-worked-programmes": _Command(
            "trialagentbench_harness.tools.export_trialdev_worked_programmes",
            "Export reproducible worked programmes for both TrialDev streams.",
        ),
        "trialeval-ablation-labels": _Command(
            "trialagentbench_harness.experiments.export_trialeval_ablation_labels",
            "Export evaluator-owned TrialEvalBench targeted-control labels.",
        ),
        "trialeval-narrative-packets": _Command(
            "trialagentbench_harness.experiments.export_trialeval_narrative_packets",
            "Export masked transcription packets from a completed narrative ablation.",
        ),
        "trialeval-normalizer-packets": _Command(
            "trialagentbench_harness.experiments.export_trialeval_normalizer_sample_packets",
            "Export a frozen cross-run masked normalizer qualification packet set.",
        ),
        "trialeval-normalizer-sample": _Command(
            "trialagentbench_harness.analysis.experiments.trialeval_normalizer_qualification",
            "Select a prospective TrialEvalBench narrative-normalizer qualification sample.",
            "select_main",
        ),
        "trialeval-normalizer-frame": _Command(
            "trialagentbench_harness.experiments.build_trialeval_normalizer_frame",
            "Build the outcome-blind TrialEvalBench narrative-normalizer sampling frame.",
        ),
        "trialeval-normalizer-observations": _Command(
            "trialagentbench_harness.experiments.build_trialeval_normalizer_observations",
            "Join and score frozen human and automated normalizer qualification evidence.",
        ),
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trialagentbench",
        description="Run, grade, analyse, and verify TrialAgentBench.",
    )
    workflows = parser.add_subparsers(dest="workflow", required=True)
    for workflow, commands in _COMMANDS.items():
        workflow_parser = workflows.add_parser(workflow)
        actions = workflow_parser.add_subparsers(dest="action", required=True)
        for action, command in commands.items():
            actions.add_parser(
                action,
                description=command.description,
                help=command.description,
                add_help=False,
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one public TrialAgentBench workflow."""

    parser = _parser()
    args, leaf_argv = parser.parse_known_args(list(argv) if argv is not None else None)
    command = _COMMANDS[args.workflow][args.action]
    try:
        return command.invoke(leaf_argv)
    except MissingOptionalDependencyError as error:
        parser.exit(2, f"trialagentbench: error: {error}\n")


def build_trace_analysis(argv: Sequence[str] | None = None) -> int:
    """Build public trace-analysis artifacts from user-supplied saved runs."""
    return _COMMANDS["analyse"]["trace"].invoke(argv or ())


def validate_trialeval_context_sufficiency(argv: Sequence[str] | None = None) -> int:
    """Validate TrialEvalBench participant-context sufficiency."""
    return _COMMANDS["verify"]["trialeval-context"].invoke(argv or ())


def validate_trialeval_diagnostic_proof_surface(argv: Sequence[str] | None = None) -> int:
    """Validate TrialEvalBench public diagnostic proof surfaces."""
    return _COMMANDS["verify"]["trialeval-diagnostics"].invoke(argv or ())


def validate_clean_room_workflow(argv: Sequence[str] | None = None) -> int:
    """Validate participant, evaluator, runtime, and verification boundaries."""
    return _COMMANDS["verify"]["clean-room"].invoke(argv or ())


def validate_trace_bundle(argv: Sequence[str] | None = None) -> int:
    """Validate a public trace-analysis bundle."""
    return _COMMANDS["verify"]["trace-bundle"].invoke(argv or ())


if __name__ == "__main__":
    raise SystemExit(main())
