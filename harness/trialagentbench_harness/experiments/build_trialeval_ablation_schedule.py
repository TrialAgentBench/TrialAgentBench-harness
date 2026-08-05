"""Build a complete TrialEval ablation schedule from participant metadata only."""

from __future__ import annotations

import argparse
import hashlib
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from trialagentbench_harness.contracts.experiments import (
    EXPERIMENT_IDENTIFIER_PATTERN_V1,
    TrialEvalAblationAnalysisConfigV1,
    TrialEvalAblationAssignmentV1,
    TrialEvalAblationScheduleV1,
    TrialEvalExperimentProtocolV1,
    TrialEvalFactorialTaskSampleV1,
    trialeval_ablation_cells_v1,
)
from trialagentbench_harness.contracts.release import TrialEvalParticipantManifestV1
from trialagentbench_harness.io import read_json_model, sha256_dir_digest, write_json_model
from trialagentbench_harness.trialeval.conditions import prompt_set_sha256_v1
from trialagentbench_harness.trialeval.data import (
    discover_participant_items,
    participant_analysis_surface_sha256,
)


def build_trialeval_ablation_schedule_v1(
    *,
    participant_root: Path,
    experiment_id: str,
    design: Literal["factorial_interface", "targeted_control"],
    experiment_design: TrialEvalExperimentProtocolV1,
    analysis_config: TrialEvalAblationAnalysisConfigV1,
    replicate_seeds: dict[str, int],
    randomization_seed: int,
    task_ids: tuple[str, ...] | None = None,
    task_sample: TrialEvalFactorialTaskSampleV1 | None = None,
) -> TrialEvalAblationScheduleV1:
    """Return a fully crossed schedule without reading evaluator metadata."""

    root = Path(participant_root)
    if analysis_config.design != design:
        raise ValueError("Ablation analysis contract design does not match the schedule design.")
    if analysis_config.experiment_design_sha256 != experiment_design.checksum:
        raise ValueError("Ablation analysis contract does not bind the supplied experiment design.")
    primary_contrasts = tuple(
        contrast
        for contrast in experiment_design.contrasts
        if contrast.contrast_id == analysis_config.primary_estimand.contrast_id
    )
    if len(primary_contrasts) != 1:
        raise ValueError("Ablation primary estimand does not identify one frozen design contrast.")
    if analysis_config.primary_estimand.metric not in primary_contrasts[0].metrics:
        raise ValueError("Ablation primary metric is not defined for its frozen design contrast.")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"TrialEval participant manifest is missing: {manifest_path}")
    manifest = TrialEvalParticipantManifestV1.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    manifest_task_ids = manifest.task_ids
    participant_release_sha256 = sha256_dir_digest(root)
    if not replicate_seeds:
        raise ValueError("replicate_seeds must be non-empty.")
    if any(
        re.fullmatch(EXPERIMENT_IDENTIFIER_PATTERN_V1, replicate_id) is None
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        for replicate_id, seed in replicate_seeds.items()
    ):
        raise ValueError("replicate_seeds requires safe replicate_id values and nonnegative integer seeds.")
    if len(set(replicate_seeds.values())) != len(replicate_seeds):
        raise ValueError("replicate_seeds values must be unique.")
    if len(replicate_seeds) < analysis_config.min_decoding_replicates:
        raise ValueError(
            "replicate_seeds does not meet the prospective decoding-replicate requirement: "
            f"required={analysis_config.min_decoding_replicates} observed={len(replicate_seeds)}."
        )
    if task_sample is not None:
        if task_sample.experiment_design_sha256 != experiment_design.checksum:
            raise ValueError("Factorial task sample does not bind the supplied experiment design.")
        if task_sample.participant_release_sha256 != participant_release_sha256:
            raise ValueError("Factorial task sample does not bind the supplied participant release.")
        if task_ids is not None and set(task_ids) != set(task_sample.task_ids):
            raise ValueError("Explicit task_ids differ from the frozen factorial task sample.")
    selected_task_ids = tuple(
        sorted(
            task_sample.task_ids
            if task_sample is not None
            else task_ids if task_ids is not None else manifest_task_ids
        )
    )
    if not selected_task_ids or len(selected_task_ids) != len(set(selected_task_ids)):
        raise ValueError("task_ids must be non-empty and unique when provided.")
    unknown = sorted(set(selected_task_ids).difference(manifest_task_ids))
    if unknown:
        raise ValueError(f"Unknown participant task_ids requested: {unknown!r}.")
    if len(selected_task_ids) < analysis_config.min_base_trial_clusters:
        raise ValueError(
            "Selected participant tasks cannot meet the prospective base-trial requirement: "
            f"required={analysis_config.min_base_trial_clusters} observed={len(selected_task_ids)}."
        )
    if analysis_config.execution_scope == "publication":
        if task_sample is None:
            raise ValueError("Publication schedule construction requires a frozen factorial task sample.")
        if design != "factorial_interface":
            raise ValueError("The frozen TrialEval publication design currently defines the factorial experiment.")
        envelope = experiment_design.compute_envelope
        if analysis_config.min_base_trial_clusters != experiment_design.precision.retained_independent_base_trials:
            raise ValueError("Publication analysis must use the frozen independent-base-trial requirement.")
        if analysis_config.min_decoding_replicates != envelope.decoding_replicates:
            raise ValueError("Publication analysis must use the frozen decoding-replicate requirement.")
        if len(replicate_seeds) != envelope.decoding_replicates:
            raise ValueError("Publication schedule decoding replicates differ from the frozen compute envelope.")
        if len(selected_task_ids) != envelope.factorial_item_count:
            raise ValueError("Publication schedule task count differs from the frozen factorial sample.")
        observed_contexts = tuple(
            sum(
                manifest.task_evidence_factors[task_id].context_configuration == context
                for task_id in selected_task_ids
            )
            for context in ("C1", "C2", "C3", "C4", "C5")
        )
        if observed_contexts != envelope.factorial_context_allocation:
            raise ValueError(
                "Publication schedule context allocation differs from the frozen factorial sample: "
                f"expected={envelope.factorial_context_allocation!r} observed={observed_contexts!r}."
            )
    items = discover_participant_items(root, task_ids=selected_task_ids)
    surface_digests = {task_id: participant_analysis_surface_sha256(items[task_id]) for task_id in selected_task_ids}

    assignments: list[TrialEvalAblationAssignmentV1] = []
    for task_id in selected_task_ids:
        factors = manifest.task_evidence_factors[task_id]
        for replicate_id, decoding_seed in sorted(replicate_seeds.items()):
            for procedure_assistance, prompt_condition, submission_interface in trialeval_ablation_cells_v1(design):
                assignment_key = (
                    f"{experiment_id}\0{task_id}\0{replicate_id}\0{factors.analysis_specification}\0"
                    f"{procedure_assistance}\0{prompt_condition}\0{submission_interface}"
                )
                assignment_id = f"A{hashlib.sha256(assignment_key.encode('utf-8')).hexdigest()[:32]}"
                assignments.append(
                    TrialEvalAblationAssignmentV1(
                        assignment_id=assignment_id,
                        task_id=task_id,
                        context_tier=factors.context_configuration,
                        data_preparation=factors.data_preparation,
                        analysis_specification=factors.analysis_specification,
                        analysis_surface_sha256=surface_digests[task_id],
                        replicate_id=replicate_id,
                        decoding_seed=decoding_seed,
                        procedure_assistance=procedure_assistance,
                        prompt_condition=prompt_condition,
                        submission_interface=submission_interface,
                    )
                )
    assignments.sort(key=lambda assignment: assignment.assignment_id)
    random.Random(randomization_seed).shuffle(assignments)
    return TrialEvalAblationScheduleV1(
        experiment_id=experiment_id,
        design=design,
        execution_scope=analysis_config.execution_scope,
        experiment_design_sha256=experiment_design.checksum,
        participant_release_sha256=participant_release_sha256,
        prompt_set_sha256=prompt_set_sha256_v1(),
        analysis_config_sha256=cast(str, analysis_config.checksum),
        factorial_task_sample_sha256=None if task_sample is None else task_sample.checksum,
        randomization_seed=randomization_seed,
        assignments=tuple(assignments),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-dir", required=True, help="Extracted TrialEval participant release root.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--design", required=True, choices=("factorial_interface", "targeted_control"))
    parser.add_argument("--experiment-design", required=True, help="Frozen TrialEval experiment-design JSON.")
    parser.add_argument("--task-sample", help="Frozen opaque factorial task-sample JSON.")
    parser.add_argument(
        "--analysis-config",
        required=True,
        help="Checksummed prospective analysis configuration JSON.",
    )
    parser.add_argument("--randomization-seed", required=True, type=int, help="Nonnegative assignment-order seed.")
    parser.add_argument(
        "--replicate",
        action="append",
        required=True,
        dest="replicates",
        metavar="ID=SEED",
        help="Opaque replicate identifier and provider decoding seed; repeat for each independent block.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Exact opaque task ID; repeat to run a participant-only subset.",
    )
    parser.add_argument("--out", required=True, help="New schedule JSON path.")
    return parser


def _parse_replicate_seeds(values: Sequence[str]) -> dict[str, int]:
    """Parse unique `ID=SEED` declarations without implicit coercion."""

    result: dict[str, int] = {}
    for value in values:
        replicate_id, separator, raw_seed = value.partition("=")
        if not separator or not replicate_id or not raw_seed.isascii() or not raw_seed.isdecimal():
            raise ValueError("--replicate must use ID=SEED with a nonnegative decimal integer.")
        if replicate_id in result:
            raise ValueError(f"Duplicate --replicate identifier: {replicate_id!r}.")
        result[replicate_id] = int(raw_seed)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Build and write one immutable participant-only schedule."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite TrialEval ablation schedule: {output}")
    analysis_config = read_json_model(
        TrialEvalAblationAnalysisConfigV1,
        Path(args.analysis_config),
    )
    experiment_design = read_json_model(TrialEvalExperimentProtocolV1, Path(args.experiment_design))
    task_sample = (
        None if args.task_sample is None else read_json_model(TrialEvalFactorialTaskSampleV1, Path(args.task_sample))
    )
    schedule = build_trialeval_ablation_schedule_v1(
        participant_root=Path(args.participant_dir),
        experiment_id=str(args.experiment_id),
        design=cast(Literal["factorial_interface", "targeted_control"], args.design),
        experiment_design=experiment_design,
        analysis_config=analysis_config,
        replicate_seeds=_parse_replicate_seeds(tuple(str(value) for value in args.replicates)),
        randomization_seed=int(args.randomization_seed),
        task_ids=tuple(str(value) for value in args.task_ids) if args.task_ids else None,
        task_sample=task_sample,
    )
    write_json_model(output, schedule)
    print(f"Wrote TrialEval ablation schedule: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
