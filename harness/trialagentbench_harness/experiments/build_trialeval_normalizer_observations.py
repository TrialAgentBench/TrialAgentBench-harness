"""Join and score frozen human and automated normalizer qualification evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_harness.analysis.experiments.trialeval_endpoint_scoring import (
    score_trialeval_ablation_submission_v1,
    trialeval_scoring_implementation_sha256_v1,
)
from trialagentbench_harness.analysis.experiments.trialeval_transcription import (
    validate_narrative_transcription_v1,
)
from trialagentbench_harness.contracts.experiments import (
    NarrativeNormalizationBatchConfigV1,
    NarrativeNormalizationBatchManifestV1,
    NarrativeQualificationPacketSetManifestV1,
    TrialEvalNarrativeTranscriptionV1,
    TrialEvalNormalizerQualificationObservationSetV1,
    TrialEvalNormalizerQualificationObservationV1,
    TrialEvalNormalizerSampleV1,
)
from trialagentbench_harness.contracts.scoring.assumption_evidence import (
    read_assumption_evidence_domains,
)
from trialagentbench_harness.experiments.narrative_normalization import NarrativeNormalizationResultV1
from trialagentbench_harness.experiments.trialeval_run_artifacts import (
    load_completed_trialeval_ablation_run,
)
from trialagentbench_harness.grading import ScoringKeyStoreV1
from trialagentbench_harness.io import read_json_model, sha256_file, sha256_path, write_json_model
from trialagentbench_harness.trialeval.data import discover_items


def _regular_descendant(root: Path, relative: str) -> Path:
    unresolved = root / relative
    if unresolved.is_symlink():
        raise ValueError(f"Qualification result path is not a regular batch descendant: {relative!r}.")
    candidate = unresolved.resolve(strict=True)
    if not candidate.is_file() or not candidate.is_relative_to(root.resolve(strict=True)):
        raise ValueError(f"Qualification result path is not a regular batch descendant: {relative!r}.")
    return candidate


def build_trialeval_normalizer_observations_v1(
    *,
    sample: TrialEvalNormalizerSampleV1,
    packet_root: Path,
    manual_transcriptions_dir: Path,
    normalization_batch_root: Path,
    run_dirs: tuple[Path, ...],
    evaluator_root: Path,
) -> TrialEvalNormalizerQualificationObservationSetV1:
    """Validate, score, and join the exact prospective normalizer sample."""

    if sample.checksum is None:
        raise ValueError("Normalizer qualification sample must be frozen with a checksum.")
    packets_root = Path(packet_root).resolve(strict=True)
    packet_manifest_path = packets_root / "manifest.json"
    packet_manifest = read_json_model(NarrativeQualificationPacketSetManifestV1, packet_manifest_path)
    if packet_manifest.sample_checksum != sample.checksum or packet_manifest.checksum is None:
        raise ValueError("Qualification packet set does not match the frozen sample.")
    packet_by_unit = {row.qualification_unit_id: row for row in packet_manifest.packets}
    sample_ids = {unit.unit_id for unit in sample.units}
    if set(packet_by_unit) != sample_ids:
        raise ValueError("Qualification packet set does not cover the exact sampled units.")

    batch_root = Path(normalization_batch_root).resolve(strict=True)
    batch_config = read_json_model(NarrativeNormalizationBatchConfigV1, batch_root / "batch_config.json")
    batch_manifest = read_json_model(NarrativeNormalizationBatchManifestV1, batch_root / "manifest.json")
    if batch_config.packet_set_manifest_sha256 != sha256_file(packet_manifest_path):
        raise ValueError("Normalization batch is not bound to the qualification packet set.")
    if batch_config.checksum is None or batch_manifest.config_checksum != batch_config.checksum:
        raise ValueError("Normalization batch manifest is not bound to its frozen configuration.")
    if batch_manifest.checksum is None:
        raise ValueError("Normalization batch manifest must be frozen with a checksum.")
    automated_records: dict[str, list[tuple[int, TrialEvalNarrativeTranscriptionV1]]] = defaultdict(list)
    for record in batch_manifest.records:
        if record.qualification_unit_id is None:
            raise ValueError("Qualification normalization records require sampled unit identities.")
        path = _regular_descendant(batch_root, record.result_file)
        if sha256_file(path) != record.result_sha256:
            raise ValueError(f"Normalization result checksum drift: {path}")
        normalization_result = read_json_model(NarrativeNormalizationResultV1, path)
        automated_records[record.qualification_unit_id].append(
            (record.repeat_index, normalization_result.transcription)
        )
    if set(automated_records) != sample_ids:
        raise ValueError("Normalization batch does not cover the exact sampled units.")

    runs = tuple(load_completed_trialeval_ablation_run(path) for path in run_dirs)
    by_run = {run.run_config.run_identity_sha256: run for run in runs}
    if set(by_run) != {unit.run_identity_sha256 for unit in sample.units}:
        raise ValueError("Observation source runs do not equal the sampled run identities.")
    evaluator = Path(evaluator_root).resolve(strict=True)
    discovered = discover_items(evaluator)
    items = {item.task_id: item for item in discovered}
    if len(items) != len(discovered):
        raise ValueError("Evaluator release contains duplicate TrialEval task IDs.")
    evaluator_task_ids = tuple(sorted(items))
    scoring_keys = ScoringKeyStoreV1.from_release(
        evaluator,
        expected_item_ids=evaluator_task_ids,
    )
    assumption_evidence = read_assumption_evidence_domains(release_root=evaluator)
    if set(assumption_evidence) != set(evaluator_task_ids):
        raise ValueError("assumption-evidence denominator must match the evaluator scoring-key denominator")
    manual_root = Path(manual_transcriptions_dir).resolve(strict=True)

    observations: list[TrialEvalNormalizerQualificationObservationV1] = []
    for unit in sample.units:
        packet_index = packet_by_unit[unit.unit_id]
        if packet_index is None:
            raise ValueError(f"Sampled unit lacks a qualification packet: {unit.unit_id!r}.")
        packet_dir = packets_root / packet_index.blinded_identity
        if packet_dir.is_symlink():
            raise ValueError(f"Qualification packet directory is a symlink: {unit.unit_id!r}.")
        report_path = _regular_descendant(
            packets_root,
            f"{packet_index.blinded_identity}/frozen_report.txt",
        )
        if sha256_file(report_path) != unit.report_sha256:
            raise ValueError(f"Qualification packet report drift: {unit.unit_id!r}.")
        report = report_path.read_text(encoding="utf-8")
        human_path = _regular_descendant(manual_root, f"{unit.unit_id}.json")
        human = read_json_model(TrialEvalNarrativeTranscriptionV1, human_path)
        automated = tuple(
            transcription for _, transcription in sorted(automated_records[unit.unit_id], key=lambda row: row[0])
        )
        for transcription in (human, *automated):
            validate_narrative_transcription_v1(
                transcription=transcription,
                frozen_report=report,
                expected_assignment_id=unit.assignment_id,
                expected_task_id=unit.task_id,
            )
        run = by_run[unit.run_identity_sha256]
        result = run.result_by_assignment().get(unit.assignment_id)
        item = items.get(unit.task_id)
        if result is None or item is None:
            raise ValueError(f"Sampled unit lacks its source result or evaluator item: {unit.unit_id!r}.")
        observations.append(
            TrialEvalNormalizerQualificationObservationV1(
                sample_unit=unit,
                masked_human_reference=human,
                automated_repeats=automated,
                masked_human_endpoint=score_trialeval_ablation_submission_v1(
                    scoring_key=scoring_keys.for_item(unit.task_id),
                    assumption_evidence=assumption_evidence[unit.task_id],
                    item=item,
                    result=result,
                    submission=human.submission,
                    normalization_source="manual_masked",
                    normalization_status=human.status,
                    normalization_failure_reason=human.abstention_reason,
                ),
                automated_endpoint=score_trialeval_ablation_submission_v1(
                    scoring_key=scoring_keys.for_item(unit.task_id),
                    assumption_evidence=assumption_evidence[unit.task_id],
                    item=item,
                    result=result,
                    submission=automated[0].submission,
                    normalization_source="automated_importer",
                    normalization_status=automated[0].status,
                    normalization_failure_reason=automated[0].abstention_reason,
                ),
            )
        )
    for run in runs:
        run.assert_unchanged()
    return TrialEvalNormalizerQualificationObservationSetV1(
        sample_checksum=sample.checksum,
        packet_set_checksum=packet_manifest.checksum,
        normalization_batch_checksum=batch_manifest.checksum,
        evaluator_release_sha256=sha256_path(evaluator),
        scoring_implementation_sha256=trialeval_scoring_implementation_sha256_v1(),
        observations=tuple(sorted(observations, key=lambda row: row.sample_unit.unit_id)),
    ).with_checksum()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--manual-dir", required=True)
    parser.add_argument("--normalization-batch", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--evaluator-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build one immutable scored normalizer-observation set."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite normalizer observations: {output}")
    observations = build_trialeval_normalizer_observations_v1(
        sample=read_json_model(TrialEvalNormalizerSampleV1, Path(args.sample)),
        packet_root=Path(args.packet_root),
        manual_transcriptions_dir=Path(args.manual_dir),
        normalization_batch_root=Path(args.normalization_batch),
        run_dirs=tuple(Path(path) for path in args.run_dir),
        evaluator_root=Path(args.evaluator_dir),
    )
    write_json_model(output, observations)
    print(f"Joined {len(observations.observations)} normalizer qualification observations: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_trialeval_normalizer_observations_v1", "main"]
