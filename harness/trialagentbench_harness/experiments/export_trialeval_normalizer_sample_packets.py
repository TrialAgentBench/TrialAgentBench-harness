"""Export one globally blinded packet set from a frozen normalizer sample."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_harness.contracts.experiments import (
    NarrativePacketIndexRowV1,
    NarrativePacketManifestV1,
    NarrativeQualificationPacketSetManifestV1,
    NarrativeReportStateV1,
    TrialEvalNormalizerSampleV1,
    manual_transcription_template_v1,
)
from trialagentbench_harness.experiments.narrative_packet_context import (
    load_narrative_participant_contexts_v1,
)
from trialagentbench_harness.experiments.trialeval_run_artifacts import (
    load_completed_trialeval_ablation_run,
)
from trialagentbench_harness.io import read_json_model, sha256_file, staged_directory, write_json, write_json_model


def _report_bytes(report: str | None) -> tuple[NarrativeReportStateV1, bytes]:
    if report is None:
        return "absent", b""
    return ("present" if report.strip() else "blank"), report.encode("utf-8")


def export_normalizer_sample_packets_v1(
    *,
    sample: TrialEvalNormalizerSampleV1,
    run_dirs: tuple[Path, ...],
    participant_root: Path,
    output_dir: Path,
) -> NarrativeQualificationPacketSetManifestV1:
    """Export exactly the sampled reports without exposing model or reference metadata."""

    if sample.checksum is None:
        raise ValueError("Normalizer qualification sample must be frozen with a checksum before packet export.")
    if not run_dirs:
        raise ValueError("Normalizer qualification packet export requires source runs.")
    runs = tuple(load_completed_trialeval_ablation_run(path) for path in run_dirs)
    by_run = {run.run_config.run_identity_sha256: run for run in runs}
    if len(by_run) != len(runs):
        raise ValueError("Normalizer qualification packet sources contain duplicate run identities.")
    required_runs = {unit.run_identity_sha256 for unit in sample.units}
    if set(by_run) != required_runs:
        raise ValueError("Normalizer qualification packet sources must equal the sampled run identities.")
    schedule_checksums = {run.run_config.schedule_checksum for run in runs}
    if len(schedule_checksums) != 1:
        raise ValueError("Normalizer qualification packet sources do not share one schedule.")
    participant_release_sha256s = {run.run_config.participant_release_sha256 for run in runs}
    if len(participant_release_sha256s) != 1:
        raise ValueError("Normalizer qualification packet sources do not share one participant release.")
    participant_release_sha256 = next(iter(participant_release_sha256s))
    participant_contexts = load_narrative_participant_contexts_v1(
        participant_root=participant_root,
        expected_release_sha256=participant_release_sha256,
        task_ids=tuple(sorted({unit.task_id for unit in sample.units})),
    )
    output = Path(output_dir).resolve()
    if any(output == run.source or output.is_relative_to(run.source) for run in runs):
        raise ValueError("Qualification packet output cannot be inside an immutable source run.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite normalizer qualification packets: {output}")

    ranked_units = tuple(
        sorted(
            sample.units,
            key=lambda unit: (
                hashlib.sha256(f"{sample.selection_seed}:{unit.unit_id}".encode()).hexdigest(),
                unit.unit_id,
            ),
        )
    )
    with staged_directory(output) as staging:
        (staging / "README.md").write_text(
            "# Masked narrative-normalizer qualification\n\n"
            "This directory contains the frozen probability sample only. Transcribers receive packet directories "
            "without run, model, score, accepted-method, or evaluator-reference metadata. Two transcribers independently "
            "transcribe source-grounded claims using the exact participant contracts in each packet; discrepancies "
            "are resolved while both transcribers remain masked.\n",
            encoding="utf-8",
        )
        packet_rows: list[NarrativePacketIndexRowV1] = []
        for index, unit in enumerate(ranked_units, start=1):
            run = by_run[unit.run_identity_sha256]
            result = run.result_by_assignment().get(unit.assignment_id)
            if result is None:
                raise ValueError(f"Sampled assignment is absent from its source run: {unit.assignment_id!r}.")
            if result.assignment.task_id != unit.task_id or run.run_config.model != unit.model_id:
                raise ValueError(f"Sampled report identity drifts from its source run: {unit.unit_id!r}.")
            report_state, report_bytes = _report_bytes(result.agent_output.report)
            report_sha256 = hashlib.sha256(report_bytes).hexdigest()
            if report_sha256 != unit.report_sha256:
                raise ValueError(f"Sampled report bytes drift from the frozen frame: {unit.unit_id!r}.")
            blinded_identity = f"masked-normalizer-{index:04d}"
            packet_dir = staging / blinded_identity
            packet_dir.mkdir()
            (packet_dir / "frozen_report.txt").write_bytes(report_bytes)
            context_path = packet_dir / "participant_context.json"
            write_json_model(context_path, participant_contexts[unit.task_id])
            packet = NarrativePacketManifestV1(
                blinded_identity=blinded_identity,
                participant_task_id=unit.task_id,
                assignment_id=unit.assignment_id,
                report_state=report_state,
                report_sha256=report_sha256,
                participant_context_sha256=sha256_file(context_path),
            )
            packet_path = packet_dir / "packet.json"
            write_json_model(packet_path, packet)
            write_json(
                packet_dir / "transcription_template.json",
                manual_transcription_template_v1(
                    assignment_id=unit.assignment_id,
                    report_sha256=report_sha256,
                    report_state=report_state,
                ),
            )
            packet_rows.append(
                NarrativePacketIndexRowV1(
                    blinded_identity=blinded_identity,
                    qualification_unit_id=unit.unit_id,
                    packet_manifest_sha256=sha256_file(packet_path),
                    report_sha256=report_sha256,
                )
            )
        for run in runs:
            run.assert_unchanged()
        source_hashes = {
            f"{run.run_config.run_identity_sha256}/{path.relative_to(run.source)}": digest
            for run in runs
            for path, digest in run.source_hashes
        }
        manifest = NarrativeQualificationPacketSetManifestV1(
            sample_checksum=sample.checksum,
            schedule_sha256=next(iter(schedule_checksums)),
            participant_release_sha256=participant_release_sha256,
            source_run_identity_sha256s=tuple(sorted(by_run)),
            source_files_sha256=dict(sorted(source_hashes.items())),
            packets=tuple(packet_rows),
        ).with_checksum()
        write_json_model(staging / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--participant-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Export one immutable cross-run masked qualification packet set."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    sample = read_json_model(TrialEvalNormalizerSampleV1, Path(args.sample))
    manifest = export_normalizer_sample_packets_v1(
        sample=sample,
        run_dirs=tuple(Path(path) for path in args.run_dir),
        participant_root=Path(args.participant_dir),
        output_dir=Path(args.out_dir),
    )
    print(f"Exported {len(manifest.packets)} masked qualification packets: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export_normalizer_sample_packets_v1", "main"]
