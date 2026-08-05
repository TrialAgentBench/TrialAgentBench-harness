"""Export blinded manual-transcription packets from a completed ablation run."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_harness.contracts.experiments import (
    NarrativePacketIndexRowV1,
    NarrativePacketManifestV1,
    NarrativePacketSetManifestV1,
    NarrativeReportStateV1,
    manual_transcription_template_v1,
)
from trialagentbench_harness.experiments.narrative_packet_context import (
    load_narrative_participant_contexts_v1,
)
from trialagentbench_harness.experiments.trialeval_run_artifacts import (
    load_completed_trialeval_ablation_run,
)
from trialagentbench_harness.io import sha256_file, staged_directory, write_json, write_json_model


def export_narrative_transcription_packets(run_dir: Path, participant_root: Path, output_dir: Path) -> Path:
    """Export one non-overwriting, blinded packet per narrative assignment."""

    run = load_completed_trialeval_ablation_run(run_dir)
    source = run.source
    output = output_dir.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Packet output must not be inside the immutable ablation run.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite narrative packet output: {output}")

    schedule = run.schedule
    run_config = run.run_config
    results = run.result_by_assignment()
    narrative_assignments = tuple(
        assignment for assignment in schedule.assignments if assignment.submission_interface == "narrative"
    )
    if not narrative_assignments:
        raise ValueError("Ablation run contains no narrative assignments to transcribe.")
    task_ids = tuple(sorted({assignment.task_id for assignment in narrative_assignments}))
    participant_contexts = load_narrative_participant_contexts_v1(
        participant_root=participant_root,
        expected_release_sha256=run_config.participant_release_sha256,
        task_ids=task_ids,
    )

    with staged_directory(output) as staging:
        (staging / "README.md").write_text(
            "# Masked narrative transcription\n\n"
            "Two transcribers independently encode each frozen report without model or evaluator-reference "
            "access. Each packet includes the exact participant task and output contracts used for "
            "normalization. Record `independent_exact_agreement` when complete canonical submissions match; "
            "otherwise resolve discrepancies while still masked and record `adjudicated_resolution`. "
            "Every extracted claim must preserve its role, evidence level, raw value, parsed value, conflict "
            "state, and exact report offsets. Only one executed or substantiated primary claim per required "
            "field can form the canonical submission; rejected, hypothetical, secondary, sensitivity, and "
            "ambiguous claims remain recorded but cannot receive primary credit.\n",
            encoding="utf-8",
        )
        packet_rows: list[NarrativePacketIndexRowV1] = []
        for index, assignment in enumerate(narrative_assignments, start=1):
            result = results[assignment.assignment_id]
            report = result.agent_output.report
            report_state: NarrativeReportStateV1
            if report is None:
                report_state = "absent"
                report_bytes = b""
            else:
                report_state = "present" if report.strip() else "blank"
                report_bytes = report.encode("utf-8")
            report_sha256 = hashlib.sha256(report_bytes).hexdigest()
            blinded_identity = f"masked-narrative-{index:04d}"
            packet_dir = staging / blinded_identity
            packet_dir.mkdir()
            (packet_dir / "frozen_report.txt").write_bytes(report_bytes)
            context_path = packet_dir / "participant_context.json"
            write_json_model(context_path, participant_contexts[assignment.task_id])
            packet_manifest = NarrativePacketManifestV1(
                blinded_identity=blinded_identity,
                participant_task_id=assignment.task_id,
                assignment_id=assignment.assignment_id,
                report_state=report_state,
                report_sha256=report_sha256,
                participant_context_sha256=sha256_file(context_path),
            )
            packet_manifest_path = packet_dir / "packet.json"
            write_json_model(packet_manifest_path, packet_manifest)
            write_json(
                packet_dir / "transcription_template.json",
                manual_transcription_template_v1(
                    assignment_id=assignment.assignment_id,
                    report_sha256=report_sha256,
                    report_state=report_state,
                ),
            )
            packet_rows.append(
                NarrativePacketIndexRowV1(
                    blinded_identity=blinded_identity,
                    packet_manifest_sha256=sha256_file(packet_manifest_path),
                    report_sha256=report_sha256,
                )
            )

        run.assert_unchanged()
        write_json_model(
            staging / "manifest.json",
            NarrativePacketSetManifestV1(
                schedule_sha256=str(schedule.checksum),
                run_identity_sha256=run_config.run_identity_sha256,
                participant_release_sha256=run_config.participant_release_sha256,
                source_files_sha256={
                    str(path.relative_to(source)): digest
                    for path, digest in sorted(run.source_hashes, key=lambda item: str(item[0]))
                },
                packets=tuple(packet_rows),
            ).with_checksum(),
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """Export narrative packets without registering a public CLI command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Immutable completed TrialEval ablation run.")
    parser.add_argument("participant_dir", help="Exact participant release used by the run.")
    parser.add_argument("output_dir", help="New narrative packet output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    output = export_narrative_transcription_packets(
        Path(args.run_dir),
        Path(args.participant_dir),
        Path(args.output_dir),
    )
    print(f"Narrative transcription packets exported to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
