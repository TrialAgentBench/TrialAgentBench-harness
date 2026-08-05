"""Assess frozen TrialEval narrative reports through one reference-blind provider."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from trialagentbench_harness.adapters import ProviderRouting, get_provider
from trialagentbench_harness.contracts.experiments import (
    DirectAssessmentBatchConfigV1,
    DirectAssessmentBatchManifestV1,
    DirectAssessmentBatchRecordV1,
)
from trialagentbench_harness.execution_policy import TRIALEVAL_RELEASE_BUDGET_V1
from trialagentbench_harness.experiments.direct_judge import (
    DirectJudgeRequestV1,
    DirectJudgeResultV1,
    judge_narrative_report_v1,
)
from trialagentbench_harness.experiments.narrative_packet_io import load_narrative_packets_v1
from trialagentbench_harness.io import (
    canonical_payload_sha256,
    read_json_model,
    sha256_file,
    write_json_model,
)
from trialagentbench_harness.util.provider_environment import load_provider_dotenv


def assess_packet_set(
    *,
    packet_root: Path,
    output_root: Path,
    config: DirectAssessmentBatchConfigV1,
    resume: bool,
) -> DirectAssessmentBatchManifestV1:
    """Assess every frozen report and retain a complete resumable denominator."""

    packet_manifest_path = packet_root / "manifest.json"
    packet_manifest, packets = load_narrative_packets_v1(packet_root)
    if config.packet_set_manifest_sha256 != sha256_file(packet_manifest_path):
        raise ValueError("Batch configuration does not match the exact packet-set manifest.")

    config_path = output_root / "batch_config.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite direct-assessment batch: {output_root}")
        existing_config = read_json_model(DirectAssessmentBatchConfigV1, config_path)
        if existing_config != config:
            raise ValueError("Existing direct-assessment batch configuration differs from the requested run.")
    else:
        output_root.mkdir(parents=True)
        write_json_model(config_path, config)

    provider = get_provider(
        config.judge_model,
        routing=ProviderRouting(provider=config.provider, openrouter_provider=config.openrouter_provider),
        send_temperature=config.send_temperature,
        timeout_s=config.timeout_seconds,
        decoding_seed=config.decoding_seed,
    )
    records: list[DirectAssessmentBatchRecordV1] = []
    for loaded in packets:
        packet_index = loaded.index
        packet = loaded.manifest
        request = DirectJudgeRequestV1(
            assignment_id=packet.assignment_id,
            task_id=packet.participant_task_id,
            raw_response=loaded.report,
            participant_context=loaded.participant_context,
            judge_model=config.judge_model,
            decoding_seed=config.decoding_seed,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
        )
        request_sha256 = canonical_payload_sha256(request.model_dump(mode="json"))
        for repeat_index in range(1, config.repeats + 1):
            result_path = output_root / "results" / packet.blinded_identity / f"repeat-{repeat_index:04d}.json"
            if result_path.exists():
                if not resume:
                    raise FileExistsError(f"Refusing to overwrite direct-assessment result: {result_path}")
                result = read_json_model(DirectJudgeResultV1, result_path)
                if result.request_sha256 != request_sha256:
                    raise ValueError(f"Existing direct-assessment result request drift: {result_path}")
            else:
                result = judge_narrative_report_v1(request=request, provider=provider)
                write_json_model(result_path, result)
            records.append(
                DirectAssessmentBatchRecordV1(
                    blinded_identity=packet.blinded_identity,
                    qualification_unit_id=packet_index.qualification_unit_id,
                    assignment_id=packet.assignment_id,
                    repeat_index=repeat_index,
                    result_file=str(result_path.relative_to(output_root)),
                    result_sha256=sha256_file(result_path),
                    status=result.status,
                )
            )

    load_narrative_packets_v1(packet_root)
    ordered = tuple(sorted(records, key=lambda row: (row.blinded_identity, row.repeat_index)))
    manifest = DirectAssessmentBatchManifestV1(
        config_checksum=cast(str, config.checksum),
        packet_count=len(packet_manifest.packets),
        repeat_count=config.repeats,
        result_count=len(ordered),
        completed_count=sum(row.status == "completed" for row in ordered),
        invalid_response_count=sum(row.status == "invalid_response" for row in ordered),
        records=ordered,
    ).with_checksum()
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        existing_manifest = read_json_model(DirectAssessmentBatchManifestV1, manifest_path)
        if existing_manifest != manifest:
            raise ValueError("Existing completed direct-assessment manifest differs from recomputation.")
    else:
        write_json_model(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet_root", help="Frozen masked narrative packet directory.")
    parser.add_argument("output_root", help="New or explicitly resumed assessment output directory.")
    parser.add_argument("--provider", choices=("openai", "openai_responses", "openrouter"), required=True)
    parser.add_argument("--openrouter-provider")
    parser.add_argument("--model", required=True)
    parser.add_argument("--decoding-seed", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--omit-temperature", action="store_true")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=TRIALEVAL_RELEASE_BUDGET_V1.maximum_completion_tokens_per_turn,
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dotenv", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run or resume a complete reference-blind semantic-assessment batch."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.provider == "openai_responses" and args.decoding_seed is not None:
        raise ValueError("--decoding-seed is not supported by --provider openai_responses")
    if args.dotenv:
        load_provider_dotenv()
    packet_root = Path(args.packet_root).resolve(strict=True)
    config = DirectAssessmentBatchConfigV1(
        packet_set_manifest_sha256=sha256_file(packet_root / "manifest.json"),
        provider=args.provider,
        openrouter_provider=args.openrouter_provider,
        judge_model=args.model,
        decoding_seed=args.decoding_seed,
        temperature=args.temperature,
        send_temperature=not args.omit_temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        repeats=args.repeats,
    ).with_checksum()
    manifest = assess_packet_set(
        packet_root=packet_root,
        output_root=Path(args.output_root).resolve(),
        config=config,
        resume=args.resume,
    )
    print(f"Assessed {manifest.result_count} frozen narrative reports: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DirectAssessmentBatchConfigV1",
    "DirectAssessmentBatchManifestV1",
    "DirectAssessmentBatchRecordV1",
    "assess_packet_set",
]
