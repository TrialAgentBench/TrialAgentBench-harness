"""Load and verify frozen TrialEval narrative packets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from trialagentbench_harness.contracts.experiments import (
    NarrativePacketIndexRowV1,
    NarrativePacketManifestV1,
    NarrativePacketSetManifestV1,
    NarrativeParticipantContextV1,
    NarrativeQualificationPacketSetManifestV1,
)
from trialagentbench_harness.io import read_json, read_json_model, sha256_file

NarrativePacketSetV1: TypeAlias = NarrativePacketSetManifestV1 | NarrativeQualificationPacketSetManifestV1


@dataclass(frozen=True)
class LoadedNarrativePacketV1:
    """One verified narrative packet and its frozen report."""

    index: NarrativePacketIndexRowV1
    manifest: NarrativePacketManifestV1
    participant_context: NarrativeParticipantContextV1
    report: str


def load_narrative_packets_v1(
    packet_root: Path,
) -> tuple[NarrativePacketSetV1, tuple[LoadedNarrativePacketV1, ...]]:
    """Load a complete packet set after verifying every source binding."""

    manifest_path = packet_root / "manifest.json"
    manifest_payload = read_json(manifest_path)
    if not isinstance(manifest_payload, dict):
        raise ValueError("Narrative packet-set manifest must be a JSON object.")
    schema_id = manifest_payload.get("schema_id")
    manifest: NarrativePacketSetV1
    if schema_id == "trialagentbench.trialeval_narrative_packet_set/v1":
        manifest = read_json_model(NarrativePacketSetManifestV1, manifest_path)
    elif schema_id == "trialagentbench.trialeval_narrative_qualification_packet_set/v1":
        manifest = read_json_model(NarrativeQualificationPacketSetManifestV1, manifest_path)
    else:
        raise ValueError(f"Unsupported narrative packet-set schema_id: {schema_id!r}.")

    packets: list[LoadedNarrativePacketV1] = []
    for row in manifest.packets:
        packet_dir = packet_root / row.blinded_identity
        packet_path = packet_dir / "packet.json"
        report_path = packet_dir / "frozen_report.txt"
        context_path = packet_dir / "participant_context.json"
        if any(path.is_symlink() for path in (packet_dir, packet_path, report_path, context_path)):
            raise ValueError(f"Narrative packet paths must not be symbolic links: {packet_dir}")
        if sha256_file(packet_path) != row.packet_manifest_sha256:
            raise ValueError(f"Narrative packet manifest drift: {packet_path}")
        packet = read_json_model(NarrativePacketManifestV1, packet_path)
        if packet.blinded_identity != row.blinded_identity:
            raise ValueError(f"Narrative packet identity drift: {packet_path}")
        if sha256_file(context_path) != packet.participant_context_sha256:
            raise ValueError(f"Narrative participant context drift: {context_path}")
        context = read_json_model(NarrativeParticipantContextV1, context_path)
        if context.task_id != packet.participant_task_id:
            raise ValueError(f"Narrative participant context task drift: {context_path}")
        report = report_path.read_text(encoding="utf-8")
        report_hash = sha256_file(report_path)
        if report_hash != row.report_sha256 or report_hash != packet.report_sha256:
            raise ValueError(f"Narrative packet report drift: {report_path}")
        packets.append(
            LoadedNarrativePacketV1(
                index=row,
                manifest=packet,
                participant_context=context,
                report=report,
            )
        )
    return manifest, tuple(packets)


__all__ = ["LoadedNarrativePacketV1", "NarrativePacketSetV1", "load_narrative_packets_v1"]
