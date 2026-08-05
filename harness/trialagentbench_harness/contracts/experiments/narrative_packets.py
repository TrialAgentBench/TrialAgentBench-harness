"""Contracts for masked TrialEval narrative-normalization packets."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import JsonValue

from trialagentbench_harness.contracts.trialeval_diagnostics import (
    TrialEvalParticipantDiagnosticDictionaryV1,
)
from trialagentbench_harness.contracts.trialeval_methods import TrialEvalParticipantMethodDictionaryV1
from trialagentbench_harness.io.checksums import canonical_payload_sha256

NarrativeReportStateV1 = Literal["present", "blank", "absent"]


class _NarrativePacketModelV1(BaseModel):
    """Strict immutable base for masked packet artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class NarrativeNormalizationBatchConfigV1(_NarrativePacketModelV1):
    """Immutable source and provider configuration for a resumable normalization batch."""

    schema_id: Literal["trialagentbench.narrative_normalization_batch_config/v1"] = (
        "trialagentbench.narrative_normalization_batch_config/v1"
    )
    packet_set_manifest_sha256: str = Field(..., min_length=64, max_length=64)
    provider: Literal["openai", "openai_responses", "openrouter"]
    openrouter_provider: str | None = None
    normalizer_model: str = Field(..., min_length=1)
    decoding_seed: int | None = Field(default=None, ge=0)
    temperature: float = Field(..., ge=0.0, le=2.0)
    send_temperature: bool
    max_tokens: int = Field(..., ge=256)
    timeout_seconds: float = Field(..., gt=0.0)
    repeats: int = Field(..., ge=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_identity(self) -> NarrativeNormalizationBatchConfigV1:
        """Require coherent routing and a valid optional checksum."""

        if (self.provider == "openrouter") != (self.openrouter_provider is not None):
            raise ValueError("OpenRouter batches require one upstream provider pin; direct batches forbid it.")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("Narrative normalization batch-config checksum mismatch.")
        return self

    def with_checksum(self) -> NarrativeNormalizationBatchConfigV1:
        """Return this configuration with its canonical checksum."""

        return self.model_copy(
            update={"checksum": canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))}
        )


class NarrativeNormalizationBatchRecordV1(_NarrativePacketModelV1):
    """One immutable normalization result in a completed batch."""

    blinded_identity: str = Field(..., pattern=r"^[A-Za-z0-9._-]+$")
    qualification_unit_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]+$")
    assignment_id: str = Field(..., min_length=1)
    repeat_index: int = Field(..., ge=1)
    result_file: str = Field(..., min_length=1)
    result_sha256: str = Field(..., min_length=64, max_length=64)
    status: Literal["complete", "abstain"]


class NarrativeNormalizationBatchManifestV1(_NarrativePacketModelV1):
    """Complete denominator and immutable record index for one normalization batch."""

    schema_id: Literal["trialagentbench.narrative_normalization_batch/v1"] = (
        "trialagentbench.narrative_normalization_batch/v1"
    )
    config_checksum: str = Field(..., min_length=64, max_length=64)
    packet_count: int = Field(..., ge=1)
    repeat_count: int = Field(..., ge=1)
    result_count: int = Field(..., ge=1)
    complete_count: int = Field(..., ge=0)
    abstain_count: int = Field(..., ge=0)
    records: tuple[NarrativeNormalizationBatchRecordV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_denominator(self) -> NarrativeNormalizationBatchManifestV1:
        """Require exact counts, canonical order, stable units, and checksum integrity."""

        ordered = tuple(sorted(self.records, key=lambda row: (row.blinded_identity, row.repeat_index)))
        keys = tuple((row.blinded_identity, row.repeat_index) for row in self.records)
        if self.records != ordered or len(keys) != len(set(keys)):
            raise ValueError("Normalization batch records must be unique and canonically ordered.")
        if self.result_count != self.packet_count * self.repeat_count or self.result_count != len(self.records):
            raise ValueError("Normalization batch result denominator is incomplete.")
        if self.complete_count + self.abstain_count != self.result_count:
            raise ValueError("Normalization batch status counts do not cover every result.")
        units_by_packet: dict[str, str | None] = {}
        for record in self.records:
            existing = units_by_packet.setdefault(record.blinded_identity, record.qualification_unit_id)
            if existing != record.qualification_unit_id:
                raise ValueError("Normalization repeats disagree on qualification unit identity.")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("Narrative normalization batch-manifest checksum mismatch.")
        return self

    def with_checksum(self) -> NarrativeNormalizationBatchManifestV1:
        """Return this manifest with its canonical checksum."""

        return self.model_copy(
            update={"checksum": canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))}
        )


class DirectAssessmentBatchConfigV1(_NarrativePacketModelV1):
    """Immutable source and provider configuration for semantic report assessment."""

    schema_id: Literal["trialagentbench.direct_assessment_batch_config/v1"] = (
        "trialagentbench.direct_assessment_batch_config/v1"
    )
    packet_set_manifest_sha256: str = Field(..., min_length=64, max_length=64)
    provider: Literal["openai", "openai_responses", "openrouter"]
    openrouter_provider: str | None = None
    judge_model: str = Field(..., min_length=1)
    decoding_seed: int | None = Field(default=None, ge=0)
    temperature: float = Field(..., ge=0.0, le=2.0)
    send_temperature: bool
    max_tokens: int = Field(..., ge=256)
    timeout_seconds: float = Field(..., gt=0.0)
    repeats: int = Field(..., ge=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_identity(self) -> DirectAssessmentBatchConfigV1:
        """Require coherent routing and a valid optional checksum."""

        if (self.provider == "openrouter") != (self.openrouter_provider is not None):
            raise ValueError("OpenRouter batches require one upstream provider pin; direct batches forbid it.")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("Direct-assessment batch-config checksum mismatch.")
        return self

    def with_checksum(self) -> DirectAssessmentBatchConfigV1:
        """Return this configuration with its canonical checksum."""

        return self.model_copy(
            update={"checksum": canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))}
        )


class DirectAssessmentBatchRecordV1(_NarrativePacketModelV1):
    """One immutable semantic-assessment result in a completed batch."""

    blinded_identity: str = Field(..., pattern=r"^[A-Za-z0-9._-]+$")
    qualification_unit_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]+$")
    assignment_id: str = Field(..., min_length=1)
    repeat_index: int = Field(..., ge=1)
    result_file: str = Field(..., min_length=1)
    result_sha256: str = Field(..., min_length=64, max_length=64)
    status: Literal["completed", "invalid_response"]


class DirectAssessmentBatchManifestV1(_NarrativePacketModelV1):
    """Complete denominator and immutable index for semantic report assessment."""

    schema_id: Literal["trialagentbench.direct_assessment_batch/v1"] = "trialagentbench.direct_assessment_batch/v1"
    config_checksum: str = Field(..., min_length=64, max_length=64)
    packet_count: int = Field(..., ge=1)
    repeat_count: int = Field(..., ge=1)
    result_count: int = Field(..., ge=1)
    completed_count: int = Field(..., ge=0)
    invalid_response_count: int = Field(..., ge=0)
    records: tuple[DirectAssessmentBatchRecordV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_denominator(self) -> DirectAssessmentBatchManifestV1:
        """Require exact counts, canonical order, stable units, and checksum integrity."""

        ordered = tuple(sorted(self.records, key=lambda row: (row.blinded_identity, row.repeat_index)))
        keys = tuple((row.blinded_identity, row.repeat_index) for row in self.records)
        if self.records != ordered or len(keys) != len(set(keys)):
            raise ValueError("Direct-assessment batch records must be unique and canonically ordered.")
        if self.result_count != self.packet_count * self.repeat_count or self.result_count != len(self.records):
            raise ValueError("Direct-assessment batch result denominator is incomplete.")
        if self.completed_count + self.invalid_response_count != self.result_count:
            raise ValueError("Direct-assessment status counts do not cover every result.")
        units_by_packet: dict[str, str | None] = {}
        for record in self.records:
            existing = units_by_packet.setdefault(record.blinded_identity, record.qualification_unit_id)
            if existing != record.qualification_unit_id:
                raise ValueError("Direct-assessment repeats disagree on qualification unit identity.")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("Direct-assessment batch-manifest checksum mismatch.")
        return self

    def with_checksum(self) -> DirectAssessmentBatchManifestV1:
        """Return this manifest with its canonical checksum."""

        return self.model_copy(
            update={"checksum": canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))}
        )


class NarrativeParticipantContextV1(_NarrativePacketModelV1):
    """Exact participant task and output contracts available during normalization."""

    schema_id: Literal["trialagentbench.trialeval_narrative_participant_context/v1"] = (
        "trialagentbench.trialeval_narrative_participant_context/v1"
    )
    task_id: str = Field(..., min_length=1)
    task_contract: dict[str, JsonValue] = Field(..., min_length=1)
    participant_submission_contract: dict[str, JsonValue] = Field(..., min_length=1)
    participant_diagnostic_dictionary: TrialEvalParticipantDiagnosticDictionaryV1
    participant_method_dictionary: TrialEvalParticipantMethodDictionaryV1
    canonical_submission_schema: dict[str, JsonValue] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_identity(self) -> NarrativeParticipantContextV1:
        """Bind both participant contracts to the packet task identity."""

        if self.task_contract.get("task_id") != self.task_id:
            raise ValueError("Narrative participant task contract does not match task_id.")
        if self.participant_submission_contract.get("task_id") != self.task_id:
            raise ValueError("Narrative participant submission contract does not match task_id.")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("Narrative participant-context checksum mismatch.")
        return self

    def with_checksum(self) -> NarrativeParticipantContextV1:
        """Return the participant context with its canonical checksum assigned."""

        return self.model_copy(
            update={"checksum": canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))}
        )


class NarrativePacketManifestV1(_NarrativePacketModelV1):
    """Blinded identity and source binding for one manual packet."""

    schema_id: Literal["trialagentbench.trialeval_narrative_packet/v1"] = (
        "trialagentbench.trialeval_narrative_packet/v1"
    )
    blinded_identity: str = Field(..., pattern=r"^[A-Za-z0-9._-]+$")
    participant_task_id: str = Field(..., min_length=1)
    assignment_id: str = Field(..., min_length=1)
    report_state: NarrativeReportStateV1
    report_file: str = "frozen_report.txt"
    report_sha256: str = Field(..., min_length=64, max_length=64)
    participant_context_file: str = "participant_context.json"
    participant_context_sha256: str = Field(..., min_length=64, max_length=64)
    transcription_template_file: str = "transcription_template.json"


class NarrativePacketIndexRowV1(_NarrativePacketModelV1):
    """One blinded packet identity and immutable report binding."""

    blinded_identity: str = Field(..., pattern=r"^[A-Za-z0-9._-]+$")
    qualification_unit_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]+$")
    packet_manifest_sha256: str = Field(..., min_length=64, max_length=64)
    report_sha256: str = Field(..., min_length=64, max_length=64)


def _validate_packet_rows(
    packets: tuple[NarrativePacketIndexRowV1, ...],
    *,
    require_qualification_units: bool,
) -> None:
    identities = tuple(row.blinded_identity for row in packets)
    if identities != tuple(sorted(identities)) or len(identities) != len(set(identities)):
        raise ValueError("Narrative packet identities must be unique and canonically ordered.")
    unit_ids = tuple(row.qualification_unit_id for row in packets)
    if require_qualification_units:
        if any(unit_id is None for unit_id in unit_ids) or len(unit_ids) != len(set(unit_ids)):
            raise ValueError("Qualification packets require one unique sampled unit identity per packet.")
    elif any(unit_id is not None for unit_id in unit_ids):
        raise ValueError("Run-level narrative packets cannot declare qualification unit identities.")


class NarrativePacketSetManifestV1(_NarrativePacketModelV1):
    """Source and denominator binding for one completed run's packet export."""

    schema_id: Literal["trialagentbench.trialeval_narrative_packet_set/v1"] = (
        "trialagentbench.trialeval_narrative_packet_set/v1"
    )
    schedule_sha256: str = Field(..., min_length=64, max_length=64)
    run_identity_sha256: str = Field(..., min_length=64, max_length=64)
    participant_release_sha256: str = Field(..., min_length=64, max_length=64)
    source_files_sha256: dict[str, str]
    packets: tuple[NarrativePacketIndexRowV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_identity(self) -> NarrativePacketSetManifestV1:
        """Reject duplicate, unordered, qualification-bound, or drifting packet sets."""

        _validate_packet_rows(self.packets, require_qualification_units=False)
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("Narrative packet-set checksum mismatch.")
        return self

    def with_checksum(self) -> NarrativePacketSetManifestV1:
        """Return this manifest with its canonical checksum assigned."""

        return self.model_copy(
            update={"checksum": canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))}
        )


class NarrativeQualificationPacketSetManifestV1(_NarrativePacketModelV1):
    """Cross-run source binding for one frozen probability-sampled packet set."""

    schema_id: Literal["trialagentbench.trialeval_narrative_qualification_packet_set/v1"] = (
        "trialagentbench.trialeval_narrative_qualification_packet_set/v1"
    )
    sample_checksum: str = Field(..., min_length=64, max_length=64)
    schedule_sha256: str = Field(..., min_length=64, max_length=64)
    participant_release_sha256: str = Field(..., min_length=64, max_length=64)
    source_run_identity_sha256s: tuple[str, ...] = Field(..., min_length=1)
    source_files_sha256: dict[str, str]
    packets: tuple[NarrativePacketIndexRowV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_identity(self) -> NarrativeQualificationPacketSetManifestV1:
        """Require exact sampled-unit coverage, source runs, and checksum integrity."""

        if self.source_run_identity_sha256s != tuple(sorted(set(self.source_run_identity_sha256s))):
            raise ValueError("Qualification packet source runs must be unique and canonically ordered.")
        _validate_packet_rows(self.packets, require_qualification_units=True)
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("Narrative qualification packet-set checksum mismatch.")
        return self

    def with_checksum(self) -> NarrativeQualificationPacketSetManifestV1:
        """Return this manifest with its canonical checksum assigned."""

        return self.model_copy(
            update={"checksum": canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))}
        )


def manual_transcription_template_v1(
    *,
    assignment_id: str,
    report_sha256: str,
    report_state: NarrativeReportStateV1,
) -> dict[str, object]:
    """Return an intentionally incomplete independent-transcription template."""

    unavailable = report_state != "present"
    return {
        "schema_id": "trialagentbench.trialeval_narrative_transcription/v1",
        "assignment_id": assignment_id,
        "report_sha256": report_sha256,
        "source": "manual_masked",
        "source_identity": "",
        "transcriber_identities": [],
        "transcription_disposition": None,
        "blinded_to_model_identity": True,
        "blinded_to_evaluator_reference": True,
        "importer_prompt_sha256": None,
        "importer_schema_sha256": None,
        "importer_response_sha256": None,
        "status": "abstain" if unavailable else None,
        "submission": None,
        "claims": [],
        "abstention_reason": (
            "No narrative report was submitted."
            if report_state == "absent"
            else "The submitted narrative report was blank." if report_state == "blank" else None
        ),
    }


__all__ = [
    "DirectAssessmentBatchConfigV1",
    "DirectAssessmentBatchManifestV1",
    "DirectAssessmentBatchRecordV1",
    "NarrativeNormalizationBatchConfigV1",
    "NarrativeNormalizationBatchManifestV1",
    "NarrativeNormalizationBatchRecordV1",
    "NarrativePacketIndexRowV1",
    "NarrativePacketManifestV1",
    "NarrativePacketSetManifestV1",
    "NarrativeParticipantContextV1",
    "NarrativeQualificationPacketSetManifestV1",
    "NarrativeReportStateV1",
    "manual_transcription_template_v1",
]
