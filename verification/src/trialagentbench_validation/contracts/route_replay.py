"""Compact independent evidence consumed by TrialEval release construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.io.checksums import sha256_file
from trialagentbench_validation.io.json import write_json_model
from trialagentbench_validation.trialeval.references.numeric import (
    PublicEvidenceNumericReferenceReportV1,
    recompute_trialeval_public_numeric_reference_v1,
)


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PublicRouteReplayRecordV1(_ContractModel):
    """Measured disagreement for one independently reproduced route."""

    route_reference_id: str = Field(min_length=1)
    route_reference_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_bundle_id: str = Field(min_length=1)
    input_bundle_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    method_composition_checksum: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    max_abs_difference: float = Field(ge=0)


class PublicRouteReplayEvidenceV1(_ContractModel):
    """Checksum-bound complete public replay evidence for credit-eligible routes."""

    schema_id: Literal["trialagentbench.public_route_replay_evidence/v1"] = (
        "trialagentbench.public_route_replay_evidence/v1"
    )
    evaluator_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participant_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: tuple[PublicRouteReplayRecordV1, ...] = Field(min_length=1)
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _canonical_records_and_checksum(self) -> PublicRouteReplayEvidenceV1:
        records = tuple(sorted(self.records, key=lambda row: row.route_reference_id))
        ids = tuple(row.route_reference_id for row in records)
        if len(ids) != len(set(ids)):
            raise ValueError(
                "public route replay records must have unique route references"
            )
        object.__setattr__(self, "records", records)
        payload = self.model_dump(mode="json", exclude={"checksum"})
        checksum = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        if self.checksum is not None and self.checksum != checksum:
            raise ValueError("public route replay evidence checksum mismatch")
        object.__setattr__(self, "checksum", checksum)
        return self


def compact_public_route_replay(
    *,
    report: PublicEvidenceNumericReferenceReportV1,
    evaluator_zip: Path,
    participant_zip: Path,
) -> PublicRouteReplayEvidenceV1:
    """Reduce a passing full replay to the release-bearing evidence contract."""

    if report.status != "pass":
        raise ValueError(f"public numeric replay failed: findings={report.findings!r}")
    records = []
    for check in report.checks:
        if check.outcome != "matched":
            raise ValueError(
                f"public route was not reproduced: {check.route_reference_id}"
            )
        differences = tuple(
            value
            for value in (
                check.abs_diff,
                check.lower_abs_diff,
                check.upper_abs_diff,
                check.standard_error_abs_diff,
                check.vector_max_abs_diff,
            )
            if value is not None
        )
        if not differences:
            raise ValueError(
                f"matched public route has no measured disagreement: "
                f"{check.route_reference_id}"
            )
        records.append(
            PublicRouteReplayRecordV1(
                route_reference_id=check.route_reference_id,
                route_reference_checksum=check.route_reference_checksum,
                input_bundle_id=check.input_bundle_id,
                input_bundle_checksum=check.input_bundle_checksum,
                method_composition_checksum=check.method_composition_checksum,
                max_abs_difference=max(differences),
            )
        )
    return PublicRouteReplayEvidenceV1(
        evaluator_sha256=sha256_file(evaluator_zip),
        participant_sha256=sha256_file(participant_zip),
        records=tuple(records),
    )


def write_public_route_replay_evidence(
    *,
    evaluator_zip: Path,
    participant_zip: Path,
    output_path: Path,
    workers: int,
) -> PublicRouteReplayEvidenceV1:
    """Independently replay public reference and write compact release evidence."""

    report = recompute_trialeval_public_numeric_reference_v1(
        evaluator_zip=evaluator_zip,
        public_zip=participant_zip,
        workers=workers,
    )
    evidence = compact_public_route_replay(
        report=report,
        evaluator_zip=evaluator_zip,
        participant_zip=participant_zip,
    )
    write_json_model(output_path, evidence)
    return evidence


__all__ = [
    "PublicRouteReplayEvidenceV1",
    "PublicRouteReplayRecordV1",
    "compact_public_route_replay",
    "write_public_route_replay_evidence",
]
