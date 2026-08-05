"""Project raw TrialEval route witnesses into the canonical grader contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.contracts.release.trialeval_runtime_surface import (
    TrialEvalSemanticSubmissionContractV1,
)
from trialagentbench_harness.contracts.scoring.assumption_evidence import (
    read_assumption_evidence_domains,
)
from trialagentbench_harness.contracts.submission import TrialEvalSubmissionV1
from trialagentbench_harness.grading import CanonicalSubmissionV1
from trialagentbench_harness.trialeval.canonicalize import (
    canonicalize_trialeval_submission_v1,
)
from trialagentbench_harness.trialeval.diagnostic_evidence import (
    validated_diagnostic_ids_v1,
)
from trialagentbench_harness.trialeval.schema import BenchmarkItem


class RawTrialEvalRouteWitnessV1(BaseModel):
    """One participant-wire witness and its public-input custody record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialeval_raw_route_witness/v1"]
    release_id: str = Field(min_length=1)
    witness_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    primary_evidence_class: Literal[
        "prescribed_execution",
        "empirical_diagnosis",
        "design_or_provenance_reasoning",
        "observed_failure_recovery",
        "evidence_insufficient",
    ]
    repair_required: bool
    fixed_question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    route_signature_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participant_input_checksums: dict[str, str] = Field(min_length=1)
    raw_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    submission: TrialEvalSubmissionV1

    @model_validator(mode="after")
    def _identity_is_bound(self) -> RawTrialEvalRouteWitnessV1:
        if self.submission.task_id != self.item_id:
            raise ValueError("raw route-witness item identity disagrees with its submission")
        if self.repair_required != (self.context_tier == "C5"):
            raise ValueError("raw route-witness repair duty must match C5 exactly")
        return self


class CanonicalTrialEvalRouteWitnessV1(BaseModel):
    """One projected canonical submission bound to its source witness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialeval_route_witness/v1"] = "trialagentbench.trialeval_route_witness/v1"
    witness_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    submission: CanonicalSubmissionV1


def _canonical_sha256(payload: object) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-root", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--witnesses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate public custody and project every raw route witness."""

    args = _parser().parse_args(argv)
    raw_witnesses = tuple(
        RawTrialEvalRouteWitnessV1.model_validate_json(line)
        for line in args.witnesses.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not raw_witnesses:
        raise ValueError("Raw TrialEval route-witness census cannot be empty")
    witness_ids = tuple(witness.witness_id for witness in raw_witnesses)
    if len(witness_ids) != len(set(witness_ids)):
        raise ValueError("Raw TrialEval route-witness census contains duplicate witness IDs")
    assumption_evidence = read_assumption_evidence_domains(release_root=args.evaluator_root)
    projected: list[CanonicalTrialEvalRouteWitnessV1] = []
    for witness in raw_witnesses:
        item_root = args.participant_root / "items" / witness.item_id
        task_path = item_root / "task.json"
        contract_path = item_root / "submission_contract.json"
        if hashlib.sha256(task_path.read_bytes()).hexdigest() != witness.fixed_question_sha256:
            raise ValueError(f"raw route witness has a stale fixed question: {witness.witness_id}")
        for relative_path, expected_checksum in witness.participant_input_checksums.items():
            observed = hashlib.sha256((item_root / relative_path).read_bytes()).hexdigest()
            if observed != expected_checksum:
                raise ValueError(
                    f"raw route witness public input checksum mismatch: {witness.witness_id}/{relative_path}"
                )
        raw_payload = witness.submission.model_dump(mode="json", exclude_none=True)
        if _canonical_sha256(raw_payload) != witness.raw_response_sha256:
            raise ValueError(f"raw route witness response checksum mismatch: {witness.witness_id}")
        contract = TrialEvalSemanticSubmissionContractV1.model_validate_json(contract_path.read_text(encoding="utf-8"))
        task = json.loads(task_path.read_text(encoding="utf-8"))
        item = BenchmarkItem(
            item_id=witness.item_id,
            task_id=witness.item_id,
            trial_name=witness.item_id,
            design_tier="undisclosed",
            design_subtype=str(task.get("design_subtype", "undisclosed")),
            assumption_tier="undisclosed",
            context_tier=witness.context_tier,
            visible_dir=item_root,
            data_dir=item_root / "data",
            task=task,
            submission_contract=contract.model_dump(mode="json"),
        )
        evidence = assumption_evidence.get(witness.item_id)
        if evidence is None:
            raise ValueError(f"raw route witness lacks assumption evidence: {witness.witness_id}")
        diagnostic_ids = validated_diagnostic_ids_v1(
            submission=witness.submission,
            item=item,
            assumption_evidence=evidence,
        )
        canonical = canonicalize_trialeval_submission_v1(
            witness.submission,
            validated_diagnostic_ids=frozenset(diagnostic_ids),
        )
        if _canonical_sha256(canonical.primary.model_dump(mode="json")) != witness.route_signature_sha256:
            raise ValueError(f"raw route witness projected to another route: {witness.witness_id}")
        projected.append(
            CanonicalTrialEvalRouteWitnessV1(
                witness_id=witness.witness_id,
                item_id=witness.item_id,
                route_id=witness.route_id,
                context_tier=witness.context_tier,
                submission=canonical,
            )
        )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite projected route witnesses: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(
                row.model_dump(mode="json", exclude_none=True),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in projected
        ),
        encoding="utf-8",
    )
    return 0


__all__ = [
    "CanonicalTrialEvalRouteWitnessV1",
    "RawTrialEvalRouteWitnessV1",
    "main",
]
