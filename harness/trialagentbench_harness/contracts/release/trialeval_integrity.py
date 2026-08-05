"""Participant-visible TrialEval data-integrity contract."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["TrialEvalPublicIntegrityPolicyV1"]


class TrialEvalPublicIntegrityPolicyV1(BaseModel):
    """Declare one uniquely repairable exact-row transport duplication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialeval.c5_integrity_policy/v1"] = (
        "trialagentbench.trialeval.c5_integrity_policy/v1"
    )
    task_id: str = Field(min_length=1)
    condition_id: Literal["exact_transport_row_duplication_v1"] = "exact_transport_row_duplication_v1"
    affected_domain: str = Field(min_length=1)
    compound_key_fields: tuple[str, ...] = Field(min_length=1)
    legitimate_repeat_semantics: str = Field(min_length=1)
    repair_contract_id: Literal["exact_transport_row_duplication_repair_v1"] = (
        "exact_transport_row_duplication_repair_v1"
    )
    repair_action: Literal["remove_one_exact_duplicate_copy"] = "remove_one_exact_duplicate_copy"
    canonical_typed_scalar_encoding_id: Literal["canonical_typed_scalar_v1"] = "canonical_typed_scalar_v1"
    canonical_compound_row_key_encoding_id: Literal["canonical_compound_row_key_v1"] = "canonical_compound_row_key_v1"
    canonical_typed_row_payload_encoding_id: Literal["canonical_typed_row_payload_v1"] = (
        "canonical_typed_row_payload_v1"
    )
    canonical_domain_content_checksum_id: Literal["canonical_domain_content_sha256_v1"] = (
        "canonical_domain_content_sha256_v1"
    )
    selected_duplicate_keys_visible: Literal[False] = False
    expected_duplicate_count_visible: Literal[False] = False
    clean_parent_checksum_visible: Literal[False] = False

    @model_validator(mode="after")
    def _canonical_fields(self) -> Self:
        if len(set(self.compound_key_fields)) != len(self.compound_key_fields):
            raise ValueError("C5 compound-key fields must be unique")
        return self
