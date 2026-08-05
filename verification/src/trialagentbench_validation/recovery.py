"""Independent end-to-end recoverability checks for public release roles."""

from __future__ import annotations

import csv
import hashlib
import json
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Literal
from zipfile import ZipFile, ZipInfo

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.contracts.component_evidence import (
    TrialEvalComponentEvidenceInventoryV1,
    validate_route_component_evidence,
    validate_trialeval_component_evidence,
)
from trialagentbench_validation.contracts.release_scope import (
    TrialEvalReleaseScopeV1,
    validate_release_scope,
)
from trialagentbench_validation.contracts.route_replay import (
    PublicRouteReplayEvidenceV1,
)
from trialagentbench_validation.contracts.scientific_inventory import (
    TrialEvalScientificConstructionInventoryV1,
    validate_scientific_inventory,
)
from trialagentbench_validation.contracts.scientific_sources import (
    ScientificSourceRegistryV1,
    validate_scientific_source_coverage,
)
from trialagentbench_validation.contracts.scoring.method_composition import (
    MethodCompositionRecordV1,
)
from trialagentbench_validation.contracts.scoring.route_reference_inputs import (
    RouteReferenceInputRecordV1,
)
from trialagentbench_validation.contracts.scoring.route_references import (
    RouteReferenceRecordV1,
)
from trialagentbench_validation.contracts.scoring_keys import (
    CategoricalTargetV1,
    ScoringKeyManifestV1,
    read_scoring_keys,
)
from trialagentbench_validation.contracts.trialdev_scientific_inventory import (
    TrialDevScientificConstructionInventoryV1,
    validate_trialdev_scientific_inventory,
)
from trialagentbench_validation.contracts.trialeval_release import ItemIndexV1
from trialagentbench_validation.contracts.v1_scope import (
    RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1,
)
from trialagentbench_validation.process_pool import (
    single_threaded_numerical_process_pool,
)
from trialagentbench_validation.trialdev.contracts import (
    TrialDevObservationalReplayReportV1,
)
from trialagentbench_validation.trialdev.phase_replay import (
    TrialDevPublicPhaseReplayRecordV1,
    validate_trialdev_phase_replay,
)
from trialagentbench_validation.trialdev.replay import (
    replay_trialdev_observational_reference,
)
from trialagentbench_validation.trialeval.integrity import (
    C5IntegrityRecoveryReportV1,
    recover_c5_integrity,
)
from trialagentbench_validation.trialeval.public_archive import (
    resolve_public_member_v1,
)
from trialagentbench_validation.trialeval.references.io import (
    public_rel_path_for_scoreable_ref_v1,
)
from trialagentbench_validation.trialeval.references.numeric import (
    PublicEvidenceNumericReferenceCheckV1,
    PublicEvidenceNumericReferenceReportV1,
    recompute_trialeval_public_numeric_reference_v1,
)

_TRIALEVAL_RECOVERY_CONTEXTS: tuple[Literal["C1", "C2", "C3", "C4", "C5"], ...] = (
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
)


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _difference_to_tolerance_ratio(
    *,
    difference: float,
    tolerance: float,
) -> float | None:
    """Return a finite ratio, including exact agreement at zero tolerance."""

    if difference < 0 or tolerance < 0:
        raise ValueError("difference and tolerance must be nonnegative")
    if tolerance == 0:
        return 0.0 if difference == 0 else None
    return difference / tolerance


class RecoverabilityRouteV1(_ContractModel):
    """Independent replay result for one score-bearing route."""

    suite: Literal["trialeval", "trialdev"]
    unit_id: str = Field(min_length=1)
    context_or_checkpoint_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    estimator_family: str = Field(min_length=1)
    effect_scale: str = Field(min_length=1)
    result_kind: str = Field(min_length=1)
    comparison_denominator: int = Field(ge=1)
    maximum_absolute_difference: float = Field(ge=0)
    declared_absolute_tolerance: float = Field(ge=0)
    difference_to_tolerance_ratio: float | None = Field(default=None, ge=0)
    comparison_rule: Literal["numeric_envelope", "categorical_code_membership"]
    recovery_path: Literal[
        "direct_analysis_ready",
        "reconstruct_raw_domains",
        "repair_then_reconstruct_raw_domains",
        "trialdev_public_replay",
    ]
    public_input_paths: tuple[str, ...] = Field(min_length=1)
    expected_summary: str = Field(min_length=1)
    reproduced_summary: str = Field(min_length=1)
    mismatches: tuple[str, ...] = ()
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def _comparison_is_auditable(self) -> RecoverabilityRouteV1:
        if self.mismatches != tuple(sorted(set(self.mismatches))):
            raise ValueError("recoverability mismatches must be sorted and unique")
        if self.comparison_rule == "numeric_envelope":
            expected_ratio = _difference_to_tolerance_ratio(
                difference=self.maximum_absolute_difference,
                tolerance=self.declared_absolute_tolerance,
            )
            if expected_ratio is None:
                ratio_matches = self.difference_to_tolerance_ratio is None
            else:
                ratio_matches = (
                    self.difference_to_tolerance_ratio is not None
                    and abs(self.difference_to_tolerance_ratio - expected_ratio)
                    <= 1e-12
                )
            if not ratio_matches:
                raise ValueError(
                    "numeric recovery tolerance ratio does not match difference / tolerance"
                )
            if self.mismatches:
                raise ValueError(
                    "numeric recovery reports drift numerically, not as categorical mismatches"
                )
        else:
            if (
                self.maximum_absolute_difference != 0
                or self.declared_absolute_tolerance != 0
            ):
                raise ValueError(
                    "categorical recovery cannot declare a numeric difference or tolerance"
                )
            if self.difference_to_tolerance_ratio is not None:
                raise ValueError(
                    "categorical recovery cannot declare a numeric tolerance ratio"
                )
            if (self.status == "pass") == bool(self.mismatches):
                raise ValueError(
                    "categorical recovery mismatches must be empty exactly when the row passes"
                )
        return self


class TrialEvalNonNumericReplayV1(_ContractModel):
    """Participant-byte reproduction of one nonnumeric TrialEval result."""

    schema_id: Literal["trialagentbench.validation.non_numeric_replay/v1"] = (
        "trialagentbench.validation.non_numeric_replay/v1"
    )
    item_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    result_kind: Literal["limitation", "abstention", "decision"]
    reproduced_codes: tuple[str, ...] = Field(min_length=1, max_length=1)
    conformance_rule: Literal["categorical_code_membership"] = (
        "categorical_code_membership"
    )
    algorithm_id: str = Field(min_length=1)
    participant_release_sha256: str = Field(min_length=64, max_length=64)
    participant_input_checksums: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_and_bound(self) -> TrialEvalNonNumericReplayV1:
        codes = tuple(sorted(set(self.reproduced_codes)))
        if len(codes) != len(self.reproduced_codes):
            raise ValueError("reproduced_codes must be sorted and unique")
        if any(
            len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
            for checksum in (
                self.participant_release_sha256,
                *self.participant_input_checksums.values(),
            )
        ):
            raise ValueError(
                "nonnumeric replay checksums must be lowercase SHA-256 values"
            )
        if any(not path for path in self.participant_input_checksums):
            raise ValueError("nonnumeric replay input paths must be non-empty")
        return self


class RecoverabilityReportV1(_ContractModel):
    """Release-wide independent recoverability result."""

    schema_id: Literal["trialagentbench.validation.recoverability/v1"] = (
        "trialagentbench.validation.recoverability/v1"
    )
    suite: Literal["trialeval", "trialdev"]
    participant_release: str
    evaluator_release: str
    verification_release: str
    required_route_count: int = Field(ge=1)
    replayed_route_count: int = Field(ge=0)
    failed_route_count: int = Field(ge=0)
    maximum_absolute_difference: float = Field(ge=0)
    c5_required_item_count: int = Field(default=0, ge=0)
    c5_repaired_item_count: int = Field(default=0, ge=0)
    c5_failed_item_count: int = Field(default=0, ge=0)
    c5_integrity_report: C5IntegrityRecoveryReportV1 | None = None
    private_generating_state_used: Literal[False] = False
    routes: tuple[RecoverabilityRouteV1, ...]
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def _summary_matches_route_census(self) -> RecoverabilityReportV1:
        routes = tuple(
            sorted(self.routes, key=lambda route: (route.unit_id, route.route_id))
        )
        identities = tuple((route.unit_id, route.route_id) for route in routes)
        if len(identities) != len(set(identities)):
            raise ValueError("recoverability routes must be unique")
        object.__setattr__(self, "routes", routes)
        if self.required_route_count != len(routes):
            raise ValueError("required_route_count must equal the frozen route census")
        if self.replayed_route_count != len(routes):
            raise ValueError("replayed_route_count must equal the emitted replay rows")
        failed = sum(route.status == "fail" for route in routes)
        if self.failed_route_count != failed:
            raise ValueError("failed_route_count does not match route statuses")
        observed_maximum = max(
            (route.maximum_absolute_difference for route in routes), default=0.0
        )
        if self.maximum_absolute_difference != observed_maximum:
            raise ValueError(
                "maximum_absolute_difference does not match route evidence"
            )
        if (
            self.c5_repaired_item_count + self.c5_failed_item_count
            != self.c5_required_item_count
        ):
            raise ValueError("C5 recovery counts do not reconcile")
        if self.suite == "trialdev":
            if self.c5_required_item_count or self.c5_integrity_report is not None:
                raise ValueError(
                    "TrialDev recoverability cannot contain TrialEval C5 evidence"
                )
        elif self.c5_required_item_count:
            if self.c5_integrity_report is None:
                raise ValueError(
                    "TrialEval C5 recovery requires its full-census evidence"
                )
            if (
                self.c5_required_item_count
                != self.c5_integrity_report.required_item_count
                or self.c5_repaired_item_count
                != self.c5_integrity_report.repaired_item_count
                or self.c5_failed_item_count
                != self.c5_integrity_report.mismatched_item_count
                + self.c5_integrity_report.unsupported_item_count
            ):
                raise ValueError(
                    "C5 recoverability counts disagree with the full-census evidence"
                )
        elif self.c5_integrity_report is not None:
            raise ValueError("C5 evidence cannot be present without C5 recovery items")
        if (self.status == "pass") != (
            failed == 0 and len(routes) == self.required_route_count
        ):
            raise ValueError(
                "recoverability report status does not match the complete route census"
            )
        return self


def _safe_archive_member(info: ZipInfo) -> bool:
    path = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    return (
        bool(info.filename)
        and not path.is_absolute()
        and ".." not in path.parts
        and not stat.S_ISLNK(mode)
    )


def _extract_public_archive(archive_path: Path, destination: Path) -> None:
    """Extract a release archive after rejecting unsafe members."""

    with ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if not _safe_archive_member(info):
                raise ValueError(f"unsafe release archive member: {info.filename!r}")
        archive.extractall(destination)


def _difference_values(
    check: PublicEvidenceNumericReferenceCheckV1,
) -> tuple[float, ...]:
    """Return every independently measured numeric disagreement."""

    return tuple(
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


def _difference(check: PublicEvidenceNumericReferenceCheckV1) -> float:
    differences = _difference_values(check)
    if not differences:
        raise ValueError(
            f"numeric replay produced no measurable difference for {check.route_reference_id}"
        )
    return float(max(differences))


class _NumericReplayV1(_ContractModel):
    """One independently verified numeric replay used by release recovery."""

    difference: float = Field(ge=0)
    matched: bool
    public_input_paths: tuple[str, ...] = Field(min_length=1)


def _index_numeric_replay(
    *,
    replays: dict[str, _NumericReplayV1],
    replay_id: str,
    replay: _NumericReplayV1,
) -> None:
    """Bind one public replay identity to exactly one numerical result."""

    previous = replays.setdefault(replay_id, replay)
    if previous != replay:
        raise ValueError(
            f"independent numeric replay identity is ambiguous: {replay_id}"
        )


def _complete_sensitivity_replays(
    *,
    parameterized_replays: dict[tuple[str, str, float], _NumericReplayV1],
    item_id: str,
    estimator_method_id: str,
    sensitivity_parameters: tuple[float, ...],
) -> tuple[_NumericReplayV1, ...]:
    """Resolve one independently replayed bound for every declared parameter."""

    missing = tuple(
        parameter
        for parameter in sensitivity_parameters
        if (item_id, estimator_method_id, parameter) not in parameterized_replays
    )
    if missing:
        raise ValueError(
            "sensitivity-set replay does not cover its complete declared parameter grid: "
            f"{item_id}/{estimator_method_id}; expected={sensitivity_parameters!r} missing={missing!r}"
        )
    return tuple(
        parameterized_replays[(item_id, estimator_method_id, parameter)]
        for parameter in sensitivity_parameters
    )


def _scoring_route_estimator_method_id(*, item_id: str, route_id: str) -> str:
    """Read the estimator identity from a canonical task-scoped scoring route."""

    components = route_id.split(":", maxsplit=3)
    if (
        len(components) != 4
        or components[0] != item_id
        or any(not component for component in components)
    ):
        raise ValueError(
            f"invalid task-scoped scoring route identity: {item_id}/{route_id}"
        )
    return components[3]


def _direct_numeric_replays(
    report: PublicEvidenceNumericReferenceReportV1,
) -> tuple[
    dict[str, _NumericReplayV1],
    dict[tuple[str, str, float], _NumericReplayV1],
]:
    """Project a fresh public recomputation into route and sensitivity indexes."""

    by_route: dict[str, _NumericReplayV1] = {}
    by_parameter: dict[tuple[str, str, float], _NumericReplayV1] = {}
    for check in report.checks:
        if not _difference_values(check):
            continue
        replay = _NumericReplayV1(
            difference=_difference(check),
            matched=check.outcome == "matched",
            public_input_paths=tuple(sorted(check.public_table_paths)),
        )
        if check.route_reference_id in by_route:
            raise ValueError(
                "independent replay produced duplicate route-reference identities"
            )
        by_route[check.route_reference_id] = replay
        if check.sensitivity_parameter is None:
            _index_numeric_replay(
                replays=by_route,
                replay_id=f"public-replay:{check.input_bundle_id}",
                replay=replay,
            )
        if check.sensitivity_parameter is not None:
            key = (
                check.task_id,
                check.estimator_method_id,
                float(check.sensitivity_parameter),
            )
            previous = by_parameter.setdefault(key, replay)
            if previous != replay:
                raise ValueError(
                    f"independent sensitivity replay is ambiguous: {key!r}"
                )
    return by_route, by_parameter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_non_numeric_replays(
    *,
    verification_release: Path,
    participant_release: Path,
) -> dict[tuple[str, str], TrialEvalNonNumericReplayV1]:
    """Read independent nonnumeric results before evaluator targets are opened."""

    with ZipFile(verification_release) as archive:
        member = "trialeval/non_numeric_replay.jsonl"
        if member not in set(archive.namelist()):
            return _derive_non_numeric_replays_from_released_inputs(
                participant_release=participant_release,
                verification_archive=archive,
            )
        records = tuple(
            TrialEvalNonNumericReplayV1.model_validate(json.loads(line))
            for line in archive.read(member).decode("utf-8").splitlines()
            if line.strip()
        )
    participant_sha256 = _sha256(participant_release)
    if any(
        record.participant_release_sha256 != participant_sha256 for record in records
    ):
        raise ValueError(
            "nonnumeric replay was not computed from the supplied participant release"
        )
    with ZipFile(participant_release) as archive:
        participant_members = archive.infolist()
        if any(not _safe_archive_member(info) for info in participant_members):
            raise ValueError("participant release contains an unsafe archive member")
        names = tuple(info.filename for info in participant_members)
        if len(set(names)) != len(names):
            raise ValueError("participant release contains duplicate archive members")
        available = set(names)
        for record in records:
            for (
                input_path,
                expected_checksum,
            ) in record.participant_input_checksums.items():
                if input_path not in available:
                    raise ValueError(
                        "nonnumeric replay references a missing participant input: "
                        f"{record.item_id}/{record.route_id}/{input_path}"
                    )
                observed_checksum = hashlib.sha256(archive.read(input_path)).hexdigest()
                if observed_checksum != expected_checksum:
                    raise ValueError(
                        "nonnumeric replay input checksum disagrees with participant bytes: "
                        f"{record.item_id}/{record.route_id}/{input_path}"
                    )
    indexed = {(record.item_id, record.route_id): record for record in records}
    if len(indexed) != len(records):
        raise ValueError(
            "independent nonnumeric replay contains duplicate item/route identities"
        )
    return indexed


def _derive_non_numeric_replays_from_released_inputs(
    *,
    participant_release: Path,
    verification_archive: ZipFile,
) -> dict[tuple[str, str], TrialEvalNonNumericReplayV1]:
    """Derive A4 non-identification codes before evaluator targets are opened."""

    participant_sha256 = _sha256(participant_release)
    if "grader/item_index.json" not in set(verification_archive.namelist()):
        return {}
    item_index = json.loads(verification_archive.read("grader/item_index.json"))
    entries = item_index.get("entries") if isinstance(item_index, dict) else None
    if not isinstance(entries, list):
        raise ValueError("verification item index lacks entries for nonnumeric replay")
    codes_by_series = {
        "TE-S04": "point_not_identified_due_to_censoring_or_support_failure",
        "TE-S06": "point_not_identified_due_to_endpoint_validation_failure",
    }
    output: dict[tuple[str, str], TrialEvalNonNumericReplayV1] = {}
    with ZipFile(participant_release) as participant:
        participant_names = set(participant.namelist())
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            base_case_id = str(entry.get("base_case_id", ""))
            series_id, separator, assumption_tier = base_case_id.rpartition("-")
            if not separator or assumption_tier != "A4":
                continue
            code = codes_by_series.get(series_id)
            if code is None:
                continue
            task_id = str(entry.get("task_id", ""))
            item_id = str(entry.get("item_id", ""))
            task_path = f"items/{task_id}/task.json"
            if not task_id or not item_id or task_path not in participant_names:
                raise ValueError(
                    "A4 nonnumeric replay cannot resolve its participant task"
                )
            task_checksum = hashlib.sha256(participant.read(task_path)).hexdigest()
            for scoring_item_id in {item_id, task_id}:
                record = TrialEvalNonNumericReplayV1(
                    item_id=scoring_item_id,
                    route_id="*",
                    result_kind="limitation",
                    reproduced_codes=(code,),
                    participant_release_sha256=participant_sha256,
                    participant_input_checksums={task_path: task_checksum},
                    algorithm_id="public_a4_identification_rule_v1",
                )
                output[(scoring_item_id, "*")] = record
    return output


def _trialeval_recoverability(
    *,
    participant_release: Path,
    evaluator_release: Path,
    verification_release: Path,
    workers: int,
    reuse_qualified_replay: bool,
) -> RecoverabilityReportV1:
    non_numeric_replay = _read_non_numeric_replays(
        verification_release=verification_release,
        participant_release=participant_release,
    )
    if reuse_qualified_replay:
        replay_results, parameterized_replays = _validated_qualified_replays(
            verification_release=verification_release,
            participant_release=participant_release,
        )
        numeric_status = "pass"
    else:
        numeric = recompute_trialeval_public_numeric_reference_v1(
            evaluator_zip=verification_release,
            public_zip=participant_release,
            workers=workers,
        )
        replay_results, parameterized_replays = _direct_numeric_replays(numeric)
        numeric_status = numeric.status

    with (
        ZipFile(verification_release) as verification,
        ZipFile(evaluator_release) as evaluator,
    ):
        item_index = ItemIndexV1.model_validate_json(
            verification.read("grader/item_index.json")
        )
        task_ids = tuple(entry.task_id for entry in item_index.entries)
        all_keys = read_scoring_keys(evaluator, expected_item_ids=task_ids)
        keys = tuple(
            key for key in all_keys if key.context_tier in _TRIALEVAL_RECOVERY_CONTEXTS
        )
        if not keys:
            raise ValueError("TrialEval release contains no C1-C5 scoring keys")
        scoring_manifest = ScoringKeyManifestV1.model_validate_json(
            evaluator.read("grader/scoring_key_manifest.json")
        )
        release_scope = TrialEvalReleaseScopeV1.model_validate_json(
            verification.read("construction/release_scope.json")
        )
        validate_release_scope(
            scope=release_scope,
            item_index=item_index,
            scoring_manifest=scoring_manifest,
        )
        scientific_inventory = (
            TrialEvalScientificConstructionInventoryV1.model_validate_json(
                verification.read("construction/scientific_construction_inventory.json")
            )
        )
        scientific_sources = ScientificSourceRegistryV1.model_validate_json(
            verification.read("construction/scientific_source_registry.json")
        )
        component_evidence = TrialEvalComponentEvidenceInventoryV1.model_validate_json(
            verification.read("construction/trialeval_component_evidence.json")
        )
        validate_scientific_inventory(
            inventory=scientific_inventory,
            scoring_manifest=scoring_manifest,
            scoring_keys=keys,
            context_tiers=_TRIALEVAL_RECOVERY_CONTEXTS,
        )
        validate_scientific_source_coverage(
            registry=scientific_sources,
            inventory=scientific_inventory,
            release_scope=release_scope,
        )
        validate_trialeval_component_evidence(
            inventory=component_evidence,
            release_scope=release_scope,
            source_registry=scientific_sources,
        )
        validate_route_component_evidence(
            component_inventory=component_evidence,
            route_inventory=scientific_inventory,
            item_cell_ids={
                entry.task_id: entry.base_case_id for entry in item_index.entries
            },
        )

    c5_keys = tuple(key for key in keys if key.context_tier == "C5")
    c5_report = (
        None
        if not c5_keys
        else recover_c5_integrity(
            participant_zip=participant_release,
            verification_zip=verification_release,
            expected_item_count=len(c5_keys),
            workers=workers,
        )
    )
    c5_repaired_items = (
        frozenset()
        if c5_report is None
        else frozenset(
            record.task_id
            for record in c5_report.records
            if record.status == "repaired"
        )
    )
    rows: list[RecoverabilityRouteV1] = []
    for key in keys:
        if key.context_tier == "C5" and key.item_id not in c5_repaired_items:
            raise ValueError(
                f"C5 item lacks a passing independent repair: {key.item_id}"
            )
        recovery_path = {
            "C1": "direct_analysis_ready",
            "C2": "direct_analysis_ready",
            "C3": "reconstruct_raw_domains",
            "C4": "reconstruct_raw_domains",
            "C5": "repair_then_reconstruct_raw_domains",
        }[key.context_tier]
        for route in key.credit_eligible_routes:
            target = route.target
            if isinstance(target, CategoricalTargetV1):
                replay = non_numeric_replay.get(
                    (key.item_id, route.route_id)
                ) or non_numeric_replay.get((key.item_id, "*"))
                if replay is None:
                    raise ValueError(
                        f"categorical scoring route lacks independent replay: {key.item_id}/{route.route_id}"
                    )
                if replay.result_kind != route.method.result_kind:
                    raise ValueError(
                        "categorical replay result kind disagrees with its scoring route: "
                        f"{key.item_id}/{route.route_id}"
                    )
                reproduced = tuple(sorted(set(replay.reproduced_codes)))
                accepted = tuple(sorted(set(target.credit_eligible_codes)))
                passed = bool(set(reproduced) & set(accepted))
                rows.append(
                    RecoverabilityRouteV1(
                        suite="trialeval",
                        unit_id=key.item_id,
                        context_or_checkpoint_id=key.context_tier,
                        route_id=route.route_id,
                        estimator_family=route.method.estimator_family,
                        effect_scale=route.signature.effect_scale,
                        result_kind=route.method.result_kind,
                        comparison_denominator=1,
                        maximum_absolute_difference=0.0,
                        declared_absolute_tolerance=0.0,
                        comparison_rule="categorical_code_membership",
                        recovery_path=recovery_path,
                        public_input_paths=tuple(
                            sorted(replay.participant_input_checksums)
                        ),
                        expected_summary=" | ".join(accepted),
                        reproduced_summary=" | ".join(reproduced),
                        mismatches=() if passed else reproduced,
                        status="pass" if passed else "fail",
                    )
                )
                continue
            envelope = target.acceptance_envelope
            if route.method.result_kind == "sensitivity_set":
                parameter_replays = _complete_sensitivity_replays(
                    parameterized_replays=parameterized_replays,
                    item_id=key.item_id,
                    estimator_method_id=_scoring_route_estimator_method_id(
                        item_id=key.item_id,
                        route_id=route.route_id,
                    ),
                    sensitivity_parameters=route.method.sensitivity_parameters,
                )
                difference = max(replay.difference for replay in parameter_replays)
                tolerance = envelope.independent_max_abs_difference
                passed = (
                    all(replay.matched for replay in parameter_replays)
                    and difference <= tolerance + 1e-15
                )
                rows.append(
                    RecoverabilityRouteV1(
                        suite="trialeval",
                        unit_id=key.item_id,
                        context_or_checkpoint_id=key.context_tier,
                        route_id=route.route_id,
                        estimator_family=route.method.estimator_family,
                        effect_scale=route.signature.effect_scale,
                        result_kind=route.method.result_kind,
                        comparison_denominator=2 * len(parameter_replays),
                        maximum_absolute_difference=difference,
                        declared_absolute_tolerance=tolerance,
                        difference_to_tolerance_ratio=_difference_to_tolerance_ratio(
                            difference=difference,
                            tolerance=tolerance,
                        ),
                        comparison_rule="numeric_envelope",
                        recovery_path=recovery_path,
                        public_input_paths=tuple(
                            sorted(
                                {
                                    path
                                    for replay in parameter_replays
                                    for path in replay.public_input_paths
                                }
                            )
                        ),
                        expected_summary=(
                            "complete sensitivity grid="
                            + ",".join(
                                f"{parameter:.2f}"
                                for parameter in route.method.sensitivity_parameters
                            )
                        ),
                        reproduced_summary=(
                            f"{2 * len(parameter_replays)} bound endpoints; "
                            f"maximum_absolute_difference={difference:.17g}"
                        ),
                        status="pass" if passed else "fail",
                    )
                )
                continue
            numeric_replay = replay_results.get(envelope.public_verification_id)
            if numeric_replay is None:
                raise ValueError(
                    f"scoring route lacks independent replay: {key.item_id}/{route.route_id}"
                )
            difference = numeric_replay.difference
            tolerance = envelope.independent_max_abs_difference
            passed = numeric_replay.matched and difference <= tolerance + 1e-15
            rows.append(
                RecoverabilityRouteV1(
                    suite="trialeval",
                    unit_id=key.item_id,
                    context_or_checkpoint_id=key.context_tier,
                    route_id=route.route_id,
                    estimator_family=route.method.estimator_family,
                    effect_scale=route.signature.effect_scale,
                    result_kind=route.method.result_kind,
                    comparison_denominator=1,
                    maximum_absolute_difference=difference,
                    declared_absolute_tolerance=tolerance,
                    difference_to_tolerance_ratio=_difference_to_tolerance_ratio(
                        difference=difference,
                        tolerance=tolerance,
                    ),
                    comparison_rule="numeric_envelope",
                    recovery_path=recovery_path,
                    public_input_paths=numeric_replay.public_input_paths,
                    expected_summary=f"verification_id={envelope.public_verification_id}",
                    reproduced_summary=f"maximum_absolute_difference={difference:.17g}",
                    status="pass" if passed else "fail",
                )
            )
    if not rows:
        raise ValueError("TrialEval evaluator contains no credit-eligible routes")
    failed = sum(row.status == "fail" for row in rows)
    return RecoverabilityReportV1(
        suite="trialeval",
        participant_release=participant_release.name,
        evaluator_release=evaluator_release.name,
        verification_release=verification_release.name,
        required_route_count=len(rows),
        replayed_route_count=len(rows),
        failed_route_count=failed,
        maximum_absolute_difference=max(
            row.maximum_absolute_difference for row in rows
        ),
        c5_required_item_count=len(c5_keys),
        c5_repaired_item_count=(
            0 if c5_report is None else c5_report.repaired_item_count
        ),
        c5_failed_item_count=(
            0
            if c5_report is None
            else c5_report.mismatched_item_count + c5_report.unsupported_item_count
        ),
        c5_integrity_report=c5_report,
        routes=tuple(rows),
        status=(
            "pass"
            if numeric_status == "pass"
            and failed == 0
            and (c5_report is None or c5_report.status == "pass")
            else "fail"
        ),
    )


def _validated_qualified_replays(
    *,
    verification_release: Path,
    participant_release: Path,
) -> tuple[
    dict[str, _NumericReplayV1],
    dict[tuple[str, str, float], _NumericReplayV1],
]:
    """Validate and reuse the complete replay that admitted the packaged routes."""

    with ZipFile(verification_release) as verification:
        evidence = PublicRouteReplayEvidenceV1.model_validate_json(
            verification.read("verification/public_route_replay_evidence.json")
        )
        route_references = tuple(
            RouteReferenceRecordV1.model_validate_json(line)
            for line in verification.read(
                "grader/domains/route_references.jsonl"
            ).splitlines()
            if line.strip()
        )
        reference_inputs = tuple(
            RouteReferenceInputRecordV1.model_validate_json(line)
            for line in verification.read(
                "grader/domains/route_reference_inputs.jsonl"
            ).splitlines()
            if line.strip()
        )
        method_compositions = (
            tuple(
                MethodCompositionRecordV1.model_validate_json(line)
                for line in verification.read(
                    "grader/domains/method_composition.jsonl"
                ).splitlines()
                if line.strip()
            )
            if "grader/domains/method_composition.jsonl" in verification.NameToInfo
            else ()
        )
    references_by_id = {row.route_reference_id: row for row in route_references}
    inputs_by_id = {row.input_bundle_id: row for row in reference_inputs}
    compositions_by_id = {row.route_reference_id: row for row in method_compositions}
    if len(references_by_id) != len(route_references):
        raise ValueError(
            "verification release contains duplicate route-reference identities"
        )
    if len(inputs_by_id) != len(reference_inputs):
        raise ValueError(
            "verification release contains duplicate route-input identities"
        )
    if len(compositions_by_id) != len(method_compositions):
        raise ValueError(
            "verification release contains duplicate method-composition identities"
        )

    table_hashes: dict[str, str] = {}
    replays: dict[str, _NumericReplayV1] = {}
    parameterized_replays: dict[tuple[str, str, float], _NumericReplayV1] = {}
    with ZipFile(participant_release) as participant:
        for record in evidence.records:
            route_reference = references_by_id.get(record.route_reference_id)
            reference_input = inputs_by_id.get(record.input_bundle_id)
            if route_reference is None or reference_input is None:
                raise ValueError(
                    "qualified replay references an absent packaged route or input bundle"
                )
            if route_reference.checksum != record.route_reference_checksum:
                raise ValueError(
                    "qualified replay route checksum does not match the packaged route"
                )
            if reference_input.checksum != record.input_bundle_checksum:
                raise ValueError(
                    "qualified replay input checksum does not match the packaged input bundle"
                )
            if record.route_reference_id not in reference_input.route_reference_ids:
                raise ValueError(
                    "qualified replay route is not declared by its input bundle"
                )
            composition = compositions_by_id.get(record.route_reference_id)
            observed_composition_checksum = (
                None if composition is None else composition.checksum
            )
            if observed_composition_checksum != record.method_composition_checksum:
                raise ValueError(
                    "qualified replay method composition does not match the packaged route"
                )
            public_paths = tuple(
                sorted(
                    public_rel_path_for_scoreable_ref_v1(table_ref.rel_path)
                    for table_ref in reference_input.required_table_refs
                )
            )
            for table_ref in reference_input.required_table_refs:
                semantic_path = public_rel_path_for_scoreable_ref_v1(table_ref.rel_path)
                observed = table_hashes.get(semantic_path)
                if observed is None:
                    member = resolve_public_member_v1(participant, semantic_path)
                    observed = hashlib.sha256(participant.read(member)).hexdigest()
                    table_hashes[semantic_path] = observed
                if observed != table_ref.sha256:
                    raise ValueError(
                        f"qualified replay input checksum does not match participant evidence: {semantic_path}"
                    )
            replay = _NumericReplayV1(
                difference=record.max_abs_difference,
                matched=True,
                public_input_paths=public_paths,
            )
            if record.route_reference_id in replays:
                raise ValueError(
                    "qualified replay contains duplicate route-reference identities"
                )
            replays[record.route_reference_id] = replay
            if route_reference.sensitivity_parameter is None:
                _index_numeric_replay(
                    replays=replays,
                    replay_id=f"public-replay:{record.input_bundle_id}",
                    replay=replay,
                )
            if route_reference.sensitivity_parameter is not None:
                key = (
                    route_reference.task_id,
                    route_reference.estimator_method_id,
                    float(route_reference.sensitivity_parameter),
                )
                previous = parameterized_replays.setdefault(key, replay)
                if previous != replay:
                    raise ValueError(
                        f"qualified sensitivity replay is ambiguous: {key!r}"
                    )
    return replays, parameterized_replays


def _replay_trialdev_method(
    request: tuple[Path, float, str],
) -> TrialDevObservationalReplayReportV1:
    """Replay one TrialDev scenario-method pair."""

    scenario_root, absolute_tolerance, method_route_id = request
    return replay_trialdev_observational_reference(
        scenario_root,
        absolute_tolerance=absolute_tolerance,
        selected_method_route_id=method_route_id,
    )


def _trialdev_method_route_ids(scenario_root: Path) -> tuple[str, ...]:
    """Read the complete public observational-method census for one scenario."""

    path = scenario_root / "public" / "observational_method_catalog.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("methods"), list):
        raise ValueError(f"invalid TrialDev observational method catalog: {path}")
    route_ids: list[str] = []
    for record in payload["methods"]:
        if not isinstance(record, dict):
            raise ValueError(f"invalid TrialDev observational method record: {path}")
        route_id = record.get("method_route_id")
        if not isinstance(route_id, str) or not route_id:
            raise ValueError(
                f"TrialDev observational method lacks a route identity: {path}"
            )
        route_ids.append(route_id)
    if not route_ids or len(route_ids) != len(set(route_ids)):
        raise ValueError(
            f"TrialDev observational method routes must be nonempty and unique: {path}"
        )
    return tuple(route_ids)


def _trialdev_recoverability(
    *,
    participant_release: Path,
    evaluator_release: Path,
    verification_release: Path,
    workers: int,
    absolute_tolerance: float,
) -> RecoverabilityReportV1:
    with tempfile.TemporaryDirectory(prefix="trialagentbench-recovery-") as temporary:
        root = Path(temporary)
        bundle = root / "bundle"
        verification = root / "verification"
        bundle.mkdir()
        verification.mkdir()
        _extract_public_archive(participant_release, bundle)
        _extract_public_archive(evaluator_release, bundle)
        _extract_public_archive(verification_release, verification)

        registry = ScientificSourceRegistryV1.model_validate_json(
            (verification / "scientific_source_registry.json").read_text(
                encoding="utf-8"
            )
        )
        scientific_inventory = (
            TrialDevScientificConstructionInventoryV1.model_validate_json(
                (verification / "scientific_construction_inventory.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        validate_trialdev_scientific_inventory(
            inventory=scientific_inventory,
            registry=registry,
            bundle_root=bundle,
        )
        scenario_roots = tuple(
            sorted(path for path in bundle.glob("scenario_s*") if path.is_dir())
        )
        if not scenario_roots:
            raise ValueError(
                "TrialDev participant release contains no scenario directories"
            )

        phase_root = verification / "phase_replay"
        required = {
            "cases.jsonl",
            "records.jsonl",
            "materialized",
        }
        missing = sorted(name for name in required if not (phase_root / name).exists())
        if missing:
            raise ValueError(
                "TrialDev verification release lacks public randomized-phase replay "
                f"artifacts: {missing!r}"
            )
        phase = validate_trialdev_phase_replay(
            bundle_root=bundle,
            materialized_root=phase_root / "materialized",
            cases_path=phase_root / "cases.jsonl",
            records_path=phase_root / "records.jsonl",
            absolute_tolerance=absolute_tolerance,
        )
        public_phase_records = tuple(
            TrialDevPublicPhaseReplayRecordV1.model_validate_json(line)
            for line in (phase_root / "records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        public_phase_by_request = {
            record.request_checksum: record for record in public_phase_records
        }
        if len(public_phase_by_request) != len(public_phase_records):
            raise ValueError(
                "TrialDev public phase replay contains duplicate request checksums"
            )
        replay_requests = tuple(
            (scenario, absolute_tolerance, method_route_id)
            for scenario in scenario_roots
            for method_route_id in _trialdev_method_route_ids(scenario)
        )
        if workers == 1:
            observational = tuple(
                _replay_trialdev_method(request) for request in replay_requests
            )
        else:
            with single_threaded_numerical_process_pool(
                workers=min(workers, len(replay_requests))
            ) as executor:
                observational = tuple(
                    executor.map(_replay_trialdev_method, replay_requests)
                )

    rows: list[RecoverabilityRouteV1] = []
    for report in observational:
        for method in report.methods:
            maximum_difference = max(
                method.maximum_utility_absolute_error,
                method.maximum_efficacy_gain_absolute_error,
                method.maximum_standard_error_absolute_error,
                method.maximum_interval_endpoint_absolute_error,
            )
            rows.append(
                RecoverabilityRouteV1(
                    suite="trialdev",
                    unit_id=report.scenario_id,
                    context_or_checkpoint_id="observational_review",
                    route_id=method.method_route_id,
                    estimator_family=method.estimator_id,
                    effect_scale="objective_utility",
                    result_kind="decision",
                    comparison_denominator=1,
                    maximum_absolute_difference=maximum_difference,
                    declared_absolute_tolerance=absolute_tolerance,
                    difference_to_tolerance_ratio=maximum_difference
                    / absolute_tolerance,
                    comparison_rule="numeric_envelope",
                    recovery_path="trialdev_public_replay",
                    public_input_paths=("public/observational_extract.parquet",),
                    expected_summary=f"absolute_tolerance={absolute_tolerance:.17g}",
                    reproduced_summary=f"maximum_absolute_difference={maximum_difference:.17g}",
                    status=method.status,
                )
            )
    for record in phase.records:
        public_record = public_phase_by_request.get(record.request_checksum)
        if public_record is None:
            raise ValueError(
                "TrialDev phase validation lacks its participant-public replay record: "
                f"{record.scenario_id}/{record.request_checksum}"
            )
        rows.append(
            RecoverabilityRouteV1(
                suite="trialdev",
                unit_id=f"{record.scenario_id}:{record.request_checksum}",
                context_or_checkpoint_id=public_record.phase_id,
                route_id=f"{record.scenario_id}:{record.request_checksum}",
                estimator_family="public_randomized_phase_replay",
                effect_scale="phase_specific",
                result_kind="decision",
                comparison_denominator=1,
                maximum_absolute_difference=record.maximum_absolute_error,
                declared_absolute_tolerance=absolute_tolerance,
                difference_to_tolerance_ratio=record.maximum_absolute_error
                / absolute_tolerance,
                comparison_rule="numeric_envelope",
                recovery_path="trialdev_public_replay",
                public_input_paths=tuple(sorted(public_record.public_source_checksums)),
                expected_summary=f"absolute_tolerance={absolute_tolerance:.17g}",
                reproduced_summary=f"maximum_absolute_difference={record.maximum_absolute_error:.17g}",
                status=record.status,
            )
        )
    if not rows:
        raise ValueError("TrialDev verification produced no score-bearing replay rows")
    failed = sum(row.status == "fail" for row in rows)
    all_reports_pass = (
        all(report.status == "pass" for report in observational)
        and phase.status == "pass"
    )
    return RecoverabilityReportV1(
        suite="trialdev",
        participant_release=participant_release.name,
        evaluator_release=evaluator_release.name,
        verification_release=verification_release.name,
        required_route_count=len(rows),
        replayed_route_count=len(rows),
        failed_route_count=failed,
        maximum_absolute_difference=max(
            row.maximum_absolute_difference for row in rows
        ),
        routes=tuple(rows),
        status="pass" if all_reports_pass and failed == 0 else "fail",
    )


def recover_release(
    *,
    participant_release: Path,
    evaluator_release: Path,
    verification_release: Path,
    workers: int = 1,
    absolute_tolerance: float = RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1,
    reuse_qualified_trialeval_replay: bool = False,
) -> RecoverabilityReportV1:
    """Independently recover score-bearing results from three public roles."""

    if absolute_tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    paths = (participant_release, evaluator_release, verification_release)
    if any(not Path(path).is_file() for path in paths):
        raise FileNotFoundError(
            "participant, evaluator, and verification releases must be files"
        )
    with ZipFile(evaluator_release) as evaluator:
        names = set(evaluator.namelist())
    if "grader/scoring_keys.jsonl" in names:
        return _trialeval_recoverability(
            participant_release=participant_release,
            evaluator_release=evaluator_release,
            verification_release=verification_release,
            workers=workers,
            reuse_qualified_replay=reuse_qualified_trialeval_replay,
        )
    if any(name.endswith("/grader/evaluation_target_register.jsonl") for name in names):
        return _trialdev_recoverability(
            participant_release=participant_release,
            evaluator_release=evaluator_release,
            verification_release=verification_release,
            workers=workers,
            absolute_tolerance=absolute_tolerance,
        )
    raise ValueError(
        "evaluator release does not identify TrialEvalBench or TrialDevBench"
    )


def write_recoverability_report(
    output_dir: Path, report: RecoverabilityReportV1
) -> None:
    """Write machine-readable and concise human-readable recovery artifacts."""

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "recoverability_report.json").write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    if report.c5_integrity_report is not None:
        (output_dir / "c5_integrity_recovery.json").write_text(
            report.c5_integrity_report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    row_payloads = [row.model_dump(mode="json") for row in report.routes]
    fieldnames = tuple(RecoverabilityRouteV1.model_fields)
    with (output_dir / "recoverability_routes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_payloads)
    pd.DataFrame(row_payloads, columns=fieldnames).to_parquet(
        output_dir / "recoverability_routes.parquet",
        index=False,
    )
    lines = [
        "# TrialAgentBench Recoverability",
        "",
        f"- Suite: `{report.suite}`",
        f"- Status: `{report.status}`",
        f"- Required routes: `{report.required_route_count}`",
        f"- Replayed routes: `{report.replayed_route_count}`",
        f"- Failed routes: `{report.failed_route_count}`",
        f"- Maximum absolute difference: `{report.maximum_absolute_difference:.12g}`",
        "- Private generating state used: `false`",
        "",
        "Each row was independently recomputed before evaluator targets were opened for comparison.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "RecoverabilityReportV1",
    "RecoverabilityRouteV1",
    "TrialEvalNonNumericReplayV1",
    "recover_release",
    "write_recoverability_report",
]
