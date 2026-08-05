"""Independently audit the released TrialDev portfolio data surface."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Literal, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialDevEpisodeRealismV1(_Record):
    """Clinical-data census for one released randomized episode."""

    episode_id: str = Field(min_length=1)
    world_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    phase_id: Literal["phase1", "phase2", "phase3"]
    row_count: int = Field(gt=0)
    follow_up_days: int = Field(gt=0)
    treated_count: int = Field(gt=0)
    control_count: int = Field(gt=0)
    efficacy_event_rate_treated: float | None = Field(default=None, ge=0.0, le=1.0)
    efficacy_event_rate_control: float | None = Field(default=None, ge=0.0, le=1.0)
    serious_ae_rate_treated: float = Field(ge=0.0, le=1.0)
    serious_ae_rate_control: float = Field(ge=0.0, le=1.0)
    discontinuation_rate_treated: float = Field(ge=0.0, le=1.0)
    discontinuation_rate_control: float = Field(ge=0.0, le=1.0)
    loss_to_follow_up_rate: float = Field(ge=0.0, le=1.0)
    terminal_event_rate: float = Field(ge=0.0, le=1.0)


class TrialDevObservationalRealismV1(_Record):
    """Usability and support census for one released observational extract."""

    world_id: str = Field(min_length=1)
    row_count: int = Field(gt=0)
    column_count: int = Field(gt=0)
    treatment_counts: dict[str, int]
    declared_adjustment_covariate_count: int = Field(gt=0)
    complete_case_rate: float = Field(ge=0.0, le=1.0)
    minimum_treatment_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_treatments(self) -> Self:
        """Require a nonempty, internally consistent treatment census."""

        if not self.treatment_counts or any(
            count <= 0 for count in self.treatment_counts.values()
        ):
            raise ValueError("Every observational treatment must contain participants.")
        if sum(self.treatment_counts.values()) != self.row_count:
            raise ValueError(
                "Observational treatment counts must sum to the extract row count."
            )
        if self.minimum_treatment_count != min(self.treatment_counts.values()):
            raise ValueError("Minimum observational treatment count is inconsistent.")
        return self


class TrialDevPortfolioReleaseAuditV1(_Record):
    """Independent integrity, realism, and usability result for one release."""

    schema_id: Literal[
        "trialagentbench.validation.trialdev_portfolio_release_audit/v1"
    ] = "trialagentbench.validation.trialdev_portfolio_release_audit/v1"
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_source_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    world_count: int = Field(ge=0)
    participant_view_count: int = Field(ge=0)
    randomized_episode_count: int = Field(ge=0)
    randomized_row_count: int = Field(ge=0)
    observational_row_count: int = Field(ge=0)
    episode_realism: tuple[TrialDevEpisodeRealismV1, ...]
    observational_realism: tuple[TrialDevObservationalRealismV1, ...]
    findings: tuple[str, ...]
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        """Bind status to canonical findings and exact aggregate counts."""

        if self.findings != tuple(sorted(set(self.findings))):
            raise ValueError("Release-audit findings must be sorted and unique.")
        if (self.status == "pass") != (not self.findings):
            raise ValueError("Release-audit status disagrees with its findings.")
        if self.randomized_episode_count != len(self.episode_realism):
            raise ValueError(
                "Randomized episode count disagrees with episode evidence."
            )
        if self.world_count != len(self.observational_realism):
            raise ValueError("World count disagrees with observational evidence.")
        if self.randomized_row_count != sum(
            row.row_count for row in self.episode_realism
        ):
            raise ValueError("Randomized row count disagrees with episode evidence.")
        if self.observational_row_count != sum(
            row.row_count for row in self.observational_realism
        ):
            raise ValueError("Observational row count disagrees with world evidence.")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _records(payload: dict[str, object], key: str) -> tuple[dict[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{key!r} must be an array of objects.")
    return tuple(row for row in value if isinstance(row, dict))


def _safe_file(root: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str):
        raise ValueError("Released artifact path must be a string.")
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != relative_path
    ):
        raise ValueError(f"Released artifact path is unsafe: {relative_path!r}.")
    path = (root / relative_path).resolve(strict=True)
    if not path.is_relative_to(root.resolve(strict=True)) or not path.is_file():
        raise ValueError(
            f"Released artifact is not a regular in-release file: {relative_path!r}."
        )
    return path


def _require_checksum(path: Path, expected: object) -> None:
    if not isinstance(expected, str) or _sha256(path) != expected:
        raise ValueError(f"Released artifact checksum mismatch: {path}.")


def _binary(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"Released table lacks required column {column!r}.")
    values = pd.to_numeric(frame[column], errors="raise")
    if not set(values.unique()).issubset({0, 1}):
        raise ValueError(f"Released column {column!r} must be binary.")
    return values.astype(int)


def _finite_time(frame: pd.DataFrame, column: str, *, follow_up_days: int) -> pd.Series:
    if column not in frame:
        raise ValueError(f"Released table lacks required column {column!r}.")
    values = pd.to_numeric(frame[column], errors="raise").astype(float)
    if (
        not values.map(math.isfinite).all()
        or bool((values < 0.0).any())
        or bool((values > follow_up_days + 1e-9).any())
    ):
        raise ValueError(
            f"Released time {column!r} lies outside the declared follow-up window."
        )
    return values


def _serious_event(safety: pd.DataFrame) -> pd.Series:
    columns = tuple(
        str(column)
        for column in safety.columns
        if str(column).startswith("AE_") and str(column).endswith("_SERIOUS")
    )
    if not columns:
        raise ValueError(
            "Released safety table lacks serious-adverse-event indicators."
        )
    components: list[pd.Series] = []
    for column in columns:
        event_column = f"{column.removesuffix('_SERIOUS')}_EVENT_E"
        event = _binary(safety, event_column)
        serious = pd.to_numeric(safety[column], errors="raise")
        if bool(serious.loc[event.eq(1)].isna().any()) or bool(
            serious.loc[event.eq(0)].notna().any()
        ):
            raise ValueError(
                f"Released seriousness {column!r} must be observed exactly when its event occurs."
            )
        observed = serious.loc[event.eq(1)]
        if not set(observed.unique()).issubset({0, 1}):
            raise ValueError(
                f"Released seriousness {column!r} must be binary when observed."
            )
        components.append(serious.fillna(0).astype(int))
    values = pd.concat(components, axis="columns")
    return pd.Series(values.max(axis=1).astype(int), index=safety.index)


def _check_event_time(
    frame: pd.DataFrame,
    *,
    event_column: str,
    time_column: str,
    follow_up_days: int,
) -> tuple[pd.Series, pd.Series]:
    event = _binary(frame, event_column)
    time = _finite_time(frame, time_column, follow_up_days=follow_up_days)
    return event, time


def _episode_realism(
    root: Path, episode: dict[str, object]
) -> TrialDevEpisodeRealismV1:
    manifest_path = _safe_file(root, episode.get("episode_manifest_relative_path"))
    _require_checksum(manifest_path, episode.get("episode_manifest_sha256"))
    manifest = _object(manifest_path)
    metadata = {
        str(row.get("metadata_id")): row for row in _records(manifest, "metadata_files")
    }
    files = {str(row.get("table_id")): row for row in _records(manifest, "files")}
    if set(files) != {"participants", "endpoints", "safety"}:
        raise ValueError(
            "Randomized episode requires participants, endpoints, and safety tables."
        )
    if set(metadata) != {"arm_mapping", "request", "execution_summary"}:
        raise ValueError(
            "Randomized episode requires arm mapping, request, and execution summary metadata."
        )
    loaded: dict[str, pd.DataFrame] = {}
    for table_id, record in files.items():
        path = _safe_file(root, record.get("relative_path"))
        _require_checksum(path, record.get("sha256"))
        loaded[table_id] = pd.read_parquet(path)
    metadata_paths: dict[str, Path] = {}
    for metadata_id, record in metadata.items():
        path = _safe_file(root, record.get("relative_path"))
        _require_checksum(path, record.get("sha256"))
        metadata_paths[metadata_id] = path
    request = _object(metadata_paths["request"])
    mapping = _object(metadata_paths["arm_mapping"])
    participants, endpoints, safety = (
        loaded[name] for name in ("participants", "endpoints", "safety")
    )
    row_count = len(participants)
    if row_count <= 0 or any(len(frame) != row_count for frame in loaded.values()):
        raise ValueError(
            "Randomized episode tables must have the same positive row count."
        )
    if row_count != episode.get("row_count") or row_count != request.get(
        "target_sample_size"
    ):
        raise ValueError(
            "Randomized episode row count disagrees with its request or release manifest."
        )
    for frame in (participants, endpoints, safety):
        if "USUBJID" not in frame or frame["USUBJID"].duplicated().any():
            raise ValueError("Every randomized table requires unique USUBJID values.")
    ids = set(participants["USUBJID"])
    if any(set(frame["USUBJID"]) != ids for frame in (endpoints, safety)):
        raise ValueError(
            "Randomized episode tables do not identify the same participants."
        )
    phase_id = str(episode.get("phase_id"))
    if phase_id not in {"phase1", "phase2", "phase3"} or phase_id != request.get(
        "phase_id"
    ):
        raise ValueError("Randomized episode phase disagrees with its request.")
    follow_up = request.get("follow_up_days")
    if isinstance(follow_up, bool) or not isinstance(follow_up, int) or follow_up <= 0:
        raise ValueError("Randomized episode follow-up must be a positive integer.")
    control_arm = mapping.get("control_arm_id")
    candidate_arms = mapping.get("candidate_arm_ids")
    drug_by_arm = mapping.get("drug_id_by_arm")
    if (
        not isinstance(control_arm, str)
        or not isinstance(candidate_arms, list)
        or len(candidate_arms) != 1
        or not isinstance(candidate_arms[0], str)
        or not isinstance(drug_by_arm, dict)
    ):
        raise ValueError(
            "Randomized episode requires one control and one candidate arm."
        )
    candidate_arm = candidate_arms[0]
    if set(participants["ARM"].astype(str)) != {control_arm, candidate_arm}:
        raise ValueError("Randomized episode arm labels disagree with arm mapping.")
    for frame in (endpoints, safety):
        if not frame["ARM"].astype(str).equals(participants["ARM"].astype(str)):
            raise ValueError("Randomized episode arm assignments differ across tables.")
    treated = participants["ARM"].astype(str).eq(candidate_arm)
    control = participants["ARM"].astype(str).eq(control_arm)
    event = _binary(endpoints, "EVENT")
    competing = _binary(endpoints, "COMPETING_EVENT")
    _finite_time(endpoints, "TIME", follow_up_days=follow_up)
    if bool(event.eq(1).mul(competing.eq(1)).any()):
        raise ValueError("Primary and competing events must be mutually exclusive.")
    if phase_id == "phase1":
        if bool(event.any()) or bool(competing.any()):
            raise ValueError(
                "Phase-1 episodes must not fabricate an unrequested efficacy endpoint."
            )
        efficacy_treated = None
        efficacy_control = None
    else:
        efficacy_treated = float(event.loc[treated].mean())
        efficacy_control = float(event.loc[control].mean())
    safety_events: dict[str, tuple[pd.Series, pd.Series]] = {}
    for event_column in ("DISCONTINUATION_E", "LTFU_E", "TERMINAL_EVENT"):
        time_column = {
            "DISCONTINUATION_E": "DISCONTINUATION_T",
            "LTFU_E": "LTFU_T",
            "TERMINAL_EVENT": "TERMINAL_TIME",
        }[event_column]
        safety_events[event_column] = _check_event_time(
            safety,
            event_column=event_column,
            time_column=time_column,
            follow_up_days=follow_up,
        )
    for column in tuple(
        str(value)
        for value in safety.columns
        if str(value).startswith("AE_") and str(value).endswith("_E")
    ):
        safety_events[column] = _check_event_time(
            safety,
            event_column=column,
            time_column=f"{column.removesuffix('_E')}_T",
            follow_up_days=follow_up,
        )
    ltfu_event, ltfu_time = safety_events["LTFU_E"]
    terminal_event, terminal_time = safety_events["TERMINAL_EVENT"]
    for event_column, (event_indicator, event_time) in safety_events.items():
        if event_column in {"LTFU_E", "TERMINAL_EVENT"}:
            continue
        observed = event_indicator.eq(1)
        if bool(
            (
                event_time.loc[observed & ltfu_event.eq(1)]
                > ltfu_time.loc[observed & ltfu_event.eq(1)] + 1e-9
            ).any()
        ):
            raise ValueError(
                f"Observed {event_column!r} occurs after loss to follow-up."
            )
        if bool(
            (
                event_time.loc[observed & terminal_event.eq(1)]
                > terminal_time.loc[observed & terminal_event.eq(1)] + 1e-9
            ).any()
        ):
            raise ValueError(
                f"Observed {event_column!r} occurs after the terminal event."
            )
    serious = _serious_event(safety)
    discontinuation = _binary(safety, "DISCONTINUATION_E")
    ltfu = _binary(safety, "LTFU_E")
    terminal = _binary(safety, "TERMINAL_EVENT")
    return TrialDevEpisodeRealismV1(
        episode_id=str(episode.get("episode_id")),
        world_id=str(episode.get("world_id")),
        asset_id=str(episode.get("asset_id")),
        phase_id=phase_id,
        row_count=row_count,
        follow_up_days=follow_up,
        treated_count=int(treated.sum()),
        control_count=int(control.sum()),
        efficacy_event_rate_treated=efficacy_treated,
        efficacy_event_rate_control=efficacy_control,
        serious_ae_rate_treated=float(serious.loc[treated].mean()),
        serious_ae_rate_control=float(serious.loc[control].mean()),
        discontinuation_rate_treated=float(discontinuation.loc[treated].mean()),
        discontinuation_rate_control=float(discontinuation.loc[control].mean()),
        loss_to_follow_up_rate=float(ltfu.mean()),
        terminal_event_rate=float(terminal.mean()),
    )


def _observational_realism(root: Path, world_id: str) -> TrialDevObservationalRealismV1:
    public = root / "worlds" / world_id / "public"
    brief = (public / "study_brief.md").read_text(encoding="utf-8")
    if "synthetic" in brief.casefold():
        raise ValueError("Participant study brief must not disclose data construction.")
    frame = pd.read_parquet(public / "observational_extract.parquet")
    method_catalog = _object(public / "observational_method_catalog.json")
    methods = _records(method_catalog, "methods")
    if not methods:
        raise ValueError(
            "Observational method catalogue must declare at least one method."
        )
    covariates = methods[0].get("adjustment_covariates")
    if (
        not isinstance(covariates, list)
        or not covariates
        or not all(isinstance(value, str) for value in covariates)
    ):
        raise ValueError("Observational method catalogue lacks adjustment covariates.")
    if any(method.get("adjustment_covariates") != covariates for method in methods):
        raise ValueError(
            "Observational methods must share one declared adjustment set."
        )
    required = {"USUBJID", "TREATMENT", *covariates}
    if not required <= set(frame):
        raise ValueError(
            "Observational extract lacks participant, treatment, or adjustment columns."
        )
    if frame.empty or frame["USUBJID"].duplicated().any():
        raise ValueError("Observational extract requires unique participants.")
    counts = {
        str(key): int(value)
        for key, value in frame["TREATMENT"].value_counts().sort_index().items()
    }
    if len(counts) != 4:
        raise ValueError(
            "Portfolio observational evidence requires one control and three investigational regimens."
        )
    return TrialDevObservationalRealismV1(
        world_id=world_id,
        row_count=len(frame),
        column_count=len(frame.columns),
        treatment_counts=counts,
        declared_adjustment_covariate_count=len(covariates),
        complete_case_rate=float(frame.loc[:, covariates].notna().all(axis=1).mean()),
        minimum_treatment_count=min(counts.values()),
    )


def audit_trialdev_portfolio_release_v1(
    release_root: Path,
) -> TrialDevPortfolioReleaseAuditV1:
    """Audit all released portfolio worlds and randomized episodes from immutable bytes."""

    root = Path(release_root).resolve(strict=True)
    manifest_path = root / "evaluator" / "release_manifest.json"
    catalogue_path = root / "participant_catalogue.json"
    manifest = _object(manifest_path)
    catalogue = _object(catalogue_path)
    episodes = _records(manifest, "episodes")
    views = _records(catalogue, "views")
    source_identity = manifest.get("source_identity")
    if not isinstance(source_identity, str) or len(source_identity) != 64:
        raise ValueError("Portfolio release manifest lacks a SHA-256 source identity.")
    if catalogue.get("source_identity") != source_identity:
        raise ValueError(
            "Participant and evaluator portfolio roles disagree on source identity."
        )
    world_ids = tuple(sorted({str(row.get("world_id")) for row in episodes}))
    findings: list[str] = []
    if len(world_ids) != 12:
        findings.append("world_count_not_12")
    if len(views) != 96:
        findings.append("participant_view_count_not_96")
    if len(episodes) != 108:
        findings.append("randomized_episode_count_not_108")
    episode_realism = tuple(_episode_realism(root, row) for row in episodes)
    observed_keys = {
        (row.world_id, row.asset_id, row.phase_id) for row in episode_realism
    }
    expected_keys: set[tuple[str, str, str]] = set()
    for world_id in world_ids:
        asset_ids = {
            row.asset_id for row in episode_realism if row.world_id == world_id
        }
        if len(asset_ids) != 3:
            findings.append("randomized_world_candidate_count_not_3")
        expected_keys.update(
            (world_id, asset_id, phase_id)
            for asset_id in asset_ids
            for phase_id in ("phase1", "phase2", "phase3")
        )
    if observed_keys != expected_keys:
        findings.append("randomized_episode_factorial_incomplete")
    observational = tuple(
        _observational_realism(root, world_id) for world_id in world_ids
    )
    return TrialDevPortfolioReleaseAuditV1(
        release_manifest_sha256=_sha256(manifest_path),
        release_source_identity=source_identity,
        world_count=len(world_ids),
        participant_view_count=len(views),
        randomized_episode_count=len(episode_realism),
        randomized_row_count=sum(row.row_count for row in episode_realism),
        observational_row_count=sum(row.row_count for row in observational),
        episode_realism=episode_realism,
        observational_realism=observational,
        findings=tuple(sorted(set(findings))),
        status="pass" if not findings else "fail",
    )


__all__ = [
    "TrialDevEpisodeRealismV1",
    "TrialDevObservationalRealismV1",
    "TrialDevPortfolioReleaseAuditV1",
    "audit_trialdev_portfolio_release_v1",
]
