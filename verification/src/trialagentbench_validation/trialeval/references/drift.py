"""Validate public-evidence reference drift dispositions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_validation.contracts.scoring.scoring_role_policy import (
    OFFICIAL_SCOREABLE_VARIANT_ROLES_V1,
)
from trialagentbench_validation.io.json import write_json_model
from trialagentbench_validation.trialeval.references.numeric import (
    PublicEvidenceReferenceDriftDispositionRowV1,
)

PublicEvidenceReferenceDriftValidationStatusV1 = Literal["pass", "fail"]


class PublicEvidenceReferenceDriftValidationFindingV1(BaseModel):
    """One finding from public-evidence reference drift validation."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., min_length=1)
    route_reference_id: str | None = None
    estimator_method_id: str | None = None
    message: str = Field(..., min_length=1)


class PublicEvidenceReferenceDriftValidationReportV1(BaseModel):
    """Validation report for public-evidence reference drift disposition rows."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal[
        "trialagentbench.public_evidence_reference_drift_validation_report/v1"
    ] = "trialagentbench.public_evidence_reference_drift_validation_report/v1"
    source_path: str
    status: PublicEvidenceReferenceDriftValidationStatusV1
    row_count: int
    no_drift_count: int
    blocked_derivation_gap_count: int
    official_scoreable_blocked_derivation_gap_count: int
    required_primary_blocked_derivation_gap_count: int
    credit_eligible_primary_alternative_blocked_derivation_gap_count: int
    non_official_blocked_derivation_gap_count: int
    score_affecting_drift_count: int
    classification_counts: dict[str, int]
    variant_role_counts: dict[str, int]
    blocked_variant_role_counts: dict[str, int]
    disposition_counts: dict[str, int]
    method_counts: dict[str, int]
    blocked_method_counts: dict[str, int]
    findings: tuple[PublicEvidenceReferenceDriftValidationFindingV1, ...]


def validate_public_evidence_reference_drift_validation_v1(
    drift_jsonl: Path,
) -> PublicEvidenceReferenceDriftValidationReportV1:
    """Validate release eligibility for public-evidence reference drift rows."""

    return validate_public_evidence_reference_drift_rows_v1(
        rows=_read_drift_rows(drift_jsonl),
        source_path=drift_jsonl.as_posix(),
    )


def validate_public_evidence_reference_drift_rows_v1(
    *,
    rows: tuple[PublicEvidenceReferenceDriftDispositionRowV1, ...],
    source_path: str,
) -> PublicEvidenceReferenceDriftValidationReportV1:
    """Validate drift rows already loaded from a public-reference replay."""

    if not rows:
        raise ValueError(
            "Public-evidence reference drift validation requires at least one row."
        )
    findings: list[PublicEvidenceReferenceDriftValidationFindingV1] = []
    for row in rows:
        if row.classification == "blocked_derivation_gap":
            if row.variant_role in OFFICIAL_SCOREABLE_VARIANT_ROLES_V1:
                findings.append(
                    PublicEvidenceReferenceDriftValidationFindingV1(
                        code="official_scoreable_blocked_derivation_gap",
                        route_reference_id=row.route_reference_id,
                        estimator_method_id=row.estimator_method_id,
                        message=(
                            "Official scoreable reference rows must be publicly replayable; "
                            f"variant_role={row.variant_role!r} remains blocked."
                        ),
                    )
                )
            if row.disposition is None or not row.required_release_action:
                findings.append(
                    PublicEvidenceReferenceDriftValidationFindingV1(
                        code="missing_disposition_for_blocked_derivation_gap",
                        route_reference_id=row.route_reference_id,
                        estimator_method_id=row.estimator_method_id,
                        message="Blocked derivation gaps require disposition and a release action.",
                    )
                )
        elif row.classification != "no_drift":
            findings.append(
                PublicEvidenceReferenceDriftValidationFindingV1(
                    code="score_affecting_or_unresolved_drift",
                    route_reference_id=row.route_reference_id,
                    estimator_method_id=row.estimator_method_id,
                    message=f"Classification {row.classification!r} is incompatible with same-release grading.",
                )
            )

    classification_counts = Counter(row.classification for row in rows)
    blocked_rows = [
        row for row in rows if row.classification == "blocked_derivation_gap"
    ]
    official_blocked_rows = [
        row
        for row in blocked_rows
        if row.variant_role in OFFICIAL_SCOREABLE_VARIANT_ROLES_V1
    ]
    score_affecting_count = sum(
        1
        for row in rows
        if row.classification not in {"no_drift", "blocked_derivation_gap"}
    )
    blocked_variant_role_counts = Counter(row.variant_role for row in blocked_rows)
    return PublicEvidenceReferenceDriftValidationReportV1(
        source_path=source_path,
        status="fail" if findings else "pass",
        row_count=len(rows),
        no_drift_count=classification_counts.get("no_drift", 0),
        blocked_derivation_gap_count=classification_counts.get(
            "blocked_derivation_gap", 0
        ),
        official_scoreable_blocked_derivation_gap_count=len(official_blocked_rows),
        required_primary_blocked_derivation_gap_count=blocked_variant_role_counts.get(
            "required_primary", 0
        ),
        credit_eligible_primary_alternative_blocked_derivation_gap_count=blocked_variant_role_counts.get(
            "credit_eligible_primary_alternative", 0
        ),
        non_official_blocked_derivation_gap_count=len(blocked_rows)
        - len(official_blocked_rows),
        score_affecting_drift_count=score_affecting_count,
        classification_counts=dict(sorted(classification_counts.items())),
        variant_role_counts=dict(
            sorted(Counter(row.variant_role for row in rows).items())
        ),
        blocked_variant_role_counts=dict(sorted(blocked_variant_role_counts.items())),
        disposition_counts=dict(
            sorted(Counter(str(row.disposition) for row in blocked_rows).items())
        ),
        method_counts=dict(
            sorted(Counter(row.estimator_method_id for row in rows).items())
        ),
        blocked_method_counts=dict(
            sorted(Counter(row.estimator_method_id for row in blocked_rows).items())
        ),
        findings=tuple(findings),
    )


def write_public_evidence_reference_drift_validation_artifacts_v1(
    *,
    drift_jsonl: Path,
    out_dir: Path,
) -> PublicEvidenceReferenceDriftValidationReportV1:
    """Write public-evidence reference drift validation artifacts."""

    report = validate_public_evidence_reference_drift_validation_v1(drift_jsonl)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_model(
        out_dir / "public_evidence_reference_drift_validation_report.json", report
    )
    (out_dir / "public_evidence_reference_drift_validation_report.md").write_text(
        render_public_evidence_reference_drift_validation_report_v1(report),
        encoding="utf-8",
    )
    return report


def render_public_evidence_reference_drift_validation_report_v1(
    report: PublicEvidenceReferenceDriftValidationReportV1,
) -> str:
    """Render a public-evidence reference drift validation report."""

    lines = [
        "# Public-Evidence Reference Drift Validation Report",
        "",
        f"- Source: `{report.source_path}`",
        f"- Status: `{report.status}`",
        f"- Rows: `{report.row_count}`",
        f"- No drift: `{report.no_drift_count}`",
        f"- Blocked derivation gaps: `{report.blocked_derivation_gap_count}`",
        f"- Official scoreable blocked derivation gaps: `{report.official_scoreable_blocked_derivation_gap_count}`",
        f"- Primary-official blocked derivation gaps: `{report.required_primary_blocked_derivation_gap_count}`",
        f"- Official-equivalent blocked derivation gaps: `{report.credit_eligible_primary_alternative_blocked_derivation_gap_count}`",
        f"- Non-official blocked derivation gaps: `{report.non_official_blocked_derivation_gap_count}`",
        f"- Score-affecting drift: `{report.score_affecting_drift_count}`",
        "",
        "## Classification Counts",
        "",
    ]
    lines.extend(
        f"- `{key}`: `{value}`"
        for key, value in sorted(report.classification_counts.items())
    )
    lines.extend(["", "## Blocked Derivation Variant Roles", ""])
    if report.blocked_variant_role_counts:
        lines.extend(
            f"- `{key}`: `{value}`"
            for key, value in sorted(report.blocked_variant_role_counts.items())
        )
    else:
        lines.append("No blocked derivation gaps.")
    lines.extend(["", "## Blocked Derivation Dispositions", ""])
    if report.disposition_counts:
        lines.extend(
            f"- `{key}`: `{value}`"
            for key, value in sorted(report.disposition_counts.items())
        )
    else:
        lines.append("No blocked derivation gaps.")
    lines.extend(["", "## Findings", ""])
    if report.findings:
        for finding in report.findings:
            lines.append(
                f"- `{finding.code}` `{finding.route_reference_id}`: {finding.message}"
            )
    else:
        lines.append("- No findings.")
    return "\n".join(lines) + "\n"


def _read_drift_rows(
    drift_jsonl: Path,
) -> tuple[PublicEvidenceReferenceDriftDispositionRowV1, ...]:
    if not drift_jsonl.is_file():
        raise FileNotFoundError(
            f"Missing public-evidence reference drift JSONL: {drift_jsonl}"
        )
    rows: list[PublicEvidenceReferenceDriftDispositionRowV1] = []
    for line_number, line in enumerate(
        drift_jsonl.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(
                PublicEvidenceReferenceDriftDispositionRowV1.model_validate_json(line)
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid drift JSONL row at {drift_jsonl}:{line_number}: {exc}"
            ) from exc
    if not rows:
        raise ValueError(
            "Public-evidence reference drift validation requires at least one row."
        )
    return tuple(rows)
